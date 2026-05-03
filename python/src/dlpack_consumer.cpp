// Copyright © 2026 Apple Inc.

#include "python/src/dlpack_consumer.h"

#include <cstring>
#include <memory>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/dtype.h"
#include "python/src/convert.h"
#include "python/src/dlpack_format.h"

namespace {

mx::Dtype dlpack_to_mlx_dtype(const nb::dlpack::dtype& dt) {
  if (dt.lanes != 1) {
    std::ostringstream msg;
    msg << "[from_dlpack] DLPack tensors with lanes != 1 are not supported "
        << "(got lanes=" << dt.lanes << ").";
    throw std::invalid_argument(msg.str());
  }
  using Code = nb::dlpack::dtype_code;
  switch (static_cast<Code>(dt.code)) {
    case Code::Bool:
      if (dt.bits == 8)
        return mx::bool_;
      break;
    case Code::Int:
      switch (dt.bits) {
        case 8:
          return mx::int8;
        case 16:
          return mx::int16;
        case 32:
          return mx::int32;
        case 64:
          return mx::int64;
      }
      break;
    case Code::UInt:
      switch (dt.bits) {
        case 8:
          return mx::uint8;
        case 16:
          return mx::uint16;
        case 32:
          return mx::uint32;
        case 64:
          return mx::uint64;
      }
      break;
    case Code::Float:
      switch (dt.bits) {
        case 16:
          return mx::float16;
        case 32:
          return mx::float32;
        case 64:
          return mx::float64;
      }
      break;
    case Code::Bfloat:
      if (dt.bits == 16)
        return mx::bfloat16;
      break;
    case Code::Complex:
      if (dt.bits == 64)
        return mx::complex64;
      break;
    default:
      break;
  }
  std::ostringstream msg;
  msg << "[from_dlpack] Unsupported DLPack dtype: code=" << int(dt.code)
      << ", bits=" << int(dt.bits) << ".";
  throw std::invalid_argument(msg.str());
}

bool is_row_contiguous(
    int32_t ndim,
    const int64_t* shape,
    const int64_t* strides) {
  if (strides == nullptr) {
    return true;
  }
  int64_t expected = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    if (strides[i] != expected) {
      return false;
    }
    expected *= shape[i];
  }
  return true;
}

struct ParsedCapsule {
  PyObject* capsule = nullptr;
  bool versioned = false;
  nb::dlpack::dltensor* tensor = nullptr;
  void* managed = nullptr; // typed by `versioned`
};

ParsedCapsule parse_capsule(PyObject* obj) {
  ParsedCapsule out;
  if (PyCapsule_IsValid(obj, "dltensor_versioned")) {
    out.versioned = true;
    auto* m = static_cast<dlpack_format::DLManagedTensorVersioned*>(
        PyCapsule_GetPointer(obj, "dltensor_versioned"));
    if (m == nullptr) {
      throw std::invalid_argument(
          "[from_dlpack] dltensor_versioned capsule is null.");
    }
    out.managed = static_cast<void*>(m);
    out.tensor = &m->dl_tensor;
  } else if (PyCapsule_IsValid(obj, "dltensor")) {
    out.versioned = false;
    auto* m = static_cast<dlpack_format::DLManagedTensor*>(
        PyCapsule_GetPointer(obj, "dltensor"));
    if (m == nullptr) {
      throw std::invalid_argument("[from_dlpack] dltensor capsule is null.");
    }
    out.managed = static_cast<void*>(m);
    out.tensor = &m->dl_tensor;
  } else {
    throw std::invalid_argument(
        "[from_dlpack] expected a PyCapsule named 'dltensor' or "
        "'dltensor_versioned'.");
  }
  out.capsule = obj;
  return out;
}

void rename_capsule_after_take(PyObject* capsule, bool versioned) {
  const char* used =
      versioned ? "used_dltensor_versioned" : "used_dltensor";
  PyCapsule_SetName(capsule, used);
  PyCapsule_SetDestructor(capsule, nullptr);
}

mx::Shape extract_shape(const nb::dlpack::dltensor& t) {
  mx::Shape shape;
  shape.reserve(t.ndim);
  for (int i = 0; i < t.ndim; ++i) {
    if (t.shape[i] > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument(
          "[from_dlpack] shape dim exceeds int32 range.");
    }
    shape.push_back(static_cast<int32_t>(t.shape[i]));
  }
  return shape;
}

mx::array build_cpu_array(
    nb::dlpack::dltensor& t,
    std::shared_ptr<DLPackOwner> owner) {
  if (!is_row_contiguous(t.ndim, t.shape, t.strides)) {
    throw std::invalid_argument(
        "[from_dlpack] non-row-contiguous DLPack strides are not supported "
        "for kDLCPU tensors yet.");
  }
  if (t.byte_offset != 0) {
    throw std::invalid_argument(
        "[from_dlpack] kDLCPU capsule with non-zero byte_offset is not "
        "supported yet.");
  }
  auto shape = extract_shape(t);
  auto dtype = dlpack_to_mlx_dtype(t.dtype);

  size_t nelems = 1;
  for (int i = 0; i < t.ndim; ++i)
    nelems *= static_cast<size_t>(t.shape[i]);
  size_t nbytes = nelems * dtype.size();

  // Allocate a fresh mlx buffer and copy the producer's bytes in. This
  // mirrors the semantics of nd_array_to_mlx_contiguous for the kDLCPU
  // path. We use the (allocator::Buffer, Shape, Dtype, Deleter) overload to
  // get an array whose status() == Status::available immediately.
  auto buffer = mx::allocator::malloc(nbytes);
  std::memcpy(
      static_cast<uint8_t*>(buffer.raw_ptr()),
      t.data,
      nbytes);
  mx::array out(buffer, std::move(shape), dtype, mx::allocator::free);

  // Done with the producer; bytes are copied.
  owner->invoke();
  return out;
}

} // namespace

void DLPackOwner::invoke() {
  if (mt_ == nullptr)
    return;
  if (versioned_) {
    auto* m = static_cast<dlpack_format::DLManagedTensorVersioned*>(mt_);
    if (m->deleter)
      m->deleter(m);
  } else {
    auto* m = static_cast<dlpack_format::DLManagedTensor*>(mt_);
    if (m->deleter)
      m->deleter(m);
  }
  mt_ = nullptr;
}

mx::array dlpack_to_mlx(nb::object obj) {
  // Accept either:
  //   * a PyCapsule (raw DLPack output),
  //   * an object that returns a PyCapsule from __dlpack__(),
  //   * an object whose __dlpack__() returns *another* object that is itself
  //     PEP-3118 / DLPack-compliant (e.g. nanobind's nb_ndarray wrapper that
  //     mlx returns from mx.array.__dlpack__). We unwrap up to N times.
  constexpr int kMaxUnwrap = 4;
  PyObject* raw = obj.ptr();
  nb::object current = obj; // own a reference for the chain

  for (int i = 0; i < kMaxUnwrap; ++i) {
    if (PyCapsule_CheckExact(raw)) {
      break;
    }
    if (!nb::hasattr(current, "__dlpack__")) {
      throw std::invalid_argument(
          "[from_dlpack] expected a PyCapsule or an object exposing "
          "__dlpack__().");
    }
    current = current.attr("__dlpack__")();
    raw = current.ptr();
  }
  if (!PyCapsule_CheckExact(raw)) {
    throw std::invalid_argument(
        "[from_dlpack] could not resolve input to a DLPack PyCapsule "
        "after repeated __dlpack__() calls.");
  }

  ParsedCapsule p = parse_capsule(raw);
  auto owner = std::make_shared<DLPackOwner>(p.versioned, p.managed);
  rename_capsule_after_take(p.capsule, p.versioned);

  auto& t = *p.tensor;
  switch (t.device.device_type) {
    case dlpack_format::kDLCPU:
      return build_cpu_array(t, owner);
    case dlpack_format::kDLMetal:
      return build_dlpack_metal_array(t, owner);
    case dlpack_format::kDLCUDA:
      throw std::invalid_argument(
          "[from_dlpack] kDLCUDA tensors are not supported by MLX. Move the "
          "tensor to host memory or to a Metal-backed framework first.");
    default: {
      std::ostringstream msg;
      msg << "[from_dlpack] unsupported DLPack device_type "
          << t.device.device_type << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}
