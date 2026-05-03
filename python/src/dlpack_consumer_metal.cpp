// Copyright © 2026 Apple Inc.

#include <Metal/Metal.hpp>

#include <memory>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/dtype.h"
#include "python/src/dlpack_consumer.h"
#include "python/src/dlpack_format.h"

namespace {

mx::Dtype dlpack_to_mlx_dtype(const nb::dlpack::dtype& dt) {
  using Code = nb::dlpack::dtype_code;
  if (dt.lanes != 1) {
    throw std::invalid_argument(
        "[from_dlpack] DLPack tensors with lanes != 1 are not supported.");
  }
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

} // namespace

mx::array build_dlpack_metal_array(
    nb::dlpack::dltensor& t,
    std::shared_ptr<DLPackOwner> owner) {
  if (!mx::metal::is_available()) {
    throw std::invalid_argument(
        "[from_dlpack] Metal device tensors require an MLX build with Metal "
        "support enabled and a Metal-capable host.");
  }
  if (t.data == nullptr) {
    throw std::invalid_argument(
        "[from_dlpack] kDLMetal capsule has null MTLBuffer pointer.");
  }
  // For kDLMetal, DLPack stipulates `data` is an MTL::Buffer*.
  auto mtl_buffer = static_cast<MTL::Buffer*>(t.data);
  if (mtl_buffer->storageMode() != MTL::StorageModeShared) {
    throw std::invalid_argument(
        "[from_dlpack] foreign MTLBuffer must use MTLStorageModeShared. MLX "
        "currently relies on shared-mode buffers for read/write access. "
        "Allocate the producer-side buffer with MTLResourceStorageModeShared "
        "before exporting via DLPack.");
  }
  if (t.byte_offset != 0) {
    throw std::invalid_argument(
        "[from_dlpack] kDLMetal capsule with non-zero byte_offset is not "
        "supported yet.");
  }
  if (!is_row_contiguous(t.ndim, t.shape, t.strides)) {
    throw std::invalid_argument(
        "[from_dlpack] non-row-contiguous DLPack strides are not supported. "
        "Reshape on the producer side before exporting.");
  }

  auto shape = extract_shape(t);
  auto dtype = dlpack_to_mlx_dtype(t.dtype);

  // Wrap the foreign MTL::Buffer* directly. The producer retains the
  // underlying allocation; we drive the capsule's deleter when the wrapping
  // mx::array (and any aliases) are destroyed.
  mx::allocator::Buffer wrapped(static_cast<void*>(mtl_buffer));
  mx::Deleter deleter = [owner](mx::allocator::Buffer) mutable {
    // Drop our shared_ptr; if this was the last reference, the owner's
    // destructor invokes the DLPack deleter.
    owner.reset();
  };

  return mx::array(wrapped, std::move(shape), dtype, std::move(deleter));
}
