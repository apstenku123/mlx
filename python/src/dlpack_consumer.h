// Copyright © 2026 Apple Inc.
#pragma once

#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "mlx/array.h"

namespace mx = mlx::core;
namespace nb = nanobind;

// Convert a DLPack capsule (or a Python object exposing __dlpack__) into an
// mx::array.
//
// Supported device types:
//   * kDLCPU (1)        : copies host bytes into a fresh mlx allocation.
//   * kDLMetal (8)      : zero-copy; wraps a foreign MTL::Buffer in shared
//                         storage mode. Non-shared buffers are rejected.
//
// All other device types raise std::invalid_argument.
//
// The returned mx::array takes ownership of the capsule's deleter. Once the
// array (and any aliases) are destroyed, the capsule's deleter is invoked
// exactly once.
mx::array dlpack_to_mlx(nb::object obj);

// A small reference-counted holder that drives the DLPack capsule's deleter
// exactly once. Kept here so the metal-glue translation unit can reach it.
class DLPackOwner {
 public:
  DLPackOwner(bool versioned, void* mt) : versioned_(versioned), mt_(mt) {}

  ~DLPackOwner() {
    invoke();
  }

  void invoke();

  // Disable copies: only one DLPackOwner may exist per managed tensor.
  DLPackOwner(const DLPackOwner&) = delete;
  DLPackOwner& operator=(const DLPackOwner&) = delete;

 private:
  bool versioned_;
  void* mt_;
};

// Build an mx::array from a DLPack tensor whose data is a foreign MTL::Buffer.
// Defined in dlpack_consumer_metal.cpp when MLX_BUILD_METAL is on, and in
// dlpack_consumer_no_metal.cpp otherwise.
mx::array build_dlpack_metal_array(
    nb::dlpack::dltensor& t,
    std::shared_ptr<DLPackOwner> owner);
