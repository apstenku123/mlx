// Copyright © 2026 Apple Inc.

#include <stdexcept>

#include "python/src/dlpack_consumer.h"

mx::array build_dlpack_metal_array(
    nb::dlpack::dltensor& /*t*/,
    std::shared_ptr<DLPackOwner> owner) {
  // Drive the producer's deleter so we don't leak its allocation.
  if (owner)
    owner->invoke();
  throw std::invalid_argument(
      "[from_dlpack] MLX was built without Metal support; cannot consume "
      "kDLMetal capsules.");
}
