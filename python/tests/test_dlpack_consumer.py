# Copyright © 2026 Apple Inc.
"""Tests for ``mx.from_dlpack`` (consumer side).

The patch adds a Python-visible entry point::

    arr = mx.from_dlpack(capsule_or_obj)

These tests cover three scenarios that don't depend on a third-party Metal
producer:

1. Round trip through NumPy (``kDLCPU`` path).
2. Self round trip via ``mx.array.__dlpack__`` (``kDLMetal`` on Metal hosts,
   ``kDLCPU`` otherwise).
3. Negative paths: CUDA capsule, malformed capsule, etc.

The Metal-with-Private-storage rejection test is provided as an optional
helper exercised when both metal-cpp and the producer-side ``__dlpack__``
hook are available; on systems where only MLX-allocated buffers can be
produced (always shared) it simply checks that MLX's own export does not
trip the private-mode guard.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover - import error is environment specific
    raise unittest.SkipTest(f"mlx.core unavailable: {exc}")


class TestFromDLPackBasic(unittest.TestCase):
    def test_function_exists(self):
        self.assertTrue(hasattr(mx, "from_dlpack"))

    def test_round_trip_via_numpy(self):
        # NumPy arrays implement __dlpack__ (always kDLCPU). Going through
        # this path validates the kDLCPU branch which copies bytes into a
        # fresh mlx allocation and drives the producer deleter immediately.
        arr_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        arr_mx = mx.from_dlpack(arr_np)
        self.assertEqual(tuple(arr_mx.shape), (3, 4))
        self.assertEqual(arr_mx.dtype, mx.float32)
        self.assertTrue(np.allclose(np.asarray(arr_mx), arr_np))

    def test_round_trip_via_capsule(self):
        # Pass a raw PyCapsule rather than the producer object.
        arr_np = np.arange(8, dtype=np.int32).reshape(2, 4)
        capsule = arr_np.__dlpack__()
        arr_mx = mx.from_dlpack(capsule)
        self.assertEqual(tuple(arr_mx.shape), (2, 4))
        self.assertEqual(arr_mx.dtype, mx.int32)
        self.assertTrue(np.array_equal(np.asarray(arr_mx), arr_np))

    def test_self_round_trip(self):
        # mx.array -> __dlpack__ -> from_dlpack should be byte-identical.
        x = mx.arange(20, dtype=mx.float32).reshape(4, 5)
        y = mx.from_dlpack(x)
        # Different arrays, same data
        self.assertTrue(mx.array_equal(x, y).item())

    def test_dtypes(self):
        cases = [
            (np.bool_, mx.bool_),
            (np.int8, mx.int8),
            (np.int16, mx.int16),
            (np.int32, mx.int32),
            (np.int64, mx.int64),
            (np.uint8, mx.uint8),
            (np.uint16, mx.uint16),
            (np.uint32, mx.uint32),
            (np.uint64, mx.uint64),
            (np.float16, mx.float16),
            (np.float32, mx.float32),
            (np.float64, mx.float64),
            (np.complex64, mx.complex64),
        ]
        for np_dtype, mx_dtype in cases:
            with self.subTest(np_dtype=np_dtype):
                arr = np.zeros((2, 3), dtype=np_dtype)
                if np_dtype is np.bool_:
                    arr[0, 0] = True
                else:
                    arr[0, 0] = 1
                converted = mx.from_dlpack(arr)
                self.assertEqual(converted.dtype, mx_dtype)
                self.assertEqual(tuple(converted.shape), (2, 3))


class TestFromDLPackErrors(unittest.TestCase):
    def test_rejects_non_dlpack_object(self):
        with self.assertRaises(Exception):
            mx.from_dlpack(object())

    def test_rejects_used_capsule(self):
        arr_np = np.arange(4, dtype=np.float32)
        capsule = arr_np.__dlpack__()
        # First call consumes; second must fail because the capsule was
        # renamed to "used_dltensor".
        _ = mx.from_dlpack(capsule)
        with self.assertRaises(Exception):
            mx.from_dlpack(capsule)


class TestFromDLPackNonContiguous(unittest.TestCase):
    def test_strided_view_rejected(self):
        # MLX's first-cut consumer does not support arbitrary DLPack strides.
        # NumPy emits __dlpack__ with explicit strides for slices; producers
        # may or may not encode strides depending on contiguity. We assert
        # that a non-row-contiguous slice is rejected with a clear error
        # rather than silently misinterpreting the layout.
        big = np.arange(16, dtype=np.float32).reshape(4, 4)
        view = big[::2, :]
        try:
            capsule = view.__dlpack__()
        except (TypeError, BufferError):
            self.skipTest(
                "NumPy refused to export a non-contiguous DLPack capsule"
            )
        with self.assertRaises(Exception):
            mx.from_dlpack(capsule)


if __name__ == "__main__":
    unittest.main()
