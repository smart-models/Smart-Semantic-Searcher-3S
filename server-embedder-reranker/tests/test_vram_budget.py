"""Tests for the shared VRAM budgeting arithmetic."""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import vram_budget  # noqa: E402


class DtypeSizeTests(unittest.TestCase):
    """Known precisions map to their size; unknown ones stay conservative."""

    def test_known_precisions(self):
        self.assertEqual(vram_budget.dtype_size_bytes("fp16"), 2)
        self.assertEqual(vram_budget.dtype_size_bytes("bf16"), 2)
        self.assertEqual(vram_budget.dtype_size_bytes("fp32"), 4)

    def test_unknown_precision_is_conservative(self):
        # Unknown precision must over-estimate, never under-estimate.
        self.assertEqual(vram_budget.dtype_size_bytes("weird"), 4)


class LogitsCostTests(unittest.TestCase):
    """The logits cost estimate matches the measured Qwen reranker cost."""

    def test_matches_measured_qwen_reranker_cost(self):
        # Qwen3-Reranker-0.6B: vocab_size 151669, fp16.
        per_token = vram_budget.logits_bytes_per_token(151669, 2)
        self.assertEqual(per_token, 303338)
        # 1000 tokens ~= 0.283 GB, the slope measured on the live server.
        self.assertAlmostEqual(per_token * 1000 / 1024**3, 0.283, places=3)


class HeadroomTests(unittest.TestCase):
    """Free VRAM accounting across pool, device and over-committed states."""

    def test_pool_free_counts_even_when_device_is_fully_committed(self):
        # Observed on the target box: device free is 0 but the allocator
        # holds 2.2 GB of unused reserved pool.
        headroom = vram_budget.usable_headroom_bytes(
            device_free_bytes=0,
            reserved_bytes=7_056_916_480,
            allocated_bytes=4_769_281_536,
        )
        self.assertEqual(headroom, 2_287_634_944)

    def test_device_free_is_added_to_pool_free(self):
        headroom = vram_budget.usable_headroom_bytes(
            device_free_bytes=100,
            reserved_bytes=500,
            allocated_bytes=400,
        )
        self.assertEqual(headroom, 200)

    def test_allocated_above_reserved_never_goes_negative(self):
        headroom = vram_budget.usable_headroom_bytes(
            device_free_bytes=0, reserved_bytes=100, allocated_bytes=300
        )
        self.assertEqual(headroom, 0)


class SafeMaxBatchTokensTests(unittest.TestCase):
    """The safe token cap respects the configured cap, reserve and floor."""

    def test_never_exceeds_configured_cap(self):
        value = vram_budget.safe_max_batch_tokens(
            headroom_bytes=10**12,
            reserve_bytes=0,
            bytes_per_token=1,
            configured_cap=8192,
        )
        self.assertEqual(value, 8192)

    def test_honours_the_reserve(self):
        # 1000 bytes headroom, 400 reserved, 100 bytes/token -> 6 tokens.
        value = vram_budget.safe_max_batch_tokens(
            headroom_bytes=1000,
            reserve_bytes=400,
            bytes_per_token=100,
            configured_cap=8192,
        )
        self.assertEqual(value, 6)

    def test_floors_at_one_when_headroom_is_exhausted(self):
        value = vram_budget.safe_max_batch_tokens(
            headroom_bytes=0,
            reserve_bytes=512,
            bytes_per_token=303338,
            configured_cap=8192,
        )
        self.assertEqual(value, 1)

    def test_zero_bytes_per_token_falls_back_to_configured_cap(self):
        value = vram_budget.safe_max_batch_tokens(
            headroom_bytes=1000,
            reserve_bytes=0,
            bytes_per_token=0,
            configured_cap=4096,
        )
        self.assertEqual(value, 4096)


import gc  # noqa: E402
import types  # noqa: E402
import weakref  # noqa: E402


class FakeOutOfMemoryError(RuntimeError):
    """Stand-in for torch's OutOfMemoryError, without importing torch."""

    pass


def _fake_torch(record):
    return types.SimpleNamespace(
        OutOfMemoryError=FakeOutOfMemoryError,
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: record.append("empty_cache"),
        ),
    )


class ExhaustionDetectionTests(unittest.TestCase):
    """Which errors count as VRAM exhaustion, and which must propagate."""

    def setUp(self):
        self.torch = _fake_torch([])

    def test_detects_torch_oom_type(self):
        exc = FakeOutOfMemoryError("CUDA out of memory")
        self.assertTrue(vram_budget.is_vram_exhaustion(exc, self.torch))

    def test_detects_expandable_segments_assert(self):
        exc = RuntimeError(
            "!handles_.at(i) INTERNAL ASSERT FAILED at "
            '"/pytorch/c10/cuda/CUDACachingAllocator.cpp":467'
        )
        self.assertTrue(vram_budget.is_vram_exhaustion(exc, self.torch))

    def test_detects_device_not_ready(self):
        exc = RuntimeError("CUDA driver error: device not ready")
        self.assertTrue(vram_budget.is_vram_exhaustion(exc, self.torch))

    def test_unrelated_runtime_error_is_not_exhaustion(self):
        # The load-bearing negative test: substring matching must not
        # swallow real bugs.
        exc = RuntimeError("shape mismatch in attention mask")
        self.assertFalse(vram_budget.is_vram_exhaustion(exc, self.torch))

    def test_unrelated_exception_type_is_not_exhaustion(self):
        self.assertFalse(vram_budget.is_vram_exhaustion(ValueError("x"), self.torch))


class RunGuardedTests(unittest.TestCase):
    """The guard converts exhaustion to a typed failure and frees memory."""

    def test_returns_the_operation_result_when_nothing_fails(self):
        result = vram_budget.run_guarded(
            lambda: [0.5, 0.25],
            batch_items=2,
            batch_tokens=100,
            bytes_per_token=4,
            torch_module=_fake_torch([]),
        )
        self.assertEqual(result, [0.5, 0.25])

    def test_unrelated_error_propagates_unchanged(self):
        def boom():
            raise RuntimeError("shape mismatch in attention mask")

        with self.assertRaises(RuntimeError) as ctx:
            vram_budget.run_guarded(
                boom,
                batch_items=1,
                batch_tokens=1,
                bytes_per_token=1,
                torch_module=_fake_torch([]),
            )
        self.assertNotIsInstance(ctx.exception, vram_budget.VramExhausted)

    def test_exhaustion_becomes_typed_failure_carrying_numbers(self):
        def boom():
            raise FakeOutOfMemoryError("CUDA out of memory")

        with self.assertRaises(vram_budget.VramExhausted) as ctx:
            vram_budget.run_guarded(
                boom,
                batch_items=8,
                batch_tokens=4240,
                bytes_per_token=303338,
                torch_module=_fake_torch([]),
            )
        failure = ctx.exception
        self.assertEqual(failure.batch_items, 8)
        self.assertEqual(failure.batch_tokens, 4240)
        self.assertEqual(failure.estimated_bytes, 4240 * 303338)
        self.assertIn("CUDA out of memory", failure.original)

    def test_empty_cache_is_called_only_on_failure(self):
        record = []
        vram_budget.run_guarded(
            lambda: None,
            batch_items=1,
            batch_tokens=1,
            bytes_per_token=1,
            torch_module=_fake_torch(record),
        )
        self.assertEqual(record, [])

        def boom():
            raise FakeOutOfMemoryError("CUDA out of memory")

        with self.assertRaises(vram_budget.VramExhausted):
            vram_budget.run_guarded(
                boom,
                batch_items=1,
                batch_tokens=1,
                bytes_per_token=1,
                torch_module=_fake_torch(record),
            )
        self.assertEqual(record, ["empty_cache"])

    def test_failure_detaches_the_original_exception(self):
        # Guards the mechanism itself: a surviving __context__ keeps the
        # dead batch's frames - and its GPU tensors - reachable.
        def boom():
            raise FakeOutOfMemoryError("CUDA out of memory")

        with self.assertRaises(vram_budget.VramExhausted) as ctx:
            vram_budget.run_guarded(
                boom,
                batch_items=1,
                batch_tokens=1,
                bytes_per_token=1,
                torch_module=_fake_torch([]),
            )
        self.assertIsNone(ctx.exception.__context__)
        self.assertIsNone(ctx.exception.__cause__)

    def test_operation_locals_are_released_while_the_failure_is_live(self):
        tracker = {}

        class Batch:
            """Placeholder object standing in for a batch the guard must release."""

            pass

        def boom():
            batch = Batch()
            tracker["ref"] = weakref.ref(batch)
            raise FakeOutOfMemoryError("CUDA out of memory")

        try:
            vram_budget.run_guarded(
                boom,
                batch_items=1,
                batch_tokens=1,
                bytes_per_token=1,
                torch_module=_fake_torch([]),
            )
        except vram_budget.VramExhausted:
            gc.collect()
            self.assertIsNone(
                tracker["ref"](),
                "batch tensors must not stay reachable through the failure",
            )
        else:
            self.fail("expected VramExhausted")


if __name__ == "__main__":
    unittest.main()
