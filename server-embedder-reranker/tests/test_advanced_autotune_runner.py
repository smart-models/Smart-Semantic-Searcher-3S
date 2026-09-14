"""Tests for the advanced autotuner's search space, gates and calibration.

Drives ``advanced_autotune_runner`` over simulated measurements so the
candidate ordering, the stability/memory gates and the final calibration
are exercised without loading a model.
"""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import advanced_autotune as schema  # noqa: E402
import advanced_autotune_quality as quality  # noqa: E402
import advanced_autotune_runner as runner  # noqa: E402
from basic_autotune import IsolatedResult  # noqa: E402


class DeferredCanaryBudgetTests(unittest.TestCase):
    """Long probes must leave time for the final quality capture."""

    def test_reservation_covers_probes_and_quality_within_existing_limits(self):
        baseline = {"measurements": [{"p50_seconds": 50.0}]}
        reserved = runner.estimate_deferred_canary_seconds(
            baseline, baseline_seconds=200.0, repeats=2, max_seconds=1800.0
        )
        # The old 150-second reservation could not finish this 240-second run.
        self.assertGreaterEqual(reserved, 180.0 + 60.0)
        self.assertLessEqual(reserved, 720.0)
        self.assertEqual(
            runner.estimate_deferred_canary_seconds(
                baseline, baseline_seconds=200.0, repeats=2, max_seconds=300.0
            ),
            120.0,
        )
        self.assertEqual(
            runner.estimate_deferred_canary_seconds(
                baseline, baseline_seconds=1000.0, repeats=2, max_seconds=1800.0
            ),
            720.0,
        )


def fingerprint(device: str = "cuda", *, memory_gib: int = 8) -> dict:
    gpus = []
    if device == "cuda":
        gpus = [
            {
                "name": "RTX Test",
                "uuid": "GPU-test",
                "compute_capability": "8.9",
                "total_memory_bytes": memory_gib * 1024**3,
                "multi_processor_count": 24,
            }
        ]
    return {
        "schema_version": 1,
        "hardware": {
            "gpus": gpus,
            "cpu": {
                "effective_cpu_count": 8,
                "instruction_sets": ["avx2", "avx512_bf16"],
            },
            "memory_limit_bytes": 16 * 1024**3,
        },
        "software": {"flash_attn": None, "openvino": None},
        "models": {
            name: {"model_id": model_id, "revision": "rev", "cached": True}
            for name, model_id in schema.ADVANCED_MODEL_IDS.items()
        },
    }


def isolated_result(
    *,
    payload=None,
    returncode=0,
    timed_out=False,
    memory_exceeded=False,
    duration_seconds=2.0,
) -> IsolatedResult:
    return IsolatedResult(
        command=("python", "worker.py"),
        returncode=returncode,
        duration_seconds=duration_seconds,
        timed_out=timed_out,
        memory_exceeded=memory_exceeded,
        peak_rss_bytes=1024,
        stdout="",
        stderr="",
        payload=payload,
    )


class FakeClock:
    """Monotonic clock that a fake isolated runner advances by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class SearchSpaceTests(unittest.TestCase):
    """How the candidate search space scales with the detected hardware."""

    def test_larger_gpu_expands_batch_and_bf16_is_capability_gated(self):
        laptop = runner.advanced_search_space(fingerprint(memory_gib=8), "quick")
        server = runner.advanced_search_space(fingerprint(memory_gib=24), "quick")

        self.assertIn("bf16", laptop["dense"]["precision"])
        self.assertEqual(max(laptop["dense"]["batch"]), 8)
        self.assertEqual(max(server["dense"]["batch"]), 32)
        self.assertEqual(laptop["dense"]["torch_compile"], ["off"])

    def test_full_cpu_exposes_runtime_candidates_and_effective_threads(self):
        space = runner.advanced_search_space(fingerprint("cpu"), "full")

        self.assertEqual(space["device"], "cpu")
        self.assertEqual(space["dense"]["runtime"], ["torch"])
        self.assertFalse(
            space["dense"]["runtime_availability"]["onnx-int8"]["available"]
        )
        self.assertEqual(max(space["dense"]["threads"]), 8)
        self.assertIn("bf16", space["dense"]["precision"])

        config = schema.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models", "SMART_EMBEDDER_AUTOTUNE": "full"},
            device_kind="cpu",
        )
        specs = runner.tuning_candidate_specs(fingerprint("cpu"), config)
        names = {item["name"] for item in specs}
        self.assertIn("cpu-intra-op-1", names)
        self.assertIn("cpu-intra-op-2", names)
        self.assertIn("cpu-inter-op-2", names)
        self.assertIn("cpu-worker-spin-passive", names)
        self.assertIn("cpu-worker-spin-active", names)
        self.assertIn("cpu-affinity-compact", names)
        self.assertIn("cpu-dense-bf16", names)
        self.assertIn("cpu-dense-onnx-fp32", names)
        onnx = next(item for item in specs if item["name"] == "cpu-dense-onnx-fp32")
        self.assertIn("missing CPU export/runtime packages", onnx["unavailable_reason"])
        parallel = next(item for item in specs if item["name"] == "hybrid-cpu-parallel")
        schedule = parallel["changes"]["schedule"]
        self.assertLessEqual(
            schedule["intra_op_threads"] * schedule["inference_lanes"], 8
        )

        topology_fingerprint = fingerprint("cpu")
        topology_fingerprint["hardware"]["cpu"].update(
            {
                "socket_count": 2,
                "numa_node_count": 2,
                "smt_active": True,
            }
        )
        topology_specs = runner.tuning_candidate_specs(topology_fingerprint, config)
        topology_names = {item["name"] for item in topology_specs}
        self.assertIn("cpu-affinity-socket-local", topology_names)
        self.assertIn("cpu-intra-op-16-oversubscribed", topology_names)

    def test_full_cpu_enables_real_onnx_candidates_when_stack_is_present(self):
        current = fingerprint("cpu")
        current["software"].update(
            {
                "onnx": "1.22.0",
                "onnxscript": "0.7.1",
                "onnxruntime": "1.27.0",
            }
        )

        space = runner.advanced_search_space(current, "full")

        self.assertEqual(
            space["dense"]["runtime"],
            ["torch", "onnx-fp32", "onnx-int8"],
        )
        self.assertTrue(
            space["dense"]["runtime_availability"]["onnx-fp32"]["available"]
        )
        json.dumps(space, allow_nan=False)
        config = schema.AdvancedAutotuneConfig.from_env(
            {"SMART_EMBEDDER_AUTOTUNE": "full"}, device_kind="cpu"
        )
        specs = runner.tuning_candidate_specs(current, config)
        onnx_specs = [
            item for item in specs if item["name"].startswith("cpu-dense-onnx")
        ]
        self.assertEqual(len(onnx_specs), 2)
        self.assertTrue(all("unavailable_reason" not in item for item in onnx_specs))
        names = [item["name"] for item in specs]
        self.assertEqual(
            names[:3],
            [
                "cpu-dense-onnx-fp32",
                "cpu-dense-onnx-int8",
                "cpu-dense-openvino",
            ],
        )
        self.assertLess(
            names.index("cpu-dense-onnx-int8"),
            names.index("dense-geometric-batch-2"),
        )
        sparse_onnx = next(
            item for item in specs if item["name"] == "cpu-sparse-onnx-fp32"
        )
        self.assertIn("LexEcho", sparse_onnx["unavailable_reason"])
        self.assertTrue(all(item["defer_long_canary"] for item in specs))

    def test_cpu_full_tuned_probe_defers_long_canary(self):
        config = schema.AdvancedAutotuneConfig.from_env(
            {"SMART_EMBEDDER_AUTOTUNE": "full"}, device_kind="cpu"
        )
        baseline = runner.baseline_candidate(fingerprint("cpu"), config)
        spec = {
            "name": "runtime",
            "target_backends": ["dense"],
            "changes": {"dense": {"runtime": "onnx-fp32"}},
            "defer_long_canary": True,
        }

        tuned = runner.materialise_candidate(baseline, baseline["selected"], spec)
        final = runner.final_long_canary_candidate(
            baseline,
            baseline_selected=baseline["selected"],
            selected=tuned["selected"],
        )

        self.assertEqual(tuned["token_lengths"], [128, 512])
        self.assertEqual(tuned["deferred_canary_token_lengths"], [1024, 2048, 8192])
        self.assertNotIn("canary_token_lengths", tuned)
        self.assertEqual(final["token_lengths"], [1024, 2048, 8192])
        self.assertEqual(final["target_backends"], ["dense"])

    def test_cpu_canary_respects_each_active_backend_max_length(self):
        current = fingerprint("cpu")
        current["limits_and_policy"] = {
            "explicit_env": {
                "QWEN_EMBED_MAX_LENGTH": "2048",
                "MILCO_MAX_LENGTH": "8192",
                "QWEN_RERANK_MAX_LENGTH": "2048",
            }
        }
        config = schema.AdvancedAutotuneConfig.from_env(
            {"SMART_EMBEDDER_AUTOTUNE": "full"}, device_kind="cpu"
        )
        baseline = runner.baseline_candidate(current, config)
        tuned = json.loads(json.dumps(baseline["selected"]))
        tuned["advanced"]["dense"]["runtime"] = "onnx-fp32"

        candidate = runner.final_long_canary_candidate(
            baseline,
            baseline_selected=baseline["selected"],
            selected=tuned,
        )

        self.assertEqual(
            baseline["deferred_canary_token_lengths_by_backend"],
            {
                "dense": [1024, 2048],
                "sparse": [1024, 2048, 8192],
                "reranker": [1024, 2048],
            },
        )
        self.assertEqual(candidate["token_lengths"], [1024, 2048])
        self.assertEqual(candidate["backend_token_lengths"], {"dense": [1024, 2048]})

    def test_quick_tuning_specs_cover_each_backend_and_token_aware_policy(self):
        config = schema.AdvancedAutotuneConfig.from_env({"HF_HOME": "/models"})
        specs = runner.tuning_candidate_specs(fingerprint(), config)
        names = {item["name"] for item in specs}

        self.assertIn("dense-bf16-sdpa", names)
        self.assertIn("sparse-bf16-weights", names)
        self.assertIn("reranker-bf16-sdpa", names)
        self.assertIn("dense-geometric-batch-policy", names)
        self.assertIn("sparse-geometric-batch-policy", names)
        self.assertIn("reranker-geometric-batch-policy", names)
        self.assertEqual(specs[0]["name"], "dense-geometric-batch-policy")
        self.assertEqual(specs[1]["name"], "hybrid-cuda-overlap")
        self.assertEqual(specs[2]["name"], "sparse-geometric-batch-policy")
        self.assertEqual(specs[3]["name"], "reranker-geometric-batch-policy")
        batch_spec = next(
            item for item in specs if item["name"] == "dense-geometric-batch-policy"
        )
        dense = batch_spec["changes"]["dense"]
        self.assertEqual(
            dense["batch_policy"][0], {"max_tokens": 128, "batch_size": 64}
        )
        self.assertEqual(dense["batch_policy"][-1]["max_tokens"], 32768)
        self.assertEqual(
            dense["max_batch_tokens_policy"][-1]["max_batch_tokens"], 32768
        )

    def test_full_batch_search_grows_one_power_of_two_per_candidate(self):
        config = schema.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models", "SMART_EMBEDDER_AUTOTUNE": "full"},
            device_kind="cuda",
        )
        specs = runner.tuning_candidate_specs(fingerprint(memory_gib=8), config)

        self.assertEqual(
            [item["name"] for item in specs[:8]],
            [
                "dense-geometric-batch-16",
                "dense-geometric-batch-32",
                "dense-geometric-batch-64",
                "hybrid-cuda-overlap",
                "sparse-geometric-batch-16",
                "sparse-geometric-batch-32",
                "sparse-geometric-batch-64",
                "reranker-geometric-batch-32",
            ],
        )
        dense = specs[:3]
        self.assertEqual([item["search_value"] for item in dense], [16, 32, 64])
        self.assertTrue(all(item["search_family"] == "dense-batch" for item in dense))
        self.assertTrue(all(item["stop_family_on_rejection"] for item in dense))
        self.assertEqual(
            dense[0]["promotion_scope"],
            {"dense": [128, 512], "hybrid": [128, 512]},
        )
        self.assertEqual(
            dense[-1]["changes"]["dense"]["batch_policy"][0],
            {"max_tokens": 128, "batch_size": 64},
        )

        large = runner.tuning_candidate_specs(fingerprint(memory_gib=24), config)
        self.assertEqual(
            [item["search_value"] for item in large[:3]],
            [64, 128, 256],
        )

    def test_full_runs_pinned_candidates_before_expensive_compile(self):
        config = schema.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models", "SMART_EMBEDDER_AUTOTUNE": "full"},
            device_kind="cuda",
        )
        names = [
            item["name"]
            for item in runner.tuning_candidate_specs(fingerprint(), config)
        ]

        self.assertLess(
            names.index("dense-pinned-nonblocking"),
            names.index("dense-compile-default"),
        )
        self.assertLess(
            names.index("sparse-pinned-nonblocking"),
            names.index("sparse-compile-default"),
        )
        self.assertLess(
            names.index("dense-double-buffer"),
            names.index("dense-compile-default"),
        )
        self.assertLess(
            names.index("sparse-double-buffer"),
            names.index("sparse-compile-default"),
        )

    def test_baseline_uses_same_geometric_external_workload_for_safe_splitting(self):
        config = schema.AdvancedAutotuneConfig.from_env({"HF_HOME": "/models"})
        candidate = runner.baseline_candidate(fingerprint(), config)

        self.assertEqual(candidate["benchmark_batches"]["dense"]["128"], 64)
        self.assertEqual(candidate["benchmark_batches"]["dense"]["8192"], 1)
        self.assertEqual(candidate["benchmark_batches"]["reranker"]["128"], 32)

        cpu = runner.baseline_candidate(fingerprint("cpu"), config)
        self.assertEqual(cpu["token_lengths"], [128, 512])
        self.assertEqual(cpu["repeats"], 1)
        self.assertEqual(cpu["deferred_canary_token_lengths"], [1024, 2048, 8192])
        self.assertNotIn("canary_token_lengths", cpu)

        full_config = schema.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models", "SMART_EMBEDDER_AUTOTUNE": "full"},
            device_kind="cpu",
        )
        cpu_full = runner.baseline_candidate(fingerprint("cpu"), full_config)
        self.assertEqual(cpu_full["token_lengths"], [128, 512])
        self.assertEqual(cpu_full["repeats"], 2)
        self.assertEqual(cpu_full["deferred_canary_token_lengths"], [1024, 2048, 8192])

    def test_gpu_baseline_includes_bounded_mixed_workload(self):
        config = schema.AdvancedAutotuneConfig.from_env({"HF_HOME": "/models"})
        candidate = runner.baseline_candidate(fingerprint(), config)

        self.assertIn("mixed", candidate["target_backends"])
        self.assertEqual(candidate["backend_token_lengths"]["mixed"], [128])
        self.assertEqual(
            candidate["benchmark_batches"]["mixed"],
            candidate["benchmark_batches"]["hybrid"],
        )

    def test_candidate_budget_scales_with_measured_backend_scope(self):
        full = runner.estimate_candidate_seconds(
            70.0, ["dense", "sparse", "hybrid", "reranker"]
        )
        hybrid = runner.estimate_candidate_seconds(70.0, ["hybrid"])
        reranker = runner.estimate_candidate_seconds(70.0, ["reranker"])

        self.assertGreater(full, hybrid)
        self.assertGreater(hybrid, reranker)
        self.assertGreaterEqual(reranker, 20.0)

    def test_batch_token_cap_scales_with_gpu_memory(self):
        quick = schema.AdvancedAutotuneConfig.from_env({"HF_HOME": "/models"})
        full = schema.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models", "SMART_EMBEDDER_AUTOTUNE": "full"},
            device_kind="cuda",
        )

        def tokens_at_512(spec):
            policy = spec["changes"]["dense"]["max_batch_tokens_policy"]
            return next(
                item["max_batch_tokens"] for item in policy if item["max_tokens"] == 512
            )

        def quick_dense(memory_gib):
            specs = runner.tuning_candidate_specs(
                fingerprint(memory_gib=memory_gib), quick
            )
            return next(
                item for item in specs if item["name"] == "dense-geometric-batch-policy"
            )

        self.assertEqual(tokens_at_512(quick_dense(8)), 8192)
        self.assertEqual(tokens_at_512(quick_dense(16)), 16384)
        self.assertEqual(tokens_at_512(quick_dense(24)), 32768)
        large_full = runner.tuning_candidate_specs(fingerprint(memory_gib=24), full)
        self.assertEqual(large_full[0]["name"], "dense-geometric-batch-64")
        self.assertEqual(tokens_at_512(large_full[0]), 32768)

    def test_default_budget_grows_to_fit_the_measured_baseline_within_its_ceiling(self):
        config = schema.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models"}, device_kind="cuda"
        )
        deadline = runner.calibration_deadline_seconds

        self.assertEqual(
            deadline(config, baseline_seconds=200.0, planned_candidate_seconds=100.0),
            300.0,
        )
        self.assertEqual(
            deadline(config, baseline_seconds=10.0, planned_candidate_seconds=20.0),
            240.0,
        )
        self.assertEqual(
            deadline(config, baseline_seconds=700.0, planned_candidate_seconds=500.0),
            720.0,
        )
        fixed = replace(config, adaptive_budget=False)
        self.assertEqual(
            deadline(fixed, baseline_seconds=700.0, planned_candidate_seconds=500.0),
            240.0,
        )

    def test_gpu_wait_puts_batch_and_coalescing_candidates_first(self):
        specs = [
            {"name": "sparse-bf16-weights"},
            {"name": "dense-geometric-batch-policy"},
            {"name": "hybrid-cuda-overlap"},
            {"name": "dense-compile-reduce-overhead"},
            {"name": "schedule-queue-batch-32"},
            {"name": "reranker-geometric-batch-policy"},
        ]

        self.assertEqual(
            [spec["name"] for spec in runner.prioritise_for_gpu_wait(specs)],
            [
                "dense-geometric-batch-policy",
                "schedule-queue-batch-32",
                "reranker-geometric-batch-policy",
                "dense-compile-reduce-overhead",
                "sparse-bf16-weights",
                "hybrid-cuda-overlap",
            ],
        )

    def test_cuda_baseline_measures_single_and_concurrent_requests(self):
        config = schema.AdvancedAutotuneConfig.from_env({"HF_HOME": "/models"})
        candidate = runner.baseline_candidate(fingerprint(), config)

        self.assertIn("requests", candidate["target_backends"])
        self.assertEqual(candidate["backend_token_lengths"]["requests"], [128])
        cpu = runner.baseline_candidate(fingerprint("cpu"), config)
        self.assertNotIn("requests", cpu["target_backends"])

    def test_quick_and_full_cuda_offer_one_coalescing_candidate(self):
        for mode in ("quick", "full"):
            config = schema.AdvancedAutotuneConfig.from_env(
                {"HF_HOME": "/models", "SMART_EMBEDDER_AUTOTUNE": mode},
                device_kind="cuda",
            )
            specs = runner.tuning_candidate_specs(fingerprint(), config)
            coalescing = [s for s in specs if s["name"] == "schedule-queue-batch-32"]

            self.assertEqual(len(coalescing), 1, mode)
            self.assertEqual(coalescing[0]["target_backends"], ["requests"])
            self.assertEqual(
                coalescing[0]["changes"],
                {"schedule": {"queue_batch": 32, "flush_timeout_ms": 10}},
            )
            self.assertEqual(coalescing[0]["promotion_scope"], {"requests": [128]})
        cpu_config = schema.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models"}, device_kind="cpu"
        )
        cpu_specs = runner.tuning_candidate_specs(fingerprint("cpu"), cpu_config)
        self.assertNotIn("schedule-queue-batch-32", {s["name"] for s in cpu_specs})


class CandidateGateTests(unittest.TestCase):
    """The gates a candidate must clear before it can be promoted."""

    def _capture(self):
        items = [*quality.QUERIES, *quality.DOCUMENTS]
        dense = {item.id: [1.0] + [0.0] * 1023 for item in items}
        sparse = {item.id: {"1": 1.0, "30522": 0.5} for item in items}
        reranker = {
            query.id: {document.id: 0.5 for document in quality.DOCUMENTS}
            for query in quality.QUERIES
        }
        return {
            "suite_version": quality.QUALITY_SUITE_VERSION,
            "dense": dense,
            "sparse": sparse,
            "reranker": reranker,
            "milco_source_view": True,
            "milco_top_k": 0,
            "source_vocab_start": 30522,
        }

    def test_identical_capture_passes_and_lexecho_change_fails(self):
        reference = self._capture()
        passed = runner.evaluate_quality_capture(reference, reference)
        changed = self._capture()
        changed["milco_source_view"] = False
        rejected = runner.evaluate_quality_capture(reference, changed)

        self.assertTrue(passed["passed"])
        self.assertFalse(rejected["passed"])
        self.assertTrue(any("LexEcho" in item for item in rejected["failures"]))

    def test_promotion_requires_three_percent_and_p95_tolerance(self):
        baseline = {
            "measurements": [
                {
                    "backend": "dense",
                    "input_tokens": 128,
                    "batch_size": 8,
                    "useful_tokens_per_second": 100.0,
                    "p95_seconds": 1.0,
                }
            ]
        }
        candidate = {
            "measurements": [
                {
                    "backend": "dense",
                    "input_tokens": 128,
                    "batch_size": 8,
                    "useful_tokens_per_second": 104.0,
                    "p95_seconds": 1.02,
                    "coefficient_of_variation": 0.02,
                }
            ]
        }
        passed = runner.evaluate_promotion(
            baseline,
            candidate,
            target_backends=["dense"],
            objective="balanced",
        )
        candidate["measurements"][0]["useful_tokens_per_second"] = 102.0
        rejected = runner.evaluate_promotion(
            baseline,
            candidate,
            target_backends=["dense"],
            objective="balanced",
        )

        self.assertTrue(passed["promoted"])
        self.assertFalse(rejected["promoted"])
        self.assertIn("below 3%", rejected["reason"])

    def test_batch_promotion_uses_changed_buckets_but_checks_all_for_safety(self):
        baseline = {
            "measurements": [
                {
                    "backend": "dense",
                    "input_tokens": length,
                    "batch_size": 8,
                    "useful_tokens_per_second": 100.0,
                    "p95_seconds": 1.0,
                }
                for length in (128, 8192)
            ]
        }
        candidate = {
            "measurements": [
                {
                    "backend": "dense",
                    "input_tokens": 128,
                    "batch_size": 8,
                    "useful_tokens_per_second": 104.0,
                    "p95_seconds": 1.0,
                    "coefficient_of_variation": 0.01,
                },
                {
                    "backend": "dense",
                    "input_tokens": 8192,
                    "batch_size": 8,
                    "useful_tokens_per_second": 100.0,
                    "p95_seconds": 1.0,
                    "coefficient_of_variation": 0.01,
                    "is_canary": True,
                },
            ]
        }

        result = runner.evaluate_promotion(
            baseline,
            candidate,
            target_backends=["dense"],
            objective="balanced",
            promotion_scope={"dense": [128]},
        )

        self.assertTrue(result["promoted"])
        self.assertEqual(result["gain_measurements"], 1)
        self.assertEqual(result["safety_measurements"], 2)

    def test_long_canary_p95_regression_rejects_fast_candidate(self):
        baseline = {
            "measurements": [
                {
                    "backend": "dense",
                    "input_tokens": length,
                    "batch_size": 8,
                    "useful_tokens_per_second": 100.0,
                    "p95_seconds": 1.0,
                }
                for length in (128, 8192)
            ]
        }
        candidate = {
            "measurements": [
                {
                    "backend": "dense",
                    "input_tokens": 128,
                    "batch_size": 8,
                    "useful_tokens_per_second": 110.0,
                    "p95_seconds": 0.95,
                    "coefficient_of_variation": 0.01,
                },
                {
                    "backend": "dense",
                    "input_tokens": 8192,
                    "batch_size": 8,
                    "useful_tokens_per_second": 100.0,
                    "p95_seconds": 1.11,
                    "coefficient_of_variation": 0.01,
                    "is_canary": True,
                },
            ]
        }

        result = runner.evaluate_promotion(
            baseline,
            candidate,
            target_backends=["dense"],
            objective="balanced",
            promotion_scope={"dense": [128]},
        )

        self.assertFalse(result["promoted"])
        self.assertEqual(result["gain_measurements"], 1)
        self.assertEqual(result["safety_measurements"], 2)
        self.assertIn("P95 regression", result["reason"])

    def test_final_canary_checks_safety_without_requiring_another_gain(self):
        baseline = {
            "measurements": [
                {
                    "backend": "dense",
                    "input_tokens": 8192,
                    "batch_size": 1,
                    "is_canary": True,
                    "p95_seconds": 10.0,
                }
            ]
        }
        candidate = {
            "measurements": [
                {
                    "backend": "dense",
                    "input_tokens": 8192,
                    "batch_size": 1,
                    "is_canary": True,
                    "p95_seconds": 10.5,
                }
            ]
        }

        passed = runner.evaluate_final_canary(
            baseline,
            candidate,
            target_backends=["dense"],
            objective="balanced",
        )
        candidate["measurements"][0]["p95_seconds"] = 11.1
        rejected = runner.evaluate_final_canary(
            baseline,
            candidate,
            target_backends=["dense"],
            objective="balanced",
        )
        recovered = runner.evaluate_final_canary(
            {"measurements": []},
            candidate,
            target_backends=["dense"],
            objective="balanced",
            expected_lengths=[8192],
        )

        self.assertTrue(passed["passed"])
        self.assertFalse(rejected["passed"])
        self.assertIn("P95 regression", rejected["reason"])
        self.assertTrue(recovered["passed"])
        self.assertIsNone(recovered["max_p95_ratio"])


class CalibrationTests(unittest.TestCase):
    """End-to-end calibration: persistence, reuse, and every failure path."""

    def _config(self, path: Path, **env) -> schema.AdvancedAutotuneConfig:
        values = {
            "SMART_EMBEDDER_ADVANCED_AUTOTUNE_PROFILE": str(path),
            "SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS": "30",
            **env,
        }
        return schema.AdvancedAutotuneConfig.from_env(values)

    def test_dry_run_checks_cache_and_does_not_launch_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            config = self._config(path)

            def fail(*args, **kwargs):
                raise AssertionError("worker should not run")

            result = runner.calibrate(
                config=config,
                fingerprint=fingerprint(),
                script_dir=SERVER_DIR,
                dry_run=True,
                isolated_runner=fail,
            )

        self.assertEqual(result["action"], "dry-run")
        self.assertEqual(
            result["candidate"]["token_lengths"],
            [128, 512, 1024, 2048, 8192],
        )
        self.assertIn("search_space", result)

    def test_missing_model_cache_falls_back_before_worker(self):
        current = fingerprint()
        current["models"]["sparse"]["cached"] = False
        with tempfile.TemporaryDirectory() as temporary:
            result = runner.calibrate(
                config=self._config(Path(temporary) / "profile.json"),
                fingerprint=current,
                script_dir=SERVER_DIR,
                isolated_runner=lambda *args, **kwargs: None,
            )

        self.assertEqual(result["action"], "fallback")
        self.assertIn("omai-research/milco-650m", result["missing_models"])

    def test_success_persists_complete_profile_then_reuses_without_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            config = self._config(path)
            selected = schema.safe_advanced_defaults("cuda")
            payload = {
                "candidate": "co-resident-safe-baseline",
                "status": "passed",
                "quality_passed": True,
                "selected": selected,
                "memory": {"peak_fraction": 0.8},
                "measurements": [],
            }
            first = runner.calibrate(
                config=config,
                fingerprint=fingerprint(),
                script_dir=SERVER_DIR,
                isolated_runner=lambda *args, **kwargs: isolated_result(
                    payload=payload
                ),
            )
            loaded = schema.load_advanced_profile(path)

            def fail(*args, **kwargs):
                raise AssertionError("reused profile must skip worker")

            second = runner.calibrate(
                config=config,
                fingerprint=fingerprint(),
                script_dir=SERVER_DIR,
                isolated_runner=fail,
            )

        self.assertEqual(first["action"], "calibrated")
        self.assertEqual(loaded["selected"], selected)
        self.assertEqual(second["action"], "reused")

    def test_timeout_and_memory_kill_are_visible_and_never_persisted(self):
        for isolated, expected in (
            (isolated_result(returncode=-1, timed_out=True), "deadline"),
            (isolated_result(returncode=-1, memory_exceeded=True), "RSS"),
        ):
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path = Path(temporary) / "profile.json"
                result = runner.calibrate(
                    config=self._config(path),
                    fingerprint=fingerprint(),
                    script_dir=SERVER_DIR,
                    isolated_runner=lambda *args, value=isolated, **kwargs: value,
                )

                self.assertEqual(result["action"], "fallback")
                self.assertIn(expected, result["reason"])
                self.assertFalse(path.exists())

    def test_quality_rejection_never_promotes_candidate(self):
        rejected = isolated_result(
            returncode=3,
            payload={
                "status": "rejected",
                "quality_passed": False,
                "quality_errors": ["LexEcho source_view mismatch"],
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            result = runner.calibrate(
                config=self._config(path),
                fingerprint=fingerprint(),
                script_dir=SERVER_DIR,
                isolated_runner=lambda *args, **kwargs: rejected,
            )

        self.assertEqual(result["action"], "fallback")
        self.assertIn("LexEcho", result["reason"])
        self.assertFalse(path.exists())

    def test_candidate_that_does_not_fit_is_skipped_and_later_ones_still_run(self):
        baseline = {
            "candidate": "co-resident-safe-baseline",
            "status": "passed",
            "quality_passed": True,
            "quality_capture": CandidateGateTests()._capture(),
            "selected": schema.safe_advanced_defaults("cuda"),
            "measurements": [],
        }
        specs = [
            {
                "name": "every-backend",
                "target_backends": ["dense", "sparse", "hybrid", "reranker", "mixed"],
                "changes": {},
            },
            {"name": "reranker-only", "target_backends": ["reranker"], "changes": {}},
        ]
        launched = []

        def isolated_runner(command, **kwargs):
            launched.append(command)
            if len(launched) == 1:
                return isolated_result(payload=baseline, duration_seconds=100.0)
            return isolated_result(payload={**baseline, "candidate": "reranker-only"})

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "tuning_candidate_specs", return_value=specs),
        ):
            result = runner.calibrate(
                config=self._config(
                    Path(temporary) / "profile.json",
                    SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS="60",
                ),
                fingerprint=fingerprint(),
                script_dir=SERVER_DIR,
                isolated_runner=isolated_runner,
            )

        records = {record["candidate"]: record for record in result["results"][1:]}
        self.assertEqual(records["every-backend"]["status"], "skipped-budget")
        self.assertNotEqual(records["reranker-only"].get("status"), "skipped-budget")
        self.assertEqual(len(launched), 2)

    def test_slow_baseline_leaves_time_for_candidates_on_the_default_budget(self):
        clock = FakeClock()
        baseline = {
            "candidate": "co-resident-safe-baseline",
            "status": "passed",
            "quality_passed": True,
            "quality_capture": CandidateGateTests()._capture(),
            "selected": schema.safe_advanced_defaults("cuda"),
            "measurements": [],
        }
        specs = [
            {"name": name, "target_backends": ["dense"], "changes": {}}
            for name in ("dense-a", "dense-b")
        ]
        timeouts = []

        def isolated_runner(command, *, timeout_seconds, **kwargs):
            timeouts.append(timeout_seconds)
            seconds = 200.0 if len(timeouts) == 1 else 20.0
            clock.now += seconds
            return isolated_result(payload=baseline, duration_seconds=seconds)

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "tuning_candidate_specs", return_value=specs),
            patch.object(runner.time, "monotonic", clock),
        ):
            config = schema.AdvancedAutotuneConfig.from_env(
                {
                    "HF_HOME": "/models",
                    "SMART_EMBEDDER_ADVANCED_AUTOTUNE_PROFILE": str(
                        Path(temporary) / "profile.json"
                    ),
                },
                device_kind="cuda",
            )
            result = runner.calibrate(
                config=config,
                fingerprint=fingerprint(),
                script_dir=SERVER_DIR,
                isolated_runner=isolated_runner,
            )

        self.assertEqual(timeouts[0], 720.0)
        self.assertEqual(len(timeouts), 3)
        self.assertTrue(result["budget"]["adaptive"])
        self.assertGreater(result["budget"]["deadline_seconds"], 240.0)
        self.assertLessEqual(result["budget"]["deadline_seconds"], 720.0)

    def test_profile_records_cpu_power_and_warns_on_throttled_cores(self):
        baseline = {
            "candidate": "co-resident-safe-baseline",
            "status": "passed",
            "quality_passed": True,
            "quality_capture": CandidateGateTests()._capture(),
            "selected": schema.safe_advanced_defaults("cuda"),
            "measurements": [],
            "cpu_frequency": {"samples": 12, "median_mhz": 800, "max_mhz": 804},
        }
        policy = {
            "available": True,
            "governors": {"powersave": 8},
            "energy_performance_preferences": {"balance_performance": 8},
            "drivers": ["intel_pstate"],
            "max_frequency_mhz": 2700,
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "tuning_candidate_specs", return_value=[]),
            patch.object(runner, "read_cpu_power_policy", return_value=policy),
        ):
            path = Path(temporary) / "profile.json"
            result = runner.calibrate(
                config=self._config(path),
                fingerprint=fingerprint(),
                script_dir=SERVER_DIR,
                isolated_runner=lambda *args, **kwargs: isolated_result(
                    payload=baseline
                ),
            )
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["host_cpu_power"]["policy"], policy)
        self.assertEqual(saved["host_cpu_power"]["measured"]["median_mhz"], 800)
        self.assertEqual(len(saved["host_cpu_power"]["warnings"]), 1)
        self.assertEqual(result["warnings"], saved["host_cpu_power"]["warnings"])

    def test_low_gpu_utilisation_reorders_candidates_and_is_recorded(self):
        baseline = {
            "candidate": "co-resident-safe-baseline",
            "status": "passed",
            "quality_passed": True,
            "quality_capture": CandidateGateTests()._capture(),
            "selected": schema.safe_advanced_defaults("cuda"),
            "measurements": [],
            "gpu_utilization": {
                "samples": 30,
                "median_percent": 10.0,
                "mean_percent": 11.0,
            },
        }
        specs = [
            {"name": "dense-bf16-sdpa", "target_backends": ["dense"], "changes": {}},
            {
                "name": "dense-geometric-batch-policy",
                "target_backends": ["dense"],
                "changes": {},
            },
        ]
        order = []

        def isolated_runner(command, **kwargs):
            payload = json.loads(command[command.index("--candidate-json") + 1])
            order.append(payload.get("name"))
            return isolated_result(
                payload={**baseline, "candidate": payload.get("name")}
            )

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "tuning_candidate_specs", return_value=specs),
        ):
            path = Path(temporary) / "profile.json"
            result = runner.calibrate(
                config=self._config(path, SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS="600"),
                fingerprint=fingerprint(),
                script_dir=SERVER_DIR,
                isolated_runner=isolated_runner,
            )
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(order[1:], ["dense-geometric-batch-policy", "dense-bf16-sdpa"])
        self.assertTrue(saved["gpu_dispatch"]["gpu_waits_on_cpu"])
        self.assertTrue(any("10%" in warning for warning in result["warnings"]))

    def test_timeout_names_candidate_and_reports_all_remaining_skips(self):
        items = [*quality.QUERIES, *quality.DOCUMENTS]
        capture = {
            "suite_version": quality.QUALITY_SUITE_VERSION,
            "dense": {item.id: [1.0] + [0.0] * 1023 for item in items},
            "sparse": {item.id: {"1": 1.0, "30522": 0.5} for item in items},
            "reranker": {
                query.id: {document.id: 0.5 for document in quality.DOCUMENTS}
                for query in quality.QUERIES
            },
            "milco_source_view": True,
            "milco_top_k": 0,
            "source_vocab_start": 30522,
        }
        baseline = {
            "candidate": "co-resident-safe-baseline",
            "status": "passed",
            "quality_passed": True,
            "quality_capture": capture,
            "selected": schema.safe_advanced_defaults("cuda"),
            "measurements": [],
        }
        specs = [
            {
                "name": name,
                "target_backends": ["dense"],
                "changes": {},
            }
            for name in ("first", "second", "third")
        ]
        calls = iter(
            [
                isolated_result(payload=baseline),
                isolated_result(returncode=-15, timed_out=True),
            ]
        )

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "tuning_candidate_specs", return_value=specs),
        ):
            result = runner.calibrate(
                config=self._config(Path(temporary) / "profile.json"),
                fingerprint=fingerprint(),
                script_dir=SERVER_DIR,
                isolated_runner=lambda *args, **kwargs: next(calls),
            )

        self.assertEqual(result["action"], "calibrated")
        self.assertEqual(result["results"][1]["candidate"], "first")
        self.assertTrue(result["results"][1]["isolated_timed_out"])
        self.assertEqual(
            [record["candidate"] for record in result["results"][2:]],
            ["second", "third"],
        )
        self.assertTrue(
            all(
                record["status"] == "skipped-deadline"
                for record in result["results"][2:]
            )
        )

    def test_batch_family_stops_after_memory_or_quality_rejection(self):
        capture = CandidateGateTests()._capture()
        baseline = {
            "candidate": "co-resident-safe-baseline",
            "status": "passed",
            "quality_passed": True,
            "quality_capture": capture,
            "selected": schema.safe_advanced_defaults("cuda"),
            "measurements": [],
        }
        bad_capture = CandidateGateTests()._capture()
        bad_capture["milco_source_view"] = False
        quality_failure = isolated_result(
            payload={
                **baseline,
                "candidate": "dense-geometric-batch-16",
                "quality_capture": bad_capture,
            }
        )
        memory_failure = isolated_result(
            returncode=-9,
            memory_exceeded=True,
        )
        specs = [
            {
                "name": "dense-geometric-batch-16",
                "target_backends": ["dense"],
                "search_family": "dense-batch",
                "search_value": 16,
                "stop_family_on_rejection": True,
                "changes": {},
            },
            {
                "name": "dense-geometric-batch-32",
                "target_backends": ["dense"],
                "search_family": "dense-batch",
                "search_value": 32,
                "stop_family_on_rejection": True,
                "changes": {},
            },
            {
                "name": "other-family",
                "target_backends": ["reranker"],
                "changes": {},
            },
        ]

        for failed, reason_fragment in (
            (memory_failure, "isolated safety gate"),
            (quality_failure, "versioned quality gate"),
        ):
            with (
                self.subTest(reason=reason_fragment),
                tempfile.TemporaryDirectory() as temporary,
            ):
                calls = iter(
                    [
                        isolated_result(payload=baseline),
                        failed,
                        isolated_result(
                            payload={**baseline, "candidate": "other-family"}
                        ),
                    ]
                )
                with patch.object(runner, "tuning_candidate_specs", return_value=specs):
                    result = runner.calibrate(
                        config=self._config(Path(temporary) / "profile.json"),
                        fingerprint=fingerprint(),
                        script_dir=SERVER_DIR,
                        isolated_runner=lambda *args, **kwargs: next(calls),
                    )

            stopped = result["results"][2]
            self.assertEqual(stopped["candidate"], "dense-geometric-batch-32")
            self.assertEqual(stopped["status"], "skipped-search-stop")
            self.assertEqual(stopped["search_family"], "dense-batch")
            self.assertIn(reason_fragment, stopped["reason"])
            self.assertEqual(result["results"][3]["candidate"], "other-family")

    def test_cpu_full_requires_final_long_canary_before_persisting(self):
        capture = CandidateGateTests()._capture()
        baseline_selected = schema.safe_advanced_defaults("cpu")
        tuned_selected = json.loads(json.dumps(baseline_selected))
        tuned_selected["advanced"]["dense"]["runtime"] = "onnx-fp32"
        baseline = {
            "candidate": "co-resident-safe-baseline",
            "status": "passed",
            "quality_passed": True,
            "quality_capture": capture,
            "selected": baseline_selected,
            "measurements": [
                {
                    "backend": "dense",
                    "input_tokens": 128,
                    "batch_size": 2,
                    "is_canary": False,
                    "p50_seconds": 1.0,
                    "p95_seconds": 1.0,
                    "useful_tokens_per_second": 100.0,
                },
                {
                    "backend": "dense",
                    "input_tokens": 8192,
                    "batch_size": 1,
                    "is_canary": True,
                    "p50_seconds": 10.0,
                    "p95_seconds": 10.0,
                    "useful_tokens_per_second": 819.2,
                },
            ],
        }
        tuned = {
            **baseline,
            "candidate": "cpu-dense-onnx-fp32",
            "selected": tuned_selected,
            "measurements": [
                {
                    "backend": "dense",
                    "input_tokens": 128,
                    "batch_size": 2,
                    "is_canary": False,
                    "p50_seconds": 0.9,
                    "p95_seconds": 0.9,
                    "coefficient_of_variation": 0.01,
                    "useful_tokens_per_second": 110.0,
                }
            ],
        }
        spec = {
            "name": "cpu-dense-onnx-fp32",
            "target_backends": ["dense"],
            "changes": {"dense": {"runtime": "onnx-fp32"}},
            "defer_long_canary": True,
        }

        for canary_p95, expected_runtime, promoted in (
            (10.5, "onnx-fp32", True),
            (11.1, "torch", False),
        ):
            with (
                self.subTest(canary_p95=canary_p95),
                tempfile.TemporaryDirectory() as temporary,
            ):
                final_canary = {
                    **baseline,
                    "candidate": "selected-final-long-canary",
                    "selected": tuned_selected,
                    "measurements": [
                        {
                            "backend": "dense",
                            "input_tokens": length,
                            "batch_size": 1,
                            "is_canary": True,
                            "p50_seconds": (canary_p95 if length == 8192 else 1.0),
                            "p95_seconds": (canary_p95 if length == 8192 else 1.0),
                            "coefficient_of_variation": 0.0,
                            "useful_tokens_per_second": length
                            / (canary_p95 if length == 8192 else 1.0),
                        }
                        for length in (1024, 2048, 8192)
                    ],
                }
                calls = iter(
                    [
                        isolated_result(payload=baseline),
                        isolated_result(payload=tuned),
                        isolated_result(payload=final_canary),
                    ]
                )
                config = schema.AdvancedAutotuneConfig.from_env(
                    {
                        "SMART_EMBEDDER_ADVANCED_AUTOTUNE_PROFILE": str(
                            Path(temporary) / "profile.json"
                        ),
                        "SMART_EMBEDDER_AUTOTUNE": "full",
                        "SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS": "300",
                    },
                    device_kind="cpu",
                )
                with patch.object(
                    runner, "tuning_candidate_specs", return_value=[spec]
                ):
                    result = runner.calibrate(
                        config=config,
                        fingerprint=fingerprint("cpu"),
                        script_dir=SERVER_DIR,
                        isolated_runner=lambda *args, **kwargs: next(calls),
                    )

            self.assertEqual(
                result["selected"]["advanced"]["dense"]["runtime"],
                expected_runtime,
            )
            self.assertEqual(
                result["results"][-1]["candidate"], "selected-final-long-canary"
            )
            self.assertEqual(result["results"][-1]["promotion"]["promoted"], promoted)


if __name__ == "__main__":
    unittest.main()
