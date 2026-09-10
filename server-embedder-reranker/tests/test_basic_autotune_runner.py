"""Tests for the basic autotuner's runner utilities and calibration.

Simulated measurements drive ``basic_autotune_runner`` end to end so the
calibration result is checked without loading BGE-M3.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import basic_autotune as autotune  # noqa: E402
import basic_autotune_runner as autotune_runner  # noqa: E402


class RunnerUtilityTests(unittest.TestCase):
    """The runner's sampling, percentile and similarity helpers."""

    def test_quick_gpu_batch_candidates_get_stability_samples(self):
        self.assertEqual(
            autotune_runner._measurement_repeats("quick", "cuda", "batch-policy"),
            2,
        )
        self.assertEqual(
            autotune_runner._measurement_repeats("quick", "cpu", "batch-policy"),
            1,
        )
        self.assertEqual(
            autotune_runner._measurement_repeats("quick", "cpu", "batch"),
            1,
        )
        self.assertEqual(
            autotune_runner._measurement_repeats("quick", "cuda", "torch-compile"),
            2,
        )
        self.assertEqual(
            autotune_runner._measurement_repeats("full", "cuda", "full-output-canary"),
            3,
        )
        self.assertEqual(
            autotune_runner._measurement_repeats("full", "cpu", "full-output-canary"),
            1,
        )

    def test_p95_uses_nearest_rank(self):
        self.assertEqual(autotune_runner._p95_ms([0.1, 0.2, 0.3]), 300.0)

    def test_dense_cosine_flattens_colbert_matrices(self):
        matrix = np.asarray([[1.0, 0.0], [0.5, 0.5]])

        self.assertAlmostEqual(autotune_runner._dense_cosine(matrix, matrix), 1.0)

    def test_sparse_diagnostics_report_rank_overlap_and_weight_delta(self):
        result = autotune_runner._sparse_diagnostics(
            {"a": 0.9, "b": 0.8, "c": 0.1},
            {"a": 0.9, "c": 0.8, "b": 0.1},
            limit=2,
        )

        self.assertFalse(result["match"])
        self.assertFalse(result["rank_match"])
        self.assertEqual(result["overlap"], 1)
        self.assertAlmostEqual(result["max_abs_delta"], 0.7)

    def test_sparse_diagnostics_accept_same_top_set_with_rank_ties(self):
        result = autotune_runner._sparse_diagnostics(
            {"a": 0.9, "b": 0.8},
            {"b": 0.9, "a": 0.8},
            limit=2,
        )

        self.assertTrue(result["match"])
        self.assertFalse(result["rank_match"])

    def test_safe_defaults_are_device_aware(self):
        cpu = autotune_runner._simulate_fingerprint("cpu")
        gpu = autotune_runner._simulate_fingerprint("gpu")

        cpu_defaults = autotune_runner.safe_defaults(cpu)
        gpu_defaults = autotune_runner.safe_defaults(gpu)

        self.assertEqual(cpu_defaults["threads"], 4)
        self.assertEqual(cpu_defaults["precision"], "fp32")
        self.assertEqual(gpu_defaults["precision"], "fp16")
        self.assertFalse(cpu_defaults["torch_compile"])
        self.assertFalse(gpu_defaults["torch_compile"])
        self.assertEqual(gpu_defaults["batch_policy"][-1]["batch_size"], 1)

    def test_simulated_dry_run_reports_search_space_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            args = autotune_runner.parse_args(
                [
                    "--simulate",
                    "gpu",
                    "--dry-run",
                    "--profile",
                    str(path),
                ]
            )

            result = autotune_runner.run(args)

            self.assertEqual(result["action"], "dry-run")
            self.assertEqual(result["search_space"]["128"][-1], 8)
            self.assertEqual(result["search_space"]["8192"], [1])
            self.assertFalse(path.exists())

    def test_dry_run_does_not_reuse_or_overwrite_existing_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            calibrated = autotune_runner.run(
                autotune_runner.parse_args(
                    ["--simulate", "cpu", "--profile", str(path), "--force"]
                )
            )
            before = path.read_bytes()

            result = autotune_runner.run(
                autotune_runner.parse_args(
                    ["--simulate", "cpu", "--profile", str(path), "--dry-run"]
                )
            )

            self.assertEqual(calibrated["action"], "calibrated")
            self.assertEqual(result["action"], "dry-run")
            self.assertEqual(path.read_bytes(), before)

    def test_blackwell_simulations_report_cc12_and_vram_scaled_spaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            for device, short_batch, long_batch in (
                ("rtx5080", 16, 2),
                ("rtx5090", 32, 4),
            ):
                path = Path(temporary) / f"{device}.json"
                fingerprint = autotune_runner._simulate_fingerprint(device)
                result = autotune_runner.run(
                    autotune_runner.parse_args(
                        ["--simulate", device, "--dry-run", "--profile", str(path)]
                    )
                )

                gpu = fingerprint["hardware"]["gpus"][0]
                self.assertEqual(gpu["compute_capability"], "12.0")
                self.assertEqual(result["search_space"]["128"][-1], short_batch)
                self.assertEqual(result["search_space"]["8192"][-1], long_batch)


class SimulatedCalibrationTests(unittest.TestCase):
    """Calibration persists then reuses a profile, on CPU and on GPU."""

    def test_cpu_calibration_persists_then_reuses_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            first_args = autotune_runner.parse_args(
                [
                    "--simulate",
                    "cpu",
                    "--profile",
                    str(path),
                    "--force",
                ]
            )
            first = autotune_runner.run(first_args)
            second = autotune_runner.run(
                autotune_runner.parse_args(
                    ["--simulate", "cpu", "--profile", str(path)]
                )
            )

            self.assertEqual(first["action"], "calibrated")
            self.assertEqual(first["selected"]["threads"], 4)
            self.assertEqual(first["selected"]["batch_policy"][0]["batch_size"], 4)
            self.assertEqual(second["action"], "reused")
            profile = autotune.load_profile(path)
            self.assertEqual(profile["status"], "complete")
            self.assertGreater(len(profile["results"]), 0)

    def test_gpu_calibration_builds_length_aware_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            result = autotune_runner.run(
                autotune_runner.parse_args(
                    [
                        "--simulate",
                        "gpu",
                        "--profile",
                        str(path),
                        "--force",
                        "--objective",
                        "throughput",
                    ]
                )
            )

            policy = {
                item["max_tokens"]: item["batch_size"]
                for item in result["selected"]["batch_policy"]
            }
            self.assertEqual(policy[128], 8)
            self.assertEqual(policy[512], 8)
            self.assertEqual(policy[8192], 1)
            self.assertFalse(result["selected"]["torch_compile"])


if __name__ == "__main__":
    unittest.main()
