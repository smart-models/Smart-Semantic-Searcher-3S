"""Tests for the basic-profile autotuner.

Covers configuration, hardware fingerprinting, profile persistence, the
workload/objective definition and the isolated executor that runs each
candidate.
"""

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import basic_autotune as autotune  # noqa: E402


def write(root: Path, relative: str, value: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


class AutotuneConfigTests(unittest.TestCase):
    """Defaults, explicit values, and safe fallback for invalid input."""

    def test_defaults_are_quick_balanced_and_cache_relative(self):
        config = autotune.AutotuneConfig.from_env({"HF_HOME": "/models"})

        self.assertEqual(config.mode, "quick")
        self.assertEqual(config.objective, "balanced")
        self.assertEqual(config.runtime, "auto")
        self.assertEqual(config.torch_compile, "auto")
        self.assertEqual(config.torch_compile_mode, "default")
        self.assertEqual(config.onnx_execution_mode, "auto")
        self.assertEqual(config.onnx_spin_policy, "auto")
        self.assertEqual(
            config.profile_path,
            str(Path("/models") / "autotune" / "hardware-profile.json"),
        )
        self.assertEqual(config.max_seconds, 120.0)

    def test_explicit_values_and_full_budget_are_resolved(self):
        config = autotune.AutotuneConfig.from_env(
            {
                "SMART_EMBEDDER_AUTOTUNE": "FULL",
                "SMART_EMBEDDER_TUNING_OBJECTIVE": "throughput",
                "SMART_EMBEDDER_AUTOTUNE_FORCE": "yes",
                "SMART_EMBEDDER_AUTOTUNE_PROFILE": "/tmp/custom.json",
                "SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS": "42.5",
                "BGE_EMBED_RUNTIME": "onnx-fp32",
                "BGE_TORCH_COMPILE": "yes",
                "BGE_TORCH_COMPILE_MODE": "reduce-overhead",
                "BGE_ONNX_EXECUTION_MODE": "parallel",
                "BGE_ONNX_SPIN_POLICY": "bounded",
            }
        )

        self.assertEqual(config.mode, "full")
        self.assertEqual(config.objective, "throughput")
        self.assertTrue(config.force)
        self.assertEqual(config.profile_path, "/tmp/custom.json")
        self.assertEqual(config.max_seconds, 42.5)
        self.assertEqual(config.runtime, "onnx-fp32")
        self.assertEqual(config.torch_compile, "on")
        self.assertEqual(config.torch_compile_mode, "reduce-overhead")
        self.assertEqual(config.onnx_execution_mode, "parallel")
        self.assertEqual(config.onnx_spin_policy, "bounded")

    def test_invalid_values_fall_back_safely(self):
        config = autotune.AutotuneConfig.from_env(
            {
                "SMART_EMBEDDER_AUTOTUNE": "dangerous",
                "SMART_EMBEDDER_TUNING_OBJECTIVE": "maximum",
                "SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS": "nan",
                "BGE_EMBED_RUNTIME": "magic",
                "BGE_TORCH_COMPILE": "sometimes",
                "BGE_TORCH_COMPILE_MODE": "magic",
                "BGE_ONNX_EXECUTION_MODE": "magic",
                "BGE_ONNX_SPIN_POLICY": "magic",
            }
        )

        self.assertEqual(config.mode, "quick")
        self.assertEqual(config.objective, "balanced")
        self.assertEqual(config.max_seconds, 120.0)
        self.assertEqual(config.runtime, "auto")
        self.assertEqual(config.torch_compile, "auto")
        self.assertEqual(config.torch_compile_mode, "default")
        self.assertEqual(config.onnx_execution_mode, "auto")
        self.assertEqual(config.onnx_spin_policy, "auto")


class HardwareFingerprintTests(unittest.TestCase):
    """What the basic fingerprint detects: cpuset, cgroup limits, GPU identity."""

    def test_cpu_list_parser_handles_ranges_and_invalid_parts(self):
        self.assertEqual(
            autotune.parse_cpu_list("0-3,8,10-11,bad,9-7"),
            {0, 1, 2, 3, 8, 10, 11},
        )

    def test_source_artifact_digest_changes_with_runtime_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "basic_autotune.py", "version one")
            first = autotune.source_artifact_digest(root)
            write(root, "basic_autotune.py", "version two")
            second = autotune.source_artifact_digest(root)

        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, second)

    def test_detects_cgroup_v2_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "cpu.max", "150000 100000\n")
            write(root, "cpuset.cpus.effective", "2-5\n")
            write(root, "memory.max", str(8 * 1024**3))

            limits = autotune.detect_cgroup_limits(root)

        self.assertEqual(limits["cpu_quota"], 1.5)
        self.assertEqual(limits["cpuset"], [2, 3, 4, 5])
        self.assertEqual(limits["memory_limit_bytes"], 8 * 1024**3)

    def test_effective_cpu_count_respects_affinity_cpuset_and_quota(self):
        cpuinfo = """
processor : 0
physical id : 0
core id : 0
model name : Test CPU
flags : avx avx2 f16c

processor : 1
physical id : 0
core id : 1
model name : Test CPU
flags : avx avx2 f16c
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cgroup = root / "cgroup"
            proc = root / "proc"
            sys_root = root / "sys"
            write(cgroup, "cpu.max", "400000 100000")
            write(cgroup, "cpuset.cpus.effective", "0-7")
            write(cgroup, "memory.max", "max")
            write(proc, "cpuinfo", cpuinfo)
            write(sys_root, "devices/system/node/online", "0-1")

            with (
                patch.object(autotune.os, "cpu_count", return_value=16),
                patch.object(
                    autotune, "_affinity_cpu_ids", return_value=list(range(6))
                ),
            ):
                detected = autotune.detect_cpu(cgroup, proc, sys_root)

        self.assertEqual(detected["effective_cpu_count"], 4)
        self.assertEqual(detected["physical_core_count"], 2)
        self.assertEqual(detected["instruction_sets"], ["avx", "avx2", "f16c"])
        self.assertEqual(detected["numa_node_count"], 2)

    def test_physical_cores_are_counted_inside_assigned_cpuset(self):
        cpuinfo = """
processor : 0
physical id : 0
core id : 0
model name : Test CPU
flags : avx2 avx512_bf16 amx_bf16

processor : 1
physical id : 0
core id : 0
model name : Test CPU
flags : avx2 avx512_bf16 amx_bf16

processor : 2
physical id : 0
core id : 1
model name : Test CPU
flags : avx2 avx512_bf16 amx_bf16

processor : 3
physical id : 0
core id : 1
model name : Test CPU
flags : avx2 avx512_bf16 amx_bf16
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / "cgroup", "cpu.max", "max 100000")
            write(root / "cgroup", "cpuset.cpus.effective", "0-1")
            write(root / "cgroup", "memory.max", "max")
            write(root / "proc", "cpuinfo", cpuinfo)
            detected = autotune.detect_cpu(root / "cgroup", root / "proc", root / "sys")

        self.assertEqual(detected["physical_core_count"], 1)
        self.assertIn("avx512_bf16", detected["instruction_sets"])
        self.assertIn("amx_bf16", detected["instruction_sets"])

    def test_gpu_detection_uses_stable_capabilities_not_free_memory(self):
        properties = types.SimpleNamespace(
            name="NVIDIA Test GPU",
            uuid="GPU-123",
            total_memory=8 * 1024**3,
            multi_processor_count=24,
        )
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: properties,
            get_device_capability=lambda index: (8, 9),
        )

        devices = autotune.detect_gpus(types.SimpleNamespace(cuda=cuda))

        self.assertEqual(
            devices,
            [
                {
                    "index": 0,
                    "name": "NVIDIA Test GPU",
                    "uuid": "GPU-123",
                    "compute_capability": "8.9",
                    "total_memory_bytes": 8 * 1024**3,
                    "multi_processor_count": 24,
                }
            ],
        )

    def test_nvidia_driver_version_is_stable_across_multiple_gpus(self):
        completed = types.SimpleNamespace(
            returncode=0,
            stdout="572.83\n572.83\n",
        )
        with patch.object(autotune.subprocess, "run", return_value=completed):
            version = autotune._nvidia_driver_version()

        self.assertEqual(version, "572.83")


class ProfilePersistenceTests(unittest.TestCase):
    """Atomic writes, locking, and every reason a profile is invalidated."""

    def _fingerprint(self, cpu_count=4):
        return {
            "schema_version": autotune.AUTOTUNE_SCHEMA_VERSION,
            "hardware": {"cpu": {"effective_cpu_count": cpu_count}},
            "software": {"model_revision": "revision-a"},
        }

    def _profile(self, fingerprint=None):
        config = autotune.AutotuneConfig.from_env({"HF_HOME": "/models"})
        return autotune.make_profile(
            fingerprint=fingerprint or self._fingerprint(),
            config=config,
            selected={"threads": 4, "runtime": "torch"},
            results=[{"tokens_per_second": 123.0}],
            duration_seconds=1.25,
        )

    def test_atomic_round_trip_and_fingerprint_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "profile.json"
            profile = self._profile()

            autotune.save_profile_locked(path, profile)
            loaded = autotune.load_profile(path)

            self.assertEqual(loaded, profile)
            self.assertTrue(autotune.profile_matches(loaded, self._fingerprint()))
            self.assertFalse(
                autotune.profile_matches(loaded, self._fingerprint(cpu_count=8))
            )
            self.assertFalse(path.with_suffix(".json.lock").exists())

    def test_profile_is_invalidated_when_tuning_contract_changes(self):
        profile = self._profile()
        matching = autotune.AutotuneConfig.from_env()
        latency = autotune.AutotuneConfig.from_env(
            {"SMART_EMBEDDER_TUNING_OBJECTIVE": "latency"}
        )
        compile_off = autotune.AutotuneConfig.from_env({"BGE_TORCH_COMPILE": "off"})
        compile_mode = autotune.AutotuneConfig.from_env(
            {"BGE_TORCH_COMPILE_MODE": "reduce-overhead"}
        )
        onnx_mode = autotune.AutotuneConfig.from_env(
            {"BGE_ONNX_EXECUTION_MODE": "parallel"}
        )

        self.assertTrue(
            autotune.profile_matches(profile, self._fingerprint(), config=matching)
        )
        self.assertFalse(
            autotune.profile_matches(profile, self._fingerprint(), config=latency)
        )
        self.assertFalse(
            autotune.profile_matches(profile, self._fingerprint(), config=compile_off)
        )
        self.assertFalse(
            autotune.profile_matches(profile, self._fingerprint(), config=compile_mode)
        )
        self.assertFalse(
            autotune.profile_matches(profile, self._fingerprint(), config=onnx_mode)
        )

    def test_corrupt_or_old_schema_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text("not-json", encoding="utf-8")
            self.assertIsNone(autotune.load_profile(path))
            path.write_text('{"schema_version": 0, "selected": {}}', encoding="utf-8")
            self.assertIsNone(autotune.load_profile(path))

    def test_second_lock_times_out_and_stale_lock_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "profile.lock"
            with autotune.ProfileLock(lock_path), self.assertRaises(TimeoutError):
                with autotune.ProfileLock(
                    lock_path,
                    timeout_seconds=0.02,
                    poll_seconds=0.005,
                ):
                    pass

            lock_path.write_text("stale", encoding="utf-8")
            old = time.time() - 100
            os.utime(lock_path, (old, old))
            with autotune.ProfileLock(lock_path, stale_seconds=1):
                self.assertTrue(lock_path.exists())
            self.assertFalse(lock_path.exists())

    def test_runtime_profile_distinguishes_active_stale_and_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            config = autotune.AutotuneConfig.from_env(
                {"SMART_EMBEDDER_AUTOTUNE_PROFILE": str(path)}
            )
            profile = autotune.make_profile(
                fingerprint=self._fingerprint(),
                config=config,
                selected={"threads": 4},
                results=[],
            )
            autotune.atomic_write_profile(path, profile)

            active = autotune.resolve_runtime_profile(config, self._fingerprint())
            stale = autotune.resolve_runtime_profile(
                config, self._fingerprint(cpu_count=8)
            )
            disabled_config = autotune.AutotuneConfig.from_env(
                {
                    "SMART_EMBEDDER_AUTOTUNE": "off",
                    "SMART_EMBEDDER_AUTOTUNE_PROFILE": str(path),
                }
            )
            disabled = autotune.resolve_runtime_profile(
                disabled_config, self._fingerprint()
            )

        self.assertEqual(active.state, "active")
        self.assertEqual(active.selected["threads"], 4)
        self.assertEqual(stale.state, "stale")
        self.assertEqual(stale.selected, {})
        self.assertEqual(disabled.state, "disabled")

    def test_batch_policy_is_clamped_sorted_and_covers_max_length(self):
        policy = autotune.normalise_batch_policy(
            [
                {"max_tokens": 512, "batch_size": 8},
                {"max_tokens": 128, "batch_size": 16},
                {"max_tokens": -1, "batch_size": 4},
                {"max_tokens": 1024, "batch_size": 9999},
            ],
            default_batch_size=1,
        )

        self.assertEqual(
            [item["max_tokens"] for item in policy], [128, 512, 1024, 8192]
        )
        self.assertEqual(policy[2]["batch_size"], 512)
        self.assertEqual(autotune.batch_size_for_tokens(policy, 64), 16)
        self.assertEqual(autotune.batch_size_for_tokens(policy, 7000), 512)


class WorkloadAndObjectiveTests(unittest.TestCase):
    """Bounded workloads, and the hard quality/memory/stability gates."""

    def _candidate(self, name, throughput, latency, **overrides):
        result = {
            "name": name,
            "quality_pass": True,
            "tokens_per_second_samples": [throughput, throughput],
            "p95_latency_ms": latency,
            "peak_memory_bytes": 50,
            "memory_limit_bytes": 100,
        }
        result.update(overrides)
        return result

    def test_quick_and_full_workloads_are_bounded(self):
        quick = autotune.calibration_workloads("quick")
        full = autotune.calibration_workloads("full")

        self.assertEqual(autotune.calibration_workloads("off"), [])
        self.assertEqual(max(max(item.token_lengths) for item in quick), 1024)
        self.assertEqual(max(max(item.token_lengths) for item in full), 8192)
        self.assertTrue(any(item.include_colbert for item in quick))
        self.assertGreater(len(full), len(quick))

    def test_objectives_select_different_valid_candidates(self):
        low_latency = self._candidate("low-latency", 100.0, 10.0)
        high_throughput = self._candidate("high-throughput", 200.0, 100.0)
        candidates = [low_latency, high_throughput]

        throughput = autotune.select_best_candidate(
            candidates, objective="throughput", max_memory_fraction=0.8
        )
        latency = autotune.select_best_candidate(
            candidates, objective="latency", max_memory_fraction=0.8
        )
        balanced = autotune.select_best_candidate(
            candidates, objective="balanced", max_memory_fraction=0.8
        )

        self.assertEqual(throughput["name"], "high-throughput")
        self.assertEqual(latency["name"], "low-latency")
        self.assertEqual(balanced["name"], "high-throughput")

    def test_quality_memory_and_stability_are_hard_gates(self):
        failed_quality = self._candidate("quality", 100.0, 10.0, quality_pass=False)
        excessive_memory = self._candidate("memory", 100.0, 10.0)
        unstable = self._candidate(
            "unstable",
            100.0,
            10.0,
            tokens_per_second_samples=[50.0, 150.0],
        )
        excessive_memory["peak_memory_bytes"] = 90

        for candidate in (failed_quality, excessive_memory, unstable):
            self.assertFalse(
                autotune.candidate_is_safe(
                    candidate, max_memory_fraction=0.8, max_cv=0.08
                )
            )
        self.assertIsNone(
            autotune.select_best_candidate(
                [failed_quality, excessive_memory, unstable],
                objective="throughput",
                max_memory_fraction=0.8,
            )
        )


class IsolatedExecutorTests(unittest.TestCase):
    """The subprocess executor's success, timeout and memory-kill paths."""

    def test_successful_worker_returns_last_json_object(self):
        code = (
            "import json; print('warmup'); "
            "print(json.dumps({'quality_pass': True, 'value': 7}))"
        )

        result = autotune.run_isolated([sys.executable, "-c", code], timeout_seconds=5)

        self.assertTrue(result.ok)
        self.assertEqual(result.payload["value"], 7)
        self.assertIn("warmup", result.stdout)

    def test_timeout_terminates_worker(self):
        result = autotune.run_isolated(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_seconds=0.05,
            poll_seconds=0.005,
        )

        self.assertTrue(result.timed_out)
        self.assertFalse(result.ok)

    def test_memory_reader_can_stop_candidate_before_timeout(self):
        result = autotune.run_isolated(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_seconds=2,
            max_rss_bytes=100,
            poll_seconds=0.005,
            rss_reader=lambda _pid: 101,
        )

        self.assertTrue(result.memory_exceeded)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.peak_rss_bytes, 101)

    def test_nonzero_worker_is_not_successful(self):
        result = autotune.run_isolated(
            [sys.executable, "-c", "raise RuntimeError('boom')"],
            timeout_seconds=5,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(result.ok)
        self.assertIn("RuntimeError", result.stderr)


class HierarchicalTunerTests(unittest.TestCase):
    """The tuner rediscovers known-good CPU and GPU policies."""

    @staticmethod
    def _measurement(throughput, latency, memory=50, quality=True):
        return {
            "quality_pass": quality,
            "tokens_per_second_samples": [throughput, throughput * 1.01],
            "p95_latency_ms": latency,
            "peak_memory_bytes": memory,
            "memory_limit_bytes": 100,
        }

    def test_cpu_tuner_rediscovers_four_threads_batch_four_single_lane(self):
        def benchmark(candidate):
            stage = candidate["stage"]
            if stage == "threads":
                throughput = {1: 100, 2: 180, 4: 260}[candidate["threads"]]
                return self._measurement(throughput, 10)
            if stage == "batch":
                batch = candidate["batch_size"]
                return self._measurement({1: 260, 2: 330, 4: 404}[batch], 10 * batch)
            lanes = candidate["concurrency"]
            throughput = 404 if lanes == 1 else 350
            return self._measurement(throughput, 40 if lanes == 1 else 45)

        outcome = autotune.tune_cpu(
            cpu={"effective_cpu_count": 4, "physical_core_count": 4},
            mode="quick",
            objective="balanced",
            benchmark=benchmark,
        )

        self.assertIsNone(outcome.fallback_reason)
        self.assertEqual(outcome.selected["threads"], 4)
        self.assertEqual(outcome.selected["inference_concurrency"], 1)
        self.assertEqual(outcome.selected["batch_policy"][0]["batch_size"], 4)
        self.assertEqual(outcome.selected["batch_policy"][-1]["batch_size"], 1)

    def test_cpu_candidate_space_bounds_threads_and_lanes(self):
        space = autotune.cpu_candidate_space(
            {"effective_cpu_count": 32, "physical_core_count": 16}, "full"
        )

        self.assertEqual(space["threads"][-1], 32)
        self.assertIn(16, space["threads"])
        self.assertEqual(space["lanes"], [1, 2, 4, 8])
        self.assertEqual(space["batches"], [1, 2, 4, 8, 16])

    def test_cpu_runtime_probe_selects_onnx_and_really_tunes_its_threads(self):
        def benchmark(candidate):
            runtime = candidate.get("runtime", "torch")
            multiplier = 2 if runtime == "onnx-fp32" else 1
            throughput = (
                100
                * multiplier
                * candidate["batch_size"]
                * candidate["threads"]
                / candidate["concurrency"]
            )
            return self._measurement(throughput, 10)

        outcome = autotune.tune_cpu(
            cpu={"effective_cpu_count": 4, "physical_core_count": 4},
            mode="quick",
            objective="throughput",
            benchmark=benchmark,
            runtimes=("torch", "onnx-fp32"),
        )

        self.assertEqual(outcome.selected["runtime"], "onnx-fp32")
        self.assertEqual(outcome.selected["threads"], 4)
        self.assertEqual(outcome.selected["inference_concurrency"], 1)
        thread_probes = [item for item in outcome.results if item["stage"] == "threads"]
        self.assertEqual([item["threads"] for item in thread_probes], [1, 2, 4])
        self.assertTrue(all(item["include_colbert"] for item in thread_probes))

    def test_quick_cpu_reserves_budget_for_medium_length_canary(self):
        class Benchmark:
            """Callable stand-in returning scripted measurements with a time budget."""

            @staticmethod
            def remaining_seconds():
                return 40.0

            def __call__(self, candidate):
                throughput = (
                    100
                    * candidate["threads"]
                    * candidate["batch_size"]
                    * candidate["concurrency"]
                )
                return HierarchicalTunerTests._measurement(throughput, 10)

        outcome = autotune.tune_cpu(
            cpu={"effective_cpu_count": 8, "physical_core_count": 4},
            mode="quick",
            objective="throughput",
            benchmark=Benchmark(),
            runtimes=("onnx-fp32",),
        )

        stages = [item["stage"] for item in outcome.results]
        self.assertNotIn("onnx-execution", stages)
        self.assertNotIn("onnx-spin", stages)
        self.assertIn("full-output-canary", stages)

    def test_cpu_canary_failure_uses_conservative_fallback(self):
        def benchmark(candidate):
            quality = not (
                candidate["stage"] == "full-output-canary"
                and candidate["token_lengths"] == [512]
            )
            return self._measurement(200, 10, quality=quality)

        outcome = autotune.tune_cpu(
            cpu={"effective_cpu_count": 4, "physical_core_count": 4},
            mode="quick",
            objective="throughput",
            benchmark=benchmark,
            runtimes=("onnx-fp32",),
        )

        self.assertIsNotNone(outcome.fallback_reason)
        self.assertEqual(outcome.selected["runtime"], "torch")
        self.assertEqual(outcome.selected["threads"], 1)
        self.assertEqual(
            outcome.selected["batch_policy"],
            [{"max_tokens": 8192, "batch_size": 1}],
        )

    def test_gpu_tuner_rediscovers_4060_style_policy(self):
        def benchmark(candidate):
            if candidate["stage"] == "precision":
                return self._measurement(1000, 10, memory=20)
            batch = candidate["batch_size"]
            token_length = candidate["token_lengths"][0]
            throughput = batch * 1000 / (1 + token_length / 8192)
            memory = min(79, 18 + batch * 7)
            return self._measurement(throughput, batch * 10, memory=memory)

        outcome = autotune.tune_gpu(
            gpu={"total_memory_bytes": 8 * 1024**3},
            mode="quick",
            objective="throughput",
            benchmark=benchmark,
        )

        policy = {
            item["max_tokens"]: item["batch_size"]
            for item in outcome.selected["batch_policy"]
        }
        self.assertEqual(outcome.selected["precision"], "fp16")
        self.assertFalse(outcome.selected["torch_compile"])
        self.assertEqual(policy[128], 8)
        self.assertEqual(policy[512], 8)
        self.assertEqual(policy[1024], 4)
        self.assertEqual(policy[2048], 1)
        self.assertEqual(policy[8192], 1)

    def test_gpu_search_space_scales_conservatively_with_vram(self):
        expected = {
            8: {128: 8, 1024: 4, 2048: 1, 8192: 1},
            16: {128: 16, 1024: 8, 2048: 4, 8192: 2},
            32: {128: 32, 1024: 16, 2048: 8, 8192: 4},
        }

        for memory_gib, buckets in expected.items():
            for token_length, maximum_batch in buckets.items():
                candidates = autotune.gpu_batch_candidates(
                    memory_gib * 1024**3, token_length, "quick"
                )
                self.assertEqual(candidates[-1], maximum_batch)

    def test_gpu_memory_gate_selects_one_step_below_unsafe_batch(self):
        def benchmark(candidate):
            if candidate["stage"] == "precision":
                return self._measurement(1000, 10, memory=20)
            batch = candidate["batch_size"]
            memory = 90 if batch == 8 else 20 + batch * 10
            return self._measurement(batch * 1000, batch * 10, memory=memory)

        outcome = autotune.tune_gpu(
            gpu={"total_memory_bytes": 8 * 1024**3},
            mode="quick",
            objective="throughput",
            benchmark=benchmark,
            memory_fraction=0.8,
        )

        self.assertEqual(outcome.selected["batch_policy"][0]["batch_size"], 4)

    def test_gpu_full_output_canary_can_reduce_selected_short_batch(self):
        def benchmark(candidate):
            if candidate["stage"] == "precision":
                return self._measurement(1000, 10, memory=20)
            batch = candidate["batch_size"]
            memory = (
                90
                if (candidate.get("include_colbert") and batch > 2)
                else 20 + batch * 5
            )
            return self._measurement(batch * 1000, batch * 10, memory=memory)

        outcome = autotune.tune_gpu(
            gpu={"total_memory_bytes": 8 * 1024**3},
            mode="quick",
            objective="throughput",
            benchmark=benchmark,
            memory_fraction=0.8,
        )

        self.assertEqual(outcome.selected["batch_policy"][0]["batch_size"], 2)
        canaries = [
            result
            for result in outcome.results
            if result["stage"] == "full-output-canary"
            and result["token_lengths"] == [128]
        ]
        self.assertEqual([result["batch_size"] for result in canaries], [8, 4, 2])

    def test_gpu_compile_auto_rejects_measured_regression(self):
        def benchmark(candidate):
            if candidate["stage"] == "torch-compile":
                throughput = 900 if candidate["torch_compile"] else 1000
                return self._measurement(throughput, 10, memory=30)
            batch = candidate["batch_size"]
            return self._measurement(batch * 1000, batch * 10, memory=30)

        outcome = autotune.tune_gpu(
            gpu={"total_memory_bytes": 8 * 1024**3},
            mode="quick",
            objective="throughput",
            benchmark=benchmark,
            torch_compile="auto",
        )

        self.assertFalse(outcome.selected["torch_compile"])
        self.assertEqual(
            outcome.selected["torch_compile_reason"],
            "gain-below-promotion-threshold",
        )

    def test_gpu_compile_auto_promotes_only_material_gain(self):
        def benchmark(candidate):
            if candidate["stage"] == "torch-compile":
                throughput = 1150 if candidate["torch_compile"] else 1000
                return self._measurement(throughput, 10, memory=35)
            batch = candidate["batch_size"]
            return self._measurement(batch * 1000, batch * 10, memory=30)

        outcome = autotune.tune_gpu(
            gpu={"total_memory_bytes": 32 * 1024**3},
            mode="quick",
            objective="throughput",
            benchmark=benchmark,
            torch_compile="auto",
        )

        self.assertTrue(outcome.selected["torch_compile"])
        self.assertEqual(
            outcome.selected["torch_compile_reason"],
            "measured-material-gain",
        )
        self.assertEqual(outcome.selected["batch_policy"][0]["batch_size"], 32)

    def test_gpu_compile_auto_keeps_eager_for_noise_sized_gain(self):
        eager = self._measurement(1000, 10, memory=30)
        compiled = self._measurement(1040, 10, memory=30)
        eager["torch_compile"] = False
        compiled["torch_compile"] = True

        enabled, reason = autotune.select_torch_compile(
            [eager, compiled],
            objective="throughput",
            max_memory_fraction=0.8,
        )

        self.assertFalse(enabled)
        self.assertEqual(reason, "gain-below-promotion-threshold")

    def test_gpu_compile_explicit_policy_skips_probe(self):
        compile_probes = []

        def benchmark(candidate):
            if candidate["stage"] == "torch-compile":
                compile_probes.append(candidate)
            batch = candidate["batch_size"]
            return self._measurement(batch * 1000, batch * 10, memory=30)

        disabled = autotune.tune_gpu(
            gpu={"total_memory_bytes": 8 * 1024**3},
            mode="quick",
            objective="throughput",
            benchmark=benchmark,
            torch_compile="off",
        )
        enabled = autotune.tune_gpu(
            gpu={"total_memory_bytes": 8 * 1024**3},
            mode="quick",
            objective="throughput",
            benchmark=benchmark,
            torch_compile="on",
        )

        self.assertEqual(compile_probes, [])
        self.assertFalse(disabled.selected["torch_compile"])
        self.assertTrue(enabled.selected["torch_compile"])


if __name__ == "__main__":
    unittest.main()
