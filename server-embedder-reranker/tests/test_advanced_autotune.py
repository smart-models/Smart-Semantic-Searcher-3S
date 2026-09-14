"""Tests for the advanced-profile autotuner's config, fingerprint and profile.

Covers how ``advanced_autotune`` reads its configuration, fingerprints the
host, and decides whether a persisted profile is still valid.
"""

import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import advanced_autotune as subject  # noqa: E402


def write(root: Path, relative: str, value: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


class AdvancedAutotuneConfigTests(unittest.TestCase):
    """The advanced tuner's config is independent and device-aware."""

    def test_profile_is_independent_and_budget_is_device_aware(self):
        gpu = subject.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models"}, device_kind="cuda"
        )
        cpu = subject.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models"}, device_kind="cpu"
        )

        self.assertEqual(gpu.max_seconds, 240.0)
        self.assertEqual(cpu.max_seconds, 360.0)
        self.assertEqual(
            gpu.profile_path,
            str(Path("/models/autotune/advanced-hardware-profile.json")),
        )
        self.assertNotEqual(Path(gpu.profile_path).name, "hardware-profile.json")

    def test_full_and_explicit_values_are_respected(self):
        config = subject.AdvancedAutotuneConfig.from_env(
            {
                "SMART_EMBEDDER_AUTOTUNE": "full",
                "SMART_EMBEDDER_TUNING_OBJECTIVE": "latency",
                "SMART_EMBEDDER_AUTOTUNE_FORCE": "true",
                "SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS": "42.5",
                "SMART_EMBEDDER_ADVANCED_AUTOTUNE_PROFILE": "/tmp/a.json",
            }
        )

        self.assertEqual(config.mode, "full")
        self.assertEqual(config.objective, "latency")
        self.assertTrue(config.force)
        self.assertEqual(config.max_seconds, 42.5)
        self.assertEqual(config.profile_path, "/tmp/a.json")

    def test_default_quick_budget_adapts_and_explicit_budget_is_fixed(self):
        default = subject.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models"}, device_kind="cuda"
        )
        explicit = subject.AdvancedAutotuneConfig.from_env(
            {
                "HF_HOME": "/models",
                "SMART_EMBEDDER_ADVANCED_AUTOTUNE_MAX_SECONDS": "240",
            },
            device_kind="cuda",
        )
        full = subject.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models", "SMART_EMBEDDER_AUTOTUNE": "full"},
            device_kind="cuda",
        )
        cpu = subject.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models"}, device_kind="cpu"
        )

        self.assertTrue(default.adaptive_budget)
        self.assertEqual(default.budget_ceiling_seconds, 720.0)
        self.assertEqual(cpu.budget_ceiling_seconds, 1080.0)
        self.assertFalse(explicit.adaptive_budget)
        self.assertEqual(explicit.budget_ceiling_seconds, 240.0)
        self.assertFalse(full.adaptive_budget)
        self.assertEqual(full.budget_ceiling_seconds, 1800.0)


class AdvancedFingerprintTests(unittest.TestCase):
    """What the advanced fingerprint covers, and what invalidates it."""

    def _roots(self, root: Path) -> tuple[Path, Path, Path]:
        cgroup = root / "cgroup"
        proc = root / "proc"
        sys_root = root / "sys"
        write(cgroup, "cpu.max", "400000 100000")
        write(cgroup, "cpuset.cpus.effective", "0-3")
        write(cgroup, "memory.max", str(12 * 1024**3))
        write(
            proc,
            "cpuinfo",
            """processor : 0
physical id : 0
core id : 0
model name : Test CPU
flags : avx2 f16c

processor : 1
physical id : 0
core id : 0
model name : Test CPU
flags : avx2 f16c

processor : 2
physical id : 1
core id : 0
model name : Test CPU
flags : avx2 f16c
""",
        )
        write(sys_root, "devices/system/node/online", "0-1")
        return cgroup, proc, sys_root

    def _cache_models(self, hf_home: Path) -> None:
        for name, model_id in subject.ADVANCED_MODEL_IDS.items():
            revision = subject.MODEL_REVISIONS[model_id]
            repo = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
            write(repo, "refs/main", revision)
            write(repo, f"snapshots/{revision}/config.json", "{}")
            if name == "sparse":
                write(repo, f"snapshots/{revision}/milco.py", "VERSION = 1")

    def test_complete_fingerprint_tracks_models_milco_policy_and_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hf_home = root / "models"
            self._cache_models(hf_home)
            cgroup, proc, sys_root = self._roots(root)
            write(root / "sources", "advanced_milco_runtime.py", "runtime-v1")
            config = subject.AdvancedAutotuneConfig.from_env({"HF_HOME": str(hf_home)})
            env = {
                "HF_HOME": str(hf_home),
                "MILCO_SOURCE_VIEW": "true",
                "MILCO_TOP_K": "0",
                "QWEN_EMBED_MAX_LENGTH": "8192",
            }
            fingerprint = subject.build_advanced_fingerprint(
                config=config,
                env=env,
                hf_home=hf_home,
                source_root=root / "sources",
                cgroup_root=cgroup,
                proc_root=proc,
                sys_root=sys_root,
            )

        self.assertTrue(all(item["cached"] for item in fingerprint["models"].values()))
        self.assertEqual(
            fingerprint["models"]["sparse"]["revision"],
            subject.MODEL_REVISIONS[subject.ADVANCED_MODEL_IDS["sparse"]],
        )
        self.assertEqual(fingerprint["hardware"]["cpu"]["socket_count"], 2)
        self.assertTrue(fingerprint["hardware"]["cpu"]["smt_active"])
        self.assertEqual(fingerprint["hardware"]["cpu"]["numa_node_count"], 2)
        self.assertEqual(fingerprint["hardware"]["memory_limit_bytes"], 12 * 1024**3)
        self.assertEqual(len(fingerprint["software"]["milco_remote_code_digest"]), 64)
        self.assertEqual(len(fingerprint["software"]["local_runtime_digest"]), 64)
        self.assertEqual(
            set(fingerprint["model_dependencies"]),
            {"milco_multilingual_tokenizer"},
        )
        self.assertTrue(fingerprint["limits_and_policy"]["milco_source_view"])
        self.assertEqual(
            fingerprint["limits_and_policy"]["explicit_env"]["QWEN_EMBED_MAX_LENGTH"],
            "8192",
        )

    def test_milco_remote_code_change_invalidates_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hf_home = root / "models"
            self._cache_models(hf_home)
            cgroup, proc, sys_root = self._roots(root)
            config = subject.AdvancedAutotuneConfig.from_env({"HF_HOME": str(hf_home)})

            def build():
                return subject.build_advanced_fingerprint(
                    config=config,
                    env={"HF_HOME": str(hf_home)},
                    hf_home=hf_home,
                    source_root=root / "sources",
                    cgroup_root=cgroup,
                    proc_root=proc,
                    sys_root=sys_root,
                )

            first = build()
            sparse = hf_home / "hub/models--omai-research--milco-650m"
            revision = subject.MODEL_REVISIONS[subject.ADVANCED_MODEL_IDS["sparse"]]
            write(sparse, f"snapshots/{revision}/milco.py", "VERSION = 2")
            second = build()

        self.assertNotEqual(
            first["software"]["milco_remote_code_digest"],
            second["software"]["milco_remote_code_digest"],
        )
        self.assertNotEqual(
            subject.fingerprint_digest(first), subject.fingerprint_digest(second)
        )

    def test_gpu_identity_and_sm_count_are_included(self):
        properties = types.SimpleNamespace(
            name="RTX Test",
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
        torch_module = types.SimpleNamespace(
            __version__="2.7.0",
            version=types.SimpleNamespace(cuda="12.6"),
            cuda=cuda,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cgroup, proc, sys_root = self._roots(root)
            config = subject.AdvancedAutotuneConfig.from_env(
                {"HF_HOME": str(root / "models")}
            )
            with patch.object(subject, "_package_version", return_value=None):
                fingerprint = subject.build_advanced_fingerprint(
                    config=config,
                    torch_module=torch_module,
                    env={"HF_HOME": str(root / "models")},
                    hf_home=root / "models",
                    source_root=root / "sources",
                    cgroup_root=cgroup,
                    proc_root=proc,
                    sys_root=sys_root,
                )

        gpu = fingerprint["hardware"]["gpus"][0]
        self.assertEqual(gpu["uuid"], "GPU-123")
        self.assertEqual(gpu["compute_capability"], "8.9")
        self.assertEqual(gpu["multi_processor_count"], 24)


class AdvancedProfileTests(unittest.TestCase):
    """Round-tripping, reusing and rejecting a persisted advanced profile."""

    def _fingerprint(self) -> dict:
        return {"schema_version": 1, "hardware": {"gpus": []}}

    def test_complete_schema_round_trips_and_reuses(self):
        selected = subject.safe_advanced_defaults("cpu")
        self.assertFalse(selected["advanced"]["dense"]["prefetch_batches"])
        self.assertFalse(selected["advanced"]["sparse"]["prefetch_batches"])
        self.assertEqual(selected["advanced"]["schedule"]["worker_spin"], "default")
        self.assertEqual(selected["advanced"]["schedule"]["affinity"], "none")
        self.assertFalse(selected["advanced"]["schedule"]["allow_oversubscription"])
        config = subject.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models"}, device_kind="cpu"
        )
        profile = subject.make_advanced_profile(
            fingerprint=self._fingerprint(),
            config=config,
            selected=selected,
            results=[{"candidate": "safe", "quality_passed": True}],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "advanced.json"
            subject.save_advanced_profile(path, profile)
            loaded = subject.load_advanced_profile(path)

        self.assertEqual(loaded["selected"], selected)
        self.assertTrue(
            subject.advanced_profile_matches(loaded, self._fingerprint(), config=config)
        )

    def test_basic_or_incomplete_profile_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile": "basic",
                        "selected": {"runtime": "torch"},
                    }
                ),
                encoding="utf-8",
            )
            loaded = subject.load_advanced_profile(path)

        self.assertIsNone(loaded)
        self.assertFalse(subject.validate_advanced_selected({"advanced": {}}))

    def test_version_one_profile_is_invalidated_for_new_schedule_evidence(self):
        """A profile predating reranker admission checks cannot be reused."""
        selected = subject.safe_advanced_defaults("cpu")
        config = subject.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models"}, device_kind="cpu"
        )
        profile = subject.make_advanced_profile(
            fingerprint=self._fingerprint(),
            config=config,
            selected=selected,
            results=[],
        )
        profile["schema_version"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "advanced.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            loaded = subject.load_advanced_profile(path)

        self.assertIsNone(loaded)

    def test_budget_and_objective_invalidate_reuse(self):
        first = subject.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models", "SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS": "240"}
        )
        second = subject.AdvancedAutotuneConfig.from_env(
            {"HF_HOME": "/models", "SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS": "241"}
        )
        profile = subject.make_advanced_profile(
            fingerprint=self._fingerprint(),
            config=first,
            selected=subject.safe_advanced_defaults("cuda"),
            results=[],
        )

        self.assertFalse(
            subject.advanced_profile_matches(
                profile, self._fingerprint(), config=second
            )
        )


class CpuPowerTests(unittest.TestCase):
    """Host CPU power policy is recorded and throttled calibration is reported."""

    def _cpufreq(self, root, cpu, governor, preference, max_khz):
        base = f"devices/system/cpu/cpu{cpu}/cpufreq"
        write(root, f"{base}/scaling_governor", governor)
        write(root, f"{base}/energy_performance_preference", preference)
        write(root, f"{base}/scaling_driver", "intel_pstate")
        write(root, f"{base}/scaling_max_freq", str(max_khz))

    def test_policy_counts_each_setting_across_assigned_cpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._cpufreq(root, 0, "powersave", "balance_performance", 2700000)
            self._cpufreq(root, 1, "performance", "performance", 2700000)
            policy = subject.read_cpu_power_policy([0, 1, 7], sys_root=root)

        self.assertTrue(policy["available"])
        self.assertEqual(policy["governors"], {"performance": 1, "powersave": 1})
        self.assertEqual(
            policy["energy_performance_preferences"],
            {"balance_performance": 1, "performance": 1},
        )
        self.assertEqual(policy["drivers"], ["intel_pstate"])
        self.assertEqual(policy["max_frequency_mhz"], 2700)

    def test_missing_cpufreq_is_reported_as_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = subject.read_cpu_power_policy([0], sys_root=Path(temporary))

        self.assertFalse(policy["available"])
        self.assertIsNone(policy["max_frequency_mhz"])

    def test_warns_only_when_measured_frequency_is_below_half_of_maximum(self):
        policy = {
            "governors": {"powersave": 224},
            "energy_performance_preferences": {"balance_performance": 224},
            "max_frequency_mhz": 2700,
        }
        slow = subject.cpu_power_warnings(policy, {"samples": 9, "median_mhz": 800})

        self.assertEqual(len(slow), 1)
        self.assertIn("800 MHz", slow[0])
        self.assertIn("powersave", slow[0])
        self.assertIn("balance_performance", slow[0])
        fast = {"samples": 9, "median_mhz": 2400}
        self.assertEqual(subject.cpu_power_warnings(policy, fast), [])
        self.assertEqual(subject.cpu_power_warnings(policy, {"samples": 0}), [])
        self.assertEqual(subject.cpu_power_warnings(policy, None), [])

    def test_active_profile_exposes_its_recorded_warnings(self):
        config = subject.AdvancedAutotuneConfig.from_env({"HF_HOME": "/models"})
        fingerprint = {"schema_version": 1, "hardware": {"gpus": []}}
        profile = subject.make_advanced_profile(
            fingerprint=fingerprint,
            config=config,
            selected=subject.safe_advanced_defaults("cuda"),
            results=[],
            host_cpu_power={"policy": {}, "measured": None, "warnings": ["slow"]},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "advanced.json"
            subject.save_advanced_profile(path, profile)
            resolved = subject.resolve_advanced_runtime_profile(
                replace(config, profile_path=str(path)), fingerprint
            )

        self.assertEqual(resolved.state, "active")
        self.assertEqual(resolved.warnings, ("slow",))


class GpuWaitDiagnosisTests(unittest.TestCase):
    """Low GPU utilisation during the probes means the GPU waits on the CPU."""

    def test_low_median_utilisation_is_diagnosed_and_warned(self):
        diagnosis = subject.gpu_wait_diagnosis({"samples": 20, "median_percent": 11.0})

        self.assertTrue(diagnosis["gpu_waits_on_cpu"])
        self.assertEqual(len(diagnosis["warnings"]), 1)
        self.assertIn("11%", diagnosis["warnings"][0])

    def test_busy_gpu_or_too_few_samples_is_not_diagnosed(self):
        busy = subject.gpu_wait_diagnosis({"samples": 20, "median_percent": 85.0})
        few = subject.gpu_wait_diagnosis({"samples": 2, "median_percent": 5.0})
        missing = subject.gpu_wait_diagnosis(None)

        for diagnosis in (busy, few, missing):
            self.assertFalse(diagnosis["gpu_waits_on_cpu"])
            self.assertEqual(diagnosis["warnings"], [])

    def test_active_profile_warnings_include_gpu_dispatch(self):
        config = subject.AdvancedAutotuneConfig.from_env({"HF_HOME": "/models"})
        fingerprint = {"schema_version": 1, "hardware": {"gpus": []}}
        profile = subject.make_advanced_profile(
            fingerprint=fingerprint,
            config=config,
            selected=subject.safe_advanced_defaults("cuda"),
            results=[],
            host_cpu_power={"policy": {}, "measured": None, "warnings": ["cpu"]},
            gpu_dispatch={"gpu_waits_on_cpu": True, "warnings": ["gpu"]},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "advanced.json"
            subject.save_advanced_profile(path, profile)
            resolved = subject.resolve_advanced_runtime_profile(
                replace(config, profile_path=str(path)), fingerprint
            )

        self.assertEqual(resolved.warnings, ("cpu", "gpu"))


if __name__ == "__main__":
    unittest.main()
