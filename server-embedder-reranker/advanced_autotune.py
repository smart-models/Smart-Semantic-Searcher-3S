"""Persistence, fingerprinting and search primitives for ADVANCED autotuning.

The ADVANCED profile is intentionally independent from the proven BGE/BASIC
autotuner.  This module is standard-library-only so a serving process can
validate a persisted decision without importing or loading model weights.

Primary interfaces are ``AdvancedAutotuneConfig``,
``build_advanced_fingerprint()``, and ``resolve_advanced_runtime_profile()``.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from basic_autotune import (
    ProfileLock,
    atomic_write_profile,
    build_fingerprint,
    fingerprint_digest,
    normalise_choice,
    parse_bool,
    positive_float,
)
from model_revisions import MODEL_REVISIONS

ADVANCED_AUTOTUNE_SCHEMA_VERSION = 2
ADVANCED_AUTOTUNE_MODES = frozenset({"off", "quick", "full"})
ADVANCED_AUTOTUNE_OBJECTIVES = frozenset({"balanced", "throughput", "latency"})
#: An unset quick budget may grow to fit the measured baseline plus the planned
#: candidates, up to this multiple of its default. An explicit budget is fixed.
ADAPTIVE_BUDGET_CEILING_FACTOR = 3.0
#: Calibration cores running below this share of their maximum frequency are
#: reported: slow single-thread kernel dispatch leaves the GPU waiting.
CPU_FREQUENCY_WARNING_RATIO = 0.5
#: A GPU busy less than this share of the baseline probes is waiting on the CPU.
GPU_WAIT_UTILIZATION_PERCENT = 50.0
#: Fewer utilisation samples than this are too few to diagnose anything.
GPU_WAIT_MIN_SAMPLES = 4
ADVANCED_MODEL_IDS = {
    "dense": "Qwen/Qwen3-Embedding-0.6B",
    "sparse": "omai-research/milco-650m",
    "reranker": "Qwen/Qwen3-Reranker-0.6B",
}
ADVANCED_RUNTIME_ARTIFACTS = (
    "advanced_autotune.py",
    "advanced_autotune_runner.py",
    "advanced_autotune_worker.py",
    "advanced_autotune_quality.py",
    "advanced_milco_runtime.py",
    "advanced_qwen_onnx_backend.py",
    "smart-embedder.py",
    "request_budget.py",
)
ADVANCED_EXPLICIT_ENV = (
    "QWEN_EMBED_RUNTIME",
    "QWEN_EMBED_MAX_LENGTH",
    "QWEN_EMBED_BATCH_SIZE",
    "QWEN_EMBED_MAX_BATCH_TOKENS",
    "QWEN_EMBED_PRECISION",
    "QWEN_EMBED_ATTENTION",
    "QWEN_EMBED_TORCH_COMPILE",
    "QWEN_EMBED_TORCH_COMPILE_MODE",
    "QWEN_EMBED_PINNED_MEMORY",
    "QWEN_EMBED_PREFETCH_BATCHES",
    "QWEN_EMBED_BATCH_POLICY",
    "QWEN_EMBED_MAX_BATCH_TOKENS_POLICY",
    "SMART_EMBEDDER_ADVANCED_ONNX_CACHE",
    "QWEN_RERANK_MAX_LENGTH",
    "QWEN_RERANK_RUNTIME",
    "QWEN_RERANK_BATCH_SIZE",
    "QWEN_RERANK_PRECISION",
    "QWEN_RERANK_ATTENTION",
    "QWEN_RERANK_TORCH_COMPILE",
    "QWEN_RERANK_TORCH_COMPILE_MODE",
    "QWEN_RERANK_BATCH_POLICY",
    "QWEN_RERANK_MAX_BATCH_TOKENS_POLICY",
    "MILCO_MAX_LENGTH",
    "MILCO_RUNTIME",
    "MILCO_BATCH_SIZE",
    "MILCO_MAX_BATCH_TOKENS",
    "MILCO_PROJECTION_CHUNK_TOKENS",
    "MILCO_SOURCE_VIEW",
    "MILCO_TOP_K",
    "MILCO_PRECISION",
    "MILCO_COMPUTE_PRECISION",
    "MILCO_TORCH_COMPILE",
    "MILCO_TORCH_COMPILE_MODE",
    "MILCO_PINNED_MEMORY",
    "MILCO_PREFETCH_BATCHES",
    "MILCO_BATCH_POLICY",
    "MILCO_MAX_BATCH_TOKENS_POLICY",
    "ADVANCED_SCHEDULE_MODE",
    "ADVANCED_INFERENCE_LANES",
    "ADVANCED_QUEUE_BATCH",
    "ADVANCED_FLUSH_TIMEOUT_MS",
    "ADVANCED_CPU_SPIN_POLICY",
    "ADVANCED_CPU_AFFINITY",
    "ADVANCED_CPU_ALLOW_OVERSUBSCRIPTION",
    "EMBEDDER_NUM_THREADS",
    "EMBEDDER_INTEROP_THREADS",
    "EMBED_LENGTH_AWARE_BATCHING",
)


@dataclass(frozen=True)
class AdvancedAutotuneConfig:
    """Operator-facing ADVANCED calibration configuration."""

    mode: str
    objective: str
    force: bool
    profile_path: str
    max_seconds: float
    adaptive_budget: bool = False

    @property
    def budget_ceiling_seconds(self) -> float:
        """Longest calibration this configuration may run, in seconds."""
        if not self.adaptive_budget:
            return self.max_seconds
        return self.max_seconds * ADAPTIVE_BUDGET_CEILING_FACTOR

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        device_kind: str = "cuda",
    ) -> "AdvancedAutotuneConfig":
        """Resolve ADVANCED calibration settings from environment values.

        Device-specific defaults are used only when an explicit time budget is
        absent or invalid.

        Args:
            env: Environment mapping. Uses the process environment when omitted.
            device_kind: Effective execution device, either ``cpu`` or ``cuda``.

        Returns:
            A validated, conservative ADVANCED autotune configuration.
        """
        values = os.environ if env is None else env
        mode = normalise_choice(
            values.get("SMART_EMBEDDER_AUTOTUNE"),
            "quick",
            ADVANCED_AUTOTUNE_MODES,
        )
        objective = normalise_choice(
            values.get("SMART_EMBEDDER_TUNING_OBJECTIVE"),
            "balanced",
            ADVANCED_AUTOTUNE_OBJECTIVES,
        )
        hf_home = values.get("HF_HOME", "/app/model_cache").strip()
        default_path = str(
            Path(hf_home) / "autotune" / "advanced-hardware-profile.json"
        )
        path = (
            values.get("SMART_EMBEDDER_ADVANCED_AUTOTUNE_PROFILE", default_path).strip()
            or default_path
        )
        default_seconds = (
            1800.0 if mode == "full" else (360.0 if device_kind == "cpu" else 240.0)
        )
        max_seconds_value = values.get("SMART_EMBEDDER_ADVANCED_AUTOTUNE_MAX_SECONDS")
        if max_seconds_value is None or not max_seconds_value.strip():
            max_seconds_value = values.get("SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS")
        explicit_budget = positive_float(max_seconds_value, 0.0) > 0.0
        return cls(
            mode=mode,
            objective=objective,
            force=parse_bool(values.get("SMART_EMBEDDER_AUTOTUNE_FORCE")),
            profile_path=path,
            max_seconds=positive_float(
                max_seconds_value,
                default_seconds,
            ),
            adaptive_budget=mode == "quick" and not explicit_budget,
        )


@dataclass(frozen=True)
class AdvancedRuntimeProfile:
    """Resolved ADVANCED profile state consumed by the serving process."""

    state: str
    selected: dict
    fingerprint_sha256: str
    profile_path: str
    fallback_reason: Optional[str] = None
    warnings: tuple[str, ...] = ()


def safe_advanced_defaults(device_kind: str) -> dict:
    """Return a complete conservative policy matching current serving code."""
    precision = "fp16" if device_kind == "cuda" else "fp32"
    sparse_compute = "bf16" if device_kind == "cuda" else "fp32"
    return {
        "advanced": {
            "dense": {
                "runtime": "torch",
                "precision": precision,
                "attention": "sdpa",
                "torch_compile": False,
                "torch_compile_mode": "default",
                "batch_policy": [],
                "max_batch_tokens_policy": [],
                "pinned_memory": False,
                "prefetch_batches": False,
            },
            "sparse": {
                "runtime": "torch",
                "precision": precision,
                "compute_precision": sparse_compute,
                "torch_compile": False,
                "torch_compile_mode": "default",
                "batch_policy": [],
                "max_batch_tokens_policy": [],
                "pinned_memory": False,
                "prefetch_batches": False,
            },
            "schedule": {
                "mode": "sequential",
                "inference_lanes": 1,
                "queue_batch": 1,
                "flush_timeout_ms": 50,
                "intra_op_threads": 4,
                "inter_op_threads": 1,
                "worker_spin": "default",
                "affinity": "none",
                "allow_oversubscription": False,
            },
            "reranker": {
                "runtime": "torch",
                "precision": precision,
                "attention": "sdpa",
                "torch_compile": False,
                "torch_compile_mode": "default",
                "batch_policy": [],
                "max_batch_tokens_policy": [],
            },
        }
    }


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_advanced_selected(selected: Any) -> bool:
    """Reject incomplete/corrupt policies before the serving process uses them."""
    if not isinstance(selected, Mapping):
        return False
    advanced = selected.get("advanced")
    if not isinstance(advanced, Mapping):
        return False
    required = {"dense", "sparse", "schedule", "reranker"}
    if not required.issubset(advanced):
        return False
    for backend in ("dense", "sparse", "reranker"):
        policy = advanced.get(backend)
        if not isinstance(policy, Mapping):
            return False
        if policy.get("runtime") not in {
            "torch",
            "onnx-fp32",
            "onnx-int8",
            "openvino",
        }:
            return False
        if policy.get("precision") not in {"fp32", "fp16", "bf16", "int8"}:
            return False
        if not isinstance(policy.get("torch_compile"), bool):
            return False
        if backend in {"dense", "sparse"} and not isinstance(
            policy.get("prefetch_batches"), bool
        ):
            return False
        for key in ("batch_policy", "max_batch_tokens_policy"):
            if not isinstance(policy.get(key), list):
                return False
    schedule = advanced["schedule"]
    return bool(
        isinstance(schedule, Mapping)
        and schedule.get("mode") in {"sequential", "overlap", "parallel"}
        and schedule.get("worker_spin") in {"default", "passive", "active"}
        and schedule.get("affinity") in {"none", "compact", "socket-local"}
        and isinstance(schedule.get("allow_oversubscription"), bool)
        and _is_positive_int(schedule.get("inference_lanes"))
        and _is_positive_int(schedule.get("queue_batch"))
        and _is_positive_int(schedule.get("flush_timeout_ms"))
        and _is_positive_int(schedule.get("intra_op_threads"))
        and _is_positive_int(schedule.get("inter_op_threads"))
    )


def _hash_files(root: Path, patterns: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    found = False
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    for path in sorted(paths, key=lambda item: item.as_posix()):
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        found = True
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest() if found else ""


def advanced_source_digest(root: Optional[Path] = None) -> str:
    """Hash the runtime sources that can affect ADVANCED calibration.

    Args:
        root: Directory containing the runtime artifacts. Defaults to this
            module's directory.

    Returns:
        A SHA-256 hex digest, or an empty string when no artifact is readable.
    """
    directory = root or Path(__file__).resolve().parent
    digest = hashlib.sha256()
    found = False
    for name in ADVANCED_RUNTIME_ARTIFACTS:
        path = directory / name
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest() if found else ""


def _snapshot_revision(snapshot: Path) -> str:
    if snapshot.parent.name == "snapshots":
        return snapshot.name
    return ""


def cached_model_metadata(
    model_id: str,
    hf_home: Path,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
) -> dict:
    """Resolve cache metadata without any network access."""
    hub_root = hub_cache or hf_home / "hub"
    repo_dir = hub_root / f"models--{model_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    candidates: list[Path] = [snapshots / revision] if revision else []
    if not revision:
        main_ref = repo_dir / "refs" / "main"
        try:
            main_revision = main_ref.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError, UnicodeError):
            main_revision = ""
        if main_revision:
            candidates.append(snapshots / main_revision)
        with contextlib.suppress(FileNotFoundError, OSError):
            candidates.extend(
                sorted(
                    (path for path in snapshots.iterdir() if path.is_dir()),
                    key=lambda path: path.name,
                    reverse=True,
                )
            )
    snapshot = next((path for path in candidates if path.is_dir()), None)
    return {
        "model_id": model_id,
        "revision": _snapshot_revision(snapshot) if snapshot else "missing",
        "cached": snapshot is not None,
        "snapshot": str(snapshot) if snapshot else None,
    }


def _cpu_topology(proc_root: Path) -> dict:
    try:
        cpuinfo = (proc_root / "cpuinfo").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeError):
        cpuinfo = ""
    sockets: set[str] = set()
    logical_by_core: dict[tuple[str, str], int] = {}
    for block in re.split(r"\n\s*\n", cpuinfo.strip()):
        fields = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip().lower()] = value.strip()
        socket = fields.get("physical id")
        core = fields.get("core id")
        if socket is not None:
            sockets.add(socket)
        if socket is not None and core is not None:
            key = (socket, core)
            logical_by_core[key] = logical_by_core.get(key, 0) + 1
    threads_per_core = max(logical_by_core.values(), default=1)
    return {
        "socket_count": len(sockets) or None,
        "threads_per_core": threads_per_core,
        "smt_active": threads_per_core > 1,
    }


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_sysfs_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return None


def read_cpu_power_policy(
    cpu_ids: Iterable[int], *, sys_root: Path = Path("/sys")
) -> dict:
    """Summarise the host CPU frequency policy for the assigned CPUs.

    Reads sysfs only; a container cannot, and must not, change host power
    management. Settings are counted so a mixed policy stays visible.

    Args:
        cpu_ids: Logical CPUs assigned to this container.
        sys_root: Root of the sysfs tree, replaceable in tests.

    Returns:
        Governor, energy preference and driver counts plus the highest maximum
        frequency in MHz; ``available`` is false when cpufreq is not exposed.
    """
    base = sys_root / "devices" / "system" / "cpu"
    governors: dict[str, int] = {}
    preferences: dict[str, int] = {}
    drivers: set[str] = set()
    max_khz = 0
    for cpu in sorted(set(cpu_ids)):
        cpufreq = base / f"cpu{cpu}" / "cpufreq"
        governor = _read_sysfs_text(cpufreq / "scaling_governor")
        if not governor:
            continue
        governors[governor] = governors.get(governor, 0) + 1
        preference = _read_sysfs_text(cpufreq / "energy_performance_preference")
        if preference:
            preferences[preference] = preferences.get(preference, 0) + 1
        driver = _read_sysfs_text(cpufreq / "scaling_driver")
        if driver:
            drivers.add(driver)
        with contextlib.suppress(TypeError, ValueError):
            max_khz = max(max_khz, int(_read_sysfs_text(cpufreq / "scaling_max_freq")))
    return {
        "available": bool(governors),
        "governors": governors,
        "energy_performance_preferences": preferences,
        "drivers": sorted(drivers),
        "max_frequency_mhz": round(max_khz / 1000) if max_khz else None,
    }


def cpu_power_warnings(
    policy: Mapping[str, Any], measured: Optional[Mapping[str, Any]]
) -> list[str]:
    """Return a warning when calibration ran on cores well below their maximum.

    Args:
        policy: Result of ``read_cpu_power_policy()``.
        measured: Frequency summary sampled while the baseline probes ran.

    Returns:
        One warning naming the measured frequency and the policy, or no warning
        when the cores ran fast enough or nothing could be measured.
    """
    maximum = policy.get("max_frequency_mhz")
    median = (measured or {}).get("median_mhz")
    if not maximum or not median or median >= maximum * CPU_FREQUENCY_WARNING_RATIO:
        return []
    governors = ", ".join(sorted(policy.get("governors") or {})) or "unknown"
    preferences = (
        ", ".join(sorted(policy.get("energy_performance_preferences") or {}))
        or "unknown"
    )
    return [
        f"calibration cores ran at a median {median:.0f} MHz, "
        f"{median / maximum:.0%} of their {maximum:.0f} MHz maximum "
        f"(governor {governors}, energy preference {preferences}); slow "
        "single-thread kernel dispatch then limits GPU throughput, so ask the "
        "host administrator to review CPU power management"
    ]


def gpu_wait_diagnosis(utilization: Optional[Mapping[str, Any]]) -> dict:
    """Decide from baseline GPU utilisation whether the GPU waits on the CPU.

    Args:
        utilization: Utilisation summary sampled while the baseline probes ran.

    Returns:
        Whether the GPU waits on host-side work, the evidence it rests on, and a
        warning when it does.
    """
    samples = int((utilization or {}).get("samples") or 0)
    median = (utilization or {}).get("median_percent")
    waits = (
        samples >= GPU_WAIT_MIN_SAMPLES
        and median is not None
        and float(median) < GPU_WAIT_UTILIZATION_PERCENT
    )
    warnings = (
        [
            f"the GPU was busy a median {float(median):.0f}% of the calibration "
            "probes: it waits on host-side work, so larger batches, request "
            "coalescing and CUDA graphs were measured first"
        ]
        if waits
        else []
    )
    return {
        "gpu_waits_on_cpu": waits,
        "median_percent": None if median is None else float(median),
        "samples": samples,
        "warnings": warnings,
    }


def build_advanced_fingerprint(
    *,
    config: AdvancedAutotuneConfig,
    torch_module: Any = None,
    env: Optional[Mapping[str, str]] = None,
    app_version: str = "",
    hf_home: Optional[Path] = None,
    source_root: Optional[Path] = None,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_root: Path = Path("/proc"),
    sys_root: Path = Path("/sys"),
) -> dict:
    """Build the complete invalidation key for an ADVANCED decision."""
    values = os.environ if env is None else env
    cache = hf_home or Path(values.get("HF_HOME", "/app/model_cache"))
    hub_cache = Path(values.get("HF_HUB_CACHE", str(cache / "hub")))
    base = build_fingerprint(
        torch_module=torch_module,
        app_version=app_version,
        model_revision="advanced-multimodel",
        artifact_digest=advanced_source_digest(source_root),
        cgroup_root=cgroup_root,
        proc_root=proc_root,
        sys_root=sys_root,
    )
    models = {
        name: cached_model_metadata(
            model_id,
            cache,
            hub_cache=hub_cache,
            revision=MODEL_REVISIONS[model_id],
        )
        for name, model_id in ADVANCED_MODEL_IDS.items()
    }
    dependencies = {
        name: cached_model_metadata(
            model_id,
            cache,
            hub_cache=hub_cache,
            revision=MODEL_REVISIONS[model_id],
        )
        for name, model_id in {
            "milco_multilingual_tokenizer": "BAAI/bge-m3-unsupervised",
        }.items()
    }
    sparse_snapshot = models["sparse"].get("snapshot")
    milco_digest = (
        _hash_files(Path(sparse_snapshot), ("*.py", "**/*.py", "config.json"))
        if sparse_snapshot
        else ""
    )
    cpu = dict(base["hardware"]["cpu"])
    cpu.update(_cpu_topology(proc_root))
    software = dict(base["software"])
    software.update(
        {
            "local_runtime_digest": advanced_source_digest(source_root),
            "milco_remote_code_digest": milco_digest,
            "flash_attn": _package_version("flash-attn"),
            "onnx": _package_version("onnx"),
            "onnxscript": _package_version("onnxscript"),
            "onnxruntime": _package_version("onnxruntime"),
            "optimum": _package_version("optimum"),
            "openvino": _package_version("openvino"),
        }
    )
    explicit = {
        name: values.get(name)
        for name in ADVANCED_EXPLICIT_ENV
        if values.get(name) is not None
    }
    return {
        "schema_version": ADVANCED_AUTOTUNE_SCHEMA_VERSION,
        "profile": "advanced",
        "hardware": {
            "cpu": cpu,
            "memory_limit_bytes": base["hardware"]["memory_limit_bytes"],
            "gpus": base["hardware"]["gpus"],
        },
        "software": software,
        "models": {
            name: {
                "model_id": value["model_id"],
                "revision": value["revision"],
                "cached": value["cached"],
            }
            for name, value in models.items()
        },
        "model_dependencies": {
            name: {
                "model_id": value["model_id"],
                "revision": value["revision"],
                "cached": value["cached"],
            }
            for name, value in dependencies.items()
        },
        "limits_and_policy": {
            "explicit_env": explicit,
            "milco_source_view": parse_bool(values.get("MILCO_SOURCE_VIEW"), True),
            "milco_top_k": values.get("MILCO_TOP_K", "0"),
        },
        "autotune": {
            "mode": config.mode,
            "objective": config.objective,
            "max_seconds": config.max_seconds,
        },
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
        },
    }


def make_advanced_profile(
    *,
    fingerprint: Mapping[str, Any],
    config: AdvancedAutotuneConfig,
    selected: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    status: str = "complete",
    fallback_reason: Optional[str] = None,
    duration_seconds: float = 0.0,
    budget: Optional[Mapping[str, Any]] = None,
    host_cpu_power: Optional[Mapping[str, Any]] = None,
    gpu_dispatch: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build a persisted ADVANCED hardware profile from measured evidence.

    Args:
        fingerprint: Hardware, software, model, and policy identity.
        config: Operator configuration used for calibration.
        selected: Complete policy selected by the calibration gates.
        results: Raw candidate evidence retained for auditability.
        status: Profile completion state.
        fallback_reason: Optional reason a conservative policy was selected.
        duration_seconds: Total calibration duration.
        budget: Optional record of the calibration time budget that applied.
        host_cpu_power: Optional host CPU power policy, measured frequency and
            warnings recorded during calibration.
        gpu_dispatch: Optional diagnosis of whether the GPU waited on the CPU.

    Returns:
        A versioned, JSON-serializable profile document.

    Raises:
        ValueError: If ``selected`` is not a complete ADVANCED policy.
    """
    if not validate_advanced_selected(selected):
        raise ValueError("invalid ADVANCED selected policy")
    profile = {
        "schema_version": ADVANCED_AUTOTUNE_SCHEMA_VERSION,
        "profile": "advanced",
        "status": status,
        "fingerprint_sha256": fingerprint_digest(fingerprint),
        "fingerprint": dict(fingerprint),
        "config": asdict(config),
        "selected": dict(selected),
        "results": [dict(result) for result in results],
        "fallback_reason": fallback_reason,
        "duration_seconds": round(max(0.0, duration_seconds), 6),
        "created_at_unix": int(time.time()),
    }
    if budget is not None:
        profile["budget"] = dict(budget)
    if host_cpu_power is not None:
        profile["host_cpu_power"] = dict(host_cpu_power)
    if gpu_dispatch is not None:
        profile["gpu_dispatch"] = dict(gpu_dispatch)
    return profile


def load_advanced_profile(path: Path) -> Optional[dict]:
    """Load a structurally valid ADVANCED profile from disk.

    Args:
        path: Profile JSON path.

    Returns:
        The decoded profile, or ``None`` when it is absent, corrupt, outdated,
        or belongs to another model profile.
    """
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(profile, dict):
        return None
    if profile.get("schema_version") != ADVANCED_AUTOTUNE_SCHEMA_VERSION:
        return None
    if profile.get("profile") != "advanced":
        return None
    if not validate_advanced_selected(profile.get("selected")):
        return None
    return profile


def advanced_profile_matches(
    profile: Optional[Mapping[str, Any]],
    fingerprint: Mapping[str, Any],
    *,
    config: AdvancedAutotuneConfig,
) -> bool:
    """Check whether a completed profile is reusable for this exact run.

    Args:
        profile: Previously loaded profile, if one exists.
        fingerprint: Current hardware and software identity.
        config: Current calibration contract.

    Returns:
        ``True`` only when identity and reuse-sensitive settings match.
    """
    if not profile or profile.get("status") != "complete":
        return False
    if profile.get("fingerprint_sha256") != fingerprint_digest(fingerprint):
        return False
    saved = profile.get("config")
    return bool(
        isinstance(saved, Mapping)
        and saved.get("mode") == config.mode
        and saved.get("objective") == config.objective
        and saved.get("max_seconds") == config.max_seconds
    )


def resolve_advanced_runtime_profile(
    config: AdvancedAutotuneConfig,
    fingerprint: dict,
    *,
    applicable: bool = True,
) -> AdvancedRuntimeProfile:
    """Resolve the persisted policy visible to the serving process.

    Args:
        config: Current ADVANCED autotune configuration.
        fingerprint: Current hardware and software fingerprint.
        applicable: Whether the active model profile uses ADVANCED tuning.

    Returns:
        Runtime state with a selected policy only when the profile is reusable.
    """
    digest = fingerprint_digest(fingerprint)
    if not applicable:
        state = "not-applicable"
        profile = None
    elif config.mode == "off":
        state = "disabled"
        profile = None
    else:
        profile = load_advanced_profile(Path(config.profile_path))
        if advanced_profile_matches(profile, fingerprint, config=config):
            return AdvancedRuntimeProfile(
                state="active",
                selected=dict(profile["selected"]),
                fingerprint_sha256=digest,
                profile_path=config.profile_path,
                fallback_reason=profile.get("fallback_reason"),
                warnings=tuple(
                    [
                        *((profile.get("host_cpu_power") or {}).get("warnings") or ()),
                        *((profile.get("gpu_dispatch") or {}).get("warnings") or ()),
                    ]
                ),
            )
        state = "stale" if profile is not None else "missing"
    return AdvancedRuntimeProfile(
        state=state,
        selected={},
        fingerprint_sha256=digest,
        profile_path=config.profile_path,
        fallback_reason=(profile or {}).get("fallback_reason"),
    )


def save_advanced_profile(path: Path, profile: Mapping[str, Any]) -> None:
    """Persist an ADVANCED profile atomically under an interprocess lock.

    Args:
        path: Destination profile path.
        profile: Complete JSON-serializable profile document.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    with ProfileLock(lock_path):
        atomic_write_profile(path, profile)


def models_are_cached(fingerprint: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Report whether every ADVANCED model snapshot is available locally.

    Args:
        fingerprint: Fingerprint containing per-model cache metadata.

    Returns:
        A readiness flag and the identifiers of any missing models.
    """
    models = fingerprint.get("models", {})
    missing = (
        [
            str(value.get("model_id", name))
            for name, value in models.items()
            if not value.get("cached")
        ]
        if isinstance(models, Mapping)
        else list(ADVANCED_MODEL_IDS.values())
    )
    return not missing, missing
