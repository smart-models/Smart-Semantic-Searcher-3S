"""Hardware-aware BASIC calibration primitives for Smart Embedder.

This module deliberately depends only on the Python standard library. Model
loading and benchmark execution live in the caller so hardware discovery,
profile invalidation and atomic persistence can be tested without downloading
weights or importing PyTorch.

Primary interfaces cover configuration, fingerprinting, profile persistence,
isolated execution, and the ``tune_cpu()``/``tune_gpu()`` search procedures.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import tempfile
import time
from contextlib import AbstractContextManager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

AUTOTUNE_SCHEMA_VERSION = 4
AUTOTUNE_MODES = frozenset({"off", "quick", "full"})
AUTOTUNE_OBJECTIVES = frozenset({"balanced", "throughput", "latency"})
AUTOTUNE_RUNTIMES = frozenset({"auto", "torch", "onnx-fp32", "onnx-int8"})
AUTOTUNE_TORCH_COMPILE_POLICIES = frozenset({"auto", "on", "off"})
AUTOTUNE_ONNX_EXECUTION_MODES = frozenset({"auto", "sequential", "parallel"})
AUTOTUNE_ONNX_SPIN_POLICIES = frozenset({"auto", "default", "bounded", "disabled"})
AUTOTUNE_TORCH_COMPILE_MODES = frozenset(
    {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
)
AUTOTUNE_ARTIFACT_FILES = (
    "basic_autotune.py",
    "basic_autotune_runner.py",
    "basic_autotune_tokens.py",
    "basic_onnx_backend.py",
    "smart-embedder.py",
    "request_budget.py",
)


def normalise_choice(value: Optional[str], default: str, allowed: frozenset) -> str:
    """Return a normalized allowed value or its conservative default."""
    choice = (value or default).strip().lower()
    return choice if choice in allowed else default


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    """Parse a conventional boolean string without raising on invalid input."""
    if value is None or not value.strip():
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    return default


def _normalise_torch_compile_policy(value: Optional[str]) -> str:
    normalised = (value or "auto").strip().lower()
    if not normalised or normalised == "auto":
        return "auto"
    if normalised in {"1", "true", "yes", "on"}:
        return "on"
    if normalised in {"0", "false", "no", "off"}:
        return "off"
    return "auto"


def positive_float(value: Optional[str], default: float) -> float:
    """Parse a finite positive float or return the supplied default."""
    try:
        parsed = float(value or "")
    except ValueError:
        return default
    return parsed if parsed > 0 and math.isfinite(parsed) else default


@dataclass(frozen=True)
class AutotuneConfig:
    """Operator-facing calibration configuration."""

    mode: str
    objective: str
    force: bool
    profile_path: str
    max_seconds: float
    runtime: str
    torch_compile: str
    torch_compile_mode: str
    onnx_execution_mode: str
    onnx_spin_policy: str

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "AutotuneConfig":
        """Resolve BASIC calibration settings from environment values.

        Args:
            env: Environment mapping. Uses the process environment when omitted.

        Returns:
            A validated configuration with conservative invalid-value fallbacks.
        """
        values = os.environ if env is None else env
        mode = normalise_choice(
            values.get("SMART_EMBEDDER_AUTOTUNE"), "quick", AUTOTUNE_MODES
        )
        objective = normalise_choice(
            values.get("SMART_EMBEDDER_TUNING_OBJECTIVE"),
            "balanced",
            AUTOTUNE_OBJECTIVES,
        )
        runtime = normalise_choice(
            values.get("BGE_EMBED_RUNTIME"), "auto", AUTOTUNE_RUNTIMES
        )
        torch_compile = _normalise_torch_compile_policy(values.get("BGE_TORCH_COMPILE"))
        torch_compile_mode = normalise_choice(
            values.get("BGE_TORCH_COMPILE_MODE"),
            "default",
            AUTOTUNE_TORCH_COMPILE_MODES,
        )
        onnx_execution_mode = normalise_choice(
            values.get("BGE_ONNX_EXECUTION_MODE"),
            "auto",
            AUTOTUNE_ONNX_EXECUTION_MODES,
        )
        onnx_spin_policy = normalise_choice(
            values.get("BGE_ONNX_SPIN_POLICY"),
            "auto",
            AUTOTUNE_ONNX_SPIN_POLICIES,
        )
        hf_home = values.get("HF_HOME", "/app/model_cache").strip()
        default_path = str(Path(hf_home) / "autotune" / "hardware-profile.json")
        profile_path = (
            values.get("SMART_EMBEDDER_AUTOTUNE_PROFILE", default_path).strip()
            or default_path
        )
        max_seconds_default = 600.0 if mode == "full" else 120.0
        return cls(
            mode=mode,
            objective=objective,
            force=parse_bool(values.get("SMART_EMBEDDER_AUTOTUNE_FORCE")),
            profile_path=profile_path,
            max_seconds=positive_float(
                values.get("SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS"),
                max_seconds_default,
            ),
            runtime=runtime,
            torch_compile=torch_compile,
            torch_compile_mode=torch_compile_mode,
            onnx_execution_mode=onnx_execution_mode,
            onnx_spin_policy=onnx_spin_policy,
        )


@dataclass(frozen=True)
class RuntimeProfile:
    """Resolved persisted profile state consumed by the serving process."""

    state: str
    selected: dict
    fingerprint_sha256: str
    profile_path: str
    fallback_reason: Optional[str] = None


def resolve_runtime_profile(
    config: AutotuneConfig,
    fingerprint: dict,
    *,
    applicable: bool = True,
) -> RuntimeProfile:
    """Resolve profile validity without ever launching calibration in-server."""
    digest = fingerprint_digest(fingerprint)
    if not applicable:
        return RuntimeProfile(
            state="not-applicable",
            selected={},
            fingerprint_sha256=digest,
            profile_path=config.profile_path,
        )
    if config.mode == "off":
        return RuntimeProfile(
            state="disabled",
            selected={},
            fingerprint_sha256=digest,
            profile_path=config.profile_path,
        )

    profile = load_profile(Path(config.profile_path))
    if profile_matches(profile, fingerprint, config=config):
        return RuntimeProfile(
            state="active",
            selected=dict(profile["selected"]),
            fingerprint_sha256=digest,
            profile_path=config.profile_path,
            fallback_reason=profile.get("fallback_reason"),
        )
    return RuntimeProfile(
        state="stale" if profile is not None else "missing",
        selected={},
        fingerprint_sha256=digest,
        profile_path=config.profile_path,
        fallback_reason=(profile or {}).get("fallback_reason"),
    )


def normalise_batch_policy(
    value: Any,
    *,
    default_batch_size: int,
    max_length: int = 8192,
    max_batch_size: int = 512,
) -> list[dict]:
    """Validate, clamp and sort a persisted length-aware batch policy."""
    policy = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, Mapping):
                continue
            try:
                token_limit = int(item.get("max_tokens"))
                item_batch = int(item.get("batch_size"))
            except (TypeError, ValueError):
                continue
            if token_limit < 1 or item_batch < 1:
                continue
            policy.append(
                {
                    "max_tokens": min(token_limit, max_length),
                    "batch_size": min(item_batch, max_batch_size),
                }
            )
    policy.sort(key=lambda item: item["max_tokens"])
    deduplicated = {item["max_tokens"]: item["batch_size"] for item in policy}
    policy = [
        {"max_tokens": token_limit, "batch_size": deduplicated[token_limit]}
        for token_limit in sorted(deduplicated)
    ]
    fallback_batch = max(1, min(int(default_batch_size), max_batch_size))
    if not policy:
        return [{"max_tokens": max_length, "batch_size": fallback_batch}]
    if policy[-1]["max_tokens"] < max_length:
        policy.append(
            {"max_tokens": max_length, "batch_size": policy[-1]["batch_size"]}
        )
    return policy


def batch_size_for_tokens(policy: list[Mapping[str, Any]], tokens: int) -> int:
    """Return the first matching bucket, falling back to the final bucket."""
    requested = max(1, int(tokens))
    for item in policy:
        if requested <= int(item["max_tokens"]):
            return int(item["batch_size"])
    return int(policy[-1]["batch_size"])


def parse_cpu_list(value: str) -> set[int]:
    """Parse Linux CPU list syntax such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    for raw_part in value.strip().split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" not in part:
            if part.isdigit():
                cpus.add(int(part))
            continue
        start_text, end_text = part.split("-", 1)
        if not start_text.isdigit() or not end_text.isdigit():
            continue
        start, end = int(start_text), int(end_text)
        if end >= start:
            cpus.update(range(start, end + 1))
    return cpus


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return None


def _positive_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def detect_cgroup_limits(cgroup_root: Path = Path("/sys/fs/cgroup")) -> dict:
    """Detect cgroup v2 limits with v1 fallbacks."""
    cpu_quota: Optional[float] = None
    cpu_max = _read_text(cgroup_root / "cpu.max")
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) >= 2 and parts[0] != "max":
            quota = _positive_int(parts[0])
            period = _positive_int(parts[1])
            if quota and period:
                cpu_quota = quota / period
    else:
        quota = _positive_int(_read_text(cgroup_root / "cpu" / "cpu.cfs_quota_us"))
        period = _positive_int(_read_text(cgroup_root / "cpu" / "cpu.cfs_period_us"))
        if quota and period:
            cpu_quota = quota / period

    cpuset_text = _read_text(cgroup_root / "cpuset.cpus.effective")
    if cpuset_text is None:
        cpuset_text = _read_text(cgroup_root / "cpuset" / "cpuset.cpus")
    cpuset = sorted(parse_cpu_list(cpuset_text or ""))

    memory_text = _read_text(cgroup_root / "memory.max")
    if memory_text is None:
        memory_text = _read_text(cgroup_root / "memory" / "memory.limit_in_bytes")
    memory_limit = None if memory_text == "max" else _positive_int(memory_text)
    # Some cgroup v1 hosts expose an effectively unlimited sentinel.
    if memory_limit is not None and memory_limit >= 1 << 60:
        memory_limit = None

    return {
        "cpu_quota": round(cpu_quota, 6) if cpu_quota is not None else None,
        "cpuset": cpuset,
        "memory_limit_bytes": memory_limit,
    }


def _parse_cpuinfo(cpuinfo: str) -> dict:
    model_name = ""
    flags: set[str] = set()
    physical_cores: set[tuple[str, str]] = set()
    processor_cores: dict[int, tuple[str, str]] = {}
    for block in re.split(r"\n\s*\n", cpuinfo.strip()):
        fields = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip().lower()] = value.strip()
        model_name = model_name or fields.get("model name", "")
        flags.update(fields.get("flags", fields.get("features", "")).split())
        physical_id = fields.get("physical id")
        core_id = fields.get("core id")
        if physical_id is not None and core_id is not None:
            core = (physical_id, core_id)
            physical_cores.add(core)
            with suppress(ValueError):
                processor_cores[int(fields.get("processor", ""))] = core
    return {
        "model": model_name or platform.processor() or platform.machine(),
        "flags": sorted(flags),
        "physical_core_count": len(physical_cores) or None,
        "processor_cores": processor_cores,
    }


def _affinity_cpu_count() -> Optional[int]:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if not callable(get_affinity):
        return None
    try:
        return len(get_affinity(0))
    except OSError:
        return None


def _affinity_cpu_ids() -> list[int]:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if not callable(get_affinity):
        return []
    try:
        return sorted(int(value) for value in get_affinity(0))
    except OSError:
        return []


def _online_numa_nodes(sys_root: Path) -> Optional[int]:
    online = _read_text(sys_root / "devices" / "system" / "node" / "online")
    if online is None:
        return None
    nodes = parse_cpu_list(online)
    return len(nodes) if nodes else None


def detect_cpu(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_root: Path = Path("/proc"),
    sys_root: Path = Path("/sys"),
) -> dict:
    """Return CPU capabilities constrained to the current container."""
    logical = os.cpu_count() or 1
    affinity_ids = _affinity_cpu_ids()
    affinity = len(affinity_ids) or _affinity_cpu_count()
    cgroup = detect_cgroup_limits(cgroup_root)
    limits = [logical]
    if affinity:
        limits.append(affinity)
    if cgroup["cpuset"]:
        limits.append(len(cgroup["cpuset"]))
    if cgroup["cpu_quota"]:
        limits.append(max(1, math.ceil(cgroup["cpu_quota"])))

    parsed = _parse_cpuinfo(_read_text(proc_root / "cpuinfo") or "")
    relevant_flags = [
        flag
        for flag in (
            "avx",
            "avx2",
            "sse4_2",
            "fma",
            "avx512f",
            "avx512_vnni",
            "avx512_bf16",
            "avx_vnni",
            "f16c",
            "bf16",
            "amx_bf16",
            "amx_int8",
            "amx_tile",
        )
        if flag in set(parsed["flags"])
    ]
    assigned_cpu_ids = set(cgroup["cpuset"] or affinity_ids)
    assigned_cores = {
        core
        for processor, core in parsed["processor_cores"].items()
        if processor in assigned_cpu_ids
    }
    physical = len(assigned_cores) or parsed["physical_core_count"]
    if physical is not None:
        physical = min(physical, min(limits))
    return {
        "architecture": platform.machine().lower(),
        "model": parsed["model"],
        "logical_cpu_count": logical,
        "physical_core_count": physical,
        "affinity_cpu_count": affinity,
        "affinity_cpu_ids": affinity_ids,
        "effective_cpu_count": min(limits),
        "cpu_quota": cgroup["cpu_quota"],
        "cpuset": cgroup["cpuset"],
        "instruction_sets": relevant_flags,
        "numa_node_count": _online_numa_nodes(sys_root),
    }


def detect_gpus(torch_module: Any = None) -> list[dict]:
    """Discover CUDA devices through an already-imported PyTorch module."""
    if torch_module is None:
        return []
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return []
    try:
        if not cuda.is_available():
            return []
        count = int(cuda.device_count())
    except (RuntimeError, TypeError, ValueError):
        return []

    devices = []
    for index in range(count):
        try:
            properties = cuda.get_device_properties(index)
            capability = cuda.get_device_capability(index)
            uuid = getattr(properties, "uuid", None)
            devices.append(
                {
                    "index": index,
                    "name": str(getattr(properties, "name", "unknown")),
                    "uuid": str(uuid) if uuid is not None else None,
                    "compute_capability": f"{capability[0]}.{capability[1]}",
                    "total_memory_bytes": int(properties.total_memory),
                    "multi_processor_count": int(
                        getattr(properties, "multi_processor_count", 0)
                    ),
                }
            )
        except (RuntimeError, TypeError, ValueError, AttributeError):
            continue
    return devices


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_artifact_digest(root: Optional[Path] = None) -> str:
    """Hash serving and calibration sources that can change a decision."""
    directory = root or Path(__file__).resolve().parent
    digest = hashlib.sha256()
    found = False
    for name in AUTOTUNE_ARTIFACT_FILES:
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


def _nvidia_driver_version() -> Optional[str]:
    """Read the host driver exposed by the NVIDIA container runtime."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    return ",".join(sorted(versions)) or None


def build_fingerprint(
    *,
    torch_module: Any = None,
    app_version: str = "",
    model_revision: str = "",
    artifact_digest: str = "",
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_root: Path = Path("/proc"),
    sys_root: Path = Path("/sys"),
) -> dict:
    """Build a stable fingerprint used to invalidate calibrated profiles."""
    cgroup = detect_cgroup_limits(cgroup_root)
    torch_version = getattr(torch_module, "__version__", None)
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    gpus = detect_gpus(torch_module)
    return {
        "schema_version": AUTOTUNE_SCHEMA_VERSION,
        "hardware": {
            "cpu": detect_cpu(cgroup_root, proc_root, sys_root),
            "memory_limit_bytes": cgroup["memory_limit_bytes"],
            "gpus": gpus,
        },
        "software": {
            "app_version": app_version,
            "model_revision": model_revision,
            "artifact_digest": artifact_digest,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch_version) if torch_version is not None else None,
            "cuda": str(cuda_version) if cuda_version is not None else None,
            "nvidia_driver": _nvidia_driver_version() if gpus else None,
            "triton": _package_version("triton"),
            "onnxruntime": _package_version("onnxruntime"),
            "onnxruntime_gpu": _package_version("onnxruntime-gpu"),
            "numpy": _package_version("numpy"),
            "flagembedding": _package_version("FlagEmbedding"),
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
            "huggingface_hub": _package_version("huggingface-hub"),
        },
    }


def fingerprint_digest(fingerprint: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest of a hardware fingerprint.

    Args:
        fingerprint: JSON-serializable hardware and software identity.

    Returns:
        Lowercase SHA-256 hexadecimal text.
    """
    payload = json.dumps(
        fingerprint,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_profile(
    *,
    fingerprint: Mapping[str, Any],
    config: AutotuneConfig,
    selected: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    status: str = "complete",
    fallback_reason: Optional[str] = None,
    duration_seconds: float = 0.0,
) -> dict:
    """Create the persisted profile envelope."""
    return {
        "schema_version": AUTOTUNE_SCHEMA_VERSION,
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


def load_profile(path: Path) -> Optional[dict]:
    """Load a structurally valid BASIC hardware profile.

    Args:
        path: Profile JSON path.

    Returns:
        The profile, or ``None`` when it is absent, corrupt, or outdated.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != AUTOTUNE_SCHEMA_VERSION:
        return None
    if not isinstance(data.get("selected"), dict):
        return None
    return data


def profile_matches(
    profile: Optional[Mapping[str, Any]],
    fingerprint: dict,
    *,
    config: Optional[AutotuneConfig] = None,
) -> bool:
    """Check whether a completed BASIC profile matches the current run.

    Args:
        profile: Previously loaded profile, if present.
        fingerprint: Current hardware and software identity.
        config: Optional reuse-sensitive calibration contract.

    Returns:
        ``True`` only when the profile can be safely reused.
    """
    if not profile or profile.get("status") != "complete":
        return False
    if profile.get("fingerprint_sha256") != fingerprint_digest(fingerprint):
        return False
    if config is None:
        return True
    profile_config = profile.get("config")
    if not isinstance(profile_config, Mapping):
        return False
    return all(
        profile_config.get(name) == getattr(config, name)
        for name in (
            "mode",
            "objective",
            "runtime",
            "torch_compile",
            "torch_compile_mode",
            "onnx_execution_mode",
            "onnx_spin_policy",
        )
    )


class ProfileLock(AbstractContextManager):
    """Cross-platform lock based on exclusive file creation."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = 30.0,
        stale_seconds: float = 900.0,
        poll_seconds: float = 0.1,
    ):
        """Configure an exclusive lock file with bounded stale recovery.

        Args:
            path: Lock-file path.
            timeout_seconds: Maximum acquisition wait.
            stale_seconds: Age after which an abandoned lock may be reclaimed.
            poll_seconds: Delay between acquisition attempts.
        """
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds
        self.poll_seconds = poll_seconds
        self._acquired = False

    def __enter__(self) -> "ProfileLock":
        """Acquire the lock or raise after the configured timeout."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_seconds:
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for profile lock {self.path}"
                    )
                time.sleep(self.poll_seconds)
                continue
            try:
                os.write(
                    descriptor,
                    f"pid={os.getpid()} created={int(time.time())}\n".encode(),
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._acquired = True
            return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Release an acquired lock without suppressing body exceptions."""
        if self._acquired:
            with suppress(FileNotFoundError):
                self.path.unlink()
            self._acquired = False
        return None


def atomic_write_profile(path: Path, profile: Mapping[str, Any]) -> None:
    """Durably replace a profile without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                profile,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def save_profile_locked(
    path: Path,
    profile: Mapping[str, Any],
    *,
    timeout_seconds: float = 30.0,
) -> None:
    """Persist a BASIC profile atomically while holding its lock.

    Args:
        path: Destination profile path.
        profile: Complete JSON-serializable profile document.
        timeout_seconds: Maximum time allowed for lock acquisition.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    with ProfileLock(lock_path, timeout_seconds=timeout_seconds):
        atomic_write_profile(path, profile)


@dataclass(frozen=True)
class WorkloadScenario:
    """A deterministic calibration workload expressed in exact token lengths."""

    name: str
    token_lengths: tuple[int, ...]
    warmups: int
    repeats: int
    include_colbert: bool = False


def calibration_workloads(mode: str) -> list[WorkloadScenario]:
    """Return bounded quick/full workloads without loading a tokenizer."""
    resolved = normalise_choice(mode, "quick", AUTOTUNE_MODES)
    if resolved == "off":
        return []
    workloads = [
        WorkloadScenario("uniform-32", (32,), 1, 3),
        WorkloadScenario("uniform-128", (128,), 1, 3),
        WorkloadScenario("uniform-512", (512,), 1, 3),
        WorkloadScenario("uniform-1024", (1024,), 1, 2),
        WorkloadScenario("mixed", (32, 128, 512, 1024), 1, 3),
        WorkloadScenario("colbert-128", (128,), 1, 2, include_colbert=True),
    ]
    if resolved == "full":
        workloads.extend(
            [
                WorkloadScenario("uniform-2048", (2048,), 1, 2),
                WorkloadScenario("uniform-4096", (4096,), 0, 1),
                WorkloadScenario("uniform-8192", (8192,), 0, 1),
                WorkloadScenario("colbert-8192", (8192,), 0, 1, include_colbert=True),
            ]
        )
    return workloads


def coefficient_of_variation(values: list[float]) -> float:
    """Measure population variability across positive observations.

    Args:
        values: Candidate measurements; non-positive values are ignored.

    Returns:
        Population standard deviation divided by the mean, or zero with fewer
        than two usable observations.
    """
    positive = [float(value) for value in values if value > 0]
    if len(positive) < 2:
        return 0.0
    mean = statistics.fmean(positive)
    return statistics.pstdev(positive) / mean if mean else math.inf


def candidate_is_safe(
    result: Mapping[str, Any],
    *,
    max_memory_fraction: float,
    max_cv: float = 0.08,
) -> bool:
    """Apply hard correctness, stability and memory gates to a result."""
    if result.get("error") or result.get("timed_out"):
        return False
    if result.get("quality_pass") is not True:
        return False
    samples = result.get("tokens_per_second_samples", [])
    if not samples or any(float(value) <= 0 for value in samples):
        return False
    if coefficient_of_variation([float(value) for value in samples]) > max_cv:
        return False
    peak = result.get("peak_memory_bytes")
    limit = result.get("memory_limit_bytes")
    if peak is not None and limit and float(peak) / float(limit) > max_memory_fraction:
        return False
    return True


def _candidate_score(result: Mapping[str, Any], objective: str) -> tuple:
    samples = [float(value) for value in result["tokens_per_second_samples"]]
    throughput = statistics.median(samples)
    p95_ms = max(float(result.get("p95_latency_ms", 0.0)), 0.001)
    peak = float(result.get("peak_memory_bytes") or 0.0)
    if objective == "throughput":
        return (throughput, -p95_ms, -peak)
    if objective == "latency":
        return (-p95_ms, throughput, -peak)
    # The square-root latency penalty prevents a tiny batch from winning solely
    # on latency while still preferring materially more responsive candidates.
    return (throughput / math.sqrt(p95_ms), throughput, -peak)


def select_best_candidate(
    results: list[Mapping[str, Any]],
    *,
    objective: str,
    max_memory_fraction: float,
    max_cv: float = 0.08,
) -> Optional[dict]:
    """Select the best safe candidate and preserve its raw evidence."""
    resolved_objective = normalise_choice(objective, "balanced", AUTOTUNE_OBJECTIVES)
    safe = [
        result
        for result in results
        if candidate_is_safe(
            result,
            max_memory_fraction=max_memory_fraction,
            max_cv=max_cv,
        )
    ]
    if not safe:
        return None
    if resolved_objective == "balanced":
        throughputs = [
            statistics.median(
                [float(value) for value in item["tokens_per_second_samples"]]
            )
            for item in safe
        ]
        latencies = [
            max(float(item.get("p95_latency_ms", 0.0)), 0.001) for item in safe
        ]
        max_throughput = max(throughputs)
        min_latency = min(latencies)

        def balanced_score(item):
            throughput = statistics.median(
                [float(value) for value in item["tokens_per_second_samples"]]
            )
            latency = max(float(item.get("p95_latency_ms", 0.0)), 0.001)
            peak = float(item.get("peak_memory_bytes") or 0.0)
            score = 0.85 * (throughput / max_throughput) + 0.15 * (
                min_latency / latency
            )
            return (score, throughput, -peak)

        return dict(max(safe, key=balanced_score))
    return dict(max(safe, key=lambda item: _candidate_score(item, resolved_objective)))


def select_torch_compile(
    results: list[Mapping[str, Any]],
    *,
    objective: str,
    max_memory_fraction: float,
    max_cv: float = 0.08,
) -> tuple[bool, str]:
    """Promote Inductor only after a safe, material measured improvement."""
    eager_results = [item for item in results if item.get("torch_compile") is False]
    compiled_results = [item for item in results if item.get("torch_compile") is True]
    eager = select_best_candidate(
        eager_results,
        objective=objective,
        max_memory_fraction=max_memory_fraction,
        max_cv=max_cv,
    )
    compiled = select_best_candidate(
        compiled_results,
        objective=objective,
        max_memory_fraction=max_memory_fraction,
        max_cv=max_cv,
    )
    if eager is None:
        return False, "missing-safe-eager-baseline"
    if compiled is None:
        return False, "compiled-candidate-unsafe-or-unavailable"

    eager_throughput = statistics.median(
        [float(value) for value in eager["tokens_per_second_samples"]]
    )
    compiled_throughput = statistics.median(
        [float(value) for value in compiled["tokens_per_second_samples"]]
    )
    eager_latency = max(float(eager.get("p95_latency_ms", 0.0)), 0.001)
    compiled_latency = max(float(compiled.get("p95_latency_ms", 0.0)), 0.001)
    throughput_ratio = compiled_throughput / eager_throughput
    latency_ratio = compiled_latency / eager_latency
    resolved = normalise_choice(objective, "balanced", AUTOTUNE_OBJECTIVES)

    if resolved == "throughput":
        promoted = throughput_ratio >= 1.05 and latency_ratio <= 1.25
    elif resolved == "latency":
        promoted = latency_ratio <= 0.95 and throughput_ratio >= 0.95
    else:
        eager_utility = eager_throughput / math.sqrt(eager_latency)
        compiled_utility = compiled_throughput / math.sqrt(compiled_latency)
        promoted = (
            compiled_utility >= eager_utility * 1.03
            and throughput_ratio >= 0.95
            and latency_ratio <= 1.15
        )

    reason = "measured-material-gain" if promoted else "gain-below-promotion-threshold"
    return promoted, reason


@dataclass(frozen=True)
class IsolatedResult:
    """Captured outcome of one bounded calibration subprocess."""

    command: tuple[str, ...]
    returncode: Optional[int]
    duration_seconds: float
    timed_out: bool
    memory_exceeded: bool
    peak_rss_bytes: Optional[int]
    stdout: str
    stderr: str
    payload: Optional[dict]

    @property
    def ok(self) -> bool:
        """Return whether the worker completed with a JSON payload."""
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.memory_exceeded
            and isinstance(self.payload, dict)
        )


def _process_rss_bytes(pid: int, proc_root: Path = Path("/proc")) -> Optional[int]:
    status = _read_text(proc_root / str(pid) / "status")
    if not status:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def _decode_limited(handle, max_bytes: int) -> str:
    handle.seek(0)
    data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        data = data[:max_bytes] + b"\n...[truncated]"
    return data.decode("utf-8", errors="replace")


def _last_json_object(output: str) -> Optional[dict]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _stop_process(process: subprocess.Popen, grace_seconds: float = 1.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)


def run_isolated(
    command: list[str],
    *,
    timeout_seconds: float,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    max_rss_bytes: Optional[int] = None,
    poll_seconds: float = 0.05,
    max_output_bytes: int = 1_000_000,
    rss_reader=None,
) -> IsolatedResult:
    """Run one candidate in a bounded subprocess and parse its last JSON line."""
    if not command:
        raise ValueError("isolated command cannot be empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_rss_bytes is not None and max_rss_bytes <= 0:
        raise ValueError("max_rss_bytes must be positive")

    process_env = os.environ.copy()
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})
    read_rss = rss_reader or _process_rss_bytes
    started = time.monotonic()
    timed_out = False
    memory_exceeded = False
    peak_rss: Optional[int] = None

    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(cwd) if cwd else None,
            env=process_env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            shell=False,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                timed_out = True
                _stop_process(process)
                break
            try:
                rss = read_rss(process.pid)
            except (OSError, ValueError):
                rss = None
            if rss is not None:
                peak_rss = max(peak_rss or 0, int(rss))
                if max_rss_bytes is not None and rss > max_rss_bytes:
                    memory_exceeded = True
                    _stop_process(process)
                    break
            time.sleep(max(0.001, poll_seconds))

        if process.poll() is None:
            _stop_process(process)
        returncode = process.returncode
        stdout = _decode_limited(stdout_file, max_output_bytes)
        stderr = _decode_limited(stderr_file, max_output_bytes)

    duration = time.monotonic() - started
    return IsolatedResult(
        command=tuple(str(part) for part in command),
        returncode=returncode,
        duration_seconds=round(duration, 6),
        timed_out=timed_out,
        memory_exceeded=memory_exceeded,
        peak_rss_bytes=peak_rss,
        stdout=stdout,
        stderr=stderr,
        payload=_last_json_object(stdout),
    )


def power_of_two_candidates(limit: int, *, include: tuple[int, ...] = ()) -> list[int]:
    """Build an ordered power-of-two search space ending at ``limit``.

    Args:
        limit: Largest permitted positive candidate.
        include: Additional in-range values that must be retained.

    Returns:
        Sorted unique candidates, or an empty list for a non-positive limit.
    """
    if limit < 1:
        return []
    values = {value for value in include if 1 <= value <= limit}
    value = 1
    while value <= limit:
        values.add(value)
        value *= 2
    values.add(limit)
    return sorted(values)


def cpu_candidate_space(cpu: Mapping[str, Any], mode: str) -> dict:
    """Build a bounded CPU search space from effective container resources."""
    effective = max(1, int(cpu.get("effective_cpu_count") or 1))
    physical = min(effective, int(cpu.get("physical_core_count") or effective))
    # Logical workers sometimes help transformer kernels and sometimes only
    # contend with their SMT sibling. Keep both topology limits measurable.
    threads = power_of_two_candidates(effective, include=(physical, effective))
    batch_cap = 16 if mode == "full" else (8 if effective >= 8 else 4)
    batch_limit = min(batch_cap, max(1, effective * 2))
    batches = power_of_two_candidates(batch_limit)
    lane_limit = min(8 if mode == "full" else 4, effective)
    lanes = power_of_two_candidates(lane_limit)
    return {"threads": threads, "batches": batches, "lanes": lanes}


def _gpu_short_batch_limit(total_memory_bytes: int) -> int:
    memory_gib = total_memory_bytes / 1024**3
    if memory_gib >= 23:
        return 32
    if memory_gib >= 15:
        return 16
    if memory_gib >= 7.5:
        return 8
    if memory_gib >= 3.5:
        return 4
    return 2


def gpu_batch_candidates(
    total_memory_bytes: int, token_length: int, mode: str
) -> list[int]:
    """Return conservative powers of two for one GPU and token bucket."""
    short_limit = _gpu_short_batch_limit(total_memory_bytes)
    if token_length > 2048:
        limit = max(1, short_limit // 8)
    elif token_length > 1024:
        limit = max(1, short_limit // 4)
        if mode != "full" and total_memory_bytes < 15 * 1024**3:
            limit = 1
    elif token_length > 512:
        limit = max(1, short_limit // 2)
    else:
        limit = short_limit
    return power_of_two_candidates(limit)


@dataclass
class TuningOutcome:
    """Selected calibration policy, evidence, and optional fallback reason."""

    selected: dict
    results: list[dict]
    fallback_reason: Optional[str] = None


def _measure_candidate(
    benchmark: Callable[[dict], Mapping[str, Any]],
    candidate: dict,
) -> dict:
    try:
        measured = dict(benchmark(dict(candidate)))
    except Exception as exc:
        measured = {"error": f"{type(exc).__name__}: {exc}"}
    return {**candidate, **measured}


def _select_stage(
    results: list[dict],
    *,
    objective: str,
    memory_fraction: float,
    max_cv: float,
) -> Optional[dict]:
    return select_best_candidate(
        results,
        objective=objective,
        max_memory_fraction=memory_fraction,
        max_cv=max_cv,
    )


def _select_cpu_stage(
    results: list[dict],
    *,
    objective: str,
    memory_fraction: float,
    max_cv: float,
) -> Optional[dict]:
    """Avoid promoting extra CPU work for throughput gains inside noise."""
    best = _select_stage(
        results,
        objective=objective,
        memory_fraction=memory_fraction,
        max_cv=max_cv,
    )
    if best is None or objective != "throughput":
        return best
    safe = [
        item
        for item in results
        if candidate_is_safe(
            item,
            max_memory_fraction=memory_fraction,
            max_cv=max_cv,
        )
    ]
    best_throughput = statistics.median(
        [float(value) for value in best["tokens_per_second_samples"]]
    )
    equivalent = [
        item
        for item in safe
        if statistics.median(
            [float(value) for value in item["tokens_per_second_samples"]]
        )
        >= best_throughput / 1.03
    ]

    def complexity(item: Mapping[str, Any]) -> tuple:
        return (
            int(item.get("threads", 1)) * int(item.get("concurrency", 1)),
            int(item.get("batch_size", 1)),
            int(item.get("concurrency", 1)),
            0 if item.get("execution_mode", "sequential") == "sequential" else 1,
            0 if item.get("spin_policy", "default") == "default" else 1,
        )

    return dict(min(equivalent, key=complexity))


def _cpu_safe_selection() -> dict:
    return {
        "runtime": "torch",
        "precision": "fp32",
        "torch_compile": False,
        "torch_compile_reason": "safe-fallback",
        "threads": 1,
        "inference_concurrency": 1,
        "onnx_execution_mode": "sequential",
        "onnx_spin_policy": "default",
        "onnx_inter_op_threads": 1,
        "batch_policy": [{"max_tokens": 8192, "batch_size": 1}],
    }


def _benchmark_has_budget(benchmark: Callable, reserve_seconds: float) -> bool:
    remaining = getattr(benchmark, "remaining_seconds", None)
    if not callable(remaining):
        return True
    try:
        return float(remaining()) >= reserve_seconds
    except (TypeError, ValueError):
        return False


def tune_cpu(
    *,
    cpu: Mapping[str, Any],
    mode: str,
    objective: str,
    benchmark: Callable[[dict], Mapping[str, Any]],
    runtimes: tuple[str, ...] = ("torch",),
    onnx_execution_mode: str = "auto",
    onnx_spin_policy: str = "auto",
    memory_fraction: float = 0.80,
) -> TuningOutcome:
    """Tune the complete BASIC output with real ORT and CPU topologies."""
    space = cpu_candidate_space(cpu, mode)
    results: list[dict] = []
    effective = max(1, int(cpu.get("effective_cpu_count") or 1))
    # Quick exploration stays short so the mandatory 512-token production
    # canary always fits inside the startup budget. Full mode expands the
    # exploratory distribution and validates every production bucket below.
    mixed_lengths = [32, 128, 512, 1024] if mode == "full" else [32, 128]
    max_cv = 0.08 if mode == "full" else 0.15
    runtime_names = tuple(dict.fromkeys(runtimes)) or ("torch",)
    selected_runtime = runtime_names[0]
    selected_execution = (
        onnx_execution_mode
        if onnx_execution_mode in {"sequential", "parallel"}
        else "sequential"
    )
    selected_spin = (
        onnx_spin_policy
        if onnx_spin_policy in {"default", "bounded", "disabled"}
        else "default"
    )
    selected_inter_op = 1
    if len(runtime_names) > 1:
        runtime_results = []
        runtime_threads = min(4, effective)
        for runtime in runtime_names:
            candidate = {
                "device": "cpu",
                "stage": "runtime",
                "runtime": runtime,
                "threads": runtime_threads,
                "concurrency": 1,
                "batch_size": 1,
                "token_lengths": [128],
                "include_colbert": True,
                "execution_mode": selected_execution,
                "spin_policy": selected_spin,
                "inter_op_threads": selected_inter_op,
            }
            result = _measure_candidate(benchmark, candidate)
            runtime_results.append(result)
            results.append(result)
        best_runtime = _select_cpu_stage(
            runtime_results,
            objective=objective,
            memory_fraction=memory_fraction,
            max_cv=max_cv,
        )
        if best_runtime is not None:
            selected_runtime = str(best_runtime["runtime"])

    thread_results = []
    for threads in space["threads"]:
        candidate = {
            "device": "cpu",
            "stage": "threads",
            "runtime": selected_runtime,
            "threads": threads,
            "concurrency": 1,
            "batch_size": 1,
            "token_lengths": [128],
            "include_colbert": True,
            "execution_mode": selected_execution,
            "spin_policy": selected_spin,
            "inter_op_threads": selected_inter_op,
        }
        result = _measure_candidate(benchmark, candidate)
        thread_results.append(result)
        results.append(result)
    best_threads = _select_cpu_stage(
        thread_results,
        objective=objective,
        memory_fraction=memory_fraction,
        max_cv=max_cv,
    )
    if best_threads is None:
        return TuningOutcome(
            selected=_cpu_safe_selection(),
            results=results,
            fallback_reason="no safe CPU thread candidate",
        )

    batch_results = []
    for batch_size in space["batches"]:
        candidate = {
            "device": "cpu",
            "stage": "batch",
            "runtime": selected_runtime,
            "threads": int(best_threads["threads"]),
            "concurrency": 1,
            "batch_size": batch_size,
            "token_lengths": mixed_lengths,
            "include_colbert": True,
            "execution_mode": selected_execution,
            "spin_policy": selected_spin,
            "inter_op_threads": selected_inter_op,
        }
        result = _measure_candidate(benchmark, candidate)
        batch_results.append(result)
        results.append(result)
    best_batch = _select_cpu_stage(
        batch_results,
        objective=objective,
        memory_fraction=memory_fraction,
        max_cv=max_cv,
    )
    if best_batch is None:
        best_batch = {**best_threads, "batch_size": 1}

    lane_results = []
    for lanes in space["lanes"]:
        threads_per_lane = max(1, min(int(best_threads["threads"]), effective // lanes))
        candidate = {
            "device": "cpu",
            "stage": "concurrency",
            "runtime": selected_runtime,
            "threads": threads_per_lane,
            "concurrency": lanes,
            "batch_size": int(best_batch["batch_size"]),
            "token_lengths": mixed_lengths,
            "include_colbert": True,
            "execution_mode": selected_execution,
            "spin_policy": selected_spin,
            "inter_op_threads": selected_inter_op,
        }
        result = _measure_candidate(benchmark, candidate)
        lane_results.append(result)
        results.append(result)
    best_lanes = _select_cpu_stage(
        lane_results,
        objective=objective,
        memory_fraction=memory_fraction,
        max_cv=max_cv,
    )
    if best_lanes is None:
        best_lanes = {
            **best_batch,
            "threads": int(best_threads["threads"]),
            "concurrency": 1,
        }

    if selected_runtime.startswith("onnx-"):
        optional_topology_budget = mode == "full" or _benchmark_has_budget(
            benchmark, 50.0
        )
        execution_modes = (
            (onnx_execution_mode,)
            if onnx_execution_mode != "auto"
            else (
                ("sequential", "parallel")
                if effective >= 8 and optional_topology_budget
                else ("sequential",)
            )
        )
        execution_results = [
            {
                **best_lanes,
                "execution_mode": selected_execution,
                "spin_policy": selected_spin,
                "inter_op_threads": selected_inter_op,
            }
        ]
        for execution_mode in execution_modes:
            if execution_mode == selected_execution:
                continue
            inter_op_threads = (
                max(1, min(4, int(best_lanes["threads"]) // 2))
                if execution_mode == "parallel"
                else 1
            )
            candidate = {
                "device": "cpu",
                "stage": "onnx-execution",
                "runtime": selected_runtime,
                "threads": int(best_lanes["threads"]),
                "concurrency": int(best_lanes["concurrency"]),
                "batch_size": int(best_batch["batch_size"]),
                "token_lengths": mixed_lengths,
                "include_colbert": True,
                "execution_mode": execution_mode,
                "spin_policy": selected_spin,
                "inter_op_threads": inter_op_threads,
            }
            result = _measure_candidate(benchmark, candidate)
            execution_results.append(result)
            results.append(result)
        best_execution = _select_cpu_stage(
            execution_results,
            objective=objective,
            memory_fraction=memory_fraction,
            max_cv=max_cv,
        )
        if best_execution is not None:
            selected_execution = str(best_execution["execution_mode"])
            selected_inter_op = int(best_execution["inter_op_threads"])

        spin_policies = (
            ()
            if onnx_spin_policy != "auto"
            else (
                ("bounded", "disabled")
                if mode == "full"
                else (("bounded",) if _benchmark_has_budget(benchmark, 45.0) else ())
            )
        )
        spin_results = [
            {
                **(best_execution or best_lanes),
                "execution_mode": selected_execution,
                "spin_policy": selected_spin,
                "inter_op_threads": selected_inter_op,
            }
        ]
        for spin_policy in spin_policies:
            candidate = {
                "device": "cpu",
                "stage": "onnx-spin",
                "runtime": selected_runtime,
                "threads": int(best_lanes["threads"]),
                "concurrency": int(best_lanes["concurrency"]),
                "batch_size": int(best_batch["batch_size"]),
                "token_lengths": mixed_lengths,
                "include_colbert": True,
                "execution_mode": selected_execution,
                "spin_policy": spin_policy,
                "inter_op_threads": selected_inter_op,
            }
            result = _measure_candidate(benchmark, candidate)
            spin_results.append(result)
            results.append(result)
        best_spin = _select_cpu_stage(
            spin_results,
            objective=objective,
            memory_fraction=memory_fraction,
            max_cv=max_cv,
        )
        if best_spin is not None:
            selected_spin = str(best_spin["spin_policy"])

    tuned_batch = int(best_batch["batch_size"])
    batch_policy = [
        {"max_tokens": 128, "batch_size": tuned_batch},
        {"max_tokens": 512, "batch_size": tuned_batch},
        {"max_tokens": 1024, "batch_size": max(1, tuned_batch // 2)},
        {"max_tokens": 2048, "batch_size": max(1, tuned_batch // 4)},
        {"max_tokens": 8192, "batch_size": 1},
    ]
    # The selected topology has already passed the full BASIC output on the
    # short exploratory distribution. Quick mode reserves its remaining time
    # for the materially more expensive 512-token canary.
    canary_lengths = [512]
    if mode == "full":
        canary_lengths.extend([1024, 2048, 8192])
    for token_length in canary_lengths:
        policy_item = next(
            item for item in batch_policy if token_length <= item["max_tokens"]
        )
        batch_size = int(policy_item["batch_size"])
        while batch_size >= 1:
            candidate = {
                "device": "cpu",
                "stage": "full-output-canary",
                "runtime": selected_runtime,
                "threads": int(best_lanes["threads"]),
                "concurrency": int(best_lanes["concurrency"]),
                "batch_size": batch_size,
                "token_lengths": [token_length],
                "include_colbert": True,
                "execution_mode": selected_execution,
                "spin_policy": selected_spin,
                "inter_op_threads": selected_inter_op,
            }
            canary = _measure_candidate(benchmark, candidate)
            results.append(canary)
            if (
                _select_stage(
                    [canary],
                    objective=objective,
                    memory_fraction=memory_fraction,
                    max_cv=max_cv,
                )
                is not None
            ):
                policy_item["batch_size"] = batch_size
                break
            if batch_size == 1:
                return TuningOutcome(
                    selected=_cpu_safe_selection(),
                    results=results,
                    fallback_reason=(
                        f"CPU full-output canary failed at {token_length} tokens"
                    ),
                )
            batch_size = max(1, batch_size // 2)

    for index in range(1, len(batch_policy)):
        batch_policy[index]["batch_size"] = min(
            batch_policy[index]["batch_size"],
            batch_policy[index - 1]["batch_size"],
        )
    return TuningOutcome(
        selected={
            "runtime": selected_runtime,
            "precision": "fp32",
            "torch_compile": False,
            "torch_compile_reason": "cpu-default-off",
            "threads": int(best_lanes["threads"]),
            "inference_concurrency": int(best_lanes["concurrency"]),
            "onnx_execution_mode": selected_execution,
            "onnx_spin_policy": selected_spin,
            "onnx_inter_op_threads": selected_inter_op,
            "batch_policy": batch_policy,
        },
        results=results,
    )


def tune_gpu(
    *,
    gpu: Mapping[str, Any],
    mode: str,
    objective: str,
    benchmark: Callable[[dict], Mapping[str, Any]],
    precisions: tuple[str, ...] = ("fp16",),
    torch_compile: str = "auto",
    memory_fraction: float = 0.80,
) -> TuningOutcome:
    """Tune precision and a length-aware microbatch policy for one GPU."""
    total_memory = int(gpu.get("total_memory_bytes") or 0)
    results: list[dict] = []
    max_cv = 0.08 if mode == "full" else 0.15
    precision_results = []
    for precision in precisions:
        candidate = {
            "device": "cuda",
            "stage": "precision",
            "precision": precision,
            "batch_size": 1,
            "token_lengths": [32, 128, 512, 1024],
        }
        result = _measure_candidate(benchmark, candidate)
        precision_results.append(result)
        results.append(result)
    best_precision = _select_stage(
        precision_results,
        objective=objective,
        memory_fraction=memory_fraction,
        max_cv=max_cv,
    )
    if best_precision is None:
        return TuningOutcome(
            selected={
                "runtime": "torch",
                "precision": "fp16",
                "torch_compile": False,
                "torch_compile_reason": "safe-fallback",
                "inference_concurrency": 1,
                "batch_policy": [{"max_tokens": 8192, "batch_size": 1}],
            },
            results=results,
            fallback_reason="no safe GPU precision candidate",
        )

    precision = str(best_precision["precision"])
    buckets = (128, 512, 1024, 2048, 8192)
    policy = []
    for token_length in buckets:
        bucket_results = []
        for batch_size in gpu_batch_candidates(total_memory, token_length, mode):
            candidate = {
                "device": "cuda",
                "stage": "batch-policy",
                "precision": precision,
                "batch_size": batch_size,
                "token_lengths": [token_length],
            }
            result = _measure_candidate(benchmark, candidate)
            bucket_results.append(result)
            results.append(result)
        best = _select_stage(
            bucket_results,
            objective=objective,
            memory_fraction=memory_fraction,
            max_cv=max_cv,
        )
        policy.append(
            {
                "max_tokens": token_length,
                "batch_size": int(best["batch_size"]) if best else 1,
            }
        )

    # A dense+sparse probe underestimates the memory footprint of the default
    # basic payload. Recheck the chosen short and maximum-length buckets with
    # ColBERT enabled, reducing the batch if a hard gate rejects it. The long
    # probe is always batch 1 in quick mode and is intentionally isolated by the
    # calibration process so an unsupported device cannot take down the server.
    fallback_reasons = []
    for token_length in (128, 8192):
        policy_item = next(
            item for item in policy if item["max_tokens"] == token_length
        )
        chosen_batch = int(policy_item["batch_size"])
        canary_results = []
        for candidate_batch in reversed(power_of_two_candidates(chosen_batch)):
            candidate = {
                "device": "cuda",
                "stage": "full-output-canary",
                "precision": precision,
                "batch_size": candidate_batch,
                "token_lengths": [token_length],
                "include_colbert": True,
            }
            result = _measure_candidate(benchmark, candidate)
            canary_results.append(result)
            results.append(result)
            safe = _select_stage(
                [result],
                objective=objective,
                memory_fraction=memory_fraction,
                max_cv=max_cv,
            )
            if safe is not None:
                policy_item["batch_size"] = candidate_batch
                break
        else:
            policy_item["batch_size"] = 1
            fallback_reasons.append(
                f"no safe full-output GPU candidate at {token_length} tokens"
            )

    compile_policy = normalise_choice(
        torch_compile,
        "auto",
        AUTOTUNE_TORCH_COMPILE_POLICIES,
    )
    if compile_policy == "on":
        compile_enabled = True
        compile_reason = "explicit-on"
    elif compile_policy == "off":
        compile_enabled = False
        compile_reason = "explicit-off"
    else:
        representative = next(item for item in policy if item["max_tokens"] == 512)
        compile_results = []
        for enabled in (False, True):
            candidate = {
                "device": "cuda",
                "stage": "torch-compile",
                "runtime": "torch",
                "precision": precision,
                "torch_compile": enabled,
                "batch_size": int(representative["batch_size"]),
                "token_lengths": [512],
                "include_colbert": True,
            }
            result = _measure_candidate(benchmark, candidate)
            compile_results.append(result)
            results.append(result)
        compile_enabled, compile_reason = select_torch_compile(
            compile_results,
            objective=objective,
            max_memory_fraction=memory_fraction,
            max_cv=max_cv,
        )

    return TuningOutcome(
        selected={
            "runtime": "torch",
            "precision": precision,
            "torch_compile": compile_enabled,
            "torch_compile_reason": compile_reason,
            "inference_concurrency": 1,
            "batch_policy": policy,
        },
        results=results,
        fallback_reason="; ".join(fallback_reasons) or None,
    )
