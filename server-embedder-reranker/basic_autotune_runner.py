"""One-shot BASIC profile hardware calibration command.

The runner is intentionally separate from the FastAPI process. A failed or
memory-hungry calibration can be bounded by the parent/orchestrator while the
server always retains conservative defaults.

``BgeCalibrationBenchmark`` produces evidence and ``run()`` owns the complete
reuse, calibration, fallback, and persistence lifecycle.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import resource
except ImportError:  # pragma: no cover - Linux container always provides it
    resource = None

from basic_autotune import (
    AutotuneConfig,
    build_fingerprint,
    cpu_candidate_space,
    fingerprint_digest,
    gpu_batch_candidates,
    load_profile,
    make_profile,
    profile_matches,
    save_profile_locked,
    source_artifact_digest,
    tune_cpu,
    tune_gpu,
)
from basic_autotune_tokens import CalibrationTokenCounter
from basic_onnx_backend import BgeM3OnnxBackend
from model_revisions import MODEL_REVISIONS, pinned_model_id

BGE_MODEL = "BAAI/bge-m3"
DEFAULT_MAX_LENGTH = 8192


class LockedTokenizer:
    """Serialize tokenizer mutation while allowing concurrent model forwards."""

    def __init__(self, tokenizer: Any):
        """Wrap a tokenizer whose calls require mutual exclusion.

        Args:
            tokenizer: Underlying tokenizer; non-call attributes are delegated.
        """
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped tokenizer."""
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        """Invoke the wrapped tokenizer while holding its mutation lock."""
        with self._lock:
            return self._tokenizer(*args, **kwargs)


def _p95_ms(seconds: list[float]) -> float:
    if not seconds:
        return 0.0
    ordered = sorted(seconds)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index] * 1000.0


def _measurement_repeats(mode: str, device: str, stage: str) -> int:
    if mode == "full":
        if device == "cpu" and stage == "full-output-canary":
            return 1
        return 3
    stable_stages = {
        "threads",
        "runtime",
        "torch-compile",
    }
    if device == "cuda":
        stable_stages.add("batch-policy")
    return 2 if stage in stable_stages else 1


def _cpu_peak_rss_bytes() -> int:
    if resource is None:
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB while macOS reports bytes. Smart Embedder containers are
    # Linux; keep the branch correct for local development as well.
    return int(value if sys.platform == "darwin" else value * 1024)


def _dense_cosine(reference, candidate) -> float:
    import numpy as np

    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _sparse_top_keys(weights: Mapping[str, Any], limit: int = 10) -> tuple[str, ...]:
    ranked = sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
    return tuple(str(key) for key, _value in ranked[:limit])


def _sparse_diagnostics(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], limit: int = 10
) -> dict:
    reference_keys = _sparse_top_keys(reference, limit)
    candidate_keys = _sparse_top_keys(candidate, limit)
    all_keys = set(reference) | set(candidate)
    max_delta = max(
        (
            abs(float(reference.get(key, 0.0)) - float(candidate.get(key, 0.0)))
            for key in all_keys
        ),
        default=0.0,
    )
    return {
        "match": set(reference_keys) == set(candidate_keys),
        "rank_match": reference_keys == candidate_keys,
        "overlap": len(set(reference_keys) & set(candidate_keys)),
        "reference_keys": list(reference_keys),
        "candidate_keys": list(candidate_keys),
        "max_abs_delta": max_delta,
    }


class BgeCalibrationBenchmark:
    """Benchmark callable consumed by the pure hierarchical tuner."""

    def __init__(
        self,
        *,
        models,
        token_counter,
        torch_module,
        device: str,
        mode: str,
        compile_mode: str,
        memory_limit_bytes: Optional[int],
        deadline: float,
    ):
        """Prepare reference outputs and bounded benchmark resources.

        Args:
            models: Runtime-name mapping of compatible BGE backends.
            token_counter: Deterministic exact-token workload generator.
            torch_module: PyTorch module used for threads, CUDA, and compilation.
            device: Effective ``cpu`` or ``cuda`` device.
            mode: Quick or full calibration mode.
            compile_mode: Inductor mode for compiled GPU candidates.
            memory_limit_bytes: Effective memory safety limit, when known.
            deadline: Absolute monotonic calibration deadline.
        """
        self.models = dict(models)
        self.reference_model = self.models["torch"]
        self.token_counter = token_counter
        self.torch = torch_module
        self.device = device
        self.mode = mode
        self.compile_mode = compile_mode
        self.memory_limit_bytes = memory_limit_bytes
        self.deadline = deadline
        self.torch_eager_inner = (
            getattr(self.reference_model, "model", None)
            if self.device == "cuda"
            else None
        )
        self.torch_compiled_inner = None
        reference_text = token_counter.texts_at_token_length(1, 128)
        reference = self._encode(
            "torch",
            reference_text,
            1,
            False,
            torch_compile=False,
        )
        self.reference_by_shape = {
            (128, False): {
                "dense": reference["dense_vecs"][0],
                "sparse": reference["lexical_weights"][0],
                "colbert": None,
            }
        }

    def remaining_seconds(self) -> float:
        """Return the non-negative time remaining before calibration expires."""
        return max(0.0, self.deadline - time.monotonic())

    def _reference(self, text: str, token_length: int, include_colbert: bool):
        key = (token_length, include_colbert)
        if key not in self.reference_by_shape:
            output = self._encode(
                "torch",
                [text],
                1,
                include_colbert,
                torch_compile=False,
            )
            self.reference_by_shape[key] = {
                "dense": output["dense_vecs"][0],
                "sparse": output["lexical_weights"][0],
                "colbert": (output["colbert_vecs"][0] if include_colbert else None),
            }
        return self.reference_by_shape[key]

    def _encode(
        self,
        runtime: str,
        texts: list[str],
        batch_size: int,
        include_colbert: bool = False,
        *,
        torch_compile: bool = False,
    ):
        model = self.models[runtime]
        if runtime == "torch" and self.device == "cuda":
            if self.torch_eager_inner is None:
                raise RuntimeError("BGE torch model has no compilable inner model")
            if torch_compile and self.torch_compiled_inner is None:
                compile_fn = getattr(self.torch, "compile", None)
                if not callable(compile_fn):
                    raise RuntimeError("torch.compile is unavailable")
                self.torch_compiled_inner = compile_fn(
                    self.torch_eager_inner,
                    dynamic=True,
                    mode=self.compile_mode,
                )
            model.model = (
                self.torch_compiled_inner if torch_compile else self.torch_eager_inner
            )
        return model.encode(
            texts,
            batch_size=batch_size,
            max_length=DEFAULT_MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=include_colbert,
        )

    def __call__(self, candidate: dict) -> Mapping[str, Any]:
        """Measure one candidate against reference quality and safety gates.

        Args:
            candidate: Runtime, topology, batch, and workload selection.

        Returns:
            Serializable quality, throughput, latency, and memory evidence.
        """
        if time.monotonic() >= self.deadline:
            return {
                "timed_out": True,
                "quality_pass": False,
                "tokens_per_second_samples": [],
            }
        threads = max(1, int(candidate.get("threads", 1)))
        if self.device == "cpu":
            self.torch.set_num_threads(threads)
            runtime = str(candidate.get("runtime", "torch"))
            model = self.models[runtime]
            configure_session = getattr(model, "configure_session", None)
            if callable(configure_session):
                configure_session(
                    threads=threads,
                    execution_mode=str(candidate.get("execution_mode", "sequential")),
                    spin_policy=str(candidate.get("spin_policy", "default")),
                    inter_op_threads=max(1, int(candidate.get("inter_op_threads", 1))),
                )

        batch_size = max(1, int(candidate.get("batch_size", 1)))
        concurrency = max(1, int(candidate.get("concurrency", 1)))
        include_colbert = bool(candidate.get("include_colbert", False))
        token_lengths = [int(value) for value in candidate["token_lengths"]]
        # Mixed candidates must see the same distribution independently of the
        # microbatch being tested. Uniform bucket probes need one complete batch.
        sample_size = (
            max(batch_size, len(token_lengths) * 2)
            if len(token_lengths) > 1
            else batch_size
        )
        texts = self.token_counter.texts_at_token_pattern(sample_size, token_lengths)
        texts.sort(key=len)
        encoded = self.token_counter.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
            add_special_tokens=True,
        )["input_ids"]
        first_token_length = len(encoded[0])
        reference = self._reference(texts[0], first_token_length, include_colbert)
        tokens_per_lane = sum(len(input_ids) for input_ids in encoded)

        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

        # Candidate-specific warm-up. Quick GPU batch selection gets two
        # measured iterations so a transient cannot choose the production
        # token bucket; CPU batch probes remain bounded by the startup budget.
        runtime = str(candidate.get("runtime", "torch"))
        compile_enabled = bool(candidate.get("torch_compile", False))
        warmup_started = time.perf_counter()
        self._encode(
            runtime,
            texts,
            batch_size,
            include_colbert,
            torch_compile=compile_enabled,
        )
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_started
        repeats = _measurement_repeats(
            self.mode,
            self.device,
            str(candidate.get("stage", "")),
        )
        durations = []
        outputs = None
        for _ in range(repeats):
            if time.monotonic() >= self.deadline:
                return {
                    "timed_out": True,
                    "quality_pass": False,
                    "tokens_per_second_samples": [],
                }
            started = time.perf_counter()
            if concurrency == 1:
                outputs = [
                    self._encode(
                        runtime,
                        texts,
                        batch_size,
                        include_colbert,
                        torch_compile=compile_enabled,
                    )
                ]
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(
                            self._encode,
                            runtime,
                            texts,
                            batch_size,
                            include_colbert,
                            torch_compile=compile_enabled,
                        )
                        for _ in range(concurrency)
                    ]
                    outputs = [future.result() for future in futures]
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)

        total_tokens = tokens_per_lane * concurrency
        throughput = [total_tokens / duration for duration in durations]
        cosine = _dense_cosine(
            reference["dense"],
            outputs[0]["dense_vecs"][0],
        )
        sparse = _sparse_diagnostics(
            reference["sparse"], outputs[0]["lexical_weights"][0]
        )
        colbert_cosine = None
        if include_colbert:
            colbert_cosine = _dense_cosine(
                reference["colbert"], outputs[0]["colbert_vecs"][0]
            )
        if self.device == "cuda":
            memory_limit = int(self.torch.cuda.get_device_properties(0).total_memory)
            peak_memory = int(self.torch.cuda.max_memory_allocated())
            try:
                free_memory, total_memory = self.torch.cuda.mem_get_info()
                peak_memory = max(peak_memory, int(total_memory - free_memory))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        else:
            peak_memory = _cpu_peak_rss_bytes()
            memory_limit = self.memory_limit_bytes

        return {
            "quality_pass": (
                cosine >= 0.99999
                and sparse["match"]
                and sparse["max_abs_delta"] <= 0.001
                and (colbert_cosine is None or colbert_cosine >= 0.999)
            ),
            "dense_cosine": cosine,
            "sparse_top10_match": sparse["match"],
            "sparse_top10_rank_match": sparse["rank_match"],
            "sparse_top10_overlap": sparse["overlap"],
            "sparse_top10_reference": sparse["reference_keys"],
            "sparse_top10_candidate": sparse["candidate_keys"],
            "sparse_max_abs_delta": sparse["max_abs_delta"],
            "colbert_cosine": colbert_cosine,
            "include_colbert": include_colbert,
            "tokens_per_second_samples": throughput,
            "warmup_seconds": warmup_seconds,
            "p95_latency_ms": _p95_ms(durations),
            "peak_memory_bytes": peak_memory,
            "memory_limit_bytes": memory_limit,
        }


def safe_defaults(fingerprint: Mapping[str, Any]) -> dict:
    """Return a conservative BASIC policy for the detected device.

    Args:
        fingerprint: Hardware fingerprint containing CPU and optional GPU data.

    Returns:
        A minimal Torch policy that does not depend on measured candidates.
    """
    hardware = fingerprint.get("hardware", {})
    gpus = hardware.get("gpus") or []
    if gpus:
        return {
            "runtime": "torch",
            "precision": "fp16",
            "torch_compile": False,
            "torch_compile_reason": "safe-default",
            "threads": None,
            "inference_concurrency": 1,
            "batch_policy": [
                {"max_tokens": 512, "batch_size": 1},
                {"max_tokens": 8192, "batch_size": 1},
            ],
        }
    cpu = hardware.get("cpu", {})
    threads = min(4, max(1, int(cpu.get("effective_cpu_count") or 1)))
    return {
        "runtime": "torch",
        "precision": "fp32",
        "torch_compile": False,
        "torch_compile_reason": "cpu-default-off",
        "threads": threads,
        "inference_concurrency": 1,
        "onnx_execution_mode": "sequential",
        "onnx_spin_policy": "default",
        "onnx_inter_op_threads": 1,
        "batch_policy": [{"max_tokens": 8192, "batch_size": 1}],
    }


def synthetic_benchmark(candidate: Mapping[str, Any]) -> dict:
    """Deterministic benchmark used by dry-run and orchestration tests."""
    device = candidate["device"]
    batch = int(candidate.get("batch_size", 1))
    threads = int(candidate.get("threads", 1))
    concurrency = int(candidate.get("concurrency", 1))
    length = max(int(value) for value in candidate["token_lengths"])
    if device == "cuda":
        throughput = 4000.0 * batch / (1.0 + length / 2048.0)
        memory_fraction = min(0.95, 0.18 + 0.07 * batch + length / 65536.0)
    else:
        thread_gain = min(threads, 4) / (1.0 + max(0, threads - 4) * 0.2)
        batch_gain = min(batch, 4) ** 0.30
        lane_penalty = 1.0 if concurrency == 1 else 0.85
        throughput = 95.0 * thread_gain * batch_gain * lane_penalty
        memory_fraction = min(0.95, 0.20 + 0.04 * batch * concurrency)
    latency = max(1.0, length * batch / max(throughput, 1.0) * 1000.0)
    return {
        "quality_pass": True,
        "tokens_per_second_samples": [throughput, throughput * 1.01],
        "p95_latency_ms": latency,
        "peak_memory_bytes": int(memory_fraction * 1000),
        "memory_limit_bytes": 1000,
    }


def _simulate_fingerprint(device: str) -> dict:
    fingerprint = build_fingerprint(
        app_version="simulation",
        model_revision=pinned_model_id(BGE_MODEL),
    )
    simulated_gpus = {
        "gpu": ("Simulated RTX 4060", "8.9", 8, 24),
        "rtx5080": ("Simulated RTX 5080", "12.0", 16, 84),
        "rtx5090": ("Simulated RTX 5090", "12.0", 32, 170),
    }
    if device in simulated_gpus:
        name, capability, memory_gib, processors = simulated_gpus[device]
        fingerprint["hardware"]["gpus"] = [
            {
                "index": 0,
                "name": name,
                "uuid": "SIMULATED",
                "compute_capability": capability,
                "total_memory_bytes": memory_gib * 1024**3,
                "multi_processor_count": processors,
            }
        ]
    else:
        fingerprint["hardware"]["gpus"] = []
        fingerprint["hardware"]["cpu"]["effective_cpu_count"] = 4
        fingerprint["hardware"]["cpu"]["physical_core_count"] = 4
        fingerprint["hardware"]["memory_limit_bytes"] = 16 * 1024**3
    return fingerprint


def _calibrate_with_benchmark(
    fingerprint: dict,
    config: AutotuneConfig,
    benchmark,
    runtimes: tuple[str, ...] = ("torch",),
) -> tuple[dict, list[dict], Optional[str]]:
    hardware = fingerprint["hardware"]
    if hardware["gpus"]:
        outcome = tune_gpu(
            gpu=hardware["gpus"][0],
            mode=config.mode,
            objective=config.objective,
            benchmark=benchmark,
            torch_compile=config.torch_compile,
        )
    else:
        outcome = tune_cpu(
            cpu=hardware["cpu"],
            mode=config.mode,
            objective=config.objective,
            benchmark=benchmark,
            runtimes=runtimes,
            onnx_execution_mode=config.onnx_execution_mode,
            onnx_spin_policy=config.onnx_spin_policy,
        )
    return outcome.selected, outcome.results, outcome.fallback_reason


def run(args: argparse.Namespace) -> dict:
    """Run or simulate one complete BASIC calibration lifecycle.

    Args:
        args: Parsed calibration command-line arguments.

    Returns:
        Reuse, dry-run, calibrated, or fallback result for the pre-start process.
    """
    config = AutotuneConfig.from_env()
    if args.mode:
        config = replace(config, mode=args.mode)
    if args.objective:
        config = replace(config, objective=args.objective)
    if args.profile:
        config = replace(config, profile_path=args.profile)
    if args.force:
        config = replace(config, force=True)

    if args.simulate:
        fingerprint = _simulate_fingerprint(args.simulate)
        torch_module = None
    else:
        import torch

        torch_module = torch
        fingerprint = build_fingerprint(
            torch_module=torch,
            app_version=args.app_version,
            model_revision=pinned_model_id(args.model_revision),
            artifact_digest=source_artifact_digest(),
        )

    if args.dry_run:
        hardware = fingerprint["hardware"]
        search_space = (
            {
                str(length): gpu_batch_candidates(
                    hardware["gpus"][0]["total_memory_bytes"],
                    length,
                    config.mode,
                )
                for length in (128, 512, 1024, 2048, 8192)
            }
            if hardware["gpus"]
            else cpu_candidate_space(hardware["cpu"], config.mode)
        )
        return {
            "action": "dry-run",
            "profile_path": config.profile_path,
            "fingerprint_sha256": fingerprint_digest(fingerprint),
            "safe_defaults": safe_defaults(fingerprint),
            "search_space": search_space,
        }

    existing = load_profile(Path(config.profile_path))
    if not config.force and profile_matches(existing, fingerprint, config=config):
        return {
            "action": "reused",
            "profile_path": config.profile_path,
            "fingerprint_sha256": fingerprint_digest(fingerprint),
            "selected": existing["selected"],
        }
    if config.mode == "off":
        return {
            "action": "disabled",
            "profile_path": config.profile_path,
            "fingerprint_sha256": fingerprint_digest(fingerprint),
            "selected": safe_defaults(fingerprint),
        }

    started = time.monotonic()
    fallback_reason = None
    results: list[dict] = []
    try:
        if args.simulate:
            benchmark = synthetic_benchmark
        else:
            from FlagEmbedding import BGEM3FlagModel
            from huggingface_hub import snapshot_download

            device = "cuda" if fingerprint["hardware"]["gpus"] else "cpu"
            if device == "cpu":
                effective = fingerprint["hardware"]["cpu"]["effective_cpu_count"]
                torch_module.set_num_threads(max(1, int(effective)))
                torch_module.set_num_interop_threads(1)
            model_snapshot = snapshot_download(
                repo_id=args.model_revision,
                revision=MODEL_REVISIONS.get(args.model_revision),
            )
            model = BGEM3FlagModel(
                model_snapshot,
                device=device,
                use_fp16=device == "cuda",
            )
            model.tokenizer = LockedTokenizer(model.tokenizer)
            counter = CalibrationTokenCounter(
                tokenizer=model.tokenizer,
                model_name=pinned_model_id(args.model_revision),
                max_length=DEFAULT_MAX_LENGTH,
            )
            models = {"torch": model}
            runtimes = ("torch",)
            if device == "cpu" and config.runtime != "torch":
                try:
                    import onnxruntime

                    snapshot_path = model_snapshot
                    models["onnx-fp32"] = BgeM3OnnxBackend(
                        snapshot_path=snapshot_path,
                        tokenizer=model.tokenizer,
                        torch_module=torch_module,
                        threads=max(1, int(effective)),
                        execution_mode=(
                            config.onnx_execution_mode
                            if config.onnx_execution_mode != "auto"
                            else "sequential"
                        ),
                        spin_policy=(
                            config.onnx_spin_policy
                            if config.onnx_spin_policy != "auto"
                            else "default"
                        ),
                        max_length=DEFAULT_MAX_LENGTH,
                        ort_module=onnxruntime,
                    )
                    runtimes = (
                        ("onnx-fp32",)
                        if config.runtime == "onnx-fp32"
                        else ("torch", "onnx-fp32")
                    )
                except Exception:
                    if config.runtime == "onnx-fp32":
                        raise
            benchmark = BgeCalibrationBenchmark(
                models=models,
                token_counter=counter,
                torch_module=torch_module,
                device=device,
                mode=config.mode,
                compile_mode=config.torch_compile_mode,
                memory_limit_bytes=fingerprint["hardware"]["memory_limit_bytes"],
                deadline=started + config.max_seconds,
            )
        selected, results, fallback_reason = _calibrate_with_benchmark(
            fingerprint,
            config,
            benchmark,
            runtimes=runtimes if not args.simulate else ("torch",),
        )
    except Exception as exc:
        selected = safe_defaults(fingerprint)
        fallback_reason = f"{type(exc).__name__}: {exc}"

    profile = make_profile(
        fingerprint=fingerprint,
        config=config,
        selected=selected,
        results=results,
        fallback_reason=fallback_reason,
        duration_seconds=time.monotonic() - started,
    )
    save_profile_locked(Path(config.profile_path), profile)
    return {
        "action": "calibrated" if fallback_reason is None else "fallback",
        "profile_path": config.profile_path,
        "fingerprint_sha256": fingerprint_digest(fingerprint),
        "selected": selected,
        "fallback_reason": fallback_reason,
        "duration_seconds": profile["duration_seconds"],
        "candidate_count": len(results),
    }


def parse_args(argv=None) -> argparse.Namespace:
    """Parse BASIC calibration command-line arguments.

    Args:
        argv: Optional argument sequence. Uses process arguments when omitted.

    Returns:
        Parsed runner options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "quick", "full"))
    parser.add_argument("--objective", choices=("balanced", "throughput", "latency"))
    parser.add_argument("--profile")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--simulate", choices=("cpu", "gpu", "rtx5080", "rtx5090"))
    parser.add_argument("--app-version", default="1.2.0")
    parser.add_argument("--model-revision", default=BGE_MODEL)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run BASIC calibration and print its JSON result.

    Args:
        argv: Optional argument sequence. Uses process arguments when omitted.

    Returns:
        Zero after emitting the result.
    """
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
