"""Pre-start calibration runner for the co-resident ADVANCED model stack.

``calibrate()`` evaluates bounded candidates and ``main()`` exposes that
lifecycle to the isolated pre-start process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from advanced_autotune import (
    AdvancedAutotuneConfig,
    advanced_profile_matches,
    build_advanced_fingerprint,
    cpu_power_warnings,
    gpu_wait_diagnosis,
    load_advanced_profile,
    make_advanced_profile,
    models_are_cached,
    read_cpu_power_policy,
    safe_advanced_defaults,
    save_advanced_profile,
)
from advanced_autotune_quality import (
    QUALITY_SUITE_VERSION,
    evaluate_dense,
    evaluate_reranker,
    evaluate_sparse,
)
from basic_autotune import IsolatedResult, run_isolated

APP_VERSION = "1.2.0"


def _device_kind(fingerprint: Mapping[str, Any]) -> str:
    return "cuda" if fingerprint.get("hardware", {}).get("gpus") else "cpu"


def _supports_bf16(fingerprint: Mapping[str, Any]) -> bool:
    if _device_kind(fingerprint) == "cuda":
        capability = fingerprint["hardware"]["gpus"][0].get("compute_capability", "0.0")
        try:
            return int(str(capability).split(".", 1)[0]) >= 8
        except ValueError:
            return False
    instruction_sets = set(
        fingerprint.get("hardware", {}).get("cpu", {}).get("instruction_sets", [])
    )
    return bool({"avx512_bf16", "amx_bf16", "bf16"} & instruction_sets)


def _cpu_onnx_availability(fingerprint: Mapping[str, Any]) -> tuple[bool, str]:
    software = fingerprint.get("software", {})
    required = ("onnx", "onnxscript", "onnxruntime")
    missing = [name for name in required if not software.get(name)]
    if missing:
        return False, "missing CPU export/runtime packages: " + ", ".join(missing)
    dense = fingerprint.get("models", {}).get("dense", {})
    if not dense.get("cached"):
        return False, "Qwen3 dense snapshot is not cached"
    return True, "local Qwen3 export is generated and validated on first use"


def advanced_search_space(fingerprint: Mapping[str, Any], mode: str) -> dict:
    """Build bounded candidates from effective hardware, never model names."""
    device = _device_kind(fingerprint)
    full = mode == "full"
    if device == "cuda":
        total = int(fingerprint["hardware"]["gpus"][0]["total_memory_bytes"])
        short_batch = (
            32
            if total >= 23 * 1024**3
            else (16 if total >= 15 * 1024**3 else 8 if total >= 7.5 * 1024**3 else 2)
        )
        precisions = ["fp16"] + (["bf16"] if _supports_bf16(fingerprint) else [])
        attention = ["sdpa", "eager"]
        if fingerprint.get("software", {}).get("flash_attn"):
            attention.append("flash_attention_2")
        compile_modes = ["off"] + (["default", "reduce-overhead"] if full else [])
        return {
            "device": device,
            "dense": {
                "runtime": ["torch"],
                "precision": precisions,
                "attention": attention,
                "torch_compile": compile_modes,
                "batch": [1, 2, 4, short_batch],
            },
            "sparse": {
                "runtime": ["torch"],
                "precision": precisions,
                "torch_compile": compile_modes,
                "batch": [1, 2, 4, short_batch],
            },
            "schedule": ["sequential", "overlap"],
            "reranker": {
                "runtime": ["torch"],
                "precision": precisions,
                "attention": attention,
                "torch_compile": compile_modes,
                "batch": [1, 2, 4, short_batch],
            },
        }

    cpu = fingerprint.get("hardware", {}).get("cpu", {})
    effective = max(1, int(cpu.get("effective_cpu_count") or 1))
    threads = {1, min(2, effective), min(4, effective), effective}
    if full and cpu.get("smt_active"):
        threads.add(effective * 2)
    inter_op_threads = {1}
    if full and effective >= 4:
        inter_op_threads.add(2)
    if full and effective >= 8:
        inter_op_threads.add(min(4, effective // 2))
    affinity = ["none"]
    if full:
        affinity.append("compact")
        if (
            int(cpu.get("socket_count") or 1) > 1
            or int(cpu.get("numa_node_count") or 1) > 1
        ):
            affinity.append("socket-local")
    onnx_available, onnx_reason = _cpu_onnx_availability(fingerprint)
    cpu_model = str(cpu.get("model") or "").lower()
    intel_cpu = "intel" in cpu_model
    runtime_availability = {
        "torch": {"available": True, "reason": "native model snapshot"},
        "onnx-fp32": {
            "available": onnx_available,
            "reason": onnx_reason,
        },
        "onnx-int8": {
            "available": onnx_available,
            "reason": onnx_reason,
        },
        "openvino": {
            "available": False,
            "reason": (
                "not applicable: OpenVINO is considered only on Intel CPUs"
                if not intel_cpu
                else (
                    "OpenVINO package and validated Qwen3 export required"
                    if not fingerprint.get("software", {}).get("openvino")
                    else "no validated Qwen3 OpenVINO backend"
                )
            ),
        },
    }
    precisions = ["fp32"] + (["bf16"] if _supports_bf16(fingerprint) else [])
    return {
        "device": device,
        "dense": {
            "runtime": [
                runtime
                for runtime in ("torch", "onnx-fp32", "onnx-int8")
                if runtime_availability[runtime]["available"]
            ],
            "runtime_availability": runtime_availability,
            "precision": precisions,
            "threads": sorted(threads),
            "batch": [1, 2, 4],
        },
        "sparse": {
            "runtime": ["torch"],
            "precision": precisions,
            "threads": sorted(threads),
            "batch": [1, 2, 4],
        },
        "schedule": ["sequential", "parallel"],
        "cpu_topology": {
            "inter_op_threads": sorted(inter_op_threads),
            "worker_spin": (["default", "passive", "active"] if full else ["default"]),
            "affinity": affinity,
            "oversubscription": bool(full and cpu.get("smt_active")),
        },
        "reranker": {
            "runtime": ["torch"],
            "precision": precisions,
            "threads": sorted(threads),
            "batch": [1, 2, 4],
        },
    }


def _power_of_two_floor(value: int) -> int:
    return 1 << (max(1, int(value)).bit_length() - 1)


#: Padded tokens per microbatch that the conservative serving default allows.
BASE_BATCH_TOKENS = 8192


def _vram_multiplier(fingerprint: Mapping[str, Any]) -> int:
    """Return the GPU memory tier shared by workloads, batch caps and token caps."""
    total = int(fingerprint["hardware"]["gpus"][0]["total_memory_bytes"])
    return 4 if total >= 23 * 1024**3 else (2 if total >= 15 * 1024**3 else 1)


def _batch_token_cap(fingerprint: Mapping[str, Any]) -> int:
    """Return the largest padded-token microbatch a calibrated policy may use."""
    if _device_kind(fingerprint) != "cuda":
        return BASE_BATCH_TOKENS
    return BASE_BATCH_TOKENS * _vram_multiplier(fingerprint)


def _gpu_geometric_batch_maps(
    fingerprint: Mapping[str, Any], lengths: list[int]
) -> dict[str, dict[str, int]]:
    """Build padded-token-equivalent request batches from measured VRAM.

    The safe serving policy may split these requests into smaller microbatches;
    a candidate can therefore be compared on the exact same external workload.
    Larger GPUs expand geometrically without claiming physical performance.
    """
    multiplier = _vram_multiplier(fingerprint)

    def policy(token_budget: int, batch_cap: int) -> dict[str, int]:
        return {
            str(length): min(
                batch_cap,
                _power_of_two_floor(max(1, token_budget // length)),
            )
            for length in lengths
        }

    embedding_budget = 8192 * multiplier
    constrained_budget = 4096 * multiplier
    return {
        "dense": policy(embedding_budget, 64 * multiplier),
        "sparse": policy(embedding_budget, 64 * multiplier),
        "hybrid": policy(constrained_budget, 32 * multiplier),
        "reranker": policy(constrained_budget, 32 * multiplier),
        "mixed": policy(constrained_budget, 32 * multiplier),
    }


def baseline_candidate(
    fingerprint: Mapping[str, Any], config: AdvancedAutotuneConfig
) -> dict:
    """Build the conservative reference candidate for the active device.

    Args:
        fingerprint: Effective hardware and software identity.
        config: ADVANCED calibration mode and objective.

    Returns:
        A co-resident baseline with bounded workloads and safe defaults.
    """
    device_kind = _device_kind(fingerprint)
    if device_kind == "cpu":
        # Long-context CPU attention can consume the entire documented FULL
        # budget (and, with all ADVANCED models resident, breach the RSS gate)
        # before any runtime is compared.  Tune on two representative regular
        # buckets, then require one isolated 1k/2k/8k canary only for the
        # selected backend(s).
        lengths = [128, 512]
    else:
        lengths = [128, 512, 1024, 2048, 8192]
    if device_kind == "cuda":
        batches = _gpu_geometric_batch_maps(fingerprint, lengths)
    else:
        batches = {
            backend: {str(length): (2 if length <= 512 else 1) for length in lengths}
            for backend in ("dense", "sparse", "hybrid", "reranker", "mixed")
        }
    repeats = 2 if config.mode == "quick" else 3
    if device_kind == "cpu":
        repeats = 1 if config.mode == "quick" else 2
    target_backends = ["dense", "sparse", "hybrid", "reranker"]
    if device_kind == "cuda" or config.mode == "full":
        target_backends.append("mixed")
    candidate = {
        "name": "co-resident-safe-baseline",
        "selected": safe_advanced_defaults(_device_kind(fingerprint)),
        "token_lengths": lengths,
        "repeats": repeats,
        "benchmark_batches": batches,
        "target_backends": target_backends,
    }
    if "mixed" in target_backends:
        candidate["backend_token_lengths"] = {
            "mixed": (
                [
                    length
                    for length in ((128, 8192) if config.mode == "full" else (128,))
                    if length in lengths
                ]
                if device_kind == "cuda"
                else [128]
            )
        }
    if device_kind == "cuda":
        candidate["target_backends"].append("requests")
        candidate.setdefault("backend_token_lengths", {})["requests"] = [128]
    if device_kind == "cpu":
        canary_buckets = (1024, 2048, 8192)
        max_lengths = {
            "dense": _configured_limit(fingerprint, "QWEN_EMBED_MAX_LENGTH", 32768),
            "sparse": _configured_limit(fingerprint, "MILCO_MAX_LENGTH", 8192),
            "reranker": _configured_limit(fingerprint, "QWEN_RERANK_MAX_LENGTH", 2048),
        }
        by_backend = {
            backend: [length for length in canary_buckets if length <= max_length]
            for backend, max_length in max_lengths.items()
        }
        candidate["deferred_canary_token_lengths_by_backend"] = by_backend
        candidate["deferred_canary_token_lengths"] = sorted(
            {length for lengths in by_backend.values() for length in lengths}
        )
    return candidate


def _configured_limit(fingerprint: Mapping[str, Any], name: str, default: int) -> int:
    explicit = fingerprint.get("limits_and_policy", {}).get("explicit_env", {})
    try:
        return max(1, int(explicit.get(name, default)))
    except (TypeError, ValueError):
        return default


def _bucket_policy(
    max_length: int,
    batch_sizes: Mapping[int, int],
    *,
    max_batch_tokens: int = BASE_BATCH_TOKENS,
) -> tuple[list[dict], list[dict]]:
    thresholds = sorted({length for length in batch_sizes if length <= max_length})
    if not thresholds or thresholds[-1] < max_length:
        thresholds.append(max_length)
    batches = []
    token_budgets = []
    last_batch = 1
    for threshold in thresholds:
        eligible = [length for length in batch_sizes if length <= threshold]
        if eligible:
            last_batch = int(batch_sizes[max(eligible)])
        batches.append({"max_tokens": threshold, "batch_size": last_batch})
        token_budgets.append(
            {
                "max_tokens": threshold,
                "max_batch_tokens": max(
                    threshold, min(max_batch_tokens, threshold * last_batch)
                ),
            }
        )
    return batches, token_budgets


def _geometric_caps(start: int, maximum: int) -> list[int]:
    """Return the strictly increasing power-of-two probes after ``start``."""
    current = _power_of_two_floor(max(1, start))
    limit = _power_of_two_floor(max(current, maximum))
    caps = []
    while current < limit:
        current *= 2
        caps.append(current)
    return caps


def _capped_batch_map(batch_sizes: Mapping[int, int], cap: int) -> dict[int, int]:
    return {
        int(length): min(max(1, int(size)), max(1, int(cap)))
        for length, size in batch_sizes.items()
    }


def _geometric_batch_family_specs(
    *,
    backend: str,
    target_backends: list[str],
    batch_sizes: Mapping[int, int],
    max_length: int,
    start_cap: int,
    max_batch_tokens: int,
) -> list[dict]:
    """Build one measurable candidate per geometric batch expansion.

    The conservative serving default already covers the 1, 2, 4, ... stages
    through ``start_cap``. FULL mode continues one power of two at a time so a
    plateau, quality failure, OOM, or memory rejection can stop only this
    backend family before a larger and less safe value is launched.
    """
    maximum = max((int(value) for value in batch_sizes.values()), default=1)
    previous = _capped_batch_map(batch_sizes, start_cap)
    specs = []
    for cap in _geometric_caps(start_cap, maximum):
        current = _capped_batch_map(batch_sizes, cap)
        changed_lengths = sorted(
            length
            for length, size in current.items()
            if length <= max_length and size > previous.get(length, 1)
        )
        previous = current
        if not changed_lengths:
            continue
        policy, token_policy = _bucket_policy(
            max_length, current, max_batch_tokens=max_batch_tokens
        )
        specs.append(
            {
                "name": f"{backend}-geometric-batch-{cap}",
                "target_backends": list(target_backends),
                "promotion_scope": {
                    target: list(changed_lengths) for target in target_backends
                },
                "search_family": f"{backend}-batch",
                "search_value": cap,
                "stop_family_on_rejection": True,
                "changes": {
                    backend: {
                        "batch_policy": policy,
                        "max_batch_tokens_policy": token_policy,
                    }
                },
            }
        )
    return specs


def tuning_candidate_specs(
    fingerprint: Mapping[str, Any], config: AdvancedAutotuneConfig
) -> list[dict]:
    """Return staged mutations; each is measured in a fresh co-resident process."""
    device = _device_kind(fingerprint)
    specs: list[dict] = []
    if device == "cuda" and _supports_bf16(fingerprint):
        specs.extend(
            [
                {
                    "name": "sparse-bf16-weights",
                    "target_backends": ["sparse", "hybrid"],
                    "changes": {"sparse": {"precision": "bf16"}},
                },
                {
                    "name": "dense-bf16-sdpa",
                    "target_backends": ["dense", "hybrid"],
                    "changes": {"dense": {"precision": "bf16"}},
                },
                {
                    "name": "reranker-bf16-sdpa",
                    "target_backends": ["reranker"],
                    "changes": {"reranker": {"precision": "bf16"}},
                },
            ]
        )
    if device == "cpu":
        cpu = fingerprint.get("hardware", {}).get("cpu", {})
        effective = max(1, int(cpu.get("effective_cpu_count") or 1))
        safe_threads = min(4, effective)
        cpu_space = advanced_search_space(fingerprint, config.mode)
        for threads in cpu_space["dense"]["threads"]:
            if threads == safe_threads:
                continue
            oversubscribed = threads > effective
            specs.append(
                {
                    "name": (
                        f"cpu-intra-op-{threads}-oversubscribed"
                        if oversubscribed
                        else f"cpu-intra-op-{threads}"
                    ),
                    "target_backends": [
                        "dense",
                        "sparse",
                        "hybrid",
                        "reranker",
                    ],
                    "changes": {
                        "schedule": {
                            "intra_op_threads": threads,
                            "inter_op_threads": 1,
                            "allow_oversubscription": oversubscribed,
                        }
                    },
                }
            )
        if config.mode == "full":
            for inter_op_threads in cpu_space["cpu_topology"]["inter_op_threads"]:
                if inter_op_threads == 1:
                    continue
                specs.append(
                    {
                        "name": f"cpu-inter-op-{inter_op_threads}",
                        "target_backends": [
                            "dense",
                            "sparse",
                            "hybrid",
                            "reranker",
                        ],
                        "changes": {
                            "schedule": {
                                "inter_op_threads": inter_op_threads,
                            }
                        },
                    }
                )
            for spin in ("passive", "active"):
                specs.append(
                    {
                        "name": f"cpu-worker-spin-{spin}",
                        "target_backends": [
                            "dense",
                            "sparse",
                            "hybrid",
                            "reranker",
                        ],
                        "changes": {"schedule": {"worker_spin": spin}},
                    }
                )
            for affinity in cpu_space["cpu_topology"]["affinity"]:
                if affinity == "none":
                    continue
                specs.append(
                    {
                        "name": f"cpu-affinity-{affinity}",
                        "target_backends": [
                            "dense",
                            "sparse",
                            "hybrid",
                            "reranker",
                        ],
                        "changes": {"schedule": {"affinity": affinity}},
                    }
                )
        if _supports_bf16(fingerprint):
            specs.extend(
                [
                    {
                        "name": "cpu-dense-bf16",
                        "target_backends": ["dense", "hybrid"],
                        "changes": {
                            "dense": {
                                "runtime": "torch",
                                "precision": "bf16",
                            }
                        },
                    },
                    {
                        "name": "cpu-sparse-bf16",
                        "target_backends": ["sparse", "hybrid"],
                        "changes": {
                            "sparse": {
                                "runtime": "torch",
                                "precision": "bf16",
                                "compute_precision": "bf16",
                            }
                        },
                    },
                    {
                        "name": "cpu-reranker-bf16",
                        "target_backends": ["reranker"],
                        "changes": {
                            "reranker": {
                                "runtime": "torch",
                                "precision": "bf16",
                            }
                        },
                    },
                ]
            )
        if config.mode == "full":
            runtime_availability = cpu_space["dense"]["runtime_availability"]
            specs.extend(
                [
                    {
                        "name": "cpu-dense-onnx-fp32",
                        "target_backends": ["dense", "hybrid"],
                        **(
                            {}
                            if runtime_availability["onnx-fp32"]["available"]
                            else {
                                "unavailable_reason": runtime_availability["onnx-fp32"][
                                    "reason"
                                ]
                            }
                        ),
                        "changes": {"dense": {"runtime": "onnx-fp32"}},
                    },
                    {
                        "name": "cpu-dense-onnx-int8",
                        "target_backends": ["dense", "hybrid"],
                        **(
                            {}
                            if runtime_availability["onnx-int8"]["available"]
                            else {
                                "unavailable_reason": runtime_availability["onnx-int8"][
                                    "reason"
                                ]
                            }
                        ),
                        "changes": {
                            "dense": {
                                "runtime": "onnx-int8",
                                "precision": "int8",
                            }
                        },
                    },
                ]
            )
            openvino_reason = runtime_availability["openvino"]["reason"]
            specs.append(
                {
                    "name": "cpu-dense-openvino",
                    "target_backends": ["dense", "hybrid"],
                    "unavailable_reason": openvino_reason,
                    "changes": {"dense": {"runtime": "openvino"}},
                }
            )
            # These entries make the Step 9 runtime audit explicit.  MILCO's
            # English-pivot + LexEcho COO projection and the Qwen reranker do
            # not yet have a separately validated ORT serving path.  Keeping
            # them as unavailable candidates is safer than silently timing the
            # Torch implementation under an ONNX label.
            sparse_ort_reason = (
                "no validated ONNX export preserves MILCO English-pivot "
                "lexical COO and LexEcho source_view semantics"
            )
            reranker_ort_reason = (
                "no validated ONNX export is shipped for the Qwen3 reranker"
            )
            for runtime in ("onnx-fp32", "onnx-int8"):
                specs.extend(
                    [
                        {
                            "name": f"cpu-sparse-{runtime}",
                            "target_backends": ["sparse", "hybrid"],
                            "unavailable_reason": sparse_ort_reason,
                            "changes": {"sparse": {"runtime": runtime}},
                        },
                        {
                            "name": f"cpu-reranker-{runtime}",
                            "target_backends": ["reranker"],
                            "unavailable_reason": reranker_ort_reason,
                            "changes": {"reranker": {"runtime": runtime}},
                        },
                    ]
                )
    if config.mode == "full":
        dense_eager = {"attention": "eager"}
        if device == "cpu":
            dense_eager.update(
                {
                    "runtime": "torch",
                    "precision": "fp32",
                    "torch_compile": False,
                }
            )
        specs.extend(
            [
                {
                    "name": "dense-eager",
                    "target_backends": ["dense", "hybrid"],
                    "changes": {"dense": dense_eager},
                },
                {
                    "name": "reranker-eager",
                    "target_backends": ["reranker"],
                    "changes": {"reranker": {"attention": "eager"}},
                },
            ]
        )
        # Low-overhead transfer candidates must run before TorchInductor. A
        # single compile probe can consume several minutes per dynamic shape;
        # placing it first left pinned-memory candidates systematically
        # unmeasured even with the documented 20–30 minute FULL budget.
        if device == "cuda":
            specs.extend(
                [
                    {
                        "name": "dense-pinned-nonblocking",
                        "target_backends": ["dense", "hybrid"],
                        "changes": {"dense": {"pinned_memory": True}},
                    },
                    {
                        "name": "sparse-pinned-nonblocking",
                        "target_backends": ["sparse", "hybrid"],
                        "changes": {"sparse": {"pinned_memory": True}},
                    },
                    {
                        "name": "dense-double-buffer",
                        "target_backends": ["dense", "hybrid"],
                        "changes": {"dense": {"prefetch_batches": True}},
                    },
                    {
                        "name": "sparse-double-buffer",
                        "target_backends": ["sparse", "hybrid"],
                        "changes": {"sparse": {"prefetch_batches": True}},
                    },
                ]
            )
        specs.extend(
            [
                {
                    "name": "dense-compile-default",
                    "target_backends": ["dense", "hybrid"],
                    "changes": {
                        "dense": {
                            **(
                                {"runtime": "torch", "precision": "fp32"}
                                if device == "cpu"
                                else {}
                            ),
                            "torch_compile": True,
                            "torch_compile_mode": "default",
                        }
                    },
                },
                {
                    "name": "sparse-compile-default",
                    "target_backends": ["sparse", "hybrid"],
                    "changes": {
                        "sparse": {
                            "torch_compile": True,
                            "torch_compile_mode": "default",
                        }
                    },
                },
                {
                    "name": "reranker-compile-default",
                    "target_backends": ["reranker"],
                    "changes": {
                        "reranker": {
                            "torch_compile": True,
                            "torch_compile_mode": "default",
                        }
                    },
                },
                {
                    "name": "dense-compile-reduce-overhead",
                    "target_backends": ["dense", "hybrid"],
                    "changes": {
                        "dense": {
                            **(
                                {"runtime": "torch", "precision": "fp32"}
                                if device == "cpu"
                                else {}
                            ),
                            "torch_compile": True,
                            "torch_compile_mode": "reduce-overhead",
                        }
                    },
                },
                {
                    "name": "sparse-compile-reduce-overhead",
                    "target_backends": ["sparse", "hybrid"],
                    "changes": {
                        "sparse": {
                            "torch_compile": True,
                            "torch_compile_mode": "reduce-overhead",
                        }
                    },
                },
                {
                    "name": "reranker-compile-reduce-overhead",
                    "target_backends": ["reranker"],
                    "changes": {
                        "reranker": {
                            "torch_compile": True,
                            "torch_compile_mode": "reduce-overhead",
                        }
                    },
                },
            ]
        )
    if device == "cuda" and fingerprint.get("software", {}).get("flash_attn"):
        specs.extend(
            [
                {
                    "name": "dense-flash-attention-2",
                    "target_backends": ["dense", "hybrid"],
                    "changes": {"dense": {"attention": "flash_attention_2"}},
                },
                {
                    "name": "reranker-flash-attention-2",
                    "target_backends": ["reranker"],
                    "changes": {"reranker": {"attention": "flash_attention_2"}},
                },
            ]
        )

    dense_max = _configured_limit(fingerprint, "QWEN_EMBED_MAX_LENGTH", 32768)
    sparse_max = _configured_limit(fingerprint, "MILCO_MAX_LENGTH", 8192)
    rerank_max = _configured_limit(fingerprint, "QWEN_RERANK_MAX_LENGTH", 32768)
    if device == "cuda":
        geometric = _gpu_geometric_batch_maps(fingerprint, [128, 512, 1024, 2048, 8192])
        dense_batches = {int(key): value for key, value in geometric["dense"].items()}
        sparse_batches = {int(key): value for key, value in geometric["sparse"].items()}
        rerank_batches = {
            int(key): value for key, value in geometric["reranker"].items()
        }
    else:
        dense_batches = {128: 2, 512: 2, 1024: 1, 2048: 1, 8192: 1}
        sparse_batches = dict(dense_batches)
        rerank_batches = dict(dense_batches)
    token_cap = _batch_token_cap(fingerprint)
    dense_policy = _bucket_policy(dense_max, dense_batches, max_batch_tokens=token_cap)
    sparse_policy = _bucket_policy(
        sparse_max, sparse_batches, max_batch_tokens=token_cap
    )
    rerank_policy = _bucket_policy(
        rerank_max, rerank_batches, max_batch_tokens=token_cap
    )
    schedule_spec = {
        "name": ("hybrid-cuda-overlap" if device == "cuda" else "hybrid-cpu-parallel"),
        "target_backends": ["hybrid", "mixed"],
        "changes": {
            "schedule": {
                "mode": "overlap" if device == "cuda" else "parallel",
                "inference_lanes": 2,
                "intra_op_threads": (
                    4
                    if device == "cuda"
                    else max(
                        1,
                        int(
                            fingerprint.get("hardware", {})
                            .get("cpu", {})
                            .get("effective_cpu_count")
                            or 1
                        )
                        // 2,
                    )
                ),
            }
        },
    }
    if config.mode == "full":
        if device == "cuda":
            multiplier = _vram_multiplier(fingerprint)
            dense_start = 8 * multiplier
            sparse_start = 8 * multiplier
            rerank_start = 16 * multiplier
        else:
            dense_start = sparse_start = rerank_start = 1
        batch_specs = [
            *_geometric_batch_family_specs(
                backend="dense",
                target_backends=["dense", "hybrid"],
                batch_sizes=dense_batches,
                max_length=dense_max,
                start_cap=dense_start,
                max_batch_tokens=token_cap,
            ),
            schedule_spec,
            *_geometric_batch_family_specs(
                backend="sparse",
                target_backends=["sparse", "hybrid"],
                batch_sizes=sparse_batches,
                max_length=sparse_max,
                start_cap=sparse_start,
                max_batch_tokens=token_cap,
            ),
            *_geometric_batch_family_specs(
                backend="reranker",
                target_backends=["reranker"],
                batch_sizes=rerank_batches,
                max_length=rerank_max,
                start_cap=rerank_start,
                max_batch_tokens=token_cap,
            ),
        ]
    else:
        batch_specs = [
            {
                "name": "dense-geometric-batch-policy",
                "target_backends": ["dense", "hybrid"],
                "promotion_scope": {"dense": [128, 512], "hybrid": [128]},
                "changes": {
                    "dense": {
                        "batch_policy": dense_policy[0],
                        "max_batch_tokens_policy": dense_policy[1],
                    }
                },
            },
            schedule_spec,
            {
                "name": "sparse-geometric-batch-policy",
                "target_backends": ["sparse", "hybrid"],
                "promotion_scope": {"sparse": [128, 512], "hybrid": [128]},
                "changes": {
                    "sparse": {
                        "batch_policy": sparse_policy[0],
                        "max_batch_tokens_policy": sparse_policy[1],
                    }
                },
            },
            {
                "name": "reranker-geometric-batch-policy",
                "target_backends": ["reranker"],
                "promotion_scope": {"reranker": [128]},
                "changes": {
                    "reranker": {
                        "batch_policy": rerank_policy[0],
                        "max_batch_tokens_policy": rerank_policy[1],
                    }
                },
            },
        ]
    specs[0:0] = batch_specs
    if device == "cuda":
        specs.insert(
            len(batch_specs),
            {
                "name": "schedule-queue-batch-32",
                "target_backends": ["requests"],
                "promotion_scope": {"requests": [128]},
                "changes": {"schedule": {"queue_batch": 32, "flush_timeout_ms": 10}},
            },
        )
    if device == "cpu" and config.mode == "full":
        # Measure portable dense runtimes immediately after the safe baseline.
        # A CPU FULL run otherwise spends its finite budget on batch/topology
        # probes before reaching the highest-impact runtime comparison.
        runtime_priority = (
            "cpu-dense-onnx-fp32",
            "cpu-dense-onnx-int8",
            "cpu-dense-openvino",
            "cpu-sparse-onnx-fp32",
            "cpu-sparse-onnx-int8",
            "cpu-reranker-onnx-fp32",
            "cpu-reranker-onnx-int8",
        )
        by_name = {item["name"]: item for item in specs}
        priority_specs = [by_name[name] for name in runtime_priority if name in by_name]
        priority_names = {item["name"] for item in priority_specs}
        specs = priority_specs + [
            item for item in specs if item["name"] not in priority_names
        ]
        # Tuned probes use the regular matrix, then the final selected policy
        # must pass one fresh isolated canary through every active max-length
        # bucket before it can be persisted.
        for spec in specs:
            spec["defer_long_canary"] = True
    return specs


def materialise_candidate(
    baseline: Mapping[str, Any], selected: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict:
    """Apply one candidate specification to an isolated baseline copy.

    Args:
        baseline: Reference workload and safety policy.
        selected: Policy promoted by earlier calibration stages.
        spec: Candidate name, target backends, and policy changes.

    Returns:
        A self-contained worker payload without shared mutable state.
    """
    candidate = copy.deepcopy(dict(baseline))
    candidate["name"] = spec["name"]
    candidate["target_backends"] = list(spec["target_backends"])
    candidate["selected"] = copy.deepcopy(dict(selected))
    advanced = candidate["selected"]["advanced"]
    for section, changes in spec.get("changes", {}).items():
        advanced[section].update(copy.deepcopy(changes))
    if spec.get("defer_long_canary"):
        deferred = {int(value) for value in candidate.get("canary_token_lengths", ())}
        if deferred:
            candidate["token_lengths"] = [
                int(value)
                for value in candidate.get("token_lengths", ())
                if int(value) not in deferred
            ]
            candidate["deferred_canary_token_lengths"] = sorted(deferred)
            candidate.pop("canary_token_lengths", None)
            candidate.pop("canary_backends", None)
    return candidate


def _changed_canary_backends(
    baseline_selected: Mapping[str, Any], selected: Mapping[str, Any]
) -> list[str]:
    baseline = baseline_selected.get("advanced", {})
    tuned = selected.get("advanced", {})
    changed = []
    for section, backend in (
        ("dense", "dense"),
        ("sparse", "sparse"),
        ("reranker", "reranker"),
    ):
        if baseline.get(section) != tuned.get(section):
            changed.append(backend)
    if baseline.get("schedule") != tuned.get("schedule"):
        changed = ["dense", "sparse", "reranker"]
    return changed


def final_long_canary_candidate(
    baseline: Mapping[str, Any],
    *,
    baseline_selected: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> Optional[dict]:
    """Build the final long-input canary for changed backends.

    Args:
        baseline: Reference workload containing deferred canary lengths.
        baseline_selected: Safe policy used for the reference measurement.
        selected: Tentative policy produced by calibration.

    Returns:
        An isolated canary candidate, or ``None`` when nothing changed.
    """
    changed_backends = _changed_canary_backends(baseline_selected, selected)
    configured_by_backend = baseline.get("deferred_canary_token_lengths_by_backend", {})
    if isinstance(configured_by_backend, Mapping) and configured_by_backend:
        by_backend = {
            backend: [int(value) for value in configured_by_backend.get(backend, ())]
            for backend in changed_backends
        }
    else:
        common = [
            int(value)
            for value in (
                baseline.get("deferred_canary_token_lengths")
                or baseline.get("canary_token_lengths", ())
            )
        ]
        by_backend = {backend: list(common) for backend in changed_backends}
    lengths = sorted({length for values in by_backend.values() for length in values})
    if not lengths or not changed_backends:
        return None
    candidate = copy.deepcopy(dict(baseline))
    candidate.update(
        {
            "name": "selected-final-long-canary",
            "selected": copy.deepcopy(dict(selected)),
            "token_lengths": lengths,
            "repeats": 1,
            "target_backends": changed_backends,
            "canary_token_lengths": lengths,
            "canary_backends": changed_backends,
        }
    )
    candidate.pop("deferred_canary_token_lengths", None)
    candidate.pop("deferred_canary_token_lengths_by_backend", None)
    candidate["backend_token_lengths"] = by_backend
    return candidate


def evaluate_quality_capture(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict:
    """Evaluate dense, sparse, and reranker captures as one quality gate.

    Args:
        reference: Baseline outputs from the versioned retrieval corpus.
        candidate: Outputs produced by the candidate runtime.

    Returns:
        Per-backend metrics and a combined pass/failure decision.
    """
    if (
        reference.get("suite_version") != QUALITY_SUITE_VERSION
        or candidate.get("suite_version") != QUALITY_SUITE_VERSION
    ):
        return {
            "passed": False,
            "failures": ["quality suite version mismatch"],
        }
    dense = evaluate_dense(reference["dense"], candidate["dense"])
    sparse = evaluate_sparse(
        reference["sparse"],
        candidate["sparse"],
        reference_source_view=bool(reference["milco_source_view"]),
        candidate_source_view=bool(candidate["milco_source_view"]),
        reference_top_k=int(reference["milco_top_k"]),
        candidate_top_k=int(candidate["milco_top_k"]),
        source_vocab_start=int(candidate["source_vocab_start"]),
    )
    reranker = evaluate_reranker(reference["reranker"], candidate["reranker"])
    failures = [
        f"{backend}: {failure}"
        for backend, result in (
            ("dense", dense),
            ("sparse", sparse),
            ("reranker", reranker),
        )
        for failure in result.failures
    ]
    return {
        "passed": not failures,
        "failures": failures,
        "dense": dense.to_dict(),
        "sparse": sparse.to_dict(),
        "reranker": reranker.to_dict(),
    }


def evaluate_promotion(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    target_backends: list[str],
    objective: str,
    promotion_scope: Optional[Mapping[str, list[int]]] = None,
) -> dict:
    """Decide whether measured performance merits candidate promotion.

    Quality and memory safety are evaluated elsewhere; this gate requires a
    material throughput gain while bounding P95 latency and variability.

    Args:
        reference: Baseline measurement payload.
        candidate: Candidate measurement payload.
        target_backends: Backends whose measurements may affect promotion.
        objective: Throughput, balanced, or latency optimization objective.
        promotion_scope: Optional backend-to-token-length gain scope. All
            comparable measurements remain part of the safety checks.

    Returns:
        Promotion decision with the measured gain and rejection reason.
    """
    reference_measurements = {
        (item["backend"], item["input_tokens"], item["batch_size"]): item
        for item in reference.get("measurements", [])
    }
    ratios = []
    gain_ratios = []
    p95_ratios = []
    variability = []
    for item in candidate.get("measurements", []):
        if item.get("backend") not in target_backends:
            continue
        key = (item["backend"], item["input_tokens"], item["batch_size"])
        baseline = reference_measurements.get(key)
        if not baseline:
            continue
        p95_ratios.append(
            float(item["p95_seconds"]) / max(float(baseline["p95_seconds"]), 1e-12)
        )
        variability.append(float(item.get("coefficient_of_variation", 0.0)))
        if item.get("is_canary"):
            continue
        ratio = float(item["useful_tokens_per_second"]) / max(
            float(baseline["useful_tokens_per_second"]), 1e-12
        )
        ratios.append(ratio)
        scoped_lengths = (
            promotion_scope.get(str(item["backend"]), []) if promotion_scope else []
        )
        if not promotion_scope or int(item["input_tokens"]) in scoped_lengths:
            gain_ratios.append(ratio)
    if not ratios or not gain_ratios:
        return {"promoted": False, "reason": "no comparable measurements"}
    median_gain = statistics.median(gain_ratios) - 1.0
    max_p95_ratio = max(p95_ratios)
    p95_limit = (
        1.25
        if objective == "throughput"
        else (1.10 if objective == "balanced" else 1.03)
    )
    promoted = (
        median_gain >= 0.03
        and max_p95_ratio <= p95_limit
        and max(variability, default=0.0) <= 0.15
    )
    if median_gain < 0.03:
        reason = "median throughput gain below 3%"
    elif max_p95_ratio > p95_limit:
        reason = "P95 regression exceeds objective tolerance"
    elif max(variability, default=0.0) > 0.15:
        reason = "run variability exceeds 15%"
    else:
        reason = "quality and performance promotion gates passed"
    return {
        "promoted": promoted,
        "reason": reason,
        "median_throughput_gain": round(median_gain, 6),
        "max_p95_ratio": round(max_p95_ratio, 6),
        "max_coefficient_of_variation": round(max(variability, default=0.0), 6),
        "gain_measurements": len(gain_ratios),
        "safety_measurements": len(p95_ratios),
    }


def candidate_command(candidate: Mapping[str, Any], *, script_dir: Path) -> list[str]:
    """Build the isolated worker command for a serialized candidate.

    Args:
        candidate: Complete candidate payload.
        script_dir: Directory containing the worker entry point.

    Returns:
        Command arguments suitable for ``subprocess`` execution.
    """
    return [
        sys.executable,
        str(script_dir / "advanced_autotune_worker.py"),
        "--candidate-json",
        json.dumps(candidate, sort_keys=True, separators=(",", ":")),
        "--script-dir",
        str(script_dir),
    ]


def _result_record(result: IsolatedResult) -> dict:
    payload = dict(result.payload or {})
    payload.update(
        {
            "isolated_returncode": result.returncode,
            "isolated_duration_seconds": result.duration_seconds,
            "isolated_timed_out": result.timed_out,
            "isolated_memory_exceeded": result.memory_exceeded,
            "isolated_peak_rss_bytes": result.peak_rss_bytes,
        }
    )
    if result.stderr:
        payload["stderr_tail"] = result.stderr[-2000:]
    return payload


def estimate_candidate_seconds(
    baseline_seconds: float, target_backends: list[str]
) -> float:
    """Budget one isolated probe from measured fixed and backend costs.

    Every candidate pays model-load and full quality-capture overhead, while
    only its target backends repeat the length matrix. Using the full baseline
    duration for every probe caused QUICK to skip small reranker/hybrid probes
    even when enough time remained.
    """
    baseline = max(1.0, float(baseline_seconds))
    fixed = min(30.0, max(15.0, baseline * 0.35))
    backend_weights = {
        "dense": 1.0,
        "sparse": 1.1,
        "hybrid": 1.5,
        "reranker": 0.8,
        "mixed": 2.0,
        "requests": 1.0,
    }
    total_weight = sum(backend_weights.values())
    selected_weight = sum(
        backend_weights.get(backend, 1.0) for backend in set(target_backends)
    )
    measured = fixed + max(0.0, baseline - fixed) * min(
        1.0, selected_weight / total_weight
    )
    return max(20.0, measured * 1.1)


def estimate_final_canary_seconds(
    baseline_payload: Mapping[str, Any],
    *,
    baseline_seconds: float,
    repeats: int,
) -> float:
    """Reserve load/quality time plus the measured long-input probes."""
    non_canary = 0.0
    canary = 0.0
    for measurement in baseline_payload.get("measurements", []):
        p50 = float(
            measurement.get("p50_seconds") or measurement.get("median_seconds") or 0.0
        )
        if measurement.get("is_canary"):
            canary += float(measurement.get("p95_seconds") or p50)
        else:
            # One warm-up plus the configured measured repetitions.
            non_canary += p50 * (max(1, int(repeats)) + 1)
    fixed = max(20.0, float(baseline_seconds) - non_canary - canary)
    return max(30.0, (fixed + canary) * 1.25)


def estimate_deferred_canary_seconds(
    baseline_payload: Mapping[str, Any],
    *,
    baseline_seconds: float,
    repeats: int,
    max_seconds: float,
) -> float:
    """Budget long probes separately from model loading and final quality checks.

    Without a long-input reference, use a baseline run as the probe allowance.
    Keep the existing floor and total-budget cap so calibration stays bounded.
    """
    overhead = estimate_final_canary_seconds(
        baseline_payload, baseline_seconds=baseline_seconds, repeats=repeats
    )
    return max(120.0, min(max_seconds * 0.4, baseline_seconds + overhead))


def prioritise_for_gpu_wait(specs: list[dict]) -> list[dict]:
    """Order candidates for a GPU that waits on the CPU, keeping each group stable.

    Larger batches and request coalescing cut the number of kernel launches
    first; CUDA graphs (compile reduce-overhead) remove launch overhead next.

    Args:
        specs: Candidate specifications in their default order.

    Returns:
        The same candidates, with those groups moved to the front.
    """

    def rank(spec: Mapping[str, Any]) -> int:
        name = str(spec.get("name", ""))
        if "geometric-batch" in name or name.startswith("schedule-queue-batch"):
            return 0
        if name.endswith("compile-reduce-overhead"):
            return 1
        return 2

    return sorted(specs, key=rank)


def calibration_deadline_seconds(
    config: AdvancedAutotuneConfig,
    *,
    baseline_seconds: float,
    planned_candidate_seconds: float,
) -> float:
    """Return the total calibration time allowed once the baseline is measured.

    An explicit budget is a fixed limit. An unset quick budget grows to fit the
    measured baseline plus every planned candidate, never below its default and
    never above ``config.budget_ceiling_seconds``: on a host whose CPU dispatches
    kernels slowly the baseline alone can consume the whole default budget.
    """
    if not config.adaptive_budget:
        return config.max_seconds
    planned = baseline_seconds + planned_candidate_seconds
    return min(config.budget_ceiling_seconds, max(config.max_seconds, planned))


def evaluate_final_canary(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    target_backends: list[str],
    objective: str,
    expected_lengths: Optional[list[int]] = None,
    expected_lengths_by_backend: Optional[Mapping[str, list[int]]] = None,
) -> dict:
    """Check long-input latency without requiring another throughput gain."""
    reference_measurements = {
        (item["backend"], item["input_tokens"], item["batch_size"]): item
        for item in reference.get("measurements", [])
        if item.get("is_canary")
    }
    ratios = []
    missing = []
    measured = set()
    for item in candidate.get("measurements", []):
        backend = str(item.get("backend"))
        if backend not in target_backends or not item.get("is_canary"):
            continue
        key = (backend, item["input_tokens"], item["batch_size"])
        measured.add((backend, int(item["input_tokens"])))
        baseline = reference_measurements.get(key)
        if baseline is not None:
            ratios.append(
                float(item["p95_seconds"]) / max(float(baseline["p95_seconds"]), 1e-12)
            )
    if expected_lengths_by_backend:
        expected = {
            (backend, int(length))
            for backend in target_backends
            for length in expected_lengths_by_backend.get(backend, ())
        }
    else:
        expected = {
            (backend, int(length))
            for backend in target_backends
            for length in (expected_lengths or [])
        }
    if expected:
        missing.extend(
            f"{backend}:{length}" for backend, length in sorted(expected - measured)
        )
    else:
        missing.extend(
            backend
            for backend in target_backends
            if not any(item[0] == backend for item in measured)
        )
    p95_limit = (
        1.25
        if objective == "throughput"
        else (1.10 if objective == "balanced" else 1.03)
    )
    max_ratio = max(ratios, default=0.0)
    passed = bool(measured) and not missing and (not ratios or max_ratio <= p95_limit)
    if missing:
        reason = "missing comparable long-canary measurements"
    elif ratios and max_ratio > p95_limit:
        reason = "long-canary P95 regression exceeds objective tolerance"
    elif not ratios:
        reason = (
            "long-input safety gate passed; the rejected/unmeasured baseline "
            "has no comparable latency"
        )
    else:
        reason = "final long-input safety gate passed"
    return {
        "passed": passed,
        "reason": reason,
        "max_p95_ratio": round(max_ratio, 6) if ratios else None,
        "p95_limit": p95_limit,
        "missing": sorted(set(missing)),
        "measurements": len(ratios),
    }


def calibrate(
    *,
    config: AdvancedAutotuneConfig,
    fingerprint: dict,
    script_dir: Path,
    dry_run: bool = False,
    isolated_runner: Callable[..., IsolatedResult] = run_isolated,
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Validate/reuse or calibrate one safe co-resident starting point."""
    started = time.monotonic()
    search_space = advanced_search_space(fingerprint, config.mode)
    candidate = baseline_candidate(fingerprint, config)
    if config.mode == "off":
        return {"action": "disabled", "reason": "autotune is off"}
    existing = load_advanced_profile(Path(config.profile_path))
    if not config.force and advanced_profile_matches(
        existing, fingerprint, config=config
    ):
        return {
            "action": "reused",
            "profile": config.profile_path,
            "fingerprint_sha256": existing["fingerprint_sha256"],
            "selected": existing["selected"],
            "reuse_seconds": round(time.monotonic() - started, 6),
        }
    cached, missing = models_are_cached(fingerprint)
    if not cached:
        return {
            "action": "fallback",
            "reason": "ADVANCED model snapshots are not all present in cache",
            "missing_models": missing,
        }
    if dry_run:
        return {
            "action": "dry-run",
            "profile": config.profile_path,
            "candidate": candidate,
            "search_space": search_space,
        }

    remaining = config.budget_ceiling_seconds - (time.monotonic() - started)
    if remaining <= 1:
        return {"action": "fallback", "reason": "calibration budget exhausted"}
    memory_limit = fingerprint.get("hardware", {}).get("memory_limit_bytes")
    max_rss = int(memory_limit * 0.9) if memory_limit else None
    isolated = isolated_runner(
        candidate_command(candidate, script_dir=script_dir),
        timeout_seconds=remaining,
        cwd=script_dir,
        env=env,
        max_rss_bytes=max_rss,
    )
    baseline_record = _result_record(isolated)
    if candidate.get("deferred_canary_token_lengths"):
        baseline_record["deferred_canary_token_lengths"] = list(
            candidate["deferred_canary_token_lengths"]
        )
    baseline_payload = isolated.payload or {}
    if not isolated.ok or not baseline_payload.get("quality_passed"):
        if isolated.timed_out:
            reason = "isolated candidate exceeded the global deadline"
        elif isolated.memory_exceeded:
            reason = "isolated candidate exceeded the RSS safety limit"
        else:
            reason = str(
                baseline_payload.get("error")
                or baseline_payload.get("quality_errors")
                or "isolated co-resident candidate failed"
            )
        return {
            "action": "fallback",
            "reason": reason,
            "results": [baseline_record],
            "search_space": search_space,
        }
    selected = baseline_payload.get("selected") or candidate["selected"]
    baseline_selected = copy.deepcopy(selected)
    results = [baseline_record]
    gpu_dispatch = gpu_wait_diagnosis(None)
    budget = {
        "configured_seconds": config.max_seconds,
        "adaptive": config.adaptive_budget,
        "ceiling_seconds": config.budget_ceiling_seconds,
    }
    cpu = fingerprint.get("hardware", {}).get("cpu", {})
    cpu_ids = (
        cpu.get("cpuset") or cpu.get("affinity_cpu_ids") or range(os.cpu_count() or 1)
    )
    cpu_policy = read_cpu_power_policy(cpu_ids)
    measured_frequency = baseline_payload.get("cpu_frequency")
    host_cpu_power = {
        "policy": cpu_policy,
        "measured": measured_frequency,
        "warnings": cpu_power_warnings(cpu_policy, measured_frequency),
    }
    reference_capture = baseline_payload.get("quality_capture")
    if isinstance(reference_capture, Mapping):
        baseline_quality = evaluate_quality_capture(
            reference_capture, reference_capture
        )
        baseline_record["quality_gate"] = baseline_quality
        capture_bytes = json.dumps(
            reference_capture,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        baseline_record["quality_capture_sha256"] = hashlib.sha256(
            capture_bytes
        ).hexdigest()
        baseline_record.pop("quality_capture", None)
        if not baseline_quality["passed"]:
            return {
                "action": "fallback",
                "reason": "safe baseline failed the versioned retrieval gate",
                "results": results,
                "search_space": search_space,
            }

        champion_payload = baseline_payload
        baseline_template = candidate
        specs = tuning_candidate_specs(fingerprint, config)
        if _device_kind(fingerprint) == "cuda":
            gpu_dispatch = gpu_wait_diagnosis(baseline_payload.get("gpu_utilization"))
        if gpu_dispatch["gpu_waits_on_cpu"]:
            specs = prioritise_for_gpu_wait(specs)
        if candidate.get("canary_token_lengths"):
            final_canary_reserve = estimate_final_canary_seconds(
                baseline_payload,
                baseline_seconds=isolated.duration_seconds,
                repeats=int(candidate.get("repeats", 1)),
            )
        elif candidate.get("deferred_canary_token_lengths"):
            # There is deliberately no 8192 reference measurement when the
            # safe Torch path cannot stay under the co-resident RSS limit.
            # Reserve a bounded fraction for a recovery canary on the selected
            # runtime instead of letting regular tuning consume the deadline.
            final_canary_reserve = estimate_deferred_canary_seconds(
                baseline_payload,
                baseline_seconds=isolated.duration_seconds,
                repeats=int(candidate.get("repeats", 1)),
                max_seconds=config.max_seconds,
            )
        else:
            final_canary_reserve = 0.0
        planned_candidate_seconds = sum(
            estimate_candidate_seconds(
                isolated.duration_seconds, list(spec["target_backends"])
            )
            + (final_canary_reserve if spec.get("defer_long_canary") else 0.0)
            for spec in specs
            if not spec.get("unavailable_reason")
        )
        elapsed_seconds = time.monotonic() - started
        deadline_seconds = calibration_deadline_seconds(
            config,
            baseline_seconds=elapsed_seconds,
            planned_candidate_seconds=planned_candidate_seconds,
        )
        budget.update(
            {
                "planned_seconds": round(
                    elapsed_seconds + planned_candidate_seconds, 6
                ),
                "deadline_seconds": round(deadline_seconds, 6),
            }
        )
        stopped_families: dict[str, str] = {}
        for spec_index, spec in enumerate(specs):
            search_family = spec.get("search_family")
            if search_family in stopped_families:
                results.append(
                    {
                        "candidate": spec["name"],
                        "status": "skipped-search-stop",
                        "search_family": search_family,
                        "search_value": spec.get("search_value"),
                        "reason": stopped_families[search_family],
                    }
                )
                continue
            if spec.get("unavailable_reason"):
                results.append(
                    {
                        "candidate": spec["name"],
                        "status": "skipped-unavailable",
                        "reason": str(spec["unavailable_reason"]),
                    }
                )
                continue
            remaining = deadline_seconds - (time.monotonic() - started)
            required_seconds = estimate_candidate_seconds(
                isolated.duration_seconds,
                list(spec["target_backends"]),
            ) + (final_canary_reserve if spec.get("defer_long_canary") else 0.0)
            if remaining < required_seconds:
                # Estimates depend on each candidate's backends, so a later one
                # can still fit: skip this candidate instead of ending the search.
                results.append(
                    {
                        "candidate": spec["name"],
                        "status": "skipped-budget",
                        "remaining_seconds": round(max(0.0, remaining), 6),
                        "estimated_seconds": round(required_seconds, 6),
                    }
                )
                continue
            tuned = materialise_candidate(baseline_template, selected, spec)
            candidate_timeout = remaining - (
                final_canary_reserve if spec.get("defer_long_canary") else 0.0
            )
            isolated_candidate = isolated_runner(
                candidate_command(tuned, script_dir=script_dir),
                timeout_seconds=max(1.0, candidate_timeout),
                cwd=script_dir,
                env=env,
                max_rss_bytes=max_rss,
            )
            candidate_record = _result_record(isolated_candidate)
            if candidate_record.get("candidate") in (None, "unknown"):
                candidate_record["candidate"] = spec["name"]
            if search_family:
                candidate_record["search_family"] = search_family
                candidate_record["search_value"] = spec.get("search_value")
            candidate_payload = isolated_candidate.payload or {}
            candidate_capture = candidate_payload.get("quality_capture")
            candidate_record.pop("quality_capture", None)
            if (
                not isolated_candidate.ok
                or not candidate_payload.get("quality_passed")
                or not isinstance(candidate_capture, Mapping)
            ):
                candidate_record["promotion"] = {
                    "promoted": False,
                    "reason": (
                        "isolated candidate failed, timed out, exceeded memory, "
                        "or returned no quality capture"
                    ),
                }
                results.append(candidate_record)
                if isolated_candidate.timed_out:
                    for pending in specs[spec_index + 1 :]:
                        results.append(
                            {
                                "candidate": pending["name"],
                                "status": "skipped-deadline",
                                "reason": (
                                    "prior isolated candidate exhausted the "
                                    "calibration deadline"
                                ),
                            }
                        )
                    break
                if search_family and spec.get("stop_family_on_rejection"):
                    stopped_families[search_family] = (
                        f"{spec['name']} failed the isolated safety gate"
                    )
                continue
            quality = evaluate_quality_capture(reference_capture, candidate_capture)
            candidate_record["quality_gate"] = quality
            if not quality["passed"]:
                candidate_record["promotion"] = {
                    "promoted": False,
                    "reason": "versioned retrieval quality gate failed",
                }
                results.append(candidate_record)
                if search_family and spec.get("stop_family_on_rejection"):
                    stopped_families[search_family] = (
                        f"{spec['name']} failed the versioned quality gate"
                    )
                continue
            promotion = evaluate_promotion(
                champion_payload,
                candidate_payload,
                target_backends=list(spec["target_backends"]),
                objective=config.objective,
                promotion_scope=spec.get("promotion_scope"),
            )
            candidate_record["promotion"] = promotion
            results.append(candidate_record)
            if promotion["promoted"]:
                selected = candidate_payload.get("selected") or tuned["selected"]
                merged = copy.deepcopy(champion_payload)
                measurements = {
                    (
                        item["backend"],
                        item["input_tokens"],
                        item["batch_size"],
                    ): item
                    for item in merged.get("measurements", [])
                }
                for item in candidate_payload.get("measurements", []):
                    measurements[
                        (
                            item["backend"],
                            item["input_tokens"],
                            item["batch_size"],
                        )
                    ] = item
                merged.update(candidate_payload)
                merged["measurements"] = list(measurements.values())
                champion_payload = merged
            elif search_family and spec.get("stop_family_on_rejection"):
                stopped_families[search_family] = (
                    f"{spec['name']} stopped on {promotion['reason']}"
                )

        final_canary = final_long_canary_candidate(
            baseline_template,
            baseline_selected=baseline_selected,
            selected=selected,
        )
        if final_canary is not None:
            remaining = deadline_seconds - (time.monotonic() - started)
            changed_backends = list(final_canary["target_backends"])
            if remaining <= 1.0:
                results.append(
                    {
                        "candidate": final_canary["name"],
                        "status": "skipped-budget",
                        "remaining_seconds": round(max(0.0, remaining), 6),
                        "promotion": {
                            "promoted": False,
                            "reason": (
                                "final long-input canary was not run; all "
                                "tuned selections were rolled back"
                            ),
                        },
                    }
                )
                selected = baseline_selected
            else:
                isolated_canary = isolated_runner(
                    candidate_command(final_canary, script_dir=script_dir),
                    timeout_seconds=remaining,
                    cwd=script_dir,
                    env=env,
                    max_rss_bytes=max_rss,
                )
                canary_record = _result_record(isolated_canary)
                if canary_record.get("candidate") in (None, "unknown"):
                    canary_record["candidate"] = final_canary["name"]
                canary_payload = isolated_canary.payload or {}
                canary_capture = canary_payload.get("quality_capture")
                canary_record.pop("quality_capture", None)
                quality = (
                    evaluate_quality_capture(reference_capture, canary_capture)
                    if isinstance(canary_capture, Mapping)
                    else {
                        "passed": False,
                        "failures": ["final canary returned no quality capture"],
                    }
                )
                canary_gate = evaluate_final_canary(
                    baseline_payload,
                    canary_payload,
                    target_backends=changed_backends,
                    objective=config.objective,
                    expected_lengths=list(final_canary["token_lengths"]),
                    expected_lengths_by_backend=final_canary.get(
                        "backend_token_lengths"
                    ),
                )
                canary_record["quality_gate"] = quality
                canary_record["canary_gate"] = canary_gate
                canary_passed = bool(
                    isolated_canary.ok
                    and canary_payload.get("quality_passed")
                    and quality["passed"]
                    and canary_gate["passed"]
                )
                canary_record["promotion"] = {
                    "promoted": canary_passed,
                    "reason": (
                        "selected policy passed the final isolated long-input gate"
                        if canary_passed
                        else (
                            "selected policy failed the final long-input gate; "
                            "all tuned selections were rolled back"
                        )
                    ),
                }
                results.append(canary_record)
                if not canary_passed:
                    selected = baseline_selected

    profile = make_advanced_profile(
        fingerprint=fingerprint,
        config=config,
        selected=selected,
        results=results,
        duration_seconds=time.monotonic() - started,
        budget=budget,
        host_cpu_power=host_cpu_power,
        gpu_dispatch=gpu_dispatch,
    )
    save_advanced_profile(Path(config.profile_path), profile)
    return {
        "action": "calibrated",
        "profile": config.profile_path,
        "fingerprint_sha256": profile["fingerprint_sha256"],
        "selected": selected,
        "duration_seconds": profile["duration_seconds"],
        "budget": budget,
        "host_cpu_power": host_cpu_power,
        "gpu_dispatch": gpu_dispatch,
        "warnings": [*host_cpu_power["warnings"], *gpu_dispatch["warnings"]],
        "results": results,
        "search_space": search_space,
        "deferred_canary_token_lengths": list(
            candidate.get("deferred_canary_token_lengths", [])
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the ADVANCED calibration command-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("off", "quick", "full"))
    parser.add_argument("--objective", choices=("balanced", "throughput", "latency"))
    parser.add_argument("--profile")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--app-version", default=APP_VERSION)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    """Run ADVANCED calibration and print its machine-readable result.

    Args:
        argv: Optional argument sequence. Uses process arguments when omitted.

    Returns:
        Zero on success or a nonzero status when calibration falls back.
    """
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ImportError:
        torch = None
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    config = AdvancedAutotuneConfig.from_env(device_kind=device)
    replacements = {
        field: argument
        for argument, field in (
            (args.mode, "mode"),
            (args.objective, "objective"),
            (args.profile, "profile_path"),
            (args.max_seconds, "max_seconds"),
        )
        if argument is not None
    }
    if args.max_seconds is not None and args.max_seconds != config.max_seconds:
        replacements["adaptive_budget"] = False
    if replacements:
        config = replace(config, **replacements)
    fingerprint = build_advanced_fingerprint(
        config=config,
        torch_module=torch,
        app_version=args.app_version,
    )
    result = calibrate(
        config=config,
        fingerprint=fingerprint,
        script_dir=Path(__file__).resolve().parent,
        dry_run=args.dry_run,
        env=os.environ,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.get("action") not in {"fallback"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
