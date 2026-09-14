"""FastAPI server for Smart Embedder embeddings and reranking.

``EmbeddingService`` routes profile-specific vector backends,
``RerankerWrapper`` scores passages, and ``app`` exposes the authenticated,
rate-limited, observable HTTP service.
"""

import asyncio
import hashlib
import logging
import os
import re
import secrets
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import orjson
import torch
from advanced_autotune import (
    AdvancedAutotuneConfig,
    build_advanced_fingerprint,
    resolve_advanced_runtime_profile,
)
from advanced_milco_runtime import memory_bounded_milco_forward
from advanced_qwen_onnx_backend import QwenDenseOnnxSession
from basic_autotune import (
    AutotuneConfig,
    batch_size_for_tokens,
    build_fingerprint,
    normalise_batch_policy,
    resolve_runtime_profile,
    source_artifact_digest,
)
from basic_onnx_backend import BgeM3OnnxBackend
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from model_revisions import MODEL_REVISIONS, pinned_model_id
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel, Field, model_validator
from request_budget import (
    BudgetExceeded,
    RequestBudget,
    estimate_embeddings,
    estimate_rerank,
)
from starlette.status import HTTP_504_GATEWAY_TIMEOUT
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
import vram_budget

# --- 1. Configuration and Logging ---

# Timestamps and levels keep concurrent inference failures diagnosable.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Server and model configuration.
# Leave this unset for one device; an explicit CUDA list enables multi-GPU placement.
MULTI_GPU_DEVICES = None
BGE_EMBEDDING_MODEL = "BAAI/bge-m3"
QWEN_DENSE_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_DENSE_VECTOR_SIZE = 1024
BGE_DENSE_VECTOR_SIZE = 1024
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
QWEN_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
QWEN_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QWEN_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MILCO_SPARSE_MODEL = "omai-research/milco-650m"

EMBEDDER_PROFILE_BASIC = "basic"
EMBEDDER_PROFILE_ADVANCED = "advanced"
SUPPORTED_PROFILES = {EMBEDDER_PROFILE_BASIC, EMBEDDER_PROFILE_ADVANCED}
EMBEDDING_CONTRACT_SCHEMA_VERSION = 1


def _resolve_profile() -> dict:
    """Resolve EMBEDDER_PROFILE into the per-type model selection."""
    configured = os.getenv("EMBEDDER_PROFILE", EMBEDDER_PROFILE_BASIC).strip().lower()
    if configured not in SUPPORTED_PROFILES:
        expected = ", ".join(sorted(SUPPORTED_PROFILES))
        raise ValueError(
            f"Unsupported EMBEDDER_PROFILE {configured!r}; expected one of: {expected}"
        )

    if configured == EMBEDDER_PROFILE_ADVANCED:
        return {
            "profile": EMBEDDER_PROFILE_ADVANCED,
            "dense": QWEN_DENSE_EMBEDDING_MODEL,
            "sparse": MILCO_SPARSE_MODEL,
            "colbert": None,
            "reranker": QWEN_RERANKER_MODEL,
        }
    return {
        "profile": EMBEDDER_PROFILE_BASIC,
        "dense": BGE_EMBEDDING_MODEL,
        "sparse": BGE_EMBEDDING_MODEL,
        "colbert": BGE_EMBEDDING_MODEL,
        "reranker": BGE_RERANKER_MODEL,
    }


ACTIVE_PROFILE = _resolve_profile()

# GPU detection and dynamic parameter configuration
has_cuda = torch.cuda.is_available()
device = "cuda" if has_cuda else "cpu"
available_gpus = torch.cuda.device_count() if has_cuda else 0

# Application version and device-aware display name (Swagger title, /metrics info).
APP_VERSION = "1.2.0"
APP_NAME = f"Smart Embedder {'GPU' if has_cuda else 'CPU'}"

# Calibration is always executed by a separate pre-start process. The serving
# process only validates and consumes a compatible persisted profile.
AUTOTUNE_CONFIG = AutotuneConfig.from_env()
AUTOTUNE_FINGERPRINT = build_fingerprint(
    torch_module=torch,
    app_version=APP_VERSION,
    model_revision=pinned_model_id(BGE_EMBEDDING_MODEL),
    artifact_digest=source_artifact_digest(),
)
AUTOTUNE_RUNTIME_PROFILE = resolve_runtime_profile(
    AUTOTUNE_CONFIG,
    AUTOTUNE_FINGERPRINT,
    applicable=ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_BASIC,
)
AUTOTUNE_SELECTED = AUTOTUNE_RUNTIME_PROFILE.selected
ADVANCED_AUTOTUNE_CONFIG = AdvancedAutotuneConfig.from_env(device_kind=device)
ADVANCED_AUTOTUNE_FINGERPRINT = build_advanced_fingerprint(
    config=ADVANCED_AUTOTUNE_CONFIG,
    torch_module=torch,
    app_version=APP_VERSION,
)
ADVANCED_AUTOTUNE_RUNTIME_PROFILE = resolve_advanced_runtime_profile(
    ADVANCED_AUTOTUNE_CONFIG,
    ADVANCED_AUTOTUNE_FINGERPRINT,
    applicable=ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_ADVANCED,
)
ADVANCED_AUTOTUNE_SELECTED = ADVANCED_AUTOTUNE_RUNTIME_PROFILE.selected
if ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_ADVANCED:
    EFFECTIVE_AUTOTUNE_CONFIG = ADVANCED_AUTOTUNE_CONFIG
    EFFECTIVE_AUTOTUNE_RUNTIME_PROFILE = ADVANCED_AUTOTUNE_RUNTIME_PROFILE
    EFFECTIVE_AUTOTUNE_SELECTED = ADVANCED_AUTOTUNE_SELECTED
else:
    EFFECTIVE_AUTOTUNE_CONFIG = AUTOTUNE_CONFIG
    EFFECTIVE_AUTOTUNE_RUNTIME_PROFILE = AUTOTUNE_RUNTIME_PROFILE
    EFFECTIVE_AUTOTUNE_SELECTED = AUTOTUNE_SELECTED
BGE_EFFECTIVE_RUNTIME = (
    AUTOTUNE_CONFIG.runtime
    if AUTOTUNE_CONFIG.runtime != "auto"
    else str(AUTOTUNE_SELECTED.get("runtime", "torch"))
)
if EFFECTIVE_AUTOTUNE_RUNTIME_PROFILE.state == "active":
    logging.info(
        "%s hardware autotune profile active: %s",
        ACTIVE_PROFILE["profile"].upper(),
        EFFECTIVE_AUTOTUNE_RUNTIME_PROFILE.fingerprint_sha256,
    )
elif EFFECTIVE_AUTOTUNE_RUNTIME_PROFILE.state not in {
    "disabled",
    "not-applicable",
}:
    logging.warning(
        "%s hardware autotune profile %s at %s; using safe defaults until "
        "pre-start calibration succeeds.",
        ACTIVE_PROFILE["profile"].upper(),
        EFFECTIVE_AUTOTUNE_RUNTIME_PROFILE.state,
        EFFECTIVE_AUTOTUNE_RUNTIME_PROFILE.profile_path,
    )
if ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_ADVANCED:
    for autotune_warning in ADVANCED_AUTOTUNE_RUNTIME_PROFILE.warnings:
        logging.warning("ADVANCED hardware autotune: %s", autotune_warning)


# Server configuration parameters (env-tunable)
def _env_int_range(
    name: str,
    default: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logging.warning(f"Invalid value for {name}, using default {default}")
        return default

    if min_value is not None and value < min_value:
        logging.warning(
            f"Value for {name} below minimum {min_value}, using default {default}"
        )
        return default

    if max_value is not None and value > max_value:
        logging.warning(
            f"Value for {name} above maximum {max_value}, using default {default}"
        )
        return default

    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logging.warning(f"Invalid value for {name}, using default {default}")
    return default


def _has_env_override(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and bool(value.strip())


def _profile_int(name: str, default: int) -> int:
    value = AUTOTUNE_SELECTED.get(name)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _advanced_section(name: str) -> Dict[str, Any]:
    advanced = ADVANCED_AUTOTUNE_SELECTED.get("advanced", {})
    if not isinstance(advanced, dict):
        return {}
    section = advanced.get(name, {})
    return section if isinstance(section, dict) else {}


def _advanced_choice(
    env_name: str,
    section: str,
    key: str,
    default: str,
    allowed: set[str],
) -> str:
    raw = os.getenv(env_name)
    if raw is not None and raw.strip():
        candidate = raw.strip().lower()
    else:
        candidate = str(_advanced_section(section).get(key, default)).lower()
    if candidate not in allowed:
        logging.warning("Invalid value for %s, using default %s", env_name, default)
        return default
    return candidate


def _advanced_bool(env_name: str, section: str, key: str, default: bool) -> bool:
    if _has_env_override(env_name):
        return _env_bool(env_name, default)
    value = _advanced_section(section).get(key)
    return value if isinstance(value, bool) else default


def _advanced_int(section: str, key: str, default: int) -> int:
    value = _advanced_section(section).get(key)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _advanced_policy_value(env_name: str, section: str, key: str) -> Any:
    raw = os.getenv(env_name)
    if raw is not None and raw.strip():
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:
            logging.warning("Invalid JSON value for %s; ignoring override", env_name)
    return _advanced_section(section).get(key)


def _effective_torch_compile() -> bool:
    policy = AUTOTUNE_CONFIG.torch_compile
    if policy == "on":
        return True
    if policy == "off":
        return False
    selected = AUTOTUNE_SELECTED.get("torch_compile")
    return selected if isinstance(selected, bool) else False


BGE_TORCH_COMPILE_ENABLED = _effective_torch_compile()


def _cache_first_model_path(model_name: str) -> str:
    """Resolve the pinned snapshot, downloading that exact revision if needed."""
    cache_root = Path(
        os.getenv(
            "HF_HUB_CACHE",
            str(Path(os.getenv("HF_HOME", "/app/model_cache")) / "hub"),
        )
    )
    repo_root = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_root = repo_root / "snapshots"
    revision = MODEL_REVISIONS.get(model_name)
    candidates = [snapshots_root / revision] if revision else []
    if not revision:
        try:
            main_revision = (
                (repo_root / "refs" / "main").read_text(encoding="utf-8").strip()
            )
        except (FileNotFoundError, OSError, UnicodeError):
            main_revision = ""
        if main_revision:
            candidates.append(snapshots_root / main_revision)
        with suppress(FileNotFoundError, OSError):
            candidates.extend(
                sorted(
                    (path for path in snapshots_root.iterdir() if path.is_dir()),
                    key=lambda path: path.name,
                    reverse=True,
                )
            )
    cached = next((path for path in candidates if path.is_dir()), None)
    if cached is not None:
        return str(cached)
    if _env_bool("HF_HUB_OFFLINE", False) or _env_bool("TRANSFORMERS_OFFLINE", False):
        revision_label = revision or "an unpinned revision"
        raise RuntimeError(
            f"Pinned model snapshot missing for {model_name}@{revision_label}; "
            "run the Smart Embedder preparation service first"
        )
    try:
        from huggingface_hub import snapshot_download

        download_options = {"repo_id": model_name}
        if revision:
            download_options["revision"] = revision
        try:
            return snapshot_download(local_files_only=True, **download_options)
        except (OSError, RuntimeError, TypeError, ValueError):
            if not revision:
                return model_name
            return snapshot_download(**download_options)
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to resolve models") from exc


def _model_revision(model_name: str) -> Optional[str]:
    """Return the configured immutable Hugging Face revision when available."""
    pinned = MODEL_REVISIONS.get(model_name)
    if pinned:
        return pinned
    resolved = Path(_cache_first_model_path(model_name))
    if resolved.parent.name == "snapshots" and resolved.name != model_name:
        return resolved.name
    return None


def build_embedding_contract(
    profile: str,
    dense_model: str,
    sparse_model: str,
) -> dict[str, Any]:
    """Build the canonical compatibility contract for stored embeddings.

    Only settings that define the vector space enter the fingerprint. Runtime
    tuning, device placement, batching and reranking are deliberately excluded.
    """
    dense_size = (
        QWEN_DENSE_VECTOR_SIZE
        if dense_model == QWEN_DENSE_EMBEDDING_MODEL
        else BGE_DENSE_VECTOR_SIZE
    )
    compatibility = {
        "schema_version": EMBEDDING_CONTRACT_SCHEMA_VERSION,
        "profile": profile,
        "dense": {
            "model": dense_model,
            "revision": _model_revision(dense_model),
            "dimension": dense_size,
            "normalization": "qdrant_cosine",
        },
        "sparse": {
            "model": sparse_model,
            "revision": _model_revision(sparse_model),
            "encoding": "indices_values",
        },
    }
    canonical = orjson.dumps(compatibility, option=orjson.OPT_SORT_KEYS)
    return {
        **compatibility,
        "fingerprint": f"sha256:{hashlib.sha256(canonical).hexdigest()}",
    }


def _load_tokenizer_cache_first(model_name: str):
    """Avoid an unnecessary network probe when model assets are already cached."""
    resolved = _cache_first_model_path(model_name)
    if resolved != model_name:
        return AutoTokenizer.from_pretrained(resolved, local_files_only=True)
    return AutoTokenizer.from_pretrained(model_name)


def _long_sequence_bge_batch_size(gpu_memory_gb: float) -> int:
    """Choose power-of-two BGE microbatches from usable GPU memory."""
    if gpu_memory_gb >= 23:
        return 32
    if gpu_memory_gb >= 15:
        return 16
    if gpu_memory_gb >= 7.5:
        return 8
    if gpu_memory_gb >= 3.5:
        return 4
    return 2


_PRECISIONS = {"fp16", "bf16", "fp32"}
_ADVANCED_RUNTIMES = {"torch", "onnx-fp32", "onnx-int8", "openvino"}
_ATTENTION_IMPLEMENTATIONS = {"sdpa", "eager", "flash_attention_2"}
_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}
ADVANCED_DENSE_RUNTIME = _advanced_choice(
    "QWEN_EMBED_RUNTIME", "dense", "runtime", "torch", _ADVANCED_RUNTIMES
)
ADVANCED_DENSE_PRECISION = _advanced_choice(
    "QWEN_EMBED_PRECISION",
    "dense",
    "precision",
    "fp16" if has_cuda else "fp32",
    _PRECISIONS,
)
ADVANCED_DENSE_ATTENTION = _advanced_choice(
    "QWEN_EMBED_ATTENTION",
    "dense",
    "attention",
    "sdpa",
    _ATTENTION_IMPLEMENTATIONS,
)
ADVANCED_DENSE_TORCH_COMPILE = _advanced_bool(
    "QWEN_EMBED_TORCH_COMPILE", "dense", "torch_compile", False
)
ADVANCED_DENSE_COMPILE_MODE = _advanced_choice(
    "QWEN_EMBED_TORCH_COMPILE_MODE",
    "dense",
    "torch_compile_mode",
    "default",
    _COMPILE_MODES,
)
ADVANCED_SPARSE_PRECISION = _advanced_choice(
    "MILCO_PRECISION",
    "sparse",
    "precision",
    "fp16" if has_cuda else "fp32",
    _PRECISIONS,
)
ADVANCED_SPARSE_RUNTIME = _advanced_choice(
    "MILCO_RUNTIME", "sparse", "runtime", "torch", _ADVANCED_RUNTIMES
)
ADVANCED_SPARSE_COMPUTE_PRECISION = _advanced_choice(
    "MILCO_COMPUTE_PRECISION",
    "sparse",
    "compute_precision",
    "bf16" if has_cuda else "fp32",
    _PRECISIONS,
)
ADVANCED_SPARSE_TORCH_COMPILE = _advanced_bool(
    "MILCO_TORCH_COMPILE", "sparse", "torch_compile", False
)
ADVANCED_SPARSE_COMPILE_MODE = _advanced_choice(
    "MILCO_TORCH_COMPILE_MODE",
    "sparse",
    "torch_compile_mode",
    "default",
    _COMPILE_MODES,
)
ADVANCED_RERANK_PRECISION = _advanced_choice(
    "QWEN_RERANK_PRECISION",
    "reranker",
    "precision",
    "fp16" if has_cuda else "fp32",
    _PRECISIONS,
)
ADVANCED_RERANK_RUNTIME = _advanced_choice(
    "QWEN_RERANK_RUNTIME", "reranker", "runtime", "torch", _ADVANCED_RUNTIMES
)
ADVANCED_RERANK_ATTENTION = _advanced_choice(
    "QWEN_RERANK_ATTENTION",
    "reranker",
    "attention",
    "sdpa",
    _ATTENTION_IMPLEMENTATIONS,
)
ADVANCED_RERANK_TORCH_COMPILE = _advanced_bool(
    "QWEN_RERANK_TORCH_COMPILE", "reranker", "torch_compile", False
)
ADVANCED_RERANK_COMPILE_MODE = _advanced_choice(
    "QWEN_RERANK_TORCH_COMPILE_MODE",
    "reranker",
    "torch_compile_mode",
    "default",
    _COMPILE_MODES,
)
ADVANCED_SCHEDULE_MODE = _advanced_choice(
    "ADVANCED_SCHEDULE_MODE",
    "schedule",
    "mode",
    "sequential",
    {"sequential", "overlap", "parallel"},
)
ADVANCED_INFERENCE_LANES = _env_int_range(
    "ADVANCED_INFERENCE_LANES",
    _advanced_int("schedule", "inference_lanes", 1),
    min_value=1,
    max_value=8,
)
ADVANCED_DENSE_PINNED_MEMORY = _advanced_bool(
    "QWEN_EMBED_PINNED_MEMORY", "dense", "pinned_memory", False
)
ADVANCED_SPARSE_PINNED_MEMORY = _advanced_bool(
    "MILCO_PINNED_MEMORY", "sparse", "pinned_memory", False
)
ADVANCED_DENSE_PREFETCH_BATCHES = _advanced_bool(
    "QWEN_EMBED_PREFETCH_BATCHES", "dense", "prefetch_batches", False
)
ADVANCED_SPARSE_PREFETCH_BATCHES = _advanced_bool(
    "MILCO_PREFETCH_BATCHES", "sparse", "prefetch_batches", False
)
ADVANCED_CPU_SPIN_POLICY = _advanced_choice(
    "ADVANCED_CPU_SPIN_POLICY",
    "schedule",
    "worker_spin",
    "default",
    {"default", "passive", "active"},
)
ADVANCED_CPU_AFFINITY = _advanced_choice(
    "ADVANCED_CPU_AFFINITY",
    "schedule",
    "affinity",
    "none",
    {"none", "compact", "socket-local"},
)
ADVANCED_CPU_ALLOW_OVERSUBSCRIPTION = _advanced_bool(
    "ADVANCED_CPU_ALLOW_OVERSUBSCRIPTION",
    "schedule",
    "allow_oversubscription",
    False,
)


def _advanced_cpu_affinity_target(
    mode: str,
    allowed_cpus,
    effective_cpu_count: int,
    cpuinfo: str,
) -> set[int]:
    """Select physical cores first, optionally within one NUMA/socket domain."""
    allowed = {int(cpu) for cpu in allowed_cpus}
    if mode == "none" or not allowed:
        return allowed
    fields: Dict[str, str] = {}
    topology: Dict[int, Tuple[str, str]] = {}
    for line in [*cpuinfo.splitlines(), ""]:
        if not line.strip():
            try:
                processor = int(fields.get("processor", ""))
            except ValueError:
                processor = -1
            if processor in allowed:
                topology[processor] = (
                    fields.get("physical id", "0"),
                    fields.get("core id", str(processor)),
                )
            fields = {}
            continue
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip().lower()] = value.strip()

    candidates = sorted(allowed)
    if mode == "socket-local" and topology:
        by_socket: Dict[str, List[int]] = defaultdict(list)
        for cpu in candidates:
            by_socket[topology.get(cpu, ("unknown", str(cpu)))[0]].append(cpu)
        candidates = max(
            by_socket.values(),
            key=lambda cpus: (
                len({topology.get(cpu, ("", str(cpu)))[1] for cpu in cpus}),
                len(cpus),
            ),
        )

    primary = []
    siblings = []
    seen_cores = set()
    for cpu in candidates:
        core = topology.get(cpu, ("unknown", str(cpu)))
        if core in seen_cores:
            siblings.append(cpu)
        else:
            seen_cores.add(core)
            primary.append(cpu)
    ordered = [*primary, *siblings]
    target_count = min(max(1, int(effective_cpu_count)), len(ordered))
    return set(ordered[:target_count])


def _apply_advanced_cpu_affinity(mode: str, effective_cpu_count: int) -> List[int]:
    get_affinity = getattr(os, "sched_getaffinity", None)
    set_affinity = getattr(os, "sched_setaffinity", None)
    if not callable(get_affinity) or not callable(set_affinity):
        return []
    try:
        allowed = set(get_affinity(0))
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        target = _advanced_cpu_affinity_target(
            mode, allowed, effective_cpu_count, cpuinfo
        )
        if mode != "none" and target and target != allowed:
            set_affinity(0, target)
        return sorted(target or allowed)
    except (OSError, UnicodeError, ValueError) as exc:
        logging.warning("Unable to apply ADVANCED CPU affinity %s: %s", mode, exc)
        return []


BGE_EMBED_MAX_LENGTH = _env_int_range(
    "BGE_EMBED_MAX_LENGTH", 8192, min_value=1, max_value=8192
)
QWEN_EMBED_MAX_LENGTH = _env_int_range(
    "QWEN_EMBED_MAX_LENGTH", 32768, min_value=1, max_value=32768
)
EMBED_LENGTH_AWARE_BATCHING = _env_bool("EMBED_LENGTH_AWARE_BATCHING", True)
BGE_RERANK_MAX_LENGTH = _env_int_range(
    "BGE_RERANK_MAX_LENGTH", 8192, min_value=1, max_value=8192
)
REQUEST_TIMEOUT = _env_int_range("REQUEST_TIMEOUT", 90, min_value=1, max_value=3600)
# Defaults tuned on RTX 4060 8GB: peak ~30-44 req/s @ conc=8 across scenarios
MAX_QUEUE_SIZE = _env_int_range("MAX_QUEUE_SIZE", 200, min_value=1, max_value=10000)
# Admission units are conservative UTF-8 bytes, not tokens (request_budget.py),
# so the floor is one full request: MAX_PASSAGES_PER_REQUEST pairs would need
# 2048 * 32768, but a client batches far smaller. 32 * QWEN_RERANK_MAX_LENGTH
# covers the largest rerank the middleware sends; below it an ordinary 20-chunk
# search is refused 413 request_budget_exceeded.
MAX_INFLIGHT_TOKENS = _env_int_range(
    "MAX_INFLIGHT_TOKENS", 1048576, min_value=1, max_value=2**40
)
MAX_INFLIGHT_OUTPUT_BYTES = _env_int_range(
    "MAX_INFLIGHT_OUTPUT_BYTES", 268435456, min_value=1, max_value=2**40
)
RATE_LIMIT_REQUESTS_PER_MINUTE = _env_int_range(
    "RATE_LIMIT_REQUESTS_PER_MINUTE", 3600, min_value=1, max_value=1000000
)
RATE_LIMIT_BURST_SIZE = _env_int_range(
    "RATE_LIMIT_BURST_SIZE", 120, min_value=1, max_value=1000000
)

RERANK_MAX_QUEUE = _env_int_range("RERANK_MAX_QUEUE", 32, min_value=1, max_value=10000)
# Hard GPU timeout for a single rerank inference. Kept strictly below
# REQUEST_TIMEOUT so the executor surfaces a 504 before the HTTP layer would.
RERANK_GPU_TIMEOUT = _env_int_range(
    "RERANK_GPU_TIMEOUT", 60, min_value=1, max_value=3600
)
QWEN_RERANK_MAX_LENGTH = _env_int_range(
    "QWEN_RERANK_MAX_LENGTH", 32768, min_value=1, max_value=32768
)
QWEN_RERANK_BATCH_SIZE = _env_int_range(
    "QWEN_RERANK_BATCH_SIZE",
    _advanced_int("reranker", "batch_size", 16),
    min_value=1,
    max_value=512,
)
MILCO_TOP_K = _env_int_range("MILCO_TOP_K", 0, min_value=0, max_value=100000)
MILCO_SOURCE_VIEW = _env_bool("MILCO_SOURCE_VIEW", True)
MILCO_MAX_LENGTH = _env_int_range(
    "MILCO_MAX_LENGTH", 8192, min_value=1, max_value=32768
)

# Structural payload limits. The queue caps the number of in-flight requests but
# not their weight; these bound a single request so one huge payload cannot pin
# the CPU/event loop in tokenization or blow up memory. Generous by default.
MAX_SENTENCES_PER_REQUEST = _env_int_range(
    "MAX_SENTENCES_PER_REQUEST", 2048, min_value=1, max_value=1000000
)
MAX_PASSAGES_PER_REQUEST = _env_int_range(
    "MAX_PASSAGES_PER_REQUEST", 2048, min_value=1, max_value=1000000
)
MAX_ITEM_BYTES = _env_int_range(
    "MAX_ITEM_BYTES", 1000000, min_value=1, max_value=100000000
)
# The three limits above are all enforced after the whole body has been read
# and deserialized, so on their own they bound nothing: 2048 sentences of
# 1 MB each is a *legal* 2 GB request, and an illegal one is still fully
# materialized before being refused. This bounds the bytes as they stream in,
# and is therefore the limit a caller actually meets first. 64 MiB sits above
# anything the middleware legitimately relays -- its own inbound cap is 25 MiB,
# and a file-ingest sub-batch is SMART_EMBEDDER_MAX_BATCH_SIZE chunks -- while
# cutting the unbounded case by orders of magnitude.
REQUEST_BODY_MAX_BYTES = _env_int_range(
    "REQUEST_BODY_MAX_BYTES", 67108864, min_value=1024, max_value=2147483648
)

# CPU intra-op thread count. On CPU a single inference parallelizes tensor ops
# (matmul, attention) across cores via OpenMP/MKL. PyTorch usually defaults to
# the physical core count, but that is not guaranteed across platforms, so set
# it explicitly. Requests are still serialized by GPU_EXECUTOR(max_workers=1),
# so one inference at a time gets all the cores. Ignored on CUDA. Default = all
# logical CPUs. Set EMBEDDER_NUM_THREADS to override (e.g. physical cores only).
_effective_cpu_count = max(
    1,
    int(AUTOTUNE_FINGERPRINT["hardware"]["cpu"].get("effective_cpu_count") or 1),
)
ADVANCED_EFFECTIVE_CPU_AFFINITY: List[int] = []
if device == "cpu":
    if ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_ADVANCED:
        if ADVANCED_CPU_SPIN_POLICY == "passive":
            os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
            os.environ["KMP_BLOCKTIME"] = "0"
        elif ADVANCED_CPU_SPIN_POLICY == "active":
            os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
            os.environ["KMP_BLOCKTIME"] = "200"
        ADVANCED_EFFECTIVE_CPU_AFFINITY = _apply_advanced_cpu_affinity(
            ADVANCED_CPU_AFFINITY,
            _effective_cpu_count,
        )
    _safe_cpu_threads = min(4, _effective_cpu_count)
    _advanced_cpu_thread_limit = _effective_cpu_count * (
        2 if ADVANCED_CPU_ALLOW_OVERSUBSCRIPTION else 1
    )
    _profile_cpu_threads = min(
        (
            _advanced_cpu_thread_limit
            if ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_ADVANCED
            else _effective_cpu_count
        ),
        (
            _advanced_int("schedule", "intra_op_threads", _safe_cpu_threads)
            if ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_ADVANCED
            else _profile_int("threads", _safe_cpu_threads)
        ),
    )
    _cpu_threads = _env_int_range(
        "EMBEDDER_NUM_THREADS",
        _profile_cpu_threads,
        min_value=1,
        max_value=(
            _advanced_cpu_thread_limit
            if ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_ADVANCED
            else 1024
        ),
    )
    _cpu_interop_threads = _env_int_range(
        "EMBEDDER_INTEROP_THREADS",
        _advanced_int("schedule", "inter_op_threads", 1),
        min_value=1,
        max_value=1024,
    )
    torch.set_num_threads(_cpu_threads)
    set_interop_threads = getattr(torch, "set_num_interop_threads", None)
    if callable(set_interop_threads):
        try:
            set_interop_threads(_cpu_interop_threads)
        except RuntimeError:
            logging.warning(
                "PyTorch inter-op threads were already initialized; keeping "
                "the existing value."
            )
    logging.info(
        "CPU inference threads set to intra=%s inter=%s "
        "(effective CPUs: %s, host logical: %s)",
        _cpu_threads,
        _cpu_interop_threads,
        _effective_cpu_count,
        os.cpu_count(),
    )


def _enforce_request_limits(items: List[str], field: str, max_items: int) -> None:
    """Reject oversized payloads with HTTP 413 before any heavy processing."""
    if len(items) > max_items:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many {field}: {len(items)} exceeds the limit of {max_items}."
            ),
        )
    for index, item in enumerate(items):
        item_bytes = len(item.encode("utf-8"))
        if item_bytes > MAX_ITEM_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"{field}[{index}] is too large: {item_bytes} bytes exceeds "
                    f"the limit of {MAX_ITEM_BYTES}."
                ),
            )


class InferenceTimeoutError(TimeoutError):
    """Signal that cooperative model work exceeded its inference deadline."""

    pass


@dataclass(frozen=True)
class InferenceDeadline:
    """Cooperative deadline checked between model micro-batches."""

    expires_at: float

    @classmethod
    def from_timeout(cls, timeout_seconds: float, safety_margin_seconds: float = 0.05):
        """Create a deadline that expires before the enclosing hard timeout."""
        return cls(time.monotonic() + max(0.0, timeout_seconds - safety_margin_seconds))

    def check(self):
        """Raise when the monotonic deadline has expired."""
        if self.remaining() == 0:
            raise InferenceTimeoutError("inference deadline expired")

    def remaining(self) -> float:
        """Return the processing budget still available to waits and workers."""
        return max(0.0, self.expires_at - time.monotonic())


def make_inference_deadline(operation_timeout: float) -> InferenceDeadline:
    """Bound cooperative work by both HTTP and operation-specific timeouts."""
    return InferenceDeadline.from_timeout(min(REQUEST_TIMEOUT, operation_timeout))


LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
RESERVED_LABEL_PREFIX = "@"
SERVICE_TOKEN_LABEL = "@middleware"
REMOVED_ENV_NAMES = {
    "API_TOKEN": "API_TOKENS (a token:label list)",
    "SMART_EMBEDDER_API_TOKEN": "SMART_EMBEDDER_API_TOKENS",
    "ALLOW_UNAUTHENTICATED_LOCAL": "SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL",
}


def parse_api_tokens(raw: str, variable: str) -> Dict[str, str]:
    """Parse ``token:label,token:label`` into a mapping of token to label.

    Args:
        raw: The raw environment value.
        variable: Name of the variable being parsed, used in error messages.

    Returns:
        Every token mapped to its audit label, in declaration order.

    Raises:
        ValueError: On a malformed entry, an out-of-charset label, a reserved
            label prefix, or a duplicate token. Never echoes the token value.
    """
    tokens: Dict[str, str] = {}
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        token, separator, label = (piece.strip() for piece in entry.partition(":"))
        if not separator or not token or not label:
            raise ValueError(f"{variable} has an entry that is not token:label")
        if label.startswith(RESERVED_LABEL_PREFIX):
            raise ValueError(
                f"{variable} label {label!r} must not start with "
                f"{RESERVED_LABEL_PREFIX!r}; that prefix is reserved for the "
                "credentials services present to each other"
            )
        if not LABEL_PATTERN.match(label):
            raise ValueError(
                f"{variable} label {label!r} must match [A-Za-z0-9._-]+. A label "
                "containing ':' usually means the token itself contains ':', "
                "which this format cannot express"
            )
        if token in tokens:
            raise ValueError(f"{variable} contains a duplicate token")
        tokens[token] = label
    return tokens


def match_bearer_token(candidate: str, tokens: Dict[str, str]) -> Optional[str]:
    """Return the audit label of the entry matching ``candidate``, if any.

    Every entry is compared even after a hit, so the time taken does not reveal
    an entry's position in the list.

    ``secrets.compare_digest`` refuses ``str`` arguments that are not
    ASCII-only, raising ``TypeError`` rather than returning False. The
    candidate arrives from an unauthenticated HTTP header (ASGI decodes
    headers latin-1, so any byte >= 0x80 lands here as a non-ASCII character),
    so comparing the strings directly turns a plain authentication failure
    into a 500. Encoding both sides to UTF-8 keeps the comparison
    constant-time -- the bytes overload has no such restriction -- while
    making every possible input a verdict instead of a crash.

    Args:
        candidate: The token presented in the ``Authorization`` header.
        tokens: The effective inbound ACL, token to label.

    Returns:
        The matched label, or ``None`` when no entry matches.
    """
    presented = candidate.encode("utf-8")
    matched: Optional[str] = None
    for known_token, label in tokens.items():
        if secrets.compare_digest(presented, known_token.encode("utf-8")):
            matched = label
    return matched


# --- Authentication ---
# Inbound bearer credentials, as a token:label list. Authentication may be
# disabled only through an explicit local-development opt-out; the secure
# default fails closed. A variable this change removed is refused outright
# rather than ignored: a stale compose file passing the old name must fail
# loudly instead of running unauthenticated.
_removed = sorted(
    name for name in REMOVED_ENV_NAMES if os.environ.get(name, "").strip()
)
if _removed:
    raise ValueError(
        "; ".join(
            f"{name} was removed -- use {REMOVED_ENV_NAMES[name]}" for name in _removed
        )
    )
API_TOKENS = parse_api_tokens(os.getenv("API_TOKENS", ""), "SMART_EMBEDDER_API_TOKENS")
# The credential the middleware presents, auto-accepted under a reserved label
# so one operator value works at both ends without being listed twice.
_service_token = os.getenv("SERVICE_TOKEN", "").strip()
if _service_token:
    if _service_token in API_TOKENS:
        raise ValueError("SERVICE_TOKEN duplicates an entry in API_TOKENS")
    API_TOKENS[_service_token] = SERVICE_TOKEN_LABEL
ALLOW_UNAUTHENTICATED_LOCAL = _env_bool(
    "SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL", False
)
if not API_TOKENS and not ALLOW_UNAUTHENTICATED_LOCAL:
    raise ValueError(
        "SMART_EMBEDDER_API_TOKENS must define at least one token:label entry "
        "unless SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL=true"
    )
EXPOSE_SMART_EMBEDDER_SWAGGER = _env_bool("EXPOSE_SMART_EMBEDDER_SWAGGER", False)
bearer_scheme = HTTPBearer(auto_error=False)
PUBLIC_AUTH_PATHS = {"/health"}
if EXPOSE_SMART_EMBEDDER_SWAGGER:
    PUBLIC_AUTH_PATHS.update({"/docs", "/redoc", "/openapi.json"})
PUBLIC_AUTH_PREFIXES = ("/docs/", "/redoc/") if EXPOSE_SMART_EMBEDDER_SWAGGER else ()


def is_public_auth_path(path: str) -> bool:
    """Return whether authentication intentionally excludes an exact path."""
    return path in PUBLIC_AUTH_PATHS or any(
        path.startswith(prefix) for prefix in PUBLIC_AUTH_PREFIXES
    )


async def require_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    """Validate the optional Bearer credential exposed through OpenAPI.

    Args:
        credentials: Parsed ``Authorization`` header, when supplied.

    Raises:
        HTTPException: If authentication is enabled and the token is missing or
            invalid.
    """
    if not API_TOKENS:
        return

    candidate = credentials.credentials if credentials else ""
    if match_bearer_token(candidate, API_TOKENS) is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


if has_cuda:
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logging.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU Memory: {gpu_mem:.2f} GB")
    logging.info(f"Available GPUs: {available_gpus}")

    embedding_batch_tuning_max_length = (
        max(BGE_EMBED_MAX_LENGTH, QWEN_EMBED_MAX_LENGTH)
        if ACTIVE_PROFILE["dense"] == QWEN_DENSE_EMBEDDING_MODEL
        else BGE_EMBED_MAX_LENGTH
    )

    # Dynamically adjust batch size based on VRAM and active embedding max length.
    if embedding_batch_tuning_max_length <= 512:
        # Can handle larger batches with shorter sequences
        device_batch_size = (
            128 if gpu_mem > 8 else 64 if gpu_mem > 6 else 32 if gpu_mem > 4 else 16
        )
        MAX_REQUESTS_IN_BATCH = 64 if gpu_mem > 8 else 32 if gpu_mem > 6 else 16
        REQUEST_FLUSH_TIMEOUT = 0.01  # More aggressive batching
    else:
        # Power-of-two microbatches cooperate with stable length grouping and
        # scale conservatively by usable VRAM. RTX 4060 8GB is validated at 8
        # through the full 8192-token BGE limit.
        device_batch_size = _long_sequence_bge_batch_size(gpu_mem)
        MAX_REQUESTS_IN_BATCH = 16
        REQUEST_FLUSH_TIMEOUT = 0.05

    _gpu_process_timeout_default = 15
else:
    logging.info("CUDA not available. Using CPU.")
    device_batch_size = 1
    MAX_REQUESTS_IN_BATCH = 8
    REQUEST_FLUSH_TIMEOUT = 0.1
    _gpu_process_timeout_default = 30

if ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_ADVANCED:
    MAX_REQUESTS_IN_BATCH = _env_int_range(
        "ADVANCED_QUEUE_BATCH",
        _advanced_int("schedule", "queue_batch", MAX_REQUESTS_IN_BATCH),
        min_value=1,
        max_value=1024,
    )
    _advanced_flush_timeout_ms = _env_int_range(
        "ADVANCED_FLUSH_TIMEOUT_MS",
        _advanced_int(
            "schedule",
            "flush_timeout_ms",
            max(1, round(REQUEST_FLUSH_TIMEOUT * 1000)),
        ),
        min_value=1,
        max_value=60000,
    )
    REQUEST_FLUSH_TIMEOUT = _advanced_flush_timeout_ms / 1000.0

# BGE embedding tuning is isolated from the advanced-profile defaults below.
# An explicit fixed batch override disables the calibrated length-aware policy.
if _has_env_override("BGE_EMBED_BATCH_SIZE"):
    batch_size = _env_int_range(
        "BGE_EMBED_BATCH_SIZE",
        device_batch_size,
        min_value=1,
        max_value=512,
    )
    BGE_EMBED_BATCH_POLICY = [
        {"max_tokens": BGE_EMBED_MAX_LENGTH, "batch_size": batch_size}
    ]
else:
    BGE_EMBED_BATCH_POLICY = normalise_batch_policy(
        AUTOTUNE_SELECTED.get("batch_policy"),
        default_batch_size=device_batch_size,
        max_length=BGE_EMBED_MAX_LENGTH,
    )
    # Compatibility alias exposed by /health and existing metrics. It denotes
    # the largest (normally shortest-sequence) microbatch in the active policy.
    batch_size = max(item["batch_size"] for item in BGE_EMBED_BATCH_POLICY)


def bge_batch_size_for_token_length(token_length: int) -> int:
    """Select the active BGE microbatch for an input token length."""
    return batch_size_for_tokens(BGE_EMBED_BATCH_POLICY, token_length)


if device == "cpu" and ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_BASIC:
    _profile_lanes = _profile_int("inference_concurrency", 1)
    _lane_limit = max(1, _effective_cpu_count // _cpu_threads)
    BGE_EMBED_INFERENCE_CONCURRENCY = _env_int_range(
        "BGE_EMBED_INFERENCE_CONCURRENCY",
        min(_profile_lanes, _lane_limit),
        min_value=1,
        max_value=max(1, _effective_cpu_count),
    )
    if _cpu_threads * BGE_EMBED_INFERENCE_CONCURRENCY > _effective_cpu_count:
        logging.warning(
            "BGE embedding lanes × threads exceeds effective CPUs; clamping "
            "concurrency from %s to %s.",
            BGE_EMBED_INFERENCE_CONCURRENCY,
            _lane_limit,
        )
        BGE_EMBED_INFERENCE_CONCURRENCY = _lane_limit
else:
    BGE_EMBED_INFERENCE_CONCURRENCY = 1

_profile_onnx_execution_mode = str(
    AUTOTUNE_SELECTED.get("onnx_execution_mode", "sequential")
)
BGE_ONNX_EXECUTION_MODE = (
    AUTOTUNE_CONFIG.onnx_execution_mode
    if AUTOTUNE_CONFIG.onnx_execution_mode != "auto"
    else _profile_onnx_execution_mode
)
if BGE_ONNX_EXECUTION_MODE not in {"sequential", "parallel"}:
    BGE_ONNX_EXECUTION_MODE = "sequential"
_profile_onnx_spin_policy = str(AUTOTUNE_SELECTED.get("onnx_spin_policy", "default"))
BGE_ONNX_SPIN_POLICY = (
    AUTOTUNE_CONFIG.onnx_spin_policy
    if AUTOTUNE_CONFIG.onnx_spin_policy != "auto"
    else _profile_onnx_spin_policy
)
if BGE_ONNX_SPIN_POLICY not in {"default", "bounded", "disabled"}:
    BGE_ONNX_SPIN_POLICY = "default"
BGE_ONNX_INTER_OP_THREADS = _profile_int("onnx_inter_op_threads", 1)

logging.info(
    "BGE embedding runtime: batch_policy=%s inference_concurrency=%s "
    "torch_compile=%s onnx_execution=%s onnx_spin=%s "
    "onnx_inter_op_threads=%s autotune_state=%s",
    BGE_EMBED_BATCH_POLICY,
    BGE_EMBED_INFERENCE_CONCURRENCY,
    BGE_TORCH_COMPILE_ENABLED,
    BGE_ONNX_EXECUTION_MODE,
    BGE_ONNX_SPIN_POLICY,
    BGE_ONNX_INTER_OP_THREADS,
    AUTOTUNE_RUNTIME_PROFILE.state,
)

# Hard GPU timeout for a single embedding inference. Defaults are device-specific
# (15s GPU / 30s CPU) but can be raised via EMBED_GPU_TIMEOUT for heavy batches or
# long sequences. Keep strictly below REQUEST_TIMEOUT so the executor surfaces a
# 504 before the HTTP layer would.
GPU_PROCESS_TIMEOUT = _env_int_range(
    "EMBED_GPU_TIMEOUT", _gpu_process_timeout_default, min_value=1, max_value=3600
)


def normalise_max_batch_tokens_policy(
    value: Any, *, default_max_batch_tokens: int, max_length: int
) -> List[Dict[str, int]]:
    """Validate and complete a length-aware padded-token policy.

    Invalid entries are discarded, duplicate length buckets keep their final
    value, and the resulting policy always covers ``max_length``.
    """
    policy: List[Dict[str, int]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                token_limit = int(item.get("max_tokens"))
                batch_tokens = int(item.get("max_batch_tokens"))
            except (TypeError, ValueError):
                continue
            if token_limit > 0 and batch_tokens > 0:
                policy.append(
                    {
                        "max_tokens": min(token_limit, max_length),
                        "max_batch_tokens": batch_tokens,
                    }
                )
    deduplicated = {item["max_tokens"]: item["max_batch_tokens"] for item in policy}
    normalised = [
        {"max_tokens": limit, "max_batch_tokens": deduplicated[limit]}
        for limit in sorted(deduplicated)
    ]
    if not normalised:
        normalised = [
            {
                "max_tokens": max_length,
                "max_batch_tokens": max(1, default_max_batch_tokens),
            }
        ]
    elif normalised[-1]["max_tokens"] < max_length:
        normalised.append(
            {
                "max_tokens": max_length,
                "max_batch_tokens": normalised[-1]["max_batch_tokens"],
            }
        )
    return normalised


def max_batch_tokens_for_length(policy: List[Dict[str, int]], token_length: int) -> int:
    """Select the first padded-token budget covering an input length."""
    requested = max(1, int(token_length))
    for item in policy:
        if requested <= item["max_tokens"]:
            return item["max_batch_tokens"]
    return policy[-1]["max_batch_tokens"]


# Advanced embedding micro-batch bounds. These bound memory in the Qwen dense
# and MILCO sparse paths by sentence count and token budget.
_qwen_embed_default_batch = _env_int_range(
    "QWEN_EMBED_BATCH_SIZE",
    max(1, min(device_batch_size, 8)),
    min_value=1,
    max_value=1000000,
)
_qwen_embed_default_tokens = _env_int_range(
    "QWEN_EMBED_MAX_BATCH_TOKENS",
    8192,
    min_value=1,
    max_value=1000000,
)
_qwen_embed_batch_policy_value = _advanced_policy_value(
    "QWEN_EMBED_BATCH_POLICY", "dense", "batch_policy"
)
_qwen_embed_tokens_policy_value = _advanced_policy_value(
    "QWEN_EMBED_MAX_BATCH_TOKENS_POLICY",
    "dense",
    "max_batch_tokens_policy",
)
QWEN_EMBED_POLICY_ACTIVE = bool(
    isinstance(_qwen_embed_batch_policy_value, list) and _qwen_embed_batch_policy_value
)
QWEN_EMBED_TOKENS_POLICY_ACTIVE = bool(
    isinstance(_qwen_embed_tokens_policy_value, list)
    and _qwen_embed_tokens_policy_value
)
QWEN_EMBED_BATCH_POLICY = normalise_batch_policy(
    _qwen_embed_batch_policy_value,
    default_batch_size=_qwen_embed_default_batch,
    max_length=QWEN_EMBED_MAX_LENGTH,
    max_batch_size=1000000,
)
QWEN_EMBED_MAX_BATCH_TOKENS_POLICY = normalise_max_batch_tokens_policy(
    _qwen_embed_tokens_policy_value,
    default_max_batch_tokens=_qwen_embed_default_tokens,
    max_length=QWEN_EMBED_MAX_LENGTH,
)
QWEN_EMBED_BATCH_SIZE = max(item["batch_size"] for item in QWEN_EMBED_BATCH_POLICY)
QWEN_EMBED_MAX_BATCH_TOKENS = max(
    item["max_batch_tokens"] for item in QWEN_EMBED_MAX_BATCH_TOKENS_POLICY
)

_milco_default_batch = _env_int_range(
    "MILCO_BATCH_SIZE",
    max(1, min(device_batch_size, 32)),
    min_value=1,
    max_value=1000000,
)
_milco_default_tokens = _env_int_range(
    "MILCO_MAX_BATCH_TOKENS",
    8192,
    min_value=1,
    max_value=1000000,
)
_milco_batch_policy_value = _advanced_policy_value(
    "MILCO_BATCH_POLICY", "sparse", "batch_policy"
)
_milco_tokens_policy_value = _advanced_policy_value(
    "MILCO_MAX_BATCH_TOKENS_POLICY",
    "sparse",
    "max_batch_tokens_policy",
)
MILCO_POLICY_ACTIVE = bool(
    isinstance(_milco_batch_policy_value, list) and _milco_batch_policy_value
)
MILCO_TOKENS_POLICY_ACTIVE = bool(
    isinstance(_milco_tokens_policy_value, list) and _milco_tokens_policy_value
)
MILCO_BATCH_POLICY = normalise_batch_policy(
    _milco_batch_policy_value,
    default_batch_size=_milco_default_batch,
    max_length=MILCO_MAX_LENGTH,
    max_batch_size=1000000,
)
MILCO_MAX_BATCH_TOKENS_POLICY = normalise_max_batch_tokens_policy(
    _milco_tokens_policy_value,
    default_max_batch_tokens=_milco_default_tokens,
    max_length=MILCO_MAX_LENGTH,
)
MILCO_BATCH_SIZE = max(item["batch_size"] for item in MILCO_BATCH_POLICY)
MILCO_MAX_BATCH_TOKENS = max(
    item["max_batch_tokens"] for item in MILCO_MAX_BATCH_TOKENS_POLICY
)
MILCO_PROJECTION_CHUNK_TOKENS = _env_int_range(
    "MILCO_PROJECTION_CHUNK_TOKENS",
    128,
    min_value=1,
    max_value=8192,
)
BGE_RERANK_BATCH_SIZE = _env_int_range(
    "BGE_RERANK_BATCH_SIZE",
    max(1, min(device_batch_size, 32)),
    min_value=1,
    max_value=512,
)
_qwen_rerank_batch_policy_value = _advanced_policy_value(
    "QWEN_RERANK_BATCH_POLICY", "reranker", "batch_policy"
)
_qwen_rerank_tokens_policy_value = _advanced_policy_value(
    "QWEN_RERANK_MAX_BATCH_TOKENS_POLICY",
    "reranker",
    "max_batch_tokens_policy",
)
QWEN_RERANK_POLICY_ACTIVE = bool(
    isinstance(_qwen_rerank_batch_policy_value, list)
    and _qwen_rerank_batch_policy_value
)
QWEN_RERANK_TOKENS_POLICY_ACTIVE = bool(
    isinstance(_qwen_rerank_tokens_policy_value, list)
    and _qwen_rerank_tokens_policy_value
)
QWEN_RERANK_BATCH_POLICY = normalise_batch_policy(
    _qwen_rerank_batch_policy_value,
    default_batch_size=QWEN_RERANK_BATCH_SIZE,
    max_length=QWEN_RERANK_MAX_LENGTH,
    max_batch_size=512,
)
QWEN_RERANK_MAX_BATCH_TOKENS_POLICY = normalise_max_batch_tokens_policy(
    _qwen_rerank_tokens_policy_value,
    default_max_batch_tokens=8192,
    max_length=QWEN_RERANK_MAX_LENGTH,
)
QWEN_RERANK_BATCH_SIZE = max(item["batch_size"] for item in QWEN_RERANK_BATCH_POLICY)


# --- Rate Limiter for Backpressure Protection ---


class RateLimiter:
    """Token bucket algorithm for rate limiting per client IP."""

    def __init__(self, requests_per_minute: int = 100, burst_size: int = 20):
        """Configure per-client refill rate and burst capacity.

        Args:
            requests_per_minute: Tokens restored per minute for each client.
            burst_size: Maximum tokens retained by one client bucket.
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.buckets = defaultdict(
            lambda: {"tokens": burst_size, "last_update": time.monotonic()}
        )

    def start_cleanup_task(self):
        """Schedule bucket cleanup inside a running event loop."""
        asyncio.create_task(self._cleanup_old_buckets())

    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client can make a request."""
        now = time.monotonic()
        bucket = self.buckets[client_id]

        time_passed = now - bucket["last_update"]
        tokens_to_add = time_passed * (self.requests_per_minute / 60)

        bucket["tokens"] = min(self.burst_size, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False

    async def _cleanup_old_buckets(self):
        """Remove unused buckets older than 1 hour."""
        while True:
            await asyncio.sleep(3600)
            now = time.monotonic()
            old_keys = [
                k for k, v in self.buckets.items() if now - v["last_update"] > 3600
            ]
            for k in old_keys:
                del self.buckets[k]


# --- Shared inference executor ---
# BASIC and the safe ADVANCED profile remain single-lane. A measured ADVANCED
# schedule may select two or more lanes so embedding and reranking can overlap;
# the isolated tuner must prove memory, quality and mixed-workload performance
# before such a profile is persisted.
_global_inference_lanes = (
    ADVANCED_INFERENCE_LANES
    if ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_ADVANCED
    else 1
)
GPU_EXECUTOR = ThreadPoolExecutor(max_workers=_global_inference_lanes)


# --- 2. Embedding Backends ---


def _token_counts(batch_data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Return useful and padded tokens for a tokenizer batch when supported."""
    attention_mask = batch_data.get("attention_mask")
    if attention_mask is None:
        return None

    try:
        if hasattr(attention_mask, "numel") and hasattr(attention_mask, "sum"):
            padded_tokens = int(attention_mask.numel())
            useful_sum = attention_mask.sum()
            useful_tokens = int(
                useful_sum.item() if hasattr(useful_sum, "item") else useful_sum
            )
        elif hasattr(attention_mask, "shape") and hasattr(attention_mask, "size"):
            padded_tokens = int(attention_mask.size)
            useful_tokens = int(attention_mask.sum())
        else:
            rows = attention_mask
            if rows and not isinstance(rows[0], (list, tuple)):
                rows = [rows]
            useful_tokens = sum(sum(int(value) for value in row) for row in rows)
            padded_tokens = sum(len(row) for row in rows)
    except (AttributeError, IndexError, TypeError, ValueError):
        logging.debug(
            "Skipping token telemetry for unsupported attention mask type %s",
            type(attention_mask).__name__,
        )
        return None

    if padded_tokens < 1:
        return None
    return useful_tokens, padded_tokens


def record_embedding_token_batch(batch_data: Dict[str, Any], backend: str) -> None:
    """Record useful and padded tokens from an existing tokenizer result."""
    counts = _token_counts(batch_data)
    if counts is None:
        return
    useful_tokens, padded_tokens = counts
    embedding_input_tokens.labels(backend=backend).inc(useful_tokens)
    embedding_padded_tokens.labels(backend=backend).inc(padded_tokens)
    embedding_tokenized_batches.labels(backend=backend).inc()


def record_advanced_token_batch(batch_data: Dict[str, Any], backend: str) -> None:
    """Record token occupancy for one ADVANCED backend tensor batch."""
    counts = _token_counts(batch_data)
    if counts is None:
        return
    useful_tokens, padded_tokens = counts
    advanced_backend_input_tokens.labels(backend=backend).inc(useful_tokens)
    advanced_backend_padded_tokens.labels(backend=backend).inc(padded_tokens)
    advanced_backend_tokenized_batches.labels(backend=backend).inc()


def observe_advanced_phase(backend: str, phase: str, started_at: float) -> None:
    """Observe a backend phase using a monotonic start timestamp."""
    advanced_backend_phase_duration.labels(backend=backend, phase=phase).observe(
        time.monotonic() - started_at
    )


@dataclass
class AdvancedDevicePhaseTimer:
    """Measure CUDA work with events without synchronizing at phase boundaries."""

    host_started_at: float
    start_event: Any = None
    end_event: Any = None
    host_duration: float = 0.0


def start_advanced_device_phase() -> AdvancedDevicePhaseTimer:
    """Start a CUDA-event or host-clock phase measurement."""
    timer = AdvancedDevicePhaseTimer(host_started_at=time.monotonic())
    if has_cuda:
        timer.start_event = torch.cuda.Event(enable_timing=True)
        timer.end_event = torch.cuda.Event(enable_timing=True)
        timer.start_event.record()
    return timer


def stop_advanced_device_phase(timer: AdvancedDevicePhaseTimer) -> None:
    """Mark the end of a device phase without forcing immediate sync."""
    if timer.end_event is not None:
        timer.end_event.record()
    else:
        timer.host_duration = time.monotonic() - timer.host_started_at


def observe_advanced_device_phase(
    backend: str, phase: str, timer: AdvancedDevicePhaseTimer
) -> None:
    """Finalize a device phase and publish its duration metric."""
    if timer.end_event is not None:
        timer.end_event.synchronize()
        duration = timer.start_event.elapsed_time(timer.end_event) / 1000.0
    else:
        duration = timer.host_duration
    advanced_backend_phase_duration.labels(backend=backend, phase=phase).observe(
        duration
    )


def begin_advanced_memory_measurement() -> None:
    """Reset CUDA peak statistics before an ADVANCED backend operation."""
    if has_cuda:
        torch.cuda.reset_peak_memory_stats()


def record_advanced_memory_measurement(backend: str) -> None:
    """Publish CUDA allocator and process peak memory for one backend."""
    if has_cuda:
        advanced_backend_gpu_memory.labels(backend=backend, kind="allocated").set(
            torch.cuda.memory_allocated()
        )
        advanced_backend_gpu_memory.labels(backend=backend, kind="reserved").set(
            torch.cuda.memory_reserved()
        )
        advanced_backend_gpu_memory.labels(backend=backend, kind="peak_allocated").set(
            torch.cuda.max_memory_allocated()
        )
        advanced_backend_gpu_memory.labels(backend=backend, kind="peak_reserved").set(
            torch.cuda.max_memory_reserved()
        )
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            peak_kib = next(
                int(line.split()[1])
                for line in status_file
                if line.startswith("VmHWM:")
            )
        advanced_backend_rss_peak.labels(backend=backend).set(peak_kib * 1024)
    except (OSError, StopIteration, ValueError):
        pass


class InstrumentedEmbeddingTokenizer:
    """Delegate to a tokenizer and observe batches it already creates."""

    def __init__(self, tokenizer: Any, backend: str):
        """Wrap a tokenizer with synchronized batch instrumentation.

        Args:
            tokenizer: Underlying tokenizer shared by inference paths.
            backend: Metric label identifying the embedding backend.
        """
        self._tokenizer = tokenizer
        self._backend = backend
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped tokenizer."""
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        """Tokenize under lock and record the resulting occupied positions."""
        with self._lock:
            batch_data = self._tokenizer(*args, **kwargs)
        record_embedding_token_batch(batch_data, self._backend)
        return batch_data

    def token_lengths(self, sentences: List[str], max_length: int) -> List[int]:
        """Count exact policy lengths without double-counting metrics."""
        with self._lock:
            encoded = self._tokenizer(
                sentences,
                padding=False,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
        return [len(token_ids) for token_ids in encoded["input_ids"]]


class BgeM3EmbeddingBackend:
    """Encapsulate the BGE-M3 model, handling loading and inference.

    This backend provides BGE-M3 dense, sparse, and ColBERT embeddings.
    """

    def __init__(
        self, model_name: str, device: str = device, devices: Optional[List[str]] = None
    ):
        """Load BGE-M3 and prepare its device-specific serving runtime.

        Args:
            model_name: Hugging Face model identifier or local snapshot.
            device: Primary Torch device.
            devices: Optional multi-device placement list.
        """
        self.model_name = model_name
        self.device = device
        self.devices = devices
        use_fp16 = self.device == "cuda" or (
            devices and any("cuda" in d for d in devices)
        )

        logging.info(
            f"Initializing model '{self.model_name}' on '{self.device}' "
            f"with FP16: {use_fp16}"
        )
        model_path = _cache_first_model_path(self.model_name)

        if devices:
            logging.info(f"Using multiple devices: {devices}")
            self.model = BGEM3FlagModel(model_path, devices=devices, use_fp16=use_fp16)
        else:
            self.model = BGEM3FlagModel(
                model_path, device=self.device, use_fp16=use_fp16
            )
        self.model.tokenizer = InstrumentedEmbeddingTokenizer(
            self.model.tokenizer, backend="bge"
        )
        uses_cuda_backend = self._uses_cuda_backend()
        self._lane_executor = (
            ThreadPoolExecutor(max_workers=BGE_EMBED_INFERENCE_CONCURRENCY)
            if not uses_cuda_backend and BGE_EMBED_INFERENCE_CONCURRENCY > 1
            else None
        )

        original_inner_model = None
        self.torch_compile_active = False
        if BGE_TORCH_COMPILE_ENABLED:
            original_inner_model = self._try_compile_inner_model()
            self.torch_compile_active = original_inner_model is not None
        else:
            logging.info("BGE torch.compile skipped (effective setting disabled).")

        # Warm up CUDA kernels to keep the first request latency predictable.
        # If torch.compile fails lazily during this first forward, restore the
        # original module and retry so startup still succeeds without compile.
        if uses_cuda_backend:
            try:
                self._warm_up()
            except Exception as exc:
                if original_inner_model is None:
                    raise
                self.model.model = original_inner_model
                self.torch_compile_active = False
                logging.warning(
                    "BGE torch.compile warm-up failed; "
                    "restored original model and retrying without compile: %s",
                    exc,
                )
                self._warm_up()
            else:
                if self.torch_compile_active:
                    logging.info(
                        "BGE torch.compile warm-up succeeded; compiled model active."
                    )
        logging.info("Model ready.")

    @staticmethod
    def _merge_encode_results(parts: List[dict]) -> dict:
        if len(parts) == 1:
            return parts[0]
        merged = {}
        for key in ("dense_vecs", "lexical_weights", "colbert_vecs"):
            # BGEM3FlagModel.encode always includes every embedding-type key, but
            # sets unrequested ones to None. The single-part fast path passes that
            # None through untouched; the merge path must skip it, or concatenating
            # across parts iterates None (e.g. colbert_vecs when return_colbert=False).
            values = [part[key] for part in parts if part.get(key) is not None]
            if not values:
                continue
            if key == "dense_vecs":
                merged[key] = np.concatenate(values, axis=0)
            else:
                merged[key] = [item for value in values for item in value]
        return merged

    def _encode_once(
        self,
        sentences: List[str],
        *,
        selected_batch_size: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict:
        return self.model.encode(
            sentences,
            batch_size=selected_batch_size,
            max_length=BGE_EMBED_MAX_LENGTH,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )

    def _encode_group(
        self,
        sentences: List[str],
        *,
        selected_batch_size: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict:
        lanes = min(BGE_EMBED_INFERENCE_CONCURRENCY, len(sentences))
        if lanes <= 1 or self._lane_executor is None:
            return self._encode_once(
                sentences,
                selected_batch_size=selected_batch_size,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert=return_colbert,
            )
        chunk_size = (len(sentences) + lanes - 1) // lanes
        chunks = [
            sentences[start : start + chunk_size]
            for start in range(0, len(sentences), chunk_size)
        ]
        futures = [
            self._lane_executor.submit(
                self._encode_once,
                chunk,
                selected_batch_size=selected_batch_size,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert=return_colbert,
            )
            for chunk in chunks
        ]
        return self._merge_encode_results([future.result() for future in futures])

    def close(self) -> None:
        """Stop the optional CPU inference-lane executor."""
        if self._lane_executor is not None:
            self._lane_executor.shutdown(wait=False, cancel_futures=True)

    def _uses_cuda_backend(self) -> bool:
        if self.device and self.device.lower().startswith("cuda"):
            return True
        return bool(
            self.devices
            and any(d and d.lower().startswith("cuda") for d in self.devices)
        )

    def _try_compile_inner_model(self):
        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            logging.warning("BGE torch.compile skipped: torch.compile unavailable.")
            return None

        inner_model = getattr(self.model, "model", None)
        if inner_model is None:
            logging.warning(
                "BGE torch.compile skipped: BGEM3FlagModel has no .model attribute."
            )
            return None

        compile_mode = AUTOTUNE_CONFIG.torch_compile_mode
        try:
            self.model.model = compile_fn(
                inner_model,
                dynamic=True,
                mode=compile_mode,
            )
        except Exception as exc:
            self.model.model = inner_model
            logging.warning("BGE torch.compile disabled: %s", exc)
            return None

        logging.info(
            "BGE torch.compile enabled (dynamic=True, mode=%s).",
            compile_mode,
        )
        return inner_model

    def _warm_up(self) -> None:
        logging.info("Performing model warm-up...")
        warmup_sentences = ["warm-up"] * batch_size
        _ = self.model.encode(
            warmup_sentences,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            max_length=BGE_EMBED_MAX_LENGTH,
        )
        _ = self.model.encode(
            warmup_sentences,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            max_length=BGE_EMBED_MAX_LENGTH,
        )

    def embed(
        self,
        sentences: List[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
        normalize_dense: bool = False,
        deadline=None,
    ) -> Dict[str, Union[np.ndarray, List]]:
        """Perform embedding of a list of sentences with configurable options.

        Generate embeddings for the provided sentences based on the specified options.
        Can selectively compute dense, sparse, and ColBERT embeddings, and optionally
        apply L2 normalization to dense vectors.

        Args:
            sentences: List of texts to process
            return_dense: If True, computes and returns dense vectors
            return_sparse: If True, computes and returns lexical weights (sparse)
            return_colbert: If True, computes and returns ColBERT vectors
            normalize_dense: If True, normalizes dense vectors (L2)
            deadline: Optional cooperative deadline checked between microbatches.

        Returns:
            Dictionary containing the requested embedding types
        """
        if deadline is not None:
            deadline.check()
        token_lengths = self.model.tokenizer.token_lengths(
            sentences, BGE_EMBED_MAX_LENGTH
        )
        groups = []
        for sentence, token_length in zip(sentences, token_lengths):
            selected_batch_size = bge_batch_size_for_token_length(token_length)
            if groups and groups[-1][0] == selected_batch_size:
                groups[-1][1].append(sentence)
            else:
                groups.append((selected_batch_size, [sentence]))

        parts = []
        for selected_batch_size, group_sentences in groups:
            if deadline is not None:
                deadline.check()
            parts.append(
                self._encode_group(
                    group_sentences,
                    selected_batch_size=selected_batch_size,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert=return_colbert,
                )
            )
        result = self._merge_encode_results(parts)

        if (
            return_dense
            and normalize_dense
            and "dense_vecs" in result
            and len(result["dense_vecs"]) > 0
        ):
            dense_vecs = np.array(result["dense_vecs"])
            norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # avoid division by zero
            result["dense_vecs"] = dense_vecs / norms

        return result


def iter_token_budget_batches(
    encoded_inputs: dict,
    max_items: int,
    max_tokens: int,
    *,
    batch_policy: Optional[List[Dict[str, int]]] = None,
    max_batch_tokens_policy: Optional[List[Dict[str, int]]] = None,
):
    """Yield sub-batches of encoded inputs bounded by item count and token budget.

    Token budget is computed as items * max_sequence_length_in_batch, which
    mirrors the padded tensor shape that the model forward pass will allocate.
    """
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    current_ids = []
    current_mask = []
    current_max_len = 0

    for ids, mask in zip(input_ids, attention_mask):
        token_count = len(ids)
        candidate_max_len = max(current_max_len, token_count)
        candidate_items = len(current_ids) + 1
        candidate_tokens = candidate_items * candidate_max_len
        item_limit = (
            batch_size_for_tokens(batch_policy, candidate_max_len)
            if batch_policy
            else max_items
        )
        token_limit = (
            max_batch_tokens_for_length(max_batch_tokens_policy, candidate_max_len)
            if max_batch_tokens_policy
            else max_tokens
        )
        over_item_budget = len(current_ids) >= item_limit
        over_token_budget = bool(current_ids) and candidate_tokens > token_limit

        if over_item_budget or over_token_budget:
            yield {"input_ids": current_ids, "attention_mask": current_mask}
            current_ids = []
            current_mask = []
            current_max_len = 0

        current_ids.append(ids)
        current_mask.append(mask)
        current_max_len = max(current_max_len, token_count)

    if current_ids:
        yield {"input_ids": current_ids, "attention_mask": current_mask}


def _chunk_token_estimate(chunk: dict) -> int:
    """Padded token count for a chunk: items * longest sequence.

    Mirrors the accounting in iter_token_budget_batches, which is also the
    shape the forward pass actually allocates.
    """
    ids = chunk["input_ids"]
    if not ids:
        return 0
    return len(ids) * max(len(item) for item in ids)


def _split_chunk(chunk: dict) -> tuple:
    """Split a chunk into two halves, biased so the first is never empty."""
    ids = chunk["input_ids"]
    mask = chunk["attention_mask"]
    midpoint = max(1, len(ids) // 2)
    return (
        {"input_ids": ids[:midpoint], "attention_mask": mask[:midpoint]},
        {"input_ids": ids[midpoint:], "attention_mask": mask[midpoint:]},
    )


def _torch_dtype_for_precision(precision: str):
    if precision == "fp16":
        return getattr(torch, "float16", None)
    if precision == "bf16":
        return getattr(torch, "bfloat16", None)
    if precision == "fp32":
        return getattr(torch, "float32", None)
    raise ValueError(f"unsupported precision {precision}")


def _move_advanced_tensor_to_device(
    value: Any, target_device: str, *, pinned_memory: bool
) -> Any:
    """Move one tokenizer tensor with an optional measured async-copy policy."""
    if not hasattr(value, "to"):
        return value
    use_async_copy = bool(pinned_memory and has_cuda)
    if use_async_copy and hasattr(value, "pin_memory"):
        try:
            value = value.pin_memory()
        except (RuntimeError, TypeError):
            use_async_copy = False
    try:
        return value.to(target_device, non_blocking=use_async_copy)
    except TypeError:
        return value.to(target_device)


def _iter_double_buffered(items, prepare: Callable[[Any], Any], *, enabled: bool):
    """Prepare the next CPU batch while the caller executes the current one.

    Tokenization of the complete request already happens before micro-batching.
    This bounded one-item look-ahead overlaps the remaining CPU pad work with
    the current model forward without retaining every padded tensor in memory.
    """
    iterator = iter(items)
    try:
        current_item = next(iterator)
    except StopIteration:
        return

    current = prepare(current_item)
    if not enabled:
        yield current
        for item in iterator:
            yield prepare(item)
        return

    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="advanced-batch-prefetch",
    )
    next_future = None
    try:
        while True:
            try:
                next_item = next(iterator)
            except StopIteration:
                yield current
                break
            next_future = executor.submit(prepare, next_item)
            yield current
            current = next_future.result()
            next_future = None
    finally:
        if next_future is not None:
            next_future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _compile_advanced_model(
    model: Any, *, enabled: bool, mode: str, backend: str
) -> Any:
    if not enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        raise RuntimeError(f"{backend} selected torch.compile but it is unavailable")
    logging.info("Compiling %s with torch.compile mode=%s", backend, mode)
    return compile_fn(model, mode=mode)


class QwenDenseEmbeddingBackend:
    """Produce dense embeddings with Qwen3-Embedding."""

    def __init__(self, model_name: str, device: str = device):
        """Load the selected Qwen dense runtime and its tokenizer.

        Args:
            model_name: Qwen model identifier or local snapshot.
            device: Torch device used by the dense backend.

        Raises:
            RuntimeError: If the selected runtime is unavailable on the device.
        """
        self.model_name = model_name
        self.device = device
        self.max_length = QWEN_EMBED_MAX_LENGTH
        if ADVANCED_DENSE_RUNTIME not in {"torch", "onnx-fp32", "onnx-int8"}:
            raise RuntimeError(
                f"Qwen dense runtime {ADVANCED_DENSE_RUNTIME!r} is unsupported"
            )
        logging.info(f"Initializing Qwen dense embedding model '{model_name}'")
        model_path = _cache_first_model_path(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Pre-trip transformers' gate: embed_dense() uses a deliberate two-step
        # encode+pad (token-budget batching between the two), so the benign
        # "fast tokenizer / use __call__" notice from tokenizer.pad() does not
        # apply. Sets transformers' own flag; the python warnings module is unused.
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        self.ort_session = None
        if ADVANCED_DENSE_RUNTIME in {"onnx-fp32", "onnx-int8"}:
            if self.device != "cpu":
                raise RuntimeError(
                    f"Qwen dense runtime {ADVANCED_DENSE_RUNTIME} is CPU-only"
                )
            if model_path == self.model_name:
                raise RuntimeError(
                    "Qwen ORT runtime requires a cached local model snapshot"
                )
            instruction_sets = tuple(
                ADVANCED_AUTOTUNE_FINGERPRINT.get("hardware", {})
                .get("cpu", {})
                .get("instruction_sets", ())
            )
            self.ort_session = QwenDenseOnnxSession(
                snapshot_path=str(model_path),
                runtime=ADVANCED_DENSE_RUNTIME,
                tokenizer=self.tokenizer,
                torch_module=torch,
                threads=_cpu_threads,
                inter_op_threads=_cpu_interop_threads,
                spin_policy=ADVANCED_CPU_SPIN_POLICY,
                instruction_sets=instruction_sets,
            )
            self.model = None
            logging.info(
                "Qwen dense embedding model ready through %s.",
                ADVANCED_DENSE_RUNTIME,
            )
            return
        dense_dtype = _torch_dtype_for_precision(ADVANCED_DENSE_PRECISION)
        model_kwargs = {"dtype": dense_dtype} if dense_dtype is not None else {}
        model_kwargs["attn_implementation"] = ADVANCED_DENSE_ATTENTION
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        self.model = self.model.to(self.device).eval()
        self.model = _compile_advanced_model(
            self.model,
            enabled=ADVANCED_DENSE_TORCH_COMPILE,
            mode=ADVANCED_DENSE_COMPILE_MODE,
            backend="qwen_dense",
        )
        logging.info("Qwen dense embedding model ready.")

    def embed_dense(
        self, sentences: List[str], normalize_dense: bool = False, deadline=None
    ) -> Dict[str, np.ndarray]:
        """Encode dense vectors under item and padded-token bounds.

        Args:
            sentences: Texts in caller order.
            normalize_dense: L2-normalize output vectors.
            deadline: Optional cooperative deadline checked between microbatches.

        Returns:
            A ``dense_vecs`` FP32 matrix with one row per input.
        """
        if not sentences:
            return {"dense_vecs": np.array([])}

        advanced_backend_requests.labels(backend="qwen_dense").inc()
        begin_advanced_memory_measurement()
        max_item_tokens = max(
            1,
            min(
                self.max_length,
                max_batch_tokens_for_length(
                    QWEN_EMBED_MAX_BATCH_TOKENS_POLICY, self.max_length
                )
                if QWEN_EMBED_TOKENS_POLICY_ACTIVE
                else QWEN_EMBED_MAX_BATCH_TOKENS,
            ),
        )
        phase_started_at = time.monotonic()
        encoded = self.tokenizer(
            sentences,
            padding=False,
            truncation=True,
            max_length=max_item_tokens,
            return_tensors=None,
        )
        observe_advanced_phase("qwen_dense", "tokenize", phase_started_at)

        chunks = iter_token_budget_batches(
            encoded,
            QWEN_EMBED_BATCH_SIZE,
            QWEN_EMBED_MAX_BATCH_TOKENS,
            batch_policy=(
                QWEN_EMBED_BATCH_POLICY if QWEN_EMBED_POLICY_ACTIVE else None
            ),
            max_batch_tokens_policy=(
                QWEN_EMBED_MAX_BATCH_TOKENS_POLICY
                if QWEN_EMBED_TOKENS_POLICY_ACTIVE
                else None
            ),
        )
        ort_session = getattr(self, "ort_session", None)

        def prepare_dense_batch(chunk):
            phase_started_at = time.monotonic()
            padded = self.tokenizer.pad(
                chunk,
                padding=True,
                return_tensors=("np" if ort_session is not None else "pt"),
            )
            observe_advanced_phase("qwen_dense", "tokenize", phase_started_at)
            return padded

        dense_parts = []
        for padded in _iter_double_buffered(
            chunks,
            prepare_dense_batch,
            enabled=ADVANCED_DENSE_PREFETCH_BATCHES,
        ):
            if deadline is not None:
                deadline.check()
            record_embedding_token_batch(padded, backend="qwen")
            record_advanced_token_batch(padded, backend="qwen_dense")
            if ort_session is not None:
                phase_started_at = time.monotonic()
                observe_advanced_phase("qwen_dense", "host_to_device", phase_started_at)
                phase_started_at = time.monotonic()
                dense_cpu = ort_session.run(padded)
                observe_advanced_phase("qwen_dense", "forward", phase_started_at)
                phase_started_at = time.monotonic()
                if normalize_dense:
                    norms = np.maximum(
                        np.linalg.norm(dense_cpu, axis=1, keepdims=True),
                        1e-12,
                    )
                    dense_cpu = dense_cpu / norms
                if dense_cpu.shape[-1] != QWEN_DENSE_VECTOR_SIZE:
                    raise RuntimeError(
                        f"{self.model_name} returned dense dimension "
                        f"{dense_cpu.shape[-1]}, expected {QWEN_DENSE_VECTOR_SIZE}"
                    )
                observe_advanced_phase("qwen_dense", "postprocess", phase_started_at)
                phase_started_at = time.monotonic()
                dense_parts.append(np.asarray(dense_cpu, dtype=np.float32))
                observe_advanced_phase("qwen_dense", "device_to_host", phase_started_at)
                observe_advanced_phase(
                    "qwen_dense", "output_conversion", phase_started_at
                )
                continue
            phase_started_at = time.monotonic()
            padded = {
                name: _move_advanced_tensor_to_device(
                    value,
                    self.device,
                    pinned_memory=ADVANCED_DENSE_PINNED_MEMORY,
                )
                for name, value in padded.items()
            }
            observe_advanced_phase("qwen_dense", "host_to_device", phase_started_at)

            forward_timer = start_advanced_device_phase()
            with torch.inference_mode():
                outputs = self.model(**padded)
            stop_advanced_device_phase(forward_timer)

            postprocess_timer = start_advanced_device_phase()
            # With left padding, -1 is the real final token for every row.
            dense_vecs = outputs.last_hidden_state[:, -1, :]
            if dense_vecs.shape[-1] != QWEN_DENSE_VECTOR_SIZE:
                raise RuntimeError(
                    f"{self.model_name} returned dense dimension "
                    f"{dense_vecs.shape[-1]}, expected {QWEN_DENSE_VECTOR_SIZE}"
                )
            if normalize_dense:
                dense_vecs = torch.nn.functional.normalize(dense_vecs, p=2, dim=1)
            stop_advanced_device_phase(postprocess_timer)

            transfer_timer = start_advanced_device_phase()
            # NumPy has no native bfloat16 dtype. Serving output is numeric JSON,
            # so convert both FP16/BF16 candidates to stable FP32 on the host.
            dense_host = dense_vecs.detach()
            # Real torch tensors expose ``float``; the lightweight tensor
            # protocol used by unit-test doubles intentionally does not.
            # Keeping the guard also makes this boundary tolerant of future
            # tensor-like accelerator adapters while BF16 torch output is
            # still converted before NumPy (which has no BF16 dtype).
            if hasattr(dense_host, "float"):
                dense_host = dense_host.float()
            dense_cpu = dense_host.cpu()
            stop_advanced_device_phase(transfer_timer)
            observe_advanced_device_phase("qwen_dense", "forward", forward_timer)
            observe_advanced_device_phase(
                "qwen_dense", "postprocess", postprocess_timer
            )
            observe_advanced_device_phase(
                "qwen_dense", "device_to_host", transfer_timer
            )
            phase_started_at = time.monotonic()
            dense_parts.append(dense_cpu.numpy())
            observe_advanced_phase("qwen_dense", "output_conversion", phase_started_at)

        if len(dense_parts) == 1:
            record_advanced_memory_measurement("qwen_dense")
            return {"dense_vecs": dense_parts[0]}
        record_advanced_memory_measurement("qwen_dense")
        return {"dense_vecs": np.concatenate(dense_parts, axis=0)}


def _milco_rows_to_lexical_weights(
    rows: List[int],
    cols: List[int],
    vals: List[float],
    n: int,
    top_k: int,
) -> List[Dict[str, float]]:
    """Group MILCO sparse COO entries into per-row {str(vocab_id): weight} dicts.

    Drops non-positive weights (learned-sparse can emit ~0). Optional top_k keeps
    only the highest-weight terms per row. Output maps 1:1 onto Qdrant sparse
    vectors via lexical_weights_to_indices().
    """
    # Single pass over the COO entries, bucketing each into its row. The earlier
    # per-row scan was O(n * nnz); this is O(nnz) and order-independent.
    grouped: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for j in range(len(rows)):
        weight = float(vals[j])
        if weight > 0:
            grouped[rows[j]].append((cols[j], weight))

    out: List[Dict[str, float]] = []
    for pairs in grouped:
        if top_k and len(pairs) > top_k:
            pairs = sorted(pairs, key=lambda p: p[1], reverse=True)[:top_k]
        out.append({str(col): val for col, val in pairs})
    return out


class MilcoSparseBackend:
    """Produce Qdrant-compatible learned-sparse vectors with MILCO."""

    def __init__(self, model_name: str, device: str = device):
        """Load the LexEcho-aware MILCO sparse runtime.

        Args:
            model_name: MILCO model identifier or local snapshot.
            device: Torch device used for sparse inference.

        Raises:
            RuntimeError: If a non-Torch runtime was selected.
        """
        self.model_name = model_name
        self.device = device
        if ADVANCED_SPARSE_RUNTIME != "torch":
            raise RuntimeError(
                "MILCO runtime "
                f"{ADVANCED_SPARSE_RUNTIME!r} is unavailable; the LexEcho-aware "
                "Torch remote-code path is required"
            )
        dtype = _torch_dtype_for_precision(ADVANCED_SPARSE_PRECISION)
        logging.info(f"Initializing MILCO sparse model '{model_name}' on '{device}'")
        model_path = _cache_first_model_path(self.model_name)
        milco_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        multilingual_checkpoint = str(
            getattr(milco_config, "multilingual_encoder_checkpoint", "")
        )
        if multilingual_checkpoint:
            cached_multilingual = _cache_first_model_path(multilingual_checkpoint)
            if cached_multilingual != multilingual_checkpoint:
                milco_config.multilingual_encoder_checkpoint = cached_multilingual
        # MILCO's English sub-model configuration and weights are embedded in
        # its own snapshot. The English tokenizer is lazy and is not used by
        # this backend, so resolving its gated SPLADE repository here would
        # impose an unnecessary Hugging Face authentication requirement.
        self.model = AutoModel.from_pretrained(
            model_path,
            config=milco_config,
            trust_remote_code=True,
            dtype=dtype,
        )
        self.model.eval().to(device)
        if ADVANCED_SPARSE_TORCH_COMPILE:
            self.model.m_model = _compile_advanced_model(
                self.model.m_model,
                enabled=True,
                mode=ADVANCED_SPARSE_COMPILE_MODE,
                backend="milco_sparse_encoder",
            )
        tokenizer = getattr(self.model, "m_tokenizer", None)
        if tokenizer is not None:
            self.tokenizer = tokenizer
            tokenizer_limit = int(getattr(self.tokenizer, "model_max_length", 8192))
            self.max_length = max(1, min(MILCO_MAX_LENGTH, tokenizer_limit))
        else:
            self.max_length = MILCO_MAX_LENGTH
        logging.info(
            "MILCO sparse model ready: source_view=%s max_length=%s "
            "max_batch_tokens=%s projection_chunk_tokens=%s",
            MILCO_SOURCE_VIEW,
            self.max_length,
            MILCO_MAX_BATCH_TOKENS,
            MILCO_PROJECTION_CHUNK_TOKENS,
        )

    def embed_sparse(
        self, sentences: List[str], deadline=None
    ) -> Dict[str, List[Dict[str, float]]]:
        """Encode Qdrant-compatible sparse weights under memory bounds.

        Args:
            sentences: Texts in caller order.
            deadline: Optional cooperative deadline checked between microbatches.

        Returns:
            Per-input lexical-weight mappings under ``lexical_weights``.
        """
        if not sentences:
            return {"lexical_weights": []}

        advanced_backend_requests.labels(backend="milco_sparse").inc()
        begin_advanced_memory_measurement()
        # Lightweight unit-test doubles and future incompatible remote-code
        # revisions retain the public MILCO path. Production snapshots expose
        # the tokenizer and internal heads used by the bounded equivalent below.
        if not hasattr(self, "tokenizer"):
            all_weights: List[Dict[str, float]] = []
            for start in range(0, len(sentences), MILCO_BATCH_SIZE):
                if deadline is not None:
                    deadline.check()
                chunk = sentences[start : start + MILCO_BATCH_SIZE]
                phase_started_at = time.monotonic()
                with torch.inference_mode():
                    reps = self.model.encode_text(
                        chunk,
                        source_view=MILCO_SOURCE_VIEW,
                    )
                observe_advanced_phase("milco_sparse", "forward", phase_started_at)
                phase_started_at = time.monotonic()
                reps = reps.coalesce()
                idx = reps.indices()
                rows = idx[0].cpu().tolist()
                cols = idx[1].cpu().tolist()
                vals = reps.values().float().cpu().tolist()
                observe_advanced_phase(
                    "milco_sparse", "device_to_host", phase_started_at
                )
                phase_started_at = time.monotonic()
                weights = _milco_rows_to_lexical_weights(
                    rows,
                    cols,
                    vals,
                    reps.shape[0],
                    MILCO_TOP_K,
                )
                observe_advanced_phase(
                    "milco_sparse", "coo_to_qdrant", phase_started_at
                )
                all_weights.extend(weights)
            record_advanced_memory_measurement("milco_sparse")
            return {"lexical_weights": all_weights}

        max_item_tokens = max(
            1,
            min(
                self.max_length,
                max_batch_tokens_for_length(
                    MILCO_MAX_BATCH_TOKENS_POLICY, self.max_length
                )
                if MILCO_TOKENS_POLICY_ACTIVE
                else MILCO_MAX_BATCH_TOKENS,
            ),
        )
        phase_started_at = time.monotonic()
        encoded = self.tokenizer(
            sentences,
            padding=False,
            truncation=True,
            max_length=max_item_tokens,
            return_tensors=None,
        )
        observe_advanced_phase("milco_sparse", "tokenize", phase_started_at)

        chunks = iter_token_budget_batches(
            encoded,
            MILCO_BATCH_SIZE,
            MILCO_MAX_BATCH_TOKENS,
            batch_policy=MILCO_BATCH_POLICY if MILCO_POLICY_ACTIVE else None,
            max_batch_tokens_policy=(
                MILCO_MAX_BATCH_TOKENS_POLICY if MILCO_TOKENS_POLICY_ACTIVE else None
            ),
        )

        def prepare_sparse_batch(chunk):
            phase_started_at = time.monotonic()
            padded = self.tokenizer.pad(
                chunk,
                padding=True,
                pad_to_multiple_of=32,
                return_tensors="pt",
            )
            observe_advanced_phase("milco_sparse", "tokenize", phase_started_at)
            return padded

        all_weights: List[Dict[str, float]] = []
        for padded in _iter_double_buffered(
            chunks,
            prepare_sparse_batch,
            enabled=ADVANCED_SPARSE_PREFETCH_BATCHES,
        ):
            if deadline is not None:
                deadline.check()
            record_embedding_token_batch(padded, backend="milco")
            record_advanced_token_batch(padded, backend="milco_sparse")
            phase_started_at = time.monotonic()
            padded = {
                name: _move_advanced_tensor_to_device(
                    value,
                    self.device,
                    pinned_memory=ADVANCED_SPARSE_PINNED_MEMORY,
                )
                for name, value in padded.items()
            }
            observe_advanced_phase("milco_sparse", "host_to_device", phase_started_at)
            device_type = self.device.split(":", 1)[0]
            compute_dtype = _torch_dtype_for_precision(
                ADVANCED_SPARSE_COMPUTE_PRECISION
            )
            forward_timer = start_advanced_device_phase()
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device_type,
                    dtype=compute_dtype,
                    enabled=ADVANCED_SPARSE_COMPUTE_PRECISION != "fp32",
                ),
            ):
                reps = memory_bounded_milco_forward(
                    self.model,
                    padded["input_ids"],
                    padded["attention_mask"],
                    projection_chunk_tokens=MILCO_PROJECTION_CHUNK_TOKENS,
                    source_view=MILCO_SOURCE_VIEW,
                )
            stop_advanced_device_phase(forward_timer)
            postprocess_timer = start_advanced_device_phase()
            reps = reps.coalesce()
            idx = reps.indices()
            stop_advanced_device_phase(postprocess_timer)
            transfer_timer = start_advanced_device_phase()
            rows_cpu = idx[0].cpu()
            cols_cpu = idx[1].cpu()
            vals_cpu = reps.values().float().cpu()
            stop_advanced_device_phase(transfer_timer)
            observe_advanced_device_phase("milco_sparse", "forward", forward_timer)
            observe_advanced_device_phase(
                "milco_sparse", "postprocess", postprocess_timer
            )
            observe_advanced_device_phase(
                "milco_sparse", "device_to_host", transfer_timer
            )
            phase_started_at = time.monotonic()
            rows = rows_cpu.tolist()
            cols = cols_cpu.tolist()
            vals = vals_cpu.tolist()
            weights = _milco_rows_to_lexical_weights(
                rows, cols, vals, reps.shape[0], MILCO_TOP_K
            )
            observe_advanced_phase("milco_sparse", "coo_to_qdrant", phase_started_at)
            all_weights.extend(weights)
        record_advanced_memory_measurement("milco_sparse")
        return {"lexical_weights": all_weights}


def length_aware_sentence_order(sentences: List[str]) -> List[int]:
    """Return a stable, zero-tokenization order that reduces padding variance."""
    return sorted(range(len(sentences)), key=lambda index: len(sentences[index]))


def restore_embedding_order(
    result: Dict[str, Union[np.ndarray, List]], order: List[int]
) -> Dict[str, Union[np.ndarray, List]]:
    """Restore backend outputs from sorted order to the original API order."""
    if order == list(range(len(order))):
        return result

    sorted_positions = [0] * len(order)
    for sorted_position, original_position in enumerate(order):
        sorted_positions[original_position] = sorted_position

    for key in ("dense_vecs", "lexical_weights", "colbert_vecs"):
        values = result.get(key)
        if values is None:
            continue
        if hasattr(values, "take") and not isinstance(values, (list, tuple)):
            result[key] = values.take(sorted_positions, axis=0)
        else:
            result[key] = [values[position] for position in sorted_positions]
    return result


class EmbeddingService:
    """Route embedding requests across the backends selected by the profile."""

    @classmethod
    def from_profile(
        cls,
        profile_models: Dict[str, Optional[str]],
        devices: Optional[List[str]] = None,
    ) -> "EmbeddingService":
        """Construct only the backends selected by a startup profile.

        Args:
            profile_models: Profile name and dense, sparse, and ColBERT models.
            devices: Optional multi-device placement for the BASIC BGE backend.

        Returns:
            A fully initialized profile-specific embedding service.

        Raises:
            RuntimeError: If a selected runtime is unavailable or unsafe.
        """
        self = cls.__new__(cls)
        self.profile = profile_models["profile"]
        self.dense_model_name = profile_models["dense"]
        self.sparse_model_name = profile_models["sparse"]
        self.colbert_model_name = profile_models["colbert"]

        advanced = self.profile == EMBEDDER_PROFILE_ADVANCED
        if advanced:
            self.bge_backend = None
        elif BGE_EFFECTIVE_RUNTIME == "onnx-fp32":
            if device != "cpu":
                raise RuntimeError(
                    "BGE_EMBED_RUNTIME=onnx-fp32 currently supports CPU only"
                )
            snapshot_path = _cache_first_model_path(BGE_EMBEDDING_MODEL)
            if snapshot_path == BGE_EMBEDDING_MODEL:
                raise RuntimeError(
                    "BGE_EMBED_RUNTIME=onnx-fp32 requires a cached BGE-M3 snapshot"
                )
            onnx_tokenizer = InstrumentedEmbeddingTokenizer(
                _load_tokenizer_cache_first(BGE_EMBEDDING_MODEL),
                backend="bge",
            )
            self.bge_backend = BgeM3OnnxBackend(
                snapshot_path=snapshot_path,
                tokenizer=onnx_tokenizer,
                torch_module=torch,
                threads=_cpu_threads,
                concurrency=BGE_EMBED_INFERENCE_CONCURRENCY,
                execution_mode=BGE_ONNX_EXECUTION_MODE,
                spin_policy=BGE_ONNX_SPIN_POLICY,
                inter_op_threads=BGE_ONNX_INTER_OP_THREADS,
                max_length=BGE_EMBED_MAX_LENGTH,
                batch_size_for_tokens=bge_batch_size_for_token_length,
            )
        elif BGE_EFFECTIVE_RUNTIME == "onnx-int8":
            raise RuntimeError(
                "BGE_EMBED_RUNTIME=onnx-int8 is disabled: quality gates failed"
            )
        else:
            self.bge_backend = BgeM3EmbeddingBackend(
                BGE_EMBEDDING_MODEL, devices=devices
            )
        self.bge_model_name = None if advanced else BGE_EMBEDDING_MODEL
        self.qwen_dense_backend = (
            QwenDenseEmbeddingBackend(self.dense_model_name)
            if self.dense_model_name == QWEN_DENSE_EMBEDDING_MODEL
            else None
        )
        self.milco_sparse_backend = (
            MilcoSparseBackend(self.sparse_model_name)
            if self.sparse_model_name == MILCO_SPARSE_MODEL
            else None
        )
        # Dedicated tokenizer for the truncation-warning path. The encode path
        # borrows the backend's own fast tokenizer on the GPU worker thread; the
        # warning path runs on the event-loop thread. Sharing one Rust fast
        # tokenizer across both threads raises "Already borrowed" under load, so
        # the warning path gets an independent instance.
        self.bge_warning_tokenizer = (
            _load_tokenizer_cache_first(self.bge_model_name)
            if self.bge_backend is not None
            else None
        )
        self.qwen_warning_tokenizer = (
            _load_tokenizer_cache_first(self.dense_model_name)
            if self.qwen_dense_backend is not None
            else None
        )
        self._hybrid_executor = None
        self._hybrid_streams = (None, None)
        if advanced and ADVANCED_SCHEDULE_MODE in {"overlap", "parallel"}:
            self._hybrid_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="advanced-hybrid",
            )
            if has_cuda and ADVANCED_SCHEDULE_MODE == "overlap":
                self._hybrid_streams = (
                    torch.cuda.Stream(device=device),
                    torch.cuda.Stream(device=device),
                )
        return self

    def close(self) -> None:
        """Close backend and hybrid executors owned by this service."""
        if self.bge_backend is not None:
            self.bge_backend.close()
        hybrid_executor = getattr(self, "_hybrid_executor", None)
        if hybrid_executor is not None:
            hybrid_executor.shutdown(wait=True, cancel_futures=True)
            self._hybrid_executor = None

    @staticmethod
    def _run_hybrid_backend(call: Callable[[], Dict], stream: Any = None) -> Dict:
        """Run one independent hybrid branch, optionally on a CUDA stream."""
        if stream is None:
            return call()
        stream.wait_stream(torch.cuda.default_stream(device=device))
        with torch.cuda.stream(stream):
            output = call()
        stream.synchronize()
        return output

    def embed(
        self,
        sentences: List[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
        normalize_dense: bool = False,
        deadline=None,
    ) -> Dict[str, Union[np.ndarray, List]]:
        """Route requested vector types and restore original input order.

        Args:
            sentences: Texts in API order.
            return_dense: Include the profile's dense vectors.
            return_sparse: Include the profile's sparse vectors.
            return_colbert: Include ColBERT vectors when supported.
            normalize_dense: L2-normalize dense vectors.
            deadline: Optional cooperative deadline shared by all backends.

        Returns:
            Requested backend outputs aligned with ``sentences``.
        """
        result: Dict[str, Union[np.ndarray, List]] = {}
        order = list(range(len(sentences)))
        backend_sentences = sentences
        if EMBED_LENGTH_AWARE_BATCHING and len(sentences) > 1:
            order = length_aware_sentence_order(sentences)
            if order != list(range(len(sentences))):
                backend_sentences = [sentences[index] for index in order]

        use_qwen_dense = self.qwen_dense_backend is not None
        use_milco = self.milco_sparse_backend is not None
        have_bge = self.bge_backend is not None
        is_advanced_hybrid = (
            return_dense and return_sparse and use_qwen_dense and use_milco
        )
        hybrid_started_at = time.monotonic() if is_advanced_hybrid else 0.0
        if is_advanced_hybrid:
            advanced_backend_requests.labels(backend="advanced_hybrid").inc()

        bge_return_dense = return_dense and not use_qwen_dense and have_bge
        bge_return_sparse = return_sparse and not use_milco and have_bge
        bge_return_colbert = return_colbert and have_bge
        needs_bge = bge_return_dense or bge_return_sparse or bge_return_colbert

        if needs_bge:
            result.update(
                self.bge_backend.embed(
                    backend_sentences,
                    return_dense=bge_return_dense,
                    return_sparse=bge_return_sparse,
                    return_colbert=bge_return_colbert,
                    normalize_dense=normalize_dense if bge_return_dense else False,
                    deadline=deadline,
                )
            )

        hybrid_executor = getattr(self, "_hybrid_executor", None)
        if is_advanced_hybrid and hybrid_executor is not None:
            dense_stream, sparse_stream = getattr(self, "_hybrid_streams", (None, None))
            dense_future = hybrid_executor.submit(
                self._run_hybrid_backend,
                lambda: self.qwen_dense_backend.embed_dense(
                    backend_sentences,
                    normalize_dense=normalize_dense,
                    deadline=deadline,
                ),
                dense_stream,
            )
            sparse_future = hybrid_executor.submit(
                self._run_hybrid_backend,
                lambda: self.milco_sparse_backend.embed_sparse(
                    backend_sentences,
                    deadline=deadline,
                ),
                sparse_stream,
            )
            try:
                result.update(dense_future.result())
                result.update(sparse_future.result())
            except BaseException:
                dense_future.cancel()
                sparse_future.cancel()
                raise
        else:
            if return_dense and use_qwen_dense:
                result.update(
                    self.qwen_dense_backend.embed_dense(
                        backend_sentences,
                        normalize_dense=normalize_dense,
                        deadline=deadline,
                    )
                )

            if return_sparse and use_milco:
                result.update(
                    self.milco_sparse_backend.embed_sparse(
                        backend_sentences, deadline=deadline
                    )
                )

        phase_started_at = time.monotonic()
        restored = restore_embedding_order(result, order)
        if is_advanced_hybrid:
            observe_advanced_phase("advanced_hybrid", "merge_restore", phase_started_at)
            observe_advanced_phase("advanced_hybrid", "pipeline", hybrid_started_at)
        return restored


# --- Prometheus Metrics ---

inference_reserved_tokens = Gauge(
    "inference_reserved_tokens", "Token-work units reserved by accepted inference"
)
inference_reserved_output_bytes = Gauge(
    "inference_reserved_output_bytes", "Estimated output bytes reserved by inference"
)


def _record_inference_reservation(cost):
    """Publish both resource gauges when the shared reservation changes."""
    inference_reserved_tokens.set(cost.tokens)
    inference_reserved_output_bytes.set(cost.output_bytes)


inference_budget = RequestBudget(
    MAX_INFLIGHT_TOKENS, MAX_INFLIGHT_OUTPUT_BYTES, _record_inference_reservation
)

# Counters
requests_total = Counter(
    "embedding_requests_total",
    "Total number of embedding requests",
    ["status", "endpoint"],
)

requests_rejected = Counter(
    "embedding_requests_rejected_total",
    "Total rejected requests",
    ["reason"],
)

sentences_processed = Counter(
    "embedding_sentences_processed_total", "Total sentences processed"
)

embedding_input_tokens = Counter(
    "embedding_input_tokens_total",
    "Post-truncation non-padding tokens submitted to embedding backends",
    ["backend"],
)

embedding_padded_tokens = Counter(
    "embedding_padded_tokens_total",
    "Padded tensor token positions submitted to embedding backends",
    ["backend"],
)

embedding_tokenized_batches = Counter(
    "embedding_tokenized_batches_total",
    "Tokenizer batches submitted to embedding backends",
    ["backend"],
)

advanced_backend_requests = Counter(
    "advanced_backend_requests_total",
    "ADVANCED backend invocations, including internal hybrid pipeline work",
    ["backend"],
)
advanced_backend_input_tokens = Counter(
    "advanced_backend_input_tokens_total",
    "Post-truncation non-padding tokens submitted to ADVANCED backends",
    ["backend"],
)
advanced_backend_padded_tokens = Counter(
    "advanced_backend_padded_tokens_total",
    "Padded tensor positions submitted to ADVANCED backends",
    ["backend"],
)
advanced_backend_tokenized_batches = Counter(
    "advanced_backend_tokenized_batches_total",
    "Tokenizer tensor batches submitted to ADVANCED backends",
    ["backend"],
)

# Histograms
request_duration = Histogram(
    "embedding_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

batch_size_histogram = Histogram(
    "embedding_batch_size",
    "Size of batches processed",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
)

gpu_inference_duration = Histogram(
    "embedding_gpu_inference_duration_seconds",
    "GPU inference time",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

advanced_backend_phase_duration = Histogram(
    "advanced_backend_phase_duration_seconds",
    "ADVANCED backend time split by tokenize, transfer, forward and post-processing",
    ["backend", "phase"],
    buckets=[
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ],
)

# Gauges
queue_size = Gauge(
    "embedding_queue_size",
    (
        "Current number of embedding requests waiting in the batching queue "
        "(updated on enqueue and dequeue)"
    ),
)

active_requests = Gauge(
    "embedding_active_requests",
    (
        "Number of embedding requests currently queued or being processed "
        "(incremented on accept, decremented on completion)"
    ),
)

gpu_memory_allocated = Gauge(
    "embedding_gpu_memory_allocated_bytes",
    "Legacy embedding label for process GPU tensor memory allocated in bytes",
)
gpu_memory_reserved = Gauge(
    "embedding_gpu_memory_reserved_bytes",
    "Legacy embedding label for process GPU memory reserved by the PyTorch "
    "caching allocator",
)
process_gpu_memory_allocated = Gauge(
    "gpu_memory_allocated_bytes",
    "Process GPU tensor memory allocated in bytes",
)
process_gpu_memory_reserved = Gauge(
    "gpu_memory_reserved_bytes",
    "Process GPU memory reserved by the PyTorch caching allocator in bytes",
)
process_gpu_memory_free = Gauge(
    "gpu_memory_free_bytes",
    "CUDA device free memory in bytes",
)
process_gpu_memory_total = Gauge(
    "gpu_memory_total_bytes",
    "CUDA device total memory in bytes",
)
advanced_backend_gpu_memory = Gauge(
    "advanced_backend_gpu_memory_bytes",
    "Current and peak PyTorch CUDA allocator memory for the last backend invocation",
    ["backend", "kind"],
)
advanced_backend_rss_peak = Gauge(
    "advanced_backend_process_rss_peak_bytes",
    "Process peak resident memory observed after the last backend invocation",
    ["backend"],
)

# Info
server_info = Info("embedding_server", "Server information")
autotune_info = Info(
    "embedding_autotune",
    "Hardware autotune profile and effective serving configuration",
)
advanced_config_info = Info(
    "advanced_effective_config",
    "Effective immutable and tunable settings for the ADVANCED profile",
)

# --- Reranker Prometheus Metrics ---

rerank_requests_total = Counter(
    "rerank_requests_total",
    "Total number of rerank requests",
    ["status"],
)

rerank_pairs_processed = Counter(
    "rerank_pairs_processed_total", "Total query-passage pairs reranked"
)

rerank_request_duration = Histogram(
    "rerank_request_duration_seconds",
    "Rerank request duration in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

rerank_inference_duration = Histogram(
    "rerank_inference_duration_seconds",
    "Reranker model inference time",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

rerank_active_requests = Gauge(
    "rerank_active_requests",
    "Number of rerank requests currently holding a slot (queued plus in-flight)",
)

rerank_requests_rejected = Counter(
    "rerank_requests_rejected_total",
    "Total rejected rerank requests",
    ["reason"],
)

rerank_batch_halvings = Counter(
    "rerank_batch_halvings_total",
    "Rerank sub-batches split after a VRAM exhaustion failure",
)

rerank_capacity_rejections = Counter(
    "rerank_capacity_rejections_total",
    "Rerank requests rejected because the batch could not be allocated",
    ["reason"],
)

# Phase-level timing histograms. These split the legacy gpu_inference_duration
# into queue wait, executor wait, model forward, response build, and HTTP
# serialization phases so dashboards can identify bottlenecks precisely.
api_batch_queue_wait_duration = Histogram(
    "api_batch_queue_wait_duration_seconds",
    "Time requests spend waiting in the async batch queue before a batch starts",
    ["endpoint"],
)
api_executor_wait_duration = Histogram(
    "api_executor_wait_duration_seconds",
    "Time between executor submit and worker start",
    ["endpoint", "operation"],
)
api_model_forward_duration = Histogram(
    "api_model_forward_duration_seconds",
    "Time spent inside model worker execution",
    ["endpoint", "operation"],
)
api_response_build_duration = Histogram(
    "api_response_build_duration_seconds",
    "Time spent building Pydantic response payloads",
    ["endpoint"],
)
api_response_serialization_duration = Histogram(
    "api_response_serialization_duration_seconds",
    "Time spent rendering HTTP JSON responses",
    ["endpoint"],
)


# GPU memory is sampled on every batch and rerank; mem_get_info() forces a
# CUDA sync, so throttle the sampling to at most once per interval.
_GPU_METRIC_MIN_INTERVAL = 1.0
_last_gpu_metric_sample = 0.0


class TimedORJSONResponse(JSONResponse):
    """Serialize with orjson while observing endpoint render latency."""

    endpoint_label = "unknown"

    def render(self, content):
        """Serialize response content and record render duration."""
        started_at = time.monotonic()
        try:
            return orjson.dumps(
                content,
                option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
            )
        finally:
            api_response_serialization_duration.labels(
                endpoint=self.endpoint_label
            ).observe(time.monotonic() - started_at)


class EmbeddingORJSONResponse(TimedORJSONResponse):
    """Timed orjson response labeled for the embedding endpoint."""

    endpoint_label = "/embeddings"


class RerankORJSONResponse(TimedORJSONResponse):
    """Timed orjson response labeled for the rerank endpoint."""

    endpoint_label = "/rerank"


def record_gpu_memory_metrics() -> None:
    """Record process-level CUDA memory metrics when CUDA is available.

    Sampling is throttled to at most once per ``_GPU_METRIC_MIN_INTERVAL``
    seconds so high request rates do not pay a CUDA sync per request.
    """
    if not has_cuda:
        return

    global _last_gpu_metric_sample
    now = time.monotonic()
    if now - _last_gpu_metric_sample < _GPU_METRIC_MIN_INTERVAL:
        return
    _last_gpu_metric_sample = now

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()

    # Keep legacy embedding-prefixed gauges for existing dashboards.
    gpu_memory_allocated.set(allocated)
    gpu_memory_reserved.set(reserved)

    process_gpu_memory_allocated.set(allocated)
    process_gpu_memory_reserved.set(reserved)
    process_gpu_memory_free.set(free)
    process_gpu_memory_total.set(total)


# --- 3. API Data Models ---


class EmbedRequestBase(BaseModel):
    """Common embedding request fields for every profile.

    Profile-specific request models add only the options that are meaningful for
    the active backend selection.
    """

    model_config = {"extra": "forbid"}

    sentences: List[str]
    return_dense: bool = True
    return_sparse: bool = True
    normalize_dense: bool = False
    sparse_as_indices: bool = False
    include_text: bool = True


class EmbedRequest(EmbedRequestBase):
    """Embedding request for the basic BGE-M3 profile."""

    return_colbert: bool = True


class AdvancedEmbedRequest(EmbedRequestBase):
    """Embedding request for the advanced profile.

    The advanced profile has no ColBERT backend, so the ColBERT toggle is not
    part of the public request schema.
    """

    @model_validator(mode="before")
    @classmethod
    def reject_colbert_request(cls, data):
        """Reject enabled ColBERT requests and discard a false legacy toggle."""
        if isinstance(data, dict) and "return_colbert" in data:
            if data["return_colbert"] is True:
                raise ValueError(
                    "return_colbert is not supported by the advanced profile"
                )
            data = dict(data)
            data.pop("return_colbert")
        return data


ActiveEmbedRequest = (
    AdvancedEmbedRequest
    if ACTIVE_PROFILE["profile"] == EMBEDDER_PROFILE_ADVANCED
    else EmbedRequest
)


def _request_return_colbert(request: EmbedRequestBase) -> bool:
    """Return whether the request can and does ask for ColBERT vectors."""
    return bool(getattr(request, "return_colbert", False))


class SparseIndicesVector(BaseModel):
    """Qdrant-compatible parallel sparse index and value arrays."""

    indices: List[int]
    values: List[float]


class EmbeddingVectors(BaseModel):
    """Group the three types of embedding vectors with optional fields.

    Contains the different embedding representations that can be generated for a text:
    dense vectors, sparse lexical weights, and ColBERT token-level embeddings.
    All fields are optional to support selective embedding generation.
    """

    dense: Optional[List[float]] = None
    sparse: Optional[Union[Dict[str, float], SparseIndicesVector]] = None
    colbert: Optional[List[List[float]]] = None


class SingleEmbeddingResponse(BaseModel):
    """Represent the complete result for a single text.

    Contains the original text, its position in the input list,
    and the generated embeddings in various formats.
    """

    id: int
    text: Optional[str] = None
    embeddings: EmbeddingVectors


class WarningTarget(BaseModel):
    """Locate the request field responsible for a non-fatal warning."""

    field: str
    index: Optional[int] = None
    pointer: str


class WarningDetails(BaseModel):
    """Describe model-specific truncation applied to one request value."""

    model: str
    max_tokens: int
    original_tokens: int
    truncated_tokens: int
    truncation_side: str = "end"


class ApiWarning(BaseModel):
    """Structured non-fatal API warning with location and details."""

    code: str
    severity: str = "warning"
    message: str
    target: WarningTarget
    details: WarningDetails


class EmbeddingsListResponse(BaseModel):
    """Main response object, with data and metadata.

    Contains a list of embedding results for all input texts,
    along with metadata about the model used and processing time.
    """

    model_config = {"protected_namespaces": ()}

    data: List[SingleEmbeddingResponse]
    model_name: str
    dense_model_name: str
    sparse_model_name: str
    colbert_model_name: Optional[str] = None
    embedding_contract_fingerprint: Optional[str] = None
    processing_time_ms: float
    warnings: List[ApiWarning] = Field(default_factory=list)


# --- Reranker Pydantic Models ---


class RerankRequest(BaseModel):
    """Model for a reranking request.

    Accepts a query and a list of passages (documents) to score.
    Returns passages sorted by relevance score descending.
    """

    query: str
    passages: List[str]
    normalize: bool = False


class RankedPassage(BaseModel):
    """A single passage with its relevance score and original index."""

    index: int
    passage: str
    score: float


class RerankResponse(BaseModel):
    """Response for a reranking request.

    Contains passages sorted by relevance score (descending),
    along with metadata about the model and processing time.
    """

    model_config = {"protected_namespaces": ()}

    results: List[RankedPassage]
    model_name: str
    processing_time_ms: float
    warnings: List[ApiWarning] = Field(default_factory=list)


class EmbedderModelsInfo(BaseModel):
    """Embedding model names by vector type."""

    dense: str
    sparse: str
    colbert: Optional[str] = None


class ModelsResponse(BaseModel):
    """Active model configuration for this server instance."""

    profile: str
    embedder: EmbedderModelsInfo
    reranker: str
    version: str
    embedding_contract: Dict[str, Any]


def lexical_weights_to_indices(lw: Dict[str, float]) -> SparseIndicesVector:
    """Convert BGE-M3 lexical weights into a Qdrant sparse vector.

    Materializing the items once guarantees that indices and values retain the
    same ordering.

    Args:
        lw: Token-ID strings mapped to their lexical weights.

    Returns:
        Matching integer indices and floating-point values for Qdrant.
    """
    items = list(lw.items())
    return SparseIndicesVector(
        indices=[int(k) for k, _ in items],
        values=[float(v) for _, v in items],
    )


def _encode_token_ids(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False, verbose=False)
    except TypeError:
        return tokenizer.encode(text, add_special_tokens=False)


def _count_tokens(tokenizer, text: str) -> int:
    return len(_encode_token_ids(tokenizer, text))


def _embedding_request_cost(request):
    """Estimate selected backends using their effective truncation limits."""
    advanced = embedding_service.profile == EMBEDDER_PROFILE_ADVANCED
    milco = embedding_service.milco_sparse_backend
    config = getattr(getattr(milco, "model", None), "config", None)
    return estimate_embeddings(
        request.sentences,
        advanced=advanced,
        dense=request.return_dense,
        sparse=request.return_sparse,
        colbert=_request_return_colbert(request),
        include_text=request.include_text,
        dense_limit=QWEN_EMBED_MAX_LENGTH if advanced else BGE_EMBED_MAX_LENGTH,
        sparse_limit=(
            getattr(milco, "max_length", MILCO_MAX_LENGTH)
            if advanced
            else BGE_EMBED_MAX_LENGTH
        ),
        english_vocab=int(getattr(config, "en_vocab_size", 30522) or 30522),
        source_view=MILCO_SOURCE_VIEW,
        sparse_top_k=MILCO_TOP_K,
    )


def _reserve_inference(cost, *, reranking=False):
    """Translate a shared admission refusal into its endpoint's metrics and HTTP."""
    try:
        return inference_budget.reserve(cost)
    except BudgetExceeded as failure:
        reason = (
            "request_budget_exceeded" if failure.oversized else "request_budget_busy"
        )
        metric = rerank_requests_rejected if reranking else requests_rejected
        metric.labels(reason=reason).inc()
        stats.increment_rejected()
        raise HTTPException(
            status_code=413 if failure.oversized else 503,
            detail={
                "code": reason,
                "message": "Inference request exceeds the admission budget"
                if failure.oversized
                else "Inference capacity is occupied; retry after active work finishes",
                "estimated_tokens": cost.tokens,
                "estimated_output_bytes": cost.output_bytes,
                "max_tokens": failure.limits.tokens,
                "max_output_bytes": failure.limits.output_bytes,
            },
            headers=None if failure.oversized else {"Retry-After": "1"},
        ) from None


def _active_embedding_tokenizers(
    request: EmbedRequestBase,
) -> List[Tuple[Any, str, int]]:
    tokenizers: List[Tuple[Any, str, int]] = []
    qwen_dense = embedding_service.qwen_dense_backend
    uses_qwen_dense = request.return_dense and qwen_dense is not None

    if uses_qwen_dense:
        tokenizers.append(
            (
                embedding_service.qwen_warning_tokenizer,
                embedding_service.dense_model_name,
                QWEN_EMBED_MAX_LENGTH,
            )
        )

    bge = embedding_service.bge_backend
    if bge is not None:
        needs_bge = (
            (request.return_dense and not uses_qwen_dense)
            or request.return_sparse
            or _request_return_colbert(request)
        )
        if needs_bge:
            tokenizers.append(
                (
                    embedding_service.bge_warning_tokenizer,
                    embedding_service.bge_model_name,
                    BGE_EMBED_MAX_LENGTH,
                )
            )

    if not tokenizers and bge is not None:
        tokenizers.append(
            (
                embedding_service.bge_warning_tokenizer,
                embedding_service.bge_model_name,
                BGE_EMBED_MAX_LENGTH,
            )
        )
    return tokenizers


def _make_truncation_warning(
    code: str,
    field: str,
    index: Optional[int],
    pointer: str,
    model_name: str,
    max_tokens: int,
    original_tokens: int,
    message: str,
) -> ApiWarning:
    return ApiWarning(
        code=code,
        message=message,
        target=WarningTarget(field=field, index=index, pointer=pointer),
        details=WarningDetails(
            model=model_name,
            max_tokens=max_tokens,
            original_tokens=original_tokens,
            truncated_tokens=max(0, original_tokens - max_tokens),
        ),
    )


def collect_embedding_truncation_warnings(request: EmbedRequest) -> List[ApiWarning]:
    """Collect request-local warnings for active embedding tokenizers.

    Inputs below the cheap byte-length threshold avoid tokenizer work. Longer
    inputs are counted with tokenizer instances independent from inference.
    """
    active_tokenizers = _active_embedding_tokenizers(request)
    warnings_out: List[ApiWarning] = []

    for index, sentence in enumerate(request.sentences):
        # Every token consumes at least one UTF-8 byte, so the token count can
        # never exceed the byte length. When the whole sentence fits the limit
        # in bytes, truncation is impossible and we skip tokenizing it entirely
        # (avoids re-tokenizing the common short input that encode() will
        # tokenize again anyway).
        sentence_byte_length = len(sentence.encode("utf-8"))
        truncation_candidates = []
        for tokenizer, model_name, max_length in active_tokenizers:
            if sentence_byte_length <= max_length:
                continue
            original_tokens = _count_tokens(tokenizer, sentence)
            if original_tokens > max_length:
                truncation_candidates.append(
                    (
                        model_name,
                        max_length,
                        original_tokens,
                        original_tokens - max_length,
                    )
                )

        if not truncation_candidates:
            continue

        model_name, max_length, original_tokens, _ = max(
            truncation_candidates,
            key=lambda item: item[3],
        )
        warning = _make_truncation_warning(
            code="input_truncated",
            field="sentences",
            index=index,
            pointer=f"/sentences/{index}",
            model_name=model_name,
            max_tokens=max_length,
            original_tokens=original_tokens,
            message="Input text was truncated to the model token limit.",
        )
        logging.warning(
            "Input truncated endpoint=/embeddings field=sentences index=%s "
            "model=%s original_tokens=%s max_tokens=%s truncated_tokens=%s",
            index,
            model_name,
            original_tokens,
            max_length,
            warning.details.truncated_tokens,
        )
        warnings_out.append(warning)

    return warnings_out


# --- Reranker Wrapper ---


def _ensure_prepare_for_model_compat(tokenizer: Any) -> None:
    """Restore the XLM-R pair builder removed from Transformers 5."""
    if callable(getattr(tokenizer, "prepare_for_model", None)):
        return

    def prepare_for_model(
        first_ids,
        second_ids=None,
        *,
        truncation=None,
        max_length=None,
        padding=False,
        **kwargs,
    ) -> dict[str, List[int]]:
        del padding, kwargs
        first = list(first_ids)
        second = list(second_ids) if second_ids is not None else None
        special_tokens = 4 if second is not None else 2
        if max_length is not None:
            content_limit = max(0, int(max_length) - special_tokens)
            if second is not None and truncation == "only_second":
                second = second[: max(0, content_limit - len(first))]
            elif (
                len(first) + (len(second) if second is not None else 0) > content_limit
            ):
                first = first[:content_limit]
                if second is not None:
                    second = second[: max(0, content_limit - len(first))]

        bos = int(tokenizer.bos_token_id)
        eos = int(tokenizer.eos_token_id)
        input_ids = [bos, *first, eos]
        if second is not None:
            input_ids.extend([eos, *second, eos])
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

    tokenizer.prepare_for_model = prepare_for_model


class RerankerWrapper:
    """Encapsulate the selected reranker model for cross-encoder scoring."""

    def __init__(self, model_name: str):
        """Load the profile-selected reranker and bind its scoring path.

        Args:
            model_name: BGE or Qwen reranker identifier or local snapshot.
        """
        self.model_name = model_name
        # Both model paths share this pool. A separate async lock serializes
        # preparation and scoring for this reranker instance across its lanes.
        self.executor = GPU_EXECUTOR
        self._score_fn: Callable[..., List[float]]

        if self.model_name == QWEN_RERANKER_MODEL:
            self._init_qwen()
        else:
            self._init_bge()

    def _init_bge(self) -> None:
        use_fp16 = has_cuda
        logging.info(
            f"Initializing BGE reranker '{self.model_name}' on '{device}' "
            f"with FP16: {use_fp16}"
        )
        model_path = _cache_first_model_path(self.model_name)
        self.model = FlagReranker(model_path, use_fp16=use_fp16)
        _ensure_prepare_for_model_compat(self.model.tokenizer)
        self._score_fn = self._score_bge

        if has_cuda:
            logging.info("Performing reranker warm-up...")
            _ = self.model.compute_score(
                [["warm-up query", "warm-up passage"]], normalize=False
            )
        logging.info("Reranker ready.")

    def _init_qwen(self) -> None:
        if ADVANCED_RERANK_RUNTIME != "torch":
            raise RuntimeError(
                "Qwen reranker runtime "
                f"{ADVANCED_RERANK_RUNTIME!r} is unavailable without a validated "
                "cached export; refusing to silently execute Torch"
            )
        logging.info(f"Initializing Qwen reranker '{self.model_name}' on '{device}'")
        model_path = _cache_first_model_path(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        # Qwen3-Reranker tokenizer ships without a pad_token; reuse EOS so
        # tokenizer.pad() does not raise. Required for batched scoring.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Pre-trip transformers' instance gate so tokenizer.pad() does not emit the
        # benign "fast tokenizer / use __call__" notice. The two-step encode+pad
        # below is intentional (per-pair prefix/suffix injection), so the hint
        # does not apply. Sets transformers' own deprecation_warnings flag rather
        # than touching the python warnings module, which the logger ignores.
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        # transformers >=4.56 renamed `torch_dtype` to `dtype`; use the current name.
        rerank_dtype = _torch_dtype_for_precision(ADVANCED_RERANK_PRECISION)
        model_kwargs: Dict[str, Any] = (
            {"dtype": rerank_dtype} if rerank_dtype is not None else {}
        )
        model_kwargs["attn_implementation"] = ADVANCED_RERANK_ATTENTION
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            .to(device)
            .eval()
        )
        logging.info(
            "Qwen reranker: attention=%s precision=%s",
            ADVANCED_RERANK_ATTENTION,
            ADVANCED_RERANK_PRECISION,
        )
        self.model.config.use_cache = False
        self.model = _compile_advanced_model(
            self.model,
            enabled=ADVANCED_RERANK_TORCH_COMPILE,
            mode=ADVANCED_RERANK_COMPILE_MODE,
            backend="qwen_reranker",
        )
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(
            QWEN_RERANKER_PREFIX, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            QWEN_RERANKER_SUFFIX, add_special_tokens=False
        )
        self._score_fn = self._score_qwen

        if has_cuda:
            logging.info("Performing reranker warm-up...")
            _ = self._score_qwen(
                [["warm-up query", "warm-up passage"]], normalize=False
            )
        logging.info("Reranker ready.")

    def score(
        self, pairs: List[List[str]], normalize: bool, deadline=None
    ) -> List[float]:
        """Compute relevance scores for a list of [query, passage] pairs.

        Args:
            pairs: List of [query, passage] pairs to score
            normalize: Backend-compatible normalization flag
            deadline: Optional InferenceDeadline for cooperative timeout

        Returns:
            List of float scores, one per pair
        """
        return self._score_fn(pairs, normalize, deadline=deadline)

    def _rerank_tokenizer(self):
        return (
            self.tokenizer
            if self.model_name == QWEN_RERANKER_MODEL
            else self.model.tokenizer
        )

    def _rerank_pair_special_tokens(self) -> int:
        tokenizer = self._rerank_tokenizer()
        if hasattr(tokenizer, "num_special_tokens_to_add"):
            return int(tokenizer.num_special_tokens_to_add(pair=True))
        return 0

    def _qwen_instruction_overhead_tokens(self) -> int:
        if self.model_name != QWEN_RERANKER_MODEL:
            return 0
        empty_instruction = self._format_qwen_instruction("", "")
        return len(_encode_token_ids(self.tokenizer, empty_instruction))

    def _qwen_max_pair_length(self) -> int:
        return max(
            1,
            QWEN_RERANK_MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

    def _rerank_content_budget(self) -> int:
        if self.model_name == QWEN_RERANKER_MODEL:
            reserved = (
                len(self.prefix_tokens)
                + len(self.suffix_tokens)
                + self._qwen_instruction_overhead_tokens()
            )
            return max(0, QWEN_RERANK_MAX_LENGTH - reserved)
        return max(0, BGE_RERANK_MAX_LENGTH - self._rerank_pair_special_tokens())

    def _decode_tokens(self, token_ids: List[int]) -> str:
        return self._rerank_tokenizer().decode(token_ids, skip_special_tokens=True)

    def _qwen_formatted_token_count(self, query: str, passage: str) -> int:
        formatted = self._format_qwen_instruction(query, passage)
        return len(_encode_token_ids(self.tokenizer, formatted))

    def _decode_token_prefix(self, token_ids: List[int], length: int) -> str:
        if length <= 0:
            return ""
        return self._decode_tokens(token_ids[:length])

    def _fit_qwen_passage_to_budget(self, query: str, passage: str) -> Tuple[str, int]:
        passage_tokens = _encode_token_ids(self.tokenizer, passage)
        max_pair_length = self._qwen_max_pair_length()
        low = 0
        high = len(passage_tokens)
        best_length = 0
        best_passage = ""

        while low <= high:
            mid = (low + high) // 2
            candidate = self._decode_token_prefix(passage_tokens, mid)
            if self._qwen_formatted_token_count(query, candidate) <= max_pair_length:
                best_length = mid
                best_passage = candidate
                low = mid + 1
            else:
                high = mid - 1

        return best_passage, best_length

    def _fit_qwen_query_to_budget(self, query: str, passage: str) -> Tuple[str, int]:
        query_tokens = _encode_token_ids(self.tokenizer, query)
        max_pair_length = self._qwen_max_pair_length()
        low = 0
        high = len(query_tokens)
        best_length = 0
        best_query = ""

        while low <= high:
            mid = (low + high) // 2
            candidate = self._decode_token_prefix(query_tokens, mid)
            if self._qwen_formatted_token_count(candidate, passage) <= max_pair_length:
                best_length = mid
                best_query = candidate
                low = mid + 1
            else:
                high = mid - 1

        return best_query, best_length

    def _upsert_rerank_warning(
        self,
        response_warnings: List[ApiWarning],
        *,
        code: str,
        field: str,
        index: Optional[int],
        pointer: str,
        max_tokens: int,
        original_tokens: int,
        message: str,
    ) -> ApiWarning:
        for warning in response_warnings:
            if (
                warning.code == code
                and warning.target.field == field
                and warning.target.index == index
            ):
                warning.details = WarningDetails(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    original_tokens=original_tokens,
                    truncated_tokens=max(0, original_tokens - max_tokens),
                )
                warning.message = message
                return warning

        warning = _make_truncation_warning(
            code=code,
            field=field,
            index=index,
            pointer=pointer,
            model_name=self.model_name,
            max_tokens=max_tokens,
            original_tokens=original_tokens,
            message=message,
        )
        response_warnings.append(warning)
        return warning

    def _ensure_qwen_pair_fits(
        self,
        query: str,
        passage: str,
        index: int,
        original_query_tokens: int,
        original_passage_tokens: int,
        response_warnings: List[ApiWarning],
    ) -> Tuple[str, str]:
        if (
            self._qwen_formatted_token_count(query, passage)
            <= self._qwen_max_pair_length()
        ):
            return query, passage

        current_passage_tokens = len(_encode_token_ids(self.tokenizer, passage))
        passage, passage_length = self._fit_qwen_passage_to_budget(query, passage)
        if passage_length < current_passage_tokens:
            warning = self._upsert_rerank_warning(
                response_warnings,
                code="passage_truncated",
                field="passages",
                index=index,
                pointer=f"/passages/{index}",
                max_tokens=min(original_passage_tokens, passage_length),
                original_tokens=original_passage_tokens,
                message="Rerank passage was truncated to the model token limit.",
            )
            logging.warning(
                "Input retruncated endpoint=/rerank field=passages index=%s "
                "model=%s original_tokens=%s max_tokens=%s truncated_tokens=%s",
                index,
                self.model_name,
                original_passage_tokens,
                warning.details.max_tokens,
                warning.details.truncated_tokens,
            )

        if (
            self._qwen_formatted_token_count(query, passage)
            <= self._qwen_max_pair_length()
        ):
            return query, passage

        current_query_tokens = len(_encode_token_ids(self.tokenizer, query))
        query, query_length = self._fit_qwen_query_to_budget(query, passage)
        if query_length < current_query_tokens:
            warning = self._upsert_rerank_warning(
                response_warnings,
                code="query_truncated",
                field="query",
                index=None,
                pointer="/query",
                max_tokens=min(original_query_tokens, query_length),
                original_tokens=original_query_tokens,
                message="Rerank query was truncated to the model token limit.",
            )
            logging.warning(
                "Input retruncated endpoint=/rerank field=query model=%s "
                "original_tokens=%s max_tokens=%s truncated_tokens=%s",
                self.model_name,
                original_query_tokens,
                warning.details.max_tokens,
                warning.details.truncated_tokens,
            )

        return query, passage

    def prepare_pairs_for_scoring(
        self, query: str, passages: List[str]
    ) -> Tuple[List[List[str]], List[ApiWarning]]:
        """Fit query-passage pairs to the active reranker's token contract.

        Passages are truncated before queries whenever possible. Qwen pairs
        are measured using the exact formatted sequence used during scoring.

        Args:
            query: Query shared by every candidate passage.
            passages: Candidate passages in original order.

        Returns:
            Prepared pairs and structured truncation warnings.
        """
        tokenizer = self._rerank_tokenizer()
        pair_budget = self._rerank_content_budget()
        query_tokens = _encode_token_ids(tokenizer, query)
        original_query_tokens = len(query_tokens)
        prepared_pairs: List[List[str]] = []
        response_warnings: List[ApiWarning] = []

        if self.model_name == QWEN_RERANKER_MODEL:
            # Qwen tokenizes one formatted prompt and then injects fixed token
            # prefixes/suffixes. Raw query+passage counts are not additive at
            # the field boundaries, so using the generic content budget first
            # can truncate an exactly fitting 8192-token sequence. Fit every
            # pair against the same formatted representation used by scoring.
            for index, passage in enumerate(passages):
                original_passage_tokens = len(_encode_token_ids(tokenizer, passage))
                prepared_query, prepared_passage = self._ensure_qwen_pair_fits(
                    query,
                    passage,
                    index,
                    original_query_tokens,
                    original_passage_tokens,
                    response_warnings,
                )
                prepared_pairs.append([prepared_query, prepared_passage])
            return prepared_pairs, response_warnings

        if len(query_tokens) > pair_budget:
            query_tokens = query_tokens[:pair_budget]
            query = self._decode_tokens(query_tokens)
            warning = _make_truncation_warning(
                code="query_truncated",
                field="query",
                index=None,
                pointer="/query",
                model_name=self.model_name,
                max_tokens=pair_budget,
                original_tokens=original_query_tokens,
                message="Rerank query was truncated to the model token limit.",
            )
            logging.warning(
                "Input truncated endpoint=/rerank field=query model=%s "
                "original_tokens=%s max_tokens=%s truncated_tokens=%s",
                self.model_name,
                original_query_tokens,
                pair_budget,
                warning.details.truncated_tokens,
            )
            response_warnings.append(warning)

        passage_budget = max(0, pair_budget - len(query_tokens))

        for index, passage in enumerate(passages):
            passage_tokens = _encode_token_ids(tokenizer, passage)
            original_passage_tokens = len(passage_tokens)
            if original_passage_tokens > passage_budget:
                passage_tokens = passage_tokens[:passage_budget]
                passage = self._decode_tokens(passage_tokens)
                warning = _make_truncation_warning(
                    code="passage_truncated",
                    field="passages",
                    index=index,
                    pointer=f"/passages/{index}",
                    model_name=self.model_name,
                    max_tokens=passage_budget,
                    original_tokens=original_passage_tokens,
                    message="Rerank passage was truncated to the model token limit.",
                )
                logging.warning(
                    "Input truncated endpoint=/rerank field=passages index=%s "
                    "model=%s original_tokens=%s max_tokens=%s truncated_tokens=%s",
                    index,
                    self.model_name,
                    original_passage_tokens,
                    passage_budget,
                    warning.details.truncated_tokens,
                )
                response_warnings.append(warning)
            prepared_pairs.append([query, passage])

        return prepared_pairs, response_warnings

    def _score_bge(
        self, pairs: List[List[str]], normalize: bool, deadline=None
    ) -> List[float]:
        if not pairs:
            return []
        all_scores: List[float] = []
        for start in range(0, len(pairs), BGE_RERANK_BATCH_SIZE):
            if deadline is not None:
                deadline.check()
            chunk = pairs[start : start + BGE_RERANK_BATCH_SIZE]

            def _compute(batch=chunk):
                return self.model.compute_score(
                    batch,
                    normalize=normalize,
                    batch_size=BGE_RERANK_BATCH_SIZE,
                    max_length=BGE_RERANK_MAX_LENGTH,
                )

            raw = vram_budget.run_guarded(
                _compute,
                batch_items=len(chunk),
                batch_tokens=len(chunk) * BGE_RERANK_MAX_LENGTH,
                bytes_per_token=0,
                torch_module=torch,
            )
            all_scores.extend(self._coerce_scores(raw))
        return all_scores

    def _score_qwen(
        self, pairs: List[List[str]], normalize: bool, deadline=None
    ) -> List[float]:
        # Qwen returns yes-probabilities; keep normalize for API-compatible calls.
        _ = normalize
        advanced_backend_requests.labels(backend="qwen_reranker").inc()
        begin_advanced_memory_measurement()
        formatted_pairs = [
            self._format_qwen_instruction(query, passage) for query, passage in pairs
        ]
        if not formatted_pairs:
            return []

        # Sort by text length to minimise padding waste; restore original order after.
        order = sorted(
            range(len(formatted_pairs)), key=lambda i: len(formatted_pairs[i])
        )
        sorted_pairs = [formatted_pairs[i] for i in order]
        phase_started_at = time.monotonic()
        encoded = self._encode_qwen_pairs(sorted_pairs)
        observe_advanced_phase("qwen_reranker", "tokenize", phase_started_at)

        all_scores: List[float] = [0.0] * len(pairs)
        processed = 0
        size_ceiling = [QWEN_RERANK_BATCH_SIZE]
        for chunk in iter_token_budget_batches(
            encoded,
            QWEN_RERANK_BATCH_SIZE,
            QWEN_RERANK_EFFECTIVE_MAX_BATCH_TOKENS,
            batch_policy=(
                QWEN_RERANK_BATCH_POLICY if QWEN_RERANK_POLICY_ACTIVE else None
            ),
            max_batch_tokens_policy=(
                QWEN_RERANK_EFFECTIVE_TOKENS_POLICY
                if QWEN_RERANK_TOKENS_POLICY_ACTIVE
                else None
            ),
        ):
            if deadline is not None:
                deadline.check()
            batch_scores = self._score_qwen_chunk_with_fallback(chunk, size_ceiling)
            for local_i, score in enumerate(batch_scores):
                all_scores[order[processed + local_i]] = score
            processed += len(batch_scores)

        record_advanced_memory_measurement("qwen_reranker")
        return all_scores

    def _score_qwen_chunk_with_fallback(self, chunk: dict, size_ceiling: list) -> list:
        """Score a chunk, halving on exhaustion until it fits or reaches 1.

        ``size_ceiling`` is a single-element list carrying the sticky limit
        across chunks of the same request. Once a smaller size is proven
        necessary, later chunks - which are longer, since pairs are sorted by
        ascending length - start there instead of re-discovering the failure.
        """
        if not RERANK_OOM_FALLBACK_ENABLED:
            return vram_budget.run_guarded(
                lambda: self._forward_qwen_chunk(chunk),
                batch_items=len(chunk["input_ids"]),
                batch_tokens=_chunk_token_estimate(chunk),
                bytes_per_token=self._logits_bytes_per_token(),
                torch_module=torch,
            )

        pending = [chunk]
        scores = []
        while pending:
            current = pending.pop(0)
            items = len(current["input_ids"])
            if items > size_ceiling[0]:
                head, tail = _split_chunk(current)
                pending.insert(0, tail)
                pending.insert(0, head)
                continue
            try:
                scores.extend(
                    vram_budget.run_guarded(
                        lambda c=current: self._forward_qwen_chunk(c),
                        batch_items=items,
                        batch_tokens=_chunk_token_estimate(current),
                        bytes_per_token=self._logits_bytes_per_token(),
                        torch_module=torch,
                    )
                )
            except vram_budget.VramExhausted:
                if items <= 1:
                    raise
                rerank_batch_halvings.inc()
                size_ceiling[0] = max(1, items // 2)
                head, tail = _split_chunk(current)
                pending.insert(0, tail)
                pending.insert(0, head)
        return scores

    def _logits_bytes_per_token(self) -> int:
        """Bytes the LM head writes per position, for failure reporting."""
        vocab_size = int(getattr(self.model.config, "vocab_size", 0) or 0)
        return vram_budget.logits_bytes_per_token(
            vocab_size, vram_budget.dtype_size_bytes(ADVANCED_RERANK_PRECISION)
        )

    def _forward_qwen_chunk(self, chunk: dict) -> list:
        """Score one padded chunk and return CPU-side probabilities.

        Every device tensor is local to this frame on purpose: when the
        allocator fails, run_guarded drops this frame with the original
        traceback, so the batch becomes collectable immediately.
        """
        phase_started_at = time.monotonic()
        inputs = self.tokenizer.pad(chunk, padding=True, return_tensors="pt")
        observe_advanced_phase("qwen_reranker", "tokenize", phase_started_at)
        record_advanced_token_batch(inputs, backend="qwen_reranker")

        phase_started_at = time.monotonic()
        inputs = {key: value.to(device) for key, value in inputs.items()}
        observe_advanced_phase("qwen_reranker", "host_to_device", phase_started_at)

        forward_timer = start_advanced_device_phase()
        with torch.inference_mode():
            # logits_to_keep=1 restricts the LM head to the final position.
            # The default (0) means "all positions" and makes the head
            # allocate [B, S, vocab] only for this code to keep the last
            # row. The tokenizer pads left (see __init__), so the final
            # position is always a real token and the result is identical.
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[
                :, -1, :
            ]
        stop_advanced_device_phase(forward_timer)

        postprocess_timer = start_advanced_device_phase()
        true_vec = logits[:, self.token_true_id]
        false_vec = logits[:, self.token_false_id]
        binary = torch.stack([false_vec, true_vec], dim=1)
        probs = torch.nn.functional.log_softmax(binary, dim=1)
        probabilities = probs[:, 1].exp().detach()
        stop_advanced_device_phase(postprocess_timer)

        transfer_timer = start_advanced_device_phase()
        probabilities_cpu = probabilities.cpu()
        stop_advanced_device_phase(transfer_timer)

        observe_advanced_device_phase("qwen_reranker", "forward", forward_timer)
        observe_advanced_device_phase("qwen_reranker", "postprocess", postprocess_timer)
        observe_advanced_device_phase("qwen_reranker", "device_to_host", transfer_timer)

        phase_started_at = time.monotonic()
        batch_scores = probabilities_cpu.tolist()
        observe_advanced_phase("qwen_reranker", "output_conversion", phase_started_at)
        return batch_scores

    def _format_qwen_instruction(self, query: str, passage: str) -> str:
        return (
            f"<Instruct>: {QWEN_RERANKER_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {passage}"
        )

    def _tokenize_qwen_pairs(
        self, formatted_pairs: List[str]
    ) -> Dict[str, torch.Tensor]:
        return self.tokenizer.pad(
            self._encode_qwen_pairs(formatted_pairs),
            padding=True,
            return_tensors="pt",
        )

    def _encode_qwen_pairs(self, formatted_pairs: List[str]) -> Dict[str, List]:
        max_pair_length = self._qwen_max_pair_length()
        # Two-step encode+pad is required so prefix/suffix tokens can be injected
        # per-pair. The fast-tokenizer "use __call__" notice from .pad() is benign
        # and already silenced at init (deprecation_warnings flag pre-tripped).
        inputs = self.tokenizer(
            formatted_pairs,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        for input_ids in inputs["input_ids"]:
            if len(input_ids) > max_pair_length:
                raise RuntimeError("Qwen rerank pair exceeds prepared token budget")
        for index, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = (
                self.prefix_tokens + input_ids + self.suffix_tokens
            )
        inputs["attention_mask"] = [
            [1] * len(input_ids) for input_ids in inputs["input_ids"]
        ]
        return inputs

    @staticmethod
    def _coerce_scores(raw) -> List[float]:
        if isinstance(raw, (int, float)):
            return [float(raw)]
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        return [float(score) for score in raw]


# --- 4. Asynchronous Request Processor with Batching ---


class RequestProcessor:
    """Manage dynamic batching of requests to maximize GPU throughput.

    Collects incoming embedding requests into batches to process them efficiently
    on the GPU. Implements asynchronous processing with configurable batch size
    and accumulation timeout.
    """

    def __init__(
        self,
        model_wrapper: Union[BgeM3EmbeddingBackend, EmbeddingService],
        max_batch_size: int,
        accumulation_timeout: float,
        stats=None,
        max_queue_size: int = MAX_QUEUE_SIZE,
    ):
        """Configure bounded asynchronous aggregation for embedding requests.

        Args:
            model_wrapper: Profile-specific embedding service.
            max_batch_size: Maximum HTTP requests combined into one worker batch.
            accumulation_timeout: Maximum batching delay in seconds.
            stats: Optional server statistics collector.
            max_queue_size: Backpressure limit for queued requests.
        """
        self.model_wrapper = model_wrapper
        self.max_batch_size = max_batch_size
        self.accumulation_timeout = accumulation_timeout
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.response_futures = {}
        self._reservations = {}
        self._admissions = set()
        # The shared pool may have measured advanced lanes. This one processor
        # still submits one embedding batch at a time.
        self.executor = GPU_EXECUTOR
        self.stats = stats
        self.is_shutting_down = False

    async def start_processing_loop(self):
        """Start the main loop that processes the queue.

        Creates an asyncio task that continuously monitors the request queue
        and processes batches of requests.
        """
        logging.info("Starting the request processing loop.")
        self.processing_loop_task = asyncio.create_task(self._processing_loop())

    async def _processing_loop(self):
        """Collect requests and process them in batches.

        Main processing loop that accumulates requests up to the maximum batch size
        or until the accumulation timeout is reached, then processes the batch.
        """
        while not self.is_shutting_down or not self.queue.empty():
            requests = []
            loop = asyncio.get_running_loop()
            start_time = loop.time()

            while len(requests) < self.max_batch_size:
                timeout = self.accumulation_timeout - (loop.time() - start_time)
                if timeout <= 0 or (self.is_shutting_down and self.queue.empty()):
                    break
                try:
                    req_id, req_data, enqueued_at = await asyncio.wait_for(
                        self.queue.get(), timeout=timeout
                    )
                    if self._get_pending_future(req_id) is not None:
                        requests.append((req_id, req_data, enqueued_at))
                    # Update on dequeue as well as enqueue so the gauge cannot
                    # remain stuck at the last observed queue depth.
                    queue_size.set(self.queue.qsize())
                except asyncio.TimeoutError:
                    break

            if requests:
                # Requests with different output contracts cannot share one
                # backend call, so aggregate only identical option sets.
                option_groups = {}
                for req_id, req_data, enqueued_at in requests:
                    options_key = (
                        req_data.return_dense,
                        req_data.return_sparse,
                        _request_return_colbert(req_data),
                        req_data.normalize_dense,
                    )
                    if options_key not in option_groups:
                        option_groups[options_key] = []
                    option_groups[options_key].append((req_id, req_data, enqueued_at))

                for options_key, group_requests in option_groups.items():
                    group_requests = [
                        item
                        for item in group_requests
                        if self._get_pending_future(item[0]) is not None
                    ]
                    if not group_requests:
                        continue
                    all_sentences = [
                        s for _, req, _ in group_requests for s in req.sentences
                    ]
                    request_ids = [req_id for req_id, _, _ in group_requests]

                    await self._run_model_on_batch(
                        all_sentences, request_ids, group_requests, options_key
                    )

    def _embed_on_worker(
        self,
        sentences,
        return_dense,
        return_sparse,
        return_colbert,
        normalize_dense,
        deadline=None,
        requests=None,
    ):
        worker_started_at = time.monotonic()
        warnings = {
            request_id: collect_embedding_truncation_warnings(request)
            for request_id, request, _ in requests or []
        }
        result = self.model_wrapper.embed(
            sentences,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert=return_colbert,
            normalize_dense=normalize_dense,
            deadline=deadline,
        )
        worker_finished_at = time.monotonic()
        if requests is not None:
            result["_warnings"] = warnings
        return result, worker_started_at, worker_finished_at

    def _get_pending_future(self, request_id: str):
        future = self.response_futures.get(request_id)
        if future is None or future.cancelled() or future.done():
            return None
        return future

    def _set_response_exception(self, request_id: str, exc: Exception):
        future = self._get_pending_future(request_id)
        if future is not None:
            future.set_exception(exc)

    async def _run_model_on_batch(
        self, all_sentences, request_ids, requests, options_key
    ):
        """Run the model on the aggregated batch and distribute the results.

        Executes the model on a batch of sentences and distributes the results back
        to the individual requests. Uses the options from the options_key for the
        entire batch since all requests in this group have the same options.

        Args:
            all_sentences: List of all sentences from all requests in the batch
            request_ids: List of request IDs corresponding to the requests
            requests: List of (request_id, request_data) tuples
            options_key: Tuple of return_dense, return_sparse,
                return_colbert, and normalize_dense.
        """
        try:
            return_dense, return_sparse, return_colbert, normalize_dense = options_key

            if self.stats:
                self.stats.update_batch()

            batch_size_histogram.observe(len(all_sentences))

            # Observe queue wait for each item in this batch.
            batch_started_at = time.monotonic()
            for _, _, enqueued_at in requests:
                api_batch_queue_wait_duration.labels(endpoint="/embeddings").observe(
                    batch_started_at - enqueued_at
                )

            # This processor awaits one embedding batch at a time. An approved
            # advanced profile may overlap that batch with rerank on another
            # lane; reranker-instance exclusion is owned by /rerank.
            gpu_start = time.time()
            submitted_at = time.monotonic()
            deadline = make_inference_deadline(GPU_PROCESS_TIMEOUT)

            future = self.executor.submit(
                self._embed_on_worker,
                all_sentences,
                return_dense,
                return_sparse,
                return_colbert,
                normalize_dense,
                deadline,
                requests,
            )
            for request_id in request_ids:
                reservation = self._reservations.get(request_id)
                if reservation is not None:
                    reservation.track(future)
            try:
                (
                    embedding_data,
                    worker_started_at,
                    worker_finished_at,
                ) = await asyncio.wait_for(
                    asyncio.wrap_future(future), timeout=GPU_PROCESS_TIMEOUT
                )
            except (asyncio.TimeoutError, InferenceTimeoutError):
                future.cancel()
                raise

            api_executor_wait_duration.labels(
                endpoint="/embeddings", operation="embed"
            ).observe(worker_started_at - submitted_at)
            api_model_forward_duration.labels(
                endpoint="/embeddings", operation="embed"
            ).observe(worker_finished_at - worker_started_at)

            # Retain the aggregate duration for backward-compatible dashboards.
            gpu_duration = time.time() - gpu_start
            gpu_inference_duration.observe(gpu_duration)

            offset = 0
            for req_id, req_data, _enqueued_at in requests:
                num_sentences = len(req_data.sentences)
                future = self._get_pending_future(req_id)
                try:
                    if future is None:
                        continue

                    result_slice = {
                        "_warnings": embedding_data.get("_warnings", {}).get(req_id, [])
                    }

                    # Extract slices for this request only when data exists.
                    if "dense_vecs" in embedding_data and req_data.return_dense:
                        dense_slice = embedding_data["dense_vecs"][
                            offset : offset + num_sentences
                        ]
                        # Backends may return arrays; responses need native lists.
                        result_slice["dense_vecs"] = (
                            dense_slice.tolist()
                            if hasattr(dense_slice, "tolist")
                            else dense_slice
                        )

                    if "lexical_weights" in embedding_data and req_data.return_sparse:
                        result_slice["lexical_weights"] = embedding_data[
                            "lexical_weights"
                        ][offset : offset + num_sentences]

                    if "colbert_vecs" in embedding_data and _request_return_colbert(
                        req_data
                    ):
                        colbert_slice = embedding_data["colbert_vecs"][
                            offset : offset + num_sentences
                        ]
                        # Backends may return arrays; responses need native lists.
                        result_slice["colbert_vecs"] = (
                            colbert_slice.tolist()
                            if hasattr(colbert_slice, "tolist")
                            else colbert_slice
                        )

                    future.set_result(result_slice)
                except Exception as e:
                    logging.error(
                        f"Error preparing response for request {req_id}: {e}",
                        exc_info=True,
                    )
                    self._set_response_exception(req_id, e)
                finally:
                    offset += num_sentences

            record_gpu_memory_metrics()

        except Exception as e:
            logging.error(f"Error during batch processing: {e}", exc_info=True)
            for req_id in request_ids:
                self._set_response_exception(req_id, e)

    async def process_request(
        self, request_data: EmbedRequestBase, *, lease=None
    ) -> Dict:
        """Own a request future through bounded admission and response waiting."""
        if self.is_shutting_down:
            raise HTTPException(status_code=503, detail="Server is shutting down")
        request_id = str(uuid4())
        response = asyncio.get_running_loop().create_future()
        self.response_futures[request_id] = response
        if lease is not None:
            self._reservations[request_id] = lease
        active_requests.inc()
        try:
            admission = asyncio.create_task(
                self.queue.put((request_id, request_data, time.monotonic()))
            )
            self._admissions.add(admission)
            try:
                done, _ = await asyncio.wait({admission}, timeout=0.5)
                if admission.cancelled():
                    raise HTTPException(
                        status_code=503, detail="Server is shutting down"
                    )
                if not done:
                    requests_rejected.labels(reason="backpressure").inc()
                    if self.stats:
                        self.stats.increment_rejected()
                    raise HTTPException(
                        status_code=503,
                        detail="Server is overloaded. Please retry later.",
                    )
                await admission
            finally:
                if not admission.done():
                    admission.cancel()
                await asyncio.gather(admission, return_exceptions=True)
                self._admissions.discard(admission)
            queue_size.set(self.queue.qsize())
            sentences_processed.inc(len(request_data.sentences))
            if self.stats:
                self.stats.update_request(len(request_data.sentences))
            return await response
        finally:
            self.response_futures.pop(request_id, None)
            self._reservations.pop(request_id, None)
            if not response.done():
                response.cancel()
            elif not response.cancelled():
                response.exception()
            active_requests.dec()

    async def graceful_shutdown(self, timeout: float = 30.0):
        """Close admission, drain accepted work, then settle remaining callers."""
        logging.info("Starting graceful shutdown...")
        deadline = time.monotonic() + timeout
        self.is_shutting_down = True
        admissions = list(self._admissions)
        for admission in admissions:
            admission.cancel()
        if admissions:
            await asyncio.gather(*admissions, return_exceptions=True)
        pending = list(self.response_futures.values())
        remaining = max(0.0, deadline - time.monotonic())
        if pending and remaining:
            await asyncio.wait(pending, timeout=remaining)
        processing = getattr(self, "processing_loop_task", None)
        if processing is not None:
            processing.cancel()
            await asyncio.gather(processing, return_exceptions=True)
        while not self.queue.empty():
            self.queue.get_nowait()
        queue_size.set(0)
        pending = list(self.response_futures.values())
        for request_id in list(self.response_futures):
            self._set_response_exception(
                request_id,
                HTTPException(status_code=503, detail="Server is shutting down"),
            )
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        # The lifespan owns the shared embedding/reranking executor. Its native
        # work must physically finish before model resources can be closed.
        logging.info("Request drain completed")


# --- 5. Server Statistics ---


class ServerStats:
    """Collect statistics on server usage.

    Tracks metrics about server usage including request counts,
    batch processing, and uptime for monitoring and diagnostics.
    """

    def __init__(self):
        """Initialize thread-safe counters and static hardware metadata."""
        self.start_time = time.time()
        self.total_requests = 0
        self.total_sentences = 0
        self.total_batches = 0
        self.gpu_info = self._get_gpu_info() if has_cuda else "CPU"
        self.rejected_requests = 0

    def _get_gpu_info(self) -> str:
        """Get information about the GPU.

        Retrieves the GPU name and memory information if available.

        Returns:
            String containing GPU name and memory or status message
        """
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"{gpu_name} ({gpu_mem:.2f} GB)"
        except Exception:
            return "GPU not available"

    def update_request(self, num_sentences: int = 1):
        """Update statistics for a new request.

        Increments the request counter and adds to the total sentence count.

        Args:
            num_sentences: Number of sentences in the request
        """
        self.total_requests += 1
        self.total_sentences += num_sentences

    def update_batch(self):
        """Update the batch counter.

        Increments the counter tracking the number of batches processed.
        """
        self.total_batches += 1

    def increment_rejected(self):
        """Track rejected requests."""
        self.rejected_requests += 1

    def get_stats(self) -> Dict:
        """Return current server statistics.

        Calculates uptime and returns a dictionary with all tracked metrics.

        Returns:
            Dictionary containing server statistics
        """
        uptime_seconds = time.time() - self.start_time
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        reserved = inference_budget.used

        return {
            "uptime": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            "uptime_seconds": int(uptime_seconds),
            "total_requests": self.total_requests,
            "total_sentences": self.total_sentences,
            "total_batches": self.total_batches,
            "rejected_requests": self.rejected_requests,
            "hardware": self.gpu_info,
            "inference_reserved_tokens": reserved.tokens,
            "inference_reserved_output_bytes": reserved.output_bytes,
        }


# --- 6. FastAPI Application Instance and Lifecycle ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server startup and shutdown within a single async context."""
    logging.info("Starting up server...")
    # Labels only: an operator must be able to see which credentials this
    # process accepts without any token value ever reaching a log line.
    _labels = sorted(API_TOKENS.values())
    logging.info(
        "inbound auth: %s",
        ", ".join(_labels) if _labels else "disabled (no tokens configured)",
    )
    await processor.start_processing_loop()
    rate_limiter.start_cleanup_task()
    logging.info("Server ready to accept requests")
    yield
    logging.info("Shutting down server...")
    await processor.graceful_shutdown(timeout=30.0)
    await asyncio.to_thread(GPU_EXECUTOR.shutdown, wait=True, cancel_futures=True)
    embedding_service.close()


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    docs_url="/docs" if EXPOSE_SMART_EMBEDDER_SWAGGER else None,
    redoc_url="/redoc" if EXPOSE_SMART_EMBEDDER_SWAGGER else None,
    openapi_url="/openapi.json" if EXPOSE_SMART_EMBEDDER_SWAGGER else None,
    lifespan=lifespan,
)

rate_limiter = RateLimiter(
    requests_per_minute=RATE_LIMIT_REQUESTS_PER_MINUTE,
    burst_size=RATE_LIMIT_BURST_SIZE,
)

# Headroom kept free for allocator overhead and transient tensors that are
# not the logits. 512 MiB by default; lower it only with measurements.
RERANK_VRAM_RESERVE_BYTES = (
    _env_int_range("SMART_EMBEDDER_VRAM_RESERVE_MB", 512, min_value=0, max_value=65536)
    * 1024
    * 1024
)

# Turn off only the batch-halving fallback; detection, the typed error and
# the cache release stay on. Set false to diagnose whether halving is
# masking a sizing problem, without redeploying a different image.
#
# Defined here, ahead of the reranker below, because RerankerWrapper._init_qwen
# scores a warm-up pair during construction (under CUDA) and that warm-up
# call reaches _score_qwen_chunk_with_fallback, which reads this flag. Unlike
# the calibrated token policy below, this is a plain env-var read with no
# dependency on the reranker, so it simply moves above it rather than being
# seeded and recomputed.
RERANK_OOM_FALLBACK_ENABLED = _env_bool("SMART_EMBEDDER_OOM_FALLBACK", True)

# Load the models and request processor at startup.
embedding_service = EmbeddingService.from_profile(
    ACTIVE_PROFILE, devices=MULTI_GPU_DEVICES
)

# Seed with the uncalibrated policy before constructing the reranker.
# RerankerWrapper._init_qwen scores a warm-up pair during construction
# (below, under CUDA), and that warm-up call reads these two globals
# through _score_qwen. Calibration itself cannot run any earlier: it needs
# the reranker's model.config.vocab_size, which only exists once the
# reranker below has been built. The warm-up pair is a single short pair,
# so scoring it against the uncalibrated budget is harmless - real traffic
# still gets the calibrated ceiling once it is recomputed below.
QWEN_RERANK_EFFECTIVE_TOKENS_POLICY = QWEN_RERANK_MAX_BATCH_TOKENS_POLICY
QWEN_RERANK_EFFECTIVE_MAX_BATCH_TOKENS = max(
    entry["max_batch_tokens"] for entry in QWEN_RERANK_EFFECTIVE_TOKENS_POLICY
)

reranker = RerankerWrapper(ACTIVE_PROFILE["reranker"])
reranker_instance_lock = asyncio.Lock()
stats = ServerStats()
processor = RequestProcessor(
    embedding_service,
    max_batch_size=MAX_REQUESTS_IN_BATCH,
    accumulation_timeout=REQUEST_FLUSH_TIMEOUT,
    stats=stats,
)


def _measure_vram_state() -> Tuple[int, int, int]:
    """Return (device_free, reserved, allocated) in bytes, zeros without CUDA."""
    if not has_cuda:
        return (0, 0, 0)
    free, _total = torch.cuda.mem_get_info()
    return (free, torch.cuda.memory_reserved(), torch.cuda.memory_allocated())


def _calibrated_rerank_token_policy() -> List[Dict[str, int]]:
    """Clamp the resolved token policy against measured VRAM headroom.

    Applied last, to whatever produced the value - environment variable,
    policy, or a persisted autotune profile - and only ever downward. A
    profile calibrated under different conditions must not outrank the
    measurement taken by the process that actually serves requests.
    """
    policy = QWEN_RERANK_MAX_BATCH_TOKENS_POLICY
    if not has_cuda or reranker.model_name != QWEN_RERANKER_MODEL:
        return policy

    device_free, reserved, allocated = _measure_vram_state()
    headroom = vram_budget.usable_headroom_bytes(device_free, reserved, allocated)
    vocab_size = int(getattr(reranker.model.config, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        return policy
    bytes_per_token = vram_budget.logits_bytes_per_token(
        vocab_size, vram_budget.dtype_size_bytes(ADVANCED_RERANK_PRECISION)
    )
    return [
        {
            **entry,
            "max_batch_tokens": vram_budget.safe_max_batch_tokens(
                headroom_bytes=headroom,
                reserve_bytes=RERANK_VRAM_RESERVE_BYTES,
                bytes_per_token=bytes_per_token,
                configured_cap=entry["max_batch_tokens"],
            ),
        }
        for entry in policy
    ]


QWEN_RERANK_EFFECTIVE_TOKENS_POLICY = _calibrated_rerank_token_policy()
QWEN_RERANK_EFFECTIVE_MAX_BATCH_TOKENS = max(
    entry["max_batch_tokens"] for entry in QWEN_RERANK_EFFECTIVE_TOKENS_POLICY
)
logging.info(
    "Reranker token budget: configured=%s effective=%s reserve_bytes=%s",
    max(e["max_batch_tokens"] for e in QWEN_RERANK_MAX_BATCH_TOKENS_POLICY),
    QWEN_RERANK_EFFECTIVE_MAX_BATCH_TOKENS,
    RERANK_VRAM_RESERVE_BYTES,
)

# Baseline headroom measured once at startup, before request traffic can
# fragment or contend for the allocator pool. Classifying a capacity failure
# against this fixed baseline - rather than against whatever headroom the
# device happens to report at failure time - is what makes the 413/503 split
# deterministic: sustained external GPU pressure lowers the *current*
# headroom without moving this number, so a request that was always
# oversized stays 413 instead of flapping to 503 under load.
_STARTUP_HEADROOM_BYTES = vram_budget.usable_headroom_bytes(*_measure_vram_state())


def _rerank_capacity_response(failure, passage_count: int):
    """Build the HTTP failure for a rerank batch that could not be allocated.

    413 takes precedence: if a single pair does not fit even against the
    headroom measured at startup, the request is permanently oversized and no
    retry can help - and 413 is outside RETRYABLE_HTTP_STATUS, so the
    middleware stops multiplying a certain failure by three.

    503 applies only when a batch of one WOULD fit against that baseline,
    i.e. the request is viable and only the current moment is bad. Without
    this precedence, sustained external GPU pressure would mark a
    permanently oversized request retryable and reintroduce the
    amplification.
    """
    # estimated_bytes uses the logits cost model, which after Task 2 is no
    # longer the dominant term - so this over-estimates. That is the safe
    # direction: over-estimating classifies a borderline request as 413
    # (rejected once, not retried) rather than as 503 (retried three times).
    per_item_bytes = failure.estimated_bytes // max(1, failure.batch_items)
    cost_known = per_item_bytes > 0

    if failure.estimated_bytes <= 0:
        # Cost unknown rather than zero: the BGE path reports no per-token
        # estimate, because a cross-encoder with a scalar head has no
        # vocabulary projection to model. Treating unknown as "fits" would
        # mark every BGE capacity failure retryable and hand the middleware
        # three attempts at a request that cannot succeed. With the batch
        # already down to one item and no cost model to argue otherwise,
        # 413 is the honest answer.
        single_item_fits_baseline = failure.batch_items > 1
    else:
        single_item_fits_baseline = per_item_bytes <= _STARTUP_HEADROOM_BYTES

    detail = {
        "passages": passage_count,
        "batch_items": failure.batch_items,
        "batch_tokens": failure.batch_tokens,
        "estimated_bytes": failure.estimated_bytes,
        "startup_headroom_bytes": _STARTUP_HEADROOM_BYTES,
    }
    if cost_known:
        # Meaningless on a BGE failure: the Qwen token-budget policy does not
        # apply to the fixed-batch BGE path, so it is only reported when the
        # per-item cost model (and therefore the value it constrains) is
        # actually known.
        detail["effective_max_batch_tokens"] = QWEN_RERANK_EFFECTIVE_MAX_BATCH_TOKENS

    if not single_item_fits_baseline:
        rerank_capacity_rejections.labels(reason="oversized").inc()
        detail["error"] = "rerank_capacity_exceeded"
        detail["message"] = (
            "A single query/passage pair exceeds the available GPU memory. "
            "Shorten the passages (see QUERY_RERANK_MAX_CHARS) or lower "
            "QWEN_RERANK_MAX_LENGTH; retrying the same request will not help."
        )
        return HTTPException(status_code=413, detail=detail)

    rerank_capacity_rejections.labels(reason="pressure").inc()
    detail["error"] = "rerank_vram_pressure"
    message = "Not enough free GPU memory for this batch right now. Retry"
    if cost_known:
        # Only meaningful when per-item cost is known: with it unknown (the
        # BGE path) there is no cost model to derive a passage count from,
        # and fabricating one produced nonsense like a byte count masquerading
        # as a passage limit.
        suggested = max(1, _STARTUP_HEADROOM_BYTES // per_item_bytes)
        detail["suggested_max_passages"] = int(suggested)
        message += f", or send at most {int(suggested)} passages per request."
    else:
        message += "."
    detail["message"] = message
    return HTTPException(status_code=503, detail=detail, headers={"Retry-After": "2"})


def _bge_torch_compile_active() -> bool:
    backend = getattr(embedding_service, "bge_backend", None)
    return bool(getattr(backend, "torch_compile_active", False))


def _model_attention(model: Any) -> str:
    config = getattr(model, "config", None)
    return str(getattr(config, "_attn_implementation", "default"))


def _reranker_total_max_length() -> int:
    return (
        QWEN_RERANK_MAX_LENGTH
        if reranker.model_name == QWEN_RERANKER_MODEL
        else BGE_RERANK_MAX_LENGTH
    )


def _advanced_effective_config() -> Dict[str, Any]:
    """Return the effective ADVANCED serving policy without secret values."""
    qwen_backend = getattr(embedding_service, "qwen_dense_backend", None)
    qwen_ort = getattr(qwen_backend, "ort_session", None)
    milco_backend = getattr(embedding_service, "milco_sparse_backend", None)
    prefix_tokens = len(getattr(reranker, "prefix_tokens", []))
    suffix_tokens = len(getattr(reranker, "suffix_tokens", []))
    rerank_max_length = _reranker_total_max_length()
    rerank_content_budget = reranker._rerank_content_budget()
    return {
        "active": embedding_service.profile == EMBEDDER_PROFILE_ADVANCED,
        "dense_model": embedding_service.dense_model_name,
        "dense_runtime": ADVANCED_DENSE_RUNTIME,
        "dense_precision": ADVANCED_DENSE_PRECISION,
        "dense_attention": (
            "n/a"
            if qwen_ort is not None
            else (
                _model_attention(qwen_backend.model)
                if qwen_backend is not None
                else "n/a"
            )
        ),
        "dense_torch_compile": (
            ADVANCED_DENSE_TORCH_COMPILE if qwen_ort is None else False
        ),
        "dense_onnx_artifact": (
            str(qwen_ort.model_path) if qwen_ort is not None else None
        ),
        "dense_pinned_memory": ADVANCED_DENSE_PINNED_MEMORY,
        "dense_prefetch_batches": ADVANCED_DENSE_PREFETCH_BATCHES,
        "dense_batch_size": QWEN_EMBED_BATCH_SIZE,
        "dense_max_batch_tokens": QWEN_EMBED_MAX_BATCH_TOKENS,
        "dense_batch_policy": QWEN_EMBED_BATCH_POLICY,
        "dense_max_batch_tokens_policy": QWEN_EMBED_MAX_BATCH_TOKENS_POLICY,
        "dense_max_length": QWEN_EMBED_MAX_LENGTH,
        "sparse_model": embedding_service.sparse_model_name,
        "sparse_runtime": ADVANCED_SPARSE_RUNTIME,
        "sparse_weights_precision": ADVANCED_SPARSE_PRECISION,
        "sparse_compute_precision": ADVANCED_SPARSE_COMPUTE_PRECISION,
        "sparse_torch_compile": ADVANCED_SPARSE_TORCH_COMPILE,
        "sparse_pinned_memory": ADVANCED_SPARSE_PINNED_MEMORY,
        "sparse_prefetch_batches": ADVANCED_SPARSE_PREFETCH_BATCHES,
        "sparse_batch_size": MILCO_BATCH_SIZE,
        "sparse_max_batch_tokens": MILCO_MAX_BATCH_TOKENS,
        "sparse_batch_policy": MILCO_BATCH_POLICY,
        "sparse_max_batch_tokens_policy": MILCO_MAX_BATCH_TOKENS_POLICY,
        "sparse_projection_chunk_tokens": MILCO_PROJECTION_CHUNK_TOKENS,
        "sparse_max_length": getattr(milco_backend, "max_length", MILCO_MAX_LENGTH),
        "sparse_english_vocab_size": int(
            getattr(
                getattr(getattr(milco_backend, "model", None), "config", None),
                "en_vocab_size",
                0,
            )
        ),
        "milco_source_view": MILCO_SOURCE_VIEW,
        "milco_top_k": MILCO_TOP_K,
        "schedule": ADVANCED_SCHEDULE_MODE,
        "inference_lanes": ADVANCED_INFERENCE_LANES,
        "queue_batch": MAX_REQUESTS_IN_BATCH,
        "flush_timeout_ms": round(REQUEST_FLUSH_TIMEOUT * 1000),
        "worker_spin": ADVANCED_CPU_SPIN_POLICY,
        "cpu_affinity_mode": ADVANCED_CPU_AFFINITY,
        "cpu_affinity_cpus": ADVANCED_EFFECTIVE_CPU_AFFINITY,
        "allow_cpu_oversubscription": ADVANCED_CPU_ALLOW_OVERSUBSCRIPTION,
        "reranker_model": reranker.model_name,
        "reranker_runtime": ADVANCED_RERANK_RUNTIME,
        "reranker_precision": ADVANCED_RERANK_PRECISION,
        "reranker_attention": _model_attention(getattr(reranker, "model", None)),
        "reranker_torch_compile": ADVANCED_RERANK_TORCH_COMPILE,
        "reranker_batch_size": QWEN_RERANK_BATCH_SIZE,
        "reranker_batch_policy": QWEN_RERANK_BATCH_POLICY,
        "reranker_max_batch_tokens_policy": (QWEN_RERANK_MAX_BATCH_TOKENS_POLICY),
        "reranker_effective_max_batch_tokens": QWEN_RERANK_EFFECTIVE_MAX_BATCH_TOKENS,
        "reranker_vram_reserve_bytes": RERANK_VRAM_RESERVE_BYTES,
        "reranker_max_length": rerank_max_length,
        "reranker_content_budget": rerank_content_budget,
        "reranker_fixed_overhead_tokens": max(
            0, rerank_max_length - rerank_content_budget
        ),
        "reranker_prefix_tokens": prefix_tokens,
        "reranker_suffix_tokens": suffix_tokens,
        "reranker_instruction": (
            QWEN_RERANKER_INSTRUCTION
            if reranker.model_name == QWEN_RERANKER_MODEL
            else ""
        ),
    }


ADVANCED_EFFECTIVE_CONFIG = _advanced_effective_config()


def _autotune_observability() -> Dict[str, Any]:
    """Describe the tuner that owns the currently active serving profile."""
    if embedding_service.profile == EMBEDDER_PROFILE_ADVANCED:
        advanced = ADVANCED_AUTOTUNE_SELECTED.get("advanced", {})
        dense = advanced.get("dense", {})
        schedule = advanced.get("schedule", {})
        dense_runtime = dense.get("runtime", "torch")
        onnx_active = str(dense_runtime).startswith("onnx-")
        return {
            "state": ADVANCED_AUTOTUNE_RUNTIME_PROFILE.state,
            "mode": ADVANCED_AUTOTUNE_CONFIG.mode,
            "objective": ADVANCED_AUTOTUNE_CONFIG.objective,
            "fingerprint_sha256": (
                ADVANCED_AUTOTUNE_RUNTIME_PROFILE.fingerprint_sha256
            ),
            "runtime": dense_runtime,
            "torch_compile": (
                bool(dense.get("torch_compile", False)) if not onnx_active else False
            ),
            "torch_compile_policy": "auto",
            "torch_compile_mode": dense.get("torch_compile_mode", "default"),
            "torch_compile_reason": "advanced-profile-or-safe-default",
            "precision": dense.get("precision", "fp16" if has_cuda else "fp32"),
            "threads": _cpu_threads if device == "cpu" else None,
            "inference_concurrency": schedule.get("inference_lanes", 1),
            "onnx_execution_mode": "sequential" if onnx_active else "n/a",
            "onnx_spin_policy": (
                schedule.get("worker_spin", "default") if onnx_active else "n/a"
            ),
            "onnx_inter_op_threads": (
                _cpu_interop_threads if onnx_active and device == "cpu" else 0
            ),
            "batch_policy": dense.get("batch_policy", []),
            "fallback_reason": (ADVANCED_AUTOTUNE_RUNTIME_PROFILE.fallback_reason),
            "profile_path": ADVANCED_AUTOTUNE_RUNTIME_PROFILE.profile_path,
            "selected": ADVANCED_AUTOTUNE_SELECTED,
        }
    return {
        "state": AUTOTUNE_RUNTIME_PROFILE.state,
        "mode": AUTOTUNE_CONFIG.mode,
        "objective": AUTOTUNE_CONFIG.objective,
        "fingerprint_sha256": AUTOTUNE_RUNTIME_PROFILE.fingerprint_sha256,
        "runtime": BGE_EFFECTIVE_RUNTIME,
        "torch_compile": _bge_torch_compile_active(),
        "torch_compile_policy": AUTOTUNE_CONFIG.torch_compile,
        "torch_compile_mode": AUTOTUNE_CONFIG.torch_compile_mode,
        "torch_compile_reason": str(
            AUTOTUNE_SELECTED.get("torch_compile_reason", "operator-or-default")
        ),
        "precision": str(
            AUTOTUNE_SELECTED.get("precision", "fp16" if has_cuda else "fp32")
        ),
        "threads": _cpu_threads if device == "cpu" else None,
        "inference_concurrency": BGE_EMBED_INFERENCE_CONCURRENCY,
        "onnx_execution_mode": BGE_ONNX_EXECUTION_MODE,
        "onnx_spin_policy": BGE_ONNX_SPIN_POLICY,
        "onnx_inter_op_threads": BGE_ONNX_INTER_OP_THREADS,
        "batch_policy": BGE_EMBED_BATCH_POLICY,
        "fallback_reason": AUTOTUNE_RUNTIME_PROFILE.fallback_reason,
        "profile_path": AUTOTUNE_RUNTIME_PROFILE.profile_path,
        "selected": AUTOTUNE_SELECTED,
    }


AUTOTUNE_EFFECTIVE_INFO = _autotune_observability()

# Bound rerank admission so executor lanes cannot accumulate an HTTP queue.
rerank_slots = asyncio.Semaphore(RERANK_MAX_QUEUE)

server_info.info(
    {
        "model": BGE_EMBEDDING_MODEL,
        "dense_embedding_model": embedding_service.dense_model_name,
        "profile": embedding_service.profile,
        "sparse_model_name": embedding_service.sparse_model_name,
        "colbert_model_name": embedding_service.colbert_model_name or "none",
        "reranker_model": reranker.model_name,
        "bge_embed_max_length": str(BGE_EMBED_MAX_LENGTH),
        "qwen_embed_max_length": str(QWEN_EMBED_MAX_LENGTH),
        "bge_rerank_max_length": str(BGE_RERANK_MAX_LENGTH),
        "qwen_rerank_max_length": str(QWEN_RERANK_MAX_LENGTH),
        "length_aware_batching": str(EMBED_LENGTH_AWARE_BATCHING),
        "bge_embed_batch_size": str(batch_size),
        "bge_embed_runtime": BGE_EFFECTIVE_RUNTIME,
        "bge_torch_compile": str(_bge_torch_compile_active()),
        "version": APP_VERSION,
        "gpu_available": str(has_cuda),
        "device": device,
    }
)
advanced_config_info.info(
    {key: str(value) for key, value in ADVANCED_EFFECTIVE_CONFIG.items()}
)
autotune_info.info(
    {
        "state": str(AUTOTUNE_EFFECTIVE_INFO["state"]),
        "mode": str(AUTOTUNE_EFFECTIVE_INFO["mode"]),
        "objective": str(AUTOTUNE_EFFECTIVE_INFO["objective"]),
        "fingerprint_sha256": str(AUTOTUNE_EFFECTIVE_INFO["fingerprint_sha256"]),
        "runtime": str(AUTOTUNE_EFFECTIVE_INFO["runtime"]),
        "torch_compile": str(AUTOTUNE_EFFECTIVE_INFO["torch_compile"]),
        "torch_compile_policy": str(AUTOTUNE_EFFECTIVE_INFO["torch_compile_policy"]),
        "torch_compile_mode": str(AUTOTUNE_EFFECTIVE_INFO["torch_compile_mode"]),
        "torch_compile_reason": str(AUTOTUNE_EFFECTIVE_INFO["torch_compile_reason"]),
        "precision": str(AUTOTUNE_EFFECTIVE_INFO["precision"]),
        "threads": str(AUTOTUNE_EFFECTIVE_INFO["threads"] or "n/a"),
        "inference_concurrency": str(AUTOTUNE_EFFECTIVE_INFO["inference_concurrency"]),
        "onnx_execution_mode": str(AUTOTUNE_EFFECTIVE_INFO["onnx_execution_mode"]),
        "onnx_spin_policy": str(AUTOTUNE_EFFECTIVE_INFO["onnx_spin_policy"]),
        "onnx_inter_op_threads": str(AUTOTUNE_EFFECTIVE_INFO["onnx_inter_op_threads"]),
        "batch_policy": orjson.dumps(AUTOTUNE_EFFECTIVE_INFO["batch_policy"]).decode(
            "utf-8"
        ),
    }
)


# Enforce the Bearer token everywhere except explicit public endpoints. The
# route dependency mirrors this policy in generated OpenAPI.
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Enforce Bearer authentication on every non-public path."""
    if not API_TOKENS or is_public_auth_path(request.url.path):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    scheme, _, token = auth_header.partition(" ")

    label = None
    if scheme.lower() == "bearer":
        label = match_bearer_token(token, API_TOKENS)
    if label is None:
        requests_rejected.labels(reason="auth").inc()
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing bearer token."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    logging.info("request authenticated: %s", label)

    return await call_next(request)


class BodySizeLimitMiddleware:
    """Reject HTTP requests once streamed body bytes exceed ``max_bytes``.

    A pure ASGI middleware rather than an ``@app.middleware("http")`` one,
    because the limit has to wrap ``receive`` and count bytes as they arrive.
    A BaseHTTPMiddleware sees the request only once the body is already in
    memory, which is the thing being prevented.

    Deliberately a copy of the middleware service's
    ``BodySizeLimitMiddleware`` rather than an import of it: services in this
    repository communicate over HTTP contracts and never share runtime
    imports. The MCP server's retry policy is duplicated for the same reason.
    """

    def __init__(self, app, max_bytes: int) -> None:
        """Wrap ``app`` and reject bodies larger than ``max_bytes``."""
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        """Enforce the body size limit on HTTP scopes, passing others through."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        total = 0
        rejected = False

        async def limited_receive():
            nonlocal total, rejected
            message = await receive()
            if message["type"] != "http.request" or rejected:
                return message
            total += len(message.get("body", b""))
            if total <= self.max_bytes:
                return message
            rejected = True
            # Attribute the rejection to the same counter the endpoint would
            # have used, so /stats and Prometheus agree with each other and
            # with the rate limiter's own accounting.
            if str(scope.get("path", "")).startswith("/rerank"):
                rerank_requests_rejected.labels(reason="request_body_too_large").inc()
            else:
                requests_rejected.labels(reason="request_body_too_large").inc()
            stats.increment_rejected()
            return {"type": "http.disconnect"}

        async def limited_send(message) -> None:
            if not rejected:
                await send(message)
                return
            if message["type"] != "http.response.start":
                return
            body = orjson.dumps(
                {"detail": (f"Request body too large. Max {self.max_bytes} bytes.")}
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})

        await self.app(scope, limited_receive, limited_send)


# Registered here on purpose. add_middleware inserts at position 0, so the
# *last* registration is the outermost layer and the effective order is
# timeout -> shutdown -> rate limit -> body size -> auth -> route. That mirrors
# the middleware service: an oversized body still costs a rate-limit token, so
# a flood of them is throttled rather than merely refused, and it is rejected
# before any token is checked, so no unauthenticated caller can make this
# process parse gigabytes.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=REQUEST_BODY_MAX_BYTES)


# Registered after the auth middleware above, which makes this the *outer* of
# the two: the token bucket is applied before authentication, not after, so
# unauthenticated floods are throttled too.
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Reject clients whose per-IP token bucket is exhausted."""
    # Core health and scraping endpoints must remain reliable under load.
    if request.url.path in ["/health", "/stats", "/metrics"]:
        return await call_next(request)

    # Use the direct connection IP for rate limiting.
    # X-Forwarded-For is ignored because clients can spoof it to bypass limits.
    # If this server runs behind a trusted reverse proxy, replace this with the
    # verified first IP from X-Forwarded-For after validating the proxy address.
    client_ip = request.client.host if request.client else "unknown"

    allowed = await rate_limiter.check_rate_limit(client_ip)

    if not allowed:
        # Attribute rejection to the endpoint-specific counter so dashboards
        # keep embedding and reranking saturation distinct.
        if request.url.path.startswith("/rerank"):
            rerank_requests_rejected.labels(reason="rate_limit").inc()
        else:
            requests_rejected.labels(reason="rate_limit").inc()
        # Keep /stats aligned with Prometheus for either monitoring path.
        stats.increment_rejected()
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    f"Rate limit exceeded. Max "
                    f"{RATE_LIMIT_REQUESTS_PER_MINUTE} requests per minute."
                ),
                "retry_after": 60,
            },
            headers={"Retry-After": "60"},
        )

    return await call_next(request)


# Reject new work after graceful shutdown begins.
@app.middleware("http")
async def shutdown_middleware(request: Request, call_next):
    """Block new requests while the service drains in-flight work."""
    if processor.is_shutting_down:
        return JSONResponse(
            status_code=503, content={"detail": "Server is shutting down"}
        )
    return await call_next(request)


# Bound the complete HTTP lifecycle independently from model deadlines.
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Return HTTP 504 when the complete request exceeds its deadline."""
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"detail": "Request timed out."},
            status_code=HTTP_504_GATEWAY_TIMEOUT,
        )


# --- 7. API Endpoints ---


@app.get("/health", status_code=200)
async def health_check():
    """Return service readiness, active models, limits, and tuning state.

    Returns:
        Health metadata for the running process and its selected profile.
    """
    gpu_status = {
        "available": has_cuda,
        "device_info": torch.cuda.get_device_name(0) if has_cuda else "CPU",
        "memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
        if has_cuda
        else 0,
        "device_count": available_gpus if has_cuda else 0,
        "multi_gpu_enabled": MULTI_GPU_DEVICES is not None,
        "configured_devices": MULTI_GPU_DEVICES if MULTI_GPU_DEVICES else [device],
    }

    return {
        "status": "healthy",
        "version": APP_VERSION,
        "gpu": gpu_status,
        "model": embedding_service.bge_model_name,
        "dense_embedding_model": embedding_service.dense_model_name,
        "reranker_model": reranker.model_name,
        "profile": embedding_service.profile,
        "sparse_model_name": embedding_service.sparse_model_name,
        "colbert_model_name": embedding_service.colbert_model_name,
        "max_input_length": BGE_EMBED_MAX_LENGTH,
        "bge_embed_max_length": BGE_EMBED_MAX_LENGTH,
        "qwen_embed_max_length": QWEN_EMBED_MAX_LENGTH,
        "bge_rerank_max_length": BGE_RERANK_MAX_LENGTH,
        "qwen_rerank_max_length": QWEN_RERANK_MAX_LENGTH,
        "reranker_content_budget": ADVANCED_EFFECTIVE_CONFIG["reranker_content_budget"],
        "reranker_fixed_overhead_tokens": ADVANCED_EFFECTIVE_CONFIG[
            "reranker_fixed_overhead_tokens"
        ],
        "reranker_prefix_tokens": ADVANCED_EFFECTIVE_CONFIG["reranker_prefix_tokens"],
        "reranker_suffix_tokens": ADVANCED_EFFECTIVE_CONFIG["reranker_suffix_tokens"],
        "reranker_instruction": ADVANCED_EFFECTIVE_CONFIG["reranker_instruction"],
        "milco_source_view": MILCO_SOURCE_VIEW,
        "milco_top_k": MILCO_TOP_K,
        "milco_max_length": MILCO_MAX_LENGTH,
        "milco_batch_size": MILCO_BATCH_SIZE,
        "milco_max_batch_tokens": MILCO_MAX_BATCH_TOKENS,
        "milco_projection_chunk_tokens": MILCO_PROJECTION_CHUNK_TOKENS,
        "milco_english_vocab_size": ADVANCED_EFFECTIVE_CONFIG[
            "sparse_english_vocab_size"
        ],
        "length_aware_batching": EMBED_LENGTH_AWARE_BATCHING,
        "bge_embed_batch_size": batch_size,
        "batch_size": batch_size,
        "max_requests_in_batch": MAX_REQUESTS_IN_BATCH,
        "advanced_config": ADVANCED_EFFECTIVE_CONFIG,
        "autotune": AUTOTUNE_EFFECTIVE_INFO,
    }


@app.get("/models", response_model=ModelsResponse, status_code=200)
async def get_models() -> ModelsResponse:
    """Return the active profile and loaded model names."""
    contract = build_embedding_contract(
        embedding_service.profile,
        embedding_service.dense_model_name,
        embedding_service.sparse_model_name,
    )
    return ModelsResponse(
        profile=embedding_service.profile,
        embedder=EmbedderModelsInfo(
            dense=embedding_service.dense_model_name,
            sparse=embedding_service.sparse_model_name,
            colbert=embedding_service.colbert_model_name,
        ),
        reranker=reranker.model_name,
        version=APP_VERSION,
        embedding_contract=contract,
    )


@app.get("/stats", status_code=200)
async def server_stats():
    """Return process-lifetime request, batching, rejection, and hardware stats.

    Returns:
        The current server statistics snapshot.
    """
    return stats.get_stats()


@app.get("/metrics")
async def metrics():
    """Return the current Prometheus exposition payload."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/embeddings/",
    response_model=EmbeddingsListResponse,
    response_class=EmbeddingORJSONResponse,
    status_code=200,
    dependencies=[Depends(require_bearer_token)],
    description=(
        f"Generate embeddings from input texts.\n\n"
        f"Active backends (selected at server startup):\n"
        f"- Dense vectors: `{embedding_service.dense_model_name}`\n"
        f"- Sparse lexical weights: `{embedding_service.sparse_model_name}`\n"
        f"- ColBERT vectors: `{embedding_service.colbert_model_name}`\n\n"
        "The active request schema follows the selected profile. With "
        "`advanced`, ColBERT vectors are disabled and `return_colbert` is not "
        "a valid request field.\n\n"
        "Request options control which embedding types are computed and "
        "returned. Common options are `return_dense`, `return_sparse`, "
        "`normalize_dense`, and `sparse_as_indices`; `basic` also supports "
        "`return_colbert`.\n\n"
        "Response metadata fields `model_name`, `dense_model_name`, "
        "`sparse_model_name`, and `colbert_model_name` report the source "
        "of each embedding type.\n\n"
        "Errors: HTTP 400 if `sentences` is empty; HTTP 503 on queue "
        "backpressure; HTTP 504 on GPU timeout; HTTP 500 on internal errors."
    ),
)
async def get_embeddings(request: ActiveEmbedRequest):
    """Generate embeddings from input texts.

    Main endpoint that processes a list of texts and returns their embeddings.
    Supports configurable options for which embedding types to compute and return.

    Args:
        request: The embedding request containing texts and options

    Returns:
        EmbeddingsListResponse: Structured response with embeddings and metadata

    Raises:
        HTTPException: For validation errors, timeouts, or internal errors
    """
    with request_duration.labels(endpoint="/embeddings").time():
        if not request.sentences:
            requests_total.labels(status="error", endpoint="/embeddings").inc()
            raise HTTPException(
                status_code=400, detail="The 'sentences' list cannot be empty."
            )

        _enforce_request_limits(
            request.sentences, "sentences", MAX_SENTENCES_PER_REQUEST
        )

        with _reserve_inference(_embedding_request_cost(request)) as lease:
            try:
                start_time = time.time()
                embedding_data = await processor.process_request(request, lease=lease)
                response_warnings = embedding_data.get("_warnings", [])

                end_time = time.time()
                processing_time_ms = (end_time - start_time) * 1000

                response_build_started_at = time.monotonic()

                response_data = []
                for i, sentence in enumerate(request.sentences):
                    embedding_kwargs = {}

                    if request.return_dense and "dense_vecs" in embedding_data:
                        embedding_kwargs["dense"] = embedding_data["dense_vecs"][i]

                    if request.return_sparse and "lexical_weights" in embedding_data:
                        lw = embedding_data["lexical_weights"][i]
                        embedding_kwargs["sparse"] = (
                            lexical_weights_to_indices(lw)
                            if request.sparse_as_indices
                            else lw
                        )

                    if (
                        _request_return_colbert(request)
                        and "colbert_vecs" in embedding_data
                    ):
                        embedding_kwargs["colbert"] = embedding_data["colbert_vecs"][i]

                    vectors = EmbeddingVectors(**embedding_kwargs)

                    response_data.append(
                        SingleEmbeddingResponse(
                            id=i,
                            text=sentence if request.include_text else None,
                            embeddings=vectors,
                        )
                    )

                requests_total.labels(status="success", endpoint="/embeddings").inc()
                response = EmbeddingsListResponse(
                    data=response_data,
                    model_name=embedding_service.dense_model_name,
                    dense_model_name=embedding_service.dense_model_name,
                    sparse_model_name=embedding_service.sparse_model_name,
                    colbert_model_name=embedding_service.colbert_model_name,
                    embedding_contract_fingerprint=build_embedding_contract(
                        embedding_service.profile,
                        embedding_service.dense_model_name,
                        embedding_service.sparse_model_name,
                    )["fingerprint"],
                    processing_time_ms=round(processing_time_ms, 2),
                    warnings=response_warnings,
                )
                api_response_build_duration.labels(endpoint="/embeddings").observe(
                    time.monotonic() - response_build_started_at
                )
                return response

            except (asyncio.TimeoutError, InferenceTimeoutError):
                requests_total.labels(status="timeout", endpoint="/embeddings").inc()
                raise HTTPException(
                    status_code=504, detail="Timeout during GPU processing."
                )
            except HTTPException:
                # Preserve deliberate API status codes, including backpressure.
                raise
            except Exception as e:
                logging.error(
                    f"Unexpected error in /embeddings/ endpoint: {e}", exc_info=True
                )
                requests_total.labels(status="error", endpoint="/embeddings").inc()
                raise HTTPException(status_code=500, detail="Internal server error.")


@app.post(
    "/rerank",
    response_model=RerankResponse,
    response_class=RerankORJSONResponse,
    status_code=200,
    dependencies=[Depends(require_bearer_token)],
    description=(
        f"Score and rank passages by relevance to a query using "
        f"`{reranker.model_name}`.\n\n"
        "Accepts a query string and a list of candidate passages. "
        "Returns all passages sorted by descending relevance score.\n\n"
        "- BGE backend: `normalize=true` applies sigmoid to map raw logits "
        "into `[0, 1]`; `normalize=false` returns the raw logit "
        "(negative values possible).\n"
        "- Qwen backend: scores are yes-probabilities in `[0, 1]`; "
        "`normalize` is kept as an API-compatible no-op.\n\n"
        "Errors: HTTP 400 if query or passages list is empty; HTTP 413 if a "
        "single query/passage pair cannot fit in GPU memory; HTTP 503 on "
        "rerank queue backpressure or transient GPU memory pressure; HTTP 504 "
        "on GPU timeout; HTTP 500 on internal model errors."
    ),
)
async def rerank(request: RerankRequest):
    """Score and rank passages using the reranker selected at startup."""
    with rerank_request_duration.time():
        if not request.query.strip():
            rerank_requests_total.labels(status="error").inc()
            raise HTTPException(
                status_code=400, detail="The 'query' field cannot be empty."
            )

        if not request.passages:
            rerank_requests_total.labels(status="error").inc()
            raise HTTPException(
                status_code=400, detail="The 'passages' list cannot be empty."
            )

        _enforce_request_limits([request.query], "query", 1)
        _enforce_request_limits(request.passages, "passages", MAX_PASSAGES_PER_REQUEST)

        cost = estimate_rerank(
            request.query,
            request.passages,
            pair_limit=QWEN_RERANK_MAX_LENGTH
            if reranker.model_name == QWEN_RERANKER_MODEL
            else BGE_RERANK_MAX_LENGTH,
        )
        with _reserve_inference(cost, reranking=True) as lease:
            # A short admission timeout prevents executor lanes from accumulating
            # an unbounded HTTP queue and matches /embeddings overload handling.
            try:
                await asyncio.wait_for(rerank_slots.acquire(), timeout=0.5)
            except asyncio.TimeoutError:
                rerank_requests_rejected.labels(reason="backpressure").inc()
                stats.increment_rejected()
                rerank_requests_total.labels(status="error").inc()
                raise HTTPException(
                    status_code=503, detail="Server is overloaded. Please retry later."
                )

            rerank_active_requests.inc()
            instance_lock = reranker_instance_lock
            reranker_lock_acquired = False
            worker_submitted = False
            try:
                start_time = time.time()
                deadline = make_inference_deadline(RERANK_GPU_TIMEOUT)

                # The selected reranker owns mutable tokenizer/model state. Keep
                # its preparation and scoring serial even when an advanced profile
                # has multiple executor lanes, while leaving another lane free for
                # embedding work.
                try:
                    await asyncio.wait_for(
                        instance_lock.acquire(), timeout=deadline.remaining()
                    )
                    reranker_lock_acquired = True
                except asyncio.TimeoutError:
                    rerank_requests_total.labels(status="timeout").inc()
                    raise HTTPException(
                        status_code=504,
                        detail="Timeout during rerank GPU processing.",
                    )

                # Preparation and scoring stay on one executor worker. The instance
                # lock is held until that worker completes, even if its caller times
                # out, so a native call cannot overlap a later rerank request.
                def _prepare_and_score():
                    worker_started_at = time.monotonic()
                    deadline.check()
                    prepare_started_at = time.monotonic()
                    pairs, warnings = reranker.prepare_pairs_for_scoring(
                        request.query,
                        request.passages,
                    )
                    if reranker.model_name == QWEN_RERANKER_MODEL:
                        observe_advanced_phase(
                            "qwen_reranker", "prepare", prepare_started_at
                        )
                    deadline.check()
                    scores = reranker.score(pairs, request.normalize, deadline=deadline)
                    worker_finished_at = time.monotonic()
                    return (
                        pairs,
                        warnings,
                        scores,
                        worker_started_at,
                        worker_finished_at,
                    )

                submitted_at = time.monotonic()
                try:
                    native = reranker.executor.submit(_prepare_and_score)
                    lease.track(native)
                    worker = asyncio.wrap_future(native)
                    worker_submitted = True

                    def _release_reranker_lock(completed):
                        """Release the instance and consume abandoned failures."""
                        instance_lock.release()
                        if not completed.cancelled():
                            completed.exception()

                    worker.add_done_callback(_release_reranker_lock)
                    (
                        pairs,
                        response_warnings,
                        scores,
                        worker_started_at,
                        worker_finished_at,
                    ) = await asyncio.wait_for(
                        asyncio.shield(worker),
                        timeout=deadline.remaining(),
                    )
                except (asyncio.TimeoutError, InferenceTimeoutError):
                    rerank_requests_total.labels(status="timeout").inc()
                    raise HTTPException(
                        status_code=504,
                        detail="Timeout during rerank GPU processing.",
                    )

                rerank_pairs_processed.inc(len(pairs))
                api_executor_wait_duration.labels(
                    endpoint="/rerank", operation="rerank"
                ).observe(worker_started_at - submitted_at)
                api_model_forward_duration.labels(
                    endpoint="/rerank", operation="rerank"
                ).observe(worker_finished_at - worker_started_at)
                # Retain the legacy aggregate for backward-compatible dashboards.
                rerank_inference_duration.observe(
                    worker_finished_at - worker_started_at
                )
                if reranker.model_name == QWEN_RERANKER_MODEL:
                    advanced_backend_phase_duration.labels(
                        backend="qwen_reranker", phase="pipeline"
                    ).observe(worker_finished_at - worker_started_at)
                record_gpu_memory_metrics()

                response_build_started_at = time.monotonic()

                # Build response sorted by score descending
                ranked = sorted(
                    [
                        RankedPassage(index=i, passage=pairs[i][1], score=scores[i])
                        for i in range(len(pairs))
                    ],
                    key=lambda r: r.score,
                    reverse=True,
                )

                processing_time_ms = (time.time() - start_time) * 1000
                rerank_requests_total.labels(status="success").inc()
                response = RerankResponse(
                    results=ranked,
                    model_name=reranker.model_name,
                    processing_time_ms=round(processing_time_ms, 2),
                    warnings=response_warnings,
                )
                api_response_build_duration.labels(endpoint="/rerank").observe(
                    time.monotonic() - response_build_started_at
                )
                return response

            except vram_budget.VramExhausted as failure:
                rerank_requests_total.labels(status="error").inc()
                logging.warning(
                    "Rerank rejected for capacity: items=%s tokens=%s bytes=%s",
                    failure.batch_items,
                    failure.batch_tokens,
                    failure.estimated_bytes,
                )
                raise _rerank_capacity_response(failure, len(request.passages))
            except HTTPException:
                raise
            except Exception as e:
                logging.error(
                    f"Unexpected error in /rerank endpoint: {e}", exc_info=True
                )
                rerank_requests_total.labels(status="error").inc()
                raise HTTPException(status_code=500, detail="Internal server error.")
            finally:
                if reranker_lock_acquired and not worker_submitted:
                    instance_lock.release()
                rerank_active_requests.dec()
                rerank_slots.release()
