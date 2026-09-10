"""Isolated co-resident probe used by the ADVANCED calibration runner.

``run_candidate()`` loads the serving stack once and returns quality,
performance, and memory evidence for one serialized candidate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import secrets
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from advanced_autotune_quality import DOCUMENTS, QUALITY_SUITE_VERSION, QUERIES


def _apply_candidate_environment(candidate: dict) -> None:
    selected = candidate.get("selected", {}).get("advanced", {})
    if not isinstance(selected, dict):
        return
    mappings = {
        "dense": {
            "runtime": "QWEN_EMBED_RUNTIME",
            "precision": "QWEN_EMBED_PRECISION",
            "attention": "QWEN_EMBED_ATTENTION",
            "torch_compile": "QWEN_EMBED_TORCH_COMPILE",
            "torch_compile_mode": "QWEN_EMBED_TORCH_COMPILE_MODE",
            "pinned_memory": "QWEN_EMBED_PINNED_MEMORY",
            "prefetch_batches": "QWEN_EMBED_PREFETCH_BATCHES",
            "batch_policy": "QWEN_EMBED_BATCH_POLICY",
            "max_batch_tokens_policy": "QWEN_EMBED_MAX_BATCH_TOKENS_POLICY",
        },
        "sparse": {
            "runtime": "MILCO_RUNTIME",
            "precision": "MILCO_PRECISION",
            "compute_precision": "MILCO_COMPUTE_PRECISION",
            "torch_compile": "MILCO_TORCH_COMPILE",
            "torch_compile_mode": "MILCO_TORCH_COMPILE_MODE",
            "pinned_memory": "MILCO_PINNED_MEMORY",
            "prefetch_batches": "MILCO_PREFETCH_BATCHES",
            "batch_policy": "MILCO_BATCH_POLICY",
            "max_batch_tokens_policy": "MILCO_MAX_BATCH_TOKENS_POLICY",
        },
        "reranker": {
            "runtime": "QWEN_RERANK_RUNTIME",
            "precision": "QWEN_RERANK_PRECISION",
            "attention": "QWEN_RERANK_ATTENTION",
            "torch_compile": "QWEN_RERANK_TORCH_COMPILE",
            "torch_compile_mode": "QWEN_RERANK_TORCH_COMPILE_MODE",
            "batch_policy": "QWEN_RERANK_BATCH_POLICY",
            "max_batch_tokens_policy": "QWEN_RERANK_MAX_BATCH_TOKENS_POLICY",
        },
        "schedule": {
            "mode": "ADVANCED_SCHEDULE_MODE",
            "inference_lanes": "ADVANCED_INFERENCE_LANES",
            "queue_batch": "ADVANCED_QUEUE_BATCH",
            "flush_timeout_ms": "ADVANCED_FLUSH_TIMEOUT_MS",
            "worker_spin": "ADVANCED_CPU_SPIN_POLICY",
            "affinity": "ADVANCED_CPU_AFFINITY",
            "allow_oversubscription": "ADVANCED_CPU_ALLOW_OVERSUBSCRIPTION",
            "intra_op_threads": "EMBEDDER_NUM_THREADS",
            "inter_op_threads": "EMBEDDER_INTEROP_THREADS",
        },
    }
    for section, fields in mappings.items():
        values = selected.get(section, {})
        if not isinstance(values, dict):
            continue
        for key, env_name in fields.items():
            if key not in values:
                continue
            value = values[key]
            if isinstance(value, (dict, list)):
                os.environ[env_name] = json.dumps(value, separators=(",", ":"))
            elif isinstance(value, bool):
                os.environ[env_name] = "true" if value else "false"
            else:
                os.environ[env_name] = str(value)
    schedule = selected.get("schedule", {})
    spin = schedule.get("worker_spin") if isinstance(schedule, dict) else None
    if spin == "passive":
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
        os.environ["KMP_BLOCKTIME"] = "0"
    elif spin == "active":
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
        os.environ["KMP_BLOCKTIME"] = "200"


def _benchmark_batch_size(candidate: dict, backend: str, length: int) -> int:
    batches = candidate.get("benchmark_batches", {})
    backend_batches = batches.get(backend, {}) if isinstance(batches, dict) else {}
    value = backend_batches.get(str(length), 1)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _run_mixed(server: Any, texts: list[str]) -> tuple[Any, Any]:
    """Measure embedding alongside a full preparation-and-scoring operation."""

    def rerank():
        pairs, _warnings = server.reranker.prepare_pairs_for_scoring(
            "multilingual retrieval", texts
        )
        return server.reranker.score(pairs, True)

    embedding_future = server.GPU_EXECUTOR.submit(
        server.embedding_service.embed,
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert=False,
        normalize_dense=True,
    )
    reranker_future = server.GPU_EXECUTOR.submit(rerank)
    return embedding_future.result(), reranker_future.result()


def _capture_quality(server: Any) -> dict:
    items = [*QUERIES, *DOCUMENTS]
    ids = [item.id for item in items]
    texts = [item.text for item in items]
    embedding = server.embedding_service.embed(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert=False,
        normalize_dense=True,
    )
    dense_values = embedding["dense_vecs"]
    if hasattr(dense_values, "tolist"):
        dense_values = dense_values.tolist()
    dense = dict(zip(ids, dense_values))
    sparse = dict(zip(ids, embedding["lexical_weights"]))
    reranker = {}
    for query in QUERIES:
        pairs, _warnings = server.reranker.prepare_pairs_for_scoring(
            query.text, [document.text for document in DOCUMENTS]
        )
        scores = server.reranker.score(pairs, normalize=True)
        reranker[query.id] = {
            document.id: float(score) for document, score in zip(DOCUMENTS, scores)
        }
    milco = server.embedding_service.milco_sparse_backend
    return {
        "suite_version": QUALITY_SUITE_VERSION,
        "dense": dense,
        "sparse": sparse,
        "reranker": reranker,
        "milco_source_view": bool(server.MILCO_SOURCE_VIEW),
        "milco_top_k": int(server.MILCO_TOP_K),
        "source_vocab_start": int(getattr(milco.model.config, "en_vocab_size", 0)),
    }


def _rss_peak_bytes() -> int | None:
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeError):
        return None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def _load_server(script_dir: Path):
    path = script_dir / "smart-embedder.py"
    spec = importlib.util.spec_from_file_location("advanced_probe_server", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import serving runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_tokens = os.environ.get("API_TOKENS")
    private_auth = not (previous_tokens or os.environ.get("SERVICE_TOKEN"))
    # Prep containers intentionally have no serving credentials. This offline
    # worker imports model wrappers, never starts an HTTP listener, and still
    # satisfies the serving module's strict authentication invariant.
    if private_auth:
        os.environ["API_TOKENS"] = f"{secrets.token_urlsafe(32)}:autotune"
    try:
        spec.loader.exec_module(module)
    finally:
        if private_auth:
            if previous_tokens is None:
                os.environ.pop("API_TOKENS", None)
            else:
                os.environ["API_TOKENS"] = previous_tokens
    return module


def _repeatable_text(tokenizer: Any, target_tokens: int) -> str:
    """Construct deterministic text close to an exact tokenizer length."""
    candidates = (
        " retrieval",
        " document",
        " multilingual",
        " information",
        " test",
    )
    unit = min(
        candidates,
        key=lambda value: abs(
            len(tokenizer.encode(value, add_special_tokens=False)) - 1
        ),
    )
    unit_ids = tokenizer.encode(unit, add_special_tokens=False)
    if not unit_ids:
        unit_ids = [int(getattr(tokenizer, "unk_token_id", 0) or 0)]
    repeated = (unit_ids * (target_tokens // len(unit_ids) + 2))[:target_tokens]
    text = tokenizer.decode(repeated, skip_special_tokens=True)
    # Decode/encode is not always length preserving. Add or trim using token IDs.
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) != target_tokens:
        repeated = (encoded + unit_ids * (target_tokens + 1))[:target_tokens]
        text = tokenizer.decode(repeated, skip_special_tokens=True)
    return text


def _cuda_memory(torch_module: Any) -> dict:
    if not torch_module.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "free_bytes": None,
            "total_bytes": None,
            "peak_fraction": 0.0,
        }
    torch_module.cuda.synchronize()
    free, total = torch_module.cuda.mem_get_info()
    peak = int(torch_module.cuda.max_memory_allocated())
    return {
        "allocated_bytes": int(torch_module.cuda.memory_allocated()),
        "reserved_bytes": int(torch_module.cuda.memory_reserved()),
        "peak_allocated_bytes": peak,
        "free_bytes": int(free),
        "total_bytes": int(total),
        "peak_fraction": peak / int(total) if total else 0.0,
    }


def _validate_dense(values: Any, expected_rows: int) -> list[str]:
    errors = []
    shape = getattr(values, "shape", ())
    if tuple(shape) != (expected_rows, 1024):
        errors.append(f"dense shape {tuple(shape)} != {(expected_rows, 1024)}")
    try:
        finite = bool(values.size) and bool((abs(values) < float("inf")).all())
    except (AttributeError, TypeError, ValueError):
        finite = False
    if not finite:
        errors.append("dense contains NaN/Inf or is empty")
    return errors


def _validate_sparse(
    rows: Any,
    *,
    expected_rows: int,
    source_view: bool,
    english_vocab_size: int,
) -> list[str]:
    errors = []
    if not isinstance(rows, list) or len(rows) != expected_rows:
        return ["sparse row count mismatch"]
    saw_source = False
    for row in rows:
        if not isinstance(row, dict) or not row:
            errors.append("sparse row is empty or not a mapping")
            continue
        for raw_index, raw_weight in row.items():
            try:
                index = int(raw_index)
                weight = float(raw_weight)
            except (TypeError, ValueError):
                errors.append("sparse entry is not numeric")
                continue
            if index < 0 or weight <= 0 or not math.isfinite(weight):
                errors.append("sparse entry is outside Qdrant invariants")
            saw_source = saw_source or index >= english_vocab_size > 0
    if source_view and english_vocab_size > 0 and not saw_source:
        errors.append("LexEcho source_view produced no source-vocabulary index")
    return errors


def _validate_scores(scores: Any, expected: int) -> list[str]:
    if not isinstance(scores, list) or len(scores) != expected:
        return ["reranker score count mismatch"]
    if any(
        not math.isfinite(float(score)) or not 0 <= float(score) <= 1
        for score in scores
    ):
        return ["reranker score is non-finite or outside [0, 1]"]
    return []


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(float(sample) for sample in samples)
    if not ordered:
        raise ValueError("at least one latency sample is required")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * min(1.0, max(0.0, quantile))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_summary(samples: list[float]) -> dict[str, float]:
    mean_elapsed = statistics.fmean(samples)
    variance = statistics.fmean((sample - mean_elapsed) ** 2 for sample in samples)
    return {
        "p50": statistics.median(samples),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
        "coefficient_of_variation": (
            math.sqrt(variance) / mean_elapsed if mean_elapsed else 0.0
        ),
    }


def run_candidate(candidate: dict, *, script_dir: Path) -> dict:
    """Measure one ADVANCED candidate with all production models resident.

    Args:
        candidate: Serialized policy, workloads, and target backend selection.
        script_dir: Directory containing ``smart-embedder.py``.

    Returns:
        Quality, latency, throughput, and memory evidence for the runner.
    """
    os.environ["EMBEDDER_PROFILE"] = "advanced"
    # Calibration itself owns process isolation; prevent nested pre-start work.
    os.environ["SMART_EMBEDDER_AUTOTUNE"] = "off"
    _apply_candidate_environment(candidate)
    started = time.monotonic()
    server = _load_server(script_dir)
    torch = server.torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    after_load = _cuda_memory(torch)

    requested_lengths = tuple(
        int(value) for value in candidate.get("token_lengths", (128, 8192))
    )
    repeats = max(1, int(candidate.get("repeats", 1)))
    dense_tokenizer = server.embedding_service.qwen_dense_backend.tokenizer
    texts = {
        length: _repeatable_text(dense_tokenizer, length)
        for length in requested_lengths
    }
    measurements = []
    quality_errors: list[str] = []
    target_backends = set(
        candidate.get("target_backends", ("dense", "sparse", "hybrid", "reranker"))
    )
    canary_lengths = {int(value) for value in candidate.get("canary_token_lengths", ())}
    canary_backends = set(
        candidate.get("canary_backends", ("dense", "sparse", "hybrid", "reranker"))
    )
    backend_token_lengths = candidate.get("backend_token_lengths", {})

    def requested_at_length(backend: str, length: int) -> bool:
        configured = (
            backend_token_lengths.get(backend)
            if isinstance(backend_token_lengths, dict)
            else None
        )
        return configured is None or length in {int(value) for value in configured}

    for length, text in texts.items():
        for backend in ("dense", "sparse", "hybrid"):
            if backend not in target_backends or not requested_at_length(
                backend, length
            ):
                continue
            is_canary = length in canary_lengths
            if is_canary and backend not in canary_backends:
                continue
            batch_size = _benchmark_batch_size(candidate, backend, length)
            batch_texts = [text] * batch_size
            elapsed_samples = []
            print(
                f"probe-start backend={backend} tokens={length} "
                f"batch={batch_size} canary={is_canary}",
                file=sys.stderr,
                flush=True,
            )
            if not is_canary:
                server.embedding_service.embed(
                    batch_texts,
                    return_dense=backend != "sparse",
                    return_sparse=backend != "dense",
                    return_colbert=False,
                    normalize_dense=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            sample_count = 1 if is_canary else repeats
            for _ in range(sample_count):
                begin = time.perf_counter()
                output = server.embedding_service.embed(
                    batch_texts,
                    return_dense=backend != "sparse",
                    return_sparse=backend != "dense",
                    return_colbert=False,
                    normalize_dense=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_samples.append(time.perf_counter() - begin)
            latency = _latency_summary(elapsed_samples)
            elapsed = latency["p50"]
            print(
                f"probe-done backend={backend} tokens={length} p50={elapsed:.6f}",
                file=sys.stderr,
                flush=True,
            )
            measurements.append(
                {
                    "backend": backend,
                    "input_tokens": length,
                    "batch_size": batch_size,
                    "is_canary": is_canary,
                    "median_seconds": round(elapsed, 6),
                    "p50_seconds": round(elapsed, 6),
                    "p95_seconds": round(latency["p95"], 6),
                    "p99_seconds": round(latency["p99"], 6),
                    "coefficient_of_variation": round(
                        latency["coefficient_of_variation"],
                        6,
                    ),
                    "useful_tokens_per_second": round(length * batch_size / elapsed, 3),
                    "requests_per_second": round(1.0 / elapsed, 6),
                    "sentences_per_second": round(batch_size / elapsed, 6),
                }
            )
            if backend != "sparse":
                quality_errors.extend(
                    _validate_dense(output.get("dense_vecs"), batch_size)
                )
            if backend != "dense":
                milco = server.embedding_service.milco_sparse_backend
                english_vocab = int(getattr(milco.model.config, "en_vocab_size", 0))
                quality_errors.extend(
                    _validate_sparse(
                        output.get("lexical_weights"),
                        expected_rows=batch_size,
                        source_view=server.MILCO_SOURCE_VIEW,
                        english_vocab_size=english_vocab,
                    )
                )

        if (
            "reranker" in target_backends
            and requested_at_length("reranker", length)
            and (length not in canary_lengths or "reranker" in canary_backends)
        ):
            is_canary = length in canary_lengths
            rerank_batch_size = _benchmark_batch_size(candidate, "reranker", length)
            rerank_pair, _ = server.reranker.prepare_pairs_for_scoring(
                "multilingual retrieval", [text] * rerank_batch_size
            )
            elapsed_samples = []
            print(
                f"probe-start backend=reranker tokens={length} "
                f"batch={rerank_batch_size} canary={is_canary}",
                file=sys.stderr,
                flush=True,
            )
            if not is_canary:
                server.reranker.score(rerank_pair, normalize=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            sample_count = 1 if is_canary else repeats
            for _ in range(sample_count):
                begin = time.perf_counter()
                scores = server.reranker.score(rerank_pair, normalize=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_samples.append(time.perf_counter() - begin)
            latency = _latency_summary(elapsed_samples)
            elapsed = latency["p50"]
            print(
                f"probe-done backend=reranker tokens={length} p50={elapsed:.6f}",
                file=sys.stderr,
                flush=True,
            )
            measurements.append(
                {
                    "backend": "reranker",
                    "input_tokens": length,
                    "batch_size": rerank_batch_size,
                    "is_canary": is_canary,
                    "median_seconds": round(elapsed, 6),
                    "p50_seconds": round(elapsed, 6),
                    "p95_seconds": round(latency["p95"], 6),
                    "p99_seconds": round(latency["p99"], 6),
                    "coefficient_of_variation": round(
                        latency["coefficient_of_variation"],
                        6,
                    ),
                    "useful_tokens_per_second": round(
                        length * rerank_batch_size / elapsed, 3
                    ),
                    "requests_per_second": round(1.0 / elapsed, 6),
                    "pairs_per_second": round(rerank_batch_size / elapsed, 6),
                }
            )
            quality_errors.extend(_validate_scores(scores, rerank_batch_size))

        if "mixed" in target_backends and requested_at_length("mixed", length):
            is_canary = length in canary_lengths
            if is_canary and "mixed" not in canary_backends:
                continue
            mixed_batch_size = _benchmark_batch_size(candidate, "mixed", length)
            mixed_texts = [text] * mixed_batch_size
            elapsed_samples = []
            print(
                f"probe-start backend=mixed tokens={length} "
                f"batch={mixed_batch_size} canary={is_canary}",
                file=sys.stderr,
                flush=True,
            )
            if not is_canary:
                _run_mixed(server, mixed_texts)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            sample_count = 1 if is_canary else repeats
            for _ in range(sample_count):
                begin = time.perf_counter()
                mixed_output, mixed_scores = _run_mixed(server, mixed_texts)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_samples.append(time.perf_counter() - begin)
            latency = _latency_summary(elapsed_samples)
            elapsed = latency["p50"]
            print(
                f"probe-done backend=mixed tokens={length} p50={elapsed:.6f}",
                file=sys.stderr,
                flush=True,
            )
            measurements.append(
                {
                    "backend": "mixed",
                    "input_tokens": length,
                    "batch_size": mixed_batch_size,
                    "is_canary": is_canary,
                    "median_seconds": round(elapsed, 6),
                    "p50_seconds": round(elapsed, 6),
                    "p95_seconds": round(latency["p95"], 6),
                    "p99_seconds": round(latency["p99"], 6),
                    "coefficient_of_variation": round(
                        latency["coefficient_of_variation"], 6
                    ),
                    "useful_tokens_per_second": round(
                        2 * length * mixed_batch_size / elapsed, 3
                    ),
                    "requests_per_second": round(2.0 / elapsed, 6),
                    "sentences_per_second": round(mixed_batch_size / elapsed, 6),
                    "pairs_per_second": round(mixed_batch_size / elapsed, 6),
                }
            )
            quality_errors.extend(
                _validate_dense(mixed_output.get("dense_vecs"), mixed_batch_size)
            )
            milco = server.embedding_service.milco_sparse_backend
            english_vocab = int(getattr(milco.model.config, "en_vocab_size", 0))
            quality_errors.extend(
                _validate_sparse(
                    mixed_output.get("lexical_weights"),
                    expected_rows=mixed_batch_size,
                    source_view=server.MILCO_SOURCE_VIEW,
                    english_vocab_size=english_vocab,
                )
            )
            quality_errors.extend(_validate_scores(mixed_scores, mixed_batch_size))

    quality_capture = _capture_quality(server)

    memory = _cuda_memory(torch)
    memory["free_after_model_load_bytes"] = after_load["free_bytes"]
    memory["rss_peak_bytes"] = _rss_peak_bytes()
    if memory["total_bytes"] and memory["peak_fraction"] > 0.85:
        quality_errors.append("peak CUDA allocation exceeded 85% safety limit")
    return {
        "candidate": candidate.get("name", "unnamed"),
        "status": "passed" if not quality_errors else "rejected",
        "quality_passed": not quality_errors,
        "quality_errors": sorted(set(quality_errors)),
        "measurements": measurements,
        "quality_capture": quality_capture,
        "memory": memory,
        "duration_seconds": round(time.monotonic() - started, 6),
        "selected": candidate.get("selected", {}),
        "co_resident_models": [
            server.embedding_service.dense_model_name,
            server.embedding_service.sparse_model_name,
            server.reranker.model_name,
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the isolated worker command-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--script-dir", default=str(Path(__file__).resolve().parent))
    return parser


def main(argv=None) -> int:
    """Execute one serialized candidate and emit exactly one JSON result.

    Args:
        argv: Optional argument sequence. Uses process arguments when omitted.

    Returns:
        Zero for a passing candidate, two for worker errors, or three for a
        measured candidate rejected by its gates.
    """
    args = build_parser().parse_args(argv)
    candidate = {}
    try:
        candidate = json.loads(args.candidate_json)
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be a JSON object")
        payload = run_candidate(candidate, script_dir=Path(args.script_dir))
    except BaseException as exc:  # isolated boundary must report CUDA OOM, too
        payload = {
            "candidate": candidate.get("name", "unknown"),
            "status": "error",
            "quality_passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
    return 0 if payload["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
