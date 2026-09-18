"""Qdrant Middleware end-to-end benchmark.

Exercises every available endpoint. Creates an isolated test collection,
runs all scenarios, then deletes it. Refuses a pre-existing collection and
only seeds or deletes after creation of the requested collection is confirmed.

Endpoint coverage:
  COLLECTIONS  POST /collections, GET /collections, GET /collections/{name},
               POST /collections/verify,
               PATCH /collections/{name}/schema-toggles,
               POST/GET /collections/{name}/payload-indexes,
               GET /collections/{name}/payload-keys
               DELETE /collections/{name}/payload-indexes/{field}  ← teardown
               DELETE /collections/{name}  ← teardown (destructive)
  POINTS       POST /points/ingest, POST /points/ingest-chunks, POST /points
               (default and insert_only update_mode), POST /points/batch
               (default and insert_only update_mode), GET /points/{id}
               (default and with_vectors), POST /points/retrieve (default and
               with_vectors), POST /points/count, POST /points/facet,
               POST /points/scroll (default and ordered),
               PATCH /points/{id}/payload, PUT /points/{id}/payload,
               PATCH /points/payload (merge by filter)
               DELETE /points/{id}, DELETE /points  ← teardown (destructive)
  QUERY        POST /query/search — fusion/rerank matrix, filters,
               score_threshold, offset, and per-request search params
               (hnsw_ef, exact, rescore, oversampling, timeout)
  MONITORING   GET /livez, /readyz, /health, /stats, /metrics

Deliberately excluded (see comments at each scenario map for why):
  - The reindex family (preflight/start/pause/resume/cancel): a real vector
    regeneration is slow, stateful, and not repeatable N times concurrently —
    covered end-to-end instead by frontend/tests/z1-reembed.spec.js.
  - Payload delete-keys / clear (single point and by filter), and
    delete-by-filter: destructive/data-losing, and repeating them
    concurrently would corrupt the fixtures other scenarios in the same run
    depend on (bench_tag, the seeded payloads read back by get/retrieve).

Usage:
    python benchmark.py --url http://localhost:6336 --token <TOKEN>
    python benchmark.py --scenarios query --requests 50 --concurrency 8
    python benchmark.py --scenarios all --no-post-check
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import os
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

import httpx

# Load the unified root environment so the benchmark can reuse the middleware token.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────

BENCH_COLLECTION = "bench-middleware-test"
# File-based ingest: large doc for setup, small for benchmark repeats
BENCH_FILE_SETUP = "response_1778232857278.json"  # initial populate (setup only)
BENCH_FILE_REPEAT = "bench_chunks.json"  # repeatable benchmark ingest
BENCH_SOURCE_FILE = "bench-file"  # source_id for file ingest scenario
BENCH_SOURCE_INLINE = "bench-perf"  # source_id for inline ingest scenario
BENCH_POINT_ID = "cafecafe-cafe-cafe-cafe-cafecafecafe"
BENCH_BATCH_IDS = [
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "cccccccc-cccc-cccc-cccc-cccccccccccc",
]

INLINE_CHUNKS = [
    {
        "id": 0,
        "text": (
            "Electric vehicle adoption is accelerating globally, "
            "driven by tightening emissions regulations and "
            "declining battery costs."
        ),
    },
    {
        "id": 1,
        "text": (
            "Operating margins improved as manufacturing efficiency "
            "programs reduced per-unit costs across all vehicle segments."
        ),
    },
    {
        "id": 2,
        "text": (
            "Hybrid search combines dense semantic embeddings with "
            "sparse BM25-style keyword signals for superior "
            "retrieval accuracy."
        ),
    },
]

QUERY_RRF = "electric vehicle revenue growth strategy"
QUERY_DBSF = "board of directors governance capital"
QUERY_FILTER = "manufacturing operations supply chain"
QUERY_TECH = "hybrid search dense sparse vector retrieval reranking"

# Filterable values that actually exist in bench_chunks.json (source_id=bench-file),
# so filter scenarios match real points instead of silently returning 0 results.
FILTER_TYPE = "paragraph"
FILTER_HEADER1 = "Technology"
FILTER_HEADER1_ALT = "Financial Overview"
FILTER_TOKEN_COUNTS = [18, 19, 20, 21, 22]
FACET_FIELD = "bench_tag"  # set on every point upsert/batch scenario writes

BENCH_COLLECTION_OPTIONS = {
    "dense_on_disk": True,
    "quantization": True,
    "sparse_idf": True,
    "sparse_on_disk": True,
}

# Payload indexes the filter/facet/order_by scenarios below rely on. Nothing
# creates these automatically (PAYLOAD_INDEXES is intentionally empty — see
# services/collection_schema.py), so setup creates them explicitly instead of
# assuming they already exist on the target collection from a prior run.
BENCH_PAYLOAD_INDEXES = [
    {"field_name": "type", "field_schema": "keyword"},
    {"field_name": "Header 1", "field_schema": "keyword"},
    {"field_name": "token_count", "field_schema": "integer"},
    {"field_name": FACET_FIELD, "field_schema": "keyword"},
]

# ── Core types ─────────────────────────────────────────────────────────────────


@dataclass
class Sample:
    """One timed request outcome (status 0 means a client-side exception)."""

    latency_ms: float
    ok: bool
    status: int = 0
    error: str = ""


@dataclass
class ScenarioResult:
    """All samples collected for one named scenario run."""

    name: str
    samples: List[Sample] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def ok_latencies(self) -> List[float]:
        """Return latencies for successful samples only."""
        return [s.latency_ms for s in self.samples if s.ok]

    @property
    def success(self) -> int:
        """Return the number of successful samples."""
        return sum(1 for s in self.samples if s.ok)

    @property
    def failed(self) -> int:
        """Return the number of failed samples."""
        return len(self.samples) - self.success

    @property
    def ok_percent(self) -> float:
        """Return the successful-sample percentage."""
        return (self.success / len(self.samples) * 100.0) if self.samples else 0.0


# ── Rendering helpers ──────────────────────────────────────────────────────────


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile with ``p`` clamped to the value range."""
    if not values:
        return 0.0
    sv = sorted(values)
    k = max(0, min(len(sv) - 1, round(p / 100.0 * (len(sv) - 1))))
    return sv[k]


def fmt_ms(v: float) -> str:
    """Format milliseconds, switching to seconds at 1000 ms."""
    return f"{v:.1f}" if v < 1000 else f"{v / 1000:.2f}s"


def fmt_int(v: int | float) -> str:
    """Format an integer with spaces as thousands separators."""
    return f"{v:,}".replace(",", " ")


def section(title: str) -> str:
    """Render a title with an underline for console output."""
    return f"{title}\n{'-' * len(title)}"


def sample_status_label(sample: Sample) -> str:
    """Label a sample by HTTP status, or ``EXC:<error>`` for exceptions."""
    if sample.status:
        return str(sample.status)
    return f"EXC:{sample.error or 'client_error'}"


def status_counts(result: ScenarioResult) -> Dict[str, int]:
    """Count samples per status label, numeric statuses first."""
    counts = Counter(sample_status_label(s) for s in result.samples)

    def sort_key(item: tuple[str, int]) -> tuple[int, str]:
        key = item[0]
        return (0, f"{int(key):04d}") if key.isdigit() else (1, key)

    return dict(sorted(counts.items(), key=sort_key))


def format_status_counts(result: ScenarioResult) -> str:
    """Render status counts as ``status=count`` pairs for the summary table."""
    return ", ".join(f"{s}={c}" for s, c in status_counts(result).items()) or "-"


def render_table(rows: List[List[str]], headers: List[str]) -> str:
    """Render an ASCII table with column widths sized to the content."""
    cols = list(zip(*[headers, *rows]))
    widths = [max(len(str(c)) for c in col) for col in cols]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def row(cells: List[str]) -> str:
        padded = [f" {c:<{w}} " for c, w in zip(cells, widths)]
        return "|" + "|".join(padded) + "|"

    out = [sep, row(headers), sep] + [row([str(c) for c in r]) for r in rows] + [sep]
    return "\n".join(out)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

ScenarioFn = Callable[[httpx.AsyncClient, str], Awaitable[Sample]]
RequestFn = Callable[[httpx.AsyncClient, str], Awaitable[httpx.Response]]


def _make_sample(start: float, resp: httpx.Response, expected: int = 200) -> Sample:
    """Convert one HTTP response into a timed Sample (ok iff status == expected)."""
    latency = (time.perf_counter() - start) * 1000
    ok = resp.status_code == expected
    error = ""
    if not ok:
        try:
            error = str(resp.json().get("detail", ""))[:200]
        except Exception:
            error = resp.text[:200]
    return Sample(latency_ms=latency, ok=ok, status=resp.status_code, error=error)


def _exc_sample(start: float, exc: Exception) -> Sample:
    """Convert a client-side exception into a failed Sample (status 0)."""
    return Sample(
        latency_ms=(time.perf_counter() - start) * 1000,
        ok=False,
        status=0,
        error=type(exc).__name__,
    )


def _scenario(do_request: RequestFn, expected: int = 200) -> ScenarioFn:
    """Wrap one HTTP call in the shared timing / Sample / exception handling."""

    async def fn(client: httpx.AsyncClient, base_url: str) -> Sample:
        start = time.perf_counter()
        try:
            return _make_sample(start, await do_request(client, base_url), expected)
        except Exception as e:
            return _exc_sample(start, e)

    return fn


def make_get(path: str, timeout: float, params: Optional[dict] = None) -> ScenarioFn:
    """Build a GET scenario."""
    return _scenario(
        lambda client, base_url: client.get(
            f"{base_url}{path}", params=params or {}, timeout=timeout
        )
    )


def make_post(
    path: str, payload: dict, timeout: float, expected: int = 200
) -> ScenarioFn:
    """Build a POST scenario; ``expected`` lets validation paths expect a 4xx."""
    return _scenario(
        lambda client, base_url: client.post(
            f"{base_url}{path}", json=payload, timeout=timeout
        ),
        expected,
    )


def make_patch(path: str, payload: dict, timeout: float) -> ScenarioFn:
    """Build a PATCH scenario."""
    return _scenario(
        lambda client, base_url: client.patch(
            f"{base_url}{path}", json=payload, timeout=timeout
        )
    )


def make_put(path: str, payload: dict, timeout: float) -> ScenarioFn:
    """Build a PUT scenario."""
    return _scenario(
        lambda client, base_url: client.put(
            f"{base_url}{path}", json=payload, timeout=timeout
        )
    )


def make_delete(path: str, timeout: float, params: Optional[dict] = None) -> ScenarioFn:
    """Build a DELETE scenario with query params."""
    return _scenario(
        lambda client, base_url: client.delete(
            f"{base_url}{path}", params=params or {}, timeout=timeout
        )
    )


def make_delete_body(path: str, payload: dict, timeout: float) -> ScenarioFn:
    """Build a DELETE scenario carrying a JSON body (httpx.delete cannot)."""
    return _scenario(
        lambda client, base_url: client.request(
            "DELETE", f"{base_url}{path}", json=payload, timeout=timeout
        )
    )


# ── Scenario catalogue ─────────────────────────────────────────────────────────
#
# Every repeatable endpoint maps to one scenario here.
# Destructive endpoints (DELETE /points/{id}, DELETE /points,
# DELETE /collections/{name}) run once in teardown — not here.


def build_scenarios(collection: str, timeout: float) -> Dict[str, ScenarioFn]:
    """Build the full scenario map targeting ``collection``."""
    return {
        # ── MONITORING ─────────────────────────────────────────────────────
        # GET /livez
        "livez": make_get("/livez", timeout),
        # GET /readyz
        "readyz": make_get("/readyz", timeout),
        # GET /health
        "health": make_get("/health", timeout),
        # GET /stats
        "stats": make_get("/stats", timeout),
        # GET /metrics
        "metrics": make_get("/metrics", timeout),
        # ── COLLECTIONS ────────────────────────────────────────────────────
        # POST /collections  (collection already exists → idempotent index repair)
        "collections_create": make_post(
            "/collections",
            {
                "collection_name": collection,
                "recreate": False,
                **BENCH_COLLECTION_OPTIONS,
            },
            timeout,
        ),
        # GET /collections
        "list_collections": make_get("/collections", timeout),
        # GET /collections/{name}
        "get_collection": make_get(f"/collections/{collection}", timeout),
        # POST /collections/verify
        "verify_collection": make_post(
            "/collections/verify",
            {"collection_name": collection},
            timeout,
        ),
        # PATCH /collections/{name}/schema-toggles  (same values → idempotent)
        "schema_toggles": make_patch(
            f"/collections/{collection}/schema-toggles",
            dict(BENCH_COLLECTION_OPTIONS),
            timeout,
        ),
        # POST /collections/{name}/payload-indexes  (existing schema → skipped,
        # exercising the C8 concurrent-create path under load)
        "payload_index_create": make_post(
            f"/collections/{collection}/payload-indexes",
            {"indexes": BENCH_PAYLOAD_INDEXES},
            timeout,
        ),
        # GET /collections/{name}/payload-indexes
        "payload_index_list": make_get(
            f"/collections/{collection}/payload-indexes", timeout
        ),
        # GET /collections/{name}/payload-keys
        "payload_keys": make_get(f"/collections/{collection}/payload-keys", timeout),
        # ── POINTS — write (idempotent) ────────────────────────────────────
        # POST /points/ingest  (bench_chunks.json, replace_source=True)
        "ingest_file": make_post(
            "/points/ingest",
            {
                "file_path": BENCH_FILE_REPEAT,
                "collection_name": collection,
                "source_id": BENCH_SOURCE_FILE,
                "replace_source": True,
            },
            timeout * 4,
        ),
        # POST /points/ingest-chunks  (inline, replace_source=True)
        "ingest_chunks": make_post(
            "/points/ingest-chunks",
            {
                "source_id": BENCH_SOURCE_INLINE,
                "collection_name": collection,
                "chunks": INLINE_CHUNKS,
                "replace_source": True,
            },
            timeout,
        ),
        # POST /points  (same ID → idempotent upsert)
        "upsert_point": make_post(
            "/points",
            {
                "collection_name": collection,
                "point_id": BENCH_POINT_ID,
                "text": "benchmark upsert point for middleware end-to-end test",
                "payload": {"bench_tag": "upsert"},
            },
            timeout,
        ),
        # POST /points/batch  (same IDs → idempotent batch upsert)
        "batch_upsert": make_post(
            "/points/batch",
            {
                "collection_name": collection,
                "points": [
                    {
                        "point_id": pid,
                        "text": f"batch benchmark point {i}",
                        "payload": {"bench_tag": "batch"},
                    }
                    for i, pid in enumerate(BENCH_BATCH_IDS)
                ],
            },
            timeout,
        ),
        # POST /points  (update_mode=insert_only on an id seeded by setup →
        # always skipped; exercises the preflight-before-embed path)
        "upsert_point_insert_only": make_post(
            "/points",
            {
                "collection_name": collection,
                "point_id": BENCH_POINT_ID,
                "text": "must never overwrite the seeded point",
                "payload": {"bench_tag": "should-not-land"},
                "update_mode": "insert_only",
            },
            timeout,
        ),
        # POST /points/batch  (update_mode=insert_only on ids seeded by setup →
        # all skipped, upserted:0)
        "batch_upsert_insert_only": make_post(
            "/points/batch",
            {
                "collection_name": collection,
                "update_mode": "insert_only",
                "points": [
                    {
                        "point_id": pid,
                        "text": "must never overwrite the seeded batch point",
                        "payload": {"bench_tag": "should-not-land"},
                    }
                    for pid in BENCH_BATCH_IDS
                ],
            },
            timeout,
        ),
        # ── POINTS — read / update ─────────────────────────────────────────
        # GET /points/{id}
        "get_point": make_get(
            f"/points/{BENCH_POINT_ID}",
            timeout,
            params={"collection_name": collection},
        ),
        # GET /points/{id}?with_vectors=true
        "get_point_with_vectors": make_get(
            f"/points/{BENCH_POINT_ID}",
            timeout,
            params={"collection_name": collection, "with_vectors": "true"},
        ),
        # POST /points/retrieve
        "retrieve_points": make_post(
            "/points/retrieve",
            {"collection_name": collection, "ids": BENCH_BATCH_IDS},
            timeout,
        ),
        # POST /points/retrieve  (with_vectors=true — measures the size/latency
        # cost of the new option against the identical call without it above)
        "retrieve_points_with_vectors": make_post(
            "/points/retrieve",
            {
                "collection_name": collection,
                "ids": BENCH_BATCH_IDS,
                "with_vectors": True,
            },
            timeout,
        ),
        # POST /points/count
        "points_count": make_post(
            "/points/count",
            {"collection_name": collection},
            timeout,
        ),
        # POST /points/facet
        "points_facet": make_post(
            "/points/facet",
            {"collection_name": collection, "key": FACET_FIELD, "limit": 20},
            timeout,
        ),
        # POST /points/scroll  (payload-only browse, filtered)
        "points_scroll": make_post(
            "/points/scroll",
            {
                "collection_name": collection,
                "filters": {FACET_FIELD: "batch"},
                "limit": 10,
            },
            timeout,
        ),
        # POST /points/scroll  (ordered by a range-indexed field)
        "points_scroll_ordered": make_post(
            "/points/scroll",
            {
                "collection_name": collection,
                "order_by": {"key": "token_count", "direction": "asc"},
                "limit": 10,
            },
            timeout,
        ),
        # PATCH /points/payload  (merge by filter — repeatable: same key/value
        # applied to the same matched set every time)
        "payload_patch_by_filter": make_patch(
            "/points/payload",
            {
                "collection_name": collection,
                "filters": {FACET_FIELD: "batch"},
                "payload": {"bench_tag_filtered": "patch"},
            },
            timeout,
        ),
        # PATCH /points/{id}/payload
        "patch_payload": make_patch(
            f"/points/{BENCH_POINT_ID}/payload",
            {"collection_name": collection, "payload": {"bench_tag": "patch"}},
            timeout,
        ),
        # PUT /points/{id}/payload
        "put_payload": make_put(
            f"/points/{BENCH_POINT_ID}/payload",
            {
                "collection_name": collection,
                "payload": {
                    "bench_tag": "put",
                    "text": "benchmark upsert point for middleware end-to-end test",
                },
            },
            timeout,
        ),
        # ── QUERY — exhaustive parameter matrix (all → POST /query/search) ──
        #
        # fusion x rerank (4 combinations)
        "query_rrf_rerank": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": True,
                "limit": 5,
                "candidate_limit": 20,
            },
            timeout,
        ),
        "query_rrf_norerank": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "candidate_limit": 20,
            },
            timeout,
        ),
        "query_dbsf_rerank": make_post(
            "/query/search",
            {
                "query": QUERY_DBSF,
                "collection_name": collection,
                "fusion": "dbsf",
                "rerank": True,
                "limit": 5,
                "candidate_limit": 20,
            },
            timeout,
        ),
        "query_dbsf_norerank": make_post(
            "/query/search",
            {
                "query": QUERY_DBSF,
                "collection_name": collection,
                "fusion": "dbsf",
                "rerank": False,
                "limit": 5,
                "candidate_limit": 20,
            },
            timeout,
        ),
        # RRF tuning — exercises the RrfQuery branch (rrf_k / rrf_weights)
        "query_rrf_k": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "candidate_limit": 20,
                "rrf_k": 10,
            },
            timeout,
        ),
        "query_rrf_weights": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "candidate_limit": 20,
                "rrf_weights": [2.0, 0.5],
            },  # positional: [dense, sparse]
            timeout,
        ),
        # Rerank tuning — explicit window size and normalize=False
        "query_rerank_top_k": make_post(
            "/query/search",
            {
                "query": QUERY_TECH,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": True,
                "limit": 5,
                "candidate_limit": 30,
                "rerank_top_k": 10,
            },
            timeout,
        ),
        "query_rerank_no_normalize": make_post(
            "/query/search",
            {
                "query": QUERY_TECH,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": True,
                "limit": 5,
                "candidate_limit": 20,
                "rerank_normalize": False,
            },
            timeout,
        ),
        # Filters — scalar MatchValue, list MatchAny (OR), space-quoted key,
        # integer field, and multi-key AND
        "query_filter_source": make_post(
            "/query/search",
            {
                "query": QUERY_FILTER,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "filters": {"source_id": BENCH_SOURCE_FILE},
            },
            timeout,
        ),
        "query_filter_type": make_post(
            "/query/search",
            {
                "query": QUERY_FILTER,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "filters": {"type": FILTER_TYPE},
            },
            timeout,
        ),
        "query_filter_header1": make_post(
            "/query/search",
            {
                "query": QUERY_TECH,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "filters": {"Header 1": FILTER_HEADER1},
            },
            timeout,
        ),
        "query_filter_match_any": make_post(
            "/query/search",
            {
                "query": QUERY_TECH,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "filters": {"Header 1": [FILTER_HEADER1, FILTER_HEADER1_ALT]},
            },  # OR
            timeout,
        ),
        "query_filter_token_count": make_post(
            "/query/search",
            {
                "query": QUERY_TECH,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "filters": {"token_count": FILTER_TOKEN_COUNTS},
            },  # integer MatchAny
            timeout,
        ),
        "query_filter_multi": make_post(
            "/query/search",
            {
                "query": QUERY_TECH,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "filters": {"type": FILTER_TYPE, "Header 1": FILTER_HEADER1},
            },  # AND
            timeout,
        ),
        # Custom payload selector
        "query_with_payload": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "with_payload": ["text", "source_id"],
            },
            timeout,
        ),
        # Wide limit / candidate_limit
        "query_limits_wide": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 20,
                "candidate_limit": 100,
            },
            timeout,
        ),
        # score_threshold — applied to the fused score, before any rerank
        "query_score_threshold": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "candidate_limit": 20,
                "score_threshold": 0.01,
            },
            timeout,
        ),
        # offset — pages fused hits; offset + limit must stay <= candidate_limit
        "query_offset": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "candidate_limit": 20,
                "offset": 5,
            },
            timeout,
        ),
        # Per-request search-param overrides: hnsw_ef, exact, rescore,
        # oversampling, timeout — grouped in one scenario since all five tune
        # the same prefetch call rather than being independent features.
        "query_search_params": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "fusion": "rrf",
                "rerank": False,
                "limit": 5,
                "candidate_limit": 20,
                "hnsw_ef": 128,
                "exact": False,
                "rescore": True,
                "oversampling": 2.0,
                "timeout": 30,
            },
            timeout,
        ),
        # Validation paths — server MUST reject these (scenario passes on the 4xx)
        "query_err_rrf_opts_dbsf": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "fusion": "dbsf",
                "rrf_k": 10,
            },  # rrf_* only valid with fusion=rrf → 400
            timeout,
            expected=400,
        ),
        "query_err_candidate_lt_limit": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "limit": 10,
                "candidate_limit": 5,
            },  # candidate_limit < limit → 422
            timeout,
            expected=422,
        ),
        # A field with no payload index: rejected by QUERY_FILTER_FIELD_ALLOWLIST
        # locally (422) when one is configured; otherwise it reaches Qdrant, which
        # rejects it itself (400, mapped from Qdrant's own "Index required but
        # not found" — see _qdrant_query_failure). This deployment runs with no
        # allowlist configured, so 400 is the expected/correct outcome here.
        "query_err_filter_not_allowed": make_post(
            "/query/search",
            {
                "query": QUERY_RRF,
                "collection_name": collection,
                "filters": {"not_indexed_field": "x"},
            },
            timeout,
            expected=400,
        ),
    }


# Ordered groups: within each group the order matters (writes before reads).
QUERY_SCENARIOS = [
    # fusion x rerank
    "query_rrf_rerank",
    "query_rrf_norerank",
    "query_dbsf_rerank",
    "query_dbsf_norerank",
    # rrf tuning
    "query_rrf_k",
    "query_rrf_weights",
    # rerank tuning
    "query_rerank_top_k",
    "query_rerank_no_normalize",
    # filters
    "query_filter_source",
    "query_filter_type",
    "query_filter_header1",
    "query_filter_match_any",
    "query_filter_token_count",
    "query_filter_multi",
    # payload selector + limits
    "query_with_payload",
    "query_limits_wide",
    # score_threshold / offset / per-request search params
    "query_score_threshold",
    "query_offset",
    "query_search_params",
    # validation (4xx expected)
    "query_err_rrf_opts_dbsf",
    "query_err_candidate_lt_limit",
    "query_err_filter_not_allowed",
]

SCENARIO_GROUPS: Dict[str, List[str]] = {
    "monitoring": ["livez", "readyz", "health", "stats", "metrics"],
    "collections": [
        "collections_create",
        "list_collections",
        "get_collection",
        "verify_collection",
        "schema_toggles",
        "payload_index_create",
        "payload_index_list",
        "payload_keys",
    ],
    "points": [
        "upsert_point",
        "upsert_point_insert_only",
        "batch_upsert",
        "batch_upsert_insert_only",
        "ingest_file",
        "ingest_chunks",
        "get_point",
        "get_point_with_vectors",
        "retrieve_points",
        "retrieve_points_with_vectors",
        "points_count",
        "points_facet",
        "points_scroll",
        "points_scroll_ordered",
        "patch_payload",
        "put_payload",
        "payload_patch_by_filter",
    ],
    "query": QUERY_SCENARIOS,
}

ALL_SCENARIOS = [s for group in SCENARIO_GROUPS.values() for s in group]


# ── Setup / teardown ───────────────────────────────────────────────────────────
#
# Setup:    create collection + seed Stellantis data so query scenarios
#           have content to search from the very first request.
# Teardown: DELETE /points/{id}, DELETE /points, DELETE /collections/{name}
#           These three destructive endpoints only run here.


async def _run_once(
    client: httpx.AsyncClient,
    base_url: str,
    name: str,
    fn: ScenarioFn,
) -> ScenarioResult:
    """Run one setup/teardown step once and print its outcome."""
    start = time.perf_counter()
    sample = await fn(client, base_url)
    wall = time.perf_counter() - start
    result = ScenarioResult(name=name, samples=[sample], wall_time_s=wall)
    status = "PASS" if sample.ok else f"FAIL({sample.status or sample.error})"
    print(f"  {name}: {status}  {sample.latency_ms:.0f} ms")
    return result


@dataclass
class CollectionOwnership:
    """Track confirmed creation even when later setup is interrupted."""

    created: bool = False


async def run_setup(
    client: httpx.AsyncClient,
    base_url: str,
    collection: str,
    timeout: float,
    ownership: Optional[CollectionOwnership] = None,
) -> List[ScenarioResult]:
    """Create the bench collection and seed all data the scenarios rely on."""
    print(section("Setup"))
    results: List[ScenarioResult] = []

    async def create_owned(client, base_url):
        """Seed only after an unambiguous creation acknowledgement."""
        start = time.perf_counter()
        try:
            existing = await client.get(
                f"{base_url}/collections/{collection}", timeout=timeout
            )
            if existing.status_code != 404:
                return Sample(0, False, existing.status_code, "collection unavailable")
            response = await client.post(
                f"{base_url}/collections",
                json={
                    "collection_name": collection,
                    "recreate": False,
                    **BENCH_COLLECTION_OPTIONS,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            created = (
                isinstance(data, dict)
                and data.get("created") is True
                and data.get("ok") is True
                and data.get("collection_name") == collection
            )
            if ownership is not None:
                ownership.created = created
            return Sample(
                (time.perf_counter() - start) * 1000,
                created,
                response.status_code,
                "" if created else "creation not confirmed",
            )
        except (httpx.HTTPError, ValueError):
            return Sample(0, False, error="creation not confirmed")

    # POST /collections
    results.append(
        await _run_once(
            client,
            base_url,
            "create_collection",
            create_owned,
        )
    )
    if not results[-1].samples[0].ok:
        print(f"  [ERROR] Cannot create '{collection}' — aborting setup.")
        return results

    # POST /collections/{name}/payload-indexes — nothing creates these
    # automatically, so the filter/facet/order_by scenarios below need them
    # created before any of them run.
    results.append(
        await _run_once(
            client,
            base_url,
            "create_payload_indexes",
            make_post(
                f"/collections/{collection}/payload-indexes",
                {"indexes": BENCH_PAYLOAD_INDEXES},
                timeout,
            ),
        )
    )

    # POST /points/ingest (Stellantis doc — rich data for realistic queries)
    results.append(
        await _run_once(
            client,
            base_url,
            "ingest_stellantis_doc",
            make_post(
                "/points/ingest",
                {
                    "file_path": BENCH_FILE_SETUP,
                    "collection_name": collection,
                    "replace_source": True,
                },
                timeout * 8,
            ),
        )
    )

    # POST /points  (seed BENCH_POINT_ID for get/patch/put/delete scenarios)
    results.append(
        await _run_once(
            client,
            base_url,
            "seed_upsert_point",
            make_post(
                "/points",
                {
                    "collection_name": collection,
                    "point_id": BENCH_POINT_ID,
                    "text": "benchmark upsert point for middleware end-to-end test",
                    "payload": {"bench_tag": "upsert"},
                },
                timeout,
            ),
        )
    )

    # POST /points/batch  (seed BENCH_BATCH_IDS for bulk_delete teardown)
    results.append(
        await _run_once(
            client,
            base_url,
            "seed_batch_upsert",
            make_post(
                "/points/batch",
                {
                    "collection_name": collection,
                    "points": [
                        {
                            "point_id": pid,
                            "text": f"batch benchmark point {i}",
                            "payload": {"bench_tag": "batch"},
                        }
                        for i, pid in enumerate(BENCH_BATCH_IDS)
                    ],
                },
                timeout,
            ),
        )
    )

    # POST /points/ingest-chunks  (seed source_id=BENCH_SOURCE_INLINE so
    # source-id filter scenarios have matching points even when 'points' is not run)
    results.append(
        await _run_once(
            client,
            base_url,
            "seed_inline_chunks",
            make_post(
                "/points/ingest-chunks",
                {
                    "source_id": BENCH_SOURCE_INLINE,
                    "collection_name": collection,
                    "chunks": INLINE_CHUNKS,
                    "replace_source": True,
                },
                timeout,
            ),
        )
    )

    # POST /points/ingest  (seed bench_chunks.json with source_id=BENCH_SOURCE_FILE
    # so type / Header 1 / token_count filter scenarios match real points regardless
    # of which scenario groups the user selects)
    results.append(
        await _run_once(
            client,
            base_url,
            "seed_file_chunks",
            make_post(
                "/points/ingest",
                {
                    "file_path": BENCH_FILE_REPEAT,
                    "collection_name": collection,
                    "source_id": BENCH_SOURCE_FILE,
                    "replace_source": True,
                },
                timeout * 4,
            ),
        )
    )

    print()
    return results


async def run_teardown(
    client: httpx.AsyncClient,
    base_url: str,
    collection: str,
    timeout: float,
) -> List[ScenarioResult]:
    """Delete the seeded points and the bench collection (destructive steps)."""
    print(section("Teardown"))
    results: List[ScenarioResult] = []

    # DELETE /points/{id}
    results.append(
        await _run_once(
            client,
            base_url,
            "delete_point",
            make_delete(
                f"/points/{BENCH_POINT_ID}",
                timeout,
                params={"collection_name": collection},
            ),
        )
    )

    # DELETE /points  (bulk)
    results.append(
        await _run_once(
            client,
            base_url,
            "bulk_delete_points",
            make_delete_body(
                "/points",
                {
                    "collection_name": collection,
                    "ids": BENCH_BATCH_IDS,
                },
                timeout,
            ),
        )
    )

    # DELETE /collections/{name}/payload-indexes/{field}  (one-shot: dropping
    # the same index twice 404s, so this cannot be a repeatable scenario)
    results.append(
        await _run_once(
            client,
            base_url,
            "delete_payload_index",
            make_delete(
                f"/collections/{collection}/payload-indexes/{FACET_FIELD}", timeout
            ),
        )
    )

    # DELETE /collections/{name}
    results.append(
        await _run_once(
            client,
            base_url,
            "delete_collection",
            make_delete(
                f"/collections/{collection}",
                timeout,
                params={"confirm_delete": collection},
            ),
        )
    )

    print()
    return results


# ── Benchmark runner ───────────────────────────────────────────────────────────


async def run_scenario(
    client: httpx.AsyncClient,
    base_url: str,
    name: str,
    fn: ScenarioFn,
    requests: int,
    concurrency: int,
    warmup: int,
) -> ScenarioResult:
    """Run one scenario ``requests`` times with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)

    async def bounded() -> Sample:
        async with sem:
            return await fn(client, base_url)

    # Warmup is bounded by the same semaphore so it cannot exceed the configured
    # concurrency (avoids skewing server cache state / tripping rate limits).
    if warmup > 0:
        await asyncio.gather(
            *[bounded() for _ in range(warmup)], return_exceptions=True
        )

    start = time.perf_counter()
    samples = await asyncio.gather(*[bounded() for _ in range(requests)])
    wall = time.perf_counter() - start

    return ScenarioResult(
        name=name,
        samples=list(samples),
        wall_time_s=wall,
    )


async def run_selected(
    client: httpx.AsyncClient,
    base_url: str,
    names: List[str],
    scenario_map: Dict[str, ScenarioFn],
    requests: int,
    concurrency: int,
    warmup: int,
    sleep_between: float,
) -> List[ScenarioResult]:
    """Run the selected scenarios sequentially, printing progress per scenario."""
    results: List[ScenarioResult] = []
    for idx, name in enumerate(names):
        if idx > 0 and sleep_between > 0:
            print(f"  Cooling down {sleep_between}s...")
            await asyncio.sleep(sleep_between)
        print(f"  Running: {name} ...")
        res = await run_scenario(
            client=client,
            base_url=base_url,
            name=name,
            fn=scenario_map[name],
            requests=requests,
            concurrency=concurrency,
            warmup=warmup,
        )
        results.append(res)
        print(
            f"    {'PASS' if res.failed == 0 else 'FAIL'} "
            f"({res.success}/{len(res.samples)} OK, "
            f"{res.wall_time_s:.2f}s, statuses: {format_status_counts(res)})"
        )
    return results


# ── Output helpers ─────────────────────────────────────────────────────────────


def summarize(results: List[ScenarioResult], concurrency: int) -> str:
    """Render the per-scenario latency/throughput summary table."""
    headers = [
        "Scenario",
        "Endpoint",
        "Result",
        "Reqs",
        "OK",
        "OK%",
        "Fail",
        "Conc",
        "Wall(s)",
        "Req/s",
        "Avg(ms)",
        "P50",
        "P95",
        "P99",
        "Min",
        "Max",
        "Statuses",
    ]
    # Map scenario name → endpoint label for reference
    endpoint_map: Dict[str, str] = {
        "livez": "GET /livez",
        "readyz": "GET /readyz",
        "health": "GET /health",
        "stats": "GET /stats",
        "metrics": "GET /metrics",
        "collections_create": "POST /collections",
        "list_collections": "GET /collections",
        "get_collection": "GET /collections/{name}",
        "verify_collection": "POST /collections/verify",
        "schema_toggles": "PATCH /collections/{name}/schema-toggles",
        "payload_index_create": "POST /collections/{name}/payload-indexes",
        "payload_index_list": "GET /collections/{name}/payload-indexes",
        "payload_keys": "GET /collections/{name}/payload-keys",
        "ingest_file": "POST /points/ingest",
        "ingest_chunks": "POST /points/ingest-chunks",
        "upsert_point": "POST /points",
        "upsert_point_insert_only": "POST /points",
        "batch_upsert": "POST /points/batch",
        "batch_upsert_insert_only": "POST /points/batch",
        "get_point": "GET /points/{id}",
        "get_point_with_vectors": "GET /points/{id}",
        "retrieve_points": "POST /points/retrieve",
        "retrieve_points_with_vectors": "POST /points/retrieve",
        "points_count": "POST /points/count",
        "points_facet": "POST /points/facet",
        "points_scroll": "POST /points/scroll",
        "points_scroll_ordered": "POST /points/scroll",
        "patch_payload": "PATCH /points/{id}/payload",
        "put_payload": "PUT /points/{id}/payload",
        "payload_patch_by_filter": "PATCH /points/payload",
        # all query_* scenarios hit POST /query/search (resolved by prefix below)
        # setup/teardown labels
        "create_collection": "POST /collections",
        "create_payload_indexes": "POST /collections/{name}/payload-indexes",
        "ingest_stellantis_doc": "POST /points/ingest",
        "seed_upsert_point": "POST /points",
        "seed_batch_upsert": "POST /points/batch",
        "seed_inline_chunks": "POST /points/ingest-chunks",
        "seed_file_chunks": "POST /points/ingest",
        "delete_point": "DELETE /points/{id}",
        "bulk_delete_points": "DELETE /points",
        "delete_payload_index": "DELETE /collections/{name}/payload-indexes/{field}",
        "delete_collection": "DELETE /collections/{name}",
    }

    def endpoint_label(name: str) -> str:
        if name.startswith("query"):
            return "POST /query/search"
        return endpoint_map.get(name, "")

    rows: List[List[str]] = []
    for r in results:
        lat = r.ok_latencies
        rps = r.success / r.wall_time_s if r.wall_time_s > 0 else 0
        if lat:
            avg = statistics.fmean(lat)
            p50 = percentile(lat, 50)
            p95 = percentile(lat, 95)
            p99 = percentile(lat, 99)
            mn, mx = min(lat), max(lat)
        else:
            avg = p50 = p95 = p99 = mn = mx = 0.0
        rows.append(
            [
                r.name,
                endpoint_label(r.name),
                "PASS" if r.failed == 0 else "FAIL",
                fmt_int(len(r.samples)),
                fmt_int(r.success),
                f"{r.ok_percent:.1f}%",
                fmt_int(r.failed),
                str(concurrency),
                f"{r.wall_time_s:.2f}",
                f"{rps:.1f}",
                fmt_ms(avg),
                fmt_ms(p50),
                fmt_ms(p95),
                fmt_ms(p99),
                fmt_ms(mn),
                fmt_ms(mx),
                format_status_counts(r),
            ]
        )
    return render_table(rows, headers)


def render_verdict(results: List[ScenarioResult]) -> str:
    """Render the one-line PASS/FAIL verdict for the whole run."""
    total = sum(len(r.samples) for r in results)
    failed = sum(r.failed for r in results)
    if failed == 0:
        return f"Verdict: PASS — {fmt_int(total)} requests completed successfully."
    names = ", ".join(r.name for r in results if r.failed)
    return (
        f"Verdict: FAIL — {fmt_int(failed)}/{fmt_int(total)} "
        f"requests failed across: {names}."
    )


def render_failure_details(results: List[ScenarioResult]) -> str:
    """Render per-scenario failure status counts and error messages."""
    lines = []
    for result in results:
        if not result.failed:
            continue
        status_totals: Dict[str, int] = {}
        messages: Dict[str, int] = {}
        for sample in result.samples:
            if sample.ok:
                continue
            label = sample_status_label(sample)
            status_totals[label] = status_totals.get(label, 0) + 1
            if sample.error:
                messages[sample.error] = messages.get(sample.error, 0) + 1
        lines.append(
            f"  {result.name}: "
            + ", ".join(f"{s}={c}" for s, c in status_totals.items())
        )
        for msg, cnt in sorted(messages.items()):
            lines.append(f"    - {msg}: {cnt}")
    return (section("Failure details") + "\n" + "\n".join(lines)) if lines else ""


def parse_prometheus_samples(text: str) -> Dict[str, float]:
    """Extract the middleware's key metric samples from Prometheus text format."""
    prefixes = (
        "qdrant_manager_requests_total",
        "qdrant_manager_requests_rejected_total",
        "qdrant_manager_request_duration_seconds",
        "qdrant_manager_vectorstore_up",
        "qdrant_collection_operations_total",
    )
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(prefixes):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            with suppress(ValueError):
                out[parts[0]] = float(parts[1])
    return out


def render_metric_delta(pre: str, post: str) -> List[str]:
    """Render before/after deltas for the key Prometheus metrics."""
    pre_m = parse_prometheus_samples(pre)
    post_m = parse_prometheus_samples(post)
    lines = []
    for key in sorted(post_m):
        delta = post_m[key] - pre_m.get(key, 0.0)
        if "_up" in key:
            lines.append(f"{key} current={post_m[key]:.0f}")
        elif delta:
            lines.append(f"{key} delta={delta:.0f} current={post_m[key]:.0f}")
    return lines


def render_server_info(info: dict) -> str:
    """Render the pre-flight /health summary table."""
    return render_table(
        [
            ["Status", str(info.get("status", "unknown"))],
            ["Qdrant", str(info.get("qdrant", "unknown"))],
            ["Collection", str(info.get("default_collection", "unknown"))],
            ["Collections", str(info.get("collections", "unknown"))],
        ],
        ["Field", "Value"],
    )


# ── Validation / resolution ────────────────────────────────────────────────────


def validate_args(args: argparse.Namespace) -> List[str]:
    """Return human-readable errors for out-of-range CLI arguments."""
    errors = []
    if args.concurrency < 1:
        errors.append("--concurrency must be >= 1")
    if args.requests < 1:
        errors.append("--requests must be >= 1")
    if args.warmup < 0:
        errors.append("--warmup must be >= 0")
    if args.timeout <= 0:
        errors.append("--timeout must be > 0")
    if args.sleep_between < 0:
        errors.append("--sleep-between must be >= 0")
    return errors


def resolve_scenarios(tokens: List[str]) -> List[str]:
    """Expand scenario/group tokens into a deduplicated, ordered name list."""
    resolved: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token == "all":
            names = ALL_SCENARIOS
        elif token in SCENARIO_GROUPS:
            names = SCENARIO_GROUPS[token]
        elif token in ALL_SCENARIOS:
            names = [token]
        else:
            raise ValueError(
                f"Unknown scenario/group: '{token}'. "
                f"Groups: {sorted(SCENARIO_GROUPS)}. "
                f"Scenarios: {ALL_SCENARIOS}."
            )
        for name in names:
            if name not in seen:
                resolved.append(name)
                seen.add(name)
    return resolved


# ── Main ──────────────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> int:
    """Run pre-flight, setup, benchmark, and teardown; return the exit code."""
    errors = validate_args(args)
    if errors:
        print("Invalid arguments:")
        for e in errors:
            print(f"  - {e}")
        return 2

    tokens = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if not tokens:
        print("No scenarios selected.")
        return 2
    try:
        selected = resolve_scenarios(tokens)
    except ValueError as e:
        print(str(e))
        return 2

    limits = httpx.Limits(
        max_connections=args.concurrency * 2,
        max_keepalive_connections=args.concurrency * 2,
    )
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}

    async with httpx.AsyncClient(headers=headers, limits=limits) as client:
        # ── Pre-flight ──────────────────────────────────────────────────────
        print(section("Pre-flight health"))
        try:
            r = await client.get(f"{args.url}/health", timeout=10.0)
            if r.status_code != 200:
                print(f"Health check failed: HTTP {r.status_code}")
                return 1
            info = r.json()
            print(render_server_info(info))
        except Exception as e:
            print(f"Cannot reach {args.url}: {e}")
            return 1
        print()

        # ── Plan ────────────────────────────────────────────────────────────
        print(section("Benchmark plan"))
        print(
            render_table(
                [
                    ["Target", args.url],
                    ["Collection", BENCH_COLLECTION],
                    ["Scenarios", ", ".join(selected)],
                    ["Requests", str(args.requests)],
                    ["Concurrency", str(args.concurrency)],
                    ["Warmup", f"{args.warmup} req/scenario"],
                    ["Timeout", f"{args.timeout:.1f}s"],
                    ["Auth", "Bearer provided" if args.token else "disabled"],
                ],
                ["Setting", "Value"],
            )
        )
        print()

        scenarios = build_scenarios(BENCH_COLLECTION, args.timeout)

        bench_results: List[ScenarioResult] = []
        setup_results: List[ScenarioResult] = []
        teardown_results: List[ScenarioResult] = []
        setup_ok = False
        ownership = CollectionOwnership()
        pre_metrics = post_metrics = ""

        # Once creation is acknowledged, teardown is attempted even if setup
        # or benchmarking is interrupted. An ambiguous response grants no ownership.
        try:
            # ── Setup ───────────────────────────────────────────────────────
            setup_results = await run_setup(
                client, args.url, BENCH_COLLECTION, args.timeout, ownership
            )
            setup_ok = all(r.samples[0].ok for r in setup_results)

            if not setup_ok:
                print("[ERROR] Setup failed — skipping benchmark.")
                print()
            else:
                # ── Benchmark ───────────────────────────────────────────────
                print(section("Benchmark"))
                if not args.no_post_check:
                    try:
                        mr = await client.get(f"{args.url}/metrics", timeout=10.0)
                        if mr.status_code == 200:
                            pre_metrics = mr.text
                    except Exception:
                        pass

                bench_results = await run_selected(
                    client=client,
                    base_url=args.url,
                    names=selected,
                    scenario_map=scenarios,
                    requests=args.requests,
                    concurrency=args.concurrency,
                    warmup=args.warmup,
                    sleep_between=args.sleep_between,
                )
                print()

                if not args.no_post_check:
                    try:
                        mr = await client.get(f"{args.url}/metrics", timeout=10.0)
                        if mr.status_code == 200:
                            post_metrics = mr.text
                    except Exception:
                        pass
        finally:
            # ── Teardown of this run's confirmed collection ────────────────
            if ownership.created:
                teardown_results = await run_teardown(
                    client, args.url, BENCH_COLLECTION, args.timeout
                )

    # ── Print results ────────────────────────────────────────────────────────
    print(section("Setup results"))
    print(summarize(setup_results, 1))
    print()

    if bench_results:
        print(section("Benchmark results"))
        print(summarize(bench_results, args.concurrency))
        print()
        print(render_verdict(bench_results))
        details = render_failure_details(bench_results)
        if details:
            print()
            print(details)
        print()

    print(section("Teardown results"))
    print(summarize(teardown_results, 1))
    print()

    if not args.no_post_check and post_metrics:
        print("Key Prometheus metric deltas:")
        lines = render_metric_delta(pre_metrics, post_metrics)
        for line in lines:
            print(f"  {line}")
        if not lines:
            print("  No key metric delta found.")
        print()

    # Rate-limit hint: a single client IP exhausting the token bucket surfaces as
    # 429s, which look like benchmark failures. Point the user at the knobs.
    rate_limited = sum(1 for r in bench_results for s in r.samples if s.status == 429)
    if rate_limited:
        print(
            f"Warning: {fmt_int(rate_limited)} request(s) hit the "
            "server rate limit (HTTP 429). "
            "Lower --requests/--concurrency, raise --sleep-between, "
            "or increase RATE_LIMIT_REQUESTS_PER_MINUTE / "
            "RATE_LIMIT_BURST_SIZE on the server."
        )
        print()

    any_bench_fail = any(r.failed > 0 for r in bench_results)
    failed_steps = [r.name for r in setup_results if r.failed] + [
        r.name for r in teardown_results if r.failed
    ]
    if failed_steps:
        print(f"Warning: setup/teardown steps failed: {', '.join(failed_steps)}")
        print()

    return 1 if (any_bench_fail or not setup_ok) else 0


def parse_args() -> argparse.Namespace:
    """Parse the benchmark CLI arguments."""
    p = argparse.ArgumentParser(description="Qdrant Middleware end-to-end benchmark")
    p.add_argument("--url", default="http://localhost:6336", help="Middleware base URL")
    p.add_argument(
        "--token",
        default=os.getenv("API_TOKEN", ""),
        help="Bearer token (reads API_TOKEN env var by default)",
    )
    p.add_argument("--concurrency", type=int, default=4, help="Concurrent requests")
    p.add_argument("--requests", type=int, default=20, help="Requests per scenario")
    p.add_argument("--warmup", type=int, default=2, help="Warmup requests per scenario")
    p.add_argument(
        "--timeout", type=float, default=120.0, help="Per-request timeout (s)"
    )
    p.add_argument(
        "--scenarios",
        default="all",
        help=(
            "Comma-separated scenario names or groups, or 'all'. "
            f"Groups: {', '.join(SCENARIO_GROUPS)}. "
            f"Scenarios: {', '.join(ALL_SCENARIOS)}."
        ),
    )
    p.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="Seconds to sleep between scenarios",
    )
    p.add_argument(
        "--no-post-check",
        action="store_true",
        help="Skip post-run /metrics diagnostics",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        exit(asyncio.run(main_async(args)))
    except KeyboardInterrupt:
        print("\nInterrupted")
        exit(130)
