# server-embedder-reranker (Smart Embedder) — the self-hosted embedding and reranking service

![Version 1.2.0](https://img.shields.io/badge/Version-1.2.0-blue)
![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green)
![CPU Support](https://img.shields.io/badge/CPU-Supported-lightgrey)
![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)

> **Read this when** you need to answer what the embedding/reranking service can
> and cannot do, look up an endpoint, a response field, an environment variable
> or a metric, explain the two model profiles or the hardware autotuner, or
> change this service's code.
> **Audience** — an AI agent working in or on this repository, and any human
> reading over its shoulder.
> **Source of truth** — this file for the HTTP contract, the profiles, the
> autotuner contract and the configuration surface; `model_revisions.py` for
> the pinned model commits; the root `.env.example` for deployment defaults.
> **Prerequisite** — the root [`README.md`](../README.md) for how the three
> services fit together, and [`../scripts/README.md`](../scripts/README.md) for
> starting the stack.

---

## Purpose and boundaries

**Smart Embedder** is a lightweight, self-hosted embedding server built for
**hybrid search** pipelines. It runs entirely on your own hardware — on an
**NVIDIA GPU** for high throughput, or on **CPU** when no GPU is available —
with no hosted inference dependency: request data is processed locally and
does not leave your machine. A cold first startup still downloads the selected
profile's pinned model snapshots from Hugging Face into the local cache.

Hybrid search combines dense vector similarity, sparse lexical matching
(BM25-style), and optional ColBERT late-interaction scoring into a single
retrieval pipeline. Smart Embedder exposes the selected profile's vector types
from one endpoint, plus a reranking endpoint to re-score candidate passages
after retrieval.

**What it is:**

- A FastAPI service wrapping **two fixed startup profiles**:
  - **basic** — [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) for dense,
    sparse and ColBERT vectors, plus `BAAI/bge-reranker-v2-m3` for reranking.
  - **advanced** —
    [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
    dense, `omai-research/milco-650m` learned-sparse, and
    [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
    reranking. **ColBERT is not loaded in this profile.**
- The only component in this stack that runs a model.

**What it is not:**

- Not a vector store. It returns vectors; it never persists them.
- Not a runtime model switcher. The profile is chosen once at process start.
- Not a chunker, a retriever or a fusion engine — those live in the
  middleware and in Qdrant.
- Not internet-facing. Its published port is hardcoded to `127.0.0.1`, and the
  right production setting withdraws that publish entirely.

The advanced profile benefits from a dedicated GPU — 8 GB+ VRAM gives the best
results. CPU execution remains supported, with conservative batch sizes applied
automatically.

| Property | Detail |
|---|---|
| **Deployment** | Unified Docker stack with CPU and NVIDIA GPU profiles |
| **Hybrid search vectors** | Profile-selected dense + sparse, with optional ColBERT in `basic` |
| **Reranking** | Cross-encoder passage reranking, same service |
| **Footprint** | `basic` fits in 8 GB VRAM; `advanced` benefits from 8 GB+ VRAM; CPU mode needs no GPU |
| **Qdrant-ready** | Sparse vectors in native `{indices, values}` format via `sparse_as_indices` |
| **Embeddings** | `basic`: BGE-M3 dense/sparse/ColBERT; `advanced`: Qwen dense + MILCO sparse, no ColBERT |
| **Authentication** | Bearer token required by default; explicit loopback-only opt-out |
| **Rate limiting** | Token bucket, 3600 req/min per IP, burst 120 |
| **Backpressure** | Embedding queue max 200, rerank slots max 32, HTTP 503 on overflow |
| **Request size** | Body capped at 64 MiB as it streams, HTTP 413 without materializing it |
| **Graceful shutdown** | 30 s drain for in-flight requests |
| **Prometheus metrics** | Counters, histograms, gauges for both models |
| **Dynamic batching** | Embedding batch size adapts to GPU VRAM and max input length at startup |

## Mental model

Ten facts explain nearly everything surprising about this service.

1. **The profile is fixed at process start.** `SMART_EMBEDDER_PROFILE` is read
   once; there is no runtime switch. Changing profile means restarting the
   stack — and existing Qdrant collections stamped with the other profile's
   embedding contract will report as incompatible. A value other than `basic`
   or `advanced` fails startup instead of silently selecting another profile.
2. **Module names are the architectural boundary.** `basic_*` belongs only to
   the BGE profile, `advanced_*` only to the Qwen/MILCO profile, and
   unprefixed modules are shared. There is no package structure enforcing this;
   the naming is the enforcement.
3. **`smart-embedder.py` is the whole app** — over 5,000 lines covering
   profile selection, batching, both model wrappers, schemas, metrics, auth and
   lifecycle. Read the relevant section for a change; never the whole file.
4. **The autotuner is a separate pre-start process**, not part of the server.
   It benchmarks candidates under quality, stability and memory gates and
   atomically persists a decision per profile. It never edits `.env`, and a
   fingerprint change invalidates its result and forces recalibration.
5. **Model repositories are pinned to immutable commit revisions** in
   `model_revisions.py`. Updating a model means bumping a pinned revision and
   rerunning the CPU/GPU smoke tests — not chasing a moving `main`.
6. **Embedding and reranking share the GPU lanes.** BASIC and safe ADVANCED
   profiles use one lane, so their forward passes never overlap; a measured
   ADVANCED profile may overlap them, with one reranker instance still
   serialized. Peak VRAM depends on that measurement — see
   [Behaviour contracts](#behaviour-contracts).
7. **Only `/health` is unauthenticated.** Every other endpoint requires a
   bearer token (documentation endpoints exist only when Swagger is explicitly
   exposed).
8. **The published host port is the only route that skips the middleware** —
   and therefore skips its bearer auth, rate limiting and
   `QUERY_RERANK_MAX_CHARS` truncation. `EXPOSE_SMART_EMBEDDER_PORT=false`
   closes it, which is the right production setting.
9. **Over-long input is truncated, not rejected.** The response carries
   structured `warnings` instead of an error; the server no longer rejects
   requests on character-count payload limits. The one thing upstream of that
   is the request-body cap in point 10 — a body too large is refused before
   anything can be truncated.
10. **The body-size cap is the limit a caller meets first.**
    `SMART_EMBEDDER_MAX_SENTENCES_PER_REQUEST` and
    `SMART_EMBEDDER_MAX_ITEM_BYTES` are checked once the body is already
    deserialized, so they bound what is *processed*, not what is *read*.
    `SMART_EMBEDDER_REQUEST_BODY_MAX_BYTES` counts bytes as they stream in and
    answers 413 without materializing the body.

## Map of the code

Flat module layout, not a package. Naming is the boundary — see Mental model,
point 2.

| Scope | Files | Responsibility |
|---|---|---|
| Shared runtime | `smart-embedder.py` | FastAPI application: profile selection, dynamic/length-aware batching, both model wrappers, request/response schemas, Prometheus metrics, authentication and lifecycle. |
| Shared startup | `autotune_entrypoint.py` | Selects and runs the BASIC or ADVANCED pre-start tuner before Uvicorn. |
| Shared tooling | `benchmark.py` | HTTP latency, throughput, token, padding and Prometheus benchmark matrix. |
| Shared models | `model_revisions.py` | The immutable Hugging Face commit revision pinned for every model — the single place a model update is expressed. |
| Shared startup | `model_prepare.py` | Prefetches each profile's models at exactly those revisions, refusing a revision mismatch rather than falling back to a branch. A local snapshot is reused only once a finished download has been recorded beside it (`.smart-embedder-complete/`), so a download interrupted mid-way resumes on the next start instead of passing as cached. That record is the file inventory the Hub listed for the revision, written only once every listed file is present, and a snapshot is reused only while each recorded file still exists. When the Hub cannot be reached and no such record exists, preparation stops with an error rather than certifying whatever folder an interrupted download left. A record from an older release lists no files and is validated again, so the first start after upgrading needs Hub access. `BAAI/bge-m3-unsupervised` is fetched without its weights: MILCO reads only that repository's tokenizer. |
| Shared startup | `model_bootstrap.py` | Init-container entry point: prepares the named volumes' ownership, then drops `CHOWN`/`SETUID`/`SETGID` before any model or network library is imported. |
| Shared runtime | `vram_budget.py` | Device headroom accounting, the safe max-batch-token cap, and the guard that turns a CUDA allocation failure into a typed, tensor-free error. |
| Shared runtime | `request_budget.py` | Token/output estimates and shared admission leases whose lifetime includes physical native work. |
| BASIC tuning | `basic_autotune.py`, `basic_autotune_runner.py`, `basic_autotune_tokens.py` | BGE hardware profiles, isolated calibration runner, deterministic calibration inputs. |
| BASIC backend | `basic_onnx_backend.py` | CPU ONNX Runtime backend for BGE dense, sparse and ColBERT output. |
| ADVANCED tuning | `advanced_autotune.py`, `advanced_autotune_runner.py`, `advanced_autotune_worker.py`, `advanced_autotune_quality.py` | Qwen/MILCO profiles, calibration orchestration, isolated measurements, quality gates. |
| ADVANCED backends | `advanced_qwen_onnx_backend.py`, `advanced_milco_runtime.py` | Qwen CPU ONNX export/runtime and exact memory-bounded MILCO projection. |
| Tests | `tests/test_*.py` | Unit suites with fakes, plus the opt-in `tests/test_server.py` runtime suite (17 checks, 18 with `--token`). |
| Dependencies | `../requirements/embedder-gpu.txt`, `../requirements/embedder-cpu.txt`, `../requirements/locks/` | Direct GPU/CPU pins and Docker target-specific transitive constraints. |
| Deployment | `../Dockerfile`, `../compose.yml`, `../.env.example`, `../pyproject.toml` | Hardened images, CPU/GPU orchestration, complete environment template, Python tooling configuration. |

### Agent reading path

Do not load `smart-embedder.py` wholesale to investigate one behaviour. Locate
the endpoint or class first, then inspect its collaborators and focused test:

| Question | Start here |
|---|---|
| Request admission, cancellation or queue draining | `get_embeddings`, `RequestProcessor`, `request_budget.py`, and request-processor/budget tests. |
| Rerank truncation, capacity or failures | `rerank`, reranker wrappers and capacity helpers in `smart-embedder.py`; reranker and performance-hardening tests. |
| Startup chooses an unexpected runtime | `autotune_entrypoint.py`, the matching profile module/runner and persisted-profile fingerprint checks. |
| ADVANCED quality or candidate promotion changes | `advanced_autotune_quality.py`, `advanced_autotune_runner.py` and isolated worker tests. |
| CPU ONNX artifact preparation or reuse fails | `advanced_qwen_onnx_backend.py` or `basic_onnx_backend.py`, according to profile; inspect init versus serving mounts before changing permissions. |
| Model identity or downloads change | `model_revisions.py`, `model_prepare.py` and bootstrap tests; this is separate from package pin maintenance. |

Trace startup and serving separately: the init stage can prepare writable
artifacts, while the serving process consumes prepared state under stricter
filesystem permissions. Preserve the distinction between a logical response
Future and native work that may still hold an admission lease. Unit tests use
fakes; model-loading, quality, memory and throughput claims need the recorded
live checks described below.

## Interface reference

All endpoints except `/health` require:

```http
Authorization: Bearer <token>
```

Documentation endpoints are present, and intentionally public, only when
`EXPOSE_SMART_EMBEDDER_SWAGGER=true`.

### `POST /embeddings/`

Generates embeddings for a list of texts.

**Request:**
```json
{
  "sentences": ["Hello world!", "Ciao mondo!"],
  "return_dense": true,
  "return_sparse": true,
  "normalize_dense": false,
  "sparse_as_indices": false,
  "include_text": true
}
```

In the `basic` profile **only**, BGE-M3 ColBERT vectors can also be requested:

```json
{
  "sentences": ["Hello world!", "Ciao mondo!"],
  "return_dense": true,
  "return_sparse": true,
  "return_colbert": true,
  "normalize_dense": false,
  "sparse_as_indices": false,
  "include_text": true
}
```

| Field | Default | Meaning |
|---|---|---|
| `sparse_as_indices` | `false` | When `true`, sparse vectors are returned in Qdrant-compatible form (`{"indices": [10, 1389, 2349], "values": [0.277, 0.292, 0.313]}`) instead of the default token-id dict. Use with `SparseVector(indices=..., values=...)` when upserting to Qdrant. |
| `include_text` | `true` | When `false`, the original text is omitted from each entry in `data`; ids and vectors are still returned. Reduces payload size. |
| `return_colbert` | `false` | **`basic` profile only.** Not a valid request field under `advanced`. |

The active embedding backends are selected by `SMART_EMBEDDER_PROFILE` at
server startup. With `basic`, `dense`, `sparse` and `colbert` all come from
`BAAI/bge-m3`. With `advanced`, `dense` comes from `Qwen/Qwen3-Embedding-0.6B`,
`sparse` comes from `omai-research/milco-650m`, and ColBERT is disabled.

**Response:**
```json
{
  "data": [
    {
      "id": 0,
      "text": "Hello world!",
      "embeddings": {
        "dense": [0.021, -0.013, ...],
        "sparse": {"12": 0.08, "435": 0.12, ...}
      }
    }
  ],
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "dense_model_name": "Qwen/Qwen3-Embedding-0.6B",
  "sparse_model_name": "omai-research/milco-650m",
  "colbert_model_name": null,
  "embedding_contract_fingerprint": "sha256:...",
  "processing_time_ms": 104.5,
  "warnings": [
    {
      "code": "input_truncated",
      "severity": "warning",
      "message": "Input text was truncated to the model token limit.",
      "target": {
        "field": "sentences",
        "index": 0,
        "pointer": "/sentences/0"
      },
      "details": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "max_tokens": 32768,
        "original_tokens": 34000,
        "truncated_tokens": 1232,
        "truncation_side": "end"
      }
    }
  ]
}
```

`embedding_contract_fingerprint` is a SHA-256 digest of the active embedding
profile. Collections created through the middleware are stamped with the same
value, so profile mismatches can be detected instead of silently mixing vector
spaces.

```bash
curl -X POST "http://127.0.0.1:8010/embeddings/" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"sentences": ["Hello world!"], "return_dense": true}'
```

### `POST /rerank`

Ranks a list of passages by relevance to a query.

**Request:**
```json
{
  "query": "What is machine learning?",
  "passages": [
    "Machine learning is a subset of AI.",
    "The weather is nice today.",
    "Deep learning uses neural networks."
  ],
  "normalize": true
}
```

**Response:**
```json
{
  "results": [
    {"index": 0, "passage": "Machine learning is a subset of AI.", "score": 0.987},
    {"index": 2, "passage": "Deep learning uses neural networks.", "score": 0.821},
    {"index": 1, "passage": "The weather is nice today.", "score": 0.003}
  ],
  "model_name": "BAAI/bge-reranker-v2-m3",
  "processing_time_ms": 52.2,
  "warnings": []
}
```

- `normalize: true` returns a score in `[0, 1]` (sigmoid).
- `normalize: false` returns a raw score (negative values possible).
- Under `advanced`, scores are Qwen yes-probabilities and `normalize` is kept
  as an API-compatible no-op.
- **Do not compare BGE `normalize: false` raw logits directly with Qwen
  scores.**
- Passages are returned sorted by **descending** score.
- `index` is the original position in the input list.
- `model_name` reports the reranker selected at startup.

For over-token query-passage pairs, the server preserves the query where
possible, truncates passages from the end, returns `200 OK`, and includes
`query_truncated` or `passage_truncated` entries in `warnings`. Warning token
counts are computed during server-side preparation; rerank inputs are then
decoded back to text and tokenized again by the model backend, so
`original_tokens`, `max_tokens` and `truncated_tokens` are **diagnostic
metadata**, not exact proof of final backend tokenization.

#### GPU rerank capacity failures (both profiles)

If a rerank batch cannot be allocated against the GPU memory measured at
startup, the server returns one of two typed errors instead of crashing or
hanging. Both share this `detail` shape, with `error`-specific fields added:

```json
{
  "detail": {
    "error": "rerank_vram_pressure",
    "message": "Not enough free GPU memory for this batch right now. Retry, or send at most 6 passages per request.",
    "passages": 64,
    "batch_items": 32,
    "batch_tokens": 262144,
    "estimated_bytes": 912345678,
    "effective_max_batch_tokens": 200000,
    "startup_headroom_bytes": 536870912,
    "suggested_max_passages": 6
  }
}
```

| Status | `error` | Retryable | Notes |
|---|---|---|---|
| `413` | `rerank_capacity_exceeded` | No | A single query/passage pair alone does not fit the headroom measured at startup. No batch size or retry can fix this; shorten passages (`QUERY_RERANK_MAX_CHARS`) or lower the active profile's rerank limit (`BGE_RERANK_MAX_LENGTH` for `basic`, `QWEN_RERANK_MAX_LENGTH` for `advanced`). `413` is outside the middleware's retryable status set, so a failure is not multiplied into three upstream calls. No `suggested_max_passages` field and no `Retry-After` header. |
| `503` | `rerank_vram_pressure` | Yes | A batch of one *would* fit the startup baseline; the current batch does not, because of transient GPU pressure. Carries a `Retry-After: 2` header and, when the per-item cost is known, a `suggested_max_passages` field estimating how many passages per request would fit. `503` is inside the middleware's retryable status set (up to 3 attempts), and the middleware honors this `Retry-After` rather than its own backoff, so those attempts are spaced 2 s apart instead of milliseconds. An MCP client may retry the resulting `502` up to 3 more times, so a sustained pressure event can still produce as many as nine `/rerank` calls for one user query — just spread over seconds rather than arriving as a burst. |

Common `detail` fields on both: `passages` (count in the original request),
`batch_items`, `batch_tokens`, `estimated_bytes` (the failing sub-batch),
`startup_headroom_bytes`, and `message` (a human-readable summary that also
names the effective lever). `effective_max_batch_tokens` and (on `503` only)
`suggested_max_passages` are included only when the per-item cost is known —
the BGE path has no per-token cost model, so a BGE capacity failure omits both
rather than reporting a fabricated number.

Detection, the typed error, and the GPU cache release on failure are **always
active**; `SMART_EMBEDDER_OOM_FALLBACK=false` only disables the automatic
batch-halving retry that normally attempts a smaller batch before giving up.

```bash
curl -X POST "http://127.0.0.1:8010/rerank" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "passages": ["ML is AI", "Nice weather"], "normalize": true}'
```

### `GET /health`

The one endpoint that needs no token. Returns server status, GPU info, active
embedding/reranker models, batch size, and the `autotune` object.

```bash
curl "http://127.0.0.1:8010/health"
```

```json
{
  "version": "1.2.0",
  "profile": "advanced",
  "model": null,
  "dense_embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "sparse_model_name": "omai-research/milco-650m",
  "colbert_model_name": null,
  "reranker_model": "Qwen/Qwen3-Reranker-0.6B"
}
```

### `GET /models`

The active profile and the model assigned to each vector type and to reranking.
Requires authentication.

```bash
curl -H "Authorization: Bearer <token>" "http://127.0.0.1:8010/models"
```

```json
{
  "profile": "advanced",
  "embedder": {
    "dense": "Qwen/Qwen3-Embedding-0.6B",
    "sparse": "omai-research/milco-650m",
    "colbert": null
  },
  "reranker": "Qwen/Qwen3-Reranker-0.6B"
}
```

### `GET /stats`

Uptime, total requests, total sentences, total batches, rejected requests,
hardware.

```bash
curl -H "Authorization: Bearer <token>" "http://127.0.0.1:8010/stats"
```

### `GET /docs`

Swagger is disabled by default. Set `EXPOSE_SMART_EMBEDDER_SWAGGER=true` to
expose interactive documentation at `http://127.0.0.1:8010/docs`. If
`SMART_EMBEDDER_API_TOKENS` is configured, Swagger shows the lock on `POST`
endpoints — use **Authorize** and enter only the token, without the `Bearer`
prefix.

### `GET /metrics`

Prometheus scraping endpoint, `text/plain`.

```bash
curl -H "Authorization: Bearer <token>" "http://127.0.0.1:8010/metrics"
```

#### Embedding metrics

| Metric | Type | Label |
|---|---|---|
| `embedding_requests_total` | Counter | `status`, `endpoint` |
| `embedding_requests_rejected_total` | Counter | `reason` |
| `embedding_sentences_processed_total` | Counter | - |
| `embedding_request_duration_seconds` | Histogram | `endpoint` |
| `embedding_batch_size` | Histogram | - |
| `embedding_gpu_inference_duration_seconds` | Histogram | - |
| `embedding_queue_size` | Gauge | - |
| `embedding_active_requests` | Gauge | - |
| `embedding_gpu_memory_allocated_bytes` | Gauge | Legacy process GPU allocated memory, kept for existing dashboards |
| `embedding_gpu_memory_reserved_bytes` | Gauge | Legacy process GPU reserved memory, kept for existing dashboards |
| `embedding_input_tokens_total` | Counter | `backend` — post-truncation non-padding tokens submitted to embedding backends |
| `embedding_padded_tokens_total` | Counter | `backend` — padded tensor token positions submitted to embedding backends |
| `embedding_tokenized_batches_total` | Counter | `backend` — tokenizer batches submitted to embedding backends |
| `embedding_server_info` | Info | Active models, profile, length limits, batching/runtime/compile state, version and device |
| `embedding_autotune_info` | Info | Profile state, fingerprint, objective, runtime, precision, threads, lanes, ONNX/compile decisions and batch policy |

Dividing `embedding_padded_tokens_total` by `embedding_input_tokens_total`
gives the padding overhead a batching change is meant to reduce; `benchmark.py`
derives exactly that ratio from these three counters.

#### Request-phase metrics

Where a request's wall-clock time actually goes, split by stage. All
histograms. `endpoint` is `/embeddings` or `/rerank`; `operation` is `embed` or
`rerank`.

| Metric | Type | Label |
|---|---|---|
| `api_batch_queue_wait_duration_seconds` | Histogram | `endpoint` — time waiting in the async batch queue before a batch starts |
| `api_executor_wait_duration_seconds` | Histogram | `endpoint`, `operation` — time between executor submit and worker start |
| `api_model_forward_duration_seconds` | Histogram | `endpoint`, `operation` — time inside model worker execution |
| `api_response_build_duration_seconds` | Histogram | `endpoint` — time building Pydantic response payloads |
| `api_response_serialization_duration_seconds` | Histogram | `endpoint` — time rendering the HTTP JSON response |

#### ADVANCED backend metrics

Exported only under the `advanced` profile. `backend` is `qwen_dense`,
`milco_sparse`, `qwen_reranker`, or `advanced_hybrid` for the pipeline that
drives the first two — so the halves of a hybrid request can be told apart,
where the `embedding_*` counters above aggregate them.

| Metric | Type | Label |
|---|---|---|
| `advanced_backend_requests_total` | Counter | `backend` — backend invocations, internal hybrid pipeline work included |
| `advanced_backend_input_tokens_total` | Counter | `backend` — post-truncation non-padding tokens |
| `advanced_backend_padded_tokens_total` | Counter | `backend` — padded tensor positions |
| `advanced_backend_tokenized_batches_total` | Counter | `backend` — tokenizer tensor batches |
| `advanced_backend_phase_duration_seconds` | Histogram | `backend`, `phase` — `tokenize`, `host_to_device`, `forward`, `device_to_host`, `postprocess`, `output_conversion`, `coo_to_qdrant`, plus `pipeline`/`merge_restore` on `advanced_hybrid`. Not every phase applies to every backend. |
| `advanced_backend_gpu_memory_bytes` | Gauge | `backend`, `kind` — `allocated`, `reserved`, `peak_allocated`, `peak_reserved` for the last invocation |
| `advanced_backend_process_rss_peak_bytes` | Gauge | `backend` — process peak resident memory after the last invocation |
| `advanced_effective_config_info` | Info | Effective immutable and tunable settings for the ADVANCED profile |

#### GPU process metrics

Process-level CUDA readings updated after embedding and rerank inference. They
include all loaded models and both endpoint paths.

| Metric | Type | Label |
|---|---|---|
| `gpu_memory_allocated_bytes` | Gauge | Process GPU tensor memory allocated by PyTorch |
| `gpu_memory_reserved_bytes` | Gauge | Process GPU memory reserved by the PyTorch caching allocator |
| `gpu_memory_free_bytes` | Gauge | CUDA device free memory from `torch.cuda.mem_get_info()` |
| `gpu_memory_total_bytes` | Gauge | CUDA device total memory from `torch.cuda.mem_get_info()` |

#### Reranker metrics

| Metric | Type | Label |
|---|---|---|
| `rerank_requests_total` | Counter | `status` |
| `rerank_requests_rejected_total` | Counter | `reason` |
| `rerank_pairs_processed_total` | Counter | - |
| `rerank_request_duration_seconds` | Histogram | - |
| `rerank_inference_duration_seconds` | Histogram | - |
| `rerank_active_requests` | Gauge | - |
| `rerank_batch_halvings_total` | Counter | - |
| `rerank_capacity_rejections_total` | Counter | `reason` |

#### Useful PromQL

```promql
# Throughput embedding (req/sec)
rate(embedding_requests_total[1m])

# Latency P95
histogram_quantile(0.95, rate(embedding_request_duration_seconds_bucket[5m]))

# Error rate (%)
rate(embedding_requests_total{status="error"}[5m]) / rate(embedding_requests_total[5m]) * 100

# GPU tensor memory allocated by PyTorch (GB)
gpu_memory_allocated_bytes / 1024 / 1024 / 1024

# GPU memory reserved by PyTorch caching allocator (GB)
gpu_memory_reserved_bytes / 1024 / 1024 / 1024

# CUDA device memory visible to the process (GB)
gpu_memory_free_bytes / 1024 / 1024 / 1024

# Reranker throughput (pairs/sec)
rate(rerank_pairs_processed_total[1m])
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'smart-embedder'
    static_configs:
      - targets: ['127.0.0.1:8010']
    metrics_path: '/metrics'
```

Recommended Grafana panels: embedding request rate
(`rate(embedding_requests_total[1m])`), latency P50/P95/P99
(`histogram_quantile`), queue size (`embedding_queue_size`), GPU memory
(`gpu_memory_allocated_bytes`, `gpu_memory_reserved_bytes`,
`gpu_memory_free_bytes`), rerank request rate
(`rate(rerank_requests_total[1m])`), and batch-size distribution
(`embedding_batch_size`).

## Configuration

Service-wide values use `SMART_EMBEDDER_*` in the unified root `.env`.
Backend-specific model tuning retains its accurate model-family names. The root
`.env.example` is the exhaustive, versioned reference; Compose maps its public
`SMART_EMBEDDER_*` service variables to the shorter internal names consumed by
Python, while `BGE_*`, `QWEN_*` and `MILCO_*` keep their names end to end.

### Service and safety controls

| Env var | Default | Description |
|---|---|---|
| `SMART_EMBEDDER_PORT` | `8010` | Loopback host port; container stays on 8000 |
| `EXPOSE_SMART_EMBEDDER_PORT` | `true` | `false` withdraws the host publish entirely; read by the launchers only |
| `SMART_EMBEDDER_PROFILE` | `basic` | Startup profile: `basic` or `advanced`; any other value fails startup. `basic` is used only when the variable is absent; the shipped `.env.example` sets `advanced` |
| `SMART_EMBEDDER_API_TOKENS` | required | Inbound bearer credentials as a `token:label` list (`API_TOKENS` in the container) |
| `SMART_EMBEDDER_CLIENT_TOKEN` | _(empty)_ | The middleware's credential, auto-accepted under the reserved label `@middleware` (`SERVICE_TOKEN` in the container) |
| `SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL` | `false` | Explicit opt-out for isolated loopback development; `false` applies only when the variable is absent, and the shipped `.env.example` sets `true` |
| `EXPOSE_SMART_EMBEDDER_SWAGGER` | `false` | Expose `/docs`, `/redoc`, and `/openapi.json` |
| `SMART_EMBEDDER_REQUEST_TIMEOUT` | `90` | Global HTTP timeout in seconds |
| `SMART_EMBEDDER_EMBED_GPU_TIMEOUT` | `60` | Embedding inference timeout; keep below the global timeout |
| `SMART_EMBEDDER_RERANK_GPU_TIMEOUT` | `60` | Rerank inference timeout; keep below the global timeout |
| `SMART_EMBEDDER_MAX_QUEUE_SIZE` | `200` | Maximum queued `/embeddings/` requests |
| `SMART_EMBEDDER_MAX_INFLIGHT_TOKENS` | `1048576` | Shared admitted token-work units across embedding and reranking |
| `SMART_EMBEDDER_MAX_INFLIGHT_OUTPUT_BYTES` | `268435456` | Shared estimated JSON output bytes; not a physical memory ceiling |
| `SMART_EMBEDDER_RERANK_MAX_QUEUE` | `32` | Maximum admitted `/rerank` requests |
| `SMART_EMBEDDER_RATE_LIMIT_REQUESTS_PER_MINUTE` | `3600` | Token-bucket refill rate per client IP |
| `SMART_EMBEDDER_RATE_LIMIT_BURST_SIZE` | `120` | Token-bucket burst capacity |
| `SMART_EMBEDDER_MAX_SENTENCES_PER_REQUEST` | `2048` | Maximum entries in one embedding request |
| `SMART_EMBEDDER_MAX_PASSAGES_PER_REQUEST` | `2048` | Maximum passages in one rerank request |
| `SMART_EMBEDDER_MAX_ITEM_BYTES` | `1000000` | Maximum UTF-8 size of one sentence, query, or passage. Checked after deserialization — see the row below |
| `SMART_EMBEDDER_REQUEST_BODY_MAX_BYTES` | `67108864` | Maximum bytes of one HTTP request body, counted as it streams in. Over it is 413 and the body is never materialized. Valid range: 1024..2147483648 |
| `SMART_EMBEDDER_LENGTH_AWARE_BATCHING` | `true` | Group inputs by estimated length before micro-batching, then restore API order |
| `EMBEDDER_NUM_THREADS` | autotuned on CPU | Explicit intra-op thread override |
| `EMBEDDER_INTEROP_THREADS` | runtime default | Explicit CPU inter-op thread override |
| `SMART_EMBEDDER_NATIVE_THREADS` | `4` | Cap for the native OpenMP/OpenBLAS/MKL pools (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` in the container). Those pools size to every host CPU, not the `cpus:` quota; on a many-core host that exceeds `pids_limit` and model loading crashes. Keep it equal to `cpus:` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,throw_on_cudamalloc_oom:True` | CUDA allocator tuning for the GPU target |
| `SMART_EMBEDDER_HF_HUB_DISABLE_XET` | `1` | Download transport for the preparation containers (`HF_HUB_DISABLE_XET`). `1` uses plain HTTP, which writes each file as it arrives and resumes it after a dropped connection; `0` allows Xet, which can be faster on a stable link but can lose a partly downloaded file once its retries run out |
| `SMART_EMBEDDER_VRAM_RESERVE_MB` | `512` | VRAM held back from the reranker token-budget calibration measured at startup. Valid range: 0..65536 |
| `SMART_EMBEDDER_OOM_FALLBACK` | `true` | `false` disables the automatic batch-halving retry on VRAM exhaustion; detection, the typed error and the GPU cache release stay active either way |

### BASIC / BGE controls

| Env var | Default | Description |
|---|---|---|
| `BGE_EMBED_MAX_LENGTH` | GPU `8192`; CPU `2048` | Dense, sparse and ColBERT input token limit; CPU Compose reads `BGE_EMBED_CPU_MAX_LENGTH` |
| `BGE_EMBED_CPU_MAX_LENGTH` | `2048` | CPU-only Compose cap that keeps long BGE requests within the HTTP timeout |
| `BGE_EMBED_BATCH_SIZE` | autotuned; safe fallback GPU `8` at 8GB, CPU `1` | Fixed embedding microbatch override |
| `BGE_EMBED_INFERENCE_CONCURRENCY` | autotuned on CPU; `1` on GPU | CPU embedding-lane override, clamped against effective CPUs and threads |
| `BGE_EMBED_RUNTIME` | `auto` | Select or measure `torch`, `onnx-fp32`, or `onnx-int8`; ONNX is CPU-only and INT8 remains quality-gated |
| `BGE_ONNX_EXECUTION_MODE` | `auto` | Measure or set `sequential`/`parallel` ORT graph scheduling |
| `BGE_ONNX_SPIN_POLICY` | `auto` | Measure or set `default`, `bounded`, or `disabled` ORT worker spinning |
| `BGE_TORCH_COMPILE` | `auto` | Measure eager versus compiled execution, or use `on`/`off` as an explicit override |
| `BGE_TORCH_COMPILE_MODE` | `default` | PyTorch Inductor compile mode |
| `BGE_RERANK_MAX_LENGTH` | `8192` | Query+passage limit; BAAI recommends 1024 for practical use |
| `BGE_RERANK_BATCH_SIZE` | `min(batch_size, 32)` | Reranker microbatch limit |

### ADVANCED / Qwen and MILCO controls

| Env var | Default | Description |
|---|---|---|
| `QWEN_EMBED_MAX_LENGTH` | GPU `32768`; CPU `2048` | Dense embedding token limit; CPU Compose reads `QWEN_EMBED_CPU_MAX_LENGTH` |
| `QWEN_EMBED_CPU_MAX_LENGTH` | `2048` | CPU-only Compose safety cap for the dense model |
| `QWEN_EMBED_BATCH_SIZE` | `min(batch_size, 8)` | Dense embedding microbatch limit |
| `QWEN_EMBED_MAX_BATCH_TOKENS` | `8192` | Dense padded-token bound per microbatch |
| `QWEN_EMBED_RUNTIME` | profile decision | `torch`, `onnx-fp32`, or `onnx-int8`; ONNX is CPU-only |
| `QWEN_EMBED_PRECISION`, `QWEN_EMBED_ATTENTION` | profile decision | Dense precision and attention implementation overrides |
| `QWEN_EMBED_TORCH_COMPILE`, `QWEN_EMBED_TORCH_COMPILE_MODE` | profile decision | Dense compile override and mode |
| `QWEN_EMBED_PINNED_MEMORY`, `QWEN_EMBED_PREFETCH_BATCHES` | profile decision | Dense transfer and bounded-prefetch overrides |
| `MILCO_MAX_LENGTH` | `8192` | Sparse input token limit |
| `MILCO_BATCH_SIZE` | `min(batch_size, 32)` | Sparse microbatch limit |
| `MILCO_MAX_BATCH_TOKENS` | `8192` | Sparse padded-token bound per microbatch |
| `MILCO_PROJECTION_CHUNK_TOKENS` | `128` | Exact memory-bounded lexical projection chunk |
| `MILCO_SOURCE_VIEW` | `true` | Add LexEcho source-language tokens to the shared English view |
| `MILCO_TOP_K` | `0` | Optional per-vector sparse pruning; `0` keeps all non-zero weights |
| `MILCO_RUNTIME`, `MILCO_PRECISION`, `MILCO_COMPUTE_PRECISION` | profile decision | Sparse runtime, storage precision, and compute precision overrides |
| `MILCO_TORCH_COMPILE`, `MILCO_TORCH_COMPILE_MODE` | profile decision | Sparse compile override and mode |
| `MILCO_PINNED_MEMORY`, `MILCO_PREFETCH_BATCHES` | profile decision | Sparse transfer and bounded-prefetch overrides |
| `QWEN_RERANK_MAX_LENGTH` | GPU `32768`; CPU `2048` | Reranker query+passage token limit. The GPU default is the model's own ceiling, and also this variable's maximum accepted value |
| `QWEN_RERANK_BATCH_SIZE` | GPU `8`; CPU `1` | Reranker microbatch limit |
| `QWEN_RERANK_RUNTIME`, `QWEN_RERANK_PRECISION`, `QWEN_RERANK_ATTENTION` | profile decision | Reranker runtime, precision, and attention overrides |
| `QWEN_RERANK_TORCH_COMPILE`, `QWEN_RERANK_TORCH_COMPILE_MODE` | profile decision | Reranker compile override and mode |
| `ADVANCED_SCHEDULE_MODE`, `ADVANCED_INFERENCE_LANES` | profile decision | Cross-backend schedule and lane-count overrides |
| `ADVANCED_QUEUE_BATCH`, `ADVANCED_FLUSH_TIMEOUT_MS` | profile decision | HTTP request coalescing bound and flush latency |
| `ADVANCED_CPU_SPIN_POLICY`, `ADVANCED_CPU_AFFINITY` | profile decision | CPU worker wait and affinity overrides |
| `ADVANCED_CPU_ALLOW_OVERSUBSCRIPTION` | `false` | Permit a calibrated CPU topology to oversubscribe effective CPUs |

Empty low-level ADVANCED overrides leave the decision to the validated hardware
profile; generated `*_BATCH_POLICY` and `*_MAX_BATCH_TOKENS_POLICY` values
should normally be managed by the tuner rather than edited by hand.

### Choosing a profile

The root stack reads one profile from `SMART_EMBEDDER_PROFILE`. The shipped
`.env.example` sets it to `advanced`; `basic` is the fallback the server and
Compose apply when the variable is absent entirely.

- **basic** (fallback when unset) — `BAAI/bge-m3` for dense, sparse and
  ColBERT, with the `BAAI/bge-reranker-v2-m3` reranker. The all-BGE behaviour.
- **advanced** — `Qwen/Qwen3-Embedding-0.6B` dense +
  `omai-research/milco-650m` learned-sparse (MILCO) +
  `Qwen/Qwen3-Reranker-0.6B` reranker. No ColBERT.

**Remote-code trust note.** The `advanced` profile loads
`omai-research/milco-650m` with `trust_remote_code=True`, which executes
third-party Python shipped in that model repository. This is an informed,
contained choice: the snapshot is pinned to an immutable commit in
`model_revisions.py`, `model_prepare.py` refuses any snapshot whose revision
differs, and the serving container is read-only and non-root. The `basic`
profile has no remote-code path at all (BGE-M3 is natively supported by
transformers). If executing any third-party model code is out of policy, use
`basic`.

MILCO sparse output is Qdrant-compatible: request `/embeddings/` with
`sparse_as_indices=true` to get `indices`/`values` you can pass straight into a
Qdrant `SparseVector`.

The Qwen dense path intentionally does **not** add query/document instruction
prefixes. This keeps the `/embeddings/` API transparent, but deployments
optimizing retrieval quality should benchmark task-specific Qwen formatting
separately before changing request semantics.

Compose supplies profile-specific conservative defaults when the variables are
not set in `.env`:

| Mode | `QWEN_RERANK_MAX_LENGTH` | `QWEN_RERANK_BATCH_SIZE` |
|---|---:|---:|
| GPU (`gpu` profile) | 32768 | 8 |
| CPU (`cpu` profile) | 2048 | 1 |

The GPU length is the model's ceiling rather than a conservative value, which
is deliberate: through the middleware the reranker never sees anything close to
it, because `QUERY_RERANK_MAX_CHARS` (default 1000 characters) truncates every
passage first. The cap matters only for a direct call to the published Smart
Embedder port, and it is the lever the `413 rerank_capacity_exceeded` error
names — a single query/passage pair at full length need not fit the VRAM
headroom measured at startup on a small GPU.

Raise `QWEN_RERANK_BATCH_SIZE` only after running a concrete matrix such as
`benchmark.py --url http://127.0.0.1:8010 --scenarios rerank --matrix-concurrency 1,2,4 --matrix-batch-size 4,8,16`.

### Embedding batch-size fallback

For `basic`, an explicit `BGE_EMBED_BATCH_SIZE` wins, then a valid
length-aware hardware profile, then the conservative device fallback below. The
fallback is calculated from available VRAM and the active embedding max length.
With `basic` that length is `BGE_EMBED_MAX_LENGTH=8192`; with `advanced`,
device defaults use the larger of `BGE_EMBED_MAX_LENGTH` and
`QWEN_EMBED_MAX_LENGTH`.

| Condition | batch_size | MAX_REQUESTS_IN_BATCH |
|---|---:|---:|
| GPU >= 23 GiB | 32 | 16 |
| GPU >= 15 GiB | 16 | 16 |
| GPU >= 7.5 GiB | 8 | 16 |
| GPU >= 3.5 GiB | 4 | 16 |
| Smaller GPU | 2 | 16 |
| CPU safe fallback | 1 | 8 |

These are fallback values, **not** the persisted BASIC policy — the autotuner
can select a different batch for each 128/512/1024/2048/8192-token bucket.

If the active embedding tuning length is `<=512`, the server switches to the
short-sequence profile:

| VRAM | batch_size | MAX_REQUESTS_IN_BATCH |
|---|---:|---:|
| > 8 GB | 128 | 64 |
| > 6 GB | 64 | 32 |
| > 4 GB | 32 | 16 |
| <= 4 GB | 16 | 16 |

`MULTI_GPU_DEVICES = None` (in `smart-embedder.py`) can be changed to
`['cuda:0', 'cuda:1']` for multi-GPU.

Overriding a rate limit ad hoc versus persistently:

```bash
# Ad-hoc (shell env)
SMART_EMBEDDER_RATE_LIMIT_REQUESTS_PER_MINUTE=10000 ./stack.sh up gpu

# Persistent — copy .env.example to .env and modify
cp .env.example .env
./stack.sh up gpu
```

The root launchers load only the root `.env`. The file is ignored by Git;
`.env.example` is the versioned template.

## Behaviour contracts

### Serving

- **HTTP JSON serialization** uses `ORJSONResponse`. Direct Python calls to
  `get_embeddings()` and `rerank()` still return Pydantic models. Optional
  response fields (`colbert`, `sparse`) remain present as JSON `null` when
  absent — they are not globally omitted.
- **Truncation, not rejection.** Texts longer than the active backend-specific
  token limit are truncated and reported in the response `warnings` array.
- **Rate limiting** — token bucket per IP,
  `SMART_EMBEDDER_RATE_LIMIT_REQUESTS_PER_MINUTE=3600`,
  `SMART_EMBEDDER_RATE_LIMIT_BURST_SIZE=120`. Over the limit: HTTP `429` with
  `Retry-After: 60`. The limiter uses the direct connection IP
  (`request.client.host`); behind a trusted reverse proxy, the middleware would
  need to extract the IP from `X-Forwarded-For`.
- **Request bodies are capped as they stream** —
  `SMART_EMBEDDER_REQUEST_BODY_MAX_BYTES=67108864` (64 MiB). The counting
  happens in an ASGI layer, so a body over the cap is answered `413` and never
  materialized; the per-item limits above run only once the body has been
  deserialized and cannot bound it. The layer sits inside the rate limiter and
  outside authentication — the same order the middleware uses — so an oversized
  body still costs a rate-limit token and is refused before any token is
  checked. Rejections appear in `/stats` and in
  `embedding_requests_rejected_total` / `rerank_requests_rejected_total` under
  `reason="request_body_too_large"`. The cap clears anything the middleware
  relays (its own inbound limit is 25 MiB); lowering it below that can refuse
  legitimate internal traffic.
- **Backpressure** — `/embeddings/` queue max
  `SMART_EMBEDDER_MAX_QUEUE_SIZE=200`; `/rerank` slots max
  `SMART_EMBEDDER_RERANK_MAX_QUEUE=32` (an admission bound on the shared
  GPU executor); acquire timeout 0.5 s. Rejections appear in both
  `/stats` (`rejected_requests`) and Prometheus
  (`embedding_requests_rejected_total` / `rerank_requests_rejected_total`).
- **Reranker serialization** — admitted requests wait for the shared reranker
  instance within their processing timeout, including preparation and scoring.
  Waiting for that instance does not trigger the 0.5 s admission rejection;
  deadline expiration returns 504. Cancellation releases the instance only
  after its submitted worker finishes.
- **Shared inference admission** — both endpoints reserve token-work units and
  estimated output bytes before warning tokenization or model work. The two
  limits above are independent: a request exceeding either limit receives 413
  (`request_budget_exceeded`); temporarily occupied capacity receives 503
  (`request_budget_busy`, `Retry-After: 1`). Existing queue limits still apply.
  A cancelled queued caller releases its lease; a submitted native future keeps
  the lease until physical completion, including failure or shutdown. The
  caller retains ownership through response construction. Warning tokenization
  runs on the existing embedding worker before its model call.

  Estimates use the larger of original/NFKC-normalized UTF-8 lengths plus an
  eight-token allowance, capped by active model limits. BASIC shares one forward
  pass; ADVANCED sums selected dense and sparse work. Rerank repeats each query
  per passage with a 1,024-unit allowance for its fixed instruction/template.
  Output estimates allow 32 bytes per dense/ColBERT coordinate, 48 per sparse
  entry, six per echoed UTF-8 byte and per-item/envelope metadata. BASIC sparse
  keys are bounded by input length; MILCO reserves its English vocabulary plus
  source-view input keys, honoring `MILCO_TOP_K` when set. The 256-MiB default
  accommodates the output reservation of the middleware's usual four concurrent
  32-item batches with these conservative sparse estimates. Input work has its
  own limit: byte-based estimates can refuse work that an exact tokenizer count
  would admit. For 413, reduce the batch, shorten inputs or disable unneeded
  output before retrying; a capacity 503 can be retried after active work drains.

  ColBERT is particularly conservative: reserving 8192 rows of 1024 coordinates
  at 32 bytes each already consumes 256 MiB before other output. Such a request
  exceeds the shipped budget even when its actual tokenizer output is shorter.
  Shorten the input or turn off ColBERT when those vectors are unnecessary;
  changing the budget requires an operator's capacity assessment.

  These are admission estimates, not RAM/VRAM or exact serialized-size caps;
  decoded request bodies, resident models, tensor workspaces, Python object
  overhead and responses awaiting transmission have separate lifetimes.
  `/stats` and Prometheus expose `inference_reserved_tokens` and
  `inference_reserved_output_bytes`; refusals use the existing rejection
  counters with the two reason codes above.
- **Timeouts** — `SMART_EMBEDDER_REQUEST_TIMEOUT=90s` is the global HTTP
  timeout (504 to the caller); `SMART_EMBEDDER_EMBED_GPU_TIMEOUT=60s` bounds
  embedding batch inference on the thread pool;
  `SMART_EMBEDDER_RERANK_GPU_TIMEOUT=60s` bounds rerank inference and should
  stay below the global timeout. Timeouts are tracked as
  `embedding_requests_total{status="timeout"}` /
  `rerank_requests_total{status="timeout"}`.
- **GPU execution** — BASIC and safe ADVANCED profiles use one GPU lane. A
  measured ADVANCED profile may use multiple lanes so embedding and reranking
  can overlap, but one reranker instance is serialized from preparation through
  scoring. Peak VRAM includes resident weights, activations and workspaces, so
  automatic overlap requires a measured profile for that host. Explicit
  operator overrides remain available but do not provide that measurement.
  Mixed tuner samples include reranker preparation and scoring on the worker,
  alongside embedding; the runtime source fingerprint invalidates decisions
  when this scheduling or measurement code changes.
- **Graceful shutdown** — closes admission and drains accepted requests for
  at most 30 s. Remaining callers receive 503 and the processing loop stops.
  The lifespan then awaits physical completion of the shared GPU executor
  before closing model resources. This native-work wait can exceed the request
  drain budget; the container's stop grace remains the final process deadline.

### Hardware autotuning

On the first start of a profile, a **separate pre-start process** fingerprints
the effective cgroup/container CPU, RAM, NVIDIA GPU, software stack and model
revision, benchmarks runtime/batch/thread candidates under quality, stability
and memory gates, and atomically persists the decision in the hardware state
volume. Later starts reuse that profile. A hardware/software change, or a
change to tuning mode, objective, runtime, compile policy or compile mode,
invalidates it.

Profile writers hold an OS lock for their entire critical section. Its
`.lock.guard` sidecar stays on disk; its presence alone does not mean a writer
is active. The OS releases ownership when the writer exits, while the temporary
`.lock` marker retains bounded stale recovery. Do not remove a guard file while
writers may be running.

**Configuration precedence:** explicit `BGE_*`/thread overrides → a valid
hardware profile → conservative device-aware defaults. **The tuner never
rewrites `.env`.** If calibration fails or times out, Uvicorn still starts with
safe defaults.

| Variable | Values / default | Purpose |
|---|---|---|
| `SMART_EMBEDDER_AUTOTUNE` | `off`, `quick`, `full`; `quick` | First-start behaviour |
| `SMART_EMBEDDER_TUNING_OBJECTIVE` | `balanced`, `throughput`, `latency`; `balanced` | Selection objective |
| `SMART_EMBEDDER_AUTOTUNE_FORCE` | `false`, `true`; `false` | Ignore a reusable profile whenever preparation runs while true |
| `SMART_EMBEDDER_AUTOTUNE_MAX_SECONDS` | `120` quick, configurable | Target measurement budget; checked between bounded candidates. ADVANCED also uses it when its own budget is unset, and its quick budget is then fixed instead of adaptive |
| `SMART_EMBEDDER_AUTOTUNE_PROFILE` | path in `/app/runtime_state/autotune` | Persisted BASIC operational profile |

For startup mode selection and a single forced preparation request, see the
[launcher controls](../scripts/README.md#startup-calibration-and-console-controls).
The environment force setting above is persistent until changed.

#### BASIC (BGE) tuner

`quick` probes deterministic token lengths, applies quality, stability and
memory gates, then stores the selected runtime, threads, inference lanes and
length-aware microbatch policy. On NVIDIA, `torch.compile` is an **A/B-measured
candidate**, not an assumed optimization. On CPU, the profile also includes the
measured ONNX graph scheduler, worker spin policy and inter-op thread count.

`auto` can select ONNX FP32 only on CPU and only after dense, sparse and
ColBERT parity gates. NVIDIA GPUs keep PyTorch FP16: the RTX 4060 audit found
ORT CUDA and TensorRT materially slower and/or too memory-intensive. **ONNX
INT8 is intentionally rejected** because it did not meet the quality gate.

On NVIDIA, compile auto-selection compares the selected 512-token full-output
batch (dense + sparse + ColBERT) in eager and compiled form. For the
`throughput` objective it promotes Inductor only with at least 5% more measured
token/s and no more than 25% P95 regression; unsafe, unstable, timed-out or
below-threshold candidates stay eager. `balanced` and `latency` use similarly
conservative gates. Quick mode uses two measured samples for every GPU batch
candidate before choosing a token bucket. The cold compile cost is excluded
from steady-state token/s but recorded in the profile as `warmup_seconds`.
Compiled artifacts use a non-persistent tmpfs and are rebuilt after container
recreation rather than being writable persistent code.

CPU quick tuning measures the production BASIC output (dense + sparse +
ColBERT), not a dense-only proxy. Torch and ONNX are compared at the same
thread count. If ONNX wins, every thread candidate rebuilds the actual ORT
session, then batch size and concurrent lanes are measured under the effective
cgroup CPU limit. The selected topology must pass a separate 512-token canary;
`full` additionally validates 1,024, 2,048 and 8,192 tokens. Unsafe quality,
memory, stability or deadline results fall back to Torch FP32, one thread, one
lane and batch one.

Larger CPU allocations get their own search. Quick mode considers the assigned
physical and logical/SMT limits, batches up to 8 and up to four lanes; full
mode extends batches to 16 and lanes to eight. On CPUs with at least eight
effective workers and enough remaining budget, ORT sequential versus parallel
execution is measured locally, as is default versus bounded spin. Explicit ONNX
policy variables skip those optional stages and become part of profile
compatibility.

This follows ONNX Runtime's official guidance that intra-op pools are fixed per
session and that parallel scheduling, spinning and affinity must be benchmarked
for the actual model and topology:
[ORT thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
ORT graph optimizations remain enabled at `ORT_ENABLE_ALL`:
[ORT graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).
The compile contract follows the official PyTorch
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
mode/dynamic-shape behaviour and
[Inductor cache guidance](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).

**Deliberately not exposed:** CPU BF16 as a serving mode — the current
FlagEmbedding BGE-M3 wrapper cannot convert its sparse/ColBERT BF16 tensors to
NumPy, and the real probe fails before parity can be established. ONNX INT8
remains rejected by the existing quality evidence. ONNX FP32 still dispatches
through the native ISA selected by ORT. CPU `torch.compile` is not assumed to
be an optimization; the slim CPU image intentionally keeps the compiler
toolchain out.

The current hardware profile uses **schema v4**. Its fingerprint includes GPU
name and UUID, compute capability, VRAM, SM count, effective CPU/container
limits and assigned core topology, CPU ISA, model revision, Python, NumPy,
PyTorch, FlagEmbedding, Transformers, tokenizers, ONNX Runtime, CUDA, NVIDIA
driver, Triton, and a SHA-256 of the serving/calibration sources. A change to
any relevant field invalidates the persisted decision and causes a new
calibration rather than carrying one host's result to different hardware or a
materially different software stack.

RTX 50xx support is intentionally measurement-driven. A 16 GiB GPU explores
short-input batches up to 16 and an at-least-23 GiB GPU up to 32; the limits
decrease with sequence length and every candidate remains subject to quality,
stability and the 80% memory gate. A different GPU always receives a new
fingerprint and local benchmark. If its compiled candidate clears the promotion
thresholds, compile is persisted; otherwise it keeps eager while still
benefiting from the locally selected batch policy. Simulated RTX 5080/5090 runs
verify this search and invalidation control flow, but do not claim unmeasured
physical token/s.

#### ADVANCED (Qwen/MILCO) tuner

The `advanced` profile has its **own** hardware profile at
`SMART_EMBEDDER_ADVANCED_AUTOTUNE_PROFILE`; it never consumes the BASIC/BGE
profile, and the same tuner never runs across both profiles. `quick` and `full`
use the same isolated pre-start, fingerprint, quality, memory and
atomic-persistence contract described above, but benchmark Qwen dense, MILCO
sparse, hybrid, Qwen reranking and mixed traffic separately, plus request
traffic through the serving queue on CUDA. `full` is an
explicit operator audit and reserves time for a final isolated long-input
canary; **if that canary fails, all tentative selections are rolled back**
rather than partially persisted. When long probes were deferred from the
baseline, their reserved time includes both a baseline-duration probe allowance
and the measured model-load/quality overhead, within the existing calibration
budget. Exhausting that window still rolls all tentative selections back.

A candidate whose estimate does not fit the remaining budget is recorded as
`skipped-budget` and the search continues with the next one, so a cheaper later
candidate (for example a reranker-only probe) is still measured.

Each candidate starts from the policy promoted so far, and it is compared with
that same policy. When an earlier promotion changed what one of its target
backends runs (request traffic, for instance, runs the dense batch policy), that
backend is first re-measured under the current policy, recorded as
`<candidate>-reference`, and its cost counts against the budget. A candidate
whose reference cannot be re-measured is recorded as `skipped-stale-reference`.

On CUDA the calibrated padded-token budget per microbatch follows the same VRAM
tier as the benchmark workloads: 8,192 tokens below 15 GiB, 16,384 from 15 GiB
and 32,768 from 23 GiB. A larger microbatch is only persisted after the memory
and quality gates pass, and the serving process still lowers the reranker
budget when its measured VRAM headroom is smaller.

An unset quick budget adapts to the host. The baseline may run for up to three
times the default budget (720 s on CUDA, 1,080 s on CPU); once it is measured,
the calibration deadline becomes the elapsed time plus the estimated cost of
every planned candidate, never below the default and never above that ceiling.
The pre-start process allows the ceiling plus 60 s, and the persisted profile
records the configured, planned and applied budget under `budget`. Setting
`SMART_EMBEDDER_ADVANCED_AUTOTUNE_MAX_SECONDS` (or the shared budget) makes the
budget a fixed limit again. Passing `--max-seconds` to
`advanced_autotune_runner.py` directly also sets a fixed limit, whatever its
value; the pre-start process never passes it.

The profile also records the host's CPU power management under
`host_cpu_power`: how many assigned CPUs use each frequency governor, energy
performance preference and driver, their maximum frequency, the groups of CPUs
sharing each of those settings with their minimum and maximum frequency limits
(`frequency_domains`), and the median and
maximum frequency sampled every 0.5 s on the core running the baseline probes.
When that median is below half of the maximum, calibration adds a warning that
the preparation container logs, and the serving process logs it again at every
start with that profile: slow single-thread kernel dispatch then limits GPU
throughput. The tuner only reads sysfs and never changes host power settings.
Those settings, down to each group's minimum and maximum frequency limits, are
part of the ADVANCED fingerprint, so changing them (for example switching to
the `performance` governor or capping some cores) recalibrates on the next
start; the sampled frequency is not.

On CUDA the worker also samples GPU utilisation with `nvidia-smi` while the
baseline probes run. A median below 50% (from at least four samples) means the
GPU waits on host-side work such as single-thread kernel dispatch; calibration
then measures larger batch policies and request coalescing first, and in full
mode CUDA-graph compilation next, keeping every other candidate in its usual
order. The diagnosis is persisted under `gpu_dispatch` and its warning is
logged with the CPU power warning. The probing thread's CPU time is not used
as the signal, because CUDA synchronisation may spin-wait on the CPU while the
GPU is still busy.

On CUDA the baseline also measures real request traffic through a fresh copy
of the serving request processor: 128-token, single-sentence dense and sparse
requests from one client and from 32 concurrent clients. The
`schedule-queue-batch-32` candidate lets that processor combine up to 32 queued
requests into one model call, waiting at most 10 ms for more to arrive. It is
promoted only when those scenarios gain throughput and the P95 gate still
holds, including for the lone client that pays the flush wait.

The offline ADVANCED worker imports the serving model wrappers without opening
an HTTP listener. When the prep container has no credentials, the import uses
a fresh, process-local random credential and restores the environment afterward.
No serving credential is passed to calibration or persisted in its results;
the serving authentication checks remain enabled. Structured worker failures
are preserved in the entrypoint's fallback result, including nonzero exit codes.

On CPU, Qwen dense compares Torch FP32 with a locally exported and
integrity-checked ONNX Runtime FP32 graph. Serving reuses a valid prepared
artifact without writing an export lock to the read-only runtime-state volume;
missing or corrupt artifacts still require the writable preparation stage.
Dynamic INT8 is measured but may
only be selected after the versioned retrieval gate; the 4-core/16-GiB physical
audit rejected INT8 for semantic drift and selected ONNX FP32. With all three
models co-resident, dense input passed the final 1,024/2,048-token canary. The
CPU Compose profile therefore defaults `QWEN_EMBED_CPU_MAX_LENGTH` to 2,048;
raising it is supported, but changes the fingerprint and requires a fresh
memory audit.

MILCO is a true learned sparse lexical backend. Its primary view projects
different languages into one shared English lexical space. With the default
`MILCO_SOURCE_VIEW=true`, LexEcho also emits source-language token indices so
proper names, rare entities and terms that should not be translated survive
retrieval. These are real sparse COO values converted directly to Qdrant
indices/values; **the tuner never changes `MILCO_TOP_K`, pruning, the English
pivot mapping or LexEcho semantics.** The projection is exact and processed in
bounded token chunks (`MILCO_PROJECTION_CHUNK_TOKENS`, default 128) so the
8,192-token sparse limit stays practical without vocabulary-wide sequence
materialization.

| Variable | Values / default | Purpose |
|---|---|---|
| `SMART_EMBEDDER_ADVANCED_AUTOTUNE_PROFILE` | `/app/runtime_state/autotune/advanced-hardware-profile.json` | Profile independent from BASIC |
| `SMART_EMBEDDER_ADVANCED_AUTOTUNE_MAX_SECONDS` | unset: quick adapts from GPU `240` / CPU `360` up to 3× | Fixed ADVANCED budget; setting it (or the shared budget) disables the adaptive quick budget |
| `QWEN_EMBED_RUNTIME` | `torch`, `onnx-fp32`, `onnx-int8` | Explicit dense runtime; ORT is CPU-only |
| `SMART_EMBEDDER_ADVANCED_ONNX_CACHE` | `/app/runtime_state/autotune/onnx` | Versioned export and manifest cache |
| `QWEN_EMBED_CPU_MAX_LENGTH` | `2048` | CPU Compose dense safety cap |
| `MILCO_MAX_LENGTH` | `8192` | Sparse input limit |
| `MILCO_SOURCE_VIEW` | `true` | Preserve the LexEcho source-language view |
| `MILCO_PROJECTION_CHUNK_TOKENS` | `128` | Exact projection chunk size |

## Capabilities and limits

**You can:**

- Embed a batch of texts into dense and sparse vectors (plus ColBERT under
  `basic`) from one call, in Qdrant-native sparse form on request.
- Rerank passages against a query with a cross-encoder, in the same service.
- Run the whole thing on CPU with no GPU at all, or on an 8 GB consumer GPU.
- Let the pre-start autotuner pick runtime, threads, lanes and a length-aware
  batch policy for your actual hardware, or override every one of those
  explicitly.
- Scrape rich Prometheus metrics including per-phase timings and padding
  efficiency.
- Run a full latency/throughput/padding benchmark matrix from `benchmark.py`.
- Detect a profile mismatch downstream through
  `embedding_contract_fingerprint`.

**You cannot:**

- Switch model profile without restarting the process.
- Request ColBERT vectors under the `advanced` profile — the field is not valid
  there.
- Compare BGE raw logits with Qwen rerank scores; they are different scales.
- Get an error instead of truncation for over-long input — check `warnings`.
- Assume multi-lane overlap is safe without the profile's measured hardware
  evidence. One reranker instance is serialized even when embedding may use
  another approved lane.
- Reach the service from another host by configuration alone; and with
  `EXPOSE_SMART_EMBEDDER_PORT=false` you cannot reach it from the host at all —
  only the middleware can, over the private Compose network.
- Run `tests/test_server.py` or `benchmark.py` against a stack started with
  `EXPOSE_SMART_EMBEDDER_PORT=false`; both target the published host port.
- Move a persisted autotune profile to different hardware — the fingerprint
  invalidates it.
- Update a model by pointing at a branch; revisions are pinned commits.

## Recipes

### 1. Configure

From the repository root, copy `.env.example` to `.env` and choose the model
preset with `SMART_EMBEDDER_PROFILE=basic` or `advanced`. Set a private
`SMART_EMBEDDER_API_TOKENS`; use
`SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL=true` only for explicit isolated
loopback development.

```powershell
Copy-Item .env.example .env
```

### 2. Start the server

```powershell
.\stack.ps1 up gpu
```

```bash
./stack.sh up gpu
```

Use `cpu` instead of `gpu` on hosts without the NVIDIA runtime. The model
profile is read from `.env` without an interactive prompt. For a standalone
deployment without middleware or Qdrant, set `EMBEDDER_ONLY=true` in the root
`.env` before running the same command; the launcher then builds and starts
only the selected Smart Embedder container.

Follow startup with `./stack.sh logs gpu` (or `.\stack.ps1 logs gpu`) from the
repository root, and wait for:

```
INFO - Reranker ready.
INFO - Server ready to accept requests
```

The server is then available at `http://127.0.0.1:8010`.

### 3. Docker specifics

**Prerequisites:** Docker Desktop / Engine meeting the
[launcher requirements](../scripts/README.md#prerequisites), and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
for the GPU profile:

```bash
nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**First startup:**

```bash
# Verify the CUDA tag exists before building
docker pull nvidia/cuda:12.6.3-runtime-ubuntu22.04

# Build and GPU startup (first time: downloads the selected profile's models)
./stack.sh up gpu

# CPU execution
./stack.sh up cpu
```

Models are saved in the named volume `smart-embedder-hf-cache`. A non-exposed
init container writes the pinned model and hardware-state volumes; the HTTP
server mounts both **read-only**. Inductor and Triton caches live in a
non-persistent executable tmpfs because they load generated native code. On a
fresh named volume, a minimal bootstrap uses only `CHOWN`, `SETUID` and
`SETGID`, then drops permanently to UID/GID 1000 before any model or network
library is imported. Subsequent starts reuse the prepared models and compatible
state.

The first Docker startup with the `advanced` profile downloads Qwen and MILCO
models into the cache volume, so it can take longer than `basic` on an empty
cache. Every runtime model is downloaded at the immutable commit recorded in
`model_revisions.py`, not at a moving branch. MILCO embeds the English
sub-model configuration and weights in its own pinned snapshot; this backend
uses only its public multilingual tokenizer and does **not** require access to
the gated `naver/splade-v3` repository.

**Useful commands:**

```bash
./stack.sh up gpu       # startup in background
./stack.sh logs gpu     # real-time logs
./stack.sh down         # stop
./stack.sh up gpu       # rebuild after code changes

# Low-level no-cache rebuild for diagnostics (the launcher is the supported path)
docker compose --env-file .env --profile gpu -f compose.yml build --no-cache
```

**Verify the GPU inside the container:**

```bash
docker compose --env-file .env --profile gpu -f compose.yml run --rm smart-embedder-gpu python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

### 4. Control host exposure

By default the server is bound to `127.0.0.1:8010` (localhost only). To change
the host port, set `SMART_EMBEDDER_PORT` in the root `.env`; the container
remains on port 8000 internally.

**Do not remove the loopback bind or expose Smart Embedder on a public IP.**
Access from another machine must use a trusted private network, personal VPN or
authenticated tunnel, with `SMART_EMBEDDER_API_TOKENS` configured.

**Withdrawing the host port entirely.** Nothing in the stack needs the
published port: the middleware reaches this service over the private Compose
network at `http://smart-embedder:8000`. The publish exists only so host-side
tooling can reach a running container.

```bash
EXPOSE_SMART_EMBEDDER_PORT=false
```

The launchers then apply `compose.embedder-private.yml`, which withdraws that
single publish. **This is the right setting for a deployment**: the published
port is the only route that reaches this service without passing through the
middleware, and therefore without its bearer auth, its rate limiting and its
`QUERY_RERANK_MAX_CHARS` truncation. `SMART_EMBEDDER_API_TOKENS` still guards
that route — unless `SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL=true`, where
this switch is the only thing left.

**What stops working:** everything targeting `http://127.0.0.1:8010` —
`tests/test_server.py`, `benchmark.py`, the `curl` examples above, and a
host-side Prometheus scrape. Turn the switch back to `true` (or run
`docker exec` inside the container) when you need them. The switch is read by
the launchers only; a bare `docker compose up` always publishes the port.

### 5. Roll out autotuning in stages

1. Start with `SMART_EMBEDDER_AUTOTUNE=off` to retain safe defaults.
2. Inspect the search space without loading models or writing the profile:

   ```powershell
   docker compose --profile cpu run --rm --entrypoint python3 `
     smart-embedder-cpu basic_autotune_runner.py --dry-run
   ```

   (`advanced_autotune_runner.py --dry-run` for the ADVANCED tuner.)
3. Validate explicit settings with autotune still `off` — for example
   `BGE_EMBED_RUNTIME=torch`, `EMBEDDER_NUM_THREADS=4` and a conservative
   `BGE_EMBED_BATCH_SIZE=1`.
4. Remove those explicit overrides, set `SMART_EMBEDDER_AUTOTUNE=quick`,
   `BGE_EMBED_RUNTIME=auto` and `BGE_TORCH_COMPILE=auto`, then apply the settings
   with the [launcher controls](../scripts/README.md#startup-calibration-and-console-controls).
   Use the CLI force action for one requested recalibration.

Check the `autotune` object returned by `/health` (and, for ADVANCED, the
`advanced_config` object plus the backend/phase Prometheus metrics), the
`embedding_autotune_info` Prometheus metric, and the `Hardware autotune
pre-start result` log line before expanding the rollout.
Restart once and require `action=reused`. Use `full` only for an explicit audit:
it adds three-sample exploratory stages and long-sequence canaries, so it is
not a first-start default. Its default budget is already longer (600 s for
BASIC, 1,800 s for ADVANCED); a budget you set yourself is a fixed limit, so
raise it rather than lower it when a full audit needs more time.

**Roll back** immediately with `SMART_EMBEDDER_AUTOTUNE=off` or explicit safe
backend overrides.

## Verification

### Unit suites (fakes only, no model downloads)

Run from the repository root:

```bash
python3 -m ruff check server-embedder-reranker
python3 -m ruff format --check server-embedder-reranker
(cd server-embedder-reranker && python3 -m unittest discover -s tests -p "test_*.py")
```

### Integration suite (needs a running server)

```bat
python tests/test_server.py
python tests/test_server.py --url http://127.0.0.1:8010
python tests/test_server.py --token <token>
```

Expected output: **17/17 tests passed**. With `--token` and
`SMART_EMBEDDER_API_TOKENS` configured, the authentication check is included
and the expected output is **18/18 tests passed**. This suite needs the
published host port, so it cannot reach a stack started with
`EXPOSE_SMART_EMBEDDER_PORT=false`.

### Benchmark

`benchmark.py` measures latency (avg/p50/p95/p99), server-side processing time,
warnings, effective input token/s, throughput, and Prometheus deltas across
scenarios. The tokenizer advertised by `/health` is loaded before the timed
window, so token counts include special tokens and post-truncation length.

```bat
py benchmark.py --concurrency 8 --requests 100 --batch-size 4
py benchmark.py --scenarios embed_dense,embed_sparse,rerank
py benchmark.py --scenarios embed_dense --matrix-input-tokens 32,128,512 --matrix-batch-size 4,8,16 --matrix-concurrency 1,4,8
```

| Flag | Default | Description |
|---|---|---|
| `--url` | `http://127.0.0.1:8010` | Server target |
| `--token` | process env, then root `.env` | Bearer token if the server requires auth; an explicit CLI value takes precedence |
| `--output-json` | `""` | Atomically save the comparable benchmark matrix as JSON |
| `--label` | `""` | Optional report label such as baseline, candidate, or canary |
| `--tokenizer-model` | active `/health` model | Override the tokenizer used for exact input token counting |
| `--tokenizer-max-length` | active `/health` limit | Override the tokenizer truncation limit |
| `--reranker-tokenizer-model` | active `/health` reranker model | Override the tokenizer used for rerank token counting |
| `--reranker-tokenizer-max-length` | active `/health` limit | Override the reranker token limit (0 = /health value) |
| `--no-token-count` | false | Disable exact token counting and omit token/s |
| `--input-tokens` | `0` | Exact tokens per synthetic embedding input; `0` uses the short default corpus |
| `--concurrency` | `8` | Concurrent requests in-flight |
| `--requests` | `100` | Requests per scenario |
| `--batch-size` | `4` | Sentences/passages per request |
| `--warmup` | `5` | Warmup requests (excluded from metrics) |
| `--timeout` | `60` | Timeout for a single request |
| `--max-batch-size` | `128` | Local guardrail on payload limits; `0` disables |
| `--scenarios` | `all` | CSV or `all`: `embed_dense,embed_sparse,embed_sparse_indices,embed_hybrid,embed_full,embed_colbert,rerank,mixed`; `embed_hybrid` is dense+sparse without ColBERT, `embed_full` includes every supported output, and `mixed` cycles heterogeneous input lengths to expose padding overhead |
| `--sleep-between` | `0` | Pause between scenarios (use `65` if a rate limit is active) |
| `--matrix-batch-size` | unset | Comma-separated batch sizes for matrix runs (e.g. `4,8,16`) |
| `--matrix-concurrency` | unset | Comma-separated concurrency levels for matrix runs (e.g. `1,2,4,8`) |
| `--matrix-input-tokens` | unset | Comma-separated exact embedding input lengths (e.g. `32,128,512`) |
| `--mixed-input-tokens` | unset | Cycle exact lengths inside each request (e.g. `32,128,512,1024`) to expose padding overhead |
| `--matrix-sleep` | `1.0` | Seconds to sleep between matrix combinations |
| `--no-post-check` | false | Skip post-run `/health` and `/metrics` diagnostics |

Output: ASCII tables for pre-flight `/health`, selected scenarios, measured
results, failure details, and post-run Prometheus deltas. Result columns are
`Reqs / OK / OK% / Fail / Conc / Wall / Req/s / Units/s / Tok/item / Tok/s /
Avg / SrvAvg / P50 / P95 / P99 / Min / Max / Warn / Statuses`. `Tok/s` is
effective successful input throughput over the measured wall-clock window;
reranking remains separate and reports `-` until pair-token accounting is
implemented.

The post-run diagnostics derive useful token/s and padded token/s over model
worker time from `embedding_input_tokens_total`,
`embedding_padded_tokens_total`, `embedding_tokenized_batches_total` and
`api_model_forward_duration_seconds`. The padding ratio is useful tokens
divided by padded tensor positions. These deltas include configured warmup
requests; use `--warmup 0` when comparing them directly with measured HTTP
`Tok/s` after the service is already warm.

The benchmark reads `SMART_EMBEDDER_API_TOKENS` from the process environment —
presenting the first entry's token — and, when it is not exported, from the
repository's single root `.env`. Before running warmups or measured requests it
verifies access to the protected endpoints.

> Default rate limits (3600 req/min, burst 120) are tuned for benchmarks on a
> single client at `conc<=16`. For extreme stress testing, set
> `SMART_EMBEDDER_RATE_LIMIT_REQUESTS_PER_MINUTE=1000000` in the root `.env`,
> then restart the selected profile.

**Matrix runs** sweep batch size and concurrency to find the tuning point for
your hardware. Like the integration suite, they need the published host port:

```bat
py benchmark.py --url http://127.0.0.1:8010 --scenarios embed_full --matrix-batch-size 4,8,16 --matrix-concurrency 1,2,4,8 --requests 20
py benchmark.py --url http://127.0.0.1:8010 --scenarios rerank --matrix-batch-size 4,8,16 --matrix-concurrency 1,2,4 --requests 20
py benchmark.py --url http://127.0.0.1:8010 --scenarios embed_hybrid --mixed-input-tokens 32,128,512,1024 --batch-size 16 --concurrency 4 --warmup 0 --requests 20
```

Each combination prints a heading
`=== input_tokens=<n> batch_size=<b> concurrency=<c> ===` followed by its own
results table, verdict and Prometheus deltas. Compare effective token/s, P95
latency, timeout count and the phase-level metrics
(`api_batch_queue_wait_duration_seconds`, `api_executor_wait_duration_seconds`,
`api_model_forward_duration_seconds`) to identify bottlenecks.

### Before opening a change

Run the unit suites, the linters, and — from the repository root — the
repository-wide checks in [`../AGENTS.md`](../AGENTS.md). Fixed test counts are
intentionally omitted from this file because the suites continue to grow.

## Failure modes

| Symptom | Cause | Next step |
|---|---|---|
| **Server won't start** | Missing CUDA or dependencies. | `python3 -c "import torch; print(torch.cuda.is_available())"`, then `python3 -m pip install -r requirements/embedder-gpu.txt --upgrade`. |
| `429 Too Many Requests` | Client exceeds the token bucket. | Raise `SMART_EMBEDDER_RATE_LIMIT_REQUESTS_PER_MINUTE` or reduce call frequency. |
| `413` with `detail: "Request body too large..."` | The body exceeded `SMART_EMBEDDER_REQUEST_BODY_MAX_BYTES` and was never read into memory. Distinct from the rerank-capacity 413, which carries a structured `error` field. | Send fewer items per request, or raise the cap together with the container's memory limit. |
| `503 Service Unavailable` | The queue is full. | Raise `SMART_EMBEDDER_MAX_QUEUE_SIZE`, or scale horizontally behind a load balancer. |
| `503` with `error: "rerank_vram_pressure"` | Transient GPU pressure; a batch of one would fit. | Retry (a `Retry-After: 2` header is sent), or send at most `suggested_max_passages` passages. |
| `413` with `error: "rerank_capacity_exceeded"` | One query/passage pair alone exceeds the startup headroom. Not retryable. | Shorten passages (`QUERY_RERANK_MAX_CHARS`) or lower `BGE_RERANK_MAX_LENGTH` (`basic`) / `QWEN_RERANK_MAX_LENGTH` (`advanced`). |
| `504 Gateway Timeout` | Inference exceeded `SMART_EMBEDDER_EMBED_GPU_TIMEOUT` or `SMART_EMBEDDER_RERANK_GPU_TIMEOUT`. | Reduce batch size, or check GPU availability. |
| **Prometheus metrics not visible** | Target unreachable or port blocked. | `curl http://127.0.0.1:8010/metrics`; verify the target in `prometheus.yml` and that the port is not firewalled. |
| **GPU not detected in the container** | NVIDIA Container Toolkit misconfigured. | `docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu22.04 nvidia-smi`; reinstall the toolkit and restart Docker if it fails. |
| `manifest for nvidia/cuda:12.6.3-runtime-ubuntu22.04 not found` | The pinned CUDA base tag moved. | Update the pinned base in the root `Dockerfile` **only after validation**. |
| **Stack does not reach the serving container on first startup** | The profile's non-exposed init container is still downloading pinned models or failed; the serving container waits for it, so its healthcheck has not started. | Inspect `./stack.sh status gpu` and `./stack.sh logs gpu` (use `cpu` for that profile), resolve the network/cache/disk failure, and run `up` again. Increasing the serving healthcheck's `start_period` does not fix an init failure. |
| **Nothing answers on `127.0.0.1:8010`** | `EXPOSE_SMART_EMBEDDER_PORT=false` withdrew the publish. | Expected. Use the middleware, or turn the switch back on for host-side tooling. |

## Invariants

- **Naming is the profile boundary.** `basic_*` must not be imported by
  `advanced_*` code or vice versa; unprefixed modules stay shared.
- **The profile is chosen once, at process start.** Do not add a runtime
  switch — collections carry an embedding contract that assumes otherwise.
- **The tuner never edits `.env`,** and never changes `MILCO_TOP_K`, pruning,
  the English pivot mapping or LexEcho semantics.
- **Model revisions are immutable commits** in `model_revisions.py`. A model
  update means bumping a revision and rerunning the CPU/GPU smoke tests.
- **Embedding and rerank share one GPU worker.** Do not parallelise them to
  "use the GPU better"; peak VRAM is the reason.
- **`/health` stays the only unauthenticated endpoint.**
- **The loopback publish is not negotiable**, and
  `EXPOSE_SMART_EMBEDDER_PORT=false` is the correct deployment posture.
  `scripts/verify_compose_security.py` validates both published-port shapes
  for CPU, all-GPU and selected-GPU deployments; see the
  [launcher reference](../scripts/README.md#automatic-startup-and-gpu-selection).
- **The serving container mounts model and state volumes read-only**; only the
  non-exposed init container writes them, and it drops `CHOWN`/`SETUID`/
  `SETGID` before importing any model or network library.
- **Compiled Torch/Triton/ONNX artifacts live on non-persistent tmpfs**, not on
  a writable persistent volume — they are generated native code.
- **Quality gates are not advisory.** A candidate that fails dense/sparse/
  ColBERT parity, the memory gate or the stability gate is not promoted, no
  matter how fast it measured.

## Extending it

- **Adding a request field** — the schemas live in `smart-embedder.py`
  alongside the endpoint. Decide explicitly whether the field is valid in both
  profiles; `return_colbert` is the precedent for one that is not.
- **Adding a backend** — follow the `basic_*` / `advanced_*` naming, register
  it in the profile's wrapper, and give it a quality gate in the matching
  autotune module before it can be selected automatically.
- **Adding a metric** — keep the existing label vocabulary (`backend`,
  `phase`, `endpoint`, `operation`) and add a row to the matching table above.
- **Adding a setting** — the root `.env.example` is the exhaustive template;
  Compose maps public `SMART_EMBEDDER_*` names to internal ones, while
  `BGE_*`/`QWEN_*`/`MILCO_*` pass through unchanged. Add the row to the
  matching table here too.
- **Updating a model** — bump the pinned revision in `model_revisions.py`,
  rebuild, and rerun the CPU **and** GPU smoke tests. A fingerprint change
  forces recalibration, which is intended.
- **Changing a dependency pin** — regenerate the matching
  `requirements/locks/*.txt` from the rebuilt, verified image
  (`python -m pip freeze --all`) and commit both files together.
- **Multi-GPU** — `MULTI_GPU_DEVICES` in `smart-embedder.py` is the entry
  point; revisit lane measurement, per-reranker serialization and memory gates
  before assigning work across devices.

## Measured baselines

Recorded evidence from real calibration and benchmark runs. These are
**measurements of specific hosts**, not hard-coded tiers, guarantees, or claims
about hardware that was never measured. They are kept because they document why
certain defaults were chosen.

### RTX 4060 Laptop 8 GB — BASIC closure

The original closure used `quick`, the `throughput` objective and
`BGE_TORCH_COMPILE=auto`. It completed 18 candidates in 51.7 s without fallback
and selected PyTorch FP16, one inference lane, and this length-aware batch
policy:

| Maximum input tokens | 128 | 512 | 1024 | 2048 | 8192 |
|---:|---:|---:|---:|---:|---:|
| Selected batch | 8 | 8 | 4 | 1 | 1 |

Two independent GPU calibrations converged to the same policy. A normal restart
then reused the production profile in 4.2 s instead of benchmarking again. The
`/health` `autotune` object reported `state=active`,
`torch_compile_policy=auto`, `torch_compile=false`,
`torch_compile_reason=gain-below-promotion-threshold`, and no fallback.

After the schema-v4 CPU work, a separate RTX 4060 regression profile completed
18 candidates in 54.71 s without fallback and kept Torch FP16, eager execution
and one lane. Its fresh local policy was `8/4/4/1/1`: at 512 tokens, batch 4
measured 29,402 median token/s versus 29,270 for batch 8, while reducing P95
from 151 ms to 70 ms. This demonstrates that the CPU ONNX controls do not enter
the CUDA path, and why GPU batch values are measurements rather than permanent
model constants.

The first cold-cache compile gate measured an eager median of 30,624 token/s
and a compiled median of 28,178 token/s: compiled lost 8.0%, regressed P95 by
9.0% and required 71.72 s of warm-up. With the persistent cache populated, the
final A/B measured 30,289 eager versus 30,153 compiled token/s (compiled
-0.45%), with compiled P95 1.4% higher and a 19.96 s warm-up. Both candidates
passed dense, sparse and ColBERT quality gates and used about 2.39 GiB, so
eager was retained strictly for performance, not correctness or memory.

The final HTTP `embed_hybrid` validation used exact 512-token inputs, request
batch 8, concurrency 8 and two runs of 30 measured requests. It completed 60/60
responses with zero warnings at 26,471 and 26,889 HTTP token/s (median 26,680),
with worker throughput of 27,529 and 29,811 token/s and padding efficiency
1.000 — 4.5% above the earlier eager A/B median of 25,540 token/s.

### Ryzen 7 7435HS, 4 CPUs / 16 GiB — CPU closure

With `quick`, `throughput` and runtime `auto`, schema-v4 calibration completed
13 candidates in 98.45 s without fallback. It selected ONNX FP32, one intra-op
thread per lane, four lanes, sequential graph execution, bounded spin, and
batch one for every length bucket. A normal restart reused that exact
source-digested profile in 4.42 s.

Final HTTP validation: exact 128-token inputs, full BASIC output, request batch
8, concurrency 2, two warm-ups, 10 measured requests.

| Measurement | Before CPU tuner | Final schema v4 | Change |
|---|---:|---:|---:|
| HTTP useful token/s | 213 | 250 | +17.4% |
| Worker useful token/s | 240 | 305 | +27.1% |
| Wall time | 48.14 s | 40.91 s | -15.0% |
| P95 latency | 10.34 s | 8.49 s | -17.9% |
| Successful responses | 10/10 | 10/10 | unchanged |

A separate selected batch-two run reached 261 HTTP / 298 worker token/s; a
batch-one/default-spin run reached 263/306. The final bounded-spin run reached
250/305 — stable model-worker throughput with normal end-to-end variation.
Because batch two was not materially faster, the 3% promotion margin correctly
retained the simpler batch-one topology.

The closure was verified with the complete embedder and middleware test suites,
repository-wide Ruff checks, Dockerfile build checks, CPU/GPU Compose
validation, real image builds, and healthy container smoke tests.

### Historical controlled baseline (2026-07-14)

Before the Transformers 5 upgrade; `basic` profile, RTX 4060 Laptop 8 GB,
server `transformers==4.57.3`, HTTP batch=16, concurrency=4, 20 measured
requests per bucket.

| Tokens/item | Dense Tok/s | Dense P95 | Dense+sparse Tok/s | Dense+sparse P95 |
|---:|---:|---:|---:|---:|
| 32 | 7,123 | 333.7ms | 7,053 | 332.7ms |
| 128 | 26,244 | 344.0ms | 25,533 | 334.4ms |
| 512 | 34,969 | 1.17s | 33,538 | 988.0ms |
| 1,024 | 34,504 | 2.38s | 33,210 | 2.50s |

A 100-request sustained dense+sparse run at 512 tokens/item delivered 31,963
Tok/s with 100% success and P95 1.26 s. GPU sampling measured 68% average
utilization (89% peak), 3,520 MiB peak VRAM, 83 W average power and 86 °C peak
temperature. A batch/concurrency sweep found batch=16, concurrency=4 to be the
maximum-throughput point; batch=8, concurrency=4 retained about 95% of that
throughput with materially lower P95. Observed throughput on this GPU is
roughly 29–45 req/s at `conc=8`, depending on the scenario.

**M1.1 padding audit** (same GPU and hybrid settings, warmup=0):

| Workload | Useful tokens | Padded tokens | Padding efficiency | HTTP Tok/s | Worker useful Tok/s | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Uniform 424 tokens/item | 135,680 | 135,680 | 100.00% | 28,595 | 30,920 | 1.28s |
| Mixed 32/128/512/1,024 (mean 424) | 135,680 | 327,680 | 41.41% | 12,844 | 13,421 | 2.54s |

At identical useful-token volume, mixed lengths caused 2.41x tensor work, 2.30x
model-worker time and a 55.1% end-to-end throughput loss — which made
length-aware request grouping the next optimization target. A separate uniform
512-token check reached 30,507 HTTP Tok/s and 33,436 worker Tok/s at 100%
padding efficiency.

**M1.2 length-aware batching** (same mixed workload, new code on both sides of
the A/B):

| Setting | Padded tokens | Padding efficiency | HTTP Tok/s | Worker useful Tok/s | P95 |
|---|---:|---:|---:|---:|---:|
| Length-aware OFF | 327,680 | 41.41% | 13,274 | 13,671 | 2.24s |
| Length-aware ON | 141,440 | 95.93% | 27,428 | 29,261 | 1.25s |

Stable character-length ordering before backend micro-batching, followed by
restoring the original API order, improved HTTP throughput by 106.6%, worker
useful-token throughput by 114.0%, reduced padded positions by 56.8% and
reduced P95 by 44.2%. The optimized mixed workload retained 97.8% of the
adjacent uniform 424-token result (28,057 HTTP Tok/s). A real OFF/ON response
comparison over heterogeneous inputs preserved text order and sparse indices
with zero maximum absolute difference for both normalized dense values and
sparse weights.

**M1.3 BGE microbatch sweep** (length-aware ON, 40 mixed requests):

| BGE microbatch | Padding efficiency | HTTP Tok/s | Worker useful Tok/s | P95 | Sampled max VRAM |
|---:|---:|---:|---:|---:|---:|
| 6 | 94.96% | 26,774 | 29,570 | 1.24s | 3,678 MiB |
| 8 | 98.24% | 29,684 | 31,713 | 1.20s | 3,572 MiB |
| 12 | 87.06% | 26,334 | 28,878 | 1.31s | 3,829 MiB |
| 16 | 84.94% | 25,548 | 27,832 | 1.65s | 3,844 MiB |

Microbatch 8 improved HTTP throughput by 10.9% and worker throughput by 7.2%
over the previous default of 6. Larger non-aligned batches crossed the
boundaries between length groups created from variable HTTP aggregation and
lost more to padding than they gained from larger GPU work units. Uniform
512-token traffic reached 32,957 HTTP Tok/s and 34,831 worker Tok/s at batch 8.
Safety probes with 8 items succeeded at 2,048, 4,096 and the full 8,192-token
limit; the latter used 5,143 MiB sampled maximum VRAM. GPU long-sequence
defaults therefore use power-of-two VRAM tiers (8 at ~8 GB, 16 at 16 GB, 32 at
24 GB), with `BGE_EMBED_BATCH_SIZE` available for hardware-specific override.

## License

This service's code is covered by the repository's GNU Affero General Public
License v3.0 — see [`../LICENSE`](../LICENSE). Section 13 matters here in
particular: this service is reached over HTTP, so exposing a modified
version to other people over a network obliges you to offer them its source.

The models are separate and keep their own licenses: `BAAI/bge-m3` and
`BAAI/bge-m3-unsupervised` are MIT; `BAAI/bge-reranker-v2-m3`,
`Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B` and
`omai-research/milco-650m` are Apache-2.0. They are downloaded at the pinned
revisions in `model_revisions.py` and never redistributed here, so using the
selected profile means accepting that profile's model licenses in addition to
this repository's.

## Related documents

| Document | Covers |
|---|---|
| [`../README.md`](../README.md) | The stack as a whole. |
| [`../scripts/README.md`](../scripts/README.md) | Building, starting, inspecting and stopping the stack. |
| [`../middleware-qdrant-manager/README.md`](../middleware-qdrant-manager/README.md) | The only in-stack caller of this service, and where `QUERY_RERANK_MAX_CHARS` lives. |
| [`../smart-qdrant-mcp-server/README.md`](../smart-qdrant-mcp-server/README.md) | The MCP surface over the middleware. |
| [`../AGENTS.md`](../AGENTS.md) | Repository-wide conventions and the full verification pass. |
| `../.env.example` | The exhaustive, versioned configuration template. |

### External references

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
