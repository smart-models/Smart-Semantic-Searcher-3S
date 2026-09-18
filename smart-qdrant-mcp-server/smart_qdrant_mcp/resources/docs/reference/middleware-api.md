# middleware-qdrant-manager — the REST API clients talk to

> **Read this when** you need to answer what the stack's HTTP API can and
> cannot do, look up an endpoint, a request field or an environment variable,
> explain hybrid search or ingest behaviour, or change middleware code.
> **Audience** — an AI agent working in or on this repository, and any human
> reading over its shoulder.
> **Source of truth** — this file for the REST contract, the configuration
> surface and the behavioural guarantees; `qdrant_middleware/core/config.py`
> for the validated ranges; the root `.env.example` for deployment defaults.
> **Prerequisite** — the root [`README.md`](../README.md) for how the three
> services fit together, and [`../scripts/README.md`](../scripts/README.md) for
> starting the stack.

---

## Purpose and boundaries

A FastAPI service that sits between clients and a Qdrant vector database. It
handles embedding, ingestion, hybrid search, point operations, vector
regeneration, and collection lifecycle so clients use one authenticated API
instead of calling Qdrant and Smart Embedder directly.

```
Client → middleware-qdrant-manager → Smart Embedder  (embed / rerank)
                                   → Qdrant           (store / search)
```

**What it owns:**

1. **Collections** — create, inspect, verify, update, reindex, or delete Qdrant
   collections pre-configured for Smart Embedder hybrid search (dense 1024-dim
   + sparse, TurboQuant, HNSW).
2. **Ingest** — accept text chunks, embed them, upsert them into Qdrant.
3. **Search** — embed a query, let Qdrant run server-side RRF/DBSF fusion,
   optionally rerank through Smart Embedder's cross-encoder, return ranked
   hits.
4. **Operate** — liveness, dependency readiness, health, runtime counters, and
   Prometheus metrics, without querying dependencies on every monitoring
   request.

**What it is not:**

- Not an embedding model host. Every vector comes from Smart Embedder.
- Not a Qdrant replacement or proxy for arbitrary Qdrant features. It exposes a
  deliberately narrower, opinionated surface.
- Not a chunker. Chunk boundaries are the caller's decision; the middleware
  stores what it is given.
- Not internet-facing. Its port is published on `127.0.0.1` unless
  `MIDDLEWARE_BIND_HOST` names a private address, which requires authentication.

## Mental model

Eight facts explain most of the surprising behaviour.

1. **Point IDs are deterministic:** `UUID5(source_id:chunk_id)`. Re-ingesting a
   source overwrites rather than duplicates. This is why every write is
   idempotent and safe to retry.
2. **Embedding finishes before the first write.** A failed embedding call
   leaves stored data untouched — there is no half-ingested state to clean up.
3. **Bearer auth is not a middleware layer.** `require_bearer_auth` is a
   route-level dependency, so it runs *inside* all four middleware layers. A
   request over `REQUEST_BODY_MAX_BYTES` is rejected before its token is ever
   checked, and the unauthenticated monitoring router still passes through
   every layer.
4. **`/readyz` reflects a cached probe**, refreshed every
   `HEALTH_PROBE_INTERVAL` seconds. It does not probe on request.
5. **A collection carries an embedding-contract fingerprint** stamped at
   creation from the active Smart Embedder profile. Switching profiles makes
   existing collections report as incompatible rather than silently mixing
   vector spaces.
6. **Vector reindex is a durable background job.** While one is active on a
   collection, mutations that could invalidate it are rejected — including
   recreate. Reindex admission and every same-collection mutation share one
   in-process lease: a write already admitted completes before preflight, and
   a later write receives the documented 409.
7. **Two settings are module constants, not `Settings` fields, on purpose.**
   `LOOPBACK_HOSTS` and `INGEST_DATA_ROOT`. pydantic-settings would populate
   any field lacking an explicit alias from the environment, which would let
   the loopback classification and the file-ingest sandbox disable themselves.
8. **`text` and `source_id` are service-owned payload fields.** No payload
   delete or clear operation can remove them.

### Middleware stack (outermost → innermost)

`timeout+metrics` → `shutdown guard` → `rate limit` → `body-size` →
[`console headers`] → `route`

`FastAPI.add_middleware` inserts at position 0, so the *last* registration in
`main.py` (`TimeoutMetricsMiddleware`) ends up outermost.

The console-headers layer exists only when `EXPOSE_FRONTEND=true`, and is
innermost because it decorates a response rather than deciding whether to
produce one: it attaches the CSP and the companion headers to `/` and
`/assets/*` on the way out, and leaves every JSON API response untouched.

## Map of the code

The runtime app is `qdrant_middleware.main:app`. `main.create_app()` builds an
independent FastAPI app per `Settings` instance — tests use it directly to get
isolated apps.

| Path | Role | Change it when |
|---|---|---|
| `qdrant_middleware/main.py` | App factory, middleware registration, router mounting. | The middleware order or a router's protection changes. |
| `qdrant_middleware/core/config.py` | `Settings` — the single source of truth for every environment variable, its default, range and cross-field invariant. Also the non-tunable module constants. | Any configuration change. |
| `qdrant_middleware/api/dependencies.py`, `api/utils.py` | Route dependencies and shared request helpers. | Authentication/readiness dependencies or common route handling change. |
| `qdrant_middleware/middleware/` | Body limits, shutdown protection, rate limiting and timeout/metrics handling. | Cross-cutting request admission or response handling changes. |
| `qdrant_middleware/schemas/` | Collection, embedding-contract and reindex models. Point/query request models also live beside their routers. | Validation or a public JSON contract changes. |
| `qdrant_middleware/core/mutation_coordinator.py`, `services/mutation_guard.py` | Coordinate collection mutations with reindex activity. | A new write path or concurrent operation is introduced. |
| `qdrant_middleware/core/lifespan.py` | Opens and closes the Qdrant client across the app lifespan. | Client lifecycle changes. |
| `qdrant_middleware/core/runtime.py` | Per-app runtime state: counters, embedder handle and lock. | Shared mutable state changes. |
| `qdrant_middleware/api/routers/` | `collections`, `points`, `query` (all behind `require_bearer_auth` + `require_operational_readiness`), and `monitoring` (unprotected). | An endpoint is added or its contract changes. |
| `qdrant_middleware/clients/embedder.py` | The Smart Embedder client. | The embedder's HTTP contract changes. |
| `qdrant_middleware/clients/retry.py` | The shared exponential-backoff retry policy for transient 5xx/429/timeout/connection errors, against both Qdrant and Smart Embedder. | Retry semantics change. |
| `qdrant_middleware/repositories/qdrant_collections.py` | Primary gateway for collection schema, metadata and payload-index I/O, including their caches and retry policy. | Collection/schema Qdrant access. |
| `qdrant_middleware/api/routers/points.py` | Point CRUD/ingest endpoints and their direct point-level Qdrant calls. | Point reads or writes. |
| `qdrant_middleware/services/` | Domain-specific query, export, health and reindex access plus filter/schema business logic. | Behaviour owned by one of those workflows. |
| `qdrant_middleware/services/vector_reindex.py` | The pausable/resumable/cancelable reindex job with durable batch-boundary checkpoints. | Reindex semantics change. |
| `frontend/dist/` | The static console, served in-process when `EXPOSE_FRONTEND=true`. **Baked into the image** — rebuild and restart before testing it. | The console changes. |
| `frontend/dist/assets/app.js`, `models.js`, `api.js`, `style.css` | Console orchestration, pure model helpers, HTTP transport and styling. | A UI interaction, payload builder or presentation changes. |
| `frontend/package.json`, `frontend/playwright*.config.js` | Browser-test commands, dependencies and separation of default/external campaigns. | Test execution or browser prerequisites change. |
| `benchmark.py` | Live API benchmark that creates and deletes an isolated collection. | Endpoint benchmark scenarios change; never treat it as an offline smoke test. |
| `frontend/tests/` | The Playwright E2E suite — see [its README](frontend/tests/README.md). | Console behaviour changes. |
| `data/` | Read-only ingest sandbox. | Never in code; stage corpus files here. |
| `exports/` | Read-write export destination. | Never in code. |
| `tests/` | pytest suite; fakes the Qdrant and Smart Embedder clients. | Always, alongside the change. |

### Agent reading path

Start with the affected endpoint in `api/routers/`, its request model and
its dependency chain; follow only the service/repository/client calls it makes.
For console changes, start with `frontend/dist/assets/models.js` for pure
validation and serialization, then the owning operation in `app.js`. There is
no separate transpiled frontend source tree to edit instead of `dist/`.

For a write, identify the mutation guard, embedding-before-write boundary,
retry policy and cache invalidation before changing it. Use
`tests/test_qdrant_access_architecture.py` to check the declared Qdrant access
owners. For a shared REST change, inspect the corresponding MCP tool and its
contract tests too. The [testbook](frontend/tests/testbook/test-book-e2e.md)
defines UI and agent acceptance; inventory assignments are not execution proof.

## Interface reference

Interactive API docs (when `EXPOSE_MIDDLEWARE_SWAGGER=true`):
`http://localhost:6336/docs`.

### Monitoring

| Method | Path | Description |
|---|---|---|
| GET | `/livez` | Liveness probe — is this process alive? |
| GET | `/readyz` | Readiness probe — are Qdrant and Smart Embedder reachable? |
| GET | `/health` | Combined dependency health check |
| GET | `/stats` | Uptime and request counters |
| GET | `/metrics` | Prometheus metrics |

`/livez` is always unauthenticated: it answers `{"status": "alive"}` and nothing
else, for the container healthcheck. The other four are unauthenticated only
while `MIDDLEWARE_BIND_HOST` is loopback, which keeps the telemetry surface
local. On a private publish they require the bearer token like every other
route, because they report collection names, models and request counters. Note
the asymmetry with
Smart Embedder, which *requires* the bearer token for its own `/stats` and
`/metrics` (its global auth middleware exempts only `/health`). Prometheus label
values here are route paths, so neither endpoint leaks per-caller data.

### Collections

| Method | Path | Description |
|---|---|---|
| GET | `/collections` | List all collections |
| GET | `/collections/{name}` | Get schema report for a collection |
| POST | `/collections` | Create a collection (or repair missing indexes) |
| POST | `/collections/verify` | Verify schema without modifying anything |
| PATCH/PUT | `/collections/{name}/schema-toggles` | Update mutable schema toggles online (`dense_on_disk`, `quantization`, `sparse_idf`, `sparse_on_disk`, `on_disk_payload`) |
| POST | `/collections/{name}/reindex/preflight` | Validate whether vectors can be regenerated without changing the collection |
| POST | `/collections/{name}/reindex` | Start confirmed background vector regeneration |
| GET | `/collections/{name}/reindex` | Read persisted reindex progress and local task state |
| POST | `/collections/{name}/reindex/pause` | Pause at the next durable batch boundary |
| POST | `/collections/{name}/reindex/resume` | Resume from the last committed offset |
| POST | `/collections/{name}/reindex/cancel` | Cancel while retaining the active source-vector generation |
| GET | `/collections/{name}/payload-indexes` | List payload indexes for a collection |
| POST | `/collections/{name}/payload-indexes` | Create user-managed payload indexes (idempotent; type changes drop and recreate) |
| DELETE | `/collections/{name}/payload-indexes/{field}` | Delete a user-managed payload index |
| GET | `/collections/{name}/payload-keys` | Sample points and list observed payload keys |
| DELETE | `/collections/{name}` | Delete a collection (requires `confirm_delete` query param equal to the collection name) |

**Create:**
```json
POST /collections
{
  "collection_name": "my-docs",
  "recreate": false,
  "dense_on_disk": true,
  "quantization": true,
  "sparse_idf": true,
  "sparse_on_disk": true
}
```

**Recreate (destructive, deletes all data):**
```json
POST /collections
{
  "collection_name": "my-docs",
  "recreate": true,
  "confirm_recreate": "my-docs",
  "dense_on_disk": true,
  "quantization": true,
  "sparse_idf": true,
  "sparse_on_disk": true
}
```

**Update mutable schema toggles online:**
```json
PATCH /collections/my-docs/schema-toggles
{
  "dense_on_disk": false,
  "quantization": true
}
```

The PATCH/PUT body is partial, but at least one of `dense_on_disk`,
`quantization`, `sparse_idf`, `sparse_on_disk` or `on_disk_payload` is
required. Qdrant applies these changes online, but they can be expensive on
large collections.

Reindex preflight is read-only. Starting a job requires `confirm_collection` to
equal the path collection name exactly. Pause and cancel take effect at durable
batch boundaries; resume continues from the last committed offset. If all batches
were committed before finalization failed, resume retries verification and cutover
without embedding the points again. While
regeneration is active, collection and point mutations that could invalidate
the job are rejected.

### Points (ingestion & CRUD)

| Method | Path | Description |
|---|---|---|
| POST | `/points/ingest` | Ingest a JSON file from the `data/` folder |
| POST | `/points/ingest-chunks` | Ingest chunks sent inline in the request body |
| POST | `/points/export` | Export a whole collection's payload as one JSON document |
| POST | `/points` | Embed and upsert a single point |
| POST | `/points/batch` | Embed and upsert a batch of points |
| POST | `/points/count` | Count points matching optional payload/ID filters |
| POST | `/points/facet` | Count distinct values of an indexed payload field |
| POST | `/points/scroll` | Page through points by payload filters (no vector search) |
| GET | `/points/{id}` | Retrieve a point's payload |
| POST | `/points/retrieve` | Retrieve multiple points by ID |
| DELETE | `/points/{id}` | Delete a single point |
| DELETE | `/points` | Bulk delete by IDs |
| POST | `/points/delete-by-filter` | Delete every point matching a non-empty filter |
| PATCH | `/points/{id}/payload` | Merge non-text fields into a point's payload |
| PUT | `/points/{id}/payload` | Replace non-text fields in a point's payload (service fields preserved) |
| POST | `/points/{id}/payload/delete-keys` | Delete non-protected payload keys from one point |
| POST | `/points/{id}/payload/clear` | Clear one payload while retaining protected service fields |
| PATCH | `/points/payload` | Merge fields into every point matching a non-empty filter |
| POST | `/points/payload/delete-keys` | Delete non-protected keys from filtered points |
| POST | `/points/payload/clear` | Clear sampled non-protected keys from filtered points |

`text` is written only by the embed-and-upsert endpoints: payload-only merge
and replace requests that include it are rejected. Use `POST /points` or
`POST /points/batch` to change text so its vector is regenerated at the same
time. `text` and `source_id` are service-owned payload fields and cannot be
removed by the delete-keys operations, including through quoted, nested or
array payload paths. Filtered clear discovers removable keys by
sampling at most 1,000 points; a key absent from that sample is left unchanged.

**Ingest a file** (the file must already be in `data/`):
```json
POST /points/ingest
{
  "file_path": "my-docs.json",
  "collection_name": "my-docs",
  "replace_source": true
}
```

**Ingest file format:**
```json
{
  "chunks": [
    { "id": 0, "text": "First chunk text...", "any_extra_field": "value" },
    { "id": 1, "text": "Second chunk text..." }
  ]
}
```

Every chunk key becomes payload. `id` is the chunker's per-source **ordinal**,
not the point id — the point id is `UUID5(source_id:id)`, derived from it. The
ordinal is stored under `chunk_id_key` (default `id`), and the middleware
*relocates* it there rather than copying, so `"chunk_id_key": "chunk_ordinal"`
leaves no `id` in the payload. It must differ from `source_id_key` and cannot
be `text`, `overflow_details`, `overlap_sentences_count` or `split_metadata`; a
breach is HTTP 400. The chunk objects themselves always use the literal key
`id` — `chunk_id_key` names where it is *stored*, not how it is *read*.

**Ingest inline chunks:**
```json
POST /points/ingest-chunks
{
  "source_id": "doc-001",
  "collection_name": "my-docs",
  "chunks": [
    { "id": 0, "text": "Hello world" }
  ]
}
```

**Export a whole collection.** `POST /points/export` walks every point of one
collection and serializes its payload — never its vectors — into a single
document, `{"chunks": [...], "metadata": {...}}`: the same shape the ingest
endpoints read, so the export round-trips back into ingestion. Each chunk is
the stored payload verbatim plus a trailing `point_id` holding the Qdrant point
id serialized as a string; it may represent either a UUID or an unsigned
integer. `point_id` is a separate key precisely because the payload may carry
the chunk's own integer `id`; drop `point_id` before re-ingesting, and supply
integer per-source ordinals under `id` if the payload has none.
There is deliberately no `filters` on this request — export is whole-collection
only.

`destination` picks where the document goes:

- `"file"` (default) — written under `EXPORT_DATA_ROOT`; the response is a
  small JSON result object describing the write:
  ```json
  POST /points/export
  { "collection_name": "my-docs" }
  ```
  ```json
  {
    "ok": true,
    "collection_name": "my-docs",
    "n_chunks": 4213,
    "path": "/app/middleware-qdrant-manager/exports/my-docs_export_2026-08-07-11-05-00_a1b2c3d4e5f6.json",
    "bytes": 5142330
  }
  ```
  An empty collection reports `"n_chunks": 0, "path": null` — nothing is
  written.
- `"body"` — the export document itself, served as an `application/json`
  attachment with an ASCII fallback and an RFC 5987 UTF-8 `filename*` parameter
  instead of the result object above:
  ```json
  POST /points/export
  { "collection_name": "my-docs", "destination": "body" }
  ```

Both destinations emit identical bytes — the export document itself, not the
response envelope shown above; only where the document lands differs.
The service derives a bounded filename from the collection name, timestamp and
random suffix, validates it as a basename, and resolves it under
`EXPORT_DATA_ROOT` before writing. This remains true even if an upstream
collection-name validator changes.

The collection's exact point count is checked **before** the walk starts, and
each scroll page is checked against the same cap before it is accumulated. Over
`EXPORT_MAX_POINTS` (default 50000) the request is refused with HTTP 413 and
nothing is written; the cap is not narrowable per-request since there is no
`filters` — raise `EXPORT_MAX_POINTS` together with the container's memory
limit, or fall back to an operator-level Qdrant volume snapshot. A failure
part-way through the walk (e.g. Qdrant becomes unreachable) returns HTTP 502
and also writes nothing — a write is all-or-nothing, there is no partial export
file on disk. JSON serialization runs off the event-loop thread and discards
each page's payload objects after encoding. `EXPORT_MAX_BYTES` (default 64 MiB)
caps the entire UTF-8 document, including metadata, escaping and formatting;
exceeding it returns 413 with nothing published. This output cap excludes the
upstream scroll page, Python object overhead and response copies, so it is not
a physical memory ceiling. The walk is not a transactional collection snapshot.
`EXPORT_MAX_CONCURRENT` (default 1) admits that many exports per app process;
excess requests receive 503 with `Retry-After: 1`, before counting or scrolling.
Cancellation drains an active serializer or filesystem worker before releasing
its reservation and never publishes a late file. File preparation and durability
flushes run outside the event loop; publication uses one atomic rename.
`POST /points/export` is exempt from the per-request timeout
middleware: a whole-collection walk is many sequential scroll pages, so the
request clock is the wrong bound for it. The point, byte and concurrency
limits bound admitted export work instead.

### Query

| Method | Path | Description |
|---|---|---|
| POST | `/query/search` | Hybrid search with optional reranking |

**Search:**
```json
POST /query/search
{
  "query": "what is retrieval augmented generation?",
  "collection_name": "my-docs",
  "limit": 5,
  "rerank": true,
  "rerank_top_k": 20
}
```

| Parameter | Default | Description |
|---|---|---|
| `query` | — | Query string |
| `collection_name` | env default | Target collection |
| `limit` | 10 | Number of results to return |
| `candidate_limit` | 50 | How many candidates to retrieve before fusion/rerank |
| `rerank` | `true` | Enable cross-encoder reranking |
| `rerank_top_k` | 20 | Number of candidates fed to the reranker |
| `rerank_normalize` | `true` | Ask Smart Embedder to normalize reranker scores |
| `fusion` | `rrf` | Fusion method: `rrf` or `dbsf` |
| `rrf_k` | `null` | Optional RRF rank constant (RRF only) |
| `rrf_weights` | `null` | Optional `[dense, sparse]` non-negative weights (RRF only) |
| `filters` | `null` | Legacy flat or advanced Qdrant-like payload filters |
| `with_payload` | `null` | Optional payload fields to return; `null` returns the full payload |
| `offset` | `0` | Offset inside the fixed fused candidate pool |
| `score_threshold` | `null` | Minimum fused score applied before reranking |
| `hnsw_ef` | env default | Per-request HNSW search breadth |
| `exact` | `null` | Override approximate (`false`) or exact (`true`) search |
| `rescore` | env default | Override quantized-vector rescoring |
| `oversampling` | env default | Override quantized-search oversampling |
| `timeout` | middleware request timeout | Qdrant query timeout in seconds |

### Payload filters

Top-level `filters` must be **either** legacy flat syntax **or** advanced
logical syntax. Mixing logical keys with legacy field keys is rejected.

**Legacy flat filters** — scalars become Qdrant `MatchValue`, arrays become
`MatchAny`, multiple fields are combined with `must`/AND:

```json
{
  "query": "rag",
  "filters": {
    "type": "paragraph",
    "source_id": ["doc-1", "doc-2"]
  }
}
```

**Advanced Qdrant-like filters:**

```json
{
  "query": "rag",
  "filters": {
    "must": [
      {"key": "type", "match": {"value": "paragraph"}},
      {"key": "token_count", "range": {"gte": 100, "lt": 500}}
    ],
    "must_not": [
      {"key": "status", "match": {"value": "deleted"}}
    ]
  }
}
```

Supported operators: `match.value`, `match.any`, `match.except`, `match.text`,
`match.text_any`, `match.phrase`, `range`, `values_count`, `is_empty`,
`is_null`, `geo_radius`, `geo_bounding_box`, `geo_polygon`, `has_id`,
`has_vector`, `nested`, `must`, `should`, `must_not`, and `min_should`.

Filter rules worth knowing before you write one:

- **Field paths and literal keys are distinct.** In filters and payload-index
  requests, `a.b` traverses an object and `items[].label` traverses an array.
  To address a literal top-level key containing that punctuation, include quotes
  in the field value (JSON: `{"key": "\"a.b\""}`). Simple names containing spaces,
  such as `Header 1`, are quoted automatically. Payload field selection keeps
  its literal top-level semantics. An existing literal index does not satisfy a
  traversal index; create the intended index explicitly if a previous version
  created a quoted literal by mistake.
- **Datetime ranges use the same public `range` JSON shape as numeric ranges.**
  Numeric bounds map to Qdrant `Range`; ISO string bounds map to
  `DatetimeRange`. In this installed `qdrant-client` version both model types
  are stored on `FieldCondition.range`. Mixed numeric/string bounds are
  rejected.
- **`has_id`** accepts the same point-id formats Qdrant accepts, including
  integers and UUID strings.
- **`has_vector`** targets vector names and is not checked against
  payload-field allowlists.
- **`is_empty` and `is_null` only accept `true`.** To express the opposite, put
  the same condition under `must_not`.
- **The max-condition counter counts every parsed condition object**, including
  the `min_should` wrapper and each condition inside it.
- **`match.text` / `match.text_any` need a Qdrant full-text payload index** on
  the field, or the query can be very slow.

### Reranker capacity errors (advanced profile)

If Smart Embedder rejects a rerank sub-batch as permanently oversized, it
returns `413` with a structured `detail` body (`error:
"rerank_capacity_exceeded"`, plus sizing fields — see the Smart Embedder
README's `/rerank` section for the full shape). The middleware forwards that
`413` and its `detail` to the caller unchanged rather than converting it to
`502`, because the body names the actionable lever (passage length, not
`rerank_top_k`) and translating it would lose that information.

This pass-through applies **only** to `413`. Every other Smart Embedder error
on this path — including the retryable `503` `rerank_vram_pressure` case, which
the retry policy already retries before giving up — is still reported as `502`.

## Configuration

Copy the root `.env.example` to `.env` and edit it. The tables below cover
every explicitly named environment setting the middleware accepts. The root
`.env.example` supplies the common deployment values; Compose replaces
`SMART_EMBEDDER_URL` with its private service DNS name.

Numeric settings carry validated ranges and cross-field invariants; the
authoritative ranges live in `.env.example` and are enforced by
`core/config.py`.

### Service, dependencies, and security

| Variable | Default | Description |
|---|---:|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint. |
| `QDRANT_API_KEY` | _(empty)_ | Optional Qdrant API key. |
| `QDRANT_COLLECTION` | _(empty)_ | Default collection; requests must supply `collection_name` when empty. |
| `QDRANT_TIMEOUT` | `30` | Qdrant client timeout in seconds. |
| `QDRANT_ON_DISK_PAYLOAD` | `true` | Store collection payload data on disk by default. |
| `QDRANT_SPARSE_FLOAT16` | `true` | Store sparse-vector values as float16. |
| `QDRANT_API_HOST` | `127.0.0.1` | Interface on which the middleware listens. |
| `MIDDLEWARE_PORT` | `6336` | Published/local middleware port. |
| `MIDDLEWARE_BIND_HOST` | `127.0.0.1` | Host address Compose publishes the port on: loopback or a private RFC1918 IPv4 literal. A non-loopback value requires authentication, and then every monitoring route except `/livez` requires the token too. It never reaches a socket; that is `QDRANT_API_HOST`. |
| `MIDDLEWARE_API_TOKENS` | _(empty)_ | Operator-provisioned inbound bearer credentials as a `token:label` list, accepted alongside `SERVICE_TOKEN`. |
| `MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL` | `false` | Permit unauthenticated access only when both the operator token list and the auto-accepted service token are empty. |
| `SERVICE_TOKEN` | _(empty)_ | Delivered by Compose from `MIDDLEWARE_CLIENT_TOKEN`: the credential the MCP server presents, auto-accepted under the reserved label `@mcp`. |
| `EXPOSE_FRONTEND` | `true` | Register the static console and `/assets/config.js`. |
| `EXPOSE_MIDDLEWARE_SWAGGER` | `true` | Expose `/docs`, `/redoc`, and `/openapi.json`. |
| `MIDDLEWARE_REQUEST_TIMEOUT` | `30` | Middleware request timeout in seconds. |
| `REQUEST_BODY_MAX_BYTES` | `26214400` | Maximum accepted HTTP request-body size. |
| `MIDDLEWARE_RATE_LIMIT_REQUESTS_PER_MINUTE` | `300` | Sustained per-client request rate. |
| `MIDDLEWARE_RATE_LIMIT_BURST_SIZE` | `180` | Per-client token-bucket capacity. |
| `SMART_EMBEDDER_URL` | `http://localhost:8010` | Smart Embedder endpoint. |
| `SMART_EMBEDDER_CLIENT_TOKEN` | _(empty)_ | Bearer token this service presents when calling Smart Embedder. |
| `SMART_EMBEDDER_TIMEOUT` | `60` | Smart Embedder request timeout in seconds. |
| `SMART_EMBEDDER_MAX_BATCH_SIZE` | `32` | Maximum texts per embedding sub-batch. |
| `SMART_EMBEDDER_RERANK_BATCH_SIZE` | `32` | Maximum passages per rerank sub-batch. |
| `SMART_EMBEDDER_EMBED_CONCURRENCY` | `4` | Maximum concurrent embedding sub-batches. |

### Query defaults and limits

| Variable | Default | Description |
|---|---:|---|
| `QUERY_DEFAULT_LIMIT` | `10` | Default number of returned hits. |
| `QUERY_MAX_LIMIT` | `50` | Maximum returned hits per request. |
| `QUERY_DEFAULT_CANDIDATE_LIMIT` | `50` | Default candidates retrieved by each hybrid query. |
| `QUERY_MAX_CANDIDATE_LIMIT` | `200` | Maximum candidate pool. |
| `QUERY_DEFAULT_RERANK` | `true` | Enable reranking by default. |
| `QUERY_DEFAULT_RERANK_TOP_K` | `20` | Default candidates sent to the reranker. |
| `QUERY_MAX_RERANK_TOP_K` | `200` | Maximum rerank window. |
| `QUERY_RERANK_MAX_CHARS` | `1000` | Maximum payload-text characters per rerank passage; `0` disables truncation. |
| `QUERY_DEFAULT_FUSION` | `rrf` | Default fusion method: `rrf` or `dbsf`. |
| `QUERY_INDEXED_ONLY` | `false` | Require indexed vector segments for searches. |
| `QUERY_HNSW_EF` | _(unset)_ | Optional default HNSW search breadth. |
| `QUERY_QUANTIZATION_RESCORE` | `true` | Rescore quantized candidates with original vectors. |
| `QUERY_QUANTIZATION_OVERSAMPLING` | `2.0` | Quantized-search oversampling factor. |
| `QUERY_FILTER_FIELD_ALLOWLIST` | _(empty)_ | Comma-separated filterable payload fields; empty allows all. |
| `QUERY_FILTER_MAX_DEPTH` | `20` | Maximum nested filter depth. |
| `QUERY_FILTER_MAX_CONDITIONS` | `100` | Maximum parsed filter conditions. |

### Ingest, export, probes, and retries

| Variable | Default | Description |
|---|---:|---|
| `INGEST_MAX_CHUNKS` | `10000` | Maximum chunks per ingest request, on both the file and the inline path. The bound that protects memory — see [Behaviour contracts](#behaviour-contracts). |
| `INGEST_MAX_FILE_BYTES` | `67108864` | Maximum file-ingest JSON size. Not the memory bound; `INGEST_MAX_CHUNKS` is. |
| `INGEST_DEFAULT_BATCH_SIZE` | `64` | Default points per Qdrant upload batch. |
| `INGEST_MAX_BATCH_SIZE` | `256` | Maximum request-selected upload batch size. |
| `INGEST_UPLOAD_CONCURRENCY` | `4` | Maximum concurrent Qdrant upload batches. |
| `EXPORT_DATA_ROOT` | `/app/middleware-qdrant-manager/exports` | Read-write directory `destination="file"` exports are written to; `./middleware-qdrant-manager/exports` on the host. |
| `EXPORT_MAX_POINTS` | `50000` | Point-count cap checked before and during the export walk; over it is HTTP 413 with nothing written. |
| `EXPORT_MAX_BYTES` | `67108864` | Entire UTF-8 export document cap, including metadata and formatting; over it is HTTP 413. |
| `EXPORT_MAX_CONCURRENT` | `1` | Active exports per app process; excess is HTTP 503 with `Retry-After: 1`. |
| `HEALTH_PROBE_INTERVAL` | `15` | Seconds between dependency probes. |
| `EMBEDDER_PROBE_TIMEOUT` | `3.0` | Timeout for each Smart Embedder probe call. |
| `RETRY_MAX_ATTEMPTS` | `3` | Total attempts for retryable operations. |
| `RETRY_BASE_DELAY` | `0.1` | Initial exponential-backoff delay in seconds, before jitter. A `Retry-After` from Smart Embedder overrides it — see [Behaviour contracts](#behaviour-contracts). |

### Collection-creation schema defaults

These describe the shape of every collection the middleware creates. Compose
does not pass them, so they apply to a natively-run middleware or after you add
them to `compose.yml` yourself. Changing one does not rewrite existing
collections — the schema check starts reporting those as divergent instead.

| Variable | Default | Description |
|---|---:|---|
| `DENSE_VECTOR_NAME` | `dense` | Name of the dense named vector. |
| `SPARSE_VECTOR_NAME` | `sparse` | Name of the sparse named vector. |
| `DENSE_DIM` | `1024` | Dense vector dimension; must match the embedding model's output. |
| `HNSW_M` | `16` | Dense HNSW edges per node; `0` disables the global graph. |
| `HNSW_EF_CONSTRUCT` | `200` | Dense HNSW build-time search breadth. |
| `HNSW_FULL_SCAN_THRESHOLD` | `10000` | Dense query-planner threshold in KB; smaller estimated vector sets favour a full scan over HNSW traversal. |
| `HNSW_INLINE_STORAGE` | `true` | Store copies of original and quantized vectors inside the HNSW index to reduce disk seeks; requires quantization. |
| `DENSE_ON_DISK` | `true` | Store dense vectors on disk. |
| `DENSE_FLOAT16` | `true` | Store dense values as float16. |
| `SPARSE_ON_DISK` | `true` | Store sparse vectors on disk. |
| `SPARSE_FULL_SCAN_THRESHOLD` | `20000` | Number of sparse vectors below which Qdrant prefers a full scan; not a KB threshold. |
| `SPARSE_IDF` | `true` | Apply IDF weighting to sparse scoring. |
| `QUANTIZATION_ENABLED` | `true` | Scalar-quantize the dense vectors. |
| `QUANTIZATION_ALWAYS_RAM` | `true` | Keep the quantized dense vectors in RAM. |
| `INDEXING_THRESHOLD` | `20000` | Vector-data size threshold in KB above which the optimizer builds an index. |
| `MEMMAP_THRESHOLD` | `20000` | Legacy memory-map threshold in KB; deprecated by Qdrant since 1.15.0. Use vector `on_disk` settings for storage placement. |
| `PREVENT_UNOPTIMIZED` | `true` | Defer visibility of new points in large unoptimized segments until optimization finishes; `wait=true` writes also wait for that visibility. |
| `SHARD_NUMBER` | `1` | Shards per collection; meaningful against a cluster. |
| `REPLICATION_FACTOR` | `1` | Replicas per shard; meaningful against a cluster. |

Qdrant owns these storage and optimizer behaviours: see its
[collection configuration API](https://api.qdrant.tech/api-reference/collections/create-collection)
and [deferred-point visibility guide](https://qdrant.tech/documentation/tutorials-operations/prevent-unoptimized-usage/).

## Behaviour contracts

These hold regardless of configuration.

- **Deterministic point IDs** — `UUID5(source_id:chunk_id)`. Re-ingesting a
  source overwrites existing points instead of duplicating them, which makes
  every ingest safe to retry.
- **Embed before write** — ingest and upsert paths finish embedding before
  their first write; a failed embedding call leaves stored data untouched.
- **Ingest is bounded by chunk count, not by bytes.** Because embedding
  finishes before the first write, one request holds every dense and sparse
  vector at once — roughly 52 KB per point at `DENSE_DIM=1024`, since a Python
  `list[float]` stores boxed floats rather than a packed array. That peak
  scales with the number of points, so `INGEST_MAX_CHUNKS` (default 10000) is
  what guards it, on both `/points/ingest` and `/points/ingest-chunks`. Over
  the limit is HTTP 413 with nothing embedded and nothing written.
  `INGEST_MAX_FILE_BYTES` bounds a different thing — how much JSON the service
  will parse — and a file well under it, full of small chunks, can still be far
  too many points. Raise `INGEST_MAX_CHUNKS` only together with the
  middleware's `mem_limit` in `compose.yml`; the container's ceiling is 1 GiB
  as shipped, and exceeding it is an OOM kill the caller sees as a dropped
  connection rather than an error.
- **`replace_source`** — when `true`, old points for the same `source_id` that
  are not in the new batch are pruned only after every new batch's original
  `wait=true` upload is durable. No second payload/vector upload is needed. It
  uses the configured `source_id_type` for both its Qdrant predicate and index,
  including exact ranges for float and datetime.
- **Background health probe** — Qdrant and Smart Embedder are checked every
  `HEALTH_PROBE_INTERVAL` seconds; `/readyz` returns 200 only while both cached
  states are reachable, and 503 otherwise.
- **HTTP timeout ownership** — before returning 504, the timeout layer cancels
  and awaits the request handler so it cannot start a later mutation. Completed
  writes are not rolled back, and durable background reindex jobs retain their
  own lifecycle. The deadline ends when response headers start; exempt ingest
  and export paths keep their existing policy.
- **Retry policy** — transient Qdrant and Smart Embedder failures (5xx, 429,
  timeouts, connection errors) are retried with exponential backoff, bounded by
  `RETRY_MAX_ATTEMPTS`. The backoff carries uniform jitter in `[0.5x, 1x]`, so
  callers that failed together do not retry in lockstep and rebuild the burst
  that caused the failure.
- **A server's `Retry-After` wins over the computed backoff.** When Smart
  Embedder answers with one — `2` on a `503` `rerank_vram_pressure`, `60` on
  its rate-limit `429` — that wait replaces `RETRY_BASE_DELAY`'s exponential
  step for the next attempt. Both RFC 9110 forms are read (delta-seconds and
  HTTP-date), and the honored value is capped at **5 seconds**: the cap is
  smaller than the 60 the embedder can ask for because this service answers
  under `MIDDLEWARE_REQUEST_TIMEOUT` (30 s by default), so waiting the full 60
  would guarantee a `504` instead of surfacing the actionable `502`. An
  unparsable, negative or absent header falls back to the jittered backoff.
  Qdrant responses are not read for this header — the retry there keeps the
  computed backoff.
- **Atomic export writes** — `/points/export` writes to a temp file in
  `exports/` and renames it into place, so a reader never sees a half-written
  export; a failure at any point deletes the temp file instead of leaving one
  behind.
- **Protected payload fields** — `text` and `source_id` survive every payload
  delete, clear and replace operation. Payload-only writes cannot set `text`:
  use an embed-and-upsert endpoint so stored text and vectors stay aligned.
- **Reindex mutual exclusion** — an active reindex job rejects collection and
  point mutations that could invalidate it. The per-collection coordinator
  serializes the admission check with the mutation itself, closing the
  start-versus-write race without blocking unrelated collections.

## Capabilities and limits

**You can:**

- Create, verify, inspect and delete collections with a fixed hybrid-search
  schema, and flip the five mutable schema toggles online without losing
  points.
- Ingest chunks inline or from a staged JSON file, with idempotent overwrite
  and optional pruning of a shrunk source.
- Run hybrid dense+sparse search with RRF or DBSF fusion, payload filters,
  score thresholds, offsets and optional cross-encoder reranking.
- Count, facet, scroll, retrieve, delete and rewrite points and payloads, by id
  or by filter.
- Regenerate every vector in a collection as a resumable background job.
- Export a whole collection's payload in a shape that round-trips back through
  ingest.
- Scrape Prometheus metrics and probe liveness/readiness without a token.

**You cannot:**

- Store or search vectors you produced elsewhere — the middleware always embeds
  through Smart Embedder, and a collection's embedding contract is stamped at
  creation.
- Ingest a file the middleware host cannot see. `data/` is a read-only mount;
  staging a file into it is an operator step outside the API.
- Ingest a whole large corpus in one call — `INGEST_MAX_CHUNKS` (default 10000)
  caps both ingest paths. Split it: point ids are deterministic, so the parts
  neither overlap nor duplicate, and re-sending one is idempotent.
- Export a *filtered* subset — export is whole-collection only, capped by
  the export limits documented above.
- Delete `text` or `source_id` from a payload.
- Change an immutable schema property (dense dimension, vector names) without
  recreating the collection.
- Mutate a collection while a reindex job is pending on it.
- Escape the ingest sandbox: paths that resolve outside
  `middleware-qdrant-manager/data/` are rejected with HTTP 400.
- Publish the service on a public or wildcard address, or on a private one
  without authentication — `MIDDLEWARE_BIND_HOST` accepts only loopback or an
  RFC1918 address, and off loopback it requires a token.

### Data directories

| Directory | Mount | Used by |
|---|---|---|
| `data/` | read-only | `POST /points/ingest`. Paths escaping it are rejected with HTTP 400. Place JSON chunk files here before ingesting. |
| `exports/` | read-write (`EXPORT_DATA_ROOT`) | `POST /points/export`. The service derives the filename itself from the collection name and a timestamp; no path, filename or content the caller supplies ever reaches either mount. |

## Recipes

**Run the stack** (from the repository root):

```powershell
.\stack.ps1 up cpu
```

```bash
./stack.sh up cpu
```

The stack listens on loopback port 6336 and includes a health check against
`/livez`. Use the `gpu` profile only on NVIDIA hosts. The full launcher
reference is [`../scripts/README.md`](../scripts/README.md).

**Install dependencies for a native run:**

```bash
python3 -m pip install -r requirements/middleware.txt
```

Requires Python 3.10+, a running Qdrant, and a running Smart Embedder. The
Docker image additionally applies `requirements/locks/middleware-py311-linux.txt`,
the complete transitive constraint captured from the verified Python 3.11 Linux
image. Regenerate that file whenever a direct middleware dependency changes.

**Ingest a corpus end to end:**

1. Stage the JSON file under `middleware-qdrant-manager/data/`.
2. `POST /collections` with the four schema toggles.
3. `POST /collections/{name}/payload-indexes` for the fields you will filter
   on.
4. `POST /points/ingest` with `file_path` and `collection_name`.
5. `POST /points/count` to confirm the number of points.
6. `POST /query/search` once as a sanity check.

**Regenerate vectors after a profile change:**

1. `POST /collections/{name}/reindex/preflight` — read-only, reports eligible
   and invalid point counts and any blockers.
2. `POST /collections/{name}/reindex` with `confirm_collection` equal to the
   collection name.
3. `GET /collections/{name}/reindex` to poll.
4. Do not mutate the collection while the job is active.

**Clear a stuck collection** (a paused or failed reindex blocks mutations):

```bash
curl -X POST -H "Authorization: Bearer <token>" \
  http://127.0.0.1:6336/collections/<name>/reindex/cancel
```

## Verification

```bash
python3 -m pytest middleware-qdrant-manager/tests -q
python3 -m pytest middleware-qdrant-manager/tests/test_settings.py -q  # one file
python3 -m pytest middleware-qdrant-manager/tests -k test_name -q      # one test
```

Tests run without a live Qdrant or Smart Embedder — they fake the clients.
`pyproject.toml` sets `pythonpath = ["middleware-qdrant-manager"]` and
`asyncio_mode = "auto"`, so imports use `qdrant_middleware.*` regardless of
cwd.

Lint, from the repository root:

```bash
python3 -m ruff check middleware-qdrant-manager
```

The console has its own end-to-end suite against a **real** running stack —
read [`frontend/tests/README.md`](frontend/tests/README.md) before running it;
it is destructive against `e2e_`-prefixed collections.

## Failure modes

| Symptom | Cause | Next step |
|---|---|---|
| `401` with no detail | The bearer token is wrong, or the *label* half of a `token:label` pair was sent instead of the token. | Send the token half. |
| `413` on a large request | Body over `REQUEST_BODY_MAX_BYTES`. Auth never ran — body-size is checked outside the route. | Split the request, or raise the limit. |
| `413` "Too many chunks" from an ingest | The request carries more than `INGEST_MAX_CHUNKS` chunks. Nothing was embedded or written. | Split the source; deterministic point ids make the parts safe to send separately. Raising the cap needs the container's `mem_limit` raised too. |
| `413` with `error: "rerank_capacity_exceeded"` | A single query/passage pair does not fit Smart Embedder's measured headroom. Not retryable. | Shorten passages (`QUERY_RERANK_MAX_CHARS`) or lower the embedder's rerank max length. |
| `413` from `/points/export` | The export exceeds its point or byte cap. Nothing was written. | Review the named cap with the memory limit, or snapshot Qdrant instead. |
| `503` from `/points/export` | The configured export concurrency is occupied. | Retry after the active export finishes. |
| `429` | Per-client rate limit. | Slow down, or raise `MIDDLEWARE_RATE_LIMIT_*`. The console shell, static assets, schema and monitoring probes are exempt; every protected operation is not. |
| `502` from a search or ingest | An unretryable (or exhausted) Smart Embedder / Qdrant failure. | Check `/health` and the embedder's logs. |
| `503` on every protected route | `/readyz` is false: a dependency is unreachable in the cached probe. | `GET /health` for which one. |
| `409` "Collection vectors are being regenerated" | A reindex job is pending on the collection. | Poll `GET .../reindex`, or cancel it. |
| `400` on ingest | The path escaped `data/`, or `chunk_id_key` collided with a reserved key. | See the ingest contract above. |
| Console shows stale UI | `frontend/dist/` is baked into the image. | Rebuild and restart the middleware. |

## Invariants

- **Raw Qdrant access is explicit and mechanically pinned.** The collection
  repository owns schema/metadata/index access; the points router owns point
  CRUD; query, export, health and reindex services own their specialized calls.
  `tests/test_qdrant_access_architecture.py` rejects new or duplicated direct
  calls until their owner, mutation guard, retry semantics and cache effects
  have been reviewed. Prefer an existing gateway over expanding the allowlist.
- **`LOOPBACK_HOSTS` and `INGEST_DATA_ROOT` stay module constants.** As
  `Settings` fields, pydantic-settings would populate them from the environment
  under their upper-cased names, letting the loopback classification and the
  ingest sandbox disable themselves.
- **The published address is loopback or private.** `MIDDLEWARE_BIND_HOST`
  defaults to `127.0.0.1` and accepts only loopback or an RFC1918 IPv4 literal;
  `core/config.py` and `scripts/verify_compose_security.py` both enforce it.
- **The effective inbound ACL must be non-empty** — `MIDDLEWARE_API_TOKENS`
  plus the `SERVICE_TOKEN` Compose delivers — or the service refuses to start.
  Serving with no authentication at all is a deliberate configuration, not an
  accident: it requires an empty list *and*
  `MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=true`. The socket bind does not
  enter into it — under Compose the container always binds `0.0.0.0` inside
  its own namespace — but the published address does: with a non-loopback
  `MIDDLEWARE_BIND_HOST` the service refuses to start unauthenticated.
- **`/assets/config.js` leaks *whether* auth is required, never the token.**
- **Embedding always precedes the first write.** Do not add a write path that
  embeds lazily.
- **Every new setting goes in `core/config.py`** with an explicit range — or is
  a module constant if it must not be operator-tunable.

## Extending it

- **A new endpoint** — router under `api/routers/`, business logic in
  `services/`. Prefer an existing Qdrant gateway; if the endpoint needs a new
  direct call, assign it to the appropriate explicit owner and update
  `tests/test_qdrant_access_architecture.py` with its mutation guard, retry and
  cache semantics. Decide deliberately whether it belongs behind
  `require_bearer_auth` +
  `require_operational_readiness` (collections/points/query) or in the
  unprotected monitoring router.
- **A new setting** — `Settings` in `core/config.py`, with a validated range, a
  row in the matching table above, and an entry in the root `.env.example`.
  Compose only passes what `compose.yml` names.
- **A new filter operator** — the filter parser in `services/`; keep the
  depth and condition counters honest, and add the operator to the supported
  list above.
- **A new dependency call** — reuse `clients/retry.py` rather than writing a
  second backoff policy.
- **Changing a dependency pin** — regenerate
  `requirements/locks/middleware-py311-linux.txt` from the rebuilt, verified
  image (`python -m pip freeze --all`) and commit both files together.
- **Changing the console** — the E2E suite is the contract; see
  [`frontend/tests/README.md`](frontend/tests/README.md).

## Related documents

| Document | Covers |
|---|---|
| [`../README.md`](../README.md) | The stack as a whole. |
| [`../scripts/README.md`](../scripts/README.md) | Building, starting, inspecting and stopping the stack. |
| [`../server-embedder-reranker/README.md`](../server-embedder-reranker/README.md) | The embedding/reranking service this one calls, including the `/rerank` error shapes referenced above. |
| [`../smart-qdrant-mcp-server/README.md`](../smart-qdrant-mcp-server/README.md) | The MCP surface over this REST API. |
| [`frontend/tests/README.md`](frontend/tests/README.md) | The destructive console E2E suite. |
| [`../AGENTS.md`](../AGENTS.md) | Repository-wide conventions for agents working on the code. |
| `../.env.example` | The exhaustive, versioned configuration template. |
