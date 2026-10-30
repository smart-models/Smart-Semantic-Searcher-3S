# Tool reference — smart-qdrant-mcp (28 tools)

All arguments go under `params`. Every tool accepts `response_format`:
`"markdown"` (default) or `"json"` — pick `json` whenever you need exact
values (scores, ids, pagination cursors, `embedder_profile`), markdown when
reading results. It selects only the text block; every result also carries
the full payload as `structuredContent`, which some clients (Claude Code)
show the model instead, making the two formats identical there. `collection_name` is optional (falls back to the
server default) on the read/search/ingest/export/upsert tools,
`qdrant_verify_collection`, `qdrant_create_collection`, `qdrant_set_payload`
and `qdrant_delete_payload_keys`; it is **required** everywhere else —
`qdrant_get_collection`, `qdrant_update_schema_toggles`,
`qdrant_delete_collection`, every payload-index and reindex tool,
`qdrant_delete_points`, `qdrant_clear_payload` — and on any gated call
(`recreate=true`, `mode="replace"`; `replace_source=true` needs an explicit
`source_id` instead), where the `confirm` must be checked against a value the
caller actually gave.

Coverage contract: these 28 tools expose **every request parameter of the
middleware's 42 REST endpoints**. The only deliberate exceptions: the
single-id `GET /points/{id}` (the tools always call the batch retrieve
endpoint, which reports `found`/`not_found` instead of 404ing) and
`GET /metrics` (served as the `qhm://metrics` resource instead). If a
parameter is not listed here, it does not exist on the wire — do not invent
it.

## Monitoring

### `qdrant_check_health`

| Param | Default | Use it well |
|---|---|---|
| `response_format` | `markdown` | Use `json` to read `embedder_profile` (`basic`/`advanced`) and `embedder_models` — the only MCP way to learn which profile is live, hence what the sparse retriever is (BGE-M3 lexical vs MILCO learned-sparse) and which cross-encoder reranks. |

Aggregates `/livez`, `/readyz`, `/health`, `/stats`. Never raises for an
unhealthy stack — it reports `ready: false`. First call whenever anything
else fails. For the same state without spending a tool call, read the
`qhm://health` resource; for Prometheus counters, `qhm://metrics`.

## Search

### `qdrant_search`

One call runs the whole pipeline: embed query (dense + sparse) → two
prefetches of `candidate_limit` each → server-side fusion (RRF default) →
optional cross-encoder rerank of the fused head → stable local paging by
(-score, id). Defaults below are the shipped `.env.example` values; all are
reconfigurable, and every omitted field keeps the middleware's configured
default for this call only.

| Param | Shipped default | Use it well |
|---|---|---|
| `query` | — (required, non-blank) | Natural language or an exact identifier — the two retrievers cover both. |
| `collection_name` | server default | A typo into a real but wrong collection returns confident-looking zero results, not an error. Check with `qdrant_list_collections`. |
| `limit` | 10 (max 50) | **Pass it explicitly** — otherwise the middleware's page size is not echoed and `has_more` is conservative (only an empty page proves the end). |
| `candidate_limit` | 50 (max 200) | The recall knob: candidates fetched **per retriever** before fusion. Raise for rare identifiers, weak recall, or deep pages — fusion can only rank what a prefetch returned. Costs latency. |
| `fusion` | `rrf` | `rrf`: rank-based, immune to score-scale differences — the robust default. `dbsf`: normalizes each retriever's raw scores (3-sigma) so magnitude counts; better only when both retrievers are well-calibrated, and one outlier score can skew a query — if DBSF rankings look unstable, raise `candidate_limit`. Choose between them on a golden query set, not by taste. |
| `rrf_k` | Qdrant's own default (2) | Only valid with `fusion="rrf"`. Smoothing constant: small k sharpens the top-rank advantage, large k (canonical paper value 60, typical range 40–80) flattens it. Change it only with evaluation evidence. |
| `rrf_weights` | equal | Only valid with `fusion="rrf"`. Exactly two non-negative floats, **positional: `[dense, sparse]`**. Upweight sparse (e.g. `[1.0, 1.5]`) for identifier/keyword-heavy corpora, dense for natural-language QA. |
| `rerank` | `true` | The cross-encoder reads query + document together in one pass — the single biggest precision lever. Keep on for quality; set `false` for cheap broad sweeps. |
| `rerank_top_k` | 20 (max 200) | Cost/precision knob: only this many fused hits pay the reranker's per-candidate forward pass. Must be `≤ candidate_limit` (else 422, shown as `Error: invalid request — …`). 10 for precision-first, larger for recall. |
| `rerank_normalize` | `true` | Keep on: reranker scores come back on a comparable 0–1 scale. |
| `filters` | none | Pre-filters **both** retrievers before fusion — hits only compete inside the filtered subset. Every key you filter on must be indexed first (`qdrant_create_payload_index`): every collection runs Qdrant strict mode, which refuses an unindexed key — or a value type the key's index cannot take — with `'<key>' has no payload index that can serve this filter`. The same holds for every other `filters`-taking tool, writes included. Grammar: `docs://reference/filters`. |
| `with_payload` | all fields | Omitting it dumps every payload field of every hit into context. Pass e.g. `["text", "source_id"]`. |
| `offset` | 0 | Deep-paging ceiling: `offset + window` must fit inside `candidate_limit`, else a 422 tells you the required value. `window` is `max(limit, rerank_top_k)` while reranking, plain `limit` with `rerank=false`. With shipped caps and rerank on that tops out near offset 180; page further with `qdrant_scroll_points` + `order_by`. |
| `score_threshold` | none | Applies to the **fused** score, before reranking — a dropped candidate never reaches the reranker. The RRF fused scale depends on `rrf_k` and on whether a hit appears in both legs, so calibrate from observed `json` scores rather than guessing a number. |
| `hnsw_ef` | unset → Qdrant falls back to `ef_construct`, **200 in this stack** (`HNSW_EF_CONSTRUCT`; Qdrant's own default is 100) | Prefetch HNSW graph breadth. Tune from 200, not from 100: 256 is a ~28% widening here, not a doubling. Lowering trades quality for speed. |
| `exact` | `false` | Bypasses HNSW for an exact scan — slow; a diagnostic, not a production setting. |
| `rescore` | `true` | Re-scores quantized candidates against full vectors. It is sent, including an explicit `false`, only when the collection was created with quantization — check `qdrant_get_collection` before tuning it. If the full vectors live on disk (`dense_on_disk=true`), rescore reads them per query: Qdrant offers disabling it as an optional latency trade ("might slightly decrease precision"), not as a recommendation. Turn it off only when disk-read latency is the measured problem. |
| `oversampling` | 2.0 | How many extra quantized candidates flow into rescore. Raise to 3–4 when quantized recall is poor. Same quantization-only caveat. |
| `timeout` | middleware's request budget | Upper bound in seconds on the Qdrant call itself. |

Response fields worth reading (`json`): `fusion`, `reranked`, `candidates`
(fused hits in this window — not the prefetch pool), `skipped_without_text`
(hits with no usable `text` payload: they skip the reranker and keep fused
order — a reason to keep `text` on every point; leaving `text` out of
`with_payload` does not cause skips, since the middleware still reads it for the
reranker and drops it from the response), `score` (fused) vs
`rerank_score`, `has_more`/`next_offset`. `total` is always null: search
never counts matches — use `qdrant_count_points` for that.

Not available through this tool (Qdrant features this stack deliberately
does not expose): per-prefetch `score_threshold`, Formula Query, `group_by`,
recommend/discover, and multivector/ColBERT search — ColBERT output is
turned off at the embedding call in both profiles. Do not propose them.

**Use the error detail to choose recovery.** `request_budget_exceeded`
(413) means the conservative input-token or output-byte admission estimate
exceeds one request's budget: reduce batch size, input length or vector output.
`request_budget_busy` (upstream 503, wrapped as 502 after middleware retries)
means accepted work holds capacity: wait for it to drain or reduce concurrency.
These estimates are not physical RAM/VRAM limits.

`rerank_capacity_exceeded` (413) means a query/passage pair did not fit the
reranker's GPU memory. It is
deterministic — retrying will not help — and the lever is passage
**length**, not `rerank_top_k`: shorten passages via `QUERY_RERANK_MAX_CHARS`
on the middleware or `QWEN_RERANK_MAX_LENGTH` on the Smart Embedder. A `502`
whose text mentions `Smart Embedder server error 503` and
`rerank_vram_pressure` is the transient counterpart — the middleware already
retried it three times — and is fixed by retrying again or lowering
`rerank_top_k` / the middleware's `SMART_EMBEDDER_RERANK_BATCH_SIZE`, not by
`qdrant_check_health`. Full text-match table: `docs://troubleshooting`.

## Collections

### `qdrant_list_collections`

| Param | Default | Use it well |
|---|---|---|
| `response_format` | `markdown` | — |

No parameters beyond format. Answers "does it exist?" before any create or
search. The `qhm://collections` resource answers the same question without a
tool call.

### `qdrant_get_collection`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | — |
| `validate` | `true` | Wire name is `validate`, not `validate_schema`. Set `false` for a cheap info-only report (no schema/contract checks). |
| `response_format` | `markdown` | — |

The report carries per-named-vector config (`dense`: size/distance/on_disk;
`sparse`: idf/on_disk), `quantization` state, `points_count`, and
`embedding_compatibility` — read it before touching `rescore`/`oversampling`
or assuming a contract.

### `qdrant_verify_collection`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | server default | — |
| `dense_on_disk`, `quantization`, `sparse_idf`, `sparse_on_disk`, `on_disk_payload` | omitted = middleware's own expectation | Pass a toggle only to check against a **different** expectation than the server's configured one. |
| `response_format` | `markdown` | — |

Never mutates, never 404s on a missing collection (reports `exists: false`).
This is the first diagnostic when `qdrant_search` refuses with a contract
409: the embedding-contract mismatch behind it lives in its `errors`.

### `qdrant_create_collection`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | server default | Explicit **required** when `recreate=true` (the confirm is checked against it). |
| `dense_on_disk` | — (**required**) | true trades RAM for latency on dense vectors. No server-side fallback exists — choose deliberately, read `docs://concepts/qdrant-data-model` first. |
| `quantization` | — (**required**) | Enables compressed vectors + makes `rescore`/`oversampling` meaningful at search time. |
| `sparse_idf` | — (**required**) | IDF modifier on sparse weights — almost always what you want for lexical retrieval. |
| `sparse_on_disk` | — (**required**) | Same memory/latency trade for sparse. |
| `on_disk_payload` | middleware default | Payload on disk: larger-than-RAM collections. |
| `recreate` | `false` | **Drops every point** first. Requires explicit `collection_name` + `confirm` == it. |
| `confirm` | none | Exact `collection_name`; zero HTTP calls on mismatch. |
| `response_format` | `markdown` | — |

On an existing collection without `recreate`, this **repairs missing payload
indexes** in place — the idempotent "make it so" call.

### `qdrant_update_schema_toggles`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | — |
| `dense_on_disk`, `quantization`, `sparse_idf`, `sparse_on_disk`, `on_disk_payload` | omitted = unchanged | Set at least one (zero toggles is a hard error). In-place and reversible, no gate — the safe way to flip a toggle without recreating. Still a mutation: refused with `blocked by an active reindex job` while a reindex is running or paused. |
| `response_format` | `markdown` | — |

### `qdrant_delete_collection`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | — |
| `confirm` | — (required) | Exact `collection_name`. Irreversible, no dry run — get an explicit user go-ahead beyond the token. |
| `response_format` | `markdown` | — |

## Payload indexes

Field names follow the filter path notation: `a.b` and `items[].label` traverse
objects/arrays; a literal key with punctuation needs explicit quotes, such as
`"\"a.b\""` in JSON. Use the same notation for index create, list, delete and
filters. A literal index cannot satisfy a traversal query. See
`docs://reference/filters` for the grammar.

### `qdrant_list_payload_keys`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | — |
| `sample_size` | 100 (1–1000) | Keys are **sampled**: a lower bound, not authoritative. Raise to catch rare keys; cost is one larger scroll. |
| `response_format` | `markdown` | — |

### `qdrant_list_payload_indexes`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | — |
| `response_format` | `markdown` | — |

Authoritative list of what is indexed today — the check before blaming a
filter for being slow, and before `qdrant_facet_points` (which refuses an
unindexed field outright).

### `qdrant_create_payload_index`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | — |
| `indexes` | — (required, ≥1) | List of `{field_name, field_schema, phrase_matching}`. `field_schema` ∈ `keyword`, `text`, `integer`, `float`, `bool`, `geo`, `datetime`, `uuid` — pick it from § Choosing `field_schema` below, not by guessing from the value's JSON type. `phrase_matching=true` is refused on any schema but `text`. No duplicate `field_name` in one call. |
| `response_format` | `markdown` | — |

A 200 can still carry per-index `failed` entries — read the buckets
(`created`/`updated`/`skipped`/`failed`), not just the ok flag.
Traversal `a.b` and literal `"a.b"` are distinct fields and may appear in the
same call; quoting a simple field such as `"lang"` does not create a second
index identity.

**Re-calling with a different schema destroys the existing index.** Qdrant has
no in-place type change, so the middleware drops the old index and creates a
new one, and reports the field under `updated`. There is **no 409 and no dry
run** — the call you thought was idempotent is the call that rebuilt the
index. Same schema is genuinely idempotent (`skipped`).

Before changing a live field's schema: `qdrant_list_payload_indexes` first, and
if the field is already indexed as something else, say so and get a decision
rather than sending the call. Between the drop and the rebuild, strict mode
refuses every filter on that field.

### `qdrant_delete_payload_index`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | — |
| `field_name` | — (required) | — |
| `confirm` | — (required) | Exact `collection_name`. No dry run. The payload values stay, but strict mode then refuses every filter on that field — searches, counts and filter-addressed writes that rely on it break until it is re-indexed. |
| `response_format` | `markdown` | — |

### Choosing `field_schema`

The schema is not a description of the value — it is the **set of operations
the field will support**. Pick it from the query you intend to run, then check
the stored values can actually carry that type.

| Need to… | Schema | Notes |
|---|---|---|
| exact match / not-in on a string (`source_id`, `language`, `section`) | `keyword` | the default choice for identifiers and enums |
| exact match on a UUID string | `uuid` | same matching as `keyword`, cheaper in memory |
| exact match on a whole number, **or** `gt`/`gte`/`lt`/`lte` | `integer` | the only match+range schema |
| `gt`/`gte`/`lt`/`lte` on a decimal | `float` | range only — Qdrant's Match applies to keyword/integer/bool, not float |
| true/false | `bool` | match only |
| date filtering with `gt`/`lt`, or ordering by time | `datetime` | RFC 3339 strings, compared as UTC. An ISO date indexed as `keyword` gives you equality and nothing else |
| word/substring search inside prose | `text` | `match.text` (all terms) / `match.text_any` (any term) |
| exact quoted phrase inside prose | `text` + `phrase_matching=true` | without the flag a phrase condition matches **nothing** |
| radius / bounding-box / polygon | `geo` | needs `{lon, lat}` values |

Derived rules — these are the ones that bite:

- **`order_by` needs a range-capable index**: `integer`, `float`, or
  `datetime`. A `keyword` index is an index and still gets rejected. The
  `qdrant_scroll_points` § below says "indexed field"; this is the precise
  version of that requirement.
- **`qdrant_facet_points` needs a match-capable index**: `keyword`, `integer`,
  `bool`, or `uuid`. Faceting a `float`, `geo` or `text` field is refused even
  though the field *is* indexed.
- **Schema must match how the value is stored.** Qdrant evaluates a condition
  only over values whose type fits it — a range condition against `"12"` stored
  as a string returns an **empty result, not an error**. An `integer` index over
  string digits is the classic silent-zero-results bug: check a sample with
  `qdrant_read_points` before indexing, and fix the payload (or index it as
  `keyword`) rather than the filter.
- **Don't index the chunk body.** `text` is the largest field and the most
  expensive index, `phrase_matching` makes it larger still, and this stack
  already searches that text through the dense+sparse vectors. Index it only
  when you need a *literal* filter that ranking cannot express, and prefer a
  short derived field (title, heading, tags) when one exists.

What this stack does **not** expose: `tokenizer`, `stemmer`, `stopwords`,
`ascii_folding`, `min_token_len`/`max_token_len`, keyword `prefix`, and the
`memory` tier (`pinned`/`cached`/`cold`, a Qdrant 1.19+ feature). A `text`
index created here therefore uses Qdrant's defaults — `word` tokenizer,
`lowercase: true`, no stemming, no stopwords, no ASCII folding — so on a
non-English corpus `correre`/`corro` and `città`/`citta` will not match each
other. If a corpus needs those, the index has to be created directly against
Qdrant, and a later `phrase_matching=true` call from this server will silently
replace it with a default-params one.

## Reindex

A background job that regenerates every point's dense+sparse vectors under
the current embedding contract. While `running`, `finalizing`, or `paused`,
mutations on that collection are rejected with 409 — intended, protecting
the checkpoint. Details: `docs://workflows/reindex`.

### `qdrant_reindex_preflight`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | — |
| `response_format` | `markdown` | — |

Read-only. Report `can_start`, `blockers`, `total_points`,
`eligible_points`, `invalid_points` (+ their ids) before any start;
`can_start=false` explains exactly why up front.

### `qdrant_start_reindex`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | — |
| `confirm` | — (required) | Exact `collection_name`. |
| `response_format` | `markdown` | — |

### `qdrant_get_reindex_status`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | Poll no faster than every few seconds. Distinguishes no-job / active / finished / `metadata_invalid` (unreadable state → `cancel` clears it). |
| `response_format` | `markdown` | — |

### `qdrant_control_reindex`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | — (required) | — |
| `action` | — (required) | `pause` keeps the checkpoint (returns when *accepted* — wait for `task_running: false` before `resume`); `resume` continues from it; `cancel` **abandons all progress** — never cancel just to get past a 409. |
| `response_format` | `markdown` | — |

## Ingest

Both tools embed server-side and derive point ids as `UUID5(source_id:id)`:
re-ingesting a source overwrites rather than duplicates. A partial failure
returns HTTP 200 with `ok: false` and `failed_at_chunk` — everything before
that position is already written; resume there, do not restart blindly.
One call carries at most `INGEST_MAX_CHUNKS` chunks (10000 shipped), file or
inline: more is refused with 413 before anything is embedded, and this server
reports it as `Error: request body too large.` — split the source across
calls sharing one `source_id`.

### `qdrant_ingest_file`

| Param | Default | Use it well |
|---|---|---|
| `file_path` | — (required) | Relative to the middleware's `data/` root **on its host**; a path resolving outside it (absolute, or `..` climbing out) is rejected (400), as is a file over `INGEST_MAX_FILE_BYTES` (64 MiB shipped, `File too large`). The file must already be staged there — no tool can write it. |
| `collection_name` | server default | — |
| `batch_size` | 64 (middleware default) | Chunks per embed/upsert round trip. Lower it (e.g. 8) for long chunks or a slow embedder to stay inside timeouts. |
| `source_id` | file name w/o extension | Explicit **required** when `replace_source=true`. |
| `source_id_key` | `"source_id"` | Payload field holding the source id. Reserved and rejected: `"text"`, `"id"`, `"overflow_details"`, `"overlap_sentences_count"`, `"split_metadata"`. |
| `source_id_type` | `"keyword"` | Payload type coercion: `keyword`, `text`, `integer`, `float`, `bool`, `datetime`, `uuid`, `geo`, `untyped`. Keep `keyword` unless you filter numerically on it. `replace_source` derives a matching predicate and index from this type. |
| `chunk_id_key` | `"id"` | Payload field holding each chunk's **ordinal**. Relocated, not copied — set it and no `id` remains in the payload. Rejected: `"text"`, the three chunker keys, and whatever `source_id_key` is. |
| `metadata` | `{}` | Fields constant across all chunks — cheaper than repeating them per chunk, and what you later filter/facet on. |
| `replace_source` | `false` | After writing the new points, deletes every existing point with this typed `source_id`, so a mistyped id wipes an unrelated source. It uses match predicates for lexical, integer, bool and UUID types and exact ranges for float and datetime; the middleware creates the `source_id` index that delete needs itself, so no prior index is required. Verify the id with `qdrant_facet_points` on the chosen source-id field first. |
| `confirm` | none | Required with `replace_source=true`: exact `source_id`. |
| `response_format` | `markdown` | — |

The JSON must be an **object** with a top-level `chunks` array — a bare
array is refused with `"JSON must have a top-level 'chunks' array"`.

#### Chunk contract (enforced server-side, HTTP 400 on any breach)

| Field | Type | Rule |
|---|---|---|
| `id` | **integer** | Required, unique within the call. The chunker's per-source **ordinal**, not the point id — the point id is `UUID5(source_id:id)`, derived from it. Stored as ordinary payload under `chunk_id_key` (default `id`). |
| `text` | **string** | Required, non-empty after `strip()`. Whitespace-only is refused (embedding it is useless). |
| any other key | any JSON type | Rides along as payload on the point. This is how you store `language`, `version`, `section`, page numbers, URLs — anything you later filter or facet on. |

The three chunker bookkeeping keys — `overflow_details`,
`overlap_sentences_count`, `split_metadata` — used to be silently **stripped**
from the payload before writing. That stripping is **currently disabled**
(`CHUNK_EXCLUDE_FIELDS` is empty in `points.py`), so they land in the payload as
sent. All three stay reserved: `source_id_key` and `chunk_id_key` both refuse
them.

`id` is not stripped but **relocated**: the middleware writes the ordinal under
`chunk_id_key` and drops the verbatim `id` copy, so renaming it never leaves a
duplicate behind. `chunk_id_key` defaults to `id`, must differ from
`source_id_key`, and cannot be `text` or one of the three keys above. The
chunk objects you send always use the literal key `id` — `chunk_id_key` names
where it is *stored*, not how it is read.

#### `metadata` — request-level fields injected into every point

`metadata` is a dict **on the request**, not on each chunk. Every key/value
in `metadata` is merged into the payload of **every** point produced by the
call, except `text` and `source_id` (protected — the service owns those).
Use it for fields that are constant across the whole source (language,
version, dataset origin) instead of repeating them on every chunk: smaller
request, single source of truth, and the fields you filter/facet on.

#### Template — copy this as a starting point

```json
{
  "chunks": [
    {
      "id": 0,
      "text": "The first chunk of text to embed. Keep ≲1000 characters — the reranker truncates at 1000.",
      "section": "intro",
      "page": 1
    },
    {
      "id": 1,
      "text": "The second chunk. Note: id is an integer unique within this call, not a UUID.",
      "section": "method",
      "page": 3
    }
  ]
}
```

With request-level `metadata` and a custom `source_id_key`:

```json
// qdrant_ingest_chunks call — params, not the file shape
{
  "chunks": [
    {"id": 0, "text": "...", "section": "intro"},
    {"id": 1, "text": "...", "section": "method"}
  ],
  "source_id": "paper-2024-v1",
  "source_id_key": "source_id",
  "source_id_type": "keyword",
  "metadata": {"language": "en", "dataset": "arxiv", "version": 1},
  "collection_name": "papers",
  "batch_size": 8
}
```

Every point written from that call carries `language=en`, `dataset=arxiv`,
`version=1`, `source_id="paper-2024-v1"`, plus the per-chunk `section` —
all in its payload, all filterable and facetable (once indexed). The chunk
`id` (0, 1) is in the payload too, under `chunk_id_key` (here the default,
so literally `id`): it is **relocated, not consumed**. It also derives the
point id (`UUID5(source_id:id)`), but that is a second use, not its only
one — index it as `integer` when you want to filter or order by position
within a source.

### `qdrant_ingest_chunks`

| Param | Default | Use it well |
|---|---|---|
| `chunks` | — (required, ≥1) | Each `{id: int, text: str, ...extra payload}`. Inline requests die at the transport's 4 MiB cap with a bare `Request body too large` — split across calls sharing one `source_id` well before that. |
| `source_id` | — (**required**) | No file name to fall back to here. |
| `collection_name`, `batch_size`, `replace_source`, `confirm`, `source_id_key`, `source_id_type`, `chunk_id_key`, `metadata`, `response_format` | as above | Identical semantics to `qdrant_ingest_file`. |

## Export

### `qdrant_export_collection`

| Param | Default | Use it well |
|---|---|---|
| `collection_name` | server default | — |
| `response_format` | `markdown` | — |

Ingest's twin: one call replaces a paginated `qdrant_scroll_points` walk. The
middleware does the walk server-side and writes a JSON file; the collection
never passes through your context window.

Two parameters this tool deliberately does **not** have:

- **No `filters`.** Export is whole-collection only — narrowing was left out
  at the endpoint's design, not omitted from the tool by accident. There is
  no way to export a slice; see the 413 row below for what to do when the
  whole collection is too large.
- **No `destination`.** The middleware endpoint backing this tool can also
  return the document in the response body, but this tool never asks for
  that — putting a whole collection into a tool result is exactly the
  context cost the endpoint exists to remove.

The document produced is `{"chunks": [...], "metadata": {...}}` — the same
shape `qdrant_ingest_chunks` expects, payload only, never vectors. Each chunk
is the stored payload verbatim plus a trailing **`point_id`** carrying the
Qdrant point id as a string — a UUID for ingested points, an unsigned integer
for a point upserted under one; it is a key of its own so it cannot collide
with a chunk's `id`. See `playbooks.md` §7 for the re-ingest id trap (strip
`point_id`, and check `id` is an integer — exports written before this split
put the UUID under `id` instead).

Returns `{"ok", "collection_name", "n_chunks", "path", "bytes"}`. `path` is
inside the **middleware container**
(`/app/middleware-qdrant-manager/exports/…`, `./middleware-qdrant-manager/exports/…`
on the host, where `./` is the stack checkout root — see `playbooks.md` §7 for
how to resolve that to an absolute path you can hand a user) and `null` when
the collection is empty (nothing written).
The middleware derives and sanitizes the timestamped, random-suffixed basename
it writes, then verifies the resolved path stays in that directory; do not
infer a future filename from the collection name.
Neither this server nor the caller can read that file back or list the
directory — no tool on this server has that capability. Point and exact
UTF-8 document-byte caps are checked throughout export, including metadata.
A cap violation returns 413; a full export slot returns 503 before scrolling;
a failed page returns 502. None publishes a file. The output cap does not
bound an upstream page or Python object overhead, and the walk is not a
transactional snapshot. Retry occupied capacity after the active export ends.

## Point reads

### `qdrant_read_points`

| Param | Default | Use it well |
|---|---|---|
| `ids` | — (required, 1–1000) | UUIDs or unsigned integers as strings. Always uses the batch endpoint: returns `found`/`not_found` lists, never 404s on one missing id. |
| `collection_name` | server default | — |
| `with_vectors` | `false` | `true` adds a **summary** of stored vectors (names and shapes only, never raw numbers) — how you confirm which embedding generation a point carries. |
| `response_format` | `markdown` | — |

### `qdrant_scroll_points`

| Param | Default | Use it well |
|---|---|---|
| `filters` | none = whole collection | Enumeration, not ranking — ordered, never scored. |
| `limit` | 20 (1–1000) | This server's deliberate default, overriding the middleware's 10, for fewer round trips. |
| `offset` | none | Page cursor: integer or the opaque `next_offset` from the previous call — pass it back untouched. **Cannot combine with `order_by`** (Qdrant rejects it). |
| `order_by` | none | `{key, direction: asc/desc, start_from}` on a field with a **range-capable** index — `integer`, `float`, or `datetime`; a `keyword` index is rejected (§ Choosing `field_schema`). Page an ordered scroll with `start_from` = first ordered value not yet seen (a number or datetime string matching the field), not a row count. |
| `with_payload` | `true` | `true` = every field (verbose), `false` = none, or a list of keys. |
| `include_total` | `true` | Adds one count call per page; set `false` on large collections when you don't need the total. |
| `collection_name` | server default | — |
| `response_format` | `markdown` | — |

For a whole collection, prefer `qdrant_export_collection` (`## Export` above)
— one call instead of N pages of this tool, with nothing landing in your
context window. Reach for a manual scroll walk only when you need a
*filtered* slice in the conversation to reason over; export has no `filters`.

### `qdrant_count_points`

| Param | Default | Use it well |
|---|---|---|
| `filters` | none = whole collection | The cheapest existence check — run it before assuming content is missing. |
| `exact` | `true` | `false` = fast approximate estimate on large collections. |
| `collection_name` | server default | — |
| `response_format` | `markdown` | — |

### `qdrant_facet_points`

| Param | Default | Use it well |
|---|---|---|
| `key` | — (required) | One payload field with a **match-capable** index — `keyword`, `integer`, `bool`, or `uuid` (§ Choosing `field_schema`). Unindexed is refused outright (it would be a full scan to list distinct values), and so is an indexed `float`/`geo`/`text` field. |
| `filters` | none | Restrict which points are faceted. |
| `limit` | 20 (1–1000) | Max distinct values. |
| `exact` | `true` | `false` for a faster approximation. |
| `collection_name` | server default | — |
| `response_format` | `markdown` | — |

How to confirm a `source_id` really exists (and its spelling) before typing
it into a `confirm`.

## Point writes

### `qdrant_upsert_points`

| Param | Default | Use it well |
|---|---|---|
| `points` | — (required, 1–10000) | Each `{point_id, text, payload}`. `point_id`: UUID or unsigned integer as string; no duplicates in one call (refused — a duplicate would silently overwrite itself). `text` is embedded server-side; blank text is rejected. One point uses the flat endpoint, more use the batch endpoint. |
| `collection_name` | server default | — |
| `update_mode` | `"upsert"` | `"upsert"` always writes; `"insert_only"` skips ids that already exist; `"update_only"` skips ids that don't. Skipped ids are **counted in the response, never silently dropped** — read the count. |
| `response_format` | `markdown` | — |

### `qdrant_delete_points`

The most dangerous tool in the server.

| Param | Default | Use it well |
|---|---|---|
| `selector` | — (required) | `"ids"` or `"filter"`. |
| `collection_name` | — (required) | Required always: the confirm target and the dry-run's counting scope. |
| `ids` | none (≤10000) | Required non-empty with `selector="ids"`. One id uses the single-point endpoint; more go out as one bulk request. |
| `filters` | none | Required non-empty with `selector="filter"` — an empty filter would match every point and is refused outright, `confirm` or not. |
| `confirm` | none | Exact `collection_name`. With `selector="filter"`, **omitting it runs a free dry-run count** — always rehearse, report the number, then confirm. With `selector="ids"` there is no dry run: read/scroll the ids first and show what will go. |
| `response_format` | `markdown` | — |

## Payload writes

Shared shape: `selector="ids"` fans out **one HTTP request per id,
non-atomically** — the response lists which ids succeeded and which failed
with why; keep going after a failure, the successful writes stay applied.
`selector="filter"` is one Qdrant-side write returning a single `matched`
count. `text` and `source_id` are protected: `replace` and `clear` never
remove them. `qdrant_set_payload` cannot set `text`; use
`qdrant_upsert_points` so the point's vector is regenerated with the new text.

### `qdrant_set_payload`

| Param | Default | Use it well |
|---|---|---|
| `selector` | — (required) | `"ids"` or `"filter"`. |
| `mode` | `"merge"` | `"merge"` adds/overwrites the given fields, keeps the rest, no gate. `"replace"` discards every unprotected field — gated: `ids` selector only, explicit `collection_name`, `confirm` == it. `filter`+`replace` is refused (no such middleware endpoint): merge instead, or resolve the filter to ids with `qdrant_scroll_points`. |
| `payload` | — (required, non-empty) | Non-text fields to write. Use `qdrant_upsert_points` to change `text`. |
| `ids` / `filters` | none | Per selector; `filters` must be non-empty. |
| `collection_name` | server default | Explicit **required** when `mode="replace"`. |
| `confirm` | none | Required only with `mode="replace"`: exact `collection_name`. |
| `response_format` | `markdown` | — |

### `qdrant_delete_payload_keys`

| Param | Default | Use it well |
|---|---|---|
| `selector` | — (required) | `"ids"` or `"filter"`. |
| `keys` | — (required, ≥1) | Qdrant payload paths to remove; blank paths are rejected. Narrower than `qdrant_clear_payload` (which takes everything but `text`/`source_id`), so reach for this first — but it **destroys those values**: there is no gate, no dry run, and up to 1000 points go in one call. Restoring them means re-setting values you still hold elsewhere. Show what will go, on which points, before calling. Protected roots (`text`, `source_id`) cannot be removed through plain, quoted, nested or array paths. |
| `ids` / `filters` | none | Per selector. |
| `collection_name` | server default | — |
| `response_format` | `markdown` | — |

### `qdrant_clear_payload`

| Param | Default | Use it well |
|---|---|---|
| `selector` | — (required) | `"ids"` or `"filter"`. |
| `ids` / `filters` | none | Per selector. By filter, the middleware discovers which non-protected keys to clear from a sample of up to 1000 matched points — a key present only beyond the sample survives. |
| `collection_name` | — (required) | — |
| `confirm` | — (required) | Exact `collection_name`, either selector. Removes everything but `text`/`source_id`; no dry run. |
| `response_format` | `markdown` | — |

## Hard caps worth memorising

| Cap | Value | Where it bites |
|---|---|---|
| MCP transport request body | 4 MiB | Inline `qdrant_ingest_chunks` — rejected before any tool runs, bare text, no `is_error`. |
| `INGEST_MAX_CHUNKS` | 10000 per ingest call | File or inline. 413 before anything is embedded, reported as `Error: request body too large.` Split across calls sharing one `source_id`. |
| `INGEST_MAX_FILE_BYTES` | 67108864 (64 MiB) | `qdrant_ingest_file` only: a larger staged file is refused with 400 `File too large`. |
| `MCP_HTTP_TIMEOUT` | 120 s | Every outbound call, ingest and export included. Only reads are retried before reporting (see `MCP_RETRY_ATTEMPTS`). |
| `qdrant_read_points` ids | 1000 | |
| `qdrant_scroll_points` / `qdrant_facet_points` limit | 1000 | |
| `qdrant_upsert_points` points / `qdrant_delete_points` ids | 10000 | |
| `qdrant_set_payload` / `qdrant_delete_payload_keys` / `qdrant_clear_payload` ids | **1000** | Lower than the point writes on purpose: these fan out one request per id (no bulk-by-ids endpoint exists), so a bigger list cannot finish or report inside one call. `selector="filter"` writes any number of points in a single request. |
| `EXPORT_MAX_POINTS` / `EXPORT_MAX_BYTES` | 50000 / 67108864 | Whole collection only. Checked during the walk; bytes include the complete encoded document. Exceeding either returns 413 with no published file. |
| `EXPORT_MAX_CONCURRENT` | 1 per middleware app | 503 before scrolling when all export slots are occupied. |
| `SMART_EMBEDDER_MAX_INFLIGHT_TOKENS` / `SMART_EMBEDDER_MAX_INFLIGHT_OUTPUT_BYTES` | 1048576 / 268435456 | Shared conservative inference admission estimates (UTF-8 bytes, not tokens; the token default covers one 32-pair rerank at full length). One oversized request gets 413; occupied capacity gets upstream 503. Active native work retains its reservation after caller cancellation. |
| Search `limit` / `candidate_limit` | 50 / 200 (shipped `QUERY_MAX_*`) | Deep paging hits the candidate ceiling — see `qdrant_search.offset`. |
| `confirm` token length | 255 chars | Over-long tokens refused locally. |
| Filter depth / conditions | 20 / 100 (configurable) | Rejected before reaching Qdrant. |
| `MCP_RETRY_ATTEMPTS` | 3 | Total attempts on transient failures for reads only — GETs and the read-only POSTs (search, scroll, count, facet, retrieve, verify, reindex preflight). Writes are sent once, never retried: after a failed or timed-out write, check with `qdrant_count_points` or `qdrant_read_points` before re-sending (ingest and upsert are idempotent, so re-sending the same chunks is safe). |

## Resources

Documentation (`docs://`, 21 in all) — read before improvising. Nine
teaching documents written against these tools:

| URI | Read it when |
|---|---|
| `docs://concepts/hybrid-search` | Understanding what dense and sparse each contribute, and why ColBERT plays no part here. |
| `docs://concepts/fusion-and-rerank` | Choosing `fusion`, `rrf_k`, `rrf_weights`, `candidate_limit`, rerank settings — the theory behind every `qdrant_search` knob. |
| `docs://concepts/qdrant-data-model` | Before creating a collection: the four toggles, HNSW, quantization, on-disk trades. |
| `docs://concepts/embedding-contract` | Before ingesting, and whenever a search or write is refused with a contract 409. |
| `docs://reference/filters` | Writing any `filters` object: legacy vs logical syntax, operators, depth/condition limits. |
| `docs://workflows/ingest` | Before the first ingest: chunk shape, sizing, staging. |
| `docs://workflows/search-recipes` | Five concrete `qdrant_search` parameterisations + the empty-results checklist. |
| `docs://workflows/reindex` | Before any reindex. |
| `docs://troubleshooting` | The full error-text → next-call table (`playbooks.md` §8 is its short form). |

Twelve application documents, byte copies of the repository's own guides and
component READMEs:

| URI | Read it when |
|---|---|
| `docs://guide/getting-started` | A user's first run: `.env`, connecting Qdrant, starting the stack, a first search. |
| `docs://guide/cli` | Telling a user how to start, stop or inspect the stack with the launchers. |
| `docs://guide/environment` | A user asks what a `.env` setting does or how a change is applied. |
| `docs://guide/architecture` | Explaining how the local stack fits together. |
| `docs://guide/hybrid-search-rerank` | Explaining hybrid search and reranking to an end user. |
| `docs://guide/embedding-models` | Which models the `basic` and `advanced` profiles run, their sources and licenses. |
| `docs://reference/middleware-api` | The REST behaviour behind a tool's response. |
| `docs://reference/launcher-scripts` | Exact launcher commands, flags and exit codes. |
| `docs://reference/smart-embedder` | Embedding and reranking behaviour, profiles, inference limits. |
| `docs://reference/mcp-server` | This server's contracts, including the one-table view of every confirmation gate. |
| `docs://reference/agent-skills` | What the client-side skills cover. |
| `docs://reference/smart-qdrant-skill` | This skill's `SKILL.md` — what the server's instructions point a client without the skill at. |

Live state (`qhm://`) — read for context without spending a tool call:
`qhm://collections`, `qhm://health`, `qhm://metrics` (Prometheus text).

Prompts (multi-tool sequences that must run in order):
`qdrant_bootstrap_collection`, `qdrant_diagnose_search_quality`,
`qdrant_safe_reindex`.

## Keeping this file aligned with the server

The server's source of truth is `smart_qdrant_mcp/tools/*.py` (input models
and gates), `tool_map.TOOL_ROUTES` (tool → endpoint), and
`resources/docs.py` (all 21 documents: the nine teaching docs in
`DOC_RESOURCES`, the twelve application copies in `APP_RESOURCES`). Its own tests (`test_prompts.py`,
`test_resources_docs.py`) fail when a prompt or doc points at something that
no longer exists. When the server adds, renames, or re-params a tool,
update this file in the same change — and verify the live surface with the
client's tool list, which is generated from those models.
