# Playbooks — smart-qdrant-mcp

Eight sequences plus an error map. Each one names the tool to call, in order.

## 0. The tools aren't there

`qdrant_*` tools missing from the client means the MCP server isn't connected —
nothing about Qdrant is broken yet.

1. The client needs its own registration pointing at
   `http://127.0.0.1:6337/mcp` with `Authorization: Bearer <token>` — in this
   repo, a local `.mcp.json` the user creates. It is gitignored, never
   shipped, and holds the user's own token, which must be listed in
   `MCP_API_TOKENS` in `.env` (unless the server runs unauthenticated:
   empty `MCP_API_TOKENS` with `MCP_ALLOW_UNAUTHENTICATED_LOCAL=true`, loopback
   only).
2. The stack must be up: `pwsh stack.ps1 up cpu` (or `gpu`) from the
   repo root, `pwsh stack.ps1 status cpu` to check.
3. `MCP_API_TOKENS`, `MIDDLEWARE_CLIENT_TOKEN`, and `SMART_EMBEDDER_CLIENT_TOKEN` must
   be pairwise distinct — a collision fails startup outright, so an "unreachable"
   server may simply have refused to boot. Check `pwsh stack.ps1 logs`.

Reconnecting is the user's action, not yours. Report which of the three it is.

## 1. Bootstrap a collection

Prefer the server's own `qdrant_bootstrap_collection` prompt. Its order:

1. `qdrant_check_health` — Qdrant and the embedder ready.
2. `qdrant_list_collections` — does it already exist? If yes, **ask before
   recreating**; `recreate=true` destroys every point.
3. `qdrant_create_collection` — all four toggles (`dense_on_disk`,
   `quantization`, `sparse_idf`, `sparse_on_disk`) are required; read
   `docs://concepts/qdrant-data-model` and choose them deliberately.
4. `qdrant_create_payload_index` — at minimum `source_id` as `keyword`, plus
   every key you plan to filter on. Every collection runs Qdrant strict mode,
   so an unindexed key cannot be filtered at all — reads and filter-addressed
   writes are refused. Choose
   each `field_schema` from the intended query, not from the value's JSON type
   (§ Choosing `field_schema` in `reference.md`): range needs
   `integer`/`float`/`datetime`, faceting needs `keyword`/`integer`/`bool`/
   `uuid`, and an index whose type disagrees with the stored values returns
   zero results without erroring. Index it now — changing a schema later drops
   and rebuilds the index.
5. `qdrant_ingest_file` or `qdrant_ingest_chunks`.
6. `qdrant_count_points` — does the count match what the ingest reported?
7. `qdrant_search` — one sanity query.

## 2. Ingest a corpus that isn't staged yet

`qdrant_ingest_file` only reads files already under
`middleware-qdrant-manager/data/`, and **no tool on this server can put one
there** — the mount is read-only and staging is deliberately an operator step.

Under ~4 MiB and already in context → skip staging, use `qdrant_ingest_chunks`,
splitting across calls that share one `source_id`. Otherwise ask the user to run:

```powershell
$dst = "middleware-qdrant-manager\data\corpus"
New-Item -ItemType Directory -Path $dst -Force | Out-Null
Copy-Item "<source>\corpus.json" -Destination "$dst\corpus.json" -Force
```

```bash
mkdir -p middleware-qdrant-manager/data/corpus
cp "<source>/corpus.json" middleware-qdrant-manager/data/corpus/
```

Then:

```json
qdrant_ingest_file {
  "file_path": "corpus/corpus.json",
  "collection_name": "docs",
  "source_id": "corpus-v1",
  "batch_size": 8
}
```

The JSON must be an object with a top-level `chunks` array, each chunk
`{"id": <int>, "text": "..."}` plus optional payload fields. Full contract,
template, and `metadata` propagation: `reference.md` § Ingest. Two rules
that bite most often: `id` must be an **integer unique within the call**
(not a UUID — the server derives the UUID), and `text` must be non-empty
after trimming (whitespace-only is refused).

**Chunk size shapes search quality.** Keep chunk `text` at ≲1000 characters:
the reranker truncates each hit at `QUERY_RERANK_MAX_CHARS` (1000 shipped),
so a longer chunk is re-scored on its head alone. Split long documents rather
than letting the reranker read half of them.

**Re-ingesting a source that shrank** leaves orphan points behind: chunk ids that
no longer appear are never revisited. Only `replace_source=true` prunes them, and
it deletes by string equality on `source_id` alone — so it needs an explicit
`source_id` and `confirm` equal to it exactly. Confirm the `source_id` with
`qdrant_facet_points` on `source_id` before you type it.

**If an ingest times out** (120 s `MCP_HTTP_TIMEOUT`), the middleware may have
finished anyway. `qdrant_count_points` first; re-sending is safe because point
ids are `UUID5(source_id:chunk_id)`.

## 3. Search returns nothing or nonsense

Prefer `qdrant_diagnose_search_quality`. Its order:

1. `qdrant_count_points`, no filter — is there any data at all?
2. `qdrant_verify_collection` — embedding-contract mismatch? It does not hide
   as empty results: the search itself fails with `Error: embedding contract
   mismatch`, and this shows what mismatched; see
   `docs://concepts/embedding-contract`.
3. `qdrant_search` with `response_format="json"` — record `fusion`, `reranked`,
   `candidates`, and the score spread.
4. Re-run with a higher `candidate_limit`; then, since `rerank` is already on
   by default, narrow it — a small `rerank_top_k` — or set `rerank=false` once
   to see what the reranker is actually contributing. Report which change moved
   the needle.
5. Filter involved? `qdrant_list_payload_indexes` (is the key indexed?) and
   `qdrant_facet_points` (do the values you filter on exist?).
6. `qdrant_scroll_points` with the same filter — does the content exist but rank
   badly, or is it absent? Different fixes.

Report as: cause, evidence, recommended change.

Tuning cheatsheet (full recipes: playbook 6): rare identifier →
`candidate_limit=200`, `rerank=true`. Precision-first → `rerank=true`,
`rerank_top_k=10`, a `score_threshold` calibrated from observed `json`
scores. Paging → fixed `limit`, increasing `offset`, identical query and
filters — until the candidate ceiling 422s (playbook 6, recipe E).

## 4. Reindex safely

Prefer `qdrant_safe_reindex`.

1. `qdrant_reindex_preflight` — report `can_start`, `blockers`, `total_points`,
   `eligible_points`, `invalid_points`. `can_start=false` → stop, explain.
2. `invalid_points > 0` → show the ids and ask before proceeding; they will not
   be reindexed.
3. `qdrant_start_reindex` with `confirm` == the collection name exactly.
4. `qdrant_get_reindex_status` with `wait_seconds=30` — each call waits for
   the job to change, so repeat it with no delay of your own. Done when
   `status` is `completed` **and** `task_running` is false; `status` can flip
   one read earlier.
5. Do not call any mutating tool on that collection meanwhile — ingest,
   upsert, payload and point writes, schema toggles, index changes: 409 is
   intended.
6. Done → `qdrant_verify_collection`, then one `qdrant_search`.

Interrupt with `qdrant_control_reindex action="pause"` (checkpoint kept,
`resume` continues) — a paused job still blocks mutations. `cancel` abandons
the progress; only completion, or a `cancel` the user approved, lifts the 409.

## 5. Delete without regret

- **By filter:** call `qdrant_delete_points` with `selector="filter"` and **no**
  `confirm`. That is a free dry-run count. Show the number to the user. Only then
  re-send with `confirm` == `collection_name`.
- **By ids:** no dry run exists. Retrieve them first (`qdrant_read_points` or
  `qdrant_scroll_points`), show what will go, then confirm.
- **A whole source:** `qdrant_delete_points` by filter on `source_id`, or
  re-ingest with `replace_source=true`.
- **Payload only:** `qdrant_delete_payload_keys` (named keys) before reaching
  for `qdrant_clear_payload` (everything but `text` and `source_id`,
  confirm-gated, no dry run). Narrower is not undoable: the named keys are
  destroyed on up to 1000 ids per call (any number by filter), with no gate
  and no dry run of its own.
  Rehearse it yourself — `qdrant_scroll_points` with the same selector, showing
  those keys — before you send it.
- **Whole collection:** `qdrant_delete_collection` — irreversible, always
  confirm-gated, and worth an explicit user go-ahead beyond the token.

## 6. Tune hybrid search quality

The defaults are sane — start bare, tune on evidence. Every knob is per-call:
nothing is stored on the collection, so a winning combination must be
repeated on every subsequent call. Theory: `docs://concepts/fusion-and-rerank`.

Baseline first: `qdrant_search` with just `query` (and `limit` +
`with_payload`). Then escalate by symptom:

- **A. Natural-language QA feels shallow** → `candidate_limit=100`,
  `rerank=true`, `rerank_top_k=25`. Recall lives in `candidate_limit`;
  precision in the reranker.
- **B. Rare identifier / exact token** (`MIDDLEWARE_API_TOKENS`, a SKU, an
  error code) → `candidate_limit=200`, `rerank=true`; if it still loses to
  prose, `rrf_weights=[1.0, 1.5]` (positional: dense, then sparse) to let the
  lexical leg count for more.
- **C. Precision-first** → `rerank=true`, `rerank_top_k=10`, plus a
  `score_threshold` chosen from a prior `response_format="json"` run's fused
  scores — the RRF scale depends on `rrf_k` and double-leg presence, so never
  guess the number.
- **D. Filtered search** → confirm the key is indexed
  (`qdrant_list_payload_indexes`; create it if not), confirm the value exists
  (`qdrant_facet_points`), then pass `filters`. Strict mode refuses an
  unindexed key outright.
- **E. Deep paging** → fixed `limit`, `offset` in steps, identical query and
  filters. The middleware 422s when `offset + max(limit, rerank_top_k)`
  exceeds `candidate_limit`: raise `candidate_limit` (max 200 shipped) as the
  error says; beyond that ceiling, switch to `qdrant_scroll_points` with
  `order_by` (paged via `start_from`).
- **F. Quantized collection** → confirm via `qdrant_get_collection`
  (`quantization` state), then `rescore=true`, `oversampling=2.0` (up to
  3–4 when recall is poor). On an unquantized collection both are silently
  ignored — tuning them there is wasted motion.
- **G. `hnsw_ef` / `exact`** → the baseline is `ef_construct`, 200 in this
  stack, so raise it meaningfully (e.g. 400) when dense-side recall is
  suspect; 256 is a marginal step from 200, not the doubling it looks like
  (`reference.md` § Search). `exact=true` only as a one-off diagnostic to
  compare against the HNSW result — if `exact` finds what HNSW missed, the
  graph walk is the problem and `hnsw_ef` is the lever.

**Choosing fusion on evidence, not taste.** Build a golden set of 5–10
representative queries with known-good expected hits. Run each at
`response_format="json"` under `fusion="rrf"` (default k, then k=60) and
`fusion="dbsf"`, plus one `rrf_weights` variant if one leg should dominate.
Record `fusion`, `candidates`, the score spread, and where each known-good
hit lands. Keep the winner as this collection's standard call pattern. DBSF
caveat: its normalization statistics come from the prefetch sample, so an
outlier can skew a single query — unstable rankings argue for a higher
`candidate_limit` or staying on RRF.

## 7. Export a collection

`qdrant_export_collection` is one call: the middleware walks every page
server-side and writes a single JSON file, so the collection's payload never
crosses into your context window the way a manual `qdrant_scroll_points` walk
would. `collection_name` is optional (falls back to the middleware's default).
There is deliberately no `filters` — export is whole-collection only, a
decision made at the endpoint's design, not a gap you can work around — and
deliberately no `destination`: the tool never asks the middleware to return
the document in the response body, because that would put the whole
collection back into your context window, exactly what the endpoint exists to
avoid.

It returns `{"ok", "collection_name", "n_chunks", "path", "bytes"}`. `path`
is inside the **middleware container**
(`/app/middleware-qdrant-manager/exports/…`, which on the host is
`./middleware-qdrant-manager/exports/…`) and is `null` when the collection is
empty — nothing is written in that case. Neither this server nor you can read
that file back or list the directory: no tool on this server has that
capability. Report the path and counts to the user and stop there; retrieving
the file from the host is their step, not yours.

**Telling the user where the file actually is.** The `path` you get back is
the *container's* view. `./middleware-qdrant-manager/exports/` is the host
side of that bind mount, and the `./` is relative to the **stack checkout
root** — the directory holding `compose.yml`, the one `pwsh stack.ps1
up` was run from. You have no filesystem access and cannot resolve it
yourself, so do not guess an absolute path and do not present one as fact.
Give the user the relative location, and if they don't know where the stack
lives, hand them the command that answers it:

```bash
docker inspect qdrant-middleware --format "{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}"
```

The line ending in `/app/middleware-qdrant-manager/exports` has the absolute
host directory on its left. Your export is the filename from `path` inside
it. Copying it somewhere else (a Downloads folder, another machine) is an
ordinary file copy the user performs — there is no tool here that moves,
renames, or deletes it, and the file stays in `exports/` until they remove
it.

**Export refusal.** A 413 names the point or encoded-byte cap exceeded
before or during the walk; no file is published. There is no `filters`
parameter to narrow this tool. Ask the operator to review the named cap and
memory capacity, or use an operator-level Qdrant snapshot outside this stack.
A 503 for occupied export capacity needs a retry after the active export
finishes. Defaults and precise accounting: `reference.md` § Export and caps.

### Re-ingesting an export — the id trap

Every exported chunk is the stored payload plus a trailing **`point_id`**: the
stored Qdrant point id as a string — for ingested points
`UUID5(source_id:chunk_id)`, a UUID; for a point upserted under an unsigned
integer, that integer. `qdrant_ingest_chunks` and `qdrant_ingest_file` expect chunk `id`
to be an **integer** (the per-source chunk ordinal), never a UUID, and reject
an unknown key nowhere — `point_id` would simply be stored as an ordinary
payload field on the new point. To round-trip an export back into a
collection:

1. Strip `point_id` from every chunk.
2. Make sure each chunk has an integer `id`, unique within the call — the
   literal key `id`, whatever `chunk_id_key` the original ingest used. An
   export taken with the default `chunk_id_key` already carries the right
   ordinal under `id`; one taken with a custom key carries it under that name
   and needs renaming back, and an older export has no ordinal at all and needs
   `id` numbered from 0 (or 1) within each `source_id`.
3. Keep `source_id` and `text` as-is; the middleware will re-embed under the
   currently active profile and recompute `UUID5(source_id:chunk_id)` — which
   **overwrites** the original points if `source_id` and the chunk ordinals
   match, rather than duplicating.

Older exports (written before the `point_id` split) carry the UUID under `id`
instead. Those need the stripping-and-renumbering the previous version of this
playbook described: an `id` that is a UUID string is the tell.

If the goal is a backup you can restore byte-for-byte (same UUIDs, same
vectors), export is the wrong tool — there is no vector-preserving dump
endpoint in this stack. Snapshot the Qdrant storage volume at the operator
level instead.

### What export is **not**

- **Not a snapshot.** The write itself is now *complete* — the middleware
  writes to a temp file and renames it into place, so you either get the
  whole document or, on a mid-walk failure, nothing at all (see the 502 row
  below); there is no such thing as a half-written export. It is still not
  *consistent*, though: the middleware's internal scroll pages are not
  isolated from concurrent writes, so points written or deleted while the
  export is running can still appear or disappear between pages. Complete,
  not consistent — say it that way, not "partial".
- **Not vector-preserving.** The document is `{"chunks": [...], "metadata":
  {...}}` — payload only, never the stored embeddings. Re-ingesting it always
  re-embeds under the active profile (see the id trap above).

## 8. Error text → next call

Keyed by the text the tool returns; the full table is `docs://troubleshooting`.

| Text | Next |
|---|---|
| `Error: this operation destroys data: set confirm to the exact <noun> '<target>'` | Zero HTTP calls were made. Verify the target is right, then resend with that exact string — not trimmed, not re-cased. |
| `Error: invalid request — <field>: <reason>` | Fix that field. `File not found: <path>` here means the file was never staged under `data/` (playbook 2); `file_path must be inside the ingest data/ directory` means a `..` or absolute path tried to leave it; `File too large` means it exceeds `INGEST_MAX_FILE_BYTES` (64 MiB shipped). |
| `Error: invalid request — '<key>' has no payload index that can serve this filter (it needs type <types>): ...` | Strict mode refused the filter. `qdrant_list_payload_indexes`: no index on `<key>` → `qdrant_create_payload_index` with one of `<types>`, then resend; indexed with another type → the filter's value type is wrong, fix the filter. |
| `Error: reindex state conflict — ...` | The job cannot take that control action in its current state (e.g. `pause` after it completed). `qdrant_get_reindex_status`, then choose again. |
| `Error: not found — ...` | `qdrant_list_collections` / `qdrant_read_points` to see what exists. |
| `Error: embedding contract mismatch — ...` | `qdrant_verify_collection`. Two fixes only: restart Smart Embedder under the original profile, or reindex (playbook 4). |
| `Error: blocked by an active reindex job — ...` | `qdrant_get_reindex_status`, then wait for the job to finish. A paused job blocks too; only completion or a user-approved `cancel` (which abandons its progress) lifts the block — never cancel just to get past it. |
| `Error: conflict — ...` | `qdrant_get_collection` to inspect actual state. |
| `Request body too large` (bare, no `Error:` prefix, no `is_error`) | Transport's 4 MiB cap, before any tool ran. Split the call or stage a file. |
| `Error: request body too large.` | Through MCP: one ingest call carried more than `INGEST_MAX_CHUNKS` chunks (10000 shipped). Split it into calls sharing one `source_id`. (The middleware's 25 MiB body limit maps to the same text but sits behind the 4 MiB transport cap.) |
| `Error: collection too large to export — ...` | The export exceeded its point/byte cap; no file was published. Ask the operator to review the named cap and capacity, or use a Qdrant snapshot. There is no filter parameter. |
| `Error: inference admission budget exceeded.` | Reduce batch size, input length or vector output before retrying. |
| `Error: inference capacity remained occupied after retries.` | Wait for active work to drain or reduce concurrency. |
| `Error: export capacity is occupied.` | Retry after the active export ends. |
| `Error: rate limited by the middleware.` | Wait, reduce concurrency, retry. |
| `Error: a dependency is not ready (Qdrant or the embedder).` | `qdrant_check_health` to see which. |
| `Error: the middleware did not respond (transient network failure)...` | A read was already retried (up to 3 attempts, `MCP_RETRY_ATTEMPTS`); a write was sent once, never retried. Retry a read. For a write, check first (`qdrant_count_points`, `qdrant_read_points`) — ingest and upsert are idempotent, so re-sending the same chunks is safe. If it persists, `qdrant_check_health`. |
| `Error: the middleware failed with status 502 — Qdrant error: ...` (from `qdrant_export_collection`) | A page of the walk failed mid-export. Nothing was written — there is no such thing as a partial export file. `qdrant_check_health` first, then retry the whole call. |
| `Error: the MCP server is not authorized against the middleware.` | Server config problem (`MIDDLEWARE_CLIENT_TOKEN`). Report it; retrying cannot fix it. |
| `Error: replace is only supported for explicit ids...` | Use `mode="merge"`, or resolve the filter to ids with `qdrant_scroll_points`. |
| `Error: selector='filter' requires a non-empty filters object...` | An empty filter would match every point. Supply real conditions or switch to ids. |
| `Error: recreate=true requires an explicit collection_name` / `mode='replace' requires...` / `replace_source=true requires an explicit source_id` | A defaulted value cannot be confirmed against. Pass it explicitly plus a matching `confirm`. |
| `Error: points contains duplicate point_id values: <ids>` | One `qdrant_upsert_points` call named the same id twice — it would silently overwrite itself. Dedupe or merge, then resend. |
| `Error: at least one schema toggle must be provided` | `qdrant_update_schema_toggles` with every toggle omitted. Set at least one. |
| `Error: payload keys cannot be blank` | A `keys` entry in `qdrant_delete_payload_keys` was whitespace. Remove it and resend. |
| `Error: phrase_matching requires field_schema 'text'` | `phrase_matching=true` on a non-`text` index. Drop it or set `field_schema="text"`. |
| `Error: invalid request — key '<field>' cannot be faceted: ...` | The field has no index, or one that cannot match values (`float`, `geo`, `text`). Faceting needs `keyword`, `integer`, `bool`, or `uuid` — see § Choosing `field_schema` in `reference.md`. `qdrant_list_payload_indexes` shows which it has. |
| A filter returns 0 hits but `qdrant_facet_points`/`qdrant_read_points` show the values exist | Type disagreement, not a missing point: Qdrant only evaluates a condition over values whose type fits it, so a range condition against digits stored as strings matches nothing and raises nothing. Read a point, compare the JSON type to the index schema, and fix the payload or the schema. |
| `Error: duplicate payload index field: <field_name>` | One `qdrant_create_payload_index` call listed the same field twice. Merge into one entry. |
| `Error executing tool ...: 1 validation error ...` (pydantic URL) | Plain shape/type mistake — wrong type, missing required field, bad enum. Check `reference.md` for that tool's params. |
