---
name: smart-qdrant-mcp
description: "Drives the smart-qdrant-mcp server's `qdrant_*` MCP tools and carries the operating rules every call relies on: hybrid search tuning, ingest discipline, collection lifecycle, payload and point writes, reindexing, destructive-operation gates and error recovery. Use when calling any `qdrant_*` tool or `qhm://` resource for any purpose, a health check or a collection listing as much as a search or an ingest, and before the first such call of a task; when searching, ingesting, inspecting, reindexing or deleting in a Qdrant collection; or when a qdrant_* call returns an error, refuses a confirm value, returns nothing, or the tools are missing from the client."
---

# Driving the smart-qdrant-mcp server

This skill governs every `smart-qdrant-mcp` call, a health check or a
collection listing as much as a search or an ingest, from the first call of a
task: the tool schemas give the arguments, these rules the practice.

## Overview

`smart-qdrant-mcp` is a stateless translation layer: 28 `qdrant_*` tools over the
middleware's REST API (port 6337 → 6336 → Smart Embedder / Qdrant). It stores
nothing itself. Every tool takes its arguments nested one level under `params`
(`{"params": {...}}`).

The server ships its own teaching material. Read it instead of guessing —
nine `docs://` resources: `docs://concepts/hybrid-search`,
`docs://concepts/fusion-and-rerank`, `docs://concepts/qdrant-data-model`,
`docs://concepts/embedding-contract`, `docs://reference/filters`,
`docs://workflows/ingest`, `docs://workflows/search-recipes`,
`docs://workflows/reindex`, `docs://troubleshooting`. It also serves the
application's own documentation, copied into the server: six guide pages
(`docs://guide/getting-started`, `docs://guide/cli`, `docs://guide/environment`,
`docs://guide/architecture`, `docs://guide/hybrid-search-rerank`, `docs://guide/embedding-models`) for how a user
runs and configures the stack, and the six component references those pages link
to (`docs://reference/middleware-api`, `docs://reference/launcher-scripts`,
`docs://reference/smart-embedder`, `docs://reference/mcp-server`,
`docs://reference/agent-skills`, `docs://reference/smart-qdrant-skill`) for the
technical detail behind them. Plus three live-state resources readable without
spending a tool call (`qhm://collections`, `qhm://health`, `qhm://metrics`). Three prompts
(`qdrant_bootstrap_collection`, `qdrant_diagnose_search_quality`,
`qdrant_safe_reindex`) already encode the multi-tool orders that are easy to get
wrong — prefer them over improvising a sequence.

## How search works in this stack

`qdrant_search` never runs one retriever: every call embeds the query twice
(dense = meaning, sparse = exact terms), fuses the two rankings server-side
(RRF by default), then — by default — re-scores the fused head with a
cross-encoder reranker. Dense finds paraphrases and synonyms; sparse finds
rare identifiers and literal strings dense would smooth away. ColBERT /
multivector search exists in neither profile here — it is turned off at the
embedding call by design, so "hybrid" always means dense + sparse + fusion +
optional rerank. Theory and tuning: `docs://concepts/hybrid-search` and
`docs://concepts/fusion-and-rerank`.

Shipped defaults (`.env.example`, all reconfigurable): `limit=10` (max 50),
`candidate_limit=50` (max 200), `fusion=rrf`, `rerank=true`,
`rerank_top_k=20`, rerank input truncated at 1000 chars per hit,
`rescore=true`, `oversampling=2.0` (both no-ops unless the collection is
quantized). Every knob and when to touch it: `reference.md` § Search.

Read the error detail on `413`: inference admission and rerank capacity
require different reductions; export has separate point/byte caps. Occupied
inference or export capacity is retryable after active work drains. A `502`
mentioning `rerank_vram_pressure` is also transient. Detail: `reference.md` §
Search, `docs://troubleshooting`.

## Pick the cheapest tool that answers the question

`qdrant_search` pays for two embeddings, a fusion, and (by default) a
cross-encoder pass. Most questions do not need it.

| You want | Tool |
|---|---|
| Points whose ids you already know | `qdrant_read_points` (≤1000 ids, no embedding) |
| A number only | `qdrant_count_points` |
| Distinct values of one indexed field | `qdrant_facet_points` |
| Enumerate / filter, no ranking | `qdrant_scroll_points` (≤1000 per page) |
| Whole collection to a file | `qdrant_export_collection` — one call, not N `qdrant_scroll_points` pages. See `playbooks.md` §7 for the re-ingest id trap and this tool's point/byte caps and 413/503/502 recovery. |
| Semantic answer from stored text | `qdrant_search` |
| Is the stack alive? | `qdrant_check_health` — the first call whenever anything fails |

## Search: three defaults to set every time

1. **Pass `limit` explicitly.** Omit it and the middleware picks its own page
   size without echoing it, so `has_more` is reported conservatively and only an
   empty page proves the end.
2. **Pass `with_payload`** (e.g. `["text", "source_id"]`). Omitted means every
   payload field on every hit, straight into context.
3. **`response_format="json"`** when diagnosing (fusion, candidates, score
   spread); markdown when reading results.

Empty results are a diagnosis, not an answer: wrong collection, over-narrow
filter, or an embedding-contract mismatch. Run `qdrant_diagnose_search_quality`.

## Destructive tools: the confirm gate

`confirm` is compared exactly — case- and whitespace-sensitive, never stripped or
normalised. A mismatch makes **zero HTTP calls**; nothing happened.

- `confirm` = the **collection name**, except for the ingest tools, where it is
  the **`source_id`**.
- `qdrant_delete_points` with `selector="filter"` and **no** `confirm` is a free
  dry-run count. Always run that first, report the count, then confirm.
- `qdrant_delete_points` with `selector="ids"`, `qdrant_delete_collection`,
  `qdrant_clear_payload`, `qdrant_delete_payload_index` have **no dry run**.
- Full gate table with every trigger: `reference.md`.

**Red flags — stop and ask the user instead:**

- Filling `confirm` on the same turn you first learned the operation is
  destructive, without showing a count or an inventory first.
- Retrying a refused `confirm` with a trimmed, lower-cased, or "yes"-style value.
- Setting `replace_source=true` to "clean up" — its delete is scoped by exact
  equality on the typed `source_id`, so a reused or mistyped id wipes an
  unrelated source wholesale.
- Cancelling a reindex job to get past a 409. That rejection protects a
  checkpoint; `pause` keeps it, `cancel` abandons the progress.

## Collection schema: what you cannot set

`qdrant_create_collection` takes the four toggles (`dense_on_disk`,
`quantization`, `sparse_idf`, `sparse_on_disk`), `on_disk_payload`, and nothing
else about the shape. Vector dimension, distance metric and vector names have
**no parameter at all** — they follow the running embedding profile (dense
dimension 1024 under both `basic` and `advanced`). A request for a different
dimension is answered by saying so, not by calling the tool and surfacing a
validation error.

Same shape of answer for `qdrant_export_collection`: it exports the whole
collection, with no filter parameter and no vector parameter, by design. A
filtered subset is a `qdrant_scroll_points` walk, not an export argument.

## Ingest routing

- Chunks already in this conversation → `qdrant_ingest_chunks`.
- File already under `middleware-qdrant-manager/data/` → `qdrant_ingest_file`
  (`file_path` is relative to that directory, on the middleware's host).
- File anywhere else → **ask the user to stage it**. No tool on this server can
  write there; the mount is read-only by design. Do not hunt for one.

Inline requests die at the transport's 4 MiB cap with a bare
`Request body too large` — no `Error:` prefix, no `is_error` flag. Split the call
or stage a file. A slow ingest can also blow `MCP_HTTP_TIMEOUT` (120 s) while the
middleware finishes writing; point ids are deterministic (`UUID5(source_id:id)`),
so verify with `qdrant_count_points` and re-send the same chunks rather than
assuming failure.

Ingest and export are mirror images, not a round trip: `qdrant_ingest_file`
reads from `data/` (read-only, operator-staged — no tool writes there);
`qdrant_export_collection` writes to `exports/` (server-written — no tool
reads it back). An exported file is **not** re-ingestible verbatim: it carries
UUID point ids, the ingest tools want integer chunk ordinals. See `playbooks.md`
§7, "Re-ingesting an export — the id trap".

## References

Detailed documentation lives in `references/`, read on demand — it is not part
of this file's always-loaded context:

- `reference.md` — all 28 tools, every parameter, gates, limits, hard caps;
  § Search covers inference admission and rerank capacity failures,
  including deterministic refusals versus retryable occupied capacity;
  § Choosing `field_schema` picks a payload index type from the operations it
  has to support, and names the two failures that are silent (wrong type over
  the stored values; a schema change that rebuilds the index).
- `playbooks.md` — bootstrap, large-corpus staging, search diagnosis and
  tuning recipes, safe reindex, safe deletion, collection export (the
  `qdrant_export_collection` call, the re-ingest id trap, and its own
  point/byte cap and capacity recovery paths), error text → next call, and
  "the tools aren't there".

## Staying aligned with the server

This skill must describe the server as it is, never as it was. The server's
source of truth is `smart_qdrant_mcp/tools/*.py` (every tool's input model,
gate, and limit — the client's tool list is generated from these),
`tool_map.TOOL_ROUTES` (tool → middleware endpoint), `resources/docs.py`
(the nine docs), and `prompts.py` (the three prompts); the server's own
tests fail when a prompt or doc references something that no longer exists.
When the server gains, renames, or re-params a tool, update
`reference.md` (and any affected playbook) in the same change, and verify
against the client's live tool list. Qdrant features this stack does not
expose (Formula Query, per-prefetch score thresholds, group_by,
recommend/discover, ColBERT) are listed as such in `reference.md` — never
propose them.
