# Troubleshooting

This table is keyed by what an agent actually sees in a tool's return
text, not by HTTP status codes — every row below maps directly to a
message `describe_error` produces, plus every refusal this MCP server
issues itself before any HTTP call is made. Every one of these arrives with
the tool call's `is_error` flag set to `true`, so a client checking that
flag (rather than pattern-matching the text) can tell the failure apart
from a genuine answer.

Every one of those self-issued refusals returns the same clean
`Error: ...` sentence, never the MCP SDK's own validation-error envelope
(`"Error executing tool ...: 1 validation error ..."`, with a
`pydantic.dev` URL). Only ordinary shape and type mistakes — a missing
required field, a value outside its allowed range, an unrecognised enum
member — still surface as that raw envelope; every cross-field business
rule listed below is raised as a guardrail refusal instead, specifically so
its message is not buried in framework noise.

One row below is an exception to that pattern: a request can be rejected by
the streamable-HTTP transport before any tool call is even attempted, so it
carries no `is_error` flag, no JSON-RPC envelope, and no `Error:` prefix. It
is marked as such where it appears.

## Middleware-reported errors

| What you see | What it means | What to call next |
|---|---|---|
| `Error: invalid request — <field>: <reason>` | The request body failed validation before the middleware attempted anything — a bad filter shape, wrong type, out-of-range value, or (for `qdrant_ingest_file`) a `file_path` not yet staged under the middleware's `data/` directory, reported as `File not found: <path>`. | Fix the named field and resend. For filter-shaped input, check `docs://reference/filters`; for a missing staged file, read `docs://workflows/ingest`. |
| `Error: not found — ...` | The named collection or point does not exist. | `qdrant_list_collections` to see what actually exists, or `qdrant_read_points` to confirm a point id. |
| `Error: embedding contract mismatch — ...` | The collection's stamped vector contract does not match the currently active embedder profile. | Read `docs://concepts/embedding-contract`, then `qdrant_verify_collection` to see exactly what mismatched. |
| `Error: blocked by an active reindex job — ...` | A running or paused reindex job is protecting this collection from mutation; this is deliberate. | `qdrant_get_reindex_status` for progress; `qdrant_control_reindex` to pause or cancel it if the mutation must proceed now. |
| `Error: reindex state conflict — ...` | The current job does not support the requested control action (for example, pause after completion). | Inspect `qdrant_get_reindex_status` before choosing another action. |
| `Error: conflict — ...` | Some other state conflict — for example, a schema toggle update did not take effect as requested. | `qdrant_get_collection` to inspect the collection's actual current state. |
| `Error: rerank capacity exceeded — ...` | A `qdrant_search` call with reranking enabled sent a query/passage pair too large to fit in the Smart Embedder's GPU memory, even alone. This is deterministic — **retrying will not help**, and it is not about how many passages were sent. | Shorten passages: lower `QUERY_RERANK_MAX_CHARS` on the middleware, or `QWEN_RERANK_MAX_LENGTH` on the Smart Embedder. Do **not** retry and do not lower `rerank_top_k` — passage count is not the lever. |
| `Error: request body too large.` | Scope: the *middleware's* own body-size limit (25 MiB by default), applied to every request regardless of tool — a distinct, much higher ceiling than the transport rejection below. In practice only a bulk write (`qdrant_upsert_points`, `qdrant_ingest_chunks`) sends enough bytes to hit it; this is unrelated to the rerank-capacity row above, which is never about request size. | Split points or chunks across several smaller calls. |
| `Request body too large` (bare text — no `Error:` prefix, no JSON-RPC envelope) | The streamable-HTTP transport rejected the request at its own 4 MiB cap, before this server's code or the middleware ever ran. This is the limit an MCP caller actually hits first on an oversized inline `qdrant_ingest_chunks` call. | Read `docs://workflows/ingest` for the size limits, then split the request across smaller calls or stage a file for `qdrant_ingest_file`. |
| `Error: rate limited by the middleware.` | Too many requests arrived too quickly. | Wait briefly, reduce concurrent calls, then retry. |
| `Error: a dependency is not ready (Qdrant or the embedder).` | Qdrant or Smart Embedder is not currently reachable from the middleware. | `qdrant_check_health` to see which dependency is down. |
| `Error: the middleware did not respond (transient network failure). Retries were already attempted...` | A timeout or connection failure, already retried once by this server's own retry policy. | Retry the call; if it keeps failing, `qdrant_check_health`. |
| `Error: the MCP server is not authorized against the middleware.` | This server's own bearer token is rejected by the middleware — a server configuration problem, not a request problem. | Report this rather than retrying; no client-side change fixes it. |
| `Error: the middleware failed with status 502 ...` mentioning `Smart Embedder server error 503` and `rerank_vram_pressure` | A `qdrant_search` rerank batch hit transient GPU memory pressure on the Smart Embedder. The middleware already retried it three times before giving up and wrapping it as a generic 502 — this is **not** a dependency outage, so `qdrant_check_health` will report everything healthy and send you looking in the wrong place. | Retry the call after a short wait, or lower `rerank_top_k` (fewer passages reach the reranker) or the middleware's `SMART_EMBEDDER_RERANK_BATCH_SIZE` (smaller sub-batches per call to the Smart Embedder). |
| `Error: collection too large to export — ...` | The whole-collection export exceeded its point or exact encoded-byte cap, possibly during the walk. No file was published. | Ask the operator to review the named cap and memory capacity; this tool has no filter parameter. |
| `Error: export capacity is occupied.` | All per-app export slots are in use. | Retry after the active export finishes. |
| `Error: inference admission budget exceeded.` | One request exceeds the conservative token or output estimate budget (413). | Reduce batch size, input length or requested vector output. |
| `Error: inference capacity remained occupied after retries.` | Native work holds the shared budget; upstream `request_budget_busy` 503 persisted through retries. | Retry after active work drains or reduce concurrent calls. |
| `Error: the middleware failed with status <N> ...` | Any other unmapped failure mode not covered by the rows above. | `qdrant_check_health` to check dependency status, then retry. |

## Refusals made by this MCP server itself (no HTTP call at all)

| What you see | What it means | What to call next |
|---|---|---|
| `Error: this operation destroys data: set confirm to the exact <noun> '<target>' to proceed` | A destructive tool (collection delete or recreate, point delete by id, ingest with `replace_source=true`, payload clear, payload index delete, or reindex start) was called with a missing or mismatched `confirm`. | Resend with `confirm` set to the exact value named, once you are certain that value is correct. |
| `Error: replace is only supported for explicit ids: the middleware has no replace-by-filter endpoint...` | `qdrant_set_payload` was called with `selector="filter"` and `mode="replace"` — a full replace cannot be expressed as one filtered write across many differing payloads. | Use `mode="merge"` instead, or resolve the filter to explicit ids with `qdrant_scroll_points` and call again with `selector="ids"`. |
| `Error: recreate=true requires an explicit collection_name so that confirm can be checked against it` | `qdrant_create_collection` was called with `recreate=true` but no explicit `collection_name` — a defaulted name cannot be checked against a confirmation token. | Resend with `collection_name` set explicitly, plus a matching `confirm`. |
| `Error: at least one schema toggle must be provided` | `qdrant_update_schema_toggles` was called with every toggle omitted, so there is nothing to change. | Resend with at least one of `dense_on_disk`, `quantization`, `sparse_idf`, `sparse_on_disk`, or `on_disk_payload` set. |
| `Error: replace_source=true requires an explicit source_id (not the file-name default) so that confirm can be checked against it` | `qdrant_ingest_file` was called with `replace_source=true` but no explicit `source_id` — the file-name default cannot be safely confirmed against. | Resend with `source_id` set explicitly, plus a matching `confirm`. |
| `Error: points contains duplicate point_id values` | `qdrant_upsert_points` was called with the same `point_id` more than once in one batch. | Remove or merge the duplicate entries and resend. |
| `Error: selector='ids' requires a non-empty ids list` | A tool that addresses points by id or filter (`qdrant_set_payload`, `qdrant_delete_payload_keys`, `qdrant_clear_payload`, `qdrant_delete_points`) was called with `selector="ids"` but an empty or missing `ids` list. | Resend with a non-empty `ids` list, or switch to `selector="filter"` with a `filters` object. |
| `Error: selector='filter' requires a non-empty filters object; an empty filter would match every point` | The same tools as above were called with `selector="filter"` but an empty or missing `filters` object, which would otherwise match every point. | Resend with a non-empty `filters` object, or switch to `selector="ids"` with an `ids` list. |
| `Error: mode='replace' requires an explicit collection_name so that confirm can be checked against it` | `qdrant_set_payload` was called with `mode="replace"` but no explicit `collection_name` — a defaulted name cannot be checked against a confirmation token. | Resend with `collection_name` set explicitly, plus a matching `confirm`. |
| `Error: payload keys cannot be blank` | `qdrant_delete_payload_keys` was called with a `keys` entry that is blank after whitespace is stripped. | Remove the blank entry and resend. |
| `Error: phrase_matching requires field_schema 'text'` | `qdrant_create_payload_index` was called with `phrase_matching=true` on an index whose `field_schema` is not `'text'`. | Either drop `phrase_matching`, or set `field_schema='text'` for that index. |
| `Error: duplicate payload index field: <field_name>` | `qdrant_create_payload_index` was called with the same `field_name` more than once in one call. | Remove or merge the duplicate entries and resend. |

## Related tools

`qdrant_check_health`, `qdrant_list_collections`, `qdrant_get_reindex_status`,
`qdrant_verify_collection`.
