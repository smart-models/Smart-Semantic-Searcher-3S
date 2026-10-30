# smart-qdrant-mcp — the MCP surface over the middleware

> **Read this when** you need to answer what an MCP client can do against this
> stack, look up a `qdrant_*` tool, a confirmation gate, a resource URI or an
> environment variable, connect a client, or change the MCP server's code.
> **Audience** — an AI agent working in or on this repository, and any human
> reading over its shoulder.
> **Source of truth** — this file for the server's shape, configuration and
> guardrails; `smart_qdrant_mcp/tools/*.py` for each tool's exact input model
> (the client's tool list is generated from those); the `docs://` resources the
> server itself serves for agent-facing teaching material.
> **Prerequisite** — [`../middleware-qdrant-manager/README.md`](../middleware-qdrant-manager/README.md),
> which owns every behaviour these tools expose.

---

## Purpose and boundaries

An MCP (Model Context Protocol) server that exposes the Qdrant middleware's
REST API to MCP clients — agents, IDE assistants, or anything else that speaks
MCP over streamable HTTP — as typed tools, documentation resources and workflow
prompts, instead of raw HTTP calls.

```
Client / MCP client → smart-qdrant-mcp (MCP, 6337) → middleware-qdrant-manager (6336) → Smart Embedder / Qdrant
```

**What it is:**

- A thin, stateless translation layer. Its complete surface covers every
  middleware operation: **28 MCP tools** map all tool-eligible routes, while
  `GET /metrics` is intentionally exposed only through a live resource. It
  also exposes **24 resources** — 21 static `docs://` documents and 3 `qhm://`
  live-state proxies — and **3 workflow prompts**.
- Optional. The stack works without it; it only adds an agent-facing surface
  over the same REST contract any HTTP client can already use.

**What it is not:**

- Not a data store. Every tool call becomes one or more HTTP requests to the
  middleware; every response is reshaped into markdown (default) or JSON.
- Not a Qdrant or Smart Embedder client. It never talks to either directly.
- Not internet-facing. See [Invariants](#invariants).

Built on `mcp==2.0.0`, whose installed SDK reports `2026-07-28` as its latest
protocol version and `2025-11-25` as its latest handshake version
(`mcp.types.version.LATEST_MODERN_VERSION` / `LATEST_HANDSHAKE_VERSION`); both
are in `SUPPORTED_PROTOCOL_VERSIONS`, so a client negotiating either succeeds.

## Mental model

1. **Every tool is a middleware call.** If a behaviour is not in the
   middleware, no tool here can produce it. Read the middleware README to
   answer "why did it do that?"; read this one to answer "which tool, with
   which arguments?".
2. **Destructive tools refuse before they call.** A missing or mismatched
   `confirm` makes zero HTTP requests. The comparison is exact — case- and
   whitespace-sensitive.
3. **The server is stateful per session and has no event store.** Correct for
   the single container `compose.yml` runs; wrong for replicas behind a load
   balancer.
4. **Two limits are transport-level, not tool-level**, so they produce errors
   with no tool message to explain them: a 4 MiB request body and one global
   outbound timeout. See [Operational limits](#operational-limits).
5. **Two addresses answer two different questions.** `MCP_BIND_HOST` is the
   published/policed address; `MCP_LISTEN_HOST` is the socket bind. Confusing
   them is the most common configuration mistake here.
6. **Documentation is served, not just written.** Twenty-one `docs://`
   documents — nine teaching documents plus the application's own guide and
   component references, copied into the package — and three `qhm://`
   live-state proxies are part of the server's surface; prefer pointing a
   client at those over restating them.

## Map of the code

| Path | Role | Change it when |
|---|---|---|
| `smart_qdrant_mcp/__main__.py`, `server.py` | Process entry point, server/ASGI assembly, handshake instructions, tool/resource/prompt registration, per-call logging and lifespan. | Startup, transport, registration or instructions change. |
| `smart_qdrant_mcp/auth.py` | Token verification and static/OAuth authentication assembly. | Inbound authentication changes. |
| `smart_qdrant_mcp/client.py`, `context.py` | Middleware HTTP gateway and the context tools receive. | HTTP lifecycle, safe-read retries or per-call dependencies change. |
| `smart_qdrant_mcp/guardrails.py` | Shared validation and exact destructive confirmation gates. | A tool's pre-transport safety contract changes. |
| `smart_qdrant_mcp/errors.py`, `formatting.py` | Middleware error translation and client-facing output formatting. | Error guidance or markdown/JSON presentation changes. |
| `smart_qdrant_mcp/config.py` | Every environment variable, read once at process start. No runtime reload. | Configuration changes. |
| `smart_qdrant_mcp/tools/*.py` | One module per tool group. Each tool's input model, confirmation gate and limits — the client's tool list is generated from these. | A tool is added, renamed or re-parametered. |
| `smart_qdrant_mcp/tool_map.py` | Tool → middleware endpoint mapping. A test compares it against the middleware's actually registered routes on every run. | The same commit as any tool change. |
| `smart_qdrant_mcp/resources/docs.py`, `resources/docs/*.md` | The nine static teaching `docs://` documents. | The teaching material changes. |
| `resources/docs/guide/`, `resources/docs/reference/` | Copies of the app's own guide pages and the component READMEs they link to, served as `docs://guide/*` and `docs://reference/*`. A test compares every copy byte-for-byte against its repository original. | Any of those originals changes — re-copy it in the same commit. |
| `smart_qdrant_mcp/resources/live.py` | The three `qhm://` live-state proxies. | A live snapshot is added. |
| `smart_qdrant_mcp/prompts.py` | The three workflow prompts. `tests/test_prompts.py` checks every tool and URI they name against reality. | A workflow's correct order changes. |
| `docs/ingest-patterns.md` | Repository-side note on how corpora reach the middleware's `data/` mount, and why staging is outside MCP. | The staging story changes. |
| `tests/` | pytest suite; fakes the middleware with `httpx.MockTransport`. | Always, alongside the change. |
| `evals/` | Fixture corpus, seed script and the ten-question evaluation set. | The tool surface changes materially. |
| `evals/harness/` | Vendored MCP evaluation client — see [its README](evals/harness/README.md). | Re-copied from upstream, not edited. |

The matching **client-side** skill lives in
[`../smart-skill/`](../smart-skill/README.md) and must be updated in the same
change as a tool rename or re-parameterisation; nothing tests that link.

### Agent reading path

For a tool change, read its input model and handler in `tools/`, follow its
calls into `client.py`, and compare the endpoint mapping in `tool_map.py`
with the middleware router. Validation and confirmation must happen before
transport where the tool contract requires it. Never add direct Qdrant or
embedder access to solve a translation-layer problem.

Inspect `errors.py` when a correct HTTP refusal becomes misleading agent
advice, and `formatting.py` when structured data is correct but presentation
loses information. Keep tool documentation, resource text and the client-side
skill aligned, while leaving their distinct ownership intact. Live agent
acceptance is specified in the
[testbook agent workflow section](../middleware-qdrant-manager/frontend/tests/testbook/test-book-e2e.md#agent-workflows);
mock transport tests and a scripted MCP seed do not prove skill-guided usage.

## Interface reference

### Client configuration

Point any MCP client that supports streamable HTTP at the server and supply a
bearer token. In the default `static` mode it must be one of the values in
`MCP_API_TOKENS`; in `oauth` mode it must be an access token accepted by the
configured RFC 7662 introspection endpoint:

```json
{
  "mcpServers": {
    "smart_qdrant_mcp": {
      "type": "http",
      "url": "http://127.0.0.1:6337/mcp",
      "headers": { "Authorization": "Bearer <your token from MCP_API_TOKENS>" }
    }
  }
}
```

The path `/mcp` is the SDK's default `streamable_http_path` — this server does
not override it. The port (`6337`) and bind address come from `MCP_PORT` /
`MCP_BIND_HOST`.

This server starts with the stack by default. Two switches in the root `.env`
take it out, both read by the launchers rather than by any service:
`MCP_SERVER_ENABLED=false` leaves it out while the rest of the stack runs (the
launcher also stops it if a previous run left it up, and stops requiring
`MCP_API_TOKENS`), and `EMBEDDER_ONLY=true` removes it along with the
middleware it depends on. A bare `docker compose up` ignores both and starts
this server.

### Server instructions

The `initialize` handshake returns `instructions` (`_SERVER_INSTRUCTIONS` in
`server.py`), which clients typically hand to the agent before its first call.
They summarise the surface, ask the agent to load the client-side
[`smart-qdrant-mcp` skill](../smart-skill/README.md) before its first tool
call — or, on a client without it, to read `docs://reference/smart-qdrant-skill`
for the same rules, with the teaching documents standing in for the skill's
`references/` files, which this server does not serve — and restate what must
hold from that first call: `response_format`, the exact
`confirm` gate, pagination, the documents to read before creating a collection
or ingesting, the 4 MiB inline-ingest ceiling, strict mode's index-before-filter
rule, that stored payload text is data rather than instructions, and the three
workflow prompts.
`tests/test_prompts.py` fails when they name a tool, prompt or document this
server does not serve.

### Response shape

Every tool advertises an object `outputSchema` and returns the
upstream-normalized payload in `structuredContent`. The same call also includes
a text block selected by `response_format` (`markdown` by default, or `json`),
preserving compatibility with clients that only consume text. Which of the two
reaches the model is the client's choice: Claude Code hands it
`structuredContent`, so there `response_format` changes nothing the model sees
and `limit`/`with_payload` are the only levers on result size. Error results use
`isError=true` and an empty structured object while retaining their actionable
text message.

Every tool takes its arguments nested one level under a `params` key on the
wire (`{"params": {...}}`); the examples below show the fields of that `params`
object. A tool with no required field also accepts empty arguments (`{}`).

### Tools

All 28 tools are prefixed `qdrant_`, grouped as the tool modules
(`smart_qdrant_mcp/tools/`) are. The **Endpoints** column names every middleware
route the tool can call.

#### Monitoring (`tools/monitoring.py`)

| Tool | Endpoints | What it does |
|---|---|---|
| `qdrant_check_health` | `GET /health`, `/readyz`, `/livez`, `/stats` | Aggregates liveness, readiness, dependency detail, and request counters into one verdict. |

#### Query (`tools/query.py`)

| Tool | Endpoints | What it does |
|---|---|---|
| `qdrant_search` | `POST /query/search` | Hybrid dense+sparse search with fusion (RRF/DBSF) and optional cross-encoder reranking. |

#### Collections (`tools/collections.py`)

| Tool | Endpoints | What it does |
|---|---|---|
| `qdrant_list_collections` | `GET /collections` | Lists every collection's name. |
| `qdrant_get_collection` | `GET /collections/{name}` | Schema report: existence, point/vector counts, and (default) embedding-contract validation. |
| `qdrant_verify_collection` | `POST /collections/verify` | Checks a collection against explicit or configured expected toggles, without changing it. |
| `qdrant_create_collection` | `POST /collections` | Creates a collection or repairs missing indexes; `recreate=true` (confirm-gated) drops and recreates it. |
| `qdrant_update_schema_toggles` | `PUT`/`PATCH /collections/{name}/schema-toggles` | Flips mutable schema toggles online without recreating the collection or losing points. |
| `qdrant_delete_collection` | `DELETE /collections/{name}` | Permanently deletes a collection and every point in it. Confirm-gated. |

#### Payload indexes (`tools/payload_indexes.py`)

| Tool | Endpoints | What it does |
|---|---|---|
| `qdrant_list_payload_keys` | `GET /collections/{name}/payload-keys` | Samples points to discover observed payload keys (a lower bound, not authoritative). |
| `qdrant_list_payload_indexes` | `GET /collections/{name}/payload-indexes` | Lists the payload indexes actually configured (authoritative). |
| `qdrant_create_payload_index` | `POST /collections/{name}/payload-indexes` | Creates one or more payload indexes, or type-repairs an existing one. |
| `qdrant_delete_payload_index` | `DELETE /collections/{name}/payload-indexes/{field}` | Permanently deletes one payload index. Confirm-gated. |

#### Reindex (`tools/reindex.py`)

| Tool | Endpoints | What it does |
|---|---|---|
| `qdrant_reindex_preflight` | `POST /collections/{name}/reindex/preflight` | Read-only readiness check: eligible/invalid point counts and blockers before starting a job. |
| `qdrant_start_reindex` | `POST /collections/{name}/reindex` | Starts a background vector-regeneration job. Confirm-gated. |
| `qdrant_get_reindex_status` | `GET /collections/{name}/reindex` | Polls job progress; distinguishes no-job, an active/finished job, and unreadable metadata. |
| `qdrant_control_reindex` | `POST /collections/{name}/reindex/{pause,resume,cancel}` | Pauses (keeps checkpoint), resumes, or cancels (abandons progress) an active job. |

#### Ingest (`tools/ingest.py`)

| Tool | Endpoints | What it does |
|---|---|---|
| `qdrant_ingest_file` | `POST /points/ingest` | Embeds and upserts chunks from a JSON file on the middleware host. `replace_source=true` is confirm-gated. |
| `qdrant_ingest_chunks` | `POST /points/ingest-chunks` | Embeds and upserts chunks supplied inline in the call. `replace_source=true` is confirm-gated. |

#### Export (`tools/export.py`)

| Tool | Endpoints | What it does |
|---|---|---|
| `qdrant_export_collection` | `POST /points/export` | Walks a whole collection server-side and writes its payload as one JSON file under the middleware's `exports/` mount. The middleware derives, sanitizes and confines its timestamped, random-suffixed filename; no caller or external validator controls an output path. Each chunk is the stored payload plus a trailing `point_id` (the Qdrant point id serialized as a string; it may represent a UUID or an unsigned integer) — strip it before re-ingesting. No `filters` (whole collection only) and no `destination` (never returns the document inline). 413 over `EXPORT_MAX_POINTS` or the exact UTF-8 `EXPORT_MAX_BYTES`; 503 while `EXPORT_MAX_CONCURRENT` slots are occupied; 502 on a failed page. Refused or failed exports publish no file. **The one tool on this server whose call results in a file being written.** |

#### Point reads (`tools/points.py`)

| Tool | Endpoints | What it does |
|---|---|---|
| `qdrant_read_points` | `GET /points/{id}`, `POST /points/retrieve` | Retrieves up to 1000 points by id; reports `found`/`not_found` rather than 404ing on a miss. |
| `qdrant_scroll_points` | `POST /points/scroll` | Pages through points by filter, ordered but never ranked. |
| `qdrant_count_points` | `POST /points/count` | Counts points matching a filter, without fetching them. |
| `qdrant_facet_points` | `POST /points/facet` | Distinct values of one indexed payload field, with counts. |

#### Point writes (`tools/points.py`)

| Tool | Endpoints | What it does |
|---|---|---|
| `qdrant_upsert_points` | `POST /points`, `POST /points/batch` | Embeds text server-side and writes 1–10000 points. |
| `qdrant_delete_points` | `DELETE /points/{id}`, `DELETE /points`, `POST /points/delete-by-filter` | Deletes by id (confirm-gated, no dry run) or by filter (confirm-gated; a free dry-run count when `confirm` is omitted). **The most dangerous tool in the server.** |

#### Payload writes (`tools/payload.py`)

| Tool | Endpoints | What it does |
|---|---|---|
| `qdrant_set_payload` | `PATCH`/`PUT /points/{id}/payload`, `PATCH /points/payload` | Merges (default, no gate) or replaces (ids only, confirm-gated) non-text payload fields. Use `qdrant_upsert_points` to change text and regenerate its vector. |
| `qdrant_delete_payload_keys` | `POST /points/{id}/payload/delete-keys`, `POST /points/payload/delete-keys` | Removes named, non-protected payload keys; protected roots stay protected through quoted, nested and array paths. |
| `qdrant_clear_payload` | `POST /points/{id}/payload/clear`, `POST /points/payload/clear` | Clears every payload field except the protected `text`/`source_id`. Confirm-gated. |

Per-id (`selector="ids"`) payload writes fan out into one HTTP request per id
and are **not atomic**: a failure partway through leaves earlier writes
applied, and the response lists exactly which ids succeeded and which failed
with why.

### Resources

Twenty-one static documents (`docs://...`, `smart_qdrant_mcp/resources/docs/`)
plus three live-state proxies (`qhm://...`,
`smart_qdrant_mcp/resources/live.py`) — 24 total. None are cached:
`mcp==2.0.0` configures cache hints per JSON-RPC method (all of
`resources/read`), not per URI, and that method is shared between the two
kinds, so caching one would mis-cache the other.

The static documents come in two groups. Nine are **teaching documents**
written against this server's tool surface. The other twelve are the
**application's own documentation** — the six end-user guide pages from the
repository's `guide/` directory, and the six component references those pages
link to for the technical detail. Those twelve are *copies* held inside
`smart_qdrant_mcp/resources/`, not paths reaching out of the package: the
container image copies `smart_qdrant_mcp/` and nothing else, so a link out
would resolve in a checkout and be missing from every deployed server. The
cost of copying is drift, and
`tests/test_resources_docs.py::test_every_app_document_matches_its_repository_original`
compares each copy byte-for-byte against its original on every run.

The guide pages are served as HTML with two read-time transforms, leaving the
files on disk byte-identical to their originals: inline `<style>` and
`<script>` are stripped (roughly 85% of each page's bytes, and nothing a
client reading the document needs), and links written as repository paths
(`../scripts/README.md#recipes`) are rewritten to the `docs://` URI serving
that same document here, so a client with no checkout can follow them.

| URI | Content |
|---|---|
| `docs://concepts/hybrid-search` | Why `qdrant_search` runs dense and sparse retrieval together, and what each contributes. |
| `docs://concepts/fusion-and-rerank` | How RRF/DBSF fusion combines two rankings, and what reranking adds. |
| `docs://concepts/qdrant-data-model` | Collections, points, payloads, named vectors, payload indexes, HNSW, quantization, on-disk toggles. |
| `docs://concepts/embedding-contract` | What a collection's stamped embedding contract is, what a mismatch looks like, and how to fix it. |
| `docs://reference/filters` | The filter syntax every `filters`-accepting tool shares, and its supported operators. |
| `docs://workflows/ingest` | The shared embed/upsert/prune contract and hazards behind `qdrant_ingest_file`/`qdrant_ingest_chunks`, plus why staging a file under `data/` happens outside MCP. |
| `docs://workflows/search-recipes` | Five concrete `qdrant_search` calls for common situations, plus an empty-results checklist. |
| `docs://workflows/reindex` | The full vector-reindex lifecycle: preflight, start, checkpointing, pause/resume/cancel. |
| `docs://troubleshooting` | Every error string this server can return, keyed by the exact text, with the next tool to call. |
| `docs://guide/getting-started` | The app's end-user guide to a first run: `.env`, connecting Qdrant, `up auto`, first search. |
| `docs://guide/cli` | The app's end-user guide to the stack launchers: entry points, CPU/GPU choice, logs, applying settings. |
| `docs://guide/environment` | The app's end-user reference for `.env`, parameter by parameter. |
| `docs://guide/architecture` | The app's end-user tour of the whole local stack and its model profiles. |
| `docs://guide/hybrid-search-rerank` | The app's plain-English walkthrough of hybrid search and reranking for end users. |
| `docs://guide/embedding-models` | The app's end-user guide to the two model profiles: each model's role, source, license, the ColBERT decision and the embedding contract. |
| `docs://reference/middleware-api` | `../middleware-qdrant-manager/README.md` — the REST API this server proxies. |
| `docs://reference/launcher-scripts` | `../scripts/README.md` — the `stack.ps1`/`stack.sh` launchers and their reference. |
| `docs://reference/smart-embedder` | `../server-embedder-reranker/README.md` — the embedding and reranking service. |
| `docs://reference/mcp-server` | This README. |
| `docs://reference/agent-skills` | `../smart-skill/README.md` — the client-side agent skills. |
| `docs://reference/smart-qdrant-skill` | `../smart-skill/smart-qdrant-mcp/SKILL.md` — the skill an agent loads to drive the `qdrant_*` tools. |
| `qhm://collections` | Live `GET /collections` snapshot. |
| `qhm://health` | Live `GET /health` snapshot. |
| `qhm://metrics` | Live Prometheus text output from the middleware (`GET /metrics`) — never a tool, since a scrape payload has no use in an agent's context window. |

### Prompts

Three fixed, numbered workflows, each pointing at the `docs://` resource that
explains its reasoning rather than restating it.

| Prompt | Arguments | Walks through |
|---|---|---|
| `qdrant_bootstrap_collection` | `collection_name`, `source_path` | Health check → check for an existing collection → create → index payload keys → ingest → verify the count → one sanity search. |
| `qdrant_diagnose_search_quality` | `collection_name`, `query` | Data present? → schema/contract valid? → run the query with diagnostics → widen candidates/enable rerank → check filter indexes and values → scroll to confirm content exists vs. ranks badly. |
| `qdrant_safe_reindex` | `collection_name` | Preflight → review blockers/invalid points → start (confirm-gated) → poll status → avoid mutating the collection while active → verify and search once complete. |

## Configuration

Every variable below is read once at process start
(`smart_qdrant_mcp/config.py`); there is no runtime reload. Defaults match
`.env.example`.

| Variable | Default | Meaning |
|---|---:|---|
| `MCP_AUTH_MODE` | `static` | Authentication backend: `static` or `oauth`. |
| `MCP_API_TOKENS` | _(empty; required in `static` mode)_ | One or more inbound bearer tokens as `token:label` pairs, comma separated. The label is the audit identity (`client_id`) logged for that token; use one token per person for individual revocation. |
| `MCP_OAUTH_ISSUER_URL` | _(empty)_ | OAuth authorization-server issuer. Required in `oauth` mode; HTTPS is mandatory except for an explicit loopback host. |
| `MCP_OAUTH_RESOURCE_SERVER_URL` | _(empty)_ | Canonical URL of this MCP resource server. Required in `oauth` mode and exposed through protected-resource metadata; HTTPS is mandatory except for an explicit loopback host. |
| `MCP_OAUTH_INTROSPECTION_URL` | _(empty)_ | RFC 7662 token-introspection endpoint. Required in `oauth` mode; HTTPS is mandatory except for an explicit loopback host. |
| `MCP_OAUTH_CLIENT_ID` / `MCP_OAUTH_CLIENT_SECRET` | _(empty)_ | Confidential-client credentials used only for introspection. Required in `oauth` mode. |
| `MCP_OAUTH_AUDIENCE` | resource-server URL | Required `aud` value in every active introspection response. |
| `MCP_OAUTH_REQUIRED_SCOPES` | _(empty)_ | Comma-separated scopes required by the MCP auth middleware. |
| `MCP_BIND_HOST` | `127.0.0.1` | The **published/policed** address: what Compose publishes the port on, what `scripts/verify_compose_security.py` checks, and what seeds the Host allowlist and the issuer/resource URL. Never reaches a socket itself. Loopback (including `::1`) or an IPv4 RFC1918 literal is accepted; public, hostname, link-local, multicast and wildcard values are refused at startup. |
| `MCP_LISTEN_HOST` | `127.0.0.1` | The **actual bind** address: what `uvicorn.run(host=...)` is called with, inside this process's own network namespace. Defaults to loopback so running outside Docker is safe by default. Unlike `MCP_BIND_HOST`, `0.0.0.0`/`::` are accepted — the `smart-qdrant-mcp` Compose service sets this to `0.0.0.0` deliberately, since binding all interfaces inside its own private namespace is exactly what the published-port restriction on `MCP_BIND_HOST` assumes. |
| `MCP_PORT` | `6337` | Listening port. |
| `MCP_MIDDLEWARE_URL` | `http://127.0.0.1:6336` | Base URL the server calls for every tool. Compose overrides this with the private-network hostname (`http://middleware:6336`); the loopback default is for running outside Docker. |
| `MCP_ALLOWED_HOSTS` | _(empty)_ | Extra `Host` header values to accept, comma separated. Loopback and `localhost` are always allowed in addition to whatever is listed here. |
| `MCP_ALLOWED_ORIGINS` | _(empty)_ | Allowed `Origin` header values, comma separated. Empty means only same-origin requests are accepted. |
| `MCP_HTTP_TIMEOUT` | `120` | Per-request timeout (seconds) for calls to the middleware. |
| `MCP_RETRY_ATTEMPTS` | `3` | Total attempts for transient 5xx/429/timeout/connection failures on safe reads. Read-only POST endpoints opt in explicitly; mutating requests are never retried automatically. |
| `MCP_RETRY_BACKOFF_SECONDS` | `0.5` | Initial exponential-backoff delay between retries. |
| `MIDDLEWARE_CLIENT_TOKEN` | _(empty, required)_ | Bearer token this server presents to the middleware, which auto-accepts it under the reserved label `@mcp`. Must differ from every token in `MCP_API_TOKENS`; the server refuses to start if any collide. |
| `SMART_EMBEDDER_CLIENT_TOKEN` | _(empty)_ | Never used to call anything — this server never talks to Smart Embedder. Read only so startup validation can also refuse a collision between it and `MCP_API_TOKENS`. |
| `MCP_ALLOW_UNAUTHENTICATED_LOCAL` | `false` | Serve without authentication when `MCP_API_TOKENS` is empty in `static` mode; requires a loopback `MCP_BIND_HOST`. |

## Behaviour contracts

These hold for every client, in every mode.

### Confirmation gates

Every destructive tool refuses to make **any** HTTP call at all when `confirm`
is missing, blank, or does not equal the target exactly — the comparison is
case-sensitive and whitespace-sensitive (`ExactString`: not stripped, not
normalised), so a padded or differently-cased value is treated the same as a
wrong one.

| Tool | Confirm required when | Must equal exactly | Dry run if omitted? |
|---|---|---|---|
| `qdrant_create_collection` | `recreate=true` | `collection_name` | No |
| `qdrant_delete_collection` | always | `collection_name` | No |
| `qdrant_delete_payload_index` | always | `collection_name` | No |
| `qdrant_start_reindex` | always | `collection_name` | No |
| `qdrant_ingest_file` | `replace_source=true` | `source_id` | No |
| `qdrant_ingest_chunks` | `replace_source=true` | `source_id` | No |
| `qdrant_delete_points` (`selector="ids"`) | always | `collection_name` | No |
| `qdrant_delete_points` (`selector="filter"`) | to execute the delete | `collection_name` | Yes — a free `POST /points/count` dry run runs instead when `confirm` is omitted |
| `qdrant_clear_payload` | always, either selector | `collection_name` | No |
| `qdrant_set_payload` | `mode="replace"` | `collection_name` | No |

`mode="replace"` discards every unprotected field on the addressed points — the
same destruction `qdrant_clear_payload` performs — so it is gated the same way:
`collection_name` must be given explicitly (not defaulted) and `confirm` must
equal it exactly.

**No gate at all:** `qdrant_update_schema_toggles`, `qdrant_set_payload` with
`mode="merge"` (the default), and `qdrant_delete_payload_keys` (either
selector). They execute immediately once their inputs validate. Deleting
payload keys does destroy the stored values and is therefore advertised to MCP
clients with `destructiveHint=true`, even though this operation deliberately
has no `confirm` parameter; the middleware still refuses any attempt to remove
the protected `text` or `source_id` fields.

### MCP-side refusals

Some failure messages come from this server itself, before any request reaches
the middleware — not from `describe_error`'s HTTP-status mapping. Blank ids, an
empty filter, too many ids for one fan-out and an unverifiable dry-run count
are refused this way; two shape how a tool is used:

1. **A missing or mismatched confirm.** Every row in the table above raises
   `Error: this operation destroys data: set confirm to the exact <noun>
   '<target>' to proceed` — zero HTTP calls are made.
2. **`qdrant_set_payload(selector="filter", mode="replace")`.** The middleware
   has a filtered merge (`PATCH /points/payload`) but no filtered replace
   endpoint — deciding which fields to keep per point cannot be done in one
   write across many differing payloads. This combination raises `Error:
   replace is only supported for explicit ids: the middleware has no
   replace-by-filter endpoint...` with zero HTTP calls; use `mode="merge"`, or
   resolve the filter to explicit ids with `qdrant_scroll_points` first.

All of them are documented, with every other error string this server
produces, in the `docs://troubleshooting` resource.

### One log line per tool call

Every tool call ends with one INFO line in `docker logs smart-qdrant-mcp`,
whatever the outcome:

```
INFO:     tool 'qdrant_search' ok 0.4s
```

The outcome is `ok` only when the result does not carry `is_error`: a failure
the tool reports (`is_error=true`) and an error the SDK raises — argument
validation, an unknown tool name — both log `error`. The duration has one
decimal. The name is written with `repr`, so a client-chosen name stays on one
line. **The arguments are never logged:** they carry query text, chunk text,
payloads, filters and confirm values. Resource reads and prompts are not
logged.

The line names no caller. Authentication logs its own line for each HTTP
request — `token verified: <label>` in `static` mode, `OAuth token verified:
<client_id>` in `oauth` mode, nothing when unauthenticated — and that is the
only per-person attribution of an MCP call: the middleware's
[operation line](../middleware-qdrant-manager/README.md#behaviour-contracts)
shows every call this server makes as `caller=@mcp`.

Uvicorn's access line for every HTTP request stays — client address, method,
path and status — but never its query string: `configure_package_logging()`
filters `uvicorn.access` so nothing after the `?` reaches a handler.

```
INFO:     127.0.0.1:58909 - "POST /mcp HTTP/1.1" 200 OK
```

The line is written by the process serving the container. A copy of the
package run any other way — for example a local stdio launch wired into a
desktop client — logs to that process's own stderr, and its calls never
appear in `docker logs`.

## Capabilities and limits

**You can:**

- Drive the full middleware surface — collections, payload indexes, reindex,
  ingest, export, point reads and writes, payload writes, hybrid search and
  health — from any MCP client over streamable HTTP.
- Preview a filter delete for free by omitting `confirm`.
- Read live collection, health and metrics snapshots as resources rather than
  tool calls.
- Choose markdown or JSON per call with `response_format`, and read the
  normalized payload from `structuredContent` either way.
- Authenticate with static bearer tokens or delegate to an RFC 7662
  introspection endpoint.

**You cannot:**

- Write a file to the middleware host. `qdrant_ingest_file` reads a file that
  must already exist under `middleware-qdrant-manager/data/`; the mount is
  read-only and staging is deliberately an operator step outside MCP.
  `qdrant_export_collection` is the one tool that causes a file to be written,
  and it chooses the path itself.
- Send more than 4 MiB in one call (transport limit — see below).
- Replace payload by filter (no such middleware endpoint).
- Reach Qdrant or Smart Embedder directly through any tool.
- Run replicas behind a load balancer without changing the session model.
- Rate-limit at this layer. This server does not rate-limit inbound requests at
  all; the middleware it calls does
  (`MIDDLEWARE_RATE_LIMIT_REQUESTS_PER_MINUTE` /
  `MIDDLEWARE_RATE_LIMIT_BURST_SIZE`), and its 429 surfaces here as
  `Error: rate limited by the middleware...`.

### Operational limits

Three defaults surprise callers, so they are stated here rather than discovered
from a failure.

**Request bodies are capped at 4 MiB.** That is the streamable-HTTP transport's
own `max_request_body_size` default, applied before any tool runs, so a large
inline `qdrant_ingest_chunks` call is rejected at the transport with no
tool-level message to explain it. Send fewer chunks per call, or write the file
under the middleware's `data/` directory and use `qdrant_ingest_file`, which
carries a path instead of the content.

**`MCP_HTTP_TIMEOUT` is one global timeout** (120 seconds by default) for every
outbound call to the middleware, ingest included. A long ingest can therefore
abort on this side while the middleware finishes it server-side: the caller
sees a transient-failure message and the points land anyway. Point ids are
deterministic, so the safe recovery is to re-send the same chunks — the write
is idempotent — or to check with `qdrant_count_points` first.

**Sessions are stateful and there is no event store.** That is correct for the
single container `compose.yml` runs: one process owns its sessions and
`restart: unless-stopped` recreates them on restart. Scaling to replicas behind
a load balancer would need `stateless_http=True` on `streamable_http_app()`,
since a session created on one replica means nothing to another.

## Recipes

### Search

**Plain search:**
```json
qdrant_search {
  "query": "what is retrieval augmented generation?",
  "collection_name": "docs",
  "limit": 5
}
```
Embeds the query, runs dense+sparse retrieval, fuses with RRF, reranks the top
candidates (default `rerank=true`), and returns up to 5 ranked hits.

**Filtered search:**
```json
qdrant_search {
  "query": "how do I rotate the token",
  "collection_name": "docs",
  "filters": {"must": [{"key": "lang", "match": {"value": "en"}}]},
  "limit": 5
}
```
Restricts results to points whose `lang` payload field matches `"en"`.
Every collection runs Qdrant strict mode, so `lang` must carry a `keyword`
index first or the call is refused. See
`docs://reference/filters` and `qdrant_list_payload_indexes`.

**High-recall search:**
```json
qdrant_search {
  "query": "rare edge case handling",
  "collection_name": "docs",
  "candidate_limit": 200,
  "rerank": true,
  "rerank_top_k": 100,
  "limit": 10
}
```
Raises how many candidates each retriever fetches before fusion
(`candidate_limit`) and how many fused hits reach the reranker
(`rerank_top_k`), trading latency for a better chance of surfacing a hit that
ranks low in either retriever alone.

### Ingest

**From a file already on the middleware host:**
```json
qdrant_ingest_file {
  "file_path": "corpus/manual.json",
  "collection_name": "docs"
}
```
Reads `middleware-qdrant-manager/data/corpus/manual.json`, embeds every chunk,
and upserts it. `source_id` defaults to the file name without its extension
(`manual`).

**Inline chunks, no file:**
```json
qdrant_ingest_chunks {
  "source_id": "manual-v1",
  "collection_name": "docs",
  "chunks": [
    {"id": 1, "text": "Qdrant stores vectors."},
    {"id": 2, "text": "The middleware embeds text before writing."}
  ]
}
```
Point ids are derived from `source_id` + each chunk's integer `id`, so
re-sending the same chunks overwrites the same points. Every chunk key is
stored as payload. `id` is the chunker's ordinal, not the point id, and lands
under `chunk_id_key` (default `"id"`) — the middleware relocates it there, so
`chunk_id_key: "chunk_ordinal"` leaves no `id` in the payload. The
chunker-internal `overflow_details` / `overlap_sentences_count` /
`split_metadata` used to be stripped and currently are not.

**Staging a large file first:**

`qdrant_ingest_file` reads a file that must already exist under
`middleware-qdrant-manager/data/`, and no tool on this server can put it there
— the mount is read-only and staging is an operator step, deliberately left
outside MCP. Copy the file first, then ingest:

```powershell
$dst = "middleware-qdrant-manager\data\corpus"
New-Item -ItemType Directory -Path $dst -Force | Out-Null
Copy-Item "<source>\promessi_sposi_chunks.json" -Destination "$dst\promessi_sposi_chunks.json" -Force
```

```bash
mkdir -p middleware-qdrant-manager/data/corpus
cp "<source>/promessi_sposi_chunks.json" middleware-qdrant-manager/data/corpus/
```

```json
qdrant_ingest_file {
  "file_path": "corpus/promessi_sposi_chunks.json",
  "collection_name": "docs",
  "source_id": "promessi-sposi",
  "batch_size": 8
}
```

The file must be a JSON object with a top-level `chunks` array — a bare array
is refused. [`docs/ingest-patterns.md`](docs/ingest-patterns.md) and the
`docs://workflows/ingest` resource both explain why no MCP tool performs the
copy step.

**Re-ingest, overwriting a shrunk source:**
```json
qdrant_ingest_chunks {
  "source_id": "manual-v1",
  "collection_name": "docs",
  "chunks": [
    {"id": 1, "text": "Updated content, chunk 2 removed."}
  ],
  "replace_source": true,
  "confirm": "manual-v1"
}
```
`confirm` must equal `source_id` exactly. On confirmation, every existing point
for `source_id="manual-v1"` not rewritten by this call is deleted after the new
points are durably written — so chunk `2` from the earlier example is pruned
rather than left as an orphan.

### Collection lifecycle

**Create:**
```json
qdrant_create_collection {
  "collection_name": "docs",
  "dense_on_disk": false,
  "quantization": true,
  "sparse_idf": true,
  "sparse_on_disk": false
}
```
All four toggles are required — there is no server-side default to fall back to
for them.

**Verify:**
```json
qdrant_verify_collection {
  "collection_name": "docs",
  "quantization": true
}
```
Checks `docs` against `quantization=true` and, for every toggle omitted, the
middleware's own configured expectation. Never mutates the collection.

**Toggle a mutable schema setting:**
```json
qdrant_update_schema_toggles {
  "collection_name": "docs",
  "dense_on_disk": true
}
```
Moves dense vectors on-disk in place; no points are touched, no confirm
required — this operation is reversible.

### Destructive operations and their gates

**Delete-by-filter dry run (no `confirm`):**
```json
qdrant_delete_points {
  "selector": "filter",
  "collection_name": "docs",
  "filters": {"must": [{"key": "source_id", "match": {"value": "manual-v1"}}]}
}
```
Because `confirm` is omitted, nothing is deleted: the tool calls
`POST /points/count` with the same filter and returns
`"Dry run: this filter matches N point(s) in 'docs'. Nothing was deleted...."`

**Delete-by-filter, confirmed:**
```json
qdrant_delete_points {
  "selector": "filter",
  "collection_name": "docs",
  "filters": {"must": [{"key": "source_id", "match": {"value": "manual-v1"}}]},
  "confirm": "docs"
}
```
`confirm` equals `collection_name` exactly, so the filter delete actually runs
this time.

**Clear payload down to protected fields:**
```json
qdrant_clear_payload {
  "selector": "ids",
  "ids": ["1", "2"],
  "collection_name": "docs",
  "confirm": "docs"
}
```
Removes every payload key except `text` and `source_id` from points `1` and
`2`. `confirm` is required unconditionally here — unlike
`qdrant_delete_points`, there is no dry run, because a clear has nothing
proportional to preview.

**Replace a point's payload down to only its protected fields:**
```json
qdrant_set_payload {
  "selector": "ids",
  "ids": ["1"],
  "mode": "replace",
  "payload": {"lang": "en"},
  "collection_name": "docs",
  "confirm": "docs"
}
```
`mode="replace"` discards every existing field on point `1` except `text` and
`source_id`, then writes `{"lang": "en"}` — gated exactly like
`qdrant_clear_payload`. `mode="merge"` (the default) needs neither.

## Verification

```bash
python3 -m pytest smart-qdrant-mcp-server/tests -q
```

Fakes only (`httpx.MockTransport`), no live middleware, no model downloads, no
network access. Fixed test counts are intentionally omitted because the suite
continues to grow. Also run from the repository root as part of the standard
check:

```bash
python3 -m ruff check smart-qdrant-mcp-server
python3 -m ruff format --check smart-qdrant-mcp-server
docker compose --env-file .env.example --profile cpu config --quiet
```

### Evaluations

A fixed, 72-chunk fixture corpus (`evals/corpus/mcp_eval_corpus.json`) and a
seed script (`evals/seed_evals.py`) build a deterministic `mcp_eval` collection
**through this server's own tools** — the same path a real client takes, not a
shortcut into the middleware. Ten verified questions
(`evals/smart_qdrant_mcp_eval.xml`) exercise a spread of tool paths: faceting,
range-filtered scrolling, cross-referencing an incident and its runbook, sparse
retrieval for a rare identifier, `is_empty` filtering, point reads by id,
payload-index introspection, per-author faceting, paging past a first page, and
a compound-filter boolean check. Every answer was computed directly from the
corpus JSON (or, for the payload-index count, from the seed script's fixed
index list) rather than by trusting any single tool run — see
`tests/test_eval_file.py` for the shape checks (exactly ten pairs, non-empty
single-value answers, no question names a tool directly).

The harness itself (`evals/harness/`) is vendored from the `mcp-builder` skill
with one runtime patch for `mcp==2.0.0`'s renamed streamable-HTTP client API —
see [`evals/harness/README.md`](evals/harness/README.md).

The seed script targets `mcp_eval`, outside the console's `e2e_` campaign
namespace. `--recreate` drops/recreates that exact collection; even the ordinary
seed uses source replacement for its fixture source. Review ownership and the
script before running it. It accepts the MCP service token, not a middleware
token, because its transport connects to the MCP endpoint.

To seed and run (use a securely supplied token rather than recording one in
shell history or shared logs):

```bash
python3 smart-qdrant-mcp-server/evals/seed_evals.py --token <MCP token> --recreate
python3 -m pip install -r smart-qdrant-mcp-server/evals/harness/requirements.txt
export ANTHROPIC_API_KEY=...   # PowerShell: $env:ANTHROPIC_API_KEY = '...'
python3 smart-qdrant-mcp-server/evals/harness/evaluation.py -t http \
  -u http://127.0.0.1:6337/mcp -H "Authorization: Bearer <MCP token>" \
  -o smart-qdrant-mcp-server/evals/report.md smart-qdrant-mcp-server/evals/smart_qdrant_mcp_eval.xml
```

**No harness run has been recorded yet.** Running the harness needs the full
stack up with models loaded and an `ANTHROPIC_API_KEY`; the ten answers were
verified by computing each one directly from the fixture corpus. The first time
someone runs the command above against a seeded stack, record the resulting
accuracy, the date, and the model used in this section. Two runs that did not
use the harness are on record: on 2026-10-30 ten Claude Code subagents (Sonnet,
`advanced` profile), one per question and blind to the answers, scored 10/10.
Note that the harness answers only the first `tool_use` block of each model
turn, so a model that issues parallel tool calls fails it with an API 400.

**Retrieval quality** is measured separately on the same seeded collection.
`evals/retrieval_golden.json` holds 32 queries in four categories of eight —
`paraphrase`, `identifier` (a literal config key or metric name),
`cross_vocabulary` (an incident's terms for its runbook, and the reverse) and
`cross_lingual` (Italian queries over the English corpus) — each labelled with
its relevant chunk ids. `evals/retrieval_eval.py` searches each query with and
without the reranker and reports hit@1, recall@5, recall@10, MRR and nDCG@10,
overall and per category. It exits non-zero when an overall metric falls more
than 0.05 below `evals/retrieval_baseline.json` for the active embedder
profile; `--write-baseline` records the current scores for that profile. It
needs the seeded stack and the MCP token, like the seed:

```bash
python3 smart-qdrant-mcp-server/evals/retrieval_eval.py --token <MCP token>
```

`tests/test_retrieval_golden.py` checks the golden set offline: every label
names a corpus chunk, every identifier query appears verbatim in a relevant
chunk, and the metric arithmetic matches a hand-computed ranking.

## Failure modes

| Symptom | Cause | Next step |
|---|---|---|
| The `qdrant_*` tools are missing from the client entirely | The server is not running, not reachable, or the client config is wrong. | Check `MCP_SERVER_ENABLED`, the URL and the token; playbook 0 in [`../smart-skill/`](../smart-skill/README.md) is the checklist. |
| `Error: this operation destroys data: set confirm...` | A confirmation gate. Zero HTTP calls were made. | Supply `confirm` equal to the named target, exactly. |
| `Error: replace is only supported for explicit ids...` | `qdrant_set_payload(selector="filter", mode="replace")`. | Use `mode="merge"`, or resolve the filter to ids with `qdrant_scroll_points`. |
| `Error: rate limited by the middleware...` | The middleware's per-client bucket, not this server's. | Slow down, or raise `MIDDLEWARE_RATE_LIMIT_*`. |
| A large inline ingest fails with no tool message | The 4 MiB transport cap. | Fewer chunks per call, or stage a file and use `qdrant_ingest_file`. |
| A long ingest reports a transient failure but the points exist | `MCP_HTTP_TIMEOUT` fired on this side while the middleware finished. | Re-send (writes are idempotent) or check with `qdrant_count_points`. |
| The server refuses to start | A token collision, insecure non-loopback OAuth URL, or `MCP_BIND_HOST` outside loopback/RFC1918 IPv4. | See [Invariants](#invariants). |
| `413` from `qdrant_export_collection` | The point or encoded-byte export cap was exceeded. | Ask the operator to review the named cap and memory capacity; no file was published. |
| Export capacity occupied / inference capacity occupied | Concurrent work holds the configured admission budget. | Retry after active work drains; reduce concurrent calls for inference. |
| Inference admission budget exceeded | One request exceeds the estimated token or output budget. | Reduce batch size, input length or requested vector output. |

Every error string this server can produce is keyed by exact text in the
`docs://troubleshooting` resource.

## Invariants

- **Token model.** In the default `static` mode, `MCP_API_TOKENS` is a
  comma-separated list of `token:label` pairs. Each token authenticates
  independently; the label becomes the access token's `client_id`, which is
  what gets logged — per-person attribution, and one person's token can be
  revoked without rotating anyone else's. At least one token is required in
  this mode unless `MCP_ALLOW_UNAUTHENTICATED_LOCAL=true` — which the shipped
  `.env.example` sets, and which additionally requires a loopback
  `MCP_BIND_HOST`; with neither, the server refuses to start. In `oauth` mode,
  opaque tokens are checked through RFC 7662 — active status, audience, expiry,
  required scopes, principal identity — failures are closed and tokens are
  never logged. Issuer, resource-server and introspection URLs require HTTPS;
  HTTP is allowed only for an explicit loopback host during local development.
- **Distinct tokens.** In `static` mode, `MCP_API_TOKENS`,
  `MIDDLEWARE_CLIENT_TOKEN` and `SMART_EMBEDDER_CLIENT_TOKEN` must all be
  pairwise distinct. A collision fails startup — a hard failure, not a warning.
- **Bind rules.** `MCP_BIND_HOST` is the published/policed address: it defaults
  to `127.0.0.1`, accepts loopback (including `::1`) or an IPv4 RFC1918 literal, seeds the
  Host allowlist and the issuer/resource URL, and is what
  `scripts/verify_compose_security.py` checks against the resolved port publish
  (Compose long-port syntax with `host_ip`, `published` and `target`) — it
  never reaches a socket. `MCP_LISTEN_HOST` is the actual bind address passed
  to `uvicorn.run(host=...)`, defaulting to `127.0.0.1` so running outside
  Docker is safe by default; the Compose service sets it explicitly to
  `0.0.0.0`, because the container listens on all interfaces inside its own
  private network namespace and the published port is what actually restricts
  reachability. Both runtime validation and the verifier reject public,
  hostname, link-local, multicast and wildcard publish addresses.
- **Host and Origin allowlists.** The SDK's `TransportSecuritySettings`
  enforces both. `Host` always accepts loopback and `localhost`, plus anything
  in `MCP_ALLOWED_HOSTS`. `Origin` accepts only what is listed in
  `MCP_ALLOWED_ORIGINS`; empty (the default) means same-origin requests only.
- **This server must not be exposed to the public internet.** It has no
  public deployment posture. In `static` mode it accepts a small set of bearer
  tokens and publishes inert local issuer metadata; in `oauth` mode it delegates
  token validation to the configured external authorization server through RFC
  7662 introspection. Neither mode changes the supported network boundary:
  reach it only over loopback, a trusted private network, a personal VPN, or an
  authenticated tunnel — the same posture `compose.yml` and the root
  `README.md` already require for the middleware and Smart Embedder.
- **`tool_map.py` stays true.** A test compares it against the middleware's
  actually registered routes on every run.
- **Destructive tools make zero HTTP calls when the gate fails.** Do not
  "helpfully" pre-fetch anything before the confirm check.

## Extending it

- **A new tool** — add the module under `tools/` with its input model and any
  gate, register its endpoint(s) in `tool_map.py`, add a row to the matching
  group table above, and update
  [`../smart-skill/smart-qdrant-mcp/references/reference.md`](../smart-skill/smart-qdrant-mcp/references/reference.md)
  in the same change. The route test fails if the endpoint is not real.
- **A new destructive operation** — decide the gate deliberately: is the change
  reversible in place? If not, it needs a `confirm` row in the table above and
  an entry in `docs://troubleshooting`.
- **A new resource or prompt** — `resources/docs.py`, `resources/live.py`, or
  `prompts.py`. `tests/test_prompts.py` verifies every tool and URI a prompt
  names, so a typo fails the suite rather than the agent.
- **A new configuration variable** — `config.py`, the table above, and the root
  `.env.example`. There is no runtime reload; document that it takes a restart.
- **Scaling out** — see the session note under
  [Operational limits](#operational-limits); replicas need `stateless_http=True`
  and a rethink of the session model.

## Related documents

| Document | Covers |
|---|---|
| [`../middleware-qdrant-manager/README.md`](../middleware-qdrant-manager/README.md) | The REST API every tool calls, and the behaviour behind every response. |
| [`../smart-skill/README.md`](../smart-skill/README.md) | The client-side agent skill that teaches an agent to use these tools well. |
| [`evals/harness/README.md`](evals/harness/README.md) | The vendored evaluation client and its one local patch. |
| [`../scripts/README.md`](../scripts/README.md) | Starting, stopping and inspecting the stack. |
| [`../README.md`](../README.md) | The stack as a whole. |
| [`../AGENTS.md`](../AGENTS.md) | Repository-wide conventions for agents working on the code. |
