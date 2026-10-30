# evals/harness — vendored MCP evaluation client

> **Read this when** you need to run an evaluation against the running MCP
> server, or you are about to edit a file in this directory and need to know
> why editing it is discouraged.
> **Audience** — an AI agent working in this repository, and any human reading
> over its shoulder.
> **Source of truth** — this file for provenance and the local patch; the
> upstream `mcp-builder` skill for the harness's own behaviour;
> [`../../README.md`](../../README.md) for the evaluation set these scripts run.
> **Prerequisite** — [`../../README.md`](../../README.md) § Verification, which
> owns the seed-and-run procedure this file supports.

---

## Purpose and boundaries

The two Python files here are **copied, not written**. They come from the
`mcp-builder` skill's `scripts/` directory
(`.superpowers/sdd/2026-07-30-mcp-qdrant-server/mcp-builder/scripts/` in
this checkout).

**What this directory is:**

- A standalone MCP client used to *drive* a running `smart-qdrant-mcp` server
  through a fixed question set and score the answers.

**What this directory is not:**

- Not a runtime dependency. The server neither imports nor ships these files.
- Not local code to evolve. It is a vendored copy carrying one deliberate
  patch; upstream changes are re-copied, not hand-merged.

## Mental model

1. **Treat this directory as third-party.** Do not edit it casually. If the
   skill ships an update, re-copy from there rather than patching drift into
   this copy.
2. **One patch exists, and it is load-bearing.** `connections.py` was changed
   to match `mcp==2.0.0`'s API. Losing that patch makes the harness fail at
   import, before any connection is attempted.
3. **Running an evaluation is not free.** It needs the full stack up with
   models loaded, a seeded `mcp_eval` collection, and an `ANTHROPIC_API_KEY`.

## Map of the directory

| File | Source | Modified? |
|---|---|---|
| `evaluation.py` | `mcp-builder/scripts/evaluation.py` | No — byte-identical copy. |
| `connections.py` | `mcp-builder/scripts/connections.py` | **Yes** — see [The applied patch](#the-applied-patch). |
| `requirements.txt` | `mcp-builder/scripts/requirements.txt` | **Yes** — one version floor bumped; see below. |
| `LICENSE.txt` | `mcp-builder/LICENSE.txt` | No — byte-identical copy (Apache 2.0; 11345 bytes as stored with LF endings, 11546 in a CRLF working copy). |

## The applied patch

`connections.py`'s HTTP transport used a v1-era client API that no longer
exists in `mcp` 2.x. It was written against 2.0.0; the repository now pins 2.2.0,
whose transport signatures are unchanged. Two
things differ between the vendored skill's assumed API and the real one:

1. **The function was renamed.**
   `mcp.client.streamable_http.streamablehttp_client` (no underscore between
   "streamable" and "http") does not exist in `mcp==2.0.0`; it is now
   `streamable_http_client` (with the underscore). The original module-level
   import raises `ImportError` immediately, before any connection is attempted
   — the same rename the seeding script (`evals/seed_evals.py`) already works
   around.
2. **The function lost its `headers=` parameter.** The real signature is
   `streamable_http_client(url: str, *, http_client: httpx2.AsyncClient | None
   = None, terminate_on_close: bool = True)`. Per its own docstring: *"To
   configure headers, authentication, or other HTTP settings, create an
   `httpx2.AsyncClient` and pass it here."* When a caller supplies
   `http_client=`, the function does **not** manage that client's lifecycle —
   it only enters/exits a client it created itself — so the caller must open
   and close it.

**The fix.** `MCPConnectionHTTP._create_context` now opens an
`httpx2.AsyncClient(headers=self.headers)` as its own `async with` block and
passes it to `streamable_http_client(url, http_client=http_client)`, composed
as a single `@asynccontextmanager` so the existing `AsyncExitStack`-based
`MCPConnection.__aenter__`/`__aexit__` needs no change at all. `import httpx2`
and `from contextlib import asynccontextmanager` were added for this.

**Scope of the patch.** Only `MCPConnectionHTTP` was touched.
`MCPConnectionStdio` (`stdio_client`) and `MCPConnectionSSE`
(`sse_client(url, headers=...)`) were verified against the installed
`mcp==2.0.0` and still match the vendored code exactly; neither was changed.
`evaluation.py` needed no change — it only imports `create_connection` and
never touches the transport layer.

**`requirements.txt`.** The `mcp>=1.1.0` floor was bumped to `mcp>=2.0.0`,
since the patched code assumes the 2.x API shape. (The pre-patch code would
have been fine on 1.x; this project ships `mcp` 2.x, so the floor matches the API the patch
assumes.)

## Capabilities and limits

**You can:**

- Run the fixed evaluation set against a live server over `stdio`, `sse`, or
  `http`.
- Re-copy any file from the upstream skill and re-apply the documented patch.

**You cannot:**

- Run this without network access to an Anthropic API endpoint — the harness
  grades answers with a model.
- Run it against a stack whose `mcp_eval` collection has not been seeded.
- Assume the patch survives a naive re-copy. Re-applying it is a manual step.

## Recipes

**Run an evaluation**

```bash
python3 -m pip install -r smart-qdrant-mcp-server/evals/harness/requirements.txt
export ANTHROPIC_API_KEY=...   # PowerShell: $env:ANTHROPIC_API_KEY = '...'
python3 smart-qdrant-mcp-server/evals/harness/evaluation.py -t http \
  -u http://127.0.0.1:6337/mcp -H "Authorization: Bearer <token>" \
  -o smart-qdrant-mcp-server/evals/report.md smart-qdrant-mcp-server/evals/smart_qdrant_mcp_eval.xml
```

Requires the full stack up (`middleware`, Smart Embedder, Qdrant, and this MCP
server) with the `mcp_eval` collection seeded via `evals/seed_evals.py`.

**Re-copy from upstream**

1. Copy the file from the skill's `scripts/` directory.
2. Re-apply the `MCPConnectionHTTP._create_context` patch above if the upstream
   file still targets the v1 API.
3. Re-run the import check below.

## Verification

After installing `requirements.txt` as shown above, the construction check
below needs no live server — it only proves the module imports and the
transport constructs without raising. Run it from the harness directory so its
sibling `connections.py` is importable:

```
$ cd smart-qdrant-mcp-server/evals/harness
$ python3 -c "import connections; c = connections.create_connection('http', url='http://127.0.0.1:6337/mcp', headers={'Authorization': 'Bearer x'}); print(type(c._create_context()))"
<class 'contextlib._AsyncGeneratorContextManager'>
```

```
$ python3 evaluation.py --help
usage: evaluation.py [-h] [-t {stdio,sse,http}] [-m MODEL] [-c COMMAND] ...
```

**Not verified here:** an actual network round-trip through `-t http` against a
live server. That needs the stack up with models loaded and an
`ANTHROPIC_API_KEY`.

## Invariants

- **Do not hand-patch drift.** Every intentional divergence from upstream is
  documented in [The applied patch](#the-applied-patch). If you add one, add it
  there too.
- **Keep the `mcp>=2.0.0` floor.** The patched code cannot run below it.
- **Leave `evaluation.py` byte-identical.** It has no reason to diverge; if it
  needs to, that is a signal to fix the harness upstream instead.

## Related documents

| Document | Covers |
|---|---|
| [`../../README.md`](../../README.md) | The MCP server, its tools, and the evaluation set (`evals/`) these scripts execute. |
| [`../../../README.md`](../../../README.md) | Running the stack the harness needs. |
