# AGENTS.md — repository guide for coding agents

This is the canonical instruction file for any AI coding agent working in this
repository. It covers what spans the whole repo; each service's own `README.md`
covers that service in depth.

**Precedence:** a direct instruction from the user beats this file; this file
beats an agent's defaults.

---

## Working agreement

- Start by checking `git status --short` and preserve changes that were already
  present. Do not reset, checkout, stash, commit, or push unrelated work.
- Work from the repository root. Keep service boundaries intact: communication
  between services goes through explicit HTTP clients and API contracts, never
  shared runtime imports.
- Read the relevant service README before changing an API, setting, Compose
  shape, model/runtime behaviour, or destructive test. Use `rg` to inspect the
  smallest relevant section; do not load the whole Smart Embedder module.
- Never place real tokens, `.env` values, model caches, generated exports, or
  local agent notes in a patch. Do not print credentials while diagnosing a
  problem.
- A change is not complete until the focused tests and checks appropriate to
  its risk have run, or the final hand-off says exactly what could not run.

## Repository shape

Three **independently-imaged services**, unified by one root Docker Compose
file, one `.env` contract and one dependency tree — not a shared Python
package.

```
Client / MCP client → smart-qdrant-mcp (MCP, 6337) ┐
Client ────────────────────────────────────────────┴→ middleware-qdrant-manager (FastAPI, 6336) → server-embedder-reranker (FastAPI, 8010)  [embed/rerank]
                                                                                                → external Qdrant                            [store/search]
```

| Path | What it is |
|---|---|
| `middleware-qdrant-manager/` | The API clients talk to. Owns collections, ingest, hybrid search, point CRUD, vector reindex, auth, rate limiting. Code in `qdrant_middleware/`, tests in `tests/`, static console in `frontend/`. |
| `server-embedder-reranker/` | "Smart Embedder": self-hosted embedding + reranking server with two fixed model profiles (`basic` = BGE-M3 family, `advanced` = Qwen/MILCO). No cloud dependency. |
| `smart-qdrant-mcp-server/` | **Optional** MCP server exposing the middleware's REST API as 28 tools plus documentation/live-state resources and workflow prompts. The stack functions without it (`MCP_SERVER_ENABLED=false`; `EMBEDDER_ONLY=true` removes it too). |
| `smart-skill/` | Client-side agent skill for the MCP server. Prompt-side documentation only — no runtime role; the stack runs identically without it. |
| `scripts/` | What the launchers call: the shared planner `stack_core.py`, `verify_token_config.py`, `verify_compose_security.py`, `verify_dependencies.py` (network-dependent OSV/SBOM audit), and their tests. |
| `stack.ps1` / `stack.sh` | **At the repository root.** The only supported way to build/start/stop the stack. |
| `requirements/` | Human-maintained direct pins (`middleware.txt`, `embedder-cpu.txt`, `embedder-gpu.txt`, `embedder-base.txt`, `shared.txt`, `mcp.txt`, `dev.txt`, `ci.txt`) plus `locks/` transitive constraints applied only inside Docker builds. |
| `.github/workflows/ci.yml` | The verification pass below, run on every push to `main` and every pull request. Add a job here when you add a suite. |
| `compose.yml` / `Dockerfile` | Multi-target build (`middleware`, `smart-embedder-cpu`, `smart-embedder-gpu`, plus non-network-facing `*-init-*` prep containers per hardware profile). |
| `compose.embedder-private.yml` | One-purpose overlay the launchers add when `EXPOSE_SMART_EMBEDDER_PORT=false`; withdraws the host port publish with `ports: !reset []` (a plain `ports: []` would concatenate, not replace). |
| `compose.gpu-device.yml` | Single physical GPU overlay; both GPU init and serving reservations use the UUID resolved by the launcher planner. |
| `docs/` | **Root-only, gitignored working material.** Untracked specs and agent notes live here; never `git add -f` them. Versioned service documentation under `smart-qdrant-mcp-server/docs/` and `smart-qdrant-mcp-server/smart_qdrant_mcp/resources/docs/` is different and must be maintained when its behaviour changes. |

## Where to look first

Read the right document instead of the whole repo. Each is the single source of
truth for its subject; none repeats another.

| Question | Document |
|---|---|
| How do I build/start/stop the stack? Which flag does what? | [`scripts/README.md`](scripts/README.md) |
| What can the REST API do? Endpoints, filters, ingest, export, config. | [`middleware-qdrant-manager/README.md`](middleware-qdrant-manager/README.md) |
| Embedding/reranking behaviour, model profiles, autotuner, metrics. | [`server-embedder-reranker/README.md`](server-embedder-reranker/README.md) |
| MCP tools, confirmation gates, resources, prompts. | [`smart-qdrant-mcp-server/README.md`](smart-qdrant-mcp-server/README.md) |
| Educational browser guides: console workflow, architecture and hybrid search. | [`guide/README.md`](guide/README.md) |
| The console E2E suite (destructive — read before running). | [`middleware-qdrant-manager/frontend/tests/README.md`](middleware-qdrant-manager/frontend/tests/README.md) |
| Every environment variable, with its shipped default. | `.env.example` (versioned, exhaustive) |

Each service README follows one fixed skeleton — `Purpose and boundaries`,
`Mental model`, `Map of the code`, `Interface reference`, `Configuration`,
`Behaviour contracts`, `Capabilities and limits`, `Recipes`, `Verification`,
`Failure modes`, `Invariants`, `Extending it`, `Related documents` — so the
right section is findable without reading the file end to end. `Capabilities
and limits` answers "can it do X?"; `Invariants` answers "may I change this?".

The root `README.md` is written for the human operator and is **owned by the
repository owner** — do not rewrite it unless asked directly.

## Common commands

Run everything from the repository root. Copy `.env.example` to `.env` before
any `up`.

### Running the stack

```powershell
pwsh stack.ps1 verify               # check .env, touch no Docker state
pwsh stack.ps1 up gpu advanced      # hardware profile + model profile
pwsh stack.ps1 up cpu -DryRun       # print the plan, run nothing
pwsh stack.ps1 down
pwsh stack.ps1 help                 # the authoritative command surface
```

```bash
bash stack.sh up cpu --wait
bash stack.sh help
```

[`scripts/README.md`](scripts/README.md) is the launcher reference — every
command, flag, precedence rule, recipe and exit code. Do not reproduce a
command list elsewhere: three documents once disagreed about which flags exist,
which is why that file is now singular and test-enforced.

### Per-service tests

Windows (PowerShell):

```powershell
py -m pytest middleware-qdrant-manager/tests -q
py -m pytest smart-qdrant-mcp-server/tests -q
py -m pytest scripts -q
Push-Location server-embedder-reranker
py -m unittest discover -s tests -p "test_*.py"
Pop-Location
```

Unix-like shells:

```bash
python3 -m pytest middleware-qdrant-manager/tests -q
python3 -m pytest smart-qdrant-mcp-server/tests -q
python3 -m pytest scripts -q
(cd server-embedder-reranker && python3 -m unittest discover -s tests -p "test_*.py")
```

These offline suites use fakes: no live Qdrant, middleware, embedder, model
download, or network is required. Narrow a run with `-k <name>` or a single
file path.

Two suites are **opt-in** and need a live stack:

```powershell
python server-embedder-reranker/tests/test_server.py --url http://127.0.0.1:8010 --token <token>
Push-Location middleware-qdrant-manager/frontend
$env:QHM_E2E_TOKEN = '<token>'; npm test
Pop-Location
```

The frontend's pure model tests do not need the stack:
`node --test middleware-qdrant-manager/frontend/tests/query-models.test.mjs`.

The Playwright suite is **destructive** against `e2e_`-prefixed Qdrant
collections — read its README first.

### Full verification pass

The check to run before opening a change. `.github/workflows/ci.yml` runs the
same set on every push to `main` and every pull request, so this is what CI
will say — running it locally only tells you sooner:

```powershell
py -m pytest middleware-qdrant-manager/tests -q
py -m pytest smart-qdrant-mcp-server/tests -q
py -m pytest scripts -q
Push-Location server-embedder-reranker
py -m unittest discover -s tests -p "test_*.py"
Pop-Location
py -m ruff check .
py -m ruff format --check .
shellcheck stack.sh
Invoke-ScriptAnalyzer -Path stack.ps1 -Severity Warning,Error
docker build --check .
docker compose --env-file .env.example --profile cpu config --quiet
docker compose --env-file .env.example --profile gpu config --quiet
```

The two launcher linters are the only tools here that are not already a
dependency of a service. Install them once: `winget install koalaman.shellcheck`
and `Install-Module PSScriptAnalyzer -Scope CurrentUser`. Both must report
nothing; `Invoke-ScriptAnalyzer` signals findings by **printing** them, not by
an exit code, so read its output rather than `$LASTEXITCODE`.

## Architecture notes

### Middleware (`middleware-qdrant-manager/qdrant_middleware/`)

- **App factory:** `main.create_app()` builds an independent FastAPI app per
  `Settings` instance (tests use it for isolated apps); `main.app` is the
  module-level instance Uvicorn serves.
- **Middleware stack, outermost → innermost:** `timeout+metrics` →
  `shutdown guard` → `rate limit` → `body-size` → [`console headers`] →
  `route`. `FastAPI.add_middleware` inserts at position 0, so the *last*
  registration in `main.py` (`TimeoutMetricsMiddleware`) ends up outermost.
  The console-headers layer is present only when `EXPOSE_FRONTEND=true` and is
  innermost by design: it decorates the response for `/` and `/assets/*` on the
  way out instead of deciding whether to produce one.
  Bearer auth (`require_bearer_auth`) is **not** one of those layers: it is a
  route-level dependency attached via
  `app.include_router(..., dependencies=protected)`, so it runs during route
  resolution — inside every one of them, body-size included.
- `core/config.py` (`Settings`) is the single source of truth for every
  environment variable; `core/lifespan.py` opens/closes the Qdrant client
  across the app lifespan; `core/runtime.py` builds per-app runtime state
  (counters, embedder handle and lock).
- Routers live under `api/routers/`. `collections`, `points` and `query` sit
  behind `require_bearer_auth` + `require_operational_readiness`; `monitoring`
  does not.
- `clients/embedder.py` talks to Smart Embedder; `clients/retry.py` implements
  the shared exponential-backoff retry policy for transient
  5xx/429/timeout/connection errors against both Qdrant and Smart Embedder.
- Raw Qdrant access has explicit owners: the collection repository owns
  schema/metadata/index access; the points router owns point CRUD; query,
  export, health and reindex services own their specialized calls.
  `tests/test_qdrant_access_architecture.py` pins every direct call and records
  the mutation guard, retry semantics and cache effects. Prefer an existing
  gateway; any allowlist expansion requires that review.
- Point IDs are deterministic — `UUID5(source_id:chunk_id)` — so re-ingesting a
  source overwrites rather than duplicates. Ingest/upsert always finishes
  embedding before the first write.
- Collections carry an embedding-contract fingerprint for the active Smart
  Embedder profile. A profile switch must report existing collections as
  incompatible rather than mixing vector spaces silently. The payload fields
  `text` and `source_id` are service-owned and cannot be cleared.
- A background health probe polls Qdrant + Smart Embedder every
  `HEALTH_PROBE_INTERVAL` seconds; `/readyz` reflects the cached state rather than
  probing per request.
- Vector reindex (`services/vector_reindex.py`) is a
  pausable/resumable/cancelable background job with durable batch-boundary
  checkpoints; an active job rejects collection/point mutations that could
  invalidate it.
- File-based ingest is sandboxed to `middleware-qdrant-manager/data/`; paths
  escaping it are rejected with HTTP 400.
- The static console (`frontend/dist/`) is served in-process when
  `EXPOSE_FRONTEND=true`. It is **baked into the image** — rebuild and restart
  the middleware before testing console changes, or you test stale assets.
  `/assets/config.js` intentionally leaks *whether* auth is required, never the
  token itself.

### Smart Embedder (`server-embedder-reranker/`)

- **Flat module layout, not a package — naming is the boundary.** `basic_*`
  belongs only to the BGE profile, `advanced_*` only to the Qwen/MILCO profile;
  unprefixed modules (`smart-embedder.py`, `autotune_entrypoint.py`,
  `benchmark.py`, `model_bootstrap.py`, `model_prepare.py`,
  `model_revisions.py`, `vram_budget.py`) are shared.
- `smart-embedder.py` (5k+ lines) is the whole FastAPI app: profile selection,
  dynamic/length-aware batching, both model wrappers, schemas, Prometheus
  metrics, auth and lifecycle. **Read the relevant section, never the whole
  file.**
- Exactly two startup profiles, chosen once via `SMART_EMBEDDER_PROFILE` at
  process start (no runtime switch): `basic` (BGE-M3 dense/sparse/ColBERT + BGE
  reranker) and `advanced` (Qwen3 dense + MILCO learned-sparse + Qwen3
  reranker, **no ColBERT**).
- A separate pre-start hardware autotuner (`autotune_entrypoint.py` dispatching
  to `basic_autotune*.py` / `advanced_autotune*.py`) fingerprints the CPU/GPU/
  software stack, benchmarks runtime/batch/thread candidates under quality +
  stability + memory gates, and atomically persists the decision per profile.
  It **never edits `.env`**; a fingerprint change invalidates the persisted
  profile and forces recalibration. The same tuner never runs across both
  profiles.
- Model repositories are pinned to **immutable Hugging Face commit revisions**
  in `model_revisions.py`. Updating a model means bumping the pinned revision
  and rerunning the CPU/GPU smoke tests — not chasing a moving `main`.
- BASIC and safe ADVANCED profiles use one GPU lane. A measured ADVANCED
  profile may use more lanes so embedding can overlap reranking, while one
  reranker instance remains serialized. Peak VRAM depends on resident weights,
  activations and workspaces; do not claim a bound without that measurement.

### MCP server (`smart-qdrant-mcp-server/smart_qdrant_mcp/`)

- A thin, stateless translation layer: every tool call becomes one or more HTTP
  requests to the middleware. It never talks to Qdrant or Smart Embedder
  directly, and holds no data of its own. It exposes 28 tools, 9 static
  documentation resources, 3 live-state resources, and 3 workflow prompts.
- `tools/*.py` define each tool's input model, confirmation gate and limits —
  the client's tool list is generated from these. `tool_map.py` maps tool →
  middleware endpoint, and tests compare it against the middleware's actually
  registered routes on every run. Tool changes also require the matching
  client-side reference update under `smart-skill/`.
- Destructive tools make **zero HTTP calls** when their `confirm` value is
  missing or does not match the target exactly (case- and
  whitespace-sensitive).
- It accepts no path, filename or file content from the caller: staging a
  corpus file under `middleware-qdrant-manager/data/` is an operator step
  outside MCP — see `smart-qdrant-mcp-server/docs/ingest-patterns.md`. The one
  write it can cause is `qdrant_export_collection`, which asks the middleware
  to write a whole-collection dump under `middleware-qdrant-manager/exports/`
  at a filename the middleware derives itself.
- A tool rename or re-parameterisation must also update
  `smart-skill/smart-qdrant-mcp/references/reference.md` in the same change.
  The link is documented but not mechanically tested; update the skill's
  playbooks when the recommended workflow changes as well.

### Cross-cutting

- All Docker/Compose config, model volumes and environment variables are
  unified at the repo root — never per-service. `.env` (untracked) holds real
  secrets; `.env.example` (versioned) is the exhaustive template both humans
  and Compose read defaults from.
- Serving containers run as non-root UID/GID 1000 with read-only root
  filesystems. A minimal init container (`CHOWN`/`SETUID`/`SETGID` only)
  prepares named volumes then drops privileges before any model or network
  library loads. Compiled Torch/Triton/ONNX artifacts live on non-persistent
  tmpfs and are rebuilt after container recreation.
- **The three services' effective inbound ACLs must be pairwise disjoint.** A
  service's ACL is its own `*_API_TOKENS` list plus the one `*_CLIENT_TOKEN`
  its caller presents, which the target auto-accepts under a reserved
  `@`-prefixed label. An empty list is a valid unauthenticated configuration
  only with that service's `*_ALLOW_UNAUTHENTICATED_LOCAL=true`.
  `scripts/verify_token_config.py` enforces the cross-service half; each
  service enforces what it can see.
- Three launcher-only switches in `.env` shape which containers run; no service
  reads them, and a bare `docker compose up` ignores all three: `EMBEDDER_ONLY`,
  `MCP_SERVER_ENABLED` (`--scale smart-qdrant-mcp=0`) and
  `EXPOSE_SMART_EMBEDDER_PORT` (adds the overlay above).
  The optional `STACK_GPU_DEVICE` selector is also launcher-owned. Automatic
  startup and selected-GPU paths use a shared host probe; the pure planner
  resolves one UUID and applies `compose.gpu-device.yml` to init and serving.
  `scripts/verify_compose_security.py` validates both published-port shapes
  for CPU, all-GPU and selected-GPU deployments: all six combinations must pass.
- Settings that must not be operator-tunable are **module constants** in
  `core/config.py`, never `Settings` fields — pydantic-settings populates any
  field lacking an explicit alias from the environment under its upper-cased
  name. `LOOPBACK_HOSTS` and `INGEST_DATA_ROOT` are constants for exactly this
  reason: as fields they would let the loopback classification and the
  file-ingest sandbox disable themselves.

## Code style and conventions

Python targets **3.10+**. Lint and format config live in the root
`pyproject.toml` and apply to all three services.

| Rule | Value |
|---|---|
| Formatter | `ruff format` — 88-character lines, spaces, double quotes |
| Lint rule sets | `E`, `W` (pycodestyle), `F` (pyflakes), `D` (pydocstyle, **google** convention) |
| `E501` | Enforced, not ignored. `ruff format` wraps what it can, so a leftover long line is a string or comment needing a human decision. |
| Docstrings | Required. Only the per-callable rules (`D102`, `D103`, `D105`, `D107`) are waived under `**/tests/**` and `scripts/test_*.py` — modules, classes and helpers still need one, and a docstring that *is* written must satisfy every formatting rule. |
| Excluded from lint | `smart-qdrant-mcp-server/evals/harness/` — vendored third-party code, kept faithful to upstream. |
| Naming | `snake_case` modules/functions/variables/test files; `PascalCase` classes; `UPPER_SNAKE_CASE` constants. |
| Tests | Files `test_<behaviour>.py`, functions `test_<expected_behaviour>()`. |

`pyproject.toml` sets `asyncio_mode = "auto"` and
`pythonpath = ["middleware-qdrant-manager", "smart-qdrant-mcp-server"]`, so
imports use `qdrant_middleware.*` and `smart_qdrant_mcp.*` regardless of the
working directory.

Match the surrounding code's comment density, naming and idiom. Keep imports
ordered and type hints consistent with neighbouring code.

### Testing

Add focused regression coverage beside the affected service. Prefer unit tests
that isolate network and model dependencies with fakes or mocks — every offline
suite in this repo already does. Run the relevant suite plus Ruff. Changes to
Compose, ports, volumes or security settings must also pass `py -m pytest
scripts -q` and Compose config validation. Run the Playwright suite from
`middleware-qdrant-manager/frontend/` when changing console behaviour.

### Documentation

- A new README under a subdirectory follows the skeleton listed in
  [Where to look first](#where-to-look-first). Omit a section that does not
  apply; never reorder.
- **`scripts/README.md` is parsed by `scripts/test_stack_readme.py`**, which
  compares its flag tables against the launchers' real `help` output in both
  directions, checks the *Accepted by* column against the option matrix, and
  asserts the exit-code contract and every troubleshooting entry are present.
  Keep those tables' column shape intact.
- Agent-facing content is written in English. Italian rationale documents under
  the untracked `docs/` are deliberate, not a bug.
- Each fact belongs in exactly one document. When two files would say the same
  thing, one of them links instead.
- The one deliberate exception is [Architecture notes](#architecture-notes)
  below: this file is loaded into every session, so it restates the small set
  of load-bearing facts an agent must hold *before* deciding which README to
  open. **Where this file and a service README disagree, the README wins** —
  and the disagreement is a bug to fix in the same change. Do not grow this
  file with detail that only matters once you have already opened the README.

## Change-risk matrix

Use the smallest verification set that proves the change, then expand it when
the change crosses a boundary:

| Change | Minimum verification |
|---|---|
| Python behaviour in one service | That service's focused pytest/unittest tests, then `ruff check` on the touched service. |
| REST endpoint, schema, auth, ingest, query, or retry | Middleware tests plus the MCP suite if `tool_map.py` or a shared contract is affected. |
| MCP tool, resource, prompt, or parameter | MCP tests and the matching `smart-skill/` reference/playbook update. |
| Launcher, `.env.example`, Compose, port, volume, profile, or security | `py -m pytest scripts -q`, both CPU/GPU Compose config checks, and the relevant Docker validation. |
| Frontend source or console behaviour | Rebuild the middleware image, run the focused Node test, and run the destructive Playwright suite only after reading its README. |
| Dependency pin or model revision | Rebuild the affected image, regenerate the matching lock from the verified image, run `py scripts\verify_dependencies.py` (network; audits the lock against OSV), and repeat both CPU and GPU smoke coverage where applicable. |

Do not call the live embedder or Playwright suite an offline test. Live checks need
an explicit running stack, real credentials, and (for Playwright) disposable
`e2e_` collections. Never run them twice without the suite's reset procedure.
`py scripts\verify_dependencies.py` is likewise not part of the offline pass:
it queries the OSV database, so it needs network access.

## Commit and pull request conventions

Git history **is** available; the remote is
`https://github.com/smart-models/Personal-Smart-Embedder`. Follow the existing
convention rather than inventing one — read `git log` when in doubt.

Commits use **Conventional Commits** with an optional scope, and a lowercase,
imperative subject that says what changed and, where it fits, why:

```
docs(middleware): rewrite README as the operator and agent reference
fix(scripts): correct the README's "Accepted by" column, and check it
feat(scripts): add stack_core.py, the launchers' shared decision core
refactor(scripts): move the launchers to the repository root
test(scripts): add --dry-run, the plan matrix, and the recovery invariant
chore(repo): add .gitattributes line-ending contract; fix stale env names
```

Types in use: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`. Scopes in use
include `scripts`, `middleware`, `points`, `query`, `reindex`, `guard`,
`client`, `repo`. A bare type with no scope is fine for repo-wide changes.

Keep commits scoped to one change. Pull requests should explain the behaviour
change, list the validation commands actually run, link the relevant issue, and
include console screenshots or API examples when applicable.

**Commit or push only when asked.** If asked while on `main`, branch first.

## Security and configuration

These are invariants, not preferences. Breaking one is a bug even when tests
pass.

- **Loopback binds are not negotiable.** The middleware and Smart Embedder
  publishes are hardcoded to `127.0.0.1` in `compose.yml`. The MCP publish is
  the one operator-settable address (`${MCP_BIND_HOST:-127.0.0.1}`): it
  defaults to loopback and accepts only loopback or an IPv4 RFC1918 literal;
  public, hostname, link-local, multicast and wildcard values fail startup.
  `MCP_ALLOW_UNAUTHENTICATED_LOCAL=true` additionally requires a loopback
  `MCP_BIND_HOST`. OAuth URLs require HTTPS except on explicit loopback hosts.
- **Keep the three ACLs pairwise disjoint** — see [Cross-cutting](#cross-cutting).
  `scripts/verify_token_config.py` refuses to start the stack on an overlap.
- **Introduce configuration in `.env.example`, never in a local `.env`.** Never
  commit `.env`, tokens, model caches, or generated runtime artifacts.
- **Model revisions and dependency locks are reproducibility controls.**
  Changing a direct dependency pin requires regenerating the matching
  `requirements/locks/*.txt` from the rebuilt, verified image
  (`python -m pip freeze --all`) and committing both files together. Updating a
  model means bumping its pinned revision in `model_revisions.py` and repeating
  the CPU **and** GPU smoke tests.
- **`EXPOSE_SMART_EMBEDDER_PORT=false` is the correct deployment posture.** The
  published embedder port is the only route that reaches it without the
  middleware's bearer auth, rate limiting and rerank truncation.
- **Never quote a token in output, a log line, or an error message.** The
  launchers' `config` command redacts credentials by default and `urls` never
  prints one; keep it that way.

## Traps

Things that have already cost time here.

| Trap | Reality |
|---|---|
| Editing a launcher flag in one shell only | `stack.ps1` and `stack.sh` are held at parity by `scripts/test_stack_parity.py`, and both must match `scripts/README.md`. |
| Putting launcher logic in the shell scripts | Everything deciding *what Compose is asked to do* lives in `scripts/stack_core.py`, once, for both. |
| "Simplifying" `down`, `logs`, `status`, `version`, `help` | They deliberately bypass the planner so stopping a stack works with `.env` absent or malformed and no Python on `PATH`. `scripts/test_stack_recovery.py` enforces it. |
| Running two stacks with `--project-name` | Container names are daemon-global; a second concurrent stack fails with "name already in use". One stack per machine. |
| Testing the console without rebuilding | `frontend/dist/` is baked into the middleware image. |
| Running the Playwright suite twice without a reset | Destructive series consume their fixtures; the second run fails on create/recreate and delete. |
| Answering a Qdrant behaviour question from memory | Fetch the current Qdrant documentation first. |
| Adding a `Settings` field for something that must stay fixed | pydantic-settings will populate it from the environment. Use a module constant. |

## Related files

| File | Role |
|---|---|
| `README.md` | The human operator's entry point. Owned by the repository owner. |
| `.env.example` | The exhaustive, versioned configuration template. |
| `scripts/README.md` | The launcher reference. |
| `LICENSE` | GNU Affero General Public License v3.0, verbatim. Never edit it, and never add a per-file licence header without being asked — the repository states its terms in this one file. |
