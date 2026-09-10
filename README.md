# Smart Semantic Searcher (3S)

Unified Docker deployment for the Qdrant middleware, Smart Embedder, and an
optional MCP server. The services remain separate processes and images; one
root Compose file, one environment contract, and one dependency tree manage
them.

```
Client / MCP client → smart-qdrant-mcp (MCP, 6337) ┐
Client ──────────────────────────────────────┴→ middleware-qdrant-manager (6336) → server-embedder-reranker (8010)
                                                                                 → external Qdrant
```

`smart-qdrant-mcp` is optional: the stack works without it, and it only adds an
agent-facing MCP surface over the same REST contract the middleware already
exposes to any HTTP client. See
[`smart-qdrant-mcp-server/README.md`](smart-qdrant-mcp-server/README.md) for its tools,
resources, prompts, and configuration. [`smart-skill/`](smart-skill/README.md)
packages the matching client-side knowledge as an agent skill — tool
selection, hybrid-search tuning, and the destructive-operation gates — for
MCP clients whose hosts load skills.

## Prerequisites

- Docker Engine/Desktop with Compose v2.24 or newer
- An external Qdrant instance reachable through `QDRANT_URL`, unless only
  Smart Embedder is enabled
- NVIDIA drivers/runtime only for the `gpu` profile

Copy `.env.example` to `.env`. It ships ready to start against a Qdrant running
on this machine's default port 6333 — the stack does not start one itself, so
that instance has to exist; `QDRANT_URL` is the only value to change if yours
lives elsewhere.

> **The shipped template runs with authentication disabled.** All three
> services accept every request without a bearer token, because the three
> `*_ALLOW_UNAUTHENTICATED_LOCAL` switches are `true` in `.env.example`. This
> is a first-run convenience, not a deployment posture. Every port stays
> published on `127.0.0.1`, so nothing outside this machine can reach the
> stack — but every process and every user on this host can read, rewrite and
> delete every collection. Turn authentication on before the stack holds
> anything real:
>
> 1. Put a `token:label` entry in `MIDDLEWARE_API_TOKENS`,
>    `SMART_EMBEDDER_API_TOKENS` and `MCP_API_TOKENS`.
> 2. Set `MIDDLEWARE_CLIENT_TOKEN` and `SMART_EMBEDDER_CLIENT_TOKEN`, the
>    credentials the services present to each other. Each must differ from
>    every token in the three lists above.
> 3. Set all three `*_ALLOW_UNAUTHENTICATED_LOCAL` back to `false`.
> 4. Run `pwsh stack.ps1 verify` (or `bash stack.sh verify`), then restart.
>
> A service whose list is empty while its switch is `false` refuses to start,
> so steps 1 and 3 cannot half-apply unnoticed.

`smart-qdrant-mcp` starts by default alongside `middleware`; set
`MCP_SERVER_ENABLED=false` to leave it out, and `EMBEDDER_ONLY=true` removes it
too. While it is enabled its default
`MCP_AUTH_MODE=static` needs a non-empty `MCP_API_TOKENS`, unless
`MCP_ALLOW_UNAUTHENTICATED_LOCAL=true`; alternatively,
`MCP_AUTH_MODE=oauth` needs the issuer, resource, introspection endpoint, and
confidential-client credentials documented in
[`smart-qdrant-mcp-server/README.md`](smart-qdrant-mcp-server/README.md). The
preflight checks the cross-service invariants: the three ACLs stay pairwise
disjoint, a service requiring authentication has a caller token, and a service
whose list is empty has its opt-out set (or, for the MCP server, is running
under `MCP_AUTH_MODE=oauth`, where the list is ignored). It does not validate
the OAuth provider settings; that mistake leaves only the MCP container
crash-looping while the other services continue. An
unauthenticated service is allowed only through its own explicit
loopback-development opt-out — `MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL`,
`SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL`, or
`MCP_ALLOW_UNAUTHENTICATED_LOCAL` — set alongside an empty list for that
service. The three services' effective ACLs must stay pairwise disjoint:
each is its `*_API_TOKENS` list plus the one `*_CLIENT_TOKEN` its caller
presents, and the launcher refuses to start on an overlap.

## Start and Operate

**[`scripts/README.md`](scripts/README.md) is the complete launcher reference** --
every command, every flag, the configuration precedence rule, the recipes, the
exit-code contract and the troubleshooting entries. What follows is the short
version and the repository-level context behind it.

The launchers keep exactly one hardware service (`cpu` or `gpu`) active. An
explicit CLI value wins; otherwise Compose resolves `STACK_PROFILE` from the
process environment and `.env`. The model profile (`basic` or `advanced`) uses
the same precedence through `SMART_EMBEDDER_PROFILE`. Profile names are
case-insensitive at the launcher boundary and normalized before Compose runs.

PowerShell examples:

```powershell
pwsh stack.ps1 verify          # check .env before anything is built
pwsh stack.ps1 up cpu
pwsh stack.ps1 up gpu advanced
pwsh stack.ps1 up -NoBuild     # use STACK_PROFILE and existing images
pwsh stack.ps1 up gpu -Wait    # return only after services are healthy
pwsh stack.ps1 urls            # endpoints, and whether each wants a token
pwsh stack.ps1 status cpu
pwsh stack.ps1 logs gpu -Tail 200
pwsh stack.ps1 restart cpu     # no rebuild
pwsh stack.ps1 down
pwsh stack.ps1 help            # also what a bare invocation prints
```

Linux/macOS equivalents:

```bash
bash stack.sh verify
bash stack.sh up cpu
bash stack.sh up gpu advanced
bash stack.sh up --no-build
bash stack.sh up gpu --wait
bash stack.sh urls
bash stack.sh status cpu
bash stack.sh logs gpu --tail 200
bash stack.sh restart cpu
bash stack.sh down
bash stack.sh help
```

`build`, `pull`, `config` and `version` complete the set; `scripts/README.md`
has the table.

`up` builds before stopping the opposite hardware service, so a failed build
leaves the running profile untouched. `--no-build`/`-NoBuild` skips that build;
`--wait`/`-Wait` waits for Compose health checks, bounded by a 600-second
`--wait-timeout`/`-WaitTimeout`. `logs` follows output by default; `--tail
N`/`-Tail N` prints the last N lines and exits. Without a hardware argument,
`status` and `logs` cover both profiles. The three switches below each have a
CLI override that beats `.env` for one run.

Set `EMBEDDER_ONLY=true` in `.env` to build and start only the selected Smart
Embedder container. The launcher removes the middleware and the opposite
hardware service, and Qdrant is not required. The default `false` starts the
complete stack.

Set `MCP_SERVER_ENABLED=false` to start everything except `smart-qdrant-mcp`.
The launcher scales that one service to zero, which also stops it if a previous
run left it running, and stops requiring `MCP_API_TOKENS`. The default `true`
keeps it in the stack. `EMBEDDER_ONLY=true` removes it either way.

Set `EXPOSE_SMART_EMBEDDER_PORT=false` to stop publishing the Smart Embedder on
the host. The launcher applies `compose.embedder-private.yml`, which withdraws
that single port publish; the middleware is unaffected because it reaches the
embedder over the private network. This closes the only route to the embedder
that skips the middleware's auth, rate limiting, and rerank truncation, so it is
the right setting for a deployment — but it also makes
`server-embedder-reranker`'s `tests/test_server.py` and `benchmark.py`
unreachable from the host, since both target that port. Both switches are read
by the launcher only: a bare `docker compose up` ignores them.

The commands that act on your configuration -- `up`, `verify`, `config`,
`build`, `pull`, `restart` and `urls` -- require `.env`. The recovery commands
`status`, `logs`, `down`, `version` and `help` use the versioned template and
remain available when `.env` is missing or malformed, and without a Python
interpreter on `PATH`. A missing local `nvidia-smi` produces a warning rather than blocking
a potentially remote GPU Docker context. Compose remains the authoritative GPU
runtime check.

The middleware is available at `http://127.0.0.1:6336`; Smart Embedder defaults
to `http://127.0.0.1:8010`, unless `EXPOSE_SMART_EMBEDDER_PORT=false` withdraws
that publish; `smart-qdrant-mcp` listens on `${MCP_BIND_HOST}:6337`,
loopback by default. Containers communicate through the private
smart-embedder:8000 DNS alias. The running containers are named
qdrant-middleware, smart-qdrant-mcp, and embedder-server-cpu or
embedder-server-gpu -- each embedder preceded by its own
embedder-server-init-* container.

Before each serving container starts, a non-network-facing init container
ensures the profile's pinned models and hardware state are ready. The serving
container mounts models and persisted state read-only; TorchInductor and Triton
use a non-persistent executable tmpfs for generated native code. A cold first
start can take time while init downloads and calibrates, while subsequent starts
reuse the verified volumes.

## Exposure Controls

Middleware exposure controls default to `true`; Smart Embedder Swagger defaults
to `false`:

- `EXPOSE_FRONTEND` controls the middleware UI and `/assets/config.js`.
- `EXPOSE_MIDDLEWARE_SWAGGER` controls middleware `/docs`, `/redoc`, and
  `/openapi.json`.
- `EXPOSE_SMART_EMBEDDER_SWAGGER` controls the same documentation paths on
  Smart Embedder.

These switches do not change application endpoints, payloads, models, or
Qdrant schemas.

## Security

The middleware and Smart Embedder publish on a fixed `127.0.0.1`; neither is
intended for direct public-IP exposure. `smart-qdrant-mcp` is the one
exception, and a narrow one: `MCP_BIND_HOST` chooses its published address so
a private VPN address is possible, but it defaults to loopback and the server
refuses to start on `0.0.0.0` or `::`. Remote access must use a trusted
private network, personal VPN, or authenticated tunnel while retaining bearer
tokens. Do not remove the loopback binds in `compose.yml`;
`scripts/verify_compose_security.py` enforces them.

Serving containers run as UID/GID 1000 on a dedicated bridge network with
read-only root filesystems, dropped Linux capabilities, `no-new-privileges`,
bounded memory/CPU/PIDs, writable tmpfs paths, and log rotation. A minimal init
bootstrap starts with only `CHOWN`, `SETUID`, and `SETGID` so a fresh named
volume can be assigned to UID/GID 1000; it drops those privileges before model
or network libraries are imported. Smart Embedder model and state volumes are
read-only in the HTTP process and writable only in that non-exposed preparation
container. Base images and Python packages are pinned. Never commit `.env`;
only `.env.example` is versioned.

## Dependencies and Reproducibility

Human-maintained direct dependencies live under `requirements/`: one file per
image (`middleware.txt`, `embedder-cpu.txt`, `embedder-gpu.txt`, `mcp.txt`),
plus `embedder-base.txt` and `shared.txt` for pins the images have in common,
and `dev.txt` / `ci.txt` for tooling that never ships. Exact overrides for a
known-vulnerable transitive dependency live separately in
`security-constraints.txt`; the release process synchronizes them into the
affected generated locks rather than passing a second constraint to Docker.
Docker applies the matching target-specific constraint under `requirements/locks/`
(`middleware-py311-linux.txt`, `embedder-cpu-py311-linux.txt`,
`embedder-gpu-py310-linux.txt`, `mcp-py311-linux.txt`); this fixes the complete
transitive environment for the Linux and Python version used by each image. If
a direct pin changes, rebuild and test the target, regenerate its constraint
from the verified image with `python -m pip freeze --all`, and commit both
changes together.

Smart Embedder model repositories are resolved at immutable Hugging Face commit
revisions declared in `server-embedder-reranker/model_revisions.py`. The cache
therefore cannot silently move to a different `main` revision between builds or
hosts. Updating a model requires changing its revision explicitly and repeating
the CPU/GPU smoke tests.

## Development and Verification

```powershell
py -m pytest middleware-qdrant-manager/tests -q
py -m pytest smart-qdrant-mcp-server/tests -q
Push-Location server-embedder-reranker
py -m unittest tests.test_reranker_wrapper tests.test_embedder_profile
Pop-Location
py -m pytest scripts/test_verify_compose_security.py -q
py -m ruff check .
py -m ruff format --check .
docker build --check .
docker compose --env-file .env.example --profile cpu config --quiet
```

API details remain in
[`middleware-qdrant-manager/README.md`](middleware-qdrant-manager/README.md),
[`server-embedder-reranker/README.md`](server-embedder-reranker/README.md),
and [`smart-qdrant-mcp-server/README.md`](smart-qdrant-mcp-server/README.md);
agent-side usage of the MCP server is covered by
[`smart-skill/README.md`](smart-skill/README.md).

## License

This project is licensed under the **GNU Affero General Public License v3.0** —
see [`LICENSE`](LICENSE) for the full text.

AGPL rather than plain GPL because this stack *is* a network service.
Section 13 of the licence covers exactly that case: if you modify it and let
other people reach your modified version over a network, they are entitled to
its source. Running it unmodified, or modifying it for your own use behind
your own loopback, obliges you to nothing.

The models the stack downloads at runtime are **not** covered by it; each
carries its own license, and none is restrictive: `BAAI/bge-m3` and
`BAAI/bge-m3-unsupervised` are MIT, while `BAAI/bge-reranker-v2-m3`,
`Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B` and
`omai-research/milco-650m` are Apache-2.0. They are fetched from Hugging Face
at the pinned revisions in
[`server-embedder-reranker/model_revisions.py`](server-embedder-reranker/model_revisions.py)
and are never redistributed by this repository.
