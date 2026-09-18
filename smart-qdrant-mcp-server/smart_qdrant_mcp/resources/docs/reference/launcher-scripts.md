# scripts — the stack launchers and their reference

> **Read this when** you need to build, start, inspect or stop the stack, look
> up a launcher flag, explain an exit code, or change how the launchers decide
> what Compose is asked to do.
> **Audience** — an AI agent working in this repository, and any human reading
> over its shoulder.
> **Source of truth** — this file for the launcher surface: every command,
> every flag, the precedence rule, the recipes, the exit-code contract and the
> troubleshooting entries. Nothing else needs to be read to run the stack.
> **Prerequisite** — none. This file stands alone; the root
> [`README.md`](../README.md) introduces the stack and gives the one-command
> quickstart.

---

## Purpose and boundaries

`stack.ps1` (Windows) and `stack.sh` (Linux, macOS, Git Bash) are the **only
supported ways** to build, start, inspect and stop the Smart Semantic Searcher
stack. They are kept at parity: the same command produces the same Compose
invocation on either platform, and a test asserts it.

**Both launchers live at the repository root**, and every command below is run
from there. What stays in `scripts/` is what they *call* rather than what you
call: the shared planner `stack_core.py`, configuration and dependency
verifiers, release-evidence helpers, their tests, and this reference.

**What the launchers are:**

- A thin, verified planning layer in front of `docker compose`. They resolve
  the configuration, run the preflight checks no single container can perform,
  and then hand Compose an argument list.

**What the launchers are not:**

- Not a replacement for Compose. Every effect they have is a Compose call you
  can see with `--dry-run`.
- Not a multi-tenant orchestrator. One stack per machine — see
  [Why `--project-name` does not give you two stacks](#why---project-name-does-not-give-you-two-stacks).
- Not a configuration generator. They never invent, write or edit `.env`.

## Mental model

Four facts explain the whole design.

1. **One planner, two shells.** Everything that decides *what Compose is asked
   to do* lives in `scripts/stack_core.py`, once, for both launchers. The shell
   scripts parse arguments and execute; they do not decide.
2. **The recovery commands deliberately bypass the planner.** `down`, `logs`,
   `status`, `version` and `help` need **neither Python nor a readable
   `.env`**. Stopping a stack must keep working when the configuration is what
   broke. `scripts/test_stack_recovery.py` enforces this; do not weaken it.
3. **`up` verifies first, and builds before it tears anything down.** A failed
   build leaves the currently running profile untouched.
4. **Launcher settings shape the stack.** The three boolean switches
   `EMBEDDER_ONLY`, `MCP_SERVER_ENABLED` and `EXPOSE_SMART_EMBEDDER_PORT`, plus
   the optional `STACK_GPU_DEVICE` selector, are read by the launchers, never
   by a service. A bare `docker compose up` ignores them.

## Map of the directory

| Path | Role | Change it when |
|---|---|---|
| `../stack.ps1` | Windows entry point: argument parsing, execution, help text. | A flag's spelling or the help text changes. |
| `../stack.sh` | POSIX entry point, at parity with the above. | Same, in the same commit. |
| `stack_core.py` | The shared planner: resolves configuration and emits the Compose argument list. | Any decision about *what Compose runs* changes. |
| `stack_gpu_probe.py` | Bounded local Docker-endpoint and NVIDIA inventory discovery for automatic startup and explicit GPU selection. | Host discovery changes; policy remains in the pure planner. |
| `verify_token_config.py` | Cross-service token preflight: the effective inbound ACLs must be pairwise disjoint. | The token model changes. |
| `verify_compose_security.py` | Compose hardening checks: public/private port shapes for CPU, all-GPU and selected-GPU deployments. | A port publish, GPU reservation or hardening property changes. |
| `verify_dependencies.py` | Vulnerability audit + SBOM for the pinned locks: queries OSV for every locked `name==version` and can emit a CycloneDX SBOM per lock. Needs network access; stdlib-only, installs nothing. | A lock in `requirements/locks/` changes. |
| `security_constraints.py` | Validates exact transitive security overrides against every generated lock and can synchronize them before images are rebuilt and frozen. | `requirements/security-constraints.txt` or an affected lock changes. |
| `release_evidence.py` | Rejects an incomplete release-evidence directory and writes its SHA-256 manifest. | The required release evidence or manifest schema changes. |
| `render_dependency_audit_issue.py` | Renders the scheduled OSV audit's vulnerable result as a stable, deduplicated issue body. | The audit report schema or issue workflow changes. |
| `stack_test_support.py` | Shared fake-Docker harness for executing the real Bash/PowerShell launchers without changing Docker state. | Launcher tests need new simulated responses or invocation assertions. |
| `test_security_constraints.py`, `test_release_evidence.py`, `test_render_dependency_audit_issue.py` | Regression coverage for security override synchronization and release/audit artifacts. | Their matching utility changes. |
| `test_branding_contract.py` | Public product naming versus stable deployment identifiers. | Branding copy or deployment identifier contracts change. |
| `test_stack_*.py`, `test_verify_*.py` | The suites that hold all of the above — including `test_stack_readme.py`, which parses **this file's tables** against the launchers' real help output. | Always, alongside the change. |

### Agent reading path

Separate parsing/execution from planning before editing a launcher: shell
argument handling belongs in both root launchers; decisions about Compose
operations belong in `stack_core.py`. Follow the matching `test_stack_*`
module and reuse `stack_test_support.py` for subprocess tests. The recovery
commands deliberately bypass the planner; do not pull them into configuration
preflight as an incidental cleanup.

The verifier and evidence utilities have separate entry points and focused
tests. Read their parser and exit-code handling before invoking them: an audit
can require network access, and a synchronization option can write lock files.
The [dependency directory guide](../requirements/README.md) explains input
ownership; this README remains the command/procedure reference. GitHub workflow
wiring is described in [.github/README.md](../.github/README.md).

## Interface reference

### TL;DR

```powershell
Copy-Item .env.example .env      # then edit it: tokens, ports, profiles
pwsh stack.ps1 verify    # is my configuration sound?
pwsh stack.ps1 up cpu    # build and start
pwsh stack.ps1 urls      # where is it, and does it want a token?
```

```bash
cp .env.example .env
bash stack.sh verify
bash stack.sh up cpu
bash stack.sh urls
```

`up` runs `verify` for you before it builds anything. Running it separately
first is useful because it needs no Docker daemon: you can check the
configuration before Docker Desktop has finished starting.

### Prerequisites

| Requirement | Why | How to check |
|---|---|---|
| Docker Engine, running | Everything the launchers do is a Compose call | `bash stack.sh version` |
| Docker Compose **2.24.4 or newer** | Environment resolution and the `!reset` / `!override` tags used by the private-port and GPU-selection overlays | `docker compose version --short` |
| Python 3.10+ on `PATH` | `up`, `verify`, `config`, `build`, `pull`, `restart` and `urls` route their decisions through `scripts/stack_core.py` | `py -V` / `python3 -V` |
| A `.env` file | The launchers never invent configuration | `Test-Path .env` |
| PowerShell 5.1+ **or** bash 3.2+ | One launcher each | `$PSVersionTable` / `bash --version` |

`down`, `logs`, `status`, `version` and `help` deliberately need **neither
Python nor a readable `.env`**. Stopping a stack must keep working when the
configuration is what broke.

An NVIDIA GPU is needed for the `gpu` hardware profile. Explicit `gpu` without
a selector keeps the advisory NVIDIA check: a missing or failing local tool
does not refuse startup because Docker may target a remote machine. Automatic
startup and GPU selection follow the local discovery rules below.

### Commands

| Command | Windows | Unix | What it does |
|---|---|---|---|
| Help | `pwsh stack.ps1 help` | `bash stack.sh help` | Prints the usage text. This is also what a bare invocation does. |
| Version | `... version` | `... version` | Launcher, Docker Engine and Compose versions, plus the required Compose minimum. Reports absence instead of failing on it. |
| Verify | `... verify` | `... verify` | Runs the cross-service token preflight and the Compose hardening checks against your `.env`, then stops. Touches no Docker state. |
| Start | `... up [cpu\|gpu\|auto] [basic\|advanced]` | same | Verifies, builds, removes the other hardware profile's services, starts the stack. |
| Build only | `... build [cpu\|gpu\|auto]` | same | Builds the images and starts nothing. Produces exactly the images `up` would. |
| Pull | `... pull [cpu\|gpu\|auto]` | same | Fetches the images this stack does not build itself. See the note under *Recipes*. |
| Restart | `... restart [cpu\|gpu]` | same | Restarts the running services without rebuilding. |
| Config | `... config [cpu\|gpu\|auto]` | same | Prints the fully resolved Compose configuration, **credentials redacted**. |
| Endpoints | `... urls` | same | Prints the published endpoints and each service's authentication state. Quotes no token. |
| Status | `... status [cpu\|gpu]` | same | `docker compose ps` for the stack's services. |
| Logs | `... logs [cpu\|gpu]` | same | Container logs. Follows by default; see `-Tail` / `--tail`. |
| Stop | `... down` | same | Stops and removes the containers. Named volumes survive unless you ask otherwise. |

The hardware profile (`cpu` / `gpu`) and model profile (`basic` / `advanced`)
are positional and optional. When omitted they come from `STACK_PROFILE` and
`SMART_EMBEDDER_PROFILE` in `.env`. Only `up`, `config`, `build` and `pull` also
accept CLI hardware `auto`; its model choice follows the automatic rules below.

### Parameters

Every Unix flag has a PowerShell parameter with the same name in PascalCase.

| Unix flag | PowerShell | Accepted by | Default | Effect |
|---|---|---|---|---|
| `--autotune <mode>` | `-Autotune <mode>` | `up`, `config` | configured mode or `quick` | Select `off`, `quick` or `full`; trims and lowercases the CLI value. |
| `--force-autotune` | `-ForceAutotune` | `up` | off | Stop the selected embedder and request one forced preparation before ordinary startup. Requires effective `quick` or `full`. |
| `--frontend` | `-Frontend` | `up`, `config` | configured `EXPOSE_FRONTEND` | Enable the middleware console. Requires the middleware. |
| `--no-frontend` | `-NoFrontend` | `up`, `config` | configured `EXPOSE_FRONTEND` | Disable the middleware console. Requires the middleware. |
| `--no-build` | `-NoBuild` | `up` | off | Reuse the existing images. |
| `--wait` | `-Wait` | `up` | off | Block until the health checks pass. |
| `--wait-timeout <s>` | `-WaitTimeout <s>` | `up` | `1800` | Positive integer seconds to wait. Zero is rejected. Implies `--wait`. Compose's own default is `0`, which means *wait forever*. Services that depend on the Smart Embedder start only after its preparation container exits, and an unset ADVANCED quick calibration may run for up to 1,080 s on CPU (720 s on CUDA) before the slowest health check's 300 s start period, so 1800 covers a first start and still lets it fail honestly. |
| `--skip-verify` | `-SkipVerify` | `up` | off | Do not run the configuration verification before building. |
| `--gpu-device <device>` | `-GpuDevice <device>` | `up`, `config`, `build`, `pull` | unset | Select one physical GPU by index, UUID or unique UUID prefix. |
| `--show-secrets` | `-ShowSecrets` | `config` | off | Print credential values in cleartext. Warns on stderr when used. |
| `--tail <n>` | `-Tail <n>` | `logs` | unset | Last *n* lines. Also accepted as `--tail=<n>`. Suppresses the implicit follow. |
| `--follow` | `-Follow` | `logs` | implicit | Keep streaming. Combinable with `--tail` and `--since`. |
| `--since <t>` | `-Since <t>` | `logs` | unset | Only entries newer than a timestamp or duration (`10m`, `2026-08-19T10:00:00`). |
| `--service <name>` | `-Service <name>` | `logs` | all | Restrict to named services. Repeat the flag, or comma-separate the names; PowerShell parameters cannot repeat, so `-Service a,b` is the spelling there and bash accepts both. |
| `--volumes` | `-Volumes` | `down` | off | **Destroys the named volumes, including the Hugging Face model cache.** Prompts for confirmation on a terminal; refused without `--yes` when there is no terminal to ask. |
| `--remove-orphans` | `-RemoveOrphans` | `down` | off | Remove containers for services no longer in the Compose file. |
| `--yes` | `-Yes` | `down` | off | Non-interactive confirmation for `--volumes`. |
| `--dry-run` | `-DryRun` | every command but `help` and `version` | off | Print the steps that would run, one argument per line, and run none of them. |
| `--env-file <path>` | `-EnvFile <path>` | every command but `help` and `version` | `.env` | Use another environment file. It must exist: there is no fallback to the template. |
| `--project-name <n>` | `-ProjectName <n>` | every command but `help` and `version` | Compose's default | Isolates volumes, networks and Compose's bookkeeping. **It does not let two stacks run side by side** — see below. |

### The three launcher switches

These boolean settings have a CLI override and an `.env` value. The GPU selector
below also has both. No service reads them; a bare `docker compose up` ignores
all three switches.

| Pair | `.env` variable | Effect when true |
|---|---|---|
| `--embedder-only` / `--full`<br>`-EmbedderOnly` / `-Full` | `EMBEDDER_ONLY` | Start only the Smart Embedder; no middleware, no MCP server. |
| `--no-mcp` / `--mcp`<br>`-NoMcp` / `-Mcp` | `MCP_SERVER_ENABLED` | `--no-mcp` leaves out `smart-qdrant-mcp` (`--scale smart-qdrant-mcp=0`). Absent from `.env` means enabled. |
| `--private-embedder` / `--public-embedder`<br>`-PrivateEmbedder` / `-PublicEmbedder` | `EXPOSE_SMART_EMBEDDER_PORT` | `--private-embedder` withdraws the Smart Embedder's host port publish by adding `compose.embedder-private.yml`. The middleware still reaches it over the private Compose network. |

Passing both halves of a pair is a **usage error**, not last-one-wins: two
different stacks were asked for and no reading of the command line satisfies
both.

### Automatic startup and GPU selection

Use `up auto` to choose hardware and models for this invocation. The launcher
queries `nvidia-smi --query-gpu=index,uuid,memory.total
--format=csv,noheader,nounits` with a five-second timeout. It compares reported
**total** memory in integer MiB, not currently free memory: **7680 MiB (7.5 GiB)**
or more chooses `advanced`; less chooses `basic` on GPU. The autotuner still
measures actual workload admission and available headroom.

Without a selector, `auto` picks the GPU with the largest total memory; ties
use ascending canonical UUID. This is not necessarily host GPU 0. The selected
GPU becomes CUDA device 0 in the container. An explicit positional model wins
over the automatic result; environment model/hardware defaults do not override
an explicit `auto` request. `STACK_PROFILE=auto` is invalid when the environment
supplies the hardware choice: `auto` is CLI-only. Nothing rewrites `.env` or
persists the result.

Use `--gpu-device 0` / `-GpuDevice 0` to select host GPU 0 explicitly. Bash also
accepts `--gpu-device=0`. Selection precedence is CLI > shell `STACK_GPU_DEVICE`
> environment-file `STACK_GPU_DEVICE` > empty. An empty CLI selector is invalid.
Use a full UUID for a persistent preference, for example
`GPU-abcdefab-1234-abcd-abcd-abcdefabcdef`; indices can change across enumeration.
`GPU-2222` is a valid unique UUID prefix, case-insensitively, with any hyphens
in their canonical UUID positions. Digits alone, including `2222`, always mean
an index. Bare `GPU-`, unprefixed hexadecimal text such as `abcd`, lists,
`all`, `none`, `void` and MIG identifiers are invalid. Ambiguous or unknown
selectors stop with a list of usable GPU identities and capacities.

| Request and facts | Result |
|---|---|
| `up auto`, 7679 MiB GPU | GPU / basic, selected UUID |
| `up auto`, 7680 MiB GPU | GPU / advanced, selected UUID |
| `up auto`, 8188 MiB GPU | GPU / advanced, selected UUID |
| `up auto`, GPU 0 = 4096 MiB, GPU 1 = 24564 MiB | GPU / advanced, GPU 1 selected |
| `up auto --gpu-device 0`, GPU 0 = 4096 MiB | GPU / basic, GPU 0 selected |
| `up auto basic`, 24564 MiB GPU | GPU / basic, explicit model wins |
| `up auto advanced`, no detected NVIDIA | CPU / advanced, explicit model wins |
| `up auto`, NVIDIA tool absent, no selector | CPU / basic, with an explanation; absence of the tool does not prove absence of NVIDIA hardware |
| `up auto`, successful query reports no devices | CPU / basic |
| `up gpu advanced --gpu-device 1` | GPU / advanced, GPU 1 selected |
| `up gpu`, no selector | Configured model, existing all-GPU reservation |
| `up cpu`, persistent selector | CPU, persistent selector ignored |
| `up cpu --gpu-device 1` | Usage error |
| Explicit selector with missing tool or unavailable GPU | Error, selector never silently ignored |
| Failed, timed-out, empty or wholly unusable query | Error before mutation; every nonzero query status, including 6, stops |
| `restart auto` or `restart --gpu-device 1` | Usage error |

Malformed individual inventory rows are excluded with a warning if usable GPUs
remain; duplicate indices or UUIDs invalidate the inventory. Only a successful
query whose complete output is `No devices were found` counts as no devices.
Startup diagnostics report hardware, model provenance, full UUID and exact MiB
before any mutation. Allocation failure later in Docker stays a Docker error;
there is no CPU or model retry.

Discovery requires the effective Docker endpoint to be a local Unix socket or
Windows named pipe. TCP (including loopback), SSH and unclassified endpoints
are refused, preventing local inventory from selecting a GPU on another host.
Process `DOCKER_CONTEXT` takes precedence over process `DOCKER_HOST`; values
inside the service environment file do not redirect the Docker CLI. Forwarded
local sockets, MIG and parallel multi-GPU placement are outside this contract.
On Linux/WSL the probe also checks `/usr/lib/wsl/lib/nvidia-smi` if PATH has none.
Windows/WSL support depends on successful UUID reservation and CUDA visibility
qualification on the target runtime; Compose configuration alone cannot prove it.

Both GPU init and serving containers receive the same single UUID through
`compose.gpu-device.yml`; the default all-GPU Compose reservation stays intact
for explicit `gpu` without selection. The private-port overlay combines with
selection. Apply a changed selector with `up --no-build` (and the desired
hardware/model); `restart` and `urls` neither rediscover the running GPU nor
apply a persistent selector. Other recovery commands also avoid discovery.

`--dry-run` performs required bounded, read-only discovery to print an accurate
plan, without executing its mutations or verifiers. `--skip-verify` skips the
configuration verifiers, including Compose reservation hardening checks; GPU
request validation, discovery and selector resolution are mandatory even with
that flag. A model change can make existing collections incompatible; see the
[middleware's embedding contracts](../middleware-qdrant-manager/README.md#mental-model).

### Startup calibration and console controls

`--autotune` / `-Autotune` selects `off`, `quick` or `full` for preparation and
serving. `full` can reuse a compatible full profile; selecting it alone does not
force fresh measurements. `off` disables calibration while retaining model
preparation. Hardware/model selection, including `auto` and GPU selection, stays
independent of tuning mode.

```powershell
pwsh stack.ps1 up auto -Autotune full -ForceAutotune -DryRun
pwsh stack.ps1 up auto -Autotune full -ForceAutotune -NoBuild -Wait
pwsh stack.ps1 up auto -NoFrontend -NoBuild
pwsh stack.ps1 config cpu -Autotune off -NoFrontend
```

```bash
bash stack.sh up auto --autotune full --force-autotune --dry-run
bash stack.sh up auto --autotune full --force-autotune --no-build --wait
bash stack.sh up auto --no-frontend --no-build
bash stack.sh config cpu --autotune off --no-frontend
```

The force flag requests one **forced preparation attempt for this invocation**.
After verification and a successful build (unless skipped), the launcher removes
the opposite hardware services as usual, stops the selected embedder and its
named init service, then runs one temporary init container. That container uses
the existing bootstrap, mounts, privilege drop and selected GPU; only its force
value is true. Ordinary dependency-managed `up` follows with force=false in both
normal init and serving containers. The normal init may run again and reuse the
new compatible profile. Serving is unavailable during this preparation.

A calibration fallback can still return success with safe defaults; this flag
does not guarantee successful measurements. If no compatible profile resulted,
ordinary preparation may make another ordinary attempt. The launcher adds no
retry. A build failure preserves the running services; a stop/preparation/up
failure aborts subsequent steps with Docker's exit code and no success summary.
No volumes or cached profiles are deleted. The health `--wait-timeout` covers
the final Compose wait, not preceding downloads or calibration.

CLI controls beat shell values, then the selected environment file, then the
existing defaults (`quick`, force=false, frontend=true). Without a mode/force
flag, the service retains responsibility for interpreting its configured mode.
A valid CLI mode overrides even an invalid configured mode. Force without a
CLI mode validates the configured mode; effective `off` is an error, including
with `--skip-verify` and `--dry-run`. Empty, multiline or unsupported CLI modes
are errors. Bash also accepts `--autotune=quick` and rejects repeated mode flags.
PowerShell missing mode values or duplicate named parameters fail in native
binding with exit 1 before the script body; Bash handles these with exit 2.
Explicit empty/unsupported values reaching either launcher return 2.

The legacy `SMART_EMBEDDER_AUTOTUNE_FORCE=true` environment setting remains
persistent on unflagged invocations: preparation ignores reusable profiles
whenever it runs while true. The CLI force action overrides this for its normal
containers, but does not change the file or caller environment. A later unflagged
`up` can therefore apply an intentionally persistent true again. Prefer the CLI
action for a single requested recalibration.

The frontend flags only toggle the middleware console and its assets. API,
Swagger, ports and MCP remain governed by their existing settings. Both frontend
flags together are an error, as is either explicit flag with effective
`EMBEDDER_ONLY=true`. A configured frontend value alone is ignored without the
middleware; `--full --no-frontend` explicitly enables middleware without console.
Without a frontend flag, existing configured boolean spellings remain accepted.

Mode and frontend overrides stay in created containers until a subsequent `up`
applies the then-effective configuration. Use `up --no-build` to apply a change
without rebuilding. `restart` retains existing container environment and does
not replay the CLI forced action. These flags are accepted only by the commands
in the parameter table; `config` remains redacted and read-only. `--dry-run`
prints resolved choices and every ordered step without executing mutations.
PowerShell restores exported environment values on every exit path; Bash exports
only inside its child process. Neither shell edits `.env`.

### Why `--project-name` does not give you two stacks

Every service in `compose.yml` carries a hardcoded `container_name`
(`qdrant-middleware`, `smart-qdrant-mcp`, `embedder-server-cpu`, …), and
container names are **daemon-global rather than project-scoped**. A second
concurrent stack therefore fails with *"name already in use"* whatever the
project is called. `--project-name` is still a real isolation flag — separate
volumes, networks and Compose state — it just is not that one. One stack per
machine.

## Configuration

### Precedence

Highest wins:

1. the command line
2. the shell environment
3. the environment file (`.env`, or `--env-file`)
4. the built-in default

Worked example, for `EMBEDDER_ONLY`:

```bash
# .env says EMBEDDER_ONLY=false
bash stack.sh up cpu                     # full stack   (.env wins)
EMBEDDER_ONLY=true bash stack.sh up cpu  # embedder only (shell wins over .env)
EMBEDDER_ONLY=true bash stack.sh up cpu --full
                                                 # full stack   (CLI wins over both)
```

The shell-versus-`.env` half of that rule is Compose's own: the launchers ask
`docker compose config --environment` for the resolved values, so quoting,
comments and blank handling have exactly one answer. The launchers implement
only the CLI half and the defaults.

Use `--dry-run` whenever you are unsure which value won — it prints the exact
argv without executing the planned operations. Automatic/selected GPU plans
still perform the bounded read-only discovery described above.

### Where the rest of the configuration lives

The launchers read `.env`; they do not define it. The exhaustive, versioned
template is the root `.env.example`, and each service's own README documents
the variables it consumes — see [Related documents](#related-documents).

## Capabilities and limits

**You can:**

- Check a configuration with no Docker daemon running (`verify`, `config`,
  `urls`).
- See exactly what would run, without running it (`--dry-run`, on every
  command but `help` and `version`).
- Switch hardware profile in one command, with the running profile safe until
  the new build succeeds.
- Run against a non-default environment file (`--env-file`) or Compose project
  (`--project-name`).
- Stop a stack whose `.env` is missing or malformed, on a machine with no
  Python.

**You cannot:**

- Run two stacks on one machine (hardcoded `container_name`s — see above).
- Run both hardware profiles at once; `up` removes the other one.
- Make the launchers write or repair `.env` — they only read it.
- Reach the Smart Embedder's host port under `--private-embedder`; that is the
  point of the switch.
- Get a token printed by `urls`, with or without `--show-secrets`.

## Recipes

**Smart Embedder only, no middleware and no MCP**

```bash
bash stack.sh up cpu --embedder-only
```

**Everything except the MCP server**

```bash
bash stack.sh up cpu --no-mcp
```

**Do not publish the Smart Embedder on the host**

```bash
bash stack.sh up cpu --private-embedder
bash stack.sh urls          # the 8010 endpoint is gone from the list
```

**Switch from CPU to GPU (or back)**

```bash
bash stack.sh up gpu
```

Nothing else is needed. `up` builds first and only then removes the opposite
profile's services, so a failed build leaves the currently running profile
untouched.

**First upgrade after the Compose project rename**

The default project is now `smart-semantic-searcher`. Because container names
are daemon-global, stop a running older checkout before updating:

```powershell
pwsh stack.ps1 down
```

```bash
bash stack.sh down
```

If the source has already been updated while the old project is still running,
target the former project explicitly:

```powershell
pwsh stack.ps1 down -ProjectName personal-smart-embedder
```

```bash
bash stack.sh down --project-name personal-smart-embedder
```

Do not add `--volumes` / `-Volumes`: the model cache and CPU/GPU state volumes
keep their existing names and data. The unused old network can be inspected
afterward with `docker network inspect personal-smart-embedder-private` and
removed only when its `Containers` object is empty.

**Refresh the images**

```bash
bash stack.sh build cpu     # rebuild from the Dockerfile
bash stack.sh pull cpu      # fetch what this stack does not build
```

`pull` normally reports that it has nothing to do: every service declares a
locally built `hybrid-manager/...` image. It stays useful the day one of them
is switched to a registry image. `build` is what refreshes the local ones.

**Full reset, model cache included**

```bash
bash stack.sh down --volumes --remove-orphans
```

This deletes the Hugging Face model cache. Rebuilding it means downloading
every model again, which is why it asks for confirmation, and why it refuses
outright when standard input is not a terminal (add `--yes` if you mean it in a
script).

**Two stacks side by side** — not supported. See *Why `--project-name` does not
give you two stacks*.

## Verification

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | Unmet prerequisite: Docker missing, daemon unreachable, Compose too old, environment file absent. |
| `2` | Usage error, or a failed configuration preflight. |
| anything else | Docker's own exit code, passed through unchanged. |

Both launcher bodies use the same exit-code contract. PowerShell native binding
can fail before that body: a missing `-Autotune` value or a duplicate named
parameter returns `1`; Bash handles these malformed arguments with `2`.
Explicit empty/unsupported values reaching the body return `2` on both shells.
The verifiers signal a violation with a non-zero exit of their own
(`verify_compose_security.py` uses `1`, `verify_token_config.py` uses `2`);
the launchers translate either to
`2`, because `1` is reserved for a prerequisite that is not met.

### Checking a change to the launchers

```powershell
py -m pytest scripts -q
shellcheck stack.sh
Invoke-ScriptAnalyzer -Path stack.ps1 -Severity Warning,Error
```

`Invoke-ScriptAnalyzer` signals findings by printing them, not by an exit code,
so read its output rather than `$LASTEXITCODE`. Install the two linters once:
`winget install koalaman.shellcheck` and
`Install-Module PSScriptAnalyzer -Scope CurrentUser`.

### Auditing dependencies

Reproducibility is not vulnerability management: the pinned locks freeze exact
versions (not artifact bytes: the locks contain no hashes) but say nothing about
whether those versions carry published CVEs.
Audit them against the OSV database (needs network; stdlib-only, installs
nothing):

```powershell
py scripts\security_constraints.py               # security overrides match every lock
py scripts\verify_dependencies.py                # audit; exits 1 on any finding
py scripts\verify_dependencies.py --sbom out\ --json-output out\audit.json
```

This runs over the Linux-targeted locks in `requirements/locks/` from any host.
`pip-audit -r` is deliberately not used: it resolves the lock by doing a
`pip install --dry-run`, which builds Linux-only wheels (uvloop) and so cannot
run on the Windows development host. Asking OSV about the pins directly is
platform-independent and never installs a package.

`requirements/security-constraints.txt` is the human-maintained input for a
vulnerable **transitive** package. It is never passed as a second Docker-build
`-c`: the target lock remains the build's one canonical constraints file. Run
`py scripts\security_constraints.py --write`, rebuild and test every affected
image, freeze each verified environment back into its lock, run the read-only
constraint check again, then run the OSV audit. The verifier's exit statuses
are deliberately distinct: `1` means vulnerabilities, `2` an internal/invalid
response error, and `3` OSV unavailable. Only `0` is a release pass.

The suite in `scripts/` is unusually opinionated about **this file**:
`test_stack_readme.py` parses the flag tables above and compares them against
the launchers' real `help` output in both directions, checks the *Accepted by*
column against the option matrix, and asserts that the exit-code contract, the
`container_name` constraint and every troubleshooting entry are still present.
A flag the README invents fails; a flag the launchers grow without being
written down fails too.

## Failure modes

**Forced autotune cannot run with mode off.** Select `quick` or `full` with the
mode flag, or remove the force flag. For unsupported mode values, the error
identifies the supplied CLI mode or configured `SMART_EMBEDDER_AUTOTUNE` source.

**Forced preparation failed after the embedder stopped.** Fix the reported
Docker/bootstrap/preparation error, then rerun `up`. The launcher leaves services
stopped after this failure, retains all volumes and does not claim a successful
startup. An autotuner fallback reported in preparation logs is a separate case:
it may allow startup with safe defaults.

**Frontend flags require the middleware.** Remove the explicit frontend flag for
an embedder-only stack, or use `--full` / `-Full` to start the middleware.

**`Docker daemon not reachable`**
The CLI is installed but the daemon is not answering. Start Docker Desktop, or
`systemctl start docker`. `config`, `verify` and `urls` keep working without
it; they resolve everything client-side.

**GPU discovery or device selection fails**
Check the NVIDIA query and selector described under
[automatic startup and GPU selection](#automatic-startup-and-gpu-selection).
Confirm the requested index or UUID exists and Docker uses a local context.
Repair a failed or timed-out NVIDIA query before retrying; an explicitly
selected GPU is never silently ignored.

**`Docker Compose 2.24.4 or newer is required`**
Upgrade Compose. The launchers use `docker compose config --environment`,
`!reset` in `compose.embedder-private.yml`, and `!override` in
`compose.gpu-device.yml`. The last tag requires 2.24.4 and replaces the all-GPU
reservation without retaining `count` beside `device_ids`.

**`bad interpreter: /usr/bin/env bash^M`**
`stack.sh` was checked out with CRLF line endings. `.gitattributes` pins `*.sh`
to LF. Preserve any unrelated work, then normalize this file only (on Windows,
use `py` in place of `python3` if needed):

```bash
python3 -c "from pathlib import Path; p = Path('stack.sh'); p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))"
bash -n stack.sh
git diff -- stack.sh
```

**`token overlap: N credential(s) open both …`**
Two services share a bearer token. Their effective inbound ACLs must be
pairwise disjoint — each service's own `*_API_TOKENS` list plus the one
`*_CLIENT_TOKEN` its caller presents. Give each its own value. The message
never quotes a token; `urls` will tell you which services require one.

**`up --wait` never returns**
It cannot hang indefinitely any more: `--wait-timeout` defaults to 1800 seconds.
If it times out, `logs --tail 200` on the service that did not become healthy
is the next step. The Smart Embedder's first start is slow by design — it
downloads and prepares models — so a cold `up` legitimately takes minutes.

**Permission errors under `middleware-qdrant-manager/exports/`**
That directory is written by the container as UID/GID 1000. When your host user
is not UID 1000, `chown 1000:1000 middleware-qdrant-manager/exports` (or
`chmod 777` it) once. The sibling `data/` is mounted read-only, so the
container never writes there — but on Linux a directory the daemon creates is
root-owned, and staging a corpus into it then needs `sudo`. It is tracked with
a `.gitkeep` so a fresh clone already owns it.

**PowerShell refuses to run the script after a ZIP download**
Windows marks downloaded files, and the default execution policy blocks them:

```powershell
Unblock-File stack.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The `Set-ExecutionPolicy` above lasts only for that PowerShell session.

**On Windows, `bash stack.sh` fails immediately**
A bare `bash` on `PATH` usually resolves to `C:\Windows\System32\bash.exe`, the
WSL launcher, which fails with `execvpe(/bin/bash) failed` when no distribution
is installed. Use Git Bash explicitly:

```powershell
& "C:\Program Files\Git\bin\bash.exe" stack.sh up cpu
```

Or just use `stack.ps1`, which is the supported Windows entry point.

## Invariants

- **`config` redacts credentials by default.** Any value whose key ends in
  `_TOKEN`, `_TOKENS`, `_SECRET`, `_PASSWORD` or `_KEY` is replaced with
  `<redacted>`, in both YAML shapes Compose emits. Empty, `null` and short
  integer values survive, because they are not secrets and hiding them would
  make `config` useless for *"did my token reach the container?"*.
  `--show-secrets` restores the old behaviour and says so on stderr.
- **Nothing sensitive travels on argv.** The resolved environment reaches
  `stack_core.py` and `verify_token_config.py` over stdin, because process
  arguments are world-readable through `ps`.
- **`urls` never prints a token**, with or without `--show-secrets`. It reports
  only whether one is required.
- **Publishes are loopback or private, never public.** The Smart Embedder
  publish is hardcoded to `127.0.0.1` in `compose.yml`. The middleware and MCP
  publishes are the operator-settable addresses (`MIDDLEWARE_BIND_HOST`,
  `MCP_BIND_HOST`): each defaults to loopback, accepts `::1` or a private
  RFC1918 address, and refuses `0.0.0.0`. Off loopback, `verify` and the
  service itself refuse to serve that service without authentication.
- **On POSIX, a group- or world-readable `.env` earns a warning.** It holds
  bearer tokens; `chmod 600 .env`. It is a warning rather than a refusal
  because the file mode is the operator's call. The check is skipped on
  Windows, where Git for Windows synthesises a mode that says nothing about the
  real ACL.
- **`verify` runs on every `up`** unless `--skip-verify` is passed. It checks
  the invariants no single container can see, because no container is given
  every service's token list.
- **The recovery commands stay Python-free and `.env`-free.**
  `test_stack_recovery.py` enforces it; do not weaken it.
- **The two launchers stay at parity.** `test_stack_parity.py` asserts that the
  same command produces the same Compose invocation on both.

## Extending it

- **Adding a flag** — add it to both launchers, to `ALLOWED` in
  `test_stack_options.py`, and to the parameter table above with a correct
  *Accepted by* cell, in one commit. The README test fails otherwise, in both
  directions.
- **Adding a command** — same, plus a row in the command table; the test
  asserts every command the help text offers appears in this file.
- **Changing what Compose is asked to do** — the change belongs in
  `stack_core.py`, not in a shell script. Keep the recovery commands out of it.
- **Adding a hardening rule** — `verify_compose_security.py`, and it must keep
  passing for both hardware profiles crossed with both
  `EXPOSE_SMART_EMBEDDER_PORT` settings, including selected GPUs (six combinations).
- **Adding a failure message** — add a troubleshooting entry above; the test
  parametrises over the topics it expects to find.

## Related documents

| Document | Covers |
|---|---|
| [`../README.md`](../README.md) | The stack as a whole, and the short version of this reference. |
| [`../middleware-qdrant-manager/README.md`](../middleware-qdrant-manager/README.md) | The middleware service: REST API, configuration, behaviour. |
| [`../server-embedder-reranker/README.md`](../server-embedder-reranker/README.md) | The Smart Embedder service: HTTP API, profiles, autotuning, configuration. |
| [`../smart-qdrant-mcp-server/README.md`](../smart-qdrant-mcp-server/README.md) | The optional MCP server: tools, resources, prompts, configuration. |
| [`../AGENTS.md`](../AGENTS.md) | Repository-wide conventions for agents working on the code. |
| `../.env.example` | The exhaustive, versioned configuration template. |
