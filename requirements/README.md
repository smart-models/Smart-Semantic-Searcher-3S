# Dependency inputs and reproducible image locks

## Purpose and boundaries

This directory owns Python dependency inputs for the three independently built
services, development/CI tooling, and target-specific resolved locks. It does
not own model revisions, browser-test packages or local environment secrets.
An agent updating a package must identify the affected image targets before
editing a pin: a successful installation on the host does not validate a Linux
CPU or CUDA image.

## Mental model

Human-maintained requirement files declare direct dependencies and shared
includes. Docker installs a target's requirements with that target's one
transitive constraints file from `locks/`. The lock is a freeze of a verified
image environment, not a universal lock for every Python/platform combination.

Security overrides are a separate maintenance input. They are synchronized into
the target locks by the verifier workflow; builds must not combine a stale lock
with a second security constraints file. Development and CI environments are
not serving images and do not substitute for rebuilding an affected target.

## Map of the code

| Path | Responsibility |
|---|---|
| [shared.txt](shared.txt) | Direct pins shared by service requirement inputs. |
| [middleware.txt](middleware.txt) | Middleware API/client dependencies, including shared inputs. |
| [mcp.txt](mcp.txt) | MCP transport/server dependencies, including shared inputs. |
| [embedder-base.txt](embedder-base.txt) | Common embedding/reranking HTTP, model and numeric dependencies. |
| [embedder-cpu.txt](embedder-cpu.txt) | Base include plus CPU Torch and ONNX packages, with Python-version markers. |
| [embedder-gpu.txt](embedder-gpu.txt) | Base include plus the CUDA Torch wheel source and pin. |
| [dev.txt](dev.txt) | Pinned pytest, asyncio test support, Ruff and the YAML parser for workflow contract tests. No serving image consumes these tooling pins. |
| [ci.txt](ci.txt) | Offline-suite environment: dev, middleware and MCP inputs plus the numeric/runtime imports used by embedder tests. It deliberately does not install the full model stack. |
| [security-constraints.txt](security-constraints.txt) | Exact transitive security overrides with rationale; input to `scripts/security_constraints.py`. |
| [locks/](locks/) | Four resolved Linux image environments; consumed as Docker build constraints. |

| Lock | Target environment |
|---|---|
| [middleware-py311-linux.txt](locks/middleware-py311-linux.txt) | Middleware, Python 3.11. |
| [mcp-py311-linux.txt](locks/mcp-py311-linux.txt) | MCP server, Python 3.11. |
| [embedder-cpu-py311-linux.txt](locks/embedder-cpu-py311-linux.txt) | CPU embedder, Python 3.11. |
| [embedder-gpu-py310-linux.txt](locks/embedder-gpu-py310-linux.txt) | GPU embedder, Python 3.10. |

## Interface reference

The root [Dockerfile](../Dockerfile) is the authority for which requirement and
lock each target installs. Requirement `-r` includes, wheel-index options and
environment markers are functional inputs; preserve them deliberately. Python
lint/test policy is in [pyproject.toml](../pyproject.toml). Frontend dependencies
belong to [frontend/package.json](../middleware-qdrant-manager/frontend/package.json),
and model revisions to
[model_revisions.py](../server-embedder-reranker/model_revisions.py).

## Behaviour contracts

Exact package versions improve reproducibility but do not pin wheel bytes:
these locks contain no artifact hashes. They also do not certify absence of
vulnerabilities. OSV auditing is a separate, network-dependent check.

Shared direct-pin changes can affect several targets. `ci.txt` explicitly
repeats the limited embedder imports it needs; review those when the corresponding
CPU/base pins change. Passing fake-backed tests does not prove model loading,
native ABI compatibility or memory behaviour in a rebuilt image.

## Recipes

For a dependency update, identify its direct/shared/transitive owner and affected
targets, change the human-maintained input, follow the synchronization/rebuild
procedure in [scripts/README.md](../scripts/README.md#auditing-dependencies),
and freeze each verified image back into its matching lock. Review unrelated
resolution changes before accepting that freeze. Keep the input and resulting
locks together in the eventual change.

Build/start/stop commands remain documented only in the
[launcher reference](../scripts/README.md). Do not overwrite Linux locks with
`pip freeze` output from a Windows host or a generic development environment.

## Verification

Read-only local contract checks, from the repository root:

```powershell
py scripts/security_constraints.py
py -m pytest scripts/test_security_constraints.py scripts/test_verify_dependencies.py -q
```

For changed pins, also rebuild the affected images, run their focused service
tests and applicable CPU/GPU smoke checks, then run the network-dependent audit
described in the launcher reference. An unreachable OSV service is not a clean
audit. No dependency installation or model download is needed merely to review
this README.

## Failure modes

| Symptom | Inspect first |
|---|---|
| Direct requirement conflicts with a lock | Input and frozen target disagree; follow the update procedure instead of adding another `-c`. |
| Linux-only package fails on a Windows host | Wrong validation environment; use the target image and the platform-independent audit. |
| CI passes but model startup fails | CI intentionally omits much of the model stack; inspect rebuilt-image smoke results. |
| Security override checker reports drift | Which target locks contain the affected package and whether they match the override. |

## Invariants

Keep CPU and CUDA targets distinct. Preserve Python markers and exact pins;
never replace reviewed pins with unbounded ranges to make a build pass. Keep
credentials and private index tokens out of requirement files. Never claim a
lock is validated without identifying the image from which it was frozen.

## Extending it

For a new image target, define its direct input, Docker installation path and
resolved lock together, then review verifier discovery, CI and SBOM coverage.
For a new security override, document the rationale and use the existing
synchronizer rather than hand-maintaining two conflicting constraint layers.

## Related documents

- [Repository change-risk rules](../AGENTS.md)
- [Automation and release evidence](../.github/AGENTS.md)
- [Middleware](../middleware-qdrant-manager/README.md)
- [Embedder](../server-embedder-reranker/README.md)
- [MCP server](../smart-qdrant-mcp-server/README.md)
