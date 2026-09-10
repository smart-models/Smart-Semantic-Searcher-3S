# Repository automation and release evidence

## Purpose and boundaries

This directory defines GitHub Actions verification and the operator release
checklist. It does not implement application behaviour or start the local stack.
An agent changing it should first identify whether the change belongs to normal
CI, scheduled dependency auditing, or release acceptance: those have different
triggers, permissions and evidence requirements.

## Mental model

Pull requests and pushes to `main` run the regular CI workflow. Its aggregate
`CI` job depends on every required job and fails for failures, cancellations or
skips. Dependency auditing runs separately because it queries an external
vulnerability database. Release checking creates an evidence bundle for a
specific candidate; the operator checklist also requires live checks that the
hosted workflow cannot establish.

## Map of the code

| File | Responsibility | Read/change it when |
|---|---|---|
| [workflows/ci.yml](workflows/ci.yml) | Python lint and service tests, frontend model/coverage and intercepted browser tests, launcher tests, Dockerfile and Compose checks, aggregate CI result. | Adding a required suite or changing supported test environments. |
| [workflows/dependency-audit.yml](workflows/dependency-audit.yml) | Scheduled/manual OSV audit, SBOM artifacts and a deduplicated vulnerability issue. | Changing audit scheduling, evidence publication or issue lifecycle. |
| [workflows/release-check.yml](workflows/release-check.yml) | Manually dispatched candidate metadata, checks, dependency audit and hashed evidence artifact. | Changing the automated subset of release acceptance. |
| [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) | Operator release steps, required artifacts and abort conditions. | Assessing whether a candidate has sufficient evidence to release. |

The workflows orchestrate existing owners rather than reimplementing them:
[scripts](../scripts/README.md) owns launcher/security/audit/evidence logic;
each service owns its tests. The [frontend test guide](../middleware-qdrant-manager/frontend/tests/README.md)
distinguishes intercepted, destructive live and external campaigns.

## Interface reference

The YAML `on`, `permissions`, `jobs`, `needs` and artifact steps are the exact
automation contract. Read them before editing a trigger or a job dependency.
The dependency audit can create, update and close issues and therefore has
`issues: write`; CI and release checking use read-only repository permissions.
Dispatching or changing a workflow is separate from modifying these files.

## Behaviour contracts

- A collected browser title is inventory, not an executed acceptance result.
- Regular CI runs selected intercepted browser suites; it does not certify
  real Qdrant mutations, model-profile switches, MCP/skill agent journeys or
  the hardware performance matrix.
- Release checking includes a network-dependent dependency audit despite its
  job's "offline" label. An unavailable OSV service is not a clean audit.
- Release manifests prove artifact presence and integrity, not the correctness
  of every assertion inside an artifact. Follow the operator checklist too.

## Verification

Review YAML syntax, triggers, job dependencies, working directories and artifact
paths together. For shared automation changes, run the focused existing tests
from the repository root, for example:

```powershell
py -m pytest scripts/test_release_evidence.py scripts/test_render_dependency_audit_issue.py scripts/test_security_constraints.py -q
```

Run the relevant service or launcher suite when its commands change. Local
checks do not establish that a workflow ran successfully on GitHub; retain the
actual workflow result for the candidate SHA when making that claim.

## Failure modes

| Symptom | Inspect first |
|---|---|
| Individual jobs pass but aggregate CI fails | Required jobs that were skipped or cancelled, and the aggregate `needs` list. |
| Audit failed without a vulnerability report | OSV/error exit classification in the verifier and the saved audit artifact. |
| Release workflow is green but release evidence is incomplete | Live/operator requirements in the checklist. |
| New suite never runs | Trigger, job condition, path/working directory and aggregate dependency wiring. |

## Invariants

Keep third-party actions pinned to full commit SHAs, minimize token permissions,
and never upload credentials or local environment files. Add required jobs to
the aggregate CI dependencies. Keep issue mutation confined to the workflow
that needs it; a documentation edit does not authorize dispatch or publication.

## Extending it

Put reusable validation in `scripts/` or the owning service and call it from the
workflow. When adding evidence, update the producer, required manifest entries
and release checklist together. Explain which parts require real hardware or
external services rather than silently treating them as covered by CI.

## Related documents

- [Repository agent agreement](../AGENTS.md)
- [Launcher and verifier reference](../scripts/README.md)
- [Dependency inputs and locks](../requirements/README.md)
