# Release candidate gate

This is the operator acceptance gate for a release of the complete stack. Keep
it alongside CI: hosted checks cannot establish real CPU/GPU model behaviour,
profile migration, destructive browser journeys, skill-guided MCP usage or
restoration of an operator's data and configuration. This file defines the
required evidence and decision; the linked service guides own execution details.

Passing this checklist does not itself publish, tag, push or deploy anything.
Those actions require the repository owner's release instruction.

## Evidence and decision contract

Create an ignored `release-evidence/<UTC timestamp>-<full commit SHA>/` bundle.
Before execution, create `00-scope.json` with the candidate SHA, required cases
and variants, hardware/profile cells, initial stack state, allowed collection
names, expected evidence files and operator. Define the complete-stack scope
before testing; do not narrow it afterward merely to hide a failure.

For each check, save sanitized stdout/stderr or tool/browser traces, the command
or procedure, UTC time, exit status, expected outcome, actual outcome and image/
profile identity. Use the artifact locations below and record individual file
paths in the scope ledger. Never include tokens, a local `.env`, confidential
collection contents or credentials embedded in command lines.

Keep outcomes distinct:

| Outcome | Release interpretation |
|---|---|
| PASS | The required assertion and its independent postconditions passed on the recorded candidate and layer. |
| FAIL | An expected assertion failed; investigate the defect or invalid test setup. A green runner with expected failures is not an all-PASS campaign. |
| BLOCKED / NOT RUN / SKIPPED | Required evidence is missing; not proof of an application defect, but insufficient for release acceptance. |
| PARTIAL | Only some required variants/layers ran; the full case is not accepted. |
| Not applicable | Written contract-based rationale established in the scope ledger; missing hardware or tools is not inapplicability. |

Expected negative responses can pass a refusal/recovery test when the exact
contract and absence of unintended effects are verified. A refused benchmark
load remains REFUSED, not a successful throughput measurement. A safe autotuner
rollback proves fallback behaviour, not completion of an unmeasured canary.

Stop the affected campaign and withhold acceptance if candidate identity drifts,
a required assertion fails, a prerequisite or evidence file is missing, fixture
reset fails, unexpected mutations occur, or secrets enter an artifact. Preserve
safe diagnostics, perform authorized cleanup/restoration, and record the blocker.
A missing GPU, service, token or network connection leaves the relevant check
BLOCKED; never substitute a mock or an old run silently.

## Required steps

1. **Freeze identity and scope.** Require a clean candidate worktree; save the
   full SHA, timestamp, operator, tool versions and initial runtime state in
   `01-candidate.txt`. Record the scope ledger before running. CPU and GPU have
   distinct image IDs; associate every result with the correct target rather
   than demanding one image digest across all four model/hardware cells.

2. **Verify the private reporting channel.** Record the repository owner's
   designated private vulnerability-reporting route and its actual availability
   in `02-private-reporting.txt`, without submitting a dummy vulnerability.
   Check repository settings as needed with an authorized administrator.
   Use the candidate's published reporting policy when present. If a policy
   file such as `SECURITY.md` is absent, do not follow a nonexistent link or
   treat a previous policy as current. A missing documented/working route is
   an administrative prerequisite; this checklist neither creates policy nor
   enables settings.

3. **Require CI for the exact SHA.** Save workflow URLs, conclusions and SHA in
   `03-ci.txt`. Require the aggregate `CI` and all its dependencies to succeed.
   Dispatch `Release check` for that same candidate when authorized and preserve
   its downloaded artifact unchanged under `03-workflow/`. A historical green
   run or an artifact from another SHA cannot be credited.

4. **Close dependency and model provenance.** Follow the
   [dependency update/audit procedure](../scripts/README.md#auditing-dependencies).
   Require the read-only security-constraint check and OSV audit to pass; retain
   `04-security-constraints.log`, `04-dependency-audit.log`,
   `04-dependency-audit.json` and the four SBOM files under `04-sbom/`.
   Verifier exits 1 (vulnerable), 2 (error) and 3 (OSV unavailable) all block
   acceptance. Record immutable model revisions and relevant runtime/profile
   fingerprints. Resolve dependency drift before freezing a new candidate;
   do not rewrite locks halfway through acceptance and keep the old SHA label.

5. **Build and identify images.** Use the supported
   [launchers](../scripts/README.md) to build the middleware, MCP, CPU embedder
   and GPU embedder targets. Verify each consumes its matching single target
   lock. Save image IDs, source identity, build outcomes and RepoDigests when
   available in `05-image-digests.txt`. A locally built image need not have a
   registry RepoDigest; record that fact instead of inventing one. Rebuild the
   middleware's baked console assets before their live acceptance run. If an
   image changes later, invalidate the affected evidence and reverify it.

6. **Require the offline and intercepted checks.** Use the current
   [CI workflow](workflows/ci.yml) and
   [repository verification contract](../AGENTS.md), not a fixed historical test
   count. Cover Python 3.10/3.11 service tests, the launcher suite, Ruff,
   shellcheck, PSScriptAnalyzer, Dockerfile and Compose checks, the complete
   Node test set (including coverage/harness tests), and CI's intercepted
   browser suites. Follow the [frontend prerequisites](../middleware-qdrant-manager/frontend/tests/README.md)
   before running Node tests that collect Playwright inventory. Store results
   or exact-SHA CI evidence references under `06-offline/`; keep actual passes,
   skips, retries and expected failures distinct.

7. **Validate deployment shapes.** Verify CPU/GPU Compose resolution, pairwise
   disjoint authentication tokens, loopback publish rules, read-only serving
   filesystems and preparation/serving volume boundaries. Exercise the planner
   combinations for MCP enabled/disabled, embedder-only, and public/private
   embedder access. Save sanitized plans and results in `07-compose/`.
   Configuration resolution alone does not establish live network reachability;
   the final isolation checks below cover that separately.

8. **Verify CPU/BASIC.** Start through the launcher with hardened authentication
   and the embedder port private. Check readiness, normal ingest/query/rerank,
   stored-data postconditions and applicable MCP translation against owned
   fixtures. Record the private-network access used for checks that need the
   embedder itself; do not expose it just to simplify a test. Save
   `08-cpu-basic/`, including active model/runtime and admission/recovery results.

9. **Verify CPU/ADVANCED.** Repeat under the ADVANCED profile and record its
   no-ColBERT capability, model identities, selected runtime and readable
   prepared artifacts in `09-cpu-advanced/`. For autotuning/runtime changes,
   require the relevant long-input canary and quality gates to complete within
   the recorded calibration budget. Inspect promotion, timeout, memory gates
   and covered backend/token-length combinations explicitly. ONNX preparation
   alone does not prove serving can read its cache on a read-only mount.

10. **Verify GPU/BASIC and GPU/ADVANCED.** Record separate results under
    `10-gpu-basic/` and `10-gpu-advanced/`: GPU identity, active profile, tuner
    fingerprint/decision, readiness, ingest/query/rerank and memory evidence.
    Follow the testbook's four-cell and performance matrix for all four cells;
    retain refusal and recovery outcomes without increasing conservative
    admission limits to manufacture accepted workloads. A missing GPU leaves
    the complete-stack release matrix incomplete. Historical benchmarks are
    context, not proof for this candidate.

11. **Execute browser acceptance with owned fixtures.** Follow the
    [frontend test guide](../middleware-qdrant-manager/frontend/tests/README.md)
    and [testbook](../middleware-qdrant-manager/frontend/tests/testbook/test-book-e2e.md).
    Initialize a fresh campaign manifest, preserve the initial collection
    inventory, seed/reset fixtures and prove exact ownership before writes.
    An `e2e_` prefix alone does not authorize adopting an existing collection.
    Save setup under `11-playwright-reset/` and results under `11-playwright/`.
    Run required default/live cases with pacing; archive each result before
    another run overwrites it. Every rerun needs the documented reset first.
    Include external profile roundtrips, cross-profile reindex/query, recovery
    and controlled outage/proxy cases required by the scope. A path passed to
    the default config does not enable externally excluded specs. Coordinate
    this step with the hardware starts above rather than running competing
    stacks or changing profile during another campaign. Include the specified
    restart/persistence and durable reindex-checkpoint assertions; a readback
    before a restart alone does not prove recovery afterward.

12. **Execute skill-guided MCP acceptance.** Follow
    [testbook §0.8](../middleware-qdrant-manager/frontend/tests/testbook/test-book-e2e.md#agent-workflows)
    for each applicable workflow. An AI agent must load the repository skill,
    use its relevant playbook and call the real MCP surface. Save skill identity,
    task, sanitized calls/results, independent postconditions and cleanup under
    `12-agent/`. Browser/REST passes and scripted MCP calls do not prove the
    skill-guided layer. Use explicit campaign-owned targets and check ownership
    before every mutation: browser interception does not guard MCP calls.
    Do not run the separate fixed-name `mcp_eval` seed as if it were an owned
    console fixture. Missing agent tooling remains BLOCKED or NOT RUN.

13. **Verify excluded-service shapes last.** Exercise
    `MCP_SERVER_ENABLED=false`, `EMBEDDER_ONLY=true`, and
    `EXPOSE_SMART_EMBEDDER_PORT=false` independently through the launcher.
    Save each scenario under `13-exclusions/`, proving the expected container
    and publish state as well as the health of services that should remain.
    A private embedder still serves the middleware internally; embedder-only
    does not require a middleware that is deliberately absent. Keep failures
    isolated to disposable fixtures and follow the documented outage procedure.

14. **Restore and account for data.** Settle only campaign-owned pending jobs,
    remove owned temporary collections/files, compare the final collection
    inventory with the protected initial inventory, and check for leaked
    reservations or unexpected errors. Restore the initial hardware/profile,
    service-enable and exposure configuration (or the explicitly requested
    stopped state). Save before/after inventories, cleanup outcomes and final
    readiness in `14-restoration/`. Never delete other campaigns' collections
    merely because they share a test prefix. Failed cleanup is recorded and
    prevents a claim that acceptance left the environment intact.

## Automation boundary and final decision

The [Release check workflow](workflows/release-check.yml) produces its own
artifact layout: `metadata.txt`, check logs, `dependency-audit.json`, `sbom/`
and `manifest.json`, retained for 90 days. It does **not** use the numbered
operator bundle layout above. Preserve the download and reference its files
from the scope ledger rather than pretending both layouts are identical.

That workflow runs Python 3.11 and only the frontend model test file; the
broader Node/intercepted-browser coverage and Python version matrix come from
regular CI. It also calls OSV, so its "offline" job label does not mean no
network. Neither workflow performs the complete live acceptance campaign.

Create `15-decision.json` with the candidate SHA, required-case/layer outcomes,
blockers, evidence references, completion time and the operator's decision.
Check every required artifact for meaningful content and a matching result.
Then run [scripts/release_evidence.py](../scripts/release_evidence.py) over the
bundle, explicitly supplying `--commit <full candidate SHA>` and a repeated
`--require <relative file>` for every expected **file** in the scope ledger,
including the scope and decision files. Directories are not valid `--require`
entries. Keep the workflow's original manifest with its downloaded artifact.

The generated `status: complete` means only that the supplied required files
exist. The tool does not discover the local Git SHA (its default outside CI is
`unknown`), check test exit statuses, reject empty logs or validate acceptance
coverage. Its hashes identify the recorded files; they cannot turn a failed or
missing assertion into PASS. Require both file completeness and an affirmative
review of all required outcomes before calling the candidate accepted. Preserve
incomplete bundles too, and record follow-up runs separately when a fix changes
the candidate.

## Maintenance

Update this checklist when CI, deployment shapes, acceptance layers or evidence
contracts change. Keep commands and detailed oracles in their owning guides.
Do not hard-code growing test counts or transfer a previous campaign's PASS to
a new SHA. Packaging optimizations such as reducing training-only transitive
dependencies belong to a separately scoped change, not an implicit waiver of
security auditing for the current candidate.
