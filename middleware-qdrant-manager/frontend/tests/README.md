# frontend/tests — console E2E suite

> **Read this when** you need to run, extend or debug the Playwright suite that
> drives the middleware's static console, or you are about to change console
> behaviour and need to know what already covers it.
> **Audience** — an AI agent working in this repository, and any human reading
> over its shoulder.
> **Source of truth** — this file for how to run the suite and what it costs;
> [`testbook/test-book-e2e.md`](testbook/test-book-e2e.md) is the single test book
> for cases, fixtures, acceptance matrices and expected results; the spec files
> show what is actually asserted.
> **Prerequisite** — [`../../README.md`](../../README.md) for the API the
> console calls, and [`../../../scripts/README.md`](../../../scripts/README.md)
> for starting the stack.

---

## Purpose and boundaries

Playwright specs that drive the console at `http://127.0.0.1:6336`, combining
live-stack checks, mocked API responses and collection/execution reporting. Most
series tests carry IDs from [`testbook/test-book-e2e.md`](testbook/test-book-e2e.md)
(`Q5`, `NF-B7`, `RE4`, …); legacy compact specs and filter/monitoring tests use
additional names. The book's §16 records the coverage review; §16.7 is the
revision-7 implementation/execution record and the single source of truth for
what is actually green. The coverage gate
(`coverage-gate.mjs` over `testbook/coverage-contract.json`) enforces
zero-unassigned cases; a collected test is not evidence that its documented
case passed until the gate credits it. The book integrates concrete acceptance
cases and reproducible fixture definitions for the reviewed gaps into each
feature section. Its 96 `TB-` cases are now backed by executable `tb-*` specs
(intercepted + live), declared backend/embedder owners, or written exclusions;
use the shared procedures and per-layer/status rules when extending the suite.
The campaign reporter retains every qualified book case, including unmapped
and unexecuted cases; NF-COV2 checks that the ledger loses no case.

**What this suite is:**

- End-to-end coverage of selected console/API operations with a live Qdrant,
  plus browser regressions whose API responses are simulated.

**What this suite is not:**

- The `*.test.mjs` files test models, ownership, setup, coverage and telemetry
  offline under `node --test`; the `.spec.js` files run under Playwright.
- The full run needs a disposable stack, Qdrant, an embedder, seeded fixtures
  and a token. Mocking one response does not make a whole spec hermetic.
- NF-COV1/NF-COV2 need no stack. X6 needs console assets and a browser,
  but mocks all API responses and performs no Qdrant writes.

## ⚠️ These tests write to Qdrant

They create, fill, recreate and delete sandbox collections. Live writes require
`QHM_E2E_RUN_ID` and a manifest created by `npm run init:run`. Logical fixture
names become `e2e_<run-id>_<suffix>`. Registration refuses an existing target;
it never adopts a collection by prefix. API helpers and the browser transport
guard refuse unowned writes before dispatch, including environment overrides.
Seed/reset tools use the same manifest and fail at the first unexpected status.
Keep the original manifest when resuming; never edit it to claim existing data.

## Mental model

Five facts explain nearly every confusing failure in this suite.

1. **Destructive series consume their fixtures.** A second consecutive full run
   without a reset fails on the create/recreate and delete series, because the
   collections they expect to be absent already exist. Reset before every full
   run.
2. **One stack, one suite.** Never run two instances at once: they share one
   Qdrant and one embedder, and the second recreates collections under the
   first one's feet. `workers: 1` only serialises *within* a run.
3. **Rate limiting can exhaust a campaign's request allowance.** Setup helpers
   fail on the actual HTTP status, and per-test telemetry records unexpected
   429 responses. Use the documented cooldown and pacing before attributing
   a failed assertion to an application defect.
4. **A pending regeneration locks its collection.** This is product behaviour,
   not a bug, and it blocks every later run until the job is cancelled.
5. **Three spec files are excluded from the default run** because they need
   process-level or cross-profile state the runner cannot create by itself.

## Map of the directory

```text
tests/
├── helpers.js            boot(), selectActive(), openTab(), apiFetch() — shared setup
├── fixtures.js           shared Playwright fixture (also applies the inter-case pacing)
├── d-helpers.js          filter-row builders for the payload-filter matrix
├── tb-live-helpers.js   api(), bootPaced(), reserve(), fresh(), snapshot() — live campaign helpers (pacing + ownership)
├── *.spec.js             one file per test-book series (see the table below)
├── *.test.mjs            pure-function unit tests and the coverage gate (node --test, not Playwright)
├── coverage-gate.mjs     zero-unassigned gate over testbook/coverage-contract.json
├── case-coverage.mjs     qualified case-key parser and ledger
├── campaign-reporter.mjs per-run collected-title/case/telemetry reporter
├── fixtures/             ingest input files
├── testbook/             the human-readable test book + coverage-contract.json
└── tools/                seed and reset scripts (seed-book.mjs, seed-reg.mjs, …)
```

The revision-7 `tb-*` specs provide executable coverage for the book's `TB-*`
cases. They come in two layers:

| Spec | Layer | Book coverage |
|---|---|---|
| `tb-ui.spec.js` | UI-mock (intercepted) | TB-UI-001–009, TB-PRO-002 truth table, TB-EXP-008/011, TB-ING-003, TB-QRY-005, NF-P20, XSS, lockdown |
| `tb-monitoring.spec.js` | UI-mock | TB-MON-001–007 |
| `tb-filter-validation.spec.js` | UI-mock | TB-FIL-001–016 |
| `tb-filters-live.spec.js` | UI-live | §8.2 A/B/C live filter matrix (183 cases) |
| `tb-payload-ids.spec.js` | UI-live + API-live | TB-PAY-001–006, TB-ID-001–005, TB-ING-001–006 |
| `tb-export-query.spec.js` | UI-live + API-live | TB-EXP-001/002/005/006/007, TB-QRY-001/003/004/006/007/008/009/010, TB-API-003 |
| `tb-reindex-live.spec.js` | UI-live + API-live | TB-REI-001–005, NF-P17, TB-QRY-002 |

The legacy `*.e2e.spec.js` and per-tab series files remain collected alongside
the `tb-*` suite; see the next table for those.

| Spec | Book series |
|---|---|
| `a-collections.spec.js` | §6 R/C/T/PI/X/D — report, create, toggles, payload indexes, export, delete |
| `b-ingest.spec.js` | §7.1 I — ingest preflight and the editable mapping table |
| `b2-ingest-indexes.spec.js` | §7.1b IX — payload indexes created after a successful ingest |
| `c-points.spec.js` | §7.2–7.6 U/B/G/DL/PL — upsert, batch, get, delete, payload |
| `d-query.spec.js` | §8.1 Q — every hybrid-search control combination |
| `d-filters.spec.js` | §8.2 QF — the payload-filter matrix |
| `e-apikey.spec.js` | §3 H — connection bar and API key popover |
| `e-monitoring.spec.js` | §9 M — badges, chart, polling lifecycle, metrics |
| `e-nav.spec.js` | §10 N — tabs, hash routing, keyboard navigation |
| `nf-newfeatures.spec.js` | §13 NF — count, payload-by-filter, order_by, threshold/offset, reference, audit findings |
| `z1-reembed.spec.js` | §5 RE — vector regeneration lifecycle |
| `z2-service.spec.js` | §2 S — readiness and outage gate (network-simulated) |
| `z3-compat-mocked.spec.js` | §4 E — embedding-compatibility lockdown (mocked report) |
| `z4-switch.spec.js` | §4/§12 E/Z — real basic↔advanced profile switch |
| `z5-real-outage.spec.js` | §2 S — outage with the embedder actually stopped |
| `z6-e7.spec.js` | E7 — cross-profile re-embed back to compatible |
| `embedding-compatibility.e2e.spec.js` | compact compatibility-lockdown regression with mocked reports |
| `ingest-preflight.e2e.spec.js` | compact ingest-preflight regression with mocked ingest calls |
| `export-filename.e2e.spec.js` | X6 — real browser downloads with mocked API responses; Unicode filenames and unchanged response bytes |
| `service-readiness.e2e.spec.js` | compact service-gate recovery regression |
| `vector-reindex.e2e.spec.js` | standalone reindex migration regression |

The four older `*.e2e.spec.js` files predate the book-ID harness and remain
collected alongside X6. They are mixed: the standalone vector reindex is live
and destructive; compatibility performs a real create/delete among its mocked
checks; preflight and readiness simulate selected responses but still load the
console and may call other live endpoints.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `QHM_E2E_TOKEN` | _(required)_ | The **token half** of one entry in `MIDDLEWARE_API_TOKENS`. For `MIDDLEWARE_API_TOKENS=k1a2b3:alice`, export `k1a2b3`, **not** `alice`. Exporting the label yields a bare `401` with no hint about the cause. Without it the suite fails at the first collection listing. |
| `QHM_BASE_URL` | `http://127.0.0.1:6336` | One HTTP origin for the runner, helpers, compact specs and setup tools; a trailing slash is normalized. |
| `QHM_E2E_RUN_ID` | _(required for writes)_ | 1–32 lowercase letters, digits or underscores. Selects names and `test-results/ownership-<run-id>.json`. |
| `QHM_E2E_COOLDOWN_SECONDS` | `15` | Cooldown `npm test` waits before Playwright starts, so setup traffic cannot starve the first specs. Set to `0` only when the middleware bucket is already known to be full. |
| `QHM_E2E_PACE_MS` | `250` | Delay the shared fixture leaves between independent cases. Raise to `700` for a full run — see [Failure modes](#failure-modes). |
| `QHM_REINDEX_COLL` | `e2e_z_reg` | Collection `z1-reembed.spec.js` regenerates. |
| `QHM_PROFILE`, `QHM_NATIVE`, `QHM_FOREIGN` | _(unset)_ | Required by `z4-switch.spec.js` only. |
| `QHM_E7_COLL` | `e2e_z_basic` | Target of `z6-e7.spec.js`; must actually be incompatible and born BASIC while the stack runs ADVANCED. The spec expects the resulting profile to be `advanced`. |

## Capabilities and limits

**You can:**

- Run the whole suite, one tab's series, or a single test-book ID.
- List collected tests without launching a stack or a browser
  (`npx playwright test --list`).
- Override pacing, origin and an explicitly owned reindex target.

**You cannot:**

- Run the full suite without a live, healthy, seeded stack; the static guards
  and fully mocked X6 are narrower exceptions.
- Run in parallel, either across workers or across processes.
- Run the three excluded specs with the default configuration, even by path;
  use the explicit configuration override below.
- Run a full suite twice without resetting the fixtures in between.

## Recipes

The `application-state`, `application-reindex`, and `application-accessibility`
specs intercept the page, local assets and every API request. They need no live
stack or token and abort unknown routes. Run these application regressions
directly to bypass the live `pretest` cooldown:

```powershell
npx playwright test tests/application-state.spec.js tests/application-reindex.spec.js tests/application-accessibility.spec.js tests/harness-browser.spec.js --workers=1
```

`application-fixtures.js` owns their response families and controlled replies.
These tests cover asynchronous state, protocol validation and keyboard behavior;
they do not replace live integration against Qdrant and the embedder.

**Prerequisites, once**

1. Stack up and healthy: `pwsh stack.ps1 up gpu` (or `cpu`) from the repo root.
2. Dependencies installed, from this `frontend/` folder:

```bash
npm install
npx playwright install chromium
```

**Run**

```bash
cd middleware-qdrant-manager/frontend

# Full suite
QHM_E2E_TOKEN='<token>' npm test

# Query tab only (hybrid search, payload filters, the four added capabilities)
QHM_E2E_TOKEN='<token>' npm run test:query

# One book ID
QHM_E2E_TOKEN='<token>' npx playwright test -g "NF-X9"

# Collection-only audit (no stack or browser launch)
npx playwright test --list
```

On 2026-09-06 after the revision-7 closure, collection discovers the full
`tb-*` plus legacy suite in the default run, including the intercepted UI
matrix (133), monitoring regressions (25), filter validation, and the
shared-origin browser check. The `tb-*-live.spec.js` suites are collected in
the default run but only pass against a seeded `seed:book` campaign; run
`npm run seed:book` first (see **Seed the `tb-*` live campaign** under
[Recipes](#recipes)). This
is an inventory, not a successful full execution. U6/PL7 assert protected text
behavior; DL10/DL11 assert idempotent absent deletion and actual deletion
counts. The book's §16.7 is the execution record.

The M-G check generates 125 samples with Playwright's clock and verifies the
120-sample window, eviction order and hover/leave behavior without a live
stack. CI also runs it separately with `--grep 'M-G —'`.

**Seed the sandbox**

Most specs expect their fixtures to exist. Choose a fresh run ID, register its
absent targets against the current inventory, then seed. Keep the run ID and
token in the shell environment for all subsequent commands:

```bash
export QHM_E2E_RUN_ID=campaign_20260905
# QHM_E2E_TOKEN must already be set; do not put real tokens in shared logs.
npm run init:run
QHM_E2E_TOKEN='<token>' npm run seed           # e2e_a_* … e2e_z_basic, indexes included
QHM_E2E_TOKEN='<token>' npm run seed:query     # query and payload-filter fixtures
QHM_E2E_TOKEN='<token>' npm run seed:reindex   # 2,000-point collection for the re-embed series
```

**Seed the `tb-*` live campaign (`seed:book`)**

The revision-7 `tb-*-live.spec.js` suites consume a single namespaced fixture
set created by `tools/seed-book.mjs`. Use a fresh run ID (for example
`book0906`) so the owned collections do not collide with the legacy
`e2e_a_*`/`e2e_c_*` fixtures:

```bash
export QHM_E2E_RUN_ID=book0906
npm run init:run
QHM_E2E_TOKEN='<token>' npm run seed:book
```

`seed:book` creates, under the owned run namespace: a 100-point filter fixture,
a 6-point edge fixture, a 3-point write fixture, a 2001-point export fixture,
an empty fixture, a 1001-point sample fixture, an alternate fixture, and two
roundtrip fixtures. The generator is the oracle — every export/ID/payload
assertion in the `tb-*` live specs compares against the seeded values, not a
hand-maintained snapshot. Re-seed only into a fresh run ID; never edit the
manifest to claim existing data.

**Reset before every full run**

```bash
QHM_E2E_TOKEN='<token>' npm run reset:collections   # e2e_a_* (create/recreate/delete series)
QHM_E2E_TOKEN='<token>' npm run reset:points        # e2e_c_* (delete/payload series)
```

**Run the three excluded specs**

`testIgnore` still applies when a file path is passed to Playwright. Use the
tracked `playwright.external.config.js` to collect the three external-state
scenarios; prepare one state at a time before execution.

In PowerShell set `$env:QHM_E2E_RUN_ID = 'campaign_20260905'` before
`npm run init:run`. A second initialization of the same manifest refuses overwrite.
Use another fresh run ID, or resume/reset only the original owned targets.

`test-results/campaign.json` contains actual collected titles, outcomes, qualified
case IDs and per-test console/page/HTTP telemetry. `NOT_MAPPED`, `NOT_RUN`,
`PARTIAL` and `SKIPPED` never become PASS. Explicit `case` annotations map a test
to a qualified key, for example `book:6.2/C1`; `book:8.2/C1` stays separate.
`requireCompleteCoverage` in `case-coverage.mjs` fails on any non-PASS case when
evaluating the final ledger. NF-COV1 is only a static control-reference inventory.
Negative browser fixtures use `fulfillResponse`/`abortResponse`; live negative
tests declare exact method/path/status with `expectHttpFailure`. Unexpected
errors fail the automatic telemetry fixture; no blanket 4xx/5xx exclusion applies.

From `frontend/`, list before executing; this must collect **10 tests in 3
files** without opening a browser or changing services:

```bash
npx playwright test --config playwright.external.config.js --list
```

Run **one file per prepared state**, not all three together. Set `QHM_E2E_TOKEN`
in the environment as for the default run.

| File | Required external state | Invocation from `frontend/` (Bash syntax) |
|---|---|---|
| `z4-switch.spec.js` | stack already switched; `QHM_NATIVE` compatible now, `QHM_FOREIGN` incompatible now; reverse their roles on switching back | `QHM_PROFILE=advanced QHM_NATIVE=<born-advanced> QHM_FOREIGN=<born-basic> npx playwright test --config playwright.external.config.js tests/z4-switch.spec.js` |
| `z6-e7.spec.js` | stack ADVANCED, disposable BASIC-born incompatible target | `QHM_E7_COLL=<e2e_collection> npx playwright test --config playwright.external.config.js tests/z6-e7.spec.js` |
| `z5-real-outage.spec.js` | embedder stopped while middleware and Qdrant stay running | `npx playwright test --config playwright.external.config.js tests/z5-real-outage.spec.js` |

In PowerShell, set variables separately (`$env:QHM_PROFILE = 'advanced'`, etc.)
then run the same `npx` command. Profile switching follows
[`scripts/README.md`](../../../scripts/README.md). After a profile switch,
wait for middleware `/readyz` to return 200 and `/health` to report the intended
`embedder_profile` before seeding or running tests. Container health can recover
before the middleware's cached readiness does; a transient 503 is a failed
setup prerequisite, not an accepted test outcome. For the dedicated outage
scenario, stop only the active `embedder-server-gpu` or `embedder-server-cpu`
container, run z5, then restart that same container in a `finally`/trap cleanup
and wait for healthy status. These are operator setup actions; z5 performs none
of them and covers neither a real Qdrant outage nor automatic restart.

`z1-reembed.spec.js` *is* part of the default run: it performs a real vector
regeneration on `QHM_REINDEX_COLL` (default `e2e_z_reg`, created by
`tools/seed-reg.mjs`). Point it at a few-hundred-point collection — on a small
one the job finishes before the pause/resume window can be observed, and the
combined lifecycle case fails. RE6 can instead skip a missed cancel window and
also accepts natural completion; its PASS alone does not prove cancellation.

**Run the pure-function tests and the coverage gate**

```bash
node --test tests/query-models.test.mjs          # pure model tests
node --test tests/coverage-gate.test.mjs         # gate unit tests
node tests/coverage-gate.mjs                     # the gate itself
```

The gate parses `testbook/test-book-e2e.md` into qualified case keys, credits
collected browser titles (default and external configs), executed Node TAP
titles, and declared backend/embedder owners, rejects unknown contract IDs and
duplicate owners, requires written exclusion reasons of at least 40
characters, marks an exclusion stale when executable evidence later appears,
and fails when unassigned cases exceed `maxUnassigned` in
`testbook/coverage-contract.json`. Revision 7 sets `maxUnassigned: 0`. The
status categories are `MAPPED`, `EXTERNAL`, `NODE`, `OWNED`, `EXCLUDED` and
`UNASSIGNED`; the gate prints a per-category count and the unassigned list.

**Run just the static coverage guards (no stack or browser)**

```bash
npx playwright test -g NF-COV --output ../../docs/e2e-static-results
```

## Verification

A change to the console is verified when:

1. `npm test` passes from a freshly seeded and reset sandbox.
2. Any new case has a matching entry in
   [`testbook/test-book-e2e.md`](testbook/test-book-e2e.md), and the test name
   carries that book ID.
3. The middleware's own suite still passes:
   `py -m pytest middleware-qdrant-manager/tests -q`.

The console is served from `frontend/dist/` **baked into the middleware image**
— rebuild and restart the middleware before running the suite, or you will test
stale assets.

## Failure modes

**A case fails on a count of `0`, and passes when its file runs alone**
Rate limiting. Against the shipped
`MIDDLEWARE_RATE_LIMIT_REQUESTS_PER_MINUTE=300`, request-heavy stretches (the
NF payload series, the Points delete/payload block) can exhaust the bucket, and
250 ms of pacing is not always enough. Re-run with `QHM_E2E_PACE_MS=700` before
looking for an application bug. The 2026-08-17 run used 700 ms.

The console shell, static assets, API schema and monitoring probes are
deliberately rate-limit exempt — loading a browser page must not consume the
same per-client bucket as the protected API operations it initiates. All
collection, point, query, ingest and reindex endpoints remain rate limited and
bearer protected. The seed/reset tools and Playwright appear to the middleware
as the *same* client through Docker's loopback-published port, so run setup to
completion before starting the suite and do not run other API-heavy jobs at the
same time.

If a deployment intentionally uses limits lower than the shipped defaults,
raise them only in a dedicated E2E stack; do not weaken the
production/private-stack defaults to accommodate tests.

**Every run fails on the same collection, including the seed script**
A re-embed job is pending on it. While a job is `running`, `paused`,
`interrupted`, `finalizing` or `failed`, the collection is locked:

```
POST /collections            (recreate) → 409 "Collection vectors are being regenerated;
                                              mutations are temporarily blocked."
POST .../reindex/preflight             → can_start: false,
                                         "A reindex job is already paused for this collection."
```

`tools/seed-reg.mjs` cancels a pending job before rebuilding and exits non-zero
if the recreate is refused, and the re-embed spec clears one before each case
that starts a job. To clear a stuck collection by hand:

```bash
curl -X POST -H "Authorization: Bearer $QHM_E2E_TOKEN" \
  http://127.0.0.1:6336/collections/<name>/reindex/cancel
```

**A bare `401` with no explanation**
`QHM_E2E_TOKEN` holds the label instead of the token. See
[Configuration](#configuration).

**The create/recreate or delete series fails on the second run**
Fixtures were not reset. See [Recipes](#recipes).

## Invariants

- **Only exact targets in the campaign manifest may be mutated.** A matching
  `e2e_` prefix never authorizes adopting an existing collection. The standalone
  reindex test and every override use the same ownership guard.
- **`workers: 1`.** The suite is not parallel-safe; do not raise it.
- **New spec names carry their test-book ID.** Legacy names and shared IDs need
  qualified mappings. Use `case-partial` annotations for checks that cover only
  part of a case; they remain PARTIAL even when their browser assertion passes.
- **Never weaken shipped middleware defaults to make a test pass.** Adjust the
  pacing, or use a dedicated E2E stack.

## Extending it

- **New console capability** — add the case to
  [`testbook/test-book-e2e.md`](testbook/test-book-e2e.md) first, then a spec
  named with the new book ID, in the series file that owns that tab.
- **Complete a reviewed gap** — implement its uniquely keyed `TB-` cases and
  variants in the [test book](testbook/test-book-e2e.md#closure).
  Keep legacy IDs section-qualified in the mapping; source references alone
  do not satisfy a case's behavioural oracle.
- **New fixture data** — ingest input files live in `tests/fixtures/`; the
  seeding scripts that build collections live in `tests/tools/`.
- **New shared step** — extend `helpers.js` (page-level) or `fixtures.js`
  (per-case setup and pacing) rather than copying setup between specs.
- **A scenario needing process-level state** — keep it out of the default run
  and document an explicit runner configuration and prerequisites; test its
  `--list` invocation so `testIgnore` cannot silently hide it.

## Related documents

| Document | Covers |
|---|---|
| [`testbook/test-book-e2e.md`](testbook/test-book-e2e.md) | The single test book: cases, IDs, deterministic data, acceptance matrices and gap-to-case mapping; §16.7 is the revision-7 execution record. |
| [`../../README.md`](../../README.md) | The middleware API the console drives, and its rate limits and endpoints. |
| [`../../../scripts/README.md`](../../../scripts/README.md) | Starting, stopping and inspecting the stack these tests need. |
| [`../../../README.md`](../../../README.md) | The stack as a whole. |

The coverage map is an inventory, not an execution certificate. Its
`browserCoverage` distinguishes full and partial declarations; `OWNED` names
an independently checked suite without importing that suite's result. The
zero-unassigned policy is pinned at zero. A passing inventory gate cannot
close an expected failure, a partial scenario or an unexecuted operator cell.
