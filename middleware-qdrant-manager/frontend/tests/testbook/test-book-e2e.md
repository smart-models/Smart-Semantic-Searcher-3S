# Comprehensive E2E Test Book — Smart Semantic Searcher (3S) web console

Console served by the manager at `http://127.0.0.1:6336`, exercised through
browser actions, agent-driven MCP/skill workflows, direct API checks, and
explicitly simulated responses.
Review date: 2026-09-06 · Revision 7 with review and agent-workflow addenda

> **Executable form:** [`../`](../) — the series specs usually carry book IDs;
> legacy compact specs and the filter/monitoring series use additional names.
> Prerequisites and seed order in [`../README.md`](../README.md);
> launched with `QHM_E2E_TOKEN=<token> npm test` from `frontend/`.
>
> **Last recorded full seeded run:** 2026-08-17, §15.1. It is historical evidence,
> not a PASS result for the current revision. The older real profile roundtrip
> is recorded separately in §15.1/§15.2; the default suite does not switch profiles.
>
> **Collection audit (2026-09-05):** **359 tests in 18 files** in the default run
> (`npx playwright test --list`). The three excluded files require a configuration
> override as well as their external prerequisites; a path alone does not bypass
> `testIgnore` (§15.5).
>
> **Single acceptance reference:** all existing series and 96 additional TB
> cases are integrated here, under their relevant feature sections. The
> [shared procedures](#status), [deterministic fixtures](#fixtures) and
> [gap-to-case map](#closure) define preparation, actions, expected results and
> evidence for every known §16 gap.
>
> **Executable coverage remains partial.** Completing the specification does
> not implement or execute its tests. NF-COV1/NF-COV2 do not prove behavioural
> coverage. Current contracts and acceptance targets blocked by open product
> defects are explicitly distinguished below; historical PASS results
> do not apply to the new cases.

This book inventories **all tabs + subtabs**, selected control combinations,
cross-cutting behaviors (busy/stale/auto-refresh), keyboard/hash navigation,
and the **Smart Embedder `basic` to `advanced` switch and back**, including lockdowns,
and the **new Qdrant features** (count, payload-by-filter, `order_by`,
`score_threshold`/`offset`, facet, `update_mode`, `with_vectors`).

**Replaces and updates:**
`test-book-collections.md` · `test-book-points.md` · `test-book-query-filters.md` ·
`test-book-monitoring.md` · `test-book-vector-reindex.md` ·
`test-book-e2e-frontend-full.md` · `test-book-e2e-frontend-full-RESULTS.md`

### Reading and execution order

Use the suite README for launch/reset commands, then follow this book's
shared procedures and the cases for the feature under test. Detailed TB cases
extend the existing series; they do not replace their assertions or inherit
their historical results.

| Subject | Detailed acceptance section |
|---|---|
| Setup and scope | [Status and common procedures](#status), [fixtures](#fixtures), [harness guards](#harness) |
| Collections | [Reindex lifecycle](#reindex), [export and re-import](#export) |
| Points | [Payload ownership, IDs and ingest](#payload) |
| Query | [Operator/type matrix and validation](#filters), [query controls, count and facet](#query) |
| Monitoring | [Metric values, chart samples and readiness](#monitoring) |
| Shared UI behaviour | [Operation/state and accessibility matrices](#crosscutting) |
| External state | [Profile roundtrip and outages](#profiles) |
| Agent workflows | [MCP and client-side skill acceptance](#agent-workflows) |
| Backend-only observability | [Forwarding, lifecycle and resource-budget acceptance](#api) |
| Run record | [Gap-to-case map, implementation order and evidence](#closure) |

---

## 0. Isolation and sandbox

- **UI** is the default interaction layer. **[API]** marks a direct contract
  check; API setup, cleanup and postcondition reads also occur inside UI tests.
  **[MOCK]** identifies simulated responses, which prove frontend behaviour
  without proving the corresponding backend operation.
- Required sandbox prefix: **`e2e_`**. Live mutations also require exact
  ownership in the campaign manifest; a prefix alone does not grant ownership.
  `vector-reindex.e2e.spec.js` uses `fixtureName("e2e_vector_reindex")`, yielding
  `e2e_<run-id>_vector_reindex` for a live campaign. Shared guards in `harness.mjs`
  validate mutation targets, including environment-selected collections.
- **Protected (NEVER destructive operations):** `cat`, `my-collection`, `test`,
  any `filters_audit_*`, `collections_audit_*`, `points_audit_*`.
- At end of run, sandbox collections remain for manual inspection.

### 0.1 Collection sandbox

Fixture names usually identify their owner (`e2e_a_*` for COLLECTIONS,
`e2e_b_*` for INGEST). Several series share read fixtures, including
`e2e_d_query`; the suite is not parallel-safe. The table distinguishes seed
scripts from fixtures created by the specs themselves.

| Collection | Created by | Use | Seed |
|---|---|---|---|
| `e2e_a_keep` | `npm run seed` | report, schema toggles, payload indexes (R/T/PI) | 10 points, indexes |
| `e2e_a_victim` | `npm run seed` | recreate / delete (series C/D) | 5 points |
| `e2e_a_empty` | `npm run seed` | export on empty collection (X3) | 0 points |
| `e2e_a_new1..4` | the UI itself, during series C | create toggles (series C) | empty |
| `e2e_a_new5` | D5, via API | separate post-delete victim | empty |
| `e2e_b_ingest`, `e2e_b_alt`, `e2e_b_null`, `e2e_b_map`, `e2e_b_pass` | `npm run seed` | ingest + editable mapping table (series I) | 0 points |
| `e2e_b_idx` | IX per-case API setup | ingest → auto-created payload indexes (§7.1b) | initially empty |
| `e2e_c_points` | `npm run seed` | upsert / batch / get / payload (U/B/G/PL) | 10 points, indexes |
| `e2e_c_victim` | `npm run seed` | bulk delete / delete-by-filter (series DL) | 12 points, indexes |
| `e2e_d_query` | `npm run seed` | hybrid search and the payload-filter matrix (Q/QF) | 12 points, indexes + `phrase_matching` |
| `e2e_d_nested` | `npm run seed:query` | array-of-object payload: same-element vs flat AND | see `tools/seed-d.mjs` |
| `e2e_d_nophrase` | `npm run seed:query` | text index **without** `phrase_matching` (QF-D) | see `tools/seed-d.mjs` |
| `e2e_e_mon` | `npm run seed` | connection bar, monitoring, navigation (H/M/N) | 3 points |
| `e2e_z_basic` | `npm run seed` | compat/lockdown, readiness, profile switch (S/E/Z) | 10 points, indexes |
| `e2e_z_reg` | `npm run seed:reindex` | re-embed lifecycle (series RE) | 2,000 points |
| `e2e_e_alt` | H-series `beforeAll`, via API | active-collection switch | empty |
| `e2e_nf_bulk` | NF per-case API setup | filtered writes, count, update modes | 12 points, indexes |
| `e2e_nf_deep` | NF-X9 API setup | deep-ranking pagination | 40 points |
| `e2e_nf_bare` | NF per-case API setup | no-index controls | 0 or 1 point, depending on the case |
| `e2e_compat_createonly` | compact compatibility spec | real create while the active report is mocked incompatible | deleted by the spec |
| `e2e_export_filename` | X6 route mocks only | download header handling | no real collection |
| `e2e_vector_reindex` | standalone reindex spec, through UI | live migration | campaign-namespaced by `fixtureName`; owned cleanup |

The profile a collection is born under is whichever the stack is running when
the seed runs; there is no permanently-ADVANCED fixture. The cross-profile
cases take their collections from the environment instead — `QHM_FOREIGN` for
`z4-switch.spec.js`, `QHM_E7_COLL` for `z6-e7.spec.js` — because a collection
born under the other profile can only be produced by restarting the stack.

Points created by `tools/seed.mjs` carry the following payload shape, covering
the 8 index types: `f_status` keyword `"s{i}"` · `f_title` text
`"alpha beta gamma {i}"` · `f_views` integer `i` · `f_price` float `i+0.5` ·
`f_flag` bool (`i` even) · `f_geo` geo `{lon:9+i*0.01, lat:45}` · `f_created`
datetime `2026-06-{i:02d}T00:00:00Z` · `f_uuid` uuid
`00000000-0000-4000-8000-{i:012d}`, plus `f_tags` keyword array (sizes 0/1/2 by
`i%3`), `f_nullable` keyword (present, `null`, or absent by `i%4`) and
`f_extra` keyword, which is deliberately **never indexed**. The collections
marked *indexes* above get all of these indexed except `f_extra`.

The seed goes through the middleware's own batch endpoint, so points are
embedded dense+sparse with whichever profile the stack is running.

### 0.2 Payload filter dataset (100 deterministic points)

Use **`e2e_book_filters`**, created from the pure generator and setup protocol
in [§0.6](#fixtures). It reproduces
the distributions used by §8.2 without touching protected `filters_audit_*`
collections. `npm run seed:book` runs `tools/seed-book.mjs`, which implements
the generator and seeds the campaign-owned fixtures. Initialize the campaign
and follow the seed/reset procedure in [the test guide](../README.md) first;
the logical filter fixture becomes `e2e_<run-id>_book_filters`.

The existing executable QF series still uses the separate 12-point
`e2e_d_query` fixture. Never apply the following totals to that collection.

The acceptance fixture contains point IDs `1..100`, loaded via batch.

| Field | Index type | Generation rule | Distribution |
|---|---|---|---|
| `text` | — (payload text, embedded) | Theme by range: 1–20 cooking, 21–40 space, 41–60 music, 61–80 sports, 81–100 programming | 5×20 |
| `status` | keyword | 1–20 `active`, 21–50 `archived`, 51–90 `draft`, 91–100 `deleted` | 20/30/40/10 |
| `tags` | keyword (array) | 1–10 `[]`; 11–40 `["red"]`; 41–80 `["red","green"]`; 81–100 `["red","green","blue"]` | size 0:10, 1:30, 2:40, 3:20 |
| `views` | integer | `views = i` | 1..100 |
| `price` | float | `price = i + 0.5` | 1.5..100.5 |
| `flag` | bool | `true` if `i` even | 50 true / 50 false |
| `description` | text **with phrase_matching** | 1–15 "the quick brown fox jumps"; 16–40 "lazy dog sleeps"; all: "document {i} corpus" | quick:15, lazy:25 |
| `notes` | text **without phrase_matching** | 1–30 "gamma marker"; all: "note {i}" | gamma:30 |
| `location` | geo | 1–25 Milan `(9.19, 45.46)`; 26–60 Rome `(12.50, 41.90)`; 61–100 NYC `(-74.00, 40.70)` | 25/35/40 |
| `created_at` | datetime | `2026-01-01T00:00:00Z + (i−1) days` | 1/day |
| `doc_uuid` | uuid | `00000000-0000-4000-8000-{i:012d}` | unique |
| `optional_note` | keyword | 1–10 field **absent**; 11–20 `null`; 21–100 `"present-{i}"` | 10/10/80 |
| `items` | — (nested objects) | `[{"label":"a","qty":i},{"label":"b","qty":100−i}]` | 2 elements/point |
| `source_id` | — | constant `e2e_book_source` | 100 |

Payload indexes to create (13): `status`:keyword · `tags`:keyword · `views`:integer ·
`price`:float · `flag`:bool · `description`:text+phrase_matching · `notes`:text ·
`location`:geo · `created_at`:datetime · `doc_uuid`:uuid · `optional_note`:keyword ·
`items[].label`:keyword · `items[].qty`:integer.

### 0.3 Ingest files for series I

| File | Content |
|---|---|
| `e2e_chunks_ok.json` | 25 valid chunks: `id`, `text`, + fields `m_str` (string), `m_int` (integers), `m_num` (float), `m_bool`, `m_date` (ISO8601), `m_uuid`, `m_geo` (`{lon,lat}`), `m_partial` (present in only 10 of 25 chunks) |
| `e2e_chunks_mixed.json` | same but `m_int` contains `"abc"` and `m_uuid` a non-UUID value (for failed coercion cases) |
| `e2e_chunks_excluded.json` | also contains `overflow_details`, `overlap_sentences_count`, `split_metadata` (excluded keys) |
| `e2e_chunks_bad.json` | malformed JSON |
| `e2e_chunks_invalid.json` | two chunks, one with a non-integer `id` (preflight blocked) |
| `e2e_chunks_idx.json` | 3 valid chunks with `prezzo` (int), `attivo` (bool), `tag` (string), `autore` (string) — the auto payload-index series (§7.1b) |
| `e2e_chunks_big.json` | large file for mid-ingest abort (I28/I29/I33/I34) and IX3 failure |
| `e2e_chunks_empty.json` | 0 chunks — preflight blocked (I32) |
| `e2e_chunks_nulls.json` | null / mixed / array / object values for coercion edge cases (I5/I20/I21/I25) |

### 0.4 Known limitations / out of UI scope

- **T-API / P-API**: the UI always sends all 5 toggles (`updateSchemaToggles`) and
  hides the phrase checkbox for non-text schemas → the per-field 400 validator and
  the 400 "phrase_matching requires field_schema 'text'" are only testable via direct
  API. They remain in the book as API-level tests (T11, PI9).
- `has_vector` **is exposed** through `(vector presence)` and a vector-name
  selector (`models.js` / `app.js`). No dedicated Playwright behaviour case
  exercises it; this is a coverage gap, not an out-of-UI capability.
- The `e2e_book_filters` empty-array expectation is 10 (§0.2); executable fixtures
  use different fields and counts. Do not copy those totals into live checks.
- Ingest assigns **deterministic UUIDs** to points: the chunk's `id` field does NOT
  become the point id — verification must locate the point via scroll/Query.
- The last re-embed job state **persists**: a visible `COMPLETED` may refer to a
  previous job. A test must first wait for a **live** state
  (`RUNNING`/`FINALIZING`/`PAUSED`), then for completion.
- The filter tree is part of the **persisted query snapshot** and survives reload:
  it must be **cleared** between independent cases, otherwise rows accumulate and a
  residual row without a value invalidates the run.
- In payload-only mode, results are **paginated at `PAYLOAD_PAGE_SIZE = 5`**:
  to compare an expected set, traverse all pages with **Next page →**.
- **Geo** cell order: radius = `lon, lat, radius(m)`; box = `TL lon, TL lat,
  BR lon, BR lat`; polygon = text field in the form `lon lat; lon lat; …` (not JSON).
- Filter rows offer **only indexed fields** (plus `(point id)` and
  `(vector presence)`): a non-indexed payload field is not selectable.
- **Match-all filter `{}`** is rejected by the UI (submit disabled) and by the
  API (400 `"filters must contain at least one condition"`) for delete-by-filter
  and for filtered payload write; an explicit action on every point must target
  a real condition, not an empty object.


<a id="status"></a>

### 0.5 Case status, scope and common procedures

Case definitions alone are **SPECIFIED / NOT RUN**. Executed results are
recorded separately in §16.7–§16.8; none inherits a PASS from a similar title
or from a different interaction layer. A table row with variants means **every variant** needs
its own result. No row may be implemented as `expect(true)`, a source-text
search, an unconditional skip, or an assertion guarded by a failed prerequisite.

| Layer | What is exercised | Required evidence |
|---|---|---|
| UI-live | Browser actions through the middleware to real Qdrant/embedding | Request status, rendered state and independent stored-data checks |
| UI-mock | Real console/browser with controlled API responses | Observed requests, DOM/downloads, no accidental live mutations |
| API-live | Middleware HTTP contract against disposable real services | Exact status/body, before/after point and schema snapshots |
| MCP-live | AI agent calls the real smart-qdrant-mcp-server tools/resources | Sanitized tool arguments/results and independent postconditions; see §0.8 |
| Skill-agent | AI agent loads smart-skill and follows its applicable playbook through MCP | Skill identity, task prompt, workflow trace and evaluated outcome; see §0.8 |
| Backend | Unit/ASGI integration with controlled clients, events and temporary paths | Call arguments, cancellation/task state, writes and filesystem effects |
| Model | Pure frontend builders/validators without a browser | Exact serialized values/errors; no UI or backend behaviour claim |
| Static | Case inventory and configuration | Explicit mapping and collected title/configuration; no claim of behaviour |

**Current contract** is the default expectation. Frontend hardening regressions
(such as rejecting unsafe JSON numbers or suppressing A→B→A stale responses)
are explicitly identified and require their own executed evidence. Rows labelled
with an audit finding retain its regression oracle after implementation; a code
change alone never supplies a PASS for every layer or variant. The historical
audit in §16 remains unchanged. A client abort is not proof that server work stopped.

Use these procedures throughout:

1. **Prepare:** create only the case's allowlisted `e2e_` collection on the
   selected base URL; assert success, exact seed count and required index types.
   Preserve a snapshot keyed by canonical point ID. A failed setup is a failed
   prerequisite, never a reason to omit an assertion. Only recreate/delete the
   explicitly owned fixture, following the README reset procedure.
2. **Exercise UI:** select that collection, open the named panel, fill controls
   using visible labels, arm a confirmation if applicable, and observe the
   relevant request before clicking. Check the actual body and HTTP status.
   Use accessible roles where possible; stable IDs are allowed for ambiguous
   controls. A screenshot or visibility check alone does not prove a write.
3. **Compare reads:** collect **all** pages, reject repeated offsets/IDs, and
   compare the exact ID set and payload projection with the fixture oracle.
   For unordered sets ignore order; for ordered results assert the specified
   ordering, allowing arbitrary ordering only among equal sort values. Treat
   IDs as decimal strings/UUIDs; never round them through JavaScript `Number`.
4. **Compare writes:** independently read the target and an unaffected control
   point; compare the complete payload and count. Negative write cases require
   no change, including no earlier valid batch item being written. Instrument
   the backend when the claim is zero embedding/Qdrant calls.
5. **Observe asynchronous work:** use a response/event barrier and poll the
   stated state with a finite deadline. UI-mock deadline: 10 seconds after
   releasing the response; live CRUD/read deadline: 60 seconds; live reindex:
   600 seconds unless the recorded preflight justifies a larger budget. A
   missed RUNNING/PAUSED/CANCELLED state is not a PASS via COMPLETED. Controlled
   backend events provide deterministic transition coverage when hardware
   finishes too quickly. Record any longer live budget before running.
6. **Finish:** disarm confirmations, cancel any case-owned pending job, record
   the resulting fixture state and remove only owned temporary files. Read-only
   fixtures stay intact. Record console exceptions, failed requests and expected
   negative responses per case; never put tokens in traces, screenshots or logs.

<a id="fixtures"></a>

### 0.6 Reproducible fixtures and independent oracles

These fixtures are additional to `npm run seed`. The JavaScript below is a
**pure data generator**, not a command that contacts or changes a service.
Implement setup using the existing authenticated API helper after TB-HAR-001
and TB-HAR-002 are satisfied; manual preparation must use the same bodies.
Generated JSON belongs in the ignored run-artifact directory, not in a patch.

For each new collection POST `/collections` with
`{collection_name: name, dense_on_disk: true, sparse_on_disk: true,
on_disk_payload: true, quantization: true, sparse_idf: true}`. Submit generated
points in batches of at most 100 to `/points/batch` using
`{collection_name: name, points: batch}`; assert each `upserted` count.
Create indexes with POST `/collections/{name}/payload-indexes`, body
`{indexes: [{field_name, field_schema, ...options}]}`. Read the report until the
expected count/index schema is visible, then independently scroll the full
collection and compare it to the generated data. Do not seed through ingest:
these fixtures require the explicit numeric point IDs shown below.

```js
const uuid = (i) => `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`;
const day = (i) => new Date(Date.UTC(2026, 0, i)).toISOString();
const themes = ["cooking", "space", "music", "sports", "programming"];
function filterPoint(i) {
  return {
    point_id: String(i),
    text: `${themes[Math.floor((i - 1) / 20)]} document ${i}`,
    payload: {
      source_id: "e2e_book_source",
      status: i <= 20 ? "active" : i <= 50 ? "archived" : i <= 90 ? "draft" : "deleted",
      tags: i <= 10 ? [] : i <= 40 ? ["red"] : i <= 80 ? ["red", "green"] : ["red", "green", "blue"],
      views: i,
      price: i + 0.5,
      flag: i % 2 === 0,
      description: `${i <= 15 ? "the quick brown fox jumps " : i <= 40 ? "lazy dog sleeps " : ""}document ${i} corpus`,
      notes: `${i <= 30 ? "gamma marker " : ""}note ${i}`,
      location: i <= 25 ? {lon: 9.19, lat: 45.46} : i <= 60 ? {lon: 12.5, lat: 41.9} : {lon: -74, lat: 40.7},
      created_at: day(i),
      doc_uuid: uuid(i),
      ...(i <= 10 ? {} : {optional_note: i <= 20 ? null : `present-${i}`}),
      items: [{label: "a", qty: i}, {label: "b", qty: 100 - i}],
    },
  };
}
const filterPoints = Array.from({length: 100}, (_, k) => filterPoint(k + 1));
const filterIndexes = [
  ["status", "keyword"], ["tags", "keyword"], ["views", "integer"],
  ["price", "float"], ["flag", "bool"], ["description", "text", true],
  ["notes", "text", false], ["location", "geo"], ["created_at", "datetime"],
  ["doc_uuid", "uuid"], ["optional_note", "keyword"],
  ["items[].label", "keyword"], ["items[].qty", "integer"],
].map(([field_name, field_schema, phrase_matching]) => ({
  field_name, field_schema,
  ...(field_schema === "text" ? {phrase_matching} : {}),
}));
const edgeTypes = [
  ["k", "keyword", "red", "green"], ["i", "integer", 1, 2],
  ["f", "float", 1.5, 2.5], ["b", "bool", false, true],
  ["t", "text", "alpha", "beta"],
  ["g", "geo", {lon: 9.19, lat: 45.46}, {lon: 12.5, lat: 41.9}],
  ["d", "datetime", day(1), day(2)], ["u", "uuid", uuid(1), uuid(2)],
];
const edgePoints = Array.from({length: 6}, (_, k) => {
  const i = k + 1;
  return {
    point_id: String(i), text: `edge document ${i}`,
    payload: Object.fromEntries(edgeTypes.flatMap(([key, , a, b]) =>
      i === 4 ? [] : [[key, i === 1 ? [] : i === 2 ? [a] : i === 3 ? [a, b] : i === 5 ? null : a]])),
  };
});
const exportPoints = Array.from({length: 2001}, (_, k) => ({
  point_id: String(k + 1), text: `export document ${k + 1}`,
  payload: {source_id: "e2e_book_export", id: k + 1, marker: `m-${k + 1}`, unicode: "à漢🙂"},
}));
```

| Fixture / ownership | Exact preparation and oracle |
|---|---|
| `e2e_book_filters` / read-only filter cases | `filterPoints`, all 13 `filterIndexes`; the distributions and §8.2 counts in this book apply exactly. No mutations or reindex in this collection |
| `e2e_book_edges` / read-only universal operators | `edgePoints`; index each `edgeTypes` key with its schema, enabling phrase matching only for `t`. IDs 1/2/3 have arrays of length 0/1/2, ID 4 is missing, ID 5 is null, ID 6 is a scalar |
| `e2e_book_write` / reset before each write variant | IDs 1,2,3: `text: "original i"`, payload `{source_id:"e2e_book_write", ordinary:"keep-i", nested:{leaf:"v"}}`; keyword index on `source_id`; IDs 1 and 2 are mutation targets, ID 3 is the control |
| `e2e_book_export` / read-only export | `exportPoints`: 2,001 points, so server page size 1,000 requires three scroll pages; compare all ID strings, text, `id`, `marker`, Unicode and `source_id` |
| `e2e_book_empty` / read-only empty states | No points; native embedding contract |
| `e2e_book_reindex` / exclusively owned lifecycle | Copy the 2,001-point export input to a new collection; use a fresh seed for each incompatible/cancel/failure variant. Snapshot every payload and active vector mapping before starting |
| `e2e_book_roundtrip_a`, `e2e_book_roundtrip_b` / each roundtrip | Empty native collections; ingest input described in TB-EXP-005/006; no dependency on export or reindex fixtures |
| `e2e_book_sample` / filtered clear only | IDs 1..1001, text `sample i`, `source_id:"e2e_book_sample"`, `ordinary:i`; keyword index on source_id; only ID 1001 has `rare:"survives"`. Verify the first 1,000 scrolled points before clearing |
| `e2e_book_alt` / stale-response control | ID 1: text `ALT ONLY`, payload `{marker:"alt-only"}`; native contract; its response is visually distinct from every other fixture |

No fixture assumes raw dense dimensions, vector names, ranking scores or model
speed. Read active vector names/dimensions from the collection report. Exact
filter oracles come from the formulas above, **not** from asking the same API
to compute both expected and actual results. Count plus first-page membership
is insufficient. Never stringify unsafe numeric IDs after a lossy JSON parse.

<a id="harness"></a>

### 0.7 Harness, isolation and case traceability

Layer: Static + UI-mock unless API-live is explicitly stated. These cases are
prerequisites for a broad destructive run, not extra authorized targets.

| Case | Action and variants | Required result |
|---|---|---|
| TB-HAR-001 | Apply the shared target guard to seed/reset helpers, every mutating spec, and `QHM_REINDEX_COLL`, `QHM_NATIVE`, `QHM_FOREIGN`, `QHM_E7_COLL`. Try `cat`, `test`, `reindex_e2e_1`, whitespace-only, and `e2e_book_write` | First four rejected before any write request; last accepted only when explicitly owned by the run. No blank override falls back to a non-sandbox collection. Verify the standalone reindex fixture uses its campaign-owned `e2e_` name |
| TB-HAR-002 | Collect/run an isolated mocked request on a non-default loopback port; record requests from browser, API helpers, legacy specs, seed/reset and assets | All use one normalized `QHM_BASE_URL`; no request to 6336 from the overridden run. Test both a trailing slash and no trailing slash |
| TB-HAR-003 | Collect default and external-state configurations without execution; resolve every applicable case key to a full test title and file | Default excludes z4/z5/z6; override includes them. One case key can map to several tests only with explicit variant labels. Missing mappings stay unimplemented, not allowlisted |
| TB-HAR-004 | Feed the coverage checker separate entries for `book:6.2/C1`, `book:8.2/C1`, `book:8.2/QF-A`, `book:9/M-G`, a generated title and a `TB-` case; then remove one mapping or turn it into a comment | Removal/comment-only reference fails mapping validation; digitless suffixes and section collisions remain distinct. A mapped but skipped/partial test never becomes complete coverage |
| TB-HAR-005 | Intentionally fail setup GET, seed POST or index creation, and inject an unexpected page exception | Case fails at the actual prerequisite/exception; no conditional payload comparison, empty substitute dataset or global console-error exemption. Expected HTTP errors are scoped to the specific method/path/status of that case |
| TB-HAR-006 | UI-mock two fixtures, abort the case midway through a pending request/job, then invoke its cleanup | Only recorded owned targets touched; pending mock handlers/listeners removed; next case starts with no stale response, armed confirmation or foreign active collection. Original stack/profile restoration recorded separately for external-state runs |

Canonical legacy keys are `book:<section>/<local ID>`; existing section
numbers and case IDs remain unchanged. Expand combined legacy
rows into one mapping per ID. New keys are the globally unique `TB-*` IDs.
Each mapped variant retains its actual file and full collected test title.
The campaign reporter reads these rows and actual Playwright results; NF-COV2
checks ledger completeness. Missing mappings remain NOT_MAPPED, and skipped or
unexecuted variants never become PASS. The 96 TB definitions remain in the
ledger even when a browser implementation is absent. `requireCompleteCoverage`
rejects any non-PASS case; NF-COV2 alone is not that execution gate.

---

<a id="agent-workflows"></a>

### 0.8 Agent-driven MCP and skill acceptance

For user-simulation campaigns, use the web console at
`http://127.0.0.1:6336/` whenever the case exposes a UI action. In addition,
exercise every applicable service workflow through an AI agent connected to
**smart-qdrant-mcp-server** and using the client-side **smart-skill**. These
are complementary layers: an MCP result cannot replace a DOM, accessibility,
validation or browser-state assertion; a REST/Playwright pass cannot prove MCP
transport, tool translation or skill-guided behaviour.

Before execution, the agent must read the repository's
[skill](../../../../smart-skill/smart-qdrant-mcp/SKILL.md), load the relevant
[playbook](../../../../smart-skill/smart-qdrant-mcp/references/playbooks.md), and
use the [tool reference](../../../../smart-skill/smart-qdrant-mcp/references/reference.md)
for parameters and gates. Connection, authentication, resources and tool
contracts remain defined once in the
[MCP README](../../../../smart-qdrant-mcp-server/README.md).
Discover the actual tools in the client and check live health before writes.
Record the loaded skill path/version or hash; merely installing or mentioning
a skill is not evidence that an agent followed it.

| Existing feature scope | Required agent workflow when exposed by the discovered tools |
|---|---|
| Collections, indexes and compatibility (§1, §6, §12) | Inspect health, collection report and schema; create/update only owned fixtures; check compatibility after operator-managed profile switches. |
| Ingest, IDs and payload (§7) | Follow the ingest playbook, read back IDs/payload/counts, and exercise applicable payload and confirmation gates. File staging remains an operator action outside MCP. |
| Search, filters, count and facet (§8, §13) | Use the appropriate search/read/count/facet tools and filter documentation; compare results with the same deterministic oracle, including empty-result diagnosis. |
| Reindex (§5) | Follow preflight, start and status/recovery steps; verify vector/schema postconditions and search. Do not cancel a job merely to bypass a conflict. |
| Export and deletion (§6, §7) | Follow the export/safe-delete playbooks and exact confirmation contracts; verify the export or deletion independently, including unaffected controls. |
| Monitoring and recovery (§3, §9, §12) | Use health/live-state resources to diagnose a controlled failure and verify recovery; browser charts and lockdown assertions still require UI evidence. |

The operator assigns the agent an ordinary user task and explicit owned targets,
not just a prewritten sequence of REST requests. Before each applicable case,
record the supported MCP equivalent and chosen playbook. Tool-side limits may
require equivalent paginated or smaller requests; record that adaptation and
never claim coverage of an unsupported parameter or of the larger workload.
UI-only actions, filesystem staging, physical profile switches and low-level
fault injection have no agent-tool substitute; record why MCP is inapplicable
to that action rather than inventing a tool.

All agent writes use the same campaign manifest, exact owned targets, fixture
reset and cleanup rules as the browser lane. The Playwright guard does not
intercept MCP calls: verify ownership before sending each mutation. Use
separate owned fixtures or reset between equivalent writes so one lane cannot
consume the other's preconditions. Follow the skill's destructive workflow and
the tool-specific confirmation/dry-run rules; preserve conservative budgets,
protected collections and token redaction. Correlate backend instrumentation
when asserting zero downstream calls; a refusal message alone is insufficient.

For each case/variant record MCP-live and Skill-agent outcomes independently:
agent/client identity, user task, skill/playbook identity, tool/resource names,
sanitized arguments and responses, observed errors, independent postconditions,
and cleanup. Mark an applicable but unexecuted layer **NOT RUN**, an unavailable
service/tool/skill **BLOCKED** with the prerequisite, and an unsupported action
**not applicable** with a contract-based reason. None counts as PASS. A script
calling MCP without a skill-guided agent proves only its MCP scope. A campaign
with missing required agent evidence remains partial even when its inventory
gate and browser suites are green.

**Evidence boundary:** §16.7–§16.8 do not record an agent-driven MCP/skill
campaign. Those historical counts receive no MCP-live or Skill-agent credit;
the applicable agent workflows remain NOT RUN until separately executed.

## 1. BASIC ↔ ADVANCED server switch procedure

The Smart Embedder runs in **one profile at a time**. The profile is set by
`SMART_EMBEDDER_PROFILE` in the root `.env` (`→ EMBEDDER_PROFILE` in the container).
BASIC and ADVANCED = **different vector spaces** ⇒ different
`embedding_contract_fingerprint` ⇒ collections born under the other profile become
**incompatible**.

**Switch (outside the UI, then return to the browser):**

```bash
# 1. stop stack
./stack.sh down        #  .\stack.ps1 down   (Windows)
# 2. edit .env → SMART_EMBEDDER_PROFILE=advanced   (or basic)
# 3. restart
./stack.sh up cpu      #  .\stack.ps1 up cpu
# 4. in browser: connection bar → "Re-test"
```

Rows marked **[SWITCH→basic]** / **[SWITCH→advanced]**: perform the switch,
wait for health `HEALTHY`, then the case.

### 1.1 Lockdown truth table

Source: `applyLockdowns()` + `renderServiceGate()` (app.js) + `collectionLockState()` (models.js).

| State | Service-gate banner | Query / Points | 01 Report | 02 Re-embed | 03 Create / Recreate | 04 Toggles / 05 Indexes | 07 Delete | 06 Export |
|---|---|---|---|---|---|---|---|---|
| `compatible` | empty | active | live | live | live (create **and** recreate) | live | live | live |
| `incompatible`, same profile, different fingerprint (**recoverable**) | "different vector space… Re-embed its vectors… create a new collection (03)" | Query inert; Points accessible with embedding writes blocked | live | **live** | **create-only** | inert | **live** | **live** |
| `incompatible`, **BASIC↔ADVANCED** (profile switch) | **suppressed (empty)** | Query inert; Points accessible with embedding writes blocked | live | **live** | **create-only** | inert | **live** | **live** |
| `incompatible` without `stored` (**foreign**) | "no verified embedding contract… still create a new collection (03)" | **inert** | live | **inert** | **create-only** | inert | **live** | **live** |
| services down (health unhealthy) | "Smart Embedder unreachable…" (**always precedes** compat) | **inert** | **inert** (entire `panel-collections`) | inert | inert | inert | inert | inert |

**Operational notes (verified):**
- **`ce-wrap` (06 Export) is NOT in the lockdown list**: remains operable with
  an incompatible collection. Verified in **X4**.
- **03 is half-locked, never fully locked** (*create-only*): `cc-wrap` stays
  interactive while `cc-recreate-control` and `cc-confirm-wrap` go `inert`,
  `cc-recreate` is unchecked + `disabled` + `aria-disabled`, `cc-submit` reads
  "Create collection", and the payload leaves with `recreate: false`. A NEW
  collection is built from the **active** contract, so it cannot inherit the
  mismatch; recreate would destroy the mismatched one and stays blocked.
  Verified in **A–D** of `embedding-compatibility.e2e.spec.js`.
- **07 Delete is never locked by an incompatible collection.** It addresses a
  name the user types in `cd-name`, not `state.active`, and `cd-submit` already
  gates on the typed confirmation alone — so the active collection's contract
  tells it nothing, the same reason the create half of 03 stays live. Locking
  it stranded the user with a collection they could neither use nor remove.
  A service outage still blocks it, through `panel-collections`. Verified in
  **E**, **E2** and **F** of `embedding-compatibility.e2e.spec.js`.
- During an **outage**, the lock is on the entire `panel-collections` and sub-wraps
  are explicitly **unlocked** (reconciliation): on return to HEALTHY the internal
  state must be correct, not "double-locked". Verified in **S6**.
- `conn-collection` is disabled **only** by outage (or empty list), **never**
  by the collection lock: during lockdown the user must be able to switch collections.
- With `opsLocked`, `ql-embedding-guard` and `pt-embedding-guard` are **emptied**
  and `ql-query-mode` is **hidden**: the banner is the sole source of "why".

---

## 2. Series S — Service readiness / outage gate

| ID | Action | Expected UI |
|---|---|---|
| S1 | Start with healthy stack | green dot, `HEALTHY`, detail `qdrant reachable · embedder reachable`; all panels active |
| S2 | Stop Smart Embedder, **Re-test** | `UNHEALTHY`, detail `… embedder unreachable`; gate "Smart Embedder unreachable"; `#panel-query/collections/points` with `inert`; `conn-collection` and `ql-run` disabled |
| S3 | During outage with an **incompatible** active collection | banner shows the outage, **NOT** "Re-embed"/"Switch to a compatible collection" (precedence) |
| S4 | Restart embedder, **Re-test** | returns to `HEALTHY`, panels reactivated, no residue |
| S5 | Stop Qdrant, **Re-test** | detail `qdrant unreachable · embedder reachable`; same outage lockdown |
| S6 | **Reconciliation**: outage (S2) → recovery (S4) with an **incompatible** active collection | after recovery: `panel-collections`, create, delete and export live; recreate controls, `cu-wrap` and `pi-wrap` inert; `cr-wrap` live only if recoverable, matching §1.1 |
| S7 | Automatic health polling (30s) without interaction | state updates on its own after a fault, without clicking Re-test |

The default `z2-service.spec.js` simulates outages with `page.route`; it does
not stop either service. `z5-real-outage.spec.js` covers an actual stopped
embedder only. A real Qdrant outage and real stop/restart roundtrip remain
operator scenarios, not default-run evidence.

---

## 3. Series H — Connection bar & API key

| ID | Action | Expected |
|---|---|---|
| H1 | `Active collection` dropdown populated on startup | list of real collections, sorted; selection persists in `localStorage` (`qhm_active_collection`) |
| H2 | Change active collection | previous collection's result-areas **cleared** (ing/up/bt/get/get-many/del/del-bulk/delf/pl/pl-keys/pl-clear + scoped `cr-*`,`cu`,`pi-*`); `delf-count` hidden; query filter tree **reset**; report 01 auto-refresh |
| H3 | Click **API key** → popover | `aria-expanded=true`; password input focused; input **emptied** on open |
| H4 | Submit with **empty** field | status "ENTER A KEY FIRST" (error); popover stays open; focus returns to input |
| H5 | Register a valid key, **Register** | status "API KEY REGISTERED"; popover auto-closes after ~800ms; **Clear** visible; key in tab-scoped `sessionStorage` (`qhm_api_key`) |
| H6 | After Register → **reload protected data** | health re-probed + collections reloaded + report/reindex/fields reloaded, **without page reload** |
| H7 | **No key** and protected collections (401) | `conn-collection-hint` visible with **"API KEY MISSING"**; select empty |
| H8 | **Wrong** key registered (401) | hint **"API KEY REJECTED"**; `apikey-btn` button with class `is-bad` (red dot **persistent**) |
| H9 | **Correct** key registered after a 401 | hint hidden; dot `is-set` (blue) that **turns off after 3s**; select populated |
| H10 | **Clear** key | status "API KEY REMOVED"; focus stays on input; protected data reloaded; hint returns to "API KEY MISSING" |
| H11 | **Esc** with popover open | popover closed; **focus returns** to the trigger button |
| H12 | **Click outside** the popover | popover closed; focus **NOT** stolen (stays where the user clicked) |
| H13 | Click the trigger with popover already open | toggle → closed |
| H14 | Reopen popover with a saved key | Clear visible; input empty again (the key is never re-shown) |

---

## 4. Series E — Embedding compatibility & BASIC↔ADVANCED switch (core)

For real switches, prepare one collection under each profile. `QHM_NATIVE`
always names the collection compatible with the **currently running** profile;
`QHM_FOREIGN` names the incompatible one. Their roles reverse on switching back.
The name `e2e_z_basic` alone does not prove which profile created it.
Default E5/E6/E9/E10 and E2/E4-mock/E8-mock use simulated reports.

| ID | Setup | Active | Expected |
|---|---|---|---|
| E1 | Embedder **BASIC** | `e2e_z_basic` | `compatible`; banner empty; Query/Points active; Report 01 green |
| E2 | **[SWITCH→advanced]** | `e2e_z_basic` (born BASIC) | `incompatible` **profile switch**: service-gate **empty**; Query **inert** + class `operations-disabled`; Points accessible, retrieval works and embedding writes are disabled; Report 01 shows `INCOMPATIBLE`; **02 Re-embed live**; **03 create-only**, 04/05 inert; **06 Export and 07 Delete live** |
| E3 | Embedder ADVANCED | the ADVANCED-born collection (`QHM_NATIVE` in this phase) | `compatible`; everything active |
| E4 | **[SWITCH→basic]** | the ADVANCED-born collection (`QHM_FOREIGN`) | same as E2, reverse profile switch |
| E5 | **[MOCK]** Same-profile fingerprint drift | collection with a stored contract | banner **shown** "different vector space… Re-embed its vectors"; same lockdown as E2 |
| E6 | **[MOCK]** Report without `stored` (**foreign**) | collection without a verified contract | banner "no verified embedding contract"; **02 Re-embed also inert** |
| E7 | From E2, **Re-embed** under ADVANCED to completion | `QHM_E7_COLL`, born BASIC | compat → `compatible`; Query/Points reactivated. Current excluded spec explicitly expects the resulting profile to be `advanced` |
| E8 | Switch active from incompatible to compatible via dropdown | `QHM_FOREIGN` → `QHM_NATIVE` | selector remains enabled; everything unlocks. Real flow in excluded `z4-switch.spec.js`; default **E8-mock** simulates the report |
| E9 | **[MOCK]** Open Points during incompatibility lockdown | incompatible | `pt-embedding-guard` empty and `ql-query-mode` hidden |
| E10 | **[MOCK]** Delay report after collection switch | target collection | compatibility-check info remains until the report arrives |

---

## 5. Series RE — 02 Re-embed vectors

Typical precondition: active collection **incompatible recoverable** (post E2/E4).
Use a collection of several hundred points (≤ ~50 and the job finishes in a few
seconds, making the Pause/Resume/Cancel windows unobservable).

| ID | Action | Expected |
|---|---|---|
| RE1 | **Run read-only preflight** | preflight result (point count, estimate); no writes |
| RE2 | Confirmation field **empty** | Start disabled |
| RE3 | Confirmation **wrong** | Start disabled; hint "Does not match" |
| RE4/5/7/9/10/11/12/13/14 | Combined lifecycle: start, pause/resume, Points lockdown, completion, payload comparison, hybrid query | Current test checks Upsert/Batch disabled and Get enabled while paused, then query success and Upsert enabled. Payload comparison covers point 1 **only if its initial GET succeeds**; complete payload equality and every mutating subtab are not proved |
| RE6 | **Cancel job** | intended: cancellation settles and points remain readable. Current spec allows `COMPLETED` too and checks the report, not point integrity; may skip if the cancel window is missed |
| RE8 | Switch active collection **during** job | intended: scoped results cleared and no stale job result applied. Current spec switches **after preflight only**, without starting a job; active-job coverage is missing |

**RE notes:**
- The last job state **persists**: a `COMPLETED` visible right after Start may refer
  to the previous job. To verify a new job, first wait for a **live** state
  (`RUNNING`/`FINALIZING`/`PAUSED`), then for completion.
- The old generation stays active until cutover and is removed after exact validation.
- During pause, the UI disables Upsert; a direct mutation must return HTTP 409.
  The combined test exercises the disabled button, not that HTTP rejection.


<a id="reindex"></a>

### 5.1 Reindex: real transitions, exclusion and vector generations

Layer: UI-live + API-live; Backend controls for deterministic pauses/failures.
Use `e2e_book_reindex`. Before **every** variant, require a successful complete
payload snapshot, exact point count, active contract/vector mapping and one
baseline query. Attach job identity/timestamps so an old COMPLETED status cannot
satisfy a new start. No assertions are conditional on a successful initial GET.

| Case | Actions | Required result |
|---|---|---|
| TB-REI-001 | Read-only preflight; start with exact confirmation; observe new RUNNING; pause at a batch boundary; observe PAUSED; resume and wait for COMPLETED | Preflight writes nothing. Durable processed count does not decrease, paused offset stays stable, resume continues from it, and terminal completed count equals seed count. Confirmation disarms; all payloads and point IDs equal baseline; active contract matches runtime; hybrid query succeeds |
| TB-REI-002 | Start a fresh job, require RUNNING then PAUSED, cancel and wait for CANCELLED | CANCELLED required, not COMPLETED; active source-vector mapping retained and every payload/ID unchanged. Reads work and mutation lock releases. A too-fast live job is reported as a missed precondition; backend event-gated test must still prove cancellation |
| TB-REI-003 | While actual job on A is RUNNING/PAUSED, select `e2e_book_alt`; release delayed status/preflight/control replies for A; return to A | Alt UI never shows A's job as its own; alt writes remain available when compatible; A job persists and is rediscovered on return. Switching collection does not implicitly cancel the server job |
| TB-REI-004 | Get and Get-many `with_vectors=true` before, during partial progress, after completion; exercise vector presence against declared source/target generations in Backend/API-live | Summaries contain names and shapes only. Processed points may carry source+target during the job; unprocessed points still have source. After cutover the report's active target is authoritative and present on all points; selector refreshes. No claim that names return to canonical defaults; no raw vector values in summary responses |
| TB-REI-005 | During RUNNING and PAUSED invoke every guarded mutation in the inventory below; repeat representative reads and a write to unrelated native collection | Each same-collection mutation rejected 409 at the API and unavailable in UI where exposed; no hidden write. Reads/report/export stay functional subject to service readiness. Unrelated collection is not serialized behind this job |
| TB-REI-006 | Backend: fail embedding or vector upload after a committed batch; restart manager with durable job metadata; resume then cancel or complete | FAILED/INTERRUPTED checkpoint retained and recoverable; no lost payloads, no premature active-generation cutover. Resume uses committed offset; active/failed job retains mutation guard until terminal cancel/complete |
| TB-REI-007 | Backend: hold finalization; inspect query/generation mapping before and after releasing cutover; inject failure immediately before metadata commit | Readers use a coherent active generation; no target made active before its data is durable. Retry/finalize follows persisted state; payloads unchanged and no orphan target advertised as active |

TB-REI-005 covers all 19 guarded routes, independently recorded: POST ingest,
POST ingest-chunks, POST upsert, POST batch, DELETE one point, DELETE bulk,
POST delete-by-filter, PATCH/PUT single payload, POST single delete-keys/clear,
PATCH filtered payload, POST filtered delete-keys/clear; POST collection
create/recreate against the occupied target, schema toggles (both PUT/PATCH
aliases), create indexes, delete index, delete collection. Reindex start/control
has its own state validation: a second start cannot create another active job.
At the UI layer also verify Get, Get-many, Count, Report and Export rather than
inferring all read availability from a single enabled button.

---

## 6. Series R / C / T / PI / X / D — COLLECTIONS tab

### 6.1 R — 01 Report

**R1–R6 have dedicated tests** in `a-collections.spec.js`, in addition to
inline report checks in C/T/K. R1 checks positive points and a numeric dense
size, not an exact count of 10 or dimension of 1024; R4 exercises one toggle
write. The checklist below describes the intended broader postconditions.

| ID | Action | Expected | Expected UI state |
|---|---|---|---|
| R1 | Active = `e2e_a_keep` (default all-true) | badge: dense on disk `true` · sparse on disk `true` · payload on disk `true` · quantization `true` · sparse idf `true` | 5 tiles first row; second row: collection name, status `green`, points `10`, dense size matching the active embedding contract |
| R2 | Via API: create an all-false collection (test C2, then return here) and select it | badge all `false` | points `0`; no findings |
| R3 | Switch collection from the connection bar | report auto-updates on every switch, without clicking anything else | points_count consistent; no residue from the previous collection |
| R4 | After every write operation (create, toggles, upsert a point from Points tab) | report auto-refreshed (points_count/badge updated) | verified alongside C/T/P tests |
| R5 | No active collection (only if reachable without touching protected ones; otherwise skip) | info alert "No collection selected." | Update toggles disabled |
| R6 | Drift findings: create with all-true toggles on an all-false collection, no recreate (=C6) | 409 → info alert pointing to 04 Schema Toggles | report 01 stays consistent with the real state |

### 6.2 C — 03 Create / Recreate

Toggles: `dense_on_disk`, `sparse_on_disk`, `payload_on_disk`, `quantization`, `sparse_idf`.

| ID | Input | Expected | Expected UI state |
|---|---|---|---|
| C1 | `e2e_a_new1`, toggles **default (all-true)** | 200 created=true | green alert "Collection created"; active → new1; report badge all-true; dropdown refreshed |
| C2 | `e2e_a_new2`, **all-false** | created=true | report badge all-false |
| C3 | `e2e_a_new3`, mix A (dense=false, quant=true, idf=false, sparse_disk=true, payload=false) | created=true | badge consistent with the mix |
| C4 | `e2e_a_new4`, mix B (dense=true, quant=false, idf=true, sparse_disk=false, payload=true) | created=true | badge consistent with the mix |
| C5 | Re-submit `e2e_a_new1` with the **same** config | 200 created=false | info alert "already exists and already has these settings — nothing to do"; no changes |
| C6 | Submit `e2e_a_new2` (all-false) with toggles **all-true**, recreate OFF | backend 409 (schema mismatch) | readable **info** alert: "Collection already exists. To change the true/false flags use 04 Schema Toggles…" — no raw JSON; collection intact |
| C7 | `e2e_a_victim` + recreate ✓, confirmation field **empty** | blocked client-side | confirmation field appears; Submit **disabled** |
| C8 | recreate ✓, wrong confirmation | blocked | Submit disabled; hint "Does not match — destructive action stays disabled." visible |
| C9 | recreate ✓, exact confirmation | 200; collection recreated | green alert; points_count → **0**; recreate checkbox auto-disarmed and confirmation cleared |
| C10 | Empty name | blocked | Submit disabled |
| C11 | Collection created under the active profile carries that profile's contract | `embedding_contract` stored on the collection; report reflects the active profile | report badge shows the contract fingerprint, not a generic label |
| C12 | **[API]** create without toggles | 400 with per-field validation (`dense_on_disk: Field required`, …): the 5 toggles are **mandatory** in the request model, not optional | n/a (UI always sends 5 values) |
| C13 | Name = `"   "` (spaces only) | trim → empty name → blocked | Submit disabled |

### 6.3 T — 04 Schema Toggles (active collection)

Each Tn: flip ONE toggle from the current state, submit, verify the badge in the
01 report, then verify points_count is unchanged. T1–T8 map directly to spec tests
in `a-collections.spec.js`; T10–T12 fill the old T6/T7/T8 cases.

| ID | Toggle flipped / Action | Expected | Expected UI state |
|---|---|---|---|
| T1 | `dense_on_disk` true→false | 200 | green alert "Schema toggles updated"; badge dense on disk `false`; points unchanged |
| T2 | `sparse_on_disk` true→false | 200 | badge sparse on disk `false` |
| T3 | `payload_on_disk` true→false | 200 | badge payload on disk `false` |
| T4 | `quantization` true→false | 200 | badge quantization `false` |
| T5 | `sparse_idf` true→false | 200 | badge sparse idf `false` |
| T6 | Update with no changes | 200 no-op | green alert "Schema toggles updated"; all badge values unchanged |
| T7 | No active collection | Update disabled | `cu-submit` disabled when `state.reportOk` is false; `cc-report` shows "no collection selected" |
| T8 | Active collection change right after Update | cu-result **cleared** (K5) | scoped result emptied on active switch |
| T10 | Widget seeding: after flips + report load, cu-* selects reflect the live state | `cu-*` selects all match the report, not static defaults | no involuntary flip on the next submit |
| T11 | **[API]**: `PATCH /collections/{name}/schema-toggles` body `{}` | 400 "at least one schema toggle must be provided" | n/a (UI always sends 5 values); current spec only bounds the status to 4xx |
| T12 | Restore: all toggles → true in a single update | 200 | badge all-true; data intact (spot check `GET /points/1`) |

### 6.4 PI — 05 Payload Indexes (active collection)

> **PI1 consolidates the 8 type rows.** The spec creates one index per schema
> type in a single test; PI2–PI10 then cover the remaining controls. PI12 and
> PI15 are separate spec tests. Table refresh is asserted inline.

| ID | Input | Expected | Expected UI state |
|---|---|---|---|
| PI1 | Create one payload index per schema type (keyword, text, integer, float, bool, geo, datetime, uuid) | all 8 created | green alert per field; table refreshed with the correct row per schema |
| PI2 | Phrase checkbox: visible only with `text` schema | selecting keyword/integer/… the checkbox disappears; with text it reappears | `pi-phrase-wrap` hidden ↔ schema ≠ text |
| PI3 | `pi3_phrase` : text, phrase_matching ✓ | created | row `pi3_phrase` / **`text (+phrase)`**, verified against the API |
| PI4 | Empty payload field → Create disabled | `pi-create` disabled with empty input | re-enabled by typing |
| PI5 | Create `pi5_upd` as keyword, then change it to integer | drop+recreate → `updated` | green alert "Index type changed: 'pi5_upd' is now integer."; table updated |
| PI6 | Delete the existing `pi5_upd` index | 200 | success alert; unsampled `pi5_upd` row disappears; delete input cleared |
| PI7 | Delete a non-existent index: `nope_never_indexed` | 404 | error alert with **server detail only**, naming the field and collection (no `HTTP 404` prefix); field NOT cleared; no crash |
| PI8 | Empty delete field → Delete disabled | `pi-delete` disabled with empty input or no active collection | re-enabled by typing |
| PI9 | **[API]** POST payload-indexes `{field_name:"f_status", field_schema:"keyword", phrase_matching:true}` (curl) | **400** "phrase_matching requires field_schema 'text'" — readable message in detail | n/a (UI prevents the case) |
| PI10 | Switch active after a create/delete index | `pi-create-result`/`pi-del-result` **cleared** (K5) | result scoped to the active collection |
| PI12 | Re-submit identical: `f_status` : keyword | skipped | **info** alert "No change" / "already indexed" |
| PI15 | Schema column `—` for sampled non-indexed fields | `f_extra` (and `text`) present in the table with `—` | correct keys+indexes merge |

### 6.5 X — 06 Export Collection

| ID | Action | Expected |
|---|---|---|
| X1 | **Export** on `e2e_a_keep` (10 points) | JSON download; `ce-result` success "Exported 10 points from '…'"; each chunk = stored payload + trailing `point_id` (the Qdrant point id — **not** `id`, which the payload now carries itself); metadata with `scope: entire-collection` (**not** a misleading `limit`) |
| X2 | Export the owned reindex fixture spanning multiple server scroll pages | Run seed:reindex first. Browser download contains the exact expected IDs, count, text and f_idx payloads, without duplicates or omissions; raw vectors absent. The case fails when its large-fixture prerequisite is missing |
| X3 | Export on `e2e_a_empty` (0 points) | **warning** alert "Collection is empty — nothing to export." and **no download** |
| X4 | **Export with an INCOMPATIBLE active collection** (state E2) | Export **remains operable** (`ce-wrap` outside the lockdown) and completes normally |
| X5 | `POST /points/export` fails (e.g. 502 mid-walk on the server) | error alert with the server's **detail**; no file downloaded |
| X6 | Export with synthetic `Content-Disposition` headers (API mocked; real console assets) | Browser prefers the UTF-8 filename, falls back for malformed percent encoding, and saves the exact response bytes; no Qdrant writes |
| X7 | Export during a service **outage** | `panel-collections` inert ⇒ button not actionable |

### 6.6 D — 07 Delete (DESTRUCTIVE)

> D3 asserts the active fallback (old D4) inline after the delete. D4 =
> "protected names never preselected" (old D7). D5/D6 are separate spec tests.

| ID | Action | Expected |
|---|---|---|
| D1 | Name in `cd-name`, confirmation **empty** | Delete disabled |
| D2 | Confirmation **wrong** | Delete disabled; hint "Does not match" |
| D3 | Confirmation **exact** `e2e_a_new4` | 200; disappears from dropdown; active falls back to an existing collection; name+confirmation fields cleared; report 01 of the new active |
| D4 | **Protected** collection | the delete form never preselects a collection (protected names stay safe) |
| D5 | Post-delete: Query tab | field/filter dropdowns consistent with the new active; no residue from the victim; run query does not crash |
| D6 | Delete a non-existent collection: `cd-name`+confirmation = `e2e_ghost` | 404; readable error alert "collection 'e2e_ghost' not found" |


<a id="export"></a>

### 6.7 Export, download, storage and re-import

Layer: UI-live/API-live; filesystem and controlled-scroll failures use Backend.
Export is a payload walk, not a transactional Qdrant snapshot. Do not promise
snapshot consistency under concurrent writes. Each page-loop case checks the
actual next-offset sequence, not just the number of scroll calls.

| Case | Data and action | Required result |
|---|---|---|
| TB-EXP-001 | Export the immutable 2,001-point fixture through UI; API body/file destinations separately | Three server pages with unique advancing offsets, no missing/duplicate IDs, all expected payloads/Unicode and `point_id` strings, metadata n_chunks=returned=2001. Raw vectors absent. Compare semantic content across separate requests; generated timestamps/elapsed/nonce can differ |
| TB-EXP-002 | Empty fixture: UI export, API body, API file | UI warns that the collection is empty and starts no download; API body contains chunks=[], n_chunks=0; file response ok=true, path=null, n_chunks=0 and no new filesystem file. No indefinite download wait |
| TB-EXP-003 | Real Unicode collection `e2e_book_à漢`; also Backend filename inputs containing slash, backslash, quotes, CR/LF, dots, reserved Windows stem and >120 UTF-8 bytes | Live Content-Disposition is ASCII/Latin-1 safe, has fallback filename plus UTF-8 filename*, browser chooses Unicode filename and bytes parse. Backend generator returns bounded basename; export target rejects escape/symlink targets. Do not depend on Qdrant accepting hostile collection names to test filesystem safety |
| TB-EXP-004 | Backend fixed generated document: body vs file bytes; two same-second exports; write/fsync/rename failure; failure on second scroll page after retry exhaustion | Body/file delivery of the **same** document is byte-identical; unique filenames do not overwrite earlier files. I/O failure 502, no completed partial file, temporary file removed; scroll failure produces no success attachment/path. Preserve pre-existing completed exports |
| TB-EXP-005 | Ingest file `e2e_book_source.json` with chunks `[{id:1,text:"alpha à漢",topic:"a"},{id:2,text:"beta",topic:"b"}]` into roundtrip A; export; transform downloaded chunks with `({point_id,source_id,...chunk}) => chunk`, wrap as `{chunks}`, save with the original filename and upload to empty B | Default ordinal/source mapping gives two points with the same deterministic IDs, texts, topics, ordinal and source_id as A. Source_id is rebuilt from the filename; exported point_id/envelope metadata are not imported as ordinary payload. Re-upload to B does not duplicate |
| TB-EXP-006 | Repeat with stored ordinal key `chunk_no` and source key `document`, source type keyword; export | Download preserves chunk_no and point_id. Direct unmodified re-import fails missing integer `id`; transform each chunk via `({point_id,document,chunk_no,...rest}) => ({...rest,id:chunk_no})`, wrap as `{chunks}`, select ordinal output key chunk_no/source output key document and preserve original filename stem. B then matches A's IDs/payloads/count. Renaming a stored ordinal is not automatic import-schema detection |
| TB-EXP-007 | Query-results download of UUID point IDs after ingest, then attempt ingest of that file | `point_id` remains UUID text and never overwrites integer ordinal. A file with valid integer id can be mapped as in TB-EXP-005; a point-only download without ordinal needs an explicit deterministic integer assignment and source filename, yielding new ingest IDs. UI explains/rejects invalid input instead of treating UUID as ordinal |
| TB-EXP-008 | Backend cap=3: count 3 then stable three-point export; count 4; API-live small fixture with isolated cap where available | At cap succeeds; preflight over cap 413 before scroll/file writes. UI renders error, starts no successful download and recovers on selecting a within-cap fixture |
| TB-EXP-009 | **Regression — audit P2·8.** Backend preflight count 2, then pages totaling 4 with cap=3; separately one-point payload above a 4096-byte test budget and multibyte payload straddling that boundary | Incremental point/UTF-8 byte budget aborts with 413 before publishing an attachment/file; no partial export, no further page retrieval after violation. EXPORT_MAX_BYTES counts the entire serialized UTF-8 document, including metadata and escaping; see the middleware README for its default and concurrency limit |
| TB-EXP-010 | **Regression — audit P2·8.** Backend control JSON serialization with a large document while issuing an unrelated health/read request | Serialization work cannot block the event loop for the whole export; a scheduled heartbeat progresses before serialization is released. Use event barriers to prove scheduling, plus recorded live latency as supporting evidence, not a hardware-specific fixed timing assertion |
| TB-EXP-011 | UI-mock hold export response; switch A→alt, release A; repeat server 502, network abort and malformed body | Old result is not presented as alt's export; no partial/success download on failed operation; button recovers and retry uses current target. Record any already completed A download as A, never relabel it. Client abort alone says nothing about continued server work |

For TB-EXP-005/006, source identity comes from the upload filename stem in
the console. The export's generated filename must **not** silently become a
new source identity: rename the upload explicitly. Keep the source mapping and
ordinal mapping in the case evidence. The console's editable mapping controls
rename output fields; they do not convert an arbitrary `chunk_no` input into
the required input `id` automatically. There is no mapping exclusion checkbox:
remove export-only/source fields in the explicit file transformation above to
avoid a duplicate synthetic source row or extra payload fields.

---

## 7. Series I / U / B / G / DL / PL — POINTS tab

Global guard: incompatible active collection or services down ⇒ `#panel-points`
**inert**. Active reindex ⇒ **differential** lockdown (series RE9–RE12).

### 7.1 I — Ingest (preflight + editable mapping table)

**Preflight table** — columns `FIELD` (read-only) · `NAME` (text input) · `TYPE` (select).

- **Synthetic first row `source_id`**: `FIELD` = `source_id` (fixed), `NAME`
  editable (default `source_id`), `TYPE` default **`keyword`**. The value is
  always the **file name stem** (shown in the "source file" badge).
- **Second synthetic row `id`** — the chunker's per-source ordinal, which is
  ordinary payload, not the point id (that is `UUID5(source_id:id)`). `FIELD` =
  `id` (fixed), `NAME` editable (default `id`), **`TYPE` fixed to `integer` and
  disabled** — the middleware writes the ordinal back unconverted. The NAME
  travels as the `chunk_id_key` request field, so the server *relocates* the
  ordinal into that key: a rename leaves no `id` duplicate in the payload, and
  the chunks themselves are sent untouched (`_validate_chunks` 400s unless every
  chunk still carries an integer `id`).
- Then one row per payload field in the file, excluding **`id`, `text`**
  (`INGEST_EXCLUDED_KEYS`) — the two keys the middleware reads off every chunk,
  which is also why no rename may TARGET either. `CHUNK_EXCLUDE_FIELDS` is
  currently empty backend-side, so `overflow_details`,
  `overlap_sentences_count` and `split_metadata` reach the payload and do get a
  row.
- `id` NAME rejects `text` and the three chunker keys
  (`INGEST_CHUNK_ID_RESERVED_KEYS`, aligned with `_validate_chunk_id_key`), plus
  any name already taken by the `source_id` row or a field row. `id` itself is
  the default and stays legal.
- The five keys `id`, `text`, `overflow_details`, `overlap_sentences_count`,
  `split_metadata` stay reserved for the `source_id` NAME
  (`INGEST_SOURCE_ID_RESERVED_KEYS`, aligned with the backend's `source_id_key`
  validation).
- **TYPE options:** `keyword`, `text`, `integer`, `float`, `bool`,
  `datetime`, `uuid`, `geo`, `— untyped`.
- **Default TYPE** from inferred type: `string→keyword`, `int→integer`,
  `float→float`, `bool→bool`; **everything else** (array, object, mixed, null)
  → **`— untyped`**.
- Preflight tiles: `chunks`, `valid`, `invalid` + context row
  `target collection` / `source file`.

**Coercion rules (applied on Confirm, for chunks that contain the field):**

| TYPE | Accepts | Result |
|---|---|---|
| `keyword` / `text` | string, number, boolean, **null** | `String(value)` (`null` → `"null"`) — array/object **rejected** |
| `integer` | integer number, string `^[+-]?\d+$` | integer — non-integer float **rejected** |
| `float` | finite number, non-empty numeric string | float — `NaN`/`Infinity` **rejected** |
| `bool` | boolean, `0`/`1`, `"true"`/`"false"` (case-insensitive, trim) | boolean |
| `datetime` | string with valid `Date.parse` | string unchanged |
| `uuid` | string matching the UUID regex | string unchanged |
| `geo` | non-array, non-null object | passthrough |
| `— untyped` | anything | passthrough unchanged |

| ID | Action | Expected |
|---|---|---|
| I1 | **Upload file** `e2e_chunks_ok.json` | auto-preflight (no Analyze button); chunks/valid/invalid tiles; context row with target collection and file name; mapping table rendered |
| I2 | Verify **`source_id` row** | first row, FIELD `source_id` read-only, NAME=`source_id`, TYPE=`keyword` |
| I3 | Verify field rows and default TYPE | `m_str→keyword`, `m_int→integer`, `m_num→float`, `m_bool→bool`; `m_date`/`m_uuid`/`m_geo` (inferred string/object) → `keyword`/`— untyped` per inference |
| I4 | Upload `e2e_chunks_excluded.json` | rows are `source_id`, `id` (synthetic, TYPE disabled), then `m_str`, `overflow_details`, `overlap_sentences_count`, `split_metadata`; `text` has **no** row |
| I5 | Mixed/null type field in the file | default TYPE **`— untyped`** |
| I6 | **Confirm** with valid default mapping | chunked ingest with progress; correct count; `skipped` tile if the backend skips duplicates |
| I7 | **Rename** a NAME | **live validation without re-render** (focus stays in the input while typing); Confirm stays enabled |
| I8 | NAME **empty** (field row) | error `field '<orig>': name is required.`; **Confirm disabled** |
| I9 | Reserved NAME — rename a field row onto **`id`** or **`text`** | error `name '<x>' is reserved` for each; Confirm disabled. The other three are valid field NAMEs here (I12 still blocks all 5 for `source_id`) |
| I10 | NAME **duplicated** across two rows | error naming the duplicated field; Confirm disabled |
| I11 | **`source_id`** row NAME empty | error `source_id: name is required.`; Confirm disabled |
| I12 | `source_id` NAME = reserved key | error `source_id: name '<x>' is reserved` |
| I13 | `source_id` NAME that **collides** with a field | error `source_id: name '<x>' duplicates field '<f>'` |
| I14 | `source_id` TYPE = `integer` with a non-numeric file stem | error `source_id → integer: '<stem>' is not convertible`; Confirm disabled |
| I15 | `source_id` TYPE = `uuid` with a non-UUID stem | same error pattern |
| I16 | TYPE `integer` on `m_int` of `e2e_chunks_mixed.json` (contains `"abc"`) | error `field 'm_int' → integer: value "abc" is not convertible` (**only the first** offending value per field, not one per chunk); Confirm disabled |
| I17 | TYPE `integer` on a non-integer float (e.g. `1.5`) | coercion error |
| I18 | TYPE `float` on an empty / non-numeric string | coercion error |
| I19 | TYPE `bool` on a value other than bool/`0`/`1`/`"true"`/`"false"` | coercion error |
| I20 | TYPE `keyword`/`text` on an **array/object** | error (lossy, rejected) |
| I21 | TYPE `keyword` on **null** | **passes**, becomes the string `"null"` (intended behavior — verify it is not mistaken for an error) |
| I22 | TYPE `datetime` on a non-parseable string | coercion error |
| I23 | TYPE `uuid` on a non-UUID string | coercion error |
| I24 | TYPE `geo` on a scalar or array | coercion error |
| I25 | TYPE `— untyped` on a field of any shape | passthrough unchanged; no error |
| I26 | `m_partial` (present in 10 of 25 chunks) with a typed TYPE | chunks lacking the field remain without it; **no error** |
| I27 | Fix all errors | errors disappear; Confirm re-enabled; ingest applies rename **and** coercion. **N.B.** ingest assigns **deterministic UUIDs** to points: the chunk's `id` field does NOT become the point id, so verification must locate the point via scroll/Query (for payload) and not via Get on the chunk's `id` |
| I28 | **[MOCK] Switch file during** ingest | client stops scheduling subsequent batches from the old file; preflight re-aligned. Does not prove rollback/cancellation of a batch already accepted by the server |
| I29 | **[MOCK] Switch active collection during** ingest | client stops scheduling old batches, sends none to the new collection, and discards stale results; an already accepted server operation may still finish |
| I30 | **Malformed** JSON file (`e2e_chunks_bad.json`) | readable error; no preflight; Confirm hidden |
| I31 | File with invalid chunks (`e2e_chunks_invalid.json`) | preflight **blocked** (invalid>0); Confirm **hidden/disabled** |
| I32 | File with 0 chunks | preflight blocked; Confirm not available |
| I33 | **[MOCK] Mid-batch failure** (second batch returns `ok:false`) | progress `failed` with "Ingest failed after N of M chunks — <detail>"; mapping table reappears with edits preserved; does not simulate a real Qdrant stop |
| I34 | Retry after I33 | Confirm re-submits from the restored state without re-uploading the file |
| I35 | Ingest **completed** | preflight **emptied** (not just hidden): a second click on Confirm cannot re-send stale chunks; report and collection fields reloaded |
| I36 | ~5s after completion | the COMPLETED block auto-resets (`ing-result` emptied), like in Re-embed |
| I37 | Switch active collection within the 5s timer | the reset does **not** clear the new collection's result |
| I38 | `id` row: NAME `id`→`id_chunk`, then Confirm | request carries `chunk_id_key: "id_chunk"`; the sent chunks are **untouched** (integer `id`, no `id_chunk`); the middleware relocates the ordinal, so the stored payload has `id_chunk` and no `id` |
| I39 | `id` row NAME: empty → `split_metadata` → `source_id` → `m_str` → back to `id` | `id: name is required` / `is reserved` / `duplicates the source_id name` / `duplicates field 'm_str'`; Confirm disabled throughout, re-enabled at `id` |

> The I series runs with **"Create payload indexes for typed fields after
> ingest" unchecked** (see §7.1b): otherwise I6/I27 would index the sandbox
> collection and the later cases would meet type-locked rows.

### 7.1b IX — Ingest → auto-created payload indexes

After a **fully successful** ingest the console creates one payload index per
typed mapping row — under its **final** name, plus the chunk ordinal as
`integer` — in a single `POST /collections/{name}/payload-indexes`. A field the
collection **already** indexes is never requested: its TYPE is pinned to the
live schema in the preflight (tooltip *"Indexed as `<schema>` — change it in
Collections → Payload Indexes"*) and it is excluded from the request, because
the middleware would otherwise **replace** an index whose type differs (drop +
recreate, reported as `updated`). The toggle above Confirm is on by default and
resets to on with every new file. Sandbox: `e2e_b_idx` (each case recreates it).

| ID | Action | Expected |
|---|---|---|
| IX1 | Real ingest with `prezzo` integer, `attivo` bool, `tag` keyword, `autore` — untyped | chips `created` for `prezzo`, `attivo`, `tag`, `source_id`, `id`; no chip for `autore`; Qdrant holds exactly those five indexes |
| IX2 | Same file, toggle **unchecked** | no `payload-indexes` request at all; no summary block; collection has no index |
| IX3 | Ingest that fails on the 2nd of 3 batches | progress `failed`; no `payload-indexes` request; no index created |
| IX4 | `tag` already indexed as `keyword` | preflight: `tag` TYPE disabled, value `keyword`, tooltip explains; request omits `tag`, includes `prezzo`; summary shows `tag` as `already indexed` |
| IX5 | Rename `prezzo` → `tag` (indexed) and back | the row locks to `keyword` in place on rename, and releases back to the chosen `integer` on rename away — without re-rendering the table |
| IX6 | Race: the manager answers `updated: ["prezzo"]` | amber chip `replaced (existing index had a different type)` + warning pointing at Collections → Payload Indexes |
| IX7 | The index call answers **409** | ingest stays `COMPLETED`; the requested fields show as `failed` with the manager's detail |

### 7.2 U — Upsert

> U1–U7 map to tests in `c-points.spec.js`; U6 asserts authoritative top-level text.
> Double-click prevention is covered separately by K1.

| ID | Action | Expected |
|---|---|---|
| U1 | point_id uuid + text + payload `{}` | Upsert enabled; 200; point with dense+sparse |
| U2 | integer point_id | 200 |
| U3 | Invalid JSON payload | parse error in `up-result`; no API call |
| U4 | id **or** text empty | Upsert **disabled** (requires both) |
| U5 | Switch active during upsert | result **discarded** (no message applied to the new collection) |
| U6 | top-level text `"base text"`, payload contains `text:"payload override"` | 200; stored `payload.text` is **`"base text"`**, matching the text embedded. |
| U7 | Same id, new text/payload | overwrite, not duplication; count unchanged; GET shows new version |

### 7.3 B — Batch

> B1–B8 map to spec tests. Double-click prevention is covered by K1.
>
> **Every request-model failure is a 400, never a 422** — `create_app` installs
> a `RequestValidationError` handler that normalises FastAPI's default. The
> PI9 likewise expects 400. A 422 only
> reaches the client from an explicit `HTTPException(422)` raised deeper in,
> such as the filter compiler (see NF-B8).
>
> Point ids must be an **unsigned integer or a UUID**: a slug is refused as a
> malformed `point_id` before the case under test is reached, so B4–B8 use
> well-formed ids throughout.

| ID | Action | Expected |
|---|---|---|
| B1 | JSON array of N valid points | 200; N points added (used for seeding) |
| B2 | `[]` empty | no-op / readable error "Input must be a non-empty JSON array of points." |
| B3 | Non-array or malformed JSON | parse error |
| B4 | Array with non-object items / missing fields / empty text | readable 400; no writes |
| B5 | Invalid id | 400; no writes |
| B6 | Duplicate IDs in the same batch | 400 `points contains duplicate point_id values`; no writes |
| B7 | Non-object payload in an item | readable 400; no crash/raw JSON |
| B8 | Batch with one existing and one new id | existing overwritten, new inserted; count grows by only 1 |

### 7.4 G — Get

> G1–G5 map to spec tests. Stale-response discarding is covered separately by K2.

| ID | Action | Expected |
|---|---|---|
| G1 | Single get existing id, `with_vectors` unchecked | 200; payload displayed without vector data; Get enabled only with non-empty id. Vector summaries are covered separately by NF-P15/NF-P16 |
| G2 | Single get non-existent id | readable "not found" |
| G3 | Get many: id per line or CSV | `get-many-count` updated; 200 multi |
| G4 | Get many empty textarea | submit disabled; count **hidden** (not "0 id(s)") |
| G5 | Payload with HTML (`<script>`) | shown as escaped text; no DOM/console execution |

### 7.5 DL — Delete

> DL1–DL9 map directly to spec tests in `c-points.spec.js`. DL10–DL12 are new
> tests filling the old DL3/DL6/DL7 gaps. **DL8** (match-all `{}`) is now
> **refused** (UI disabled + API 400), not executed — corrected from Rev. 2.

| ID | Action | Expected |
|---|---|---|
| DL1 | Single delete existing id | 200; point removed; id field cleared; report -1 |
| DL2 | Single delete empty id | disabled |
| DL3 | Bulk delete: ids + **exact** confirmation | `del-count` shows "N id(s) parsed."; 200; points removed; input and confirmation cleared; counter hidden |
| DL4 | Bulk delete wrong/empty confirmation | disabled; hint "Does not match" |
| DL5 | Delete by filter: JSON filter → **Count matches** | correct count; Count enabled only with a filter |
| DL6 | Delete by filter exact confirmation → **Delete by filter** | 200; only matches removed (`e2e_c_victim`) |
| DL7 | Delete by filter wrong confirmation | disabled |
| DL8 | Filter `{}` (match-all) + confirmation | **refused** — UI disabled; API 400 "filters must contain at least one condition" |
| DL9 | Switch active after typing the confirmation | confirmation no longer valid for the new collection (label updated); button disabled |
| DL10 | Single delete non-existent | idempotent 200; success clears the field, point count stays unchanged |
| DL11 | Bulk submits one existing and one absent valid ID | success: `deleted=1` counts the actual deletion; point count decreases by one |
| DL12 | Bulk with an invalid id | 400; no deletes executed (see the §7.3 note: request-model failures are 400) |

> Bulk delete is **`DELETE /points`** with a JSON body — there is no
> `/points/bulk` route, and that path falls through to
> `DELETE /points/{point_id}` with `point_id="bulk"`, producing a misleading
> malformed-id error.

### 7.6 PL — Payload

> PL1–PL8 map to spec tests; PL7 asserts atomic rejection of text mutation.
> Selected stale-response cases are covered separately by K2.

| ID | Action | Expected |
|---|---|---|
| PL1 | Mode **MERGE (PATCH)** on existing id | 200; payload preserved + merge |
| PL2 | Mode **REPLACE (PUT)** | 200; payload replaced; `text` and `source_id` automatically preserved |
| PL3 | Invalid JSON payload | parse error; no API call |
| PL4 | Delete keys: keys per line/CSV | `pl-keys-count`; 200; keys removed; enabled only with keys |
| PL5 | Clear payload exact confirmation | 200; only `text` + `source_id` remain (protected) |
| PL6 | Clear payload wrong/empty confirmation | disabled; hint |
| PL7 | REPLACE including `text`, `source_id` and other new fields | **400** because `text` requires an embedding upsert; the entire request is refused and the existing payload stays unchanged. A request changing `source_id` alone is not rejected by the text guard |
| PL8 | PATCH/PUT non-existent point | readable 404; no implicit creation |


<a id="payload"></a>

### 7.7 Payload ownership, ID fidelity and ingest

Reset `e2e_book_write` for each independent mutation. Use `has_id:[1,2]` for
filtered scope and point 3 as a non-match control. Layer: UI-live plus API-live
variants for syntax the UI cannot submit. Backend counters are required where
the oracle says zero embedding/write calls.

| Case | Data and action | Required result |
|---|---|---|
| TB-PAY-001 | Upsert ID 1 with top-level text `base text`, payload `{text:"spoof", ordinary:"new"}`; repeat through Batch IDs 1 and 2 | Stored text is `base text`, ordinary field is `new`; captured embedder input is top-level text. Batch does not reintroduce the opposite dict precedence. Repairs main U6 |
| TB-PAY-002 | PATCH, PUT, filtered MERGE with `{text:"spoof", ordinary:"new"}` | HTTP 400, readable service-owned-field message; complete snapshots unchanged for targets and control. Repairs PL7; no metadata subset is applied before rejection |
| TB-PAY-003 | Delete keys, both point and filter scopes: plain `text`, `source_id`; JSON-path quoted forms; `"text"[]`, `"text".sub`, `text.sub`; path containing literal `"\u0074ext"` or `"\u0073ource_id"`; fully escaped quoted root | Each protected-root variant 400, unchanged payload. Distinguish a literal backslash-u in the path from JSON transport decoding: capture raw body as well as decoded key. Include ordinary key in the same request to prove atomic rejection |
| TB-PAY-004 | Both delete scopes: blank/whitespace path, NUL suffix, unmatched quote, malformed array path; then valid `ordinary` and `nested.leaf` | Malformed syntax rejected 400 before writes. Valid ordinary paths removed only from selected targets; protected fields and control unchanged. A literal backslash field is a separate payload name, not proof of bypassing `text` protection |
| TB-PAY-005 | PATCH ordinary field; PUT `{ordinary:"replacement"}`; PUT `{source_id:"e2e_book_changed"}`; single-point Clear | PATCH retains other keys; PUT removes omitted ordinary fields but preserves omitted text/source_id. Explicit source_id replacement remains allowed by the current contract. Clear retains only protected fields actually present; none is fabricated |
| TB-PAY-006 | Filter Clear on `e2e_book_sample`, filter source_id=e2e_book_sample; verify all 1001 afterward | `ordinary` removed throughout, text/source_id preserved; `rare` on point 1001 survives because it was outside the sampled key set. UI reference explains this limit. Do not assert full arbitrary-key clearing |
| TB-ID-001 | Batch contains point IDs `"1"`/`"01"`/`"0001"`, or upper/lower UUID spellings of the same UUID | 400 for each duplicate batch, before embedding/writes; point count/payload unchanged. Duplicate detection follows canonical IDs rather than original spelling |
| TB-ID-002 | Upsert/Get/Get-many/query/download/delete IDs `"0"`, `"9007199254740991"`, `"9007199254740993"`, `"18446744073709551615"`, and a UUID | Each ID round-trips exactly as text/canonical UUID; no rounding, duplicate collapse or wrong-point mutation. Assert raw response/download ID representation as well as rendered text. Large decimal values in JSON inputs are quoted |
| TB-ID-003 | Negative point ID, uint64 max+1, fractional point ID, malformed UUID and valid+invalid batch pair | 400 and no writes. Separate client validation from server rejection in results; do not accept an unhandled exception or an arbitrary 4xx as the expected contract |
| TB-ID-004 | Ingest chunk `id` values true, `"1"`, null, fraction; duplicate integer IDs; `text` null/number/object/empty/whitespace; valid first + invalid second chunk | API returns400 before embedding/writes. Browser preflight blocks malformed types/text, unsafe integer ordinals and duplicate ordinals before submission. Chunk ordinal strings are not converted from numeric aliases. This differs from string point IDs used by Batch |
| TB-ID-005 | Ingest signed integer ordinals -1, 0, 1 with valid text; repeat source and ordinals | Accepted by the current integer-ordinal contract; deterministic UUID point IDs, count remains 3 after repeat. A negative ordinal is not a negative Qdrant point ID; do not invent a non-negative restriction |
| TB-ING-001 | Editable mapping for every type: keyword/text numeric scalar, integer `"42"`, float `"1.5"`, bool `"true"`, datetime RFC 3339, UUID, geo object, untyped nested object | Successful request has the selected types and renamed keys; stored payload and generated index type agree. Control all failures separately: integer `"abc"`, bad UUID, invalid bool, blank numeric string, structured value mapped to keyword/text. UI rejects coercion with row-specific messages before ingest |
| TB-ING-002 | Rename ordinary field to another mapped field, ordinal to text/source key, source key to ordinal/text, blank name; choose untyped for an optional field; toggle type after error | Collision/reserved/blank name blocks confirmation and sends no ingest. Untyped preserves the value and requests no new index; it does not drop the field. Correcting mapping clears its error and updates planned indexes, without stale rows. Removing a field requires editing the input file, not an invented exclusion checkbox |
| TB-ING-003 | Ingest succeeds then index phase fails on one new field; repair the missing index through Collections → Payload Indexes; change collection during delayed index result | Show ingested count and distinct index failure; do not label data ingestion as rolled back or blindly repeat it. Manual repair changes only the missing/failed index. Old collection response cannot mutate new collection's UI/index list. No dedicated ingest index-retry button is assumed |
| TB-ING-004 | API-live `replace_source=true`: for each source type keyword/integer/float/datetime/uuid/bool/text/untyped use source `e2e-book`/`42`/`1.5`/`2026-01-01T00:00:00Z`/UUID 1/`true`/`e2ebook`/`e2e-book`; ingest ordinals 1,2, then only 2 with changed text | Source point count 2→1; surviving point has new text, unrelated source unchanged; the `source_id` index follows the middleware's declared mapping (`points.py` `_SOURCE_ID_INDEX_SCHEMAS`): keyword, `text` and `untyped` sources are indexed as `keyword`, the other types as themselves. Float and datetime predicates tested via backend call capture for exact typed range/UTC semantics. Console ingestion remains additive; it has no replace-source switch |
| TB-ING-005 | Backend, shared per-app coordinator: concurrent same-source replacements with disjoint ordinals; hold first mutation at a barrier, start second, release sequentially | Second cannot enter mutation body until first exits; final state is one complete latest replacement, never empty/mixed. Repeat mutation versus reindex start. Required coordinator omission fails loudly rather than constructing an independent lock |
| TB-ING-006 | Backend failure injection: final embedding batch fails; later Qdrant upload fails; prune fails after successful upload | Embedding failure produces zero writes. Upload failure reports partial progress accurately and never prunes old source; prune failure does not report a wholly successful replacement. Preserve original source/control points as the operation's phase contract requires |

For TB-ID-002/004, unsafe JSON **number** inputs also need a dedicated UI-mock
acceptance target: detect precision loss and reject with a readable request to
use string point IDs, or preserve the lexical value with lossless parsing.
Silent rounding is a failure. For ingest ordinals beyond the browser's exact
integer range, rejection before submission is an acceptable UI boundary;
do not change the server's signed-integer contract merely to mirror JavaScript.

---

## 8. Series Q / QF — QUERY tab

Guard: incompatible collection or services down ⇒ `#panel-query` **inert** and
`ql-query-mode` hidden.

### 8.1 Q — Hybrid search controls and selected combinations

| ID | Combination | Expected |
|---|---|---|
| Q1 | fusion **rrf**, rrf_k OFF, weights OFF, rerank ON, normalize ON | coherent request JSON; ranked results |
| Q2 | rrf + **Set rrf_k ✓** | `rrf_k` in the JSON; range 1..10000 respected |
| Q3 | rrf + **Set rrf_weights ✓**: slider sparse◄►dense + weight scale | caption "Qdrant weights [x, y]" updated **live**; weights in the JSON |
| Q4 | fusion **dbsf** | rrf_k irrelevant/disabled; JSON uses dbsf |
| Q5 | dbsf + candidate_limit = 1 | info: "With one candidate, each DBSF branch normalizes it to 0.5, making fusion tie-prone; increase candidate_limit." |
| Q6 | rerank OFF | `rerank_top_k`/`normalize` not applied |
| Q7 | rerank ON + **Set rerank_top_k ✓** | `rerank_top_k` in the JSON, bounded by the runtime query configuration and `candidate_limit` |
| Q8 | rerank ON + **normalize OFF** | `rerank_normalize=false` |
| Q9 | limit / candidate_limit at the boundaries (1, max) and out of range | `required`/min/max validation; Run blocked on invalid input. **N.B.** both fields are **disabled by design** when the query is empty (payload-only mode): they govern only hybrid search |
| Q10 | **Empty query + filters present** | caption "HYBRID SEARCH IS DISABLED — PAYLOAD-ONLY MODE FILTERING"; filtered retrieval (scroll) ok |
| Q11 | **Empty query + no filters** | info: "Empty query and no filters: the first page of all points will be returned, ordered by ID." |
| Q12 | **Returned Payload**: select/deselect `with_payload` fields | JSON and results reflect the selection; `text` is forced/disabled only for hybrid search with reranking. Payload-only mode allows deselecting it |
| Q13 | **Download results as JSON ✓** → Run | file downloaded **in addition to** the render |
| Q14 | **[MOCK]** Run with a server error | readable `.alert-error` in `ql-results`; retry succeeds when the mock is removed |
| Q15 | Preview `ql-request-json` | reflects **live** every control change before Run |
| Q16 | Switch active collection after a Run | previous results not presented as current; filters and payload-keys reset |

### 8.2 QF — Payload filters (full matrix)

Mode: query field **empty** → banner "PAYLOAD-ONLY MODE", button "Retrieve points",
`POST /points/scroll` (fixed page size 5). The expected count is read from the
`returned` tile (= total) and the page count.

> **Acceptance fixture: `e2e_book_filters` (§0.2).** The
> [detailed filter cases (§8.3)](#filters) define the common browser procedure,
> missing operator branches and validation cases. Every count below applies
> only to that 100-point fixture, never to
> the 12-point `e2e_d_query` used by the current executable spec.
>
> `d-filters.spec.js` has 19 consolidated tests, with **partial** coverage of
> this matrix. Expectations such as `ids((i) => i > 9)` depend on the fixed
> 12-point fixture definition; changing the seed still requires reviewing them.
>
> Missing cases extend beyond integer *any/except* (A5/A6): polygon holes,
> negative phrase results, UUID lists, open array-count bounds, and nested
> same-element semantics are also absent from this spec. `F12` builds an
> ordinary logical group, and `QF-A` only checks that the nested button is
> visible. Most C-series validation rows have no browser assertion here.
> The executable inventory below and §16 make those limits explicit.
>
> The `A`/`B`/`C` sub-series below are local to §8.2. They are unrelated to
> series B (Batch, §7.3), C (Create, §6.2) and D (Delete, §6.6) — `C1` means
> two different cases depending on the section. The specs sidestep the clash by
> naming these tests `F01`–`F12`, `QF-A`–`QF-D` and `Q01`–`Q02`, so book IDs in
> this section alone do not appear in test names.

**Current executable inventory** (`d-filters.spec.js`):

| Spec ID | What its body asserts |
|---|---|
| F01 | Keyword eq/any/except, exact returned ID sets |
| F02 | Text all-words/any-words and one successful phrase |
| F03 | Integer equality and gt/gte/lt/lte/between; no integer any/except |
| F03b | Exclusive integer range |
| F04 | Float gte and between |
| F05 | Both boolean values |
| F06 | Radius with partial membership checks, box and closed polygon with exact IDs; no hole |
| F07 | Datetime between and one UUID equality; no UUID list |
| F08 | Array count with two bounds, including exclusive bounds; no open-bound case |
| F09 | Empty arrays and explicit null distinct from missing; no `is_empty` on nullable/missing fields |
| F10 | Point-ID list of small numeric IDs |
| F11 | MUST plus MUST NOT, and two SHOULD rows with minimum 1/2 |
| F12 | One logical group; no same-element group |
| QF-A | Add condition and logical group; nested button visibility only |
| QF-B | Root min-should control shown/hidden as SHOULD rows change |
| QF-C | Unindexed `f_extra` absent from field options |
| QF-D | Phrase option disabled with an explanatory title when its index lacks phrase matching |
| Q01 | Payload-only filter and exact returned IDs |
| Q02 | Filter retained across RRF with/without rerank and DBSF with rerank; no weighted-RRF branch despite its title |

**Series A — operator × type:**

| ID | Filter row (GROUP=MUST unless noted) | Expected | UI notes |
|---|---|---|---|
| A1 | `status` equals `active` | **20** | 4 pages |
| A2 | `status` any of `active, deleted` | **30** | |
| A3 | `status` any value except `active, archived, draft` | **10** | scalar field; array semantics differ, see TB-FIL-002 |
| A4 | `views` equals `50` | **1** | 1 page, "End of results" |
| A5 | `views` any of `1, 2, 3` | **3** | |
| A6 | `views` any value except `1, 2, 3` | **97** | |
| A7 | `views` greater than `90` | **10** | |
| A8 | `views` at least `90` | **11** | |
| A9 | `views` less than `11` | **10** | |
| A10 | `views` at most `10` | **10** | |
| A11 | `views` between `10`–`19` | **10** | |
| A12 | `price` greater than `50.5` | **50** | float: no eq/any |
| A13 | `price` between `10.5`–`19.5` | **10** | |
| A14 | `flag` is `true` | **50** | select true/false |
| A15 | `description` contains all words `quick fox` | **15** | |
| A16 | `description` contains any word `quick lazy` | **40** | |
| A17 | `description` exact phrase `quick brown fox` | **15** | option enabled (phrase index) |
| A18 | `description` exact phrase `fox brown quick` | **0** | coherent empty state |
| A19 | `notes` contains all words `gamma` | **30** | |
| A20 | `notes`: "exact phrase" option **disabled** (tooltip: index lacks phrase_matching) | — | UI only |
| A21 | `location` within radius: lon `9.19` lat `45.46` r `1000` | **25** | |
| A22 | `location` within box: TL `8`,`47` → BR `14`,`40` | **60** | |
| A23 | `location` within polygon `6 50; 16 50; 16 36; 6 36` | **60** | ring auto-closed |
| A24 | same as A23 + hole `\| 11 43; 14 43; 14 40.5; 11 40.5` | **25** | Rome excluded by the hole |
| A25 | `created_at` at least `2026-04-01T00:00:00Z` | **10** | full RFC 3339 value |
| A26 | `created_at` between `2026-01-01T00:00:00Z`–`2026-01-31T00:00:00Z` | **31** | inclusive bounds |
| A27 | `doc_uuid` equals `00000000-0000-4000-8000-000000000001` | **1** | |
| A28 | `doc_uuid` any of (uuid 1..5, CSV) | **5** | |
| A29 | `tags` array size between `2`–`2` | **40** | |
| A30 | `tags` array size min `2`, max empty | **60** | open bound ok |
| A31 | `tags` is empty | **10** | |
| A32 | `optional_note` is null | **10** | |
| A33 | `optional_note` is empty | **20** | absent+null |
| A34 | `(point id)` any of `1, 2, 3` | **3** | has_id |

**Series B — combinations and modes:**

| ID | Setup | Expected | Notes |
|---|---|---|---|
| B1 | MUST `status`=`active` + MUST `views` ≤ `10` | **10** | AND |
| B2 | MUST NOT `status`=`active` | **80** | |
| B3 | SHOULD `views` ≤ `20` + SHOULD `flag` is `true` (min_should=1) | **60** | union 20+50−10 |
| B4 | same as B3 but `min should matches` = 2 | **10** | even ≤20 |
| B5 | MUST `tags` any `red` + MUST NOT `flag` is `true` | **45** | odd 11–100 |
| B6 | MUST·SAME ELEMENT `items[].label`=`a` + MUST·SAME ELEMENT `items[].qty` ≥ `90` | **11** | nested, same element |
| B7 | same conditions as B6 but both plain MUST | **21** | i≤10 ∪ i≥90 — difference vs B6 |
| B8 | Only 1 SHOULD row → `min should matches` input **hidden** | — | appears only with at least two SHOULD rows |
| B9 | Hybrid query: query `ricetta di cucina italiana` + MUST `status`=`active`, limit 5 | ≤5 hits, all `status=active` | score bars, tiles fusion/reranked/candidates |
| B10 | Payload-only + `with_payload`: deselect all except `status` | payload card = only `status` | text is selectable in this mode; forced only for hybrid rerank |
| B11 | `status`=`draft` (40) → pagination: Next to page 8, then Previous back to 1 | 8 pages ×5 | "Page x of 8", End of results at p.8, Previous disabled at p.1 |
| B12 | No filters, empty query | **100**, 20 pages | info alert "first page of all points", ordered by id |

**Series C — edge cases and validation:**

| ID | Action | Expected |
|---|---|---|
| C1 | `views` between `20`–`10` | error "between min cannot exceed max", Run **disabled** |
| C2 | `views` between `20`–(empty) | error "between needs both min and max" |
| C3 | `created_at` at least `2026-13-99` | error "bound must be an RFC 3339 datetime" (per-component range validation, not just shape) |
| C4 | polygon `0 0; 1 1` (2 points) | error "needs at least 3 distinct points" |
| C5 | polygon not closed (A23 without repeating the first point) | auto-closure: request JSON with first==last, count 60 |
| C6 | polygon with malformed hole (`\| 1 1; 2`) | error "polygon ring 2: …" |
| C7 | geo radius r=`-5` | error "radius must be positive (meters)" |
| C8 | geo radius lat=`91` | error "lat must be between -90 and 90" |
| C9 | Selected scalar/list operation with empty value | readable "value is required" error; Run disabled (`is_empty`/`is_null` require no value and remain valid) |
| C10 | Fill a value, then change OPERATION | values cleared |
| C11 | Fill a value, then change FIELD | values cleared + operation renormalized |
| C12 | `views` equals `abc` | error "value must be an integer" |
| C13 | `views` any of `1, abc` | error "values must all be an integer" |
| C14 | `flag` without true/false selection | "value is required" error; Run disabled |
| C15 | `(point id)` any of `abc` | error "ids must be unsigned integers or UUIDs" |
| C16 | `doc_uuid` equals `not-a-uuid` | error "value must be a UUID" |
| C17 | 2 SHOULD rows, min_should=2, then remove 1 SHOULD row | effective minimum returns to 1; control hidden; request has no stale min_count=2 |
| C18 | payload-only: any `limit` in the UI | request JSON `limit: 5` fixed (page size) |
| C19 | hybrid query: limit 10 > candidate_limit 5 | error "candidate_limit must be ≥ limit", Run disabled |
| C20 | Switch collection with active filters on the other's fields | filter tree and old results reset; options loaded for the new collection, as checked by Q16 |
| C21 | Switch collection during an in-flight request | old collection's results NOT rendered |

**Additional QF cases:**

| ID | Control | Expected |
|---|---|---|
| QF-A | **+ Condition** / **+ Logical group** / **+ Same-element group** | rows built; "Same-element group" enabled only in valid context |
| QF-B | min-should | `ql-minshould` appears only with ≥2 SHOULD rows |
| QF-C | Inspect filter field choices for **non-indexed** `f_extra` | field not offered; the browser case does not submit a backend request for it |
| QF-D | "exact phrase" filter on a text field **without** a phrase index | option not offered / readable error (contrast with PI2 which enables it) |


<a id="filters"></a>

### 8.3 Filters: full operator/type and validation coverage

Layer: UI-live for positive semantics, UI-mock for request-count validation
and delayed replies. Run §8.2 A1–A34, B1–B12 and C1–C21 on
`e2e_book_filters`, using the following common procedure: clear the query text,
reset the filter tree, add the listed condition/group through the UI, inspect
the compiled request, retrieve all pages, and compare IDs, total and page
boundaries. Each legacy row needs its own qualified result. Additional rows
below cover missing operators and untested branches, not just missing labels.

The allowed operator matrix is pinned to
[models.js](../../dist/assets/models.js). For every schema, assert exactly this
set, including disabled phrase matching where applicable. Internal operator
keys identify controls unambiguously; use the displayed labels when clicking.

| Schema | Non-universal operators | Universal operators |
|---|---|---|
| keyword | `eq`, `any`, `except` | `values_count`, `is_empty`, `is_null` |
| integer | `eq`, `any`, `except`, `gt`, `gte`, `lt`, `lte`, `between`, `range` | same three |
| float | `gt`, `gte`, `lt`, `lte`, `between`, `range` | same three |
| bool | `eq_bool` | same three |
| text | `text`, `text_any`, `phrase` | same three |
| geo | `geo_radius`, `geo_box`, `geo_polygon` | same three |
| datetime | `gt`, `gte`, `lt`, `lte`, `between`, `range` | same three |
| uuid | `eq`, `any`, `except` | same three |
| point ID | `id_any` | none |
| vector presence | `has_vector` | none |

Qdrant distinguishes matching any array element from matching one nested
object; `except` on an array is not a blanket exclusion of that array.
Null, missing and empty arrays also need distinct cases. These semantic
checks follow the official [filter reference](https://qdrant.tech/documentation/search/filtering/)
and the repository's filter builders.

| Case | Data and action | Exact oracle / required observation |
|---|---|---|
| TB-FIL-001 | Inspect every row in the operator matrix, with `description` and `notes` as the two text variants | No float equality, bool range or non-indexed metadata field; pseudo-fields always offered. Phrase enabled only for the phrase index. Selecting a different type resets incompatible operators/values |
| TB-FIL-002 | On `tags`, select `except red`; separately use MUST NOT `tags any red` | First IDs 41..100 (60): each has a non-red value. Second IDs 1..10 (10). This prevents mislabelling `except` as "none of" |
| TB-FIL-003 | Integer any `1,2,3`, except `1,2,3`; UUID any UUIDs 1..5, except UUIDs 1..5 | Exact IDs 1..3, 4..100, 1..5, 6..100 respectively. Repeat keyword/integer/UUID lists using valid JSON-array input; inspect string versus integer wire types |
| TB-FIL-004 | `price` with `gt/gte/lt/lte 50.5`; custom range lower 10.5 and upper 19.5 in all four inclusive/exclusive combinations | Counts 50/51/49/50. Range IDs 10..19, 11..19, 10..18, 11..18 respectively. Single lower >99.5 gives ID 100; single upper <2.5 gives ID 1 |
| TB-FIL-005 | `views` custom range lower 10, upper 19 in all four inclusive/exclusive combinations; open >99 and <2; equal inclusive bounds 50/50 | Same range sets as TB-FIL-004; IDs 100, 1, 50 for the three remaining variants |
| TB-FIL-006 | `created_at` with `gt/gte/lt/lte` at day 50 (`2026-02-19T00:00:00Z`); four custom-bound combinations day 10..19 | Counts 50/51/49/50; range IDs as TB-FIL-004. Repeat day-50 threshold as `2026-02-19T01:00:00+01:00`: same sets. Open >day99 yields ID100, <day2 yields ID1, inclusive day50/day50 yields ID50; use generator day(i) for full RFC3339 values |
| TB-FIL-007 | `flag=false`; text all `quick fox`, any `quick lazy`, phrase `quick brown fox`, reversed phrase `fox brown quick`; attempt phrase on `notes` | IDs odd, 1..15, 1..40, 1..15, empty respectively. Disabled phrase choice explains missing phrase index and cannot send a phrase request |
| TB-FIL-008 | Main A21..A24 geometry, repeat polygon with explicit closure and automatic closure, then add the Rome hole | Exact sets 1..25, 1..60, 1..60, 1..25. Request rings have matching first/last coordinate and correct longitude/latitude orientation; auto-closure adds one endpoint, not a duplicate ring |
| TB-FIL-009 | On each of the eight fields of `e2e_book_edges`, restrict point IDs to 1..3 and run values_count: equal 0, equal 1, equal 2, lower-only ≥1, upper-only ≤1, exclusive >0/<2 | Exact sets `{1}`, `{2}`, `{3}`, `{2,3}`, `{1,2}`, `{2}` for every type. Scalar ID 6 has count 1; include it in a separate explicit scalar variant |
| TB-FIL-010 | On every edge field run `is_empty`, then `is_null`, without any value input | `is_empty` IDs `{1,4,5}`; `is_null` ID `{5}`. Changing from a scalar operator to a unary operator removes the value requirement |
| TB-FIL-011 | Main B6 nested label=a and qty≥90, then flat B7; negate the entire nested group | Nested IDs 90..100; flat IDs 1..10 plus 90..100; NOT nested IDs 1..89. Wire filter uses one nested array root and relative child paths, not independent nested conditions |
| TB-FIL-012 | MUST status=active AND logical group (SHOULD views≤5, SHOULD views≥15); nest that group inside another MUST group; set inner minimum to 2 then back to 1 | IDs 1..5 plus 15..20 (11); nested parentheses preserve set; minimum 2 returns empty. Root and child minima do not overwrite each other |
| TB-FIL-013 | Choose point-ID and vector pseudo-fields inside/outside a same-element group; switch array root and remove a populated group | Nested selectors only offer indexed descendants valid for that group; invalid point-ID nesting is unavailable/blocked. Removed children contribute no conditions. No stale field path survives a root change |
| TB-FIL-014 | Main C1..C21 plus numeric fraction in integer input; `NaN`/`Infinity` in float; negative/fractional array count; both count bounds empty; UUID JSON list containing a number; malformed JSON list | Required invalid variants disable Run with a field-associated readable error and issue **zero** search/scroll requests. Correcting one value re-enables Run and sends only the corrected filter |
| TB-FIL-015 | Geo longitude 181, latitude -91, radius 0, repeated-point ring with <3 distinct vertices, malformed second ring; datetime invalid date/leap-day/timezone | Same no-request validation; retain the user's input for correction. Valid leap day `2024-02-29T00:00:00Z` is accepted even when it returns no matches |
| TB-FIL-016 | Capture runtime filter-depth/condition maxima; construct at-limit and limit+1 trees through the UI; set minimum SHOULD > number of children | At-limit tree compiles; excess is rejected with the configured bound, no request. Removing a SHOULD child clamps the effective minimum; <2 children hides the control and removes stale min_count from the body |

For TB-FIL-014/015, if a malformed variant reaches the API, record a frontend
validation failure even when the server rejects it. Do not replace a desired
no-request assertion with “any 4xx”. Out-of-range calendar tests must be based
on actual instants, not lexicographic comparison of differing timezone strings.

<a id="query"></a>

### 8.4 Query controls, vector presence, count and facet

Layer: UI-live unless otherwise marked. Use the immutable filter fixture;
repeat control serialization with UI-mock to avoid asserting unstable ANN ranks.

| Case | Action and variants | Required result |
|---|---|---|
| TB-QRY-001 | `(vector presence)`: inspect offered dense/sparse names against the fresh fixture's report; select each and combine with status=active; inspect a nonexistent name | Each offered populated vector matches 100 points, with status filter20; nonexistent name not offered. First valid name is auto-selected, so no artificial blank-option UI action. Separate Model variant with empty vector value returns validation error |
| TB-QRY-002 | Repeat vector presence after reindex completes and fields/report refresh; use TB-REI-004 for intermediate vector presence | Active generation names are selectable and return every point; removed generation not retained. The selector is seeded from config.js at startup; `tb-reindex-live.spec.js` compares it with the report's `active_vectors` after a completed job and fails if they diverge (a same-profile regeneration keeps the names, so this run is not proof of a cross-profile refresh) |
| TB-QRY-003 | Cross `fusion=rrf/dbsf` × `rerank=true/false` × `with_payload=all/selected` with query `cooking`, status=active, limit=5, candidates=20 | Eight results recorded; ≤5 hits all active, no duplicate IDs; body retains the same filter. Text forced into projection only when rerank requires it; fusion/rerank indicators agree with the submitted options |
| TB-QRY-004 | RRF default versus weights `[0.8,0.2]` and k=60; switch to DBSF and back; toggle rerank off/on | UI-mock body maps dense then sparse weights, uses RRF-only options only with RRF, and restores valid state after switches. Run valid variants live for acceptance; never assert that changing weights must change rank on this dataset |
| TB-QRY-005 | Runtime limit/candidate/top-k boundaries: minimum, configured maximum, max+1, zero, fraction, candidate<limit and top-k>candidate | Valid boundary bodies accepted; invalid UI states block Run and identify the actual bound. Record fetched config, not a hardcoded 200. Payload-only remains fixed at five and does not send hybrid-only controls |
| TB-QRY-006 | Payload-only status=draft, visit all eight pages and return; change filter to active; order by views ascending then descending with limit=5; hybrid pagination separately | Unordered draft has 40 unique IDs; changing filter resets offsets/page 1. Ordered page IDs 1..5 or 100..96 without filter; no unsupported Next cursor is fabricated. Hybrid offset is independent of scroll cursor |
| TB-QRY-007 | Query download in unordered payload-only, ordered payload-only, hybrid (rerank on/off) | File contains respectively all matching pages, the ordered returned window, or the hybrid returned window. Exact `point_id` strings and payload projection agree with displayed/requested mode; no silent claim that a top-k file is a whole-collection export |
| TB-QRY-008 | UI-mock deterministic scores 0.9/0.5/0.1 with threshold 0.5 and offset 0/1; clear threshold and reset offset, then run valid options live | Mocked request carries exact threshold/offset; UI labels reset, zero-result case is coherent. Backend forwarding/ranking-stage assertions belong to TB-API-002; do not infer their correctness from rendered mock scores |
| TB-QRY-009 | Count with/without status=active, exact on/off; facet status/flag/doc_uuid/views with exact on and limit100; repeat facet limit2 and filter status=active | Exact counts 100/20; exact status values 20/30/40/10, bool 50/50, 100 unique UUID/views each1. Limit2 returns two entries; status facet with active filter returns active:20 only. Approximate mode is labelled and forwarded; do not require equality on arbitrary larger datasets |
| TB-QRY-010 | Facet key dropdown, unsupported text/geo/float/datetime via API, no-index collection, empty collection, zero-match filter | UI offers keyword/integer/bool/uuid indexes only; no-index state explains absence. Unsupported API facet type returns400 with field-specific reason. Empty/zero-match yields no facet values and count 0, without stale rows from the previous collection |

---

## 9. Series M — MONITORING tab

**Read-only:** the MONITORING tab mutates neither collections nor points.

M1–M19 are a checklist mapped to the named M-A/M-B/etc. tests, not 19
independently executed cases. Coverage is partial: M-B2 checks only the Qdrant
badge's consistency when reachable; M-F checks chart visibility. M-G uses the
real console with intercepted endpoints and 125 deterministic samples to verify
120 retained samples, eviction/order, hover values and pointer-leave cleanup.
M21 watches `/stats` and `/health` for 401s only. See §16 for historical gaps;
the remaining badge and metric-value checks still need separate evidence.

### 9.1 Badges and model tiles

| ID | Action | Expected |
|---|---|---|
| M1 | Open `#monitoring` | MONITORING tab selected, panel visible, polling started |
| M2 | Health badges on first tick | 5 tiles: livez UP, readyz UP, qdrant UP, embedder UP, health UP |
| M3 | Badge ↔ API comparison | each badge consistent with the real HTTP status of the endpoint |
| M4 | Model tiles | profile + dense/sparse/reranker shown when embedder is reachable |
| M5 | Embedder unknown/unreachable | tiles replaced by a readable info alert |

### 9.2 Statistics and chart

| ID | Action | Expected |
|---|---|---|
| M6 | Stats tiles | uptime, total_requests, rejected_requests present; total relative to the baseline |
| M7 | Chart first sample | caption "Collecting samples — the chart appears after the second refresh."; no SVG until series < 2 |
| M8 | Chart after ≥ 2 samples | SVG with polyline, area, dot, title "total requests", first/last timestamp ticks |
| M9 | Hover on the chart | crosshair visible + tooltip `hh:mm:ss · N requests`; leave → hidden |
| M10 | Series cap | series kept at max 120 samples (`SERIES_MAX`) |
| M11 | Reset counters | baseline realigned to current raw values; series cleared; total_requests visible → ~0; immediate tick |

### 9.3 Metrics table and polling

| ID | Action | Expected |
|---|---|---|
| M12 | Metrics table | columns endpoint/requests/errors/p50/p95; at least one row after activity |
| M13 | Documented filter | only endpoints present in openapi.json and with requests > 0 after reset |
| M14 | p50/p95 | values from histogram when present, `—` otherwise; no NaN |
| M15 | Prometheus parsing | `#` comments ignored, malformed lines discarded, no exceptions |
| M16 | Polling 5 s active | multiple ticks while the tab stays open (one /stats request every ~5 s) |
| M17 | Stop on tab exit | leaving MONITORING the timer stops: no new ticks |
| M18 | No overlap | a slow tick does not queue parallel ticks (`monitorTickRunning` guard) |
| M19 | Re-entry | re-entering the tab polling restarts without duplicate timers |

### 9.4 Lifecycle and profile

| ID | Action | Expected |
|---|---|---|
| M20 | JS console | zero exceptions; only deliberate 4xx elsewhere. Covered by K6-style console-error guard in `e-monitoring.spec.js` |
| M21 | Monitoring requests | no 401 on `/stats` and `/health` during the observed polling window; on the loopback publish this book targets these monitoring endpoints are public, so this is not a bearer-auth proof |
| M22 | Escaping | detail/dynamic values escaped, no DOM injection. Verified via route mock with malicious payload |
| M-A | Open Monitoring | `mon-badges` live (livez/readyz/qdrant/embedder/health) |
| M-B | **Embedder Models** | `mon-models` lists dense/sparse/reranker of the **active profile** |
| M-C | **[SWITCH→advanced]** + Re-test + Monitoring | models and `embedder_profile` reflect `advanced` |
| M-D | **[SWITCH→basic]** | return to `basic` |
| M-E | First refresh (single sample) | caption "Collecting samples — the chart appears after the second refresh." |
| M-F | From the second refresh onward | chart drawn; crosshair and tooltip on hover |
| M-G | Long session (>120 samples) | series **truncated at 120** points (no unbounded growth) |
| M-H | Leave the Monitoring tab and re-enter | polling **stopped** on exit and restarted on re-entry; **no duplicate timer** |
| M-I | Slow backend during polling | overlapping ticks **skipped** (`monitorTickRunning`): no out-of-order render |
| M-J | **Reset counters** | counters zeroed; the series/chart restarts consistently |
| M-K | Per-endpoint metrics | `mon-table` (endpoint, requests, errors, p50, p95) populated after exercising query/points; counts consistent with the actions taken |


<a id="monitoring"></a>

### 9.5 Monitoring and readiness with measurable oracles

Layer: UI-mock for deterministic timing/data; one UI-live coherence pass per
active profile. Freeze the browser clock and fulfill all five monitoring routes
(`/livez`, `/readyz`, `/health`, `/stats`, `/metrics`) plus OpenAPI as appropriate;
do not leave four routes accidentally talking to real services.

| Case | Controlled data and action | Required result |
|---|---|---|
| TB-MON-001 | Feed 125 completed ticks five seconds apart with total_requests=1..125, starting with no samples; inspect chart after ticks 1,120,121,125 | Stored/rendered sample count 1/120/120/120; at tick125 exactly samples6..125 remain, in timestamp order. Inspect polyline point coordinates or an equivalent observable series representation, not count of polyline elements. Distinct values/times make eviction/order verifiable |
| TB-MON-002 | Hover first/middle/last retained samples then leave chart; test zero samples and constant-valued samples | Tooltip reports that sample's time/value, crosshair aligns; leave hides it; empty/constant scale has no NaN/Infinity coordinates or exceptions. Do not assert only an SVG exists |
| TB-MON-003 | Stats total10, then Reset UI view, then total13; compare metrics body before/after reset | Display delta3 and chart reset with no prior samples carried over; no server reset/mutation request; server counter remains13. Endpoint deltas reset coherently. New browser context starts without inheriting an old view baseline |
| TB-MON-004 | Metrics fixture below, then no histogram, malformed unrelated lines, zero-count endpoint, and OpenAPI failure | `/query/search`: requests10, errors2, p50 `0.3000`, p95 `0.4800` seconds; absent histogram shows `—`. Invalid unrelated lines ignored without throwing. Zero-request rows suppressed; OpenAPI success limits rows to documented paths, failure does not blank all valid metrics |
| TB-MON-005 | Coherent healthy responses; readiness503 while livez200; unhealthy dependency; network error; recovery; BASIC and ADVANCED model descriptors | Each badge/tile agrees with its own endpoint semantics; liveness is not readiness. Service banner has precedence over incompatibility and disappears on recovery without lingering inert layers. Model tiles show current profile and absent optional models as absent, never stale BASIC values under ADVANCED |
| TB-MON-006 | Hold one poll response across two intervals; navigate away/back; hide/show page where polling lifecycle handles visibility | No overlapping aggregate poll while previous tick is pending; tab re-entry leaves one timer, not duplicates. Stale/failed samples do not overwrite fresh health with success. Verify actual request times/counts, not a source constant |
| TB-MON-007 | Change total_requests from100 to2 (backend restart), with and without an active Reset baseline | UI must not show negative counters/NaN or stale profile. Record current post-reset clamp-to-zero behaviour when raw count is below baseline; do not claim the reset view automatically rebases to2. A future rebasing change requires an explicit new oracle |

Metrics fixture for TB-MON-004 (OpenAPI includes `/query/search`):

```text
qdrant_manager_requests_total{endpoint="/query/search",status="200"} 8
qdrant_manager_requests_total{endpoint="/query/search",status="502"} 2
qdrant_manager_request_duration_seconds_bucket{endpoint="/query/search",le="0.1"} 0
qdrant_manager_request_duration_seconds_bucket{endpoint="/query/search",le="0.5"} 10
qdrant_manager_request_duration_seconds_bucket{endpoint="/query/search",le="+Inf"} 10
```

---

## 10. Series N — Navigation, hash and keyboard

| ID | Action | Expected |
|---|---|---|
| N0 | Open the console at a 320 px viewport | title and top-bar brand use Smart Semantic Searcher; the brand stays inside the viewport |
| N1 | Click the 4 tabs | corresponding panel visible, others `hidden`; single `aria-selected` |
| N2 | Direct open with `#collections`, `#points`, `#monitoring`, `#query` | the corresponding tab is active on load |
| N3 | **Unknown** hash (`#pippo`) or absent | fallback to **query** |
| N4 | Tab switch | URL updated with `history.replaceState` (the hash changes **without** adding history entries — Back must not navigate between tabs) |
| N5 | Manual hash edit in the address bar | `hashchange` activates the corresponding tab |
| N6 | Focus on a tab + **ArrowRight / ArrowLeft** | moves **and activates** the adjacent tab, with wrap (last→first, first→last) |
| N7 | **Home** / **End** | first / last tab |
| N8 | Roving tabindex | only the selected tab has `tabIndex=0`; others `-1`; **Tab** exits the tablist instead of traversing the 4 tabs |
| N9 | The **API key** button is not part of the tablist | Arrow does not reach it (it is outside `role="tablist"`) |
| N10 | Enter/exit Monitoring via keyboard | polling started/stopped as on click (cross-ref M-H) |
| N11 | Points subtabs (Ingest/Upsert/Batch/Get/Delete/Payload/Count) | click changes panel; `aria-selected` consistent; each of the 7 subtabs carries `aria-controls` for its panel and each panel `aria-labelledby` for its subtab |

---

## 11. Series K — Cross-cutting (busy, stale, auto-refresh)

To be verified **by sampling on every operation**, not just once.

The rows below are acceptance targets, not proof that every listed operation
is exercised. K1 in Points covers ten submit buttons, NF-B11 the three filtered
writes, and the Query/Collections K1 cases cover Run/Create. K2 in Points covers
five operations plus the separate U5 test; Query has its own K2. Neither
`fixtures.js` nor `watchConsole()` enforces K6 automatically across the suite:
each spec must actually assert the collected errors.

| ID | Behavior | How to verify | Expected |
|---|---|---|---|
| K1 | **Double-click prevention** (`withBusy`) | rapid double-click on: `ing-confirm-submit`, `up-submit`, `bt-submit`, `get-submit`, `get-many-submit`, `del-submit`, `del-bulk-submit`, `delf-submit`, `pl-submit`, `pl-del-keys-submit`, `pl-clear-submit`, `cc-submit`, `cu-submit`, `pi-create`, `pi-delete`, `cd-submit`, `ce-export`, `ql-run`, `cr-preflight`, `cr-start` | the button disables during the operation; **only one** request is sent |
| K2 | **Stale response discarding** | start a slow operation and switch active collection before the response — on upsert, batch, get, get-many, delete, bulk delete, delete-by-filter, payload update/keys/clear, export, ingest, reindex, query | the old collection's result is **not** rendered as if it were the new one's |
| K3 | **Auto-refresh of Report 01** | after: create, recreate, schema toggles, create/delete index, upsert, batch, ingest, delete points, delete-by-filter, reindex completed | `cc-report` (points_count/badge) updated **without** manual action |
| K4 | **Refresh of payload keys/indexed fields** | after ingest/upsert with new payload fields, then index creation | `with_payload` reflects discovered keys; Query filters include them only once indexed. Dedicated subsets exist in `b-ingest.spec.js` / `b2-ingest-indexes.spec.js`; TB-UI-009 defines the full observation |
| K5 | **Result scoping** | after every active collection switch | `cr-result`, `cr-preflight-result`, `cu-result`, `pi-create-result`, `pi-del-result` **cleared**; create/delete collection results (03/07) **not** cleared (they are tied to the typed name, not the active collection) |
| K6 | **No console errors** | DevTools open for the entire run | zero unhandled errors/exceptions (excluding deliberate HTTP noise: induced 409/404/500) |
| K7 | **Destructive confirmations** | all 6 (recreate, delete collection, bulk delete, delete-by-filter, clear payload, re-embed start) | label shows the expected name; trim/case handled consistently; hint "Does not match" visible only when needed |


<a id="crosscutting"></a>

### 11.1 Operation/state, accessibility and security matrices

Run **every row** of the operation inventory against TB-UI-001/002/003/004.
They are parameterized cases; one Upsert test does not cover the other rows.
Use `e2e_book_write` as A, `e2e_book_alt` as B; typed-name operations use an
additional owned `e2e_book_named`. UI-mock controls timing/error variants;
UI-live verifies success postconditions after per-case reset.

| Operation variants | Principal action control | Success oracle / scope |
|---|---|---|
| Create / recreate | cc-submit | Name exists with requested schema / recreated empty; list/report refresh; scope typed name |
| Schema toggles | cu-submit | All five reported toggles match submitted values; active scope |
| Index create / delete | pi-create / pi-delete | Schema and filter/facet choices refresh; active scope |
| Collection delete | cd-submit | Typed name removed; active selection falls back coherently |
| Export | ce-export | TB-EXP byte/content oracle; active scope |
| Reindex preflight / start / pause / resume / cancel | cr-preflight / cr-start / cr-pause / cr-resume / cr-cancel | TB-REI states and scopes; preflight read-only |
| Ingest / Upsert / Batch | ing-confirm-submit / up-submit / bt-submit | Count/payload/type postconditions and report/field refresh; active scope |
| Get / Get-many | get-submit / get-many-submit | Exact found/missing IDs and payload/vector projection; active scope |
| Single / bulk / filtered delete | del-submit / del-bulk-submit / delf-submit | Selected IDs removed, control retained; active scope |
| Filtered-delete count / payload-filter count | delf-count-submit / pl-filter-count-submit | Exact matched count, zero mutations |
| Payload MERGE / REPLACE / keys / clear, point and filter scope where offered | pl-submit / pl-del-keys-submit / pl-clear-submit | TB-PAY ownership and non-match control; filtered REPLACE not offered |
| Count / Facet | cnt-submit / cnt-facet-submit | TB-QRY-009/010; active scope |
| Query run / page navigation / download | ql-run and visible page controls | Correct mode/window/IDs; reset cursor on new query/filter/collection |

| Case | Procedure applied to every inventory variant | Required result |
|---|---|---|
| TB-UI-001 | Hold relevant API response, double-click and press Enter while action pending, then release success; repeat failure | Exactly one principal operation request; action disabled/busy while pending, recovers according to validation/lock after success/failure. Index follow-up, polling and report refresh are separately counted, not misclassified as duplicate operations |
| TB-UI-002 | Hold A response, switch to B, release A, then initiate a B operation; repeat A→B→A while first A request is pending | No stale result overwrites the latest user intent, no stale field/report response resets B. Typed-name create/delete results retain explicit target. Record A→B→A as a distinct acceptance variant: collection-name equality alone cannot identify an obsolete request |
| TB-UI-003 | Simulate 401,429,504,502, malformed JSON and network failure one at a time; then recover and retry | Specific readable error, no false success, no uncaught rejection; controls recover and retry targets the current form. Preserve correction inputs unless success explicitly disarms them. No blind automatic replay of a destructive request with uncertain completion |
| TB-UI-004 | Exercise initial/empty, loading, error and success states at 320×720,768×1024,1440×900; use only keyboard for one pass | Visible labels, usable horizontal scrolling for wide tables, no clipped confirmation/action or page-level forced overflow; focused input/button reachable, focus visible, disabled reason discoverable, errors associated with their controls. JSON/result panels remain readable at 200% zoom |
| TB-UI-005 | For recreate, collection delete, bulk delete, filtered delete, clear point/filter payload and reindex start: blank, wrong case, exact, padded exact confirmation; switch collection/typed target; complete success | Console uses trimmed, case-sensitive equality; nonempty expected name required. Invalid states issue zero requests, scope change invalidates old target, success disarms. API confirmation equality is checked separately in TB-API-003. Single-point delete has no typed confirmation in the current UI |
| TB-UI-006 | Add/remove dynamic filter groups; edit mapping rows and replace the uploaded file; operate select/listbox/details/popover using Tab, Shift+Tab, Enter/Space, Escape | Focus remains on a live meaningful control after removal/closure/file replacement; no keyboard trap; labels include field/group context. Test nested filters, source/ordinal mapping and errors, not just static tab navigation |
| TB-UI-007 | Render payload/text, field name, error message and mocked collection label containing `<img src=x onerror=...>`, quotes, ampersands and Unicode | Content displayed as text; no script execution, unexpected network request or injected interactive DOM. Use an inert test marker callback, never a credential-exfiltration URL |
| TB-UI-008 | API key popover with synthetic token: save, reload, clear, wrong token, two independent browser contexts; inspect config.js, visible errors and captured artifact fields | Correct auth state and request header application, no cross-context credential contamination, no actual token in config.js/log/error/URL/screenshot. Clear removes the stored credential and updates auth-required UI; authenticated test context is not evidence for deliberately unauthenticated configuration |
| TB-UI-009 | After each data/index/schema mutation, observe automatic report and field refresh; introduce a new payload key and then index it | Counts/report refresh without manual action; payload projection offers discovered payload key, filter/facet selectors offer it only when indexed and schema eligible. Do not require an unindexed payload key to be filterable |

These are acceptance targets for observable console quality. A newly exposed
failure (for example the A→B→A request-generation case) is a product finding;
do not weaken the oracle because an older test checked only A→B.

---

## 12. Series Z — Full BASIC↔ADVANCED roundtrip

This is an **operator workflow**, not an automated roundtrip. `z4-switch.spec.js`
observes an already-switched stack and existing native/foreign collections; it
does not create/seed both profiles, restart services or perform Z7 cleanup.
The current `z6-e7.spec.js` migration specifically checks BASIC → ADVANCED.

| ID | Step | Expected |
|---|---|---|
| Z1 | Embedder **BASIC**: create `e2e_z_basic`, indexes (PI1), seed via Batch, hybrid query (Q1) | all green; consistent results |
| Z2 | **[SWITCH→advanced]**, Re-test | `e2e_z_basic` → incompatible profile switch: banner suppressed, Query inert, Points retrieval available with embedding writes blocked, 02 Re-embed live, **06 Export still live** (X4) |
| Z3 | Create a collection under ADVANCED, seed, query | operates normally while `e2e_z_basic` stays locked |
| Z4 | Active=`e2e_z_basic`, **Re-embed** (RE4→RE7) under ADVANCED; during the job open Points | RE9–RE11: mutations disabled, Get available; on job completion compat OK and Query/Points reactivated |
| Z5 | **[SWITCH→basic]**, Re-test | the ADVANCED-born collection **and** `e2e_z_basic` (re-embedded ADVANCED) → both incompatible profile switch; recoverable via Re-embed |
| Z6 | Monitoring after each switch | models/profile always aligned with the active profile (M-C/M-D) |
| Z7 | Cleanup: delete `e2e_a_new*`, `e2e_a_victim`, `e2e_a_empty` (never the protected ones) | dropdown clean; `e2e_z_basic`, `e2e_a_keep` and `e2e_d_query` left for inspection |


<a id="profiles"></a>

### 12.1 External-state profile and outage acceptance

Layer: UI-live/API-live with deliberate external service transitions. Follow
the launcher/profile procedure in this book and the excluded-suite recipe
in the README. One stack at a time. Record baseline hardware/profile before
starting and restore it in cleanup even if the middle phase fails. No switch
or outage is performed while editing this specification.

| Case | Steps | Required result |
|---|---|---|
| TB-PRO-001 | On BASIC create native fixture; switch to ADVANCED, create another native fixture; compare both; reindex BASIC fixture under ADVANCED; switch back to BASIC and reindex it again | Both directions observed: native compatible, foreign recoverable, profile-switch banner suppression matches §1.1; create-only/export/delete availability correct. Post-reindex payloads/IDs unchanged, contract/vectors match current runtime; queries and monitoring recover after each cutover |
| TB-PRO-002 | Separate same-profile fingerprint drift and unverified/foreign collection, plus each state combined with an unhealthy service response | Full §1.1 truth table: recoverable drift shows reindex guidance; unverified contract blocks reindex; service outage takes precedence. UI-mock proves matrix; live profile-switch result is not evidence for these different states |
| TB-PRO-003 | Stop embedder using the supported launcher procedure, keep middleware available, inspect livez/readyz/health and console; restore; repeat real Qdrant outage in an isolated test environment | Liveness remains distinct from readiness; operations gate according to dependency availability, visible explanation and eventual recovery. No stale incompatible overlay or multiple monitoring timers after recovery |
| TB-PRO-004 | Schedule TB-PRO-001 and focused ingest/query/reindex smoke for CPU/BASIC, CPU/ADVANCED, GPU/BASIC, GPU/ADVANCED | Four distinct result cells with actual hardware/model evidence. Unavailable hardware/profile is BLOCKED with reason, not silently converted to a mocked pass or included in an overall “all profiles green” claim |

---

## 13. Series NF — New Qdrant features exposed

Reference: the executable coverage audit in `nf-newfeatures.spec.js` and the
control-to-reference contracts described below.
Most cases are UI-driven, with API setup/postcondition reads. Declared `[API]`
cases exercise contracts directly, and NF-COV1/NF-COV2 inspect local source only.

### 13.1 NF-A — Points → Count (new subtab)

| ID | Action | Expected |
|---|---|---|
| NF-A1 | Open Points → **Count** subtab | panel visible; `aria-selected` consistent; filter pre-filled `{}`, `exact` ✓ |
| NF-A2 | **Count points** with filter `{}` | counts the **entire** collection; success alert with the real total and the word `exact` |
| NF-A3 | Filter on an indexed field (e.g. `f_flag = true`) | count = number of matching points (verifiable against the Query in payload-only) |
| NF-A4 | `exact` unchecked | response reports `estimated`; no error |
| NF-A5 | Malformed JSON filter | readable error in `cnt-result`; no request sent |
| NF-A6 | Filter on a **non-indexed** field | readable backend error (Qdrant requires the index); no crash |
| NF-A7 | Switch active collection | `cnt-result` **cleared** (K5) |
| NF-A8 | Double-click on **Count points** | only one request (K1) |
| NF-A9 | The Count subtab contains no destructive actions | no `btn-danger`, no confirmation field |

### 13.2 NF-B — Payload: scope POINT ID | FILTER

| ID | Action | Expected |
|---|---|---|
| NF-B1 | Points → Payload: default scope | **POINT ID** selected; `point_id` field visible; filter hidden |
| NF-B2 | Select **FILTER** | `point_id` field hidden, `filter JSON` visible; caption "Every point matching the filter…" |
| NF-B3 | In FILTER scope, **REPLACE** | REPLACE radio **disabled**, selection forced to MERGE; caption explains REPLACE requires a single point |
| NF-B4 | FILTER + **Count matches** | count of matching points, shown before writing |
| NF-B5 | FILTER + MERGE with payload `{k:v}` | 200; message reports **how many points** were updated; verify on Qdrant: only matches have the new key |
| NF-B6 | FILTER + **Delete keys** | key removed **only** from matching points; count in the message |
| NF-B7 | FILTER + **Clear payload** with exact confirmation | sampled non-protected keys removed only from matches; `text`/`source_id` retained when present. A key absent from the collection sample can survive; the current fixture proves the small, fully sampled case only |
| NF-B8 | FILTER with filter `{}` (match-all) | submit **disabled** on the UI side; **[API]** 400 "filters must contain at least one condition" |
| NF-B9 | FILTER + Delete keys on a protected key (`text`/`source_id`) | 400 "Cannot delete protected payload fields"; no writes |
| NF-B10 | Return to POINT ID scope | per-id merge behavior remains available (see the PL series) |
| NF-B11 | Double-click on the three submit buttons in FILTER scope | only one request per button (K1) |
| NF-B12 | Switch active collection in FILTER scope | results and count cleared (K5) |

### 13.3 NF-C — Query: order_by on payload-only scroll

| ID | Action | Expected |
|---|---|---|
| NF-C1 | Query with text (hybrid mode) | `Set order_by` **disabled**; `title` "Payload-only mode only…" |
| NF-C2 | Empty query (payload-only) | `Set order_by` enabled; key/direction/start_from unlock only after the checkbox |
| NF-C3 | The `key` options | **only range-indexed** integer/float/datetime fields (NF-X1), not every indexed field |
| NF-C4 | order_by `f_views` **desc** → Run | results in descending order on the field (not by id) |
| NF-C5 | order_by `f_views` **asc** | ascending order |
| NF-C6 | With order_by active | pagination does not offer **Next page**; the indicator reports "Ordered by <key> <dir>" instead of "Page N of M" |
| NF-C7 | `start_from` valued | the page starts from the indicated value |
| NF-C8 | order_by on a collection without indexed fields | select shows "no orderable field"; checkbox forced off; Run does not send `order_by` |
| NF-C9 | **[API]** `order_by` + `offset` together | **422** with a message pointing to `order_by.start_from` |
| NF-C10 | Preview `ql-request-json` with order_by active | contains `order_by` with key and direction |
| NF-C11 | **UI position** | `order_by` applies **only** in payload-only, so it is not in §01 Hybrid Search but in **§03 Returned Payload**, after the `with_payload` list and before §04: a proper block with `divider` + subhead "Result order" and `fieldset legend="order_by"` |

### 13.4 NF-D — Query: score_threshold and offset on hybrid search

| ID | Action | Expected |
|---|---|---|
| NF-D1 | Empty query (payload-only) | `Set score_threshold` and `Set offset` **disabled** (they act only on hybrid search) with `title` "Hybrid search only…" |
| NF-D2 | Query with text | both enabled; numeric fields stay disabled until the checkbox (enable-toggle) |
| NF-D3 | `score_threshold` with a low value (e.g. 0.1) | results ≥ threshold; no result below the threshold |
| NF-D4 | `score_threshold` high but reachable | **reduced** set vs the baseline, all above the threshold |
| NF-D5 | `score_threshold` impossible (e.g. 99) | **0 results** without error: readable empty response |
| NF-D6 | `offset` = 2 with limit unchanged | the results window **shifts**: the first 2 of the baseline disappear, later elements appear |
| NF-D7 | **Where the explanations live** | the page carries no inline captions for the 3 new controls; the **Parameter reference** contains an entry for `score_threshold`, `offset`, `order_by`, `start_from` with the facts (score is **fused**, **before rerank**, mode restrictions, `offset` incompatible with `order_by`, `start_from` inclusive). No example numbers |
| NF-D8 | Preview `ql-request-json` | contains `score_threshold` and `offset` only when their respective checkboxes are active |
| NF-D9 | Invalid values (negative offset) | Run blocked with a readable validation error |
| NF-D10 | Query snapshot | toggling the checkbox on/off, the state of the two fields is consistent after a tab switch |
| NF-D7b | **Tooltip on disabled controls in payload-only mode** | explains why `score_threshold` and `offset` are disabled (hybrid search only) |
| NF-D11 | **UI position** | `score_threshold` and `offset` act on the **fused** result, so they belong with the fusion block and **before** the rerank controls: the DOM order must read `fusion → rrf_k → score_threshold → offset → rerank → rerank_top_k` (pipeline: retrieve → fuse → cut/page → rerank) |

### 13.5 NF-R — Feature reference (Query · Collections · Points)

| ID | Verification | Expected |
|---|---|---|
| NF-R1 | **Query coverage** | a card exists for **every** control in the Query panel. Control→card map: `query` · `collection_name` · `limit` · `candidate_limit` · `fusion` · `rrf_k` · `rrf_weights` · `score_threshold` · `offset` · `rerank` · `rerank_top_k` · `rerank_normalize` · `filters — GROUP/FIELD/OPERATION/VALUE` · `min should matches` · `with_payload` · `order_by` · `start_from` · `Request JSON` · `Download results as JSON` · `Results` · `Availability`. The test **also fails** if a new control appears without a card |
| NF-R2 | **Query style consistency** | every description is **180–450 characters**, is prose (not a list), ends as a sentence, and the max/min ratio stays **< 2.6×** |
| NF-R3 | **Collections coverage** | a card exists for every control in the Collections panel. Includes: `phrase matching`, `payload index field to delete`, `06 Export Collection`, `type-to-confirm`, `Availability`. Also verifies that the **numbering** follows the panel: Export is 06 and Delete is 07 |
| NF-R4 | **Collections style consistency** | same band as Query: 180–450 characters, prose, complete sentence, spread max/min < 2.6× |
| NF-R5 | **Points coverage** | a card exists for every control in the Points panel, including the three additions (`update_mode`, `with_vectors`, `Facet`). The 7 inline captions removed from the page are already covered by existing cards — no information lost |
| NF-R6 | **Points style consistency** | same band as Query and Collections: 180–450 characters, prose, complete sentence, spread max/min < 2.6× |
| NF-R7 | **Monitoring coverage** | a card exists for every rendered badge/tile and for Reset counters, models, chart, metrics and auto-refresh |
| NF-R8 | **Monitoring style consistency** | same prose, length and completeness rules as the other three panels |

### 13.6 NF-X — Final Query page audit findings

| ID | Verification | Expected |
|---|---|---|
| NF-X1 | `order_by` keys | only fields with a **range index** (`integer`, `float`, `datetime`). A keyword/bool/geo/uuid/text field is indexed but **not orderable**: it must not even be offered |
| NF-X2 | Collection without orderable fields | the select shows "no orderable field", the checkbox is disarmed and the payload does not contain `order_by` |
| NF-X3 | `start_from` typed on the field | on `integer` a non-integer value, on `datetime` a non-parseable date → readable error that **names the field**, Run blocked; with a valid value the error disappears |
| NF-X4 | **[API]** `start_from` incompatible | **422** with a useful message — never a **500** (the ValidationError must be intercepted) |
| NF-X5 | `offset` at 0 with the toggle active | the payload contains `"offset": 0`: otherwise the preview would contradict the checkbox |
| NF-X6 | `offset` window exceeds `candidate_limit` | rejected with guidance to raise `candidate_limit`; a valid offset re-enables Run |
| NF-X6b | Valid offset window within the candidate budget but beyond the actual matching set | intended: empty page without error. No dedicated assertion in the current NF-X6 test |
| NF-X7 | Download consistent with the mode | payload-only unordered: **all** pages in the file. Ordered: **one** page, because Qdrant does not paginate an ordering with `offset` — the file contains exactly what is on screen |
| NF-X8 | `offset` field ceiling | the `max` attribute follows `candidate_limit` minus the window |
| NF-X9 | **[API]** Page tiling | consecutive pages exactly tile the same ranking as a single deep run; equal-score RRF hits use a deterministic point-id tie-break |
| NF-X10 | **[API]** `with_payload` omitted | returns **all** payload, not a fixed list of six keys |
| NF-X11 | **[API]** Echo in response | `offset` and `score_threshold` present; absent values are 0 and null |
| NF-X12 | **[API]** Search params and timeout per request | current test accepts `hnsw_ef`, `exact`, `rescore`, `oversampling` and `timeout: 30`, and rejects invalid `hnsw_ef`, `oversampling`, `timeout` → 400. Acceptance does not prove forwarding or enforcement; TB-API-002 specifies those separate assertions |

### 13.7 NF-Z — UI/UX theme conformance

| ID | Verification | Expected |
|---|---|---|
| NF-Z1 | New controls use theme patterns | `.field` / `.grid-3` / `fieldset+legend` / `.radio-row` / `.check` / `.caption`; no inline styles |
| NF-Z2 | Enable-toggle pattern | dependent fields start `disabled` and enable only with the checkbox |
| NF-Z3 | The Count subtab | uses the same `.subtabs` with `role="tab"` and panel `role="tabpanel"` + `hidden` |
| NF-Z4 | Result areas | `aria-live` on new result containers (`cnt-result`) |
| NF-Z5 | No new destructive action without type-to-confirm | Clear payload by filter reuses the existing confirmation |
| NF-Z6 | Focus ring | `:focus-visible` visible on all new controls |

### 13.8 NF-P — Points: facet, update_mode, with_vectors

| ID | Verification | Expected |
|---|---|---|
| NF-P1 | Count → **Facet**: select `key` | populated only with indexed fields **keyword / integer / bool / uuid** (`facetableFields`); `text`/`geo`/`float`/`datetime` do **not** appear |
| NF-P2 | Collection without facetable fields | select shows "no facetable field", **disabled**; **Get facet** disabled |
| NF-P3 | Facet on a keyword/bool field | `value \| count` table with all distinct values and their count; no points retrieved (read-only) |
| NF-P4 | `exact` unchecked | response still valid (approximate estimate on the Qdrant side) |
| NF-P5 | **[API]** Facet on a **non-indexed** field | **400** with Qdrant's real reason in the message, never a 502 with the raw body |
| NF-P6 | **[API]** `key` empty/whitespace | 400 "key cannot be blank" |
| NF-P7 | **[API]** `limit` out of range (0 or >1000) | 400 validation |
| NF-P8 | Switch active collection | `cnt-facet-result` cleared; select recomputed on the new collection's fields (K5) |
| NF-P9 | Upsert → `update_mode` | fieldset radio-row UPSERT / INSERT ONLY / UPDATE ONLY, default UPSERT (behavior unchanged) |
| NF-P10 | Upsert with `INSERT ONLY` on an **existing** `point_id` | no writes; **warning** alert "already exists — skipped (insert_only)"; the original payload stays intact |
| NF-P11 | Upsert with `UPDATE ONLY` on a **non-existent** `point_id` | no writes; warning alert "does not exist — skipped (update_only)"; the point is not created |
| NF-P12 | Upsert with `UPDATE ONLY` on an existing `point_id` | writes normally (same outcome as UPSERT) |
| NF-P13 | Batch → `update_mode` with a mix of existing/new ids | response reports **upserted** and **skipped** separately; points kept out by `insert_only`/`update_only` are the expected ones |
| NF-P14 | Batch entirely discarded (`insert_only` on only existing ids) | UI shows `upserted: 0`, `skipped: N`; current browser test does not observe embedder calls. The no-embedding contract needs separate backend instrumentation |
| NF-P15 | Get (single) → `with_vectors` | checkbox unchecked: no `vectors:` line; checked: line `vectors: <name> (dense, N-d) · <name> (sparse, M nonzero)`, never the raw vector numbers |
| NF-P16 | Get many → `with_vectors` | same summary for each point in the list; absent ids remain simply omitted |
| NF-P17 | `with_vectors` during and after re-embed | names/shapes agree with the report's **active** generation before and after cutover; do not assume a return to canonical names. Scripted in `tb-reindex-live.spec.js` (paused job → Get / Get-many with `with_vectors`; completed job → active names) |
| NF-P18 | Theme conformance | the three controls use `.field`/`fieldset+legend`/`.radio-row`/`.check`; the facet table uses `.table-wrap`+`<table>` like the other product tables; no inline styles |
| NF-P19 | Parameter reference | coverage and style verified by **NF-R5/NF-R6** alongside all other POINTS_HELP cards |
| NF-P20 | Regression | Upsert/Batch/Get/Get many/Count without touching the new controls behave exactly as before (§7 U/B/G, §13.1 NF-A) |

NF-P17 requires a real re-embed job and additional assertions in the RE series;
the existing tests do not make it redundant.

### 13.9 NF-COV — Executable coverage guard

| ID | Verification | Expected |
|---|---|---|
| NF-COV1 | Search spec source for each static control's `id`, `name`, `data-pt-tab` | fails for a missing **text reference**. A comment, an assertion of visibility, or an excluded spec can satisfy it; dynamically generated controls and action/branch coverage are not proved |
| NF-COV2 | Compare qualified book cases against the campaign reporter's actual collection/result ledger | Every case retained, including 96 TB cases; duplicate section-local IDs and digitless suffixes stay distinct. PASS requires all mapped variants to have passed. NOT_MAPPED, NOT_RUN, SKIPPED and FAIL remain explicit; this structural check does not assert complete execution or assertion quality |


<a id="api"></a>

### 13.10 Backend observability and open audit lifecycle findings

These rows complement browser coverage. Fakes/events must exercise the real
handler/batcher code; mocking the entire handler cannot prove its cancellation
or forwarding behaviour. Do not add product-only instrumentation to the UI.

| Case | Layer, action and controls | Required result |
|---|---|---|
| TB-API-001 | Backend: batch all existing IDs with insert_only, all absent IDs with update_only, then mixed batch | Fully skipped variants report upserted0/skippedN and invoke embedder/upsert zero times. Mixed variant embeds/writes only eligible IDs. API-live/UI result alone does not prove the skipped compute work |
| TB-API-002 | Backend capture `query_points`/prefetch arguments for hnsw_ef256, exact=true, rescore=false, oversampling4, timeout30; quantized/unquantized and omitted/override variants | Supplied options reach their documented Qdrant location; omitted values use settings; unquantized search has no quantization params. `rescore:false` is an explicit false on a quantized collection, not silently omitted. Timeout30 passed, not discarded. Invalid values return400 before query; NF-X12 already covers acceptance of timeout30 |
| TB-API-003 | API-live: empty schema-toggle PATCH, non-text phrase_matching, malformed batch; recreate/delete/reindex confirmation empty/wrong/padded/exact | First three use400, not any4xx or stale422 titles. Recreate/delete/reindex mismatched confirmation returns400, exact confirmation proceeds on a valid target. No writes on rejection. Browser trim behaviour does not authorize padded raw API confirmation |
| TB-LIFE-001 | **Regression — audit P2·6.** Backend embed batcher: cancel one queued request, one submitted request, then let surviving batch members finish; repeat for reranker | Cancelled callers' futures reach a terminal state, pending maps/queue counters return to baseline, surviving results keep input correspondence. No set_result on a cancelled future, no retained future/task after drain. Native inference already running may finish; claiming forced GPU cancellation is not required |
| TB-LIFE-002 | **Regression — audit P2·7.** Backend/ASGI non-exempt handler held just before a write, middleware request timeout expires, then release work barrier | Client receives504; handler is cancelled and does not issue a **new** write after cancellation. Await cleanup and prove counters/locks released. Already committed writes are not rolled back. Export's explicit timeout exemption remains distinct |
| TB-LIFE-003 | **Regression — audit P3·15.** Backend shutdown while one job is queued and another is in inference; stop accepting new work, release native inference and await lifespan completion | Every awaiting caller completes/fails/cancels explicitly, queues drain, worker tasks/executor/client resources close without orphan work. Bound the test's wait and fail on leaked tasks; preserve durable reindex checkpoint rather than pretending shutdown completed the job |
| TB-LIFE-004 | Backend timeout/cancel while waiting for collection mutation lock, and while owning it; next request for same and different collections | Cancelled waiter never enters later; owner's finally releases lock; next legitimate mutation progresses and unrelated collection is independent. No private-coordinator fallback masks missing dependency wiring |

Lifecycle rows above are correctness acceptance targets, with no requirement
to kill running native code. The remaining audit optimization B concerns
token/output admission budgets, including BASIC ColBERT. It belongs to the
embedder layer, not the console; its bounded specification is:

| Case | Layer and workload | Required result / status |
|---|---|---|
| TB-PER-001 | Backend controlled estimator: token/output reservation budget4096; submit estimated jobs2048+2048, then1; release one reservation and retry; repeat failure/cancel | At-budget work admitted, budget+1 refused before inference, reservation released exactly once on every terminal path, retry admitted. Separate input-token and output-byte accounting; explicit BASIC ColBERT output enabled/disabled variants. The embedder README defines the admission settings and estimate assumptions; these are not physical RAM/VRAM bounds |
| TB-PER-002 | API-live benchmark after admission policy exists: native-profile input lengths32/256/1024 tokens truncated to supported maximum, concurrency1/2/4, two warmups plus five measured batches per cell; BASIC return_colbert false/true explicitly, ADVANCED dense/sparse only | Record tokenizer counts, admitted/rejected requests, latency, resident/peak RAM/VRAM, output bytes and result order/shape/finite values. Same fixture and hardware for baseline/candidate; stop escalating a cell at refusal or configured resource budget, never induce OOM as a pass condition. No universal throughput gain required; policy acceptance requires bounded reservations, graceful refusal and recovery, with quality unchanged within existing model gates |

Read the embedder README before implementing or executing these API-live
variants. Unavailable hardware or an unimplemented admission policy is BLOCKED,
not a browser-test gap disguised as a passing benchmark.

---

## 14. Exit criteria

For a release acceptance run, each applicable case must:

1. execute **visible UI actions** wherever available, plus the applicable
   MCP-live and Skill-agent workflows in [§0.8](#agent-workflows); declared
   non-UI assertions retain their own layer;
2. **no console errors** in the browser (K6);
3. UI response/state **consistent** with the test data and the §1.1 truth table;
4. BASIC↔ADVANCED switches produce exactly the expected lockdowns: banner
   **suppressed** for profile-switch, **shown** for same-profile drift,
   Re-embed live **only** if recoverable, Export **always** live;
5. record whether it ran against live services, mocked responses or only static
   sources, and report skips/exclusions separately;
6. satisfy its stated postcondition, not merely have an ID in NF-COV2. Include
   the [gap-to-case map (§16.6)](#closure) cases and matrix
   variants mapped to the applicable §16 gaps. Record implementation, execution
   and product blockers separately before claiming complete acceptance coverage.

Failures recorded by **test ID** with the visible UI response and screenshot.

---

## 15. Consolidated results

### 15.1 Last recorded full run (2026-08-17, historical)

**355 PASS · 0 FAIL · 1 SKIP** in 15.1 minutes (356 tests in 17 files).

> Default run only (`npm test`): `z4-switch.spec.js`, `z5-real-outage.spec.js`
> and `z6-e7.spec.js` were **not** run — they need a real
> `SMART_EMBEDDER_PROFILE` switch + stack restart, or the embedder container
> stopped by hand (see §15.5). Everything else ran, including
> `z1-reembed.spec.js` (real 2,000-point regeneration on `e2e_z_reg`).
> Stack: `qdrant-middleware` + `embedder-server-gpu`, external Qdrant, profile
> **`advanced`** throughout (no switch this run). Sandbox dropped and reseeded
> from empty (`npm run seed`, `seed:query`, `seed:reindex`), then
> `reset:collections` + `reset:points` immediately before the run.
>
> Run with `QHM_E2E_PACE_MS=700` and a 150 s cooldown. **This matters:** at the
> shipped 250 ms pacing the suite trips the middleware rate limiter
> (300 req/min) in its request-heavy stretches, and a 429 surfaces as an empty
> result set or a `Rate limit exceeded` alert rather than an obvious failure.
> Four cases (`NF-B5`, `NF-B7`, `NF-B10`, `NF-D8`) failed that way in the
> earlier run of the day and pass consistently once paced.

¹ `X2` (multi-page export) was skipped. Its recorded reason, "no sandbox
collection exceeds one page", conflicts with the 2,000-point reindex fixture
described above. The spec skips unconditionally; see the current gap in §16.

**The 22 failures of the earlier 2026-08-17 run were all triaged to the test
suite, the book, or the environment — no application defect was found.** The
fixes landed in the same commit as this section:

| Cause | IDs | Fix |
|---|---|---|
| Expected 422; the app normalises every request-model failure to **400** (`create_app`), as §7.3 now records | `B4` `B5` `B6` `B7` `DL12` | expectation corrected in spec + book |
| Point ids that are neither an unsigned integer nor a UUID, refused before the case under test | `U6` `B8` `G5` `DL11` `PL7` `PL8` | well-formed ids |
| Selectors that never existed (`#pl-mode-replace`, `#pl-payload`) | `PL7` `PL8` | real ids (`input[name="pl-mode"][value="PUT"]`, `#pl-json`) |
| Wrong endpoint: `/points/bulk` falls through to `DELETE /points/{point_id}` | `DL12` | `DELETE /points` with a body |
| Wrong result container for a bulk delete | `DL11` | `#del-bulk-result` |
| Read of an asynchronous refresh without waiting for it | `T12` `K4` | `expect.poll`, as `R4` already did |
| Fixture consumed by an earlier case | `G1` (`U7` overwrote point 1) · `D5` (`D3` already deleted `e2e_a_new4`) | dedicated ids / own victim collection `e2e_a_new5` |
| Assertion incompatible with documented behaviour: `#ql-errors` legitimately carries the `Q11` hint | `D5` | assert absence of `.alert-error`, not emptiness |
| Self-contradictory assertion: correctly escaped text necessarily contains `onerror` | `M22` | assert no `img` / `[onerror]` node exists |
| Hardcoded a deployment-dependent ceiling (`.env` here sets `QUERY_MAX_CANDIDATE_LIMIT=2500`, `.env.example` 200) | `Q9` | read both ceilings from `window.QHM_CONFIG` |
| Rate limiting under full-suite load (see the pacing note above) | `NF-B5` `NF-B7` `NF-B10` `NF-D8` | pacing |
| Reference entry 456 chars against a 450 cap — **the only product change** | `NF-R4` | "06 Export Collection" trimmed to 444 |

One documentation defect was also corrected: `_coerce_point_id`'s docstring in
`api/routers/points.py` claimed pydantic would surface a 422.

The previous historical run (2026-07-24/25, 283 PASS · 0 FAIL · 1 SKIP, with a
real BASIC↔ADVANCED switch) is superseded by this section; see git history for
its detail if needed.

### 15.2 Previous runs of the original books (consolidated)

| Original book | Date | Cases | Result | Bugs found |
|---|---|---|---|---|
| Collections | 2026-07-12 | 54 | 51 PASS, 2 SKIP, 1 FAIL→fixed | X1: stale alerts on active collection switch (fix `app.js`) |
| Points | 2026-07-12 | 78 | 78/78 PASS | BUG-1: `replace_source` on strict-mode collection; BUG-2: text whitespace-only accepted |
| Query/Filters | 2026-07-11 | 67 | 66 PASS, 1 FAIL→fixed | C3: `DATETIME_RE` shape-only, `isValidDatetime` with per-component range |
| Monitoring | 2026-07-12 | 22 | 22/22 PASS | none |
| Re-embed | 2026-07-22 | 13 | 13/13 PASS | none |

### 15.3 Application bugs found and fixed (historical)

**BUG-1 — a 401 erases the stored active collection** · ✅ FIXED
`setActiveCollection()` no longer persists an empty name; `refreshCollections()` uses
the stored preference as fallback. The preference survives the 401.

**BUG-2 — export interrupted by collection switch stuck on "Exporting…"** · ✅ FIXED
The cancellation branch was dead code. Now `renderCancelled()` is used at both exit
points for collection switch.

**BUG-3 — `replace_source` breaks on strict-mode collection without `source_id` index** · ✅ FIXED
New `ensure_source_id_index()` in `qdrant_collections.py`, called only when
`replace_source` is active.

**BUG-4 — text whitespace-only accepted by single upsert and batch** · ✅ FIXED
Validator `_require_non_blank_text` added to `UpsertPointRequest` and `BatchUpsertItem`.

**BUG-5 — `isValidDatetime` accepted year `0000`** · ✅ FIXED
Guard `Number(year) < 1` added, aligned with pydantic parsing (MINYEAR=1).

**X-1 — `order_by` offered non-orderable fields** · ✅ FIXED
`ORDERABLE_SCHEMAS` and `orderableFields()`: only `integer`, `float`, `datetime`.

**X-2 — mistyped `start_from` returned HTTP 500** · ✅ FIXED
`ValidationError` intercepted and mapped to 422; frontend validates against the field's schema.

**X-3 — `offset` at 0 was not sent** · ✅ FIXED
Removed the `> 0` condition.

**X-4 — reference promised a download that the ordered mode does not do** · ✅ FIXED
Corrected the reference text, not the code.

### 15.4 Book expectation corrections (app correct, book wrong)

| ID | The book said | Real behavior | Correction applied |
|---|---|---|---|
| **C12** | 400 with "at least one schema toggle" | 400 with per-field validation: toggles are **mandatory** | ✅ corrected in this book |
| **PI9/PI10** | 422 | **400**; the message `phrase_matching requires field_schema 'text'` is correct | ✅ corrected in this book |
| **PI14** | alert with a generic title | the rendering shows the **server detail alone** (`payload index '…' not found in collection '…'`), no `HTTP 404` prefix | ✅ corrected in this book |
| **Q9/limit** | `limit`/`candidate_limit` always settable | in payload-only they are **disabled by design** | ✅ corrected in this book |
| **X3** | empty export → JSON with 0 points | warning "Collection is empty" and **no download** | ✅ corrected in this book |
| **§8.1** | payload-only results in a single view | the filtered set is **paginated** at 5 per page | ✅ note added in §0.4 |
| **RE** | — | the last job state **persists** | ✅ note added in §0.4 and §5 |
| **Filters** | — | the filter tree survives reload: must be cleared | ✅ note added in §0.4 |
| **Ingest** | chunk `id` = point id | ingest assigns **deterministic UUIDs** | ✅ note added in §0.4 and I27 |
| **DL11** (Rev. 3) | filter `{}` match-all → total removal | match-all is **refused** (UI disabled + API 400 "filters must contain at least one condition") | ✅ corrected in §7.5 DL8 and §0.4 |

### 15.5 Specs excluded from the default run

Three files are excluded by `testIgnore` because they require external state.
**A path alone does not enable them**: on 2026-09-05 the formerly documented
invocation returned `No tests found`. Use the configuration override and
per-file recipes in [`../README.md`](../README.md#recipes), which were checked
with `--list` (10 tests in 3 files). Listing does not exercise the scenarios.

| File | Requires |
|---|---|
| `z4-switch.spec.js` | stack already switched; correctly assigned native/foreign fixtures; run once per profile |
| `z6-e7.spec.js` | ADVANCED stack and a disposable BASIC-born incompatible collection |
| `z5-real-outage.spec.js` | embedder stopped, middleware and Qdrant running; restore the embedder afterwards |

`z1-reembed.spec.js` **is** part of the default run: it performs a real vector
regeneration on `QHM_REINDEX_COLL` (default `e2e_z_reg`, 2,000 points).

### 15.6 Spec → book series map

| Spec | Book series |
|---|---|
| `a-collections.spec.js` | §6 R/C/T/PI/X/D |
| `b-ingest.spec.js` | §7.1 I |
| `b2-ingest-indexes.spec.js` | §7.1b IX |
| `c-points.spec.js` | §7.2–7.6 U/B/G/DL/PL |
| `d-query.spec.js` | §8.1 Q |
| `d-filters.spec.js` | §8.2 QF |
| `e-apikey.spec.js` | §3 H |
| `e-monitoring.spec.js` | §9 M |
| `e-nav.spec.js` | §10 N |
| `nf-newfeatures.spec.js` | §13 NF |
| `z1-reembed.spec.js` | §5 RE |
| `z2-service.spec.js` | §2 S |
| `z3-compat-mocked.spec.js` | §4 E |
| `z4-switch.spec.js` | §4/§12 E/Z |
| `z5-real-outage.spec.js` | §2 S |
| `z6-e7.spec.js` | E7 |
| `embedding-compatibility.e2e.spec.js` | compact compatibility regression (mocked reports) |
| `export-filename.e2e.spec.js` | X6 — two browser-download cases with every API response mocked |
| `ingest-preflight.e2e.spec.js` | compact ingest/preflight regression |
| `service-readiness.e2e.spec.js` | compact readiness/recovery regression |
| `vector-reindex.e2e.spec.js` | standalone reindex migration regression |

The four pre-existing `*.e2e.spec.js` specs plus X6 are collected by the default
runner. The standalone reindex spec is live; compact compatibility also creates
and deletes a real collection. Do not classify this whole group as mocked.
`query-models.test.mjs` runs separately with `node --test`.

---

## 16. Current coverage review (2026-09-05)

This revision reviews spec bodies, runner configuration, fixture scripts,
console source and relevant middleware contracts. **No full live Playwright
run, profile switch or outage was performed in this review.** Historical PASS
counts in §15 do not validate the current tree, and this documentation change
does not repair the test implementations listed below.

### 16.1 Verification performed

| Check | Observed result | What it proves |
|---|---|---|
| Default `playwright test --list` | 359 tests / 18 files | Collection only; not execution or assertion quality |
| Old excluded-file invocation using paths only | exit 1, `No tests found` | The original run recipe was incorrect |
| Explicit environment configuration, `--list` | 10 tests / 3 files | Corrected recipe collects the external-state cases |
| NF-COV1 / NF-COV2 | 2 PASS | Current source-reference and ID guards pass, with the limitations in §13.9 |
| `node --test tests/query-models.test.mjs` | 68 PASS | Pure frontend model regressions, including incomplete filters and polygon serialization; not browser interaction |
| Middleware `test_points_crud_extra.py` + `test_payload_paths.py` | 41 PASS | Offline payload ownership/path/CRUD checks supporting the corrected U6/PL7 contracts |

### 16.2 Corrections needed in the executable suite

| Priority | Area | Evidence and required follow-up |
|---|---|---|
| **P2** | U6 / PL7: obsolete success expectations | [c-points.spec.js](../c-points.spec.js) still expects metadata to overwrite embedded text and payload REPLACE to change text. U6 must assert top-level text wins; PL7 must assert 400 and unchanged payload. Also exercise PATCH and filtered MERGE rejection. Do not revert product fixes to satisfy these old assertions |
| **P2** | Sandbox enforcement and URL consistency | Historical finding addressed by [harness.mjs](../harness.mjs): shared `BASE_URL`, campaign names and ownership validation now guard live mutations. [vector-reindex.e2e.spec.js](../vector-reindex.e2e.spec.js) uses `fixtureName("e2e_vector_reindex")`. Keep TB-HAR-001/002 as regression acceptance; the former prefix/URL limitations are not current prerequisites |
| **P2** | Filter semantics and validation | [d-filters.spec.js](../d-filters.spec.js) does not execute the whole §8.2 matrix. Add integer any/except, negative phrase, polygon hole/auto-close, UUID list, open array bounds, null/missing `is_empty`, real same-element versus flat matching, and browser validation cases. Node serializer checks do not prove the UI-to-Qdrant path |
| **P2** | Vector-presence filter | `(vector presence)` is rendered and compiled by [app.js](../../dist/assets/app.js) and [models.js](../../dist/assets/models.js), but no Playwright case selects a vector and asserts the returned set. Include valid/missing vector names and refresh after generation changes |
| **P2** | Monitoring claims exceed assertions | M-G in [e-monitoring.spec.js](../e-monitoring.spec.js) counts polyline **elements**, not samples, and only checks that fetched source is a string. Feed >120 deterministic samples and verify eviction/order; add chart hover/leave, full badge/API coherence and metric-value assertions. M-F and M-K currently prove rendering, not all the listed chart/metrics semantics |
| **P2** | Reindex lifecycle and vector visibility | RE8 starts no job; RE13 conditionally skips payload comparison if the initial GET fails; RE6 also accepts natural completion. Require the precondition to succeed, observe a real running job before switching, require actual cancellation when testing cancel, and compare payloads/vector summaries. NF-P17 is an additional missing assertion, not redundant coverage |
| **P2** | Export pagination | X2 is unconditionally skipped despite the 2,000-point reindex seed. Provide a dedicated read fixture (or explicitly share a stable one), walk multiple server pages, and compare exact IDs/payloads. X6 proves browser header selection and bytes, not server Unicode header construction, filesystem confinement or large-export limits |
| **P2** | Coverage guards | NF-COV1 accepts source text, even comments. NF-COV2 merges IDs such as §8.2 C1 and §6.2 C1, overlooks names without digits and relies on broad allowlists. Use explicit section-qualified case mappings and collected test titles, and record partial/manual coverage rather than adding allowlist entries to make a missing test green |
| **P3** | Cross-cutting invariants | Expand K1/K2 to the remaining listed operations and enforce K6 through a shared fixture with narrowly identified expected errors. Merely calling `watchConsole()` does not assert its result. The current per-file subsets are described in §11 |
| **P3** | Titles and externally observed contracts | B4–B7 still say 422 in titles although their bodies expect 400. T11 permits any 4xx. NF-X12 only proves selected request acceptance/rejection; NF-P14 does not observe embedding calls. Tighten the assertions at the layer that can observe each contract |

### 16.3 Additional acceptance cases now specified

These are missing or partial **browser acceptance cases**, not assertions that
the corresponding backend feature is broken. Middleware unit/ASGI tests remain
the appropriate layer for internal forwarding, locking and failure injection.
The normative [gap-to-case map (§16.6)](#closure) maps
every row in §16.2 and this section to concrete cases. Their executable status
remains open until the corresponding assertions are implemented and run.

| Capability | Required scenario and observable result | Current evidence boundary |
|---|---|---|
| Payload protection after the audit | Upsert/Batch with conflicting `payload.text`; PATCH/PUT/filter MERGE with text; delete-keys with plain, quoted, escaped, nested and array forms of `text`/`source_id`; permitted ordinary keys still work | Offline backend coverage exists; browser U6/PL7 are stale and NF-B9 covers plain `text` only |
| Identifier/input fidelity | Batch aliases such as `1`/`01`, normalized UUID duplicates, uint64 values beyond JavaScript's safe integer range; ingest boolean/string IDs and non-string text. Valid IDs must stay lossless; invalid or duplicate inputs refused before writes. Signed integer chunk ordinals, including negative values, remain valid and generate UUID point IDs | Selected model/backend tests exist; current B6 and I31 do not cover this full boundary through the browser; TB-ID-001–005 specify it |
| Download → re-import | Export an ingested corpus with default and renamed chunk-ID mapping, then re-import using a defined ordinal/source mapping; compare text, metadata and count. Separately test query-result downloads containing UUID point IDs | X1/X6 and NF-X7 download/read files but do not re-upload them; a JSON download is not proof of an ingest roundtrip |
| Export limits and interruption | Empty, multi-page, over-cap, collection growing during scroll, server failure and collection switch; assert absence of partial downloads and readable recovery | X3/X5/X7 cover subsets. Incremental point/byte limits remain an open product audit item, not a claimed implemented contract |
| Filtered clear sampling | A non-protected key appearing only beyond the sample must survive, and the reference must describe that limitation; both protected fields survive when present | NF-B7's small fixture cannot establish behaviour outside the sampled set |
| Request timing and admission | Double-click, delayed response after collection switch, visible 429/timeout, retry and cancelled client; assert current-collection results and button recovery | Do not infer cancellation of server writes from browser stale-result suppression (I28/I29/K2) |
| Profile roundtrip | Execute §12 on both profiles, record per-phase native/foreign contracts and re-embed results, then verify monitoring and cleanup | z4 only observes externally prepared state; z6 currently asserts BASIC → ADVANCED; the default run proves no real switch |
| UI states and accessibility | Keyboard/focus and disabled reasons for dynamic filter/mapping controls; narrow viewport across each panel; loading, empty, error and success states for each operation | N0 checks the brand at 320 px; N/NF-Z cover selected controls, not a complete responsive/accessibility audit |

### 16.4 Recording the next run

Record the tested commit and any uncommitted product changes, image rebuild,
active profile, service/dependency versions, seed/reset actions, collection
targets, pacing/cooldown and exact command. Never include tokens or local `.env`
contents. Report PASS/FAIL/SKIP per spec/ID and list external-state scenarios
separately. Retain screenshots for failures and identify mocked responses.

Close a §16 gap only after its actual postcondition has an assertion and the
appropriate check ran; do not use the raw number of collected tests as a
percentage of functional coverage.

### 16.5 Specification completion and unification (revisions 5–6)

The detailed acceptance sections in this book add fresh deterministic
fixtures, a complete operator/type matrix, operation/state matrices, explicit
lifecycle and failure oracles, and a [gap-to-case mapping](#closure). They also
define acceptance targets for the still-open export resource bounds and
cancellation/shutdown audit findings without claiming those product fixes exist.

Revision 5 performs document consistency and fixture checks only. No live
Playwright run, service restart, profile switch or new acceptance-case execution
is implied. §16.1 remains the evidence from the earlier revision-4 review.
Revision 6 integrates the complete specification into this one file, preserving
all 96 TB case IDs and their actions/oracles. Only document consistency and
fixture arithmetic were checked for this merge; the E2E suite was not run.

Further source checks corrected two revision-4 statements: NF-X12 already
accepts a valid `timeout: 30` (forwarding/enforcement still need separate
assertions), and negative **chunk ordinals** are valid signed integers, unlike
negative numeric **point IDs**. The detailed query section also records that
vector-name options currently come from startup configuration, making generation
refresh an explicit frontend acceptance target.

<a id="closure"></a>

### 16.6 Gap closure map and execution record

Every known review gap now has an acceptance specification. The mapping
below does not replace the historical evidence inventory in §16.2/16.3.
The execution record for the closure batches is §16.7.

| Review gap | Normative cases / matrix |
|---|---|
| §16.2 U6/PL7 stale assertions | TB-PAY-001, TB-PAY-002, TB-PAY-005 |
| §16.2 sandbox/base URL | TB-HAR-001, TB-HAR-002, TB-HAR-006 |
| §16.2 filter semantics/validation | Main §8.2 qualified A/B/C cases and TB-FIL-001–016 |
| §16.2 vector presence | TB-QRY-001, TB-QRY-002, TB-REI-004 |
| §16.2 monitoring | TB-MON-001–007 |
| §16.2 reindex lifecycle/vector visibility | TB-REI-001–007 and guarded mutation inventory |
| §16.2 export pagination | TB-EXP-001–004, TB-EXP-008–011 |
| §16.2 coverage guards | TB-HAR-003–005 and qualified mapping rules |
| §16.2 cross-cutting invariants | TB-UI-001–009 and every operation/state variant |
| §16.2 titles/observable contracts | TB-API-001–003; correct B4–B7 titles when implementing |
| §16.3 payload protection | TB-PAY-001–006 |
| §16.3 identifier/input fidelity | TB-ID-001–005, TB-ING-001–002 |
| §16.3 download/re-import | TB-EXP-005–007, TB-QRY-007 |
| §16.3 export limits/interruption | TB-EXP-008–011 |
| §16.3 sampled clear | TB-PAY-006 |
| §16.3 request timing/admission | TB-UI-001–003, TB-ING-005–006, TB-LIFE-001–004 |
| §16.3 profile roundtrip | TB-PRO-001–004 |
| §16.3 dynamic UI/accessibility | TB-UI-004–008 |

Implementation order: shared harness/target guards and stale assertions first;
pure fixtures and backend observability next; deterministic mocked browser
matrices next; live filters/CRUD/export/reindex next; external profile/outage
matrix last. Open-product targets need their own fixes and focused regressions
before their acceptance rows can pass. This order adds no permission to run
or mutate services during the documentation task.

Each future run records one row per qualified case **and variant**:

| Field | Required content |
|---|---|
| Identity | Case key, variant, full spec title/file or manual procedure, book revision |
| Build/environment | Commit + relevant dirty files; image identity, rebuilt assets, middleware/Qdrant/embedder versions, hardware/model profile; no secrets |
| Setup | Base URL, owned collection names, fixture version/formula, exact initial count/indexes, reset action |
| Layer | UI-live / UI-mock / API-live / MCP-live / Skill-agent / Backend / Model / Static, including every mocked route and controlled fault; agent evidence follows §0.8 |
| Oracle evidence | Exact IDs/count/payload differences, request status/fields, transitions and assertion locations; sanitized screenshot/trace if relevant |
| Result | PASS / FAIL / BLOCKED / NOT RUN; skipped test recorded with reason and does not count as PASS |
| Cleanup | Pending jobs settled, owned artifacts accounted for, starting profile restored, unexpected requests/console errors listed |

Specification completeness here is bounded to the reviewed console contracts,
known audit findings and matrices above. It is not a claim that all possible
inputs, browsers, distributed deployments or resource conditions have been
proved. A newly discovered contract/branch must receive a case and oracle
before any claim of complete acceptance for the revised scope.

### 16.7 Implementation and execution record (revision 7, 2026-09-06)

This addendum records inventory assignment and the historical execution evidence for the §16.6 gap map.
It is the single source of truth for "what is actually green at revision 7";
§16.6 stays the normative gap map and is not edited retroactively. Every row
below is reproducible from the repository with the commands in
`frontend/tests/README.md` and `.github/workflows/ci.yml`.

**Coverage contract.** `frontend/tests/testbook/coverage-contract.json` is the
machine-readable gate. It parses this book into qualified case keys, credits
collected browser titles (default and external configs), executed Node TAP titles, and
declared backend/embedder owners, rejects unknown IDs and duplicate owners,
requires written exclusion reasons of at least 40 characters, marks an
exclusion stale when executable evidence later appears, and fails the run when
unassigned cases exceed `maxUnassigned`. Revision 7 sets
`maxUnassigned: 0`. The gate is `frontend/tests/coverage-gate.mjs`, its unit
tests are `coverage-gate.test.mjs`, and the campaign reporter is
`campaign-reporter.mjs`.

**Status categories at revision 7.**

| Category | Count | Meaning |
|---|---|---|
| `MAPPED` | 497 | A collected default-config browser title claims the case; no execution outcome is implied. |
| `EXTERNAL` | 12 | A collected external-config title claims the case; execution needs the prepared profile/outage state. |
| `OWNED` | 16 | A declared middleware pytest or embedder unittest owner claims the case. Its suite checks marker collection and runs independently; this inventory gate does not read its results. |
| `NODE` | 5 | A passing `node --test` TAP title owns the case (pure model/contract tests). |
| `EXCLUDED` | 6 | Operator-only or environment-dependent; a written reason ≥40 chars is on file and no executable evidence has appeared. |
| `UNASSIGNED` | 0 | No owner and no exclusion — the gate fails if this is non-zero. |

**Executable suites and their last recorded result.**

| Suite | File(s) | Layer | Last result |
|---|---|---|---|
| Live filter matrix | `tb-filters-live.spec.js` | UI-live | 183/183 passed |
| Live payload/IDs/ingest | `tb-payload-ids.spec.js` | UI-live + API-live | historically reported green with a TB-ID-004 expected failure; this is not an all-PASS result. See §16.8 for follow-up verification |
| Live export/query | `tb-export-query.spec.js` | UI-live + API-live | 55 passed |
| Live reindex + vector presence | `tb-reindex-live.spec.js` | UI-live + API-live | 7 passed (TB-REI-001–005, NF-P17, TB-QRY-002) |
| Intercepted UI matrix | `tb-ui.spec.js` | UI-mock | 129 passed (TB-UI-001–009, TB-PRO-002 truth table, TB-EXP-008/011, TB-ING-003, TB-QRY-005, NF-P20, XSS, lockdown) |
| Monitoring regressions | `tb-monitoring.spec.js` | UI-mock | 25 passed (TB-MON-001–007) |
| Filter validation | `tb-filter-validation.spec.js` | UI-mock | historical run included the TB-FIL-014 expected failure; see §16.8 |
| Node contract/model tests | `*.test.mjs` | Node | passing (coverage gate + query model + case-coverage) |
| Middleware backend owners | `middleware-qdrant-manager/tests/test_testbook_backend.py`, `test_testbook_contract.py` | Backend | passing (export cap, payload protection, contract shape) |
| Embedder owners | `server-embedder-reranker/tests/test_testbook_contract.py`, `test_request_processor_lifecycle.py`, `book_cases.py` | Model | passing (admission budget, request-processor lifecycle, book cases) |

**Live campaign invariants (enforced by `tb-live-helpers.js` and the harness).**

- A live run requires `QHM_E2E_TOKEN` and an initialized `QHM_E2E_RUN_ID`
  (`npm run init:run`); the manifest is written to
  `frontend/test-results/ownership-<run_id>.json`.
- Fixture collections are namespaced by the run id
  (`e2e_<run_id>_book_filters`, …). Mutations outside owned collections are
  rejected by the harness before any request is sent.
- Read-only endpoints (query, scroll, count, retrieve, facet) are allowed
  against any collection; only writes are ownership-gated.
- `bootPaced`/`reserve` pace the suite to stay under the middleware's
  300 req/min rate limit. Rate-limit exhaustion is an environmental condition
  to control with pacing, not a silent pass.
- Pending reindex jobs are settled in cleanup; owned artifacts are accounted
  for; the starting profile is restored.

**Seed fixtures (`npm run seed:book`).** `tools/seed-book.mjs` creates, under
the owned run namespace: a 100-point filter fixture, a 6-point edge fixture, a
3-point write fixture, a 2001-point export fixture, an empty fixture, a
1001-point sample fixture, an alternate fixture, and two roundtrip fixtures.
The generator is the oracle: every export/ID/payload assertion compares
against the seeded values, not a hand-maintained snapshot.

**Written exclusions (the six `EXCLUDED` rows).** Each has a ≥40-character
reason in `coverage-contract.json` and is reviewed for staleness on every gate
run.

| Case | Reason summary |
|---|---|
| `book:12/Z1` | BASIC-side operator roundtrip setup and query. |
| `book:12/Z3` | ADVANCED-side setup/query after the real profile switch. |
| `book:12/Z5` | Second profile switch and reverse compatibility check. |
| `book:12/Z7` | Operator roundtrip cleanup. |
| `TB-PRO-004` | Four hardware/profile cells, requiring separately recorded operator evidence. |
| `TB-PER-002` | Performance benchmark campaign under conservative admission policy. |

If executable evidence later appears for any excluded case, the gate marks
the exclusion stale and fails until the exclusion is removed and the case is mapped or owned.

**What revision 7 does not claim.**

- The live suites prove behaviour for the `basic` profile on the seeded
  fixtures at the recorded commit. They are not proof for every profile,
  browser, dataset size, or distributed deployment. A cross-profile refresh of
  the vector selector (TB-QRY-002) is recorded as a same-profile regeneration
  that keeps the names; a real cross-profile refresh needs the external
  profile-switch config.
- The historical intercepted artifact has 334 results: 333 passed and one
  expected failure (TB-FIL-014). Expected failures remain defects in the case
  ledger even when Playwright reports the overall run successful.
- The external profile/outage/proxy cases remain operator-run via
  `playwright.external.config.js`; they are `EXTERNAL`, not `MAPPED`, and do
  not run in the default CI lane.

### 16.8 Revision-7 review fixes (2026-09-06)

The inventory gate is explicitly labeled `kind: inventory`. `browserCoverage`
is `full`, `partial` or `none` and describes the declared browser scope only;
it never certifies execution. The independent campaign ledger retains PASS,
PARTIAL, FAIL, NOT_RUN and SKIPPED. External scenarios and the six operator
exclusions remain outstanding until their campaign evidence is recorded.

The follow-up removes expected-failure markers for TB-FIL-014 and TB-ID-004:
local field validation retains error associations during no-op edits, and the
shared ingest validator refuses duplicate or unsafe ordinals before transport.
An intercepted ingest test now asserts the visible error as well as zero calls.
The embedder deadline test now consumes and checks its response Future, and
the zero-unassigned limit is pinned to zero independently of inventory counts.

Live and performance campaign results must be recorded with their run ID and
profile; historical revision-7 results are not credited to the changed assets.

Verification of the follow-up: 336 intercepted browser tests passed without
expected failures; Node 98 passed; middleware 592 passed; embedder 353 passed with the deadline
Future consumed and no unhandled-Future error. Ruff check and format passed.
The campaign reporter now includes actual outcome counts and separately counts
expected failures; a green runner status does not imply every case passed.

Fresh live campaign `review0906`, GPU/ADVANCED, rebuilt assets: export/query
55/55, filters 183/183, payload/IDs 83/83, reindex 7/7. All 328 results were
actual passes, with zero skips, unexpected outcomes or flaky tests (15.1 min).
The campaign manifest protected 73 preexisting collections. Cleanup removed only
manifest-owned fixtures: 73 collections before and after, none missing and no
owned collection remaining.

Operator follow-up on the same campaign: profile-switch checks passed 7/7 on
BASIC and 7/7 on ADVANCED; cross-profile reindex plus a real hybrid query passed
1/1; real embedder outage passed 2/2 with livez=200 and readyz=503. The outage
telemetry permits only the expected 503 responses on the three declared GET
endpoints. These runs do not establish the separate Qdrant-outage or proxy
scenarios, nor remove partial annotations from broader cases.

The four hardware/profile cells completed live smoke, admission/recovery and a
bounded benchmark (32/256/1024 tokens, concurrency 1/2/4). Actual benchmark cell
outcomes, with shipped conservative request limits retained:

| Hardware/profile | PASS | REFUSED | NOT_RUN_AFTER_REFUSAL |
|---|---:|---:|---:|
| CPU/ADVANCED | 18 | 0 | 0 |
| CPU/BASIC | 25 | 2 | 0 |
| GPU/ADVANCED | 17 | 1 | 0 |
| GPU/BASIC | 23 | 2 | 2 |

Refusals remain explicit outcomes, not successful workload measurements. Small
recovery requests succeeded and reservations drained in all four cells. This
operator evidence does not convert the six inventory exclusions into automated
coverage or claim throughput for refused workloads.

The first full CPU/ADVANCED calibration used the unchanged 1800-second budget.
All seven long probes completed, but the 150.8-second final reservation expired
before quality capture; promotion was rejected and tuned selections rolled
back. The deferred-canary scheduler now reserves probe time plus the existing
load/quality estimate, retaining the 120-second floor, 40-percent cap, total
time budget and RSS/quality gates. Full revalidation completed in 1617.6 seconds:
the selected-policy canary completed in 72.3 seconds, with promotion and quality
gates passing and no timeout or memory breach. It covered the changed dense and
reranker backends at 1024/2048 tokens; it is not a new sparse-8192 measurement
or a comparative long-input latency claim against an unmeasured baseline.

That run exposed a separate serving defect: ONNX cache reuse attempted to create
an export lock on the read-only runtime-state volume. Cache reuse now verifies
the requested artifact before acquiring any write lock; missing/corrupt exports
retain the existing locked atomic preparation path. FP32/INT8 regression tests
exercise read-only reuse and corruption regeneration. Physical startup
verification passed in a separate full-mode run with explicit ONNX FP32 and a
900-second calibration limit: 605.3 seconds total, final canary promoted without
timeout, real CPU startup and smoke/admission/recovery passed. The runtime-state
mount remained read-only. GPU/ADVANCED was restored with readyz=200. A separate quick
600-second CPU run started successfully and passed smoke/admission/recovery,
but its canary timed out and tuned selections were rolled back. That shorter
run is not credited as a canary pass and did not exercise ONNX selection.
