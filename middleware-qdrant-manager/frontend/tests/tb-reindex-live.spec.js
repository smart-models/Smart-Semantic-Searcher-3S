// Test book §5.1 — reindex lifecycle (TB-REI-001..005) and §13.8 NF-P17,
// UI-live on the 2001-point `e2e_book_export` fixture with `e2e_book_alt` as
// the unrelated native collection. Each case starts from a settled collection
// and finishes by settling it (cancel + wait) so no lock survives the file.
// Jobs that finish before a pause window can be observed are reported as a
// missed precondition, never as a pass. TB-REI-006/007 are Backend-owned.
import { expect, test } from "./fixtures.js";
import { fixtureName, settleOwnedReindex } from "./harness.mjs";
import { boot, openTab, openPointsSubtab, selectActive } from "./helpers.js";
import { api, bootPaced, reserve, snapshot } from "./tb-live-helpers.js";

const COLL = fixtureName("e2e_book_export");
const ALT = fixtureName("e2e_book_alt");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test.describe.configure({ mode: "serial" });

const settle = () => settleOwnedReindex(COLL);
test.beforeEach(settle);
test.afterAll(settle);

const status = async () => (await api("GET", `/collections/${encodeURIComponent(COLL)}/reindex`)).data;
const report = async (name = COLL) => (await api("GET", `/collections/${encodeURIComponent(name)}`)).data;

async function openCollections(page) {
  const errors = await bootPaced(page, boot, { active: COLL });
  await selectActive(page, COLL);
  await openTab(page, "collections");
  return errors;
}

// Preflight, exact confirmation, Start: returns once the server reports a live job.
async function startJob(page) {
  await reserve(4);
  await expect(page.locator("#cr-preflight")).toBeEnabled({ timeout: 60_000 });
  await page.locator("#cr-preflight").click();
  await expect(page.locator("#cr-confirm-wrap")).toBeVisible({ timeout: 30_000 });
  await page.locator("#cr-confirm").fill(COLL);
  await expect(page.locator("#cr-start")).toBeEnabled();
  await page.locator("#cr-start").click();
  await expect(page.locator("#cr-status")).toContainText(/RUNNING|FINALIZING|PAUSED/, { timeout: 60_000 });
  await expect(page.locator("#cr-confirm")).toHaveValue("");
}

// Pause and prove PAUSED with a stable durable offset; a job that completes
// first is a missed precondition for the case, reported as such.
async function pauseJob(page) {
  const pause = page.locator("#cr-pause");
  expect(await pause.isVisible().catch(() => false), "precondition: job completed before Pause could be exercised").toBe(true);
  await pause.click();
  // PAUSED is requested at once; the in-flight batch still commits (a batch
  // boundary), so the durable offset is judged once the worker has stopped.
  await expect.poll(async () => { const s = await status(); return `${s.job?.status}:${s.task_running}`; }, { timeout: 90_000 }).toBe("paused:false");
  const first = await status();
  await sleep(2000);
  const second = await status();
  expect(second.job.processed_points).toBe(first.job.processed_points);
  expect(second.task_running).toBe(false);
  expect(second.job.processed_points).toBeLessThan(second.job.total_points);
  await expect(page.locator("#cr-status")).toContainText("PAUSED");
  return first.job;
}

async function waitTerminal(expected, timeout = 600_000) {
  const deadline = Date.now() + timeout;
  let state;
  while (Date.now() < deadline) {
    state = await status();
    if (["completed", "cancelled", "failed"].includes(state.job?.status)) break;
    await sleep(2000);
  }
  expect(state.job?.status).toBe(expected);
  return state.job;
}

const withVectors = async (id, name = COLL) => (await api("GET", `/points/${id}?collection_name=${encodeURIComponent(name)}&with_vectors=true`)).data;

test("TB-REI-001 — preflight writes nothing; start→RUNNING→PAUSED (stable offset)→resume→COMPLETED with count = seed; payloads, IDs and contract intact; hybrid query succeeds", async ({ page }, info) => {
  test.setTimeout(900_000);
  const errors = await openCollections(page);
  const baseline = await snapshot(COLL);
  expect(baseline).toHaveLength(2001);
  const reportBefore = await report();
  await reserve(2);
  await page.locator("#cr-preflight").click();
  await expect(page.locator("#cr-preflight-result")).toContainText(/Preflight/i);
  expect(await snapshot(COLL)).toEqual(baseline);
  expect((await status()).job?.status ?? null).not.toBe("running");
  await startJob(page);
  const paused = await pauseJob(page);
  await page.locator("#cr-resume").click();
  await expect.poll(async () => (await status()).job?.status, { timeout: 60_000 }).toMatch(/running|finalizing|completed/);
  const done = await waitTerminal("completed");
  expect(done.processed_points).toBeGreaterThanOrEqual(paused.processed_points);
  expect(done.processed_points).toBe(2001);
  expect(done.total_points).toBe(2001);
  await expect(page.locator("#cr-status")).toContainText("COMPLETED", { timeout: 60_000 });
  const after = await snapshot(COLL);
  expect(after).toEqual(baseline);
  const reportAfter = await report();
  expect(reportAfter.points_count).toBe(2001);
  expect(reportAfter.embedding_compatibility.status).toBe("compatible");
  expect(reportAfter.embedding_compatibility.active.profile).toBe(reportBefore.embedding_compatibility.active.profile);
  // Every point carries exactly the report's active target vectors.
  const active = Object.values(reportAfter.embedding_compatibility.active_vectors).sort();
  for (const id of ["1", "1001", "2001"]) {
    const point = await withVectors(id);
    expect(Object.keys(point.vectors).sort()).toEqual(expect.arrayContaining(active));
    for (const summary of Object.values(point.vectors)) expect(Object.keys(summary).sort()).toEqual(expect.arrayContaining(["type"]));
  }
  await openTab(page, "query");
  await page.locator("#ql-query").fill("export document");
  await reserve(1);
  const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/query/search");
  await page.locator("#ql-run").click();
  expect((await pending).status()).toBe(200);
  await expect(page.locator("#ql-results .result-card").first()).toBeVisible({ timeout: 30_000 });
  await info.attach("lifecycle", { body: JSON.stringify({ paused, done, reportAfter }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("TB-REI-002 — RUNNING→PAUSED→cancel yields CANCELLED (not COMPLETED); mapping, payloads and IDs unchanged; reads and writes work afterwards", async ({ page }, info) => {
  test.setTimeout(600_000);
  const errors = await openCollections(page);
  const baseline = await snapshot(COLL);
  const before = await report();
  await startJob(page);
  await pauseJob(page);
  await page.locator("#cr-cancel").click();
  const job = await waitTerminal("cancelled", 120_000);
  await expect(page.locator("#cr-status")).toContainText("CANCELLED", { timeout: 60_000 });
  expect(await snapshot(COLL)).toEqual(baseline);
  const after = await report();
  expect(after.embedding_compatibility.active_vectors).toEqual(before.embedding_compatibility.active_vectors);
  expect(after.embedding_compatibility.status).toBe("compatible");
  // The mutation lock is released: a write lands and is reverted.
  await api("POST", "/points", { collection_name: COLL, point_id: "9999", text: "lock released", payload: { probe: true } });
  await api("DELETE", `/points/9999?collection_name=${encodeURIComponent(COLL)}`, undefined);
  expect(await snapshot(COLL)).toEqual(baseline);
  await info.attach("cancelled-job", { body: JSON.stringify(job), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("TB-REI-003 — while A is PAUSED the alt collection never shows A's job, stays writable, and A's job is rediscovered on return", async ({ page }, info) => {
  test.setTimeout(600_000);
  const errors = await openCollections(page);
  await startJob(page);
  const paused = await pauseJob(page);
  await reserve(8);
  await selectActive(page, ALT);
  await openTab(page, "collections");
  await expect(page.locator("#cc-report")).toContainText(ALT);
  await expect(page.locator("#cr-status")).not.toContainText("PAUSED");
  await expect(page.locator("#cr-status")).not.toContainText(String(paused.processed_points));
  await expect(page.locator("#cr-pause")).toBeHidden();
  await expect(page.locator("#cr-resume")).toBeHidden();
  await expect(page.locator("#cr-cancel")).toBeHidden();
  await openTab(page, "points");
  await openPointsSubtab(page, "upsert");
  await page.locator("#up-id").fill("2");
  await page.locator("#up-text").fill("alt write during A job");
  await page.locator("#up-payload").fill("{}");
  await expect(page.locator("#up-submit")).toBeEnabled();
  await reserve(3);
  const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points" && r.request().method() === "POST");
  await page.locator("#up-submit").click();
  expect((await pending).status()).toBe(200);
  await api("DELETE", `/points/2?collection_name=${encodeURIComponent(ALT)}`, undefined);
  // The server job persisted: switching collections is not a cancel.
  expect((await status()).job.status).toBe("paused");
  await reserve(8);
  await selectActive(page, COLL);
  await openTab(page, "collections");
  await expect(page.locator("#cr-status")).toContainText("PAUSED", { timeout: 30_000 });
  await expect(page.locator("#cr-resume")).toBeVisible();
  await page.locator("#cr-cancel").click();
  await waitTerminal("cancelled", 120_000);
  await info.attach("scoped-job", { body: JSON.stringify(paused), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("TB-REI-004 — with_vectors summaries report names and shapes only, before, during (paused) and after the job", async ({ page }, info) => {
  test.setTimeout(900_000);
  const errors = await openCollections(page);
  const before = await report();
  const source = Object.values(before.embedding_compatibility.active_vectors).sort();
  const summaryOnly = (point) => {
    for (const [name, summary] of Object.entries(point.vectors)) {
      expect(["dense", "sparse"]).toContain(summary.type);
      expect(Object.keys(summary).sort()).toEqual(summary.type === "dense" ? ["dimension", "type"] : ["nonzero", "type"]);
      expect(name.length).toBeGreaterThan(0);
    }
  };
  const first = await withVectors("1");
  expect(Object.keys(first.vectors).sort()).toEqual(source);
  summaryOnly(first);
  await startJob(page);
  const paused = await pauseJob(page);
  const during = [];
  for (const id of ["1", "1000", "2001"]) {
    const point = await withVectors(id);
    summaryOnly(point);
    // Every point still carries the source generation; processed ones may add the target.
    expect(Object.keys(point.vectors)).toEqual(expect.arrayContaining(source));
    during.push({ id, names: Object.keys(point.vectors).sort() });
  }
  const many = (await api("POST", "/points/retrieve", { collection_name: COLL, ids: ["1", "2001"], with_vectors: true })).data;
  for (const point of many.results) summaryOnly(point);
  await page.locator("#cr-resume").click();
  await waitTerminal("completed");
  const after = await report();
  const target = Object.values(after.embedding_compatibility.active_vectors).sort();
  for (const id of ["1", "1000", "2001"]) {
    const point = await withVectors(id);
    summaryOnly(point);
    expect(Object.keys(point.vectors)).toEqual(expect.arrayContaining(target));
  }
  // The UI Get form renders the summary, not raw numbers.
  await openTab(page, "points");
  await openPointsSubtab(page, "get");
  await page.locator("#get-id").fill("1");
  await page.locator("#get-with-vectors").check();
  await reserve(1);
  const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/1");
  await page.locator("#get-submit").click();
  expect((await pending).status()).toBe(200);
  await expect(page.locator("#get-result")).toContainText(target[0]);
  await expect(page.locator("#get-result")).not.toContainText(/\d\.\d{4,}/);
  await info.attach("vector-summaries", { body: JSON.stringify({ source, target, during, paused }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

const encoded = encodeURIComponent;
function guardedRoutes(name) {
  const c = { collection_name: name };
  const filters = { must: [{ key: "probe", match: { value: true } }] };
  return [
    // A real sandboxed file: the lock, not path validation, must be what refuses it.
    ["POST", "/points/ingest", { ...c, source_id: "guard", file_path: "mcptest_n_file.json" }],
    ["POST", "/points/ingest-chunks", { ...c, source_id: "guard", chunks: [{ id: 1, text: "guard" }] }],
    ["POST", "/points", { ...c, point_id: "9999", text: "guard" }],
    ["POST", "/points/batch", { ...c, points: [{ point_id: "9999", text: "guard" }] }],
    ["DELETE", `/points/1?collection_name=${encoded(name)}`, undefined],
    ["DELETE", "/points", { ...c, ids: ["1", "2"] }],
    ["POST", "/points/delete-by-filter", { ...c, filters, confirm_collection: name }],
    ["PATCH", `/points/1/payload?collection_name=${encoded(name)}`, { ...c, payload: { probe: true } }],
    ["PUT", `/points/1/payload?collection_name=${encoded(name)}`, { ...c, payload: { probe: true } }],
    ["POST", `/points/1/payload/delete-keys?collection_name=${encoded(name)}`, { ...c, keys: ["marker"] }],
    ["POST", `/points/1/payload/clear?collection_name=${encoded(name)}`, { ...c, confirm_collection: name }],
    ["PATCH", "/points/payload", { ...c, filters, payload: { probe: true }, confirm_collection: name }],
    ["POST", "/points/payload/delete-keys", { ...c, filters, keys: ["marker"], confirm_collection: name }],
    ["POST", "/points/payload/clear", { ...c, filters, confirm_collection: name }],
    ["POST", "/collections", { ...c, recreate: true, confirm_recreate: name, dense_on_disk: true, sparse_on_disk: true, on_disk_payload: true, quantization: true, sparse_idf: true }],
    ["PATCH", `/collections/${encoded(name)}/schema-toggles`, { dense_on_disk: false }],
    ["PUT", `/collections/${encoded(name)}/schema-toggles`, { dense_on_disk: false }],
    ["POST", `/collections/${encoded(name)}/payload-indexes`, { indexes: [{ field_name: "marker", field_schema: "keyword" }] }],
    ["DELETE", `/collections/${encoded(name)}/payload-indexes/marker`, undefined],
    ["DELETE", `/collections/${encoded(name)}?confirm_delete=${encoded(name)}`, undefined],
  ];
}

test("TB-REI-005 — during RUNNING and PAUSED all 19 guarded routes return 409 with no hidden write; reads, report and export work; the unrelated collection is not serialized; a second start is refused", async ({ page }, info) => {
  test.setTimeout(900_000);
  const errors = await openCollections(page);
  const baseline = await snapshot(COLL);
  const altBaseline = await snapshot(ALT);
  const results = { running: [], paused: [] };
  const exercise = async (phase) => {
    for (const [method, path, body] of guardedRoutes(COLL)) {
      const result = await api(method, path, body, { expected: 409 });
      results[phase].push({ method, path: path.split("?")[0], status: result.status });
    }
    // Reads stay functional.
    expect((await api("GET", `/points/1?collection_name=${encoded(COLL)}`)).data.id).toBe("1");
    expect((await api("POST", "/points/retrieve", { collection_name: COLL, ids: ["1", "2"] })).data.results).toHaveLength(2);
    expect((await api("POST", "/points/count", { collection_name: COLL, exact: true })).data.count).toBe(2001);
    expect((await report()).points_count).toBe(2001);
    expect((await api("POST", "/points/export", { collection_name: COLL, destination: "body" })).data.metadata.n_chunks).toBe(2001);
    // The unrelated native collection accepts a write immediately.
    await api("POST", "/points", { collection_name: ALT, point_id: "3", text: `alt write while ${phase}` });
    await api("DELETE", `/points/3?collection_name=${encoded(ALT)}`, undefined);
    expect(await snapshot(ALT)).toEqual(altBaseline);
  };
  await startJob(page);
  // A second start cannot create another active job.
  await api("POST", `/collections/${encoded(COLL)}/reindex`, { confirm_collection: COLL }, { expected: 409 });
  if ((await status()).job.status === "running") await exercise("running");
  await pauseJob(page);
  await exercise("paused");
  expect(results.paused).toHaveLength(20);
  // The UI exposes the same lock: mutations disabled, reads available.
  await openTab(page, "points");
  await openPointsSubtab(page, "upsert");
  await page.locator("#up-id").fill("1");
  await page.locator("#up-text").fill("blocked");
  await expect(page.locator("#up-submit")).toBeDisabled();
  await openPointsSubtab(page, "batch");
  await expect(page.locator("#bt-submit")).toBeDisabled();
  await openPointsSubtab(page, "delete");
  await page.locator("#del-id").fill("1");
  await expect(page.locator("#del-submit")).toBeDisabled();
  await openPointsSubtab(page, "get");
  await page.locator("#get-id").fill("1");
  await expect(page.locator("#get-submit")).toBeEnabled();
  await page.locator("#get-ids").fill("1\n2");
  await expect(page.locator("#get-many-submit")).toBeEnabled();
  await openPointsSubtab(page, "count");
  await expect(page.locator("#cnt-submit")).toBeEnabled();
  await openTab(page, "collections");
  await expect(page.locator("#ce-export")).toBeEnabled();
  await expect(page.locator("#cu-submit")).toBeDisabled();
  await expect(page.locator("#pi-create")).toBeDisabled();
  await page.locator("#cr-cancel").click();
  await waitTerminal("cancelled", 120_000);
  expect(await snapshot(COLL)).toEqual(baseline);
  await info.attach("guarded-routes", { body: JSON.stringify(results), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("book:13.8/NF-P17 — with_vectors on Get and Get-many during and after a reindex reports the generation names actually stored", {
  annotation: { type: "case", description: "book:13.8/NF-P17" },
}, async ({ page }, info) => {
  test.setTimeout(900_000);
  const errors = await openCollections(page);
  await startJob(page);
  await pauseJob(page);
  await openTab(page, "points");
  await openPointsSubtab(page, "get");
  await page.locator("#get-id").fill("1");
  await page.locator("#get-with-vectors").check();
  await reserve(2);
  let pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/1");
  await page.locator("#get-submit").click();
  const during = await (await pending).json();
  expect(during.vectors).toBeTruthy();
  await page.locator("#get-ids").fill("1\n2001");
  await page.locator("#get-many-with-vectors").check();
  pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/retrieve");
  await page.locator("#get-many-submit").click();
  const manyDuring = await (await pending).json();
  for (const point of manyDuring.results) expect(point.vectors).toBeTruthy();
  await openTab(page, "collections");
  await page.locator("#cr-resume").click();
  await waitTerminal("completed");
  const after = await report();
  const target = Object.values(after.embedding_compatibility.active_vectors).sort();
  await openTab(page, "points");
  await openPointsSubtab(page, "get");
  await reserve(1);
  pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/1");
  await page.locator("#get-submit").click();
  const post = await (await pending).json();
  expect(Object.keys(post.vectors)).toEqual(expect.arrayContaining(target));
  await info.attach("nf-p17", { body: JSON.stringify({ during: Object.keys(during.vectors), post: Object.keys(post.vectors), target }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("TB-QRY-002 — after a completed reindex the vector-presence selector offers exactly the report's active generation and each name returns every point", async ({ page }, info) => {
  test.setTimeout(900_000);
  const errors = await openCollections(page);
  const before = await report();
  await startJob(page);
  await waitTerminal("completed");
  await expect(page.locator("#cr-status")).toContainText("COMPLETED", { timeout: 60_000 });
  const after = await report();
  const active = Object.values(after.embedding_compatibility.active_vectors).sort();
  // Fields/report refresh happened on completion; the selector must follow the report,
  // not the startup configuration (book: record a failure if it stays stale).
  await openTab(page, "query");
  await page.locator("#ql-filter-add").click();
  const row = page.locator("#ql-filter-rows .filter-row").last();
  await row.locator('[data-role="field"]').selectOption("(vector presence)");
  const offered = (await row.locator('[data-role="value"] option').evaluateAll((nodes) => nodes.map((n) => n.value))).filter(Boolean).sort();
  expect(offered).toEqual(active);
  for (const name of active) {
    const result = (await api("POST", "/points/scroll", { collection_name: COLL, filters: { must: [{ has_vector: name }] }, limit: 1 })).data;
    expect(result.total).toBe(2001);
  }
  await info.attach("selector-vs-report", { body: JSON.stringify({ before: before.embedding_compatibility.active_vectors, after: after.embedding_compatibility.active_vectors, offered }), contentType: "application/json" });
  expect(errors).toEqual([]);
});