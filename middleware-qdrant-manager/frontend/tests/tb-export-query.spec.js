// Test book §7.6 export (TB-EXP-001/002/005/006/007), §7.9 query
// (TB-QRY-001/003/004/006/007/008/009/010) and §7.11 TB-API-003 — live against
// the `e2e_book_*` fixtures (npm run seed:book). Every export oracle is the
// generator; every ID is compared as a string; file-destination exports are
// read back from the bind-mounted exports directory.
// TB-EXP-003/004/008/009/010 and TB-API-001/002 are Backend-owned (middleware pytest).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "./fixtures.js";
import { fixtureName } from "./harness.mjs";
import { boot, openTab, openPointsSubtab, selectActive } from "./helpers.js";
import { clearFilters, condition, waitFieldsLoaded } from "./d-helpers.js";
import { api, bootPaced, fresh, reserve, snapshot, SCHEMA } from "./tb-live-helpers.js";
import { exportPoints } from "./tools/seed-book.mjs";

const FILTERS = fixtureName("e2e_book_filters");
const EXPORT = fixtureName("e2e_book_export");
const EMPTY = fixtureName("e2e_book_empty");
const exportDir = fileURLToPath(new URL("../../exports/", import.meta.url));
const active = { must: [{ key: "status", match: { value: "active" } }] };
const range = (a, b) => Array.from({ length: b - a + 1 }, (_, k) => String(a + k));

async function download(page, collection) {
  await selectActive(page, collection);
  await openTab(page, "collections");
  await reserve(2);
  const pending = page.waitForEvent("download");
  const response = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/export");
  await page.locator("#ce-export").click();
  const [file, res] = await Promise.all([pending, response]);
  expect(res.status()).toBe(200);
  const bytes = fs.readFileSync(await file.path());
  await expect(page.locator("#ce-result")).toContainText(/Exported/);
  return { body: JSON.parse(bytes), bytes, filename: file.suggestedFilename(), headers: res.headers() };
}

// Every point exactly once, as {point_id, text, ...payload}, ids as strings.
function exportOracle(body, points) {
  expect(body.chunks.length).toBe(points.length);
  expect(body.metadata.n_chunks).toBe(points.length);
  const expected = new Map(points.map((p) => [p.point_id, { ...p.payload, text: p.text, point_id: p.point_id }]));
  const seen = new Set();
  for (const chunk of body.chunks) {
    expect(typeof chunk.point_id).toBe("string");
    expect(seen.has(chunk.point_id)).toBe(false);
    seen.add(chunk.point_id);
    expect(chunk).toEqual(expected.get(chunk.point_id));
  }
}

async function openQuery(page) {
  const errors = await bootPaced(page, boot, { active: FILTERS });
  await waitFieldsLoaded(page);
  await expect(page.locator("#ql-payload-fields input")).toHaveCount(14);
  await clearFilters(page);
  await page.locator("#ql-query").fill("");
  return errors;
}

// ── TB-EXP ───────────────────────────────────────────────────────────────────

let uiExportChunks = null;
for (const mode of ["UI", "body", "file"]) {
  test(`TB-EXP-001/${mode} — the 2001-point export is complete, exact and identical across destinations`, async ({ page }, info) => {
    test.setTimeout(120000);
    let result;
    if (mode === "UI") {
      await bootPaced(page, boot, { active: EXPORT });
      result = await download(page, EXPORT);
      uiExportChunks = result.body.chunks;
    } else {
      const { data, headers } = await api("POST", "/points/export", { collection_name: EXPORT, destination: mode });
      if (mode === "body") result = { body: data, bytes: Buffer.from(JSON.stringify(data)), headers };
      else {
        expect(data.n_chunks).toBe(2001);
        const filename = path.basename(data.path);
        expect(filename.startsWith(`${EXPORT}_export_`)).toBe(true);
        const bytes = fs.readFileSync(path.join(exportDir, filename));
        expect(bytes.length).toBe(data.bytes);
        result = { body: JSON.parse(bytes), bytes, filename };
        // Separate requests agree semantically; timestamps/nonces may differ (TB-EXP-004 pins bytes).
        if (uiExportChunks) expect(result.body.chunks).toEqual(uiExportChunks);
      }
    }
    exportOracle(result.body, exportPoints);
    // Three Qdrant pages (1000 + 1000 + 1) with no repeated offset: 2001 unique ids.
    expect(new Set(result.body.chunks.map((c) => c.point_id)).size).toBe(2001);
    await info.attach("export-evidence", { body: JSON.stringify({ mode, bytes: result.bytes.length, filename: result.filename, headers: result.headers, n_chunks: result.body.chunks.length }), contentType: "application/json" });
  });
}

test("TB-EXP-002 — an empty collection yields no download, an empty body and no new file", async ({ page }, info) => {
  const errors = await bootPaced(page, boot, { active: EMPTY });
  await openTab(page, "collections");
  let downloads = 0;
  page.on("download", () => downloads++);
  await reserve(1);
  await page.locator("#ce-export").click();
  await expect(page.locator("#ce-result")).toContainText(/empty/i);
  expect(downloads).toBe(0);
  const before = fs.readdirSync(exportDir).sort();
  const body = (await api("POST", "/points/export", { collection_name: EMPTY, destination: "body" })).data;
  expect(body.chunks).toEqual([]);
  expect(body.metadata.n_chunks).toBe(0);
  const file = (await api("POST", "/points/export", { collection_name: EMPTY, destination: "file" })).data;
  expect(file).toMatchObject({ ok: true, n_chunks: 0, path: null });
  expect(fs.readdirSync(exportDir).sort()).toEqual(before);
  await info.attach("empty-responses", { body: JSON.stringify({ body, file }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

async function upload(page, name, chunks, custom) {
  await selectActive(page, name);
  await openTab(page, "points");
  await openPointsSubtab(page, "ingest");
  await page.locator("#ing-file").setInputFiles({ name: "e2e_book_source.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ chunks })) });
  await expect(page.locator("#ing-map-table")).toBeVisible();
  if (custom) {
    await page.locator('#ing-map-table tr[data-source-id="true"] .ing-map-name').fill("document");
    await page.locator('#ing-map-table tr[data-chunk-id="true"] .ing-map-name').fill("chunk_no");
  }
  await reserve(3);
  const wait = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/ingest-chunks");
  await page.locator("#ing-confirm-submit").click();
  expect((await wait).status()).toBe(200);
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");
  await expect(page.locator("#ing-confirm-submit")).toBeHidden();
}

for (const custom of [false, true]) {
  test(`${custom ? "TB-EXP-006" : "TB-EXP-005"} — ingest → export → ${custom ? "custom-name transform" : "strip ids"} → re-import reproduces the collection and is idempotent`, async ({ page }, info) => {
    test.setTimeout(240000);
    const a = await fresh("e2e_book_roundtrip_a"), b = await fresh("e2e_book_roundtrip_b");
    const errors = await bootPaced(page, boot, { active: a });
    const chunks = [{ id: 1, text: "alpha à漢", topic: "a" }, { id: 2, text: "beta", topic: "b" }];
    await upload(page, a, chunks, custom);
    const before = await snapshot(a);
    const file = await download(page, a);
    if (custom) {
      // The raw export cannot be re-imported as-is: the console preflight refuses it.
      await selectActive(page, b);
      await openTab(page, "points");
      await openPointsSubtab(page, "ingest");
      await page.locator("#ing-file").setInputFiles({ name: "e2e_book_source.json", mimeType: "application/json", buffer: file.bytes });
      await expect(page.locator("#ing-confirm-submit")).toBeHidden();
      await expect(page.locator("#ing-preflight-result")).toContainText(/invalid|integer|id/i);
    }
    const transformed = file.body.chunks.map((chunk) => custom
      ? (({ point_id, document, chunk_no, ...rest }) => ({ ...rest, id: chunk_no }))(chunk)
      : (({ point_id, source_id, ...rest }) => rest)(chunk));
    await upload(page, b, transformed, custom);
    expect(await snapshot(b)).toEqual(before);
    await upload(page, b, transformed, custom);
    expect(await snapshot(b)).toEqual(before);
    await info.attach("roundtrip", { body: JSON.stringify({ custom, before, transformed }), contentType: "application/json" });
    expect(errors).toEqual([]);
  });
}

test("TB-EXP-007 — a second export click while one is pending is ignored and no duplicate download occurs", async ({ page }) => {
  const errors = await bootPaced(page, boot, { active: EXPORT });
  await openTab(page, "collections");
  let requests = 0, downloads = 0;
  page.on("request", (r) => { if (new URL(r.url()).pathname === "/points/export") requests++; });
  page.on("download", () => downloads++);
  await reserve(2);
  const button = page.locator("#ce-export");
  const pending = page.waitForEvent("download");
  await button.click();
  await expect(button).toBeDisabled();
  await button.click({ force: true }).catch(() => {});
  await pending;
  await expect(page.locator("#ce-result")).toContainText(/Exported/);
  await expect(button).toBeEnabled();
  expect(requests).toBe(1);
  expect(downloads).toBe(1);
  expect(errors).toEqual([]);
});

// ── TB-QRY ───────────────────────────────────────────────────────────────────

test("TB-QRY-001 — the vector-presence filter offers exactly the report's active vectors and every point has each", async ({ page }, info) => {
  const errors = await openQuery(page);
  const report = (await api("GET", `/collections/${encodeURIComponent(FILTERS)}`)).data;
  await condition(page, { field: "(vector presence)", op: "has_vector" });
  const offered = (await page.locator('#ql-filter-rows [data-role="value"] option').evaluateAll((nodes) => nodes.map((n) => n.value))).filter(Boolean);
  const declared = Object.values(report.embedding_compatibility.active_vectors);
  expect(offered.sort()).toEqual([...declared].sort());
  const results = [];
  for (const name of offered) {
    for (const filtered of [false, true]) {
      const filters = { must: [{ has_vector: name }, ...(filtered ? active.must : [])] };
      const result = (await api("POST", "/points/scroll", { collection_name: FILTERS, filters, limit: 1000 })).data;
      const expected = range(1, filtered ? 20 : 100).sort();
      expect(result.results.map((p) => String(p.id)).sort()).toEqual(expected);
      results.push({ name, filtered, count: expected.length });
    }
  }
  await info.attach("vector-presence", { body: JSON.stringify({ declared, offered, results }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

for (const fusion of ["rrf", "dbsf"]) for (const rerank of [true, false]) for (const projection of ["all", "selected"]) {
  test(`TB-QRY-003/${fusion}-${rerank ? "rerank" : "norerank"}-${projection} — hybrid request carries the controls; hits honour the filter and projection`, async ({ page }, info) => {
    const errors = await openQuery(page);
    await page.locator("#ql-query").fill("cooking");
    await page.locator("#ql-limit").fill("5");
    await page.locator("#ql-cand").fill("20");
    await page.locator(`input[name="ql-fusion"][value="${fusion}"]`).check();
    await page.locator("#ql-rerank").setChecked(rerank);
    const choices = page.locator("#ql-payload-fields input");
    for (let i = 0; i < await choices.count(); i++) {
      const input = choices.nth(i);
      if (await input.isEnabled()) await input.setChecked(projection === "all" || await input.inputValue() === "status");
    }
    await condition(page, { field: "status", op: "eq", value: "active" });
    await reserve(1);
    const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/query/search");
    await page.locator("#ql-run").click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const body = await response.json();
    const sent = response.request().postDataJSON();
    expect(sent).toMatchObject({ query: "cooking", limit: 5, candidate_limit: 20, fusion, rerank, filters: active });
    const results = body.results;
    expect(results.length).toBeGreaterThan(0);
    expect(results.length).toBeLessThanOrEqual(5);
    expect(new Set(results.map((p) => String(p.id))).size).toBe(results.length);
    for (const result of results) expect(result.payload.status).toBe("active");
    const projected = rerank ? ["status", "text"] : ["status"];
    if (projection === "selected") {
      expect([...sent.with_payload].sort()).toEqual(projected);
      for (const hit of results) expect(Object.keys(hit.payload).sort()).toEqual(projected);
    }
    await expect(page.locator("#ql-results .result-card")).toHaveCount(results.length);
    await expect(page.locator("#ql-results")).toContainText(fusion);
    await expect(page.locator("#ql-results .metric").filter({ has: page.locator(".metric-label", { hasText: /^reranked$/ }) }).locator(".metric-value")).toHaveText(String(rerank));
    await info.attach("hybrid-request-response", { body: JSON.stringify({ sent, body }), contentType: "application/json" });
    expect(errors).toEqual([]);
  });
}

for (const variant of ["default", "weights-and-k", "dbsf-back-rrf"]) {
  test(`TB-QRY-004/${variant} — valid RRF controls are forwarded exactly and the toggle path leaves no residue`, async ({ page }, info) => {
    const errors = await openQuery(page);
    await page.locator("#ql-query").fill("cooking");
    if (variant !== "default") {
      await page.locator("#ql-rrfk-en").check();
      await page.locator("#ql-rrfk").fill("60");
      await page.locator("#ql-weights-en").check();
      await page.locator("#ql-balance").fill("0.8");
      await page.locator("#ql-weight-scale").fill("0.5");
      if (variant === "dbsf-back-rrf") {
        await page.locator('input[name="ql-fusion"][value="dbsf"]').check();
        await page.locator('input[name="ql-fusion"][value="rrf"]').check();
      }
    }
    await reserve(1);
    const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/query/search");
    await page.locator("#ql-run").click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const sent = response.request().postDataJSON();
    if (variant === "default") {
      expect(sent).not.toHaveProperty("rrf_k");
      expect(sent).not.toHaveProperty("rrf_weights");
    } else expect(sent).toMatchObject({ fusion: "rrf", rrf_k: 60, rrf_weights: [0.8, 0.2] });
    await expect(page.locator("#ql-results .result-card").first()).toBeVisible();
    await info.attach("rrf-live", { body: JSON.stringify({ sent, body: await response.json() }), contentType: "application/json" });
    expect(errors).toEqual([]);
  });
}

test("TB-QRY-006 — the draft walk covers 8 pages of 5 in id order, Previous returns, a new filter resets, order_by pages by views", async ({ page }, info) => {
  test.setTimeout(180000);
  const errors = await openQuery(page);
  await condition(page, { field: "status", op: "eq", value: "draft" });
  async function click(control) {
    await reserve(1);
    const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/scroll");
    await control.click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const body = await response.json();
    const expected = body.results.map((p) => String(p.id));
    await expect.poll(async () => (await page.locator("#ql-results .result-id").allTextContents()).map((x) => x.trim())).toEqual(expected);
    return body;
  }
  let body = await click(page.locator("#ql-run"));
  const ids = body.results.map((p) => String(p.id));
  for (let n = 2; n <= 8; n++) {
    body = await click(page.locator('[data-query-page="next"]'));
    ids.push(...body.results.map((p) => String(p.id)));
    await expect(page.locator(".pagination-page")).toContainText(`Page ${n} of 8`);
  }
  expect(ids).toEqual(range(51, 90));
  await expect(page.locator('[data-query-page="next"]')).toHaveCount(0);
  for (let n = 7; n >= 1; n--) {
    await click(page.locator('[data-query-page="previous"]'));
    await expect(page.locator(".pagination-page")).toContainText(`Page ${n} of 8`);
  }
  await clearFilters(page);
  await condition(page, { field: "status", op: "eq", value: "active" });
  body = await click(page.locator("#ql-run"));
  expect(body.results.map((p) => String(p.id))).toEqual(range(1, 5));
  await expect(page.locator(".pagination-page")).toContainText("Page 1 of 4");
  await clearFilters(page);
  await page.locator("#ql-order-en").check();
  await page.locator("#ql-order-key").selectOption("views");
  for (const direction of ["asc", "desc"]) {
    await page.locator(`input[name="ql-order-dir"][value="${direction}"]`).check();
    body = await click(page.locator("#ql-run"));
    expect(body.results.map((p) => String(p.id))).toEqual(direction === "asc" ? range(1, 5) : ["100", "99", "98", "97", "96"]);
    // order_by pages by start_from, not by Qdrant offsets.
    expect(body.next_page_offset).toBeNull();
    await expect(page.locator('[data-query-page="next"]')).toHaveCount(0);
  }
  await info.attach("draft-pages", { body: JSON.stringify({ ids, count: ids.length }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

for (const mode of ["payload", "ordered", "hybrid"]) {
  test(`TB-QRY-007/${mode} — the download holds all matching pages (payload-only) or the returned window (ordered/hybrid), ids as strings`, async ({ page }, info) => {
    const errors = await openQuery(page);
    await page.locator("#ql-query").fill(mode === "hybrid" ? "cooking" : "");
    if (mode === "payload") await condition(page, { field: "status", op: "eq", value: "draft" });
    if (mode === "ordered") {
      await page.locator("#ql-order-en").check();
      await page.locator("#ql-order-key").selectOption("views");
    }
    if (mode === "hybrid") {
      await page.locator("#ql-limit").fill("5");
      await condition(page, { field: "status", op: "eq", value: "active" });
    }
    await page.locator("#ql-download").check();
    await reserve(1);
    const pending = page.waitForEvent("download");
    await page.locator("#ql-run").click();
    const file = await pending;
    const body = JSON.parse(fs.readFileSync(await file.path(), "utf8"));
    const ids = body.chunks.map((p) => p.point_id);
    expect(ids.every((i) => typeof i === "string")).toBe(true);
    expect(new Set(ids).size).toBe(ids.length);
    const shown = (await page.locator("#ql-results .result-id").allTextContents()).map((x) => x.trim());
    if (mode === "payload") {
      // Unordered payload-only: every matching page, not just the displayed one.
      expect([...ids].sort((a, b) => Number(a) - Number(b))).toEqual(range(51, 90));
      expect(ids).toEqual(expect.arrayContaining(shown));
      for (const chunk of body.chunks) expect(chunk.status).toBe("draft");
    } else if (mode === "ordered") {
      expect(ids).toEqual(range(1, 5));
      expect(ids).toEqual(shown);
    }
    else {
      expect(ids.length).toBe(5);
      expect([...ids].sort()).toEqual([...shown].sort());
      for (const chunk of body.chunks) expect(chunk.status).toBe("active");
    }
    await info.attach("download-scope", { body: JSON.stringify({ mode, filename: file.suggestedFilename(), ids }), contentType: "application/json" });
    expect(errors).toEqual([]);
  });
}

for (const offset of [0, 1]) {
  test(`TB-QRY-008/offset${offset} — score_threshold 0.5 and offset ${offset} are accepted and echoed by the service`, async ({ page }, info) => {
    const errors = await openQuery(page);
    await page.locator("#ql-query").fill("cooking");
    await page.locator("#ql-threshold-en").check();
    await page.locator("#ql-threshold").fill("0.5");
    await page.locator("#ql-offset-en").check();
    await page.locator("#ql-offset").fill(String(offset));
    await reserve(1);
    const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/query/search");
    await page.locator("#ql-run").click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(response.request().postDataJSON()).toMatchObject({ score_threshold: 0.5, offset });
    expect(body).toMatchObject({ score_threshold: 0.5, offset });
    for (const hit of body.results) expect(hit.score).toBeGreaterThanOrEqual(0.5);
    await info.attach("threshold-live", { body: JSON.stringify({ sent: response.request().postDataJSON(), body }), contentType: "application/json" });
    expect(errors).toEqual([]);
  });
}

async function openCount(page) {
  const errors = await bootPaced(page, boot, { active: FILTERS });
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  return errors;
}

for (const exact of [true, false]) for (const filtered of [true, false]) {
  test(`TB-QRY-009/count-${exact ? "exact" : "approx"}-${filtered ? "filtered" : "all"} — the count is ${filtered ? 20 : 100} and the exact flag is forwarded`, async ({ page }, info) => {
    const errors = await openCount(page);
    await page.locator("#cnt-json").fill(JSON.stringify(filtered ? active : {}));
    await page.locator("#cnt-exact").setChecked(exact);
    await reserve(1);
    const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/count");
    await page.locator("#cnt-submit").click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(body.count).toBe(filtered ? 20 : 100);
    expect(response.request().postDataJSON().exact).toBe(exact);
    await expect(page.locator("#cnt-result")).toContainText(String(body.count));
    await info.attach("count", { body: JSON.stringify(body), contentType: "application/json" });
    expect(errors).toEqual([]);
  });
}

for (const key of ["status", "flag", "doc_uuid", "views"]) {
  test(`TB-QRY-009/facet-${key} — complete value frequencies through the browser`, async ({ page }, info) => {
    const errors = await openCount(page);
    await expect(page.locator("#cnt-facet-key")).toBeEnabled();
    await page.locator("#cnt-facet-key").selectOption(key);
    await page.locator("#cnt-facet-limit").fill("100");
    await reserve(1);
    const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/facet");
    await page.locator("#cnt-facet-submit").click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const body = await response.json();
    const counts = Object.fromEntries(body.hits.map((x) => [String(x.value), x.count]));
    if (key === "status") expect(counts).toEqual({ active: 20, archived: 30, draft: 40, deleted: 10 });
    else if (key === "flag") expect(counts).toEqual({ true: 50, false: 50 });
    else {
      expect(body.hits.length).toBe(100);
      expect(body.hits.every((h) => h.count === 1)).toBe(true);
    }
    await expect(page.locator("#cnt-facet-result tbody tr")).toHaveCount(body.hits.length);
    await info.attach("facet", { body: JSON.stringify(body), contentType: "application/json" });
    expect(errors).toEqual([]);
  });
}

for (const [variant, body, expected] of [["limit2", { key: "status", limit: 2 }, 2], ["active", { key: "status", filters: active, limit: 100 }, 1]]) {
  test(`TB-QRY-009/facet-${variant} — limit and filter shape the facet subset exactly`, async () => {
    const result = (await api("POST", "/points/facet", { collection_name: FILTERS, exact: true, ...body })).data;
    expect(result.hits.length).toBe(expected);
    if (variant === "active") expect(result.hits).toEqual([{ value: "active", count: 20 }]);
    else expect(result.hits.map((h) => h.count)).toEqual([40, 30]);
  });
}

for (const key of ["description", "location", "price", "created_at"]) {
  test(`TB-QRY-010/unsupported-${key} — a facet on a non-facetable index is an explicit 400 naming the key`, async () => {
    const result = await api("POST", "/points/facet", { collection_name: FILTERS, key }, { expected: 400 });
    expect(JSON.stringify(result.data)).toContain(key);
  });
}

test("TB-QRY-010 — the facet dropdown offers only facetable keys, an empty result is empty, and a switch clears stale rows", async ({ page }) => {
  const errors = await openCount(page);
  await expect(page.locator("#cnt-facet-key")).toBeEnabled();
  const options = await page.locator("#cnt-facet-key option").evaluateAll((nodes) => nodes.map((n) => n.value));
  for (const key of ["description", "location", "price", "created_at"]) expect(options).not.toContain(key);
  for (const key of ["status", "views", "flag", "doc_uuid"]) expect(options).toContain(key);
  await page.locator("#cnt-facet-key").selectOption("status");
  await reserve(1);
  let pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/facet");
  await page.locator("#cnt-facet-submit").click();
  expect((await pending).status()).toBe(200);
  await expect(page.locator("#cnt-facet-result tbody tr")).toHaveCount(4);
  const result = (await api("POST", "/points/facet", { collection_name: FILTERS, key: "status", filters: { must: [{ key: "views", range: { gt: 100 } }] } })).data;
  expect(result.hits).toEqual([]);
  await reserve(6);
  await page.locator("#conn-collection").selectOption(EMPTY);
  await expect(page.locator("#cnt-facet-key")).toBeDisabled();
  await expect(page.locator("#cnt-facet-result")).toHaveText("");
  expect(errors).toEqual([]);
});

// ── TB-API-003 ───────────────────────────────────────────────────────────────

const NAMED = fixtureName("e2e_book_named");
async function seedNamed() {
  await api("POST", "/collections", { collection_name: NAMED, recreate: true, confirm_recreate: NAMED, ...SCHEMA });
  await api("POST", "/points", { collection_name: NAMED, point_id: "1", text: "confirmation control" });
  return snapshot(NAMED);
}

for (const [label, method, path_, body] of [
  ["empty-toggles", "PATCH", `/collections/${NAMED}/schema-toggles`, {}],
  ["non-text-phrase", "POST", `/collections/${NAMED}/payload-indexes`, { indexes: [{ field_name: "field", field_schema: "keyword", phrase_matching: true }] }],
  ["malformed-batch", "POST", "/points/batch", { collection_name: NAMED, points: [{ point_id: "1", text: "valid must not write" }, { point_id: "bad", text: "invalid" }] }],
]) {
  test(`TB-API-003/${label} — exact HTTP 400 and no mutation`, async ({}, info) => {
    const before = await seedNamed();
    const result = await api(method, path_, body, { expected: 400 });
    expect(await snapshot(NAMED)).toEqual(before);
    await info.attach("response", { body: JSON.stringify(result.data), contentType: "application/json" });
  });
}

for (const operation of ["recreate", "delete", "reindex"]) {
  for (const [label, confirm] of [["empty", ""], ["wrong", "wrong"], ["padded", ` ${NAMED} `], ["exact", NAMED]]) {
    test(`TB-API-003/${operation}-${label} — raw API confirmation is exact string equality`, async ({}, info) => {
      test.setTimeout(120000);
      const before = await seedNamed();
      const expected = label === "exact" ? 200 : 400;
      let result;
      if (operation === "recreate") result = await api("POST", "/collections", { collection_name: NAMED, recreate: true, confirm_recreate: confirm, ...SCHEMA }, { expected });
      else if (operation === "delete") result = await api("DELETE", `/collections/${NAMED}?confirm_delete=${encodeURIComponent(confirm)}`, undefined, { expected });
      else result = await api("POST", `/collections/${NAMED}/reindex`, { confirm_collection: confirm }, { expected });
      if (label !== "exact") expect(await snapshot(NAMED)).toEqual(before);
      else if (operation === "delete") await api("GET", `/collections/${NAMED}`, undefined, { expected: 404 });
      else if (operation === "recreate") expect(await snapshot(NAMED)).toEqual([]);
      else {
        let state;
        for (let n = 0; n < 60; n++) {
          state = (await api("GET", `/collections/${NAMED}/reindex`)).data;
          if (["completed", "cancelled", "failed"].includes(state.job?.status)) break;
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
        expect(state.job.status).toBe("completed");
        expect(await snapshot(NAMED)).toEqual(before);
      }
      await info.attach("confirmation", { body: JSON.stringify({ operation, label, status: result.status, body: result.data }), contentType: "application/json" });
    });
  }
}
