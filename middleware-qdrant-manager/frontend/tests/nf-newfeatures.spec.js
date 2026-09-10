import { fixtureName, requireResults } from "./harness.mjs";
import { expectHttpFailure } from "./telemetry.mjs";
// §13 Serie NF — the four newly exposed Qdrant capabilities:
//   NF-A  Points → Count subtab
//   NF-B  Payload scope POINT ID | FILTER
//   NF-C  order_by on the payload-only scroll
//   NF-D  score_threshold + offset on the hybrid search
//   NF-Z  theme/UX conformance of the new controls
import { expect, test } from "./fixtures.js";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { boot, selectActive, openTab, openPointsSubtab, apiFetch, setupFetch, BASE_URL, TOKEN } from "./helpers.js";

const TESTS_DIR = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_DIR = path.dirname(TESTS_DIR);
const QUERY_COLL = fixtureName("e2e_d_query");   // 12 points, fully indexed, read fixture
const BULK_COLL = fixtureName("e2e_nf_bulk");    // recreated per test group, written to
const DEEP_COLL = fixtureName("e2e_nf_deep");    // dedicated deep-ranking pagination fixture

const api = (request, method, path, body) => apiFetch(request, method, path, body);

// Rebuild the bulk sandbox: 12 points, f_flag true on even ids, indexes on the
// fields the filters use.
async function reseedBulk(request) {
  await setupFetch(request, "POST", "/collections", {
    collection_name: BULK_COLL, recreate: true, confirm_recreate: BULK_COLL,
    dense_on_disk: true, sparse_on_disk: true, on_disk_payload: true,
    quantization: true, sparse_idf: true,
  });
  const points = Array.from({ length: 12 }, (_, k) => {
    const i = k + 1;
    return {
      point_id: String(i),
      text: `bulk document ${i} about payload operations`,
      payload: { f_flag: i % 2 === 0, f_views: i, f_status: `s${i}` },
    };
  });
  await setupFetch(request, "POST", "/points/batch", { collection_name: BULK_COLL, points });
  for (const [field, schema] of [["f_flag", "bool"], ["f_views", "integer"], ["f_status", "keyword"]]) {
    await setupFetch(request, "POST", `/collections/${BULK_COLL}/payload-indexes`,
      { indexes: [{ field_name: field, field_schema: schema }] });
  }
}

async function reseedDeepRanking(request) {
  const created = await api(request, "POST", "/collections", {
    collection_name: DEEP_COLL, recreate: true, confirm_recreate: DEEP_COLL,
    dense_on_disk: true, sparse_on_disk: true, on_disk_payload: true,
    quantization: true, sparse_idf: true,
  });
  expect(created.status()).toBe(200);
  const points = Array.from({ length: 40 }, (_, k) => {
    const i = k + 1;
    return {
      point_id: String(i),
      text: `deep ranking document ${i} about hybrid retrieval and ranking`,
      payload: { sequence: i },
    };
  });
  const seeded = await api(request, "POST", "/points/batch", {
    collection_name: DEEP_COLL,
    points,
  });
  expect(seeded.status()).toBe(200);
}
async function payloadsOf(request, collection) {
  const res = await setupFetch(request, "POST", "/points/scroll",
    { collection_name: collection, limit: 50, with_payload: true, include_total: false });
  return requireResults(await res.json());
}

// loadCollectionFields() fetches the payload indexes asynchronously after the
// collection becomes active; the facet key select re-renders once that
// resolves. Reading its options right after opening the Count subtab races
// that fetch — the select still shows the static "no facetable field"
// placeholder until it does. Tests against a collection that does have
// facetable fields must wait for the real options before asserting on them.
async function waitForFacetKeys(page) {
  await expect
    .poll(async () => page.locator("#cnt-facet-key option").count(), { timeout: 30_000 })
    .toBeGreaterThan(1);
}

// ════════════════════ NF-A — Count subtab ════════════════════

test("NF-A1 — the Count subtab opens with an empty filter and exact checked", async ({ page }) => {
  const errors = await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await expect(page.locator('[data-pt-tab="count"]')).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#cnt-json")).toHaveValue("{}");
  await expect(page.locator("#cnt-exact")).toBeChecked();
  await expect(page.locator("#cnt-submit")).toBeEnabled();
  expect(errors).toEqual([]);
});

test("NF-A2 — counting with an empty filter counts the whole collection", async ({ page, request }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await page.locator("#cnt-submit").click();
  await expect(page.locator("#cnt-result .alert-success")).toContainText("12 point(s)");
  await expect(page.locator("#cnt-result")).toContainText("exact");
  const rep = await (await api(request, "GET", `/collections/${QUERY_COLL}`)).json();
  expect(rep.points_count).toBe(12);
});

test("NF-A3 — counting with a filter matches the filtered subset", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await page.locator("#cnt-json").fill('{"must":[{"key":"f_flag","match":{"value":true}}]}');
  await page.locator("#cnt-submit").click();
  // Fixture: f_flag true on the 6 even ids.
  await expect(page.locator("#cnt-result .alert-success")).toContainText("6 point(s)");
  await expect(page.locator("#cnt-result")).toContainText("matching the filter");
});

test("NF-A4 — unchecking exact reports an estimate", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await page.locator("#cnt-exact").uncheck();
  await page.locator("#cnt-submit").click();
  await expect(page.locator("#cnt-result")).toContainText("estimated");
});

test("NF-A5 — malformed filter JSON is reported locally, no request sent", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  let calls = 0;
  await page.route("**/points/count", (route) => { calls += 1; return route.fallback(); });
  await page.locator("#cnt-json").fill('{"must": [');
  await page.locator("#cnt-submit").click();
  await expect(page.locator("#cnt-result")).toContainText(/invalid filter json/i);
  expect(calls).toBe(0);
});

test("NF-A6 — a non-indexed field yields a readable backend error", async ({ page }) => {
  expectHttpFailure(page, 'POST', '/points/count', 400);
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await page.locator("#cnt-json").fill('{"must":[{"key":"f_extra","match":{"value":"x1"}}]}');
  await page.locator("#cnt-submit").click();
  await expect(page.locator("#cnt-result .alert-error")).toContainText(/index/i);
});

test("NF-A7 — switching the active collection clears the count result (K5)", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await page.locator("#cnt-submit").click();
  await expect(page.locator("#cnt-result")).not.toBeEmpty();
  const options = await page.locator("#conn-collection option").allTextContents();
  const other = options.find((o) => o !== QUERY_COLL);
  test.skip(!other, "no second collection");
  await selectActive(page, other);
  await expect(page.locator("#cnt-result")).toBeEmpty();
});

test("NF-A8 — double-clicking Count fires a single request (K1)", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  let calls = 0;
  await page.route("**/points/count", async (route) => {
    calls += 1;
    await new Promise((r) => setTimeout(r, 900));
    return route.fallback();
  });
  await page.locator("#cnt-submit").click({ clickCount: 2 });
  await expect(page.locator("#cnt-result")).not.toBeEmpty({ timeout: 30_000 });
  expect(calls).toBe(1);
});

test("NF-A9 — the Count subtab carries no destructive control", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await expect(page.locator("#pt-count .btn-danger")).toHaveCount(0);
  await expect(page.locator("#pt-count input[id*='confirm']")).toHaveCount(0);
});

// ════════════════════ NF-B — Payload scope ════════════════════

test("NF-B1/B2/B3 — scope switch reveals the filter and locks REPLACE", async ({ page }) => {
  const errors = await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "payload");

  // B1: default scope is point id. The explanation lives in the Feature
  // reference now, not an inline caption (NF-P13 checks it is there).
  await expect(page.locator('input[name="pl-scope"][value="point"]')).toBeChecked();
  await expect(page.locator("#pl-id-field")).toBeVisible();
  await expect(page.locator("#pl-filter-field")).toBeHidden();

  // B2: filter scope.
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await expect(page.locator("#pl-id-field")).toBeHidden();
  await expect(page.locator("#pl-filter-field")).toBeVisible();

  // B3: REPLACE is not offered for a bulk write.
  await expect(page.locator('input[name="pl-mode"][value="PUT"]')).toBeDisabled();
  await expect(page.locator('input[name="pl-mode"][value="PATCH"]')).toBeChecked();
  expect(errors).toEqual([]);
});

test("NF-B4 — Count matches previews the scope of a bulk write", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "payload");
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await expect(page.locator("#pl-filter-count-submit")).toBeDisabled();
  await page.locator("#pl-filter-json").fill('{"must":[{"key":"f_flag","match":{"value":true}}]}');
  await expect(page.locator("#pl-filter-count-submit")).toBeEnabled();
  await page.locator("#pl-filter-count-submit").click();
  await expect(page.locator("#pl-filter-count")).toContainText("6 point(s) match");
});

test("NF-B5 — MERGE by filter writes only to the matching points", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "payload");
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await page.locator("#pl-filter-json").fill('{"must":[{"key":"f_flag","match":{"value":true}}]}');
  await page.locator("#pl-json").fill('{"bulk_tag":"on"}');
  await expect(page.locator("#pl-submit")).toBeEnabled();
  await page.locator("#pl-submit").click();
  await expect(page.locator("#pl-result .alert-success")).toContainText("6 point(s)");

  const points = await payloadsOf(request, BULK_COLL);
  const tagged = points.filter((p) => "bulk_tag" in p.payload);
  expect(tagged.length).toBe(6);
  for (const p of tagged) expect(p.payload.f_flag).toBe(true);
  // The pre-existing payload is preserved (merge, not replace).
  for (const p of tagged) expect(p.payload).toHaveProperty("f_views");
});

test("NF-B6 — Delete keys by filter removes the key only where it matched", async ({ page, request }) => {
  await reseedBulk(request);
  await api(request, "PATCH", "/points/payload", {
    collection_name: BULK_COLL,
    filters: { must: [{ key: "f_flag", match: { value: true } }] },
    payload: { bulk_tag: "on" },
  });
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "payload");
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await page.locator("#pl-filter-json").fill('{"must":[{"key":"f_flag","match":{"value":true}}]}');
  await page.locator("#pl-keys").fill("bulk_tag");
  await expect(page.locator("#pl-del-keys-submit")).toBeEnabled();
  await page.locator("#pl-del-keys-submit").click();
  await expect(page.locator("#pl-keys-result .alert-success")).toContainText("6 point(s)");

  const points = await payloadsOf(request, BULK_COLL);
  expect(points.filter((p) => "bulk_tag" in p.payload).length).toBe(0);
  // Untouched fields survive.
  expect(points.every((p) => "f_views" in p.payload)).toBe(true);
});

test("NF-B7 — Clear payload by filter keeps protected fields and spares non-matches", {
  // NF-Z5: the filtered clear reuses the existing type-to-confirm gate.
  annotation: { type: "case", description: "book:13.7/NF-Z5" },
}, async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "payload");
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await page.locator("#pl-filter-json").fill('{"must":[{"key":"f_flag","match":{"value":true}}]}');
  await expect(page.locator("#pl-clear-submit")).toBeDisabled();
  await page.locator("#pl-clear-confirm").fill(BULK_COLL);
  await expect(page.locator("#pl-clear-submit")).toBeEnabled();
  await page.locator("#pl-clear-submit").click();
  await expect(page.locator("#pl-clear-result .alert-success")).toContainText("6 point(s)");

  const points = await payloadsOf(request, BULK_COLL);
  const cleared = points.filter((p) => p.payload.f_flag === undefined);
  const kept = points.filter((p) => p.payload.f_flag === false);
  expect(cleared.length).toBe(6);
  expect(kept.length).toBe(6);
  for (const p of cleared) expect(Object.keys(p.payload).sort()).toEqual(["text"]);
  for (const p of kept) expect(p.payload).toHaveProperty("f_views");
});

test("NF-B8 — a match-all filter cannot be submitted, and the API refuses it", async ({ page, request }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "payload");
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await page.locator("#pl-json").fill('{"x":1}');
  // Left at "{}" the UI keeps every bulk write disabled.
  await expect(page.locator("#pl-submit")).toBeDisabled();
  await expect(page.locator("#pl-filter-count-submit")).toBeDisabled();
  // [API] the backend refuses it too, so this is not a client-only guard.
  // An empty object is stopped by the request model (min_length=1), exactly as
  // DeleteByFilterRequest does…
  const empty = await api(request, "PATCH", "/points/payload",
    { collection_name: QUERY_COLL, filters: {}, payload: { x: 1 } });
  expect(empty.status()).toBe(400);
  expect(JSON.stringify(await empty.json())).toMatch(/at least 1 item/i);
  // …and a filter that is shaped but carries no condition is stopped by the
  // shared filter compiler, with the field named in the message (422).
  const hollow = await api(request, "PATCH", "/points/payload",
    { collection_name: QUERY_COLL, filters: { must: [] }, payload: { x: 1 } });
  expect(hollow.status()).toBe(422);
  expect(JSON.stringify(await hollow.json())).toMatch(/must be a non-empty list/i);
});

test("NF-B9 — deleting a protected key by filter is refused", async ({ page, request }) => {
  expectHttpFailure(page, 'POST', '/points/payload/delete-keys', 400);
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "payload");
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await page.locator("#pl-filter-json").fill('{"must":[{"key":"f_flag","match":{"value":true}}]}');
  await page.locator("#pl-keys").fill("text");
  await page.locator("#pl-del-keys-submit").click();
  await expect(page.locator("#pl-keys-result .alert-error")).toContainText(/protected payload fields/i);
  const points = await payloadsOf(request, BULK_COLL);
  expect(points.every((p) => "text" in p.payload)).toBe(true);
});

test("NF-B10 — the point-id scope still behaves as before", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "payload");
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await page.locator('input[name="pl-scope"][value="point"]').check();
  await expect(page.locator('input[name="pl-mode"][value="PUT"]')).toBeEnabled();
  await page.locator("#pl-id").fill("1");
  await page.locator("#pl-json").fill('{"single":"yes"}');
  await page.locator("#pl-submit").click();
  await expect(page.locator("#pl-result .alert-success")).toContainText("point '1'");
  const points = await payloadsOf(request, BULK_COLL);
  expect(points.filter((p) => p.payload.single === "yes").length).toBe(1);
});

test("NF-B11 — double-click protection on the three filtered writes (K1)", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "payload");
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await page.locator("#pl-filter-json").fill('{"must":[{"key":"f_flag","match":{"value":true}}]}');

  for (const [button, route, prep] of [
    ["#pl-submit", "**/points/payload", async () => page.locator("#pl-json").fill('{"a":1}')],
    ["#pl-del-keys-submit", "**/points/payload/delete-keys", async () => page.locator("#pl-keys").fill("a")],
    ["#pl-clear-submit", "**/points/payload/clear", async () => page.locator("#pl-clear-confirm").fill(BULK_COLL)],
  ]) {
    let calls = 0;
    await page.route(route, async (r) => {
      if (r.request().url().endsWith("/points/payload")
          && r.request().method() !== "PATCH") return r.continue();
      calls += 1;
      await new Promise((res) => setTimeout(res, 900));
      return r.continue();
    });
    await prep();
    await page.locator(button).click({ clickCount: 2 });
    await page.waitForTimeout(2000);
    expect(calls, `${button} fired ${calls} requests`).toBe(1);
    await page.unroute(route);
  }
});

test("NF-B12 — changing collection clears the filtered payload results (K5)", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "payload");
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await page.locator("#pl-filter-json").fill('{"must":[{"key":"f_flag","match":{"value":true}}]}');
  await page.locator("#pl-filter-count-submit").click();
  await expect(page.locator("#pl-filter-count")).not.toBeEmpty();
  await selectActive(page, QUERY_COLL);
  await expect(page.locator("#pl-filter-count")).toBeHidden();
  await expect(page.locator("#pl-result")).toBeEmpty();
});

// ════════════════════ NF-C — order_by ════════════════════

async function openQueryPayloadOnly(page) {
  const errors = await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "query");
  await page.locator("#ql-query").fill("");
  await expect
    .poll(async () => page.locator("#ql-payload-fields input").count(), { timeout: 30_000 })
    .toBeGreaterThan(5);
  for (let g = 0; g < 40; g += 1) {
    const remove = page.locator('#ql-filter-rows [data-action="remove"]');
    if (!(await remove.count())) break;
    await remove.first().click();
  }
  return errors;
}

async function resultIds(page) {
  return (await page.locator("#ql-results .result-id").allTextContents()).map((t) => t.trim());
}

test("NF-C1 — order_by is disabled while a hybrid query is typed", async ({ page }) => {
  await openQueryPayloadOnly(page);
  await page.locator("#ql-query").fill("hybrid vector search");
  await expect(page.locator("#ql-order-en")).toBeDisabled();
  // The page carries no inline explanation any more (it lives in the Parameter
  // reference); only the reason it is unavailable, as a title.
  await expect(page.locator("#ql-order-en"))
    .toHaveAttribute("title", /payload-only mode only/i);
});

test("NF-C2/C3 — payload-only unlocks order_by; keys are the indexed fields only", async ({ page }) => {
  await openQueryPayloadOnly(page);
  await expect(page.locator("#ql-order-en")).toBeEnabled();
  await expect(page.locator("#ql-order-key")).toBeDisabled();
  await page.locator("#ql-order-en").check();
  await expect(page.locator("#ql-order-key")).toBeEnabled();
  await expect(page.locator("#ql-order-start")).toBeEnabled();
  const keys = await page.locator("#ql-order-key option").allTextContents();
  expect(keys).toContain("f_views");
  // f_extra exists in the payload but has no index: Qdrant cannot order by it.
  expect(keys).not.toContain("f_extra");
});

test("NF-C4/C5/C6/C10 — ordered scroll sorts by the field and drops offset paging", async ({ page }) => {
  await openQueryPayloadOnly(page);
  await page.locator("#ql-order-en").check();
  await page.locator("#ql-order-key").selectOption("f_views");
  await page.locator('input[name="ql-order-dir"][value="desc"]').check();
  // C10: the preview carries the spec.
  await expect(page.locator("#ql-request-json")).toContainText('"order_by"');
  await expect(page.locator("#ql-request-json")).toContainText('"f_views"');
  await expect(page.locator("#ql-request-json")).toContainText('"desc"');

  await page.locator("#ql-run").click();
  await expect.poll(async () => (await resultIds(page)).length, { timeout: 60_000 }).toBeGreaterThan(0);
  // f_views equals the numeric id, so desc order is 12, 11, 10 …
  const desc = (await resultIds(page)).map(Number);
  expect(desc).toEqual([...desc].sort((a, b) => b - a));
  expect(desc[0]).toBe(12);

  // C6: no offset pages for an ordered scroll.
  await expect(page.locator('[data-query-page="next"]')).toHaveCount(0);
  await expect(page.locator(".pagination-page")).toContainText("Ordered by f_views desc");

  // C5: ascending.
  await page.locator('input[name="ql-order-dir"][value="asc"]').check();
  await page.locator("#ql-run").click();
  await expect.poll(async () => (await resultIds(page))[0], { timeout: 60_000 }).toBe("1");
  const asc = (await resultIds(page)).map(Number);
  expect(asc).toEqual([...asc].sort((a, b) => a - b));
});

test("NF-C7 — start_from resumes the ordered page at the given value", async ({ page }) => {
  await openQueryPayloadOnly(page);
  await page.locator("#ql-order-en").check();
  await page.locator("#ql-order-key").selectOption("f_views");
  await page.locator('input[name="ql-order-dir"][value="desc"]').check();
  await page.locator("#ql-order-start").fill("8");
  await page.locator("#ql-run").click();
  await expect.poll(async () => (await resultIds(page))[0], { timeout: 60_000 }).toBe("8");
  const ids = (await resultIds(page)).map(Number);
  expect(Math.max(...ids)).toBeLessThanOrEqual(8);
});

test("NF-C8 — a collection with no orderable field offers no order key", async ({ page, request }) => {
  const bare = fixtureName("e2e_nf_bare");
  await setupFetch(request, "POST", "/collections", {
    collection_name: bare, recreate: true, confirm_recreate: bare,
    dense_on_disk: true, sparse_on_disk: true, on_disk_payload: true,
    quantization: true, sparse_idf: true,
  });
  await setupFetch(request, "POST", "/points/batch", {
    collection_name: bare,
    points: [{ point_id: "1", text: "bare point", payload: { plain: "x" } }],
  });
  await boot(page, { active: bare });
  await selectActive(page, bare);
  await openTab(page, "query");
  await page.locator("#ql-query").fill("");
  await expect(page.locator("#ql-order-key")).toContainText("no orderable field");
  await expect(page.locator("#ql-order-en")).not.toBeChecked();
});

test("NF-C11 — order_by lives in section 03, after the payload field list", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "query");
  // It only applies to the payload-only scroll, so it must NOT sit in 01 Hybrid
  // Search (where it is dead as soon as a query is typed) but in 03, after the
  // with_payload picker and before 04 Run Query.
  const placement = await page.evaluate(() => {
    const order = document.getElementById("ql-order-field");
    const payload = document.getElementById("ql-payload-fields");
    const rerank = document.getElementById("ql-rerank");
    const run = document.getElementById("ql-request-json");
    const after = (a, b) =>
      Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
    const sections = [...document.querySelectorAll("#panel-query .section")];
    // Section whose header precedes the order block most closely.
    const owning = sections.filter((s) => after(s, order)).pop();
    return {
      afterPayloadFields: after(payload, order),
      afterRerank: after(rerank, order),
      beforeRun: after(order, run),
      section: owning ? owning.textContent.replace(/\s+/g, " ").trim() : null,
    };
  });
  expect(placement.afterPayloadFields, "order_by must follow with_payload").toBe(true);
  expect(placement.afterRerank, "order_by must not sit in the rerank block").toBe(true);
  expect(placement.beforeRun, "order_by must precede 04 Run Query").toBe(true);
  expect(placement.section).toContain("Returned Payload");
  // Its own labelled group with the theme's subhead separator.
  await expect(page.locator("#ql-order-field > legend")).toHaveText("order_by");
  await expect(page.locator("#panel-query h3.subhead", { hasText: "Result order" })).toHaveCount(1);
});

test("NF-C9 — [API] order_by together with offset is refused with guidance", async ({ request }) => {
  const res = await api(request, "POST", "/points/scroll", {
    collection_name: QUERY_COLL, limit: 3, offset: 3,
    order_by: { key: "f_views", direction: "desc" },
  });
  expect(res.status()).toBe(422);
  expect(JSON.stringify(await res.json())).toMatch(/start_from/);
});

// ════════════════════ NF-D — score_threshold + offset ════════════════════

async function openQueryHybrid(page) {
  const errors = await openQueryPayloadOnly(page);
  await page.locator("#ql-query").fill("hybrid vector search and retrieval");
  await page.locator("#ql-rerank").uncheck();
  return errors;
}

test("NF-D1 — both controls are disabled in payload-only mode", async ({ page }) => {
  await openQueryPayloadOnly(page);
  await expect(page.locator("#ql-threshold-en")).toBeDisabled();
  await expect(page.locator("#ql-offset-en")).toBeDisabled();
});

test("NF-D2 — a typed query unlocks them, fields stay gated by the checkbox", async ({ page }) => {
  await openQueryHybrid(page);
  await expect(page.locator("#ql-threshold-en")).toBeEnabled();
  await expect(page.locator("#ql-offset-en")).toBeEnabled();
  await expect(page.locator("#ql-threshold")).toBeDisabled();
  await expect(page.locator("#ql-offset")).toBeDisabled();
  await page.locator("#ql-threshold-en").check();
  await expect(page.locator("#ql-threshold")).toBeEnabled();
  await page.locator("#ql-offset-en").check();
  await expect(page.locator("#ql-offset")).toBeEnabled();
});

async function runHybrid(page) {
  const response = page.waitForResponse(
    (r) => new URL(r.url()).pathname === "/query/search" && r.request().method() === "POST",
    { timeout: 60_000 });
  await page.locator("#ql-run").click();
  const result = await response;
  expect(result.status(), 'Hybrid result prerequisite').toBe(200);
  return requireResults(await result.json()).map((h) => ({ id: String(h.id), score: h.score }));
}

test("NF-D3/D4 — score_threshold keeps only hits at or above the cut", async ({ page }) => {
  await openQueryHybrid(page);
  const baseline = await runHybrid(page);
  expect(baseline.length).toBeGreaterThan(2);

  // Cut just under the third score: the result set must shrink and stay above it.
  const cut = baseline[2].score;
  await page.locator("#ql-threshold-en").check();
  await page.locator("#ql-threshold").fill(String(cut));
  await expect(page.locator("#ql-request-json")).toContainText('"score_threshold"');
  const cutHits = await runHybrid(page);
  expect(cutHits.length).toBeLessThan(baseline.length);
  for (const hit of cutHits) expect(hit.score).toBeGreaterThanOrEqual(cut - 1e-9);
});

test("NF-D5 — an unreachable threshold returns an empty, readable result", async ({ page }) => {
  await openQueryHybrid(page);
  await page.locator("#ql-threshold-en").check();
  await page.locator("#ql-threshold").fill("99");
  const hits = await runHybrid(page);
  expect(hits.length).toBe(0);
  await expect(page.locator("#ql-results")).not.toBeEmpty();
  await expect(page.locator("#ql-errors .alert-error")).toHaveCount(0);
});

test("NF-D6 — offset shifts the hybrid result window", async ({ page }) => {
  await openQueryHybrid(page);
  const baseline = await runHybrid(page);
  expect(baseline.length).toBeGreaterThan(3);
  await page.locator("#ql-offset-en").check();
  await page.locator("#ql-offset").fill("2");
  await expect(page.locator("#ql-request-json")).toContainText('"offset"');
  const shifted = await runHybrid(page);
  expect(shifted.length).toBeGreaterThan(0);
  // The first two baseline hits are gone and the window starts at baseline[2].
  expect(shifted[0].id).toBe(baseline[2].id);
  expect(shifted.map((h) => h.id)).not.toContain(baseline[0].id);
});

test("NF-D7 — the explanations live in the Parameter reference, not inline", async ({ page }) => {
  await openQueryHybrid(page);
  // Full explanations stay in the reference; accessible disabled reasons are
  // cleared when a control becomes available.
  await expect(page.locator("#ql-threshold-caption")).toHaveCount(0);
  await expect(page.locator("#ql-order-caption")).toHaveCount(0);
  await page.locator('#ql-offset-en').check();
  await expect(page.locator('#ql-offset')).toBeEnabled();
  await expect(page.locator("#ql-offset-field .caption")).toBeHidden();
  // Only the unavailability reason is surfaced inline, as a title.
  await expect(page.locator("#ql-threshold-en")).not.toHaveAttribute("title", /./);

  // The reference carries an entry per new parameter, each stating the facts the
  // captions used to carry (fused score, before rerank, mode restrictions).
  const reference = page.locator("#query-reference");
  await page.locator("#panel-query details.reference").first()
    .evaluate((el) => { el.open = true; });
  for (const term of ["score_threshold", "offset", "order_by", "start_from"]) {
    await expect(reference).toContainText(term);
  }
  const text = (await reference.textContent()) || "";
  expect(text).toMatch(/FUSED score/i);
  expect(text).toMatch(/before any rerank/i);
  expect(text).toMatch(/Hybrid search only/i);
  expect(text).toMatch(/refuses offset together with order_by/i);
  expect(text).toMatch(/inclusive/i);
  // The retracted draft numbers must not resurface anywhere.
  expect(text).not.toMatch(/0\.50 · 0\.33/);
});

test("NF-D7b — payload-only mode explains why the hybrid-only controls are off", async ({ page }) => {
  await openQueryPayloadOnly(page);
  for (const id of ["#ql-threshold-en", "#ql-offset-en"]) {
    await expect(page.locator(id)).toBeDisabled();
    await expect(page.locator(id)).toHaveAttribute("title", /hybrid search only/i);
  }
});

test("NF-D11 — threshold/offset sit with the fusion block, before the rerank controls", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "query");
  // They act on the fused result, so the DOM order must read
  // fusion → threshold/offset → rerank.
  const order = await page.evaluate(() => {
    const ids = ["ql-rrfk-en", "ql-threshold-en", "ql-offset-en", "ql-rerank", "ql-topk-en"];
    const nodes = ids.map((id) => document.getElementById(id));
    return ids.map((id, i) => ({
      id,
      pos: nodes.slice(0, i).every((prev) =>
        prev.compareDocumentPosition(nodes[i]) & Node.DOCUMENT_POSITION_FOLLOWING),
    }));
  });
  for (const entry of order) {
    expect(entry.pos, `${entry.id} is out of pipeline order`).toBe(true);
  }
});

test("NF-D8 — the preview carries each field only while its toggle is on", async ({ page }) => {
  await openQueryHybrid(page);
  await expect(page.locator("#ql-request-json")).not.toContainText('"score_threshold"');
  await expect(page.locator("#ql-request-json")).not.toContainText('"offset"');
  await page.locator("#ql-threshold-en").check();
  await expect(page.locator("#ql-request-json")).toContainText('"score_threshold"');
  await page.locator("#ql-threshold-en").uncheck();
  await expect(page.locator("#ql-request-json")).not.toContainText('"score_threshold"');
});

test("NF-D9 — a negative offset blocks the run with a readable error", async ({ page }) => {
  await openQueryHybrid(page);
  await page.locator("#ql-offset-en").check();
  await page.locator("#ql-offset").fill("-5");
  await expect(page.locator("#ql-errors")).toContainText(/offset/i);
  await expect(page.locator("#ql-run")).toBeDisabled();
});

test("NF-D10 — the toggles survive a tab round-trip", async ({ page }) => {
  await openQueryHybrid(page);
  await page.locator("#ql-threshold-en").check();
  await page.locator("#ql-threshold").fill("0.25");
  await openTab(page, "points");
  await openTab(page, "query");
  await expect(page.locator("#ql-threshold-en")).toBeChecked();
  await expect(page.locator("#ql-threshold")).toHaveValue("0.25");
  await expect(page.locator("#ql-threshold")).toBeEnabled();
});

// ════════════════════ NF-Z — theme conformance ════════════════════

test("NF-Z1/Z2/Z3/Z4 — new controls follow the theme patterns", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "query");

  // Z1: theme classes, no inline styles on the new blocks.
  for (const id of ["#ql-threshold-field", "#ql-offset-field", "#ql-order-field"]) {
    await expect(page.locator(id)).toHaveClass(/field/);
    expect(await page.locator(id).getAttribute("style")).toBeNull();
  }
  // order_by is a grouped control: fieldset + legend, like `fusion`.
  expect(await page.locator("#ql-order-field").evaluate((el) => el.tagName)).toBe("FIELDSET");
  await expect(page.locator("#ql-order-field > legend")).toHaveText("order_by");
  await expect(page.locator("#ql-order-field .radio-row")).toHaveCount(1);

  // Z2: enable-toggle — dependent controls start disabled.
  await expect(page.locator("#ql-threshold")).toBeDisabled();
  await expect(page.locator("#ql-offset")).toBeDisabled();
  await expect(page.locator("#ql-order-key")).toBeDisabled();

  // Z3: the Count subtab is wired like the other subtabs.
  await openTab(page, "points");
  const countTab = page.locator('[data-pt-tab="count"]');
  await expect(countTab).toHaveAttribute("role", "tab");
  await expect(page.locator("#pt-count")).toHaveAttribute("role", "tabpanel");
  await expect(page.locator("#pt-count")).toBeHidden();
  await countTab.click();
  await expect(page.locator("#pt-count")).toBeVisible();

  // Z4: live region on the new result area.
  await expect(page.locator("#cnt-result")).toHaveAttribute("aria-live", "polite");

  // Z5: the only destructive bulk action reuses the existing confirmation.
  await openPointsSubtab(page, "payload");
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await expect(page.locator("#pl-clear-submit")).toHaveClass(/btn-danger/);
  await expect(page.locator("#pl-clear-confirm")).toBeVisible();
  await expect(page.locator("#pl-clear-submit")).toBeDisabled();
});

test("NF-Z6 — the new controls take a visible focus ring", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "query");
  await page.locator("#ql-threshold-en").focus();
  const outline = await page.locator("#ql-threshold-en").evaluate((el) => {
    const s = getComputedStyle(el);
    return `${s.outlineStyle}|${s.outlineWidth}`;
  });
  expect(outline).not.toBe("none|0px");
});

// ════════════════════ NF-R — Parameter reference ════════════════════

// Every interactive control the Query panel exposes, mapped to the reference
// card that must explain it. Several controls share one card on purpose (the
// three enable-toggles of a parameter, the filter row cells, run/errors).
const QUERY_CONTROL_DOCS = {
  "ql-query": "query",
  "ql-collection": "collection_name",
  "ql-limit": "limit",
  "ql-cand": "candidate_limit",
  "ql-rrfk": "rrf_k",
  "ql-rrfk-en": "rrf_k",
  "ql-balance": "rrf_weights",
  "ql-weight-scale": "rrf_weights",
  "ql-weights-en": "rrf_weights",
  "ql-threshold": "score_threshold",
  "ql-threshold-en": "score_threshold",
  "ql-offset": "offset",
  "ql-offset-en": "offset",
  "ql-rerank": "rerank",
  "ql-topk": "rerank_top_k",
  "ql-topk-en": "rerank_top_k",
  "ql-normalize": "rerank_normalize",
  "ql-minshould": "min should matches",
  "ql-payload-fields": "with_payload",
  "ql-order-en": "order_by",
  "ql-order-key": "order_by",
  "ql-order-start": "start_from",
  "ql-request-json": "Request JSON",
  "ql-download": "Download results as JSON",
  "ql-results": "Results",
  "ql-embedding-guard": "Availability",
};

test("NF-R1 — the Parameter reference documents every control the Query panel exposes", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "query");
  await expect(page.locator("#query-reference .param-card").first()).toBeAttached();

  const cards = await page.locator("#query-reference .param-name").allTextContents();
  const names = cards.map((t) => t.trim());
  // 1. Every control has a card.
  for (const [control, card] of Object.entries(QUERY_CONTROL_DOCS)) {
    expect(names, `${control} has no reference card ("${card}")`).toContain(card);
  }
  // 2. No control present in the panel is missing from the map above — this is
  //    what catches a NEW control shipped without documentation.
  const controls = await page.evaluate(() => [...document.querySelectorAll(
    "#panel-query input, #panel-query select, #panel-query textarea")]
    .map((el) => el.id)
    .filter(Boolean));
  const map = QUERY_CONTROL_DOCS;
  const undocumented = controls.filter((id) =>
    !(id in map) && !/^ql-(filter|run|order-dir|minshould-field)/.test(id));
  expect(undocumented, `controls with no reference entry: ${undocumented.join(", ")}`).toEqual([]);
  // 3. Filter-row cells are covered by their four dedicated cards.
  for (const card of ["filters — GROUP", "filters — FIELD", "filters — OPERATION", "filters — VALUE"]) {
    expect(names).toContain(card);
  }
});

test("NF-R2 — reference entries keep one house style: 180-450 chars, prose, no stubs", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "query");
  const entries = await page.locator("#query-reference .param-card").evaluateAll((cards) =>
    cards.map((c) => ({
      name: c.querySelector(".param-name").textContent.trim(),
      len: c.querySelector(".param-desc").textContent.trim().length,
      text: c.querySelector(".param-desc").textContent.trim(),
    })));
  expect(entries.length).toBeGreaterThan(20);
  for (const e of entries) {
    expect(e.len, `"${e.name}" is ${e.len} chars — outside the house band`)
      .toBeGreaterThanOrEqual(180);
    expect(e.len, `"${e.name}" is ${e.len} chars — outside the house band`)
      .toBeLessThanOrEqual(450);
    // Prose, not a bullet dump, and a real sentence.
    expect(e.text, `"${e.name}" is not prose`).not.toMatch(/^[-•*]/);
    expect(e.text, `"${e.name}" does not end as a sentence`).toMatch(/[.)]$/);
  }
  // The spread is what makes the panel read evenly; keep it tight.
  const lens = entries.map((e) => e.len);
  expect(Math.max(...lens) / Math.min(...lens),
    `length spread is ${(Math.max(...lens) / Math.min(...lens)).toFixed(1)}x`).toBeLessThan(2.6);
});

// ════════════════════ NF-X — findings from the Query page audit ════════════════════

test("NF-X1 — order_by only offers fields Qdrant can actually sort (range index)", async ({ page }) => {
  await openQueryPayloadOnly(page);
  await page.locator("#ql-order-en").check();
  const keys = await page.locator("#ql-order-key option").allTextContents();
  // Sortable: integer, float, datetime. Indexed but NOT sortable: keyword, bool,
  // geo, uuid, text — Qdrant answers "No range index for `order_by` key" for
  // those, so offering them would only produce a raw 400.
  expect(keys.sort()).toEqual(["f_created", "f_price", "f_views"]);
  for (const notSortable of ["f_status", "f_flag", "f_geo", "f_uuid", "f_tags", "f_title"]) {
    expect(keys, `${notSortable} has no range index and must not be offered`)
      .not.toContain(notSortable);
  }
});

test("NF-X2 — a collection with no range-indexed field disarms order_by", async ({ page }) => {
  // e2e_d_nophrase carries keyword + text indexes only.
  await boot(page, { active: fixtureName("e2e_d_nophrase") });
  await selectActive(page, fixtureName("e2e_d_nophrase"));
  await openTab(page, "query");
  await page.locator("#ql-query").fill("");
  await expect(page.locator("#ql-order-key")).toContainText("no orderable field");
  await expect(page.locator("#ql-order-en")).not.toBeChecked();
  await expect(page.locator("#ql-request-json")).not.toContainText('"order_by"');
});

test("NF-X3 — start_from is validated against the ordered field's type", async ({ page }) => {
  await openQueryPayloadOnly(page);
  await page.locator("#ql-order-en").check();

  await page.locator("#ql-order-key").selectOption("f_views");
  await page.locator("#ql-order-start").fill("not-a-number");
  await expect(page.locator("#ql-errors")).toContainText("start_from must be an integer for 'f_views'");
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-order-key").selectOption("f_created");
  await page.locator("#ql-order-start").fill("lunedi");
  await expect(page.locator("#ql-errors")).toContainText("start_from must be a date for 'f_created'");
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-order-start").fill("2026-06-05T00:00:00Z");
  await expect(page.locator("#ql-errors")).not.toContainText("start_from");
  await expect(page.locator("#ql-run")).toBeEnabled();
});

test("NF-X4 — [API] a mistyped start_from is a 422, never a 500", async ({ request }) => {
  const res = await api(request, "POST", "/points/scroll", {
    collection_name: QUERY_COLL, limit: 5,
    order_by: { key: "f_views", start_from: "not-a-number" },
  });
  expect(res.status(), "bad input must not surface as a server error").toBe(422);
  expect(JSON.stringify(await res.json())).toMatch(/start_from must be a number or a datetime/);
});

test("NF-X5 — offset is sent whenever its toggle is on, including 0", async ({ page }) => {
  await openQueryHybrid(page);
  await page.locator("#ql-offset-en").check();
  await page.locator("#ql-offset").fill("0");
  // Skipping offset:0 would make the preview disagree with the checkbox, while
  // score_threshold already sends its 0.
  await expect(page.locator("#ql-request-json")).toContainText('"offset": 0');
  await page.locator("#ql-threshold-en").check();
  await page.locator("#ql-threshold").fill("0");
  await expect(page.locator("#ql-request-json")).toContainText('"score_threshold": 0');
});

test("NF-X6 — an offset deeper than candidate_limit is refused with the number to raise", async ({ page }) => {
  await openQueryHybrid(page);
  await page.locator("#ql-offset-en").check();
  await page.locator("#ql-offset").fill("9999");
  // Previously this returned an empty page indistinguishable from "no matches";
  // fusion cannot rank a point neither branch retrieved, so the page is refused
  // and the message names the field to change.
  await expect(page.locator("#ql-errors")).toContainText("needs candidate_limit ≥");
  await expect(page.locator("#ql-errors")).toContainText("Raise candidate_limit");
  await expect(page.locator("#ql-run")).toBeDisabled();

  // Within candidate_limit it runs normally again.
  await page.locator("#ql-offset").fill("2");
  await expect(page.locator("#ql-errors")).not.toContainText("candidate_limit ≥");
  await expect(page.locator("#ql-run")).toBeEnabled();
});

test("NF-X8 — the offset field's ceiling tracks candidate_limit", async ({ page }) => {
  await openQueryHybrid(page);   // rerank off, limit 10 by default
  await page.locator("#ql-offset-en").check();
  const ceiling = async () => Number(await page.locator("#ql-offset").getAttribute("max"));
  const limit = Number(await page.locator("#ql-limit").inputValue());
  await page.locator("#ql-cand").fill("50");
  expect(await ceiling()).toBe(50 - limit);
  await page.locator("#ql-cand").fill("200");
  expect(await ceiling()).toBe(200 - limit);
});

test("NF-X9 — [API] consecutive pages tile into one deep ranking", async ({ request }) => {
  await reseedDeepRanking(request);
  const body = (extra) => ({
    collection_name: DEEP_COLL, query: "hybrid retrieval and ranking",
    candidate_limit: 200, rerank: false, ...extra,
  });
  const ids = async (extra) => {
    const res = await api(request, "POST", "/query/search", body(extra));
    expect(res.status()).toBe(200);
    return requireResults(await res.json()).map((h) => String(h.id));
  };
  const paged = [];
  for (const offset of [0, 5, 10, 15]) paged.push(...await ids({ limit: 5, offset }));
  const single = await ids({ limit: 20 });

  expect(paged).toHaveLength(20);
  // Equal-score RRF hits receive the same deterministic id tie-break in every
  // request, so adjacent pages must exactly tile the single deep ranking.
  expect(paged).toEqual(single);
});

test("NF-X10 — [API] omitting with_payload returns the whole payload", async ({ request }) => {
  const res = await api(request, "POST", "/query/search",
    { collection_name: QUERY_COLL, query: "hybrid vector search", limit: 1, rerank: false });
  expect(res.status()).toBe(200);
  const payload = (await res.json()).results[0].payload;
  // Used to be a hardcoded six-key shortlist, which silently dropped everything else.
  for (const key of ["text", "f_status", "f_views", "f_price", "f_flag", "f_geo",
    "f_created", "f_uuid", "f_extra", "f_tags", "f_title"]) {
    expect(Object.keys(payload), `${key} missing from the default projection`).toContain(key);
  }
});

test("NF-X11 — [API] the response echoes offset and score_threshold", async ({ request }) => {
  const res = await api(request, "POST", "/query/search", {
    collection_name: QUERY_COLL, query: "hybrid vector search",
    limit: 3, rerank: false, offset: 2, score_threshold: 0.1,
  });
  const body = await res.json();
  expect(body.offset).toBe(2);
  expect(body.score_threshold).toBeCloseTo(0.1);
  // Absent by default, so a caller can tell "not applied" from "applied as 0".
  const plain = await (await api(request, "POST", "/query/search",
    { collection_name: QUERY_COLL, query: "x", limit: 1, rerank: false })).json();
  expect(plain.offset).toBe(0);
  expect(plain.score_threshold).toBeNull();
});

test("NF-X12 — [API] prefetch search params are settable per request and bounded", async ({ request }) => {
  const call = (extra) => api(request, "POST", "/query/search",
    { collection_name: QUERY_COLL, query: "hybrid", limit: 2, rerank: false, ...extra });
  for (const ok of [{ hnsw_ef: 256 }, { exact: true }, { rescore: false }, { oversampling: 4.0 }]) {
    expect((await call(ok)).status(), JSON.stringify(ok)).toBe(200);
  }
  for (const bad of [{ hnsw_ef: 0 }, { oversampling: 0.5 }, { timeout: 0 }]) {
    expect((await call(bad)).status(), JSON.stringify(bad)).toBe(400);
  }
  // A per-request timeout is accepted and bounds the Qdrant call.
  expect((await call({ timeout: 30 })).status()).toBe(200);
});

test("NF-X7 — the download matches what the mode can page", async ({ page }) => {
  const readDownload = async () => {
    const wait = page.waitForEvent("download", { timeout: 60_000 });
    await page.locator("#ql-run").click();
    const stream = await (await wait).createReadStream();
    let raw = "";
    for await (const chunk of stream) raw += chunk;
    return JSON.parse(raw);
  };

  // Unordered payload-only: every page is walked, so all 12 points land in the file
  // even though the screen shows 5.
  await openQueryPayloadOnly(page);
  await page.locator("#ql-download").check();
  const all = await readDownload();
  expect(all.chunks.length).toBe(12);
  expect(all.metadata.mode).toBe("payload-only");

  // Ordered: Qdrant serves one page (no offset paging with order_by), so the file
  // holds exactly the page on screen — as the reference states.
  await openQueryPayloadOnly(page);
  await page.locator("#ql-download").check();
  await page.locator("#ql-order-en").check();
  await page.locator("#ql-order-key").selectOption("f_views");
  await page.locator('input[name="ql-order-dir"][value="desc"]').check();
  const ordered = await readDownload();
  expect(ordered.chunks.length).toBe(5);
  // The stored point id rides under `point_id`; `id` belongs to the payload.
  expect(ordered.chunks.map((c) => String(c.point_id))).toEqual(["12", "11", "10", "9", "8"]);
});

// ════════════════════ NF-R3/R4 — Collections Feature reference ════════════════════

// Every control the Collections panel exposes, mapped to the card that documents it.
// Groups of controls legitimately share one card (the five toggles appear in both 03
// and 04; the three type-to-confirm fields share one entry).
const COLLECTIONS_CONTROL_DOCS = {
  "cc-report": "01 Report",
  "cr-preflight": "02 Re-embed Vectors",
  "cr-pause": "pause / resume / cancel",
  "cr-resume": "pause / resume / cancel",
  "cr-cancel": "pause / resume / cancel",
  "cr-start": "02 Re-embed Vectors",
  "cr-confirm": "type-to-confirm",
  "cc-name": "collection_name",
  "cc-dense": "dense_on_disk",
  "cc-sparse-disk": "sparse_on_disk",
  "cc-payload-disk": "payload_on_disk",
  "cc-quant": "quantization",
  "cc-idf": "sparse_idf",
  "cc-recreate": "recreate",
  "cc-confirm": "type-to-confirm",
  "cc-submit": "03 Create / Recreate",
  "cu-dense": "dense_on_disk",
  "cu-sparse-disk": "sparse_on_disk",
  "cu-payload-disk": "payload_on_disk",
  "cu-quant": "quantization",
  "cu-idf": "sparse_idf",
  "cu-submit": "04 Schema Toggles",
  "pi-table": "05 Payload Indexes",
  "pi-field": "payload field / field schema",
  "pi-schema": "payload field / field schema",
  "pi-phrase": "phrase matching",
  "pi-create": "05 Payload Indexes",
  "pi-del-field": "payload index field to delete",
  "pi-delete": "payload index field to delete",
  "ce-export": "06 Export Collection",
  "cd-name": "07 Delete",
  "cd-confirm": "type-to-confirm",
  "cd-submit": "07 Delete",
};

test("NF-R3 — the Feature reference documents every control the Collections panel exposes", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "collections");
  await expect(page.locator("#collections-reference .param-card").first()).toBeAttached();

  const names = (await page.locator("#collections-reference .param-name").allTextContents())
    .map((t) => t.trim());
  for (const [control, card] of Object.entries(COLLECTIONS_CONTROL_DOCS)) {
    expect(names, `${control} has no reference card ("${card}")`).toContain(card);
  }
  // Section numbers must track the panel: Export took 06 and Delete moved to 07, so a
  // card still called "06 Delete" would point at the wrong section.
  const labels = await page.locator("#panel-collections .section").evaluateAll((hs) =>
    hs.map((h) => h.textContent.replace(/\s+/g, " ").trim()));
  expect(labels.some((l) => l.includes("06") && /Export/i.test(l))).toBe(true);
  expect(labels.some((l) => l.includes("07") && /Delete/i.test(l))).toBe(true);
  expect(names).toContain("06 Export Collection");
  expect(names).toContain("07 Delete");
  expect(names, "stale numbering: Delete is section 07").not.toContain("06 Delete");

  // A new control shipped without documentation fails here.
  const controls = await page.evaluate(() => [...document.querySelectorAll(
    "#panel-collections input, #panel-collections select, #panel-collections textarea")]
    .map((el) => el.id)
    .filter(Boolean));
  const map = COLLECTIONS_CONTROL_DOCS;
  const undocumented = controls.filter((id) => !(id in map));
  expect(undocumented, `controls with no reference entry: ${undocumented.join(", ")}`).toEqual([]);
});

test("NF-R4 — Collections reference entries keep the same house style as Query", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "collections");
  const entries = await page.locator("#collections-reference .param-card").evaluateAll((cards) =>
    cards.map((c) => ({
      name: c.querySelector(".param-name").textContent.trim(),
      text: c.querySelector(".param-desc").textContent.trim(),
    })));
  expect(entries.length).toBeGreaterThan(18);
  for (const e of entries) {
    expect(e.text.length, `"${e.name}" is ${e.text.length} chars — outside the house band`)
      .toBeGreaterThanOrEqual(180);
    expect(e.text.length, `"${e.name}" is ${e.text.length} chars — outside the house band`)
      .toBeLessThanOrEqual(450);
    expect(e.text, `"${e.name}" is not prose`).not.toMatch(/^[-•*]/);
    expect(e.text, `"${e.name}" does not end as a sentence`).toMatch(/[.)]$/);
  }
  const lens = entries.map((e) => e.text.length);
  expect(Math.max(...lens) / Math.min(...lens),
    `length spread is ${(Math.max(...lens) / Math.min(...lens)).toFixed(1)}x`).toBeLessThan(2.6);
});

// ════════════════════ NF-P — facet, update_mode, with_vectors ════════════════════
// docs/audit-points-coverage-2026-07.md recommendations 1-3.

test("NF-P1 — the facet key select lists only keyword/integer/bool/uuid fields", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await waitForFacetKeys(page);
  const options = await page.locator("#cnt-facet-key option").allTextContents();
  // e2e_d_query indexes f_status(keyword) f_views(integer) f_flag(bool) f_uuid(uuid)
  // facetable, and f_title(text) f_geo(geo) f_price(float) f_created(datetime) not.
  expect(options).toEqual(expect.arrayContaining(["f_status", "f_views", "f_flag", "f_uuid"]));
  expect(options).not.toEqual(expect.arrayContaining(["f_title", "f_geo", "f_price", "f_created"]));
});

test("NF-P2 — a collection with no facetable field disables the control", async ({ page, request }) => {
  const empty = fixtureName("e2e_nf_bare");
  await api(request, "POST", "/collections", {
    collection_name: empty, recreate: true, confirm_recreate: empty,
    dense_on_disk: true, sparse_on_disk: true, on_disk_payload: true,
    quantization: true, sparse_idf: true,
  });
  await boot(page, { active: empty });
  await selectActive(page, empty);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await expect(page.locator("#cnt-facet-key")).toBeDisabled();
  await expect(page.locator("#cnt-facet-key")).toContainText("no facetable field");
  await expect(page.locator("#cnt-facet-submit")).toBeDisabled();
});

test("NF-P3 — facet on a keyword field returns every distinct value and its count", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await waitForFacetKeys(page);
  await page.locator("#cnt-facet-key").selectOption("f_status");
  await page.locator("#cnt-facet-submit").click();
  await expect(page.locator("#cnt-facet-result table")).toBeVisible();
  const rows = await page.locator("#cnt-facet-result tbody tr").count();
  // 12 points, distinct f_status per point (s1..s12) in the read fixture.
  expect(rows).toBe(12);
});

test("NF-P4 — exact deselected still returns a valid facet", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await waitForFacetKeys(page);
  await page.locator("#cnt-facet-key").selectOption("f_status");
  await page.locator("#cnt-facet-exact").uncheck();
  await page.locator("#cnt-facet-submit").click();
  await expect(page.locator("#cnt-facet-result table")).toBeVisible();
  await expect(page.locator("#cnt-facet-result tbody tr").first()).toBeVisible();
});

test("NF-P5 — [API] facet on an unindexed field is a 400 naming the field, never a raw 502", async ({ request }) => {
  const res = await api(request, "POST", "/points/facet",
    { collection_name: QUERY_COLL, key: "no_such_field", limit: 10 });
  expect(res.status()).toBe(400);
  const body = await res.json();
  expect(body.detail).toContain("no_such_field");
  expect(body.detail).not.toContain("Raw response");
});

test("NF-P6 — [API] a blank facet key is a 400 validation error", async ({ request }) => {
  const res = await api(request, "POST", "/points/facet",
    { collection_name: QUERY_COLL, key: "   ", limit: 10 });
  expect(res.status()).toBe(400);
});

test("NF-P7 — [API] limit outside 1-1000 is a 400 validation error", async ({ request }) => {
  const tooLow = await api(request, "POST", "/points/facet",
    { collection_name: QUERY_COLL, key: "f_status", limit: 0 });
  expect(tooLow.status()).toBe(400);
  const tooHigh = await api(request, "POST", "/points/facet",
    { collection_name: QUERY_COLL, key: "f_status", limit: 1001 });
  expect(tooHigh.status()).toBe(400);
});

test("NF-P8 — switching the active collection clears the facet result and recomputes the key select", async ({ page, request }) => {
  const bare = fixtureName("e2e_nf_bare");
  await api(request, "POST", "/collections", {
    collection_name: bare, recreate: true, confirm_recreate: bare,
    dense_on_disk: true, sparse_on_disk: true, on_disk_payload: true,
    quantization: true, sparse_idf: true,
  });
  await api(request, "POST", "/points/batch", {
    collection_name: bare, points: [{ point_id: "1", text: "bare point", payload: { plain: "x" } }],
  });
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "count");
  await waitForFacetKeys(page);
  await page.locator("#cnt-facet-key").selectOption("f_status");
  await page.locator("#cnt-facet-submit").click();
  await expect(page.locator("#cnt-facet-result table")).toBeVisible();

  await selectActive(page, bare);
  await expect(page.locator("#cnt-facet-result")).toBeEmpty();
  await expect(page.locator("#cnt-facet-key")).toBeDisabled();
});

test("NF-P9 — Upsert: update_mode defaults to UPSERT and overwrites as before", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "upsert");
  await expect(page.locator('input[name="up-mode"][value="upsert"]')).toBeChecked();
  await page.locator("#up-id").fill("1");
  await page.locator("#up-text").fill("overwritten by NF-P9");
  await page.locator("#up-submit").click();
  await expect(page.locator("#up-result .alert-success")).toBeVisible();
  const payloads = await payloadsOf(request, BULK_COLL);
  const p1 = payloads.find((p) => p.id === "1");
  expect(p1.payload.text).toBe("overwritten by NF-P9");
});

test("NF-P10 — Upsert: INSERT ONLY skips an existing id and leaves it untouched", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "upsert");
  await page.locator("#up-id").fill("2");
  await page.locator("#up-text").fill("must not land");
  await page.locator('input[name="up-mode"][value="insert_only"]').check();
  await page.locator("#up-submit").click();
  await expect(page.locator("#up-result .alert-warning")).toContainText(/already exists/i);
  await expect(page.locator("#up-result .alert-warning")).toContainText(/insert_only/i);
  const payloads = await payloadsOf(request, BULK_COLL);
  const p2 = payloads.find((p) => p.id === "2");
  expect(p2.payload.text).not.toBe("must not land");
});

test("NF-P11 — Upsert: UPDATE ONLY skips a new id instead of creating it", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "upsert");
  await page.locator("#up-id").fill("999");
  await page.locator("#up-text").fill("ghost point");
  await page.locator('input[name="up-mode"][value="update_only"]').check();
  await page.locator("#up-submit").click();
  await expect(page.locator("#up-result .alert-warning")).toContainText(/does not exist/i);
  await expect(page.locator("#up-result .alert-warning")).toContainText(/update_only/i);
  const res = await api(request, "GET", `/points/999?collection_name=${BULK_COLL}`);
  expect(res.status()).toBe(404);
});

test("NF-P12 — Upsert: UPDATE ONLY on an existing id writes normally", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "upsert");
  await page.locator("#up-id").fill("4");
  await page.locator("#up-text").fill("updated in place by NF-P12");
  await page.locator('input[name="up-mode"][value="update_only"]').check();
  await page.locator("#up-submit").click();
  await expect(page.locator("#up-result .alert-success")).toBeVisible();
  const payloads = await payloadsOf(request, BULK_COLL);
  expect(payloads.find((p) => p.id === "4").payload.text).toBe("updated in place by NF-P12");
});

test("NF-P13 — Batch: update_mode reports upserted and skipped separately", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "batch");
  await page.locator('input[name="bt-mode"][value="insert_only"]').check();
  await page.locator("#bt-json").fill(JSON.stringify([
    { point_id: "3", text: "existing, must be skipped", payload: {} },
    { point_id: "50", text: "new point", payload: {} },
  ]));
  await page.locator("#bt-submit").click();
  await expect(page.locator("#bt-result .alert-success")).toContainText("1 point(s) upserted");
  await expect(page.locator("#bt-result .alert-success")).toContainText(/1 skipped/i);
  await expect(page.locator("#bt-result .alert-success")).toContainText(/insert_only/i);
  const payloads = await payloadsOf(request, BULK_COLL);
  expect(payloads.find((p) => p.id === "3").payload.text).not.toBe("existing, must be skipped");
  expect(payloads.find((p) => p.id === "50")).toBeTruthy();
});

test("NF-P14 — Batch: entirely skipped reports upserted:0 without failing", async ({ page, request }) => {
  await reseedBulk(request);
  await boot(page, { active: BULK_COLL });
  await selectActive(page, BULK_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "batch");
  await page.locator('input[name="bt-mode"][value="insert_only"]').check();
  await page.locator("#bt-json").fill(JSON.stringify([
    { point_id: "5", text: "already there", payload: {} },
    { point_id: "6", text: "also already there", payload: {} },
  ]));
  await page.locator("#bt-submit").click();
  await expect(page.locator("#bt-result .alert-success")).toContainText("0 point(s) upserted");
  await expect(page.locator("#bt-result .alert-success")).toContainText(/2 skipped/i);
});

test("NF-P15 — Get: with_vectors adds a name/shape summary, never raw numbers", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "get");
  await page.locator("#get-id").fill("1");
  await page.locator("#get-submit").click();
  await expect(page.locator("#get-result")).not.toBeEmpty();
  await expect(page.locator("#get-result")).not.toContainText("vectors:");

  await page.locator("#get-with-vectors").check();
  await page.locator("#get-submit").click();
  // The handler updates the DOM asynchronously after the click resolves;
  // wait for the new content instead of racing it with a bare innerText read.
  await expect(page.locator("#get-result")).toContainText("vectors:");
  const text = await page.locator("#get-result").innerText();
  expect(text).toMatch(/vectors:.*dense.*\d+-d/i);
  expect(text).toMatch(/sparse.*\d+ nonzero/i);
  // Never the raw vector array: no long run of comma-separated decimals.
  expect(text).not.toMatch(/-?0\.\d+,\s*-?0\.\d+,\s*-?0\.\d+/);
});

test("NF-P16 — Get many: with_vectors summarizes every returned point", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await selectActive(page, QUERY_COLL);
  await openTab(page, "points");
  await openPointsSubtab(page, "get");
  await page.locator("#get-ids").fill("1, 2, 3");
  await page.locator("#get-many-with-vectors").check();
  await page.locator("#get-many-submit").click();
  await expect(page.locator("#get-many-result")).toContainText("3 of 3 id(s) found");
  const text = await page.locator("#get-many-result").innerText();
  const vectorLines = (text.match(/vectors:/g) || []).length;
  expect(vectorLines).toBe(3);
});

test("NF-P18 — theme conformance: new controls use the shared patterns", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "points");
  await openPointsSubtab(page, "upsert");
  // update_mode is a grouped control: fieldset + legend + radio-row, like order_by.
  expect(await page.locator("#up-mode-field").evaluate((el) => el.tagName)).toBe("FIELDSET");
  await expect(page.locator("#up-mode-field > legend")).toHaveText("update_mode");
  await expect(page.locator("#up-mode-field .radio-row")).toHaveCount(1);
  expect(await page.locator("#up-mode-field").getAttribute("style")).toBeNull();
  await openPointsSubtab(page, "batch");
  expect(await page.locator("#bt-mode-field").evaluate((el) => el.tagName)).toBe("FIELDSET");
  await expect(page.locator("#bt-mode-field .radio-row")).toHaveCount(1);
  await openPointsSubtab(page, "get");
  // with_vectors is a plain toggle: label.check, same pattern as exact on Count.
  const parentClass = await page.locator("#get-with-vectors").evaluate((el) => el.closest("label")?.className);
  expect(parentClass).toContain("check");
  await openPointsSubtab(page, "count");
  await expect(page.locator("#cnt-facet-result")).toHaveAttribute("aria-live", "polite");
});

// ════════════════════ NF-R5/R6 — Points Feature reference ════════════════════
// Same two-part check as NF-R3/R4 for Collections: every control has a card
// (including the three added by NF-P), and every card keeps house style.

const POINTS_CONTROL_DOCS = {
  "ing-file": "chunks JSON file",
  "ing-create-indexes": "Create payload indexes for typed fields after ingest",
  "up-id": "point_id",
  "up-text": "text",
  "up-payload": "payload JSON",
  "bt-json": "Batch",
  "get-id": "Get",
  "get-with-vectors": "with_vectors",
  "get-ids": "Get",
  "get-many-with-vectors": "with_vectors",
  "del-id": "Delete",
  "del-ids": "ids",
  "del-confirm": "Delete",
  "delf-json": "Delete by filter",
  "delf-confirm": "Delete by filter",
  "pl-id": "point_id",
  "pl-filter-json": "scope: POINT ID | FILTER",
  "pl-json": "Payload",
  "pl-keys": "Delete keys",
  "pl-clear-confirm": "Clear payload",
  "cnt-json": "Count",
  "cnt-exact": "Count",
  "cnt-facet-key": "Facet",
  "cnt-facet-limit": "Facet",
  "cnt-facet-exact": "Facet",
};

test("NF-R5 — the Feature reference documents every control the Points panel exposes", {
  annotation: { type: "case", description: "book:13.8/NF-P19" },
}, async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "points");
  await expect(page.locator("#points-reference .param-card").first()).toBeAttached();

  const names = (await page.locator("#points-reference .param-name").allTextContents())
    .map((t) => t.trim());
  for (const [control, card] of Object.entries(POINTS_CONTROL_DOCS)) {
    expect(names, `${control} has no reference card ("${card}")`).toContain(card);
  }
  expect(names).toContain("update_mode");
  expect(names).toContain("with_vectors");
  expect(names).toContain("Facet");

  // A new control shipped without documentation fails here.
  const controls = await page.evaluate(() => [...document.querySelectorAll(
    "#panel-points input, #panel-points select, #panel-points textarea")]
    .map((el) => el.id)
    .filter(Boolean));
  const undocumented = controls.filter((id) => !(id in POINTS_CONTROL_DOCS));
  expect(undocumented, `controls with no reference entry: ${undocumented.join(", ")}`).toEqual([]);
});

test("NF-R6 — Points reference entries keep the same house style as Query and Collections", {
  annotation: { type: "case", description: "book:13.8/NF-P19" },
}, async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "points");
  const entries = await page.locator("#points-reference .param-card").evaluateAll((cards) =>
    cards.map((c) => ({
      name: c.querySelector(".param-name").textContent.trim(),
      text: c.querySelector(".param-desc").textContent.trim(),
    })));
  expect(entries.length).toBeGreaterThan(18);
  for (const e of entries) {
    expect(e.text.length, `"${e.name}" is ${e.text.length} chars — outside the house band`)
      .toBeGreaterThanOrEqual(180);
    expect(e.text.length, `"${e.name}" is ${e.text.length} chars — outside the house band`)
      .toBeLessThanOrEqual(450);
    expect(e.text, `"${e.name}" is not prose`).not.toMatch(/^[-•*]/);
    expect(e.text, `"${e.name}" does not end as a sentence`).toMatch(/[.)]$/);
  }
  const lens = entries.map((e) => e.text.length);
  expect(Math.max(...lens) / Math.min(...lens),
    `length spread is ${(Math.max(...lens) / Math.min(...lens)).toFixed(1)}x`).toBeLessThan(2.6);
});

// ════════════════════ NF-R7/R8 — Monitoring Feature reference ════════════════════
// Monitoring has one real control (Reset counters) and otherwise renders every
// label at runtime rather than as static form fields, so coverage is checked
// against the rendered badge/tile text instead of a static-HTML id map.

const MONITORING_RENDERED_LABELS = [
  "livez", "readyz", "qdrant", "embedder", "health",
  "uptime", "total_requests", "rejected_requests",
];

test("NF-R7 — the Feature reference documents every badge, tile, and control the Monitoring panel renders", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "monitoring");
  await expect(page.locator("#mon-badges")).not.toBeEmpty({ timeout: 30_000 });
  await expect(page.locator("#mon-stats")).not.toBeEmpty({ timeout: 30_000 });
  await expect(page.locator("#monitoring-reference .param-card").first()).toBeAttached();

  const names = (await page.locator("#monitoring-reference .param-name").allTextContents())
    .map((t) => t.trim());
  for (const label of MONITORING_RENDERED_LABELS) {
    expect(names, `"${label}" is rendered but has no reference card`).toContain(label);
  }
  expect(names).toContain("Embedder Models");
  expect(names).toContain("Reset counters");
  expect(names).toContain("Per-endpoint metrics");
  expect(names).toContain("requests chart");
  expect(names).toContain("Auto-refresh");

  // mon-reset is the one static control on this tab; it must have its own card.
  const controls = await page.evaluate(() => [...document.querySelectorAll(
    "#panel-monitoring input, #panel-monitoring select, #panel-monitoring textarea, #panel-monitoring button")]
    .map((el) => el.id)
    .filter(Boolean));
  expect(controls).toContain("mon-reset");
});

test("NF-R8 — Monitoring reference entries keep the same house style as Query, Collections, and Points", async ({ page }) => {
  await boot(page, { active: QUERY_COLL });
  await openTab(page, "monitoring");
  const entries = await page.locator("#monitoring-reference .param-card").evaluateAll((cards) =>
    cards.map((c) => ({
      name: c.querySelector(".param-name").textContent.trim(),
      text: c.querySelector(".param-desc").textContent.trim(),
    })));
  expect(entries.length).toBeGreaterThan(11);
  for (const e of entries) {
    expect(e.text.length, `"${e.name}" is ${e.text.length} chars — outside the house band`)
      .toBeGreaterThanOrEqual(180);
    expect(e.text.length, `"${e.name}" is ${e.text.length} chars — outside the house band`)
      .toBeLessThanOrEqual(450);
    expect(e.text, `"${e.name}" is not prose`).not.toMatch(/^[-•*]/);
    expect(e.text, `"${e.name}" does not end as a sentence`).toMatch(/[.)]$/);
  }
  const lens = entries.map((e) => e.text.length);
  expect(Math.max(...lens) / Math.min(...lens),
    `length spread is ${(Math.max(...lens) / Math.min(...lens)).toFixed(1)}x`).toBeLessThan(2.6);
});

// ═══════════════════ NF-COV1 — executable UI coverage contract ════════════════

test("NF-COV1 — every interactive console control has a static spec reference", async () => {
  const html = readFileSync(path.join(FRONTEND_DIR, "dist", "index.html"), "utf8");
  const specSource = readdirSync(TESTS_DIR)
    .filter((name) => name.endsWith(".spec.js"))
    .map((name) => readFileSync(path.join(TESTS_DIR, name), "utf8"))
    .join("\n");

  const tags = html.match(/<(?:button|input|select|textarea)\b[^>]*>/gi) || [];
  const contracts = new Set();
  for (const tag of tags) {
    for (const attribute of ["id", "name", "data-pt-tab"]) {
      const match = tag.match(new RegExp(`\\b${attribute}="([^"]+)"`, "i"));
      if (match) contracts.add(match[1]);
    }
  }

  const uncovered = [...contracts]
    .filter((contract) => !specSource.includes(contract))
    .sort();
  expect(
    uncovered,
    `interactive controls without an E2E reference: ${uncovered.join(", ")}`,
  ).toEqual([]);
});

// The reporter uses actual Playwright collection and results; no source regex.
test("NF-COV2 — qualified execution ledger retains every book case", async () => {
  const { bookCases } = await import('./case-coverage.mjs');
  const cases = bookCases(readFileSync(path.join(TESTS_DIR, 'testbook/test-book-e2e.md'), 'utf8'));
  const ledger = JSON.parse(readFileSync(path.join(FRONTEND_DIR, 'test-results/campaign.json'), 'utf8'));
  expect(ledger.cases.map((item) => item.key)).toEqual(cases.map((item) => item.key));
  expect(new Set(ledger.cases.map((item) => item.key)).size).toBe(cases.length);
  expect(cases.filter((item) => item.id.startsWith('TB-'))).toHaveLength(96);
  for (const item of ledger.cases.filter((item) => item.status === 'PASS')) {
    expect(item.variants.length).toBeGreaterThan(0);
    for (const variant of item.variants) {
      expect(ledger.results.some((result) => result.id === variant.id && result.status === 'passed')).toBe(true);
    }
  }
});
