// Test book §8.2 Series A/B and §8.3 TB-FIL-002..012 on the 100-point
// `e2e_book_filters` fixture and the 6-point `e2e_book_edges` fixture (§0.6).
// UI-live: every row is built through the console, every page of the
// payload-only scroll is walked, offsets/ids are rejected when repeated and
// the exact ID set is compared with the generator formulas — never with the
// API. Seed with `npm run seed:book` (owned, QHM_E2E_RUN_ID) before running.
// Validation rows (C1–C21, TB-FIL-001/013–016) live in tb-filter-validation.spec.js.
import { expect, test } from "./fixtures.js";
import { fixtureName } from "./harness.mjs";
import { boot } from "./helpers.js";
import { clearFilters, condition, requestJson, rows, setRow, waitFieldsLoaded } from "./d-helpers.js";
import { day, uuid } from "./tools/seed-book.mjs";
import { bootPaced, reserve } from "./tb-live-helpers.js";

const FILTERS = fixtureName("e2e_book_filters");
const EDGES = fixtureName("e2e_book_edges");
const range = (a, b) => Array.from({ length: Math.max(0, b - a + 1) }, (_, k) => String(a + k));
const all = range(1, 100);
const sorted = (ids) => [...ids].map(String).sort((x, y) => Number(x) - Number(y));

async function open(page, collection = FILTERS) {
  const errors = await bootPaced(page, boot, { active: collection });
  await waitFieldsLoaded(page);
  await clearFilters(page);
  await page.locator("#ql-query").fill("");
  return errors;
}

// Walk every payload-only page; reject repeated offsets and ids; return the ids
// plus the raw requests/responses for wire-shape assertions.
async function collect(page, info) {
  const ids = [], evidence = [], offsets = new Set();
  const step = async (control) => {
    await reserve(1);
    const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/scroll" && r.request().method() === "POST");
    await control.click();
    const response = await pending;
    const body = await response.json();
    expect(response.status(), JSON.stringify(body)).toBe(200);
    const pageIds = body.results.map((point) => String(point.id));
    evidence.push({ request: response.request().postDataJSON(), ids: pageIds, payloads: body.results.map((point) => point.payload),
      next_offset: body.next_page_offset, total: body.total });
    await expect.poll(async () => (await page.locator("#ql-results .result-id").allTextContents()).map((t) => t.trim())).toEqual(pageIds);
    ids.push(...pageIds);
    if (body.next_page_offset != null) {
      expect(offsets.has(String(body.next_page_offset)), "repeated scroll offset").toBe(false);
      offsets.add(String(body.next_page_offset));
    }
  };
  await expect(page.locator("#ql-run")).toBeEnabled();
  await step(page.locator("#ql-run"));
  for (let guard = 0; guard < 30; guard++) {
    const next = page.locator('#ql-results [data-query-page="next"]');
    if (!(await next.count())) break;
    await step(next);
  }
  expect(new Set(ids).size, "duplicate ids across pages").toBe(ids.length);
  await info.attach("requests-and-exact-ids", { body: JSON.stringify(evidence, null, 2), contentType: "application/json" });
  return { ids: sorted(ids), evidence };
}

// One book row: build the listed conditions and compare the exact ID set.
function row(id, label, specs, expected, { collection = FILTERS, after = null, annotation } = {}) {
  const options = annotation ? { annotation } : {};
  test(`${id} — ${label}`, options, async ({ page }, info) => {
    const errors = await open(page, collection);
    for (const spec of Array.isArray(specs) ? specs : [specs]) await condition(page, spec);
    const result = await collect(page, info);
    expect(result.ids).toEqual(sorted(expected));
    expect(result.evidence[0].total).toBe(expected.length);
    if (after) await after(page, result);
    expect(errors).toEqual([]);
  });
}

// ── Series A — operator × type ───────────────────────────────────────────────

row("A1", "status equals active → 20 over 4 pages", { field: "status", op: "eq", value: "active" }, range(1, 20),
  { after: async (page, { evidence }) => expect(evidence).toHaveLength(4) });
row("A2", "status any of active, deleted → 30", { field: "status", op: "any", value: "active, deleted" }, [...range(1, 20), ...range(91, 100)]);
row("A3", "status except active, archived, draft → 10", { field: "status", op: "except", value: "active, archived, draft" }, range(91, 100));
row("A4", "views equals 50 → 1 page ending immediately", { field: "views", op: "eq", value: "50" }, ["50"],
  { after: async (page, { evidence }) => { expect(evidence).toHaveLength(1); await expect(page.locator("#ql-results")).toContainText(/End of results/i); } });
row("A5", "views any of 1, 2, 3 → 3", { field: "views", op: "any", value: "1, 2, 3" }, range(1, 3));
row("A6", "views except 1, 2, 3 → 97", { field: "views", op: "except", value: "1, 2, 3" }, range(4, 100));
row("A7", "views greater than 90 → 10", { field: "views", op: "gt", value: "90" }, range(91, 100));
row("A8", "views at least 90 → 11", { field: "views", op: "gte", value: "90" }, range(90, 100));
row("A9", "views less than 11 → 10", { field: "views", op: "lt", value: "11" }, range(1, 10));
row("A10", "views at most 10 → 10", { field: "views", op: "lte", value: "10" }, range(1, 10));
row("A11", "views between 10 and 19 → 10", { field: "views", op: "between", value: "10", value2: "19" }, range(10, 19));
row("A12", "price greater than 50.5 → 50", { field: "price", op: "gt", value: "50.5" }, range(51, 100));
row("A13", "price between 10.5 and 19.5 → 10", { field: "price", op: "between", value: "10.5", value2: "19.5" }, range(10, 19));
row("A14", "flag is true → 50", { field: "flag", op: "eq_bool", value: "true" }, all.filter((i) => Number(i) % 2 === 0));
row("A15", "description contains all words quick fox → 15", { field: "description", op: "text", value: "quick fox" }, range(1, 15));
row("A16", "description contains any word quick lazy → 40", { field: "description", op: "text_any", value: "quick lazy" }, range(1, 40));
row("A17", "description exact phrase quick brown fox → 15", { field: "description", op: "phrase", value: "quick brown fox" }, range(1, 15));
row("A18", "description exact phrase fox brown quick → 0 with a coherent empty state", { field: "description", op: "phrase", value: "fox brown quick" }, [],
  { after: async (page) => { await expect(page.locator("#ql-results .result-id")).toHaveCount(0); await expect(page.locator("#ql-results .alert-error")).toHaveCount(0); } });
row("A19", "notes contains all words gamma → 30", { field: "notes", op: "text", value: "gamma" }, range(1, 30));

test("A20 — notes offers exact phrase disabled because its index lacks phrase_matching", async ({ page }) => {
  const errors = await open(page);
  const index = await condition(page, { field: "notes" });
  const option = rows(page).nth(index).locator('[data-role="op"] option[value="phrase"]');
  await expect(option).toBeDisabled();
  await expect(option).toHaveAttribute("title", /index lacks phrase_matching/);
  await setRow(page, index, { field: "description" });
  await expect(rows(page).nth(index).locator('[data-role="op"] option[value="phrase"]')).toBeEnabled();
  expect(errors).toEqual([]);
});

row("A21", "location within 1000 m of Milan → 25", { field: "location", op: "geo_radius", value: "9.19", value2: "45.46", value3: "1000" }, range(1, 25));
row("A22", "location within the box TL 8,47 → BR 14,40 → 60", { field: "location", op: "geo_box", value: "8", value2: "47", value3: "14", value4: "40" }, range(1, 60));
row("A23", "location within the open polygon (auto-closed) → 60", { field: "location", op: "geo_polygon", value: "6 50; 16 50; 16 36; 6 36" }, range(1, 60),
  { after: async (page, { evidence }) => {
    const ring = evidence[0].request.filters.must[0].geo_polygon.exterior.points;
    expect(ring).toHaveLength(5);
    expect(ring[0]).toEqual(ring.at(-1));
  } });
row("A24", "polygon with the Rome hole → 25", { field: "location", op: "geo_polygon", value: "6 50; 16 50; 16 36; 6 36 | 11 43; 14 43; 14 40.5; 11 40.5" }, range(1, 25),
  { after: async (page, { evidence }) => {
    const polygon = evidence[0].request.filters.must[0].geo_polygon;
    expect(polygon.interiors).toHaveLength(1);
    expect(polygon.interiors[0].points[0]).toEqual({ lon: 11, lat: 43 });
  } });
row("A25", "created_at at least 2026-04-01 → 10", { field: "created_at", op: "gte", value: "2026-04-01T00:00:00Z" }, range(91, 100));
row("A26", "created_at between Jan 1 and Jan 31 inclusive → 31", { field: "created_at", op: "between", value: day(1), value2: day(31) }, range(1, 31));
row("A27", "doc_uuid equals uuid 1 → 1", { field: "doc_uuid", op: "eq", value: uuid(1) }, ["1"]);
row("A28", "doc_uuid any of uuids 1..5 (CSV) → 5", { field: "doc_uuid", op: "any", value: range(1, 5).map(Number).map(uuid).join(", ") }, range(1, 5));
row("A29", "tags array size between 2 and 2 → 40", { field: "tags", op: "values_count", value: "2", value2: "2" }, range(41, 80));
row("A30", "tags array size at least 2 with open upper bound → 60", { field: "tags", op: "values_count", value: "2", value2: "" }, range(41, 100));
row("A31", "tags is empty → 10", { field: "tags", op: "is_empty" }, range(1, 10));
row("A32", "optional_note is null → 10", { field: "optional_note", op: "is_null" }, range(11, 20));
row("A33", "optional_note is empty (absent + null) → 20", { field: "optional_note", op: "is_empty" }, range(1, 20));
row("A34", "(point id) any of 1, 2, 3 → 3", { field: "(point id)", op: "id_any", value: "1, 2, 3" }, range(1, 3),
  { after: async (page, { evidence }) => expect(evidence[0].request.filters).toEqual({ must: [{ has_id: [1, 2, 3] }] }) });

// ── Series B — combinations and modes ────────────────────────────────────────

row("B1", "MUST status=active AND MUST views ≤ 10 → 10",
  [{ field: "status", op: "eq", value: "active" }, { field: "views", op: "lte", value: "10" }], range(1, 10));
row("B2", "MUST NOT status=active → 80", { clause: "must_not", field: "status", op: "eq", value: "active" }, range(21, 100));

for (const [id, minimum, expected] of [["B3", 1, all.filter((i) => Number(i) <= 20 || Number(i) % 2 === 0)], ["B4", 2, all.filter((i) => Number(i) <= 20 && Number(i) % 2 === 0)]]) {
  test(`${id} — SHOULD views ≤ 20 + SHOULD flag true with min should ${minimum} → ${expected.length}`, async ({ page }, info) => {
    const errors = await open(page);
    await condition(page, { clause: "should", field: "views", op: "lte", value: "20" });
    await condition(page, { clause: "should", field: "flag", op: "eq_bool", value: "true" });
    await expect(page.locator("#ql-minshould-field")).toBeVisible();
    await page.locator("#ql-minshould").fill(String(minimum));
    await page.locator("#ql-minshould").dispatchEvent("input");
    const { ids, evidence } = await collect(page, info);
    expect(ids).toEqual(sorted(expected));
    if (minimum === 2) expect(evidence[0].request.filters.min_should.min_count).toBe(2);
    else expect(JSON.stringify(evidence[0].request)).not.toContain("min_count");
    expect(errors).toEqual([]);
  });
}

row("B5", "MUST tags any red AND MUST NOT flag true → 45",
  [{ field: "tags", op: "any", value: "red" }, { clause: "must_not", field: "flag", op: "eq_bool", value: "true" }], range(11, 100).filter((i) => Number(i) % 2));

// Same-element (B6) versus flat (B7): the discriminating pair of the matrix.
async function nestedGroup(page, clause = "must") {
  await page.locator("#ql-filter-nested-add").click();
  const group = page.locator("#ql-filter-rows .filter-group-card.is-nested").first();
  await group.locator('[data-role="nested-root"]').selectOption("items");
  if (clause !== "must") await group.locator(':scope > .filter-group-toolbar > [data-role="clause"]').selectOption(clause);
  await setRow(page, 0, { field: "items[].label", op: "eq", value: "a" });
  await group.locator(':scope > .filter-group-actions > [data-action="add-condition"]').click();
  await setRow(page, 1, { field: "items[].qty", op: "gte", value: "90" });
  return group;
}

test("B6 — same-element items[].label=a AND items[].qty ≥ 90 → 11", {
  annotation: { type: "case-partial", description: "TB-FIL-011" },
}, async ({ page }, info) => {
  const errors = await open(page);
  await nestedGroup(page);
  const { ids, evidence } = await collect(page, info);
  expect(ids).toEqual(range(90, 100));
  // One nested root with relative child paths, not two independent conditions.
  expect(evidence[0].request.filters).toEqual({ must: [{ nested: { key: "items", filter: { must: [
    { key: "label", match: { value: "a" } }, { key: "qty", range: { gte: 90 } }] } } }] });
  expect(errors).toEqual([]);
});

row("B7", "flat MUST items[].label=a AND MUST items[].qty ≥ 90 → 21 (i ≤ 10 ∪ i ≥ 90)",
  [{ field: "items[].label", op: "eq", value: "a" }, { field: "items[].qty", op: "gte", value: "90" }], [...range(1, 10), ...range(90, 100)],
  { annotation: { type: "case-partial", description: "TB-FIL-011" } });

test("TB-FIL-011 — negating the whole same-element group selects the complement 1..89", async ({ page }, info) => {
  const errors = await open(page);
  await nestedGroup(page, "must_not");
  const { ids, evidence } = await collect(page, info);
  expect(ids).toEqual(range(1, 89));
  expect(evidence[0].request.filters.must_not[0].nested.key).toBe("items");
  expect(errors).toEqual([]);
});

test("B8 — a single SHOULD row hides the min should control", async ({ page }) => {
  const errors = await open(page);
  await condition(page, { clause: "should", field: "views", op: "lte", value: "20" });
  await expect(page.locator("#ql-minshould-field")).toBeHidden();
  await condition(page, { clause: "should", field: "flag", op: "eq_bool", value: "true" });
  await expect(page.locator("#ql-minshould-field")).toBeVisible();
  expect(errors).toEqual([]);
});

test("B9 — hybrid query with MUST status=active and limit 5 returns ≤ 5 active hits with fusion tiles", async ({ page }) => {
  const errors = await open(page);
  await page.locator("#ql-query").fill("ricetta di cucina italiana");
  await page.locator("#ql-limit").fill("5");
  await page.locator("#ql-limit").dispatchEvent("input");
  await condition(page, { field: "status", op: "eq", value: "active" });
  const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/query/search");
  await page.locator("#ql-run").click();
  const body = await (await pending).json();
  expect(body.results.length).toBeLessThanOrEqual(5);
  expect(body.results.length).toBeGreaterThan(0);
  for (const hit of body.results) {
    expect(hit.payload.status).toBe("active");
    expect(Number(hit.id)).toBeLessThanOrEqual(20);
  }
  await expect(page.locator("#ql-results .score-fill").first()).toBeVisible();
  await expect(page.locator("#ql-results")).toContainText(/fusion/i);
  await expect(page.locator("#ql-results")).toContainText(/candidates/i);
  expect(errors).toEqual([]);
});

test("B10 — payload-only with only status selected returns payload cards with just status", async ({ page }, info) => {
  const errors = await open(page);
  const boxes = page.locator("#ql-payload-fields input");
  // The sampled key list arrives asynchronously and re-renders everything selected.
  await expect(boxes).toHaveCount(14);
  for (let index = 0; index < await boxes.count(); index++) {
    const box = boxes.nth(index);
    if (await box.isEnabled()) await box.setChecked(await box.inputValue() === "status");
  }
  await expect(page.locator('#ql-payload-fields input[value="status"]')).toBeChecked();
  await expect(page.locator('#ql-payload-fields input[value="text"]')).not.toBeChecked();
  await condition(page, { field: "views", op: "eq", value: "50" });
  const { evidence } = await collect(page, info);
  expect(evidence[0].request.with_payload).toEqual(["status"]);
  expect(evidence[0].ids).toEqual(["50"]);
  // The projection is honoured end to end: the stored point carries only status.
  expect(evidence[0].payloads).toEqual([{ status: "archived" }]);
  const card = page.locator("#ql-results .result-card").first();
  await expect(card).toContainText("50");
  await expect(card).not.toContainText("views");
  expect(errors).toEqual([]);
});

test("B11 — status=draft paginates 8 pages of 5 with Previous back to page 1", async ({ page }, info) => {
  const errors = await open(page);
  await condition(page, { field: "status", op: "eq", value: "draft" });
  const { ids, evidence } = await collect(page, info);
  expect(ids).toEqual(range(51, 90));
  expect(evidence).toHaveLength(8);
  await expect(page.locator("#ql-results")).toContainText("Page 8 of 8");
  await expect(page.locator("#ql-results")).toContainText(/End of results/i);
  for (let back = 7; back >= 1; back--) {
    await page.locator('#ql-results [data-query-page="previous"]').click();
    await expect(page.locator("#ql-results")).toContainText(`Page ${back} of 8`);
  }
  await expect(page.locator('#ql-results [data-query-page="previous"]')).toBeDisabled();
  expect(errors).toEqual([]);
});

row("B12", "no filters and empty query → 100 points, 20 pages, ordered by id", [], all,
  { after: async (page, { evidence }) => {
    expect(evidence).toHaveLength(20);
    await expect(page.locator("#ql-errors .alert-info")).toContainText("first page of all points");
  } });

// ── §8.3 TB-FIL-002..010 ─────────────────────────────────────────────────────

row("TB-FIL-002/array-except", "tags except red → 41..100 (each has a non-red value)", { field: "tags", op: "except", value: "red" }, range(41, 100));
row("TB-FIL-002/must-not-any", "MUST NOT tags any red → 1..10", { clause: "must_not", field: "tags", op: "any", value: "red" }, range(1, 10));

for (const [field, list, yes, no, wire] of [
  ["views", [1, 2, 3], range(1, 3), range(4, 100), "number"],
  ["doc_uuid", range(1, 5).map(Number).map(uuid), range(1, 5), range(6, 100), "string"],
  ["status", ["active", "deleted"], [...range(1, 20), ...range(91, 100)], range(21, 90), "string"],
]) {
  for (const format of ["csv", "json"]) for (const op of ["any", "except"]) {
    row(`TB-FIL-003/${field}-${op}-${format}`, `${field} ${op} list as ${format}`,
      { field, op, value: format === "csv" ? list.join(",") : JSON.stringify(list) }, op === "any" ? yes : no,
      { after: async (page, { evidence }) => {
        const sent = evidence[0].request.filters.must[0].match[op];
        expect(sent).toEqual(list);
        for (const value of sent) expect(typeof value).toBe(wire);
      } });
  }
}

for (const [id, field, bound] of [["TB-FIL-004", "price", (i) => i + 0.5], ["TB-FIL-005", "views", (i) => i], ["TB-FIL-006", "created_at", day]]) {
  for (const [op, a, b] of [["gt", 51, 100], ["gte", 50, 100], ["lt", 1, 49], ["lte", 1, 50]]) {
    row(`${id}/${op}-threshold`, `${field} ${op} day/value 50`, { field, op, value: String(bound(50)) }, range(a, b));
  }
  for (const lowerOp of ["gte", "gt"]) for (const upperOp of ["lte", "lt"]) {
    row(`${id}/${lowerOp}-${upperOp}`, `${field} custom range 10..19 ${lowerOp}/${upperOp}`,
      { field, op: "range", lowerOp, upperOp, value: String(bound(10)), value2: String(bound(19)) },
      range(lowerOp === "gt" ? 11 : 10, upperOp === "lt" ? 18 : 19),
      { after: async (page, { evidence }) => expect(Object.keys(evidence[0].request.filters.must[0].range).sort()).toEqual([lowerOp, upperOp].sort()) });
  }
  row(`${id}/open-lower`, `${field} > value 99 → id 100`, { field, op: "range", lowerOp: "gt", value: String(bound(99)), value2: "" }, ["100"]);
  row(`${id}/open-upper`, `${field} < value 2 → id 1`, { field, op: "range", upperOp: "lt", value: "", value2: String(bound(2)) }, ["1"]);
  row(`${id}/equal-bounds`, `${field} inclusive 50..50 → id 50`, { field, op: "range", lowerOp: "gte", upperOp: "lte", value: String(bound(50)), value2: String(bound(50)) }, ["50"]);
}
for (const [op, a, b] of [["gt", 51, 100], ["gte", 50, 100], ["lt", 1, 49], ["lte", 1, 50]]) {
  row(`TB-FIL-006/${op}-timezone`, `created_at ${op} day 50 expressed as +01:00 → same set`, { field: "created_at", op, value: "2026-02-19T01:00:00+01:00" }, range(a, b));
}

row("TB-FIL-007/bool-false", "flag = false → odd ids", { field: "flag", op: "eq_bool", value: "false" }, all.filter((i) => Number(i) % 2));
for (const [op, value, expected] of [["text", "quick fox", range(1, 15)], ["text_any", "quick lazy", range(1, 40)], ["phrase", "quick brown fox", range(1, 15)], ["phrase", "fox brown quick", []]]) {
  row(`TB-FIL-007/${op}-${value.replaceAll(" ", "-")}`, `description ${op} "${value}"`, { field: "description", op, value }, expected);
}
test("TB-FIL-007/notes-phrase — the disabled phrase choice cannot send a phrase request", async ({ page }) => {
  const errors = await open(page);
  let phraseRequests = 0;
  page.on("request", (r) => { if (/\/points\/scroll$/.test(new URL(r.url()).pathname) && (r.postData() || "").includes('"phrase"')) phraseRequests++; });
  const index = await condition(page, { field: "notes", op: "text", value: "gamma" });
  const phrase = rows(page).nth(index).locator('[data-role="op"] option[value="phrase"]');
  await expect(phrase).toBeDisabled();
  await expect(phrase).toHaveAttribute("title", /index lacks phrase_matching/);
  await expect(page.locator("#ql-run")).toBeEnabled();
  await page.locator("#ql-run").click();
  await expect(page.locator("#ql-results .result-id").first()).toBeVisible();
  expect(phraseRequests).toBe(0);
  expect(errors).toEqual([]);
});

for (const [label, spec, expected, check] of [
  ["radius", { op: "geo_radius", value: "9.19", value2: "45.46", value3: "1000" }, range(1, 25), null],
  ["box", { op: "geo_box", value: "8", value2: "47", value3: "14", value4: "40" }, range(1, 60), null],
  ["polygon-auto", { op: "geo_polygon", value: "6 50; 16 50; 16 36; 6 36" }, range(1, 60), 5],
  ["polygon-closed", { op: "geo_polygon", value: "6 50; 16 50; 16 36; 6 36; 6 50" }, range(1, 60), 5],
  ["polygon-hole", { op: "geo_polygon", value: "6 50; 16 50; 16 36; 6 36 | 11 43; 14 43; 14 40.5; 11 40.5" }, range(1, 25), 5],
]) {
  row(`TB-FIL-008/${label}`, `location ${label}`, { field: "location", ...spec }, expected, { after: async (page, { evidence }) => {
    if (check === null) return;
    const polygon = evidence[0].request.filters.must[0].geo_polygon;
    const ring = polygon.exterior.points;
    // Closure adds exactly one endpoint (never a duplicate ring) and keeps lon/lat orientation.
    expect(ring).toHaveLength(check);
    expect(ring[0]).toEqual(ring.at(-1));
    expect(ring[0]).toEqual({ lon: 6, lat: 50 });
    if (label === "polygon-hole") expect(polygon.interiors[0].points.at(-1)).toEqual(polygon.interiors[0].points[0]);
    else expect(polygon.interiors).toBeUndefined();
  } });
}

for (const field of ["k", "i", "f", "b", "t", "g", "d", "u"]) {
  for (const [label, spec, expected] of [
    ["zero", { value: "0", value2: "0" }, ["1"]], ["one", { value: "1", value2: "1" }, ["2"]], ["two", { value: "2", value2: "2" }, ["3"]],
    ["lower", { value: "1", value2: "" }, ["2", "3"]], ["upper", { value: "", value2: "1" }, ["1", "2"]],
    ["exclusive", { lowerOp: "gt", upperOp: "lt", value: "0", value2: "2" }, ["2"]],
  ]) {
    row(`TB-FIL-009/${field}-${label}`, `values_count ${label} on ${field} restricted to ids 1..3`,
      [{ field: "(point id)", op: "id_any", value: "1,2,3" }, { field, op: "values_count", ...spec }], expected, { collection: EDGES });
  }
  row(`TB-FIL-009/${field}-scalar`, `scalar id 6 has values_count 1 on ${field}`,
    [{ field: "(point id)", op: "id_any", value: "6" }, { field, op: "values_count", value: "1", value2: "1" }], ["6"], { collection: EDGES });
  row(`TB-FIL-010/${field}-empty`, `is_empty on ${field} → 1, 4, 5`, { field, op: "is_empty" }, ["1", "4", "5"], { collection: EDGES });
  row(`TB-FIL-010/${field}-null`, `is_null on ${field} → 5`, { field, op: "is_null" }, ["5"], { collection: EDGES });
}

test("TB-FIL-010/scalar-to-unary — switching a scalar operator to a unary one drops the value requirement in place", async ({ page }, info) => {
  const errors = await open(page, EDGES);
  const index = await condition(page, { field: "k", op: "eq", value: "" });
  await expect(page.locator("#ql-run")).toBeDisabled();
  await setRow(page, index, { op: "is_null" });
  await expect(rows(page).nth(index).locator('[data-role^="value"]')).toHaveCount(0);
  const { ids } = await collect(page, info);
  expect(ids).toEqual(["5"]);
  expect(errors).toEqual([]);
});

// ── TB-FIL-012 — nested logical groups and independent minima ───────────────

test("TB-FIL-012 — status=active AND (SHOULD views ≤ 5, SHOULD views ≥ 15) → 11; minimum 2 → empty; 2→1 restores; root and child minima stay independent", async ({ page }, info) => {
  const errors = await open(page);
  await condition(page, { field: "status", op: "eq", value: "active" });
  await page.locator("#ql-filter-group-add").click();
  let group = page.locator("#ql-filter-rows .filter-group-card").first();
  // Nest the group inside another MUST group: parentheses preserve the set.
  await group.locator(".filter-row [data-action=\"remove\"]").click();
  await group.locator(':scope > .filter-group-actions > [data-action="add-group"]').click();
  group = group.locator(".filter-group-card").first();
  await setRow(page, 1, { clause: "should", field: "views", op: "lte", value: "5" });
  await group.locator(':scope > .filter-group-actions > [data-action="add-condition"]').click();
  await setRow(page, 2, { clause: "should", field: "views", op: "gte", value: "15" });
  const minimum = group.locator('[data-role="group-min"]');
  await expect(minimum).toBeVisible();
  const first = await collect(page, info);
  expect(first.ids).toEqual([...range(1, 5), ...range(15, 20)]);
  await minimum.fill("2");
  await minimum.dispatchEvent("input");
  await expect.poll(async () => JSON.stringify(await requestJson(page))).toContain('"min_count":2');
  const second = await collect(page, info);
  expect(second.ids).toEqual([]);
  await minimum.fill("1");
  await minimum.dispatchEvent("input");
  await expect.poll(async () => JSON.stringify(await requestJson(page))).not.toContain("min_count");
  const third = await collect(page, info);
  expect(third.ids).toEqual(first.ids);
  // The root has no SHOULD rows: its control stays hidden and untouched.
  await expect(page.locator("#ql-minshould-field")).toBeHidden();
  await expect(page.locator("#ql-minshould")).toHaveValue("1");
  expect(errors).toEqual([]);
});
