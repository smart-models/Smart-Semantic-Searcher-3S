// Test book §8.2 Series C (C1–C21) and §8.3 TB-FIL-001/013/014/015/016:
// client-side filter validation and operator matrix, fully intercepted.
// Every invalid variant must disable Run, associate a readable error with the
// row's value control and send zero search/scroll requests; a corrected value
// re-enables Run and sends exactly one request carrying the corrected filter.
// Positive semantics against the 100-point fixture live in tb-filters-live.spec.js.
import { expect, test } from "./fixtures.js";
import { A, B, openConsole } from "./application-fixtures.js";
import { addCondition, clearFilters, condition, requestJson, rows, setRow } from "./d-helpers.js";
import { fulfillResponse } from "./telemetry.mjs";

// §0.2 payload indexes of e2e_book_filters.
const BOOK_INDEXES = [
  ["status", "keyword"], ["tags", "keyword"], ["views", "integer"], ["price", "float"], ["flag", "bool"],
  ["description", "text", true], ["notes", "text", false], ["location", "geo"], ["created_at", "datetime"],
  ["doc_uuid", "uuid"], ["optional_note", "keyword"], ["items[].label", "keyword"], ["items[].qty", "integer"],
].map(([field_name, field_schema, phrase_matching]) => ({ field_name, field_schema,
  ...(field_schema === "text" ? { phrase_matching } : {}) }));
const BOOK_KEYS = ["text", "source_id", "status", "tags", "views", "price", "flag", "description", "notes",
  "location", "created_at", "doc_uuid", "optional_note", "items", "f_extra"];
const UNIVERSAL = ["values_count", "is_empty", "is_null"];
// §8.3 operator matrix, pinned to models.js.
const OPERATORS = {
  status: ["eq", "any", "except", ...UNIVERSAL], tags: ["eq", "any", "except", ...UNIVERSAL],
  views: ["eq", "any", "except", "gt", "gte", "lt", "lte", "between", "range", ...UNIVERSAL],
  price: ["gt", "gte", "lt", "lte", "between", "range", ...UNIVERSAL], flag: ["eq_bool", ...UNIVERSAL],
  description: ["text", "text_any", "phrase", ...UNIVERSAL], notes: ["text", "text_any", "phrase", ...UNIVERSAL],
  location: ["geo_radius", "geo_box", "geo_polygon", ...UNIVERSAL],
  created_at: ["gt", "gte", "lt", "lte", "between", "range", ...UNIVERSAL],
  doc_uuid: ["eq", "any", "except", ...UNIVERSAL], "(point id)": ["id_any"], "(vector presence)": ["has_vector"],
};

async function openFilters(page, options = {}) {
  const mock = await openConsole(page, { tab: "query", indexes: BOOK_INDEXES, payloadKeys: BOOK_KEYS, ...options });
  await expect(page.locator("#ql-filter-add")).toBeEnabled();
  await clearFilters(page);
  await page.locator("#ql-query").fill("");
  // Deterministic one-point replies: these tests judge requests, not ranking.
  mock.intercept = async (route, entry) => {
    if (entry.path !== "/points/scroll" && entry.path !== "/query/search") return false;
    const point = { id: "1", payload: { text: "mock point" }, vectors: null };
    const page_ = { collection_name: entry.body.collection_name, returned: 1, total: 1, next_page_offset: null, results: [point] };
    await fulfillResponse(route, { json: entry.path === "/points/scroll" ? page_
      : { ...page_, fusion: "rrf", reranked: false, candidates: 1, skipped_without_text: 0, results: [{ ...point, score: 0.9 }] } });
    return true;
  };
  mock.searches = () => mock.requestsFor("POST", "/points/scroll").length + mock.requestsFor("POST", "/query/search").length;
  return mock;
}

const run = (page) => page.locator("#ql-run");
const errors = (page) => page.locator("#ql-errors .alert-error");

// The row's value controls must be flagged and point at the error region.
async function expectRowInvalid(page, index, message) {
  await expect(run(page)).toBeDisabled();
  await expect(errors(page)).toHaveText([message]);
  const controls = rows(page).nth(index).locator('[data-role^="value"]');
  const count = await controls.count();
  expect(count).toBeGreaterThan(0);
  for (let position = 0; position < count; position++) {
    await expect(controls.nth(position)).toHaveAttribute("aria-invalid", "true");
    await expect(controls.nth(position)).toHaveAttribute("aria-describedby", /\bql-errors\b/);
  }
}

async function expectRowValid(page, index) {
  await expect(run(page)).toBeEnabled();
  await expect(errors(page)).toHaveCount(0);
  const controls = rows(page).nth(index).locator('[data-role^="value"]');
  for (let position = 0; position < await controls.count(); position++) {
    await expect(controls.nth(position)).not.toHaveAttribute("aria-invalid", "true");
  }
}

// Run once and return the single request body the console sent.
async function runOnce(page, mock) {
  const before = mock.searches();
  await run(page).click();
  await expect(page.locator("#ql-results .alert, #ql-results .result-id").first()).toBeVisible();
  await expect.poll(() => mock.searches()).toBe(before + 1);
  const scroll = mock.requestsFor("POST", "/points/scroll"), search = mock.requestsFor("POST", "/query/search");
  return (scroll.at(-1) || search.at(-1)).body;
}

// One invalid row, one correction: the C-series contract in one procedure.
async function invalidThenCorrected(page, mock, spec, message, fix, expectedFilter) {
  const index = await condition(page, spec);
  await expectRowInvalid(page, index, message);
  expect(mock.searches()).toBe(0);
  if (fix) {
    await setRow(page, index, fix);
    await expectRowValid(page, index);
    const body = await runOnce(page, mock);
    expect(body.limit).toBe(5);
    expect(body.filters).toEqual(expectedFilter);
    expect(mock.searches()).toBe(1);
  }
  expect(mock.errors).toEqual([]);
  expect(mock.unknown).toEqual([]);
}

const must = (condition) => ({ must: [condition] });

// ── Series C ─────────────────────────────────────────────────────────────────

test("C1 — between with min above max is refused and Run stays disabled", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "views", op: "between", value: "20", value2: "10" },
    "views: between min cannot exceed max", { value: "10", value2: "19" },
    must({ key: "views", range: { gte: 10, lte: 19 } }));
});

test("C2 — between with one empty bound names both bounds", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "views", op: "between", value: "20" },
    "views: between needs both min and max", { value2: "25" }, must({ key: "views", range: { gte: 20, lte: 25 } }));
});

test("TB-FIL-014/no-op edit keeps both between bounds associated with the error", async ({ page }) => {
  const mock = await openFilters(page);
  const index = await condition(page, { field: "views", op: "between", value: "20" });
  await expectRowInvalid(page, index, "views: between needs both min and max");
  await setRow(page, index, { value2: "" });
  await expectRowInvalid(page, index, "views: between needs both min and max");
  expect(mock.searches()).toBe(0);
});

test("C3 — a datetime bound outside the calendar is not an RFC 3339 instant", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "created_at", op: "gte", value: "2026-13-99" },
    "created_at: bound must be an RFC 3339 datetime", { value: "2026-04-01T00:00:00Z" },
    must({ key: "created_at", range: { gte: "2026-04-01T00:00:00Z" } }));
});

test("C4 — a two-point polygon needs at least 3 distinct points", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "location", op: "geo_polygon", value: "0 0; 1 1" },
    "location: needs at least 3 distinct points", { value: "6 50; 16 50; 16 36; 6 36" },
    must({ key: "location", geo_polygon: { exterior: { points: [
      { lon: 6, lat: 50 }, { lon: 16, lat: 50 }, { lon: 16, lat: 36 }, { lon: 6, lat: 36 }, { lon: 6, lat: 50 }] } } }));
});

test("C5 — an open ring is auto-closed in the request with first == last", async ({ page }) => {
  const mock = await openFilters(page);
  const index = await condition(page, { field: "location", op: "geo_polygon", value: "6 50; 16 50; 16 36; 6 36" });
  await expectRowValid(page, index);
  const preview = await requestJson(page);
  const ring = preview.filters.must[0].geo_polygon.exterior.points;
  expect(ring).toHaveLength(5);
  expect(ring[0]).toEqual(ring.at(-1));
  expect(ring.slice(0, 4)).toEqual([{ lon: 6, lat: 50 }, { lon: 16, lat: 50 }, { lon: 16, lat: 36 }, { lon: 6, lat: 36 }]);
  const body = await runOnce(page, mock);
  expect(body.filters.must[0].geo_polygon.exterior.points).toEqual(ring);
  expect(body.filters.must[0].geo_polygon.interiors).toBeUndefined();
  expect(mock.errors).toEqual([]);
});

test("C6 — a malformed hole is reported as polygon ring 2", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock,
    { field: "location", op: "geo_polygon", value: "6 50; 16 50; 16 36; 6 36 | 1 1; 2" },
    "location: polygon ring 2: each polygon point must be 'lon lat'",
    { value: "6 50; 16 50; 16 36; 6 36 | 11 43; 14 43; 14 40.5; 11 40.5" },
    must({ key: "location", geo_polygon: {
      exterior: { points: [{ lon: 6, lat: 50 }, { lon: 16, lat: 50 }, { lon: 16, lat: 36 }, { lon: 6, lat: 36 }, { lon: 6, lat: 50 }] },
      interiors: [{ points: [{ lon: 11, lat: 43 }, { lon: 14, lat: 43 }, { lon: 14, lat: 40.5 }, { lon: 11, lat: 40.5 }, { lon: 11, lat: 43 }] }],
    } }));
});

test("C7 — a negative radius is refused", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "location", op: "geo_radius", value: "9.19", value2: "45.46", value3: "-5" },
    "location: radius must be positive (meters)", { value3: "1000" },
    must({ key: "location", geo_radius: { center: { lon: 9.19, lat: 45.46 }, radius: 1000 } }));
});

test("C8 — a latitude of 91 is out of range", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "location", op: "geo_radius", value: "9.19", value2: "91", value3: "1000" },
    "location: lat must be between -90 and 90", { value2: "45.46" },
    must({ key: "location", geo_radius: { center: { lon: 9.19, lat: 45.46 }, radius: 1000 } }));
});

test("C9 — a scalar or list operation with an empty value is required, while unary operators stay valid", async ({ page }) => {
  const mock = await openFilters(page);
  const index = await condition(page, { field: "status", op: "any", value: "" });
  await expectRowInvalid(page, index, "status: value is required");
  await setRow(page, index, { op: "eq" });
  await expectRowInvalid(page, index, "status: value is required");
  await setRow(page, index, { op: "is_empty" });
  await expect(rows(page).nth(index).locator('[data-role^="value"]')).toHaveCount(0);
  await expect(run(page)).toBeEnabled();
  await expect(errors(page)).toHaveCount(0);
  const body = await runOnce(page, mock);
  expect(body.filters).toEqual(must({ key: "status", is_empty: true }));
  expect(mock.errors).toEqual([]);
});

test("C10 — changing the OPERATION clears the typed value", async ({ page }) => {
  const mock = await openFilters(page);
  const index = await condition(page, { field: "views", op: "eq", value: "50" });
  await expect(rows(page).nth(index).locator('[data-role="value"]')).toHaveValue("50");
  await setRow(page, index, { op: "gt" });
  await expect(rows(page).nth(index).locator('[data-role="value"]')).toHaveValue("");
  await expectRowInvalid(page, index, "views: value is required");
  expect(mock.searches()).toBe(0);
});

test("C11 — changing the FIELD clears the value and renormalizes the operation", async ({ page }) => {
  const mock = await openFilters(page);
  const index = await condition(page, { field: "views", op: "gt", value: "90" });
  await setRow(page, index, { field: "status" });
  const row = rows(page).nth(index);
  expect(await row.locator('[data-role="op"] option').evaluateAll((nodes) => nodes.map((n) => n.value))).toEqual(OPERATORS.status);
  expect(OPERATORS.status).not.toContain("gt");
  await expect(row.locator('[data-role="value"]')).toHaveValue("");
  await expectRowInvalid(page, index, "status: value is required");
  expect(mock.searches()).toBe(0);
});

test("C12 — an integer equality with text is refused", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "views", op: "eq", value: "abc" },
    "views: value must be an integer", { value: "50" }, must({ key: "views", match: { value: 50 } }));
});

test("C13 — an integer list with one non-integer is refused as a whole", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "views", op: "any", value: "1, abc" },
    "views: values must all be an integer", { value: "1, 2, 3" }, must({ key: "views", match: { any: [1, 2, 3] } }));
});

test("C14 — a bool row without a true/false selection is required", async ({ page }) => {
  const mock = await openFilters(page);
  const index = await condition(page, { field: "flag", op: "eq_bool" });
  await expectRowInvalid(page, index, "flag: value is required");
  expect(mock.searches()).toBe(0);
  await setRow(page, index, { value: "true" });
  await expectRowValid(page, index);
  const body = await runOnce(page, mock);
  expect(body.filters).toEqual(must({ key: "flag", match: { value: true } }));
});

test("C15 — point ids must be unsigned integers or UUIDs", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "(point id)", op: "id_any", value: "abc" },
    "(point id): ids must be unsigned integers or UUIDs", { value: "1, 2, 3" }, must({ has_id: [1, 2, 3] }));
});

test("C16 — a uuid equality needs a UUID", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "doc_uuid", op: "eq", value: "not-a-uuid" },
    "doc_uuid: value must be a UUID", { value: "00000000-0000-4000-8000-000000000001" },
    must({ key: "doc_uuid", match: { value: "00000000-0000-4000-8000-000000000001" } }));
});

test("C17 — removing a SHOULD row clamps min_should back to 1, hides the control and sends no stale min_count", async ({ page }) => {
  const mock = await openFilters(page);
  await condition(page, { clause: "should", field: "views", op: "lte", value: "20" });
  await condition(page, { clause: "should", field: "flag", op: "eq_bool", value: "true" });
  await expect(page.locator("#ql-minshould-field")).toBeVisible();
  await page.locator("#ql-minshould").fill("2");
  await page.locator("#ql-minshould").dispatchEvent("input");
  expect((await requestJson(page)).filters.min_should).toEqual({ min_count: 2, conditions: expect.any(Array) });
  await rows(page).nth(1).locator('[data-action="remove"]').click();
  await expect(rows(page)).toHaveCount(1);
  await expect(page.locator("#ql-minshould-field")).toBeHidden();
  await expect(page.locator("#ql-minshould")).toHaveValue("1");
  const body = await runOnce(page, mock);
  expect(JSON.stringify(body)).not.toContain("min_count");
  expect(body.filters).toEqual({ should: [{ key: "views", range: { lte: 20 } }] });
  expect(mock.errors).toEqual([]);
});

test("C18 — payload-only requests carry the fixed page size 5 whatever the limit control says", async ({ page }) => {
  const mock = await openFilters(page);
  await condition(page, { field: "status", op: "eq", value: "draft" });
  await expect(page.locator("#ql-limit")).toBeDisabled();
  const body = await runOnce(page, mock);
  expect(body.limit).toBe(5);
  expect(body).not.toHaveProperty("candidate_limit");
  expect(mock.requestsFor("POST", "/query/search")).toEqual([]);
});

test("C19 — a hybrid limit above candidate_limit blocks Run with the bound named", async ({ page }) => {
  const mock = await openFilters(page);
  await page.locator("#ql-query").fill("ricetta di cucina italiana");
  await page.locator("#ql-limit").fill("10");
  await page.locator("#ql-cand").fill("5");
  await page.locator("#ql-cand").dispatchEvent("input");
  await expect(run(page)).toBeDisabled();
  await expect(errors(page)).toHaveText(["candidate_limit must be greater than or equal to limit"]);
  await expect(page.locator("#ql-cand")).toHaveAttribute("aria-invalid", "true");
  expect(mock.searches()).toBe(0);
  await page.locator("#ql-cand").fill("20");
  await page.locator("#ql-cand").dispatchEvent("input");
  await expect(run(page)).toBeEnabled();
  await expect(errors(page)).toHaveCount(0);
  const body = await runOnce(page, mock);
  expect(body).toMatchObject({ limit: 10, candidate_limit: 20, query: "ricetta di cucina italiana" });
  expect(mock.errors).toEqual([]);
});

test("C20 — switching collection resets the filter tree and results and loads the new field options", async ({ page }) => {
  const mock = await openFilters(page);
  await condition(page, { field: "status", op: "eq", value: "active" });
  await runOnce(page, mock);
  await expect(page.locator("#ql-results .result-id")).toHaveCount(1);
  mock.indexes = [{ field_name: "other_field", field_schema: "keyword" }];
  await page.locator("#conn-collection").selectOption(B);
  await expect(page.locator("#ql-collection")).toHaveValue(B);
  await expect(rows(page)).toHaveCount(0);
  await expect(page.locator("#ql-results .result-id")).toHaveCount(0);
  await expect(page.locator("#ql-filter-add")).toBeEnabled();
  const index = await addCondition(page);
  const offered = await rows(page).nth(index).locator('[data-role="field"] option').evaluateAll((nodes) => nodes.map((n) => n.value));
  expect(offered).toContain("other_field");
  expect(offered).not.toContain("status");
  expect(mock.errors).toEqual([]);
});

test("C21 — a reply for the previous collection arriving after a switch is never rendered", async ({ page }) => {
  const mock = await openFilters(page);
  await condition(page, { field: "status", op: "eq", value: "active" });
  const held = mock.deferReply("POST", "/points/scroll");
  await run(page).click();
  await held.started;
  await page.locator("#conn-collection").selectOption(B);
  await expect(page.locator("#ql-collection")).toHaveValue(B);
  held.resolve({ collection_name: A, returned: 1, total: 1, next_page_offset: null,
    results: [{ id: "77", payload: { text: "stale from A" } }] });
  await expect(page.locator("#ql-results")).not.toContainText("stale from A");
  await expect(page.locator("#ql-results .result-id")).toHaveCount(0);
  await expect(run(page)).toBeEnabled();
  expect(mock.errors).toEqual([]);
});

// ── §8.3 ─────────────────────────────────────────────────────────────────────

test("TB-FIL-001 — every schema offers exactly its operator set; phrase is disabled only without a phrase index; a type change resets the row", async ({ page }) => {
  const mock = await openFilters(page);
  const index = await addCondition(page);
  const row = rows(page).nth(index);
  const offered = await row.locator('[data-role="field"] option').evaluateAll((nodes) => nodes.map((n) => n.value));
  expect(offered).toEqual(expect.arrayContaining(Object.keys(OPERATORS)));
  expect(offered).not.toContain("f_extra");
  expect(offered).not.toContain("source_id");
  for (const [field, operators] of Object.entries(OPERATORS)) {
    await setRow(page, index, { field });
    expect(await row.locator('[data-role="op"] option').evaluateAll((nodes) => nodes.map((n) => n.value)), field).toEqual(operators);
    if (field === "description" || field === "notes") {
      const phrase = row.locator('[data-role="op"] option[value="phrase"]');
      expect(await phrase.isDisabled()).toBe(field === "notes");
      if (field === "notes") await expect(phrase).toHaveAttribute("title", /index lacks phrase_matching/);
    }
  }
  // A float value typed for `price` does not survive a change to an integer field.
  await setRow(page, index, { field: "price", op: "gt", value: "50.5" });
  await expectRowValid(page, index);
  await setRow(page, index, { field: "views" });
  await expect(row.locator('[data-role="value"]')).toHaveValue("");
  expect(await row.locator('[data-role="op"]').inputValue()).not.toBe("gt_float");
  await expectRowInvalid(page, index, "views: value is required");
  expect(mock.searches()).toBe(0);
  expect(mock.errors).toEqual([]);
});

test("TB-FIL-013 — pseudo-fields are unavailable inside a same-element group, a root change drops stale paths and removed groups contribute nothing", async ({ page }) => {
  const mock = await openFilters(page);
  await page.locator("#ql-filter-nested-add").click();
  const group = page.locator("#ql-filter-rows .filter-group-card.is-nested").first();
  await expect(group).toBeVisible();
  const root = group.locator('[data-role="nested-root"]');
  await expect(root).toBeVisible();
  expect(await root.locator("option").evaluateAll((nodes) => nodes.map((n) => n.value))).toEqual(["items"]);
  // A same-element group is born with one condition row.
  await expect(group.locator(".filter-row")).toHaveCount(1);
  const nestedRow = group.locator(".filter-row").first();
  const nestedFields = await nestedRow.locator('[data-role="field"] option').evaluateAll((nodes) => nodes.map((n) => n.value));
  expect(nestedFields.sort()).toEqual(["items[].label", "items[].qty"]);
  expect(nestedFields).not.toContain("(point id)");
  expect(nestedFields).not.toContain("(vector presence)");
  await setRow(page, 0, { field: "items[].label", op: "eq", value: "a" });
  const preview = await requestJson(page);
  expect(preview.filters).toEqual({ must: [{ nested: { key: "items", filter: { must: [{ key: "label", match: { value: "a" } }] } } }] });
  // A populated group removed as a whole leaves no condition behind.
  await group.locator('[data-action="remove"]').first().click();
  await expect(page.locator("#ql-filter-rows .filter-row")).toHaveCount(0);
  expect((await requestJson(page)).filters).toBeUndefined();
  // Outside a group the pseudo-fields are offered again.
  const index = await addCondition(page);
  const offered = await rows(page).nth(index).locator('[data-role="field"] option').evaluateAll((nodes) => nodes.map((n) => n.value));
  expect(offered).toEqual(expect.arrayContaining(["(point id)", "(vector presence)"]));
  expect(mock.searches()).toBe(0);
  expect(mock.errors).toEqual([]);
});

const FIL014 = [
  ["integer fraction", { field: "views", op: "eq", value: "1.5" }, "views: value must be an integer"],
  ["float NaN", { field: "price", op: "gte", value: "NaN" }, "price: bound must be a number"],
  ["float Infinity", { field: "price", op: "gte", value: "Infinity" }, "price: bound must be a number"],
  ["negative array count", { field: "tags", op: "values_count", value: "-1", value2: "2" }, "tags: array size min must be a non-negative integer"],
  ["fractional array count", { field: "tags", op: "values_count", value: "1.5", value2: "2" }, "tags: array size min must be a non-negative integer"],
  ["both count bounds empty", { field: "tags", op: "values_count", value: "", value2: "" }, "tags: value is required"],
  ["UUID JSON list with a number", { field: "doc_uuid", op: "any", value: "[1]" }, "doc_uuid: JSON list values must all be UUID strings"],
  ["malformed JSON list", { field: "status", op: "any", value: '["active"' }, "status: value must be valid JSON array syntax"],
];
for (const [label, spec, message] of FIL014) {
  test(`TB-FIL-014/${label} — invalid input disables Run with a row-associated error and sends nothing`, async ({ page }) => {
    const mock = await openFilters(page);
    await invalidThenCorrected(page, mock, spec, message, null, null);
  });
}

test("TB-FIL-014/correction — fixing one invalid row re-enables Run and sends only the corrected filter", async ({ page }) => {
  const mock = await openFilters(page);
  await invalidThenCorrected(page, mock, { field: "tags", op: "values_count", value: "-1", value2: "2" },
    "tags: array size min must be a non-negative integer", { value: "2", value2: "2" },
    must({ key: "tags", values_count: { gte: 2, lte: 2 } }));
});

const FIL015 = [
  ["longitude 181", { field: "location", op: "geo_radius", value: "181", value2: "0", value3: "100" }, "location: lon must be between -180 and 180"],
  ["latitude -91", { field: "location", op: "geo_radius", value: "0", value2: "-91", value3: "100" }, "location: lat must be between -90 and 90"],
  ["radius 0", { field: "location", op: "geo_radius", value: "0", value2: "0", value3: "0" }, "location: radius must be positive (meters)"],
  ["repeated-point ring", { field: "location", op: "geo_polygon", value: "0 0; 1 1; 0 0; 1 1" }, "location: needs at least 3 distinct points"],
  ["malformed second ring", { field: "location", op: "geo_polygon", value: "0 0; 0 1; 1 1 | 1 1; 2" }, "location: polygon ring 2: each polygon point must be 'lon lat'"],
  ["invalid date", { field: "created_at", op: "gte", value: "2026-13-99" }, "created_at: bound must be an RFC 3339 datetime"],
  ["invalid leap day", { field: "created_at", op: "gte", value: "2026-02-29T00:00:00Z" }, "created_at: bound must be an RFC 3339 datetime"],
  ["invalid timezone", { field: "created_at", op: "gte", value: "2026-01-01T00:00:00+25:00" }, "created_at: bound must be an RFC 3339 datetime"],
];
for (const [label, spec, message] of FIL015) {
  test(`TB-FIL-015/${label} — geometry and calendar validation sends nothing and keeps the input`, async ({ page }) => {
    const mock = await openFilters(page);
    const index = await condition(page, spec);
    await expectRowInvalid(page, index, message);
    expect(mock.searches()).toBe(0);
    // The user's input is retained for correction, not wiped.
    await expect(rows(page).nth(index).locator('[data-role="value"]')).toHaveValue(spec.value);
    expect(mock.errors).toEqual([]);
  });
}

test("TB-FIL-015/valid leap day — 2024-02-29T00:00:00Z is accepted and sent as typed", async ({ page }) => {
  const mock = await openFilters(page);
  const index = await condition(page, { field: "created_at", op: "gte", value: "2024-02-29T00:00:00Z" });
  await expectRowValid(page, index);
  const body = await runOnce(page, mock);
  expect(body.filters).toEqual(must({ key: "created_at", range: { gte: "2024-02-29T00:00:00Z" } }));
});

test("TB-FIL-016 — runtime depth/condition maxima admit an at-limit tree and refuse limit+1 without a request; SHOULD minima clamp", async ({ page }) => {
  const mock = await openFilters(page, { config: { queryFilterMaxDepth: 20, queryFilterMaxConditions: 4 } });
  const maxima = await page.evaluate(() => ({ depth: window.QHM_CONFIG.queryFilterMaxDepth, conditions: window.QHM_CONFIG.queryFilterMaxConditions }));
  expect(maxima).toEqual({ depth: 20, conditions: 4 });
  for (let index = 0; index < maxima.conditions; index++) {
    await condition(page, { clause: "should", field: "views", op: "eq", value: String(index + 1) });
  }
  await expect(run(page)).toBeEnabled();
  await expect(errors(page)).toHaveCount(0);
  await condition(page, { field: "status", op: "eq", value: "active" });
  await expect(run(page)).toBeDisabled();
  await expect(errors(page)).toHaveText([`filters exceed maximum condition count ${maxima.conditions}`]);
  expect(mock.searches()).toBe(0);
  await rows(page).nth(4).locator('[data-action="remove"]').click();
  await expect(run(page)).toBeEnabled();
  // Minimum SHOULD above the number of children is clamped to that number.
  const minShould = page.locator("#ql-minshould");
  await expect(minShould).toHaveAttribute("max", "4");
  await minShould.fill("4");
  await minShould.dispatchEvent("input");
  await rows(page).nth(3).locator('[data-action="remove"]').click();
  await expect(minShould).toHaveValue("3");
  await expect(minShould).toHaveAttribute("max", "3");
  await rows(page).nth(2).locator('[data-action="remove"]').click();
  await rows(page).nth(1).locator('[data-action="remove"]').click();
  await expect(page.locator("#ql-minshould-field")).toBeHidden();
  const body = await runOnce(page, mock);
  expect(JSON.stringify(body)).not.toContain("min_count");
  expect(mock.errors).toEqual([]);
});
