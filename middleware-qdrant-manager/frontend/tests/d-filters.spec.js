import { fixtureName } from "./harness.mjs";
// §8.2 Serie QF — payload filter matrix, driven through the Query UI against
// e2e_d_query (12 points, ids "1".."12"). Assertions are on the RETURNED IDS.
//
// Fixture, for i = 1..12:
//   f_status  keyword  = `s{i}`
//   f_title   text     = `alpha beta gamma {i}`   (phrase_matching enabled)
//   f_views   integer  = i
//   f_price   float    = i + 0.5
//   f_flag    bool     = (i % 2 === 0)
//   f_geo     geo      = { lon: 9 + i*0.01, lat: 45 }
//   f_created datetime = 2026-06-{i:02}T00:00:00Z
//   f_uuid    uuid     = 00000000-0000-4000-8000-{i:012}
//   f_extra   keyword  = `x{i}`   — NOT indexed (QF-C)
//   f_tags    keyword[]= ["a","b"] if i%3==0 · [] if i%3==1 · ["a"] if i%3==2
//   f_nullable         = null if i%4==0 · `v{i}` if i%4==1 · ABSENT otherwise
import { expect, test } from "./fixtures.js";
import { boot, selectActive, openTab } from "./helpers.js";

const COLL = fixtureName("e2e_d_query");
const ALL = Array.from({ length: 12 }, (_, k) => k + 1);

const tagsOf = (i) => (i % 3 === 0 ? ["a", "b"] : i % 3 === 1 ? [] : ["a"]);
const ids = (pred) => ALL.filter(pred).map(String).sort();

async function openQuery(page) {
  const errors = await boot(page, { active: COLL });
  await selectActive(page, COLL);
  await openTab(page, "query");
  // Payload-only mode: the query stays empty, so the scroll path is used and
  // #ql-limit is (by design) disabled — the page size comes from
  // PAYLOAD_PAGE_SIZE in models.js, which comfortably covers the 12 fixtures.
  await page.locator("#ql-query").fill("");
  // The indexed-field list loads asynchronously; a row added before it lands is
  // re-rendered with the default field, silently dropping the test's selection.
  await expect(page.locator("#ql-payload-fields input").first()).toBeAttached({ timeout: 30_000 });
  await expect
    .poll(async () => page.locator("#ql-payload-fields input").count(), { timeout: 30_000 })
    .toBeGreaterThan(5);
  // The query snapshot (including the filter tree) is restored across reloads,
  // so a previous case's rows would otherwise accumulate and invalidate the run.
  await clearFilters(page);
  return errors;
}

async function clearFilters(page) {
  for (let guard = 0; guard < 40; guard += 1) {
    const remove = page.locator('#ql-filter-rows [data-action="remove"]');
    if (!(await remove.count())) break;
    await remove.first().click();
  }
  await expect(page.locator("#ql-filter-rows .filter-row")).toHaveCount(0);
  await expect(page.locator("#ql-filter-rows .filter-group-card")).toHaveCount(0);
}

const rows = (page) => page.locator("#ql-filter-rows .filter-row");

async function addRow(page) {
  const before = await rows(page).count();
  await page.locator("#ql-filter-add").click();
  await expect(rows(page)).toHaveCount(before + 1);
  return rows(page).nth(before);
}

// Configure one condition row and VERIFY it stuck. A field/payload load that
// lands after the selection re-renders the row and resets it to the default
// field (observed race, see the report), so apply, verify, and retry.
async function applyRow(page, row, { clause, field, op, values = [], lowerOp, upperOp }) {
  if (clause) await row.locator('[data-role="clause"]').selectOption(clause);
  if (field) await row.locator('[data-role="field"]').selectOption(field);
  if (op) await row.locator('[data-role="op"]').selectOption(op);
  if (lowerOp) await row.locator('[data-role="lower-op"]').selectOption(lowerOp);
  if (upperOp) await row.locator('[data-role="upper-op"]').selectOption(upperOp);
  const roles = ["value", "value2", "value3", "value4"];
  for (let i = 0; i < values.length; i += 1) {
    const cell = row.locator(`[data-role="${roles[i]}"]`);
    const tag = await cell.evaluate((el) => el.tagName.toLowerCase());
    if (tag === "select") await cell.selectOption(String(values[i]));
    else await cell.fill(String(values[i]));
  }
}

async function setRow(page, row, cfg) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    await applyRow(page, row, cfg);
    await page.waitForTimeout(250);
    const okField = !cfg.field
      || (await row.locator('[data-role="field"]').inputValue()) === cfg.field;
    const okOp = !cfg.op || (await row.locator('[data-role="op"]').inputValue()) === cfg.op;
    if (okField && okOp) return;
  }
  throw new Error(`filter row did not keep field=${cfg.field} op=${cfg.op}`);
}

// Run the query and return the sorted list of returned point ids.
// Payload-only scroll pages at PAYLOAD_PAGE_SIZE (5), so walk every page through
// the UI pagination control before comparing the id set.
async function run(page) {
  await expect(page.locator("#ql-run")).toBeEnabled();
  await page.locator("#ql-run").click();
  // Either result cards or a "no results"/error block must appear.
  await expect
    .poll(async () => (await page.locator("#ql-results").textContent()) || "", { timeout: 60_000 })
    .not.toBe("");
  // A validation error means the row was not configured as intended: fail loudly
  // instead of silently reporting an empty id set.
  const errText = (await page.locator("#ql-errors").textContent()) || "";
  expect(errText, `query refused: ${errText}`).not.toMatch(/is required|validation/i);

  const out = new Set();
  const pageIds = async () =>
    (await page.locator("#ql-results .result-id").allTextContents()).map((t) => t.trim());
  for (const id of await pageIds()) out.add(id);
  for (let guard = 0; guard < 50; guard += 1) {
    const next = page.locator('[data-query-page="next"]');
    if (!(await next.count())) break;
    const before = (await pageIds()).join(",");
    await next.click();
    await expect.poll(async () => (await pageIds()).join(","), { timeout: 30_000 }).not.toBe(before);
    for (const id of await pageIds()) out.add(id);
  }
  return [...out].sort();
}

async function single(page, cfg) {
  const row = await addRow(page);
  await setRow(page, row, cfg);
  return run(page);
}

test.describe("QF — payload filters", () => {
  test("F01 — keyword: equals / any of / any value except", async ({ page }) => {
    await openQuery(page);
    expect(await single(page, { field: "f_status", op: "eq", values: ["s3"] })).toEqual(["3"]);

    await openQuery(page);
    expect(await single(page, { field: "f_status", op: "any", values: ["s1, s5, s9"] }))
      .toEqual(["1", "5", "9"].sort());

    await openQuery(page);
    expect(await single(page, { field: "f_status", op: "except", values: ["s1, s2"] }))
      .toEqual(ids((i) => i > 2));
  });

  test("F02 — text: contains all words / any word / exact phrase", async ({ page }) => {
    await openQuery(page);
    expect(await single(page, { field: "f_title", op: "text", values: ["alpha 7"] })).toEqual(["7"]);

    await openQuery(page);
    expect(await single(page, { field: "f_title", op: "text_any", values: ["3 8"] }))
      .toEqual(["3", "8"].sort());

    await openQuery(page);
    // phrase_matching is enabled on this collection's text index (seed).
    const phraseRow = await addRow(page);
    await setRow(page, phraseRow, { field: "f_title", op: "phrase", values: ["alpha beta gamma 4"] });
    await expect(phraseRow.locator('[data-role="op"]')).toHaveValue("phrase");
    expect(await run(page)).toEqual(["4"]);
  });

  test("F03 — integer: exact, list, except, and every range comparator", async ({ page }) => {
    await openQuery(page);
    expect(await single(page, { field: "f_views", op: "eq", values: [6] })).toEqual(["6"]);

    await openQuery(page);
    expect(await single(page, { field: "f_views", op: "gt", values: [9] })).toEqual(ids((i) => i > 9));

    await openQuery(page);
    expect(await single(page, { field: "f_views", op: "gte", values: [9] })).toEqual(ids((i) => i >= 9));

    await openQuery(page);
    expect(await single(page, { field: "f_views", op: "lt", values: [3] })).toEqual(ids((i) => i < 3));

    await openQuery(page);
    expect(await single(page, { field: "f_views", op: "lte", values: [3] })).toEqual(ids((i) => i <= 3));

    await openQuery(page);
    expect(await single(page, { field: "f_views", op: "between", values: [4, 6] }))
      .toEqual(ids((i) => i >= 4 && i <= 6));
  });

  test("F03b — integer custom range with exclusive bounds", async ({ page }) => {
    await openQuery(page);
    expect(await single(page, {
      field: "f_views", op: "range", values: [4, 6], lowerOp: "gt", upperOp: "lt",
    })).toEqual(["5"]);
  });

  test("F04 — float ranges", async ({ page }) => {
    await openQuery(page);
    expect(await single(page, { field: "f_price", op: "gte", values: [10.5] }))
      .toEqual(ids((i) => i + 0.5 >= 10.5));

    await openQuery(page);
    expect(await single(page, { field: "f_price", op: "between", values: [2.5, 4.5] }))
      .toEqual(ids((i) => i + 0.5 >= 2.5 && i + 0.5 <= 4.5));
  });

  test("F05 — bool", async ({ page }) => {
    await openQuery(page);
    expect(await single(page, { field: "f_flag", op: "eq_bool", values: ["true"] }))
      .toEqual(ids((i) => i % 2 === 0));

    await openQuery(page);
    expect(await single(page, { field: "f_flag", op: "eq_bool", values: ["false"] }))
      .toEqual(ids((i) => i % 2 !== 0));
  });

  test("F06 — geo: radius, box, polygon", async ({ page }) => {
    // Points sit on lat 45, lon 9.01 … 9.12 (about 785 m apart per 0.01°).
    // Cell order is lon, lat, radius(m) — not lat/lon.
    await openQuery(page);
    const radius = await single(page, {
      field: "f_geo", op: "geo_radius", values: [9.01, 45, 1200],
    });
    expect(radius.length).toBeGreaterThan(0);
    expect(radius).toContain("1");
    expect(radius).not.toContain("12");

    // Box cells are TL lon, TL lat, BR lon, BR lat.
    await openQuery(page);
    const box = await single(page, {
      field: "f_geo", op: "geo_box", values: [8.9, 45.5, 9.045, 44.5],
    });
    expect(box).toEqual(ids((i) => 9 + i * 0.01 <= 9.045));

    // Polygon uses the "lon lat; lon lat; …" text form, not JSON.
    await openQuery(page);
    const polygon = await single(page, {
      field: "f_geo", op: "geo_polygon",
      values: ["8.9 44.5; 9.045 44.5; 9.045 45.5; 8.9 45.5; 8.9 44.5"],
    });
    expect(polygon).toEqual(ids((i) => 9 + i * 0.01 <= 9.045));
  });

  test("F07 — datetime and uuid", async ({ page }) => {
    await openQuery(page);
    expect(await single(page, {
      field: "f_created", op: "between",
      values: ["2026-06-03T00:00:00Z", "2026-06-05T00:00:00Z"],
    })).toEqual(["3", "4", "5"].sort());

    await openQuery(page);
    expect(await single(page, {
      field: "f_uuid", op: "eq", values: ["00000000-0000-4000-8000-000000000008"],
    })).toEqual(["8"]);
  });

  test("F08 — array size range, inclusive and exclusive", async ({ page }) => {
    await openQuery(page);
    expect(await single(page, { field: "f_tags", op: "values_count", values: [2, 2] }))
      .toEqual(ids((i) => tagsOf(i).length === 2));

    await openQuery(page);
    expect(await single(page, {
      field: "f_tags", op: "values_count", values: [0, 2], lowerOp: "gt", upperOp: "lt",
    })).toEqual(ids((i) => tagsOf(i).length === 1));
  });

  test("F09 — is empty vs is null keep their distinction", async ({ page }) => {
    await openQuery(page);
    // is_empty: empty array OR missing field.
    const empty = await single(page, { field: "f_tags", op: "is_empty" });
    expect(empty).toEqual(ids((i) => tagsOf(i).length === 0));

    await openQuery(page);
    // is_null matches ONLY the explicit nulls (i % 4 === 0), never the absent ones.
    const nulls = await single(page, { field: "f_nullable", op: "is_null" });
    expect(nulls).toEqual(ids((i) => i % 4 === 0));
    for (const i of ALL.filter((n) => n % 4 !== 0 && n % 4 !== 1)) {
      expect(nulls, `absent f_nullable must not match is_null (id ${i})`).not.toContain(String(i));
    }
  });

  test("F10 — filter by point id", async ({ page }) => {
    await openQuery(page);
    const row = await addRow(page);
    await setRow(page, row, { field: "__point_id__", op: "id_any", values: ["2, 4, 6"] })
      .catch(async () => {
        // The point-id pseudo-field label may differ; pick the option that is
        // not a payload key.
        const opts = await row.locator('[data-role="field"] option').allTextContents();
        const idOpt = opts.find((o) => /id/i.test(o) && !/uuid/i.test(o));
        await row.locator('[data-role="field"]').selectOption({ label: idOpt });
        await row.locator('[data-role="op"]').selectOption("id_any");
        await row.locator('[data-role="value"]').fill("2, 4, 6");
      });
    expect(await run(page)).toEqual(["2", "4", "6"].sort());
  });

  test("F11 — MUST / SHOULD / MUST NOT and root min_should", async ({ page }) => {
    // MUST + MUST NOT
    await openQuery(page);
    const r1 = await addRow(page);
    await setRow(page, r1, { clause: "must", field: "f_views", op: "gte", values: [5] });
    const r2 = await addRow(page);
    await setRow(page, r2, { clause: "must_not", field: "f_flag", op: "eq_bool", values: ["true"] });
    expect(await run(page)).toEqual(ids((i) => i >= 5 && i % 2 !== 0));

    // Two SHOULDs, min_should = 1 → union
    await openQuery(page);
    const s1 = await addRow(page);
    await setRow(page, s1, { clause: "should", field: "f_status", op: "eq", values: ["s1"] });
    const s2 = await addRow(page);
    await setRow(page, s2, { clause: "should", field: "f_status", op: "eq", values: ["s2"] });
    await expect(page.locator("#ql-minshould-field")).toBeVisible();
    expect(await run(page)).toEqual(["1", "2"].sort());

    // min_should = 2 with mutually exclusive SHOULDs → empty
    await page.locator("#ql-minshould").fill("2");
    expect(await run(page)).toEqual([]);
  });

  test("F12 — logical group and same-element nesting", async ({ page }) => {
    await openQuery(page);
    await page.locator("#ql-filter-group-add").click();
    const group = page.locator("#ql-filter-rows .filter-group-card").first();
    await expect(group).toBeVisible();
    const gRow = group.locator(".filter-row").first();
    await setRow(page, gRow, { clause: "should", field: "f_views", op: "lt", values: [3] });
    await group.locator('[data-action="add-condition"]').click();
    const gRow2 = group.locator(".filter-row").nth(1);
    await setRow(page, gRow2, { clause: "should", field: "f_views", op: "gt", values: [10] });
    await expect(group.locator('[data-role="group-min"]')).toBeVisible();
    expect(await run(page)).toEqual(ids((i) => i < 3 || i > 10));
  });

  test("QF-A — the three add buttons build the expected structures", async ({ page }) => {
    await openQuery(page);
    await page.locator("#ql-filter-add").click();
    await expect(rows(page)).toHaveCount(1);
    await page.locator("#ql-filter-group-add").click();
    await expect(page.locator("#ql-filter-rows .filter-group-card")).toHaveCount(1);
    // Same-element requires an array root in context; the button reflects that.
    const nested = page.locator("#ql-filter-nested-add");
    await expect(nested).toBeVisible();
  });

  test("QF-B — min should appears only with 2+ SHOULD rows", async ({ page }) => {
    await openQuery(page);
    const r1 = await addRow(page);
    await setRow(page, r1, { clause: "should", field: "f_status", op: "eq", values: ["s1"] });
    await expect(page.locator("#ql-minshould-field")).toBeHidden();
    const r2 = await addRow(page);
    await setRow(page, r2, { clause: "should", field: "f_status", op: "eq", values: ["s2"] });
    await expect(page.locator("#ql-minshould-field")).toBeVisible();
    await r2.locator('[data-role="clause"]').selectOption("must");
    await expect(page.locator("#ql-minshould-field")).toBeHidden();
  });

  test("QF-C — a non-indexed field is not offered as a filter field", async ({ page }) => {
    await openQuery(page);
    const row = await addRow(page);
    const fields = await row.locator('[data-role="field"] option').allTextContents();
    // f_extra exists in the payload but has no index: it must not be selectable.
    expect(fields).not.toContain("f_extra");
    expect(fields).toContain("f_status");
  });

  test("QF-D — exact phrase is disabled on a text index without phrase_matching", async ({ page, request }) => {
    // e2e_d_nophrase: text index created WITHOUT phrase_matching.
    const name = fixtureName("e2e_d_nophrase");
    await boot(page, { active: name });
    await selectActive(page, name);
    await openTab(page, "query");
    await clearFilters(page);
    const row = await addRow(page);
    // Field options arrive with the async payload-keys load.
    await expect
      .poll(async () => row.locator('[data-role="field"] option').allTextContents(),
        { timeout: 30_000 })
      .toContain("f_title");
    await row.locator('[data-role="field"]').selectOption("f_title");
    const phrase = row.locator('[data-role="op"] option[value="phrase"]');
    await expect(phrase).toHaveAttribute("disabled", "");
    await expect(phrase).toHaveAttribute("title", /phrase_matching/);
  });

  test("Q01 — empty-query scroll with a filter retrieves the filtered set", async ({ page }) => {
    await openQuery(page);
    await expect(page.locator("#ql-query-mode")).toContainText(/PAYLOAD-ONLY/i);
    expect(await single(page, { field: "f_views", op: "lte", values: [4] }))
      .toEqual(ids((i) => i <= 4));
  });

  test("Q02 — filters survive hybrid RRF / weighted RRF / DBSF / rerank toggles", async ({ page }) => {
    for (const cfg of [
      { fusion: "rrf", rerank: true },
      { fusion: "rrf", rerank: false },
      { fusion: "dbsf", rerank: true },
    ]) {
      await openQuery(page);
      await page.locator("#ql-query").fill("hybrid vector search");
      await page.locator(`input[name="ql-fusion"][value="${cfg.fusion}"]`).check();
      const rerank = page.locator("#ql-rerank");
      if ((await rerank.isChecked()) !== cfg.rerank) await rerank.setChecked(cfg.rerank);
      const row = await addRow(page);
      await setRow(page, row, { field: "f_views", op: "lte", values: [4] });
      const got = await run(page);
      expect(got.length, `fusion=${cfg.fusion} rerank=${cfg.rerank}`).toBeGreaterThan(0);
      for (const id of got) expect(Number(id)).toBeLessThanOrEqual(4);
    }
  });
});
