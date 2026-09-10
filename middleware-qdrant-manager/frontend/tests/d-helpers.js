// Query-tab helpers shared by tests/d-query.spec.js and tests/d-filters.spec.js.
import { expect } from "@playwright/test";
import { fixtureName } from "./harness.mjs";

export const D_QUERY = fixtureName("e2e_d_query");
export const D_NESTED = fixtureName("e2e_d_nested");
export const D_NOPHRASE = fixtureName("e2e_d_nophrase");

// Wait until the collection's payload indexes/keys have landed: until then the
// filter FIELD dropdowns are empty and "Run" is disabled ("Loading fields…").
export async function waitFieldsLoaded(page) {
  await expect(page.locator("#ql-run")).not.toHaveText(/Loading fields/i, { timeout: 30_000 });
  await expect(page.locator("#ql-filter-add")).toBeEnabled();
}

export function requestJson(page) {
  return page.locator("#ql-request-json").textContent().then((t) => JSON.parse(t || "{}"));
}

export function rows(page) {
  return page.locator("#ql-filter-rows .filter-row");
}

export async function addCondition(page) {
  const before = await rows(page).count();
  await page.locator("#ql-filter-add").click();
  await expect(rows(page)).toHaveCount(before + 1);
  return before; // index of the row just added
}

// Set a root-level condition row. Field/op/clause changes re-render the tree,
// so every control is re-located right before use.
export async function setRow(page, index, spec) {
  const row = () => rows(page).nth(index);
  if (spec.clause) await row().locator('[data-role="clause"]').selectOption(spec.clause);
  if (spec.field) await row().locator('[data-role="field"]').selectOption(spec.field);
  if (spec.op) await row().locator('[data-role="op"]').selectOption(spec.op);
  if (spec.lowerOp) await row().locator('[data-role="lower-op"]').selectOption(spec.lowerOp);
  if (spec.upperOp) await row().locator('[data-role="upper-op"]').selectOption(spec.upperOp);
  for (const role of ["value", "value2", "value3", "value4"]) {
    const v = spec[role];
    if (v === undefined) continue;
    const control = row().locator(`[data-role="${role}"]`);
    const tag = await control.evaluate((el) => el.tagName);
    if (tag === "SELECT") await control.selectOption(String(v));
    else await control.fill(String(v));
  }
}

// One-shot: append a condition and configure it.
export async function condition(page, spec) {
  const index = await addCondition(page);
  await setRow(page, index, spec);
  return index;
}

export async function runQuery(page) {
  await expect(page.locator("#ql-run")).toBeEnabled();
  await page.locator("#ql-run").click();
  await expect(page.locator("#ql-results .alert")).toBeVisible({ timeout: 60_000 });
  await expect(page.locator("#ql-run")).not.toHaveText(/Running/i, { timeout: 60_000 });
}

export function idsOnPage(page) {
  return page.locator("#ql-results .result-id").allTextContents();
}

// Payload-only retrieval pages at PAYLOAD_PAGE_SIZE=5, so a filter matching
// more than five points needs the pagination walked before its ID set can be
// compared with the fixture. Returns ids as numbers where possible.
export async function collectAllIds(page) {
  const seen = [];
  for (let guard = 0; guard < 40; guard += 1) {
    seen.push(...(await idsOnPage(page)));
    const next = page.locator('#ql-results [data-query-page="next"]');
    if (!(await next.count())) break;
    await next.click();
    await expect(page.locator('#ql-results [data-query-page="next"], #ql-results .pagination-end'))
      .toBeVisible({ timeout: 30_000 });
    await expect(page.locator("#ql-results")).not.toContainText("Loading…", { timeout: 30_000 });
  }
  return seen;
}

export function asSorted(ids) {
  return [...ids].map((v) => Number(v)).sort((a, b) => a - b);
}

// Drive a single root-level condition and return the sorted matching ID set.
export async function filterIds(page, spec) {
  await clearFilters(page);
  await condition(page, spec);
  await runQuery(page);
  return asSorted(await collectAllIds(page));
}

export async function clearFilters(page) {
  for (let guard = 0; guard < 30; guard += 1) {
    const remove = page.locator('#ql-filter-rows [data-action="remove"]').first();
    if (!(await remove.count())) break;
    await remove.click();
  }
  await expect(page.locator("#ql-filter-rows .filter-row")).toHaveCount(0);
}

export function alertTexts(page, root = "#ql-errors") {
  return page.locator(`${root} .alert`).allTextContents();
}
