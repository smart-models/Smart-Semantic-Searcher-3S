import { fulfillResponse } from "./telemetry.mjs";
import { test, expect } from "./fixtures.js";

for (const [width, zoom] of [[320, 1], [768, 1], [1440, 1], [1440, 2]]) {
  test(`disabled explanations remain usable at ${width}px and ${zoom * 100}% CSS zoom`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await openConsole(page, { tab: "collections" });
    await page.evaluate((zoom) => { document.body.style.zoom = String(zoom); }, zoom);
    const explanation = page.locator("#cc-submit-reason");
    await explanation.scrollIntoViewIfNeeded();
    await expect(explanation).toBeVisible();
    await expect(page.locator("#cc-submit")).toHaveAccessibleDescription(/collection name/i);
    const bounds = await explanation.boundingBox();
    expect(bounds.x).toBeGreaterThanOrEqual(0);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(width + 1);
    await page.locator("#cc-name").fill("e2e_application_new");
    await expect(page.locator("#cc-submit")).toBeEnabled();
  });
}
import { openConsole, openPoints } from "./application-fixtures.js";
import { A, operations, openOperation, matches } from "./application-fixtures.js";

test("mobile filter group toolbars use two columns", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 1000 });
  await openConsole(page);
  await page.locator("#ql-filter-group-add").click();
  const toolbars = page.locator(".filter-group-card.no-min-should .filter-group-toolbar");
  await expect(toolbars.first()).toBeVisible();
  const columns = await toolbars.evaluateAll((elements) => elements.map((element) =>
    getComputedStyle(element).gridTemplateColumns.split(" ").length));
  expect(columns.length).toBeGreaterThan(0);
  expect(columns.every((count) => count === 2)).toBe(true);
});

const jsonInputs = { upsert: "up-payload", batch: "bt-json", "filtered delete": "delf-json", "filtered-delete count": "delf-json", "payload-filter count": "pl-filter-json", "payload point merge": "pl-json", "payload point replace": "pl-json", "payload filter merge": "pl-json", "payload filter keys": "pl-filter-json", "payload filter clear": "pl-filter-json", count: "cnt-json" };
for (const op of operations) {
  test(`${op.name}: errors can be read and pending state is explained`, async ({ page }) => {
    const mock = await openOperation(page, op);
    const resultId = op.name === "filtered-delete count" ? "delf-result" : op.result;
    if (jsonInputs[op.name]) {
      const control = page.locator(`#${jsonInputs[op.name]}`);
      const original = await control.inputValue();
      await control.fill("{invalid JSON");
      await page.locator(`#${op.button}`).click();
      await expect(control).toHaveAttribute("aria-invalid", "true");
      await expect(control).toHaveAccessibleDescription(/JSON/i);
      await control.fill(original);
      await expect(control).not.toHaveAttribute("aria-invalid", "true");
    }
    const late = mock.deferReply(op.method, op.path || `/collections/${A}${op.suffix}`);
    try {
      await page.locator(`#${op.button}`).click();
      await late.started;
      await expect(page.locator(`#${op.button}`)).toBeDisabled();
      const pending = op.button === "ql-run" ? /Running|progress/i : /progress/i;
      await expect(page.locator(`#${op.button}`)).toHaveAccessibleDescription(pending);
      late.resolve({ detail: "Operation unavailable. Refresh its state before retrying." }, 502);
      await expect(page.locator(`#${resultId}`)).toContainText("Operation unavailable");
      await expect(page.locator(`#${resultId}`)).toHaveAttribute("aria-live", "polite");
      expect(mock.errors).toEqual([]);
    } finally { late.resolve({}, 502); }
  });
}

test("structured server validation identifies the affected input only", async ({ page }) => {
  const mock = await openOperation(page, operations.find((op) => op.name === "upsert"));
  mock.intercept = async (route, request) => {
    if (!matches(operations.find((op) => op.name === "upsert"), request)) return false;
    await fulfillResponse(route, { status: 400, json: { detail: [{ loc: ["body", "point_id"], msg: "Point ID is invalid" }] } }); return true;
  };
  await page.locator("#up-submit").click();
  await expect(page.locator("#up-id")).toHaveAttribute("aria-invalid", "true");
  await expect(page.locator("#up-id")).toHaveAccessibleDescription(/Point ID is invalid/i);
  await expect(page.locator("#up-text")).not.toHaveAttribute("aria-invalid", "true");
});

test("filter editing and removal keep useful keyboard focus", async ({ page }) => {
  await openConsole(page);
  await page.locator("#ql-filter-add").click();
  const field = page.locator('[data-role="field"]');
  await field.focus();
  await field.selectOption("category");
  await expect(field).toBeFocused();
  const operation = page.locator('[data-role="op"]');
  await operation.focus();
  await operation.selectOption("any");
  await expect(operation).toBeFocused();
  await page.locator('[data-action="remove"]').focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#ql-filter-add")).toBeFocused();
});

test("batch validation describes its editor and clears on correction", async ({ page }) => {
  await openConsole(page);
  await openPoints(page, "batch");
  await page.locator("#bt-json").fill('[{"point_id":9007199254740993,"text":"keep"}]');
  await page.locator("#bt-submit").click();
  await expect(page.locator("#bt-json")).toHaveAttribute("aria-invalid", "true");
  await expect(page.locator("#bt-json")).toHaveAccessibleDescription(/quoted ID string/i);
  await page.locator("#bt-json").fill('[{"point_id":"9007199254740993","text":"keep"}]');
  await expect(page.locator("#bt-json")).not.toHaveAttribute("aria-invalid", "true");
});

test("a disabled create button has a visible explanation", async ({ page }) => {
  await openConsole(page, { tab: "collections" });
  await expect(page.locator("#cc-submit")).toBeDisabled();
  await expect(page.locator("#cc-submit")).toHaveAccessibleDescription(/collection name/i);
  await expect(page.locator("#cc-submit-reason")).toBeVisible();
});

for (const direction of ["next", "previous"]) {
  test(`pagination ${direction} retains keyboard focus and announces errors`, async ({ page }) => {
    const mock = await openConsole(page);
    mock.intercept = async (route, request) => {
      if (request.path !== "/points/scroll") return false;
      const second = Boolean(request.body.offset);
      await fulfillResponse(route, { json: {
        collection_name: A, returned: 1, total: 2,
        results: [{ id: second ? "2" : "1", payload: { text: "page" } }],
        next_page_offset: second ? null : "2",
      } }); return true;
    };
    await page.locator("#ql-run").click();
    const next = page.locator('[data-query-page="next"]');
    await expect(next).toBeEnabled();
    await next.focus();
    await page.keyboard.press("Enter");
    const previous = page.locator('[data-query-page="previous"]');
    await expect(previous).toBeFocused();
    if (direction === "next") {
      await previous.press("Enter");
      await expect(next).toBeFocused();
    }
    mock.intercept = async (route, request) => {
      if (request.path !== "/points/scroll") return false;
      await fulfillResponse(route, { status: 502, json: { detail: "Page unavailable" } }); return true;
    };
    await page.keyboard.press("Enter");
    await expect(page.locator("#ql-results")).toContainText("Page unavailable");
    await expect(page.locator("#ql-results")).toBeFocused();
    await expect(page.locator("#ql-results")).toHaveAttribute("aria-live", "polite");
  });
}

test("mapping controls have names and reserved source errors identify the row", async ({ page }) => {
  await openOperation(page, operations.find((op) => op.name === "ingest"));
  const source = page.getByRole("textbox", { name: "source_id name", exact: true });
  await expect(page.getByRole("combobox", { name: "source_id type", exact: true })).toHaveCount(1);
  await expect(page.getByRole("textbox", { name: "id name", exact: true })).toHaveCount(1);
  await source.fill("text");
  await expect(source).toHaveAttribute("aria-invalid", "true");
  await expect(source).toHaveAccessibleDescription(/reserved/i);
  await source.fill("source_id");
  await expect(source).not.toHaveAttribute("aria-invalid", "true");
});
