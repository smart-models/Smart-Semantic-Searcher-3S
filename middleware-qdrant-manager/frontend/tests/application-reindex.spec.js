import { fulfillResponse } from "./telemetry.mjs";
import { test, expect } from "./fixtures.js";
import { A, B, compatible, openConsole, openPoints } from "./application-fixtures.js";

for (const status of ["running", "paused", "interrupted", "failed", "finalizing"]) {
  test(`collection mutations are guarded during ${status}`, async ({ page }) => {
    await openConsole(page, { tab: "collections", job: { status, processed_points: 64, total_points: 512 } });
    await page.locator("#pi-field").fill("category");
    await page.locator("#pi-del-field").fill("category");
    for (const id of ["cu-submit", "pi-create", "pi-delete"]) await expect(page.locator(`#${id}`)).toBeDisabled();
    await page.locator("#cc-name").fill(A);
    await page.locator("#cc-recreate").check();
    await page.locator("#cc-confirm").fill(A);
    await expect(page.locator("#cc-submit")).toBeDisabled();
    await page.locator("#cd-name").fill(A);
    await page.locator("#cd-confirm").fill(A);
    await expect(page.locator("#cd-submit")).toBeDisabled();
    await page.locator("#cd-name").fill(B);
    await page.locator("#cd-confirm").fill(B);
    await expect(page.locator("#cd-submit")).toBeEnabled();
    await page.locator("#cc-recreate").uncheck();
    await page.locator("#cc-name").fill("e2e_application_new");
    await expect(page.locator("#cc-submit")).toBeEnabled();
  });
}

test("paused profile migration allows point reads", async ({ page }) => {
  const mock = await openConsole(page, { compatibility: {
    ...compatible, status: "incompatible", active: { profile: "advanced", fingerprint: "next" },
  }, job: { status: "paused", processed_points: 64, total_points: 512 } });
  await openPoints(page, "get");
  await expect(page.locator("#panel-points")).not.toHaveAttribute("inert", "");
  await page.locator("#get-id").fill("7");
  await page.locator("#get-ids").fill("7,8");
  await expect(page.locator("#get-submit")).toBeEnabled();
  await expect(page.locator("#get-many-submit")).toBeEnabled();
  mock.intercept = async (route, req) => {
    if (req.path === "/points/7") { await fulfillResponse(route, { json: { id: "7", collection_name: A, payload: { text: "retained" }, vectors: null } }); return true; }
    return false;
  };
  await page.locator("#get-submit").click();
  await expect(page.locator("#get-result")).toContainText("retained");
  await openPoints(page, "upsert");
  await page.locator("#up-id").fill("7");
  await page.locator("#up-text").fill("text");
  await expect(page.locator("#up-submit")).toBeDisabled();
  expect(mock.unknown).toEqual([]);
});

test("vector options follow metadata cutover and retain an invalid old filter", async ({ page }) => {
  const mock = await openConsole(page, { job: { job_id: "cutover", status: "running", total_points: 10, processed_points: 1 } });
  await page.locator("#ql-filter-add").click();
  await page.locator('[data-role="field"]').selectOption("(vector presence)");
  const vector = page.locator('[data-role="value"]');
  await expect(vector).toHaveValue("dense");
  mock.compatibility.active_vectors = { dense: "dense_g2", sparse: "sparse_g2" };
  mock.job = { job_id: "cutover", status: "completed", total_points: 10, processed_points: 10 };
  await page.clock.runFor(1000);
  await expect(vector.locator("option")).toContainText(["Select a current vector", "dense_g2", "sparse_g2"]);
  await expect(page.locator("#ql-errors")).toContainText("unavailable");
  await expect(page.locator("#ql-run")).toBeDisabled();
  await vector.selectOption("dense_g2");
  await expect(page.locator("#ql-run")).toBeEnabled();
  expect(mock.errors).toEqual([]);
});
