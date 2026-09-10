// Test book §2 — S2/S5 driven by a REAL outage: the runner stops the Smart
// Embedder container before this file runs, so nothing is mocked here.
import { expect, test } from "./fixtures.js";
import { BASE_URL, TOKEN } from "./helpers.js";
import { expectHttpFailure } from "./telemetry.mjs";

const PANELS = ["#panel-query", "#panel-collections", "#panel-points"];

test.beforeEach(async ({ page, request }) => {
  await expect.poll(async () => (await request.get(`${BASE_URL}/readyz`)).status(),
    { timeout: 60_000 }).toBe(503);
  expect((await request.get(`${BASE_URL}/livez`)).status()).toBe(200);
  for (const path of ["/health", "/readyz", "/collections"]) {
    expectHttpFailure(page, "GET", path, 503);
  }
});

test("S2-real — with the embedder container stopped the console locks every panel", {
  // TB-PRO-003 also demands livez/readyz inspection, recovery and a real
  // Qdrant outage; the stopped-embedder console lockdown is one part of it.
  annotation: { type: "case-partial", description: "TB-PRO-003" },
}, async ({ page }) => {
  await page.addInitScript((t) => sessionStorage.setItem("qhm_api_key", t), TOKEN);
  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-health-status")).toHaveText("UNHEALTHY", { timeout: 60_000 });
  await expect(page.locator("#conn-health-detail")).toContainText("embedder unreachable");
  await expect(page.locator("#service-gate")).toContainText(/unreachable/i);
  for (const p of PANELS) await expect(page.locator(p)).toHaveAttribute("inert", "");
  await expect(page.locator("#conn-collection")).toBeDisabled();
  await expect(page.locator("#ql-run")).toBeDisabled();
});

test("S2-real — Monitoring degrades without crashing while the embedder is down", async ({ page }) => {
  await page.addInitScript((t) => sessionStorage.setItem("qhm_api_key", t), TOKEN);
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`${BASE_URL}/#monitoring`);
  await expect(page.locator("#mon-badges")).not.toBeEmpty({ timeout: 60_000 });
  await expect(page.locator("#mon-models .alert-info")).toContainText(/unreachable/i);
  await page.waitForTimeout(6000);
  expect(errors).toEqual([]);
});
