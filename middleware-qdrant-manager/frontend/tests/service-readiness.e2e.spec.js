import { fulfillResponse } from "./telemetry.mjs";
import { expect, test } from "./fixtures.js";

import { BASE_URL, prepareConsole } from "./helpers.js";

test("service outage disables every operational panel until health recovers", async ({
  page,
}) => {
  await prepareConsole(page);
  let healthy = false;
  await page.route("**/health", async (route) => {
    const body = healthy
      ? {
          status: "healthy",
          qdrant: "reachable",
          embedder: "reachable",
          collections: 0,
        }
      : {
          status: "unhealthy",
          qdrant: "reachable",
          embedder: "unreachable",
          detail: "Smart Embedder unreachable.",
        };
    await fulfillResponse(route, {
      status: healthy ? 200 : 503,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });

  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-health-status")).toHaveText("UNHEALTHY");
  await expect(page.locator("#conn-health-detail")).toContainText(
    "qdrant reachable · embedder unreachable",
  );
  await expect(page.locator("#service-gate")).toContainText(
    "Smart Embedder unreachable",
  );
  // Precedence: while services are down, the outage banner wins even if a report
  // would otherwise report the collection as incompatible.
  await expect(page.locator("#service-gate")).not.toContainText(
    "Switch to a compatible collection",
  );
  await expect(page.locator("#service-gate")).not.toContainText(
    "Re-embed its vectors",
  );
  for (const selector of [
    "#panel-query",
    "#panel-collections",
    "#panel-points",
  ]) {
    await expect(page.locator(selector)).toHaveAttribute("inert", "");
  }
  await expect(page.locator("#conn-collection")).toBeDisabled();
  await expect(page.locator("#ql-run")).toBeDisabled();

  healthy = true;
  await page.locator("#conn-retest").click();

  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
  for (const selector of [
    "#panel-query",
    "#panel-collections",
    "#panel-points",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("inert", "");
  }
});
