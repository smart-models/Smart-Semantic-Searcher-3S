import { fulfillResponse, transformJsonResponse } from "./telemetry.mjs";
import { fixtureName } from "./harness.mjs";
// Test book §2 — Serie S (service readiness / outage gate).
// S2/S5/S6 are driven by a REAL outage: the runner stops the embedder / Qdrant
// container between phases. Set QHM_REAL_OUTAGE=1 only when no other agent runs.
import { expect, test } from "./fixtures.js";
import { boot, prepareConsole, BASE_URL } from "./helpers.js";

const COLL = fixtureName("e2e_z_basic");
test.beforeEach(async ({ page }) => {
  await prepareConsole(page, { active: COLL });
});

const PANELS = ["#panel-query", "#panel-collections", "#panel-points"];

test("S1 — healthy stack leaves every panel operational", async ({ page }) => {
  const errors = await boot(page, { active: COLL });
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
  await expect(page.locator("#conn-health-detail")).toContainText("qdrant reachable");
  await expect(page.locator("#conn-health-detail")).toContainText("embedder reachable");
  for (const p of PANELS) await expect(page.locator(p)).not.toHaveAttribute("inert", "");
  await expect(page.locator("#service-gate")).toBeEmpty();
  expect(errors).toEqual([]);
});

// S2 with a simulated 503 (the real-outage variant is driven by the runner).
test("S2 — embedder unreachable locks every operational panel", async ({ page }) => {
  await page.route("**/health", (route) => fulfillResponse(route, {
    status: 503, contentType: "application/json",
    body: JSON.stringify({
      status: "unhealthy", qdrant: "reachable", embedder: "unreachable",
      detail: "Smart Embedder unreachable.",
    }),
  }));
  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-health-status")).toHaveText("UNHEALTHY");
  await expect(page.locator("#conn-health-detail")).toContainText("embedder unreachable");
  await expect(page.locator("#service-gate")).toContainText("Smart Embedder unreachable");
  for (const p of PANELS) await expect(page.locator(p)).toHaveAttribute("inert", "");
  await expect(page.locator("#conn-collection")).toBeDisabled();
  await expect(page.locator("#ql-run")).toBeDisabled();
});

test("S3 — outage banner wins over an incompatible-collection banner", async ({ page }) => {
  await page.route("**/health", (route) => fulfillResponse(route, {
    status: 503, contentType: "application/json",
    body: JSON.stringify({
      status: "unhealthy", qdrant: "reachable", embedder: "unreachable",
      detail: "Smart Embedder unreachable.",
    }),
  }));
  await page.route(/\/collections\/[^/?]+$/, async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return transformJsonResponse(route, (json) => ({
      ...json, embedding_compatibility: {
      status: "incompatible",
      stored: { profile: "BASIC", fingerprint: "sha256:a" },
      active: { profile: "BASIC", fingerprint: "sha256:b" },
      message: "different vector spaces",
    },
    }));
  });
  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#service-gate")).toContainText("Smart Embedder unreachable");
  await expect(page.locator("#service-gate")).not.toContainText("Re-embed its vectors");
  await expect(page.locator("#service-gate")).not.toContainText("compatible collection");
});

test("S4 — recovery via Re-test restores every panel", async ({ page }) => {
  let down = true;
  await page.route("**/health", (route) => {
    if (down) {
      return fulfillResponse(route, {
        status: 503, contentType: "application/json",
        body: JSON.stringify({
          status: "unhealthy", qdrant: "reachable", embedder: "unreachable",
          detail: "Smart Embedder unreachable.",
        }),
      });
    }
    return route.fallback();
  });
  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-health-status")).toHaveText("UNHEALTHY");
  down = false;
  await page.locator("#conn-retest").click();
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY", { timeout: 30_000 });
  for (const p of PANELS) await expect(page.locator(p)).not.toHaveAttribute("inert", "");
  await expect(page.locator("#service-gate")).toBeEmpty();
});

test("S5 — qdrant unreachable produces the same lockdown", async ({ page }) => {
  await page.route("**/health", (route) => fulfillResponse(route, {
    status: 503, contentType: "application/json",
    body: JSON.stringify({
      status: "unhealthy", qdrant: "unreachable", embedder: "reachable",
      detail: "Qdrant unreachable.",
    }),
  }));
  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-health-detail")).toContainText("qdrant unreachable");
  for (const p of PANELS) await expect(page.locator(p)).toHaveAttribute("inert", "");
});

// S6 — the reconciliation case: outage unlocks the collection sub-wraps, so on
// recovery with an INCOMPATIBLE active collection they must be locked again.
test("S6 — post-recovery reconciliation re-locks the collection sub-wraps", async ({ page }) => {
  let down = true;
  await page.route("**/health", (route) => {
    if (down) {
      return fulfillResponse(route, {
        status: 503, contentType: "application/json",
        body: JSON.stringify({
          status: "unhealthy", qdrant: "reachable", embedder: "unreachable",
          detail: "Smart Embedder unreachable.",
        }),
      });
    }
    return route.fallback();
  });
  await page.route(/\/collections\/[^/?]+$/, async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return transformJsonResponse(route, (json) => ({
      ...json, embedding_compatibility: {
      status: "incompatible",
      stored: { profile: "BASIC", fingerprint: "sha256:a" },
      active: { profile: "BASIC", fingerprint: "sha256:b" },
      message: "different vector spaces",
    },
    }));
  });
  await page.goto(`${BASE_URL}/#collections`);
  await expect(page.locator("#conn-health-status")).toHaveText("UNHEALTHY");
  down = false;
  await page.locator("#conn-retest").click();
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY", { timeout: 30_000 });
  await expect(page.locator("#panel-collections")).not.toHaveAttribute("inert", "");
  for (const id of ["#cc-recreate-control", "#cu-wrap", "#pi-wrap"]) {
    await expect(page.locator(id)).toHaveAttribute("inert", "");
  }
  // 07 Delete works on a typed name, so an incompatible active
  // collection never locks it.
  await expect(page.locator("#cd-wrap")).not.toHaveAttribute("inert", "");
  // 03 keeps its create half live: a new collection is built from the ACTIVE
  // contract, so it cannot inherit the mismatch.
  await expect(page.locator("#cc-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cr-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#ce-wrap")).not.toHaveAttribute("inert", "");
});

test("S7 — health is re-polled automatically without pressing Re-test", async ({ page }) => {
  let hits = 0;
  await page.route("**/health", (route) => { hits += 1; return route.fallback(); });
  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY", { timeout: 30_000 });
  const first = hits;
  // The console polls health on a 30s timer; allow one tick plus slack.
  await page.waitForTimeout(35_000);
  expect(hits).toBeGreaterThan(first);
});
