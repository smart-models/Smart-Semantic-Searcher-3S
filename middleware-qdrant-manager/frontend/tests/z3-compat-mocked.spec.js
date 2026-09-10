import { fulfillResponse, transformJsonResponse } from "./telemetry.mjs";
import { fixtureName } from "./harness.mjs";
// Test book §4 — Serie E, the cases the book allows to drive with a mocked
// report ([API] mock): E5 (same-profile drift), E6 (foreign collection),
// E9/E10 (guard suppression and loading transient).
import { expect, test } from "./fixtures.js";
import { BASE_URL, TOKEN } from "./helpers.js";

const COLL = fixtureName("e2e_z_basic");

async function mountReport(page, compat) {
  await page.route(/\/collections\/[^/?]+$/, async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return transformJsonResponse(route, (json) => ({
      ...json, embedding_compatibility: compat,
    }));
  });
}

async function bootWith(page, compat) {
  await page.addInitScript(({ t, c }) => {
    sessionStorage.setItem("qhm_api_key", t);
    localStorage.setItem("qhm_active_collection", c);
  }, { t: TOKEN, c: COLL });
  await mountReport(page, compat);
  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY", { timeout: 30_000 });
}

const DRIFT = {
  status: "incompatible",
  stored: { profile: "BASIC", fingerprint: "sha256:stored" },
  active: { profile: "BASIC", fingerprint: "sha256:active" },
  message: "Active embedder and collection use different vector spaces.",
};
const PROFILE_SWITCH = {
  status: "incompatible",
  stored: { profile: "BASIC", fingerprint: "sha256:stored" },
  active: { profile: "ADVANCED", fingerprint: "sha256:active" },
  message: "Active embedder and collection use different vector spaces.",
};
const FOREIGN = {
  status: "incompatible",
  stored: null,
  active: { profile: "BASIC", fingerprint: "sha256:active" },
  message: "Collection has no verified embedding contract.",
};

test("E5 — same-profile fingerprint drift SHOWS the recoverable banner", async ({ page }) => {
  await bootWith(page, DRIFT);
  await expect(page.locator("#service-gate")).toContainText("different vector space");
  await expect(page.locator("#service-gate")).toContainText("Re-embed its vectors");
  await expect(page.locator("#panel-query")).toHaveAttribute("inert", "");
  await expect(page.locator("#panel-query")).toHaveClass(/operations-disabled/);
  await page.locator("#tab-points").click();
  await expect(page.locator("#panel-points")).not.toHaveAttribute("inert", "");
  await page.locator('[data-pt-tab="upsert"]').click();
  await page.locator("#up-id").fill("1");
  await page.locator("#up-text").fill("blocked mutation");
  await expect(page.locator("#up-submit")).toBeDisabled();
  await page.locator('[data-pt-tab="get"]').click();
  await page.locator("#get-id").fill("1");
  await expect(page.locator("#get-submit")).toBeEnabled();
  await page.locator("#tab-collections").click();
  await expect(page.locator("#cr-wrap")).not.toHaveAttribute("inert", "");
  for (const id of ["#cc-recreate-control", "#cu-wrap", "#pi-wrap"]) {
    await expect(page.locator(id)).toHaveAttribute("inert", "");
  }
  // 07 Delete works on a typed name, so an incompatible active
  // collection never locks it.
  await expect(page.locator("#cd-wrap")).not.toHaveAttribute("inert", "");
  // §1.1: 03 is partially locked — recreate goes, creating a new collection stays.
  await expect(page.locator("#cc-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-name")).toBeEnabled();
  // §1.1: Export is deliberately outside the lockdown list.
  await expect(page.locator("#ce-wrap")).not.toHaveAttribute("inert", "");
});

test("E2/E4-mock — BASIC<->ADVANCED profile switch SUPPRESSES the banner", async ({ page }) => {
  await bootWith(page, PROFILE_SWITCH);
  await expect(page.locator("#service-gate")).toBeEmpty();
  await expect(page.locator("#panel-query")).toHaveAttribute("inert", "");
  await page.locator("#tab-collections").click();
  await expect(page.locator("#cc-report")).toContainText(/INCOMPATIBLE/i);
  await expect(page.locator("#cr-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#ce-wrap")).not.toHaveAttribute("inert", "");
});

test("E6 — foreign collection (no stored contract) also locks Re-embed", async ({ page }) => {
  await bootWith(page, FOREIGN);
  await expect(page.locator("#service-gate")).toContainText("no verified embedding contract");
  await expect(page.locator("#panel-query")).toHaveAttribute("inert", "");
  await page.locator("#tab-collections").click();
  await expect(page.locator("#cr-wrap")).toHaveAttribute("inert", "");
});

test("E9 — inline guards are emptied while the banner owns the reason", async ({ page }) => {
  await bootWith(page, DRIFT);
  await expect(page.locator("#ql-embedding-guard")).toBeEmpty();
  await expect(page.locator("#ql-query-mode")).toBeHidden();
  await page.locator("#tab-points").click();
  await expect(page.locator("#pt-embedding-guard")).toBeEmpty();
});

test("E10 — compatibility check shows a loading transient", async ({ page }) => {
  await page.addInitScript(({ t, c }) => {
    sessionStorage.setItem("qhm_api_key", t);
    localStorage.setItem("qhm_active_collection", c);
  }, { t: TOKEN, c: COLL });
  // Delay the report so the loading state is observable.
  await page.route(/\/collections\/[^/?]+$/, async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    await new Promise((r) => setTimeout(r, 2500));
    await route.fallback();
  });
  await page.goto(`${BASE_URL}/#points`);
  await expect(page.locator("#pt-embedding-guard")).toContainText(
    "Checking collection compatibility", { timeout: 10_000 });
});

test("E8-mock — switching from incompatible to compatible collection unlocks the panels", async ({ page }) => {
  // Mount DRIFT only for the incompatible collection; other collections pass through.
  const driftOnly = async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return transformJsonResponse(route, (json) => ({
      ...json,
      ...(json.collection_name === COLL ? { embedding_compatibility: DRIFT } : {}),
    }));
  };
  await page.addInitScript(({ t, c }) => {
    sessionStorage.setItem("qhm_api_key", t);
    localStorage.setItem("qhm_active_collection", c);
  }, { t: TOKEN, c: COLL });
  await page.route(/\/collections\/[^/?]+$/, driftOnly);
  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY", { timeout: 30_000 });

  // Verify lockdown on the incompatible collection
  await expect(page.locator("#panel-query")).toHaveAttribute("inert", "");
  await expect(page.locator("#conn-collection")).toBeEnabled();

  // Switch to e2e_a_keep (compatible, real report passes through)
  await page.locator("#conn-collection").selectOption(fixtureName("e2e_a_keep"));
  // After switching to a compatible collection, panels must unlock
  await expect(page.locator("#panel-query")).not.toHaveAttribute("inert", "", { timeout: 20_000 });
  await page.locator("#tab-collections").click();
  await expect(page.locator("#panel-collections")).not.toHaveAttribute("inert", "", { timeout: 20_000 });
  await page.locator("#tab-points").click();
  await expect(page.locator("#panel-points")).not.toHaveAttribute("inert", "", { timeout: 20_000 });
});
