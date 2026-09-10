import { fixtureName } from "./harness.mjs";
// §10 Serie N — navigation, hash and keyboard (agent E scope).
import { test, expect } from "./fixtures.js";
import { boot, prepareConsole, BASE_URL, watchConsole } from "./helpers.js";

const TABS = ["query", "collections", "points", "monitoring"];
const ACTIVE = fixtureName("e2e_e_mon");

async function selectedTab(page) {
  return page.evaluate(() => {
    const sel = [...document.querySelectorAll('.tabs [role="tab"]')]
      .filter((t) => t.getAttribute("aria-selected") === "true");
    return { count: sel.length, id: sel[0] ? sel[0].id : null };
  });
}

test("N0 — product brand and title fit a 320px viewport", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 800 });
  await boot(page, { active: ACTIVE });

  await expect(page).toHaveTitle("Smart Semantic Searcher");
  await expect(page.locator(".brand")).toHaveText("SMART SEMANTIC SEARCHER");
  const bounds = await page.locator(".brand").evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return { left: rect.left, right: rect.right, viewport: window.innerWidth };
  });
  expect(bounds.left).toBeGreaterThanOrEqual(0);
  expect(bounds.right).toBeLessThanOrEqual(bounds.viewport);
});

test.describe("N — navigation / hash / keyboard", () => {
  test("N1 — clicking the 4 tabs shows one panel, hides the others, single aria-selected", async ({ page }) => {
    const errors = await boot(page, { active: ACTIVE });
    for (const tab of TABS) {
      await page.locator(`#tab-${tab}`).click();
      await expect(page.locator(`#panel-${tab}`)).toBeVisible();
      for (const other of TABS.filter((t) => t !== tab)) {
        await expect(page.locator(`#panel-${other}`)).toBeHidden();
      }
      const sel = await selectedTab(page);
      expect(sel).toEqual({ count: 1, id: `tab-${tab}` });
    }
    expect(errors, `console errors: ${errors.join(" | ")}`).toEqual([]);
  });

  test("N2 — deep links #query #collections #points #monitoring activate the tab at load", async ({ page }) => {
    for (const tab of TABS) {
      await page.addInitScript((t) => {
        sessionStorage.setItem("qhm_api_key", t);
      }, process.env.QHM_E2E_TOKEN || process.env.QHM_TOKEN || "");
      await page.goto(`${BASE_URL}/#${tab}`);
      await expect(page.locator(`#panel-${tab}`)).toBeVisible();
      expect(await selectedTab(page)).toEqual({ count: 1, id: `tab-${tab}` });
    }
  });

  test("N3 — unknown hash (#pippo) and empty hash fall back to query", async ({ page }) => {
    await prepareConsole(page, { active: ACTIVE });
    await page.goto(`${BASE_URL}/#pippo`);
    await expect(page.locator("#panel-query")).toBeVisible();
    expect(await selectedTab(page)).toEqual({ count: 1, id: "tab-query" });
    // activateTab normalises the hash back to the fallback tab.
    expect(await page.evaluate(() => window.location.hash)).toBe("#query");

    await page.goto(`${BASE_URL}/`);
    await expect(page.locator("#panel-query")).toBeVisible();
    expect(await selectedTab(page)).toEqual({ count: 1, id: "tab-query" });
  });

  test("N4 — tab changes use history.replaceState (Back does not walk the tabs)", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    const before = await page.evaluate(() => history.length);
    for (const tab of ["collections", "points", "monitoring", "query", "points"]) {
      await page.locator(`#tab-${tab}`).click();
      await expect(page.locator(`#panel-${tab}`)).toBeVisible();
      expect(await page.evaluate(() => window.location.hash)).toBe(`#${tab}`);
    }
    const after = await page.evaluate(() => history.length);
    expect(after, "history entries added by tab switching").toBe(before);
    expect(await page.evaluate(() => history.state && history.state.tab)).toBe("points");
  });

  test("N5 — manual hash edit fires hashchange and activates the tab", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    for (const tab of ["monitoring", "collections", "points", "query"]) {
      await page.evaluate((t) => { window.location.hash = `#${t}`; }, tab);
      await expect(page.locator(`#panel-${tab}`)).toBeVisible();
      expect(await selectedTab(page)).toEqual({ count: 1, id: `tab-${tab}` });
    }
  });

  test("N6 — ArrowRight / ArrowLeft move and activate the adjacent tab, with wrap", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    await page.locator("#tab-query").focus();
    // Forward through all four and wrap back to query.
    for (const expected of ["collections", "points", "monitoring", "query"]) {
      await page.keyboard.press("ArrowRight");
      await expect(page.locator(`#panel-${expected}`)).toBeVisible();
      expect(await selectedTab(page)).toEqual({ count: 1, id: `tab-${expected}` });
      expect(await page.evaluate(() => document.activeElement.id)).toBe(`tab-${expected}`);
    }
    // Backward wrap: query → monitoring.
    for (const expected of ["monitoring", "points", "collections", "query"]) {
      await page.keyboard.press("ArrowLeft");
      await expect(page.locator(`#panel-${expected}`)).toBeVisible();
      expect(await selectedTab(page)).toEqual({ count: 1, id: `tab-${expected}` });
    }
  });

  test("N7 — Home / End jump to the first / last tab", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    await page.locator("#tab-points").focus();
    await page.keyboard.press("End");
    await expect(page.locator("#panel-monitoring")).toBeVisible();
    expect(await page.evaluate(() => document.activeElement.id)).toBe("tab-monitoring");
    await page.keyboard.press("Home");
    await expect(page.locator("#panel-query")).toBeVisible();
    expect(await page.evaluate(() => document.activeElement.id)).toBe("tab-query");
  });

  test("N8 — roving tabindex: only the selected tab has tabIndex 0; Tab exits the tablist", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    const readIdx = () => page.evaluate(() =>
      [...document.querySelectorAll('.tabs [role="tab"]')].map((t) => [t.id, t.tabIndex]));
    expect(await readIdx()).toEqual([
      ["tab-query", 0], ["tab-collections", -1], ["tab-points", -1], ["tab-monitoring", -1],
    ]);
    await page.locator("#tab-points").click();
    expect(await readIdx()).toEqual([
      ["tab-query", -1], ["tab-collections", -1], ["tab-points", 0], ["tab-monitoring", -1],
    ]);
    // Tab from the selected tab must leave the tablist, not visit the siblings.
    await page.locator("#tab-points").focus();
    await page.keyboard.press("Tab");
    const next = await page.evaluate(() => ({
      id: document.activeElement.id,
      inTablist: Boolean(document.activeElement.closest('[role="tablist"]')),
    }));
    expect(next.inTablist, `Tab landed on ${next.id} inside the tablist`).toBe(false);
  });

  test("N9 — the API key button is outside the tablist and unreachable by Arrow", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    expect(await page.evaluate(() =>
      Boolean(document.getElementById("apikey-btn").closest('[role="tablist"]')))).toBe(false);
    await page.locator("#tab-monitoring").click();
    await page.locator("#tab-monitoring").focus();
    await page.keyboard.press("ArrowRight");
    expect(await page.evaluate(() => document.activeElement.id)).toBe("tab-query");
    await page.keyboard.press("ArrowLeft");
    expect(await page.evaluate(() => document.activeElement.id)).toBe("tab-monitoring");
  });

  test("N10 — entering / leaving Monitoring by keyboard starts and stops polling", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    let statsHits = 0;
    page.on("request", (r) => { if (r.url().endsWith("/stats")) statsHits += 1; });

    // Enter monitoring via End (keyboard only).
    await page.locator("#tab-query").focus();
    await page.keyboard.press("End");
    await expect(page.locator("#panel-monitoring")).toBeVisible();
    await page.waitForTimeout(11_000);
    const whileOpen = statsHits;
    expect(whileOpen, "polling must start on keyboard entry").toBeGreaterThanOrEqual(2);

    // Leave via Home (keyboard only) — polling must stop.
    await page.keyboard.press("Home");
    await expect(page.locator("#panel-query")).toBeVisible();
    const atExit = statsHits;
    await page.waitForTimeout(12_000);
    expect(statsHits - atExit, "ticks after leaving Monitoring by keyboard").toBe(0);
  });

  test("N11 — Points subtabs switch panel with coherent aria-selected", async ({ page }) => {
    const errors = await boot(page, { active: ACTIVE });
    await page.locator("#tab-points").click();
    const names = ["ingest", "upsert", "batch", "get", "delete", "payload"];
    for (const name of names) {
      await page.locator(`[data-pt-tab="${name}"]`).click();
      await expect(page.locator(`#pt-${name}`)).toBeVisible();
      for (const other of names.filter((n) => n !== name)) {
        await expect(page.locator(`#pt-${other}`)).toBeHidden();
      }
      const sel = await page.evaluate(() => {
        const s = [...document.querySelectorAll("[data-pt-tab]")]
          .filter((t) => t.getAttribute("aria-selected") === "true");
        return { count: s.length, name: s[0] ? s[0].dataset.ptTab : null };
      });
      expect(sel).toEqual({ count: 1, name });
    }

    // Every subtab must name its panel and every panel must name its subtab back:
    // data-pt-tab drives the JS, this pair is what maps tab → panel for a reader.
    const pairs = [
      ["subtab-ingest", "pt-ingest"],
      ["subtab-upsert", "pt-upsert"],
      ["subtab-batch", "pt-batch"],
      ["subtab-get", "pt-get"],
      ["subtab-delete", "pt-delete"],
      ["subtab-payload", "pt-payload"],
      ["subtab-count", "pt-count"],
    ];
    for (const [tabId, panelId] of pairs) {
      await expect(page.locator(`#${tabId}`)).toHaveAttribute("aria-controls", panelId);
      await expect(page.locator(`#${panelId}`)).toHaveAttribute("aria-labelledby", tabId);
    }

    expect(errors, `console errors: ${errors.join(" | ")}`).toEqual([]);
  });
});
