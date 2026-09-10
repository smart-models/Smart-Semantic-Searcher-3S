import { fixtureName } from "./harness.mjs";
import { expectHttpFailure } from "./telemetry.mjs";
// §3 Serie H — connection bar & API key popover (agent E scope).
import { test, expect, request as pwRequest } from "./fixtures.js";
import { boot, BASE_URL, TOKEN, setupFetch } from "./helpers.js";

const ACTIVE = fixtureName("e2e_e_mon");
const ALT = fixtureName("e2e_e_alt");
const WRONG_KEY = "definitely-not-the-right-key-0000";

test.beforeAll(async () => {
  // A second sandbox collection so H2 can switch without touching other agents'.
  const ctx = await pwRequest.newContext();
  try {
    const list = await setupFetch(ctx, "GET", "/collections");
    const names = (await list.json()).collections;
    expect(Array.isArray(names)).toBe(true);
    if (!names.includes(ALT)) await setupFetch(ctx, "POST", "/collections", {
      collection_name: ALT,
      dense_on_disk: true,
      sparse_on_disk: true,
      payload_on_disk: true,
      quantization: true,
      sparse_idf: true,
    });
  } finally { await ctx.dispose(); }
});

async function openPopover(page) {
  await page.locator("#apikey-btn").click();
  await expect(page.locator("#apikey-popover")).toBeVisible();
}

async function enterRegisteredKey(page) {
  // Keep a real credential out of Playwright's value-bearing fill/error log.
  await page.locator('#apikey-input').evaluate((input, value) => {
    input.value = value;
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }, TOKEN);
}

function expectRejectedCredentials(page) {
  for (const path of ['/collections', `/collections/${ACTIVE}`,
    `/collections/${ACTIVE}/payload-keys`, `/collections/${ACTIVE}/payload-indexes`,
    `/collections/${ACTIVE}/reindex`]) expectHttpFailure(page, 'GET', path, 401);
}

test.describe("H — connection bar & API key", () => {
  test.beforeEach(async ({ page }, info) => {
    if (/^H(?:5|6|7|8|9|10) —/.test(info.title)) {
      expectRejectedCredentials(page);
    }
  });
  test("H1 — active collection dropdown populated, sorted, persisted in localStorage", async ({ page }) => {
    const errors = await boot(page, { active: ACTIVE });
    await expect(page.locator("#conn-collection")).toBeEnabled();
    const options = await page.locator("#conn-collection option").allTextContents();
    expect(options.length).toBeGreaterThan(1);
    const sorted = [...options].sort((a, b) =>
      a.toLowerCase() < b.toLowerCase() ? -1 : a.toLowerCase() > b.toLowerCase() ? 1 : 0);
    expect(options, "dropdown order").toEqual(sorted);
    expect(options).toContain(ACTIVE);
    await expect(page.locator("#conn-collection")).toHaveValue(ACTIVE);
    expect(await page.evaluate(() => localStorage.getItem("qhm_active_collection"))).toBe(ACTIVE);

    // Selection persists across a reload without the initScript seeding it.
    await page.locator("#conn-collection").selectOption(ALT);
    await expect(page.locator("#ql-collection")).toHaveValue(ALT);
    expect(await page.evaluate(() => localStorage.getItem("qhm_active_collection"))).toBe(ALT);
    expect(errors, `console errors: ${errors.join(" | ")}`).toEqual([]);
  });

  test("H2 — changing active collection clears scoped result areas and the filter tree", async ({ page }) => {
    expectHttpFailure(page, "GET", "/points/999999999", 404);
    await boot(page, { active: ACTIVE });
    // Populate a scoped result area with a read-only Get on a missing id.
    await page.locator("#tab-points").click();
    await page.locator('[data-pt-tab="get"]').click();
    await page.locator("#get-id").fill("999999999");
    await page.locator("#get-submit").click();
    await expect(page.locator("#get-result")).not.toBeEmpty();

    // Build a filter row on the Query tab.
    await page.locator("#tab-query").click();
    await page.locator("#ql-filter-add").click();
    const rowsBefore = await page.locator("#ql-filter-rows .filter-row").count();
    expect(rowsBefore).toBeGreaterThan(0);

    await page.locator("#conn-collection").selectOption(ALT);
    await expect(page.locator("#ql-collection")).toHaveValue(ALT);
    await expect(page.locator("#get-result")).toBeEmpty();
    await expect(page.locator("#delf-count")).toBeHidden();
    expect(await page.locator("#ql-filter-rows .filter-row").count()).toBe(0);
    for (const id of ["cr-result", "cr-preflight-result", "cu-result",
      "pi-create-result", "pi-del-result", "up-result", "bt-result",
      "get-many-result", "del-result", "del-bulk-result", "delf-result",
      "pl-result", "pl-keys-result", "pl-clear-result", "ing-result"]) {
      await expect(page.locator(`#${id}`), id).toBeEmpty();
    }
  });

  test("H3 — clicking API key opens the popover: aria-expanded true, input focused and empty", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    await expect(page.locator("#apikey-btn")).toHaveAttribute("aria-expanded", "false");
    await openPopover(page);
    await expect(page.locator("#apikey-btn")).toHaveAttribute("aria-expanded", "true");
    await expect(page.locator("#apikey-input")).toBeFocused();
    await expect(page.locator("#apikey-input")).toHaveValue("");
    // With a key stored the placeholder is masked, never the key itself.
    await expect(page.locator("#apikey-input")).toHaveAttribute("placeholder", "••••••••");
  });

  test("H4 — empty submit shows ENTER A KEY FIRST, keeps the popover open, refocuses the input", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    await openPopover(page);
    await page.locator("#apikey-input").fill("");
    await page.locator("#apikey-save").click();
    await expect(page.locator("#apikey-status")).toHaveText("ENTER A KEY FIRST");
    await expect(page.locator("#apikey-status")).toHaveClass(/error/);
    await expect(page.locator("#apikey-popover")).toBeVisible();
    await expect(page.locator("#apikey-input")).toBeFocused();
    // Still open a second later: no auto-close on the error path.
    await page.waitForTimeout(1200);
    await expect(page.locator("#apikey-popover")).toBeVisible();
  });

  test("H5 — Register a valid key: REGISTERED, Clear visible, stored, auto-close after ~800ms", async ({ page }) => {
    await boot(page, { active: ACTIVE, token: "" });
    await openPopover(page);
    await enterRegisteredKey(page);
    await page.locator("#apikey-save").click();
    await expect(page.locator("#apikey-status")).toHaveText("API KEY REGISTERED");
    await expect(page.locator("#apikey-status")).toHaveClass(/ok/);
    await expect(page.locator("#apikey-clear")).toBeVisible();
    expect(await page.evaluate((value) => sessionStorage.getItem("qhm_api_key") === value, TOKEN)).toBe(true);
    // Transition: still open shortly after, closed after the 800ms timer.
    await page.waitForTimeout(300);
    await expect(page.locator("#apikey-popover")).toBeVisible();
    await page.waitForTimeout(900);
    await expect(page.locator("#apikey-popover")).toBeHidden();
    await expect(page.locator("#apikey-btn")).toHaveAttribute("aria-expanded", "false");
  });

  test("H6 — Register reloads protected data without a page reload", async ({ page }) => {
    await boot(page, { active: ACTIVE, token: "" });
    await expect(page.locator("#conn-collection-hint")).toHaveText("API KEY MISSING — OPEN API KEY AND REGISTER A KEY");
    // Marker survives only if the document is never reloaded.
    await page.evaluate(() => { window.__noReload = true; });
    const hits = [];
    page.on("request", (r) => {
      const u = new URL(r.url()).pathname;
      if (["/health", "/collections"].includes(u)) hits.push(u);
    });
    await openPopover(page);
    await enterRegisteredKey(page);
    await page.locator("#apikey-save").click();
    await expect(page.locator("#conn-collection-hint")).toBeHidden();
    await expect(page.locator("#conn-collection")).toHaveValue(ACTIVE);
    await expect(page.locator("#cc-report")).not.toBeEmpty();
    expect(await page.evaluate(() => window.__noReload === true), "page was reloaded").toBe(true);
    expect(hits, `re-probed endpoints: ${hits.join(",")}`).toEqual(
      expect.arrayContaining(["/health", "/collections"]));
  });

  test("H7 — no key: hint shows API KEY MISSING and the select stays empty", async ({ page }) => {
    await boot(page, { active: ACTIVE, token: "" });
    const hint = page.locator("#conn-collection-hint");
    await expect(hint).toBeVisible();
    await expect(hint).toHaveText("API KEY MISSING — OPEN API KEY AND REGISTER A KEY");
    await expect(page.locator("#conn-collection")).toBeDisabled();
    expect(await page.locator("#conn-collection option").allTextContents()).toEqual(["— none —"]);
    await expect(page.locator("#apikey-btn")).toHaveClass(/is-bad/);
  });

  test("H8 — wrong key: hint shows API KEY REJECTED and the dot stays is-bad", async ({ page }) => {
    await boot(page, { active: ACTIVE, token: WRONG_KEY });
    const hint = page.locator("#conn-collection-hint");
    await expect(hint).toBeVisible();
    await expect(hint).toHaveText("API KEY REJECTED — OPEN API KEY AND REGISTER THE CORRECT KEY");
    await expect(page.locator("#apikey-btn")).toHaveClass(/is-bad/);
    await expect(page.locator("#apikey-btn")).not.toHaveClass(/is-set/);
    // Persistent: still red well past the 3s transient window.
    await page.waitForTimeout(4000);
    await expect(page.locator("#apikey-btn")).toHaveClass(/is-bad/);
  });

  test("H9 — a correct key after a 401 hides the hint and flashes is-set for 3s", async ({ page }) => {
    await boot(page, { active: ACTIVE, token: WRONG_KEY });
    await expect(page.locator("#conn-collection-hint")).toHaveText("API KEY REJECTED — OPEN API KEY AND REGISTER THE CORRECT KEY");
    await openPopover(page);
    await enterRegisteredKey(page);
    await page.locator("#apikey-save").click();
    await expect(page.locator("#conn-collection-hint")).toBeHidden();
    await expect(page.locator("#apikey-btn")).not.toHaveClass(/is-bad/);
    // Blue dot present right after acceptance…
    await expect(page.locator("#apikey-btn")).toHaveClass(/is-set/, { timeout: 5000 });
    const setAt = Date.now();
    // …and gone after the 3s timer (checked at ~1.5s it is still on).
    await page.waitForTimeout(1500);
    await expect(page.locator("#apikey-btn")).toHaveClass(/is-set/);
    await page.waitForTimeout(2200);
    await expect(page.locator("#apikey-btn")).not.toHaveClass(/is-set/);
    expect(Date.now() - setAt).toBeGreaterThan(3000);
    await expect(page.locator("#conn-collection")).toHaveValue(ACTIVE);
    expect((await page.locator("#conn-collection option").allTextContents()).length)
      .toBeGreaterThan(1);
  });

  test("H10 — Clear removes the key: REMOVED, focus stays on the input, data reloaded", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    await openPopover(page);
    await expect(page.locator("#apikey-clear")).toBeVisible();
    await page.locator("#apikey-clear").click();
    await expect(page.locator("#apikey-status")).toHaveText("API KEY REMOVED");
    await expect(page.locator("#apikey-status")).toHaveClass(/ok/);
    await expect(page.locator("#apikey-input")).toBeFocused();
    await expect(page.locator("#apikey-popover")).toBeVisible();
    expect(await page.evaluate(() => sessionStorage.getItem("qhm_api_key"))).toBeNull();
    await expect(page.locator("#apikey-clear")).toBeHidden();
    await expect(page.locator("#conn-collection-hint")).toHaveText("API KEY MISSING — OPEN API KEY AND REGISTER A KEY");
    await expect(page.locator("#conn-collection")).toBeDisabled();
  });

  test("H11 — Escape closes the popover and returns focus to the trigger", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    await openPopover(page);
    await page.keyboard.press("Escape");
    await expect(page.locator("#apikey-popover")).toBeHidden();
    await expect(page.locator("#apikey-btn")).toHaveAttribute("aria-expanded", "false");
    await expect(page.locator("#apikey-btn")).toBeFocused();
  });

  test("H12 — outside pointerdown closes the popover without stealing focus", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    await openPopover(page);
    // Click a focusable control outside the popover that the popover does not
    // overlay (#conn-retest sits underneath it in the connection bar).
    await page.locator("#tab-collections").click();
    await expect(page.locator("#apikey-popover")).toBeHidden();
    const focused = await page.evaluate(() => document.activeElement.id);
    expect(focused, "focus was stolen by the popover trigger").not.toBe("apikey-btn");
  });

  test("H13 — clicking the trigger while open toggles the popover closed", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    await openPopover(page);
    await page.locator("#apikey-btn").click();
    await expect(page.locator("#apikey-popover")).toBeHidden();
    await expect(page.locator("#apikey-btn")).toHaveAttribute("aria-expanded", "false");
    await page.locator("#apikey-btn").click();
    await expect(page.locator("#apikey-popover")).toBeVisible();
  });

  test("H14 — reopening with a stored key shows Clear and an empty input", async ({ page }) => {
    await boot(page, { active: ACTIVE });
    await openPopover(page);
    await page.locator("#apikey-input").fill("leftover-typing");
    await page.keyboard.press("Escape");
    await expect(page.locator("#apikey-popover")).toBeHidden();
    await openPopover(page);
    await expect(page.locator("#apikey-clear")).toBeVisible();
    await expect(page.locator("#apikey-input")).toHaveValue("");
    await expect(page.locator("#apikey-input")).toHaveAttribute("placeholder", "••••••••");
    await expect(page.locator("#apikey-status")).toHaveText("");
  });
});

// Regression guard for BUG-1 (test book §3 H1/H6/H9): a 401 must NOT erase the
// persisted active collection, and registering the right key must restore it.
test("BUG-1 — a 401 keeps the persisted active collection, which is restored on auth", async ({ page }) => {
  expectRejectedCredentials(page);
  await page.addInitScript((active) => {
    sessionStorage.setItem("qhm_api_key", "WRONG-KEY");
    localStorage.setItem("qhm_active_collection", active);
  }, ACTIVE);
  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-collection-hint")).toHaveText("API KEY REJECTED — OPEN API KEY AND REGISTER THE CORRECT KEY");
  expect(await page.evaluate(() => localStorage.getItem("qhm_active_collection")))
    .toBe(fixtureName("e2e_e_mon"));

  // Registering the correct key restores the user's collection, not the first one.
  await page.locator("#apikey-btn").click();
  await enterRegisteredKey(page);
  await page.locator("#apikey-save").click();
  await expect(page.locator("#conn-collection-hint")).toBeHidden();
  await expect(page.locator("#conn-collection")).toHaveValue(fixtureName("e2e_e_mon"));
});
