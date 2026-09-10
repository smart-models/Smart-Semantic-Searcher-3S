import { fixtureOverride } from "./harness.mjs";
import { expect, test } from "./fixtures.js";
import { boot, selectActive, openTab, openPointsSubtab, BASE_URL, TOKEN } from "./helpers.js";

const COLL = fixtureOverride("QHM_E7_COLL", "e2e_z_basic");

test("E7 — re-embedding a cross-profile collection makes it compatible again", {
  // Z4 is the same re-embed under ADVANCED; its Points-lockdown-during-job
  // clause (RE9-RE11) is not observed here, hence partial.
  annotation: { type: "case-partial", description: "book:12/Z4" },
}, async ({ page, request }) => {
  test.setTimeout(900_000);
  const errors = await boot(page, { active: COLL });
  await selectActive(page, COLL);

  // Before: locked by the profile switch, banner suppressed, Re-embed live.
  await expect(page.locator("#panel-query")).toHaveAttribute("inert", "");
  await expect(page.locator("#service-gate")).toBeEmpty();
  await openTab(page, "collections");
  await expect(page.locator("#cc-report")).toContainText(/INCOMPATIBLE/i);
  await expect(page.locator("#cr-wrap")).not.toHaveAttribute("inert", "");

  // Run the regeneration to completion.
  await page.locator("#cr-preflight").click();
  await expect(page.locator("#cr-confirm-wrap")).toBeVisible();
  await page.locator("#cr-confirm").fill(COLL);
  await expect(page.locator("#cr-start")).toBeEnabled();
  await page.locator("#cr-start").click();
  // A previous job leaves a persisted COMPLETED status, so wait for the NEW job
  // to enter a live state first; only then wait for its completion.
  await expect(page.locator("#cr-status")).toContainText(/RUNNING|FINALIZING|PAUSED/,
    { timeout: 120_000 });
  await expect(page.locator("#cr-status")).toContainText("COMPLETED", { timeout: 780_000 });

  // After: the stored contract matches the active embedder.
  const rep = await (await request.get(`${BASE_URL}/collections/${COLL}`,
    { headers: { Authorization: `Bearer ${TOKEN}` } })).json();
  expect(rep.embedding_compatibility.status).toBe("compatible");
  expect(rep.embedding_compatibility.stored.profile).toBe("advanced");

  // And the UI unlocks without a reload.
  await expect(page.locator("#cc-report")).not.toContainText("INCOMPATIBLE", { timeout: 60_000 });
  await expect(page.locator("#panel-query")).not.toHaveAttribute("inert", "");
  await openTab(page, "points");
  await openPointsSubtab(page, "upsert");
  await page.locator("#up-id").fill("1");
  await page.locator("#up-text").fill("writable again after re-embed");
  await expect(page.locator("#up-submit")).toBeEnabled();
  await openTab(page, "query");
  await page.locator("#ql-query").fill("profile roundtrip");
  const queried = page.waitForResponse((response) => new URL(response.url()).pathname === "/query/search"
    && response.request().method() === "POST");
  await page.locator("#ql-run").click();
  const response = await queried;
  expect(response.status()).toBe(200);
  expect((await response.json()).results.length).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});
