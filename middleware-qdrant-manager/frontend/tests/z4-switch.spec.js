import { fixtureOverride } from "./harness.mjs";
// Test book §4 (E1–E4, E7, E8) and §12 (Z1–Z7): the REAL BASIC<->ADVANCED
// profile switch. The runner sets QHM_PROFILE to the profile the stack is
// currently running, and QHM_NATIVE to the collection born under that profile.
import { expect, test } from "./fixtures.js";
import { boot, selectActive, openTab, openPointsSubtab, BASE_URL, TOKEN } from "./helpers.js";

const PROFILE = process.env.QHM_PROFILE || "basic";
const NATIVE = fixtureOverride("QHM_NATIVE", "e2e_z_basic");      // compatible now
const FOREIGN = fixtureOverride("QHM_FOREIGN");                // born under the other profile

test(`E-native — (${PROFILE}) collection born under the active profile is compatible`, {
  // E1 is the BASIC-born native check, E3 the ADVANCED-born one; the running
  // profile decides which row this execution proves.
  annotation: { type: "case", description: PROFILE === "advanced" ? "book:4/E3" : "book:4/E1" },
}, async ({ page }) => {
  const errors = await boot(page, { active: NATIVE });
  await selectActive(page, NATIVE);
  await expect(page.locator("#service-gate")).toBeEmpty();
  await expect(page.locator("#panel-query")).not.toHaveAttribute("inert", "");
  await openTab(page, "collections");
  await expect(page.locator("#cc-report")).not.toContainText("INCOMPATIBLE");
  expect(errors).toEqual([]);
});

test(`E-native — (${PROFILE}) health reports the active embedder profile`, async ({ request }) => {
  const r = await request.get(`${BASE_URL}/health`);
  const j = await r.json();
  expect(j.embedder_profile).toBe(PROFILE);
});

test(`E2/E4 — (${PROFILE}) cross-profile collection is locked with the banner suppressed`, {
  annotation: [
    { type: "case", description: "book:12/Z2" },
    // TB-PRO-001 needs both directions and the post-reindex checks; one
    // profile's lockdown observation is a partial contribution.
    { type: "case-partial", description: "TB-PRO-001" },
  ],
}, async ({ page }) => {
  test.skip(!FOREIGN, "no cross-profile collection provided");
  const errors = await boot(page, { active: FOREIGN });
  await selectActive(page, FOREIGN);
  // Profile switch: the banner is deliberately suppressed.
  await expect(page.locator("#service-gate")).toBeEmpty();
  await expect(page.locator("#panel-query")).toHaveAttribute("inert", "");
  await expect(page.locator("#panel-query")).toHaveClass(/operations-disabled/);
  await page.locator("#tab-points").click();
  await expect(page.locator("#panel-points")).not.toHaveAttribute("inert", "");
  await openPointsSubtab(page, "upsert");
  await page.locator("#up-id").fill("1");
  await page.locator("#up-text").fill("blocked vector write");
  await expect(page.locator("#up-submit")).toBeDisabled();
  await openPointsSubtab(page, "get");
  await page.locator("#get-id").fill("1");
  await expect(page.locator("#get-submit")).toBeEnabled();
  const retrieved = page.waitForResponse((response) => new URL(response.url()).pathname === "/points/1"
    && response.request().method() === "GET");
  await page.locator("#get-submit").click();
  expect((await retrieved).status()).toBe(200);
  await openTab(page, "collections");
  await expect(page.locator("#cc-report")).toContainText(/INCOMPATIBLE/i);
  await expect(page.locator("#cr-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#ce-wrap")).not.toHaveAttribute("inert", "");
  for (const id of ["#cc-recreate-control", "#cu-wrap", "#pi-wrap"]) {
    await expect(page.locator(id)).toHaveAttribute("inert", "");
  }
  // 07 Delete works on a typed name, so an incompatible active
  // collection never locks it.
  await expect(page.locator("#cd-wrap")).not.toHaveAttribute("inert", "");
  // A profile switch is exactly when a new, compatible collection is needed:
  // 03 keeps its create half live.
  await expect(page.locator("#cc-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-submit")).toHaveText(/create collection/i);
  expect(errors).toEqual([]);
});

test(`E8 — (${PROFILE}) the collection selector stays usable during a lockdown`, async ({ page }) => {
  test.skip(!FOREIGN, "no cross-profile collection provided");
  await boot(page, { active: FOREIGN });
  await selectActive(page, FOREIGN);
  await expect(page.locator("#conn-collection")).toBeEnabled();
  await selectActive(page, NATIVE);
  await expect(page.locator("#panel-query")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#service-gate")).toBeEmpty();
});

test(`X4 — (${PROFILE}) Export still works on an incompatible collection`, async ({ page }) => {
  test.skip(!FOREIGN, "no cross-profile collection provided");
  await boot(page, { active: FOREIGN });
  await selectActive(page, FOREIGN);
  await openTab(page, "collections");
  const download = page.waitForEvent("download", { timeout: 60_000 }).catch(() => null);
  await page.locator("#ce-export").click();
  // "Exporting… scanning every point" is the intermediate state: poll for the
  // terminal message instead of reading the first render.
  await expect
    .poll(async () => (await page.locator("#ce-result").textContent()) || "", { timeout: 90_000 })
    .toMatch(/Exported|Partial export|empty/i);
  await download;
});

test(`M-C/M-D — (${PROFILE}) Monitoring reflects the active profile`, {
  annotation: { type: "case", description: "book:12/Z6" },
}, async ({ page }) => {
  await boot(page, { active: NATIVE });
  await openTab(page, "monitoring");
  await expect(page.locator("#mon-models")).not.toBeEmpty({ timeout: 30_000 });
  await expect(page.locator("#mon-models")).toContainText(new RegExp(PROFILE, "i"));
});

// Guard against a spurious pass of the "native" case: the FOREIGN collection
// must really be reported INCOMPATIBLE, and the NATIVE one really compatible.
test(`E-sanity — (${PROFILE}) the report itself distinguishes native from foreign`, async ({ page }) => {
  test.skip(!FOREIGN, "no cross-profile collection provided");
  await boot(page, { active: FOREIGN });
  await selectActive(page, FOREIGN);
  await openTab(page, "collections");
  await expect(page.locator("#cc-report")).toContainText(/INCOMPATIBLE/i);
  await selectActive(page, NATIVE);
  await expect(page.locator("#cc-report")).not.toContainText(/INCOMPATIBLE/i, { timeout: 30_000 });
  await expect(page.locator("#panel-collections")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-wrap")).not.toHaveAttribute("inert", "");
});
