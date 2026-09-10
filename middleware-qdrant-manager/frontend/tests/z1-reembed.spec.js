import { fixtureOverride, settleOwnedReindex } from "./harness.mjs";
// Test book §5 — Serie RE (02 Re-embed Vectors) + the differential Points lockdown.
// Runs against e2e_z_basic. Must NOT run while the parallel agents are active.
import { expect, test } from "./fixtures.js";
import { boot, selectActive, openTab, openPointsSubtab, setupFetch } from "./helpers.js";

// Default to the 2,000-point fixture from tools/seed-reg.mjs: on a small collection
// the job finishes before the pause/cancel window can be observed.
const COLL = fixtureOverride("QHM_REINDEX_COLL", "e2e_z_reg");

test.describe.configure({ mode: "serial" });

async function openCollections(page) {
  const errors = await boot(page, { active: COLL });
  await selectActive(page, COLL);
  await openTab(page, "collections");
  return errors;
}

async function report(request) {
  const r = await setupFetch(request, 'GET', `/collections/${COLL}`);
  const body = await r.json();
  expect(Number.isSafeInteger(body.points_count)).toBe(true);
  return body;
}

test("RE1 — read-only preflight runs and writes nothing", async ({ page, request }) => {
  const errors = await openCollections(page);
  const before = await report(request);
  await expect(page.locator("#cr-preflight")).toBeEnabled();
  await page.locator("#cr-preflight").click();
  await expect(page.locator("#cr-preflight-result")).toContainText(/Preflight/i);
  const after = await report(request);
  expect(JSON.stringify(after.points_count ?? after.report?.points_count))
    .toEqual(JSON.stringify(before.points_count ?? before.report?.points_count));
  expect(errors).toEqual([]);
});

test("RE2 — empty confirmation keeps Start disabled", async ({ page }) => {
  await openCollections(page);
  await page.locator("#cr-preflight").click();
  await expect(page.locator("#cr-confirm-wrap")).toBeVisible();
  await expect(page.locator("#cr-start")).toBeVisible();
  await expect(page.locator("#cr-start")).toBeDisabled();
});

test("RE3 — wrong confirmation keeps Start disabled and shows the hint", async ({ page }) => {
  await openCollections(page);
  await page.locator("#cr-preflight").click();
  await expect(page.locator("#cr-confirm-wrap")).toBeVisible();
  await page.locator("#cr-confirm").fill(`${COLL}x`);
  await expect(page.locator("#cr-start")).toBeDisabled();
  await expect(page.locator("#cr-confirm-hint")).toBeVisible();
});

// A pending job holds a lock on the collection: preflight then refuses with "A
// reindex job is already paused for this collection" and no amount of retrying
// clears it. Cancel it so each case starts from a startable collection.
async function clearPendingJob() { await settleOwnedReindex(COLL); }

test.afterAll(clearPendingJob);

async function startJob(page) {
  await expect(page.locator("#cr-preflight")).toBeEnabled({ timeout: 60_000 });
  await page.locator("#cr-preflight").click();
  await expect(page.locator("#cr-confirm-wrap")).toBeVisible({ timeout: 30_000 });
  await page.locator("#cr-confirm").fill(COLL);
  await expect(page.locator("#cr-start")).toBeEnabled();
  await page.locator("#cr-start").click();
  // The status of the LAST job persists, so a COMPLETED already on screen may
  // belong to a previous run: require a live state for the job just started.
  await expect(page.locator("#cr-status")).toContainText(
    /RUNNING|FINALIZING|PAUSED/, { timeout: 120_000 });
}

test("RE6 — cancel stops the job and leaves the collection readable", async ({ page, request }) => {
  test.setTimeout(300_000);
  await clearPendingJob(request);
  await openCollections(page);
  await startJob(page);
  if (!(await page.locator("#cr-cancel").isVisible().catch(() => false))) {
    test.skip(true, "job finished before cancel could be exercised");
  }
  await page.locator("#cr-cancel").click();
  await expect(page.locator("#cr-status")).toContainText(
    /CANCELLED|COMPLETED/, { timeout: 120_000 });
  const rep = await report(request);
  expect(rep.exists ?? true).toBeTruthy();
});

test("RE4/5/7/9/10/11/12/13/14 — lifecycle, pause/resume, differential Points lockdown, verify payload unchanged, hybrid query post-job", async ({ page, request }) => {
  test.setTimeout(900_000);
  await clearPendingJob(request);
  const errors = await openCollections(page);

  // RE13 prep: snapshot point 1 before the job
  const pre = await setupFetch(request, "GET", `/points/1?collection_name=${COLL}`);
  const preBody = await pre.json();
  expect(preBody.payload).toBeTruthy();

  await startJob(page);

  await expect(page.locator("#cr-status .reindex-progress")).toBeVisible();

  // Pause immediately so the worker cannot finish while the test navigates to
  // Points. A paused job is a real mutation-lock state and makes RE9/RE10
  // deterministic even on fast GPUs.
  const pauseVisible = await page.locator("#cr-pause").isVisible().catch(() => false);
  expect(pauseVisible, "Pause never became visible — job too short to test RE5").toBe(true);
  if (pauseVisible) await page.locator("#cr-pause").click();
  await expect(page.locator("#cr-status")).toContainText(
    /PAUSED|COMPLETED|FINALIZING/, { timeout: 90_000 });
  const settled = await Promise.race([
    page.locator("#cr-resume").waitFor({ state: "visible", timeout: 90_000 })
      .then(() => "resumable").catch(() => null),
    expect(page.locator("#cr-status")).toContainText("COMPLETED", { timeout: 90_000 })
      .then(() => "completed").catch(() => null),
  ]);
  expect(settled, "pause left the job neither resumable nor completed").not.toBeNull();
  expect(settled, "job finished before the lockdown window could be observed")
    .toBe("resumable");

  if (settled === "resumable") {
    await openTab(page, "points");
    await expect(page.locator("#pt-embedding-guard")).toContainText(
      "Point retrieval remains available", { timeout: 15_000 });
    await openPointsSubtab(page, "upsert");
    await page.locator("#up-id").fill("1");
    await page.locator("#up-text").fill("blocked while reindexing");
    await expect(page.locator("#up-submit")).toBeDisabled();
    await openPointsSubtab(page, "batch");
    await expect(page.locator("#bt-submit")).toBeDisabled();
    await openPointsSubtab(page, "get");
    await page.locator("#get-id").fill("1");
    await expect(page.locator("#get-submit")).toBeEnabled();
    await page.locator("#get-submit").click();
    await expect(page.locator("#get-result")).not.toBeEmpty({ timeout: 30_000 });
    await openTab(page, "collections");
    await page.locator("#cr-resume").click();
    await expect(page.locator("#cr-status")).toContainText(
      /RUNNING|FINALIZING|COMPLETED/, { timeout: 90_000 });
  }
  await expect(page.locator("#cr-status")).toContainText("COMPLETED", { timeout: 780_000 });

  // RE13: point 1 payload/text must be unchanged after re-embed
  const post = await setupFetch(request, "GET", `/points/1?collection_name=${COLL}`);
  expect((await post.json()).payload).toEqual(preBody.payload);

  // RE14: hybrid query must return results using the new active generation
  await openTab(page, "query");
  await page.locator("#ql-query").fill("test query");
  await page.locator("#ql-run").click();
  await expect(page.locator("#ql-errors")).toBeEmpty({ timeout: 30_000 });
  await expect(page.locator("#ql-results .result-card").first()).toBeVisible({ timeout: 30_000 });

  await openTab(page, "points");
  await openPointsSubtab(page, "upsert");
  await page.locator("#up-id").fill("1");
  await page.locator("#up-text").fill("unblocked after reindex");
  await expect(page.locator("#up-submit")).toBeEnabled();
  expect(errors).toEqual([]);
});

test("RE8 — switching the active collection during a job clears the scoped results", async ({ page }) => {
  test.setTimeout(300_000);
  await openCollections(page);
  await page.locator("#cr-preflight").click();
  await expect(page.locator("#cr-preflight-result")).toContainText(/Preflight/i);
  const options = await page.locator("#conn-collection option").allTextContents();
  const other = options.find((o) => o !== COLL);
  test.skip(!other, "no second collection available");
  await selectActive(page, other);
  await expect(page.locator("#cr-preflight-result")).toBeEmpty();
});
