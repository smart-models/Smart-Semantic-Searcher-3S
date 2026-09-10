import { expect, test } from "./fixtures.js";

import { BASE_URL, fixtureName, cleanupOwnedCollection } from "./harness.mjs";

test("Collections Re-embed Vectors completes a safe UI migration with owned cleanup", async ({
  page,
}) => {
  test.setTimeout(600_000);
  const token = process.env.QHM_E2E_TOKEN || "";
  const collection = fixtureName("e2e_vector_reindex");
  await page.addInitScript((apiToken) => {
    if (apiToken) sessionStorage.setItem("qhm_api_key", apiToken);
  }, token);

  const openTab = async (name) => {
    await page.locator(`#tab-${name}`).click();
    await expect(page.locator(`#panel-${name}`)).toBeVisible();
  };

  try {
    await page.goto(`${BASE_URL}/#collections`, {
      waitUntil: "domcontentloaded",
    });
    await expect(page.locator("#conn-health-status")).toContainText(
      /HEALTHY|READY/,
      { timeout: 30_000 },
    );
    await expect(page.locator("#conn-collection option").first()).toBeAttached({
      timeout: 30_000,
    });

    const duplicateIds = await page.evaluate(() => {
      const ids = [...document.querySelectorAll("[id]")].map((node) => node.id);
      return ids.filter((id, index) => ids.indexOf(id) !== index);
    });
    expect(duplicateIds).toEqual([]);

    await page.locator("#cc-name").fill(collection);
    await expect(page.locator("#cc-submit")).toBeEnabled();
    await page.locator("#cc-submit").click();
    await expect(page.locator("#cc-create-result")).toContainText(
      "Collection created",
      { timeout: 30_000 },
    );
    await expect(page.locator("#conn-collection")).toHaveValue(collection);

    await openTab("points");
    await page.locator('[data-pt-tab="batch"]').click();
    const points = Array.from({ length: 500 }, (_, index) => ({
      point_id: String(index + 1),
      text: `document ${index + 1} about hybrid dense sparse retrieval`,
      payload: {
        category: index % 2 ? "odd" : "even",
        sequence: index + 1,
      },
    }));
    await page.locator("#bt-json").fill(JSON.stringify(points));
    await page.locator("#bt-submit").click();
    await expect(page.locator("#bt-result")).toContainText(
      "500 point(s) upserted",
      { timeout: 240_000 },
    );

    await openTab("collections");
    await expect(page.locator("#cr-preflight")).toBeEnabled();
    await expect(page.locator("#cr-start")).toBeHidden();
    await page.locator("#cr-preflight").click();
    await expect(page.locator("#cr-preflight-result")).toContainText(
      "Preflight passed",
      { timeout: 120_000 },
    );
    await expect(page.locator("#cr-preflight-result")).toContainText("500");
    await expect(page.locator("#cr-confirm-wrap")).toBeVisible();

    await page.locator("#cr-confirm").fill(`${collection}_wrong`);
    await expect(page.locator("#cr-start")).toBeDisabled();
    await expect(page.locator("#cr-confirm-hint")).toBeVisible();
    await page.locator("#cr-confirm").fill(collection);
    await expect(page.locator("#cr-start")).toBeEnabled();
    await page.locator("#cr-start").click();
    // Job state is reported only by the status badge, not a separate message.
    await expect(page.locator("#cr-status")).toContainText(
      /Vector regeneration: (RUNNING|FINALIZING|COMPLETED)/,
      { timeout: 30_000 },
    );

    await expect(page.locator("#cr-pause")).toBeVisible();
    await page.locator("#cr-pause").click();
    // Pause commits the batch in flight, so the job may also reach the end while
    // stopping: accept either outcome instead of assuming it settles on PAUSED.
    await expect(page.locator("#cr-status")).toContainText(/PAUSED|COMPLETED/, {
      timeout: 120_000,
    });
    const finishedWhilePausing = /COMPLETED/.test(
      (await page.locator("#cr-status").textContent()) || "",
    );
    test.skip(
      finishedWhilePausing,
      "job completed before the pause took effect — nothing left to resume",
    );
    // Resume stays hidden while the worker is still releasing the batch it had
    // started; on 500 points that can outlast the pause itself, so give it the
    // same budget rather than 30s.
    await expect(page.locator("#cr-resume")).toBeVisible({
      timeout: 120_000,
    });

    await openTab("points");
    await page.locator('[data-pt-tab="upsert"]').click();
    await page.locator("#up-id").fill("999");
    await page.locator("#up-text").fill("this write must be blocked while paused");
    await page.locator("#up-payload").fill('{"category":"blocked"}');
    await expect(page.locator("#pt-embedding-guard")).toContainText(
      "Point mutations are temporarily disabled",
    );
    await expect(page.locator("#up-submit")).toBeDisabled();

    await openTab("collections");
    await expect(page.locator("#cr-resume")).toBeVisible();
    await page.locator("#cr-resume").click();
    await expect(page.locator("#cr-status")).toContainText("COMPLETED", {
      timeout: 180_000,
    });
    await expect(page.locator("#cr-status")).toContainText("100%");
    await expect(page.locator("#cr-status")).toContainText("500 / 500");
    await expect(page.locator("#cc-report")).toContainText("COMPATIBLE");

    await openTab("points");
    await page.locator('[data-pt-tab="get"]').click();
    await page.locator("#get-id").fill("2");
    await page.locator("#get-submit").click();
    await expect(page.locator("#get-result")).toContainText("document 2", {
      timeout: 30_000,
    });
    await expect(page.locator("#get-result")).toContainText('"category": "odd"');

    await openTab("query");
    await page.locator("#ql-query").fill("hybrid dense sparse retrieval");
    await page.locator("#ql-run").click();
    await expect(page.locator("#ql-results .result-card").first()).toBeVisible({
      timeout: 120_000,
    });

    const visibleActionBoxes = await page
      .locator(".reindex-actions .btn:visible")
      .evaluateAll((buttons) => buttons.map((button) => {
        const box = button.getBoundingClientRect();
        return { left: box.left, right: box.right, top: box.top, bottom: box.bottom };
      }));
    for (let index = 1; index < visibleActionBoxes.length; index += 1) {
      const previous = visibleActionBoxes[index - 1];
      const current = visibleActionBoxes[index];
      const overlaps = previous.left < current.right
        && previous.right > current.left
        && previous.top < current.bottom
        && previous.bottom > current.top;
      expect(overlaps).toBe(false);
    }
  } finally {
    await cleanupOwnedCollection(collection);
  }

});
