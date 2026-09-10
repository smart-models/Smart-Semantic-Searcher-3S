import { fulfillResponse } from "./telemetry.mjs";
import { expect, test } from "./fixtures.js";

import { BASE_URL, fixtureName } from "./harness.mjs";
import { prepareConsole } from "./helpers.js";

async function bootstrap(page) {
  await prepareConsole(page, { active: fixtureName("e2e_a_keep") });
  // Force servicesReady=true deterministically: without this, a really-down
  // embedder would make panels inert via the OUTAGE path, so the Ingest
  // preflight assertions would pass/fail for the wrong reason. Mirrors
  // service-readiness.e2e.spec.js / embedding-compatibility.e2e.spec.js.
  await page.route("**/health", async (route) => {
    await fulfillResponse(route, {
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "healthy",
        qdrant: "reachable",
        embedder: "reachable",
        collections: 1,
      }),
    });
  });
}

const VALID_FILE = JSON.stringify({
  chunks: [
    { id: 1, text: "alpha", source: "doc", page: 1 },
    { id: 2, text: "beta", source: "doc", page: 2, tag: "intro" },
  ],
});
const INVALID_FILE = JSON.stringify({
  chunks: [{ id: "x", text: "" }],
});

async function selectIngestFile(page, name, contents) {
  await page.locator("#ing-file").setInputFiles({
    name,
    mimeType: "application/json",
    buffer: Buffer.from(contents),
  });
}

async function openIngestSubtab(page) {
  await page.goto(`${BASE_URL}/#points`);
  await expect(page.locator("#conn-collection option").first()).toBeAttached({
    timeout: 30_000,
  });
  await expect(page.locator("#conn-collection")).not.toHaveValue("");
  await page.locator('[data-pt-tab="ingest"]').click();
}

async function mockIngest(page, { ok = true } = {}) {
  await page.route("**/points/ingest-chunks", async (route) => {
    const body = route.request().postDataJSON();
    const n = Array.isArray(body.chunks) ? body.chunks.length : 0;
    if (!ok) {
      await fulfillResponse(route, { status: 200, contentType: "application/json",
        json: { ok: false, ingested: 0, skipped: 0, collection_name: body.collection_name,
                error: "boom" } });
      return;
    }
    await fulfillResponse(route, { status: 200, contentType: "application/json",
      json: { ok: true, ingested: n, skipped: 0, collection_name: body.collection_name } });
  });
}

test("uploading a valid file auto-runs preflight and reveals confirm", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);

  const preflight = page.locator("#ing-preflight-result");
  await expect(preflight.locator(".alert-success")).toContainText("Preflight passed");
  await expect(preflight).toContainText("chunks");
  await expect(preflight.locator("th", { hasText: "name" })).toBeVisible();
  await expect(preflight.locator('tr[data-field="source"] td').first()).toBeVisible();
  await expect(page.locator("#ing-confirm-submit")).toBeVisible();
});

test("an invalid file blocks the confirm button", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "bad.json", INVALID_FILE);

  await expect(page.locator("#ing-preflight-result .alert-error"))
    .toContainText("Preflight blocked");
  await expect(page.locator("#ing-confirm-submit")).toBeHidden();
});

test("swapping a valid file for an invalid one re-hides confirm", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);
  await expect(page.locator("#ing-confirm-submit")).toBeVisible();

  await selectIngestFile(page, "bad.json", INVALID_FILE);
  await expect(page.locator("#ing-preflight-result .alert-error"))
    .toContainText("Preflight blocked");
  await expect(page.locator("#ing-confirm-submit")).toBeHidden();
});

test("editing a payload name to a reserved key blocks confirm", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);
  await expect(page.locator("#ing-confirm-submit")).toBeVisible();

  await page.locator('#ing-map-table tbody tr[data-field="source"]')
    .locator(".ing-map-name").fill("id");
  await expect(page.locator("#ing-map-errors")).toContainText("reserved");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("choosing an incompatible type blocks confirm", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);

  const row = page.locator('#ing-map-table tbody tr[data-field="source"]');
  await row.locator(".ing-map-type").selectOption("integer");
  await expect(page.locator("#ing-map-errors")).toContainText("not convertible");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("a valid rename and type keeps confirm enabled", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);

  const row = page.locator('#ing-map-table tbody tr[data-field="source"]');
  await row.locator(".ing-map-name").fill("source_name");
  await row.locator(".ing-map-type").selectOption("keyword");
  await expect(page.locator("#ing-map-errors")).toBeEmpty();
  await expect(page.locator("#ing-confirm-submit")).toBeEnabled();
});

test("the context badge reads 'source file' with the full .json filename", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);

  const preflight = page.locator("#ing-preflight-result");
  await expect(preflight).toContainText("source file");
  await expect(preflight).toContainText("chunks.json");
});

test("source_id FIELD reads 'source_id' and NAME defaults to the key 'source_id'", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);

  const row = page.locator('#ing-map-table tbody tr[data-source-id="true"]');
  await expect(row.locator("td").first()).toHaveText("source_id");
  await expect(row.locator(".ing-map-name")).toHaveValue("source_id");
});

test("the type dropdown offers keyword", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);

  const options = page.locator('#ing-map-table tbody tr[data-source-id="true"] .ing-map-type option');
  await expect(options.filter({ hasText: "keyword" })).toHaveCount(1);
});

test("clearing the source_id name blocks confirm", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);
  await expect(page.locator("#ing-confirm-submit")).toBeVisible();

  await page.locator('#ing-map-table tbody tr[data-source-id="true"]')
    .locator(".ing-map-name").fill("");
  await expect(page.locator("#ing-map-errors")).toContainText("name is required");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("renaming source_id to a reserved key blocks confirm", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);
  await expect(page.locator("#ing-confirm-submit")).toBeVisible();

  await page.locator('#ing-map-table tbody tr[data-source-id="true"]')
    .locator(".ing-map-name").fill("text");
  await expect(page.locator("#ing-map-errors")).toContainText("reserved");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("uuid type on a non-uuid source_id blocks confirm", async ({ page }) => {
  await bootstrap(page);
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);

  await page.locator('#ing-map-table tbody tr[data-source-id="true"]')
    .locator(".ing-map-type").selectOption("uuid");
  await expect(page.locator("#ing-map-errors")).toContainText("not convertible");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("ingest shows a progress bar and a COMPLETED state", async ({ page }) => {
  await bootstrap(page);
  await mockIngest(page, { ok: true });
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);
  // This case exercises the progress display; index creation has live coverage
  // in b2-ingest-indexes and must not escape the simulated ingest operation.
  await page.locator("#ing-create-indexes").uncheck();
  await page.locator("#ing-confirm-submit").click();

  await expect(page.locator("#ing-result .reindex-progress")).toBeVisible();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");
  await expect(page.locator("#ing-preflight-result")).toBeEmpty();
});

test("a failing batch shows FAILED and restores the table", async ({ page }) => {
  await bootstrap(page);
  await mockIngest(page, { ok: false });
  await openIngestSubtab(page);
  await selectIngestFile(page, "chunks.json", VALID_FILE);
  await page.locator("#ing-confirm-submit").click();

  await expect(page.locator("#ing-result")).toContainText("failed");
  await expect(page.locator("#ing-preflight-result")).toBeVisible();
});
