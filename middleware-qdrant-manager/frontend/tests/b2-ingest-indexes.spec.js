import { fulfillResponse } from "./telemetry.mjs";
import { fixtureName } from "./harness.mjs";
// Test book §7.1b — Points → Ingest → auto-created payload indexes (IX1..IX7).
// Sandbox collection: e2e_b_idx (created by this file, not by the seed tool).
//
// IX1 runs a real ingest (the embedder does the work); every other case mocks
// /points/ingest-chunks so it exercises only the index phase.
import { expect, test } from "./fixtures.js";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { boot, openTab, openPointsSubtab, setupFetch } from "./helpers.js";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(DIR, "fixtures");
const COLL = fixtureName("e2e_b_idx");

// A fresh collection per case: the index phase asserts on which indexes exist,
// so a leftover one from the previous case would make the next one lie.
async function resetCollection(request) {
  await setupFetch(request, "DELETE", `/collections/${COLL}?confirm_delete=${COLL}`, undefined, [200, 404]);
  const created = await setupFetch(request, "POST", "/collections", {
    collection_name: COLL,
    dense_on_disk: true,
    sparse_on_disk: true,
    on_disk_payload: true,
    quantization: true,
    sparse_idf: true,
  });
  expect(created.ok(), "sandbox collection created").toBeTruthy();
}

async function indexMap(request) {
  const res = await setupFetch(request, "GET", `/collections/${COLL}/payload-indexes`);
  expect(res.ok()).toBeTruthy();
  const body = await res.json();
  return Object.fromEntries(
    body.indexes.map((i) => [i.field_name, i.field_schema]));
}

async function start(page) {
  await boot(page, { active: COLL });
  await openTab(page, "points");
  await openPointsSubtab(page, "ingest");
  await expect(page.locator("#conn-collection")).toHaveValue(COLL);
}

async function uploadAndWait(page, file = "e2e_chunks_idx.json") {
  await page.locator("#ing-file").setInputFiles(path.join(DATA, file));
  await expect(page.locator("#ing-map-table")).toBeVisible();
}

function row(page, field) {
  return page.locator(`#ing-map-table tbody tr[data-field="${field}"]`);
}
async function setType(page, field, type) {
  await row(page, field).locator(".ing-map-type").selectOption(type);
}
function chip(page, field) {
  return page.locator(`#ing-index-summary .chip[data-field="${field}"]`);
}

// Fulfil every ingest batch with a success (or with `plan(i)` when given), so a
// case can reach the index phase without paying for real embeddings.
async function mockIngest(page, plan) {
  await page.route("**/points/ingest-chunks", async (route) => {
    const body = route.request().postDataJSON();
    const n = Array.isArray(body.chunks) ? body.chunks.length : 0;
    const seen = mockIngest.calls.get(page) || 0;
    mockIngest.calls.set(page, seen + 1);
    const json = plan
      ? plan(seen, body)
      : { ok: true, ingested: n, skipped: 0, collection_name: body.collection_name };
    await fulfillResponse(route, {
      status: json.__status || 200,
      contentType: "application/json",
      body: JSON.stringify(json),
    });
  });
}
mockIngest.calls = new Map();

// Record (and optionally answer) the payload-index request the ingest fires.
async function watchIndexCalls(page, respond) {
  const seen = [];
  await page.route(`**/collections/${COLL}/payload-indexes`, async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const body = route.request().postDataJSON();
    seen.push(body);
    if (!respond) return route.fallback();
    const json = respond(body);
    await fulfillResponse(route, {
      status: json.__status || 200,
      contentType: "application/json",
      body: JSON.stringify(json),
    });
  });
  return seen;
}

async function ingest(page) {
  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result .alert")).toContainText(/completed|failed/i);
}

test.describe("§7.1b Ingest → auto payload indexes", () => {
  test("IX1 — typed fields become payload indexes after a real ingest", async ({ page, request }) => {
    await resetCollection(request);
    await start(page);
    await uploadAndWait(page);
    await setType(page, "prezzo", "integer");
    await setType(page, "attivo", "bool");
    await setType(page, "tag", "keyword");
    await setType(page, "autore", "untyped");
    await expect(page.locator("#ing-create-indexes")).toBeChecked();
    await ingest(page);

    await expect(chip(page, "prezzo")).toHaveAttribute("data-status", "created");
    await expect(chip(page, "attivo")).toHaveAttribute("data-status", "created");
    await expect(chip(page, "tag")).toHaveAttribute("data-status", "created");
    await expect(chip(page, "source_id")).toHaveAttribute("data-status", "created");
    await expect(chip(page, "id")).toHaveAttribute("data-status", "created");
    await expect(chip(page, "autore")).toHaveCount(0);

    // AC-1: the indexes really exist in Qdrant, ordinal included.
    expect(await indexMap(request)).toEqual({
      prezzo: "integer",
      attivo: "bool",
      tag: "keyword",
      source_id: "keyword",
      id: "integer",
    });

    // AC-6: the console reloaded its field list, so the new fields are
    // filterable without a manual refresh (strict mode lists indexed fields
    // only, so their presence here IS the proof the index registered).
    await openTab(page, "query");
    await expect(page.locator("#ql-filter-add")).toBeEnabled();
    await page.locator("#ql-filter-add").click();
    const fields = await page.locator('#ql-filter-rows [data-role="field"] option')
      .evaluateAll((els) => els.map((e) => e.value));
    expect(fields).toEqual(expect.arrayContaining(["prezzo", "attivo", "tag"]));
  });

  test("IX2 — the toggle off skips the index phase entirely", async ({ page, request }) => {
    await resetCollection(request);
    await start(page);
    await mockIngest(page);
    const calls = await watchIndexCalls(page);
    await uploadAndWait(page);
    await setType(page, "prezzo", "integer");
    await page.locator("#ing-create-indexes").uncheck();
    await ingest(page);

    expect(calls).toHaveLength(0);
    await expect(page.locator("#ing-index-summary")).toHaveCount(0);
    expect(await indexMap(request)).toEqual({});
  });

  test("IX3 — a failed ingest creates no index", async ({ page, request }) => {
    await resetCollection(request);
    await start(page);
    // 130 chunks = 3 batches of 64: the second one fails, so the ingest stops
    // half-way and the index phase must never run.
    await mockIngest(page, (i, body) => (i === 1
      ? { ok: false, error: "simulated batch failure" }
      : { ok: true, ingested: body.chunks.length, skipped: 0 }));
    const calls = await watchIndexCalls(page);
    await uploadAndWait(page, "e2e_chunks_big.json");
    await setType(page, "m_int", "integer");
    await ingest(page);

    await expect(page.locator("#ing-result .alert")).toContainText(/failed/i);
    expect(calls).toHaveLength(0);
    await expect(page.locator("#ing-index-summary")).toHaveCount(0);
    expect(await indexMap(request)).toEqual({});
  });

  test("IX4 — an already-indexed field is locked in the preflight and never requested", async ({ page, request }) => {
    await resetCollection(request);
    await setupFetch(request, "POST", `/collections/${COLL}/payload-indexes`, {
      indexes: [{ field_name: "tag", field_schema: "keyword" }],
    });
    await start(page);
    await mockIngest(page);
    const calls = await watchIndexCalls(page);
    await uploadAndWait(page);

    // FR-2bis: preset to the existing schema, not editable, and it says why.
    const tagType = row(page, "tag").locator(".ing-map-type");
    await expect(tagType).toBeDisabled();
    await expect(tagType).toHaveValue("keyword");
    await expect(tagType).toHaveAttribute(
      "title", /Indexed as keyword — change it in Collections → Payload Indexes/);

    await setType(page, "prezzo", "integer");
    await ingest(page);

    expect(calls).toHaveLength(1);
    const requested = calls[0].indexes.map((i) => i.field_name);
    expect(requested).not.toContain("tag");
    expect(requested).toContain("prezzo");
    await expect(chip(page, "tag")).toHaveAttribute("data-status", "skipped");
    await expect(chip(page, "tag")).toContainText("already indexed");
    await expect(chip(page, "prezzo")).toHaveAttribute("data-status", "created");
    expect((await indexMap(request)).tag).toBe("keyword");
  });

  test("IX5 — renaming onto an indexed name locks the row, renaming away releases it", async ({ page, request }) => {
    await resetCollection(request);
    await setupFetch(request, "POST", `/collections/${COLL}/payload-indexes`, {
      indexes: [{ field_name: "tag", field_schema: "keyword" }],
    });
    await start(page);
    await uploadAndWait(page);

    const prezzoType = row(page, "prezzo").locator(".ing-map-type");
    await setType(page, "prezzo", "integer");
    await expect(prezzoType).toBeEnabled();

    await row(page, "prezzo").locator(".ing-map-name").fill("tag");
    await expect(prezzoType).toBeDisabled();
    await expect(prezzoType).toHaveValue("keyword");

    await row(page, "prezzo").locator(".ing-map-name").fill("prezzo");
    await expect(prezzoType).toBeEnabled();
    await expect(prezzoType).toHaveValue("integer");
  });

  test("IX6 — a race that replaces an index is reported as replaced with a warning", async ({ page, request }) => {
    await resetCollection(request);
    await start(page);
    await mockIngest(page);
    // Another client indexed `prezzo` as keyword between preflight and upload:
    // the middleware drops and recreates it, and reports it under `updated`.
    await watchIndexCalls(page, () => ({
      ok: true,
      collection_name: COLL,
      created: ["attivo"],
      updated: ["prezzo"],
      skipped: [],
      failed: [],
    }));
    await uploadAndWait(page);
    await setType(page, "prezzo", "integer");
    await setType(page, "attivo", "bool");
    await ingest(page);

    await expect(chip(page, "prezzo")).toHaveAttribute("data-status", "updated");
    await expect(chip(page, "prezzo")).toContainText("replaced");
    await expect(chip(page, "attivo")).toHaveAttribute("data-status", "created");
    await expect(page.locator("#ing-index-summary .caption.warn"))
      .toContainText(/replaced/i);
  });

  test("IX7 — a 409 marks the fields failed and leaves the ingest completed", async ({ page, request }) => {
    await resetCollection(request);
    await start(page);
    await mockIngest(page);
    await watchIndexCalls(page, () => ({
      __status: 409,
      detail: "Payload index schema mismatch for 'prezzo'",
    }));
    await uploadAndWait(page);
    await setType(page, "prezzo", "integer");
    await ingest(page);

    // FR-4: the ingest itself still counts as completed.
    await expect(page.locator("#ing-result .alert")).toContainText(/completed/i);
    await expect(chip(page, "prezzo")).toHaveAttribute("data-status", "failed");
    await expect(page.locator("#ing-index-summary .caption.warn"))
      .toContainText(/schema mismatch/i);
  });

  // K4 — Refresh of payload keys/indexed fields after ingest with new payload
  // fields. The Query panel's with_payload checkboxes and filter field list
  // must include the new fields once the ingest completes.
  test("K4 — new payload fields appear in the Query with_payload list after ingest", async ({ page, request }) => {
    await resetCollection(request);
    await start(page);
    await uploadAndWait(page);
    await setType(page, "prezzo", "integer");
    await setType(page, "attivo", "bool");
    await setType(page, "tag", "keyword");
    await ingest(page);
    await expect(page.locator("#ing-result .alert")).toContainText(/completed/i);

    // Move to the Query panel and verify the new fields are listed.
    await openTab(page, "query");
    // The list is filled from a collection sampling round-trip, so it is empty
    // for the first few hundred ms after the panel opens.
    await expect
      .poll(async () => page.locator("#ql-payload-fields input")
        .evaluateAll((els) => els.map((e) => e.value)), { timeout: 15_000 })
      .toEqual(expect.arrayContaining(["prezzo", "attivo", "tag"]));
  });
});
