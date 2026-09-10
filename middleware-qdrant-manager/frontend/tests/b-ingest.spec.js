import { fulfillResponse } from "./telemetry.mjs";
import { fixtureName, requireResults } from "./harness.mjs";
// Test book §7.1 — Points → Ingest (I1..I37) + spot-checks K1/K3/K6.
// Sandbox collections: e2e_b_ingest, e2e_b_alt, e2e_b_null, e2e_b_map, e2e_b_pass.
import { expect, test } from "./fixtures.js";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { boot, openTab, openPointsSubtab, selectActive, apiFetch, setupFetch } from "./helpers.js";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(DIR, "fixtures");

const MAIN = fixtureName("e2e_b_ingest");
const ALT = fixtureName("e2e_b_alt");

// K6 accumulator: every boot() pushes its page's console errors here.
const ALL_CONSOLE_ERRORS = [];

async function start(page, active = MAIN) {
  const errors = await boot(page, { active });
  ALL_CONSOLE_ERRORS.push(errors);
  await openTab(page, "points");
  await openPointsSubtab(page, "ingest");
  await expect(page.locator("#conn-collection")).toHaveValue(active);
  return errors;
}

// This series is about the preflight and the mapping table, so it opts out of
// the post-ingest index phase (covered by b2-ingest-indexes.spec.js): letting
// it run would index the sandbox collection as a side effect of I6/I27, and the
// later cases would then meet type-locked rows instead of free dropdowns.
// The toggle resets to checked on every file, so it is unchecked per upload.
async function upload(page, file) {
  await page.locator("#ing-file").setInputFiles(path.join(DATA, file));
  await page.locator("#ing-create-indexes").uncheck();
}

async function uploadAndWait(page, file) {
  await upload(page, file);
  await expect(page.locator("#ing-map-table")).toBeVisible();
}

// Exact label -> value map of the metric tiles inside a container.
async function tiles(page, containerId) {
  return page.evaluate((id) => {
    const out = {};
    for (const m of document.querySelectorAll(`#${id} .metric`)) {
      out[m.querySelector(".metric-label").textContent] =
        m.querySelector(".metric-value").textContent;
    }
    return out;
  }, containerId);
}

async function mapErrors(page) {
  return page.locator("#ing-map-errors p").allTextContents();
}

function row(page, field) {
  return page.locator(`#ing-map-table tbody tr[data-field="${field}"]`);
}
function srcRow(page) {
  return page.locator('#ing-map-table tbody tr[data-source-id="true"]');
}

async function setType(page, field, type) {
  await row(page, field).locator(".ing-map-type").selectOption(type);
}
async function setName(page, field, name) {
  await row(page, field).locator(".ing-map-name").fill(name);
}

// Intercept the ingest endpoint. `plan(i, body)` returns the JSON body to
// fulfil with for the i-th (0-based) batch. Records every request body.
async function mockIngest(page, plan) {
  const seen = [];
  await page.route("**/points/ingest-chunks", async (route) => {
    const body = route.request().postDataJSON();
    const i = seen.length;
    seen.push(body);
    const n = Array.isArray(body.chunks) ? body.chunks.length : 0;
    const json = plan
      ? plan(i, body)
      : { ok: true, ingested: n, skipped: 0, collection_name: body.collection_name };
    if (json && json.__delay) await new Promise((r) => setTimeout(r, json.__delay));
    await fulfillResponse(route, {
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: true, ingested: n, skipped: 0, collection_name: body.collection_name,
        ...(json || {}),
      }),
    });
  });
  return seen;
}

// ── I1 ────────────────────────────────────────────────────────────────────────
test("I1 — upload auto-runs preflight and renders the mapping table", async ({ page }) => {
  await start(page);
  // Exercise the visible control, not only the hidden file input used by the
  // remaining mapping-focused cases.
  const chooserPromise = page.waitForEvent("filechooser");
  await page.locator("#ing-file-btn").click();
  const chooser = await chooserPromise;
  await chooser.setFiles(path.join(DATA, "e2e_chunks_ok.json"));
  await expect(page.locator("#ing-map-table")).toBeVisible();

  await expect(page.locator("#ing-preflight-result .alert-success"))
    .toContainText("Preflight passed — file ready to ingest");
  const t = await tiles(page, "ing-preflight-result");
  expect(t).toMatchObject({
    chunks: "25", valid: "25", invalid: "0",
    "target collection": MAIN, "source file": "e2e_chunks_ok.json",
  });
  // No separate Analyze step exists.
  await expect(page.locator("#pt-ingest button", { hasText: /analy[sz]e/i })).toHaveCount(0);
  await expect(page.locator("#ing-confirm-submit")).toBeVisible();
  await expect(page.locator("#ing-confirm-submit")).toBeEnabled();
});

// ── I2 ────────────────────────────────────────────────────────────────────────
test("I2 — synthetic source_id row: FIELD fixed, NAME editable, TYPE keyword, value = file stem", async ({ page }) => {
  await start(page);
  const seen = await mockIngest(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");

  const first = page.locator("#ing-map-table tbody tr").first();
  await expect(first).toHaveAttribute("data-source-id", "true");
  await expect(first.locator("td").first()).toHaveText("source_id");
  // FIELD cell is a plain read-only cell (no form control).
  await expect(first.locator("td").first().locator("input, select")).toHaveCount(0);
  await expect(first.locator(".ing-map-name")).toHaveValue("source_id");
  await expect(first.locator(".ing-map-name")).toBeEditable();
  await expect(first.locator(".ing-map-type")).toHaveValue("keyword");

  // The value actually sent is the filename stem.
  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");
  expect(seen[0].source_id).toBe("e2e_chunks_ok");
  expect(seen[0].source_id_key).toBe("source_id");
  expect(seen[0].source_id_type).toBe("keyword");
});

// ── I3 ────────────────────────────────────────────────────────────────────────
test("I3 — field rows and inferred default TYPE", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");

  const got = await page.evaluate(() => {
    const out = {};
    for (const tr of document.querySelectorAll("#ing-map-table tbody tr")) {
      if (tr.dataset.sourceId === "true") continue;
      out[tr.dataset.field] = tr.querySelector(".ing-map-type").value;
    }
    return out;
  });
  expect(got).toEqual({
    // The chunk-ordinal row is synthetic like source_id, but unlike it keeps a
    // data-field, so it shows up here with its fixed type.
    id: "integer",
    m_str: "keyword", m_int: "integer", m_num: "float", m_bool: "bool",
    m_date: "keyword", m_uuid: "keyword", m_geo: "untyped", m_partial: "keyword",
  });
  // TYPE option set is the documented one.
  const opts = await row(page, "m_str").locator(".ing-map-type option")
    .evaluateAll((els) => els.map((e) => `${e.value}|${e.textContent}`));
  expect(opts).toEqual([
    "keyword|keyword", "text|text", "integer|integer", "float|float", "bool|bool",
    "datetime|datetime", "uuid|uuid", "geo|geo", "untyped|— untyped",
  ]);
});

// ── I4 ────────────────────────────────────────────────────────────────────────
test("I4 — excluded keys are not mappable rows", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_excluded.json");

  // `id` has a synthetic row of its own (second, after source_id); `text` has
  // none. The chunker-internal keys are ordinary rows while CHUNK_EXCLUDE_FIELDS
  // is empty backend-side.
  const fields = await page.locator("#ing-map-table tbody tr")
    .evaluateAll((els) => els.map((e) => e.dataset.field));
  expect(fields).toEqual([
    "source_id", "id", "m_str", "overflow_details", "overlap_sentences_count", "split_metadata",
  ]);
  await expect(row(page, "text")).toHaveCount(0);
  // The ordinal row's TYPE is fixed — the server stores it unconverted.
  await expect(row(page, "id").locator(".ing-map-type")).toBeDisabled();
});

// ── I5 ────────────────────────────────────────────────────────────────────────
test("I5 — mixed / null / array / object fields default to — untyped", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_nulls.json");

  const got = await page.evaluate(() => {
    const out = {};
    for (const tr of document.querySelectorAll("#ing-map-table tbody tr")) {
      if (tr.dataset.sourceId === "true") continue;
      out[tr.dataset.field] = tr.querySelector(".ing-map-type").value;
    }
    return out;
  });
  expect(got).toEqual({
    id: "integer",
    m_null: "untyped", m_mixed: "untyped", m_arr: "untyped", m_obj: "untyped",
  });
  expect(await mapErrors(page)).toEqual([]);
});

// ── I6 ────────────────────────────────────────────────────────────────────────
test("I6 — real ingest with the default mapping (progress + counts) [+K3]", async ({ page }) => {
  test.setTimeout(420_000);
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");

  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result .reindex-progress")).toBeVisible();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED", { timeout: 360_000 });
  const t = await tiles(page, "ing-result");
  expect(t.progress).toBe("100%");
  expect(t.chunks).toBe("25 / 25");
  expect(t).toHaveProperty("skipped");

  // K3 — Report 01 auto-refreshed without any manual action.
  await openTab(page, "collections");
  const r = await tiles(page, "cc-report");
  expect(Number(r.points)).toBeGreaterThanOrEqual(25);
});

// ── I7 ────────────────────────────────────────────────────────────────────────
test("I7 — renaming validates live without re-rendering the table", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");

  await page.evaluate(() => { document.getElementById("ing-map-table").dataset.marker = "keep"; });
  const input = row(page, "m_str").locator(".ing-map-name");
  await input.click();
  await input.press("End");
  await input.pressSequentially("_renamed", { delay: 30 });

  await expect(input).toBeFocused();
  await expect(input).toHaveValue("m_str_renamed");
  // Same DOM node survived every keystroke → no re-render.
  await expect(page.locator("#ing-map-table")).toHaveAttribute("data-marker", "keep");
  expect(await mapErrors(page)).toEqual([]);
  await expect(page.locator("#ing-confirm-submit")).toBeEnabled();
});

// ── I8 ────────────────────────────────────────────────────────────────────────
test("I8 — empty field NAME blocks confirm", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await setName(page, "m_str", "");
  expect(await mapErrors(page)).toContain("field 'm_str': name is required");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

// ── I9 ────────────────────────────────────────────────────────────────────────
// A field NAME may only clash with what applyIngestFieldMapping passes through
// untouched. The three chunker-internal keys are ordinary payload names while
// CHUNK_EXCLUDE_FIELDS is empty, so they are accepted here — I12 still blocks
// all five for the source_id row, which the middleware validates separately.
test("I9 — the two reserved NAMEs are blocked", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  for (const reserved of ["id", "text"]) {
    await setName(page, "m_str", reserved);
    expect(await mapErrors(page), `reserved=${reserved}`)
      .toContain(`field 'm_str': name '${reserved}' is reserved`);
    await expect(page.locator("#ing-confirm-submit"), `reserved=${reserved}`).toBeDisabled();
  }
  await setName(page, "m_str", "m_str");
  await expect(page.locator("#ing-confirm-submit")).toBeEnabled();
});

// ── I10 ───────────────────────────────────────────────────────────────────────
test("I10 — duplicate NAME between two rows blocks confirm", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await setName(page, "m_int", "m_str");
  expect(await mapErrors(page))
    .toContain("field 'm_int': name 'm_str' duplicates field 'm_str'");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

// ── I11..I15 (source_id row) ──────────────────────────────────────────────────
test("I11 — empty source_id NAME blocks confirm", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await srcRow(page).locator(".ing-map-name").fill("");
  expect(await mapErrors(page)).toContain("source_id: name is required");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("I12 — source_id NAME set to a reserved key", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  for (const reserved of
    ["id", "text", "overflow_details", "overlap_sentences_count", "split_metadata"]) {
    await srcRow(page).locator(".ing-map-name").fill(reserved);
    expect(await mapErrors(page), `reserved=${reserved}`)
      .toContain(`source_id: name '${reserved}' is reserved`);
    await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
  }
});

test("I13 — source_id NAME colliding with a field NAME", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await srcRow(page).locator(".ing-map-name").fill("m_str");
  expect(await mapErrors(page))
    .toContain("source_id: name 'm_str' duplicates field 'm_str'");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("I14 — source_id TYPE integer with a non-numeric file stem", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await srcRow(page).locator(".ing-map-type").selectOption("integer");
  expect(await mapErrors(page))
    .toContain("source_id → integer: 'e2e_chunks_ok' is not convertible");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("I15 — source_id TYPE uuid with a non-UUID stem", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await srcRow(page).locator(".ing-map-type").selectOption("uuid");
  expect(await mapErrors(page))
    .toContain("source_id → uuid: 'e2e_chunks_ok' is not convertible");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

// ── I16 ───────────────────────────────────────────────────────────────────────
test("I16 — only the FIRST offending value per field is reported", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_mixed.json");
  await setType(page, "m_int", "integer");

  const errs = await mapErrors(page);
  expect(errs).toContain('field \'m_int\' → integer: value "abc" is not convertible');
  expect(errs.filter((e) => e.includes("m_int"))).toHaveLength(1);
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

// ── I17..I24 coercion matrix ──────────────────────────────────────────────────
test("I17 — integer on a non-integer float is rejected", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await setType(page, "m_num", "integer");
  expect(await mapErrors(page))
    .toContain("field 'm_num' → integer: value 1.5 is not convertible");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("I18 — float on a non-numeric string is rejected", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await setType(page, "m_str", "float");
  expect(await mapErrors(page))
    .toContain('field \'m_str\' → float: value "s1" is not convertible');
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("I19 — bool on a value outside bool/0/1/'true'/'false' is rejected", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_mixed.json");
  await setType(page, "m_bool", "bool");
  expect(await mapErrors(page))
    .toContain('field \'m_bool\' → bool: value "maybe" is not convertible');
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("I20 — keyword/text on array or object is rejected", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_nulls.json");
  await setType(page, "m_arr", "keyword");
  expect(await mapErrors(page))
    .toContain("field 'm_arr' → keyword: value [1,2] is not convertible");
  await setType(page, "m_arr", "untyped");
  await setType(page, "m_obj", "text");
  expect(await mapErrors(page))
    .toContain('field \'m_obj\' → text: value {"k":"v"} is not convertible');
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("I21 — keyword on null PASSES and becomes the string \"null\"", async ({ page }) => {
  test.setTimeout(300_000);
  await start(page, fixtureName("e2e_b_null"));
  await uploadAndWait(page, "e2e_chunks_nulls.json");
  await setType(page, "m_null", "keyword");

  expect(await mapErrors(page)).toEqual([]);
  await expect(page.locator("#ing-confirm-submit")).toBeEnabled();

  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED", { timeout: 240_000 });

  // Ingest assigns deterministic UUID point ids (the chunk "id" is NOT the point
  // id), so read the stored payload back with a scroll, then confirm through the
  // UI Get subtab using the real id.
  const scrolled = await setupFetch(page.request, "POST", "/points/scroll",
    { collection_name: fixtureName("e2e_b_null"), limit: 10, with_payload: true });
  const hits = requireResults(await scrolled.json());
  const withNull = hits.filter((h) => "m_null" in (h.payload || {}));
  expect(withNull.length, "no ingested point carries m_null").toBeGreaterThan(0);
  for (const h of withNull) expect(h.payload.m_null).toBe("null");
  await openPointsSubtab(page, "get");
  await page.locator("#get-id").fill(String(withNull[0].id));
  await page.locator("#get-submit").click();
  await expect(page.locator("#get-result")).toContainText('"m_null": "null"', { timeout: 60_000 });
});

test("I22 — datetime on an unparsable string is rejected", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_mixed.json");
  await setType(page, "m_date", "datetime");
  expect(await mapErrors(page))
    .toContain('field \'m_date\' → datetime: value "lunedi" is not convertible');
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("I23 — uuid on a non-UUID string is rejected", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_mixed.json");
  await setType(page, "m_uuid", "uuid");
  expect(await mapErrors(page))
    .toContain('field \'m_uuid\' → uuid: value "not-a-uuid" is not convertible');
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

test("I24 — geo on an array or a scalar is rejected", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_mixed.json");
  await setType(page, "m_geo", "geo");
  expect(await mapErrors(page))
    .toContain("field 'm_geo' → geo: value [1,2] is not convertible");
  await setType(page, "m_geo", "untyped");
  await setType(page, "m_int", "geo");
  expect(await mapErrors(page))
    .toContain("field 'm_int' → geo: value 1 is not convertible");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
});

// ── I25 ───────────────────────────────────────────────────────────────────────
test("I25 — — untyped passes any shape through unchanged", async ({ page }) => {
  await start(page);
  const seen = await mockIngest(page);
  await uploadAndWait(page, "e2e_chunks_nulls.json");
  expect(await mapErrors(page)).toEqual([]);

  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");
  const sent = seen[0].chunks;
  expect(sent[0].m_null).toBeNull();
  expect(sent[0].m_mixed).toBe(1);
  expect(sent[1].m_mixed).toBe("two");
  expect(sent[2].m_arr).toEqual([1, 2]);
  expect(sent[2].m_obj).toEqual({ k: "v" });
});

// ── I26 ───────────────────────────────────────────────────────────────────────
test("I26 — a field present in only some chunks does not error", async ({ page }) => {
  await start(page);
  const seen = await mockIngest(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");

  await expect(row(page, "m_partial").locator(".ing-map-type")).toHaveValue("keyword");
  await setType(page, "m_partial", "text");
  expect(await mapErrors(page)).toEqual([]);
  await expect(page.locator("#ing-confirm-submit")).toBeEnabled();

  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");
  const sent = seen[0].chunks;
  expect(sent.filter((c) => "m_partial" in c)).toHaveLength(10);
  expect(sent[0].m_partial).toBe("p1");
  expect(sent[24]).not.toHaveProperty("m_partial");
});

// ── I27 ───────────────────────────────────────────────────────────────────────
test("I27 — fixing every error re-enables confirm; rename + coercion land in Qdrant",
  async ({ page, request }) => {
    test.setTimeout(420_000);
    await start(page, fixtureName("e2e_b_map"));
    await uploadAndWait(page, "e2e_chunks_ok.json");

    // Break it first.
    await setType(page, "m_int", "uuid");
    await setName(page, "m_num", "text");
    expect((await mapErrors(page)).length).toBeGreaterThanOrEqual(2);
    await expect(page.locator("#ing-confirm-submit")).toBeDisabled();

    // Fix everything + apply the real renames / coercions under test.
    await setType(page, "m_int", "integer");
    await setName(page, "m_num", "m_num_f");
    await setName(page, "m_str", "meta_label");
    await setType(page, "m_date", "datetime");
    await setType(page, "m_uuid", "uuid");
    await setType(page, "m_geo", "geo");
    await srcRow(page).locator(".ing-map-name").fill("doc_src");

    expect(await mapErrors(page)).toEqual([]);
    await expect(page.locator("#ing-confirm-submit")).toBeEnabled();

    await page.locator("#ing-confirm-submit").click();
    await expect(page.locator("#ing-result")).toContainText("COMPLETED", { timeout: 360_000 });

    // Ingest assigns deterministic UUID point ids, so locate the chunk by its
    // renamed payload instead of assuming the chunk "id" became the point id.
    const scrolled = await apiFetch(request, "POST", "/points/scroll",
      { collection_name: fixtureName("e2e_b_map"), limit: 100, with_payload: true });
    expect(scrolled.status()).toBe(200);
    const hits = requireResults(await scrolled.json());
    const hit = hits.find((h) => (h.payload || {}).meta_label === "s1");
    expect(hit, "chunk 1 not found after ingest").toBeTruthy();

    // Verify through the UI (Get), using the real point id …
    await openPointsSubtab(page, "get");
    await page.locator("#get-id").fill(String(hit.id));
    await page.locator("#get-submit").click();
    const got = page.locator("#get-result");
    await expect(got).toContainText('"meta_label": "s1"', { timeout: 60_000 });
    await expect(got).toContainText('"doc_src": "e2e_chunks_ok"');
    await expect(got).toContainText('"m_num_f": 1.5');
    await expect(got).not.toContainText('"m_str"');

    // … and confirm the stored payload directly.
    const payload = hit.payload || {};
    expect(payload.meta_label).toBe("s1");
    expect(payload.doc_src).toBe("e2e_chunks_ok");
    expect(payload.m_int).toBe(1);
    expect(payload).not.toHaveProperty("m_str");
    expect(payload).not.toHaveProperty("m_num");
  });

// ── I28 ───────────────────────────────────────────────────────────────────────
test("I28 — swapping the file mid-ingest aborts the loop", async ({ page }) => {
  await start(page);
  const seen = await mockIngest(page, () => ({ __delay: 2500 }));
  await uploadAndWait(page, "e2e_chunks_big.json");

  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("INGESTING");
  await page.waitForTimeout(400);
  await upload(page, "e2e_chunks_ok.json");
  await page.waitForTimeout(6000);

  expect(seen).toHaveLength(1);                    // no batch of the old file after the swap
  expect(seen[0].chunks).toHaveLength(64);
  await expect(page.locator("#ing-preflight-result")).toContainText("e2e_chunks_ok.json");
  const t = await tiles(page, "ing-preflight-result");
  expect(t.chunks).toBe("25");
});

// ── I29 ───────────────────────────────────────────────────────────────────────
test("I29 — changing the active collection mid-ingest aborts the loop", async ({ page }) => {
  await start(page);
  const seen = await mockIngest(page, () => ({ __delay: 2500 }));
  await uploadAndWait(page, "e2e_chunks_big.json");

  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("INGESTING");
  await page.waitForTimeout(400);
  await selectActive(page, ALT);
  await page.waitForTimeout(6000);

  expect(seen).toHaveLength(1);
  expect(seen.every((b) => b.collection_name === MAIN)).toBe(true);
  expect(seen.some((b) => b.collection_name === ALT)).toBe(false);
  expect(await page.locator("#ing-result").innerHTML()).toBe("");
});

// ── I30 ───────────────────────────────────────────────────────────────────────
test("I30 — malformed JSON: readable error, no preflight, confirm hidden", async ({ page }) => {
  await start(page);
  await upload(page, "e2e_chunks_bad.json");
  await expect(page.locator("#ing-result .alert-error"))
    .toContainText("'e2e_chunks_bad.json' is not a valid JSON file");
  expect(await page.locator("#ing-preflight-result").innerHTML()).toBe("");
  await expect(page.locator("#ing-confirm-submit")).toBeHidden();
});

// ── I31 ───────────────────────────────────────────────────────────────────────
test("I31 — invalid chunks: preflight blocked, confirm hidden", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_invalid.json");
  await expect(page.locator("#ing-preflight-result .alert-error"))
    .toContainText("Preflight blocked ingest");
  const t = await tiles(page, "ing-preflight-result");
  expect(t).toMatchObject({ chunks: "2", valid: "1", invalid: "1" });
  await expect(page.locator("#ing-preflight-result"))
    .toContainText("chunk[0]: 'id' must be an int");
  await expect(page.locator("#ing-confirm-submit")).toBeHidden();
});

// ── I32 ───────────────────────────────────────────────────────────────────────
test("I32 — zero chunks: preflight blocked, confirm unavailable", async ({ page }) => {
  await start(page);
  await upload(page, "e2e_chunks_empty.json");
  await expect(page.locator("#ing-preflight-result .alert-error"))
    .toContainText("Preflight blocked ingest");
  const t = await tiles(page, "ing-preflight-result");
  expect(t).toMatchObject({ chunks: "0", valid: "0", invalid: "0" });
  await expect(page.locator("#ing-preflight-result")).toContainText("'chunks' is empty");
  await expect(page.locator("#ing-confirm-submit")).toBeHidden();
});

// ── I33 ───────────────────────────────────────────────────────────────────────
test("I33 — mid-batch failure restores the mapping table with edits preserved", async ({ page }) => {
  await start(page);
  const seen = await mockIngest(page, (i) =>
    i === 1 ? { ok: false, ingested: 0, error: "simulated batch failure" } : null);
  await uploadAndWait(page, "e2e_chunks_big.json");
  await setName(page, "m_str", "renamed_str");
  await setType(page, "m_int", "float");

  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result .alert-error")).toContainText(
    "Ingest failed after 64 of 130 chunks — simulated batch failure");
  expect(seen).toHaveLength(2);

  const pre = page.locator("#ing-preflight-result");
  await expect(pre).toBeVisible();
  await expect(page.locator("#ing-map-table")).toBeVisible();
  await expect(row(page, "m_str").locator(".ing-map-name")).toHaveValue("renamed_str");
  await expect(row(page, "m_int").locator(".ing-map-type")).toHaveValue("float");
  const t = await tiles(page, "ing-result");
  expect(t.chunks).toBe("64 / 130");
});

// ── I34 ───────────────────────────────────────────────────────────────────────
test("I34 — retry after a mid-batch failure re-sends without re-uploading", async ({ page }) => {
  await start(page);
  // Phase 1 fails the 2nd batch; phase 2 (the retry) lets everything through.
  const seen2 = [];
  let phase = 1;
  await page.route("**/points/ingest-chunks", async (route) => {
    const body = route.request().postDataJSON();
    seen2.push({
      phase, n: body.chunks.length,
      renamed: "renamed_str" in body.chunks[0] && !("m_str" in body.chunks[0]),
    });
    const fail = phase === 1 && seen2.filter((s) => s.phase === 1).length === 2;
    await fulfillResponse(route, {
      status: 200, contentType: "application/json",
      body: JSON.stringify(fail
        ? { ok: false, ingested: 0, skipped: 0, collection_name: body.collection_name,
            error: "simulated batch failure" }
        : { ok: true, ingested: body.chunks.length, skipped: 0,
            collection_name: body.collection_name }),
    });
  });

  await uploadAndWait(page, "e2e_chunks_big.json");
  await setName(page, "m_str", "renamed_str");
  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result .alert-error")).toContainText("Ingest failed after 64");

  // Retry — same file input untouched.
  phase = 2;
  await expect(page.locator("#ing-confirm-submit")).toBeEnabled();
  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");
  const t = await tiles(page, "ing-result");
  expect(t.chunks).toBe("130 / 130");
  expect(seen2.filter((s) => s.phase === 2)).toHaveLength(3);
  expect(seen2.filter((s) => s.phase === 2).map((s) => s.n)).toEqual([64, 64, 2]);
  // The retried batches carry the rename made before the failure.
  expect(seen2.every((s) => s.renamed)).toBe(true);
});

// ── I35 ───────────────────────────────────────────────────────────────────────
test("I35 — a completed ingest EMPTIES the preflight (not merely hides it)", async ({ page }) => {
  await start(page);
  const seen = await mockIngest(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");

  const pre = page.locator("#ing-preflight-result");
  expect(await pre.innerHTML()).toBe("");
  expect(await pre.evaluate((el) => el.hidden)).toBe(false);
  await expect(page.locator("#ing-map-table")).toHaveCount(0);
  // Confirm is hidden AND disabled → stale chunks cannot be re-sent.
  await expect(page.locator("#ing-confirm-submit")).toBeHidden();
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
  await page.locator("#ing-confirm-submit").dispatchEvent("click");
  await page.waitForTimeout(500);
  expect(seen).toHaveLength(1);
});

// ── I36 ───────────────────────────────────────────────────────────────────────
test("I36 — the COMPLETED block auto-resets after ~5s", async ({ page }) => {
  await start(page);
  await mockIngest(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");

  await page.waitForTimeout(3000);
  expect(await page.locator("#ing-result").innerHTML()).not.toBe("");
  await expect
    .poll(async () => page.locator("#ing-result").innerHTML(), { timeout: 8000 })
    .toBe("");
});

// ── I37 ───────────────────────────────────────────────────────────────────────
test("I37 — switching collection inside the 5s window does not wipe the new result", async ({ page }) => {
  await start(page);
  await mockIngest(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");
  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");

  await page.waitForTimeout(500);
  await selectActive(page, ALT);
  // Produce a result that belongs to the NEW collection.
  await upload(page, "e2e_chunks_bad.json");
  await expect(page.locator("#ing-result")).toContainText("not a valid JSON file");

  await page.waitForTimeout(6500);   // past the original 5s timer
  await expect(page.locator("#ing-result")).toContainText("not a valid JSON file");
});

// ── I38 ───────────────────────────────────────────────────────────────────────
test("I38 — the id row's NAME travels as chunk_id_key, chunks untouched", async ({ page }) => {
  await start(page);
  const seen = await mockIngest(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");

  await expect(row(page, "id").locator(".ing-map-name")).toHaveValue("id");
  expect(await mapErrors(page)).toEqual([]);

  await setName(page, "id", "id_chunk");
  expect(await mapErrors(page)).toEqual([]);
  await page.locator("#ing-confirm-submit").click();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");

  // The client never rewrites the chunk: _validate_chunks 400s unless `id` is
  // still an integer, and the point id derives from it. The middleware relocates
  // the ordinal to chunk_id_key on write, so no duplicate reaches the payload.
  expect(seen[0].chunk_id_key).toBe("id_chunk");
  expect(typeof seen[0].chunks[0].id).toBe("number");
  expect(seen[0].chunks[0].id_chunk).toBeUndefined();
});

// ── I39 ───────────────────────────────────────────────────────────────────────
test("I39 — the id row rejects a reserved or colliding NAME", async ({ page }) => {
  await start(page);
  await uploadAndWait(page, "e2e_chunks_ok.json");

  await setName(page, "id", "");
  // Rendered messages drop the trailing period (uppercaseMessage).
  expect(await mapErrors(page)).toContain("id: name is required");
  await expect(page.locator("#ing-confirm-submit")).toBeDisabled();

  await setName(page, "id", "split_metadata");
  expect(await mapErrors(page)).toContain("id: name 'split_metadata' is reserved");

  await setName(page, "id", "source_id");
  expect(await mapErrors(page)).toContain("id: name 'source_id' duplicates the source_id name");

  await setName(page, "id", "m_str");
  expect(await mapErrors(page)).toContain("id: name 'm_str' duplicates field 'm_str'");

  await setName(page, "id", "id");
  expect(await mapErrors(page)).toEqual([]);
  await expect(page.locator("#ing-confirm-submit")).toBeEnabled();
});

// ── K1 ────────────────────────────────────────────────────────────────────────
test("K1 — double-click on Confirm fires a single ingest", async ({ page }) => {
  await start(page);
  const seen = await mockIngest(page, () => ({ __delay: 1200 }));
  await uploadAndWait(page, "e2e_chunks_ok.json");

  const btn = page.locator("#ing-confirm-submit");
  await btn.dblclick();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED", { timeout: 20_000 });
  expect(seen).toHaveLength(1);
});

// ── K6 ────────────────────────────────────────────────────────────────────────
test("K6 — no browser console errors across the ingest series", async () => {
  const flat = ALL_CONSOLE_ERRORS.flat();
  expect(flat, `console errors: ${JSON.stringify(flat, null, 2)}`).toEqual([]);
});
