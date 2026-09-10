import { transformJsonResponse } from "./telemetry.mjs";
import { fulfillResponse, abortResponse } from "./telemetry.mjs";
import { fixtureName, fixtureOverride } from "./harness.mjs";
import { expectHttpFailure } from "./telemetry.mjs";
// Test book §6 — Tab COLLECTIONS (C / T / PI / X / D) + §11 cross-cutting spot checks.
// Sandbox: e2e_a_* only.
import { test, expect } from "./fixtures.js";
import fs from "node:fs";
import { boot, openTab, apiFetch, setupFetch, BASE_URL, TOKEN } from "./helpers.js";

const KEEP = fixtureName("e2e_a_keep");
const VICTIM = fixtureName("e2e_a_victim");
const EMPTY = fixtureName("e2e_a_empty");

const bootA = (page, active) => boot(page, { active });

async function openCollections(page) {
  await openTab(page, "collections");
}

async function reportMetrics(page) {
  return page.evaluate(() => {
    const out = {};
    document.querySelectorAll("#cc-report .metric").forEach((m) => {
      const l = m.querySelector(".metric-label")?.textContent?.trim();
      const v = m.querySelector(".metric-value")?.textContent?.trim();
      if (l) out[l] = v;
    });
    return out;
  });
}

async function expectReportFor(page, name) {
  await expect
    .poll(async () => (await reportMetrics(page))["collection name"], { timeout: 25_000 })
    .toBe(name);
}

async function indexTable(page) {
  return page.evaluate(() => {
    const out = {};
    document.querySelectorAll("#pi-table tbody tr").forEach((tr) => {
      const cells = [...tr.querySelectorAll("td")].map((td) => td.textContent.trim());
      if (cells.length >= 2) out[cells[0]] = cells[1];
    });
    return out;
  });
}

// Create-form toggles. Keys mirror the book's toggle names.
const CC_IDS = {
  dense_on_disk: "#cc-dense",
  sparse_on_disk: "#cc-sparse-disk",
  payload_on_disk: "#cc-payload-disk",
  quantization: "#cc-quant",
  sparse_idf: "#cc-idf",
};
const REPORT_LABEL = {
  dense_on_disk: "dense on disk",
  sparse_on_disk: "sparse on disk",
  payload_on_disk: "payload on disk",
  quantization: "quantization",
  sparse_idf: "sparse idf",
};
const CU_IDS = {
  dense_on_disk: "#cu-dense",
  sparse_on_disk: "#cu-sparse-disk",
  payload_on_disk: "#cu-payload-disk",
  quantization: "#cu-quant",
  sparse_idf: "#cu-idf",
};

async function setCreateToggles(page, toggles) {
  for (const [k, v] of Object.entries(toggles)) {
    await page.locator(CC_IDS[k]).selectOption(String(v));
  }
}

async function expectBadges(page, toggles) {
  const m = await reportMetrics(page);
  for (const [k, v] of Object.entries(toggles)) {
    expect(m[REPORT_LABEL[k]], `report badge '${REPORT_LABEL[k]}'`).toBe(String(v));
  }
}

function countRequests(page, pathname, method) {
  const hits = [];
  page.on("request", (r) => {
    try {
      const u = new URL(r.url());
      if (u.pathname === pathname && (!method || r.method() === method)) hits.push(r.url());
    } catch { /* ignore */ }
  });
  return hits;
}

const ALL_TRUE = {
  dense_on_disk: true, sparse_on_disk: true, payload_on_disk: true,
  quantization: true, sparse_idf: true,
};
const ALL_FALSE = {
  dense_on_disk: false, sparse_on_disk: false, payload_on_disk: false,
  quantization: false, sparse_idf: false,
};
const MIX_A = {
  dense_on_disk: false, sparse_on_disk: true, payload_on_disk: false,
  quantization: true, sparse_idf: false,
};
const MIX_B = {
  dense_on_disk: true, sparse_on_disk: false, payload_on_disk: true,
  quantization: false, sparse_idf: true,
};

// ───────────────────────── §6.1 C — 03 Create / Recreate ─────────────────────

test("C1 — create e2e_a_new1 with default toggles (all-true)", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  // default toggles must already read true (backend defaults, set in initCollections)
  for (const id of Object.values(CC_IDS)) {
    await expect(page.locator(id)).toHaveValue("true");
  }
  const posts = countRequests(page, "/collections", "POST");
  await page.locator("#cc-name").fill(fixtureName("e2e_a_new1"));
  await page.locator("#cc-submit").click();
  await expect(page.locator("#cc-create-result .alert-success")).toContainText(
    /collection created/i);
  // active switches to the new collection + report auto-refresh (K3)
  await expect(page.locator("#ql-collection")).toHaveValue(fixtureName("e2e_a_new1"));
  await expectReportFor(page, fixtureName("e2e_a_new1"));
  await expectBadges(page, ALL_TRUE);
  expect((await reportMetrics(page)).points).toBe("0");
  // dropdown refresh
  await expect(page.locator(`#conn-collection option[value="${fixtureName("e2e_a_new1")}"]`)).toHaveCount(1);
  expect(posts.length, "exactly one POST /collections").toBe(1);
});

test("C2 — create e2e_a_new2 with all-false toggles", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cc-name").fill(fixtureName("e2e_a_new2"));
  await setCreateToggles(page, ALL_FALSE);
  await page.locator("#cc-submit").click();
  await expect(page.locator("#cc-create-result .alert-success")).toContainText(
    /collection created/i);
  await expectReportFor(page, fixtureName("e2e_a_new2"));
  await expectBadges(page, ALL_FALSE);
});

test("C3 — create e2e_a_new3 with mix A", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cc-name").fill(fixtureName("e2e_a_new3"));
  await setCreateToggles(page, MIX_A);
  await page.locator("#cc-submit").click();
  await expect(page.locator("#cc-create-result .alert-success")).toContainText(
    /collection created/i);
  await expectReportFor(page, fixtureName("e2e_a_new3"));
  await expectBadges(page, MIX_A);
});

test("C4 — create e2e_a_new4 with mix B", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cc-name").fill(fixtureName("e2e_a_new4"));
  await setCreateToggles(page, MIX_B);
  await page.locator("#cc-submit").click();
  await expect(page.locator("#cc-create-result .alert-success")).toContainText(
    /collection created/i);
  await expectReportFor(page, fixtureName("e2e_a_new4"));
  await expectBadges(page, MIX_B);
});

test("C5 — re-submit e2e_a_new1 with the same config → created=false, info alert", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cc-name").fill(fixtureName("e2e_a_new1"));
  await setCreateToggles(page, ALL_TRUE);
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname === "/collections"
      && r.request().method() === "POST"),
    page.locator("#cc-submit").click(),
  ]);
  expect(resp.status()).toBe(200);
  expect((await resp.json()).created).toBe(false);
  await expect(page.locator("#cc-create-result .alert-info")).toContainText(
    /already exists and already has these settings/i);
  await expect(page.locator("#cc-create-result .alert-info")).toContainText(/nothing to do/i);
});

test("C6 — submit existing e2e_a_new2 with different toggles, recreate OFF → 409 info", async ({ page }) => {
  expectHttpFailure(page, "POST", "/collections", 409);
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cc-name").fill(fixtureName("e2e_a_new2"));
  await setCreateToggles(page, ALL_TRUE); // new2 is all-false
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname === "/collections"
      && r.request().method() === "POST"),
    page.locator("#cc-submit").click(),
  ]);
  expect(resp.status()).toBe(409);
  await expect(page.locator("#cc-create-result .alert-info")).toContainText(
    /04 Schema Toggles/i);
  await expect(page.locator("#cc-create-result .alert-error")).toHaveCount(0);
  // collection intact: still all-false
  const report = await setupFetch(page.request, "GET", `/collections/${fixtureName("e2e_a_new2")}`);
  const data = await report.json();
  expect(data.dense.on_disk).toBe(false);
  expect(data.sparse.on_disk).toBe(false);
  expect(data.on_disk_payload).toBe(false);
});

test("C7 — recreate armed with an empty confirmation → Submit disabled", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cc-name").fill(VICTIM);
  await page.locator("#cc-recreate").check();
  await expect(page.locator("#cc-confirm-wrap")).toBeVisible();
  await expect(page.locator("#cc-confirm-label")).toContainText(VICTIM); // K7 label
  await expect(page.locator("#cc-confirm")).toHaveValue("");
  await expect(page.locator("#cc-submit")).toBeDisabled();
});

test("C8 — recreate with a wrong confirmation → Submit disabled + hint", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cc-name").fill(VICTIM);
  await page.locator("#cc-recreate").check();
  await page.locator("#cc-confirm").fill(fixtureName("e2e_a_victi"));
  await expect(page.locator("#cc-confirm-hint")).toBeVisible();
  await expect(page.locator("#cc-confirm-hint")).toContainText(/does not match/i);
  await expect(page.locator("#cc-submit")).toBeDisabled();
});

test("C9 — recreate e2e_a_victim with the exact confirmation", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  // precondition: victim has points
  const before = await (await setupFetch(page.request, "GET", `/collections/${VICTIM}`)).json();
  expect(before.points_count).toBeGreaterThan(0);

  await page.locator("#cc-name").fill(VICTIM);
  await page.locator("#cc-recreate").check();
  await page.locator("#cc-confirm").fill(VICTIM);
  await expect(page.locator("#cc-confirm-hint")).toBeHidden();
  await expect(page.locator("#cc-submit")).toBeEnabled();
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname === "/collections"
      && r.request().method() === "POST"),
    page.locator("#cc-submit").click(),
  ]);
  expect(resp.status()).toBe(200);
  expect((await resp.json()).created).toBe(true);
  await expect(page.locator("#cc-create-result .alert-success")).toContainText(
    /collection created/i);
  // recreate auto-disarmed + confirmation emptied
  await expect(page.locator("#cc-recreate")).not.toBeChecked();
  await expect(page.locator("#cc-confirm")).toHaveValue("");
  await expect(page.locator("#cc-confirm-wrap")).toBeHidden();
  // points_count → 0 (K3 auto-refresh)
  await expectReportFor(page, VICTIM);
  await expect
    .poll(async () => (await reportMetrics(page)).points, { timeout: 20_000 })
    .toBe("0");
});

test("C10 — empty collection name → Submit disabled", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expect(page.locator("#cc-name")).toHaveValue("");
  await expect(page.locator("#cc-submit")).toBeDisabled();
  await page.locator("#cc-name").fill(fixtureName("e2e_a_tmp"));
  await expect(page.locator("#cc-submit")).toBeEnabled();
  await page.locator("#cc-name").fill("   ");
  await expect(page.locator("#cc-submit")).toBeDisabled();
});

test("C11 — collection created under the active profile carries that profile's contract", async ({ page }) => {
  // The BASIC↔ADVANCED switch itself is out of scope for this agent (no .env /
  // container changes allowed): only the observable half is asserted here.
  await bootA(page, fixtureName("e2e_a_new1"));
  await openCollections(page);
  await expectReportFor(page, fixtureName("e2e_a_new1"));
  const m = await reportMetrics(page);
  expect(m["embedding contract"]).toBe("COMPATIBLE");
  expect(["basic", "advanced"]).toContain(m["active profile"]);
  expect(m["stored profile"]).toBe(m["active profile"]);
  await expect(page.locator("#service-gate")).toBeEmpty();
});

test("C12 — [API] create without the schema toggles → 400", async ({ request }) => {
  const resp = await apiFetch(request, "POST", "/collections",
    { collection_name: fixtureName("e2e_a_c12probe"), recreate: false });
  expect(resp.status()).toBe(400);
  const body = await resp.json();
  const msg = JSON.stringify(body);
  // The real contract, now also what the book states (§6.1 C12): the five toggles
  // are REQUIRED fields, so the refusal is a per-field validation error and not a
  // single "at least one schema toggle" message.
  expect(msg).toMatch(/Field required/);
  for (const toggle of ["dense_on_disk", "sparse_on_disk", "quantization", "sparse_idf"]) {
    expect(msg, `${toggle} not named in the refusal`).toContain(toggle);
  }
});

test("C13 — spaces-only name trims to empty and blocks Submit", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cc-name").fill("   ");
  await expect(page.locator("#cc-submit")).toBeDisabled();
});

// ───────────────────────── §6.2 T — 04 Schema Toggles ────────────────────────

for (const [idx, key] of Object.entries({
  1: "dense_on_disk", 2: "sparse_on_disk", 3: "payload_on_disk",
  4: "quantization", 5: "sparse_idf",
})) {
  test(`T${idx} — flip ${key} on e2e_a_new1 and Update`, async ({ page }) => {
    await bootA(page, fixtureName("e2e_a_new1"));
    await openCollections(page);
    await expectReportFor(page, fixtureName("e2e_a_new1"));
    const current = await page.locator(CU_IDS[key]).inputValue();
    const next = current === "true" ? "false" : "true";
    await page.locator(CU_IDS[key]).selectOption(next);
    const [resp] = await Promise.all([
      page.waitForResponse((r) => new URL(r.url()).pathname
        === `/collections/${fixtureName("e2e_a_new1")}/schema-toggles`),
      page.locator("#cu-submit").click(),
    ]);
    expect(resp.status()).toBe(200);
    await expect(page.locator("#cu-result .alert-success")).toContainText(
      /schema toggles updated/i);
    // K3: report badge refreshes with no manual action
    await expect
      .poll(async () => (await reportMetrics(page))[REPORT_LABEL[key]], { timeout: 20_000 })
      .toBe(next);
  });
}

test("T6 — Update with no changes → readable 200 no-op", async ({ page }) => {
  await bootA(page, fixtureName("e2e_a_new1"));
  await openCollections(page);
  await expectReportFor(page, fixtureName("e2e_a_new1"));
  const before = await reportMetrics(page);
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname
      === `/collections/${fixtureName("e2e_a_new1")}/schema-toggles`),
    page.locator("#cu-submit").click(),
  ]);
  expect(resp.status()).toBe(200);
  await expect(page.locator("#cu-result .alert-success")).toContainText(
    /schema toggles updated/i);
  const after = await reportMetrics(page);
  for (const label of Object.values(REPORT_LABEL)) expect(after[label]).toBe(before[label]);
});

test("T7 — no active collection → Update disabled", async ({ page }) => {
  // Force the "no collection" state by serving an empty collection list.
  await page.route((u) => u.pathname === "/collections", async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return fulfillResponse(route, {
      status: 200, contentType: "application/json",
      body: JSON.stringify({ ok: true, collections: [], count: 0 }),
    });
  });
  await bootA(page, null);
  await openCollections(page);
  await expect(page.locator("#cc-report .alert-info")).toContainText(/no collection selected/i);
  await expect(page.locator("#cu-submit")).toBeDisabled();
});

test("T8 — active collection change right after Update empties cu-result (K5)", async ({ page }) => {
  await bootA(page, fixtureName("e2e_a_new1"));
  await openCollections(page);
  await expectReportFor(page, fixtureName("e2e_a_new1"));
  await page.locator("#cu-submit").click();
  await expect(page.locator("#cu-result .alert-success")).toBeVisible();
  await page.locator("#conn-collection").selectOption(EMPTY);
  await expect(page.locator("#cu-result")).toBeEmpty();
});

test("T10 — widget seeding: cu-* selects reflect the live state after loadReport", async ({ page }) => {
  // After T1–T5 flipped every toggle to false on e2e_a_new1, reload the report
  // and verify the update-form selects match the report, not static defaults.
  await bootA(page, fixtureName("e2e_a_new1"));
  await openCollections(page);
  await expectReportFor(page, fixtureName("e2e_a_new1"));
  const m = await reportMetrics(page);
  for (const [key, id] of Object.entries(CU_IDS)) {
    const actual = await page.locator(id).inputValue();
    expect(actual, `cu-* select for '${key}' should match the report badge`).toBe(m[REPORT_LABEL[key]]);
  }
});

test("T11 — [API] PATCH schema-toggles with empty body → 400", async ({ page }) => {
  // The UI always sends all 5 toggles; this case is only reachable via direct API.
  await bootA(page, fixtureName("e2e_a_keep"));
  await openCollections(page);
  const resp = await apiFetch(page.request, "PATCH",
    `/collections/${KEEP}/schema-toggles`, {});
  expect(resp.status()).toBe(400);
  expect(resp.status()).toBeLessThan(500);
  const body = await resp.json();
  expect(JSON.stringify(body)).toMatch(/schema toggle/i);
});

test("T12 — restore all toggles to true in a single update, data intact", async ({ page }) => {
  // After T1–T5 all toggles are false. Restore all → true in one submit.
  await bootA(page, fixtureName("e2e_a_new1"));
  await openCollections(page);
  await expectReportFor(page, fixtureName("e2e_a_new1"));
  for (const id of Object.values(CU_IDS)) {
    await page.locator(id).selectOption("true");
  }
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname
      === `/collections/${fixtureName("e2e_a_new1")}/schema-toggles`),
    page.locator("#cu-submit").click(),
  ]);
  expect(resp.status()).toBe(200);
  await expect(page.locator("#cu-result .alert-success")).toContainText(
    /schema toggles updated/i);
  // Report 01 is re-fetched after the write, so the success alert lands before
  // the refreshed badges do — poll, exactly as R4 does for the same badge.
  for (const label of Object.values(REPORT_LABEL)) {
    await expect
      .poll(async () => (await reportMetrics(page))[label],
        { timeout: 20_000, message: `report badge '${label}' after restore` })
      .toBe("true");
  }
});

// ───────────────────────── §6.3 PI — 05 Payload Indexes ──────────────────────

const PI_TYPES = ["keyword", "text", "integer", "float", "bool", "geo", "datetime", "uuid"];

test("PI1 — create one payload index per schema type", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  for (const schema of PI_TYPES) {
    const field = `pi1_${schema}`;
    await page.locator("#pi-field").fill(field);
    await page.locator("#pi-schema").selectOption(schema);
    const [resp] = await Promise.all([
      page.waitForResponse((r) => new URL(r.url()).pathname
        === `/collections/${KEEP}/payload-indexes` && r.request().method() === "POST"),
      page.locator("#pi-create").click(),
    ]);
    expect(resp.status(), `create index ${field}`).toBe(200);
    await expect(page.locator("#pi-create-result .alert-success")).toContainText(
      new RegExp(`index created: '${field}' as ${schema}`, "i"));
  }
  await expect
    .poll(async () => {
      const t = await indexTable(page);
      return PI_TYPES.every((s) => t[`pi1_${s}`] === s);
    }, { timeout: 20_000 })
    .toBe(true);
});

test("PI2 — phrase-matching checkbox appears only for schema=text", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#pi-schema").selectOption("keyword");
  await expect(page.locator("#pi-phrase-wrap")).toBeHidden();
  await page.locator("#pi-schema").selectOption("text");
  await expect(page.locator("#pi-phrase-wrap")).toBeVisible();
  for (const s of ["integer", "float", "bool", "geo", "datetime", "uuid"]) {
    await page.locator("#pi-schema").selectOption(s);
    await expect(page.locator("#pi-phrase-wrap")).toBeHidden();
  }
});

test("PI3 — create a text index with phrase matching enabled", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  await page.locator("#pi-field").fill("pi3_phrase");
  await page.locator("#pi-schema").selectOption("text");
  await page.locator("#pi-phrase").check();
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname
      === `/collections/${KEEP}/payload-indexes` && r.request().method() === "POST"),
    page.locator("#pi-create").click(),
  ]);
  expect(resp.status()).toBe(200);
  await expect(page.locator("#pi-create-result .alert-success")).toContainText(
    /index created: 'pi3_phrase' as text/i);
  await expect
    .poll(async () => (await indexTable(page))["pi3_phrase"], { timeout: 20_000 })
    .toBe("text (+phrase)");
  const api = await (await setupFetch(page.request, "GET",
    `/collections/${KEEP}/payload-indexes`)).json();
  const row = api.indexes.find((i) => i.field_name === "pi3_phrase");
  expect(row.phrase_matching).toBe(true);
});

test("PI4 — empty payload field → Create disabled", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expect(page.locator("#pi-field")).toHaveValue("");
  await expect(page.locator("#pi-create")).toBeDisabled();
  await page.locator("#pi-field").fill("x");
  await expect(page.locator("#pi-create")).toBeEnabled();
  await page.locator("#pi-field").fill("  ");
  await expect(page.locator("#pi-create")).toBeDisabled();
});

test("PI5 — re-create an existing index with a different schema", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  await page.locator("#pi-field").fill("pi5_upd");
  await page.locator("#pi-schema").selectOption("keyword");
  await page.locator("#pi-create").click();
  await expect(page.locator("#pi-create-result .alert-success")).toContainText(
    /index created: 'pi5_upd' as keyword/i);
  await page.locator("#pi-schema").selectOption("integer");
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname
      === `/collections/${KEEP}/payload-indexes` && r.request().method() === "POST"),
    page.locator("#pi-create").click(),
  ]);
  expect(resp.status()).toBe(200);
  await expect(page.locator("#pi-create-result .alert-success")).toContainText(
    /index type changed: 'pi5_upd' is now integer/i);
  await expect
    .poll(async () => (await indexTable(page))["pi5_upd"], { timeout: 20_000 })
    .toBe("integer");
});

test("PI6 — delete an existing payload index", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  await expect
    .poll(async () => (await indexTable(page))["pi5_upd"], { timeout: 20_000 })
    .toBeTruthy();
  await page.locator("#pi-del-field").fill("pi5_upd");
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname
      === `/collections/${KEEP}/payload-indexes/pi5_upd`
      && r.request().method() === "DELETE"),
    page.locator("#pi-delete").click(),
  ]);
  expect(resp.status()).toBe(200);
  await expect(page.locator("#pi-del-result .alert-success")).toContainText(
    /payload index deleted: pi5_upd/i);
  await expect(page.locator("#pi-del-field")).toHaveValue("");
  await expect
    .poll(async () => (await indexTable(page))["pi5_upd"], { timeout: 20_000 })
    .toBeUndefined();
});

test("PI7 — delete a never-indexed field → readable error, no crash", async ({ page }) => {
  expectHttpFailure(page, "DELETE", `/collections/${KEEP}/payload-indexes/nope_never_indexed`, 404);
  const errors = await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  await page.locator("#pi-del-field").fill("nope_never_indexed");
  await page.locator("#pi-delete").click();
  // The rendered error is the server detail alone ("payload index 'x' not
  // found in collection 'y'") — the bare "HTTP 404" status is not shown.
  await expect(page.locator("#pi-del-result .alert-error")).toContainText(
    /payload index 'nope_never_indexed' not found in collection/i);
  await expect(page.locator("#pi-del-result .alert-error")).not.toContainText(/HTTP/);
  // the field is NOT cleared on failure, so the user can fix the typo
  await expect(page.locator("#pi-del-field")).toHaveValue("nope_never_indexed");
  expect(errors.filter((e) => /uncaught|is not a function/i.test(e))).toEqual([]);
});

test("PI8 — empty delete field → Delete disabled", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expect(page.locator("#pi-del-field")).toHaveValue("");
  await expect(page.locator("#pi-delete")).toBeDisabled();
  await page.locator("#pi-del-field").fill("f_status");
  await expect(page.locator("#pi-delete")).toBeEnabled();
  await page.locator("#pi-del-field").fill(" ");
  await expect(page.locator("#pi-delete")).toBeDisabled();
});

test("PI9 — [API] phrase_matching on a non-text schema", async ({ request }) => {
  const resp = await apiFetch(request, "POST", `/collections/${KEEP}/payload-indexes`,
    { indexes: [{ field_name: "f_status", field_schema: "keyword", phrase_matching: true }] });
  const body = JSON.stringify(await resp.json());
  expect(body).toMatch(/phrase_matching requires field_schema 'text'/);
  // 400, not 422: the check lives in the endpoint's own validation, not in the
  // request model. Matches §6.3 PI9 of the book.
  expect(resp.status()).toBe(400);
});

test("PI10 — active collection change empties pi-create-result / pi-del-result (K5)", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  await page.locator("#pi-field").fill("pi10_tmp");
  await page.locator("#pi-schema").selectOption("keyword");
  await page.locator("#pi-create").click();
  await expect(page.locator("#pi-create-result .alert-success")).toBeVisible();
  await page.locator("#pi-del-field").fill("pi10_tmp");
  await page.locator("#pi-delete").click();
  await expect(page.locator("#pi-del-result .alert-success")).toBeVisible();
  await page.locator("#conn-collection").selectOption(EMPTY);
  await expect(page.locator("#pi-create-result")).toBeEmpty();
  await expect(page.locator("#pi-del-result")).toBeEmpty();
});

test("PI12 — re-submit an identical index → info 'No change'", async ({ page }) => {
  // e2e_a_keep has f_status indexed as keyword from seed. Re-creating it
  // identically should report skipped, not created.
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  await page.locator("#pi-field").fill("f_status");
  await page.locator("#pi-schema").selectOption("keyword");
  await page.locator("#pi-create").click();
  await expect(page.locator("#pi-create-result .alert-info")).toContainText(
    /No change|already indexed/i, { timeout: 15_000 });
});

test("PI15 — schema column shows '—' for sampled non-indexed fields", async ({ page }) => {
  // e2e_a_keep has f_extra deliberately never indexed.
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  const table = await indexTable(page);
  expect(table["f_extra"], "f_extra should show '—' schema").toBe("—");
});

// ───────────────────────── §6.4 X — 06 Export Collection ─────────────────────

test("X1 — export e2e_a_keep (10 points) downloads a JSON with entire-collection scope", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 30_000 }),
    page.locator("#ce-export").click(),
  ]);
  await expect(page.locator("#ce-result .alert-success")).toContainText(
    new RegExp(`exported 10 points from '${KEEP}'`, "i"));
  expect(download.suggestedFilename()).toMatch(new RegExp(`^${KEEP}_export_`));
  const file = await download.path();
  const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
  expect(parsed.chunks).toHaveLength(10);
  // The stored point id rides under `point_id`; `id` belongs to the payload.
  expect(parsed.chunks.every((c) => typeof c.point_id === "string")).toBe(true);
  expect(parsed.chunks.every((c) => typeof c.text === "string")).toBe(true);
  expect(parsed.metadata.n_chunks).toBe(10);
  expect(parsed.metadata.collection_name).toBe(KEEP);
  expect(parsed.metadata.request.scope).toBe("entire-collection");
  expect(Object.hasOwn(parsed.metadata.request, "limit")).toBe(false);
});

test("X2 — export across multiple real pages (>1000 points)", async ({ page }) => {
  const collection = fixtureOverride('QHM_REINDEX_COLL', 'e2e_z_reg');
  const report = await setupFetch(page.request, 'GET', `/collections/${collection}`);
  const count = (await report.json()).points_count;
  expect(count, 'Run seed:reindex before the export pagination case').toBeGreaterThan(1000);
  await bootA(page, collection);
  await openCollections(page);
  await expectReportFor(page, collection);
  const [download] = await Promise.all([
    page.waitForEvent('download', { timeout: 30_000 }),
    page.locator('#ce-export').click(),
  ]);
  const parsed = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
  expect(parsed.metadata.collection_name).toBe(collection);
  expect(parsed.metadata.n_chunks).toBe(count);
  expect(parsed.chunks).toHaveLength(count);
  expect(new Set(parsed.chunks.map((chunk) => chunk.point_id)).size).toBe(count);
  expect(parsed.chunks.map((chunk) => chunk.point_id).sort()).toEqual(
    Array.from({ length: count }, (_, index) => String(index + 1)).sort());
  for (const chunk of parsed.chunks) {
    expect(typeof chunk.point_id).toBe('string');
    expect(chunk.f_idx).toBe(Number(chunk.point_id));
    expect(chunk.text).toContain(`document ${chunk.point_id} paragraph`);
    expect(chunk).not.toHaveProperty('vector');
    expect(chunk).not.toHaveProperty('vectors');
  }
});

test("X3 — export an empty collection → warning, no download", async ({ page }) => {
  await bootA(page, EMPTY);
  await openCollections(page);
  await expectReportFor(page, EMPTY);
  let downloaded = false;
  page.on("download", () => { downloaded = true; });
  await page.locator("#ce-export").click();
  await expect(page.locator("#ce-result .alert-warning")).toContainText(
    /collection is empty/i);
  await expect(page.locator("#ce-result .alert-warning")).toContainText(
    /nothing to export/i);
  await page.waitForTimeout(1500);
  expect(downloaded).toBe(false);
});

test("X4 — export stays operable while the active collection is INCOMPATIBLE", async ({ page }) => {
  // The BASIC↔ADVANCED switch is out of scope, so the incompatible report is
  // injected at the network layer; everything downstream is the real app.
  await page.route((u) => u.pathname === `/collections/${KEEP}`, async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return transformJsonResponse(route, (json) => {
      json.embedding_compatibility.status = "incompatible";
      json.embedding_compatibility.message =
        "Collection uses a different vector space than the active embedder.";
      json.embedding_compatibility.stored.fingerprint = `sha256:${"a".repeat(64)}`;
      return json;
    });
  });
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  // lockdown truth table §1.1 (recoverable, same profile → banner shown)
  await expect(page.locator("#service-gate .alert-error")).toContainText(
    /different vector space/i);
  for (const id of ["cc-recreate-control", "cu-wrap", "pi-wrap"]) {
    await expect(page.locator(`#${id}`)).toHaveAttribute("inert", "");
  }
  // 07 Delete works on a typed name, so an incompatible active
  // collection never locks it.
  await expect(page.locator("#cd-wrap")).not.toHaveAttribute("inert", "");
  // 03 is half-locked, not locked: creating a NEW collection stays possible.
  await expect(page.locator("#cc-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-recreate")).toBeDisabled();
  await expect(page.locator("#cr-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#ce-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#conn-collection")).toBeEnabled();
  // and the export still completes
  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 30_000 }),
    page.locator("#ce-export").click(),
  ]);
  await expect(page.locator("#ce-result .alert-success")).toContainText(
    /exported 10 points/i);
  expect(download.suggestedFilename()).toContain(KEEP);
});

test("X5 — a failed export surfaces the server's error, no download", async ({ page }) => {
  // The walk (and any mid-walk failure) now happens server-side inside a
  // single POST /points/export; the client has nothing left to retry or
  // partially assemble, so a failure is just one error response.
  await page.route((u) => u.pathname === "/points/export", (route) => fulfillResponse(route, {
    status: 502, contentType: "application/json",
    body: JSON.stringify({ detail: "Qdrant scroll failed mid-export" }),
  }));
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  let downloaded = false;
  page.on("download", () => { downloaded = true; });
  await page.locator("#ce-export").click();
  await expect(page.locator("#ce-result .alert-error")).toContainText(
    /qdrant scroll failed mid-export/i);
  await page.waitForTimeout(1000);
  expect(downloaded).toBe(false);
});

test("X7 — during a service outage panel-collections is inert", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  await page.route((u) => u.pathname === "/health", (route) => fulfillResponse(route, {
    status: 200, contentType: "application/json",
    body: JSON.stringify({
      status: "unhealthy", qdrant: "reachable", embedder: "unreachable",
      collections: 9, embedder_profile: "basic",
    }),
  }));
  await page.locator("#conn-retest").click();
  await expect(page.locator("#conn-health-status")).toHaveText("UNHEALTHY");
  await expect(page.locator("#service-gate .alert-error")).toContainText(
    /Smart Embedder unreachable/i);
  await expect(page.locator("#panel-collections")).toHaveAttribute("inert", "");
  await expect(page.locator("#panel-collections")).toHaveAttribute("aria-disabled", "true");
  await expect(page.locator("#conn-collection")).toBeDisabled();
  // no export request can be issued from an inert panel
  const exports = countRequests(page, "/points/export", "POST");
  await page.locator("#ce-export").click({ timeout: 4000 }).catch(() => {});
  await page.waitForTimeout(1000);
  expect(exports.length).toBe(0);
});

// ───────────────────────── §6.5 D — 07 Delete ────────────────────────────────

test("D1 — delete with an empty confirmation → Delete disabled", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cd-name").fill(fixtureName("e2e_a_new3"));
  await expect(page.locator("#cd-confirm-label")).toContainText(fixtureName("e2e_a_new3")); // K7
  await expect(page.locator("#cd-confirm")).toHaveValue("");
  await expect(page.locator("#cd-submit")).toBeDisabled();
});

test("D2 — delete with a wrong confirmation → Delete disabled + hint", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cd-name").fill(fixtureName("e2e_a_new3"));
  await page.locator("#cd-confirm").fill(fixtureName("e2e_a_new3").slice(0, -1));
  await expect(page.locator("#cd-confirm-hint")).toBeVisible();
  await expect(page.locator("#cd-confirm-hint")).toContainText(/does not match/i);
  await expect(page.locator("#cd-submit")).toBeDisabled();
  // trailing whitespace is trimmed on both sides (K7)
  await page.locator("#cd-confirm").fill(`  ${fixtureName("e2e_a_new3")}  `);
  await expect(page.locator("#cd-confirm-hint")).toBeHidden();
  await expect(page.locator("#cd-submit")).toBeEnabled();
  // case sensitivity: uppercase must NOT satisfy the gate
  await page.locator("#cd-confirm").fill(fixtureName("e2e_a_new3").toUpperCase());
  await expect(page.locator("#cd-submit")).toBeDisabled();
});

test("D3 — delete e2e_a_new4 with the exact confirmation", async ({ page }) => {
  await bootA(page, fixtureName("e2e_a_new4"));
  await openCollections(page);
  await expectReportFor(page, fixtureName("e2e_a_new4"));
  await page.locator("#cd-name").fill(fixtureName("e2e_a_new4"));
  await page.locator("#cd-confirm").fill(fixtureName("e2e_a_new4"));
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname === `/collections/${fixtureName("e2e_a_new4")}`
      && r.request().method() === "DELETE"),
    page.locator("#cd-submit").click(),
  ]);
  expect(resp.status()).toBe(200);
  await expect(page.locator("#cd-result .alert-success")).toContainText(
    `Collection deleted: ${fixtureName("e2e_a_new4")}`);
  await expect(page.locator("#cd-name")).toHaveValue("");
  await expect(page.locator("#cd-confirm")).toHaveValue("");
  await expect(page.locator(`#conn-collection option[value="${fixtureName("e2e_a_new4")}"]`)).toHaveCount(0);
  // active falls back to the first collection alphabetically
  const expectedFallback = await page.locator("#conn-collection option").evaluateAll(
    (options) => options.map((option) => option.value).filter(Boolean)[0],
  );
  expect(expectedFallback).toBeTruthy();
  await expect(page.locator("#ql-collection")).not.toHaveValue(fixtureName("e2e_a_new4"));
  await expect(page.locator("#ql-collection")).toHaveValue(expectedFallback);
});

test("D4 — the delete form never preselects a collection (protected names stay safe)", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  // cd-name is not seeded from the active collection nor from any list
  await expect(page.locator("#cd-name")).toHaveValue("");
  await expect(page.locator("#cd-confirm")).toHaveValue("");
  await expect(page.locator("#cd-submit")).toBeDisabled();
  const html = await page.locator("#cd-wrap").innerHTML();
  for (const protectedName of ["cat", "my-collection", "test"]) {
    expect(html).not.toContain(`>${protectedName}<`);
  }
  // no <select>/datalist of collections exists inside the delete section
  expect(await page.locator("#cd-wrap select, #cd-wrap datalist").count()).toBe(0);
});

test("D5 — post-delete: Query tab fields consistent with new active, query does not crash", async ({ page }) => {
  // Its own victim: D3 already consumed e2e_a_new4, so booting on that name
  // would wait out the report poll on a collection that no longer exists.
  // recreate:true keeps the case self-healing if a previous run died here.
  const DOOMED = fixtureName("e2e_a_new5");
  const created = await apiFetch(page.request, "POST", "/collections", {
    collection_name: DOOMED, recreate: true, confirm_recreate: DOOMED,
    dense_on_disk: true, sparse_on_disk: true, payload_on_disk: true,
    quantization: true, sparse_idf: true,
  });
  expect(created.status()).toBe(200);
  await bootA(page, DOOMED);
  await openCollections(page);
  await expectReportFor(page, DOOMED);
  await page.locator("#cd-name").fill(DOOMED);
  await page.locator("#cd-confirm").fill(DOOMED);
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname === `/collections/${DOOMED}`),
    page.locator("#cd-submit").click(),
  ]);
  expect(resp.status()).toBe(200);
  await openTab(page, "query");
  await page.locator("#ql-query").fill("");
  await expect(page.locator("#ql-run")).toBeEnabled();
  await page.locator("#ql-run").click();
  // #ql-errors is not empty here and must not be: an empty query with no
  // filters legitimately carries the Q11 ordered-by-ID hint as .alert-info.
  // What must be absent is a real failure, which the panel renders as an
  // .alert-error — in #ql-errors for a refused request, in #ql-results for a
  // failed one (Q14).
  await expect(page.locator("#ql-results .alert-success")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator("#ql-errors .alert-error")).toHaveCount(0);
  await expect(page.locator("#ql-results .alert-error")).toHaveCount(0);
});

test("D6 — delete e2e_ghost (non-existent) → readable 404", async ({ page }) => {
  expectHttpFailure(page, "DELETE", `/collections/${fixtureName("e2e_ghost")}`, 404);
  await bootA(page, KEEP);
  await openCollections(page);
  await page.locator("#cd-name").fill(fixtureName("e2e_ghost"));
  await page.locator("#cd-confirm").fill(fixtureName("e2e_ghost"));
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname
      === `/collections/${fixtureName("e2e_ghost")}`),
    page.locator("#cd-submit").click(),
  ]);
  expect(resp.status()).toBe(404);
  await expect(page.locator("#cd-result .alert-error")).toBeVisible({ timeout: 15_000 });
});

// ───────────────────────── §6.1 R — 01 Report ───────────────────────────────

test("R1 — Report 01 badges and tiles for the default all-true collection", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  await expectBadges(page, ALL_TRUE);
  const m = await reportMetrics(page);
  expect(m["status"]).toBe("green");
  expect(Number(m["points"])).toBeGreaterThan(0);
  expect(m["dense size"]).toMatch(/^\d/);
});

test("R2 — all-false collection shows all-false badges and zero points", async ({ page }) => {
  // e2e_a_new2 was created all-false by C2 and never re-toggled.
  await bootA(page, fixtureName("e2e_a_new2"));
  await openCollections(page);
  await expectReportFor(page, fixtureName("e2e_a_new2"));
  await expectBadges(page, ALL_FALSE);
  const m = await reportMetrics(page);
  expect(m["points"]).toBe("0");
});

test("R3 — switching the active collection from the connection bar refreshes the report", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  const before = await reportMetrics(page);
  await page.locator("#conn-collection").selectOption(fixtureName("e2e_a_new2"));
  await expectReportFor(page, fixtureName("e2e_a_new2"));
  const after = await reportMetrics(page);
  // No residue: the new report shows the new collection's badges, not the old ones.
  for (const label of Object.values(REPORT_LABEL)) {
    expect(after[label], `report badge '${label}' after switch`).toBe("false");
    expect(before[label], `report badge '${label}' before switch`).toBe("true");
  }
});

test("R4 — report auto-refreshes after a schema-toggle write (K3 in the Report context)", async ({ page }) => {
  await bootA(page, fixtureName("e2e_a_new1"));
  await openCollections(page);
  await expectReportFor(page, fixtureName("e2e_a_new1"));
  const before = (await reportMetrics(page))["dense on disk"];
  const next = before === "true" ? "false" : "true";
  await page.locator(CU_IDS.dense_on_disk).selectOption(next);
  await page.locator("#cu-submit").click();
  await expect
    .poll(async () => (await reportMetrics(page))["dense on disk"], { timeout: 20_000 })
    .toBe(next);
  // restore
  await page.locator(CU_IDS.dense_on_disk).selectOption(before);
  await page.locator("#cu-submit").click();
  await expect
    .poll(async () => (await reportMetrics(page))["dense on disk"], { timeout: 20_000 })
    .toBe(before);
});

test("R5 — no active collection shows the info alert and disables Update", async ({ page }) => {
  // Same empty-list mock strategy as T7.
  await page.route((u) => u.pathname === "/collections", async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return fulfillResponse(route, {
      status: 200, contentType: "application/json",
      body: JSON.stringify({ ok: true, collections: [], count: 0 }),
    });
  });
  await bootA(page, null);
  await openCollections(page);
  await expect(page.locator("#cc-report .alert-info")).toContainText(/no collection selected/i);
  await expect(page.locator("#cu-submit")).toBeDisabled();
});

test("R6 — a 409 drift finding keeps Report 01 consistent with the real state", async ({ page }) => {
  expectHttpFailure(page, "POST", "/collections", 409);
  // Re-run the C6 scenario and assert the report still reflects e2e_a_new2's
  // real (all-false) state, not the rejected all-true submission.
  await bootA(page, fixtureName("e2e_a_new2"));
  await openCollections(page);
  await expectReportFor(page, fixtureName("e2e_a_new2"));
  await page.locator("#cc-name").fill(fixtureName("e2e_a_new2"));
  await setCreateToggles(page, ALL_TRUE);
  const [resp] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname === "/collections"
      && r.request().method() === "POST"),
    page.locator("#cc-submit").click(),
  ]);
  expect(resp.status()).toBe(409);
  await expect(page.locator("#cc-create-result .alert-info")).toContainText(
    /04 Schema Toggles/i);
  // The report must still show the real all-false state.
  await expectReportFor(page, fixtureName("e2e_a_new2"));
  await expectBadges(page, ALL_FALSE);
});

// ───────────────────────── §11 K — cross-cutting ─────────────────────────────

test("K1 — double click fires a single request (cc-submit)", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  const posts = countRequests(page, "/collections", "POST");
  await page.locator("#cc-name").fill(fixtureName("e2e_a_new1"));
  await page.locator("#cc-submit").dblclick();
  await expect(page.locator("#cc-create-result .alert-info")).toBeVisible();
  await page.waitForTimeout(1500);
  expect(posts.length).toBe(1);
});

test("K1 — double click fires a single request (cu-submit)", async ({ page }) => {
  await bootA(page, fixtureName("e2e_a_new1"));
  await openCollections(page);
  await expectReportFor(page, fixtureName("e2e_a_new1"));
  const hits = countRequests(page, `/collections/${fixtureName("e2e_a_new1")}/schema-toggles`);
  await page.locator("#cu-submit").dblclick();
  await expect(page.locator("#cu-result .alert-success")).toBeVisible();
  await page.waitForTimeout(1500);
  expect(hits.length).toBe(1);
});

test("K1 — double click fires a single request (pi-create / pi-delete)", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  const creates = countRequests(page, `/collections/${KEEP}/payload-indexes`, "POST");
  await page.locator("#pi-field").fill("k1_dbl");
  await page.locator("#pi-schema").selectOption("keyword");
  await page.locator("#pi-create").dblclick();
  await expect(page.locator("#pi-create-result .alert-success")).toBeVisible();
  await page.waitForTimeout(1500);
  expect(creates.length, "POST payload-indexes").toBe(1);

  const dels = countRequests(page, `/collections/${KEEP}/payload-indexes/k1_dbl`, "DELETE");
  await page.locator("#pi-del-field").fill("k1_dbl");
  await page.locator("#pi-delete").dblclick();
  await expect(page.locator("#pi-del-result .alert-success")).toBeVisible();
  await page.waitForTimeout(1500);
  expect(dels.length, "DELETE payload-indexes").toBe(1);
});

test("K1 — double click fires a single request (ce-export)", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  const exports = countRequests(page, "/points/export", "POST");
  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 30_000 }),
    page.locator("#ce-export").dblclick(),
  ]);
  expect(download.suggestedFilename()).toContain(KEEP);
  await page.waitForTimeout(1500);
  expect(exports.length).toBe(1);
});

test("K1 — double click fires a single request (cd-submit)", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  const dels = countRequests(page, `/collections/${fixtureName("e2e_a_new3")}`, "DELETE");
  await page.locator("#cd-name").fill(fixtureName("e2e_a_new3"));
  await page.locator("#cd-confirm").fill(fixtureName("e2e_a_new3"));
  await page.locator("#cd-submit").dblclick();
  await expect(page.locator("#cd-result .alert-success")).toContainText(
    `Collection deleted: ${fixtureName("e2e_a_new3")}`);
  await page.waitForTimeout(1500);
  expect(dels.length).toBe(1);
});

test("K5 — cr-*/cu/pi results cleared on active change, cc/cd results kept", async ({ page }) => {
  await bootA(page, KEEP);
  await openCollections(page);
  await expectReportFor(page, KEEP);
  // produce a 03-create result (name-scoped) and a 05-index result (active-scoped)
  await page.locator("#cc-name").fill(fixtureName("e2e_a_new1"));
  await page.locator("#cc-submit").click();
  await expect(page.locator("#cc-create-result .alert-info")).toBeVisible();
  await page.locator("#conn-collection").selectOption(KEEP);
  await expectReportFor(page, KEEP);
  await page.locator("#pi-field").fill("k5_tmp");
  await page.locator("#pi-schema").selectOption("keyword");
  await page.locator("#pi-create").click();
  await expect(page.locator("#pi-create-result .alert-success")).toBeVisible();

  await page.locator("#conn-collection").selectOption(EMPTY);
  await expect(page.locator("#pi-create-result")).toBeEmpty();
  await expect(page.locator("#pi-del-result")).toBeEmpty();
  await expect(page.locator("#cu-result")).toBeEmpty();
  await expect(page.locator("#cr-result")).toBeEmpty();
  await expect(page.locator("#cr-preflight-result")).toBeEmpty();
  // 03/07 results are name-scoped and deliberately survive
  await expect(page.locator("#cc-create-result")).not.toBeEmpty();
  // cleanup of the temp index
  await setupFetch(page.request, "DELETE", `/collections/${KEEP}/payload-indexes/k5_tmp`);
});

test("K6 — every Collections result retains clean per-test telemetry", async () => {
  const ledger = JSON.parse(fs.readFileSync(new URL('../test-results/campaign.json', import.meta.url), 'utf8'));
  const ids = new Set(ledger.collected.filter((item) => item.file === 'a-collections.spec.js').map((item) => item.id));
  const results = ledger.results.filter((item) => ids.has(item.id));
  for (const result of results.filter((item) => item.status !== 'skipped')) {
    expect(result.telemetry).not.toBeNull();
    expect(result.telemetry.unexpected).toEqual([]);
  }
});
