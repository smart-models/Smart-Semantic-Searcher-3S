import { fixtureName } from "./harness.mjs";
import { expectHttpFailure } from "./telemetry.mjs";
// Series C — POINTS tab, everything except Ingest.
// Test book: tests/testbook/test-book-e2e.md §7.2 (U), §7.3 (B), §7.4 (G),
// §7.5 (DL), §7.6 (PL) + the applicable §11 cross-cutting rules (K1,K2,K3,K6,K7).
import { test, expect } from "./fixtures.js";
import { boot, selectActive, openTab, openPointsSubtab, apiFetch, setupFetch, BASE_URL } from "./helpers.js";

const C = fixtureName("e2e_c_points");   // non-destructive sandbox (10 pts, ids "1".."10")
const V = fixtureName("e2e_c_victim");   // destructive sandbox (12 pts, ids "1".."12")


// ── small utilities ───────────────────────────────────────────────────────────

function pathOf(req) {
  try { return new URL(req.url()).pathname; } catch { return ""; }
}

/** Record every request matching `predicate` while `fn` runs. */
async function recordRequests(page, predicate, fn) {
  const seen = [];
  const onReq = (r) => { if (predicate(r)) seen.push(`${r.method()} ${pathOf(r)}`); };
  page.on("request", onReq);
  try { await fn(); } finally { page.off("request", onReq); }
  return seen;
}

/** Delay (server-side round trip) every request matching `predicate` by `ms`. */
async function withDelay(page, predicate, ms, fn) {
  const handler = async (route) => {
    if (predicate(route.request())) await new Promise((r) => setTimeout(r, ms));
    await route.fallback();
  };
  await page.route("**/*", handler);
  try { return await fn(); } finally { await page.unroute("**/*", handler); }
}

/**
 * K1 probe: fire three synchronous clicks on a button whose handler is wrapped
 * in withBusy(); exactly one request must leave the page.
 */
async function tripleClickRequests(page, buttonId, predicate, delayMs = 900) {
  await expect(page.locator(`#${buttonId}`)).toBeEnabled();
  let seen = [];
  await withDelay(page, predicate, delayMs, async () => {
    seen = await recordRequests(page, predicate, async () => {
      await page.evaluate((id) => {
        const b = document.getElementById(id);
        b.click(); b.click(); b.click();
      }, buttonId);
      await page.waitForTimeout(delayMs + 1200);
    });
  });
  return seen;
}

/** Read the "points" metric tile from Report 01 (#cc-report). */
function reportPoints(page) {
  return page.evaluate(() => {
    const tiles = [...document.querySelectorAll("#cc-report .metric")];
    const tile = tiles.find((m) => m.querySelector(".metric-label")?.textContent.trim() === "points");
    return tile ? tile.querySelector(".metric-value").textContent.trim() : null;
  });
}

async function apiGetPoint(request, collection, id) {
  const r = await apiFetch(request, "GET", `/points/${encodeURIComponent(id)}?collection_name=${collection}`);
  return { status: r.status(), body: r.ok() ? await r.json() : await r.text() };
}

async function apiPointsCount(request, collection) {
  const r = await setupFetch(request, "GET", `/collections/${collection}`);
  const count = (await r.json()).points_count;
  expect(Number.isSafeInteger(count) && count >= 0, "Malformed points count oracle").toBe(true);
  return count;
}

async function apiSeed(request, collection, points) {
  const r = await apiFetch(request, "POST", "/points/batch", { collection_name: collection, points });
  return { status: r.status(), body: await r.text() };
}

// ── fixtures ──────────────────────────────────────────────────────────────────

test.beforeEach(async ({ page }) => {
  await boot(page, { active: C });
  await openTab(page, "points");
});

// ══════════════════════════ §7.2 U — Upsert ═══════════════════════════════════

test("U1 — upsert with uuid point_id + text + empty payload", async ({ page, request }) => {
  const uuid = "11111111-1111-4111-8111-111111111111";
  await openPointsSubtab(page, "upsert");
  await page.fill("#up-id", uuid);
  await page.fill("#up-text", "zeta unique upsert probe kappa lambda for the c series");
  await page.fill("#up-payload", "{}");
  await expect(page.locator("#up-submit")).toBeEnabled();
  await page.click("#up-submit");
  await expect(page.locator("#up-result")).toContainText(`Point '${uuid}' upserted into '${C}'`);

  // Effect check: the point exists…
  const got = await apiGetPoint(request, C, uuid);
  expect(got.status).toBe(200);
  expect(got.body.payload.text).toContain("zeta unique upsert probe kappa");

  // …and it is retrievable through the hybrid (dense+sparse) query path.
  const q = await apiFetch(request, "POST", "/query/search", {
    collection_name: C, query: "zeta unique upsert probe kappa lambda", limit: 5,
  });
  expect(q.status()).toBe(200);
  const ids = (await q.json()).results.map((r) => String(r.id));
  expect(ids).toContain(uuid);
});

test("U2 — upsert with integer point_id", async ({ page, request }) => {
  await openPointsSubtab(page, "upsert");
  await page.fill("#up-id", "101");
  await page.fill("#up-text", "integer id upsert probe for the c series");
  await page.fill("#up-payload", '{"c_marker":"u2"}');
  await page.click("#up-submit");
  await expect(page.locator("#up-result")).toContainText(`Point '101' upserted into '${C}'`);
  const got = await apiGetPoint(request, C, "101");
  expect(got.status).toBe(200);
  expect(got.body.payload.c_marker).toBe("u2");
});

test("U3 — invalid payload JSON is reported locally, no request is sent", async ({ page }) => {
  await openPointsSubtab(page, "upsert");
  await page.fill("#up-id", "102");
  await page.fill("#up-text", "payload parse probe");
  await page.fill("#up-payload", "{not json");
  const isUpsert = (r) => r.method() === "POST" && pathOf(r) === "/points";
  const seen = await recordRequests(page, isUpsert, async () => {
    await page.click("#up-submit");
    await expect(page.locator("#up-result")).toContainText(/Invalid payload JSON/i);
    await page.waitForTimeout(500);
  });
  expect(seen).toEqual([]);
});

test("U4 — Upsert requires both id and text", async ({ page }) => {
  await openPointsSubtab(page, "upsert");
  await page.fill("#up-id", "");
  await page.fill("#up-text", "");
  await expect(page.locator("#up-submit")).toBeDisabled();
  await page.fill("#up-id", "103");
  await expect(page.locator("#up-submit")).toBeDisabled();
  await page.fill("#up-id", "");
  await page.fill("#up-text", "text only");
  await expect(page.locator("#up-submit")).toBeDisabled();
  await page.fill("#up-id", "103");
  await expect(page.locator("#up-submit")).toBeEnabled();
  // whitespace-only must not count as filled
  await page.fill("#up-id", "   ");
  await expect(page.locator("#up-submit")).toBeDisabled();
});

test("U5 — active collection changed mid-upsert: result discarded (K2)", async ({ page }) => {
  await openPointsSubtab(page, "upsert");
  await page.fill("#up-id", "104");
  await page.fill("#up-text", "stale upsert probe");
  await page.fill("#up-payload", "{}");
  const isUpsert = (r) => r.method() === "POST" && pathOf(r) === "/points";
  await withDelay(page, isUpsert, 2500, async () => {
    await page.click("#up-submit");
    await page.waitForTimeout(300);
    await selectActive(page, V);
    await page.waitForTimeout(3500);
  });
  await expect(page.locator("#up-result")).toHaveText("");
});

test("U6 — top-level text remains authoritative over payload text", async ({ page, request }) => {
  // Qdrant point ids are an unsigned integer or a UUID; a slug is refused with
  // a 400 before the case under test is ever reached.
  const id = "6006";
  await openPointsSubtab(page, "upsert");
  await page.locator("#up-id").fill(id);
  await page.locator("#up-text").fill("base text");
  await page.locator("#up-payload").fill('{"text":"payload override"}');
  await page.locator("#up-submit").click();
  await expect(page.locator("#up-result .alert-success")).toBeVisible({ timeout: 15_000 });
  const { status, body } = await apiGetPoint(request, C, id);
  expect(status).toBe(200);
  expect(body.payload.text).toBe("base text");
});

test("U7 — same id with new text overwrites without duplicating", async ({ page, request }) => {
  // A dedicated id, not a seeded one: an upsert replaces the whole payload, so
  // overwriting point "1" here would strip the payload G1/G5 read back later.
  const id = "7777";
  expect((await apiSeed(request, C, [{ point_id: id, text: "first version", payload: {} }])).status).toBe(200);
  const before = await apiPointsCount(request, C);
  await openPointsSubtab(page, "upsert");
  await page.locator("#up-id").fill(id);
  await page.locator("#up-text").fill("overwritten text");
  await page.locator("#up-payload").fill("{}");
  await page.locator("#up-submit").click();
  await expect(page.locator("#up-result .alert-success")).toBeVisible({ timeout: 15_000 });
  const after = await apiPointsCount(request, C);
  expect(after).toBe(before);
  const { body } = await apiGetPoint(request, C, id);
  expect(body.payload.text).toBe("overwritten text");
});

// ══════════════════════════ §7.3 B — Batch ════════════════════════════════════

test("B1 — batch upsert of N valid points", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "batch");
  const points = [101, 102, 103].map((i) => ({
    point_id: String(i),
    text: `batch seeded point ${i} alpha beta gamma`,
    payload: { f_status: `b${i}`, f_views: i },
  }));
  const before = await apiPointsCount(request, V);
  await page.fill("#bt-json", JSON.stringify(points, null, 2));
  await page.click("#bt-submit");
  await expect(page.locator("#bt-result")).toContainText(`3 point(s) upserted into '${V}'`);
  expect(await apiPointsCount(request, V)).toBe(before + 3);
  const got = await apiGetPoint(request, V, "102");
  expect(got.status).toBe(200);
  expect(got.body.payload.f_status).toBe("b102");
});

test("B2 — empty array is refused with a readable message, no request", async ({ page }) => {
  await openPointsSubtab(page, "batch");
  await page.fill("#bt-json", "[]");
  const isBatch = (r) => r.method() === "POST" && pathOf(r) === "/points/batch";
  const seen = await recordRequests(page, isBatch, async () => {
    await page.click("#bt-submit");
    await expect(page.locator("#bt-result")).toContainText(/non-empty JSON array of points/i);
    await page.waitForTimeout(400);
  });
  expect(seen).toEqual([]);
});

test("B3 — malformed / non-array JSON is a parse error", async ({ page }) => {
  await openPointsSubtab(page, "batch");
  const isBatch = (r) => r.method() === "POST" && pathOf(r) === "/points/batch";
  const seen = await recordRequests(page, isBatch, async () => {
    await page.fill("#bt-json", '[{"point_id":');
    await page.click("#bt-submit");
    await expect(page.locator("#bt-result")).toContainText(/Invalid JSON/i);
    await page.fill("#bt-json", '{"point_id":"1"}');
    await page.click("#bt-submit");
    await expect(page.locator("#bt-result")).toContainText(/non-empty JSON array of points/i);
    await page.waitForTimeout(400);
  });
  expect(seen).toEqual([]);
});

test("B4 — non-object items, missing fields, empty text → readable 400, no writes", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "batch");
  const before = await apiPointsCount(request, V);
  const resp = await apiSeed(request, V, [
    { point_id: "4004", text: "" },
    "not-an-object",
  ]);
  // Every request-model failure is normalised to 400 by the app-wide
  // RequestValidationError handler (main.py), not left at FastAPI's 422.
  expect(resp.status).toBe(400);
  expect(resp.body).toMatch(/[Ii]nput|[Nn]on-empty|[Vv]alid|[Ff]ield/);
  const after = await apiPointsCount(request, V);
  expect(after).toBe(before);
});

test("B5 — invalid point_id → 400, no writes", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "batch");
  const before = await apiPointsCount(request, V);
  const resp = await apiSeed(request, V, [{ point_id: null, text: "b5" }]);
  expect(resp.status).toBe(400);
  const after = await apiPointsCount(request, V);
  expect(after).toBe(before);
});

test("B6 — duplicate point_ids in the same batch → 400", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "batch");
  // The ids must be well-formed, otherwise the point_id format error fires
  // first and the duplicate check under test is never reached.
  const resp = await apiSeed(request, V, [
    { point_id: "6161", text: "first" },
    { point_id: "6161", text: "second" },
  ]);
  expect(resp.status).toBe(400);
  expect(resp.body).toMatch(/duplicate/i);
});

test("B7 — non-object payload in a batch item → readable 400", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "batch");
  const before = await apiPointsCount(request, V);
  const resp = await apiSeed(request, V, [{ point_id: "7171", text: "ok", payload: "not-an-object" }]);
  expect(resp.status).toBe(400);
  const after = await apiPointsCount(request, V);
  expect(after).toBe(before);
});

test("B8 — batch with one existing and one new id: existing overwritten, count +1", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "batch");
  const before = await apiPointsCount(request, V);
  const resp = await apiSeed(request, V, [
    { point_id: "1", text: "updated existing", payload: {} },
    { point_id: "8008", text: "new point", payload: {} },
  ]);
  expect(resp.status).toBe(200);
  const after = await apiPointsCount(request, V);
  expect(after).toBe(before + 1);
  const got = await apiGetPoint(request, V, "1");
  expect(got.status).toBe(200);
  expect(got.body.payload.text).toBe("updated existing");
});

// ══════════════════════════ §7.4 G — Get ══════════════════════════════════════

test("G1 — single get of an existing id", async ({ page }) => {
  await openPointsSubtab(page, "get");
  await expect(page.locator("#get-submit")).toBeDisabled(); // enabled only with an id
  await page.fill("#get-id", "1");
  await expect(page.locator("#get-submit")).toBeEnabled();
  await page.click("#get-submit");
  await expect(page.locator("#get-result")).toContainText("id:");
  await expect(page.locator("#get-result")).toContainText(C);
  await expect(page.locator("#get-result")).toContainText('"f_status": "s1"');
  await expect(page.locator("#get-result")).toContainText('"f_views": 1');
  // NOTE: the console renders id + collection + payload only — no vectors.
  // No vector arrays are rendered. (The word "vector" appears inside the seeded
  // point text, so match the JSON keys, not the free text.)
  const body = await page.locator("#get-result").textContent();
  expect(body).not.toMatch(/"dense"\s*:|"sparse"\s*:|"vectors?"\s*:/i);
});

test("G2 — single get of a missing id is a readable not-found", async ({ page }) => {
  expectHttpFailure(page, "GET", "/points/987654", 404);
  await openPointsSubtab(page, "get");
  await page.fill("#get-id", "987654");
  await page.click("#get-submit");
  await expect(page.locator("#get-result .alert-error")).toBeVisible();
  await expect(page.locator("#get-result")).toContainText(/not found/i);
  await expect(page.locator("#get-result")).toContainText("987654");
});

test("G3 — get many by ids, one per line and CSV", async ({ page }) => {
  await openPointsSubtab(page, "get");
  await page.fill("#get-ids", "1\n2\n3");
  await expect(page.locator("#get-many-count")).toHaveText("3 id(s) parsed.");
  await page.click("#get-many-submit");
  await expect(page.locator("#get-many-result")).toContainText("3 of 3 id(s) found");

  await page.fill("#get-ids", "4, 5, 987654");
  await expect(page.locator("#get-many-count")).toHaveText("3 id(s) parsed.");
  await page.click("#get-many-submit");
  await expect(page.locator("#get-many-result")).toContainText("2 of 3 id(s) found");
  await expect(page.locator("#get-many-result")).toContainText("1 not present");
});

test("G4 — empty get-many textarea: submit disabled, count hidden", async ({ page }) => {
  await openPointsSubtab(page, "get");
  await page.fill("#get-ids", "");
  await expect(page.locator("#get-many-submit")).toBeDisabled();
  await expect(page.locator("#get-many-count")).toBeHidden();
  expect(await page.locator("#get-many-count").textContent()).toBe("");
  await page.fill("#get-ids", "   \n  ,  ");
  await expect(page.locator("#get-many-submit")).toBeDisabled();
  await expect(page.locator("#get-many-count")).toBeHidden();
});

test("G5 — payload with HTML is escaped, no DOM/console execution", async ({ page, request }) => {
  const id = "5005";
  expect((await apiSeed(request, C, [{ point_id: id, text: "safe", payload: { html: "<script>alert(1)</script>" } }])).status).toBe(200);
  await openPointsSubtab(page, "get");
  await page.locator("#get-id").fill(id);
  await page.locator("#get-submit").click();
  await expect(page.locator("#get-result")).toBeVisible({ timeout: 15_000 });
  const resultHtml = await page.locator("#get-result").innerHTML();
  // The script tag must not be live — it must be escaped or absent from innerHTML
  expect(resultHtml).not.toContain("<script>");
  expect(resultHtml).toContain("&lt;script&gt;");
});

// ══════════════════════════ §7.5 DL — Delete ══════════════════════════════════

test("DL1 — single delete of an existing id", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  await page.fill("#del-id", "12");
  await expect(page.locator("#del-submit")).toBeEnabled();
  await page.click("#del-submit");
  await expect(page.locator("#del-result")).toContainText(`Point '12' deleted from '${V}'`);
  await expect(page.locator("#del-id")).toHaveValue("");
  const got = await apiGetPoint(request, V, "12");
  expect(got.status).toBe(404);
});

test("DL2 — single delete with an empty id is disabled", async ({ page }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  await page.fill("#del-id", "");
  await expect(page.locator("#del-submit")).toBeDisabled();
  await page.fill("#del-id", "  ");
  await expect(page.locator("#del-submit")).toBeDisabled();
});

test("DL3 — bulk delete with exact confirmation", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  const before = await apiPointsCount(request, V);
  await page.fill("#del-ids", "1\n2");
  await expect(page.locator("#del-count")).toHaveText("2 id(s) parsed.");
  await page.fill("#del-confirm", V);
  await expect(page.locator("#del-confirm-hint")).toBeHidden();
  await expect(page.locator("#del-bulk-submit")).toBeEnabled();
  await page.click("#del-bulk-submit");
  await expect(page.locator("#del-bulk-result")).toContainText(`2 point(s) deleted from '${V}'`);
  // The confirmation is disarmed after a successful destructive run.
  await expect(page.locator("#del-ids")).toHaveValue("");
  await expect(page.locator("#del-confirm")).toHaveValue("");
  expect((await apiGetPoint(request, V, "1")).status).toBe(404);
  expect((await apiGetPoint(request, V, "2")).status).toBe(404);
  expect((await apiGetPoint(request, V, "3")).status).toBe(200);
  expect(await apiPointsCount(request, V)).toBe(before - 2);
});

test("DL4 — bulk delete with wrong/empty confirmation stays disabled", async ({ page }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  await page.fill("#del-ids", "4\n5");
  await page.fill("#del-confirm", "");
  await expect(page.locator("#del-bulk-submit")).toBeDisabled();
  await expect(page.locator("#del-confirm-hint")).toBeHidden(); // empty ⇒ no hint yet
  await page.fill("#del-confirm", fixtureName("e2e_c_vict"));
  await expect(page.locator("#del-bulk-submit")).toBeDisabled();
  await expect(page.locator("#del-confirm-hint")).toBeVisible();
  await expect(page.locator("#del-confirm-hint")).toContainText("Does not match");
  await page.fill("#del-confirm", C); // a real, but wrong, collection name
  await expect(page.locator("#del-bulk-submit")).toBeDisabled();
  await expect(page.locator("#del-confirm-hint")).toBeVisible();
});

test("DL5 — delete by filter: Count matches", async ({ page }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  await page.fill("#delf-json", "{}");
  await expect(page.locator("#delf-count-submit")).toBeDisabled(); // no filter ⇒ no count
  await page.fill("#delf-json", '{"f_status":"s3"}');
  await expect(page.locator("#delf-count-submit")).toBeEnabled();
  await page.click("#delf-count-submit");
  await expect(page.locator("#delf-count")).toHaveText("1 point(s) match this filter.");
});

test("DL6 — delete by filter with exact confirmation removes only the matches", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  const before = await apiPointsCount(request, V);
  await page.fill("#delf-json", '{"f_status":"s3"}');
  await page.fill("#delf-confirm", V);
  await expect(page.locator("#delf-submit")).toBeEnabled();
  await page.click("#delf-submit");
  await expect(page.locator("#delf-result")).toContainText(`1 point(s) deleted from '${V}' by filter`);
  await expect(page.locator("#delf-json")).toHaveValue("{}");
  await expect(page.locator("#delf-confirm")).toHaveValue("");
  expect((await apiGetPoint(request, V, "3")).status).toBe(404);
  expect((await apiGetPoint(request, V, "4")).status).toBe(200);
  expect(await apiPointsCount(request, V)).toBe(before - 1);
});

test("DL7 — delete by filter with a wrong confirmation stays disabled", async ({ page }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  await page.fill("#delf-json", '{"f_status":"s5"}');
  await page.fill("#delf-confirm", "nope");
  await expect(page.locator("#delf-submit")).toBeDisabled();
  await expect(page.locator("#delf-confirm-hint")).toBeVisible();
  await page.fill("#delf-confirm", "");
  await expect(page.locator("#delf-submit")).toBeDisabled();
});

test("DL8 — match-all filter {} is refused, not executed", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  const before = await apiPointsCount(request, V);
  await page.fill("#delf-json", "{}");
  await page.fill("#delf-confirm", V);
  // UI gate: an empty filter never arms the destructive button.
  await expect(page.locator("#delf-submit")).toBeDisabled();
  await expect(page.locator("#delf-count-submit")).toBeDisabled();
  // [API] the backend refuses it too, so there is no bypass.
  const r = await apiFetch(request, "POST", "/points/delete-by-filter", { collection_name: V, filters: {} });
  expect(r.status()).toBeGreaterThanOrEqual(400);
  expect(await apiPointsCount(request, V)).toBe(before);
});

test("DL9 — confirmation typed for one collection does not carry to another", async ({ page }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  await page.fill("#del-ids", "9\n10");
  await page.fill("#del-confirm", V);
  await expect(page.locator("#del-bulk-submit")).toBeEnabled();
  await expect(page.locator("#del-confirm-label")).toHaveText(`Type '${V}' to confirm bulk delete`);

  await selectActive(page, C);
  await expect(page.locator("#del-confirm-label")).toHaveText(`Type '${C}' to confirm bulk delete`);
  await expect(page.locator("#del-bulk-submit")).toBeDisabled();
  await expect(page.locator("#delf-confirm-label")).toHaveText(`Type '${C}' to confirm delete by filter`);
  await expect(page.locator("#pl-clear-confirm-label")).toHaveText(`Type '${C}' to confirm clear payload`);
  // Reported separately: the "Does not match" hint is not re-evaluated here.
  const hintHidden = await page.locator("#del-confirm-hint").isHidden();
  test.info().annotations.push({ type: "DL9-hint-state", description: `hint hidden after switch = ${hintHidden}` });
});

test("DL10 — single delete of an absent id is idempotent and leaves the count unchanged", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  await page.fill("#del-id", "9999");
  const before = await apiPointsCount(request, V);
  await page.locator("#del-submit").click();
  await expect(page.locator("#del-result .alert-success")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("#del-id")).toHaveValue("");
  expect(await apiPointsCount(request, V)).toBe(before);
});

test("DL11 — bulk delete counts actual deletions when one submitted ID is absent", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  // One real point (seeded here, so an earlier delete cannot make this flaky)
  // plus one id that is well-formed but absent. A slug would be refused as a
  // malformed point_id and never exercise the "non-existent" path.
  expect((await apiSeed(request, V, [{ point_id: "9911", text: "dl11 probe", payload: {} }])).status).toBe(200);
  const before = await apiPointsCount(request, V);
  await page.fill("#del-ids", "9911\n999999");
  await page.fill("#del-confirm", V);
  await page.locator("#del-bulk-submit").click();
  await expect(page.locator("#del-bulk-result .alert-success")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("#del-bulk-result")).toContainText("1 point(s) deleted");
  expect(await apiPointsCount(request, V)).toBe(before - 1);
});

test("DL12 — bulk with an invalid id → 400, no deletes executed", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  const before = await apiPointsCount(request, V);
  // Bulk delete is DELETE /points with a body; there is no /points/bulk route,
  // and that path would fall through to DELETE /points/{point_id}.
  const resp = await apiFetch(request, "DELETE", "/points",
    { collection_name: V, ids: ["invalid id"] });
  expect(resp.status()).toBe(400);
  const after = await apiPointsCount(request, V);
  expect(after).toBe(before);
});

// ══════════════════════════ §7.6 PL — Payload ═════════════════════════════════

test("PL1 — MERGE (PATCH) keeps the existing payload", async ({ page, request }) => {
  await openPointsSubtab(page, "payload");
  await page.fill("#pl-id", "3");
  await page.check('input[name="pl-mode"][value="PATCH"]');
  await page.fill("#pl-json", '{"pl_merge":"m1","f_views":333}');
  await expect(page.locator("#pl-submit")).toBeEnabled();
  await page.click("#pl-submit");
  await expect(page.locator("#pl-result")).toContainText("Payload merged into point '3'");
  const { body } = await apiGetPoint(request, C, "3");
  expect(body.payload.pl_merge).toBe("m1");
  expect(body.payload.f_views).toBe(333);
  expect(body.payload.f_status).toBe("s3");   // preserved
  expect(body.payload.text).toContain("alpha beta gamma");
});

test("PL2 — REPLACE (PUT) substitutes the payload", async ({ page, request }) => {
  await openPointsSubtab(page, "payload");
  await page.fill("#pl-id", "4");
  await page.check('input[name="pl-mode"][value="PUT"]');
  await page.fill("#pl-json", '{"pl_only":"r1"}');
  await page.click("#pl-submit");
  await expect(page.locator("#pl-result")).toContainText("Payload replaced on point '4'");
  const { body } = await apiGetPoint(request, C, "4");
  expect(body.payload.pl_only).toBe("r1");
  expect(body.payload.f_status).toBeUndefined();
  expect(body.payload.f_views).toBeUndefined();
  // protected service fields are carried forward by the backend
  expect(body.payload.text).toContain("alpha beta gamma");
});

test("PL3 — invalid payload JSON is reported locally, no request", async ({ page }) => {
  await openPointsSubtab(page, "payload");
  await page.fill("#pl-id", "5");
  await page.fill("#pl-json", "{oops");
  const isPayload = (r) => /^\/points\/[^/]+\/payload$/.test(pathOf(r));
  const seen = await recordRequests(page, isPayload, async () => {
    await page.click("#pl-submit");
    await expect(page.locator("#pl-result")).toContainText(/Invalid JSON/i);
    await page.waitForTimeout(400);
  });
  expect(seen).toEqual([]);
});

test("PL4 — delete payload keys", async ({ page, request }) => {
  await openPointsSubtab(page, "payload");
  await page.fill("#pl-id", "5");
  await page.fill("#pl-keys", "");
  await expect(page.locator("#pl-del-keys-submit")).toBeDisabled();
  await expect(page.locator("#pl-keys-count")).toBeHidden();
  await page.fill("#pl-keys", "f_extra\nf_tags, f_price");
  await expect(page.locator("#pl-keys-count")).toHaveText("3 key(s) parsed.");
  await expect(page.locator("#pl-del-keys-submit")).toBeEnabled();
  await page.click("#pl-del-keys-submit");
  await expect(page.locator("#pl-keys-result")).toContainText("3 key(s) removed from point '5'");
  await expect(page.locator("#pl-keys")).toHaveValue("");
  const { body } = await apiGetPoint(request, C, "5");
  expect(body.payload.f_extra).toBeUndefined();
  expect(body.payload.f_tags).toBeUndefined();
  expect(body.payload.f_price).toBeUndefined();
  expect(body.payload.f_status).toBe("s5"); // untouched
});

test("PL5 — clear payload keeps only text and source_id", async ({ page, request }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "payload");
  const before = await apiGetPoint(request, V, "5");
  expect(Object.keys(before.body.payload).length).toBeGreaterThan(2);
  await page.fill("#pl-id", "5");
  await page.fill("#pl-clear-confirm", V);
  await expect(page.locator("#pl-clear-confirm-hint")).toBeHidden();
  await expect(page.locator("#pl-clear-submit")).toBeEnabled();
  await page.click("#pl-clear-submit");
  await expect(page.locator("#pl-clear-result")).toContainText("Payload cleared on point '5' (protected fields kept)");
  await expect(page.locator("#pl-clear-confirm")).toHaveValue("");
  const after = await apiGetPoint(request, V, "5");
  const keys = Object.keys(after.body.payload).sort();
  expect(keys.every((k) => k === "text" || k === "source_id")).toBe(true);
  expect(keys).toContain("text");
  expect(after.body.payload.text).toBe(before.body.payload.text);
});

test("PL6 — clear payload with wrong/empty confirmation stays disabled", async ({ page }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "payload");
  await page.fill("#pl-id", "6");
  await page.fill("#pl-clear-confirm", "");
  await expect(page.locator("#pl-clear-submit")).toBeDisabled();
  await expect(page.locator("#pl-clear-confirm-hint")).toBeHidden();
  await page.fill("#pl-clear-confirm", fixtureName("e2e_c_victi"));
  await expect(page.locator("#pl-clear-submit")).toBeDisabled();
  await expect(page.locator("#pl-clear-confirm-hint")).toBeVisible();
  // and with no point id, even an exact confirmation is not enough
  await page.fill("#pl-clear-confirm", V);
  await page.fill("#pl-id", "");
  await expect(page.locator("#pl-clear-submit")).toBeDisabled();
});

test("PL7 — REPLACE containing text is refused atomically", async ({ page, request }) => {
  expectHttpFailure(page, "PUT", "/points/7007/payload", 400);
  const id = "7007";
  expect((await apiSeed(request, C, [{ point_id: id, text: "original", payload: { x: 1 } }])).status).toBe(200);
  const original = (await apiGetPoint(request, C, id)).body.payload;
  await openPointsSubtab(page, "payload");
  await page.fill("#pl-id", id);
  await page.check('input[name="pl-mode"][value="PUT"]');
  await page.fill("#pl-json", '{"text":"explicit","source_id":"explicit-src","y":2}');
  const [response] = await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname === "/points/7007/payload"
      && r.request().method() === "PUT"),
    page.locator("#pl-submit").click(),
  ]);
  expect(response.status()).toBe(400);
  await expect(page.locator("#pl-result .alert-error")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("#pl-json")).toHaveValue('{"text":"explicit","source_id":"explicit-src","y":2}');
  const { body } = await apiGetPoint(request, C, id);
  expect(body.payload).toEqual(original);
});

test("PL8 — PATCH/PUT on a non-existent point → readable 404, no implicit creation", async ({ page, request }) => {
  expectHttpFailure(page, "PATCH", "/points/999997/payload", 404);
  await openPointsSubtab(page, "payload");
  await page.fill("#pl-id", "999997");
  await page.fill("#pl-json", '{"z":99}');
  await page.locator("#pl-submit").click();
  await expect(page.locator("#pl-result .alert-error")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("#pl-result")).toContainText(/not found/i);
  const got = await apiGetPoint(request, C, "999997");
  expect(got.status).toBe(404);
});

// ══════════════════════════ §11 K — cross-cutting ═════════════════════════════

test("K1 — double-click prevention on every Points submit button", async ({ page, request }) => {
  test.setTimeout(300_000);
  expect((await apiSeed(request, V, [901, 902, 903, 904, 905].map((i) => ({
    point_id: String(i),
    text: `k1 fixture point ${i}`,
    payload: { f_status: i === 905 ? "k1z" : `k1_${i}`, f_extra: "e", f_other: "o" },
  })))).status).toBe(200);
  await selectActive(page, V);
  const report = {};

  // up-submit
  await openPointsSubtab(page, "upsert");
  await page.fill("#up-id", "911");
  await page.fill("#up-text", "k1 double click upsert");
  await page.fill("#up-payload", "{}");
  report["up-submit"] = await tripleClickRequests(
    page, "up-submit", (r) => r.method() === "POST" && pathOf(r) === "/points");

  // bt-submit
  await openPointsSubtab(page, "batch");
  await page.fill("#bt-json", '[{"point_id":"912","text":"k1 double click batch","payload":{}}]');
  report["bt-submit"] = await tripleClickRequests(
    page, "bt-submit", (r) => r.method() === "POST" && pathOf(r) === "/points/batch");

  // get-submit / get-many-submit
  await openPointsSubtab(page, "get");
  await page.fill("#get-id", "901");
  report["get-submit"] = await tripleClickRequests(
    page, "get-submit", (r) => r.method() === "GET" && pathOf(r) === "/points/901");
  await page.fill("#get-ids", "901,902");
  report["get-many-submit"] = await tripleClickRequests(
    page, "get-many-submit", (r) => r.method() === "POST" && pathOf(r) === "/points/retrieve");

  // del-submit / del-bulk-submit / delf-submit
  await openPointsSubtab(page, "delete");
  await page.fill("#del-id", "901");
  report["del-submit"] = await tripleClickRequests(
    page, "del-submit", (r) => r.method() === "DELETE" && pathOf(r) === "/points/901");
  await page.fill("#del-ids", "902,903");
  await page.fill("#del-confirm", V);
  report["del-bulk-submit"] = await tripleClickRequests(
    page, "del-bulk-submit", (r) => r.method() === "DELETE" && pathOf(r) === "/points");
  await page.fill("#delf-json", '{"f_status":"k1z"}');
  await page.fill("#delf-confirm", V);
  report["delf-submit"] = await tripleClickRequests(
    page, "delf-submit", (r) => r.method() === "POST" && pathOf(r) === "/points/delete-by-filter");

  // pl-submit / pl-del-keys-submit / pl-clear-submit
  await openPointsSubtab(page, "payload");
  await page.fill("#pl-id", "904");
  await page.fill("#pl-json", '{"k1":"x"}');
  report["pl-submit"] = await tripleClickRequests(
    page, "pl-submit", (r) => pathOf(r) === "/points/904/payload");
  await page.fill("#pl-keys", "f_extra");
  report["pl-del-keys-submit"] = await tripleClickRequests(
    page, "pl-del-keys-submit", (r) => pathOf(r) === "/points/904/payload/delete-keys");
  await page.fill("#pl-clear-confirm", V);
  report["pl-clear-submit"] = await tripleClickRequests(
    page, "pl-clear-submit", (r) => pathOf(r) === "/points/904/payload/clear");

  const offenders = Object.entries(report).filter(([, reqs]) => reqs.length !== 1);
  expect(offenders, `K1 request counts: ${JSON.stringify(report)}`).toEqual([]);
});

test("K2 — stale responses are discarded when the active collection changes", async ({ page }) => {
  test.setTimeout(300_000);
  const cases = [
    { name: "get", sub: "get", setup: async () => { await page.fill("#get-id", "1"); },
      btn: "get-submit", result: "get-result",
      pred: (r) => r.method() === "GET" && pathOf(r) === "/points/1" },
    { name: "get-many", sub: "get", setup: async () => { await page.fill("#get-ids", "1,2"); },
      btn: "get-many-submit", result: "get-many-result",
      pred: (r) => r.method() === "POST" && pathOf(r) === "/points/retrieve" },
    { name: "payload-update", sub: "payload",
      setup: async () => { await page.fill("#pl-id", "6"); await page.fill("#pl-json", '{"k2":"x"}'); },
      btn: "pl-submit", result: "pl-result",
      pred: (r) => pathOf(r) === "/points/6/payload" },
    { name: "payload-keys", sub: "payload",
      setup: async () => { await page.fill("#pl-id", "6"); await page.fill("#pl-keys", "k2"); },
      btn: "pl-del-keys-submit", result: "pl-keys-result",
      pred: (r) => pathOf(r) === "/points/6/payload/delete-keys" },
    { name: "single-delete", sub: "delete", setup: async () => { await page.fill("#del-id", "10"); },
      btn: "del-submit", result: "del-result",
      pred: (r) => r.method() === "DELETE" && pathOf(r) === "/points/10" },
  ];
  const leaked = [];
  for (const c of cases) {
    await selectActive(page, C);
    await openPointsSubtab(page, c.sub);
    await c.setup();
    await withDelay(page, c.pred, 2500, async () => {
      await page.click(`#${c.btn}`);
      await page.waitForTimeout(300);
      await selectActive(page, V);
      await page.waitForTimeout(3500);
    });
    const text = (await page.locator(`#${c.result}`).textContent()) || "";
    if (text.trim() !== "") leaked.push(`${c.name}: "${text.trim().slice(0, 120)}"`);
  }
  expect(leaked, `stale results rendered under the new collection: ${JSON.stringify(leaked)}`).toEqual([]);
});

test("K3 — Report 01 auto-refreshes after point mutations", async ({ page, request }) => {
  await selectActive(page, C);
  await expect.poll(() => reportPoints(page)).not.toBeNull();
  const before = Number(await reportPoints(page));
  expect(Number.isFinite(before)).toBe(true);

  // upsert of a brand new id must bump the report without any manual refresh
  await openPointsSubtab(page, "upsert");
  await page.fill("#up-id", "301");
  await page.fill("#up-text", "k3 auto refresh probe");
  await page.fill("#up-payload", "{}");
  await page.click("#up-submit");
  await expect(page.locator("#up-result")).toContainText("upserted into");
  await expect.poll(async () => Number(await reportPoints(page)), { timeout: 20_000 }).toBe(before + 1);

  // and the same after a delete
  await openPointsSubtab(page, "delete");
  await page.fill("#del-id", "301");
  await page.click("#del-submit");
  await expect(page.locator("#del-result")).toContainText("deleted from");
  await expect.poll(async () => Number(await reportPoints(page)), { timeout: 20_000 }).toBe(before);
  expect((await apiGetPoint(request, C, "301")).status).toBe(404);
});

test("K7 — destructive confirmations: label, trim and case handling", async ({ page }) => {
  await selectActive(page, V);
  await openPointsSubtab(page, "delete");
  await expect(page.locator("#del-confirm-label")).toHaveText(`Type '${V}' to confirm bulk delete`);
  await expect(page.locator("#delf-confirm-label")).toHaveText(`Type '${V}' to confirm delete by filter`);
  await openPointsSubtab(page, "payload");
  await expect(page.locator("#pl-clear-confirm-label")).toHaveText(`Type '${V}' to confirm clear payload`);

  // bulk delete: trim accepted, wrong case rejected
  await openPointsSubtab(page, "delete");
  await page.fill("#del-ids", "8");
  await page.fill("#del-confirm", `  ${V}  `);
  await expect(page.locator("#del-confirm-hint")).toBeHidden();
  await expect(page.locator("#del-bulk-submit")).toBeEnabled();
  await page.fill("#del-confirm", V.toUpperCase());
  await expect(page.locator("#del-confirm-hint")).toBeVisible();
  await expect(page.locator("#del-bulk-submit")).toBeDisabled();
  await page.fill("#del-confirm", "");
  await expect(page.locator("#del-confirm-hint")).toBeHidden();

  // delete by filter
  await page.fill("#delf-json", '{"f_status":"s8"}');
  await page.fill("#delf-confirm", ` ${V}`);
  await expect(page.locator("#delf-confirm-hint")).toBeHidden();
  await expect(page.locator("#delf-submit")).toBeEnabled();
  await page.fill("#delf-confirm", V.toUpperCase());
  await expect(page.locator("#delf-confirm-hint")).toBeVisible();
  await expect(page.locator("#delf-submit")).toBeDisabled();

  // clear payload
  await openPointsSubtab(page, "payload");
  await page.fill("#pl-id", "8");
  await page.fill("#pl-clear-confirm", `${V} `);
  await expect(page.locator("#pl-clear-confirm-hint")).toBeHidden();
  await expect(page.locator("#pl-clear-submit")).toBeEnabled();
  await page.fill("#pl-clear-confirm", V.toUpperCase());
  await expect(page.locator("#pl-clear-confirm-hint")).toBeVisible();
  await expect(page.locator("#pl-clear-submit")).toBeDisabled();
});
