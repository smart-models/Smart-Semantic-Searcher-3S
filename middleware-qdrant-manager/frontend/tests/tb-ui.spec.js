// Test book §11.1 — operation/state, accessibility and security matrices
// (TB-UI-001/002/003/005/007/008/009, plus TB-EXP-011 and TB-PRO-002).
// UI-mock: every row of the §11.1 operation inventory (`operations` in
// application-fixtures.js) is exercised with held, failed, malformed and
// aborted replies; requests are counted per principal endpoint so follow-up
// report/field refreshes are never mistaken for duplicate operations.
import { expect, test } from "./fixtures.js";
import { abortResponse, fulfillResponse } from "./telemetry.mjs";
import { A, B, compatible, matches, openConsole, openOperation, openPoints, operations, setupOperation, success } from "./application-fixtures.js";

const principal = (op) => op.path || `/collections/${A}${op.suffix}`;
const resultOf = (op) => op.name === "filtered-delete count" ? "delf-result" : op.result;
const button = (page, op) => page.locator(`#${op.button}`);
const idle = async (page, op) => expect(button(page, op)).not.toHaveAttribute("data-busy", "true");

// ── TB-UI-001 — one principal request per user intent ────────────────────────

for (const op of operations) {
  test(`TB-UI-001/${op.name} — double-click and Enter while pending send one request; success and failure both recover`, async ({ page }) => {
    const mock = await openOperation(page, op);
    const pathname = principal(op);
    const downloads = [];
    page.on("download", (download) => downloads.push(download));
    for (const outcome of ["success", "failure"]) {
      const held = mock.deferReply(op.method, pathname);
      const before = mock.requestsFor(op.method, pathname).length;
      const control = button(page, op);
      await control.click();
      const request = await held.started;
      await expect(control).toBeDisabled();
      await control.dblclick({ force: true });
      await control.focus().catch(() => {});
      await page.keyboard.press("Enter");
      await page.waitForTimeout(150);
      expect(mock.requestsFor(op.method, pathname).length, `${outcome}: requests while pending`).toBe(before + 1);
      const received = page.waitForResponse((r) => r.url().split("?")[0].endsWith(pathname) && r.request().method() === op.method);
      if (outcome === "success") held.resolve(success(op, request, mock));
      else held.resolve({ detail: "Operation failed for the test." }, 500);
      await received;
      await idle(page, op);
      if (outcome === "failure") {
        await expect(page.locator(`#${resultOf(op)}`)).toContainText("Operation failed for the test");
        await expect(control).toBeEnabled();
      } else if (op.confirm) {
        // A completed destructive action disarms its confirmation.
        await expect(page.locator(`#${op.confirm}`)).toHaveValue("");
        if (op.name === "reindex start") break;
        await setupOperation(op, mock);
      }
      if (!(await control.isEnabled())) await setupOperation(op, mock);
    }
    expect(mock.requestsFor(op.method, pathname).length).toBe(op.name === "reindex start" ? 1 : 2);
    if (op.download) expect(downloads.length).toBe(1);
    expect(mock.errors).toEqual([]);
  });
}

// ── TB-UI-002 — obsolete replies never overwrite the latest intent ──────────

for (const op of operations.filter((item) => !item.typed)) {
  test(`TB-UI-002/${op.name} — a reply for A released after switching to B is dropped and B then works`, async ({ page }) => {
    const mock = await openOperation(page, op);
    const pathname = principal(op);
    const held = mock.deferReply(op.method, pathname);
    await button(page, op).click();
    const request = await held.started;
    await page.locator("#conn-collection").selectOption(B);
    await expect(page.locator("#cc-report")).toContainText(B);
    const result = page.locator(`#${resultOf(op)}`);
    const before = await result.textContent();
    const received = page.waitForResponse((r) => r.url().split("?")[0].endsWith(pathname) && r.request().method() === op.method);
    held.resolve(success(op, request, mock));
    await received;
    await idle(page, op);
    await expect(result).toHaveText(before);
    // The report for B was not reset by A's late reply.
    await expect(page.locator("#cc-report")).toContainText(B);
    await expect(page.locator("#conn-collection")).toHaveValue(B);
    expect(mock.requestsFor(op.method, pathname)).toHaveLength(1);
    expect(mock.errors).toEqual([]);
  });
}

for (const op of operations.filter((item) => item.typed)) {
  test(`TB-UI-002/${op.name} — a typed-name result keeps its explicit target across an active-collection change`, async ({ page }) => {
    const mock = await openOperation(page, op);
    const pathname = principal(op);
    const held = mock.deferReply(op.method, pathname);
    await button(page, op).click();
    const request = await held.started;
    mock.collections = op.name === "collection delete" ? [A] : [A, B];
    await page.locator("#conn-collection").selectOption(op.name === "collection delete" ? A : B);
    const received = page.waitForResponse((r) => r.url().split("?")[0].endsWith(pathname) && r.request().method() === op.method);
    held.resolve(success(op, request, mock));
    await received;
    await idle(page, op);
    // Create/recreate/delete are scoped to the typed name: the result renders for that
    // target (delete names it and disarms the form; create keeps the typed name).
    if (op.name === "collection delete") await expect(page.locator(`#${op.result}`)).toContainText(`Collection deleted: ${B}`);
    else {
      await expect(page.locator(`#${op.result}`)).toContainText(/created/i);
      await expect(page.locator("#cc-name")).toHaveValue(B);
    }
    expect(mock.errors).toEqual([]);
  });
}

// ── TB-UI-003 — every failure mode is readable and recoverable ──────────────

// 401 comes last: after it the console requires a key before sending anything else.
const FAILURES = [
  ["502", { status: 502, json: { detail: "Upstream embedder unavailable." } }],
  ["504", { status: 504, json: { detail: "Upstream request timed out." } }],
  ["malformed", { status: 200, contentType: "application/json", body: "{not json" }],
  ["network", null],
  ["429", { status: 429, json: { detail: "Rate limit exceeded. Max 300 req/min.", retry_after: 60 } }],
  ["401", { status: 401, json: { detail: "Invalid or missing bearer token." } }],
];
for (const op of operations) {
  test(`TB-UI-003/${op.name} — 401/429/502/504, malformed JSON and a network failure each show a specific error, keep the form and allow a retry`, async ({ page }) => {
    const mock = await openOperation(page, op);
    let mode = null;
    mock.intercept = async (route, request) => {
      if (!matches(op, request) || mode === null) return false;
      if (mode === "valid") { await fulfillResponse(route, { json: success(op, request, mock) }); return true; }
      const failure = FAILURES.find(([label]) => label === mode)[1];
      if (failure === null) await abortResponse(route); else await fulfillResponse(route, failure);
      return true;
    };
    const result = page.locator(`#${resultOf(op)}`);
    const seen = new Set();
    for (const [label] of FAILURES) {
      mode = label;
      const before = mock.requestsFor(op.method, principal(op)).length;
      const previous = await result.textContent();
      await button(page, op).click();
      await expect.poll(() => mock.requestsFor(op.method, principal(op)).length, `${label}: exactly one attempt`).toBe(before + 1);
      await idle(page, op);
      await expect.poll(() => result.textContent()).not.toBe(previous);
      // Filter counts report in a plain caption; every other region uses an error alert.
      if (op.name !== "payload-filter count") await expect(result.locator(".alert-error")).toBeVisible();
      const text = (await (op.name === "payload-filter count" ? result : result.locator(".alert-error").first()).textContent()).trim();
      // Distinct failures produce distinct explanations, never a generic one.
      const expectation = { 401: /401|token|api key/i, 429: /429|rate limit/i, 502: /embedder unavailable/i, 504: /timed out/i,
        malformed: /unexpected response|invalid|parse/i, network: /network|unreachable|failed|fetch|offline/i }[label];
      expect(text, label).toMatch(expectation);
      seen.add(text);
      expect(mock.requestsFor(op.method, principal(op)).length, `${label}: no automatic replay`).toBe(before + 1);
      await expect(button(page, op)).toBeEnabled();
      if (op.confirm) await expect(page.locator(`#${op.confirm}`)).toHaveValue(op.typed ? B : A);
    }
    expect(seen.size).toBe(FAILURES.length);
    // After the 401 the console demands a credential before it sends again.
    await page.locator("#apikey-btn").click();
    await page.locator("#apikey-input").fill("synthetic-test-token");
    await page.locator("#apikey-save").click();
    mode = "valid";
    const before = mock.requestsFor(op.method, principal(op)).length;
    await button(page, op).click();
    await expect.poll(() => mock.requestsFor(op.method, principal(op)).length).toBe(before + 1);
    await idle(page, op);
    await expect(result.locator(".alert-error")).toHaveCount(0);
    expect(mock.errors).toEqual([]);
    expect(mock.unknown).toEqual([]);
  });
}

// ── TB-UI-005 — typed confirmations: exact trimmed equality ─────────────────

const CONFIRMS = operations.filter((op) => op.confirm);
for (const op of CONFIRMS) {
  test(`TB-UI-005/${op.name} — blank, wrong-case and padded-wrong confirmations send nothing; exact/padded-exact arm; a scope change disarms; success disarms`, async ({ page }) => {
    const mock = await openOperation(page, op);
    const pathname = principal(op);
    const target = op.typed ? B : A;
    const confirm = page.locator(`#${op.confirm}`);
    const control = button(page, op);
    for (const [label, value, armed] of [["blank", "", false], ["wrong case", target.toUpperCase(), false], ["other name", `${target}x`, false], ["padded exact", `  ${target}  `, true], ["exact", target, true]]) {
      await confirm.fill(value);
      if (armed) await expect(control, label).toBeEnabled(); else await expect(control, label).toBeDisabled();
      if (!armed) {
        await control.click({ force: true }).catch(() => {});
        await page.keyboard.press("Enter");
      }
    }
    expect(mock.requestsFor(op.method, pathname)).toHaveLength(0);
    // The visible expected name is the exact target.
    const label = page.locator(`[for="${op.confirm}"], #${op.confirm}-label`).first();
    await expect(label).toContainText(target);
    if (!op.typed) {
      // Changing the active collection invalidates the old confirmation.
      await page.locator("#conn-collection").selectOption(B);
      await expect(page.locator("#cc-report")).toContainText(B);
      await expect(label).toContainText(B);
      await expect(control).toBeDisabled();
      expect(mock.requestsFor(op.method, pathname)).toHaveLength(0);
      await page.locator("#conn-collection").selectOption(A);
      await expect(page.locator("#cc-report")).toContainText(A);
      await setupOperation(op, mock);
    } else {
      const nameInput = page.locator(op.name === "collection delete" ? "#cd-name" : "#cc-name");
      await nameInput.fill(`${B}_other`);
      await expect(control).toBeDisabled();
      await nameInput.fill(B);
      await expect(control).toBeEnabled();
    }
    mock.intercept = async (route, request) => {
      if (!matches(op, request)) return false;
      await fulfillResponse(route, { json: success(op, request, mock) }); return true;
    };
    await control.click();
    await idle(page, op);
    expect(mock.requestsFor(op.method, pathname)).toHaveLength(1);
    await expect(confirm).toHaveValue("");
    if (op.name === "recreate") await expect(page.locator("#cc-recreate")).not.toBeChecked();
    else await expect(control).toBeDisabled();
    expect(mock.errors).toEqual([]);
  });
}

// ── TB-UI-007 — hostile content renders as text ──────────────────────────────

const HOSTILE = `<img src=x onerror="window.__xssMarker=(window.__xssMarker||0)+1"> "quoted" & ampersand à漢🙂 <b>bold</b>`;
test("TB-UI-007 — hostile payload text, field names, error messages and collection labels render inert", async ({ page }) => {
  const hostileCollection = `e2e_application_${HOSTILE}`;
  const mock = await openConsole(page, { tab: "collections", payloadKeys: ["text", HOSTILE] });
  mock.collections = [A, B, hostileCollection];
  await page.addInitScript(() => { window.__xssMarker = 0; });
  const marker = () => page.evaluate(() => window.__xssMarker || 0);
  mock.intercept = async (route, request) => {
    if (request.method === "GET" && request.path === `/points/${encodeURIComponent("101")}`) {
      await fulfillResponse(route, { json: { collection_name: A, id: "101", payload: { text: HOSTILE, [HOSTILE]: HOSTILE }, vectors: null } });
      return true;
    }
    if (request.method === "GET" && request.path === "/points/102") {
      await fulfillResponse(route, { status: 404, json: { detail: `Point not found: ${HOSTILE}` } });
      return true;
    }
    return false;
  };
  // Payload text and key.
  await openPoints(page, "get");
  await page.locator("#get-id").fill("101");
  await page.locator("#get-submit").click();
  await expect(page.locator("#get-result")).toContainText("& ampersand à漢🙂");
  await expect(page.locator("#get-result")).toContainText("onerror=");
  expect(await page.locator("#get-result img").count()).toBe(0);
  expect(await page.locator("#get-result b").count()).toBe(0);
  // Error message.
  await page.locator("#get-id").fill("102");
  await page.locator("#get-submit").click();
  await expect(page.locator("#get-result .alert-error")).toContainText("Point not found");
  expect(await page.locator("#get-result img").count()).toBe(0);
  // Collection label in the dropdown and report header.
  await page.locator("#conn-refresh, #conn-collection").first().dispatchEvent("change").catch(() => {});
  await page.reload();
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
  const option = page.locator(`#conn-collection option[value="${hostileCollection.replaceAll('"', '\\"')}"]`);
  await expect(option).toHaveCount(1);
  expect(await option.textContent()).toContain('"quoted" & ampersand');
  expect(await page.locator("#conn-collection img").count()).toBe(0);
  // The payload-field label in Query.
  await page.locator("#tab-query").click();
  await expect(page.locator("#ql-payload-fields label").filter({ hasText: "ampersand" })).toHaveCount(1);
  expect(await page.locator("#ql-payload-fields img").count()).toBe(0);
  expect(await marker()).toBe(0);
  expect(mock.unknown.filter((url) => /x$/.test(url))).toEqual([]);
  expect(mock.errors).toEqual([]);
});

// ── TB-UI-008 — API key handling ─────────────────────────────────────────────

test("TB-UI-008 — a synthetic token is stored, applied as a bearer header, survives reload, is clearable and never leaks", async ({ page, browser }) => {
  const token = "synthetic-test-token-0123456789";
  const mock = await openConsole(page, { tab: "collections", config: { authRequired: true } });
  const authHeaders = () => mock.page.evaluate(() => null);
  const seen = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.origin === "http://application.test" && !url.pathname.startsWith("/assets/") && url.pathname !== "/") {
      seen.push({ path: url.pathname, auth: request.headers().authorization || null });
    }
  });
  await page.locator("#apikey-btn").click();
  await expect(page.locator("#apikey-popover")).toBeVisible();
  await page.locator("#apikey-input").fill(token);
  await page.locator("#apikey-save").click();
  await expect(page.locator("#apikey-status")).toContainText(/registered|saved|active/i);
  await page.locator("#conn-collection").selectOption(B);
  await expect(page.locator("#cc-report")).toContainText(B);
  const withToken = seen.filter((entry) => entry.auth !== null);
  expect(withToken.length).toBeGreaterThan(0);
  for (const entry of withToken) expect(entry.auth).toBe(`Bearer ${token}`);
  // The token never appears in config.js, in the URL or in the DOM text.
  const config = await page.evaluate(() => fetch("/assets/config.js").then((r) => r.text()));
  expect(config).not.toContain(token);
  expect(page.url()).not.toContain(token);
  expect(await page.locator("body").innerText()).not.toContain(token);
  // Reload keeps the credential for this context.
  seen.length = 0;
  await page.reload();
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
  await expect.poll(() => seen.filter((entry) => entry.auth === `Bearer ${token}`).length).toBeGreaterThan(0);
  // A second browser context does not inherit it.
  const other = await browser.newContext();
  const otherPage = await other.newPage();
  const otherSeen = [];
  otherPage.on("request", (request) => { const auth = request.headers().authorization; if (auth) otherSeen.push(auth); });
  const otherMock = await openConsole(otherPage, { tab: "collections", config: { authRequired: true } });
  await otherPage.locator("#conn-collection").selectOption(B);
  await expect(otherPage.locator("#cc-report")).toContainText(B);
  expect(otherSeen).toEqual([]);
  await expect(otherPage.locator("#apikey-btn")).toBeVisible();
  expect(otherMock.errors).toEqual([]);
  await other.close();
  // Clear removes the stored credential.
  await page.locator("#apikey-btn").click();
  await page.locator("#apikey-clear").click();
  seen.length = 0;
  await page.locator("#conn-collection").selectOption(A);
  await expect(page.locator("#cc-report")).toContainText(A);
  await expect.poll(() => seen.length).toBeGreaterThan(0);
  expect(seen.every((entry) => entry.auth === null)).toBe(true);
  expect(await page.evaluate(() => JSON.stringify(Object.entries(localStorage)))).not.toContain(token);
  expect(mock.errors).toEqual([]);
  void authHeaders;
});

// ── TB-UI-009 — automatic refresh after mutations ───────────────────────────

test("TB-UI-009 — a mutation refreshes the report and payload keys automatically; a new key is projectable at once and filterable only once indexed", async ({ page }) => {
  const mock = await openConsole(page, { tab: "query" });
  let points = 10;
  mock.intercept = async (route, request) => {
    if (request.method === "GET" && request.path === `/collections/${A}`) {
      await fulfillResponse(route, { json: { collection_name: A, exists: true, status: "green", points_count: points, dense: { size: 1024, on_disk: true },
        sparse: { on_disk: true, idf: true }, quantization: "turbo", on_disk_payload: true, schema_ok: true, embedding_compatibility: compatible } });
      return true;
    }
    if (request.method === "POST" && request.path === "/points") {
      points += 1;
      mock.payloadKeys = [...mock.payloadKeys, "brand_new"];
      await fulfillResponse(route, { json: { ok: true, collection_name: A, point_id: "500", skipped: false } });
      return true;
    }
    if (request.method === "POST" && request.path === `/collections/${A}/payload-indexes`) {
      mock.indexes = [...mock.indexes, { field_name: "brand_new", field_schema: "keyword" }];
      await fulfillResponse(route, { json: { ok: true, collection_name: A, created: ["brand_new"], updated: [], skipped: [], failed: [] } });
      return true;
    }
    return false;
  };
  await expect(page.locator("#cc-report")).toContainText("10");
  const reportGets = () => mock.requestsFor("GET", `/collections/${A}`).length;
  const keyGets = () => mock.requestsFor("GET", `/collections/${A}/payload-keys`).length;
  const reportsBefore = reportGets(), keysBefore = keyGets();
  await openPoints(page, "upsert");
  await page.locator("#up-id").fill("500");
  await page.locator("#up-text").fill("a brand new point");
  await page.locator("#up-payload").fill('{"brand_new":"yes"}');
  await page.locator("#up-submit").click();
  await expect(page.locator("#up-result")).toContainText(/upserted/i);
  // No manual action: the report and the key list are re-read.
  await expect.poll(reportGets).toBeGreaterThan(reportsBefore);
  await expect.poll(keyGets).toBeGreaterThan(keysBefore);
  await expect(page.locator("#cc-report")).toContainText("11");
  await page.locator("#tab-query").click();
  await expect(page.locator('#ql-payload-fields input[value="brand_new"]')).toHaveCount(1);
  await page.locator("#ql-filter-add").click();
  let offered = await page.locator('#ql-filter-rows [data-role="field"] option').evaluateAll((nodes) => nodes.map((n) => n.value));
  expect(offered).not.toContain("brand_new");
  // Index it: the filter (and facet) selectors now offer the field.
  await page.locator("#tab-collections").click();
  await page.locator("#pi-field").fill("brand_new");
  await page.locator("#pi-schema").selectOption("keyword");
  await page.locator("#pi-create").click();
  await expect(page.locator("#pi-create-result .alert-success")).toBeVisible();
  await page.locator("#tab-query").click();
  await expect.poll(async () => page.locator('#ql-filter-rows [data-role="field"] option').evaluateAll((nodes) => nodes.map((n) => n.value))).toContain("brand_new");
  await openPoints(page, "count");
  await expect(page.locator('#cnt-facet-key option[value="brand_new"]')).toHaveCount(1);
  expect(mock.errors).toEqual([]);
});

// ── TB-EXP-011 — export while a regeneration job is active ──────────────────

for (const status of ["running", "paused", "interrupted"]) {
  test(`TB-EXP-011/${status} — export stays available and delivers the current vectors' payload while a job is ${status}`, async ({ page }) => {
    const mock = await openConsole(page, { tab: "collections", job: { job_id: "test-job", status, total_points: 10, processed_points: 4 } });
    const downloads = [];
    page.on("download", (download) => downloads.push(download));
    mock.intercept = async (route, request) => {
      if (request.method !== "POST" || request.path !== "/points/export") return false;
      await fulfillResponse(route, { json: { chunks: [{ point_id: "1", text: "during job" }], metadata: { n_chunks: 1, collection_name: A } } });
      return true;
    };
    await expect(page.locator("#ce-export")).toBeEnabled();
    // Mutating controls are locked by the job; export is read-only and stays live.
    await expect(page.locator("#cu-submit")).toBeDisabled();
    await page.locator("#ce-export").click();
    await expect.poll(() => downloads.length).toBe(1);
    await expect(page.locator("#ce-result")).toContainText(/Exported/);
    expect(mock.requestsFor("POST", "/points/export")).toHaveLength(1);
    expect(mock.errors).toEqual([]);
  });
}

// ── TB-PRO-002 — §1.1 truth table ────────────────────────────────────────────
// Lockdown model (app.js applyLockdowns/renderServiceGate): regions carry the
// `inert` attribute when locked; #service-gate is the single explanatory banner.

const basicActive = { profile: "basic", fingerprint: "same" };
const STATES = {
  compatible: { compatibility: compatible, banner: null, reindex: true, points: true, toggles: true },
  drift: { compatibility: { status: "incompatible", stored: { profile: "basic", fingerprint: "old" }, active: basicActive, active_vectors: compatible.active_vectors },
    banner: /different vector space[\s\S]*Re-embed/, reindex: true, points: true, toggles: false },
  "profile-switch": { compatibility: { status: "incompatible", stored: { profile: "advanced", fingerprint: "other" }, active: basicActive, active_vectors: compatible.active_vectors },
    banner: null, reindex: true, points: true, toggles: false },
  unverified: { compatibility: { status: "incompatible", stored: null, active: basicActive, active_vectors: compatible.active_vectors },
    banner: /no verified embedding contract/, reindex: false, points: false, toggles: false },
};
const inert = (page, id) => page.locator(`#${id}`).evaluate((el) => el.hasAttribute("inert"));

for (const [state, expected] of Object.entries(STATES)) {
  for (const outage of [false, true]) {
    test(`TB-PRO-002/${state}${outage ? "-outage" : ""} — banner, reindex availability and region locks follow the §1.1 truth table`, async ({ page }) => {
      const mock = await openConsole(page, { tab: "collections", compatibility: expected.compatibility });
      const gate = page.locator("#service-gate");
      if (outage) {
        mock.health = { status: "unhealthy", qdrant: "reachable", embedder: "unreachable", collections: 2 };
        await page.locator("#conn-retest").click();
        await expect(page.locator("#conn-health-status")).toHaveText("UNHEALTHY");
        // The outage explanation takes precedence over any compatibility overlay.
        await expect(gate.locator(".alert-error")).toContainText("Smart Embedder unreachable. Collection, point, and query operations are disabled");
        expect(await inert(page, "panel-collections")).toBe(true);
        expect(await inert(page, "panel-points")).toBe(true);
        expect(await inert(page, "panel-query")).toBe(true);
        await expect(page.locator("#conn-collection")).toBeDisabled();
        // Recovery removes the lock and the banner with no residue.
        mock.health = { status: "healthy", qdrant: "reachable", embedder: "reachable", collections: 2 };
        await page.locator("#conn-retest").click();
        await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
        expect(await inert(page, "panel-collections")).toBe(false);
      }
      if (expected.banner) await expect(gate.locator(".alert-error")).toContainText(expected.banner);
      else await expect(gate).toBeEmpty();
      expect(await inert(page, "cr-wrap"), "reindex region").toBe(!expected.reindex);
      expect(await inert(page, "panel-points"), "points region").toBe(!expected.points);
      expect(await inert(page, "panel-query"), "query region").toBe(state !== "compatible");
      expect(await inert(page, "cu-wrap"), "schema toggles").toBe(!expected.toggles);
      expect(await inert(page, "pi-wrap"), "payload indexes").toBe(!expected.toggles);
      // Create (new collection), delete (typed name) and export stay available in every state.
      expect(await inert(page, "cc-wrap")).toBe(false);
      expect(await inert(page, "cd-wrap")).toBe(false);
      await expect(page.locator("#ce-export")).toBeEnabled();
      expect(await inert(page, "cc-recreate-control")).toBe(state !== "compatible");
      expect(mock.errors).toEqual([]);
    });
  }
}
// ── TB-EXP-008 (UI clause) — over-cap export ────────────────────────────────

test("TB-EXP-008 — a 413 export renders the cap error, starts no download and recovers on a within-cap collection", async ({ page }) => {
  const mock = await openConsole(page, { tab: "collections" });
  const downloads = [];
  page.on("download", (download) => downloads.push(download));
  mock.intercept = async (route, request) => {
    if (request.method !== "POST" || request.path !== "/points/export") return false;
    if (request.body.collection_name === A) {
      await fulfillResponse(route, { status: 413, json: { detail: "Collection 'e2e_application_a' has 4 points, above EXPORT_MAX_POINTS=3 — raise the limit or export a smaller collection." } });
    } else {
      await fulfillResponse(route, { json: { chunks: [{ point_id: "1", text: "within cap" }], metadata: { n_chunks: 1, collection_name: B } } });
    }
    return true;
  };
  await page.locator("#ce-export").click();
  await expect(page.locator("#ce-result .alert-error")).toContainText("EXPORT_MAX_POINTS=3");
  await expect(page.locator("#ce-export")).toBeEnabled();
  expect(downloads).toEqual([]);
  await page.locator("#conn-collection").selectOption(B);
  await expect(page.locator("#cc-report")).toContainText(B);
  await expect(page.locator("#ce-result")).toBeEmpty();
  await page.locator("#ce-export").click();
  await expect.poll(() => downloads.length).toBe(1);
  await expect(page.locator("#ce-result")).toContainText(/Exported/);
  expect(mock.requestsFor("POST", "/points/export")).toHaveLength(2);
  expect(mock.errors).toEqual([]);
});

// ── TB-ING-003 — ingest succeeds, one index fails ───────────────────────────

test("TB-ING-003 — an index-phase failure is reported apart from the completed ingest; repair touches only the failed index; a switch drops the late index reply", async ({ page }) => {
  const mock = await openConsole(page, { tab: "query", indexes: [] });
  const chunks = [{ id: 1, text: "typed rows", good_field: "a", bad_field: "b" }];
  let indexMode = "fail";
  const held = { promise: null, resolve: null };
  mock.intercept = async (route, request) => {
    if (request.method === "POST" && request.path === "/points/ingest-chunks") {
      await fulfillResponse(route, { json: { ok: true, collection_name: request.body.collection_name, ingested: 1, skipped: 0 } });
      return true;
    }
    if (request.method === "POST" && /\/payload-indexes$/.test(request.path)) {
      const fields = request.body.indexes.map((item) => item.field_name);
      if (indexMode === "hold") await held.promise;
      const failed = indexMode === "fail" ? fields.filter((field) => field === "bad_field") : [];
      const created = fields.filter((field) => !failed.includes(field));
      mock.indexes = [...mock.indexes, ...created.map((field) => ({ field_name: field, field_schema: "keyword" }))];
      await fulfillResponse(route, { json: { ok: failed.length === 0, collection_name: A, created, updated: [], skipped: [], failed } });
      return true;
    }
    return false;
  };
  const ingest = async (rows = chunks, fields = ["good_field", "bad_field"]) => {
    await openPoints(page, "ingest");
    await page.locator("#ing-file").setInputFiles({ name: "typed.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ chunks: rows })) });
    await expect(page.locator("#ing-map-table")).toBeVisible();
    for (const field of fields) {
      // A field the collection already indexes is locked to that type.
      const type = page.locator(`#ing-map-table tr[data-field="${field}"] .ing-map-type`);
      if (await type.isEnabled()) await type.selectOption("keyword");
    }
    await expect(page.locator("#ing-create-indexes")).toBeChecked();
    await page.locator("#ing-confirm-submit").click();
  };
  await ingest();
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");
  const summary = page.locator("#ing-index-summary");
  await expect(summary).toContainText(/1 failed/);
  await expect(summary.locator('[data-field="bad_field"]')).toHaveAttribute("data-status", "failed");
  await expect(summary.locator('[data-field="good_field"]')).toHaveAttribute("data-status", "created");
  await expect(page.locator("#ing-result")).not.toContainText(/rolled back/i);
  expect(mock.requestsFor("POST", "/points/ingest-chunks")).toHaveLength(1);
  expect(mock.requestsFor("POST", `/collections/${A}/payload-indexes`)).toHaveLength(1);
  // Manual repair creates only the missing index.
  indexMode = "ok";
  await page.locator("#tab-collections").click();
  await page.locator("#pi-field").fill("bad_field");
  await page.locator("#pi-schema").selectOption("keyword");
  await page.locator("#pi-create").click();
  await expect(page.locator("#pi-create-result .alert-success")).toBeVisible();
  const repair = mock.requestsFor("POST", `/collections/${A}/payload-indexes`).at(-1);
  expect(repair.body.indexes.map((item) => item.field_name)).toEqual(["bad_field"]);
  expect(mock.requestsFor("POST", "/points/ingest-chunks")).toHaveLength(1);
  // A collection switch during a delayed index reply: the old reply cannot touch B.
  indexMode = "hold";
  held.promise = new Promise((resolve) => { held.resolve = resolve; });
  await ingest([{ id: 2, text: "late index", late_field: "z" }], ["late_field"]);
  await expect(page.locator("#ing-result")).toContainText(/Creating payload indexes/);
  await page.locator("#conn-collection").selectOption(B);
  await expect(page.locator("#cc-report")).toContainText(B);
  const indexesBefore = mock.requestsFor("GET", `/collections/${B}/payload-indexes`).length;
  held.resolve();
  await page.waitForTimeout(300);
  await expect(page.locator("#ing-index-summary")).toHaveCount(0);
  expect(mock.requestsFor("POST", `/collections/${B}/payload-indexes`)).toHaveLength(0);
  expect(mock.requestsFor("GET", `/collections/${B}/payload-indexes`).length).toBe(indexesBefore);
  expect(mock.errors).toEqual([]);
});

// ── TB-QRY-005 — runtime boundaries from the fetched config ─────────────────

test("TB-QRY-005 — limit/candidate/top-k boundaries come from the fetched config: minima and maxima send, max+1/zero/fraction/candidate<limit/top-k>candidate block Run naming the bound; payload-only stays at 5", async ({ page }) => {
  const config = { queryMaxLimit: 40, queryMaxCandidateLimit: 300, queryMaxRerankTopK: 150, queryDefaultLimit: 10, queryDefaultCandidateLimit: 50, queryDefaultRerankTopK: 20 };
  const mock = await openConsole(page, { tab: "query", config });
  const fetched = await page.evaluate(() => ({ ...window.QHM_CONFIG }));
  expect(fetched).toMatchObject(config);
  mock.intercept = async (route, request) => {
    if (request.path !== "/query/search" && request.path !== "/points/scroll") return false;
    const point = { id: "1", payload: { text: "hit" }, vectors: null };
    await fulfillResponse(route, { json: request.path === "/points/scroll"
      ? { collection_name: A, returned: 1, total: 1, results: [point], next_page_offset: null }
      : { collection_name: A, returned: 1, total: 1, fusion: "rrf", reranked: true, candidates: 1, skipped_without_text: 0, results: [{ ...point, score: 0.5 }], next_page_offset: null } });
    return true;
  };
  const run = page.locator("#ql-run"), errors = page.locator("#ql-errors .alert-error");
  const set = async (id, value) => { await page.locator(`#${id}`).fill(String(value)); await page.locator(`#${id}`).dispatchEvent("input"); };
  await page.locator("#ql-query").fill("boundary");
  await page.locator("#ql-rerank").check();
  await page.locator("#ql-topk-en").check();
  const valid = [
    ["limit-min", 1, fetched.queryDefaultCandidateLimit, fetched.queryDefaultRerankTopK],
    ["limit-max", fetched.queryMaxLimit, fetched.queryMaxCandidateLimit, fetched.queryMaxRerankTopK],
    ["candidate-min", 1, 1, 1], ["top-k-min", 5, 20, 1],
    ["all-max", fetched.queryMaxLimit, fetched.queryMaxCandidateLimit, fetched.queryMaxRerankTopK],
  ];
  for (const [label, limit, candidate, topK] of valid) {
    await set("ql-limit", limit); await set("ql-cand", candidate); await set("ql-topk", topK);
    await expect(run, label).toBeEnabled();
    await expect(errors).toHaveCount(0);
    const before = mock.requestsFor("POST", "/query/search").length;
    await run.click();
    await expect.poll(() => mock.requestsFor("POST", "/query/search").length).toBe(before + 1);
    expect(mock.requestsFor("POST", "/query/search").at(-1).body).toMatchObject({ limit, candidate_limit: candidate, rerank_top_k: topK });
  }
  const invalid = [
    ["limit max+1", { "ql-limit": fetched.queryMaxLimit + 1 }, new RegExp(`limit must be an integer between 1 and ${fetched.queryMaxLimit}`)],
    ["limit zero", { "ql-limit": 0 }, /limit must be an integer between 1/],
    ["limit fraction", { "ql-limit": 2.5 }, /limit must be an integer/],
    ["candidate max+1", { "ql-limit": 5, "ql-cand": fetched.queryMaxCandidateLimit + 1 }, new RegExp(`candidate_limit must be an integer between 1 and ${fetched.queryMaxCandidateLimit}`)],
    ["candidate below limit", { "ql-limit": 10, "ql-cand": 5 }, /candidate_limit must be greater than or equal to limit/],
    ["top-k max+1", { "ql-limit": 5, "ql-cand": 50, "ql-topk": fetched.queryMaxRerankTopK + 1 }, new RegExp(`rerank_top_k must be an integer between 1 and ${fetched.queryMaxRerankTopK}`)],
    ["top-k zero", { "ql-topk": 0 }, /rerank_top_k must be an integer between 1/],
  ];
  for (const [label, values, pattern] of invalid) {
    for (const [id, value] of Object.entries(values)) await set(id, value);
    await expect(run, label).toBeDisabled();
    await expect(errors.first()).toHaveText(pattern);
  }
  const searches = mock.requestsFor("POST", "/query/search").length;
  // Payload-only: the page size is fixed at five and hybrid-only controls are not sent.
  await set("ql-topk", 20); await set("ql-cand", 50); await set("ql-limit", 7);
  await page.locator("#ql-query").fill("");
  await expect(page.locator("#ql-limit")).toBeDisabled();
  await expect(run).toBeEnabled();
  await run.click();
  await expect.poll(() => mock.requestsFor("POST", "/points/scroll").length).toBe(1);
  const scroll = mock.requestsFor("POST", "/points/scroll")[0].body;
  expect(scroll.limit).toBe(5);
  for (const key of ["candidate_limit", "rerank_top_k", "rerank", "fusion", "query"]) expect(scroll).not.toHaveProperty(key);
  expect(mock.requestsFor("POST", "/query/search")).toHaveLength(searches);
  expect(mock.errors).toEqual([]);
});

// ── TB-UI-004 — responsive states and a keyboard-only pass ──────────────────

for (const [width, height] of [[320, 720], [768, 1024], [1440, 900]]) {
  test(`TB-UI-004/${width}x${height} — initial, loading, error and success states keep labels visible, no page-level overflow, focus visible and errors associated`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    const mock = await openConsole(page, { tab: "points" });
    const noOverflow = async (label) => {
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow, `${label}: page-level horizontal overflow`).toBeLessThanOrEqual(0);
    };
    await noOverflow("initial");
    for (const tab of ["collections", "query", "monitoring", "points"]) {
      await page.locator(`#tab-${tab}`).click();
      await expect(page.locator(`#panel-${tab}`)).toBeVisible();
      await noOverflow(tab);
    }
    await openPoints(page, "get");
    const field = page.locator("#get-id");
    await expect(page.locator('label[for="get-id"], label:has(#get-id)').first()).toBeVisible();
    // Loading: the pending button explains itself and is not clipped.
    await field.fill("1");
    const held = mock.deferReply("GET", "/points/1");
    await page.locator("#get-submit").click();
    await held.started;
    await expect(page.locator("#get-submit")).toBeDisabled();
    await expect(page.locator("#get-submit")).toHaveAccessibleDescription(/progress/i);
    expect(await page.locator("#get-submit").boundingBox()).not.toBeNull();
    // Error: readable, associated with the result region.
    held.resolve({ detail: "Point not found: 1" }, 404);
    await expect(page.locator("#get-result .alert-error")).toContainText("Point not found");
    await expect(page.locator("#get-result")).toHaveAttribute("aria-live", "polite");
    await noOverflow("error");
    // Success at this width: the JSON panel is inside a scrollable container, not the page.
    mock.intercept = async (route, request) => {
      if (request.method !== "GET" || request.path !== "/points/1") return false;
      await fulfillResponse(route, { json: { collection_name: A, id: "1", payload: { text: "x".repeat(400), category: "wide" }, vectors: null } });
      return true;
    };
    await page.locator("#get-submit").click();
    await expect(page.locator("#get-result")).toContainText("wide");
    await noOverflow("success");
    // Keyboard-only: Tab reaches the input and the button, focus is visible.
    await page.locator("#get-id").focus();
    await page.keyboard.press("Tab");
    const focusedAfterTab = await page.evaluate(() => ({ id: document.activeElement?.id, visible: document.activeElement?.matches(":focus-visible") }));
    expect(focusedAfterTab.id).toBeTruthy();
    expect(focusedAfterTab.visible).toBe(true);
    let reached = false;
    for (let step = 0; step < 6 && !reached; step++) {
      reached = await page.evaluate(() => document.activeElement?.id === "get-submit");
      if (!reached) await page.keyboard.press("Tab");
    }
    expect(reached, "Get button reachable by keyboard").toBe(true);
    // Disabled reason is discoverable: clear the id and read the button's description.
    await page.locator("#get-id").fill("");
    await expect(page.locator("#get-submit")).toBeDisabled();
    await expect(page.locator("#get-submit")).toHaveAccessibleDescription(/point id/i);
    // 200% zoom keeps the result panel readable; the book pairs zoom with the
    // tablet/desktop widths (320 px at 200% is a 160 px viewport, outside the matrix).
    if (width >= 768) {
      await page.evaluate(() => { document.documentElement.style.zoom = "2"; });
      await expect(page.locator("#get-result")).toBeVisible();
      await noOverflow("zoom");
      await page.evaluate(() => { document.documentElement.style.zoom = ""; });
    }
    expect(mock.errors).toEqual([]);
  });
}

// ── TB-UI-006 — dynamic controls by keyboard ────────────────────────────────

test("TB-UI-006 — nested filter add/remove, mapping edits, file replacement, select/details/popover keep focus on live controls with no trap", async ({ page }) => {
  const mock = await openConsole(page, { tab: "query" });
  const active = () => page.evaluate(() => {
    const el = document.activeElement;
    return { id: el?.id || "", role: el?.dataset?.role || "", action: el?.dataset?.action || "", tag: el?.tagName, inDoc: document.body.contains(el) && el !== document.body };
  });
  // Nested group: add, edit via keyboard, remove → focus lands on a live control.
  await page.locator("#ql-filter-nested-add").focus();
  await page.keyboard.press("Enter");
  const group = page.locator("#ql-filter-rows .filter-group-card.is-nested").first();
  await expect(group).toBeVisible();
  const rowField = group.locator('.filter-row [data-role="field"]').first();
  await expect(rowField).toHaveAttribute("aria-label", /.+/);
  await rowField.focus();
  await page.keyboard.press("Tab");
  expect((await active()).role).toBe("op");
  await page.keyboard.press("Shift+Tab");
  expect((await active()).role).toBe("field");
  await group.locator('.filter-row [data-action="remove"]').first().focus();
  await page.keyboard.press("Enter");
  let focus = await active();
  expect(focus.inDoc, "focus left the document after a row removal").toBe(true);
  expect(focus.tag).toMatch(/BUTTON|SELECT|INPUT/);
  await group.locator(':scope > .filter-group-toolbar > [data-action="remove"]').focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#ql-filter-rows .filter-group-card")).toHaveCount(0);
  focus = await active();
  expect(focus.inDoc).toBe(true);
  expect(focus.tag).toMatch(/BUTTON|SELECT|INPUT/);
  // Mapping rows: rename by keyboard, error names the row, file replacement keeps a live focus.
  await openPoints(page, "ingest");
  await page.locator("#ing-file").setInputFiles({ name: "first.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ chunks: [{ id: 1, text: "a", alpha: "x" }] })) });
  await expect(page.locator("#ing-map-table")).toBeVisible();
  const name = page.locator('#ing-map-table tr[data-field="alpha"] .ing-map-name');
  await expect(name).toHaveAccessibleName(/alpha/);
  await name.focus();
  await page.keyboard.press("Control+A");
  await page.keyboard.type("text");
  await expect(page.locator("#ing-map-errors")).toContainText(/field 'alpha'/);
  await page.keyboard.press("Tab");
  focus = await active();
  expect(focus.inDoc).toBe(true);
  // A keyboard user replaces the file from the file control itself.
  await page.locator("#ing-file").focus();
  await page.locator("#ing-file").setInputFiles({ name: "second.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ chunks: [{ id: 1, text: "b", beta: "y" }] })) });
  await expect(page.locator('#ing-map-table tr[data-field="beta"]')).toBeVisible();
  await expect(page.locator('#ing-map-table tr[data-field="alpha"]')).toHaveCount(0);
  focus = await active();
  expect(focus.inDoc).toBe(true);
  expect(focus.id).toBe("ing-file");
  // Popover: open with Enter, close with Escape, focus returns to the trigger.
  await page.locator("#apikey-btn").focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#apikey-popover")).toBeVisible();
  expect((await active()).id).toBe("apikey-input");
  await page.keyboard.press("Escape");
  await expect(page.locator("#apikey-popover")).toBeHidden();
  expect((await active()).id).toBe("apikey-btn");
  // Details: toggled by Enter and Space on its summary.
  const summary = page.locator("details > summary:visible").first();
  await expect(summary).toBeVisible();
  await summary.focus();
  const details = summary.locator("..");
  const open = () => details.evaluate((el) => el.open);
  const before = await open();
  await page.keyboard.press("Enter");
  await expect.poll(open).toBe(!before);
  await page.keyboard.press("Space");
  await expect.poll(open).toBe(before);
  expect(mock.errors).toEqual([]);
});

// ── NF-P20 — untouched new controls leave the legacy requests unchanged ────

test("book:13.8/NF-P20 — Upsert, Batch, Get, Get-many and Count without touching the new controls send the pre-feature request shapes", {
  annotation: { type: "case", description: "book:13.8/NF-P20" },
}, async ({ page }) => {
  const mock = await openConsole(page, { tab: "points" });
  mock.intercept = async (route, request) => {
    const replies = {
      "POST /points": { ok: true, collection_name: A, point_id: "101", skipped: false },
      "POST /points/batch": { ok: true, collection_name: A, upserted: 1, skipped: 0 },
      "GET /points/101": { collection_name: A, id: "101", payload: { text: "t" }, vectors: null },
      "POST /points/retrieve": { collection_name: A, returned: 1, total: 1, results: [{ id: "101", payload: { text: "t" }, vectors: null }], next_page_offset: null },
      "POST /points/count": { ok: true, collection_name: A, count: 1 },
    };
    const reply = replies[`${request.method} ${request.path}`];
    if (!reply) return false;
    await fulfillResponse(route, { json: reply });
    return true;
  };
  const urls = [];
  page.on("request", (r) => { const u = new URL(r.url()); if (u.pathname === "/points/101") urls.push(u.search); });
  await openPoints(page, "upsert");
  await page.locator("#up-id").fill("101"); await page.locator("#up-text").fill("t"); await page.locator("#up-payload").fill("{}");
  await page.locator("#up-submit").click();
  await expect(page.locator("#up-result")).toContainText(/upserted/);
  expect(mock.requestsFor("POST", "/points")[0].body).toEqual({ collection_name: A, point_id: "101", text: "t", payload: {}, update_mode: "upsert" });
  await openPoints(page, "batch");
  await page.locator("#bt-json").fill('[{"point_id":"101","text":"t"}]');
  await page.locator("#bt-submit").click();
  await expect(page.locator("#bt-result")).toContainText(/upserted/);
  expect(mock.requestsFor("POST", "/points/batch")[0].body).toEqual({ collection_name: A, points: [{ point_id: "101", text: "t" }], update_mode: "upsert" });
  await openPoints(page, "get");
  await page.locator("#get-id").fill("101");
  await page.locator("#get-submit").click();
  await expect(page.locator("#get-result")).toContainText("101");
  expect(urls.every((search) => !/with_vectors=true/.test(search))).toBe(true);
  await page.locator("#get-ids").fill("101");
  await page.locator("#get-many-submit").click();
  await expect(page.locator("#get-many-result")).toContainText("101");
  expect(mock.requestsFor("POST", "/points/retrieve")[0].body).toMatchObject({ collection_name: A, ids: ["101"] });
  expect(mock.requestsFor("POST", "/points/retrieve")[0].body.with_vectors ?? false).toBe(false);
  await openPoints(page, "count");
  await page.locator("#cnt-submit").click();
  await expect(page.locator("#cnt-result")).toContainText("1");
  const count = mock.requestsFor("POST", "/points/count")[0].body;
  expect(count).toMatchObject({ collection_name: A, exact: true });
  expect(count.filters ?? {}).toEqual({});
  expect(mock.errors).toEqual([]);
});
for (const ids of [[1, 1], [9007199254740992]]) {
  test(`TB-ID-004/preflight-${ids.join("-")} — invalid ordinals keep ingest disabled without transport`, async ({ page }) => {
    const mock = await openConsole(page, { tab: "points" });
    await page.locator('[data-pt-tab="ingest"]').click();
    const content = JSON.stringify({ chunks: ids.map((id) => ({ id, text: "text" })) });
    await page.locator("#ing-file").setInputFiles({ name: "invalid.json", mimeType: "application/json", buffer: Buffer.from(content) });
    await expect(page.locator("#pt-ingest")).toContainText(/duplicate|safe integer/);
    await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
    expect(mock.requestsFor("POST", "/points/ingest-chunks")).toHaveLength(0);
    expect(mock.errors).toEqual([]);
  });
}
