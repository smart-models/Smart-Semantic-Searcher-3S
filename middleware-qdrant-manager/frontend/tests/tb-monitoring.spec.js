// Test book §9 — Monitoring with measurable oracles (M1–M19, TB-MON-001..007).
// Fully intercepted (UI-mock layer): the real console runs against controlled
// replies for all five monitoring routes and OpenAPI, under a frozen clock.
// Rows that need a real profile switch (M-C/M-D) or live services stay in
// e-monitoring.spec.js / z4-switch.spec.js.
import { expect, test } from "./fixtures.js";
import { openConsole } from "./application-fixtures.js";
import { abortResponse, fulfillResponse } from "./telemetry.mjs";

const TICK = 5000;
const BOOK_METRICS = [
  'qdrant_manager_requests_total{endpoint="/query/search",status="200"} 8',
  'qdrant_manager_requests_total{endpoint="/query/search",status="502"} 2',
  'qdrant_manager_request_duration_seconds_bucket{endpoint="/query/search",le="0.1"} 0',
  'qdrant_manager_request_duration_seconds_bucket{endpoint="/query/search",le="0.5"} 10',
  'qdrant_manager_request_duration_seconds_bucket{endpoint="/query/search",le="+Inf"} 10',
].join("\n");
const OPENAPI = { openapi: "3.1.0", paths: { "/query/search": {}, "/points/count": {}, "/livez": {} } };
const MODELS = {
  basic: { dense: "BAAI/bge-m3", sparse: "BAAI/bge-m3", reranker: "BAAI/bge-reranker-v2-m3" },
  advanced: { dense: "Qwen/Qwen3-Embedding-0.6B", sparse: "MILCO", reranker: "Qwen/Qwen3-Reranker-0.6B" },
};

// One monitoring session. `state` is mutable so a test changes replies between
// ticks; every route the console polls is answered from it.
async function openMonitoring(page, overrides = {}) {
  const state = {
    total: 0, rejected: 0, uptime: "1h", metrics: "", openapi: OPENAPI,
    livez: 200, readyz: 200, healthStatus: 200, profile: "basic", models: MODELS.basic,
    qdrant: "reachable", embedder: "reachable", healthState: "healthy", detail: "",
    network: null, ...overrides,
  };
  // OpenAPI is read once at boot, so it must be part of the boot fixture.
  const mock = await openConsole(page, { tab: "query", openapi: state.openapi });
  mock.intercept = async (route, entry) => {
    if (state.network && state.network.includes(entry.path)) { await abortResponse(route); return true; }
    if (entry.path === "/stats") {
      await fulfillResponse(route, { json: { total_requests: state.total, rejected_requests: state.rejected, uptime: state.uptime } });
      return true;
    }
    if (entry.path === "/metrics") { await fulfillResponse(route, { contentType: "text/plain", body: state.metrics }); return true; }
    if (entry.path === "/livez") { await fulfillResponse(route, { status: state.livez, json: { status: state.livez === 200 ? "alive" : "dead" } }); return true; }
    if (entry.path === "/readyz") { await fulfillResponse(route, { status: state.readyz, json: { status: state.readyz === 200 ? "ready" : "not ready" } }); return true; }
    if (entry.path === "/health") {
      const body = { status: state.healthState, qdrant: state.qdrant, embedder: state.embedder, collections: 2,
        embedder_profile: state.profile, embedder_models: state.models, ...(state.detail ? { detail: state.detail } : {}) };
      await fulfillResponse(route, { status: state.healthStatus, json: body }); return true;
    }
    return false;
  };
  mock.state = state;
  mock.statsCalls = () => mock.requestsFor("GET", "/stats").length;
  return mock;
}

// Badge tile by its exact label; the health tile detail may mention other labels.
function badgeValue(page, label) {
  return page.locator("#mon-badges .metric").filter({ has: page.locator(".metric-label", { hasText: new RegExp(`^${label}$`) }) }).locator(".metric-value");
}

// Every committed tick rewrites #mon-badges. Clearing it first makes the five
// tiles prove that *this* tick finished: a re-entry otherwise still shows the
// previous visit's tiles, and the next interval would hit the in-flight guard.
async function clearBadges(page) {
  await page.locator("#mon-badges").evaluate((node) => { node.innerHTML = ""; });
}

async function enterMonitoring(page, mock) {
  const before = mock.statsCalls();
  await clearBadges(page);
  await page.locator("#tab-monitoring").click();
  await expect(page.locator("#panel-monitoring")).toBeVisible();
  await expect.poll(() => mock.statsCalls()).toBe(before + 1);
  await expect(page.locator("#mon-badges .metric")).toHaveCount(5);
}

// Advance the frozen clock by whole intervals. With `mock`, one /stats request
// per interval is required and its tick must have rendered before returning.
async function tick(page, count = 1, mock = null) {
  const before = mock ? mock.statsCalls() : 0;
  if (mock) await clearBadges(page);
  await page.clock.runFor(TICK * count);
  if (!mock) return;
  await expect.poll(() => mock.statsCalls()).toBe(before + count);
  await expect(page.locator("#mon-badges .metric")).toHaveCount(5);
}

function seriesPoints(page) {
  return page.locator("#mon-chart-plot .chart-line").evaluate((line) =>
    Array.from(line.points, (point) => ({ x: point.x, y: point.y })));
}

async function hoverAt(page, fraction) {
  const chart = page.locator("#mon-chart-plot svg");
  await chart.scrollIntoViewIfNeeded();
  const box = await chart.boundingBox();
  const points = await seriesPoints(page);
  const index = Math.round(fraction * (points.length - 1));
  await page.mouse.move(box.x + points[index].x, box.y + 100);
  return index;
}

function finish(mock) {
  expect(mock.unknown).toEqual([]);
  expect(mock.errors).toEqual([]);
}

// ── §9.1 badges and model tiles ──────────────────────────────────────────────

test("M1 — opening #monitoring selects the tab, shows the panel and starts polling", async ({ page }) => {
  const mock = await openMonitoring(page);
  expect(mock.statsCalls()).toBe(0);
  await enterMonitoring(page, mock);
  await expect(page.locator("#tab-monitoring")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#panel-query")).toBeHidden();
  expect(mock.statsCalls()).toBe(1);
  await tick(page, 1, mock);
  await expect.poll(() => mock.statsCalls()).toBe(2);
  finish(mock);
});

test("M2 — the first tick renders five UP badges for livez, readyz, qdrant, embedder and health", async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  const values = page.locator("#mon-badges .metric-value");
  await expect(page.locator("#mon-badges .metric-label")).toHaveText(["livez", "readyz", "qdrant", "embedder", "health"]);
  await expect(values).toHaveText(["UP", "UP", "UP", "UP", "UP"]);
  for (let index = 0; index < 5; index++) await expect(values.nth(index)).toHaveClass(/\bup\b/);
  finish(mock);
});

test("M3 — every badge follows the HTTP status or field of its own endpoint", {
  annotation: { type: "case-partial", description: "TB-MON-005" },
}, async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  const badge = (label) => badgeValue(page, label);
  mock.state.readyz = 503;
  await tick(page, 1, mock);
  await expect(badge("readyz")).toHaveText("DOWN");
  await expect(badge("livez")).toHaveText("UP");
  mock.state.readyz = 200; mock.state.qdrant = "unreachable"; mock.state.healthState = "unhealthy"; mock.state.detail = "qdrant unreachable · embedder reachable";
  await tick(page, 1, mock);
  await expect(badge("qdrant")).toHaveText("DOWN");
  await expect(badge("embedder")).toHaveText("UP");
  await expect(badge("health")).toHaveText("DOWN qdrant unreachable · embedder reachable");
  await expect(badge("health")).toHaveClass(/\bdown\b/);
  mock.state.qdrant = "reachable"; mock.state.healthState = "healthy"; mock.state.detail = "";
  await tick(page, 1, mock);
  await expect(page.locator("#mon-badges .metric-value")).toHaveText(["UP", "UP", "UP", "UP", "UP"]);
  finish(mock);
});

test("M4 — model tiles show the active profile and its dense/sparse/reranker identities", async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  const models = page.locator("#mon-models .metric");
  await expect(models.locator(".metric-label")).toHaveText(["profile", "dense model", "sparse model", "reranker model"]);
  await expect(models.locator(".metric-value")).toHaveText(["basic", MODELS.basic.dense, MODELS.basic.sparse, MODELS.basic.reranker]);
  finish(mock);
});

test("M5 — an unreachable or unknown embedder replaces the model tiles with a readable info alert", async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  mock.state.embedder = "unreachable"; mock.state.healthState = "unhealthy";
  await tick(page, 1, mock);
  await expect(page.locator("#mon-models .alert-info")).toHaveText("Embedder unreachable — model info unavailable");
  await expect(page.locator("#mon-models .metric")).toHaveCount(0);
  mock.state.embedder = "unknown";
  await tick(page, 1, mock);
  await expect(page.locator("#mon-models .alert-info")).toHaveText("Embedder probe not available on this manager");
  await expect(badgeValue(page, "embedder")).toHaveText("DOWN unknown");
  finish(mock);
});

// ── §9.2 statistics and chart ────────────────────────────────────────────────

test("M6 — stats tiles show uptime, total_requests and rejected_requests relative to the baseline", async ({ page }) => {
  const mock = await openMonitoring(page, { total: 40, rejected: 3, uptime: "2h" });
  await enterMonitoring(page, mock);
  const tile = (label) => page.locator("#mon-stats .metric", { hasText: label }).locator(".metric-value");
  await expect(tile("uptime")).toHaveText("2h");
  await expect(tile("total_requests")).toHaveText("40");
  await expect(tile("rejected_requests")).toHaveText("3");
  await page.locator("#mon-reset").click();
  await expect(tile("total_requests")).toHaveText("0");
  mock.state.total = 47; mock.state.rejected = 4;
  await tick(page, 1, mock);
  await expect(tile("total_requests")).toHaveText("7");
  await expect(tile("rejected_requests")).toHaveText("1");
  finish(mock);
});

test("M7 — one sample shows the collecting caption and no SVG", async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  await expect(page.locator("#mon-chart-plot .caption")).toHaveText("Collecting samples — the chart appears after the second refresh.");
  await expect(page.locator("#mon-chart-plot svg")).toHaveCount(0);
  finish(mock);
});

test("M8 — from the second sample the chart carries line, area, dot, title and both timestamp ticks", async ({ page }) => {
  const mock = await openMonitoring(page, { total: 5 });
  await enterMonitoring(page, mock);
  mock.state.total = 9;
  await tick(page, 1, mock);
  const svg = page.locator("#mon-chart-plot svg");
  await expect(svg).toBeVisible();
  await expect(svg.locator(".chart-line")).toHaveCount(1);
  await expect(svg.locator(".chart-area")).toHaveCount(1);
  await expect(svg.locator(".chart-dot")).toHaveCount(1);
  await expect(svg.locator(".chart-title")).toHaveText("total requests");
  await expect(svg).toHaveAttribute("aria-label", /^Total requests over time, from 5 at \d\d:\d\d:\d\d to 9 at \d\d:\d\d:\d\d$/);
  const ticks = await svg.locator("text.chart-tick").allTextContents();
  expect(ticks.filter((text) => /^\d\d:\d\d:\d\d$/.test(text))).toHaveLength(2);
  finish(mock);
});

test("M9 — hovering shows the crosshair and a `hh:mm:ss · N requests` tooltip; leaving hides both", async ({ page }) => {
  const mock = await openMonitoring(page, { total: 5 });
  await enterMonitoring(page, mock);
  mock.state.total = 9;
  await tick(page, 1, mock);
  await hoverAt(page, 1);
  await expect(page.locator("#mon-chart-cross")).toBeVisible();
  await expect(page.locator("#mon-chart-tip")).toHaveText(/^\d\d:\d\d:\d\d · 9 requests$/);
  await page.mouse.move(0, 0);
  await expect(page.locator("#mon-chart-cross")).toBeHidden();
  await expect(page.locator("#mon-chart-tip")).toBeHidden();
  finish(mock);
});

test("M11 — Reset counters realigns the baseline, clears the series and ticks immediately", async ({ page }) => {
  const mock = await openMonitoring(page, { total: 10 });
  await enterMonitoring(page, mock);
  mock.state.total = 12;
  await tick(page, 1, mock);
  await expect(page.locator("#mon-chart-plot svg")).toBeVisible();
  const before = mock.statsCalls();
  await page.locator("#mon-reset").click();
  await expect.poll(() => mock.statsCalls()).toBe(before + 1);
  await expect(page.locator("#mon-stats .metric", { hasText: "total_requests" }).locator(".metric-value")).toHaveText("0");
  await expect(page.locator("#mon-chart-plot .caption")).toContainText("Collecting samples");
  finish(mock);
});

// ── §9.3 metrics table and polling ───────────────────────────────────────────

test("M12 — the metrics table has endpoint/requests/errors/p50/p95 columns and a row per active endpoint", async ({ page }) => {
  const mock = await openMonitoring(page, { metrics: BOOK_METRICS });
  await enterMonitoring(page, mock);
  await expect(page.locator("#mon-table thead th")).toHaveText(["endpoint", "requests", "errors", "p50 (s)", "p95 (s)"]);
  await expect(page.locator("#mon-table tbody tr")).toHaveCount(1);
  await expect(page.locator("#mon-table tbody tr td").first()).toHaveText("/query/search");
  finish(mock);
});

test("M13 — only documented endpoints with requests above zero since the reset are listed", async ({ page }) => {
  const mock = await openMonitoring(page, { metrics: [
    BOOK_METRICS,
    'qdrant_manager_requests_total{endpoint="/undocumented",status="200"} 4',
    'qdrant_manager_requests_total{endpoint="/points/count",status="200"} 0',
  ].join("\n") });
  await enterMonitoring(page, mock);
  await expect(page.locator("#mon-table tbody tr td:first-child")).toHaveText(["/query/search"]);
  await page.locator("#mon-reset").click();
  await expect(page.locator("#mon-table .alert-info")).toHaveText("No request metrics yet");
  mock.state.metrics = [BOOK_METRICS, 'qdrant_manager_requests_total{endpoint="/points/count",status="200"} 3'].join("\n");
  await tick(page, 1, mock);
  await expect(page.locator("#mon-table tbody tr td:first-child")).toHaveText(["/points/count"]);
  finish(mock);
});

test("M14 — p50/p95 come from the histogram when present and read `—` otherwise, never NaN", async ({ page }) => {
  const mock = await openMonitoring(page, { metrics: [
    BOOK_METRICS, 'qdrant_manager_requests_total{endpoint="/points/count",status="200"} 3',
  ].join("\n") });
  await enterMonitoring(page, mock);
  const rows = page.locator("#mon-table tbody tr");
  await expect(rows.nth(0).locator("td")).toHaveText(["/points/count", "3", "0", "—", "—"]);
  await expect(rows.nth(1).locator("td")).toHaveText(["/query/search", "10", "2", "0.3000", "0.4800"]);
  expect(await page.locator("#mon-table").textContent()).not.toMatch(/NaN|Infinity/);
  finish(mock);
});

test("M15 — comment and malformed Prometheus lines are ignored without exceptions", async ({ page }) => {
  const mock = await openMonitoring(page, { metrics: [
    "# HELP qdrant_manager_requests_total total", "# TYPE qdrant_manager_requests_total counter",
    "garbage without a value", 'qdrant_manager_requests_total{endpoint="/query/search",status="200"} not_a_number',
    BOOK_METRICS, "",
  ].join("\n") });
  await enterMonitoring(page, mock);
  await expect(page.locator("#mon-table tbody tr td")).toHaveText(["/query/search", "10", "2", "0.3000", "0.4800"]);
  finish(mock);
});

test("M16 — while the tab stays open one /stats request lands every five seconds", async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  const start = mock.statsCalls();
  for (let index = 1; index <= 4; index++) {
    await tick(page, 1, mock);
    await expect.poll(() => mock.statsCalls()).toBe(start + index);
  }
  await page.clock.runFor(TICK - 1);
  expect(mock.statsCalls()).toBe(start + 4);
  finish(mock);
});

test("M17 — leaving Monitoring stops the timer: no further ticks", async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  await tick(page, 1, mock);
  await page.locator("#tab-query").click();
  const atLeave = mock.statsCalls();
  await tick(page, 3);
  expect(mock.statsCalls()).toBe(atLeave);
  finish(mock);
});

test("M18 — a slow tick never queues a parallel tick", {
  annotation: { type: "case-partial", description: "TB-MON-006" },
}, async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  const held = mock.deferReply("GET", "/stats");
  await tick(page);
  await held.started;
  const during = mock.statsCalls();
  await tick(page, 2);
  expect(mock.statsCalls()).toBe(during);
  held.resolve({ total_requests: 3, rejected_requests: 0, uptime: "1h" });
  await expect(page.locator("#mon-stats .metric", { hasText: "total_requests" }).locator(".metric-value")).toHaveText("3");
  await tick(page, 1, mock);
  await expect.poll(() => mock.statsCalls()).toBe(during + 1);
  finish(mock);
});

test("M19 — re-entering restarts polling with exactly one timer", async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  for (let round = 0; round < 2; round++) {
    await page.locator("#tab-query").click();
    // Re-entry issues exactly one immediate tick, then one per interval.
    await enterMonitoring(page, mock);
  }
  const base = mock.statsCalls();
  await tick(page, 1, mock);
  await tick(page, 1, mock);
  expect(mock.statsCalls()).toBe(base + 2);
  finish(mock);
});

// ── §9.5 measurable oracles ──────────────────────────────────────────────────

test("TB-MON-001 — 125 ticks retain exactly samples 6..125 in timestamp order", async ({ page }) => {
  const mock = await openMonitoring(page, { total: 1 });
  await enterMonitoring(page, mock);
  const svg = page.locator("#mon-chart-plot svg");
  const expectSamples = async (count, first, last) => {
    if (count < 2) { await expect(page.locator("#mon-chart-plot .caption")).toBeVisible(); return; }
    await expect(svg).toHaveAttribute("aria-label", new RegExp(`from ${first} at .* to ${last} at `));
    const points = await seriesPoints(page);
    expect(points).toHaveLength(count);
    expect(points.every((p, i) => i === 0 || (p.x > points[i - 1].x && p.y < points[i - 1].y))).toBe(true);
  };
  await expectSamples(1, 1, 1);
  for (let value = 2; value <= 125; value++) {
    mock.state.total = value;
    await tick(page, 1, mock);
    await expect(page.locator("#mon-stats .metric", { hasText: "total_requests" }).locator(".metric-value")).toHaveText(String(value));
    if (value === 120) await expectSamples(120, 1, 120);
    if (value === 121) await expectSamples(120, 2, 121);
  }
  await expectSamples(120, 6, 125);
  finish(mock);
});

test("TB-MON-002 — tooltip and crosshair report the hovered sample; empty and constant series stay finite", async ({ page }) => {
  const mock = await openMonitoring(page, { total: 7 });
  await enterMonitoring(page, mock);
  for (const value of [7, 7, 7]) { mock.state.total = value; await tick(page, 1, mock); }
  // Constant-valued series: one gridline, finite coordinates, no NaN anywhere.
  const svgHtml = await page.locator("#mon-chart-plot").innerHTML();
  expect(svgHtml).not.toMatch(/NaN|Infinity/);
  expect(await page.locator("#mon-chart-plot .chart-grid").count()).toBe(1);
  for (const value of [9, 12, 20]) { mock.state.total = value; await tick(page, 1, mock); }
  const expected = [7, 7, 7, 7, 9, 12, 20];
  for (const fraction of [0, 0.5, 1]) {
    const index = await hoverAt(page, fraction);
    await expect(page.locator("#mon-chart-tip")).toHaveText(new RegExp(`^\\d\\d:\\d\\d:\\d\\d · ${expected[index]} requests$`));
    await expect(page.locator("#mon-chart-cross")).toBeVisible();
    const left = await page.locator("#mon-chart-cross").evaluate((el) => parseFloat(el.style.left));
    expect(Number.isFinite(left)).toBe(true);
  }
  await page.mouse.move(0, 0);
  await expect(page.locator("#mon-chart-tip")).toBeHidden();
  // Zero samples after a reset: hovering the (absent) chart raises nothing.
  await page.locator("#mon-reset").click();
  await expect(page.locator("#mon-chart-plot .caption")).toBeVisible();
  const wrap = await page.locator("#mon-chart").boundingBox();
  await page.mouse.move(wrap.x + wrap.width / 2, wrap.y + wrap.height / 2);
  await expect(page.locator("#mon-chart-tip")).toBeHidden();
  finish(mock);
});

test("TB-MON-003 — Reset is a view-only baseline: display delta, no server mutation, no inheritance across contexts", async ({ page, browser }) => {
  const mock = await openMonitoring(page, { total: 10, metrics: BOOK_METRICS });
  await enterMonitoring(page, mock);
  const total = page.locator("#mon-stats .metric", { hasText: "total_requests" }).locator(".metric-value");
  await expect(total).toHaveText("10");
  await page.locator("#mon-reset").click();
  await expect(total).toHaveText("0");
  await expect(page.locator("#mon-table .alert-info")).toHaveText("No request metrics yet");
  mock.state.total = 13;
  mock.state.metrics = BOOK_METRICS.replace('status="200"} 8', 'status="200"} 11');
  await tick(page, 1, mock);
  await expect(total).toHaveText("3");
  await expect(page.locator("#mon-table tbody tr td").nth(1)).toHaveText("3");
  expect(mock.requestsFor("POST", "/stats")).toEqual([]);
  expect(mock.requestsFor("POST", "/metrics")).toEqual([]);
  expect(mock.state.total).toBe(13);
  // The series restarted at the reset: only post-reset samples are charted.
  await expect(page.locator("#mon-chart-plot svg")).toHaveAttribute("aria-label", /from 0 at .* to 3 at /);
  const other = await browser.newContext();
  try {
    const fresh = await other.newPage();
    const freshMock = await openMonitoring(fresh, { total: 13 });
    await enterMonitoring(fresh, freshMock);
    await expect(fresh.locator("#mon-stats .metric", { hasText: "total_requests" }).locator(".metric-value")).toHaveText("13");
    finish(freshMock);
  } finally { await other.close(); }
  finish(mock);
});

test("TB-MON-004 — the book fixture yields requests 10, errors 2, p50 0.3000, p95 0.4800; OpenAPI failure keeps valid rows", async ({ page }) => {
  const mock = await openMonitoring(page, { metrics: BOOK_METRICS, openapi: null });
  await enterMonitoring(page, mock);
  // OpenAPI failed: rows are not filtered by documentation, and not blanked.
  await expect(page.locator("#mon-table tbody tr td")).toHaveText(["/query/search", "10", "2", "0.3000", "0.4800"]);
  mock.state.metrics = [
    BOOK_METRICS.split("\n").slice(0, 2).join("\n"),
    'qdrant_manager_requests_total{endpoint="/points/count",status="200"} 0',
    "unrelated garbage line", 'qdrant_manager_request_duration_seconds_bucket{endpoint="/query/search",le="0.1"} not_a_number',
  ].join("\n");
  await tick(page, 1, mock);
  await expect(page.locator("#mon-table tbody tr")).toHaveCount(1);
  await expect(page.locator("#mon-table tbody tr td")).toHaveText(["/query/search", "10", "2", "—", "—"]);
  finish(mock);
});

test("TB-MON-005 — liveness is not readiness, the outage banner takes precedence and recovery leaves no residue", async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  const badge = (label) => badgeValue(page, label);
  mock.state.readyz = 503;
  await tick(page, 1, mock);
  await expect(badge("livez")).toHaveText("UP");
  await expect(badge("readyz")).toHaveText("DOWN");
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
  // Unhealthy dependency: the console-wide gate appears and outranks the
  // incompatibility banner an incompatible collection would otherwise show.
  mock.state.readyz = 200; mock.state.embedder = "unreachable"; mock.state.healthState = "unhealthy";
  mock.state.detail = "qdrant reachable · embedder unreachable";
  mock.compatibility.status = "incompatible";
  await tick(page, 1, mock);
  await expect(badge("embedder")).toHaveText("DOWN");
  await expect(page.locator("#conn-health-status")).toHaveText("UNHEALTHY");
  await expect(page.locator("#service-gate")).toContainText(/unreachable/i);
  await expect(page.locator("#service-gate")).not.toContainText(/Re-embed|different vector space/i);
  await expect(page.locator("#panel-query")).toHaveAttribute("inert", "");
  // Network error on /health: stale success is not painted over the failure.
  mock.state.network = ["/health"];
  await tick(page, 1, mock);
  await expect(badge("health")).toHaveText(/^DOWN/);
  // Recovery under ADVANCED: tiles show the current profile, never stale BASIC values.
  mock.state.network = null; mock.state.embedder = "reachable"; mock.state.healthState = "healthy"; mock.state.detail = "";
  mock.state.profile = "advanced"; mock.state.models = MODELS.advanced; mock.compatibility.status = "compatible";
  await tick(page, 1, mock);
  await expect(page.locator("#mon-badges .metric-value")).toHaveText(["UP", "UP", "UP", "UP", "UP"]);
  await expect(page.locator("#mon-models .metric-value")).toHaveText(["advanced", MODELS.advanced.dense, MODELS.advanced.sparse, MODELS.advanced.reranker]);
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
  await expect(page.locator("#service-gate")).toBeEmpty();
  await expect(page.locator("#panel-query")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#panel-monitoring")).not.toHaveAttribute("inert", "");
  // Absent optional model reads as absent, not as the previous value.
  mock.state.models = { ...MODELS.advanced, reranker: null };
  await tick(page, 1, mock);
  await expect(page.locator("#mon-models .metric", { hasText: "reranker model" }).locator(".metric-value")).toHaveText("—");
  finish(mock);
});

test("TB-MON-006 — a poll held across two intervals never overlaps; away/back leaves one timer; stale failure does not overwrite fresh health", async ({ page }) => {
  const mock = await openMonitoring(page);
  await enterMonitoring(page, mock);
  const held = mock.deferReply("GET", "/stats");
  await tick(page);
  await held.started;
  const during = mock.statsCalls();
  await tick(page, 2);
  expect(mock.statsCalls(), "overlapping aggregate poll while a tick is pending").toBe(during);
  held.resolve({ total_requests: 2, rejected_requests: 0, uptime: "1h" });
  await expect(page.locator("#mon-stats .metric", { hasText: "total_requests" }).locator(".metric-value")).toHaveText("2");
  await page.locator("#tab-query").click();
  await enterMonitoring(page, mock);
  const base = mock.statsCalls();
  for (let index = 0; index < 3; index++) await tick(page, 1, mock);
  expect(mock.statsCalls()).toBe(base + 3);
  // Observe consumption of this poll's five response bodies. A MessageChannel
  // task runs after the promise continuations that decide whether it can render;
  // frozen-clock advancement alone does not wait for those continuations.
  await page.evaluate(() => {
    const remaining = new Set(["/livez", "/readyz", "/health", "/stats", "/metrics"]);
    const readText = Response.prototype.text;
    window.restoreMonitorResponseReader = () => { Response.prototype.text = readText; };
    window.monitorPollSettled = new Promise((resolve) => {
      Response.prototype.text = async function (...args) {
        const text = await readText.apply(this, args);
        const pathname = new URL(this.url).pathname;
        // The deliberately newer Re-test response must not settle the old poll.
        if (pathname !== "/health" || this.status === 503) remaining.delete(pathname);
        if (remaining.size === 0) {
          window.restoreMonitorResponseReader();
          const channel = new MessageChannel();
          channel.port1.onmessage = () => {
            channel.port1.close(); channel.port2.close(); resolve();
          };
          channel.port2.postMessage(null);
        }
        return text;
      };
    });
  });
  // Re-test starts a newer health request while the old monitor poll is held.
  // Merely observing the existing HEALTHY label would not establish staleness.
  const stale = mock.deferReply("GET", "/health");
  try {
    await tick(page);
    await stale.started;
    mock.state.profile = "advanced";
    await page.locator("#conn-retest").click();
    await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
    await expect(page.locator("#conn-health-detail")).toContainText("[advanced]");
    stale.resolve({ status: "unhealthy", qdrant: "unreachable", embedder: "unreachable" }, 503);
    await page.evaluate(() => window.monitorPollSettled);
    await expect(page.locator("#mon-badges .metric-value")).toHaveText(["UP", "UP", "UP", "UP", "UP"]);
    await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
    await expect(page.locator("#conn-health-detail")).toContainText("[advanced]");
  } finally {
    stale.resolve({});
    await page.evaluate(() => window.restoreMonitorResponseReader());
  }
  finish(mock);
});

test("TB-MON-007 — a backend restart that lowers total_requests shows no negative or NaN counter", async ({ page }) => {
  const mock = await openMonitoring(page, { total: 100 });
  await enterMonitoring(page, mock);
  const total = page.locator("#mon-stats .metric", { hasText: "total_requests" }).locator(".metric-value");
  await expect(total).toHaveText("100");
  mock.state.total = 2;
  await tick(page, 1, mock);
  await expect(total).toHaveText("2");
  // With an active baseline the current contract clamps to zero; it does not
  // rebase to the restarted value (recorded behaviour, see the book row).
  mock.state.total = 100;
  await tick(page, 1, mock);
  await page.locator("#mon-reset").click();
  await expect(total).toHaveText("0");
  mock.state.total = 2;
  await tick(page, 1, mock);
  await expect(total).toHaveText("0");
  await expect(page.locator("#mon-models .metric-value").first()).toHaveText("basic");
  expect(await page.locator("#mon-stats").textContent()).not.toMatch(/-\d|NaN/);
  finish(mock);
});
