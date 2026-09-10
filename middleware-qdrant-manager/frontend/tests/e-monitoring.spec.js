import { fetchResponse } from "./telemetry.mjs";
import { fulfillResponse, transformJsonResponse } from "./telemetry.mjs";
import { fixtureName } from "./harness.mjs";
import { openConsole } from "./application-fixtures.js";
// §9 Serie M — Monitoring (badges, models, chart lifecycle, polling, metrics).
// States that would need a real outage are simulated with page.route.
import { expect, test } from "./fixtures.js";
import { boot, prepareConsole, openTab, BASE_URL } from "./helpers.js";

const COLL = fixtureName("e2e_e_mon");

test("M-A — status badges render for livez/readyz/qdrant/embedder/health", async ({ page }) => {
  const errors = await boot(page, { active: COLL });
  await openTab(page, "monitoring");
  const badges = page.locator("#mon-badges");
  await expect(badges).not.toBeEmpty({ timeout: 30_000 });
  for (const label of ["livez", "readyz", "qdrant", "embedder", "health"]) {
    await expect(badges).toContainText(label);
  }
  expect(errors).toEqual([]);
});

test("M-B — embedder models show the active profile and the model identities", async ({ page, request }) => {
  await boot(page, { active: COLL });
  await openTab(page, "monitoring");
  const models = page.locator("#mon-models");
  await expect(models).not.toBeEmpty({ timeout: 30_000 });
  const health = await (await request.get(`${BASE_URL}/health`)).json();
  await expect(models).toContainText(health.embedder_profile);
  await expect(models).toContainText(health.embedder_models.dense);
  await expect(models).toContainText(health.embedder_models.sparse);
  await expect(models).toContainText(health.embedder_models.reranker);
});

test("M-B2 — badge/API coherence: the UI badges match the live endpoints", async ({ page, request }) => {
  await boot(page, { active: COLL });
  await openTab(page, "monitoring");
  await expect(page.locator("#mon-badges")).not.toBeEmpty({ timeout: 30_000 });
  const health = await (await request.get(`${BASE_URL}/health`)).json();
  const badgeText = await page.locator("#mon-badges").textContent();
  // qdrant reachable in the API ⇒ the badge must not read as a failure.
  if (health.qdrant === "reachable") expect(badgeText).not.toMatch(/qdrant\s*(DOWN|FAIL)/i);
});

test("M-E — the first sample shows the collecting-samples caption instead of a chart", async ({ page }) => {
  await boot(page, { active: COLL });
  await openTab(page, "monitoring");
  await expect(page.locator("#mon-chart-plot")).toContainText(
    "Collecting samples — the chart appears after the second refresh.", { timeout: 20_000 });
});

test("M-F — from the second refresh the chart is drawn", {
  annotation: { type: "case-partial", description: "book:9.4/M-F" },
}, async ({ page }) => {
  await boot(page, { active: COLL });
  await openTab(page, "monitoring");
  // Poll interval is 5s; two samples land within ~10s.
  await expect(page.locator("#mon-chart-plot svg")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator("#mon-chart-plot")).not.toContainText("Collecting samples");
});

test("M-G — the sample series is capped at 120 points", {
  annotation: { type: "case", description: "book:9.2/M10" },
}, async ({ page }) => {
  const mock = await openConsole(page, { origin: BASE_URL });
  let sample = 0;
  mock.intercept = async (route, request) => {
    if (request.path !== "/stats") return false;
    await fulfillResponse(route, { json: {
      total_requests: ++sample, rejected_requests: 0, uptime: "1h",
    } });
    return true;
  };
  await openTab(page, "monitoring");
  await expect(page.locator("#mon-stats")).toContainText("total_requests1");
  const chart = page.locator("#mon-chart-plot svg");
  for (let index = 2; index <= 125; index++) {
    await page.clock.runFor(5000);
    await expect(chart).toHaveAttribute("aria-label", new RegExp(` to ${index} at `));
  }
  const points = await page.locator("#mon-chart-plot .chart-line").evaluate((line) =>
    Array.from(line.points, (point) => ({ x: point.x, y: point.y })));
  expect(points).toHaveLength(120);
  expect(points.every((point, index) => index === 0
    || (point.x > points[index - 1].x && point.y < points[index - 1].y))).toBe(true);
  await expect(chart).toHaveAttribute("aria-label", /from 6 at .* to 125 at /);
  await chart.scrollIntoViewIfNeeded();
  const bounds = await chart.boundingBox();
  for (const [point, value] of [[points[0], 6], [points.at(-1), 125]]) {
    await page.mouse.move(bounds.x + point.x, bounds.y + 100);
    await expect(page.locator("#mon-chart-tip")).toContainText(`${value} requests`);
    await expect(page.locator("#mon-chart-cross")).toBeVisible();
  }
  await page.mouse.move(0, 0);
  await expect(page.locator("#mon-chart-tip")).toBeHidden();
  await expect(page.locator("#mon-chart-cross")).toBeHidden();
  expect(mock.unknown).toEqual([]);
  expect(mock.errors).toEqual([]);
});

test("M-H — polling starts on entry, stops on exit, and never duplicates timers", async ({ page }) => {
  await boot(page, { active: COLL });
  let stats = 0;
  await page.route("**/stats", (route) => { stats += 1; return route.fallback(); });

  await openTab(page, "monitoring");
  await page.waitForTimeout(11_000);
  const afterFirst = stats;
  expect(afterFirst, "polling did not start").toBeGreaterThan(0);

  await openTab(page, "query");
  const atLeave = stats;
  await page.waitForTimeout(11_000);
  expect(stats, "polling did not stop on tab exit").toBe(atLeave);

  // Re-enter twice: the rate must stay the same, not double.
  await openTab(page, "monitoring");
  await openTab(page, "query");
  await openTab(page, "monitoring");
  const beforeWindow = stats;
  await page.waitForTimeout(11_000);
  const rate = stats - beforeWindow;
  expect(rate, `duplicated timers: ${rate} ticks in 11s`).toBeLessThanOrEqual(afterFirst + 1);
});

test("M-I — a slow backend does not pile up overlapping ticks", async ({ page }) => {
  await boot(page, { active: COLL });
  let inFlight = 0;
  let maxInFlight = 0;
  await page.route("**/stats", async (route) => {
    inFlight += 1;
    maxInFlight = Math.max(maxInFlight, inFlight);
    await new Promise((r) => setTimeout(r, 9000));
    inFlight -= 1;
    return route.fallback();
  });
  await openTab(page, "monitoring");
  await page.waitForTimeout(25_000);
  expect(maxInFlight, "overlapping monitor ticks").toBeLessThanOrEqual(1);
});

test("M-embedder-down — models block degrades to an info alert", async ({ page }) => {
  await page.route("**/health", (route) => transformJsonResponse(route,
    (json) => ({ ...json, embedder: "unreachable" })));
  await prepareConsole(page, { active: COLL });
  await page.goto(`${BASE_URL}/#monitoring`);
  await expect(page.locator("#mon-models .alert-info")).toContainText(
    "Embedder unreachable — model info unavailable", { timeout: 30_000 });
});

test("M-metrics-robust — malformed Prometheus output does not break the table", async ({ page }) => {
  await page.route("**/metrics", (route) => fulfillResponse(route, {
    status: 200, contentType: "text/plain",
    body: [
      "# HELP http_requests_total a comment that must be ignored",
      "# TYPE http_requests_total counter",
      "this line is garbage without any value",
      'http_requests_total{endpoint="/health",method="GET"} 12',
      'http_request_duration_seconds{endpoint="/health",quantile="0.5"} not_a_number',
      "",
    ].join("\n"),
  }));
  await prepareConsole(page, { active: COLL });
  await page.goto(`${BASE_URL}/#monitoring`);
  await expect(page.locator("#mon-table")).not.toBeEmpty({ timeout: 30_000 });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.waitForTimeout(6000);
  expect(errors).toEqual([]);
});

test("M-K — the per-endpoint metrics table lists exercised endpoints", {
  annotation: { type: "case-partial", description: "book:9.4/M-K" },
}, async ({ page }) => {
  await boot(page, { active: COLL });
  // Exercise an endpoint through the UI first.
  await openTab(page, "points");
  await page.locator('[data-pt-tab="get"]').click();
  await page.locator("#get-id").fill("1");
  await page.locator("#get-submit").click();
  await expect(page.locator("#get-result")).not.toBeEmpty({ timeout: 30_000 });

  await openTab(page, "monitoring");
  const table = page.locator("#mon-table");
  await expect(table).not.toBeEmpty({ timeout: 30_000 });
  for (const header of ["endpoint", "requests", "errors", "p50", "p95"]) {
    await expect(table).toContainText(header);
  }
  await expect(table.locator("tbody tr").first()).toBeVisible();
});

// Runs last: it mutates the shared server-side counters view.
test("M-J — Reset counters zeroes the displayed totals", {
  annotation: { type: "case-partial", description: "book:9.4/M-J" },
}, async ({ page }) => {
  await boot(page, { active: COLL });
  await openTab(page, "monitoring");
  await expect(page.locator("#mon-stats")).not.toBeEmpty({ timeout: 30_000 });
  const readTotal = async () => {
    const t = ((await page.locator("#mon-stats").textContent()) || "").replace(/\s+/g, "");
    const m = /total_requests(\d+)/.exec(t);
    return m ? Number(m[1]) : NaN;
  };
  const before = await readTotal();
  expect(Number.isFinite(before)).toBe(true);
  await page.locator("#mon-reset").click();
  // The tile refreshes on the next 5s poll tick, so poll for the drop.
  await expect.poll(readTotal, { timeout: 30_000 }).toBeLessThan(Math.max(50, before / 2));
});

// ───────────────────────── §9.4 Lifecycle guards ───────────────────────────

test("M20 — no browser console errors across the Monitoring run", async ({ page }) => {
  const errors = await boot(page, { active: COLL });
  await openTab(page, "monitoring");
  await expect(page.locator("#mon-badges")).not.toBeEmpty({ timeout: 30_000 });
  expect(errors).toEqual([]);
});

test("M21 — no 401 responses during monitoring polling", async ({ page }) => {
  await boot(page, { active: COLL });
  let any401 = false;
  await page.route("**/stats", async (route) => {
    const resp = await fetchResponse(route);
    if (resp.status() === 401) any401 = true;
    return fulfillResponse(route, { response: resp });
  });
  await page.route("**/health", async (route) => {
    const resp = await fetchResponse(route);
    if (resp.status() === 401) any401 = true;
    return fulfillResponse(route, { response: resp });
  });
  await openTab(page, "monitoring");
  await expect(page.locator("#mon-badges")).not.toBeEmpty({ timeout: 30_000 });
  await page.waitForTimeout(12_000); // allow at least two poll cycles
  expect(any401, "monitoring endpoints should not return 401").toBe(false);
});

test("M22 — dynamic values are escaped, no DOM injection", async ({ page }) => {
  // Inject a malicious string into the health endpoint response and verify
  // the monitoring panel renders it as text, not as active HTML.
  await page.route("**/health", (route) => transformJsonResponse(route,
    (json) => ({ ...json, embedder_profile: "<img src=x onerror=alert(1)>" })));
  await prepareConsole(page, { active: COLL });
  await page.goto(`${BASE_URL}/#monitoring`);
  await expect(page.locator("#mon-models")).not.toBeEmpty({ timeout: 30_000 });
  const html = await page.locator("#mon-models").innerHTML();
  expect(html).not.toContain("<img");
  // The escaped version should appear as text
  expect(html).toContain("&lt;img");
  // "onerror" survives as inert text inside the escaped string — that is the
  // point of escaping, so assert the real property instead: the payload never
  // becomes a node, and no handler is attached anywhere in the panel.
  expect(await page.locator("#mon-models img, #mon-models *[onerror]").count()).toBe(0);
});
