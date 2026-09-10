import { fulfillResponse } from "./telemetry.mjs";
import { fixtureName } from "./harness.mjs";
// Test book §8.1 (Q1–Q16) + §11 spot checks (K1, K2, K6) — Tab QUERY.
// Sandbox: e2e_d_query only (read-only fixture, 12 points id 1..12).
import { test, expect } from "./fixtures.js";
import { boot, selectActive } from "./helpers.js";
import {
  D_QUERY, waitFieldsLoaded, requestJson, condition, clearFilters, runQuery,
  idsOnPage, collectAllIds, asSorted, alertTexts,
} from "./d-helpers.js";

const QUERY_TEXT = "hybrid vector search";

test.beforeEach(async ({ page }) => {
  await boot(page, { active: D_QUERY });
  await waitFieldsLoaded(page);
});

test("Q1 — rrf, rrf_k off, weights off, rerank on, normalize on", async ({ page }) => {
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await expect(page.locator("#ql-query-mode")).toHaveText(
    "HYBRID SEARCH IS ACTIVE - DENSE + SPARSE RETRIEVAL AND OPTIONAL RERANKING");
  await expect(page.locator("#ql-rerank")).toBeChecked();
  await expect(page.locator("#ql-normalize")).toBeChecked();

  const payload = await requestJson(page);
  expect(payload).toMatchObject({
    query: QUERY_TEXT,
    collection_name: D_QUERY,
    fusion: "rrf",
    rerank: true,
    rerank_normalize: true,
    limit: 10,
    candidate_limit: 50,
  });
  expect(payload).not.toHaveProperty("rrf_k");
  expect(payload).not.toHaveProperty("rrf_weights");
  expect(payload).not.toHaveProperty("rerank_top_k");
  expect(payload.with_payload).toContain("text");

  await runQuery(page);
  await expect(page.locator("#ql-results .alert-success")).toContainText(/Query completed/i);
  const ids = await idsOnPage(page);
  expect(ids.length).toBe(10); // limit 10 out of 12 fixture points
  expect(new Set(ids).size).toBe(10);
  await expect(page.locator("#ql-results .metric").first()).toContainText("fusion");
  await expect(page.locator("#ql-results")).toContainText("rrf");
});

test("Q2 — rrf + Set rrf_k puts rrf_k in the payload and enforces 1..10000", async ({ page }) => {
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await page.locator("#ql-rrfk-en").check();
  await expect(page.locator("#ql-rrfk")).toBeEnabled();
  await page.locator("#ql-rrfk").fill("60");
  expect(await requestJson(page)).toMatchObject({ fusion: "rrf", rrf_k: 60 });

  await page.locator("#ql-rrfk").fill("0");
  expect(await alertTexts(page)).toEqual(
    expect.arrayContaining([expect.stringMatching(/rrf_k must be an integer between 1 and 10000/i)]));
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-rrfk").fill("10001");
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-rrfk").fill("10000");
  expect(await requestJson(page)).toMatchObject({ rrf_k: 10000 });
  await expect(page.locator("#ql-run")).toBeEnabled();
  await page.locator("#ql-rrfk").fill("1");
  expect(await requestJson(page)).toMatchObject({ rrf_k: 1 });

  await page.locator("#ql-rrfk").fill("60");
  await runQuery(page);
  await expect(page.locator("#ql-results .alert-success")).toBeVisible();
});

test("Q3 — rrf_weights slider + weight scale update the caption live and reach the payload", async ({ page }) => {
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await page.locator("#ql-weights-en").check();
  await expect(page.locator("#ql-balance")).toBeEnabled();
  await expect(page.locator("#ql-balance-caption")).toHaveText("Qdrant weights [1.00, 1.00]");
  expect(await requestJson(page)).toMatchObject({ rrf_weights: [1, 1] });

  await page.locator("#ql-balance").fill("0.75");
  await expect(page.locator("#ql-balance-caption")).toHaveText("Qdrant weights [1.50, 0.50]");
  expect(await requestJson(page)).toMatchObject({ rrf_weights: [1.5, 0.5] });

  await page.locator("#ql-weight-scale").fill("2");
  await expect(page.locator("#ql-balance-caption")).toHaveText("Qdrant weights [3.00, 1.00]");
  expect(await requestJson(page)).toMatchObject({ rrf_weights: [3, 1] });

  // Out-of-range scale: caption switches to the bound message and Run is blocked.
  await page.locator("#ql-weight-scale").fill("0");
  await expect(page.locator("#ql-balance-caption")).toHaveText(/Weight scale must be/i);
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-weight-scale").fill("1");
  await page.locator("#ql-balance").fill("0.25");
  await expect(page.locator("#ql-balance-caption")).toHaveText("Qdrant weights [0.50, 1.50]");
  await runQuery(page);
  await expect(page.locator("#ql-results .alert-success")).toBeVisible();
});

test("Q4 — fusion dbsf hides/ignores rrf_k and rrf_weights", async ({ page }) => {
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await page.locator("#ql-rrfk-en").check();
  await page.locator("#ql-rrfk").fill("77");
  await page.locator('input[name="ql-fusion"][value="dbsf"]').check();

  await expect(page.locator("#ql-rrfk-field")).toBeHidden();
  await expect(page.locator("#ql-weights-field")).toBeHidden();
  const payload = await requestJson(page);
  expect(payload).toMatchObject({ fusion: "dbsf" });
  expect(payload).not.toHaveProperty("rrf_k");
  expect(payload).not.toHaveProperty("rrf_weights");

  await runQuery(page);
  await expect(page.locator("#ql-results")).toContainText("dbsf");
});

test("Q5 — dbsf with candidate_limit=1 warns about ties", async ({ page }) => {
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await page.locator('input[name="ql-fusion"][value="dbsf"]').check();
  await page.locator("#ql-limit").fill("1");
  await page.locator("#ql-cand").fill("1");

  const texts = await alertTexts(page);
  expect(texts.join(" | ")).toMatch(
    /With one candidate, each DBSF branch normalizes it to 0\.5, making fusion tie-prone; increase candidate_limit/i);
  // It is an info hint, not a validation error: the query must stay runnable.
  await expect(page.locator("#ql-errors .alert-info").first()).toBeVisible();
  await expect(page.locator("#ql-errors .alert-error")).toHaveCount(0);
  await expect(page.locator("#ql-run")).toBeEnabled();

  // The warning must be specific to dbsf + candidate_limit 1.
  await page.locator("#ql-cand").fill("2");
  expect((await alertTexts(page)).join(" | ")).not.toMatch(/tie-prone/i);
  await page.locator("#ql-cand").fill("1");
  await page.locator('input[name="ql-fusion"][value="rrf"]').check();
  expect((await alertTexts(page)).join(" | ")).not.toMatch(/tie-prone/i);
});

test("Q6 — rerank OFF drops rerank_top_k", async ({ page }) => {
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await page.locator("#ql-topk-en").check();
  await page.locator("#ql-topk").fill("7");
  expect(await requestJson(page)).toMatchObject({ rerank: true, rerank_top_k: 7 });

  await page.locator("#ql-rerank").uncheck();
  await expect(page.locator("#ql-topk-en")).toBeDisabled();
  await expect(page.locator("#ql-normalize")).toBeDisabled();
  const payload = await requestJson(page);
  expect(payload.rerank).toBe(false);
  expect(payload).not.toHaveProperty("rerank_top_k");

  await runQuery(page);
  await expect(page.locator("#ql-results")).toContainText("reranked");
});

test("Q7 — rerank ON + Set rerank_top_k (1..200, <= candidate_limit)", async ({ page }) => {
  await page.locator("#ql-query").fill(QUERY_TEXT);
  const max = await page.locator("#ql-topk").getAttribute("max");
  expect(max).toBe("200");
  await page.locator("#ql-topk-en").check();
  await page.locator("#ql-topk").fill("5");
  expect(await requestJson(page)).toMatchObject({ rerank_top_k: 5 });

  await page.locator("#ql-topk").fill("0");
  expect((await alertTexts(page)).join(" | ")).toMatch(/rerank_top_k must be an integer between 1 and 200/i);
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-topk").fill("201");
  await expect(page.locator("#ql-run")).toBeDisabled();

  // 200 is in range but exceeds the default candidate_limit of 50.
  await page.locator("#ql-topk").fill("200");
  expect((await alertTexts(page)).join(" | ")).toMatch(/rerank_top_k cannot exceed candidate_limit/i);
  await page.locator("#ql-cand").fill("200");
  await expect(page.locator("#ql-run")).toBeEnabled();
  expect(await requestJson(page)).toMatchObject({ rerank_top_k: 200, candidate_limit: 200 });

  await page.locator("#ql-topk").fill("5");
  await page.locator("#ql-cand").fill("50");
  await runQuery(page);
  await expect(page.locator("#ql-results .alert-success")).toBeVisible();
});

test("Q8 — rerank ON + normalize OFF sends rerank_normalize=false", async ({ page }) => {
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await page.locator("#ql-normalize").uncheck();
  expect(await requestJson(page)).toMatchObject({ rerank: true, rerank_normalize: false });
  await runQuery(page);
  await expect(page.locator("#ql-results .alert-success")).toBeVisible();
});

test("Q9 — limit / candidate_limit bounds block Run when invalid", async ({ page }) => {
  await page.locator("#ql-query").fill(QUERY_TEXT);
  // Both ceilings are operator-configurable (QUERY_MAX_LIMIT /
  // QUERY_MAX_CANDIDATE_LIMIT) and the console mirrors whatever the server
  // reports, so read them back instead of hardcoding the shipped defaults.
  const cfg = await page.evaluate(() => ({
    limit: window.QHM_CONFIG.queryMaxLimit,
    cand: window.QHM_CONFIG.queryMaxCandidateLimit,
  }));
  expect(await page.locator("#ql-limit").getAttribute("max")).toBe(String(cfg.limit));
  expect(await page.locator("#ql-cand").getAttribute("max")).toBe(String(cfg.cand));

  await page.locator("#ql-limit").fill("1");
  expect(await requestJson(page)).toMatchObject({ limit: 1 });
  await expect(page.locator("#ql-run")).toBeEnabled();

  await page.locator("#ql-limit").fill(String(cfg.limit));
  await expect(page.locator("#ql-run")).toBeEnabled();

  await page.locator("#ql-limit").fill(String(cfg.limit + 1));
  expect((await alertTexts(page)).join(" | "))
    .toMatch(new RegExp(`limit must be an integer between 1 and ${cfg.limit}`, "i"));
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-limit").fill("0");
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-limit").fill("10");
  await page.locator("#ql-cand").fill("0");
  expect((await alertTexts(page)).join(" | "))
    .toMatch(new RegExp(`candidate_limit must be an integer between 1 and ${cfg.cand}`, "i"));
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-cand").fill(String(cfg.cand + 1));
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-cand").fill("5"); // below limit 10
  expect((await alertTexts(page)).join(" | ")).toMatch(/candidate_limit must be greater than or equal to limit/i);
  await expect(page.locator("#ql-run")).toBeDisabled();

  await page.locator("#ql-cand").fill(String(cfg.cand));
  await expect(page.locator("#ql-run")).toBeEnabled();
});

test("Q10 — empty query + filters switches to payload-only filtering", async ({ page }) => {
  await expect(page.locator("#ql-query")).toHaveValue("");
  await expect(page.locator("#ql-query-mode")).toHaveText(
    "HYBRID SEARCH IS DISABLED - PAYLOAD-ONLY MODE FILTERING");
  await condition(page, { field: "f_views", op: "gte", value: "10" });

  const payload = await requestJson(page);
  expect(payload).toMatchObject({ collection_name: D_QUERY });
  expect(payload).not.toHaveProperty("query");
  expect(payload).not.toHaveProperty("fusion");
  expect(payload.filters).toEqual({ must: [{ key: "f_views", range: { gte: 10 } }] });
  // The "no filters" hint must be gone once a filter exists.
  expect((await alertTexts(page)).join(" | ")).not.toMatch(/ordered by ID/i);

  await expect(page.locator("#ql-run")).toHaveText("Retrieve points");
  await runQuery(page);
  await expect(page.locator("#ql-results .alert-success")).toContainText(/Retrieval completed/i);
  expect(asSorted(await collectAllIds(page))).toEqual([10, 11, 12]);
});

test("Q11 — empty query and no filters reports the ordered-by-ID first page", async ({ page }) => {
  await expect(page.locator("#ql-query")).toHaveValue("");
  await expect(page.locator("#ql-filter-rows .filter-row")).toHaveCount(0);
  expect((await alertTexts(page)).join(" | ")).toMatch(
    /Empty query and no filters: the first page of all points will be returned, ordered by ID/i);
  await expect(page.locator("#ql-errors .alert-info").first()).toBeVisible();
  await expect(page.locator("#ql-run")).toBeEnabled();

  await runQuery(page);
  expect(asSorted(await idsOnPage(page))).toEqual([1, 2, 3, 4, 5]); // PAYLOAD_PAGE_SIZE
  expect(asSorted(await collectAllIds(page))).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

  // Adding a filter must retract the hint.
  await condition(page, { field: "f_status", op: "eq", value: "s1" });
  expect((await alertTexts(page)).join(" | ")).not.toMatch(/ordered by ID/i);
});

test("Q12 — with_payload selection reaches the payload and the rendered results", async ({ page }) => {
  const boxes = page.locator("#ql-payload-fields input");
  await expect(boxes.first()).toBeVisible();
  const values = await boxes.evaluateAll((els) => els.map((e) => e.value));
  expect(values).toEqual(expect.arrayContaining(["f_status", "f_views", "text"]));

  // Payload-only mode: nothing is locked, so deselect everything but f_status.
  for (const value of values) {
    if (value !== "f_status") await page.locator(`#ql-payload-fields input[value="${value}"]`).uncheck();
  }
  expect(await requestJson(page)).toMatchObject({ with_payload: ["f_status"] });

  await runQuery(page);
  const first = await page.locator("#ql-results .payload-details code").first().textContent();
  expect(Object.keys(JSON.parse(first))).toEqual(["f_status"]);

  // Re-select one field and confirm the payload follows.
  await page.locator('#ql-payload-fields input[value="f_views"]').check();
  const payload = await requestJson(page);
  expect(payload.with_payload).toEqual(expect.arrayContaining(["f_status", "f_views"]));
  expect(payload.with_payload).not.toContain("text");

  // Hybrid + rerank must force "text" back in (the reranker reads it).
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await expect(page.locator('#ql-payload-fields input[value="text"]')).toBeDisabled();
  expect((await requestJson(page)).with_payload).toContain("text");
});

test("Q13 — Download results as JSON saves a file in addition to rendering", async ({ page }) => {
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await page.locator("#ql-download").check();
  const downloadPromise = page.waitForEvent("download", { timeout: 60_000 });
  await runQuery(page);
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(new RegExp(`^${fixtureName("e2e_d_query")}_query_results_.*\\.json$`));
  await expect(page.locator("#ql-results .result-card").first()).toBeVisible();
});

test("Q14 — server error renders readably without crashing the panel", async ({ page }) => {
  // The 500 below is injected by this test; the browser logs it as a network
  // error, which is not an application K6 failure.
  await page.route("**/query/search", (route) =>
    fulfillResponse(route, { status: 500, contentType: "application/json",
      body: JSON.stringify({ detail: "forced e2e failure" }) }));
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await runQuery(page);
  await expect(page.locator("#ql-results .alert-error")).toBeVisible();
  await expect(page.locator("#ql-results")).toContainText(/forced e2e failure/i);
  await expect(page.locator("#panel-query")).toBeVisible();

  // Recovery: unroute and re-run, the panel must still work.
  await page.unroute("**/query/search");
  await runQuery(page);
  await expect(page.locator("#ql-results .alert-success")).toBeVisible();
});

test("Q15 — ql-request-json preview tracks every control live, before Run", async ({ page }) => {
  // Payload-only baseline.
  expect(await requestJson(page)).toMatchObject({ collection_name: D_QUERY, limit: 5 });

  await page.locator("#ql-query").fill("alpha");
  expect(await requestJson(page)).toMatchObject({ query: "alpha", fusion: "rrf", limit: 10 });

  await page.locator("#ql-limit").fill("3");
  await page.locator("#ql-cand").fill("30");
  expect(await requestJson(page)).toMatchObject({ limit: 3, candidate_limit: 30 });

  await page.locator('input[name="ql-fusion"][value="dbsf"]').check();
  expect(await requestJson(page)).toMatchObject({ fusion: "dbsf" });
  await page.locator('input[name="ql-fusion"][value="rrf"]').check();

  await page.locator("#ql-rrfk-en").check();
  await page.locator("#ql-rrfk").fill("40");
  expect(await requestJson(page)).toMatchObject({ rrf_k: 40 });

  await page.locator("#ql-weights-en").check();
  await page.locator("#ql-balance").fill("0.6");
  expect((await requestJson(page)).rrf_weights).toEqual([1.2, 0.8]);

  await page.locator("#ql-normalize").uncheck();
  expect(await requestJson(page)).toMatchObject({ rerank_normalize: false });

  await condition(page, { field: "f_status", op: "eq", value: "s4" });
  expect((await requestJson(page)).filters).toEqual({ must: [{ key: "f_status", match: { value: "s4" } }] });

  // Deselecting a payload field is reflected immediately too.
  await page.locator('#ql-payload-fields input[value="f_geo"]').uncheck();
  expect((await requestJson(page)).with_payload).not.toContain("f_geo");

  // Nothing was run: results stay empty.
  await expect(page.locator("#ql-results")).toBeEmpty();
});

test("Q16 — switching active collection after a Run resets results, filters and payload keys", async ({ page }) => {
  await condition(page, { field: "f_status", op: "eq", value: "s2" });
  await runQuery(page);
  expect(await idsOnPage(page)).toEqual(["2"]);

  await selectActive(page, fixtureName("e2e_d_nested"));
  await expect(page.locator("#ql-results")).toBeEmpty();
  await expect(page.locator("#ql-filter-rows .filter-row")).toHaveCount(0);
  await waitFieldsLoaded(page);
  const values = await page.locator("#ql-payload-fields input").evaluateAll(
    (els) => els.map((e) => e.value));
  expect(values).toContain("label");
  expect(values).not.toContain("f_status");
  expect(await requestJson(page)).toMatchObject({ collection_name: fixtureName("e2e_d_nested") });
});

test("K1 — double-click on Run fires a single request", async ({ page }) => {
  let calls = 0;
  await page.route("**/query/search", async (route) => {
    calls += 1;
    await new Promise((r) => setTimeout(r, 1500));
    await route.fallback();
  });
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await expect(page.locator("#ql-run")).toBeEnabled();
  await page.locator("#ql-run").click({ clickCount: 2, delay: 30 });
  await expect(page.locator("#ql-run")).toHaveText("Running…");
  await expect(page.locator("#ql-run")).toBeDisabled();
  await expect(page.locator("#ql-results .alert-success")).toBeVisible({ timeout: 60_000 });
  expect(calls).toBe(1);
});

test("K2 — a response for the previous collection is not rendered after a switch", async ({ page }) => {
  await page.route("**/query/search", async (route) => {
    await new Promise((r) => setTimeout(r, 3000));
    await route.fallback();
  });
  await page.locator("#ql-query").fill(QUERY_TEXT);
  await page.locator("#ql-run").click();
  await expect(page.locator("#ql-run")).toHaveText("Running…");
  await selectActive(page, fixtureName("e2e_d_nested"));
  await page.waitForTimeout(5000);
  await expect(page.locator("#ql-results")).toBeEmpty();
  await expect(page.locator("#ql-collection")).toHaveValue(fixtureName("e2e_d_nested"));
});
