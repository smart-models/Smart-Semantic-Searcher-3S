// Shared helpers for the Smart Semantic Searcher console test book run.
import { expect } from "@playwright/test";
import { BASE_URL, TOKEN, guardMutation, requireStatus } from "./harness.mjs";
import { startTelemetry } from "./telemetry.mjs";

export { BASE_URL, TOKEN };

// Console errors are an exit criterion (K6): collect them per page.
export function watchConsole(page) {
  // The automatic fixture owns and disposes the sole page telemetry collector.
  return startTelemetry(page).messages;
}

// Boot the console with the API key already registered and a chosen active
// collection, so a test starts from a deterministic state.
export async function prepareConsole(page, { active = null, token = TOKEN } = {}) {
  await page.addInitScript(
    ({ apiToken, activeName }) => {
      if (apiToken) sessionStorage.setItem("qhm_api_key", apiToken);
      else sessionStorage.removeItem("qhm_api_key");
      if (activeName) localStorage.setItem("qhm_active_collection", activeName);
      else localStorage.removeItem("qhm_active_collection");
    },
    { apiToken: token, activeName: active },
  );
}

export async function boot(page, options = {}) {
  await prepareConsole(page, options);
  const errors = watchConsole(page);
  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY", { timeout: 30_000 });
  return errors;
}

export async function selectActive(page, name) {
  await page.locator("#conn-collection").selectOption(name);
  await expect(page.locator("#ql-collection")).toHaveValue(name);
}

export async function openTab(page, tab) {
  await page.locator(`#tab-${tab}`).click();
  await expect(page.locator(`#panel-${tab}`)).toBeVisible();
}

export async function openPointsSubtab(page, name) {
  await page.locator(`[data-pt-tab="${name}"]`).click();
  await expect(page.locator(`#pt-${name}`)).toBeVisible();
}

// Direct API call helper for [API] setup/teardown steps (declared in the book).
export async function apiFetch(request, method, path, body) {
  guardMutation(method, path, body);
  return request.fetch(`${BASE_URL}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}),
    },
    ...(body === undefined ? {} : { data: body }),
  });
}

export async function setupFetch(request, method, path, body, expected = [200]) {
  const response = await apiFetch(request, method, path, body);
  requireStatus(response, method, path, expected);
  return response;
}
