// E2E runner for the console test book (tests/testbook/test-book-e2e.md).
//
// These specs drive the REAL console against a running stack — nothing is
// stubbed except the few cases the book marks as network-simulated. They create
// and delete collections prefixed `e2e_`; see tests/README.md before running.
import { defineConfig } from "@playwright/test";
import { fileURLToPath } from "node:url";
import { BASE_URL } from "./tests/harness.mjs";

export default defineConfig({
  testDir: fileURLToPath(new URL("./tests", import.meta.url)),
  // Only .spec.js: query-models.test.mjs is a plain node:test module, not a
  // Playwright spec, and must not be collected here.
  testMatch: /.*\.spec\.js$/,
  // Excluded from a default run because they need state the runner cannot create:
  // a real SMART_EMBEDDER_PROFILE switch with a stack restart (z4, z6) or the
  // embedder container stopped (z5). Use --config playwright.external.config.js
  // and the prepared scenario's path; a path alone does not bypass testIgnore.
  // See tests/README.md for the external prerequisites.
  testIgnore: [
    "**/z4-switch.spec.js",
    "**/z5-real-outage.spec.js",
    "**/z6-e7.spec.js",
  ],
  timeout: 120_000,
  expect: { timeout: 15_000 },
  // Serial on purpose: the suite shares one Qdrant instance and one embedder, and
  // several cases assert on collection-scoped state that parallel workers would
  // trample.
  fullyParallel: false,
  workers: 1,
  retries: 0,
  outputDir: fileURLToPath(new URL("./test-results/artifacts", import.meta.url)),
  reporter: [["list"], [fileURLToPath(new URL("./tests/campaign-reporter.mjs", import.meta.url))]],
  use: {
    baseURL: BASE_URL,
    headless: true,
    actionTimeout: 20_000,
    navigationTimeout: 30_000,
    trace: "off",
    screenshot: "only-on-failure",
  },
});
