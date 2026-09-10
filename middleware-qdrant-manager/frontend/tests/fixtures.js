import {
  expect,
  request,
  test as base,
} from "@playwright/test";
import { BASE_URL, guardMutation } from "./harness.mjs";
import { startTelemetry } from "./telemetry.mjs";

const rawPaceMs = process.env.QHM_E2E_PACE_MS ?? "250";
const paceMs = Number(rawPaceMs);

if (!Number.isFinite(paceMs) || paceMs < 0) {
  throw new Error("QHM_E2E_PACE_MS must be a non-negative number.");
}

export const test = base.extend({
  campaignTelemetry: [async ({ page }, use, testInfo) => {
    const telemetry = startTelemetry(page);
    try { await use(); }
    finally {
      telemetry.dispose();
      const unexpected = telemetry.unexpected();
      await testInfo.attach('campaign-telemetry', { contentType: 'application/json',
        body: Buffer.from(JSON.stringify({ events: telemetry.events, expected: telemetry.expected,
          expectedNetwork: telemetry.expectedNetwork, unexpected })) });
      expect(unexpected, 'Unexpected browser or HTTP errors (K6)').toEqual([]);
    }
  }, { auto: true }],
  mutationOwnership: [async ({ context }, use) => {
    const violations = [];
    const guard = async (route) => {
      const request = route.request();
      try {
        let body;
        if (request.postData()) body = request.postDataJSON();
        guardMutation(request.method(), request.url(), body);
      } catch (error) {
        violations.push(error.message);
        await route.abort("blockedbyclient");
        return;
      }
      await route.continue();
    };
    await context.route(`${BASE_URL}/**`, guard);
    try { await use(); }
    finally {
      await context.unroute(`${BASE_URL}/**`, guard);
      expect(violations, "Requests rejected by the campaign ownership guard").toEqual([]);
    }
  }, { auto: true }],
  rateLimitPacing: [async ({}, use) => {
    await use();
    if (paceMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, paceMs));
    }
  }, { auto: true }],
});

export { expect, request };
