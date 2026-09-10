import { fetchResponse } from "./telemetry.mjs";
import { fulfillResponse, abortResponse } from "./telemetry.mjs";
import { fixtureName } from "./harness.mjs";
import { expect, test } from "./fixtures.js";

import { BASE_URL, setupCall } from "./harness.mjs";
// Created for real by the "create-only" case below and dropped again in the same
// test; prefixed e2e_ like every other collection this suite touches.
const CREATED = fixtureName("e2e_compat_createonly");
const ACTIVE = fixtureName("e2e_a_keep");

const DRIFT = {
  status: "incompatible",
  stored: { profile: "ADVANCED", fingerprint: "sha256:stored" },
  active: { profile: "ADVANCED", fingerprint: "sha256:active" },
  active_vectors: { dense: "dense", sparse: "sparse" },
  message: "Active embedder and collection use different vector spaces.",
};

async function dropCollection(name) {
  await setupCall('DELETE', `/collections/${name}?confirm_delete=${name}`, undefined, { expected: [200, 404] });
}

function mountReportRoute(page, compat) {
  return page.route(/\/collections\/[^/?]+$/, async (route) => {
    if (route.request().method() !== "GET") {
      await route.fallback();
      return;
    }
    try {
      const response = await fetchResponse(route);
      const report = await response.json();
      report.embedding_compatibility = compat;
      await fulfillResponse(route, { response, json: report });
    } catch (error) {
      // The page or the whole context went away while this report request was
      // still in flight (typical at teardown): nobody is waiting for the answer,
      // and letting the rejection escape would fail the test that already passed.
      if (!page.isClosed()) throw error;
    }
  });
}

async function bootstrap(page) {
  const token = process.env.QHM_E2E_TOKEN || "";
  await page.addInitScript(({ apiToken, active }) => {
    localStorage.setItem("qhm_active_collection", active);
    if (apiToken) sessionStorage.setItem("qhm_api_key", apiToken);
  }, { apiToken: token, active: ACTIVE });
  // Force servicesReady=true deterministically: without this, a really-down
  // embedder would make panels inert via the OUTAGE path, so the collection-lock
  // assertions (esp. the active==null "not inert" case) would pass/fail for the
  // wrong reason. Mirrors service-readiness.e2e.spec.js.
  await page.route("**/health", async (route) => {
    await fulfillResponse(route, {
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "healthy",
        qdrant: "reachable",
        embedder: "reachable",
        collections: 1,
      }),
    });
  });
}

async function expectRecoverablePoints(page) {
  await page.locator('#tab-points').click();
  await expect(page.locator('#panel-points')).not.toHaveAttribute('inert', '');
  await page.locator('[data-pt-tab="upsert"]').click();
  await page.locator('#up-id').fill('1');
  await page.locator('#up-text').fill('blocked mutation');
  await expect(page.locator('#up-submit')).toBeDisabled();
  await page.locator('[data-pt-tab="get"]').click();
  await page.locator('#get-id').fill('1');
  await expect(page.locator('#get-submit')).toBeEnabled();
}

test("recoverable incompatible collection: mutations locked, reads and Re-embed stay live", async ({ page }) => {
  await bootstrap(page);
  // Same profile, different fingerprint (e.g. model/revision upgrade within
  // ADVANCED) — the general recoverable banner case, distinct from the
  // basic<->advanced profile-switch case which suppresses the banner below.
  await mountReportRoute(page, {
    status: "incompatible",
    stored: { profile: "ADVANCED", fingerprint: "sha256:stored" },
    active: { profile: "ADVANCED", fingerprint: "sha256:active" },
    active_vectors: { dense: "dense", sparse: "sparse" },
    message: "Active embedder and collection use different vector spaces.",
  });

  await page.goto(`${BASE_URL}/#query`);
  await expect(page.locator("#conn-collection")).not.toHaveValue("");
  // Selector stays usable so the user can switch away.
  await expect(page.locator("#conn-collection")).toBeEnabled();
  // Banner is the single source of the reason.
  await expect(page.locator("#service-gate")).toContainText("different vector space");
  await expect(page.locator("#service-gate")).toContainText("Re-embed its vectors");

  // Vector-space queries and mutations lock; point retrieval stays available.
  await expect(page.locator("#panel-query")).toHaveAttribute("inert", "");
  await expect(page.locator("#panel-query")).toHaveClass(/operations-disabled/);
  await expectRecoverablePoints(page);

  // Collections: Report live, 02 Re-embed live (recoverable), 03/04/05/06 locked.
  await page.locator("#tab-collections").click();
  await expect(page.locator("#cc-report")).toContainText("INCOMPATIBLE");
  await expect(page.locator("#cc-report")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cr-wrap")).not.toHaveAttribute("inert", "");
  // 03 is only PARTIALLY locked: creating a new collection stays available,
  // recreate does not.
  await expect(page.locator("#cc-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-recreate-control")).toHaveAttribute("inert", "");
  await expect(page.locator("#cu-wrap")).toHaveAttribute("inert", "");
  await expect(page.locator("#pi-wrap")).toHaveAttribute("inert", "");
  // 07 Delete addresses a typed name, so it is not locked by the active
  // collection being incompatible -- see the E/F cases below.
  await expect(page.locator("#cd-wrap")).not.toHaveAttribute("inert", "");
});

test("basic<->advanced profile switch: banner suppressed, lockdown and Re-embed unchanged", async ({ page }) => {
  await bootstrap(page);
  await mountReportRoute(page, {
    status: "incompatible",
    stored: { profile: "BASIC", fingerprint: "sha256:stored" },
    active: { profile: "ADVANCED", fingerprint: "sha256:active" },
    active_vectors: { dense: "dense", sparse: "sparse" },
    message: "Active embedder and collection use different vector spaces.",
  });

  await page.goto(`${BASE_URL}/#query`);
  // Banner suppressed for this specific mismatch — no text at all.
  await expect(page.locator("#service-gate")).toBeEmpty();
  // Query stays locked; read-only point retrieval is independent of vector space.
  await expect(page.locator("#panel-query")).toHaveAttribute("inert", "");
  await expectRecoverablePoints(page);

  // Collections: Re-embed (02) still live — recoverable, unaffected by the
  // suppressed banner.
  await page.locator("#tab-collections").click();
  await expect(page.locator("#cr-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-recreate-control")).toHaveAttribute("inert", "");
  // The suppressed banner must not cost the user the create form: a profile
  // switch is exactly when a new, compatible collection is needed.
  await expect(page.locator("#cc-name")).toBeEnabled();
  await expect(page.locator("#cc-submit")).toHaveText(/create collection/i);
});

test("foreign incompatible collection: Re-embed locked too, contract-only banner", async ({ page }) => {
  await bootstrap(page);
  await mountReportRoute(page, {
    status: "incompatible",
    stored: null,
    active: { profile: "ADVANCED", fingerprint: "sha256:active" },
    active_vectors: { dense: "dense", sparse: "sparse" },
    message: "Collection has no verified embedding contract; re-embed required.",
  });

  await page.goto(`${BASE_URL}/#collections`);
  await expect(page.locator("#service-gate")).toContainText("no verified embedding contract");
  // A foreign collection has no stored contract to regenerate from, so the banner
  // states only that — it must NOT offer the recoverable remedies (re-embedding it
  // or switching away), which belong to the same-profile drift case.
  await expect(page.locator("#service-gate")).not.toContainText("Re-embed its vectors");
  await expect(page.locator("#service-gate")).not.toContainText("compatible collection");
  await expect(page.locator("#service-gate")).not.toContainText("Delete");
  // Foreign: 02 Re-embed is locked as well.
  await expect(page.locator("#cr-wrap")).toHaveAttribute("inert", "");
  await expect(page.locator("#cd-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-report")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#conn-collection")).toBeEnabled();
});

test("incompatible with active==null does not lock (readiness owns it)", async ({ page }) => {
  await bootstrap(page);
  await mountReportRoute(page, {
    status: "incompatible",
    stored: { profile: "BASIC", fingerprint: "sha256:stored" },
    active: null,
    message: "Active Smart Embedder contract is unavailable; compatibility cannot be verified.",
  });

  await page.goto(`${BASE_URL}/#query`);
  // No compat lockdown: the panel is not inert from the collection path, and the
  // banner shows no incompatibility copy (readiness path handles embedder outage).
  await expect(page.locator("#panel-query")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#service-gate")).not.toContainText("Switch to a compatible collection");
  // active == null never locks, so 03 keeps its full recreate-capable shape.
  await page.locator("#tab-collections").click();
  await expect(page.locator("#cc-recreate-control")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-recreate")).toBeEnabled();
});

// ── Create-only mode on an embedder-mismatched collection ────────────────────
// A new collection is built from the ACTIVE embedding contract, so it cannot
// inherit the mismatch: creating one stays available while the destructive
// recreate — which would wipe the mismatched collection — does not.

test("A — create stays live on an incompatible collection, recreate does not", async ({ page }) => {
  await bootstrap(page);
  await mountReportRoute(page, DRIFT);

  await page.goto(`${BASE_URL}/#collections`);
  await expect(page.locator("#cc-report")).toContainText("INCOMPATIBLE");

  // The create half of 03 is fully interactive.
  await expect(page.locator("#cc-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-name")).toBeEnabled();
  for (const id of ["#cc-dense", "#cc-quant", "#cc-idf", "#cc-sparse-disk", "#cc-payload-disk"]) {
    await expect(page.locator(id)).toBeEnabled();
  }

  // The recreate half is locked, unchecked and announced as such.
  await expect(page.locator("#cc-recreate-control")).toHaveAttribute("inert", "");
  await expect(page.locator("#cc-recreate-control")).toHaveAttribute("aria-disabled", "true");
  await expect(page.locator("#cc-recreate")).toBeDisabled();
  await expect(page.locator("#cc-recreate")).toHaveAttribute("aria-disabled", "true");
  await expect(page.locator("#cc-recreate")).not.toBeChecked();
  await expect(page.locator("#cc-confirm")).toBeDisabled();
  await expect(page.locator("#cc-confirm-wrap")).toBeHidden();

  // Submit becomes available on a name alone, and says what it will do.
  await expect(page.locator("#cc-submit")).toBeDisabled();
  await page.locator("#cc-name").fill(CREATED);
  await expect(page.locator("#cc-submit")).toBeEnabled();
  await expect(page.locator("#cc-submit")).toHaveText("Create collection");

  // The banner still states why the rest is locked, and now says what is left.
  await expect(page.locator("#service-gate")).toContainText("different vector space");
  await expect(page.locator("#service-gate")).toContainText(/create a new collection/i);
});

test("B — the new collection is really created with recreate:false", async ({ page }) => {
  await bootstrap(page);
  await mountReportRoute(page, DRIFT);

  const bodies = [];
  await page.route("**/collections", async (route) => {
    if (route.request().method() === "POST") {
      bodies.push(route.request().postDataJSON());
    }
    await route.fallback();
  });

  try {
    await page.goto(`${BASE_URL}/#collections`);
    await expect(page.locator("#cc-report")).toContainText("INCOMPATIBLE");
    await page.locator("#cc-name").fill(CREATED);
    await page.locator("#cc-submit").click();

    await expect(page.locator("#cc-create-result")).toContainText(/created|already exists/i);
    expect(bodies).toHaveLength(1);
    expect(bodies[0].collection_name).toBe(CREATED);
    expect(bodies[0].recreate).toBe(false);
    expect(bodies[0]).not.toHaveProperty("confirm_recreate");

    // …and the console now knows about it.
    await expect(page.locator("#conn-collection")).toHaveValue(CREATED);
    await expect(page.locator(`#conn-collection option[value="${CREATED}"]`))
      .toHaveCount(1);
  } finally {
    await dropCollection(CREATED);
  }
});

test("C — recreate cannot be armed by click or keyboard while locked", async ({ page }) => {
  await bootstrap(page);
  await mountReportRoute(page, DRIFT);

  await page.goto(`${BASE_URL}/#collections`);
  await expect(page.locator("#cc-recreate")).toBeDisabled();

  // force:true bypasses Playwright's own actionability guard, so this asserts the
  // page's behavior instead of the test runner's.
  await page.locator("#cc-recreate").click({ force: true }).catch(() => {});
  await expect(page.locator("#cc-recreate")).not.toBeChecked();

  // inert takes the checkbox out of the focus order, so even a scripted focus()
  // cannot land on it — and Space therefore never reaches it.
  const focused = await page.evaluate(() => {
    const box = document.getElementById("cc-recreate");
    box.focus();
    return document.activeElement === box;
  });
  expect(focused).toBe(false);
  await page.keyboard.press("Space");
  await expect(page.locator("#cc-recreate")).not.toBeChecked();
  await expect(page.locator("#cc-confirm-wrap")).toBeHidden();
});

test("D — a compatible collection keeps the full create/recreate form", async ({ page }) => {
  await bootstrap(page);
  await mountReportRoute(page, {
    status: "compatible",
    stored: { profile: "ADVANCED", fingerprint: "sha256:same" },
    active: { profile: "ADVANCED", fingerprint: "sha256:same" },
    active_vectors: { dense: "dense", sparse: "sparse" },
    message: "",
  });

  await page.goto(`${BASE_URL}/#collections`);
  // The selector fills asynchronously; reading it earlier yields "".
  await expect(page.locator("#conn-collection")).not.toHaveValue("");
  await expect(page.locator("#service-gate")).toBeEmpty();
  await expect(page.locator("#cc-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-recreate-control")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cc-submit")).toHaveText("Submit create/recreate");

  const name = await page.locator("#conn-collection").inputValue();
  await page.locator("#cc-name").fill(name);
  await page.locator("#cc-recreate").check();
  await expect(page.locator("#cc-recreate")).toBeChecked();
  await expect(page.locator("#cc-confirm-wrap")).toBeVisible();
  // Type-to-confirm still gates the destructive path.
  await expect(page.locator("#cc-submit")).toBeDisabled();
  await page.locator("#cc-confirm").fill(name);
  await expect(page.locator("#cc-submit")).toBeEnabled();
});

test("E — 07 Delete stays live on an incompatible collection", async ({ page }) => {
  await bootstrap(page);
  await mountReportRoute(page, DRIFT);

  await page.goto(`${BASE_URL}/#collections`);
  await expect(page.locator("#cc-report")).toContainText("INCOMPATIBLE");

  // 07 addresses a name the user types, never state.active, so an incompatible
  // active collection tells it nothing -- the same reason the create half of 03
  // stays live. Locking it would strand the user with a collection they cannot
  // use and cannot remove.
  await expect(page.locator("#cd-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cd-wrap")).toHaveAttribute("aria-disabled", "false");
  await expect(page.locator("#cd-name")).toBeEnabled();
  await expect(page.locator("#cd-confirm")).toBeEnabled();

  // 04 and 05 still go: both operate on the active collection itself.
  await expect(page.locator("#cu-wrap")).toHaveAttribute("inert", "");
  await expect(page.locator("#pi-wrap")).toHaveAttribute("inert", "");
});

test("E2 — 07 Delete stays live for a foreign collection too", async ({ page }) => {
  await bootstrap(page);
  await mountReportRoute(page, {
    status: "incompatible",
    stored: null,
    active: { profile: "ADVANCED", fingerprint: "sha256:active" },
    active_vectors: { dense: "dense", sparse: "sparse" },
    message: "Collection has no verified embedding contract; re-embed required.",
  });

  await page.goto(`${BASE_URL}/#collections`);
  await expect(page.locator("#service-gate")).toContainText("no verified embedding contract");
  // Foreign locks 02 Re-embed -- there is no stored contract to regenerate
  // from -- but deleting by name is unaffected by that.
  await expect(page.locator("#cr-wrap")).toHaveAttribute("inert", "");
  await expect(page.locator("#cd-wrap")).not.toHaveAttribute("inert", "");
  await expect(page.locator("#cd-name")).toBeEnabled();
});

test("F — the destructive confirmation still gates 07 while incompatible", async ({ page }) => {
  await bootstrap(page);
  await mountReportRoute(page, DRIFT);

  await page.goto(`${BASE_URL}/#collections`);
  await expect(page.locator("#cd-wrap")).not.toHaveAttribute("inert", "");

  // Unlocking the section must not weaken the type-to-confirm gate.
  await expect(page.locator("#cd-submit")).toBeDisabled();
  await page.locator("#cd-name").fill(CREATED);
  await expect(page.locator("#cd-submit")).toBeDisabled();
  await page.locator("#cd-confirm").fill("not-the-name");
  await expect(page.locator("#cd-submit")).toBeDisabled();
  await page.locator("#cd-confirm").fill(CREATED);
  await expect(page.locator("#cd-submit")).toBeEnabled();
});
