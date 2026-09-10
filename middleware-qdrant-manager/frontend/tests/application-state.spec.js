import { fulfillResponse } from "./telemetry.mjs";
import { test, expect } from "./fixtures.js";
import { A, B, openConsole, openPoints } from "./application-fixtures.js";
import { operations, openOperation, matches, success } from "./application-fixtures.js";

test("wrong optional report types produce a protocol error without a render crash", async ({ page }) => {
  const op = operations.find((op) => op.name === "create");
  const mock = await openOperation(page, op);
  let field;
  mock.intercept = async (route, request) => {
    if (!matches(op, request)) return false;
    const body = success(op, request, mock);
    body.report[field] = "not an array";
    await fulfillResponse(route, { json: body }); return true;
  };
  for (field of ["findings", "errors", "suggested_fixes"]) {
    await page.locator("#cc-submit").click();
    await expect(page.locator("#cc-create-result")).toContainText("unexpected response");
    await expect(page.locator("#cc-name")).toHaveValue(B);
    expect(mock.errors).toEqual([]);
  }
});

test("changing the API token invalidates a pending point read", async ({ page }) => {
  const mock = await openConsole(page);
  await openPoints(page, "get");
  await page.locator("#get-id").fill("7");
  const late = mock.deferReply("GET", "/points/7");
  try {
    await page.locator("#get-submit").click(); await late.started;
    await page.locator("#apikey-btn").click();
    await page.locator("#apikey-input").fill("synthetic-test-token");
    await page.locator("#apikey-save").click();
    const received = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/7");
    late.resolve({ collection_name: A, id: "7", payload: { text: "obsolete token result" } });
    await received;
    await expect(page.locator("#get-result")).not.toContainText("obsolete token result");
    expect(mock.errors).toEqual([]);
  } finally { late.resolve({}); }
});

for (const change of ["replacement file", "token"]) {
  test(`a late file read cannot replace state after ${change}`, async ({ page }) => {
    const mock = await openConsole(page);
    await openPoints(page, "ingest");
    await page.evaluate(() => {
      const read = File.prototype.text;
      File.prototype.text = function () {
        if (this.name === "old.json") return new Promise((resolve) => { window.releaseOldFile = resolve; });
        return read.call(this);
      };
    });
    await page.locator("#ing-file").setInputFiles({ name: "old.json", mimeType: "application/json", buffer: Buffer.from("{}") });
    await expect.poll(() => page.evaluate(() => typeof window.releaseOldFile)).toBe("function");
    if (change === "replacement file") {
      await page.locator("#ing-file").setInputFiles({ name: "new.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ chunks: [{ id: 1, text: "new file" }] })) });
      await expect(page.locator("#ing-preflight-result")).toContainText("new.json");
    } else {
      await page.locator("#apikey-btn").click();
      await page.locator("#apikey-input").fill("synthetic-test-token");
      await page.locator("#apikey-save").click();
    }
    const before = await page.locator("#ing-preflight-result").textContent();
    await page.evaluate(() => window.releaseOldFile(JSON.stringify({ chunks: [{ id: 7, text: "obsolete file" }] })));
    await expect(page.locator("#ing-preflight-result")).toHaveText(before);
    expect(mock.requestsFor("POST", "/points/ingest-chunks")).toHaveLength(0);
    expect(mock.errors).toEqual([]);
  });
}

test("editing a download query stops pagination after an obsolete page", async ({ page }) => {
  const mock = await openConsole(page), downloads = [];
  page.on("download", (download) => downloads.push(download));
  await page.locator("#ql-download").check();
  const first = mock.deferReply("POST", "/points/scroll");
  await page.locator("#ql-run").click(); await first.started;
  const second = mock.deferReply("POST", "/points/scroll");
  const reply = (id) => ({ collection_name: A, returned: 1, total: 3, results: [{ id, payload: { text: "page" } }], next_page_offset: String(Number(id) + 1) });
  try {
    first.resolve(reply("1")); await second.started;
    await page.locator("#ql-query").fill("changed query");
    const received = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/scroll");
    second.resolve(reply("2")); await received;
    await expect(page.locator("#ql-run")).toBeEnabled();
    expect(mock.requestsFor("POST", "/points/scroll")).toHaveLength(2);
    expect(downloads).toEqual([]);
    expect(mock.errors).toEqual([]);
  } finally { first.resolve({}); second.resolve({}); }
});

for (const op of operations) {
  test(`${op.name}: obsolete replies cannot commit after leaving and returning`, async ({ page }) => {
    const mock = await openOperation(page, op);
    const pathname = op.path || `/collections/${A}${op.suffix}`;
    const late = mock.deferReply(op.method, pathname);
    try {
      await page.locator(`#${op.button}`).click();
      const request = await late.started;
      if (op.typed) {
        const input = page.locator(op.name === "collection delete" ? "#cd-name" : "#cc-name");
        await input.fill("e2e_application_corrected");
        await input.fill(B);
      } else {
        await page.locator("#conn-collection").selectOption(B);
        await page.locator("#conn-collection").selectOption(A);
      }
      const resultId = op.name === "filtered-delete count" ? "delf-result" : op.result;
      const before = await page.locator(`#${resultId}`).textContent();
      const received = page.waitForResponse((r) => r.url().split("?")[0].endsWith(pathname) && r.request().method() === op.method);
      late.resolve(success(op, request, mock));
      await received;
      await expect(page.locator(`#${op.button}`)).not.toHaveAttribute("data-busy", "true");
      await expect(page.locator(`#${resultId}`)).toHaveText(before);
      expect(mock.requestsFor(op.method, pathname)).toHaveLength(1);
      expect(mock.errors).toEqual([]);
    } finally { late.resolve({}); }
  });

  test(`${op.name}: malformed HTTP 200 preserves the form and valid responses work`, async ({ page }) => {
    const mock = await openOperation(page, op);
    let body;
    mock.intercept = async (route, request) => {
      if (!matches(op, request)) return false;
      await fulfillResponse(route, { json: body === "valid" ? success(op, request, mock) : body });
      return true;
    };
    const result = page.locator(`#${op.name === "filtered-delete count" ? "delf-result" : op.result}`);
    for (const invalid of [null, 7, [], {}, { ok: true }]) {
      body = invalid;
      await page.locator(`#${op.button}`).click();
      await expect(result).toContainText("unexpected response");
      if (op.confirm) await expect(page.locator(`#${op.confirm}`)).toHaveValue(op.typed ? B : A);
    }
    body = "valid";
    await page.locator(`#${op.button}`).click();
    await expect(result).not.toContainText("unexpected response");
    expect(mock.errors).toEqual([]);
    expect(mock.unknown).toEqual([]);
  });
}

test("unsafe batch JSON ID is refused before HTTP and remains editable", async ({ page }) => {
  const mock = await openConsole(page);
  await openPoints(page, "batch");
  const input = '[{"point_id":18446744073709551615,"text":"keep"}]';
  await page.locator("#bt-json").fill(input);
  await page.locator("#bt-submit").click();
  await expect(page.locator("#bt-result")).toContainText("quoted ID string");
  await expect(page.locator("#bt-json")).toHaveValue(input);
  expect(mock.requestsFor("POST", "/points/batch")).toHaveLength(0);
});

test("bulk deletion reports the server count and disarms a successful confirmation", async ({ page }) => {
  const mock = await openConsole(page);
  await openPoints(page, "delete");
  mock.intercept = async (route, request) => {
    if (request.method !== "DELETE" || request.path !== "/points") return false;
    await fulfillResponse(route, { json: { ok: true, collection_name: A, deleted: 1 } }); return true;
  };
  await page.locator("#del-ids").fill("101,102");
  await page.locator("#del-confirm").fill(A);
  await page.locator("#del-bulk-submit").click();
  await expect(page.locator("#del-bulk-result")).toContainText("1 point(s) deleted");
  await expect(page.locator("#del-confirm")).toHaveValue("");
  expect(mock.requestsFor("DELETE", "/points")[0].body.ids).toHaveLength(2);
});

for (const change of ["collection ABA", "edit while pending"]) {
  test(`late delete preserves correction state after ${change}`, async ({ page }) => {
    const mock = await openConsole(page);
    await openPoints(page, "delete");
    await page.locator("#del-id").fill("7");
    const late = mock.deferReply("DELETE", "/points/7");
    try {
      await page.locator("#del-submit").click();
      await late.started;
      if (change === "collection ABA") {
        await page.locator("#conn-collection").selectOption(B);
        await page.locator("#conn-collection").selectOption(A);
      }
      await page.locator("#del-id").fill("8");
      const received = page.waitForResponse((r) => r.request().method() === "DELETE");
      late.resolve({ deleted: true, id: "7", collection_name: A });
      await received;
      await expect(page.locator("#del-submit")).toBeEnabled();
      await expect(page.locator("#del-id")).toHaveValue("8");
      await expect(page.locator("#del-result")).toBeEmpty();
    } finally { late.resolve({ deleted: true, collection_name: A, id: "7" }); }
  });
}

test("an obsolete export cannot initiate a download", async ({ page }) => {
  const mock = await openConsole(page, { tab: "collections" });
  const late = mock.deferReply("POST", "/points/export"), downloads = [];
  page.on("download", (download) => downloads.push(download));
  try {
    await page.locator("#ce-export").click();
    await late.started;
    await page.locator("#conn-collection").selectOption(B);
    await page.locator("#conn-collection").selectOption(A);
    const received = page.waitForResponse((r) => r.url().endsWith("/points/export"));
    late.resolve({ chunks: [{ point_id: "7", text: "old" }], metadata: { n_chunks: 1, collection_name: A } });
    await received;
    await expect(page.locator("#ce-export")).toBeEnabled();
    expect(downloads).toEqual([]);
    await expect(page.locator("#ce-result")).not.toContainText("Exported");
  } finally { late.resolve({}); }
});

for (const first of ["probe", "monitor"]) {
  test(`an older ${first} health response cannot overwrite newer health`, async ({ page }) => {
    const mock = await openConsole(page);
    const late = mock.deferReply("GET", "/health");
    try {
      await page.locator(first === "probe" ? "#conn-retest" : "#tab-monitoring").click();
      await late.started;
      mock.health = { status: "unhealthy", qdrant: "reachable", embedder: "unreachable" };
      await page.locator(first === "probe" ? "#tab-monitoring" : "#conn-retest").click();
      await expect(page.locator("#conn-health-status")).toHaveText("UNHEALTHY");
      const received = page.waitForResponse((r) => r.url().endsWith("/health"));
      late.resolve({ status: "healthy", qdrant: "reachable", embedder: "reachable" });
      await received;
      await expect(page.locator("#conn-health-status")).toHaveText("UNHEALTHY");
    } finally { late.resolve({}); }
  });
}

test("editing ingest options while uploading prevents an obsolete index phase", async ({ page }) => {
  const op = operations.find((op) => op.name === "ingest");
  const mock = await openOperation(page, op);
  await page.locator("#ing-create-indexes").check();
  const late = mock.deferReply("POST", "/points/ingest-chunks");
  try {
    await page.locator("#ing-confirm-submit").click();
    const request = await late.started;
    await page.locator("#ing-create-indexes").uncheck();
    await page.locator("#ing-create-indexes").check();
    const received = page.waitForResponse((r) => r.url().endsWith("/points/ingest-chunks"));
    late.resolve(success(op, request, mock));
    await received;
    expect(mock.requestsFor("POST", `/collections/${A}/payload-indexes`)).toHaveLength(0);
    expect(mock.errors).toEqual([]);
  } finally { late.resolve({}); }
});

test("an old finally does not unlock a newer request", async ({ page }) => {
  const mock = await openConsole(page);
  await openPoints(page, "get");
  await page.locator("#get-id").fill("7");
  const old = mock.deferReply("GET", "/points/7"), newer = mock.deferReply("GET", "/points/8");
  try {
    await page.locator("#get-submit").click(); await old.started;
    await page.locator("#get-id").fill("8");
    await page.locator("#get-submit").click(); await newer.started;
    const received = page.waitForResponse((r) => r.url().includes("/points/7"));
    old.resolve({ collection_name: A, id: "7", payload: {} }); await received;
    await expect(page.locator("#get-submit")).toBeDisabled();
    newer.resolve({ collection_name: A, id: "8", payload: { text: "newer result" } });
    await expect(page.locator("#get-result")).toContainText("newer result");
    await expect(page.locator("#get-submit")).toBeEnabled();
  } finally { old.resolve({}); newer.resolve({}); }
});
