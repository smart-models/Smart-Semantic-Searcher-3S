// Test book §7.7 — payload ownership, ID fidelity and ingest (TB-PAY-001..006,
// TB-ID-001..005, TB-ING-001/002/004). UI-live where the console can submit
// the input, API-live for syntax it cannot. `e2e_book_write` is reset before
// every mutation; point 3 is the untouched control. Seed with `npm run seed:book`.
// TB-ING-005/006 are Backend-owned (middleware pytest) and not claimed here.
import { expect, test } from "./fixtures.js";
import { fixtureName } from "./harness.mjs";
import { boot, openTab, openPointsSubtab } from "./helpers.js";
import { clearFilters, condition } from "./d-helpers.js";
import { api, bootPaced, fresh, snapshot } from "./tb-live-helpers.js";
import { writePoints } from "./tools/seed-book.mjs";
import fs from "node:fs";

const WRITE = fixtureName("e2e_book_write");
const SAMPLE = fixtureName("e2e_book_sample");
const filter = { must: [{ has_id: ["1", "2"] }] };
const original = writePoints.map((point) => ({ ...point, payload: { ...point.payload, source_id: WRITE } }));

async function reset() {
  await api("POST", "/points/batch", { collection_name: WRITE, points: original });
  return snapshot(WRITE);
}
const unchanged = async (before) => expect(await snapshot(WRITE)).toEqual(before);
const endpoint = (id, suffix) => `/points/${id}${suffix}?collection_name=${encodeURIComponent(WRITE)}`;

async function ui(page, panel, collection = WRITE) {
  const errors = await bootPaced(page, boot, { active: collection });
  await openTab(page, "points");
  await openPointsSubtab(page, panel);
  return errors;
}

// ── TB-PAY ───────────────────────────────────────────────────────────────────

for (const batch of [false, true]) {
  test(`TB-PAY-001/${batch ? "batch" : "single"} — the top-level text is authoritative over payload.text`, async ({ page }, info) => {
    const before = await reset();
    const errors = await ui(page, batch ? "batch" : "upsert");
    if (batch) {
      await page.locator("#bt-json").fill(JSON.stringify([1, 2].map((i) => ({ point_id: String(i), text: "base text", payload: { text: "spoof", ordinary: "new" } }))));
    } else {
      await page.locator("#up-id").fill("1");
      await page.locator("#up-text").fill("base text");
      await page.locator("#up-payload").fill(JSON.stringify({ text: "spoof", ordinary: "new" }));
    }
    const pending = page.waitForResponse((r) => new URL(r.url()).pathname === (batch ? "/points/batch" : "/points") && r.request().method() === "POST");
    await page.locator(batch ? "#bt-submit" : "#up-submit").click();
    expect((await pending).status()).toBe(200);
    const after = await snapshot(WRITE);
    for (const point of after) {
      if (["1", ...(batch ? ["2"] : [])].includes(String(point.id))) expect(point.payload).toEqual({ text: "base text", ordinary: "new" });
    }
    expect(after.find((p) => String(p.id) === "3")).toEqual(before.find((p) => String(p.id) === "3"));
    await info.attach("stored-payload", { body: JSON.stringify(after), contentType: "application/json" });
    expect(errors).toEqual([]);
  });
}

for (const mode of ["PATCH", "PUT", "filter"]) {
  test(`TB-PAY-002/${mode} — a text mutation is refused atomically with a readable service-owned-field message`, async ({}, info) => {
    const before = await reset();
    const result = mode === "filter"
      ? await api("PATCH", "/points/payload", { collection_name: WRITE, filters: filter, payload: { text: "spoof", ordinary: "new" }, confirm_collection: WRITE }, { expected: 400 })
      : await api(mode, endpoint("1", "/payload"), { collection_name: WRITE, payload: { text: "spoof", ordinary: "new" } }, { expected: 400 });
    expect(JSON.stringify(result.data)).toMatch(/text/i);
    await unchanged(before);
    await info.attach("response", { body: JSON.stringify(result.data), contentType: "application/json" });
  });
}

const protectedKeys = ["text", "source_id", '"text"', '"source_id"', '"text"[]', '"text".sub', "text.sub", '"\\u0074ext"', '"\\u0073ource_id"', '"\\u0074\\u0065\\u0078\\u0074"'];
for (const scope of ["point", "filter"]) {
  for (const [index, key] of protectedKeys.entries()) {
    test(`TB-PAY-003/${scope}-path${index} — protected root ${JSON.stringify(key)} is refused with the ordinary key in the same request`, async ({}, info) => {
      const before = await reset();
      const body = { keys: ["ordinary", key], ...(scope === "filter" ? { collection_name: WRITE, filters: filter, confirm_collection: WRITE } : { collection_name: WRITE }) };
      const raw = JSON.stringify(body);
      // A literal backslash-u in the path is distinct from JSON transport decoding.
      expect(raw.includes("\\\\u") || !key.includes("\\u")).toBe(true);
      const result = await api("POST", scope === "point" ? endpoint("1", "/payload/delete-keys") : "/points/payload/delete-keys", body, { expected: 400 });
      await unchanged(before);
      await info.attach("raw-body-and-response", { body: JSON.stringify({ body: raw, response: result.data }), contentType: "application/json" });
    });
  }
  for (const [index, key] of ["", " ", "ordinary\0", '"ordinary', "ordinary[abc]"].entries()) {
    test(`TB-PAY-004/${scope}-malformed${index} — malformed path syntax is refused before any write`, async () => {
      const before = await reset();
      await api("POST", scope === "point" ? endpoint("1", "/payload/delete-keys") : "/points/payload/delete-keys",
        { keys: ["ordinary", key], ...(scope === "filter" ? { collection_name: WRITE, filters: filter, confirm_collection: WRITE } : { collection_name: WRITE }) }, { expected: 400 });
      await unchanged(before);
    });
  }
  test(`TB-PAY-004/${scope}-valid — ordinary and nested.leaf are removed only from the selected targets`, async () => {
    const before = await reset();
    await api("POST", scope === "point" ? endpoint("1", "/payload/delete-keys") : "/points/payload/delete-keys",
      { keys: ["ordinary", "nested.leaf"], ...(scope === "filter" ? { collection_name: WRITE, filters: filter, confirm_collection: WRITE } : { collection_name: WRITE }) });
    const after = await snapshot(WRITE);
    for (const point of after) {
      if (String(point.id) === "1" || (scope === "filter" && String(point.id) === "2")) {
        expect(point.payload).toEqual({ text: `original ${point.id}`, source_id: WRITE, nested: {} });
      } else expect(point).toEqual(before.find((p) => p.id === point.id));
    }
  });
}

for (const [label, method, payload, expected] of [
  ["merge", "PATCH", { ordinary: "changed" }, { text: "original 1", source_id: WRITE, ordinary: "changed", nested: { leaf: "v" } }],
  ["replace", "PUT", { ordinary: "replacement" }, { text: "original 1", source_id: WRITE, ordinary: "replacement" }],
  ["source-replace", "PUT", { source_id: "e2e_book_changed" }, { text: "original 1", source_id: "e2e_book_changed" }],
]) {
  test(`TB-PAY-005/${label} — exact resulting payload and untouched control`, async () => {
    const before = await reset();
    await api(method, endpoint("1", "/payload"), { collection_name: WRITE, payload });
    const after = await snapshot(WRITE);
    expect(after.find((p) => String(p.id) === "1").payload).toEqual(expected);
    expect(after.find((p) => String(p.id) === "3")).toEqual(before.find((p) => String(p.id) === "3"));
  });
}

test("TB-PAY-005/clear — a single-point clear keeps only the protected fields actually present", async () => {
  const before = await reset();
  await api("POST", endpoint("1", "/payload/clear"), { collection_name: WRITE, confirm_collection: WRITE });
  const after = await snapshot(WRITE);
  expect(after.find((p) => String(p.id) === "1").payload).toEqual({ text: "original 1", source_id: WRITE });
  expect(after.find((p) => String(p.id) === "3")).toEqual(before.find((p) => String(p.id) === "3"));
});

test("TB-PAY-006 — a filtered clear over 1001 points preserves text/source_id and the unsampled rare key", async ({ page }, info) => {
  const first = (await api("POST", "/points/scroll", { collection_name: SAMPLE, limit: 1000 })).data.results;
  expect(first.length).toBe(1000);
  expect(first.some((p) => String(p.id) === "1001" || "rare" in p.payload)).toBe(false);
  const errors = await ui(page, "payload", SAMPLE);
  await page.locator('input[name="pl-scope"][value="filter"]').check();
  await page.locator("#pl-filter-json").fill(JSON.stringify({ must: [{ key: "source_id", match: { value: "e2e_book_sample" } }] }));
  await page.locator("#pl-clear-confirm").fill(SAMPLE);
  const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/payload/clear");
  await page.locator("#pl-clear-submit").click();
  const response = await pending;
  expect(response.status()).toBe(200);
  const after = await snapshot(SAMPLE);
  expect(after.length).toBe(1001);
  for (const point of after) {
    expect(point.payload).toEqual({ text: `sample ${point.id}`, source_id: "e2e_book_sample", ...(String(point.id) === "1001" ? { rare: "survives" } : {}) });
  }
  await info.attach("sample-clear", { body: JSON.stringify({ response: await response.json(), rarePoint: after.find((p) => String(p.id) === "1001"), count: after.length }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

// ── TB-ID ────────────────────────────────────────────────────────────────────

for (const ids of [["1", "01", "0001"], ["abcdefab-cdef-4abc-8def-abcdefabcdef", "ABCDEFAB-CDEF-4ABC-8DEF-ABCDEFABCDEF"]]) {
  test(`TB-ID-001/${ids[0]} — canonical duplicates in one batch are refused before any write`, async () => {
    const before = await reset();
    await api("POST", "/points/batch", { collection_name: WRITE, points: ids.map((point_id) => ({ point_id, text: "duplicate must not write" })) }, { expected: 400 });
    await unchanged(before);
  });
}

const boundaryIds = ["0", "9007199254740991", "9007199254740993", "18446744073709551615", "abcdefab-cdef-4abc-8def-abcdefabcdef"];

test("TB-ID-002/batch-roundtrip — boundary ids round-trip exactly through batch, get, get-many, query download and delete", async ({ page }, info) => {
  const name = await fresh("e2e_book_ids");
  const errors = await ui(page, "batch", name);
  await page.locator("#bt-json").fill(JSON.stringify(boundaryIds.map((point_id) => ({ point_id, text: `ID ${point_id}`, payload: { marker: point_id } }))));
  let pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/batch");
  await page.locator("#bt-submit").click();
  expect((await pending).status()).toBe(200);
  const stored = await snapshot(name);
  expect(stored.map((p) => String(p.id)).sort()).toEqual([...boundaryIds].sort());
  await openPointsSubtab(page, "get");
  for (const id of boundaryIds) {
    await page.locator("#get-id").fill(id);
    pending = page.waitForResponse((r) => new URL(r.url()).pathname === `/points/${id}` && r.request().method() === "GET");
    await page.locator("#get-submit").click();
    const response = await pending;
    expect(response.status()).toBe(200);
    expect(String((await response.json()).id)).toBe(id);
    await expect(page.locator("#get-result")).toContainText(id);
  }
  await page.locator("#get-ids").fill(boundaryIds.join("\n"));
  pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/retrieve");
  await page.locator("#get-many-submit").click();
  expect((await (await pending).json()).results.map((p) => String(p.id)).sort()).toEqual([...boundaryIds].sort());
  await openTab(page, "query");
  await clearFilters(page);
  await page.locator("#ql-query").fill("");
  await condition(page, { field: "(point id)", op: "id_any", value: boundaryIds.join(",") });
  await page.locator("#ql-download").check();
  const downloaded = page.waitForEvent("download");
  pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/scroll");
  await page.locator("#ql-run").click();
  expect((await pending).status()).toBe(200);
  const file = JSON.parse(fs.readFileSync(await (await downloaded).path(), "utf8"));
  expect(file.chunks.map((p) => p.point_id).sort()).toEqual([...boundaryIds].sort());
  await expect(page.locator("#ql-results .result-id")).toHaveCount(5);
  await openTab(page, "points");
  await openPointsSubtab(page, "delete");
  for (const id of boundaryIds) {
    await page.locator("#del-id").fill(id);
    pending = page.waitForResponse((r) => new URL(r.url()).pathname === `/points/${id}` && r.request().method() === "DELETE");
    await page.locator("#del-submit").click();
    expect((await pending).status()).toBe(200);
  }
  expect(await snapshot(name)).toEqual([]);
  await info.attach("exact-id-evidence", { body: JSON.stringify({ stored, download: file }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("TB-ID-002/single-upsert — each boundary id is sent exactly as typed through the single form", async ({ page }, info) => {
  const name = await fresh("e2e_book_ids");
  const errors = await ui(page, "upsert", name);
  const bodies = [];
  for (const id of boundaryIds) {
    await page.locator("#up-id").fill(id);
    await page.locator("#up-text").fill(`boundary ${id}`);
    await page.locator("#up-payload").fill("{}");
    const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points" && r.request().method() === "POST");
    await page.locator("#up-submit").click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const sent = response.request().postDataJSON();
    expect(sent.point_id).toBe(id);
    bodies.push(sent);
  }
  const after = await snapshot(name);
  expect(after.map((p) => String(p.id)).sort()).toEqual([...boundaryIds].sort());
  for (const point of after) expect(point.payload.text).toBe(`boundary ${point.id}`);
  await info.attach("single-boundaries", { body: JSON.stringify({ bodies, after }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

test("TB-ID-002/unsafe-json-number — a numeric point id beyond the safe range is refused by the console before submission", async ({ page }, info) => {
  const errors = await ui(page, "batch");
  let requests = 0;
  page.on("request", (r) => { if (/\/points\/batch$/.test(new URL(r.url()).pathname)) requests++; });
  await page.locator("#bt-json").fill('[{"point_id":9007199254740993,"text":"unsafe numeric ID"}]');
  await expect(page.locator("#bt-submit")).toBeDisabled();
  expect(requests).toBe(0);
  await info.attach("validation", { body: await page.locator("#pt-batch").innerText() });
  expect(errors).toEqual([]);
});

for (const id of ["-1", "18446744073709551616", "1.5", "not-a-uuid"]) {
  for (const batch of [false, true]) {
    test(`TB-ID-003/${id}-${batch ? "batch" : "single"} — an invalid point id is a 400 with no write, even beside a valid item`, async () => {
      const before = await reset();
      const point = { point_id: id, text: "invalid" };
      await api("POST", batch ? "/points/batch" : "/points",
        batch ? { collection_name: WRITE, points: [{ point_id: "1", text: "must not write" }, point] } : { collection_name: WRITE, ...point }, { expected: 400 });
      await unchanged(before);
    });
  }
}

for (const [label, chunk] of [
  ["true", { id: true, text: "a" }], ["string", { id: "1", text: "a" }], ["null", { id: null, text: "a" }], ["fraction", { id: 1.5, text: "a" }],
  ["text-null", { id: 2, text: null }], ["text-number", { id: 2, text: 2 }], ["text-object", { id: 2, text: {} }], ["text-empty", { id: 2, text: "" }], ["text-blank", { id: 2, text: " " }],
]) {
  test(`TB-ID-004/${label} — a valid chunk followed by an invalid one writes nothing`, async () => {
    const before = await snapshot(WRITE);
    await api("POST", "/points/ingest-chunks", { collection_name: WRITE, source_id: "e2e-invalid", chunks: [{ id: 9, text: "valid must not write" }, chunk] }, { expected: 400 });
    await unchanged(before);
  });
}

test("TB-ID-004/duplicate-ordinal — duplicate integer ordinals are refused before any write", async () => {
  const before = await snapshot(WRITE);
  await api("POST", "/points/ingest-chunks", { collection_name: WRITE, source_id: "e2e-invalid", chunks: [{ id: 1, text: "a" }, { id: 1, text: "b" }] }, { expected: 400 });
  await unchanged(before);
});

for (const label of ["unsafe-ingest-ordinal", "duplicate-ordinal-preflight"]) {
  test(`TB-ID-004/${label} — the browser preflight refuses lossy or duplicate ordinals without a request`, async ({ page }, info) => {
    const errors = await ui(page, "ingest");
    let requests = 0;
    page.on("request", (r) => { if (/\/points\/ingest-chunks$/.test(new URL(r.url()).pathname)) requests++; });
    const content = label.startsWith("duplicate") ? '[{"id":1,"text":"first"},{"id":1,"text":"second"}]' : '[{"id":9007199254740993,"text":"unsafe ordinal"}]';
    await page.locator("#ing-file").setInputFiles({ name: "e2e_unsafe.json", mimeType: "application/json", buffer: Buffer.from(`{"chunks":${content}}`) });
    await expect(page.locator("#ing-confirm-submit")).not.toBeEnabled();
    expect(requests).toBe(0);
    await info.attach("validation", { body: await page.locator("#pt-ingest").innerText() });
    expect(errors).toEqual([]);
  });
}

test("TB-ID-005 — signed ordinals -1/0/1 are valid, deterministic and idempotent on repeat", async () => {
  const name = await fresh("e2e_book_ingest");
  const body = { collection_name: name, source_id: "e2e-signed", chunks: [-1, 0, 1].map((id) => ({ id, text: `signed ${id}` })) };
  await api("POST", "/points/ingest-chunks", body);
  const before = await snapshot(name);
  expect(before.length).toBe(3);
  await api("POST", "/points/ingest-chunks", body);
  expect(await snapshot(name)).toEqual(before);
  expect(before.map((p) => p.payload.id).sort()).toEqual([-1, 0, 1]);
  for (const point of before) expect(String(point.id)).toMatch(/^[0-9a-f-]{36}$/);
});

// ── TB-ING ───────────────────────────────────────────────────────────────────

const sources = [["keyword", "e2e-book", "e2e-control"], ["integer", "42", "43"], ["float", "1.5", "2.5"],
  ["datetime", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], ["uuid", "00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000002"],
  ["bool", "true", "false"], ["text", "e2ebook", "controlbook"], ["untyped", "e2e-book", "e2e-control"]];
for (const [type, source, control] of sources) {
  test(`TB-ING-004/${type} — replace_source shrinks the source 2→1, keeps the control and indexes the declared type`, async ({}, info) => {
    const name = await fresh("e2e_book_ingest");
    const body = { collection_name: name, source_id_type: type };
    await api("POST", "/points/ingest-chunks", { ...body, source_id: control, chunks: [{ id: 1, text: "control" }] });
    await api("POST", "/points/ingest-chunks", { ...body, source_id: source, chunks: [{ id: 1, text: "old1" }, { id: 2, text: "old2" }] });
    const before = await snapshot(name);
    expect(before.length).toBe(3);
    const controlPoint = before.find((p) => p.payload.text === "control");
    const result = await api("POST", "/points/ingest-chunks", { ...body, source_id: source, replace_source: true, chunks: [{ id: 2, text: "new2" }] });
    const after = await snapshot(name);
    expect(after.length).toBe(2);
    expect(after.find((p) => p.id === controlPoint.id)).toEqual(controlPoint);
    const changed = after.find((p) => p.id !== controlPoint.id);
    expect(changed.payload.text).toBe("new2");
    expect(changed.payload.id).toBe(2);
    const indexes = (await api("GET", `/collections/${encodeURIComponent(name)}/payload-indexes`)).data.indexes;
    const schema = indexes.find((item) => item.field_name === "source_id")?.field_schema;
    // points.py _SOURCE_ID_INDEX_SCHEMAS: text and untyped sources are indexed as keyword.
    expect(schema).toBe(["text", "untyped"].includes(type) ? "keyword" : type);
    await info.attach("replacement-evidence", { body: JSON.stringify({ type, source, before, after, indexes, response: result.data }), contentType: "application/json" });
  });
}

test("TB-ING-001 — every mapping type is applied, stored payload and generated index type agree", async ({ page }, info) => {
  const name = await fresh("e2e_book_ingest");
  const errors = await ui(page, "ingest", name);
  const values = { kw: 42, tx: 42, integer: "42", float: "1.5", boolean: "true", date: "2026-01-01T00:00:00Z", uid: "00000000-0000-4000-8000-000000000001", geo: { lon: 9.19, lat: 45.46 }, raw: { nested: { leaf: "keep" } } };
  const types = { kw: "keyword", tx: "text", integer: "integer", float: "float", boolean: "bool", date: "datetime", uid: "uuid", geo: "geo", raw: "untyped" };
  await page.locator("#ing-file").setInputFiles({ name: "e2e_book_mapping.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ chunks: [{ id: 1, text: "mapped", ...values }] })) });
  await expect(page.locator("#ing-map-table")).toBeVisible();
  for (const [field, type] of Object.entries(types)) {
    const row = page.locator(`#ing-map-table tr[data-field="${field}"]`);
    await row.locator(".ing-map-name").fill(`mapped_${field}`);
    await row.locator(".ing-map-type").selectOption(type);
  }
  const indexed = page.waitForResponse((r) => new URL(r.url()).pathname === `/collections/${name}/payload-indexes` && r.request().method() === "POST");
  const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/ingest-chunks");
  await page.locator("#ing-confirm-submit").click();
  const response = await pending;
  expect(response.status()).toBe(200);
  await expect(page.locator("#ing-result")).toContainText("COMPLETED");
  expect((await indexed).status()).toBe(200);
  const stored = await snapshot(name);
  expect(stored.length).toBe(1);
  expect(stored[0].payload).toEqual({ text: "mapped", source_id: "e2e_book_mapping", id: 1, mapped_kw: "42", mapped_tx: "42", mapped_integer: 42, mapped_float: 1.5, mapped_boolean: true, mapped_date: values.date, mapped_uid: values.uid, mapped_geo: values.geo, mapped_raw: values.raw });
  const schemas = Object.fromEntries((await api("GET", `/collections/${encodeURIComponent(name)}/payload-indexes`)).data.indexes.map((i) => [i.field_name, i.field_schema]));
  for (const [field, type] of Object.entries(types)) {
    if (type === "untyped") expect(schemas).not.toHaveProperty(`mapped_${field}`);
    else expect(schemas[`mapped_${field}`]).toBe(type);
  }
  await info.attach("mapping", { body: JSON.stringify({ request: response.request().postDataJSON(), stored, schemas }), contentType: "application/json" });
  expect(errors).toEqual([]);
});

for (const [label, field, type, value, message] of [
  ["integer-abc", "n", "integer", "abc", /not convertible/],
  ["bad-uuid", "u", "uuid", "not-a-uuid", /not convertible/],
  ["invalid-bool", "b", "bool", "maybe", /not convertible/],
  ["blank-numeric", "f", "float", "", /not convertible/],
  ["structured-keyword", "s", "keyword", { nested: true }, /not convertible/],
]) {
  test(`TB-ING-001/${label} — a failing coercion blocks Confirm with a row-specific message and no request`, async ({ page }) => {
    const errors = await ui(page, "ingest");
    let requests = 0;
    page.on("request", (r) => { if (/\/points\/ingest-chunks$/.test(new URL(r.url()).pathname)) requests++; });
    await page.locator("#ing-file").setInputFiles({ name: "e2e_book_coercion.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ chunks: [{ id: 1, text: "coerce", [field]: value }] })) });
    await expect(page.locator("#ing-map-table")).toBeVisible();
    await page.locator(`#ing-map-table tr[data-field="${field}"] .ing-map-type`).selectOption(type);
    await expect(page.locator("#ing-map-errors")).toContainText(new RegExp(`field '${field}'`));
    await expect(page.locator("#ing-map-errors")).toContainText(message);
    await expect(page.locator("#ing-confirm-submit")).toBeDisabled();
    expect(requests).toBe(0);
    expect(errors).toEqual([]);
  });
}

test("TB-ING-002 — collisions, reserved and blank names block Confirm; untyped preserves the value; correcting clears the error", async ({ page }) => {
  const name = await fresh("e2e_book_ingest");
  const errors = await ui(page, "ingest", name);
  let requests = 0;
  page.on("request", (r) => { if (/\/points\/ingest-chunks$/.test(new URL(r.url()).pathname)) requests++; });
  await page.locator("#ing-file").setInputFiles({ name: "e2e_book_rename.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ chunks: [{ id: 1, text: "rename", alpha: "a", beta: { deep: true }, gamma: 3 }, { id: 2, text: "second", alpha: "b", gamma: 4 }] })) });
  await expect(page.locator("#ing-map-table")).toBeVisible();
  const nameOf = (field) => page.locator(`#ing-map-table tr[data-field="${field}"] .ing-map-name`);
  const errorsBox = page.locator("#ing-map-errors");
  const confirm = page.locator("#ing-confirm-submit");
  await nameOf("alpha").fill("gamma");
  await expect(errorsBox).toContainText(/duplicat/i);
  await expect(confirm).toBeDisabled();
  await nameOf("alpha").fill("");
  await expect(errorsBox).toContainText("field 'alpha': name is required");
  await expect(confirm).toBeDisabled();
  await nameOf("alpha").fill("text");
  await expect(errorsBox).toContainText(/reserved/);
  await nameOf("id").fill("source_id");
  await expect(errorsBox).toContainText(/duplicates the source_id name/);
  await nameOf("id").fill("text");
  await expect(errorsBox).toContainText(/reserved/);
  await nameOf("source_id").fill("gamma");
  await expect(errorsBox).toContainText(/source_id: name 'gamma' duplicates field/);
  expect(requests).toBe(0);
  // Correct everything: the errors clear and the untyped optional field is kept as-is.
  await nameOf("alpha").fill("alpha_renamed");
  await nameOf("id").fill("id");
  await nameOf("source_id").fill("source_id");
  await page.locator('#ing-map-table tr[data-field="beta"] .ing-map-type').selectOption("untyped");
  await expect(errorsBox).toBeEmpty();
  await expect(confirm).toBeEnabled();
  const indexed = page.waitForResponse((r) => new URL(r.url()).pathname === `/collections/${name}/payload-indexes` && r.request().method() === "POST");
  const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/points/ingest-chunks");
  await confirm.click();
  expect((await pending).status()).toBe(200);
  const indexRequest = (await indexed).request().postDataJSON();
  expect(indexRequest.indexes.map((item) => item.field_name)).not.toContain("beta");
  const stored = await snapshot(name);
  expect(stored.find((p) => p.payload.id === 1).payload).toMatchObject({ alpha_renamed: "a", beta: { deep: true }, gamma: 3 });
  expect(stored.find((p) => p.payload.id === 2).payload).not.toHaveProperty("beta");
  expect(errors).toEqual([]);
});
