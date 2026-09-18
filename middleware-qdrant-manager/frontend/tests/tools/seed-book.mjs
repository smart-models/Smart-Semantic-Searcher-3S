// [API] setup for the test book's deterministic acceptance fixtures (§0.6).
// The generators below are the book's own, verbatim in semantics: every §8.2
// count and every TB-FIL/TB-EXP/TB-PAY oracle is derived from them, never from
// the API. Writes go through the ownership manifest (QHM_E2E_RUN_ID); each
// collection is cleaned and rebuilt, then read back and compared to the data.
//
//   node tests/tools/seed-book.mjs            # every e2e_book_* fixture
//   node tests/tools/seed-book.mjs --only e2e_book_filters
import { cleanupOwnedCollection, fixtureName, setupCall as call } from "../harness.mjs";
import { pathToFileURL } from "node:url";
import { deepStrictEqual } from "node:assert";

export const uuid = (i) => `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`;
export const day = (i) => new Date(Date.UTC(2026, 0, i)).toISOString();

export function verifyStoredPayload(name, point, payload) {
  const expected = { ...point.payload, text: point.text };
  deepStrictEqual(payload, expected, `${name}: point ${point.point_id} payload differs from the generator`);
}
const themes = ["cooking", "space", "music", "sports", "programming"];

export function filterPoint(i) {
  return {
    point_id: String(i),
    text: `${themes[Math.floor((i - 1) / 20)]} document ${i}`,
    payload: {
      source_id: "e2e_book_source",
      status: i <= 20 ? "active" : i <= 50 ? "archived" : i <= 90 ? "draft" : "deleted",
      tags: i <= 10 ? [] : i <= 40 ? ["red"] : i <= 80 ? ["red", "green"] : ["red", "green", "blue"],
      views: i,
      price: i + 0.5,
      flag: i % 2 === 0,
      description: `${i <= 15 ? "the quick brown fox jumps " : i <= 40 ? "lazy dog sleeps " : ""}document ${i} corpus`,
      notes: `${i <= 30 ? "gamma marker " : ""}note ${i}`,
      location: i <= 25 ? { lon: 9.19, lat: 45.46 } : i <= 60 ? { lon: 12.5, lat: 41.9 } : { lon: -74, lat: 40.7 },
      created_at: day(i),
      doc_uuid: uuid(i),
      ...(i <= 10 ? {} : { optional_note: i <= 20 ? null : `present-${i}` }),
      items: [{ label: "a", qty: i }, { label: "b", qty: 100 - i }],
    },
  };
}
export const filterPoints = Array.from({ length: 100 }, (_, k) => filterPoint(k + 1));
export const filterIndexes = [
  ["status", "keyword"], ["tags", "keyword"], ["views", "integer"],
  ["price", "float"], ["flag", "bool"], ["description", "text", true],
  ["notes", "text", false], ["location", "geo"], ["created_at", "datetime"],
  ["doc_uuid", "uuid"], ["optional_note", "keyword"],
  ["items[].label", "keyword"], ["items[].qty", "integer"],
].map(([field_name, field_schema, phrase_matching]) => ({
  field_name, field_schema,
  ...(field_schema === "text" ? { phrase_matching } : {}),
}));

export const edgeTypes = [
  ["k", "keyword", "red", "green"], ["i", "integer", 1, 2],
  ["f", "float", 1.5, 2.5], ["b", "bool", false, true],
  ["t", "text", "alpha", "beta"],
  ["g", "geo", { lon: 9.19, lat: 45.46 }, { lon: 12.5, lat: 41.9 }],
  ["d", "datetime", day(1), day(2)], ["u", "uuid", uuid(1), uuid(2)],
];
export const edgePoints = Array.from({ length: 6 }, (_, k) => {
  const i = k + 1;
  return {
    point_id: String(i), text: `edge document ${i}`,
    payload: Object.fromEntries(edgeTypes.flatMap(([key, , a, b]) =>
      i === 4 ? [] : [[key, i === 1 ? [] : i === 2 ? [a] : i === 3 ? [a, b] : i === 5 ? null : a]])),
  };
});
export const edgeIndexes = edgeTypes.map(([field_name, field_schema]) => ({
  field_name, field_schema, ...(field_schema === "text" ? { phrase_matching: true } : {}),
}));

export const exportPoints = Array.from({ length: 2001 }, (_, k) => ({
  point_id: String(k + 1), text: `export document ${k + 1}`,
  payload: { source_id: "e2e_book_export", id: k + 1, marker: `m-${k + 1}`, unicode: "à漢🙂" },
}));
export const writePoints = [1, 2, 3].map((i) => ({
  point_id: String(i), text: `original ${i}`,
  payload: { source_id: "e2e_book_write", ordinary: `keep-${i}`, nested: { leaf: "v" } },
}));
export const samplePoints = Array.from({ length: 1001 }, (_, k) => ({
  point_id: String(k + 1), text: `sample ${k + 1}`,
  payload: { source_id: "e2e_book_sample", ordinary: k + 1, ...(k + 1 === 1001 ? { rare: "survives" } : {}) },
}));
export const altPoints = [{ point_id: "1", text: "ALT ONLY", payload: { marker: "alt-only" } }];

const SOURCE_INDEX = [{ field_name: "source_id", field_schema: "keyword" }];
export const BOOK_FIXTURES = {
  e2e_book_filters: { points: filterPoints, indexes: filterIndexes },
  e2e_book_edges: { points: edgePoints, indexes: edgeIndexes },
  e2e_book_write: { points: writePoints, indexes: SOURCE_INDEX },
  e2e_book_export: { points: exportPoints, indexes: [] },
  e2e_book_empty: { points: [], indexes: [] },
  e2e_book_sample: { points: samplePoints, indexes: SOURCE_INDEX },
  e2e_book_alt: { points: altPoints, indexes: [] },
  e2e_book_roundtrip_a: { points: [], indexes: [] },
  e2e_book_roundtrip_b: { points: [], indexes: [] },
};

const BATCH = 100;

function chunks(list, size) {
  const out = [];
  for (let start = 0; start < list.length; start += size) out.push(list.slice(start, start + size));
  return out;
}

async function scrollAll(name) {
  const seen = new Map();
  let offset = null;
  for (;;) {
    const page = await call("POST", "/points/scroll", { collection_name: name, limit: 1000, ...(offset === null ? {} : { offset }) });
    for (const point of page.data.results) seen.set(String(point.id), point.payload);
    if (page.data.next_page_offset == null) break;
    offset = page.data.next_page_offset;
  }
  return seen;
}

// Build one owned fixture from scratch and prove it equals the generator.
export async function seedBookFixture(logical, { points, indexes } = BOOK_FIXTURES[logical]) {
  const name = fixtureName(logical);
  await cleanupOwnedCollection(name);
  await call("POST", "/collections", {
    collection_name: name, dense_on_disk: true, sparse_on_disk: true,
    on_disk_payload: true, quantization: true, sparse_idf: true,
  });
  let upserted = 0;
  for (const batch of chunks(points, BATCH)) {
    const result = await call("POST", "/points/batch", { collection_name: name, points: batch });
    if (result.data.upserted !== batch.length) throw new Error(`${name}: batch upserted ${result.data.upserted}, expected ${batch.length}`);
    upserted += result.data.upserted;
  }
  if (indexes.length) {
    const created = await call("POST", `/collections/${encodeURIComponent(name)}/payload-indexes`, { indexes });
    const failed = created.data.failed || [];
    if (failed.length) throw new Error(`${name}: payload indexes failed: ${JSON.stringify(failed)}`);
  }
  const report = await call("GET", `/collections/${encodeURIComponent(name)}`);
  if (report.data.points_count !== points.length) throw new Error(`${name}: report counts ${report.data.points_count}, seeded ${points.length}`);
  const schema = await call("GET", `/collections/${encodeURIComponent(name)}/payload-indexes`);
  const live = new Map(schema.data.indexes.map((item) => [item.field_name, item.field_schema]));
  for (const item of indexes) {
    if (live.get(item.field_name) !== item.field_schema) throw new Error(`${name}: index ${item.field_name} is ${live.get(item.field_name)}, expected ${item.field_schema}`);
  }
  const stored = await scrollAll(name);
  if (stored.size !== points.length) throw new Error(`${name}: scrolled ${stored.size} points, expected ${points.length}`);
  for (const point of points) {
    const payload = stored.get(point.point_id);
    verifyStoredPayload(name, point, payload);
  }
  console.log(`${name}: ${upserted} points, ${indexes.length} indexes, read back and verified`);
  return name;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const only = process.argv.indexOf("--only");
  const targets = only === -1 ? Object.keys(BOOK_FIXTURES) : [process.argv[only + 1]];
  for (const logical of targets) {
    if (!BOOK_FIXTURES[logical]) throw new Error(`Unknown book fixture ${logical}`);
    await seedBookFixture(logical);
  }
}
