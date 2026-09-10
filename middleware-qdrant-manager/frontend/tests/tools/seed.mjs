import { fixtureName } from "../harness.mjs";
// [API] setup — creates the per-agent sandbox collections and seeds points.
// Declared as setup in the test book (§0), not as a test case.
import { setupCall as call } from "../harness.mjs";


const point = (i) => ({
  point_id: String(i),
  text: `alpha beta gamma document ${i} about hybrid vector search and retrieval`,
  payload: {
    f_status: `s${i}`,
    f_title: `alpha beta gamma ${i}`,
    f_views: i,
    f_price: i + 0.5,
    f_flag: i % 2 === 0,
    f_geo: { lon: 9 + i * 0.01, lat: 45 },
    f_created: `2026-06-${String(i).padStart(2, "0")}T00:00:00Z`,
    f_uuid: `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`,
    f_extra: `x${i}`,
    f_tags: i % 3 === 0 ? ["a", "b"] : i % 3 === 1 ? [] : ["a"],
    ...(i % 4 === 0 ? { f_nullable: null } : i % 4 === 1 ? { f_nullable: `v${i}` } : {}),
  },
});

const INDEXES = [
  ["f_status", "keyword"], ["f_title", "text"], ["f_views", "integer"],
  ["f_price", "float"], ["f_flag", "bool"], ["f_geo", "geo"],
  ["f_created", "datetime"], ["f_uuid", "uuid"], ["f_tags", "keyword"],
  ["f_nullable", "keyword"],
];

const PLAN = [
  { name: fixtureName("e2e_a_keep"), points: 10, indexes: true },
  { name: fixtureName("e2e_a_victim"), points: 5, indexes: false },
  { name: fixtureName("e2e_a_empty"), points: 0, indexes: false },
  { name: fixtureName("e2e_b_ingest"), points: 0, indexes: false },
  { name: fixtureName("e2e_b_alt"), points: 0, indexes: false },
  { name: fixtureName("e2e_b_null"), points: 0, indexes: false },
  { name: fixtureName("e2e_b_map"), points: 0, indexes: false },
  { name: fixtureName("e2e_b_pass"), points: 0, indexes: false },
  { name: fixtureName("e2e_c_points"), points: 10, indexes: true },
  { name: fixtureName("e2e_c_victim"), points: 12, indexes: true },
  { name: fixtureName("e2e_d_query"), points: 12, indexes: true, phrase: true },
  { name: fixtureName("e2e_e_mon"), points: 3, indexes: false },
  { name: fixtureName("e2e_z_basic"), points: 10, indexes: true },
];

for (const spec of PLAN) {
  const created = await call("POST", "/collections", {
    collection_name: spec.name, dense_on_disk: true, sparse_on_disk: true,
    on_disk_payload: true, quantization: true, sparse_idf: true,
  });
  console.log(`create ${spec.name}: ${created.status} ${JSON.stringify(created.data).slice(0, 120)}`);
  if (spec.points > 0) {
    const pts = Array.from({ length: spec.points }, (_, k) => point(k + 1));
    const up = await call("POST", "/points/batch", { collection_name: spec.name, points: pts });
    console.log(`  seed ${spec.points} pts: ${up.status} ${JSON.stringify(up.data).slice(0, 120)}`);
  }
  if (spec.indexes) {
    for (const [field, schema] of INDEXES) {
      const item = { field_name: field, field_schema: schema };
      if (spec.phrase && schema === "text") item.phrase_matching = true;
      const ix = await call("POST", `/collections/${spec.name}/payload-indexes`, { indexes: [item] });
      if (!ix.ok) console.log(`  index ${field}/${schema}: ${ix.status} ${JSON.stringify(ix.data).slice(0, 160)}`);
    }
    console.log(`  indexes done`);
  }
}
const list = await call("GET", "/collections");
console.log("FINAL:", JSON.stringify(list.data));
