import { fixtureName } from "../harness.mjs";
// [API] setup for the D (Query) agent: two extra e2e_d_* fixtures.
//  - e2e_d_nested   : array-of-object payload so "+ Same-element group" becomes
//                     available (F12 / QF-A) and nested vs flat AND differ.
//  - e2e_d_nophrase : a text index WITHOUT phrase_matching (QF-D).
import { setupCall as call } from "../harness.mjs";


async function make(name, points, indexes) {
  const c = await call("POST", "/collections", {
    collection_name: name, dense_on_disk: true, sparse_on_disk: true,
    on_disk_payload: true, quantization: true, sparse_idf: true,
  });
  console.log(`create ${name}: ${c.status}`);
  const up = await call("POST", "/points/batch", { collection_name: name, points });
  console.log(`  seed ${points.length}: ${up.status} ${JSON.stringify(up.data).slice(0, 120)}`);
  for (const item of indexes) {
    const ix = await call("POST", `/collections/${name}/payload-indexes`, { indexes: [item] });
    console.log(`  index ${item.field_name}: ${ix.status} ${ix.ok ? "" : JSON.stringify(ix.data).slice(0, 200)}`);
  }
}

// Same-element discriminator: only points 2 and 3 have ONE element that is
// both name=red AND qty=2. Point 1 has red and qty=2 in DIFFERENT elements,
// so a flat AND matches it but a nested (same-element) filter must not.
await make(fixtureName("e2e_d_nested"), [
  { point_id: "1", text: "nested alpha document one",
    payload: { items: [{ name: "red", qty: 1 }, { name: "blue", qty: 2 }], label: "n1" } },
  { point_id: "2", text: "nested alpha document two",
    payload: { items: [{ name: "red", qty: 2 }, { name: "blue", qty: 1 }], label: "n2" } },
  { point_id: "3", text: "nested alpha document three",
    payload: { items: [{ name: "red", qty: 2 }], label: "n3" } },
  { point_id: "4", text: "nested alpha document four",
    payload: { items: [{ name: "blue", qty: 5 }], label: "n4" } },
], [
  { field_name: "items[].name", field_schema: "keyword" },
  { field_name: "items[].qty", field_schema: "integer" },
  { field_name: "label", field_schema: "keyword" },
]);

await make(fixtureName("e2e_d_nophrase"), [
  { point_id: "1", text: "alpha beta gamma one",
    payload: { f_title: "alpha beta gamma 1", f_status: "s1" } },
  { point_id: "2", text: "alpha beta gamma two",
    payload: { f_title: "gamma beta alpha 2", f_status: "s2" } },
], [
  // deliberately NO phrase_matching
  { field_name: "f_title", field_schema: "text" },
  { field_name: "f_status", field_schema: "keyword" },
]);

console.log("FINAL:", JSON.stringify((await call("GET", "/collections")).data));
