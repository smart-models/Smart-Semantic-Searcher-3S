import { fixtureName } from "../harness.mjs";
import { setupCall as call } from "../harness.mjs";

for (const n of [fixtureName("e2e_a_new1"),fixtureName("e2e_a_new2"),fixtureName("e2e_a_new3"),fixtureName("e2e_a_new4"),fixtureName("e2e_a_new5"),fixtureName("e2e_a_c12probe")]) {
  const r = await call("DELETE", `/collections/${n}?confirm_delete=${n}`, undefined, { expected: [200, 404] });
  console.log("del", n, r.status);
}
// e2e_a_victim must hold 5 points again for C9.
const pts = Array.from({length:5},(_,k)=>({point_id:String(k+1),
  text:`victim chunk ${k+1} alpha beta`, payload:{v_tag:`t${k+1}`}}));
console.log("reseed victim:", (await call("POST","/points/batch",{collection_name:fixtureName("e2e_a_victim"),points:pts})).status);
const chk = await call("GET",`/collections/${fixtureName("e2e_a_victim")}`);
console.log("victim points:", chk.data.points_count);

// Drop indexes created by the PI series so PI1/PI3 see a clean slate.
const ix = await call("GET", `/collections/${fixtureName("e2e_a_keep")}/payload-indexes`);
if (!Array.isArray(ix.data.indexes)) throw new Error('Malformed payload-index inventory.');
for (const i of ix.data.indexes) {
  if (/^(pi[0-9]_|k5_tmp)/.test(i.field_name)) {
    const r = await call("DELETE", `/collections/${fixtureName("e2e_a_keep")}/payload-indexes/${i.field_name}`);
    console.log("dropix", i.field_name, r.status);
  }
}
