import { fixtureName } from "../harness.mjs";
// [API] setup — rebuilds e2e_z_reg with enough points for the re-embed series
// (z1-reembed): on a small collection the job finishes before the pause, resume
// and cancel windows can be observed.
import { setupCall as call, cleanupOwnedCollection } from "../harness.mjs";

const COLL=fixtureName("e2e_z_reg");
const N=2000;

await cleanupOwnedCollection(COLL);

const created = await call("POST", "/collections", {
  collection_name: COLL, recreate: true, confirm_recreate: COLL,
  dense_on_disk: true, sparse_on_disk: true, on_disk_payload: true,
  quantization: true, sparse_idf: true,
});
console.log("create:", created.status);
// Fail loudly: a silently refused recreate leaves the old data in place and the
// next run ends up debugging the wrong thing — which is exactly what happened.
if (created.status !== 200) {
  console.error("recreate refused:", JSON.stringify(created.data).slice(0, 300));
  process.exit(1);
}

for (let off=0; off<N; off+=50) {
  const pts=Array.from({length:Math.min(50,N-off)},(_,k)=>({point_id:String(off+k+1),
    text:`document ${off+k+1} paragraph about hybrid retrieval, dense and sparse vectors, reranking, and payload filtering in vector databases`,
    payload:{f_idx:off+k+1}}));
  const r=await call("POST","/points/batch",{collection_name:COLL,points:pts});
  process.stdout.write(`${off+pts.length}:${r.status} `);
  if (r.status !== 200) { console.error("\nseed failed:", JSON.stringify(r.data).slice(0,200)); process.exit(1); }
}
const points = (await call("GET",`/collections/${COLL}`)).data.points_count;
console.log("\npoints:", points);
if (points !== N) { console.error(`expected ${N} points, got ${points}`); process.exit(1); }
