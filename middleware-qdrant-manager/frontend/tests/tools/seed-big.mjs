import { fixtureName } from "../harness.mjs";
import { setupCall as call } from "../harness.mjs";

const N=400;
console.log("create:", (await call("POST","/collections",{collection_name:fixtureName("e2e_z_big"),recreate:true,
  confirm_recreate:fixtureName("e2e_z_big"),dense_on_disk:true,sparse_on_disk:true,on_disk_payload:true,
  quantization:true,sparse_idf:true})).status);
for (let off=0; off<N; off+=50) {
  const pts=Array.from({length:Math.min(50,N-off)},(_,k)=>({point_id:String(off+k+1),
    text:`document ${off+k+1} paragraph about hybrid retrieval, dense and sparse vectors, reranking, and payload filtering in vector databases`,
    payload:{f_idx:off+k+1}}));
  const r=await call("POST","/points/batch",{collection_name:fixtureName("e2e_z_big"),points:pts});
  process.stdout.write(`${off+pts.length}:${r.status} `);
}
console.log("\npoints:", (await call("GET",`/collections/${fixtureName("e2e_z_big")}`)).data.points_count);
