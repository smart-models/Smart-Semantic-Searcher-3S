import { fixtureName } from "../harness.mjs";
import { setupCall as call } from "../harness.mjs";

const point=(i)=>({point_id:String(i),
  text:`alpha beta gamma document ${i} about hybrid retrieval and ranking`,
  payload:{f_status:`s${i}`,f_title:`alpha beta gamma ${i}`,f_views:i,f_price:i+0.5,
    f_flag:i%2===0,f_geo:{lon:9+i*0.01,lat:45},
    f_created:`2026-06-${String(i).padStart(2,"0")}T00:00:00Z`,
    f_uuid:`00000000-0000-4000-8000-${String(i).padStart(12,"0")}`,f_extra:`x${i}`,
    f_tags:i%3===0?["a","b"]:i%3===1?[]:["a"],
    ...(i%4===0?{f_nullable:null}:i%4===1?{f_nullable:`v${i}`}:{})}});
for (const [name,n] of [[fixtureName("e2e_c_points"),10],[fixtureName("e2e_c_victim"),12]]) {
  // Full rewrite: recreate so deleted ids and cleared payloads come back.
  const rc = await call("POST","/collections",{collection_name:name,recreate:true,
    confirm_recreate:name,dense_on_disk:true,sparse_on_disk:true,on_disk_payload:true,
    quantization:true,sparse_idf:true});
  const up = await call("POST","/points/batch",
    {collection_name:name,points:Array.from({length:n},(_,k)=>point(k+1))});
  for (const [f,sch] of [["f_status","keyword"],["f_title","text"],["f_views","integer"],
    ["f_price","float"],["f_flag","bool"],["f_geo","geo"],["f_created","datetime"],
    ["f_uuid","uuid"],["f_tags","keyword"],["f_nullable","keyword"]]) {
    await call("POST",`/collections/${name}/payload-indexes`,{indexes:[{field_name:f,field_schema:sch}]});
  }
  const chk = await call("GET",`/collections/${name}`);
  console.log(name,"recreate",rc.status,"seed",up.status,"points",chk.data.points_count);
}
