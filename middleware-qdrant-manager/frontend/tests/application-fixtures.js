import { startTelemetry, fulfillResponse } from "./telemetry.mjs";
// Isolated console fixtures: local assets and explicit API replies only.
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect } from "@playwright/test";

export const A = "e2e_application_a", B = "e2e_application_b";
const ORIGIN = "http://application.test";
const DIST = fileURLToPath(new URL("../dist/", import.meta.url));
export const compatible = {
  status: "compatible", stored: { profile: "basic", fingerprint: "same" },
  active: { profile: "basic", fingerprint: "same" },
  active_vectors: { dense: "dense", sparse: "sparse" },
};

const DEFAULT_INDEXES = [{ field_name: "category", field_schema: "keyword" }, { field_name: "items[].name", field_schema: "keyword" }];

export async function openConsole(page, { tab = "query", compatibility = compatible, job = null, origin = ORIGIN,
  openapi = { openapi: "3.1.0", paths: {} }, indexes = DEFAULT_INDEXES, payloadKeys = ["text", "source_id", "category"],
  config = {} } = {}) {
  const requests = [], unknown = [], pending = new Map();
  const telemetry = startTelemetry(page, { origin });
  const mock = {
    // OpenAPI is fetched once at boot; `null` answers it with a 500 so a test
    // can observe the console without a documented-path filter. `indexes` and
    // `payloadKeys` answer every collection's field lookups; `config` is merged
    // into window.QHM_CONFIG (runtime limits the console reads at boot).
    page, compatibility: structuredClone(compatibility), job, collections: [A, B], openapi, indexes, payloadKeys,
    health: { status: "healthy", qdrant: "reachable", embedder: "reachable", collections: 2 },
    requestsFor: (method, pathname) => requests.filter((r) => r.method === method && r.path === pathname),
    intercept: null, unknown,
    get errors() { return telemetry.events.filter((event) => event.type === "pageerror").map((event) => event.message); },
    deferReply(method, pathname) {
      let release, received;
      const response = new Promise((resolve) => { release = resolve; });
      const started = new Promise((resolve) => { received = resolve; });
      pending.set(`${method} ${pathname}`, { response, received });
      return { started, resolve: (json, status = 200) => release({ json, status }) };
    },
  };
  // Install a minute early: the installed clock runs until paused, and under
  // load pausing at the install instant fails with "fast-forward to the past".
  await page.clock.install({ time: new Date("2026-09-05T11:59:00Z") });
  await page.clock.pauseAt(new Date("2026-09-05T12:00:00Z"));
  await page.addInitScript((active) => localStorage.setItem("qhm_active_collection", active), A);
  await page.route("**/*", async (route) => {
    const req = route.request(), url = new URL(req.url()), pathname = url.pathname;
    if (url.origin !== origin) { unknown.push(req.url()); return route.abort(); }
    if (pathname === "/assets/config.js") return fulfillResponse(route, {
      contentType: "text/javascript", body: `window.QHM_CONFIG=${JSON.stringify({ authRequired: false, ...config })};`,
    });
    if (pathname === "/" || pathname.startsWith("/assets/")) {
      const filename = path.resolve(DIST, pathname === "/" ? "index.html" : pathname.slice(1));
      if (!filename.startsWith(DIST)) return route.abort();
      return fulfillResponse(route, { path: filename });
    }
    const entry = { method: req.method(), path: pathname, body: req.postDataJSON() };
    requests.push(entry);
    const key = `${entry.method} ${pathname}`, deferred = pending.get(key);
    if (deferred) {
      pending.delete(key); deferred.received(entry);
      return fulfillResponse(route, await deferred.response);
    }
    if (mock.intercept && await mock.intercept(route, entry)) return;
    const send = (json) => fulfillResponse(route, { json });
    if (pathname === "/health") return send(mock.health);
    if (pathname === "/livez") return send({ status: "alive" });
    if (pathname === "/readyz") return send({ status: "ready" });
    if (pathname === "/stats") return send({ total_requests: 0, rejected_requests: 0, uptime: "1h" });
    if (pathname === "/metrics") return fulfillResponse(route, { contentType: "text/plain", body: "# metrics\n" });
    if (pathname === "/openapi.json") {
      if (mock.openapi === null) return fulfillResponse(route, { status: 500, json: { detail: "openapi unavailable" } });
      return send(mock.openapi);
    }
    if (key === "GET /collections") return send({ collections: mock.collections });
    const name = decodeURIComponent(pathname.split("/")[2] || "");
    if (entry.method === "GET" && pathname.endsWith("/payload-indexes")) return send({ collection_name: name, indexes: mock.indexes });
    if (pathname.endsWith("/payload-keys")) return send({ collection_name: name, keys: mock.payloadKeys });
    if (entry.method === "GET" && pathname.endsWith("/reindex")) return send({
      collection_name: name, job: mock.job, task_running: mock.job?.status === "running",
    });
    if (entry.method === "GET" && /^\/collections\/[^/]+$/.test(pathname)) return send({
      collection_name: name, exists: true, status: "green", points_count: 10,
      dense: { size: 1024, on_disk: true }, sparse: { on_disk: true, idf: true },
      quantization: "turbo", on_disk_payload: true, schema_ok: true,
      embedding_compatibility: mock.compatibility,
    });
    unknown.push(key); return route.abort();
  });
  await page.goto(`${origin}/#${tab}`);
  await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
  await expect(page.locator("#conn-collection")).toHaveValue(A);
  await expect(page.locator("#cc-report")).toContainText(A);
  return mock;
}

export async function openPoints(page, tab) {
  await page.locator("#tab-points").click();
  await page.locator(`[data-pt-tab="${tab}"]`).click();
}

// Response families come from middleware routers and their Pydantic models.
// Keep these valid: malformed fixtures must not accidentally test only failures.
export function success(op, request, mock) {
  const target = op.typed ? B : A;
  const base = { ok: true, collection_name: target };
  const point = { id: "101", payload: { text: "marker response", category: "test" }, vectors: null };
  const page = { collection_name: target, returned: 1, total: 1, results: [point], next_page_offset: null };
  if (op.name === "export") return { chunks: [{ point_id: "101", text: "marker response" }], metadata: { n_chunks: 1, collection_name: target } };
  if (op.name === "reindex preflight") return { ...base, can_start: true, total_points: 10, eligible_points: 10, invalid_points: 0, invalid_point_ids: [], blockers: [] };
  if (op.name.startsWith("reindex ")) {
    const status = { "reindex start": "running", "reindex pause": "paused", "reindex resume": "running", "reindex cancel": "cancelled" }[op.name];
    return { ...base, job: { job_id: "test-job", status, processed_points: 1, total_points: 10 }, task_running: status === "running" };
  }
  if (["create", "recreate", "schema toggles"].includes(op.name)) return { ...base, created: op.name !== "schema toggles", report: {
    collection_name: target, exists: true, schema_ok: true, points_count: 0,
    dense: { size: 1024 }, sparse: {}, quantization: "turbo", embedding_compatibility: compatible,
  } };
  if (op.name === "collection delete") return { ...base, deleted: true };
  if (op.name === "index create") return { ...base, created: ["category"], updated: [], skipped: [], failed: [] };
  if (op.name === "index delete") return { ...base, deleted: true, field_name: "category" };
  if (op.name === "get") return { ...base, ...point };
  if (op.name === "get-many" || op.name === "query scroll" || op.name === "filtered-delete count") return page;
  if (op.name.startsWith("query ")) return { ...page, fusion: "rrf", reranked: false, candidates: 1, skipped_without_text: 0, results: [{ ...point, score: 0.9 }] };
  if (op.name === "facet") return { ...base, key: "category", hits: [{ value: "marker response", count: 1 }] };
  if (op.name.endsWith("count")) return { ...base, count: 1 };
  if (op.name === "upsert") return { ...base, point_id: "101", skipped: false };
  if (op.name === "batch") return { ...base, upserted: 1, skipped: 0 };
  if (op.name === "ingest") return { ...base, ingested: 1, skipped: 0 };
  if (op.tab === "payload") return op.filter ? { ...base, matched: 1 } : { ...base, point_id: "101" };
  return { ...base, deleted: 1 };
}

export async function openOperation(page, op) {
  const job = ["reindex pause", "reindex resume", "reindex cancel"].includes(op.name)
    ? { job_id: "test-job", status: op.name === "reindex resume" ? "paused" : "running", total_points: 10, processed_points: 1 } : null;
  const mock = await openConsole(page, { job });
  await setupOperation(op, mock);
  return mock;
}

const filter='{"must":[{"key":"category","match":{"value":"test"}}]}';
export const operations=[
 {name:'create',button:'cc-submit',result:'cc-create-result',tab:'collections',method:'POST',path:'/collections',typed:true},
 {name:'recreate',button:'cc-submit',result:'cc-create-result',tab:'collections',method:'POST',path:'/collections',typed:true,confirm:'cc-confirm'},
 {name:'schema toggles',button:'cu-submit',result:'cu-result',tab:'collections',method:'PATCH',suffix:'/schema-toggles'},
 {name:'index create',button:'pi-create',result:'pi-create-result',tab:'collections',method:'POST',suffix:'/payload-indexes'},
 {name:'index delete',button:'pi-delete',result:'pi-del-result',tab:'collections',method:'DELETE',suffix:'/payload-indexes/category'},
 {name:'collection delete',button:'cd-submit',result:'cd-result',tab:'collections',method:'DELETE',path:`/collections/${B}`,typed:true,confirm:'cd-confirm'},
 {name:'export',button:'ce-export',result:'ce-result',tab:'collections',method:'POST',path:'/points/export'},
 ...['preflight','start','pause','resume','cancel'].map(action=>({name:`reindex ${action}`,button:`cr-${action}`,result:action==='preflight'?'cr-preflight-result':'cr-result',tab:'collections',method:'POST',suffix:`/reindex${action==='start'?'':`/${action}`}`,...(action==='start'?{confirm:'cr-confirm'}:{})})),
 {name:'ingest',button:'ing-confirm-submit',result:'ing-result',tab:'ingest',method:'POST',path:'/points/ingest-chunks'},
 {name:'upsert',button:'up-submit',result:'up-result',tab:'upsert',method:'POST',path:'/points'},
 {name:'batch',button:'bt-submit',result:'bt-result',tab:'batch',method:'POST',path:'/points/batch'},
 {name:'get',button:'get-submit',result:'get-result',tab:'get',method:'GET',path:'/points/101'},
 {name:'get-many',button:'get-many-submit',result:'get-many-result',tab:'get',method:'POST',path:'/points/retrieve'},
 {name:'single delete',button:'del-submit',result:'del-result',tab:'delete',method:'DELETE',path:'/points/101'},
 {name:'bulk delete',button:'del-bulk-submit',result:'del-bulk-result',tab:'delete',method:'DELETE',path:'/points',confirm:'del-confirm'},
 {name:'filtered delete',button:'delf-submit',result:'delf-result',tab:'delete',method:'POST',path:'/points/delete-by-filter',confirm:'delf-confirm'},
 {name:'filtered-delete count',button:'delf-count-submit',result:'delf-count',tab:'delete',method:'POST',path:'/points/scroll'},
 {name:'payload-filter count',button:'pl-filter-count-submit',result:'pl-filter-count',tab:'payload',method:'POST',path:'/points/count',filter:true},
 ...['point','filter'].flatMap(scope=>['merge','keys','clear',...(scope==='point'?['replace']:[])].map(action=>({name:`payload ${scope} ${action}`,button:action==='keys'?'pl-del-keys-submit':action==='clear'?'pl-clear-submit':'pl-submit',result:action==='keys'?'pl-keys-result':action==='clear'?'pl-clear-result':'pl-result',tab:'payload',method:action==='merge'?'PATCH':action==='replace'?'PUT':'POST',path:`/points${scope==='point'?'/101':''}/payload${action==='keys'?'/delete-keys':action==='clear'?'/clear':''}`,filter:scope==='filter',...(action==='clear'?{confirm:'pl-clear-confirm'}:{})}))),
 {name:'count',button:'cnt-submit',result:'cnt-result',tab:'count',method:'POST',path:'/points/count'},
 {name:'facet',button:'cnt-facet-submit',result:'cnt-facet-result',tab:'count',method:'POST',path:'/points/facet'},
 {name:'query search',button:'ql-run',result:'ql-results',tab:'query',method:'POST',path:'/query/search'},
 {name:'query scroll',button:'ql-run',result:'ql-results',tab:'query',method:'POST',path:'/points/scroll'},
 {name:'query download',button:'ql-run',result:'ql-results',tab:'query',method:'POST',path:'/query/search',download:true},
];
export const matches=(op,r)=>r.method===op.method&&(op.path?r.path===op.path:r.path.endsWith(op.suffix));

export async function setupOperation(op,m,target=A) {
  const p=m.page;
  if(['query','collections'].includes(op.tab))await p.locator(`#tab-${op.tab}`).click();else await openPoints(p,op.tab);
  if(['create','recreate'].includes(op.name)){await p.locator('#cc-name').fill(B);await p.locator('#cc-recreate').setChecked(op.name==='recreate');}
  if(op.name==='index create')await p.locator('#pi-field').fill('category');
  if(op.name==='index delete')await p.locator('#pi-del-field').fill('category');
  if(op.name==='collection delete')await p.locator('#cd-name').fill(B);
  if(op.name==='reindex start'){
    const previous=m.intercept;m.intercept=async(route,r)=>{if(r.path.endsWith('/reindex/preflight')){await fulfillResponse(route, {json:success(operations.find(o=>o.name==='reindex preflight'),r,m)});return true;}return previous?previous(route,r):false;};
    await p.locator('#cr-preflight').click();await expect(p.locator('#cr-start')).toBeVisible();m.intercept=previous;
  }
  if(op.name==='ingest'){
    await p.locator('#ing-file').setInputFiles({name:'ui_mock.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({chunks:[{id:1,text:'a test point',category:'test'}]}))});
    await p.locator('#ing-create-indexes').uncheck();await expect(p.locator('#ing-confirm-submit')).toBeVisible();
  }
  if(op.name==='upsert'){await p.locator('#up-id').fill('101');await p.locator('#up-text').fill('a test point');await p.locator('#up-payload').fill('{"category":"test"}');}
  if(op.name==='batch')await p.locator('#bt-json').fill('[{"point_id":"101","text":"a test point","payload":{"category":"test"}}]');
  if(op.name==='get')await p.locator('#get-id').fill('101');
  if(op.name==='get-many')await p.locator('#get-ids').fill('101\n102');
  if(op.name==='single delete')await p.locator('#del-id').fill('101');
  if(op.name==='bulk delete')await p.locator('#del-ids').fill('101\n102');
  if(op.name.startsWith('filtered'))await p.locator('#delf-json').fill(filter);
  if(op.tab==='payload'){
    await p.locator(`input[name="pl-scope"][value="${op.filter?'filter':'point'}"]`).check();
    if(op.filter)await p.locator('#pl-filter-json').fill(filter);else await p.locator('#pl-id').fill('101');
    await p.locator('#pl-json').fill('{"category":"updated"}');await p.locator('#pl-keys').fill('category');
    if(!op.filter)await p.locator(`input[name="pl-mode"][value="${op.name.endsWith('replace')?'PUT':'PATCH'}"]`).check();
  }
  if(op.name==='facet')await p.locator('#cnt-facet-key').selectOption('category');
  if(op.name.startsWith('query ')){await p.locator('#ql-query').fill(op.name==='query scroll'?'':'a test point');await p.locator('#ql-download').setChecked(Boolean(op.download));}
  if(op.confirm)await p.locator(`#${op.confirm}`).fill(op.typed?B:target);
  await expect(p.locator(`#${op.button}`)).toBeEnabled();
}
