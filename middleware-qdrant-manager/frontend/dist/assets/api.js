// ── API client for Smart Semantic Searcher ──
// ES module — every method maps one-to-one to a manager endpoint and resolves
// to {ok, status, data, error, elapsed}; network errors are captured, not thrown.
import { responseValidators as valid } from "/assets/models.js?v=71";

// The console is served by the manager itself, so the API base URL is always
// the page origin; it is configured server-side via .env, never in the UI.
const BASE_URL = (window.location.origin || "http://127.0.0.1:6336").replace(/\/+$/, "");

function getBaseUrl() {
  return BASE_URL;
}

// The bearer token is the API key the user registers from the topbar
// popover (see initApiKey in app.js); it must be the token half of one entry
// in MIDDLEWARE_API_TOKENS or protected endpoints answer 401.
function getToken() {
  return sessionStorage.getItem("qhm_api_key") || "";
}

// Server-provided flag: true when the manager enforces bearer auth.
function authRequired() {
  return Boolean(window.QHM_CONFIG && window.QHM_CONFIG.authRequired);
}

function headers(json = true) {
  const h = {};
  if (json) h["Content-Type"] = "application/json";
  const token = getToken();
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

async function request(method, path, { body, params, raw = false, validate = null } = {}) {
  const url = new URL(BASE_URL + path);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) url.searchParams.set(k, v);
    }
  }
  const started = performance.now();
  try {
    const resp = await fetch(url.toString(), {
      method,
      headers: headers(true),
      body: body ? JSON.stringify(body) : undefined,
    });
    const elapsed = (performance.now() - started) / 1000;
    const text = await resp.text();
    let data;
    if (raw && resp.ok) {
      // The caller needs the exact bytes (JSON.parse + JSON.stringify would
      // reorder integer-like keys, and the export must be byte-identical).
      // Error bodies still parse, so apiErrorHtml keeps rendering `detail`.
      data = text;
    } else {
      try { data = JSON.parse(text); } catch { data = text; }
    }
    let error = resp.ok ? null : `HTTP ${resp.status}`;
    if (resp.status === 401) {
      error = getToken()
        ? "API key rejected by the manager — register the correct one."
        : "API key required — register it from the API KEY button.";
    }
    const validBody = !resp.ok || !validate || validate(data);
    if (!validBody) error = "The manager returned an unexpected response. Refresh the operation state before retrying.";
    return { ok: resp.ok && validBody, status: resp.status, data, error, elapsed, headers: resp.headers };
  } catch (err) {
    const elapsed = (performance.now() - started) / 1000;
    // Raw fetch errors ("TypeError: Failed to fetch") are noise to users;
    // the console keeps the original for debugging.
    console.warn("API request failed:", err);
    return {
      ok: false,
      status: null,
      data: null,
      error: "Cannot reach the manager (network error)",
      elapsed,
      headers: null,
    };
  }
}

const enc = encodeURIComponent;
const scoped = (validate, name) => (data) => validate(data)
  && (!name || data.collection_name === name);

export const api = {
  getBaseUrl,
  getToken,
  authRequired,

  // Query
  search(payload) {
    return request("POST", "/query/search", { validate: scoped(valid.search, payload.collection_name), body: payload });
  },
  scrollPoints(payload) {
    return request("POST", "/points/scroll", { validate: scoped(valid.scroll, payload.collection_name), body: payload });
  },

  // Collections
  getCollections() {
    return request("GET", "/collections", { validate: valid.collections });
  },
  getCollection(name) {
    return request("GET", `/collections/${enc(name)}`, { validate: scoped(valid.report, name) });
  },
  createCollection(payload) {
    return request("POST", "/collections", { validate: scoped(valid.collectionOperation, payload.collection_name), body: payload });
  },
  verifyCollection(payload) {
    return request("POST", "/collections/verify", { validate: scoped(valid.collectionOperation, payload.collection_name), body: payload });
  },
  updateSchemaToggles(name, payload, method = "PATCH") {
    const verb = String(method).toUpperCase() === "PUT" ? "PUT" : "PATCH";
    return request(verb, `/collections/${enc(name)}/schema-toggles`, { validate: scoped(valid.collectionOperation, name), body: payload });
  },
  preflightReindex(name) {
    return request("POST", `/collections/${enc(name)}/reindex/preflight`, { validate: scoped(valid.preflight, name) });
  },
  startReindex(name, confirmCollection) {
    return request("POST", `/collections/${enc(name)}/reindex`, { validate: scoped(valid.reindexAction, name),
      body: { confirm_collection: confirmCollection },
    });
  },
  getReindexStatus(name) {
    return request("GET", `/collections/${enc(name)}/reindex`, { validate: scoped(valid.reindex, name) });
  },
  pauseReindex(name) {
    return request("POST", `/collections/${enc(name)}/reindex/pause`, { validate: scoped(valid.reindexAction, name) });
  },
  resumeReindex(name) {
    return request("POST", `/collections/${enc(name)}/reindex/resume`, { validate: scoped(valid.reindexAction, name) });
  },
  cancelReindex(name) {
    return request("POST", `/collections/${enc(name)}/reindex/cancel`, { validate: scoped(valid.reindexAction, name) });
  },
  getPayloadIndexes(name) {
    return request("GET", `/collections/${enc(name)}/payload-indexes`, { validate: scoped(valid.indexes, name) });
  },
  getPayloadKeys(name, sampleSize = 100) {
    return request("GET", `/collections/${enc(name)}/payload-keys`, { validate: scoped(valid.keys, name),
      params: { sample_size: sampleSize },
    });
  },
  createPayloadIndex(name, payload) {
    return request("POST", `/collections/${enc(name)}/payload-indexes`, { validate: scoped(valid.indexCreate, name), body: payload });
  },
  deletePayloadIndex(name, fieldName) {
    return request("DELETE", `/collections/${enc(name)}/payload-indexes/${enc(fieldName)}`, { validate: scoped(valid.indexDelete, name) });
  },
  deleteCollection(name, confirmDelete) {
    return request("DELETE", `/collections/${enc(name)}`, { validate: scoped(valid.collectionDelete, name),
      params: { confirm_delete: confirmDelete },
    });
  },

  // Points
  ingestChunks(payload) {
    return request("POST", "/points/ingest-chunks", { validate: scoped(valid.ingest, payload.collection_name), body: payload });
  },
  upsertPoint(payload) {
    return request("POST", "/points", { validate: scoped(valid.upsert, payload.collection_name), body: payload });
  },
  upsertBatch(payload) {
    return request("POST", "/points/batch", { validate: scoped(valid.batch, payload.collection_name), body: payload });
  },
  getPoint(pointId, collectionName, withVectors = false) {
    return request("GET", `/points/${enc(pointId)}`, { validate: scoped(valid.point, collectionName),
      params: { collection_name: collectionName, with_vectors: withVectors },
    });
  },
  retrievePoints(body) {
    return request("POST", "/points/retrieve", { validate: scoped(valid.retrieve, body.collection_name), body });
  },
  deletePoint(pointId, collectionName) {
    return request("DELETE", `/points/${enc(pointId)}`, { validate: scoped(valid.deleted, collectionName),
      params: { collection_name: collectionName },
    });
  },
  deletePoints(body) {
    return request("DELETE", "/points", { validate: scoped(valid.deleted, body.collection_name), body });
  },
  deletePointsByFilter(body) {
    return request("POST", "/points/delete-by-filter", { validate: scoped(valid.deleted, body.collection_name), body });
  },
  updatePayload(pointId, body, method) {
    const verb = String(method).toUpperCase() === "PUT" ? "PUT" : "PATCH";
    return request(verb, `/points/${enc(pointId)}/payload`, { validate: scoped(valid.payload, body.collection_name), body });
  },
  deletePayloadKeys(pointId, body) {
    return request("POST", `/points/${enc(pointId)}/payload/delete-keys`, { validate: scoped(valid.payload, body.collection_name), body });
  },
  clearPayload(pointId, body) {
    return request("POST", `/points/${enc(pointId)}/payload/clear`, { validate: scoped(valid.payload, body.collection_name), body });
  },
  // Bulk payload writes: same operations scoped by filter instead of by point id.
  // There is no filtered REPLACE — see patch_payload_by_filter in points.py.
  updatePayloadByFilter(body) {
    return request("PATCH", "/points/payload", { validate: scoped(valid.payloadFilter, body.collection_name), body });
  },
  deletePayloadKeysByFilter(body) {
    return request("POST", "/points/payload/delete-keys", { validate: scoped(valid.payloadFilter, body.collection_name), body });
  },
  clearPayloadByFilter(body) {
    return request("POST", "/points/payload/clear", { validate: scoped(valid.payloadFilter, body.collection_name), body });
  },
  countPoints(body) {
    return request("POST", "/points/count", { validate: scoped(valid.count, body.collection_name), body });
  },
  facetPoints(body) {
    return request("POST", "/points/facet", { validate: scoped(valid.facet, body.collection_name), body });
  },
  exportCollection(payload) {
    return request("POST", "/points/export", { validate: (data) => valid.export(data, payload.collection_name), body: payload, raw: true });
  },

  // Monitoring
  openapi() { return request("GET", "/openapi.json", { validate: valid.openapi }); },
  livez() { return request("GET", "/livez", { validate: valid.live }); },
  readyz() { return request("GET", "/readyz", { validate: valid.ready }); },
  health() { return request("GET", "/health", { validate: valid.health }); },
  stats() { return request("GET", "/stats", { validate: valid.stats }); },
  metrics() { return request("GET", "/metrics", { validate: valid.metrics }); },
};
