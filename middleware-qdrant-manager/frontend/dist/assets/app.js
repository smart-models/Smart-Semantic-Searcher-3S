// ── Smart Semantic Searcher console ──
// One-page UI: Query, Collections, Points, Monitoring. All API traffic
// goes through assets/api.js; all pure logic lives in assets/models.js.

// Version queries keep module caches in sync with index.html's ?v= bump.
import { api } from "/assets/api.js?v=63";
import {
  COLLECTIONS_HELP,
  COMMON_PAYLOAD_METADATA,
  DEFAULTS,
  FILTER_GROUPS,
  FILTER_GROUP_LABELS,
  POINT_ID_FIELD,
  VECTOR_FIELD,
  FILTER_OP_LABELS,
  RRF_WEIGHT_SCALE_MAX,
  RRF_WEIGHT_SCALE_MIN,
  MONITORING_HELP,
  POINTS_HELP,
  buildQueryParamHelp,
  buildBatchPayload,
  validateBatchPointIds,
  buildBulkDeleteBody,
  buildClearPayloadBody,
  buildCountPayload,
  buildCreatePayload,
  buildDeleteByFilterBody,
  buildFilterCountPayload,
  buildFilterCondition,
  buildIngestChunksPayload,
  buildIngestIndexRequests,
  buildPayloadFilterClearBody,
  buildPayloadFilterKeysBody,
  buildPayloadFilterUpdateBody,
  buildPayloadKeysDeleteBody,
  buildPayloadUpdateBody,
  buildRequestPayload,
  orderableFields,
  facetableFields,
  buildFacetPayload,
  buildRetrieveBody,
  buildScrollPayload,
  buildUpsertPayload,
  chunkArray,
  coercePayloadValue,
  INGEST_CHUNK_ID_RESERVED_KEYS,
  INGEST_SOURCE_ID_RESERVED_KEYS,
  parseTokenList,
  chooseActiveCollection,
  collectionSchemaMessages,
  confirmationSatisfied,
  createQueryRequestCoordinator,
  createRequestCoordinator,
  activeVectorNames,
  literalPayloadIndexField,
  defaultQueryState,
  downloadFilename,
  displayEmbedderProfile,
  embeddingCompatibilityGate,
  collectionLockState,
  filterOpsFor,
  filterValueKind,
  parseEndpointMetrics,
  normalizeQueryConfig,
  nestedRootsForFields,
  rrfWeightsFromBalance,
  schemaForField,
  quantizationEnabled,
  reportToggleDefaults,
  sortedCollectionNames,
  summarizeIngestChunks,
  summarizeIngestIndexResult,
  applyIngestFieldMapping,
  validateQueryState,
} from "/assets/models.js?v=72";

// ── DOM helpers ───────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);
const queryConfig = normalizeQueryConfig(window.QHM_CONFIG || {});
const queryCoordinator = createQueryRequestCoordinator();
const healthCoordinator = createRequestCoordinator();
const collectionsCoordinator = createRequestCoordinator();
// Each form owns its revisions. Typed collection targets are independent of
// the active selection; point operations share their visible subtab's inputs.
const operationScopes = new Map();
const busyOwners = new WeakMap();

function operationScope(button) {
  const selector = button.closest('[role="tabpanel"][id^="pt-"], [id$="-wrap"]');
  const element = selector || button;
  if (!operationScopes.has(element)) {
    operationScopes.set(element, {
      coordinator: createRequestCoordinator(), buttons: new Set(),
      active: !["cc-wrap", "cd-wrap"].includes(element.id),
    });
  }
  const scope = operationScopes.get(element);
  scope.buttons.add(button);
  return scope;
}

function invalidateOperations({ activeOnly = false, input = null } = {}) {
  for (const [element, scope] of operationScopes) {
    if (activeOnly && !scope.active) continue;
    if (input && !element.contains(input)) continue;
    scope.coordinator.changed();
    for (const button of scope.buttons) {
      busyOwners.delete(button);
      delete button.dataset.busy;
    }
  }
  if (!input || $("pt-ingest").contains(input)) {
    state.ingestRunId += 1;
    clearIngestCompleteTimer();
  }
}

function emptyFilterTree() {
  return { kind: "root", id: "root", label: "Payload filters", minShould: 1, children: [] };
}

function cloneFilterTree(node) {
  return {
    ...node,
    children: Array.isArray(node?.children)
      ? node.children.map((child) => cloneFilterTree(child))
      : [],
  };
}

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function uppercaseMessage(message) {
  return String(message).replace(/\.+\s*$/, "");
}

function alertHtml(type, message) {
  // Messages render uppercase without a trailing period (style choice).
  return `<p class="alert alert-${type}">${esc(uppercaseMessage(message))}</p>`;
}

function jsonBlock(data) {
  return `<pre class="json-block"><code>${esc(JSON.stringify(data, null, 2))}</code></pre>`;
}

// Describes a point's vectors by name and shape only — never the raw numbers.
// Absent when with_vectors was not requested (backend leaves it null).
function vectorsSummaryHtml(vectors) {
  if (!vectors || typeof vectors !== "object") return "";
  const parts = Object.entries(vectors).map(([name, info]) => {
    if (info?.type === "dense") return `${esc(name)} (dense, ${info.dimension}-d)`;
    if (info?.type === "sparse") return `${esc(name)} (sparse, ${info.nonzero} nonzero)`;
    return `${esc(name)} (${esc(info?.type ?? "unknown")})`;
  });
  return `<p class="caption"><strong>vectors:</strong> ${parts.join(" · ") || "none"}</p>`;
}

function metricTiles(entries) {
  const tiles = entries.map(([label, value]) =>
    `<div class="metric"><div class="metric-label">${esc(label)}</div>` +
    `<div class="metric-value">${esc(value === null || value === undefined ? "—" : value)}</div></div>`
  ).join("");
  return `<div class="metric-row">${tiles}</div>`;
}

function fillReference(containerId, items) {
  $(containerId).innerHTML = Object.entries(items).map(([name, desc]) =>
    `<div class="param-card"><div class="param-name">${esc(name)}</div>` +
    `<div class="param-desc">${esc(desc)}</div></div>`
  ).join("");
}

// Type-to-confirm field: keeps the destructive button disabled until the typed
// text matches the expected name, mirroring the backend confirmation gate.
function wireConfirm(inputId, hintId, getExpected, onChange) {
  $(inputId).addEventListener("input", () => {
    const typed = $(inputId).value;
    const matched = confirmationSatisfied(typed, getExpected());
    $(hintId).hidden = !typed || matched;
    associateFieldError($(inputId), hintId, Boolean(typed && !matched));
    onChange();
  });
  return () => confirmationSatisfied($(inputId).value, getExpected());
}

function formatScore(value) {
  if (value === null || value === undefined) return "n/a";
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(4) : String(value);
}

function scoreToPct(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  return Math.max(0, Math.min(1, num)) * 100;
}

// ── Shared state ──────────────────────────────────────────────────────────────

const ACTIVE_KEY = "qhm_active_collection";

const state = {
  collections: [],
  active: localStorage.getItem(ACTIVE_KEY) || "",
  filterTree: emptyFilterTree(),
  filterCtr: 0,
  vectorNames: [...new Set([
    window.QHM_CONFIG?.denseVectorName || "dense",
    window.QHM_CONFIG?.sparseVectorName || "sparse",
  ])],
  // Payload keys sampled from the active collection's points; empty until the
  // first fetch succeeds (no static fallback — avoids showing fake fields).
  payloadKeys: [],
  // Indexed payload fields offered by the console filter selectors
  // and their index schema, which decides the operations a row can offer.
  indexedFields: [],
  fieldSchemas: {},
  // Text fields whose index has phrase_matching: only these can use the
  // 'exact phrase' operation.
  phraseFields: new Set(),
  collectionFieldsLoading: false,
  collectionFieldsLoadId: 0,
  // Whether the last report fetch for the active collection succeeded; the
  // Schema Toggles widgets are only trustworthy (and submittable) when true.
  reportOk: false,
  reportLoadId: 0,
  embeddingCompatibility: null,
  embeddingCompatibilityLoading: Boolean(localStorage.getItem(ACTIVE_KEY)),
  servicesReady: false,
  serviceStatusMessage:
    "Checking Qdrant and Smart Embedder availability…",
  reindexPreflight: null,
  reindexStatus: null,
  reindexPollTimer: null,
  reindexCompleteTimer: null,
  reindexLoadId: 0,
  ingestFile: null,
  ingestPreflight: null,
  ingestCompleteTimer: null,
  // Client-side only (no env var, no backend flag): create the payload indexes
  // for the typed mapping rows once every batch of an ingest has landed.
  ingestCreateIndexes: true,
  // Monotonic id of the running upload; the post-ingest index phase uses it to
  // notice that a newer ingest has taken over.
  ingestRunId: 0,
  monitorTimer: null,
  monitorSeries: [],
  // View-only reset baselines: displayed counts = raw counter − baseline.
  // Raw snapshots are kept so "Reset counters" can baseline the latest values.
  statsBaseline: { total: 0, rejected: 0 },
  endpointBaseline: new Map(),
  lastRawStats: null,
  lastRawRows: null,
  // Documented API paths from /openapi.json; the per-endpoint table only
  // shows these (null until loaded = no path filtering).
  openapiPaths: null,
  healthTimer: null,
  // Last successful /health payload (embedder models, profile, …); consumed by
  // the QUERY results download to stamp provenance metadata into the file.
  health: null,
};

function activeEmbeddingGate() {
  return embeddingCompatibilityGate(state.embeddingCompatibility, {
    loading: state.embeddingCompatibilityLoading,
    hasCollection: Boolean(state.active),
  });
}

function activeReindexBlocksMutations() {
  return state.reindexStatus?.task_running === true
    || REINDEX_BLOCKING_STATUSES.has(state.reindexStatus?.job?.status);
}

function activeCollectionLock() {
  return collectionLockState(state.embeddingCompatibility, {
    servicesReady: state.servicesReady,
  });
}

// 03 Create / Recreate degrades to create-only under exactly the same condition
// that locks the other collection regions: an incompatible ACTIVE collection.
function createOnlyMode() {
  return activeCollectionLock().locked;
}

function setRegionLocked(id, locked) {
  const el = $(id);
  el.classList.toggle("operations-disabled", locked);
  el.toggleAttribute("inert", locked);
  el.setAttribute("aria-disabled", String(locked));
}

// Single owner of the inert/dimming lockdown for BOTH the service-readiness outage
// and the incompatible-collection cases. They share the same attributes on the same
// elements, so a single function must reconcile them (otherwise the collection path,
// running with servicesReady=false → locked=false, would strip the inert the outage
// path just set).
function applyLockdowns() {
  const ready = state.servicesReady;
  const { locked: collLocked, recoverable } = activeCollectionLock();
  const opsLocked = !ready || collLocked;
  setRegionLocked("panel-query", opsLocked);
  setRegionLocked("panel-points", !ready || (collLocked && !recoverable));
  if (!ready) {
    setRegionLocked("panel-collections", true);
    for (const id of [
      "cr-wrap", "cc-wrap", "cc-recreate-control", "cc-confirm-wrap",
      "cu-wrap", "pi-wrap", "cd-wrap",
    ]) {
      setRegionLocked(id, false);
    }
  } else {
    setRegionLocked("panel-collections", false);
    setRegionLocked("cr-wrap", collLocked && !recoverable);
    // 03 Create / Recreate is only half-locked: a NEW collection is built from
    // the ACTIVE embedding contract, so it cannot inherit the mismatch and stays
    // creatable. Only recreate — destructive on the mismatched collection — goes.
    setRegionLocked("cc-wrap", false);
    setRegionLocked("cc-recreate-control", collLocked);
    setRegionLocked("cc-confirm-wrap", collLocked);
    setRegionLocked("cu-wrap", collLocked);
    setRegionLocked("pi-wrap", collLocked);
    // 07 Delete is never locked by an incompatible collection. It addresses a
    // name the user types, not state.active, and its submit already gates on
    // the typed confirmation alone -- so the active collection's contract tells
    // it nothing, exactly as with the create half of 03 above. Locking it
    // stranded the user with a collection they could neither use nor remove.
    // The outage branch above still blocks it, through panel-collections.
    setRegionLocked("cd-wrap", false);
  }
  // conn-collection: disabled only by outage, never by collection lock.
  $("conn-collection").disabled = !ready || state.collections.length === 0;
  // The banner is the single source of the "why"; suppress inline guards/caption
  // whenever query/points are locked.
  if (opsLocked) {
    $("ql-embedding-guard").innerHTML = "";
    if (!ready || !recoverable) $("pt-embedding-guard").innerHTML = "";
  }
  $("ql-query-mode").hidden = opsLocked;
  renderServiceGate();
}

function renderServiceGate() {
  if (!state.servicesReady) {
    $("service-gate").innerHTML = alertHtml(
      "error",
      `${state.serviceStatusMessage} Collection, point, and query operations are disabled.`,
    );
    return;
  }
  const { locked, recoverable } = activeCollectionLock();
  if (locked) {
    // Basic<->advanced profile switch is an expected fingerprint drift, not a
    // surprise: suppress the recoverable banner for that specific case only.
    // Data operations stay disabled and Re-embed (02) stays available either way.
    const compat = state.embeddingCompatibility;
    const storedProfile = displayEmbedderProfile(compat?.stored?.profile);
    const activeProfile = displayEmbedderProfile(compat?.active?.profile);
    const isProfileSwitch = recoverable
      && storedProfile !== "—" && activeProfile !== "—"
      && storedProfile !== activeProfile;
    if (isProfileSwitch) {
      $("service-gate").innerHTML = "";
      return;
    }
    $("service-gate").innerHTML = alertHtml(
      "error",
      recoverable
        ? "Active collection uses a different vector space than the current Smart Embedder. Point retrieval remains available. Re-embed its vectors (02), create a new collection (03) or switch to a compatible collection."
        : "Active collection has no verified embedding contract, data operations are disabled. You can still create a new collection (03).",
    );
    return;
  }
  $("service-gate").innerHTML = "";
}

function setDisabledReason(ids, reason) {
  for (const id of ids) {
    const control = $(id);
    const reasonId = `${id}-reason`;
    let caption = $(reasonId);
    if (!caption && reason) {
      caption = document.createElement("span");
      caption.id = reasonId;
      caption.className = "caption disabled-reason";
      control.insertAdjacentElement("afterend", caption);
    }
    if (caption) {
      caption.textContent = reason;
      caption.hidden = !reason;
    }
    describeControl(control, reasonId, Boolean(reason));
    if (reason) control.title = reason;
    else control.removeAttribute("title");
  }
}

function describeControl(control, id, enabled) {
  if (!control) return;
  const tokens = new Set((control.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean));
  if (enabled) tokens.add(id); else tokens.delete(id);
  if (tokens.size) control.setAttribute("aria-describedby", [...tokens].join(" "));
  else control.removeAttribute("aria-describedby");
}

function associateFieldError(control, id, invalid) {
  if (!control) return;
  const previous = control.dataset.validationError;
  if (previous) describeControl(control, previous, false);
  if (invalid) {
    control.dataset.validationError = id;
    control.setAttribute("aria-invalid", "true");
    describeControl(control, id, true);
  } else if (previous) {
    delete control.dataset.validationError;
    control.removeAttribute("aria-invalid");
  }
}

function associateResponseErrors(button, result) {
  if (![400, 422].includes(result?.status) || !Array.isArray(result.data?.detail)) return;
  const scope = button.closest('[role="tabpanel"][id^="pt-"], [id$="-wrap"]');
  if (!scope) return;
  const error = [...scope.querySelectorAll('[id$="-result"]')].find((node) => node.querySelector(".alert-error"));
  if (!error) return;
  const fields = {
    point_id: /^(up|get|del|pl)-id$/, ids: /^(get|del)-ids$/,
    collection_name: /^(cc|cd)-name$/, field_name: /^pi-(field|del-field)$/,
    payload: /^(up-payload|pl-json)$/, points: /^bt-json$/,
    filters: /^(delf-json|pl-filter-json|cnt-json)$/, keys: /^pl-keys$/,
    text: /^up-text$/, confirm_collection: /^cr-confirm$/,
  };
  for (const item of result.data.detail) {
    const key = item?.loc?.find((part) => fields[part]);
    if (!key) continue;
    for (const control of scope.querySelectorAll("input[id], textarea[id], select[id]")) {
      if (fields[key].test(control.id)) associateFieldError(control, error.id, true);
    }
  }
}

function operationReasons(reasons) {
  for (const [id, missing] of Object.entries(reasons)) {
    const button = $(id);
    const busy = button.dataset.busy === "true" || (id === "ql-run" && queryCoordinator.isBusy());
    const reason = !state.servicesReady ? state.serviceStatusMessage
      : busy ? "Request in progress. Wait for its response."
      : button.disabled ? missing : "";
    button.setAttribute("aria-busy", String(busy));
    setDisabledReason([id], reason);
  }
}

function setOperationalReadiness(ready, message) {
  state.servicesReady = ready;
  state.serviceStatusMessage = message;
  updateQueryUI();
  updateCollectionsUI();
  updatePointsUI();
  applyLockdowns();
}

// Result areas whose messages describe the active collection's schema state
// (02 Re-embed, 04 Schema Toggles, 05 Payload Indexes). Collections 03/06 results are keyed
// to the typed name instead (and create/delete themselves change the active
// collection), so they are deliberately not in this list.
const ACTIVE_SCOPED_RESULTS = [
  "cr-result", "cr-preflight-result",
  "cu-result", "pi-create-result", "pi-del-result",
];

function clearActiveScopedResults() {
  for (const id of ACTIVE_SCOPED_RESULTS) $(id).innerHTML = "";
  resetReindexUI();
}

function setActiveCollection(name) {
  const changed = name !== state.active;
  state.active = name;
  // A transient failure (missing/rejected API key, outage) leaves the collection
  // list empty and calls this with "": persisting that would erase the user's
  // choice for good, so keep the stored preference and let chooseActiveCollection
  // restore it once the list loads again.
  if (name) localStorage.setItem(ACTIVE_KEY, name);
  $("ql-collection").value = name;
  if (changed) {
    invalidateOperations({ activeOnly: true });
    collectionsCoordinator.changed();
    markQueryChanged();
    // Results from the previous collection must not be shown as if current.
    for (const id of ["ce-result", "ing-result", "ing-preflight-result", "up-result", "bt-result", "get-result",
                      "get-many-result", "del-result", "del-bulk-result",
                      "delf-result", "pl-result", "pl-keys-result",
                      "pl-clear-result", "cnt-result", "cnt-facet-result"]) {
      $(id).innerHTML = "";
    }
    $("delf-count").hidden = true;
    $("pl-filter-count").hidden = true;
    clearActiveScopedResults();
    // Filter rows reference the previous collection's indexed fields; kept
    // rows would silently filter on fields the new collection may not have.
    state.filterTree = emptyFilterTree();
    state.payloadKeys = [];
    state.vectorNames = [];
    state.indexedFields = [];
    state.fieldSchemas = {};
    state.phraseFields = new Set();
    state.embeddingCompatibility = null;
    state.embeddingCompatibilityLoading = Boolean(name);
    state.ingestPreflight = null;
    clearIngestCompleteTimer();
    $("ing-preflight-result").hidden = false;
    renderPayloadFieldOptions();
    renderOrderByOptions();
    renderFacetKeyOptions();
    renderFilterTree();
    loadReport();
    loadReindexStatus();
    loadCollectionFields();
  }
  updateQueryUI();
  updateCollectionsUI();
  updatePointsUI();
  applyLockdowns();
}

// ── Connection bar ────────────────────────────────────────────────────────────

function populateCollectionSelect(select, collections, active) {
  if (collections.length) {
    select.disabled = !state.servicesReady;
    select.innerHTML = collections.map((name) =>
      `<option value="${esc(name)}"${name === active ? " selected" : ""}>${esc(name)}</option>`
    ).join("");
  } else {
    select.innerHTML = `<option value="">— none —</option>`;
    select.disabled = true;
  }
}

// Last observed auth verdict (null = not probed yet) and the timer that
// turns off the transient blue "key accepted" dot.
let apikeyAuthOk = null;
let apikeyDotTimer;

async function refreshCollections() {
  const run = collectionsCoordinator.begin();
  const result = await api.getCollections();
  if (!collectionsCoordinator.canCommit(run)) return;
  // Collections are bearer-protected: without a valid API key the select
  // stays empty, so say why right under it.
  const hint = $("conn-collection-hint");
  const unauthorized = result.status === 401;
  // Topbar dot mirrors the auth state: red = key missing or rejected
  // (persistent); blue = key just accepted (flashes for 3s, then off).
  const apikeyBtn = $("apikey-btn");
  apikeyBtn.classList.toggle("is-bad", unauthorized);
  if (unauthorized) {
    clearTimeout(apikeyDotTimer);
    apikeyBtn.classList.remove("is-set");
  } else if (apikeyAuthOk !== true && api.getToken()) {
    clearTimeout(apikeyDotTimer);
    apikeyBtn.classList.add("is-set");
    apikeyDotTimer = setTimeout(() => apikeyBtn.classList.remove("is-set"), 3000);
  }
  apikeyAuthOk = !unauthorized;
  hint.hidden = !unauthorized;
  hint.textContent = !unauthorized
    ? ""
    : api.getToken()
      ? "API KEY REJECTED — OPEN API KEY AND REGISTER THE CORRECT KEY"
      : "API KEY MISSING — OPEN API KEY AND REGISTER A KEY";
  const names = result.ok && result.data && Array.isArray(result.data.collections)
    ? result.data.collections
    : [];
  state.collections = sortedCollectionNames(names);
  // After a transient failure state.active is "" while the stored preference
  // still holds the user's choice, so fall back to it: recovering auth must
  // return to the collection they had selected, not to the first one.
  const preferred = state.active || localStorage.getItem(ACTIVE_KEY) || "";
  const active = state.collections.length
    ? chooseActiveCollection(preferred, state.collections)
    : "";
  // Only the connection bar selects the collection; the per-section
  // collection_name fields are informational mirrors (disabled inputs).
  populateCollectionSelect($("conn-collection"), state.collections, active);
  setActiveCollection(active);
}

async function probeHealth() {
  const run = healthCoordinator.begin();
  setOperationalReadiness(
    false,
    "Checking Qdrant and Smart Embedder availability…",
  );
  const chip = $("conn-health");
  chip.classList.remove("ok", "bad");
  $("conn-health-status").textContent = "PROBING…";
  $("conn-health-detail").textContent = "";
  const result = await api.health();
  if (!healthCoordinator.canCommit(run)) return;
  applyHealthResult(result);
}

function applyHealthResult(result) {
  const previouslyReady = state.servicesReady;
  const chip = $("conn-health");
  chip.classList.remove("ok", "bad");
  const data = result.data;
  const isDict = data !== null && typeof data === "object" && !Array.isArray(data);
  const status = isDict ? String(data.status || "unknown").toLowerCase() : "";
  const qdrant = isDict ? String(data.qdrant || "unknown").toLowerCase() : "";
  const embedder = isDict ? String(data.embedder || "unknown").toLowerCase() : "";
  state.health = isDict ? data : null;
  const ok = result.ok
    && status === "healthy"
    && qdrant === "reachable"
    && embedder === "reachable";
  chip.classList.add(ok ? "ok" : "bad");
  if (isDict) {
    $("conn-health-status").textContent = ok ? "HEALTHY" : "UNHEALTHY";
    const profile = data.embedder_profile ? ` [${data.embedder_profile}]` : "";
    $("conn-health-detail").textContent =
      `qdrant ${qdrant} · embedder ${embedder}${profile} · ` +
      `${data.collections ?? "—"} collections · ${result.elapsed.toFixed(3)}s`;
  } else if (result.ok) {
    $("conn-health-status").textContent = "HEALTHY";
    $("conn-health-detail").textContent = `${result.elapsed.toFixed(3)}s`;
  } else {
    $("conn-health-status").textContent = "UNAVAILABLE";
    const bareStatus = /^HTTP \d{3}$/.test(result.error || "");
    $("conn-health-detail").textContent =
      bareStatus || !result.error ? "Health check failed" : result.error;
  }
  const unavailable = [
    qdrant !== "reachable" ? "Qdrant" : null,
    embedder !== "reachable" ? "Smart Embedder" : null,
  ].filter(Boolean);
  setOperationalReadiness(
    ok,
    ok
      ? ""
      : unavailable.length
        ? `${unavailable.join(" and ")} unreachable.`
        : "Service health could not be verified.",
  );
  if (ok && !previouslyReady) {
    void refreshCollections();
  }
}

function initConnection() {
  $("conn-retest").addEventListener("click", probeHealth);
  $("conn-collection").addEventListener("change", (event) => {
    setActiveCollection(event.target.value);
  });
  state.healthTimer = setInterval(probeHealth, 30000);
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

const TABS = ["query", "collections", "points", "monitoring"];

function activateTab(name) {
  const tab = TABS.includes(name) ? name : "query";
  for (const t of TABS) {
    const button = $(`tab-${t}`);
    button.setAttribute("aria-selected", String(t === tab));
    button.tabIndex = t === tab ? 0 : -1;
    $(`panel-${t}`).hidden = t !== tab;
  }
  if (history.state?.tab !== tab) {
    history.replaceState({ tab }, "", `#${tab}`);
  }
  if (tab === "monitoring") startMonitor();
  else stopMonitor();
}

// ARIA tabs pattern: roving tabindex plus Arrow/Home/End keyboard navigation.
// Arrow moves focus and activates the tab (automatic activation).
function initTablistKeyboard(tablist) {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  const syncTabindex = () => {
    for (const t of tabs) {
      t.tabIndex = t.getAttribute("aria-selected") === "true" ? 0 : -1;
    }
  };
  tablist.addEventListener("keydown", (event) => {
    const index = tabs.indexOf(document.activeElement);
    if (index === -1) return;
    let target;
    if (event.key === "ArrowRight") target = tabs[(index + 1) % tabs.length];
    else if (event.key === "ArrowLeft") target = tabs[(index - 1 + tabs.length) % tabs.length];
    else if (event.key === "Home") target = tabs[0];
    else if (event.key === "End") target = tabs[tabs.length - 1];
    else return;
    event.preventDefault();
    target.click();
    target.focus();
    syncTabindex();
  });
  // Child click handlers update aria-selected before this bubbles up.
  tablist.addEventListener("click", syncTabindex);
  syncTabindex();
}

function initTabs() {
  for (const t of TABS) {
    $(`tab-${t}`).addEventListener("click", () => activateTab(t));
  }
  window.addEventListener("hashchange", () =>
    activateTab(window.location.hash.replace("#", "")));
  activateTab(window.location.hash.replace("#", "") || "query");
  initTablistKeyboard(document.querySelector(".tabs"));
}

// ── API key ─────────────────────────────────────────────────────────────────

// The key is scoped to this tab's session and attached by api.js as the bearer
// token on every request: it must be the token half of one entry in
// MIDDLEWARE_API_TOKENS, otherwise protected endpoints answer 401. It survives
// reloads but not closing the tab/browser and is not shared with another tab.
const APIKEY_STORAGE_KEY = "qhm_api_key";

function loadApiKey() {
  return sessionStorage.getItem(APIKEY_STORAGE_KEY) || "";
}

function saveApiKey(value) {
  invalidateOperations();
  markQueryChanged();
  collectionsCoordinator.changed();
  healthCoordinator.changed();
  state.reportLoadId += 1;
  state.reindexLoadId += 1;
  state.collectionFieldsLoadId += 1;
  if (value) sessionStorage.setItem(APIKEY_STORAGE_KEY, value);
  else sessionStorage.removeItem(APIKEY_STORAGE_KEY);
}

function initApiKey() {
  const btn = $("apikey-btn");
  const popover = $("apikey-popover");
  const input = $("apikey-input");
  const clearBtn = $("apikey-clear");
  const status = $("apikey-status");
  let closeTimer;

  const setStatus = (text, kind = "") => {
    status.textContent = text;
    status.className = `caption apikey-status${kind ? ` ${kind}` : ""}`;
  };

  // Reflects storage into the UI: clear button, placeholder. The stored key
  // is never echoed back; the topbar dot is owned by refreshCollections
  // (red = rejected/missing, transient blue = accepted).
  const syncState = () => {
    const hasKey = Boolean(loadApiKey());
    clearBtn.hidden = !hasKey;
    input.placeholder = hasKey ? "••••••••" : "type or paste the key";
    setStatus("");
  };

  function openPopover() {
    clearTimeout(closeTimer);
    popover.hidden = false;
    btn.setAttribute("aria-expanded", "true");
    input.value = "";
    syncState();
    input.focus();
  }

  // Focus returns to the trigger on scripted closes (Esc / after register);
  // an outside pointerdown keeps focus where the user clicked.
  function closePopover({ refocus = true } = {}) {
    if (popover.hidden) return;
    popover.hidden = true;
    btn.setAttribute("aria-expanded", "false");
    if (refocus) btn.focus();
  }

  btn.addEventListener("click", () =>
    (popover.hidden ? openPopover() : closePopover()));

  // A real <form> gives Enter-to-submit for free, on every keyboard.
  $("apikey-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const value = input.value.trim();
    if (!value) {
      setStatus("ENTER A KEY FIRST", "error");
      input.focus();
      return;
    }
    saveApiKey(value);
    input.value = "";
    syncState();
    setStatus("API KEY REGISTERED", "ok");
    closeTimer = setTimeout(() => closePopover(), 800);
    reloadProtectedData();
  });

  clearBtn.addEventListener("click", () => {
    saveApiKey("");
    syncState();
    setStatus("API KEY REMOVED", "ok");
    input.focus();
    reloadProtectedData();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !popover.hidden) closePopover();
  });

  document.addEventListener("pointerdown", (event) => {
    if (!popover.hidden
        && !popover.contains(event.target)
        && !btn.contains(event.target)) {
      closePopover({ refocus: false });
    }
  });

  syncState();
}

// ── Query ─────────────────────────────────────────────────────────────────

let scrollPagination = null;

function markQueryChanged() {
  queryCoordinator.changed();
  scrollPagination = null;
  $("ql-results").innerHTML = "";
}

function readQueryState() {
  const isRrf = document.querySelector('input[name="ql-fusion"]:checked').value === "rrf";
  // A disabled query box (incompatible collection) never contributes a hybrid
  // query, regardless of any stale text left in the field.
  const queryText = $("ql-query").disabled ? "" : $("ql-query").value;
  const rerankOn = Boolean(queryText.trim()) && $("ql-rerank").checked;
  const rrfkEn = $("ql-rrfk-en").checked;
  const weightsEn = $("ql-weights-en").checked;
  const topkEn = $("ql-topk-en").checked;
  const balance = parseFloat($("ql-balance").value);
  const weightScale = $("ql-weight-scale").valueAsNumber;
  return {
    query: queryText,
    collectionName: state.active,
    limit: $("ql-limit").valueAsNumber,
    candidateLimit: $("ql-cand").valueAsNumber,
    fusion: isRrf ? "rrf" : "dbsf",
    rrfK: rrfkEn && isRrf ? $("ql-rrfk").valueAsNumber : null,
    rrfBalance: weightsEn && isRrf ? balance : null,
    rrfWeightScale: weightsEn && isRrf ? weightScale : 1,
    rerank: rerankOn,
    rerankTopK: topkEn && rerankOn ? $("ql-topk").valueAsNumber : null,
    rerankNormalize: $("ql-normalize").checked,
    // score_threshold / offset only reach Qdrant on the hybrid path; order_by
    // only on the payload-only scroll path, so each is read for its own mode.
    scoreThreshold: $("ql-threshold-en").checked && queryText.trim()
      ? $("ql-threshold").valueAsNumber : null,
    offset: $("ql-offset-en").checked && queryText.trim()
      ? $("ql-offset").valueAsNumber : null,
    orderByEnabled: $("ql-order-en").checked && !queryText.trim(),
    orderByKey: $("ql-order-key").value,
    orderByDirection:
      document.querySelector('input[name="ql-order-dir"]:checked')?.value || "asc",
    orderByStartFrom: $("ql-order-start").value,
    filterTree: cloneFilterTree(state.filterTree),
    vectorNames: state.vectorNames,
    filterRows: [],
    payloadFields: [...document.querySelectorAll("#ql-payload-fields input:checked")]
      .map((box) => box.value),
  };
}

function updateQueryUI() {
  const embeddingGate = activeEmbeddingGate();
  // hybridActive tracks whether a query has been typed; it only governs which
  // hybrid controls are enabled here. Panel-wide lockdown (incompatible
  // collection or service outage) is owned separately by applyLockdowns().
  const hybridActive = Boolean($("ql-query").value.trim());
  const isRrf = document.querySelector('input[name="ql-fusion"]:checked').value === "rrf";
  const rerankOn = $("ql-rerank").checked;
  $("ql-limit").disabled = !hybridActive;
  $("ql-cand").disabled = !hybridActive;
  for (const radio of document.querySelectorAll('input[name="ql-fusion"]')) {
    radio.disabled = !hybridActive;
  }
  $("ql-rerank").disabled = !hybridActive;
  $("ql-rrfk-field").hidden = !isRrf;
  $("ql-weights-field").hidden = !isRrf;
  $("ql-rrfk-en").disabled = !hybridActive || !isRrf;
  if (!$("ql-rrfk-en").checked) $("ql-rrfk").value = "2";
  $("ql-rrfk").disabled =
    !hybridActive || !isRrf || !$("ql-rrfk-en").checked;
  $("ql-weights-en").disabled = !hybridActive || !isRrf;
  if (!$("ql-weights-en").checked) $("ql-balance").value = "0.5";
  if (!$("ql-weights-en").checked) $("ql-weight-scale").value = "1";
  $("ql-balance").disabled =
    !hybridActive || !isRrf || !$("ql-weights-en").checked;
  $("ql-weight-scale").disabled =
    !hybridActive || !isRrf || !$("ql-weights-en").checked;
  $("ql-topk-en").disabled = !hybridActive || !rerankOn;
  if (!$("ql-topk-en").checked) {
    const candidateLimit = $("ql-cand").valueAsNumber;
    $("ql-topk").value = Number.isInteger(candidateLimit) && candidateLimit > 0
      ? Math.min(
        queryConfig.queryDefaultRerankTopK,
        candidateLimit,
        queryConfig.queryMaxRerankTopK)
      : queryConfig.queryDefaultRerankTopK;
  }
  $("ql-topk").disabled =
    !hybridActive || !rerankOn || !$("ql-topk-en").checked;
  $("ql-normalize").disabled = !hybridActive || !rerankOn;
  // score_threshold and offset act on the fused hybrid result; order_by acts on
  // the payload-only scroll. Each set is therefore live in one mode only. What
  // each one does is documented once in the Parameter reference; here only the
  // reason a control is unavailable is surfaced, as a title (same idiom as
  // setDisabledReason on the Points buttons).
  $("ql-threshold-en").disabled = !hybridActive;
  $("ql-threshold").disabled = !hybridActive || !$("ql-threshold-en").checked;
  // score_threshold is compared against the FUSED score, whose scale depends
  // on which fusion ran: RRF scores are 1/(k+rank), always in (0, 1]; DBSF
  // normalizes each retriever to a z-score (mean/3-sigma), so values land in
  // roughly [-3, 3]. Advertise that range via min/max (spinner step bounds +
  // native :invalid styling) without rewriting whatever the user typed here —
  // deliberately out-of-range values (e.g. an unreachable cut to assert an
  // empty result) still reach the request untouched. Carrying a value across a
  // fusion switch is the one case that is reset, at the radio's own listener.
  const thresholdInput = $("ql-threshold");
  const [thresholdMin, thresholdMax] = isRrf ? [0, 1] : [-3, 3];
  thresholdInput.min = String(thresholdMin);
  thresholdInput.max = String(thresholdMax);
  $("ql-offset-en").disabled = !hybridActive;
  $("ql-offset").disabled = !hybridActive || !$("ql-offset-en").checked;
  // Deepest well-founded page for the current settings: candidate_limit has to
  // cover offset+window, so the reachable offset moves with candidate_limit.
  const rerankWindow = rerankOn
    ? Math.max($("ql-limit").valueAsNumber || 0, $("ql-topk").valueAsNumber || 0)
    : $("ql-limit").valueAsNumber || 0;
  $("ql-offset").max = String(
    Math.max(0, ($("ql-cand").valueAsNumber || 0) - rerankWindow));
  setDisabledReason(
    ["ql-threshold-en", "ql-offset-en"],
    hybridActive ? "" : "Hybrid search only — type a query to cut or page fused hits",
  );
  $("ql-order-en").disabled = hybridActive;
  const orderOn = !hybridActive && $("ql-order-en").checked;
  $("ql-order-key").disabled = !orderOn;
  $("ql-order-start").disabled = !orderOn;
  for (const radio of document.querySelectorAll('input[name="ql-order-dir"]')) {
    radio.disabled = !orderOn;
  }
  setDisabledReason(
    ["ql-order-en"],
    hybridActive ? "Payload-only mode only — clear the query to order a filtered scroll" : "",
  );
  // Caption visibility (hidden while opsLocked) is handled by applyLockdowns();
  // this just keeps the text in sync with the current hybrid/payload-only mode.
  $("ql-query-mode").textContent = hybridActive
    ? "HYBRID SEARCH IS ACTIVE - DENSE + SPARSE RETRIEVAL AND OPTIONAL RERANKING"
    : "HYBRID SEARCH IS DISABLED - PAYLOAD-ONLY MODE FILTERING";

  $("ql-embedding-guard").innerHTML =
    embeddingGate.status === "loading"
      ? alertHtml("info", "Checking collection compatibility with the active embedder…")
      : "";

  const balance = parseFloat($("ql-balance").value);
  const weightScale = $("ql-weight-scale").valueAsNumber;
  if (Number.isFinite(weightScale)
      && weightScale >= RRF_WEIGHT_SCALE_MIN
      && weightScale <= RRF_WEIGHT_SCALE_MAX) {
    const [denseWeight, sparseWeight] =
      rrfWeightsFromBalance(balance, weightScale);
    $("ql-balance-caption").textContent =
      `Qdrant weights [${denseWeight.toFixed(2)}, ${sparseWeight.toFixed(2)}]`;
  } else {
    $("ql-balance-caption").textContent =
      `Weight scale must be ${RRF_WEIGHT_SCALE_MIN}–${RRF_WEIGHT_SCALE_MAX}.`;
  }

  // min_should only becomes meaningful with 2+ root SHOULD conditions.
  // Keep the implicit value at 1 and hide the control below that threshold.
  const shouldCount = state.filterTree.children
    .filter((node) => node.clause === "should").length;
  const minShouldCap = Math.max(1, shouldCount);
  const minShouldField = $("ql-minshould-field");
  const minShouldInput = $("ql-minshould");
  minShouldInput.max = String(minShouldCap);
  if ((Math.trunc(minShouldInput.valueAsNumber) || 1) > minShouldCap) {
    minShouldInput.value = String(minShouldCap);
  }
  minShouldInput.disabled = shouldCount < 2;
  minShouldField.hidden = shouldCount < 2;
  if (shouldCount < 2) minShouldInput.value = "1";
  state.filterTree.minShould = shouldCount < 2 ? 1 : minShouldInput.valueAsNumber;
  minShouldInput.title = shouldCount < 2
    ? "Add 2+ SHOULD rows to require a minimum number of matches (Qdrant min_should)."
    : `Require at least this many of the ${shouldCount} SHOULD rows to match (max ${shouldCount}).`;

  const current = readQueryState();
  const errors = validateQueryState(current, state.fieldSchemas, queryConfig);
  const payload = errors.length
    ? { validation_errors: errors }
    : (current.query.trim()
      ? buildRequestPayload(current, state.fieldSchemas)
      : buildScrollPayload(current, state.fieldSchemas));
  $("ql-request-json").textContent = JSON.stringify(payload, null, 2);
  const noFilter = !current.query.trim() && current.filterTree.children.length === 0;
  const dbsfWarning = current.query.trim() && current.fusion === "dbsf"
    && current.candidateLimit === 1
    ? alertHtml(
      "info",
      "With one candidate, each DBSF branch normalizes it to 0.5, making fusion tie-prone; increase candidate_limit.")
    : "";
  $("ql-errors").innerHTML = errors.map((err) => alertHtml("error", err)).join("") +
    dbsfWarning +
    (state.collectionFieldsLoading
      ? alertHtml("info", "Loading collection payload fields…") : "") +
    (noFilter ? alertHtml("info", "Empty query and no filters: the first page of all points will be returned, ordered by ID.") : "");
  for (const [id, key] of Object.entries({ "ql-limit": "limit ", "ql-cand": "candidate_limit", "ql-rrfk": "rrf_k", "ql-topk": "rerank_top_k", "ql-offset": "offset ", "ql-order-key": "order_by", "ql-order-start": "start_from" })) {
    associateFieldError($(id), "ql-errors", errors.some((message) => message.startsWith(key)));
  }
  for (const row of $("ql-filter-rows").querySelectorAll(".filter-row")) {
    const entry = findFilterEntry(row.dataset.nodeId);
    if (!entry) continue;
    const [, error] = buildFilterCondition(entry.node, rowSchema(entry.node));
    const invalidVector = entry.node.op === "has_vector" && !state.vectorNames.includes(entry.node.rawValue);
    for (const control of row.querySelectorAll('[data-role^="value"]')) {
      associateFieldError(control, "ql-errors", Boolean(error || invalidVector));
    }
  }
  $("ql-run").disabled = errors.length > 0
    || queryCoordinator.isBusy()
    || state.collectionFieldsLoading
    || !state.servicesReady
    || (hybridActive && embeddingGate.blocked);
  if (hybridActive && embeddingGate.blocked) {
    $("ql-run").title = embeddingGate.message;
  } else {
    $("ql-run").removeAttribute("title");
  }
  $("ql-run").textContent = state.collectionFieldsLoading
    ? "Loading fields…"
    : (hybridActive && embeddingGate.blocked
      ? "Hybrid query unavailable"
    : (queryCoordinator.isBusy()
      ? "Running…"
      : (hybridActive ? "Run hybrid query" : "Retrieve points")));
  operationReasons({ "ql-run": errors[0] || embeddingGate.message || "Loading collection payload fields." });
}

// Sampled payload keys drive the with_payload checkboxes; the filter FIELD
// dropdowns offer indexed fields. Unindexed filtering through the API depends
// on the target collection's strict-mode policy; do not assume it permits scans.

function filterFieldOptions(current) {
  const options = [POINT_ID_FIELD, VECTOR_FIELD, ...state.indexedFields];
  if (current && !options.includes(current)) options.push(current);
  return options;
}

let lastPayloadKeysSig = null;

// Qdrant orders a scroll only on a RANGE index, so the key list is narrower than
// the indexed set: a keyword or bool field is indexed yet not sortable, and
// offering it would only earn a raw "No range index for `order_by` key" from the
// server.
function renderOrderByOptions() {
  const select = $("ql-order-key");
  const previous = select.value;
  const fields = orderableFields(state.fieldSchemas);
  select.innerHTML = fields.length
    ? fields.map((field) =>
      `<option value="${esc(field)}"${field === previous ? " selected" : ""}>${esc(field)}</option>`
    ).join("")
    : `<option value="">no orderable field</option>`;
  if (!fields.length) {
    $("ql-order-en").checked = false;
  }
}

function renderFacetKeyOptions() {
  const select = $("cnt-facet-key");
  const previous = select.value;
  const fields = facetableFields(state.fieldSchemas);
  select.disabled = !fields.length;
  select.innerHTML = fields.length
    ? fields.map((field) =>
      `<option value="${esc(field)}"${field === previous ? " selected" : ""}>${esc(field)}</option>`
    ).join("")
    : `<option value="">no facetable field</option>`;
}

function renderPayloadFieldOptions() {
  // A new collection (or a changed key set) starts with everything selected;
  // within the same key set the user's deselections are preserved.
  const sig = state.payloadKeys.join(" ");
  const freshKeys = sig !== lastPayloadKeysSig;
  lastPayloadKeysSig = sig;
  const checked = new Set(
    [...document.querySelectorAll("#ql-payload-fields input:checked")].map((box) => box.value));
  const rerankOn = !$("ql-query").disabled
    && Boolean($("ql-query").value.trim()) && $("ql-rerank").checked;
  const fields = [...state.payloadKeys];
  if (rerankOn && !fields.includes("text")) fields.push("text");
  // No sampled payload fields (and no rerank-forced "text"): tell the user
  // plainly instead of showing an empty "with_payload" control.
  $("ql-payload-legend").textContent =
    fields.length ? "with_payload" : "no payload fields in this collection";
  $("ql-payload-fields").innerHTML = fields.map((field) => {
    // The reranker reads payload text: while rerank is active the field must be
    // selected even when collection sampling did not discover it.
    const locked = rerankOn && field === "text";
    const isChecked = locked || freshKeys || checked.has(field);
    return `<label class="check"><input type="checkbox" value="${esc(field)}"` +
      `${isChecked ? " checked" : ""}${locked ? " disabled" : ""} /> ${esc(field)}</label>`;
  }).join("");
}

// Fetch sampled payload keys and payload indexes together, then refresh every
// consumer: with_payload checkboxes, filter FIELD dropdowns (indexed fields
// only), integer-coercion set, and the Payload Indexes table.
async function loadCollectionFields() {
  const target = state.active;
  const loadId = ++state.collectionFieldsLoadId;
  state.collectionFieldsLoading = Boolean(target);
  $("ql-filter-nested-add").disabled = true;
  updateQueryUI();
  let keys = [];
  let indexes = [];
  let indexesResult = null;
  if (target) {
    const [indexesRes, keysRes] = await Promise.all([
      api.getPayloadIndexes(target),
      api.getPayloadKeys(target),
    ]);
    // The user may have switched collection while the requests were in
    // flight; rendering would show the wrong collection's fields.
    if (target !== state.active || loadId !== state.collectionFieldsLoadId) return;
    indexesResult = indexesRes;
    if (keysRes.ok && keysRes.data && Array.isArray(keysRes.data.keys)) {
      keys = keysRes.data.keys;
    }
    if (indexesRes.ok && indexesRes.data && Array.isArray(indexesRes.data.indexes)) {
      indexes = indexesRes.data.indexes;
    }
  }
  state.payloadKeys = keys.length ? keys : [];
  // The index schema drives the operations each filter row offers and how
  // values are typed (integers stay ints or MatchValue never hits the index).
  state.fieldSchemas = {};
  state.phraseFields = new Set();
  for (const item of indexes) {
    if (item && typeof item === "object" && item.field_name) {
      state.fieldSchemas[item.field_name] = item.field_schema;
      if (item.phrase_matching) state.phraseFields.add(item.field_name);
    }
  }
  state.indexedFields = Object.keys(state.fieldSchemas);
  state.collectionFieldsLoading = false;

  // Point-id filters need no payload index, so the button stays enabled;
  // the title still nudges toward indexing when payload filtering is wanted.
  $("ql-filter-add").disabled = false;
  $("ql-filter-group-add").disabled = false;
  $("ql-filter-nested-add").disabled = nestedRootsForFields(state.indexedFields).length === 0;
  $("ql-filter-add").title = state.indexedFields.length
    ? ""
    : "Only point ID and vector presence are filterable — index payload fields in Collections → Payload Indexes.";

  renderPayloadFieldOptions();
  renderOrderByOptions();
  renderFacetKeyOptions();
  renderFilterTree();
  updateQueryUI();
  updatePointsUI();
  // An ingest preflight already on screen has to pick up the index list the
  // moment it lands — it decides which mapping rows are type-locked.
  refreshIngestIndexLocks();

  if (!target) {
    $("pi-table").innerHTML = alertHtml("info", "Select a collection to manage payload indexes.");
  } else if (indexesResult.ok && indexesResult.data !== null
             && typeof indexesResult.data === "object") {
    $("pi-table").innerHTML = indexesTable(indexes, keys);
  } else {
    $("pi-table").innerHTML =
      apiErrorHtml(indexesResult, "Payload indexes unavailable");
  }
}

function rowSchema(row) {
  return schemaForField(row.fieldName, state.fieldSchemas);
}

// Keep the row's operation valid for its field's schema (e.g. after the user
// switches from an integer field to a text field).
function resetRowValues(row) {
  row.rawValue = "";
  row.rawValue2 = "";
  row.rawValue3 = "";
  row.rawValue4 = "";
  row.lowerOp = "gte";
  row.upperOp = "lte";
}

function normalizeRowOp(row) {
  const ops = filterOpsFor(rowSchema(row));
  if (!ops.includes(row.op)) row.op = ops[0] || "eq";
  // 'exact phrase' needs a text index built with phrase_matching.
  if (row.op === "phrase" && !state.phraseFields.has(row.fieldName?.trim())) {
    row.op = ops[0] || "eq";
  }
}

function filterValueCell(row) {
  const kind = filterValueKind(row.op);
  if (kind === "none") {
    return `<span class="filter-novalue" aria-hidden="true">—</span>`;
  }
  if (kind === "bool") {
    const hasBoolValue = row.rawValue === "true" || row.rawValue === "false";
    return `<select data-role="value" aria-label="filter value">
      <option value="" disabled${hasBoolValue ? "" : " selected"}>select true/false</option>
      <option value="true"${row.rawValue === "true" ? " selected" : ""}>true</option>
      <option value="false"${row.rawValue === "false" ? " selected" : ""}>false</option>
    </select>`;
  }
  if (kind === "vector") {
    const options = state.vectorNames.map((name) =>
      `<option value="${esc(name)}"${row.rawValue === name ? " selected" : ""}>${esc(name)}</option>`
    ).join("");
    const missing = state.vectorNames.includes(row.rawValue) ? ""
      : `<option value="${esc(row.rawValue || "")}" selected disabled>Select a current vector</option>`;
    return `<select data-role="value" aria-label="vector name">${missing}${options}</select>`;
  }
  if (kind === "range") {
    const count = row.op === "values_count";
    return `<select class="filter-comparator" data-role="lower-op" aria-label="lower comparator">
        <option value="gte"${row.lowerOp !== "gt" ? " selected" : ""}>≥</option>
        <option value="gt"${row.lowerOp === "gt" ? " selected" : ""}>&gt;</option>
      </select>
      <input type="text" data-role="value" aria-label="lower bound"
             placeholder="${count ? "min count" : "lower"}" value="${esc(row.rawValue ?? "")}" />
      <select class="filter-comparator" data-role="upper-op" aria-label="upper comparator">
        <option value="lte"${row.upperOp !== "lt" ? " selected" : ""}>≤</option>
        <option value="lt"${row.upperOp === "lt" ? " selected" : ""}>&lt;</option>
      </select>
      <input type="text" data-role="value2" aria-label="upper bound"
             placeholder="${count ? "max count" : "upper"}" value="${esc(row.rawValue2 ?? "")}" />`;
  }
  if (kind === "pair") {
    const [minPh, maxPh] = row.op === "values_count"
      ? ["min count", "max count"]
      : ["min", "max"];
    return `<input type="text" data-role="value" aria-label="minimum"
              placeholder="${minPh}" value="${esc(row.rawValue ?? "")}" />
      <input type="text" data-role="value2" aria-label="maximum"
              placeholder="${maxPh}" value="${esc(row.rawValue2 ?? "")}" />`;
  }
  if (kind === "geo") {
    return `<input type="text" data-role="value" aria-label="longitude"
              placeholder="lon" value="${esc(row.rawValue ?? "")}" />
      <input type="text" data-role="value2" aria-label="latitude"
              placeholder="lat" value="${esc(row.rawValue2 ?? "")}" />
      <input type="text" data-role="value3" aria-label="radius in meters"
              placeholder="radius m" value="${esc(row.rawValue3 ?? "")}" />`;
  }
  if (kind === "geo_box") {
    return `<input type="text" data-role="value" aria-label="top-left longitude"
              placeholder="TL lon" value="${esc(row.rawValue ?? "")}" />
      <input type="text" data-role="value2" aria-label="top-left latitude"
              placeholder="TL lat" value="${esc(row.rawValue2 ?? "")}" />
      <input type="text" data-role="value3" aria-label="bottom-right longitude"
              placeholder="BR lon" value="${esc(row.rawValue3 ?? "")}" />
      <input type="text" data-role="value4" aria-label="bottom-right latitude"
              placeholder="BR lat" value="${esc(row.rawValue4 ?? "")}" />`;
  }
  const placeholder = row.op === "any" || row.op === "except"
    ? 'value1, value2 or ["value, one"]'
    : row.op === "text" || row.op === "text_any"
      ? "words …"
      : row.op === "phrase"
        ? "exact phrase …"
        : row.op === "geo_polygon"
          ? "lon lat; lon lat; … | lon lat; lon lat; …"
          : row.op === "id_any"
            ? "id1, id2, …"
            : "filter value …";
  return `<input type="text" data-role="value" aria-label="filter value"
            placeholder="${placeholder}" value="${esc(row.rawValue ?? "")}" />`;
}

function clauseOptions(selected) {
  return FILTER_GROUPS.map((clause) =>
    `<option value="${clause}"${clause === selected ? " selected" : ""}>` +
    `${esc(FILTER_GROUP_LABELS[clause])}</option>`
  ).join("");
}

function fieldsForContext(nestedRoot, current = "") {
  const fields = nestedRoot
    ? state.indexedFields.filter((field) => field.startsWith(`${nestedRoot}[].`))
    : filterFieldOptions(current);
  if (current && !fields.includes(current)) fields.push(current);
  return fields;
}

function rootsForContext(nestedRoot = null) {
  const roots = nestedRootsForFields(state.indexedFields);
  return nestedRoot
    ? roots.filter((root) => root.startsWith(`${nestedRoot}[].`))
    : roots;
}

function newCondition(nestedRoot = null) {
  const firstField = nestedRoot
    ? fieldsForContext(nestedRoot)[0] || ""
    : state.indexedFields[0] || POINT_ID_FIELD;
  const row = {
    kind: "condition",
    id: state.filterCtr++,
    clause: "must",
    fieldName: firstField,
    op: filterOpsFor(schemaForField(firstField, state.fieldSchemas))[0] || "eq",
    rawValue: firstField === VECTOR_FIELD ? state.vectorNames[0] : "",
    rawValue2: "",
    rawValue3: "",
    rawValue4: "",
    lowerOp: "gte",
    upperOp: "lte",
  };
  return row;
}

function newLogicalGroup(nestedRoot = null) {
  return {
    kind: "group",
    id: state.filterCtr++,
    clause: "must",
    label: "Logical group",
    minShould: 1,
    children: [newCondition(nestedRoot)],
  };
}

function newNestedGroup(parentRoot = null) {
  const root = rootsForContext(parentRoot)[0] || "";
  return {
    kind: "nested",
    id: state.filterCtr++,
    clause: "must",
    label: "Same-element group",
    root,
    minShould: 1,
    children: root ? [newCondition(root)] : [],
  };
}

function findFilterEntry(id, group = state.filterTree, parent = null, nestedRoot = null) {
  if (String(group.id) === String(id)) return { node: group, parent, nestedRoot };
  const childContext = group.kind === "nested" ? group.root : nestedRoot;
  for (const child of group.children || []) {
    if (String(child.id) === String(id)) {
      return { node: child, parent: group, nestedRoot: childContext };
    }
    if (child.kind === "group" || child.kind === "nested") {
      const found = findFilterEntry(id, child, group, childContext);
      if (found) return found;
    }
  }
  return null;
}

function renderConditionRow(row, nestedRoot) {
  normalizeRowOp(row);
  const fieldOpts = fieldsForContext(nestedRoot, row.fieldName).map((field) =>
    `<option value="${esc(field)}"${field === row.fieldName ? " selected" : ""}>${esc(field)}</option>`
  ).join("");
  const opOpts = filterOpsFor(rowSchema(row)).map((op) => {
    const blocked = op === "phrase" && !state.phraseFields.has(row.fieldName);
    return `<option value="${esc(op)}"${op === row.op ? " selected" : ""}` +
      `${blocked ? ` disabled title="index lacks phrase_matching — recreate it in Collections → Payload Indexes"` : ""}>` +
      `${esc(FILTER_OP_LABELS[op] || op)}</option>`;
  }).join("");
  return `<div class="filter-row" data-node-id="${row.id}">
    <select data-role="clause" aria-label="filter clause">${clauseOptions(row.clause)}</select>
    <select data-role="field" aria-label="filter field">${fieldOpts}</select>
    <select data-role="op" aria-label="filter operation">${opOpts}</select>
    <span class="filter-value">${filterValueCell(row)}</span>
    <button type="button" class="btn-icon" data-action="remove" title="Remove this filter row">✕</button>
  </div>`;
}

function renderGroupCard(group, parentRoot) {
  const nested = group.kind === "nested";
  const contextRoot = nested ? group.root : parentRoot;
  const shouldCount = group.children.filter((child) => child.clause === "should").length;
  if (shouldCount < 2) group.minShould = 1;
  const rootOptions = rootsForContext(parentRoot).map((root) =>
    `<option value="${esc(root)}"${root === group.root ? " selected" : ""}>${esc(root)}[]</option>`
  ).join("");
  const children = group.children.map((child) =>
    child.kind === "condition"
      ? renderConditionRow(child, contextRoot)
      : renderGroupCard(child, contextRoot)
  ).join("");
  const nestedButtonDisabled = rootsForContext(contextRoot).length ? "" : " disabled";
  const minShouldControl = shouldCount >= 2
    ? `<label class="filter-min-should">min should matches
        <input type="number" data-role="group-min" min="1" max="${shouldCount}"
               step="1" value="${esc(group.minShould)}" />
      </label>`
    : "";
  return `<section class="filter-group-card${nested ? " is-nested" : ""}` +
    `${shouldCount < 2 ? " no-min-should" : ""}" data-node-id="${group.id}">
    <div class="filter-group-toolbar">
      <select data-role="clause" aria-label="group clause">${clauseOptions(group.clause)}</select>
      <span class="filter-group-title">${nested ? "SAME ELEMENT IN" : "LOGICAL GROUP"}</span>
      ${nested
        ? `<select data-role="nested-root" aria-label="array field">${rootOptions}</select>`
        : ""}
      ${minShouldControl}
      <button type="button" class="btn-icon" data-action="remove" title="Remove this group">✕</button>
    </div>
    <div class="filter-group-body">${children}</div>
    <div class="filter-group-actions">
      <button type="button" class="btn btn-compact" data-action="add-condition">+ Condition</button>
      <button type="button" class="btn btn-compact" data-action="add-group">+ Logical group</button>
      <button type="button" class="btn btn-compact" data-action="add-nested"${nestedButtonDisabled}>+ Same-element</button>
    </div>
  </section>`;
}

function renderFilterTree() {
  const container = $("ql-filter-rows");
  const focused = document.activeElement;
  const node = focused?.closest("[data-node-id]");
  const restore = container.contains(focused) && node ? {
    id: node.dataset.nodeId, role: focused.dataset.role, action: focused.dataset.action,
    start: focused.selectionStart, end: focused.selectionEnd,
    neighbors: [...container.querySelectorAll("[data-node-id]")].map((el) => el.dataset.nodeId),
  } : null;
  const restoreFocus = () => {
    if (!restore) return;
    const selector = restore.role ? `[data-role="${restore.role}"]` : `[data-action="${restore.action}"]`;
    const same = container.querySelector(`[data-node-id="${restore.id}"] ${selector}`);
    const position = restore.neighbors.indexOf(restore.id);
    const neighbors = [...restore.neighbors.slice(position + 1), ...restore.neighbors.slice(0, position).reverse()];
    const nearby = neighbors.map((id) => container.querySelector(`[data-node-id="${id}"] [data-role="value"], [data-node-id="${id}"] [data-action="add-condition"]`)).find(Boolean);
    const next = same || nearby || $("ql-filter-add");
    next.focus({ preventScroll: true });
    if (next === same && restore.start !== null && typeof next.setSelectionRange === "function") {
      next.setSelectionRange(restore.start, restore.end);
    }
  };
  if (!state.filterTree.children.length) {
    container.innerHTML = "";
    restoreFocus();
    return;
  }
  const head = `<div class="filter-head" aria-hidden="true">` +
    `<span>GROUP</span><span>FIELD</span><span>OPERATION</span><span>VALUE</span><span></span></div>`;
  container.innerHTML = head + state.filterTree.children.map((node) =>
    node.kind === "condition"
      ? renderConditionRow(node, null)
      : renderGroupCard(node, null)
  ).join("");
  restoreFocus();
}

function applyQueryState(snapshot) {
  $("ql-query").value = snapshot.query;
  $("ql-limit").value = snapshot.limit;
  $("ql-cand").value = snapshot.candidateLimit;
  document.querySelector(`input[name="ql-fusion"][value="${snapshot.fusion}"]`).checked = true;
  $("ql-rrfk-en").checked = snapshot.rrfK !== null;
  $("ql-rrfk").value = snapshot.rrfK !== null ? snapshot.rrfK : 2;
  const weightsEn = snapshot.rrfBalance !== null;
  $("ql-weights-en").checked = weightsEn;
  $("ql-balance").value = snapshot.rrfBalance !== null ? snapshot.rrfBalance : 0.5;
  $("ql-weight-scale").value = snapshot.rrfWeightScale ?? 1;
  $("ql-rerank").checked = snapshot.rerank;
  $("ql-topk-en").checked = snapshot.rerankTopK !== null;
  $("ql-topk").value = snapshot.rerankTopK !== null
    ? snapshot.rerankTopK : queryConfig.queryDefaultRerankTopK;
  $("ql-normalize").checked = snapshot.rerankNormalize;
  $("ql-threshold-en").checked = snapshot.scoreThreshold !== null
    && snapshot.scoreThreshold !== undefined;
  if (snapshot.scoreThreshold !== null && snapshot.scoreThreshold !== undefined) {
    $("ql-threshold").value = snapshot.scoreThreshold;
  }
  $("ql-offset-en").checked = snapshot.offset !== null && snapshot.offset !== undefined;
  if (snapshot.offset !== null && snapshot.offset !== undefined) {
    $("ql-offset").value = snapshot.offset;
  }
  $("ql-order-en").checked = Boolean(snapshot.orderByEnabled);
  if (snapshot.orderByKey) $("ql-order-key").value = snapshot.orderByKey;
  const orderDir = document.querySelector(
    `input[name="ql-order-dir"][value="${snapshot.orderByDirection || "asc"}"]`);
  if (orderDir) orderDir.checked = true;
  $("ql-order-start").value = snapshot.orderByStartFrom || "";
  // Payload checkboxes are governed by the collection's sampled keys
  // (renderPayloadFieldOptions), not by the snapshot.
  const legacyRows = Array.isArray(snapshot.filterRows) ? snapshot.filterRows : [];
  state.filterTree = snapshot.filterTree
    ? cloneFilterTree(snapshot.filterTree)
    : {
        ...emptyFilterTree(),
        minShould: snapshot.minShould ?? 1,
        children: legacyRows.map((row) => ({
          ...row,
          kind: "condition",
          id: state.filterCtr++,
          clause: FILTER_GROUPS.includes(row.group) ? row.group : "must",
          lowerOp: row.lowerOp || "gte",
          upperOp: row.upperOp || "lte",
        })),
      };
  let maxId = state.filterCtr;
  const ensureIds = (group) => {
    for (const child of group.children || []) {
      if (!Number.isInteger(child.id)) child.id = maxId++;
      else maxId = Math.max(maxId, child.id + 1);
      if (child.kind === "group" || child.kind === "nested") ensureIds(child);
    }
  };
  ensureIds(state.filterTree);
  state.filterCtr = maxId;
  $("ql-minshould").value = state.filterTree.minShould ?? 1;
  renderFilterTree();
  renderPayloadFieldOptions();
  updateQueryUI();
}

function renderResultCard(rank, hit, scored = true) {
  const payload = hit.payload || {};
  const chips = COMMON_PAYLOAD_METADATA
    .filter((key) => payload[key] !== null && payload[key] !== undefined)
    .map((key) => `<span class="chip">${esc(key)}: ${esc(payload[key])}</span>`)
    .join("");
  const text = typeof payload.text === "string" && payload.text.trim()
    ? `<div class="result-text">${esc(payload.text.trim())}</div>`
    : "";
  const scoreGroup = (label, value) =>
    `<div class="score-group"><div class="score-label">${label}</div>` +
    `<div class="score-bar-wrap"><div class="score-track">` +
    `<div class="score-fill" style="width:${scoreToPct(value).toFixed(1)}%"></div></div>` +
    `<span class="score-val">${esc(formatScore(value))}</span></div></div>`;
  return `<div class="result-card">
      <div class="result-rank">#${rank}</div>
      <div class="result-body">
        <code class="result-id">${esc(hit.id ?? "unknown")}</code>
        ${scored ? `<div class="score-row">${scoreGroup("Vector Score", hit.score)}${scoreGroup("Rerank Score", hit.rerank_score)}</div>` : ""}
        <div class="chips">${chips}</div>
        ${text}
      </div>
    </div>
    <details class="payload-details"><summary>Payload JSON</summary>${jsonBlock(payload)}</details>`;
}

// Walk every scroll page for a payload-only query, starting from the already
// fetched first page. Returns {hits, complete}: `complete` is false when a page
// request failed (partial data) or PAGE_CAP was hit, so callers can flag the
// export instead of claiming success. Returns null if the active collection
// changed mid-walk (abort the download). The Run button is disabled while busy,
// so no newer run can supersede this one; only a collection switch can
// invalidate it. PAGE_CAP bounds a server that never returns a null offset.
async function collectAllScrollHits(basePayload, firstData, target, run) {
  const PAGE_CAP = 100000;
  const hits = [...(firstData.results || [])];
  let offset = firstData.next_page_offset ?? null;
  let pages = 0;
  while (offset !== null && offset !== undefined) {
    if (!queryCoordinator.canCommit(run) || target !== state.active) return null;
    if (pages >= PAGE_CAP) return { hits, complete: false };
    const res = await api.scrollPoints({ ...basePayload, include_total: false, offset });
    if (!queryCoordinator.canCommit(run)) return null;
    if (!res.ok || !res.data || typeof res.data !== "object") {
      return { hits, complete: false };
    }
    hits.push(...(res.data.results || []));
    offset = res.data.next_page_offset ?? null;
    pages += 1;
  }
  return { hits, complete: true };
}

// Convert a query/scroll response into the RAPTOR chunks-file shape
// ({"chunks": [{...payload, point_id}], "metadata": {…}}) and save it, so a
// result set round-trips back into ingestion. Each hit's payload already holds
// `text` (plus any extra keys such as token_count); spreading it before
// `point_id` mirrors the reference file's key order. A trailing `metadata`
// block stamps provenance — the exact request JSON that produced them, plus
// embedder models/profile.
function downloadResultsAsChunks(result, collectionName, request, modeOverride = null) {
  const data = result && result.data;
  const hits = data && Array.isArray(data.results) ? data.results : [];
  if (!hits.length) return;  // nothing to save — skip the empty file
  // The stored point id goes last under `point_id` — not `id`, which the
  // payload now carries itself since ingest stopped stripping a chunk's own
  // `id` (CHUNK_EXCLUDE_FIELDS in points.py); spreading the payload first and
  // then writing `id` would destroy the chunk ordinal. Always present: a hit
  // without one would otherwise drop the key entirely under JSON.stringify.
  const chunks = hits.map((hit) => ({ ...(hit.payload || {}), point_id: hit.id ?? null }));

  const tokenCounts = chunks
    .map((c) => (typeof c.token_count === "number" ? c.token_count : null))
    .filter((n) => n !== null);
  const health = state.health || {};
  const models = health.embedder_models || {};
  const isScroll = !(data && Object.hasOwn(data, "fusion"));
  const now = new Date();
  const metadata = {
    n_chunks: chunks.length,
    ...(tokenCounts.length ? {
      avg_tokens: tokenCounts.reduce((a, b) => a + b, 0) / tokenCounts.length,
      max_tokens: Math.max(...tokenCounts),
      min_tokens: Math.min(...tokenCounts),
    } : {}),
    collection_name: collectionName || null,
    source: "hybrid-manager-query",
    generated_at: now.toISOString(),
    mode: modeOverride ?? (isScroll ? "payload-only" : "hybrid"),
    // The exact query payload as sent (query, filters, with_payload, fusion, …)
    // — one self-contained record, so its fields are not duplicated elsewhere.
    request: request ?? null,
    returned: data.returned ?? chunks.length,
    elapsed_s: typeof result.elapsed === "number" ? result.elapsed : null,
    embedder_profile: health.embedder_profile || null,
    embedder_models: {
      dense: models.dense || null,
      sparse: models.sparse || null,
      reranker: models.reranker || null,
    },
  };

  const blob = new Blob([JSON.stringify({ chunks, metadata }, null, 2)],
    { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  const suffix = modeOverride === "collection-export" ? "export" : "query_results";
  // Compact filesystem-safe timestamp (2026-07-24-14-30-05) so every download
  // is unique and recognizable at a glance.
  const stamp = now.toISOString().replace(/\.\d+Z$/, "").replace(/[:T]/g, "-");
  anchor.download = `${collectionName || "collection"}_${suffix}_${stamp}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function renderQueryResponse(result) {
  const container = $("ql-results");
  const data = result.data;
  if (result.ok && data !== null && typeof data === "object" && !Array.isArray(data)) {
    const isScroll = !Object.hasOwn(data, "fusion");
    const summary = metricTiles(isScroll ? [
      ["mode", "payload only"],
      ["returned", scrollPagination?.total ?? data.total ?? data.returned],
      ["elapsed_s", result.elapsed.toFixed(3)],
    ] : [
      ["fusion", data.fusion],
      ["reranked", data.reranked],
      ["fused candidates", data.candidates],
      ["returned", data.returned],
      ["skipped_no_text", data.skipped_without_text],
      ["elapsed_s", result.elapsed.toFixed(3)],
    ]);
    const rankStart = isScroll
      ? ((scrollPagination?.page ?? 1) - 1) * (scrollPagination?.basePayload.limit ?? data.returned)
      : 0;
    const hits = (data.results || [])
      .map((hit, index) => renderResultCard(rankStart + index + 1, hit, !isScroll))
      .join("");
    const totalPages = Math.max(1, Math.ceil(
      (scrollPagination?.total ?? data.total ?? data.returned) /
      (scrollPagination?.basePayload.limit ?? 1)
    ));
    // An ordered scroll has no offset-based pages: reporting "Page 1 of N" there
    // would promise pages the API cannot serve, so name the mode instead.
    const ordered = Boolean(scrollPagination?.basePayload?.order_by);
    const pagination = isScroll ? `<nav class="query-pagination" aria-label="Payload results pages">
      <button type="button" class="btn btn-ghost" data-query-page="previous"
              ${scrollPagination?.history.length && !ordered ? "" : "disabled"}>← Previous page</button>
      <span class="pagination-page">${ordered
        ? `Ordered by ${esc(scrollPagination.basePayload.order_by.key)} ${esc(scrollPagination.basePayload.order_by.direction)}`
        : `Page ${esc(scrollPagination?.page ?? 1)} of ${esc(totalPages)}`}</span>
      ${data.next_page_offset === null || data.next_page_offset === undefined
        ? `<span class="pagination-end">End of results</span>`
        : `<button type="button" class="btn btn-primary" data-query-page="next">Next page →</button>`}
    </nav>` : "";
    container.innerHTML =
      alertHtml("success", `${isScroll ? "Retrieval" : "Query"} completed in ${result.elapsed.toFixed(3)}s`) +
      `<h2 class="section"><span class="section-num">05</span><span class="section-label">Results</span></h2>` +
      summary + hits + pagination +
      `<details class="raw-details"><summary>Raw response JSON</summary>${jsonBlock(data)}</details>`;
  } else if (result.ok) {
    container.innerHTML = alertHtml("warning", "Unexpected non-JSON response") + jsonBlock(data);
  } else {
    container.innerHTML = apiErrorHtml(result, "Query failed");
  }
}

async function loadScrollPage(offset, pagination) {
  if (!pagination) return { accepted: false, ok: false };
  const payload = { ...pagination.basePayload, include_total: false };
  if (offset !== null && offset !== undefined) payload.offset = offset;
  const result = await api.scrollPoints(payload);
  if (scrollPagination !== pagination) {
    return { accepted: false, ok: result.ok };
  }
  if (result.ok && result.data && typeof result.data === "object") {
    pagination.currentOffset = offset;
    pagination.nextOffset = result.data.next_page_offset ?? null;
  }
  renderQueryResponse(result);
  return { accepted: true, ok: result.ok };
}

function initQueryLab() {
  fillReference("query-reference", buildQueryParamHelp(queryConfig));

  $("ql-limit").max = String(queryConfig.queryMaxLimit);
  $("ql-cand").max = String(queryConfig.queryMaxCandidateLimit);
  $("ql-topk").max = String(queryConfig.queryMaxRerankTopK);

  renderPayloadFieldOptions();

  const filterContainer = $("ql-filter-rows");
  filterContainer.addEventListener("input", (event) => {
    const nodeEl = event.target.closest("[data-node-id]");
    if (!nodeEl) return;
    const entry = findFilterEntry(nodeEl.dataset.nodeId);
    if (!entry) return;
    const node = entry.node;
    const role = event.target.dataset.role;
    let rerender = false;
    if (role === "clause") {
      node.clause = event.target.value;
      rerender = true;
    } else if (role === "field") {
      node.fieldName = event.target.value;
      resetRowValues(node);
      // A new field can mean a new schema: re-render so the operation list
      // and value editor match it.
      normalizeRowOp(node);
      if (node.fieldName === VECTOR_FIELD) node.rawValue = state.vectorNames[0];
      rerender = true;
    } else if (role === "op") {
      node.op = event.target.value;
      resetRowValues(node);
      rerender = true;
    } else if (role === "value") {
      node.rawValue = event.target.value;
    } else if (role === "value2") {
      node.rawValue2 = event.target.value;
    } else if (role === "value3") {
      node.rawValue3 = event.target.value;
    } else if (role === "value4") {
      node.rawValue4 = event.target.value;
    } else if (role === "lower-op") {
      node.lowerOp = event.target.value;
    } else if (role === "upper-op") {
      node.upperOp = event.target.value;
    } else if (role === "group-min") {
      node.minShould = event.target.valueAsNumber;
    } else if (role === "nested-root") {
      node.root = event.target.value;
      node.children = [newCondition(node.root)];
      node.minShould = 1;
      rerender = true;
    }
    if (rerender) renderFilterTree();
    markQueryChanged();
    updateQueryUI();
  });
  filterContainer.addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const nodeEl = button.closest("[data-node-id]");
    const entry = nodeEl ? findFilterEntry(nodeEl.dataset.nodeId) : null;
    if (!entry) return;
    const action = button.dataset.action;
    const targetRoot = entry.node.kind === "nested"
      ? entry.node.root
      : entry.nestedRoot;
    if (action === "remove" && entry.parent) {
      entry.parent.children = entry.parent.children.filter(
        (child) => String(child.id) !== String(entry.node.id));
    } else if (action === "add-condition") {
      entry.node.children.push(newCondition(targetRoot));
    } else if (action === "add-group") {
      entry.node.children.push(newLogicalGroup(targetRoot));
    } else if (action === "add-nested") {
      entry.node.children.push(newNestedGroup(targetRoot));
    }
    renderFilterTree();
    markQueryChanged();
    updateQueryUI();
  });
  $("ql-filter-add").addEventListener("click", () => {
    state.filterTree.children.push(newCondition());
    renderFilterTree();
    markQueryChanged();
    updateQueryUI();
  });
  $("ql-filter-group-add").addEventListener("click", () => {
    state.filterTree.children.push(newLogicalGroup());
    renderFilterTree();
    markQueryChanged();
    updateQueryUI();
  });
  $("ql-filter-nested-add").addEventListener("click", () => {
    state.filterTree.children.push(newNestedGroup());
    renderFilterTree();
    markQueryChanged();
    updateQueryUI();
  });

  // RRF and DBSF put the fused score on different scales, so a threshold typed
  // for one is meaningless under the other (and, being a valid number, would
  // otherwise ride along into the request unnoticed). Switching fusion resets
  // it to 0 — a no-op cut on both scales — rather than carrying the stale
  // value over. Only a real selection change fires here, and restoreSnapshot
  // sets .checked programmatically, which fires nothing: restored thresholds
  // survive. The bubbled change then re-renders the preview via the delegated
  // handler below.
  for (const radio of document.querySelectorAll('input[name="ql-fusion"]')) {
    radio.addEventListener("change", () => {
      $("ql-threshold").value = "0";
    });
  }

  // Any other control change re-renders preview + validation.
  for (const type of ["input", "change"]) {
    $("panel-query").addEventListener(type, (event) => {
      if (event.target.closest("#ql-filter-rows")) return; // handled above
      if (event.target.closest("#ql-results")) return;
      if (event.target.id === "ql-rerank" || event.target.id === "ql-query") {
        renderPayloadFieldOptions();
      }
      markQueryChanged();
      updateQueryUI();
    });
  }

  $("ql-run").addEventListener("click", async () => {
    const current = readQueryState();
    const errors = validateQueryState(current, state.fieldSchemas, queryConfig);
    if (errors.length) return;
    const run = queryCoordinator.begin();
    const target = state.active;
    let pendingPagination = null;
    updateQueryUI();
    try {
      let result;
      let requestPayload;
      if (current.query.trim()) {
        requestPayload = buildRequestPayload(current, state.fieldSchemas);
        result = await api.search(requestPayload);
      } else {
        requestPayload = buildScrollPayload(current, state.fieldSchemas);
        pendingPagination = {
          basePayload: requestPayload,
          currentOffset: null,
          nextOffset: null,
          total: null,
          history: [],
          page: 1,
        };
        result = await api.scrollPoints(pendingPagination.basePayload);
        if (result.ok && result.data && typeof result.data === "object") {
          pendingPagination.nextOffset = result.data.next_page_offset ?? null;
          pendingPagination.total = result.data.total ?? result.data.returned;
        }
      }
      if (queryCoordinator.canCommit(run) && target === state.active) {
        scrollPagination = pendingPagination;
        renderQueryResponse(result);
        if (result.ok && $("ql-download").checked) {
          // Hybrid search returns a single capped page; payload-only scroll is
          // paginated, so walk every page before saving — the user wants the
          // full result set, not just what the UI shows.
          if (current.query.trim()) {
            downloadResultsAsChunks(result, target, requestPayload);
          } else {
            const walk = await collectAllScrollHits(
              requestPayload, result.data, target, run);
            if (walk !== null && queryCoordinator.canCommit(run)) {
              downloadResultsAsChunks(
                { ok: true, elapsed: result.elapsed,
                  data: { ...result.data, results: walk.hits,
                          returned: walk.hits.length } },
                target, requestPayload);
            }
          }
        }
      }
    } catch (error) {
      if (queryCoordinator.canCommit(run) && target === state.active) {
        scrollPagination = null;
        renderQueryResponse({
          ok: false,
          status: null,
          data: null,
          error: error instanceof Error ? error.message : "Unexpected query error",
          elapsed: 0,
        });
      }
    } finally {
      queryCoordinator.finish(run);
      updateQueryUI();
    }
  });

  $("ql-results").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-query-page]");
    if (!button || !scrollPagination) return;
    const pagination = scrollPagination;
    const snapshot = {
      currentOffset: pagination.currentOffset,
      nextOffset: pagination.nextOffset,
      total: pagination.total,
      history: [...pagination.history],
      page: pagination.page,
    };
    let offset;
    if (button.dataset.queryPage === "next") {
      if (pagination.nextOffset === null) return;
      pagination.history.push(pagination.currentOffset);
      offset = pagination.nextOffset;
      pagination.page += 1;
    } else {
      if (!pagination.history.length) return;
      offset = pagination.history.pop();
      pagination.page -= 1;
    }
    for (const control of $("ql-results").querySelectorAll("[data-query-page]")) {
      control.disabled = true;
    }
    button.textContent = "Loading…";
    const outcome = await loadScrollPage(offset, pagination);
    if (outcome.accepted && !outcome.ok) Object.assign(pagination, snapshot);
    if (outcome.accepted && (document.activeElement === button || document.activeElement === document.body)) {
      const container = $("ql-results");
      const next = container.querySelector(`[data-query-page="${button.dataset.queryPage}"]:not(:disabled)`)
        || container.querySelector("[data-query-page]:not(:disabled)") || container;
      if (next === container) next.tabIndex = -1;
      next.focus({ preventScroll: true });
    }
  });

  applyQueryState(defaultQueryState(queryConfig));
}

// ── Collections ───────────────────────────────────────────────────────────────

function toggleValues(prefix) {
  return {
    denseOnDisk: $(`${prefix}-dense`).value === "true",
    quantization: $(`${prefix}-quant`).value === "true",
    sparseIdf: $(`${prefix}-idf`).value === "true",
    sparseOnDisk: $(`${prefix}-sparse-disk`).value === "true",
    payloadOnDisk: $(`${prefix}-payload-disk`).value === "true",
  };
}

function reportHtml(report) {
  const dense = report.dense || {};
  const sparse = report.sparse || {};
  const bool = (v) => (v ? "true" : "false");
  const tri = (v) => (v === null || v === undefined ? "—" : bool(v));
  const dash = (v) => (v === null || v === undefined ? "—" : v);
  const [findings, fixes] = collectionSchemaMessages(report);
  const compatibility = report.embedding_compatibility || {};
  const storedContract = compatibility.stored || {};
  const activeContract = compatibility.active || {};
  const shortFingerprint = (value) => {
    const raw = String(value || "").replace(/^sha256:/, "");
    return raw ? `${raw.slice(0, 8)}…${raw.slice(-6)}` : "—";
  };
  const contractType = compatibility.status === "compatible"
    ? "success"
    : "error";
  const contractLabel = String(
    compatibility.status || "incompatible"
  ).replaceAll("_", " ").toUpperCase();
  const captions = (banner, type, messages) => messages.length
    ? alertHtml(type, banner) + messages.map((m) => `<p class="caption">${esc(m)}</p>`).join("")
    : "";
  // Foreign collection (incompatible, no stored contract — not created by this
  // app): schema findings/fixes compare against OUR expected schema (named
  // dense/sparse vectors, indexing_threshold, …), which is meaningless noise
  // for a collection this app never wrote. Suppress the contract message and
  // both caption blocks; the metric tiles above already show INCOMPATIBLE.
  const isForeign = compatibility.status === "incompatible" && !compatibility.stored;
  return metricTiles([
    ["dense on disk", bool(dense.on_disk)],
    ["sparse on disk", bool(sparse.on_disk)],
    ["payload on disk", tri(report.on_disk_payload)],
    ["quantization", bool(quantizationEnabled(report.quantization ?? ""))],
    ["sparse idf", bool(String(sparse.modifier || "").toLowerCase() === "idf")],
  ]) + metricTiles([
    ["collection name", report.collection_name || "—"],
    ["status", report.status || "—"],
    ["points", dash(report.points_count)],
    ["dense size", dash(dense.size)],
  ]) + metricTiles([
    ["embedding contract", contractLabel],
    ["stored profile", displayEmbedderProfile(storedContract.profile)],
    ["active profile", displayEmbedderProfile(activeContract.profile)],
    ["fingerprint", shortFingerprint(storedContract.fingerprint)],
  ]) +
    (isForeign
      ? ""
      : alertHtml(contractType,
        compatibility.message || "Embedding contract information is unavailable.") +
        captions("Schema findings", "warning", findings) +
        captions("Suggested fixes", "info", fixes));
}

const REINDEX_BLOCKING_STATUSES = new Set([
  "running", "paused", "interrupted", "finalizing", "failed",
]);
const REINDEX_CONTROLLABLE_STATUSES = new Set([
  "running", "paused", "interrupted", "failed",
]);

function clearReindexPoll() {
  if (state.reindexPollTimer !== null) {
    clearTimeout(state.reindexPollTimer);
    state.reindexPollTimer = null;
  }
  if (state.reindexCompleteTimer !== null) {
    clearTimeout(state.reindexCompleteTimer);
    state.reindexCompleteTimer = null;
  }
}

function resetReindexUI() {
  clearReindexPoll();
  state.reindexLoadId += 1;
  state.reindexPreflight = null;
  state.reindexStatus = null;
  $("cr-confirm").value = "";
  $("cr-confirm-wrap").hidden = true;
  $("cr-start").hidden = true;
  $("cr-status").innerHTML = state.active
    ? alertHtml("info", "Loading vector regeneration status…")
    : alertHtml("info", "No collection selected.");
}

function renderReindexPreflight(preflight) {
  const blockers = Array.isArray(preflight?.blockers) ? preflight.blockers : [];
  const type = preflight?.can_start ? "success" : "error";
  const title = preflight?.can_start
    ? "Preflight passed — collection is ready"
    : "Preflight blocked vector regeneration";
  $("cr-preflight-result").innerHTML =
    alertHtml(type, title) +
    metricTiles([
      ["total points", preflight?.total_points ?? "—"],
      ["eligible", preflight?.eligible_points ?? "—"],
      ["invalid text", preflight?.invalid_points ?? "—"],
      ["target profile", preflight?.target_contract?.profile ?? "—"],
      ["Qdrant", preflight?.qdrant_version ?? "—"],
    ]) +
    blockers.map((message) =>
      `<p class="caption warn">${esc(uppercaseMessage(message))}</p>`).join("");
}

const INGEST_TYPE_OPTIONS =
  ["keyword", "text", "integer", "float", "bool", "datetime", "uuid", "geo", "untyped"];

// summarizeIngestChunks infers JS types; map them to a sensible default dropdown
// selection for field rows. Anything unmapped (array/object/mixed/null) → untyped.
const INFERRED_TO_TYPE = { string: "keyword", int: "integer", float: "float", bool: "bool" };

// A field the active collection already indexes cannot pick its own type: the
// manager would replace the existing index (drop + recreate) on the next
// ingest. The row is pinned to the live schema and says where to change it.
function ingestIndexLockTitle(schema) {
  return `Indexed as ${schema} — change it in Collections → Payload Indexes`;
}

function ingestTypeSelect(selected, lockedSchema = null, field = "source_id") {
  const values = lockedSchema && !INGEST_TYPE_OPTIONS.includes(lockedSchema)
    // A schema this console has no option for (a future Qdrant type) still has
    // to be shown, or the row would silently display the wrong type.
    ? [...INGEST_TYPE_OPTIONS, lockedSchema]
    : INGEST_TYPE_OPTIONS;
  const active = lockedSchema || selected;
  const opts = values.map((t) => {
    const label = t === "untyped" ? "— untyped" : t;
    const sel = t === active ? " selected" : "";
    return `<option value="${t}"${sel}>${label}</option>`;
  }).join("");
  const attrs = lockedSchema
    ? ` disabled title="${esc(ingestIndexLockTitle(lockedSchema))}"`
    : "";
  return `<select class="ing-map-type" aria-label="${esc(field)} type" data-choice="${esc(selected)}"${attrs}>${opts}</select>`;
}

// Editable payload mapping table. The first two rows are synthetic — source_id
// and the chunk ordinal `id` — with an editable NAME; the rest are the chunk
// payload fields. Rendered once per upload; edits are read back from the DOM.
function ingestMappingTable(fields) {
  // The source_id row behaves like a field row: NAME renames the payload key
  // (default "source_id"); its value is always the filename stem (fixed, not
  // shown here — the "source file" badge shows the file).
  const sourceRow =
    `<tr data-field="source_id" data-source-id="true"` +
    `${state.fieldSchemas.source_id ? ` data-indexed="${esc(state.fieldSchemas.source_id)}"` : ""}>` +
    `<td>source_id</td>` +
    `<td><input type="text" class="ing-map-name" aria-label="source_id name" spellcheck="false" value="source_id" /></td>` +
    `<td>${ingestTypeSelect("keyword", state.fieldSchemas.source_id || null)}</td></tr>`;
  // The chunk ordinal is ordinary payload whose NAME travels as `chunk_id_key`,
  // so the middleware relocates it rather than storing it twice. TYPE is fixed:
  // an ordinal is the integer the point id derives from, and the server writes
  // it back unconverted.
  const chunkIdRow =
    `<tr data-field="id" data-chunk-id="true">` +
    `<td>id</td>` +
    `<td><input type="text" class="ing-map-name" aria-label="id name" spellcheck="false" value="id" /></td>` +
    `<td><select class="ing-map-type" aria-label="id type" disabled title="A chunk ordinal is always stored as an integer">` +
    `<option value="integer" selected>integer</option></select></td></tr>`;
  const fieldRows = fields.map((f) => {
    const defaultType = INFERRED_TO_TYPE[f.type] || "untyped";
    // The lock follows the row's FINAL name, which starts out as the JSON key
    // and is re-evaluated in place on every rename (syncIngestIndexLock).
    const indexed = state.fieldSchemas[literalPayloadIndexField(f.name)] || null;
    return `<tr data-field="${esc(f.name)}"` +
      `${indexed ? ` data-indexed="${esc(indexed)}"` : ""}>` +
      `<td>${esc(f.name)}</td>` +
      `<td><input type="text" class="ing-map-name" aria-label="${esc(f.name)} name" spellcheck="false" value="${esc(f.name)}" /></td>` +
      `<td>${ingestTypeSelect(defaultType, indexed, f.name)}</td></tr>`;
  }).join("");
  return `<div class="table-wrap"><table id="ing-map-table">` +
    `<thead><tr><th scope="col">field</th><th scope="col">name</th>` +
    `<th scope="col">type</th></tr></thead>` +
    `<tbody>${sourceRow}${chunkIdRow}${fieldRows}</tbody></table></div>` +
    `<div id="ing-map-errors" role="alert"></div>`;
}

function renderIngestPreflight(summary, ctx) {
  const ok = summary.total > 0 && summary.invalid === 0;
  const title = ok
    ? "Preflight passed — file ready to ingest"
    : "Preflight blocked ingest";
  const contextRow =
    `<div class="metric-row">` +
    `<div class="metric"><div class="metric-label">target collection</div>` +
    `<div class="metric-value">${esc(ctx.active || "—")}</div></div>` +
    `<div class="metric"><div class="metric-label">source file</div>` +
    `<div class="metric-value">${esc(ctx.fileName || "—")}</div></div></div>`;
  $("ing-preflight-result").innerHTML =
    alertHtml(ok ? "success" : "error", title) +
    metricTiles([
      ["chunks", summary.total],
      ["valid", summary.valid],
      ["invalid", summary.invalid],
    ]) +
    contextRow +
    ingestMappingTable(summary.fields) +
    summary.errors.map((e) =>
      `<p class="caption warn">${esc(uppercaseMessage(e))}</p>`).join("");
}

function clearIngestCompleteTimer() {
  if (state.ingestCompleteTimer !== null) {
    clearTimeout(state.ingestCompleteTimer);
    state.ingestCompleteTimer = null;
  }
}

// Progress UI for the chunked ingest upload — mirrors the Re-embed (02) visual:
// an alert, the shared .reindex-progress bar, and metric tiles. Writes #ing-result.
function renderIngestProgress({ status, done, total, skipped, message }, extraHtml = "") {
  const percent = status === "completed"
    ? 100
    : total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  const type = status === "completed" ? "success"
    : status === "failed" ? "error" : "info";
  const shown = status === "completed" ? total : done;
  $("ing-result").innerHTML =
    alertHtml(type, message || `Ingest: ${status.toUpperCase()}`) +
    `<div class="reindex-progress" role="progressbar" aria-label="Ingest progress" ` +
    `aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}">` +
    `<div class="reindex-progress-bar" style="width:${percent}%"></div></div>` +
    metricTiles([
      ["progress", `${percent}%`],
      ["chunks", `${shown} / ${total}`],
      ["skipped", skipped],
    ]) +
    extraHtml;
}

// Chip colour per outcome: created reads like the other "up" states (blue),
// a replaced index is a warning the user did not ask for (amber), a failure is
// red, and an untouched (already indexed) field stays the neutral chip.
const INGEST_INDEX_CHIP_CLASS = {
  created: "chip up",
  updated: "chip warn",
  failed: "chip down",
  skipped: "chip",
};

const INGEST_INDEX_STATUS_LABEL = {
  created: "created",
  updated: "replaced (existing index had a different type)",
  failed: "failed",
  skipped: "already indexed",
};

// The index report shares the completed block's lifecycle — it is a result
// line, not extra state to keep around (PRD §16.3).
const INGEST_RESULT_HOLD_MS = 5000;

function ingestIndexPendingHtml() {
  return `<div id="ing-index-summary">` +
    `<p class="caption">Creating payload indexes… index creation may take a ` +
    `while on a large collection.</p></div>`;
}

// Final index report, appended under the ingest progress. Ingest itself has
// already succeeded here — anything shown below is about the indexes only.
function ingestIndexSummaryHtml(summary) {
  const { counts } = summary;
  const headline = [
    counts.created ? `${counts.created} created` : "",
    counts.updated ? `${counts.updated} replaced` : "",
    counts.failed ? `${counts.failed} failed` : "",
    counts.skipped ? `${counts.skipped} already indexed` : "",
  ].filter(Boolean).join(", ") || "nothing to do";
  const chips = summary.rows.map((row) =>
    `<span class="${INGEST_INDEX_CHIP_CLASS[row.status]}" data-field="${esc(row.field)}" ` +
    `data-status="${row.status}">${esc(row.field)} (${esc(row.schema)}) — ` +
    `${esc(INGEST_INDEX_STATUS_LABEL[row.status])}</span>`).join("");
  const warnings = [];
  if (summary.replaced) {
    warnings.push("An existing index was replaced with a different type — " +
      "verify it in Collections → Payload Indexes.");
  }
  if (summary.reason) warnings.push(summary.reason);
  return `<div id="ing-index-summary">` +
    `<p class="caption">Payload indexes: ${esc(headline)}</p>` +
    `<div class="chips">${chips}</div>` +
    warnings.map((message) =>
      `<p class="caption warn">${esc(uppercaseMessage(message))}</p>`).join("") +
    `<p class="caption">Manage manually → Collections → Payload Indexes</p>` +
    `</div>`;
}

// Post-ingest phase: create a payload index for every typed mapping row, in
// one batch call. Runs only on a fully successful ingest, so a half-written
// collection never gains indexes. Returns the rendered summary, or null when
// the toggle is off / the mapping types nothing / the run was abandoned
// (collection switched, file swapped) — the caller uses it only to decide how
// long the completed block stays on screen.
async function runIngestIndexPhase(
  progress, { fieldMapping, sourceMap, chunkIdKey, target, runId, current },
) {
  if (!current() || !state.ingestCreateIndexes) return null;
  const { requests, alreadyIndexed } = buildIngestIndexRequests(
    fieldMapping, sourceMap, chunkIdKey, state.fieldSchemas);
  if (!requests.length && !alreadyIndexed.length) return null;

  let response = { created: [], updated: [], skipped: [], failed: [] };
  let reason = "";
  if (requests.length) {
    renderIngestProgress(progress, ingestIndexPendingHtml());
    const result = await api.createPayloadIndex(target, { indexes: requests });
    // Same guard as the upload loop: a collection switch or a newer ingest
    // started mid-call makes this report belong to a screen the user has left.
    if (!current() || target !== state.active || state.ingestRunId !== runId) return null;
    if (result.ok && result.data && typeof result.data === "object") {
      response = result.data;
    } else {
      response = null;
      reason = apiErrorText(result, "payload index creation failed");
    }
  }
  const summary = summarizeIngestIndexResult(
    response, requests, alreadyIndexed, reason);
  console.log("Ingest payload indexes:", summary.counts, summary.rows);
  renderIngestProgress(progress, ingestIndexSummaryHtml(summary));
  return summary;
}

// Re-evaluate one mapping row's type lock after a rename, in place: the rename
// handler deliberately does not re-render the table (it would drop the caret),
// so the select is patched instead. The chunk-ordinal row is skipped — its type
// is fixed at `integer` whatever the collection already indexes.
// Re-apply the locks to a mapping table that is already on screen. The index
// list arrives asynchronously, so a preflight rendered while it was still in
// flight starts unlocked; the constraint has to appear as soon as the data is
// there (and disappear again on a collection whose fields are not indexed).
function refreshIngestIndexLocks() {
  const rows = document.querySelectorAll("#ing-map-table tbody tr");
  if (!rows.length) return;
  for (const tr of rows) syncIngestIndexLock(tr);
  // A lock can change a row's effective type, so the coercion check must rerun.
  validateIngestMapping();
}

function syncIngestIndexLock(tr) {
  if (!tr || tr.dataset.chunkId === "true") return;
  const select = tr.querySelector(".ing-map-type");
  const nameInput = tr.querySelector(".ing-map-name");
  if (!select || !nameInput) return;
  const indexed = state.fieldSchemas[literalPayloadIndexField(nameInput.value)] || null;
  if (indexed === (tr.dataset.indexed || null)) return;
  if (indexed) {
    // Remember what the user had picked so releasing the lock can restore it.
    if (!tr.dataset.indexed) select.dataset.choice = select.value;
    if (![...select.options].some((o) => o.value === indexed)) {
      select.add(new Option(indexed, indexed));
    }
    select.value = indexed;
    select.disabled = true;
    select.title = ingestIndexLockTitle(indexed);
    tr.dataset.indexed = indexed;
  } else {
    select.value = select.dataset.choice || "untyped";
    select.disabled = false;
    select.removeAttribute("title");
    delete tr.dataset.indexed;
  }
}

function readIngestMapping() {
  const rows = [...document.querySelectorAll("#ing-map-table tbody tr")];
  const fieldMapping = [];
  let sourceId = { key: "source_id", type: "keyword" };
  let chunkIdKey = "id";
  for (const tr of rows) {
    const name = tr.querySelector(".ing-map-name").value;
    const type = tr.querySelector(".ing-map-type").value;
    if (tr.dataset.sourceId === "true") {
      sourceId = { key: name, type };
    } else if (tr.dataset.chunkId === "true") {
      chunkIdKey = name;
    } else {
      fieldMapping.push({ field: tr.dataset.field, name, type });
    }
  }
  return { fieldMapping, sourceId, chunkIdKey };
}

// Validate the source_id row like a field row: its NAME (the payload key —
// empty / reserved / colliding with a field NAME) and its TYPE coercion against
// the fixed value (the filename stem).
function validateSourceIdMapping(sourceId, fieldMapping, stem) {
  const errors = [];
  const key = String(sourceId.key ?? "").trim();
  if (!key) {
    errors.push("source_id: name is required.");
  } else if (INGEST_SOURCE_ID_RESERVED_KEYS.has(key)) {
    errors.push(`source_id: name '${key}' is reserved`);
  } else {
    const clash = fieldMapping.find((m) => String(m.name ?? "").trim() === key);
    if (clash) {
      errors.push(`source_id: name '${key}' duplicates field '${clash.field}'`);
    }
  }
  if (sourceId.type !== "untyped") {
    const res = coercePayloadValue(stem, sourceId.type);
    if (!res.ok) {
      errors.push(`source_id → ${sourceId.type}: '${stem}' is not convertible`);
    }
  }
  return errors;
}

// Validate the chunk-ordinal row's NAME (the payload key it travels under as
// `chunk_id_key`). Mirrors _validate_chunk_id_key plus the collision check in
// points.py, so the preflight blocks exactly what the middleware would 400 on.
// `id` stays legal here — it is the default, not a clobber.
function validateChunkIdMapping(chunkIdKey, fieldMapping, sourceId) {
  const errors = [];
  const key = String(chunkIdKey ?? "").trim();
  if (!key) return ["id: name is required."];
  if (INGEST_CHUNK_ID_RESERVED_KEYS.has(key)) {
    errors.push(`id: name '${key}' is reserved`);
    return errors;
  }
  if (key === String(sourceId.key ?? "").trim()) {
    errors.push(`id: name '${key}' duplicates the source_id name`);
  }
  const clash = fieldMapping.find((m) => String(m.name ?? "").trim() === key);
  if (clash) errors.push(`id: name '${key}' duplicates field '${clash.field}'`);
  return errors;
}

// Re-validate the current mapping against the parsed chunks without re-rendering
// the table (which would drop input focus). Writes any errors below the table
// and flips state.ingestPreflight.mappingValid, then refreshes button state.
function validateIngestMapping() {
  const preflight = state.ingestPreflight;
  if (!preflight) return;
  const table = $("ing-map-table");
  if (!table) {
    preflight.mappingValid = true;
    updatePointsUI();
    return;
  }
  const { fieldMapping, sourceId, chunkIdKey } = readIngestMapping();
  const { errors: fieldErrors } = applyIngestFieldMapping(preflight.chunks, fieldMapping);
  const srcErrors = validateSourceIdMapping(
    sourceId, fieldMapping, fileStem(preflight.fileName));
  const chunkIdErrors = validateChunkIdMapping(chunkIdKey, fieldMapping, sourceId);
  const errors = [...srcErrors, ...chunkIdErrors, ...fieldErrors];
  preflight.mappingValid = errors.length === 0;
  $("ing-map-errors").innerHTML =
    errors.map((e) =>
      `<p class="caption warn">${esc(uppercaseMessage(e))}</p>`).join("");
  for (const row of table.querySelectorAll("[data-field]")) {
    const rowErrors = row.dataset.sourceId ? srcErrors : row.dataset.chunkId ? chunkIdErrors
      : fieldErrors.filter((message) => message.includes(row.dataset.field));
    for (const control of row.querySelectorAll("input, select")) {
      associateFieldError(control, "ing-map-errors", rowErrors.length > 0);
    }
  }
  updatePointsUI();
}

function renderReindexStatus(data) {
  state.reindexStatus = data && typeof data === "object" ? data : null;
  const job = state.reindexStatus?.job;
  if (!job) {
    $("cr-status").innerHTML = "";
    updateCollectionsUI();
    updatePointsUI();
    return;
  }
  const total = Number(job.total_points) || 0;
  const processed = Number(job.processed_points) || 0;
  const percent = job.status === "completed"
    ? 100
    : total > 0 ? Math.min(100, Math.round((processed / total) * 100)) : 0;
  const messageType = job.status === "completed"
    ? "success"
    : job.status === "failed"
      ? "error"
      : job.status === "paused" || job.status === "interrupted"
        ? "warning"
        : "info";
  $("cr-status").innerHTML =
    alertHtml(messageType, `Vector regeneration: ${String(job.status).toUpperCase()}`) +
    `<div class="reindex-progress" role="progressbar" aria-label="Vector regeneration progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}">
      <div class="reindex-progress-bar" style="width:${percent}%"></div>
    </div>` +
    metricTiles([
      ["progress", `${percent}%`],
      ["processed", `${processed} / ${total}`],
      ["target profile", job.target_contract?.profile ?? "—"],
      ["worker", state.reindexStatus.task_running ? "RUNNING" : "IDLE"],
    ]) +
    (job.error ? alertHtml("error", job.error) : "");
  updateCollectionsUI();
  updatePointsUI();
}

async function loadReindexStatus(target = state.active) {
  clearReindexPoll();
  if (!target) {
    resetReindexUI();
    updateCollectionsUI();
    return;
  }
  const loadId = ++state.reindexLoadId;
  const previousStatus = state.reindexStatus?.job?.status;
  const result = await api.getReindexStatus(target);
  if (target !== state.active || loadId !== state.reindexLoadId) return;
  if (!result.ok) {
    state.reindexStatus = null;
    $("cr-status").innerHTML =
      apiErrorHtml(result, "Vector regeneration status unavailable");
    updateCollectionsUI();
    updatePointsUI();
    return;
  }
  renderReindexStatus(result.data);
  const job = result.data?.job;
  if (job?.status === "completed") {
    if (previousStatus !== "completed") {
      loadReport();
      loadCollectionFields();
    }
    // Show the COMPLETED block briefly, then drop back to the initial state.
    // clearReindexPoll() at the top already cancelled any prior timer, so a
    // refetch within the window simply restarts the 5s countdown.
    state.reindexCompleteTimer = setTimeout(() => {
      state.reindexCompleteTimer = null;
      if (target !== state.active || loadId !== state.reindexLoadId) return;
      renderReindexStatus(null);
    }, 5000);
  }
  if (result.data?.task_running
      || job?.status === "running"
      || job?.status === "finalizing") {
    state.reindexPollTimer = setTimeout(
      () => loadReindexStatus(target),
      1000,
    );
  }
}

// Seed the Schema Toggles widgets from the live report: all five toggles are
// sent on update, so starting from static defaults would silently flip
// settings the user never touched.
function seedSchemaToggles(report) {
  const current = reportToggleDefaults(report, {
    denseOnDisk: DEFAULTS.denseOnDisk,
    quantization: DEFAULTS.quantization,
    sparseIdf: DEFAULTS.sparseIdf,
    sparseOnDisk: DEFAULTS.sparseOnDisk,
    payloadOnDisk: DEFAULTS.payloadOnDisk,
  });
  $("cu-dense").value = String(current.denseOnDisk);
  $("cu-quant").value = String(current.quantization);
  $("cu-idf").value = String(current.sparseIdf);
  $("cu-sparse-disk").value = String(current.sparseOnDisk);
  $("cu-payload-disk").value = String(current.payloadOnDisk);
}

async function loadReport() {
  const target = state.active;
  const loadId = ++state.reportLoadId;
  const toggleScope = operationScope($("cu-submit")).coordinator;
  const toggleRun = toggleScope.snapshot();
  if (!target) {
    $("cc-report").innerHTML = alertHtml("info", "No collection selected.");
    state.reportOk = false;
    state.embeddingCompatibility = null;
    state.embeddingCompatibilityLoading = false;
    seedSchemaToggles(null);
    updateCollectionsUI();
    updateQueryUI();
    updatePointsUI();
    applyLockdowns();
    return;
  }
  state.embeddingCompatibility = null;
  state.embeddingCompatibilityLoading = true;
  updateQueryUI();
  updatePointsUI();
  applyLockdowns();
  const result = await api.getCollection(target);
  // Ignore responses that arrive after the user switched collection.
  if (target !== state.active || loadId !== state.reportLoadId) return;
  if (result.ok && result.data !== null && typeof result.data === "object") {
    $("cc-report").innerHTML = reportHtml(result.data);
    state.reportOk = true;
    state.embeddingCompatibility = result.data.embedding_compatibility || null;
    updateVectorNames(activeVectorNames(result.data));
    if (!toggleScope || toggleScope.isCurrent(toggleRun)) seedSchemaToggles(result.data);
  } else {
    $("cc-report").innerHTML = apiErrorHtml(result, "Report unavailable");
    // Without a live report the toggle widgets would show static defaults;
    // applying those blindly could flip settings, so Update stays disabled.
    state.reportOk = false;
    state.embeddingCompatibility = null;
    updateVectorNames([]);
    seedSchemaToggles(null);
  }
  state.embeddingCompatibilityLoading = false;
  updateCollectionsUI();
  updateQueryUI();
  updatePointsUI();
  applyLockdowns();
}

function updateVectorNames(names) {
  if (JSON.stringify(names) === JSON.stringify(state.vectorNames)) return;
  state.vectorNames = names;
  markQueryChanged();
  renderFilterTree();
}

// One row per payload field observed in the collection, merged with the
// index list: fields without an index yet show "—" in the schema column.
function indexesTable(indexes, allKeys) {
  const schemaByField = new Map(
    (Array.isArray(indexes) ? indexes : [])
      .filter((item) => item && typeof item === "object" && item.field_name)
      .map((item) => [item.field_name,
        item.phrase_matching ? "text (+phrase)" : item.field_schema]));
  const fields = [...new Set([...allKeys.map(literalPayloadIndexField), ...schemaByField.keys()])]
    .sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
  if (!fields.length) return alertHtml("info", "No payload fields found.");
  const rows = fields.map((field) =>
    `<tr><td>${esc(field)}</td><td>${esc(schemaByField.get(field) ?? "—")}</td></tr>`
  ).join("");
  return `<div class="table-wrap"><table>
    <thead><tr><th scope="col">payload field</th><th scope="col">index schema</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

// Failure rendering for Collections and Points: a clear one-line message plus
// any detail strings from the backend — never a raw JSON dump.
function apiErrorHtml(result, fallback = "Operation failed") {
  const detail = result.data && typeof result.data === "object"
    ? result.data.detail : null;
  let detailMsg = "";
  let detailErrors = [];
  if (typeof detail === "string") {
    detailMsg = detail;
  } else if (Array.isArray(detail)) {
    // FastAPI validation errors (422): [{loc, msg, type}, ...]
    detailErrors = detail.map((item) =>
      item && typeof item === "object" && item.msg ? String(item.msg) : String(item));
  } else if (detail && typeof detail === "object") {
    detailMsg = detail.message ? String(detail.message) : "";
    if (Array.isArray(detail.errors)) detailErrors = detail.errors;
  }
  // A bare "HTTP <status>" is noise to users: the server detail (or the
  // caller's fallback) carries the explanation. Human-phrased errors (401,
  // network) keep flowing through unchanged.
  const bareStatus = /^HTTP \d{3}$/.test(result.error || "");
  const headline = bareStatus
    ? detailMsg || fallback
    : (result.error || fallback) + (detailMsg ? ` — ${detailMsg}` : "");
  return alertHtml("error", headline) +
    detailErrors.map((m) => `<p class="caption">${esc(String(m))}</p>`).join("");
}

// Plain-text version of the API error detail, for messages embedded inside
// another alert (e.g. the ingest progress FAILED line). Handles FastAPI 422
// array details so they don't stringify to "[object Object]".
function apiErrorText(result, fallback = "request failed") {
  const detail = result.data && typeof result.data === "object"
    ? result.data.detail : null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) =>
      item && typeof item === "object" && item.msg ? String(item.msg) : String(item)
    ).join("; ");
  }
  if (detail && typeof detail === "object" && detail.message) return String(detail.message);
  // No usable detail: a bare "HTTP <status>" is noise — prefer the fallback.
  return /^HTTP \d{3}$/.test(result.error || "") ? fallback : result.error || fallback;
}

// Disable a button while its async handler runs so a double click cannot fire
// the same operation twice (recreate/delete are destructive).
async function withBusy(button, task) {
  if (button.disabled) return;
  const { coordinator } = operationScope(button);
  const run = coordinator.begin();
  busyOwners.set(button, run);
  button.disabled = true;
  button.dataset.busy = "true";
  button.setAttribute("aria-busy", "true");
  setDisabledReason([button.id], "Request in progress. Wait for its response.");
  let response;
  try {
    await task((result) => {
      if (!coordinator.isCurrent(run)) return false;
      if (result) response = result;
      return true;
    });
    if (coordinator.isCurrent(run)) associateResponseErrors(button, response);
  } finally {
    coordinator.finish(run);
    if (busyOwners.get(button) === run) {
      busyOwners.delete(button);
      delete button.dataset.busy;
    }
    updateCollectionsUI();
    updatePointsUI();
  }
}


function updateCollectionsUI() {
  $("ce-export").disabled = $("ce-export").dataset.busy === "true"
    || !state.servicesReady || !state.active;
  const reindexJob = state.reindexStatus?.job;
  const reindexStatus = reindexJob?.status;
  const reindexTaskRunning = state.reindexStatus?.task_running === true;
  const preflightReady = state.reindexPreflight?.can_start === true
    && state.reindexPreflight?.collection_name === state.active;
  const reindexBlocked = activeReindexBlocksMutations();
  $("cr-preflight").disabled = $("cr-preflight").dataset.busy === "true"
    || !state.servicesReady
    || !state.active
    || reindexBlocked;
  $("cr-confirm-wrap").hidden = !preflightReady;
  $("cr-start").hidden = !preflightReady;
  $("cr-confirm-label").innerHTML =
    `Type '<span class="literal">${esc(state.active || "the collection name")}</span>' to start vector regeneration`;
  $("cr-start").disabled = $("cr-start").dataset.busy === "true"
    || !state.servicesReady
    || !preflightReady
    || !confirmationSatisfied($("cr-confirm").value, state.active)
    || reindexBlocked;
  $("cr-pause").hidden = reindexStatus !== "running";
  $("cr-pause").disabled = $("cr-pause").dataset.busy === "true"
    || !state.servicesReady;
  $("cr-resume").hidden = !["paused", "interrupted", "failed"].includes(reindexStatus)
    || reindexTaskRunning;
  $("cr-resume").disabled = $("cr-resume").dataset.busy === "true"
    || !state.servicesReady;
  $("cr-cancel").hidden = !REINDEX_CONTROLLABLE_STATUSES.has(reindexStatus);
  $("cr-cancel").disabled = $("cr-cancel").dataset.busy === "true"
    || !state.servicesReady;

  const name = $("cc-name").value.trim();
  // Create-only: the active collection is embedder-mismatched, so recreate is
  // withdrawn (see applyLockdowns) while creating a new collection stays legal.
  const createOnly = createOnlyMode();
  const recreateBox = $("cc-recreate");
  if (createOnly && recreateBox.checked) {
    recreateBox.checked = false;
    $("cc-confirm").value = "";
  }
  recreateBox.disabled = createOnly;
  recreateBox.setAttribute("aria-disabled", String(createOnly));
  $("cc-confirm").disabled = createOnly;
  $("cc-recreate-note").hidden = !createOnly;
  $("cc-submit").textContent = createOnly ? "Create collection" : "Submit create/recreate";
  const recreate = recreateBox.checked && !createOnly;
  $("cc-confirm-wrap").hidden = !recreate;
  $("cc-confirm-label").innerHTML = `Type '<span class="literal">${esc(name || "the collection name")}</span>' to confirm recreate`;
  const confirmed = confirmationSatisfied($("cc-confirm").value, name);
  $("cc-submit").disabled = $("cc-submit").dataset.busy === "true"
    || !state.servicesReady
    || !name
    || (recreate && name === state.active && reindexBlocked)
    || (recreate && !confirmed);

  const delName = $("cd-name").value.trim();
  $("cd-confirm-label").innerHTML = `Type '<span class="literal">${esc(delName || "the collection name")}</span>' to confirm delete`;
  $("cd-submit").disabled = $("cd-submit").dataset.busy === "true"
    || !state.servicesReady
    || (delName === state.active && reindexBlocked)
    || !confirmationSatisfied($("cd-confirm").value, delName);

  $("pi-create").disabled = $("pi-create").dataset.busy === "true"
    || reindexBlocked
    || !state.servicesReady
    || !state.active
    || !$("pi-field").value.trim();
  $("pi-delete").disabled = $("pi-delete").dataset.busy === "true"
    || reindexBlocked
    || !state.servicesReady
    || !state.active
    || !$("pi-del-field").value.trim();
  $("cu-submit").disabled = $("cu-submit").dataset.busy === "true"
    || reindexBlocked
    || !state.servicesReady
    || !state.active
    || !state.reportOk;
  const blocked = "Collection mutations are disabled while vector regeneration is active.";
  operationReasons({
    "cc-submit": recreate && name === state.active && reindexBlocked ? blocked : "Enter a collection name and the exact confirmation when recreating.",
    "cd-submit": delName === state.active && reindexBlocked ? blocked : "Enter the collection name and its exact deletion confirmation.",
    "cu-submit": reindexBlocked ? blocked : "A current collection report is required before updating schema toggles.",
    "pi-create": reindexBlocked ? blocked : "Select a collection and enter a payload field.",
    "pi-delete": reindexBlocked ? blocked : "Select a collection and enter the payload index field to delete.",
    "ce-export": "Select a collection to export.",
    "cr-preflight": reindexBlocked ? blocked : "Select a collection for preflight.",
    "cr-start": reindexBlocked ? blocked : "Run preflight and enter the exact collection name.",
    "cr-pause": "Pause is unavailable in the current job state.",
    "cr-resume": "Resume is unavailable in the current job state.",
    "cr-cancel": "Cancel is unavailable in the current job state.",
  });
}

function initCollections() {
  fillReference("collections-reference", COLLECTIONS_HELP);
  for (const id of ["cc-dense", "cc-quant", "cc-idf", "cc-sparse-disk", "cc-payload-disk"]) {
    $(id).value = "true"; // backend defaults; the API stays the source of truth
  }

  for (const id of ["cc-name", "cc-recreate", "cd-name", "pi-field", "pi-del-field"]) {
    $(id).addEventListener("input", updateCollectionsUI);
  }
  wireConfirm("cc-confirm", "cc-confirm-hint", () => $("cc-name").value, updateCollectionsUI);
  wireConfirm("cd-confirm", "cd-confirm-hint", () => $("cd-name").value, updateCollectionsUI);
  wireConfirm("cr-confirm", "cr-confirm-hint", () => state.active, updateCollectionsUI);

  $("cr-preflight").addEventListener("click", () =>
    withBusy($("cr-preflight"), async (current) => {
      const target = state.active;
      state.reindexPreflight = null;
      $("cr-confirm").value = "";
      $("cr-preflight-result").innerHTML =
        alertHtml("info", "Scanning every point and validating the active embedder…");
      updateCollectionsUI();
      const result = await api.preflightReindex(target);
      if (!current(result)) return;
      if (result.ok) {
        state.reindexPreflight = result.data;
        renderReindexPreflight(result.data);
      } else {
        $("cr-preflight-result").innerHTML =
          apiErrorHtml(result, "Vector regeneration preflight failed");
      }
      updateCollectionsUI();
    }));

  $("cr-start").addEventListener("click", () =>
    withBusy($("cr-start"), async (current) => {
      const target = state.active;
      const result = await api.startReindex(
        target,
        $("cr-confirm").value.trim(),
      );
      if (!current(result)) return;
      if (result.ok) {
        state.reindexPreflight = null;
        $("cr-confirm").value = "";
        $("cr-preflight-result").innerHTML = "";
        // The status badge below reports the job state; no separate message.
        $("cr-result").innerHTML = "";
        renderReindexStatus(result.data);
        await loadReindexStatus(target);
      } else {
        $("cr-result").innerHTML =
          apiErrorHtml(result, "Vector regeneration start failed");
      }
    }));

  const reindexControl = (buttonId, action, errorLabel) => {
    $(buttonId).addEventListener("click", () =>
      withBusy($(buttonId), async (current) => {
        const target = state.active;
        const result = await action(target);
        if (!current(result)) return;
        if (result.ok) {
          // The status badge reflects the new job state; no separate message.
          $("cr-result").innerHTML = "";
          renderReindexStatus(result.data);
          await loadReindexStatus(target);
        } else {
          $("cr-result").innerHTML =
            apiErrorHtml(result, `${errorLabel} failed`);
        }
      }));
  };
  reindexControl("cr-pause", api.pauseReindex, "Pause");
  reindexControl("cr-resume", api.resumeReindex, "Resume");
  reindexControl("cr-cancel", api.cancelReindex, "Cancel");

  $("cc-submit").addEventListener("click", () => withBusy($("cc-submit"), async (current) => {
    const name = $("cc-name").value.trim();
    const createOnly = createOnlyMode();
    const recreate = $("cc-recreate").checked && !createOnly;
    // Send what the user actually typed so the backend confirmation gate stays
    // meaningful instead of being auto-satisfied by the UI.
    const payload = buildCreatePayload(
      name, recreate, recreate ? $("cc-confirm").value.trim() : null, toggleValues("cc"),
      { createOnly });
    const result = await api.createCollection(payload);
    if (!current(result)) return;
    if (result.ok) {
      // Badges live in 01 Report; switch the active collection to the one
      // just created/updated so the report (and every tab) points at it.
      // created=false means the collection was already there with a matching
      // schema — say so instead of claiming an operation happened.
      const created = result.data && typeof result.data === "object"
        && result.data.created === true;
      $("cc-create-result").innerHTML = created
        ? alertHtml("success", "Collection created")
        : alertHtml("info",
            "Collection already exists and already has these settings — nothing to do.");
      // Disarm recreate so a stray second click cannot destroy the fresh
      // collection without retyping the confirmation.
      $("cc-recreate").checked = false;
      $("cc-confirm").value = "";
      // state.active is pre-set below, so setActiveCollection sees no change
      // and skips its clear; recreate keeps the same name too. Stale 04/05
      // messages would otherwise survive pointing at destroyed/old state.
      clearActiveScopedResults();
      setActiveCollection(name);
      await refreshCollections();
      if (!current()) return;
      await Promise.all([
        loadReport(),
        loadReindexStatus(),
        loadCollectionFields(),
      ]);
    } else if (result.status === 409 && !recreate) {
      // Not an error worth a red alert + JSON dump: the collection simply
      // already exists — point the user at the online toggle editor instead.
      // In create-only mode that editor is locked too, so the advice changes:
      // pick a free name, or select a compatible collection first.
      $("cc-create-result").innerHTML = alertHtml("info", createOnly
        ? "Collection already exists. While the active collection does not match "
          + "the Smart Embedder, editing or recreating it is disabled — use a "
          + "different name, or select a compatible collection first."
        : "Collection already exists. To change the true/false flags use "
          + "04 Schema Toggles (online, non-destructive). Recreate is only for "
          + "structural changes and deletes all points.");
    } else {
      // A 409 on recreate=true means the rebuilt collection failed schema
      // verification — a real error, not an "already exists" hint.
      $("cc-create-result").innerHTML =
        apiErrorHtml(result, "Collection create failed");
    }
  }));

  $("cu-submit").addEventListener("click", () => withBusy($("cu-submit"), async (current) => {
    const values = toggleValues("cu");
    const result = await api.updateSchemaToggles(state.active, {
      dense_on_disk: values.denseOnDisk,
      quantization: values.quantization,
      sparse_idf: values.sparseIdf,
      sparse_on_disk: values.sparseOnDisk,
      on_disk_payload: values.payloadOnDisk,
    });
    if (!current(result)) return;
    if (result.ok) {
      // The refreshed schema shows up in the 01 Report badges via loadReport();
      // repeating it here would duplicate the readout.
      $("cu-result").innerHTML = alertHtml("success", "Schema toggles updated");
      await loadReport();
    } else {
      $("cu-result").innerHTML =
        apiErrorHtml(result, "Schema toggles update failed");
    }
  }));

  // phrase_matching is a text-index feature: show the checkbox only there.
  $("pi-schema").addEventListener("change", () => {
    $("pi-phrase-wrap").hidden = $("pi-schema").value !== "text";
  });
  $("pi-create").addEventListener("click", () => withBusy($("pi-create"), async (current) => {
    const field = $("pi-field").value.trim();
    const schema = $("pi-schema").value;
    const result = await api.createPayloadIndex(state.active, {
      indexes: [{
        field_name: field,
        field_schema: schema,
        ...(schema === "text" && $("pi-phrase").checked
          ? { phrase_matching: true }
          : {}),
      }],
    });
    if (!current(result)) return;
    if (result.ok) {
      const data = result.data && typeof result.data === "object" ? result.data : {};
      if (data.failed.length || !data.ok) {
        $("pi-create-result").innerHTML = alertHtml("error", `Payload indexes failed: ${data.failed.join(", ") || "request not completed"}. Refresh indexes before retrying.`);
      } else if (data.updated.length) {
        $("pi-create-result").innerHTML = alertHtml("success",
          `Index type changed: '${data.updated.join(", ")}' is now ${schema}.`);
      } else if (data.created?.length) {
        $("pi-create-result").innerHTML = alertHtml("success",
          `Index created: '${data.created.join(", ")}' as ${schema}.`);
      } else {
        $("pi-create-result").innerHTML = alertHtml("info",
          `No change: '${(data.skipped || [field]).join(", ")}' is already indexed as ${schema}.`);
      }
      await loadCollectionFields();
    } else {
      $("pi-create-result").innerHTML =
        apiErrorHtml(result, "Payload index create failed");
    }
  }));
  $("pi-delete").addEventListener("click", () => withBusy($("pi-delete"), async (current) => {
    const field = $("pi-del-field").value.trim();
    const result = await api.deletePayloadIndex(state.active, field);
    if (!current(result)) return;
    if (result.ok) {
      $("pi-del-result").innerHTML =
        alertHtml("success", `Payload index deleted: ${field}`);
      $("pi-del-field").value = "";
      await loadCollectionFields();
    } else {
      $("pi-del-result").innerHTML =
        apiErrorHtml(result, "Payload index delete failed");
    }
  }));

  $("cd-submit").addEventListener("click", () => withBusy($("cd-submit"), async (current) => {
    const name = $("cd-name").value.trim();
    const result = await api.deleteCollection(name, $("cd-confirm").value.trim());
    if (!current(result)) return;
    if (result.ok) {
      $("cd-result").innerHTML =
        alertHtml("success", `Collection deleted: ${name}`);
      // Disarm the confirmation so the button cannot fire again by accident.
      $("cd-name").value = "";
      $("cd-confirm").value = "";
      await refreshCollections();
      if (!current()) return;
      await Promise.all([
        loadReport(),
        loadReindexStatus(),
        loadCollectionFields(),
      ]);
    } else {
      $("cd-result").innerHTML =
        apiErrorHtml(result, "Collection delete failed");
    }
  }));

  // Full-collection export: the middleware walks every point and returns the
  // assembled document, which the browser saves. The walk used to live here;
  // it moved server-side so the same producer serves the console and the MCP
  // tool, and the file cannot drift between them.
  $("ce-export").addEventListener("click", () => withBusy($("ce-export"), async (current) => {
    const target = state.active;
    if (!target) {
      $("ce-result").innerHTML = alertHtml("warning", "No active collection to export.");
      return;
    }
    $("ce-result").innerHTML = alertHtml("info", "Exporting… scanning every point.");
    const res = await api.exportCollection({
      collection_name: target, destination: "body",
    });
    if (!current(res)) return;
    if (!res.ok) {
      $("ce-result").innerHTML = apiErrorHtml(res, "Collection export failed");
      return;
    }
    let parsed;
    try { parsed = JSON.parse(res.data); } catch { parsed = null; }
    const nChunks = parsed?.metadata?.n_chunks ?? 0;
    if (!nChunks) {
      $("ce-result").innerHTML =
        alertHtml("warning", "Collection is empty — nothing to export.");
      return;
    }
    // The filename is the server's; parsing it out keeps one naming authority.
    const disposition = res.headers?.get("content-disposition") || "";
    const filename = downloadFilename(disposition, `${target}_export.json`);
    // res.data is the untouched response text — saving the parsed object
    // instead would risk re-serializing it differently.
    const blob = new Blob([res.data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    $("ce-result").innerHTML =
      alertHtml("success", `Exported ${nChunks} points from '${esc(target)}'.`);
  }));
}

// ── Points ────────────────────────────────────────────────────────────────────

const PT_TABS = ["ingest", "upsert", "batch", "get", "delete", "payload", "count"];

function parseJsonInto(raw, resultId, label = "Invalid JSON", controlId = null) {
  try {
    const value = JSON.parse(raw);
    associateFieldError($(controlId), resultId, false);
    return value;
  } catch (err) {
    $(resultId).innerHTML = alertHtml("error", `${label}: ${err.message}`);
    associateFieldError($(controlId), resultId, true);
    return undefined;
  }
}

// Points-tab mutation rendering: a clear ok/ko message, never a JSON dump.
// `target` is the collection the operation was issued against — responses
// arriving after the user switched collection are discarded.
function renderPointsResult(containerId, target, result, successMsg) {
  if (target !== state.active) return;
  if (result.ok) {
    $(containerId).innerHTML = alertHtml("success", successMsg);
    // Every mutation makes the Collections report (points count) and the
    // sampled payload keys stale.
    loadReport();
    loadCollectionFields();
  } else {
    $(containerId).innerHTML = apiErrorHtml(result);
  }
}

function fileStem(name) {
  const base = String(name).split("/").pop();
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

function parsedBulkIds() {
  return $("del-ids").value
    .replaceAll(",", "\n")
    .split("\n")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

function updatePointsUI() {
  $("cnt-submit").disabled = $("cnt-submit").dataset.busy === "true"
    || !state.servicesReady || !state.active;
  const embeddingGate = activeEmbeddingGate();
  const reindexBlocked = activeReindexBlocksMutations();
  const reindexReason =
    "Point mutations are temporarily disabled while vector regeneration is active.";
  const vectorWriteReason = reindexBlocked ? reindexReason : embeddingGate.message;
  const vectorWriteBlocked = reindexBlocked || embeddingGate.blocked;
  $("pt-embedding-guard").innerHTML = reindexBlocked
    ? alertHtml(
      "warning",
      `${reindexReason} Point retrieval remains available.`,
    )
    : (embeddingGate.status === "loading"
      ? alertHtml("info", "Checking collection compatibility with the active embedder…")
      : "");

  const ingestPreflightOk = Boolean(state.ingestPreflight)
    && state.ingestPreflight.summary.total > 0
    && state.ingestPreflight.summary.invalid === 0
    && state.ingestPreflight.mappingValid !== false;
  $("ing-confirm-submit").hidden = !ingestPreflightOk;
  $("ing-confirm-submit").disabled = $("ing-confirm-submit").dataset.busy === "true"
    || !state.servicesReady
    || vectorWriteBlocked
    || !ingestPreflightOk;

  $("up-submit").disabled = $("up-submit").dataset.busy === "true"
    || !state.servicesReady
    || vectorWriteBlocked
    || !$("up-id").value.trim()
    || !$("up-text").value.trim();
  $("bt-submit").disabled = $("bt-submit").dataset.busy === "true"
    || !state.servicesReady
    || vectorWriteBlocked;
  $("get-submit").disabled = $("get-submit").dataset.busy === "true"
    || !state.servicesReady
    || !$("get-id").value.trim();
  $("del-submit").disabled = $("del-submit").dataset.busy === "true"
    || !state.servicesReady
    || reindexBlocked
    || !$("del-id").value.trim();

  const ids = parsedBulkIds();
  // Only meaningful once something has been typed; "0 id(s)" at rest is noise.
  $("del-count").hidden = !ids.length;
  $("del-count").textContent = ids.length ? `${ids.length} id(s) parsed.` : "";
  $("del-confirm-label").innerHTML =
    `Type '<span class="literal">${esc(state.active || "the collection name")}</span>' to confirm bulk delete`;
  const bulkConfirmOk = confirmationSatisfied($("del-confirm").value, state.active);
  $("del-bulk-submit").disabled = $("del-bulk-submit").dataset.busy === "true"
    || !state.servicesReady
    || reindexBlocked
    || !(ids.length && bulkConfirmOk);

  // Payload scope: one point by id, or every point matching a filter. REPLACE
  // stays point-scoped — a full replace carries that point's own text/source_id
  // forward, which one bulk write cannot do across many points.
  const plScope = document.querySelector('input[name="pl-scope"]:checked')?.value || "point";
  const plByFilter = plScope === "filter";
  const plFilterText = $("pl-filter-json").value.trim();
  const plHasFilter = plFilterText.length > 0 && plFilterText !== "{}";
  $("pl-id-field").hidden = plByFilter;
  $("pl-filter-field").hidden = !plByFilter;
  const plReplace = document.querySelector('input[name="pl-mode"]:checked')?.value === "PUT";
  const plReplaceRadio = document.querySelector('input[name="pl-mode"][value="PUT"]');
  if (plReplaceRadio) plReplaceRadio.disabled = plByFilter;
  if (plByFilter && plReplace) {
    document.querySelector('input[name="pl-mode"][value="PATCH"]').checked = true;
  }
  $("pl-filter-count-submit").disabled =
    $("pl-filter-count-submit").dataset.busy === "true"
    || !state.servicesReady
    || !plHasFilter;
  $("pl-submit").disabled = $("pl-submit").dataset.busy === "true"
    || !state.servicesReady
    || reindexBlocked
    || (plByFilter ? !plHasFilter : !$("pl-id").value.trim());

  // Get many by ids — read-only, gated only on service readiness.
  const getIds = parseTokenList($("get-ids").value);
  $("get-many-count").hidden = !getIds.length;
  $("get-many-count").textContent = getIds.length ? `${getIds.length} id(s) parsed.` : "";
  $("get-many-submit").disabled = $("get-many-submit").dataset.busy === "true"
    || !state.servicesReady
    || !getIds.length;

  // Facet — read-only, same gating as Count: only needs a key to ask about.
  $("cnt-facet-submit").disabled = $("cnt-facet-submit").dataset.busy === "true"
    || !$("cnt-facet-key").value;

  // Delete by filter — count is read-only; the delete needs a match + confirm.
  const filterText = $("delf-json").value.trim();
  const hasFilterText = filterText.length > 0 && filterText !== "{}";
  $("delf-count-submit").disabled = $("delf-count-submit").dataset.busy === "true"
    || !state.servicesReady
    || !hasFilterText;
  $("delf-confirm-label").innerHTML =
    `Type '<span class="literal">${esc(state.active || "the collection name")}</span>' to confirm delete by filter`;
  const filterConfirmOk = confirmationSatisfied($("delf-confirm").value, state.active);
  $("delf-submit").disabled = $("delf-submit").dataset.busy === "true"
    || !state.servicesReady
    || reindexBlocked
    || !(hasFilterText && filterConfirmOk);

  // Payload — delete keys and clear address one point by id.
  const plId = $("pl-id").value.trim();
  const payloadKeys = parseTokenList($("pl-keys").value);
  $("pl-keys-count").hidden = !payloadKeys.length;
  $("pl-keys-count").textContent = payloadKeys.length ? `${payloadKeys.length} key(s) parsed.` : "";
  $("pl-del-keys-submit").disabled = $("pl-del-keys-submit").dataset.busy === "true"
    || !state.servicesReady
    || reindexBlocked
    || !payloadKeys.length
    || (plByFilter ? !plHasFilter : !plId);
  $("pl-clear-confirm-label").innerHTML =
    `Type '<span class="literal">${esc(state.active || "the collection name")}</span>' to confirm clear payload`;
  const clearConfirmOk = confirmationSatisfied($("pl-clear-confirm").value, state.active);
  $("pl-clear-submit").disabled = $("pl-clear-submit").dataset.busy === "true"
    || !state.servicesReady
    || reindexBlocked
    || !clearConfirmOk
    || (plByFilter ? !plHasFilter : !plId);

  setDisabledReason(
    ["ing-confirm-submit", "up-submit", "bt-submit"],
    vectorWriteBlocked ? vectorWriteReason : "",
  );
  setDisabledReason(
    ["del-submit", "del-bulk-submit", "delf-submit", "pl-submit",
      "pl-del-keys-submit", "pl-clear-submit"],
    reindexBlocked ? reindexReason : "",
  );
  operationReasons({
    "ing-confirm-submit": vectorWriteBlocked ? vectorWriteReason : "Upload a valid chunks file and correct its field mapping.",
    "up-submit": vectorWriteBlocked ? vectorWriteReason : "Enter a point ID and nonempty text.",
    "bt-submit": vectorWriteBlocked ? vectorWriteReason : "Select a collection to upsert points.",
    "get-submit": "Enter a point ID.", "get-many-submit": "Enter at least one point ID.",
    "del-submit": reindexBlocked ? reindexReason : "Enter a point ID to delete.",
    "del-bulk-submit": reindexBlocked ? reindexReason : "Enter point IDs and the exact collection name to confirm deletion.",
    "delf-count-submit": "Enter a nonempty filter to count matching points.",
    "delf-submit": reindexBlocked ? reindexReason : "Enter a nonempty filter and the exact collection name to confirm deletion.",
    "pl-filter-count-submit": "Enter a nonempty payload filter.",
    "pl-submit": reindexBlocked ? reindexReason : "Enter a point ID or a nonempty filter for the selected scope.",
    "pl-del-keys-submit": reindexBlocked ? reindexReason : "Enter payload keys and a point ID or nonempty filter.",
    "pl-clear-submit": reindexBlocked ? reindexReason : "Enter a target and the exact collection name to confirm clearing payload.",
    "cnt-submit": "Select a collection to count points.",
    "cnt-facet-submit": "Select an indexed field for facets.",
  });
}

function initPoints() {
  fillReference("points-reference", POINTS_HELP);

  const subtabs = document.querySelectorAll("[data-pt-tab]");
  for (const button of subtabs) {
    button.addEventListener("click", () => {
      for (const other of subtabs) {
        other.setAttribute("aria-selected", String(other === button));
      }
      for (const t of PT_TABS) {
        $(`pt-${t}`).hidden = t !== button.dataset.ptTab;
      }
    });
  }
  initTablistKeyboard(document.querySelector(".subtabs"));

  $("panel-points").addEventListener("input", updatePointsUI);

  // Ingest — the native file input is visually hidden (its browser-localized
  // "choose file / no file chosen" text is replaced by this English button).
  // Selecting a file runs the preflight automatically (no separate Analyze
  // step); source_id is derived from the file name, batch_size is fixed, and
  // ingest is always additive (no destructive replace_source).
  $("ing-file-btn").addEventListener("click", () => $("ing-file").click());
  $("ing-file").addEventListener("change", async () => {
    const file = $("ing-file").files[0] || null;
    state.ingestFile = file;
    state.ingestPreflight = null;
    // Picking a new file abandons whatever the previous run is still doing,
    // including an index phase that outlived its upload.
    state.ingestRunId += 1;
    const fileRunId = state.ingestRunId;
    // Each new file starts from the default: create the indexes it types.
    state.ingestCreateIndexes = true;
    $("ing-create-indexes").checked = true;
    clearIngestCompleteTimer();
    $("ing-preflight-result").hidden = false;
    $("ing-preflight-result").innerHTML = "";
    $("ing-result").innerHTML = "";
    if (!file) {
      updatePointsUI();
      return;
    }
    let obj;
    try {
      obj = JSON.parse(await file.text());
    } catch {
      if (fileRunId !== state.ingestRunId) return;
      $("ing-result").innerHTML = alertHtml("error",
        `'${file.name}' is not a valid JSON file.`);
      updatePointsUI();
      return;
    }
    if (fileRunId !== state.ingestRunId) return;
    const sourceId = fileStem(file.name);
    const summary = summarizeIngestChunks(obj);
    state.ingestPreflight = {
      fileName: file.name,
      chunks: Array.isArray(obj.chunks) ? obj.chunks : [],
      summary,
    };
    renderIngestPreflight(summary, { active: state.active, sourceId, fileName: file.name });
    validateIngestMapping();
  });

  // Live re-validation on any mapping edit — delegated so it survives the
  // table being re-rendered on each new upload.
  $("ing-preflight-result").addEventListener("input", (e) => {
    if (!e.target.classList.contains("ing-map-name")) return;
    // The rename may have moved the row onto (or off) an existing index.
    syncIngestIndexLock(e.target.closest("tr"));
    validateIngestMapping();
  });
  $("ing-preflight-result").addEventListener("change", (e) => {
    if (!e.target.classList.contains("ing-map-type")) return;
    // Remember the pick so a temporary index lock can hand it back later.
    e.target.dataset.choice = e.target.value;
    validateIngestMapping();
  });

  $("ing-create-indexes").addEventListener("change", () => {
    state.ingestCreateIndexes = $("ing-create-indexes").checked;
  });

  $("ing-confirm-submit").addEventListener("click", () => withBusy($("ing-confirm-submit"), async (current) => {
    const preflight = state.ingestPreflight;
    if (!preflight || preflight.summary.total === 0 || preflight.summary.invalid > 0) return;
    const stem = fileStem(preflight.fileName);
    const { fieldMapping, sourceId: sourceMap, chunkIdKey } = readIngestMapping();
    const { chunks: mappedChunks, errors: fieldErrors } =
      applyIngestFieldMapping(preflight.chunks, fieldMapping);
    const srcErrors = validateSourceIdMapping(sourceMap, fieldMapping, stem);
    const chunkIdErrors = validateChunkIdMapping(chunkIdKey, fieldMapping, sourceMap);
    const mapErrors = [...srcErrors, ...chunkIdErrors, ...fieldErrors];
    if (mapErrors.length) {
      $("ing-map-errors").innerHTML =
        mapErrors.map((e) =>
          `<p class="caption warn">${esc(uppercaseMessage(e))}</p>`).join("");
      return;
    }
    clearIngestCompleteTimer();
    // Identifies this upload for the async index phase, which outlives the
    // preflight object the upload loop guards on.
    const runId = ++state.ingestRunId;
    const batchSize = DEFAULTS.ingestBatchSize;
    const groups = chunkArray(mappedChunks, batchSize);
    const total = mappedChunks.length;
    const target = state.active;
    // Hide the table (DOM/edits preserved) so the progress replaces it.
    $("ing-preflight-result").hidden = true;
    let done = 0;
    let skipped = 0;
    renderIngestProgress({ status: "ingesting", done, total, skipped });
    for (const group of groups) {
      const payload = buildIngestChunksPayload(
        group, target, batchSize, false, stem, sourceMap.key, sourceMap.type,
        chunkIdKey);
      const result = await api.ingestChunks(payload);
      if (!current(result)) return;
      if (target !== state.active || state.ingestPreflight !== preflight) return;  // collection switched or file swapped mid-upload
      const data = result.ok && result.data && typeof result.data === "object"
        ? result.data : null;
      if (!result.ok || (data && data.ok === false)) {
        const detail = data && data.ok === false
          ? (data.error || "unknown error")
          : apiErrorText(result);
        renderIngestProgress({
          status: "failed", done, total, skipped,
          message: `Ingest failed after ${done} of ${total} chunks — ${detail}`,
        });
        $("ing-preflight-result").hidden = false;  // restore table for retry
        loadReport();
        loadCollectionFields();
        return;
      }
      // Advance by chunks SENT (not the backend's ingested count) so the bar
      // tracks upload position and reaches 100% cleanly; skips are a side tile.
      done += group.length;
      skipped += data && Number.isFinite(data.skipped) ? data.skipped : 0;
      renderIngestProgress({ status: "ingesting", done, total, skipped });
    }
    renderIngestProgress({ status: "completed", done: total, total, skipped });
    // A completed ingest invalidates the preview: clear it so stale chunks
    // cannot be re-sent by re-clicking Confirm. Done before the index phase —
    // the upload is over, and the table must not linger through an extra call.
    state.ingestPreflight = null;
    $("ing-preflight-result").innerHTML = "";
    $("ing-preflight-result").hidden = false;
    updatePointsUI();
    // Every batch landed: create the payload indexes for the typed rows. A
    // failure here never demotes the ingest — it is reported on its own.
    await runIngestIndexPhase(
      { status: "completed", done: total, total, skipped },
      { fieldMapping, sourceMap, chunkIdKey, target, runId, current });
    if (!current() || target !== state.active) return;
    loadReport();
    loadCollectionFields();
    // Show the COMPLETED block briefly, then drop back to the initial state
    // (mirrors the Re-embed 5s auto-reset). The index report shares that
    // lifecycle rather than earning a longer one of its own — the timer starts
    // after the index phase, so the summary gets the full window.
    state.ingestCompleteTimer = setTimeout(() => {
      state.ingestCompleteTimer = null;
      if (!current() || target !== state.active) return;
      $("ing-result").innerHTML = "";
    }, INGEST_RESULT_HOLD_MS);
  }));

  // Upsert
  $("up-submit").addEventListener("click", () => withBusy($("up-submit"), async (current) => {
    const payloadObj = parseJsonInto($("up-payload").value || "{}", "up-result", "Invalid payload JSON", "up-payload");
    if (payloadObj === undefined) return;
    const pointId = $("up-id").value.trim();
    const updateMode = document.querySelector('input[name="up-mode"]:checked')?.value || "upsert";
    const body = buildUpsertPayload(state.active, pointId, $("up-text").value, payloadObj, updateMode);
    const target = state.active;
    const result = await api.upsertPoint(body);
    if (!current(result)) return;
    if (result.ok && result.data?.skipped) {
      $("up-result").innerHTML = alertHtml("warning",
        `Point '${pointId}' ${updateMode === "insert_only" ? "already exists" : "does not exist"} `
        + `— skipped (${updateMode}).`);
      return;
    }
    renderPointsResult("up-result", target, result,
      `Point '${pointId}' upserted into '${target}'.`);
  }));

  // Batch
  $("bt-submit").addEventListener("click", () => withBusy($("bt-submit"), async (current) => {
    const points = parseJsonInto($("bt-json").value || "[]", "bt-result", "Invalid JSON", "bt-json");
    if (points === undefined) return;
    if (!Array.isArray(points) || !points.length) {
      $("bt-result").innerHTML = alertHtml("error", "Input must be a non-empty JSON array of points.");
      associateFieldError($("bt-json"), "bt-result", true);
      return;
    }
    const target = state.active;
    const updateMode = document.querySelector('input[name="bt-mode"]:checked')?.value || "upsert";
    const idError = validateBatchPointIds(points);
    if (idError) {
      $("bt-result").innerHTML = alertHtml("error", idError);
      associateFieldError($("bt-json"), "bt-result", true);
      return;
    }
    const result = await api.upsertBatch(buildBatchPayload(target, points, updateMode));
    if (!current(result)) return;
    const upserted = result.data?.upserted;
    const skipped = result.data && typeof result.data === "object"
      ? (result.data.skipped || 0) : 0;
    renderPointsResult("bt-result", target, result,
      `${upserted} point(s) upserted into '${target}'`
      + (skipped ? `, ${skipped} skipped (${updateMode}).` : "."));
  }));

  // Get
  $("get-submit").addEventListener("click", () => withBusy($("get-submit"), async (current) => {
    const target = state.active;
    const withVectors = $("get-with-vectors").checked;
    const result = await api.getPoint($("get-id").value.trim(), target, withVectors);
    if (!current(result)) return;
    if (result.ok && result.data !== null && typeof result.data === "object") {
      const point = result.data;
      $("get-result").innerHTML =
        `<p class="caption"><strong>id:</strong> <code>${esc(point.id ?? "—")}</code>` +
        ` · <strong>collection:</strong> ${esc(point.collection_name ?? "—")}</p>` +
        vectorsSummaryHtml(point.vectors) +
        jsonBlock(point.payload || {});
    } else {
      $("get-result").innerHTML = apiErrorHtml(result, "Point not found");
    }
  }));

  // Get many by ids
  $("get-many-submit").addEventListener("click", () => withBusy($("get-many-submit"), async (current) => {
    const ids = parseTokenList($("get-ids").value);
    if (!ids.length) return;
    const target = state.active;
    const withVectors = $("get-many-with-vectors").checked;
    const result = await api.retrievePoints(buildRetrieveBody(target, ids, withVectors));
    if (!current(result)) return;
    if (result.ok && result.data && Array.isArray(result.data.results)) {
      const { returned, results } = result.data;
      const missing = ids.length - returned;
      const head = `<p class="caption"><strong>${returned}</strong> of ${ids.length} id(s) found` +
        `${missing > 0 ? ` · ${missing} not present` : ""}.</p>`;
      $("get-many-result").innerHTML = returned
        ? head + results.map((point) =>
          `<p class="caption"><strong>id:</strong> <code>${esc(point.id ?? "—")}</code></p>` +
          vectorsSummaryHtml(point.vectors) +
          jsonBlock(point.payload || {})).join("")
        : head;
    } else {
      $("get-many-result").innerHTML = apiErrorHtml(result, "Retrieve failed");
    }
  }));

  // Delete
  $("del-submit").addEventListener("click", () => withBusy($("del-submit"), async (current) => {
    const pointId = $("del-id").value.trim();
    const target = state.active;
    const result = await api.deletePoint(pointId, target);
    if (!current(result)) return;
    renderPointsResult("del-result", target, result,
      `Point '${pointId}' deleted from '${target}'.`);
    if (result.ok) $("del-id").value = "";
  }));
  wireConfirm("del-confirm", "del-confirm-hint", () => state.active, updatePointsUI);
  $("del-bulk-submit").addEventListener("click", () => withBusy($("del-bulk-submit"), async (current) => {
    const ids = parsedBulkIds();
    const target = state.active;
    const result = await api.deletePoints(buildBulkDeleteBody(target, ids));
    if (!current(result)) return;
    const deleted = result.data?.deleted;
    renderPointsResult("del-bulk-result", target, result,
      `${deleted} point(s) deleted from '${target}'.`);
    if (result.ok) {
      // Disarm the confirmation so the button cannot fire again by accident.
      $("del-ids").value = "";
      $("del-confirm").value = "";
    }
  }));

  // Delete by filter
  wireConfirm("delf-confirm", "delf-confirm-hint", () => state.active, updatePointsUI);
  $("delf-count-submit").addEventListener("click", () => withBusy($("delf-count-submit"), async (current) => {
    const filters = parseJsonInto($("delf-json").value || "{}", "delf-result", "Invalid filter JSON", "delf-json");
    if (filters === undefined) return;
    const target = state.active;
    const result = await api.scrollPoints(buildFilterCountPayload(target, filters));
    if (!current(result)) return;
    if (result.ok && result.data && typeof result.data === "object") {
      $("delf-count").hidden = false;
      $("delf-count").textContent = `${result.data.total ?? 0} point(s) match this filter.`;
      $("delf-result").innerHTML = "";
    } else {
      $("delf-count").hidden = true;
      $("delf-result").innerHTML = apiErrorHtml(result, "Count failed");
    }
  }));
  $("delf-submit").addEventListener("click", () => withBusy($("delf-submit"), async (current) => {
    const filters = parseJsonInto($("delf-json").value || "{}", "delf-result", "Invalid filter JSON", "delf-json");
    if (filters === undefined) return;
    const target = state.active;
    const result = await api.deletePointsByFilter(buildDeleteByFilterBody(target, filters));
    if (!current(result)) return;
    const deleted = result.data && typeof result.data === "object"
      ? result.data.deleted : "?";
    renderPointsResult("delf-result", target, result,
      `${deleted} point(s) deleted from '${target}' by filter.`);
    if (result.ok) {
      $("delf-json").value = "{}";
      $("delf-confirm").value = "";
      $("delf-count").hidden = true;
    }
  }));

  // Payload scope helpers: every payload write reads the scope from the same
  // radio, so the three handlers below stay symmetric.
  const payloadByFilter = () =>
    document.querySelector('input[name="pl-scope"]:checked')?.value === "filter";
  const payloadFilterObj = (resultId) =>
    parseJsonInto($("pl-filter-json").value || "{}", resultId, "Invalid filter JSON", "pl-filter-json");

  // Count matches for the filter scope — read-only preflight before a bulk write.
  $("pl-filter-count-submit").addEventListener("click", () =>
    withBusy($("pl-filter-count-submit"), async (current) => {
      const filters = payloadFilterObj("pl-result");
      if (filters === undefined) return;
      const target = state.active;
      const result = await api.countPoints(buildCountPayload(target, filters, true));
      if (!current(result)) return;
      $("pl-filter-count").hidden = false;
      $("pl-filter-count").textContent = result.ok && result.data
        ? `${result.data.count} point(s) match in '${target}'.`
        : apiErrorText(result);
    }));

  // Payload update
  $("pl-submit").addEventListener("click", () => withBusy($("pl-submit"), async (current) => {
    const payloadObj = parseJsonInto($("pl-json").value || "{}", "pl-result", "Invalid JSON", "pl-json");
    if (payloadObj === undefined) return;
    const target = state.active;
    if (payloadByFilter()) {
      const filters = payloadFilterObj("pl-result");
      if (filters === undefined) return;
      const result = await api.updatePayloadByFilter(
        buildPayloadFilterUpdateBody(target, filters, payloadObj));
      if (!current(result)) return;
      renderPointsResult("pl-result", target, result,
        `Payload merged into ${result.data?.matched ?? 0} point(s) in '${target}'.`);
      return;
    }
    const method = document.querySelector('input[name="pl-mode"]:checked').value;
    const pointId = $("pl-id").value.trim();
    const body = buildPayloadUpdateBody(target, payloadObj);
    const result = await api.updatePayload(pointId, body, method);
    if (!current(result)) return;
    renderPointsResult("pl-result", target, result,
      method === "PUT"
        ? `Payload replaced on point '${pointId}'.`
        : `Payload merged into point '${pointId}'.`);
  }));

  // Delete payload keys
  $("pl-del-keys-submit").addEventListener("click", () => withBusy($("pl-del-keys-submit"), async (current) => {
    const keys = parseTokenList($("pl-keys").value);
    if (!keys.length) return;
    const target = state.active;
    if (payloadByFilter()) {
      const filters = payloadFilterObj("pl-keys-result");
      if (filters === undefined) return;
      const result = await api.deletePayloadKeysByFilter(
        buildPayloadFilterKeysBody(target, filters, keys));
      if (!current(result)) return;
      renderPointsResult("pl-keys-result", target, result,
        `${keys.length} key(s) removed from ${result.data?.matched ?? 0} point(s) in '${target}'.`);
      if (result.ok) $("pl-keys").value = "";
      return;
    }
    const pointId = $("pl-id").value.trim();
    const result = await api.deletePayloadKeys(pointId, buildPayloadKeysDeleteBody(target, keys));
    if (!current(result)) return;
    renderPointsResult("pl-keys-result", target, result,
      `${keys.length} key(s) removed from point '${pointId}'.`);
    if (result.ok) $("pl-keys").value = "";
  }));

  // Clear payload
  wireConfirm("pl-clear-confirm", "pl-clear-confirm-hint", () => state.active, updatePointsUI);
  $("pl-clear-submit").addEventListener("click", () => withBusy($("pl-clear-submit"), async (current) => {
    const target = state.active;
    if (payloadByFilter()) {
      const filters = payloadFilterObj("pl-clear-result");
      if (filters === undefined) return;
      const result = await api.clearPayloadByFilter(
        buildPayloadFilterClearBody(target, filters));
      if (!current(result)) return;
      renderPointsResult("pl-clear-result", target, result,
        `Payload cleared on ${result.data?.matched ?? 0} point(s) in '${target}' `
        + "(protected fields kept).");
      if (result.ok) $("pl-clear-confirm").value = "";
      return;
    }
    const pointId = $("pl-id").value.trim();
    const result = await api.clearPayload(pointId, buildClearPayloadBody(target));
    if (!current(result)) return;
    renderPointsResult("pl-clear-result", target, result,
      `Payload cleared on point '${pointId}' (protected fields kept).`);
    if (result.ok) $("pl-clear-confirm").value = "";
  }));

  // Count points — read-only, never adjacent to a destructive action.
  $("cnt-submit").addEventListener("click", () => withBusy($("cnt-submit"), async (current) => {
    const raw = $("cnt-json").value.trim();
    // An empty filter is meaningful here: it counts the whole collection.
    const filters = raw === "" || raw === "{}"
      ? null
      : parseJsonInto(raw, "cnt-result", "Invalid filter JSON", "cnt-json");
    if (filters === undefined) return;
    const target = state.active;
    const exact = $("cnt-exact").checked;
    const result = await api.countPoints(buildCountPayload(target, filters, exact));
    if (!current(result)) return;
    $("cnt-result").innerHTML = result.ok && result.data
      ? alertHtml("success",
        `${result.data.count} point(s) in '${esc(target)}'`
        + `${filters ? " matching the filter" : ""} — ${exact ? "exact" : "estimated"}.`)
      : apiErrorHtml(result, "Count failed");
  }));

  // Facet — read-only, listed key options track the active collection's
  // keyword/integer/bool/uuid indexes (loadCollectionFields/renderFacetKeyOptions).
  $("cnt-facet-submit").addEventListener("click", () => withBusy($("cnt-facet-submit"), async (current) => {
    const key = $("cnt-facet-key").value;
    if (!key) return;
    const target = state.active;
    const limit = $("cnt-facet-limit").valueAsNumber || 20;
    const exact = $("cnt-facet-exact").checked;
    const result = await api.facetPoints(buildFacetPayload(target, key, limit, exact));
    if (!current(result)) return;
    if (result.ok && result.data && Array.isArray(result.data.hits)) {
      const { hits } = result.data;
      $("cnt-facet-result").innerHTML = hits.length
        ? `<div class="table-wrap"><table>
          <thead><tr><th scope="col">value</th><th scope="col">count</th></tr></thead>
          <tbody>${hits.map((hit) =>
            `<tr><td>${esc(String(hit.value))}</td><td>${esc(hit.count)}</td></tr>`).join("")}</tbody>
          </table></div>`
        : alertHtml("info", "No values found.");
    } else {
      $("cnt-facet-result").innerHTML = apiErrorHtml(result, "Facet failed");
    }
  }));

  updatePointsUI();
}

// ── Monitoring ────────────────────────────────────────────────────────────────

const SERIES_MAX = 120;

// Status badge tile: same look as metricTiles, with the up/down state color
// on the value.
function statusTile(label, ok, detail = "") {
  return `<div class="metric"><div class="metric-label">${esc(label)}</div>` +
    `<div class="metric-value ${ok ? "up" : "down"}">` +
    `${esc(`${ok ? "UP" : "DOWN"}${detail ? ` ${detail}` : ""}`)}</div></div>`;
}

// Chart geometry shared by the renderer and the hover layer. The series color
// (#3f92f5) is the app blue snapped into the dark-mode lightness band and
// validated for chroma and 3:1 contrast against the panel surface.
// Width follows the container at render time (1 SVG unit = 1 CSS pixel) so the
// plot always fills the panel instead of being letterboxed by a fixed viewBox.
let CHART_W = 600;
const CHART_H = 340;
const CHART_PAD = { top: 22, right: 16, bottom: 26, left: 46 };

function renderMonitorChart() {
  const plot = $("mon-chart-plot");
  CHART_W = Math.max(320, plot.clientWidth || 600);
  plot.innerHTML = chartSvg(state.monitorSeries);
}

const fmtCompact = new Intl.NumberFormat("en", {
  notation: "compact", maximumFractionDigits: 1,
});

function chartScales(series) {
  const values = series.map((s) => s.requests);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const innerW = CHART_W - CHART_PAD.left - CHART_PAD.right;
  const innerH = CHART_H - CHART_PAD.top - CHART_PAD.bottom;
  return {
    min, max,
    x: (i) => CHART_PAD.left + (i / (series.length - 1)) * innerW,
    y: (v) => CHART_H - CHART_PAD.bottom - ((v - min) / span) * innerH,
  };
}

function chartSvg(series) {
  if (series.length < 2) {
    return `<p class="caption">Collecting samples — the chart appears after the second refresh.</p>`;
  }
  const { min, max, x, y } = chartScales(series);
  const pts = series.map((s, i) => `${x(i).toFixed(1)},${y(s.requests).toFixed(1)}`);
  const first = series[0];
  const last = series[series.length - 1];
  const lastX = x(series.length - 1).toFixed(1);
  const lastY = y(last.requests).toFixed(1);
  const baseline = (CHART_H - CHART_PAD.bottom).toFixed(1);
  // Three recessive gridlines with value ticks; one line when the series is flat.
  const ticks = max === min ? [min] : [min, (min + max) / 2, max];
  const grid = ticks.map((v) =>
    `<line x1="${CHART_PAD.left}" y1="${y(v).toFixed(1)}" x2="${CHART_W - CHART_PAD.right}" y2="${y(v).toFixed(1)}" class="chart-grid" />` +
    `<text x="${CHART_PAD.left - 8}" y="${(y(v) + 3).toFixed(1)}" class="chart-tick" text-anchor="end">${esc(fmtCompact.format(v))}</text>`
  ).join("");
  return `<svg viewBox="0 0 ${CHART_W} ${CHART_H}" role="img"
      aria-label="Total requests over time, from ${esc(String(first.requests))} at ${esc(first.t)} to ${esc(String(last.requests))} at ${esc(last.t)}">
    ${grid}
    <path class="chart-area" d="M${pts.join(" L")} L${lastX},${baseline} L${x(0).toFixed(1)},${baseline} Z" />
    <polyline class="chart-line" points="${pts.join(" ")}" />
    <circle class="chart-dot" cx="${lastX}" cy="${lastY}" r="3.5" />
    <text x="${CHART_PAD.left}" y="12" class="chart-title">total requests</text>
    <text x="${CHART_W - CHART_PAD.right}" y="12" class="chart-value" text-anchor="end">${esc(fmtCompact.format(last.requests))}</text>
    <text x="${CHART_PAD.left}" y="${CHART_H - 6}" class="chart-tick">${esc(first.t)}</text>
    <text x="${CHART_W - CHART_PAD.right}" y="${CHART_H - 6}" class="chart-tick" text-anchor="end">${esc(last.t)}</text>
  </svg>`;
}

// Crosshair + tooltip on the requests chart. Listeners live on the persistent
// wrapper; the SVG inside is replaced on every refresh tick.
async function loadOpenapiPaths() {
  const result = await api.openapi();
  const paths = result.ok && result.data && typeof result.data === "object"
    ? result.data.paths : null;
  if (paths && typeof paths === "object") {
    state.openapiPaths = new Set(Object.keys(paths));
  }
}

function initMonitorReset() {
  $("mon-reset").addEventListener("click", () => {
    state.statsBaseline = { ...(state.lastRawStats || { total: 0, rejected: 0 }) };
    state.endpointBaseline = new Map((state.lastRawRows || []).map((row) =>
      [row.endpoint, { requests: row.requests, errors: row.errors }]));
    state.monitorSeries = [];
    renderMonitorChart();
    monitorTick();
  });
}

function initMonitorChart() {
  const wrap = $("mon-chart");
  const cross = $("mon-chart-cross");
  const tip = $("mon-chart-tip");
  const hide = () => { cross.hidden = true; tip.hidden = true; };
  wrap.addEventListener("pointerleave", hide);
  // Redraw at the new width on resize; between resizes the SVG stretches,
  // so the redraw keeps 1 unit = 1 pixel and the text crisp.
  window.addEventListener("resize", () => {
    if (state.monitorSeries.length >= 2 && wrap.clientWidth) {
      hide();
      renderMonitorChart();
    }
  });
  wrap.addEventListener("pointermove", (event) => {
    const svg = wrap.querySelector("svg");
    const series = state.monitorSeries;
    if (!svg || series.length < 2) { hide(); return; }
    const rect = svg.getBoundingClientRect();
    const wrapRect = wrap.getBoundingClientRect();
    const scaleX = rect.width / CHART_W;
    const scaleY = rect.height / CHART_H;
    const innerW = CHART_W - CHART_PAD.left - CHART_PAD.right;
    const vx = (event.clientX - rect.left) / scaleX;
    const idx = Math.max(0, Math.min(series.length - 1,
      Math.round(((vx - CHART_PAD.left) / innerW) * (series.length - 1))));
    const { x } = chartScales(series);
    const px = rect.left - wrapRect.left + x(idx) * scaleX;
    cross.style.left = `${px.toFixed(1)}px`;
    cross.style.top = `${(rect.top - wrapRect.top + CHART_PAD.top * scaleY).toFixed(1)}px`;
    cross.style.height =
      `${((CHART_H - CHART_PAD.top - CHART_PAD.bottom) * scaleY).toFixed(1)}px`;
    cross.hidden = false;
    tip.textContent = `${series[idx].t} · ${series[idx].requests.toLocaleString("en")} requests`;
    tip.hidden = false;
    // Clamp the tooltip inside the wrapper, flipping side near the right edge.
    const tipW = tip.offsetWidth;
    const flip = px + 12 + tipW > wrapRect.width;
    tip.style.left = `${(flip ? px - tipW - 12 : px + 12).toFixed(1)}px`;
    tip.style.top = `${(rect.top - wrapRect.top + 14 * scaleY).toFixed(1)}px`;
  });
}

function metricsTable(rows) {
  if (!rows.length) return alertHtml("info", "No request metrics yet.");
  const body = rows.map((r) =>
    `<tr><td>${esc(r.endpoint)}</td>` +
    `<td class="num">${Math.round(r.requests)}</td>` +
    `<td class="num">${Math.round(r.errors)}</td>` +
    `<td class="num">${r.p50 === null ? "—" : r.p50.toFixed(4)}</td>` +
    `<td class="num">${r.p95 === null ? "—" : r.p95.toFixed(4)}</td></tr>`
  ).join("");
  return `<div class="table-wrap"><table>
    <thead><tr><th scope="col">endpoint</th><th scope="col">requests</th>
    <th scope="col">errors</th><th scope="col">p50 (s)</th><th scope="col">p95 (s)</th></tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

let monitorTickRunning = false;

async function monitorTick() {
  // A slow backend must not pile up overlapping refreshes: each tick would
  // race the previous one and render out of order.
  if (monitorTickRunning) return;
  monitorTickRunning = true;
  try {
    await monitorTickInner();
  } finally {
    monitorTickRunning = false;
  }
}

async function monitorTickInner() {
  const run = healthCoordinator.begin();
  const [livez, readyz, health, stats, metrics] = await Promise.all([
    api.livez(), api.readyz(), api.health(), api.stats(), api.metrics(),
  ]);
  if (!healthCoordinator.canCommit(run)) return;
  applyHealthResult(health);

  const healthDict = health.data !== null && typeof health.data === "object" ? health.data : {};
  const healthOk = health.ok && String(healthDict.status || "").toLowerCase() === "healthy";
  const healthDetail = !healthOk
    ? String(healthDict.detail || healthDict.status || "") : "";
  const embedderState = String(healthDict.embedder || "unknown").toLowerCase();
  const embedderOk = embedderState === "reachable";
  $("mon-badges").innerHTML = `<div class="metric-row">` +
    statusTile("livez", livez.ok) +
    statusTile("readyz", readyz.ok) +
    statusTile("qdrant", String(healthDict.qdrant || "").toLowerCase() === "reachable") +
    statusTile("embedder", embedderOk, embedderState === "unknown" ? "unknown" : "") +
    statusTile("health", healthOk, healthDetail) +
    `</div>`;

  // Model identity as metric badges, consistent with the other panels.
  const models = healthDict.embedder_models || {};
  $("mon-models").innerHTML = embedderOk
    ? metricTiles([
        ["profile", healthDict.embedder_profile || "—"],
        ["dense model", models.dense || "—"],
        ["sparse model", models.sparse || "—"],
        ["reranker model", models.reranker || "—"],
      ])
    : alertHtml("info", embedderState === "unknown"
        ? "Embedder probe not available on this manager."
        : "Embedder unreachable — model info unavailable.");

  if (stats.ok && stats.data !== null && typeof stats.data === "object") {
    const rawTotal = Number(stats.data.total_requests || 0);
    const rawRejected = Number(stats.data.rejected_requests || 0);
    state.lastRawStats = { total: rawTotal, rejected: rawRejected };
    // Counts are shown relative to the last view-only reset; a raw counter
    // below its baseline means the server restarted, so show it as-is.
    const total = Math.max(0, rawTotal - state.statsBaseline.total);
    const rejected = Math.max(0, rawRejected - state.statsBaseline.rejected);
    $("mon-stats").innerHTML = metricTiles([
      ["uptime", stats.data.uptime ?? "—"],
      ["total_requests", total],
      ["rejected_requests", stats.data.rejected_requests === undefined ? "—" : rejected],
    ]);
    state.monitorSeries.push({
      t: new Date().toLocaleTimeString("en-GB"), requests: total,
    });
    state.monitorSeries = state.monitorSeries.slice(-SERIES_MAX);
    renderMonitorChart();
  } else {
    $("mon-stats").innerHTML = alertHtml("error",
      `Stats unavailable${stats.error ? ` — ${stats.error}` : ""}`);
  }

  if (metrics.ok && typeof metrics.data === "string") {
    const rows = parseEndpointMetrics(metrics.data);
    state.lastRawRows = rows;
    const adjusted = rows.map((row) => {
      const base = state.endpointBaseline.get(row.endpoint);
      return base
        ? {
            ...row,
            requests: Math.max(0, row.requests - base.requests),
            errors: Math.max(0, row.errors - base.errors),
          }
        : row;
    });
    // Only documented API endpoints (Swagger paths), and only ones that have
    // actually been called since the last reset.
    const visible = adjusted.filter((row) =>
      row.requests > 0
      && (!state.openapiPaths || state.openapiPaths.has(row.endpoint)));
    $("mon-table").innerHTML = metricsTable(visible);
  } else {
    $("mon-table").innerHTML = alertHtml("error",
      `Metrics unavailable${metrics.error ? ` — ${metrics.error}` : ""}`);
  }
}

function startMonitor() {
  if (state.monitorTimer !== null) return;
  monitorTick();
  state.monitorTimer = setInterval(monitorTick, 5000);
}

function stopMonitor() {
  if (state.monitorTimer !== null) {
    clearInterval(state.monitorTimer);
    state.monitorTimer = null;
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────

// Re-probes everything gated by bearer auth after the key changes, so the
// console recovers (or degrades) without a manual page reload.
function reloadProtectedData() {
  probeHealth();
  refreshCollections().then(() =>
    Promise.all([loadReport(), loadReindexStatus(), loadCollectionFields()]));
}

function init() {
  for (const result of document.querySelectorAll('[id$="-result"], #ql-results, #delf-count, #pl-filter-count')) {
    if (!result.closest('[aria-live], [role="alert"], [role="status"]')) result.setAttribute("aria-live", "polite");
  }
  for (const type of ["input", "change"]) {
    document.addEventListener(type, (event) => {
      // Local validators own these errors, including edits with no value change.
      const errorId = event.target.dataset.validationError;
      if (errorId !== "ql-errors" && errorId !== "ing-map-errors") {
        associateFieldError(event.target, errorId, false);
      }
      invalidateOperations({ input: event.target });
      updateCollectionsUI();
      updatePointsUI();
    }, true);
  }
  fillReference("monitoring-reference", MONITORING_HELP);
  initConnection();
  initQueryLab();
  initCollections();
  initPoints();
  initMonitorChart();
  initMonitorReset();
  loadOpenapiPaths();
  initTabs();
  initApiKey();
  probeHealth();
  refreshCollections().then(() =>
    Promise.all([loadReport(), loadReindexStatus(), loadCollectionFields()]));
}

init();
