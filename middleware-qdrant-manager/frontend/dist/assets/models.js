// ── Pure logic: query building/validation, console builders,
//    Prometheus parsing. Direct port of the Streamlit dashboard's
//    query_models.py / console_models.py. No DOM access here.

// Filter operations per payload index schema, mirroring what Qdrant supports
// for each field type (match: keyword/integer/bool/uuid; range:
// integer/float/datetime; full-text/phrase: text; geo_radius: geo) plus the
// universal conditions values_count (array length) and is_empty / is_null.
export const FILTER_OPS = {
  keyword: ["eq", "any", "except", "values_count", "is_empty", "is_null"],
  integer: ["eq", "any", "except", "gt", "gte", "lt", "lte", "between", "range", "values_count", "is_empty", "is_null"],
  float: ["gt", "gte", "lt", "lte", "between", "range", "values_count", "is_empty", "is_null"],
  bool: ["eq_bool", "values_count", "is_empty", "is_null"],
  text: ["text", "text_any", "phrase", "values_count", "is_empty", "is_null"],
  geo: ["geo_radius", "geo_box", "geo_polygon", "values_count", "is_empty", "is_null"],
  datetime: ["gt", "gte", "lt", "lte", "between", "range", "values_count", "is_empty", "is_null"],
  uuid: ["eq", "any", "except", "values_count", "is_empty", "is_null"],
  point_id: ["id_any"],
  vector: ["has_vector"],
};

export const FILTER_OP_LABELS = {
  eq: "equals",
  any: "any of",
  except: "any value except",
  gt: "greater than",
  gte: "at least (≥)",
  lt: "less than",
  lte: "at most (≤)",
  between: "between",
  range: "custom range",
  text: "contains all words",
  text_any: "contains any word",
  phrase: "exact phrase",
  eq_bool: "is",
  values_count: "array size range",
  geo_radius: "within radius (m)",
  geo_box: "within box",
  geo_polygon: "within polygon",
  id_any: "any of (ids)",
  has_vector: "is present",
  is_empty: "is empty",
  is_null: "is null",
};

export function filterOpsFor(schema) {
  return FILTER_OPS[schema] ?? FILTER_OPS.keyword;
}

// Logical group of a filter row. Qdrant combines the three lists with AND:
// all MUST conditions, at least one SHOULD condition, no MUST NOT condition.
export const FILTER_GROUPS = ["must", "should", "must_not"];

export const FILTER_GROUP_LABELS = {
  must: "MUST",
  should: "SHOULD",
  must_not: "MUST NOT",
};

// Row-level group choices: the three outer groups plus the two same-element
// variants. A "same element" row targets an array path ("diet[].food") and is
// merged with its siblings into one Qdrant nested condition per array root.
export const ROW_GROUPS = ["must", "should", "must_not", "must_nested", "must_not_nested"];

export const ROW_GROUP_LABELS = {
  must: "MUST",
  should: "SHOULD",
  must_not: "MUST NOT",
  must_nested: "MUST · SAME ELEMENT",
  must_not_nested: "MUST NOT · SAME ELEMENT",
};

// What the VALUE cell must render for an operation:
// "single" one input (comma-separated for any/except), "pair" min+max,
// "bool" a true/false select, "geo" lon+lat+radius, "geo_box" the two box
// corners (top-left lon/lat, bottom-right lon/lat), "none" no value at all.
export function filterValueKind(op) {
  if (op === "is_empty" || op === "is_null") return "none";
  if (op === "between") return "pair";
  if (op === "range" || op === "values_count") return "range";
  if (op === "eq_bool") return "bool";
  if (op === "has_vector") return "vector";
  if (op === "geo_radius") return "geo";
  if (op === "geo_box") return "geo_box";
  return "single";
}

// Static fallback shown until payload keys are sampled from the collection.
export const COMMON_PAYLOAD_FIELDS = ["text", "source_id", "type", "Header 1", "Header 2", "token_count"];
export const COMMON_PAYLOAD_METADATA = ["Header 1", "Header 2", "type", "source_id", "token_count"];

// Backend defaults mirrored from qdrant_middleware.core.config (fallbacks when
// the UI runs against a manager with custom env, the backend stays the truth).
export const DEFAULTS = {
  denseOnDisk: true,
  quantization: true,
  sparseIdf: true,
  sparseOnDisk: true,
  payloadOnDisk: true,
  rerankTopK: 20,
  ingestBatchSize: 64,
  ingestMaxBatchSize: 256,
};

export const QUERY_DEFAULTS = Object.freeze({
  queryDefaultLimit: 10,
  queryMaxLimit: 50,
  queryDefaultCandidateLimit: 50,
  queryMaxCandidateLimit: 200,
  queryDefaultFusion: "rrf",
  queryDefaultRerank: true,
  queryDefaultRerankTopK: 20,
  queryMaxRerankTopK: 200,
  queryRerankMaxChars: 1000,
  queryFilterMaxDepth: 20,
  queryFilterMaxConditions: 100,
});

export const RRF_WEIGHT_SCALE_MIN = 0.01;
export const RRF_WEIGHT_SCALE_MAX = 1_000_000;

function configuredInteger(raw, key) {
  const value = Number(raw?.[key]);
  return Number.isInteger(value) && value >= 1 ? value : QUERY_DEFAULTS[key];
}

export function normalizeQueryConfig(raw = {}) {
  const queryMaxLimit = configuredInteger(raw, "queryMaxLimit");
  const queryMaxCandidateLimit = configuredInteger(raw, "queryMaxCandidateLimit");
  const queryMaxRerankTopK = configuredInteger(raw, "queryMaxRerankTopK");
  return {
    queryDefaultLimit: Math.min(
      configuredInteger(raw, "queryDefaultLimit"), queryMaxLimit),
    queryMaxLimit,
    queryDefaultCandidateLimit: Math.min(
      configuredInteger(raw, "queryDefaultCandidateLimit"), queryMaxCandidateLimit),
    queryMaxCandidateLimit,
    queryDefaultFusion: raw?.queryDefaultFusion === "dbsf" ? "dbsf" : "rrf",
    queryDefaultRerank: typeof raw?.queryDefaultRerank === "boolean"
      ? raw.queryDefaultRerank : QUERY_DEFAULTS.queryDefaultRerank,
    queryDefaultRerankTopK: Math.min(
      configuredInteger(raw, "queryDefaultRerankTopK"), queryMaxRerankTopK),
    queryMaxRerankTopK,
    queryRerankMaxChars: Number.isInteger(Number(raw?.queryRerankMaxChars))
      && Number(raw.queryRerankMaxChars) >= 0
      ? Number(raw.queryRerankMaxChars) : QUERY_DEFAULTS.queryRerankMaxChars,
    queryFilterMaxDepth: configuredInteger(raw, "queryFilterMaxDepth"),
    queryFilterMaxConditions: configuredInteger(raw, "queryFilterMaxConditions"),
  };
}

// ── Query state ───────────────────────────────────────────────────────────────

export function defaultQueryState(config = QUERY_DEFAULTS) {
  const queryConfig = normalizeQueryConfig(config);
  return {
    query: "",
    collectionName: "",
    limit: queryConfig.queryDefaultLimit,
    candidateLimit: queryConfig.queryDefaultCandidateLimit,
    fusion: queryConfig.queryDefaultFusion,
    rrfK: null,
    rrfBalance: null,
    rrfWeightScale: 1,
    rerank: queryConfig.queryDefaultRerank,
    rerankTopK: null,
    rerankNormalize: true,
    minShould: 1,
    filterRows: [],          // [{group, fieldName, op, rawValue..rawValue4}]
    payloadFields: [],
  };
}

export function createRequestCoordinator() {
  let revision = 0;
  let runCounter = 0;
  let activeRun = null;
  return {
    changed() {
      revision += 1;
    },
    begin() {
      activeRun = { id: ++runCounter, revision };
      return activeRun;
    },
    snapshot() {
      return { id: runCounter, revision };
    },
    isBusy() {
      return activeRun !== null && activeRun.revision === revision;
    },
    canCommit(run) {
      return activeRun !== null
        && run.id === activeRun.id
        && run.revision === revision;
    },
    isCurrent(run) {
      return run.id === runCounter && run.revision === revision;
    },
    finish(run) {
      if (activeRun !== null && activeRun.id === run.id) activeRun = null;
    },
  };
}

export const createQueryRequestCoordinator = createRequestCoordinator;

const isRecord = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const isCount = (value) => Number.isInteger(value) && value >= 0;
const isString = (value) => typeof value === "string";
const strings = (value) => Array.isArray(value) && value.every(isString);
const nullable = (test, value) => value === null || test(value);
const hasCollection = (value) => isRecord(value) && isString(value.collection_name);
const mutation = (value) => hasCollection(value) && value.ok === true;
const point = (value) => isRecord(value) && isString(value.id)
  && nullable(isRecord, value.payload)
  && (value.vectors === undefined || nullable(isRecord, value.vectors));
const resultPage = (value) => hasCollection(value) && isCount(value.returned)
  && Array.isArray(value.results) && value.results.every(point)
  && value.returned === value.results.length;
const report = (value) => hasCollection(value) && typeof value.exists === "boolean"
  && nullable((v) => typeof v === "boolean", value.schema_ok)
  && nullable(isCount, value.points_count) && nullable(isRecord, value.dense)
  && nullable(isRecord, value.sparse) && isString(value.quantization)
  && [value.findings, value.errors, value.suggested_fixes].every(
    (items) => items === undefined || nullable(Array.isArray, items))
  && nullable((c) => isRecord(c) && isString(c.status), value.embedding_compatibility);
const reindex = (value) => hasCollection(value) && typeof value.task_running === "boolean"
  && nullable((job) => isRecord(job) && isString(job.status)
    && isCount(job.processed_points) && isCount(job.total_points), value.job);

// Validate only the response fields consumed by the console. Additional server
// fields are allowed; transport status and original data remain in the envelope.
export const responseValidators = {
  collections: (v) => isRecord(v) && strings(v.collections),
  report,
  collectionOperation: (v) => mutation(v) && typeof v.created === "boolean" && report(v.report)
    && v.report.collection_name === v.collection_name,
  collectionDelete: (v) => mutation(v) && v.deleted === true,
  indexes: (v) => hasCollection(v) && Array.isArray(v.indexes)
    && v.indexes.every((i) => isRecord(i) && isString(i.field_name) && isString(i.field_schema)),
  keys: (v) => hasCollection(v) && strings(v.keys),
  indexCreate: (v) => hasCollection(v) && typeof v.ok === "boolean"
    && [v.created, v.updated, v.skipped, v.failed].every(strings),
  indexDelete: (v) => mutation(v) && isString(v.field_name) && v.deleted === true,
  reindex,
  reindexAction: (v) => reindex(v) && typeof v.ok === "boolean",
  preflight: (v) => hasCollection(v) && typeof v.can_start === "boolean"
    && [v.total_points, v.eligible_points, v.invalid_points].every(isCount)
    && strings(v.blockers) && strings(v.invalid_point_ids),
  upsert: (v) => mutation(v) && isString(v.point_id) && typeof v.skipped === "boolean",
  batch: (v) => mutation(v) && isCount(v.upserted) && isCount(v.skipped),
  ingest: (v) => hasCollection(v) && typeof v.ok === "boolean"
    && isCount(v.ingested) && isCount(v.skipped) && (v.ok || isString(v.error)),
  point: (v) => hasCollection(v) && point(v),
  retrieve: resultPage,
  scroll: (v) => resultPage(v) && nullable(isCount, v.total)
    && nullable((offset) => isString(offset) || isCount(offset), v.next_page_offset),
  search: (v) => resultPage(v) && isString(v.fusion) && typeof v.reranked === "boolean"
    && isCount(v.candidates) && isCount(v.skipped_without_text)
    && v.results.every((hit) => Number.isFinite(hit.score)
      && (hit.rerank_score === undefined || nullable(Number.isFinite, hit.rerank_score))),
  deleted: (v) => mutation(v) && isCount(v.deleted),
  payload: (v) => mutation(v) && isString(v.point_id),
  payloadFilter: (v) => mutation(v) && isCount(v.matched),
  count: (v) => mutation(v) && isCount(v.count),
  facet: (v) => mutation(v) && isString(v.key) && Array.isArray(v.hits)
    && v.hits.every((hit) => isRecord(hit) && isCount(hit.count)
      && ["string", "number", "boolean"].includes(typeof hit.value)),
  health: (v) => isRecord(v) && [v.status, v.qdrant, v.embedder].every(isString),
  live: (v) => isRecord(v) && v.status === "alive",
  ready: (v) => isRecord(v) && v.status === "ready",
  stats: (v) => isRecord(v) && isCount(v.total_requests) && isCount(v.rejected_requests)
    && isString(v.uptime),
  openapi: (v) => isRecord(v) && isString(v.openapi) && isRecord(v.paths),
  metrics: isString,
  export: (text, name) => {
    let v;
    try { v = JSON.parse(text); } catch { return false; }
    return isRecord(v) && Array.isArray(v.chunks) && v.chunks.every(isRecord)
      && isRecord(v.metadata) && isCount(v.metadata.n_chunks)
      && v.metadata.n_chunks === v.chunks.length && v.metadata.collection_name === name;
  },
};

export function activeVectorNames(report) {
  const active = report?.embedding_compatibility?.active_vectors;
  return active && [active.dense, active.sparse].every((name) => isString(name) && name.length)
    ? [...new Set([active.dense, active.sparse])] : [];
}

export function splitCsvValues(raw) {
  return String(raw)
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

export function parseListInput(raw) {
  const text = String(raw ?? "").trim();
  if (!text) return [[], null];
  if (!text.startsWith("[")) return [splitCsvValues(text), null];
  try {
    const parsed = JSON.parse(text);
    if (!Array.isArray(parsed)) return [null, "value must be a JSON array."];
    return [parsed, null];
  } catch {
    return [null, "value must be valid JSON array syntax."];
  }
}

export const POINT_ID_FIELD = "(point id)";
export const VECTOR_FIELD = "(vector presence)";
export const PAYLOAD_PAGE_SIZE = 5;

// Schema driving a row's operation list. The point-id pseudo-field has its
// own schema so FILTER_OPS can offer id_any only.
export function schemaForField(fieldName, fieldSchemas = {}) {
  const field = String(fieldName ?? "").trim();
  if (field === POINT_ID_FIELD) return "point_id";
  if (field === VECTOR_FIELD) return "vector";
  return fieldSchemas[field] || "keyword";
}

// Split an array path at its first "[]": "diet[].food" → {root: "diet",
// inner: "food"}. Returns null when the path has no "[]" or either side is
// empty (same-element rows require both).
function splitArrayPath(field) {
  const idx = field.indexOf("[]");
  if (idx <= 0) return null;
  const root = field.slice(0, idx);
  let inner = field.slice(idx + 2);
  if (inner.startsWith(".")) inner = inner.slice(1);
  return inner ? { root, inner } : null;
}

// Return every array-object root addressable by a nested filter. For example,
// "orders[].items[].sku" exposes "orders" and "orders[].items".
export function nestedRootsForFields(fields = []) {
  const roots = new Set();
  for (const rawField of fields) {
    const field = String(rawField ?? "");
    let from = 0;
    while (true) {
      const index = field.indexOf("[]", from);
      if (index < 1) break;
      roots.add(field.slice(0, index));
      from = index + 2;
    }
  }
  return [...roots].sort((a, b) => a.localeCompare(b));
}

function parseIntStrict(value) {
  const trimmed = String(value).trim();
  return /^[+-]?\d+$/.test(trimmed) ? parseInt(trimmed, 10) : null;
}

function parseFloatStrict(value) {
  const trimmed = String(value).trim();
  if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(trimmed)) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

// RFC 3339 subset accepted by Qdrant: date required, time and timezone
// optional, 'T' or space separator, seconds with up to 6 decimals.
const DATETIME_RE =
  /^(\d{4})-(\d{2})-(\d{2})(?:[Tt ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d{1,6})?)?(?:[Zz]|[+-](\d{2}):(\d{2}))?)?$/;

// Shape alone lets impossible datetimes through ("2026-13-99" has month 13),
// and Date.parse can't be the gate either: it rolls over short months
// ("2026-02-31" parses). Check each component's range like the backend does.
function isValidDatetime(raw) {
  const match = DATETIME_RE.exec(raw);
  if (!match) return false;
  const [, year, month, day, hour, minute, second, offsetH, offsetM] = match;
  if (Number(year) < 1) return false; // backend datetime parsing has MINYEAR 1
  if (Number(month) < 1 || Number(month) > 12) return false;
  const daysInMonth = new Date(Date.UTC(Number(year), Number(month), 0)).getUTCDate();
  if (Number(day) < 1 || Number(day) > daysInMonth) return false;
  if (hour !== undefined && Number(hour) > 23) return false;
  if (minute !== undefined && Number(minute) > 59) return false;
  if (second !== undefined && Number(second) > 59) return false;
  if (offsetH !== undefined && (Number(offsetH) > 23 || Number(offsetM) > 59)) return false;
  return true;
}

export function parseNumericPointId(raw) {
  if (typeof raw !== "string" && typeof raw !== "number") return [null, null];
  if (typeof raw === "number" && (!Number.isSafeInteger(raw) || raw < 0)) {
    return [null, "numeric ids must be safe unsigned integers; use a quoted ID string for large values."];
  }
  const value = String(raw).trim();
  if (!/^\d+$/.test(value)) return [null, null];
  const number = BigInt(value);
  if (number > 18446744073709551615n) {
    return [null, "numeric ids must fit unsigned 64-bit range."];
  }
  return [number <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value, null];
}

export function validateBatchPointIds(points) {
  for (const [index, point] of points.entries()) {
    const raw = point?.point_id;
    const [, error] = parseNumericPointId(raw);
    if (error) return `Point ${index + 1}: ${error}`;
    if (typeof raw === "number") {
      return `Point ${index + 1}: point_id must be a quoted ID string in batch JSON.`;
    }
  }
  return null;
}

const UUID_RE =
  /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;

// Parse one scalar for a schema: integers stay strict ints (a string "123"
// never matches an integer index), floats accept any numeric literal, uuids
// must be canonical 8-4-4-4-12 hex.
function parseScalar(schema, raw) {
  if (schema === "integer") return parseIntStrict(raw);
  if (schema === "float") return parseFloatStrict(raw);
  const value = String(raw).trim();
  if (!value) return null;
  if (schema === "uuid" && !UUID_RE.test(value)) return null;
  return value;
}

const NUMBER_NAME = { integer: "an integer", float: "a number", uuid: "a UUID" };

// Parse one polygon ring: "lon lat; lon lat; …" with ≥3 distinct points,
// range-checked and auto-closed. Returns [points, error].
function parseRing(text) {
  const points = [];
  for (const pair of text.split(";").map((p) => p.trim()).filter(Boolean)) {
    const nums = pair.split(/[\s,]+/).filter(Boolean).map(parseFloatStrict);
    if (nums.length !== 2 || nums.some((n) => n === null)) {
      return [null, "each polygon point must be 'lon lat'."];
    }
    const [lon, lat] = nums;
    if (lon < -180 || lon > 180) return [null, "lon must be between -180 and 180."];
    if (lat < -90 || lat > 90) return [null, "lat must be between -90 and 90."];
    points.push({ lon, lat });
  }
  const distinct = new Set(points.map((p) => `${p.lon},${p.lat}`));
  if (distinct.size < 3) return [null, "needs at least 3 distinct points."];
  const first = points[0];
  const last = points[points.length - 1];
  // Qdrant requires closed rings (first point == last point).
  if (first.lon !== last.lon || first.lat !== last.lat) points.push({ ...first });
  return [points, null];
}

// Compile one filter row into a middleware advanced-filter condition.
// Returns [condition, error]: [null, null] means "incomplete row, skip".
export function buildFilterCondition(row, schema = "keyword") {
  const field = String(row.fieldName ?? "").trim();
  if (!field) return [null, "filter field is required."];
  const op = row.op;
  const raw = String(row.rawValue ?? "").trim();
  const raw2 = String(row.rawValue2 ?? "").trim();
  const bad = (msg) => [null, `${field}: ${msg}`];

  if (field === POINT_ID_FIELD) {
    const [parts, listError] = parseListInput(raw);
    if (listError) return bad(listError);
    if (!parts.length) return bad("value is required.");
    const ids = [];
    for (const part of parts) {
      const [id, idError] = parseNumericPointId(part);
      if (idError) return bad(idError);
      if (id !== null) {
        ids.push(id);
      } else {
        const value = String(part).trim();
        if (!UUID_RE.test(value)) return bad("ids must be unsigned integers or UUIDs.");
        ids.push(value);
      }
    }
    return [{ has_id: ids }, null];
  }
  if (field === VECTOR_FIELD) {
    if (!raw) return bad("select a vector name.");
    return [{ has_vector: raw }, null];
  }
  if (op === "is_empty" || op === "is_null") {
    return [{ key: field, [op]: true }, null];
  }
  if (op === "eq_bool") {
    if (!raw) return bad("value is required.");
    if (raw !== "true" && raw !== "false") return bad("value must be true or false.");
    return [{ key: field, match: { value: raw === "true" } }, null];
  }
  if (op === "eq") {
    if (!raw) return bad("value is required.");
    const value = parseScalar(schema, raw);
    if (value === null) return bad(`value must be ${NUMBER_NAME[schema] || "set"}.`);
    return [{ key: field, match: { value } }, null];
  }
  if (op === "any" || op === "except") {
    const [parts, listError] = parseListInput(raw);
    if (listError) return bad(listError);
    if (!parts.length) return bad("value is required.");
    if ((schema === "keyword" || schema === "uuid")
        && parts.some((part) => typeof part !== "string")) {
      const expected = schema === "uuid" ? "UUID strings" : "strings";
      return bad("JSON list values must all be " + expected + ".");
    }
    const values = parts.map((part) => parseScalar(schema, part));
    if (values.some((value) => value === null)) {
      return bad(`values must all be ${NUMBER_NAME[schema] || "set"}.`);
    }
    return [{ key: field, match: { [op]: values } }, null];
  }
  if (op === "gt" || op === "gte" || op === "lt" || op === "lte") {
    if (!raw) return bad("value is required.");
    if (schema === "datetime") {
      if (!isValidDatetime(raw)) return bad("bound must be an RFC 3339 datetime.");
      return [{ key: field, range: { [op]: raw } }, null];
    }
    const value = parseScalar(schema === "float" ? "float" : "integer", raw);
    if (value === null) return bad(`bound must be ${NUMBER_NAME[schema] || "a number"}.`);
    return [{ key: field, range: { [op]: value } }, null];
  }
  if (op === "between") {
    if (!raw && !raw2) return bad("value is required.");
    if (!raw || !raw2) return bad("between needs both min and max.");
    if (schema === "datetime") {
      // No min>max check: lexicographic comparison across timezone formats
      // is unreliable; Qdrant handles empty ranges gracefully.
      if (!isValidDatetime(raw) || !isValidDatetime(raw2)) {
        return bad("bounds must be RFC 3339 datetimes.");
      }
      return [{ key: field, range: { gte: raw, lte: raw2 } }, null];
    }
    const numberSchema = schema === "float" ? "float" : "integer";
    const min = parseScalar(numberSchema, raw);
    const max = parseScalar(numberSchema, raw2);
    if (min === null || max === null) {
      return bad(`bounds must be ${NUMBER_NAME[schema] || "numbers"}.`);
    }
    if (min > max) return bad("between min cannot exceed max.");
    return [{ key: field, range: { gte: min, lte: max } }, null];
  }
  if (op === "range") {
    if (!raw && !raw2) return bad("value is required.");
    const lowerOp = row.lowerOp === "gt" ? "gt" : "gte";
    const upperOp = row.upperOp === "lt" ? "lt" : "lte";
    const bounds = {};
    if (raw) {
      if (schema === "datetime") {
        if (!isValidDatetime(raw)) return bad("lower bound must be an RFC 3339 datetime.");
        bounds[lowerOp] = raw;
      } else {
        const value = parseScalar(schema === "float" ? "float" : "integer", raw);
        if (value === null) return bad("lower bound must be a number.");
        bounds[lowerOp] = value;
      }
    }
    if (raw2) {
      if (schema === "datetime") {
        if (!isValidDatetime(raw2)) return bad("upper bound must be an RFC 3339 datetime.");
        bounds[upperOp] = raw2;
      } else {
        const value = parseScalar(schema === "float" ? "float" : "integer", raw2);
        if (value === null) return bad("upper bound must be a number.");
        bounds[upperOp] = value;
      }
    }
    const lower = bounds.gt ?? bounds.gte;
    const upper = bounds.lt ?? bounds.lte;
    if (lower !== undefined && upper !== undefined && lower > upper) {
      return bad("range lower bound cannot exceed upper bound.");
    }
    return [{ key: field, range: bounds }, null];
  }
  if (op === "text" || op === "text_any" || op === "phrase") {
    if (!raw) return bad("value is required.");
    return [{ key: field, match: { [op]: raw } }, null];
  }
  if (op === "values_count") {
    // Array length bounds; either side may be left empty for open ranges.
    if (!raw && !raw2) return bad("value is required.");
    const bounds = {};
    if (raw) {
      const min = parseIntStrict(raw);
      if (min === null || min < 0) return bad("array size min must be a non-negative integer.");
      bounds[row.lowerOp === "gt" ? "gt" : "gte"] = min;
    }
    if (raw2) {
      const max = parseIntStrict(raw2);
      if (max === null || max < 0) return bad("array size max must be a non-negative integer.");
      bounds[row.upperOp === "lt" ? "lt" : "lte"] = max;
    }
    const lower = bounds.gt ?? bounds.gte;
    const upper = bounds.lt ?? bounds.lte;
    if (lower !== undefined && upper !== undefined && lower > upper) {
      return bad("array size min cannot exceed max.");
    }
    return [{ key: field, values_count: bounds }, null];
  }
  if (op === "geo_box") {
    const raw3 = String(row.rawValue3 ?? "").trim();
    const raw4 = String(row.rawValue4 ?? "").trim();
    if (!raw && !raw2 && !raw3 && !raw4) return bad("value is required.");
    const lon1 = parseFloatStrict(raw);
    const lat1 = parseFloatStrict(raw2);
    const lon2 = parseFloatStrict(raw3);
    const lat2 = parseFloatStrict(raw4);
    if (lon1 === null || lat1 === null || lon2 === null || lat2 === null) {
      return bad("box needs numeric top-left lon/lat and bottom-right lon/lat.");
    }
    if (lon1 < -180 || lon1 > 180 || lon2 < -180 || lon2 > 180) {
      return bad("lon must be between -180 and 180.");
    }
    if (lat1 < -90 || lat1 > 90 || lat2 < -90 || lat2 > 90) {
      return bad("lat must be between -90 and 90.");
    }
    if (lat1 < lat2) return bad("top-left lat must be north of (≥) bottom-right lat.");
    return [{
      key: field,
      geo_bounding_box: {
        top_left: { lon: lon1, lat: lat1 },
        bottom_right: { lon: lon2, lat: lat2 },
      },
    }, null];
  }
  if (op === "geo_polygon") {
    if (!raw) return bad("value is required.");
    // Rings separated by "|": first is the exterior, the rest are holes.
    const rings = raw.split("|").map((r) => r.trim()).filter(Boolean);
    const parsed = [];
    for (const [index, ring] of rings.entries()) {
      const [points, ringError] = parseRing(ring);
      if (ringError) {
        return bad(rings.length > 1 ? `polygon ring ${index + 1}: ${ringError}` : ringError);
      }
      parsed.push({ points });
    }
    const geoPolygon = { exterior: parsed[0] };
    if (parsed.length > 1) geoPolygon.interiors = parsed.slice(1);
    return [{ key: field, geo_polygon: geoPolygon }, null];
  }
  if (op === "geo_radius") {
    const raw3 = String(row.rawValue3 ?? "").trim();
    if (!raw && !raw2 && !raw3) return bad("value is required.");
    const lon = parseFloatStrict(raw);
    const lat = parseFloatStrict(raw2);
    const radius = parseFloatStrict(raw3);
    if (lon === null || lat === null || radius === null) {
      return bad("geo radius needs numeric lon, lat, and radius.");
    }
    if (lon < -180 || lon > 180) return bad("lon must be between -180 and 180.");
    if (lat < -90 || lat > 90) return bad("lat must be between -90 and 90.");
    if (radius <= 0) return bad("radius must be positive (meters).");
    return [{ key: field, geo_radius: { center: { lon, lat }, radius } }, null];
  }
  return bad(`operation '${op}' is not supported. Select another operation.`);
}

// Compile all rows into per-group condition lists: [filter, errors].
// ``filter`` is {must?, should?, must_not?, min_should?} with only non-empty
// lists, or null when no row produced a condition. Incomplete rows are
// skipped. Same-element rows (group must_nested / must_not_nested) are
// bucketed by (outer group, array root) and each bucket becomes one Qdrant
// nested condition whose inner filter ANDs the bucket's conditions. When
// minShouldCount > 1 the should list migrates to Qdrant's min_should clause.
export function buildFilterGroups(rows, fieldSchemas = {}, minShouldCount = 1) {
  const groups = { must: [], should: [], must_not: [] };
  const nestedBuckets = new Map(); // "outer root" (split at first space)
  const errors = [];
  for (const row of rows) {
    const field = String(row.fieldName ?? "").trim();
    const schema = schemaForField(field, fieldSchemas);
    const rowGroup = ROW_GROUPS.includes(row.group) ? row.group : "must";
    if (rowGroup === "must_nested" || rowGroup === "must_not_nested") {
      if (!field) continue;
      const path = splitArrayPath(field);
      if (!path) {
        errors.push(`${field}: same-element filters need an array path containing "[]".`);
        continue;
      }
      const [condition, error] = buildFilterCondition({ ...row, fieldName: path.inner }, schema);
      if (error) errors.push(error);
      if (!condition) continue;
      const bucketKey = JSON.stringify([
        rowGroup === "must_nested" ? "must" : "must_not",
        path.root,
        String(row.nestedGroupId ?? "legacy"),
      ]);
      if (!nestedBuckets.has(bucketKey)) nestedBuckets.set(bucketKey, []);
      nestedBuckets.get(bucketKey).push(condition);
      continue;
    }
    const [condition, error] = buildFilterCondition(row, schema);
    if (condition) groups[rowGroup].push(condition);
    if (error) errors.push(error);
  }
  for (const [bucketKey, conditions] of nestedBuckets) {
    const [outer, root] = JSON.parse(bucketKey);
    groups[outer].push({ nested: { key: root, filter: { must: conditions } } });
  }
  const filter = {};
  for (const group of FILTER_GROUPS) {
    if (groups[group].length) filter[group] = groups[group];
  }
  if (minShouldCount > 1 && filter.should) {
    if (filter.should.length < minShouldCount) {
      errors.push(
        `Minimum SHOULD matches (${minShouldCount}) cannot exceed the number of SHOULD conditions (${filter.should.length}).`);
    } else {
      filter.min_should = { conditions: filter.should, min_count: minShouldCount };
      delete filter.should;
    }
  }
  return [Object.keys(filter).length ? filter : null, errors];
}

function relativeNestedField(fieldName, nestedRoot) {
  if (!nestedRoot) return [String(fieldName ?? "").trim(), null];
  const prefix = `${nestedRoot}[].`;
  const field = String(fieldName ?? "").trim();
  if (!field.startsWith(prefix) || field.length === prefix.length) {
    return [null, `${field || "filter"}: field must be inside ${nestedRoot}[].`];
  }
  return [field.slice(prefix.length), null];
}

function relativeNestedRoot(root, parentRoot) {
  if (!parentRoot) return [root, null];
  const prefix = `${parentRoot}[].`;
  if (!root.startsWith(prefix) || root.length === prefix.length) {
    return [null, `${root}: nested group must be inside ${parentRoot}[].`];
  }
  return [root.slice(prefix.length), null];
}

// Compile the visual filter tree. Each node belongs to one clause in its
// parent; group nodes become recursive Qdrant Filter conditions and nested
// nodes become same-element NestedCondition objects.
export function buildFilterTree(root, fieldSchemas = {}) {
  const errors = [];

  const compileGroup = (group, nestedRoot = null) => {
    const buckets = { must: [], should: [], must_not: [] };
    const children = Array.isArray(group?.children) ? group.children : [];
    if (!children.length && group?.kind !== "root") {
      errors.push(`${group?.label || "Filter group"}: add at least one condition.`);
    }

    for (const node of children) {
      const clause = FILTER_GROUPS.includes(node?.clause) ? node.clause : "must";
      let condition = null;

      if (node?.kind === "condition") {
        const [relativeField, pathError] = relativeNestedField(
          node.fieldName,
          nestedRoot,
        );
        if (pathError) {
          errors.push(pathError);
          continue;
        }
        const schema = schemaForField(node.fieldName, fieldSchemas);
        const [built, error] = buildFilterCondition(
          { ...node, fieldName: relativeField },
          schema,
        );
        condition = built;
        if (error) errors.push(error);
      } else if (node?.kind === "group") {
        condition = compileGroup(node, nestedRoot);
      } else if (node?.kind === "nested") {
        const rootName = String(node.root ?? "").trim();
        if (!rootName) {
          errors.push("Same-element group: select an array field.");
          continue;
        }
        const [relativeRoot, rootError] = relativeNestedRoot(rootName, nestedRoot);
        if (rootError) {
          errors.push(rootError);
          continue;
        }
        const nestedFilter = compileGroup(node, rootName);
        if (nestedFilter) {
          condition = { nested: { key: relativeRoot, filter: nestedFilter } };
        }
      } else {
        errors.push("The filter contains an unsupported node. Remove and recreate the affected filter group.");
      }

      if (condition) buckets[clause].push(condition);
    }

    const filter = {};
    if (buckets.must.length) filter.must = buckets.must;
    if (buckets.must_not.length) filter.must_not = buckets.must_not;
    if (buckets.should.length) {
      const minCount = group?.minShould ?? 1;
      if (!Number.isInteger(minCount) || minCount < 1) {
        errors.push(`${group?.label || "Filter group"}: minimum SHOULD matches must be a positive integer.`);
      } else if (minCount > buckets.should.length) {
        errors.push(
          `${group?.label || "Filter group"}: minimum SHOULD matches (${minCount}) ` +
          `cannot exceed the number of SHOULD conditions (${buckets.should.length}).`,
        );
      } else if (minCount > 1) {
        filter.min_should = {
          conditions: buckets.should,
          min_count: minCount,
        };
      } else {
        filter.should = buckets.should;
      }
    }
    return Object.keys(filter).length ? filter : null;
  };

  if (!root || root.kind !== "root") {
    return [null, ["Filter tree root is invalid."]];
  }
  return [compileGroup(root), errors];
}

export function filterTreeStats(root) {
  let conditions = 0;
  let maxDepth = 1;

  const visitGroup = (group, filterDepth) => {
    maxDepth = Math.max(maxDepth, filterDepth);
    const children = Array.isArray(group?.children) ? group.children : [];
    const shouldCount = children.filter((child) => child.clause === "should").length;
    const usesMinShould = shouldCount > 0 && Number(group?.minShould ?? 1) > 1;
    if (usesMinShould) {
      conditions += 1; // backend counts the MinShould wrapper
      maxDepth = Math.max(maxDepth, filterDepth + 1);
    }
    for (const node of children) {
      conditions += 1;
      const conditionDepth = filterDepth +
        (usesMinShould && node.clause === "should" ? 2 : 1);
      maxDepth = Math.max(maxDepth, conditionDepth);
      if (node.kind === "group") visitGroup(node, conditionDepth + 1);
      if (node.kind === "nested") visitGroup(node, conditionDepth + 2);
    }
  };

  if (root?.kind === "root") visitGroup(root, 1);
  return { conditions, maxDepth };
}

function buildStateFilter(state, fieldSchemas) {
  if (state.filterTree) return buildFilterTree(state.filterTree, fieldSchemas);
  return buildFilterGroups(state.filterRows, fieldSchemas, state.minShould ?? 1);
}

export function buildPayloadFields(state) {
  const fields = [];
  for (const fieldName of state.payloadFields) {
    const stripped = fieldName.trim();
    if (stripped && !fields.includes(stripped)) fields.push(stripped);
  }
  return fields.length ? fields : null;
}

export function rrfWeightsFromBalance(balance, scale = 1) {
  if (typeof balance !== "number"
      || !Number.isFinite(balance) || balance < 0 || balance > 1) {
    throw new RangeError("RRF balance must be between 0 and 1.");
  }
  if (typeof scale !== "number" || !Number.isFinite(scale)
      || scale < RRF_WEIGHT_SCALE_MIN || scale > RRF_WEIGHT_SCALE_MAX) {
    throw new RangeError(
      "RRF weight scale must be between " +
      RRF_WEIGHT_SCALE_MIN + " and " + RRF_WEIGHT_SCALE_MAX + ".");
  }
  return [
    Number((2 * balance * scale).toPrecision(12)),
    Number((2 * (1 - balance) * scale).toPrecision(12)),
  ];
}

export function buildRequestPayload(state, fieldSchemas = {}) {
  const payload = {
    query: state.query.trim(),
    collection_name: state.collectionName.trim(),
    limit: state.limit,
    candidate_limit: state.candidateLimit,
    fusion: state.fusion,
    rerank: state.rerank,
    rerank_normalize: state.rerankNormalize,
  };
  if (state.fusion === "rrf" && state.rrfK !== null) payload.rrf_k = state.rrfK;
  if (state.fusion === "rrf" && state.rrfBalance !== null) {
    payload.rrf_weights = rrfWeightsFromBalance(
      state.rrfBalance, state.rrfWeightScale);
  }
  if (state.rerank && state.rerankTopK !== null) {
    payload.rerank_top_k = state.rerankTopK;
  }
  // Both land on Qdrant's fused result, before any rerank.
  if (state.scoreThreshold !== null && state.scoreThreshold !== undefined) {
    payload.score_threshold = state.scoreThreshold;
  }
  // Sent whenever the toggle is on, including 0: skipping it there would make the
  // request preview disagree with the checkbox (score_threshold already sends 0).
  if (state.offset !== null && state.offset !== undefined) {
    payload.offset = state.offset;
  }

  const [filter] = buildStateFilter(state, fieldSchemas);
  if (filter !== null) payload.filters = filter;

  const payloadFields = buildPayloadFields(state) || [];
  if (state.rerank && !payloadFields.includes("text")) payloadFields.push("text");
  if (payloadFields.length) payload.with_payload = payloadFields;
  return payload;
}

export function buildScrollPayload(state, fieldSchemas = {}) {
  const payload = {
    collection_name: state.collectionName.trim(),
    limit: PAYLOAD_PAGE_SIZE,
  };
  const [filter] = buildStateFilter(state, fieldSchemas);
  if (filter !== null) payload.filters = filter;
  const payloadFields = buildPayloadFields(state);
  if (payloadFields !== null) payload.with_payload = payloadFields;
  const orderBy = buildOrderBy(state);
  if (orderBy !== null) payload.order_by = orderBy;
  return payload;
}

// Qdrant orders a scroll only on a RANGE index, so keyword, bool, geo, uuid and
// text fields are not sortable however they are indexed — asking for one answers
// "No range index for `order_by` key". Only these schemas may reach order_by.
export const ORDERABLE_SCHEMAS = new Set(["integer", "float", "datetime"]);

export function orderableFields(fieldSchemas = {}) {
  return Object.keys(fieldSchemas)
    .filter((field) => ORDERABLE_SCHEMAS.has(fieldSchemas[field]))
    .sort();
}

// Qdrant facets only keyword, integer, bool and uuid indexes — text, geo,
// float and datetime answer "No appropriate index for faceting".
export const FACETABLE_SCHEMAS = new Set(["keyword", "integer", "bool", "uuid"]);

export function facetableFields(fieldSchemas = {}) {
  return Object.keys(fieldSchemas)
    .filter((field) => FACETABLE_SCHEMAS.has(fieldSchemas[field]))
    .sort();
}

// Ordered scroll spec, or null when ordering is off. Qdrant refuses offset with
// order_by, so an ordered page is a single page paged via start_from instead.
export function buildOrderBy(state) {
  const key = String(state.orderByKey ?? "").trim();
  if (!state.orderByEnabled || !key) return null;
  const spec = {
    key,
    direction: state.orderByDirection === "desc" ? "desc" : "asc",
  };
  const raw = String(state.orderByStartFrom ?? "").trim();
  if (raw !== "") {
    const numeric = Number(raw);
    spec.start_from = Number.isFinite(numeric) && /^[+-]?[\d.]+$/.test(raw)
      ? numeric
      : raw;
  }
  return spec;
}

// Bulk payload bodies: the filtered counterparts of the per-point ones.
export function buildPayloadFilterUpdateBody(collectionName, filters, payload) {
  return { collection_name: collectionName, filters, payload };
}

export function buildPayloadFilterKeysBody(collectionName, filters, keys) {
  return { collection_name: collectionName, filters, keys };
}

export function buildPayloadFilterClearBody(collectionName, filters) {
  return { collection_name: collectionName, filters };
}

export function buildCountPayload(collectionName, filters, exact = true) {
  const body = { collection_name: collectionName, exact };
  if (filters !== null && filters !== undefined) body.filters = filters;
  return body;
}

export function buildFacetPayload(collectionName, key, limit = 20, exact = true) {
  return { collection_name: collectionName, key, limit, exact };
}

export function validateQueryState(state, fieldSchemas = {}, config = QUERY_DEFAULTS) {
  const queryConfig = normalizeQueryConfig(config);
  const errors = [];
  const semantic = Boolean(state.query.trim());
  if (!state.collectionName.trim()) errors.push("Collection name is required.");
  // Mirror the backend request bounds so out-of-range values typed directly
  // into the number inputs fail here instead of as an opaque 422.
  if (semantic && (!Number.isInteger(state.limit)
      || state.limit < 1 || state.limit > queryConfig.queryMaxLimit)) {
    errors.push(
      "limit must be an integer between 1 and " + queryConfig.queryMaxLimit + ".");
  }
  if (semantic && (!Number.isInteger(state.candidateLimit)
      || state.candidateLimit < 1
      || state.candidateLimit > queryConfig.queryMaxCandidateLimit)) {
    errors.push(
      "candidate_limit must be an integer between 1 and " +
      queryConfig.queryMaxCandidateLimit + ".");
  }
  if (semantic && state.rrfK !== null && (!Number.isInteger(state.rrfK)
      || state.rrfK < 1 || state.rrfK > 10000)) {
    errors.push("rrf_k must be an integer between 1 and 10000.");
  }
  if (semantic && state.rerankTopK !== null
      && (!Number.isInteger(state.rerankTopK)
        || state.rerankTopK < 1
        || state.rerankTopK > queryConfig.queryMaxRerankTopK)) {
    errors.push(
      "rerank_top_k must be an integer between 1 and " +
      queryConfig.queryMaxRerankTopK + ".");
  }
  if (semantic && state.candidateLimit < state.limit) {
    errors.push("candidate_limit must be greater than or equal to limit.");
  }
  if (semantic && state.rerankTopK !== null && state.rerankTopK > state.candidateLimit) {
    errors.push("rerank_top_k cannot exceed candidate_limit.");
  }
  if (semantic && state.fusion === "dbsf"
      && (state.rrfK !== null || state.rrfBalance !== null)) {
    errors.push("rrf_k and rrf_weights are supported only when fusion is rrf.");
  }
  if (semantic && state.rrfBalance !== null
      && (!Number.isFinite(state.rrfBalance)
        || state.rrfBalance < 0 || state.rrfBalance > 1)) {
    errors.push("RRF balance must be a number between 0 and 1.");
  }
  if (semantic && state.rrfBalance !== null
      && (!Number.isFinite(state.rrfWeightScale)
        || state.rrfWeightScale < RRF_WEIGHT_SCALE_MIN
        || state.rrfWeightScale > RRF_WEIGHT_SCALE_MAX)) {
    errors.push(
      "RRF weight scale must be a number between " +
      RRF_WEIGHT_SCALE_MIN + " and " + RRF_WEIGHT_SCALE_MAX + ".");
  }
  if (!Number.isInteger(state.minShould ?? 1) || (state.minShould ?? 1) < 1) {
    errors.push("Minimum SHOULD matches must be a positive integer.");
  }
  if (semantic && state.scoreThreshold !== null && state.scoreThreshold !== undefined
      && !Number.isFinite(state.scoreThreshold)) {
    errors.push("score_threshold must be a number.");
  }
  if (semantic && state.offset !== null && state.offset !== undefined
      && (!Number.isInteger(state.offset) || state.offset < 0)) {
    errors.push("offset must be a non-negative integer.");
  }
  // A fused hit at position N exists only if it was inside one of the two
  // candidate lists, so a page at `offset` is well-founded only when
  // candidate_limit already covers offset+window. Checked here so the user sees
  // which number to raise rather than an empty or arbitrary page.
  if (semantic && Number.isInteger(state.offset) && state.offset > 0) {
    const window = state.rerank
      ? Math.max(state.limit, state.rerankTopK
        ?? Math.min(queryConfig.queryDefaultRerankTopK, state.candidateLimit))
      : state.limit;
    const needed = state.offset + window;
    if (needed > state.candidateLimit) {
      errors.push(
        `offset ${state.offset} needs candidate_limit ≥ ${needed} (currently ` +
        `${state.candidateLimit}): fusion can only rank a point one of the two ` +
        `branches retrieved. Raise candidate_limit (max ` +
        `${queryConfig.queryMaxCandidateLimit}) or lower the offset.`);
    }
  }
  // An ordered scroll needs a key; the select only lists orderable fields, so an
  // empty value means the collection has none.
  if (!semantic && state.orderByEnabled && !String(state.orderByKey ?? "").trim()) {
    errors.push("order_by needs a payload field with a range index (integer, float or datetime).");
  }
  // start_from is typed by the ordered field: Qdrant accepts a number or a
  // datetime there, never a free string, so a mismatch is caught here instead of
  // travelling to the API as an unusable value.
  const orderKey = String(state.orderByKey ?? "").trim();
  const rawStart = String(state.orderByStartFrom ?? "").trim();
  if (!semantic && state.orderByEnabled && orderKey && rawStart !== "") {
    const schema = fieldSchemas[orderKey];
    if (schema === "integer" && !/^[+-]?\d+$/.test(rawStart)) {
      errors.push(`start_from must be an integer for '${orderKey}'.`);
    } else if (schema === "float" && !Number.isFinite(Number(rawStart))) {
      errors.push(`start_from must be a number for '${orderKey}'.`);
    } else if (schema === "datetime" && Number.isNaN(Date.parse(rawStart))) {
      errors.push(
        `start_from must be a date for '${orderKey}', for example 2026-06-05T00:00:00Z.`);
    }
  }
  const [, filterRowErrors] = buildStateFilter(state, fieldSchemas);
  errors.push(...filterRowErrors);
  if (Array.isArray(state.vectorNames)) {
    const checkVectors = (nodes) => {
      for (const node of nodes) {
        if (node.op === "has_vector" && !state.vectorNames.includes(node.rawValue)) {
          errors.push(`Vector '${node.rawValue || ""}' is unavailable. Select a current collection vector.`);
        }
        if (node.children) checkVectors(node.children);
      }
    };
    checkVectors(state.filterTree?.children || state.filterRows || []);
  }
  if (state.filterTree) {
    const stats = filterTreeStats(state.filterTree);
    if (stats.maxDepth > queryConfig.queryFilterMaxDepth) {
      errors.push(
        `filters exceed maximum depth ${queryConfig.queryFilterMaxDepth}.`);
    }
    if (stats.conditions > queryConfig.queryFilterMaxConditions) {
      errors.push(
        `filters exceed maximum condition count ${queryConfig.queryFilterMaxConditions}.`);
    }
  }
  return errors;
}

// ── Console builders ──────────────────────────────────────────────────────────

export function confirmationSatisfied(typed, expected) {
  return typed.trim() === expected.trim() && expected.trim().length > 0;
}

function withOptionalToggles(base, toggles) {
  const { denseOnDisk, quantization, sparseIdf, sparseOnDisk, payloadOnDisk } = toggles;
  if (denseOnDisk !== null && denseOnDisk !== undefined) base.dense_on_disk = denseOnDisk;
  if (quantization !== null && quantization !== undefined) base.quantization = quantization;
  if (sparseIdf !== null && sparseIdf !== undefined) base.sparse_idf = sparseIdf;
  if (sparseOnDisk !== null && sparseOnDisk !== undefined) base.sparse_on_disk = sparseOnDisk;
  if (payloadOnDisk !== null && payloadOnDisk !== undefined) base.on_disk_payload = payloadOnDisk;
  return base;
}

// createOnly is the embedder-mismatch mode of 03 Create / Recreate: creating a
// new collection stays legal (the backend builds it from the ACTIVE contract),
// while recreate — destructive on the mismatched collection — is stripped here
// too, not only hidden in the UI.
export function buildCreatePayload(
  collectionName, recreate, confirmRecreate, toggles, { createOnly = false } = {},
) {
  const effectiveRecreate = createOnly ? false : Boolean(recreate);
  const payload = { collection_name: collectionName, recreate: effectiveRecreate };
  if (effectiveRecreate && confirmRecreate) payload.confirm_recreate = confirmRecreate;
  return withOptionalToggles(payload, toggles);
}

export function quantizationEnabled(quantRaw) {
  const inactive = ["disabled", "none", "null", ""];
  return Boolean(quantRaw) && !inactive.includes(String(quantRaw).trim().toLowerCase());
}

// Seeds the Schema Toggles widgets with the collection's live state so an
// update doesn't silently overwrite toggles the user didn't touch.
export function reportToggleDefaults(report, fallback) {
  if (!report || typeof report !== "object") return { ...fallback };
  const dense = report.dense || {};
  const sparse = report.sparse || {};
  const boolOr = (value, def) => (value !== null && value !== undefined ? Boolean(value) : def);
  return {
    denseOnDisk: boolOr(dense.on_disk, fallback.denseOnDisk),
    quantization: "quantization" in report
      ? quantizationEnabled(report.quantization)
      : fallback.quantization,
    sparseIdf: report.sparse !== null && report.sparse !== undefined
      ? String(sparse.modifier || "").toLowerCase().includes("idf")
      : fallback.sparseIdf,
    sparseOnDisk: boolOr(sparse.on_disk, fallback.sparseOnDisk),
    payloadOnDisk: boolOr(report.on_disk_payload, fallback.payloadOnDisk),
  };
}

// Fail closed while the UI cannot certify that vector operations use the
// same embedding space as the selected collection. Read-only and payload-only
// operations use a separate path and remain available.
export function embeddingCompatibilityGate(
  compatibility,
  { loading = false, hasCollection = true } = {},
) {
  if (!hasCollection) {
    return {
      blocked: true,
      status: "no_collection",
      message: "Select a collection to use vector operations.",
    };
  }
  if (loading) {
    return {
      blocked: true,
      status: "loading",
      message: "Checking collection compatibility with the active embedder…",
    };
  }

  const status = typeof compatibility === "string"
    ? compatibility
    : compatibility?.status;
  if (status === "compatible") {
    return { blocked: false, status, message: "" };
  }

  const messages = {
    incompatible:
      "Vector operations are disabled because this collection does not match the active embedder. Re-embed its vectors to continue.",
  };
  return {
    blocked: true,
    status: status || "unknown",
    message: messages[status]
      || "Vector operations are disabled until collection compatibility can be verified.",
  };
}

// Full data-operation lockdown decision for the active collection. Distinct from
// embeddingCompatibilityGate (which only gated vector ops): an incompatible
// collection now inhibits every data operation, mirroring the service-unreachable
// lockdown. active == null is an embedder-availability problem the readiness gate
// owns, so it never locks here (and would otherwise render the wrong banner).
export function collectionLockState(compatibility, { servicesReady = false } = {}) {
  if (!servicesReady) return { locked: false, recoverable: false };
  const c = compatibility; // object or null (report shape)
  const incompatible = c?.status === "incompatible";
  const locked = incompatible && Boolean(c?.active);
  const recoverable = locked && Boolean(c?.stored);
  return { locked, recoverable };
}

export function displayEmbedderProfile(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return ["basic", "advanced"].includes(normalized) ? normalized : "—";
}

export function buildUpsertPayload(collectionName, pointId, text, payload, updateMode = "upsert") {
  return {
    collection_name: collectionName,
    point_id: pointId,
    text,
    payload,
    update_mode: updateMode,
  };
}

export function buildBatchPayload(collectionName, points, updateMode = "upsert") {
  return { collection_name: collectionName, points, update_mode: updateMode };
}

export function buildBulkDeleteBody(collectionName, ids) {
  return { collection_name: collectionName, ids };
}

export function buildRetrieveBody(collectionName, ids, withVectors = false) {
  return { collection_name: collectionName, ids, with_vectors: withVectors };
}

export function buildDeleteByFilterBody(collectionName, filters) {
  return { collection_name: collectionName, filters };
}

export function buildFilterCountPayload(collectionName, filters) {
  return {
    collection_name: collectionName,
    filters,
    limit: 1,
    with_payload: false,
    include_total: true,
  };
}

export function buildPayloadUpdateBody(collectionName, payload) {
  return { collection_name: collectionName, payload };
}

export function buildPayloadKeysDeleteBody(collectionName, keys) {
  return { collection_name: collectionName, keys };
}

export function buildClearPayloadBody(collectionName) {
  return { collection_name: collectionName };
}

export function downloadFilename(disposition, fallback) {
  const extended = /filename\*\s*=\s*UTF-8''([^;]+)/i.exec(disposition);
  if (extended) {
    try {
      return decodeURIComponent(extended[1]);
    } catch {
      // Fall through to the ASCII filename parameter below.
    }
  }
  const basic = /filename\s*=\s*"([^"]+)"/i.exec(disposition);
  return basic ? basic[1] : fallback;
}

// Split a comma/newline-separated free-text list into trimmed non-empty items.
// Shared by the bulk-id and payload-key inputs.
export function parseTokenList(raw) {
  return String(raw)
    .replaceAll(",", "\n")
    .split("\n")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

export function buildIngestChunksPayload(
  chunks, collectionName, batchSize, replaceSource, sourceId,
  sourceIdKey = "source_id", sourceIdType = "keyword", chunkIdKey = "id",
) {
  return {
    chunks,
    source_id: sourceId,
    source_id_key: sourceIdKey,
    source_id_type: sourceIdType,
    chunk_id_key: chunkIdKey,
    collection_name: collectionName,
    batch_size: batchSize,
    replace_source: replaceSource,
  };
}

// Contract mirrors the backend: top-level {"chunks": [...]}, each chunk needs
// an integer ``id`` and a non-empty ``text``.
export function validateIngestChunks(obj) {
  if (typeof obj !== "object" || obj === null || Array.isArray(obj) || !("chunks" in obj)) {
    return ["Missing 'chunks' array."];
  }
  const chunks = obj.chunks;
  if (!Array.isArray(chunks)) return ["'chunks' must be a list."];
  if (!chunks.length) return ["'chunks' is empty."];
  const errors = [];
  const ordinals = new Set();
  chunks.forEach((chunk, i) => {
    if (typeof chunk !== "object" || chunk === null || Array.isArray(chunk)) {
      errors.push(`chunk[${i}]: must be an object`);
      return;
    }
    const cid = chunk.id;
    if (typeof cid !== "number" || !Number.isInteger(cid)) {
      errors.push(`chunk[${i}]: 'id' must be an int`);
    } else if (!Number.isSafeInteger(cid)) {
      errors.push(`chunk[${i}]: 'id' must be a safe integer`);
    } else if (ordinals.has(cid)) {
      errors.push(`chunk[${i}]: duplicate 'id' ${cid}`);
    } else {
      ordinals.add(cid);
    }
    const text = chunk.text;
    if (typeof text !== "string" || !text.trim()) {
      errors.push(`chunk[${i}]: 'text' must be a non-empty string`);
    }
  });
  return errors;
}

// Client-side preflight summary for the Ingest subtab. Reuses
// validateIngestChunks for the error list and per-chunk validity, then builds a
// payload-field schema for every key stored as payload (i.e. all keys except the
// structural ``id`` and ``text``).
export function summarizeIngestChunks(obj) {
  const errors = validateIngestChunks(obj);
  const chunks = obj && typeof obj === "object" && !Array.isArray(obj)
    && Array.isArray(obj.chunks) ? obj.chunks : [];
  const total = chunks.length;

  // Per-chunk errors are prefixed ``chunk[<i>]:``; count distinct bad indices so
  // valid/invalid stay consistent with validateIngestChunks.
  const invalidIdx = new Set();
  for (const message of errors) {
    const match = /^chunk\[(\d+)\]/.exec(message);
    if (match) invalidIdx.add(Number(match[1]));
  }
  const invalid = total ? invalidIdx.size : 0;
  const valid = total - invalid;

  const inferType = (v) => {
    if (v === null) return "null";
    if (Array.isArray(v)) return "array";
    if (typeof v === "number") return Number.isInteger(v) ? "int" : "float";
    if (typeof v === "boolean") return "bool";
    if (typeof v === "object") return "object";
    return "string";
  };
  const truncate = (v) => {
    const s = typeof v === "string" ? v : JSON.stringify(v);
    return s.length > 40 ? `${s.slice(0, 39)}…` : s;
  };

  const stats = new Map();
  for (const chunk of chunks) {
    if (!chunk || typeof chunk !== "object" || Array.isArray(chunk)) continue;
    for (const [key, value] of Object.entries(chunk)) {
      if (INGEST_EXCLUDED_KEYS.has(key)) continue;
      const type = inferType(value);
      let stat = stats.get(key);
      if (!stat) {
        stat = { count: 0, type, sample: undefined };
        stats.set(key, stat);
      }
      stat.count += 1;
      if (stat.type !== type) stat.type = "mixed";
      if (stat.sample === undefined && value !== null && value !== undefined) {
        stat.sample = truncate(value);
      }
    }
  }

  const fields = [...stats.entries()].map(([name, stat]) => ({
    name,
    type: stat.type,
    coverage: `${stat.count}/${total}`,
    sample: stat.sample === undefined ? "—" : stat.sample,
  }));

  return { total, valid, invalid, errors, fields };
}

// Split a list into consecutive groups of `size` for chunked upload. A
// non-positive or non-finite size falls back to 1; a non-array input yields [].
export function chunkArray(list, size) {
  if (!Array.isArray(list)) return [];
  const step = Number.isFinite(size) && size >= 1 ? Math.trunc(size) : 1;
  const groups = [];
  for (let i = 0; i < list.length; i += step) {
    groups.push(list.slice(i, i + step));
  }
  return groups;
}

const PAYLOAD_UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Coerce a payload value to a chosen Qdrant type. Returns {ok:true, value} or
// {ok:false}. ``untyped`` is handled by the caller and never passed here.
export function coercePayloadValue(value, type) {
  const fail = { ok: false };
  switch (type) {
    case "keyword":
    case "text":
      if (value === null || typeof value === "string"
        || typeof value === "number" || typeof value === "boolean") {
        return { ok: true, value: String(value) };
      }
      return fail;
    case "integer":
      if (typeof value === "number") {
        return Number.isInteger(value) ? { ok: true, value } : fail;
      }
      if (typeof value === "string" && /^[+-]?\d+$/.test(value.trim())) {
        return { ok: true, value: parseInt(value.trim(), 10) };
      }
      return fail;
    case "float":
      if (typeof value === "number") {
        return Number.isFinite(value) ? { ok: true, value } : fail;
      }
      if (typeof value === "string" && value.trim() !== ""
        && Number.isFinite(Number(value))) {
        return { ok: true, value: Number(value) };
      }
      return fail;
    case "bool":
      if (typeof value === "boolean") return { ok: true, value };
      if (value === 0 || value === 1) return { ok: true, value: value === 1 };
      if (typeof value === "string") {
        const v = value.trim().toLowerCase();
        if (v === "true") return { ok: true, value: true };
        if (v === "false") return { ok: true, value: false };
      }
      return fail;
    case "datetime":
      if (typeof value === "string" && !Number.isNaN(Date.parse(value))) {
        return { ok: true, value };
      }
      return fail;
    case "uuid":
      return typeof value === "string" && PAYLOAD_UUID_RE.test(value)
        ? { ok: true, value }
        : fail;
    case "geo":
      return value !== null && typeof value === "object" && !Array.isArray(value)
        ? { ok: true, value }
        : fail;
    default:
      return fail;
  }
}

// Names the backend refuses as the source_id payload key: it would clobber a
// chunk-critical field. Mirrors {"text", "id"} | CHUNK_RESERVED_FIELDS in
// points.py exactly, so the preflight blocks what the middleware would 400 on.
export const INGEST_SOURCE_ID_RESERVED_KEYS = new Set([
  "id", "text", "overflow_details", "overlap_sentences_count", "split_metadata",
]);

// Names the backend refuses as the chunk-ordinal payload key. Mirrors
// `_validate_chunk_id_key` in points.py: `id` is the legitimate default and
// stays allowed, unlike for source_id_key where it would overwrite the ordinal.
export const INGEST_CHUNK_ID_RESERVED_KEYS = new Set([
  "text", "overflow_details", "overlap_sentences_count", "split_metadata",
]);

// Keys the preflight neither lists as ordinary field rows nor accepts as a
// rename target. `id` and `text` are what the middleware reads off every chunk
// (`_validate_chunks` in points.py), so neither may be renamed away nor landed
// on. `id` still gets a row of its own — a synthetic one alongside source_id,
// since its payload key travels as the `chunk_id_key` request field rather than
// as a client-side rename.
export const INGEST_EXCLUDED_KEYS = new Set(["id", "text"]);

// Apply an ingest field mapping to the parsed chunks: rename each mapped
// field's key (field -> name) and coerce its values to the chosen type (unless
// ``untyped``). id/text and any unmapped keys pass through unchanged — the
// middleware reads both off the chunk, and the ordinal's payload key is chosen
// with the ``chunk_id_key`` request field instead of by rewriting the chunk.
// Name problems (empty / reserved / duplicate) and coercion failures (first
// failing value per field) are collected in ``errors``; when ``errors`` is
// non-empty the returned ``chunks`` is [] and MUST NOT be used.
export function applyIngestFieldMapping(chunks, mapping) {
  const errors = [];
  const seenNames = new Map();
  for (const m of mapping) {
    const name = String(m.name ?? "").trim();
    if (!name) {
      errors.push(`field '${m.field}': name is required.`);
      continue;
    }
    if (INGEST_EXCLUDED_KEYS.has(name)) {
      errors.push(`field '${m.field}': name '${name}' is reserved`);
      continue;
    }
    if (seenNames.has(name)) {
      errors.push(`field '${m.field}': name '${name}' duplicates field '${seenNames.get(name)}'`);
      continue;
    }
    seenNames.set(name, m.field);
  }

  const mappingByField = new Map(mapping.map((m) => [m.field, m]));
  const coercionFlagged = new Set();
  const out = [];
  for (const chunk of Array.isArray(chunks) ? chunks : []) {
    if (!chunk || typeof chunk !== "object" || Array.isArray(chunk)) {
      out.push(chunk);
      continue;
    }
    const next = {};
    for (const [key, value] of Object.entries(chunk)) {
      const m = mappingByField.get(key);
      if (key === "id" || key === "text" || !m) {
        next[key] = value;
        continue;
      }
      const name = String(m.name ?? "").trim();
      const nameUsable = name && !INGEST_EXCLUDED_KEYS.has(name);
      if (m.type === "untyped") {
        if (nameUsable) next[name] = value;
        continue;
      }
      const res = coercePayloadValue(value, m.type);
      if (!res.ok) {
        if (!coercionFlagged.has(m.field)) {
          coercionFlagged.add(m.field);
          errors.push(`field '${name || m.field}' → ${m.type}: value ${JSON.stringify(value)} is not convertible`);
        }
        continue;
      }
      if (nameUsable) next[name] = res.value;
    }
    out.push(next);
  }

  return { chunks: errors.length ? [] : out, errors };
}

// Ingest writes top-level keys; its indexes must preserve literal path syntax.
export function literalPayloadIndexField(name) {
  const key = String(name ?? "").trim();
  return /[.\[\]"]/.test(key) ? JSON.stringify(key) : key;
}

function canonicalIndexField(name) {
  const path = String(name ?? "").trim();
  if (path.startsWith('"')) {
    try { return literalPayloadIndexField(JSON.parse(path)); } catch { /* Quoted segment with a traversal suffix. */ }
  }
  return path;
}

// Payload indexes to create once an ingest has completed. Each typed mapping
// row becomes one request item under its FINAL name, alongside source_id (only
// when typed) and the chunk ordinal, which is always `integer` because the
// middleware writes it back unconverted.
//
// A field the collection already indexes is NEVER requested: the middleware
// replaces an index whose type differs (Qdrant has no in-place type change, so
// it drops and recreates) and reports it as `updated`. Excluding those fields
// client-side is what keeps an ingest from silently rewriting an index; they
// are returned in `alreadyIndexed` with the schema they actually have, for the
// summary. Changing one stays a manual Collections → Payload Indexes action.
export function buildIngestIndexRequests(
  fieldMapping, sourceId, chunkIdKey, existingSchemas = {},
) {
  const wanted = [];
  // A row with no type at all is treated like `untyped`: a request item with
  // no field_schema would fail the whole batch, valid fields included.
  const sourceKey = String(sourceId?.key ?? "").trim();
  if (sourceKey && sourceId.type && sourceId.type !== "untyped") {
    wanted.push({ field_name: literalPayloadIndexField(sourceKey), field_schema: sourceId.type });
  }
  const ordinalKey = String(chunkIdKey ?? "").trim();
  if (ordinalKey) {
    wanted.push({ field_name: literalPayloadIndexField(ordinalKey), field_schema: "integer" });
  }
  for (const m of Array.isArray(fieldMapping) ? fieldMapping : []) {
    const name = String(m?.name ?? "").trim();
    if (!name || !m.type || m.type === "untyped") continue;
    wanted.push({ field_name: literalPayloadIndexField(name), field_schema: m.type });
  }

  const schemas = existingSchemas && typeof existingSchemas === "object"
    ? existingSchemas : {};
  const requests = [];
  const alreadyIndexed = [];
  const seen = new Set();
  for (const item of wanted) {
    const canonical = canonicalIndexField(item.field_name);
    if (seen.has(canonical)) continue;
    seen.add(canonical);
    const existing = schemas[item.field_name] ?? schemas[canonical];
    if (existing) {
      alreadyIndexed.push({ field_name: item.field_name, field_schema: existing });
    } else {
      requests.push(item);
    }
  }
  return { requests, alreadyIndexed };
}

// Rows are grouped by outcome so the summary reads top-down like the PRD's
// example: what changed first, what was left alone last.
const INGEST_INDEX_STATUS_RANK = {
  created: 0, updated: 1, failed: 2, skipped: 3,
};

// Fold a PayloadIndexCreateResponse together with the fields the client kept
// out of the request (already indexed) into one render-ready summary. A failed
// call — `response` null, `reason` set — marks every requested field `failed`;
// the excluded ones stay `skipped`, since nothing was attempted for them.
export function summarizeIngestIndexResult(
  response, requests, alreadyIndexed = [], reason = "",
) {
  const data = response && typeof response === "object" ? response : null;
  const bucketStatus = new Map();
  if (data) {
    for (const status of ["created", "updated", "skipped", "failed"]) {
      const list = Array.isArray(data[status]) ? data[status] : [];
      for (const field of list) {
        bucketStatus.set(canonicalIndexField(field), status);
      }
    }
  }

  const rows = (Array.isArray(requests) ? requests : []).map((item) => ({
    field: item.field_name,
    schema: item.field_schema,
    // A requested field missing from every bucket means the manager answered a
    // shape this console does not understand — report it rather than hide it.
    status: data
      ? bucketStatus.get(canonicalIndexField(item.field_name)) || "failed"
      : "failed",
  }));
  for (const item of Array.isArray(alreadyIndexed) ? alreadyIndexed : []) {
    rows.push({
      field: item.field_name, schema: item.field_schema, status: "skipped",
    });
  }
  rows.sort((a, b) =>
    INGEST_INDEX_STATUS_RANK[a.status] - INGEST_INDEX_STATUS_RANK[b.status]);

  const counts = { created: 0, updated: 0, skipped: 0, failed: 0 };
  for (const row of rows) counts[row.status] += 1;
  return {
    rows,
    counts,
    // An `updated` field is an index this ingest overwrote: the summary has to
    // say so loudly, because no one asked for it (see FR-5 in the PRD).
    replaced: counts.updated > 0,
    reason: data ? "" : String(reason || ""),
  };
}

export function collectionSchemaMessages(report) {
  const pick = (item, keys) => {
    if (typeof item === "object" && item !== null) {
      for (const key of keys) {
        if (item[key]) return String(item[key]);
      }
      return JSON.stringify(item);
    }
    return String(item);
  };
  const findings = (report.findings || report.errors || []).map((item) =>
    pick(item, ["message", "field", "category"]));
  const fixes = (report.suggested_fixes || []).map((item) =>
    pick(item, ["description", "endpoint", "type"]));
  return [findings, fixes];
}

// ── Collection name helpers ───────────────────────────────────────────────────

export function sortedCollectionNames(collections) {
  const seen = new Set();
  for (const name of collections) {
    const trimmed = String(name).trim();
    if (trimmed) seen.add(trimmed);
  }
  return [...seen].sort((a, b) =>
    a.toLowerCase() < b.toLowerCase() ? -1 : a.toLowerCase() > b.toLowerCase() ? 1 : 0);
}

export function chooseActiveCollection(current, collections) {
  const names = sortedCollectionNames(collections);
  const trimmed = current.trim();
  if (names.includes(trimmed)) return trimmed;
  return names.length ? names[0] : "";
}

// ── Prometheus parsing ────────────────────────────────────────────────────────

// Estimate a quantile from cumulative histogram buckets:
// [[upperBound, cumulativeCount], ...] sorted ascending, +Inf included.
export function estimateQuantile(buckets, q) {
  if (!buckets.length) return 0;
  const total = buckets[buckets.length - 1][1];
  if (total <= 0) return 0;
  const target = q * total;
  let prevBound = 0;
  let prevCount = 0;
  for (const [bound, cum] of buckets) {
    if (cum >= target) {
      if (bound === Infinity) return prevBound;
      const span = cum - prevCount;
      if (span <= 0) return bound;
      const frac = (target - prevCount) / span;
      return prevBound + frac * (bound - prevBound);
    }
    prevBound = bound;
    prevCount = cum;
  }
  return buckets[buckets.length - 1][0];
}

function parsePromLine(line) {
  const match = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)/.exec(line);
  if (!match) return null;
  const labels = {};
  if (match[2]) {
    const labelRe = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"/g;
    let lm;
    while ((lm = labelRe.exec(match[2])) !== null) {
      labels[lm[1]] = lm[2].replace(/\\(["\\n])/g, (_, c) => (c === "n" ? "\n" : c));
    }
  }
  const value = match[3] === "+Inf" ? Infinity : parseFloat(match[3]);
  return { name: match[1], labels, value };
}

// Parse Prometheus exposition text into per-endpoint rows:
// {endpoint, requests, errors, p50, p95}. p50/p95 are null without a histogram.
export function parseEndpointMetrics(text) {
  const requests = new Map();
  const errors = new Map();
  const hist = new Map();

  for (const rawLine of String(text).split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const sample = parsePromLine(line);
    if (!sample || Number.isNaN(sample.value)) continue;

    if (sample.name === "qdrant_manager_requests_total") {
      const ep = sample.labels.endpoint || "?";
      requests.set(ep, (requests.get(ep) || 0) + sample.value);
      if (String(sample.labels.status || "").startsWith("5")) {
        errors.set(ep, (errors.get(ep) || 0) + sample.value);
      }
    } else if (sample.name === "qdrant_manager_request_duration_seconds_bucket") {
      const ep = sample.labels.endpoint || "?";
      const le = sample.labels.le || "+Inf";
      const bound = le === "+Inf" || le === "Inf" ? Infinity : parseFloat(le);
      if (!hist.has(ep)) hist.set(ep, []);
      hist.get(ep).push([bound, sample.value]);
    }
  }

  const endpoints = [...new Set([...requests.keys(), ...hist.keys()])].sort();
  return endpoints.map((ep) => {
    const buckets = (hist.get(ep) || []).slice().sort((a, b) => a[0] - b[0]);
    const hasHist = buckets.length > 0;
    return {
      endpoint: ep,
      requests: requests.get(ep) || 0,
      errors: errors.get(ep) || 0,
      p50: hasHist ? estimateQuantile(buckets, 0.5) : null,
      p95: hasHist ? estimateQuantile(buckets, 0.95) : null,
    };
  });
}

// ── Help copy ─────────────────────────────────────────────────────────────────

export const PARAM_HELP = {
  // House style for every entry below: two to four sentences, roughly 200-400
  // characters. Open with what the control does, then the constraint or caveat
  // that actually bites. No lists, no worked examples longer than a clause.
  "query": "Text to search for, in natural language or keywords. BGE-M3 embeds it into a dense (semantic) and a sparse (lexical) vector, and it is multilingual, so the query language need not match the documents. Leave it empty to switch the tab to payload-only retrieval: nothing is embedded, the filters drive a plain scroll, and the hybrid-only controls turn themselves off.",
  "collection_name": "The Qdrant collection every request on this tab targets. It is read-only here: choose it from Active collection in the connection bar, which the Collections and Points tabs follow too. Changing it clears the results and the filter rows, since both belong to the collection they were built against.",
  "limit": "How many hits the hybrid search finally returns, after fusion and any rerank. Payload-only retrieval ignores it and walks a fixed page of 5 points instead, so the field is unavailable while the query is empty.",
  "candidate_limit": "How many candidates each branch, dense and sparse, fetches from Qdrant before fusion and rerank. It must be at least limit; raising it improves recall and costs latency, because both branches grow together. Hybrid search only.",
  "fusion": "How the dense and sparse rankings become a single list. RRF merges by rank position, so differing score scales cannot distort it; DBSF normalizes the two score distributions and sums them, which keeps score magnitude meaningful but is more sensitive to outliers. RRF is the safer default.",
  "rrf_k": "Smoothing constant for RRF. Higher values flatten the advantage of the leading positions, so mid-ranked candidates weigh relatively more; lower values sharpen the head of the list. Left unset, Qdrant uses 2. Available with RRF only.",
  "rrf_weights": "Tilts fusion toward one branch. The slider sets the dense/sparse balance and the scale is a common multiplier, so Qdrant receives [2 × balance × scale, 2 × (1 − balance) × scale] — 50/50 at scale 1 sends [1,1], 75/25 at scale 2 sends [3,1]. These are weights, not shares of the final score. A weight of 0 removes a branch's contribution but still runs its prefetch, so zero-score points can surface in the tail. RRF only.",
  "score_threshold": "Minimum score a hit must reach to be returned, so weak matches are dropped instead of padding the page. Qdrant applies it to the fused score — the value the cards show while rerank is off — and therefore before any rerank. Because the useful value depends on the fusion in use, read it off an unfiltered run rather than guessing; set too high, it legitimately returns nothing. Hybrid search only.",
  "offset": "Skips the first N fused hits: limit 10 with offset 10 gives the second page. Applied after fusion and before rerank, so a reranked page is reranked within its own window. Depth is bounded by candidate_limit — fusion only ranks points a branch retrieved — so a deeper offset is refused naming the value to raise. Where fused scores tie, a point can shift between adjacent pages. Hybrid search only.",
  "rerank": "Second pass with the Smart Embedder cross-encoder, which re-scores the head candidates against the query using their payload text. Slower and more precise than fusion alone. Hits without usable text cannot be scored: they are kept below the reranked ones and counted as skipped_no_text.",
  "rerank_top_k": "How many leading candidates the reranker actually evaluates. It cannot exceed candidate_limit, and everything below the window keeps its fusion order, so raising it buys precision deeper into the list at a proportional cost in latency.",
  "rerank_normalize": "Maps rerank scores into [0,1] so they read the same across queries. Disable it to see the cross-encoder's raw output, which is what you want when calibrating a score_threshold or comparing two runs directly.",
  "filters — GROUP": "MUST, SHOULD and MUST NOT combine the conditions at the current level. A logical group adds parentheses around its own conditions; a same-element group makes all of its conditions target one single dictionary inside an object array, rather than matching across different elements.",
  "filters — FIELD": "Only indexed payload fields appear, because these collections run in Qdrant strict mode and an unindexed field cannot be filtered. (point id) and (vector presence) are always available since they need no index. Inside a same-element group the list narrows to the indexed paths below the chosen [] array.",
  "filters — OPERATION": "The operations offered follow the field's index type: exact or list match, numeric and datetime ranges, full-text all/any/phrase, geo radius, box and polygon, array-size bounds, is empty and is null. Note that match-except means any stored value outside the list; to forbid every listed value use MUST NOT with any of.",
  "filters — VALUE": "Lists accept comma-separated text or a JSON array, and JSON is the way to pass values that contain commas. Custom ranges and array sizes take independent lower and upper bounds, each switchable between strict and inclusive. Polygon rings are lon lat pairs separated by semicolons, with holes separated by a pipe.",
  "min should matches": "Appears once two or more SHOULD conditions exist at the same level, and sets how many of them a point must satisfy. At 1 it behaves like a plain SHOULD, meaning any of them; raising it demands that many simultaneously, which is how you express \"at least two of these\" without writing every combination by hand.",
  "with_payload": "Which payload fields come back with each hit, offered from the keys sampled on the collection's real points. Everything is selected by default. While rerank is active text is included and locked, because the cross-encoder reads it to score the candidates.",
  "order_by": "Sorts a payload-only retrieval by an indexed payload field instead of by point id, which is a deterministic UUID and therefore unordered in practice. Only indexed fields are offered, since Qdrant needs the field index to sort. Qdrant refuses offset together with order_by, so an ordered run is served as one page and the pager is replaced by the ordering summary. Payload-only retrieval only.",
  "start_from": "Optional resume point, and the only way to page an ordered run: it means \"begin at this value\", not \"skip N rows\". Put the last value you saw here to get the following block. It is inclusive, so that boundary row repeats — unavoidable with duplicate values, and precisely why Qdrant pages by value: it jumps straight there through the index and stays correct while points are written. Leave it empty for the first page.",
  "Request JSON": "Live preview of the exact body that Run sends to the API, rebuilt on every change so what you read is what will be sent. Validation errors appear underneath it and keep Run disabled until they are resolved.",
  "Download results as JSON": "Saves payloads plus point_id as a chunks file, with request, mode and embedder metadata. Re-ingesting requires integer chunk ordinals and the original source identity; exported point_id values identify stored points separately. Payload-only retrieval walks every page before saving. An ordered run exports the single page shown on screen.",
  "Results": "Hits best-first, each with its fusion score, its rerank score when reranked, metadata chips and the full payload. The candidates tile counts the fused hits the middleware received, not everything the two prefetches examined. Payload-only retrieval is paged rather than ranked: Previous and Next walk 5 points at a time, and the footer shows the page or, with order_by set, the active ordering.",
  "Availability": "The whole tab goes inert when the services are unreachable or when the active collection was written with a different embedding contract than the current Smart Embedder — searching it would compare incompatible vectors. The banner above states which case applies; re-embed the collection or switch to a compatible one to restore it.",
};

export function buildQueryParamHelp(config = QUERY_DEFAULTS) {
  const queryConfig = normalizeQueryConfig(config);
  return {
    ...PARAM_HELP,
    // Same entries, with the live server bounds spliced in. Wording and length
    // stay in the band the static texts use.
    limit: "How many hits the hybrid search finally returns, after fusion and any " +
      "rerank (1-" + queryConfig.queryMaxLimit + "). Payload-only retrieval " +
      "ignores it and walks a fixed page of 5 points instead, so the field is " +
      "unavailable while the query is empty.",
    candidate_limit: "How many candidates each branch, dense and sparse, fetches " +
      "from Qdrant before fusion and rerank (1-" +
      queryConfig.queryMaxCandidateLimit + "). It must be at least limit; raising " +
      "it improves recall and costs latency, because both branches grow together. " +
      "Hybrid search only.",
    rerank: PARAM_HELP.rerank + " Each candidate's text is truncated to " +
      queryConfig.queryRerankMaxChars + " characters (0 = unlimited).",
    rerank_top_k: "How many leading candidates the reranker actually evaluates (1-" +
      queryConfig.queryMaxRerankTopK + ", default " +
      queryConfig.queryDefaultRerankTopK + ", capped by the current " +
      "candidate_limit). Everything below the window keeps its fusion order, so " +
      "raising it buys precision deeper into the list at a proportional cost in " +
      "latency.",
  };
}

// Same house style as PARAM_HELP: two to four sentences, roughly 200-400
// characters, opening with what the control does and closing on the constraint
// that actually bites. Entries follow the order the panel reads in.
export const COLLECTIONS_HELP = {
  "Active collection": "Connection-bar selector shared by the whole console. Sections 01, 02, 04 and 05 always act on it, and the Points and Query tabs follow it too. Sections 03 and 07 are the exception: they work on the name typed into their own field, so they can create or delete a collection that is not the active one.",
  "Availability": "Sections lock themselves rather than fail. With the services unreachable the whole tab goes inert. With an active collection written under a different embedding contract, 03, 04, 05 and 07 lock while 01 Report stays readable and 02 Re-embed stays available to fix it — unless the collection carries no stored contract at all, in which case there is nothing to regenerate from and 02 locks as well. 06 Export is read-only and always stays live.",
  "01 Report": "Live readout of the active collection: the five storage toggles, point count, dense vector size, and the embedding contract. It compares the profile and models recorded in Qdrant metadata against the running Smart Embedder and reports compatible or incompatible. It refreshes on its own after every write from any section.",
  "02 Re-embed Vectors": "Regenerates dense and sparse vectors from each point's payload text, which is how a collection moves to a new embedding contract without losing data. Preflight scans the whole collection read-only first. The current vectors keep serving queries until the new generation validates exactly, and only then does metadata switch Query and Points over.",
  "pause / resume / cancel": "Pause stops at the end of the batch in flight and keeps the checkpoint, so Resume continues instead of restarting. Cancel abandons the target generation and drops its temporary vector schemas. While a job is resumable, point writes that would change vectors stay blocked; reads keep working.",
  "type-to-confirm": "Every destructive action here — recreate, delete, and starting a regeneration — stays disabled until you type the exact collection name. It is deliberate friction, not validation theatre: the same gate exists on the API, so a mistyped name is refused server-side too. The field clears itself after a successful run.",
  "03 Create / Recreate": "Creates a collection with the five toggles chosen here and makes it active. Re-submitting the same name with identical settings does nothing. With different toggles it is refused (409) and you are pointed at 04 Schema Toggles, because changing them online never needs a rebuild. The vector structure itself is fixed: one dense plus one sparse vector, named from the embedding contract.",
  "collection_name": "Name of the collection to create or recreate. Free text and independent of the active collection — it does not have to exist yet. On a successful create the new collection becomes active and the connection-bar list refreshes.",
  "recreate": "DESTRUCTIVE. Drops the collection and rebuilds it empty, losing every point. Only structural changes need it (vector size, distance, shard layout); the five toggles do not — use 04 instead. Requires the exact name typed to confirm, and the checkbox disarms itself after the run.",
  "dense_on_disk": "Keeps dense vectors on disk instead of in RAM. Saves memory in proportion to the collection size, at a small latency cost per search since the vectors are read from storage. Sensible default for large collections; leave it off when the whole set fits comfortably in memory.",
  "sparse_on_disk": "Keeps the sparse index on disk instead of in RAM. Same trade as dense_on_disk and usually cheaper, because a sparse posting list is far smaller than the dense vectors it accompanies.",
  "payload_on_disk": "Keeps payload values on disk instead of in RAM. The cost is usually negligible: the operating system caches the hot pages, and filtering does not read payload values at all — it uses the in-RAM payload indexes from section 05.",
  "quantization": "Compresses dense vectors with TurboQuant (4-bit, about 8x smaller), which cuts memory and speeds up search at some cost in recall. The recall is bought back at query time with rescore and oversampling, so treat this as the storage half of a decision whose other half lives in the Query tab.",
  "sparse_idf": "Applies the IDF modifier to the sparse vector, so a term that is rare across the collection weighs more than a common one. Recommended with BGE-M3 sparse, which is trained expecting it; without it frequent tokens dominate the lexical score.",
  "04 Schema Toggles": "Changes the five storage toggles on the active collection in place — no rebuild, no data loss — while Qdrant rewrites storage in the background. The widgets always load the collection's current values, so what you see is the real state and not the last thing you submitted.",
  "05 Payload Indexes": "Indexes on payload fields. They are not optional here: these collections run in Qdrant strict mode, so a field without an index cannot be filtered at all. The table lists every payload field sampled from real points, with an em dash in the schema column for the ones not indexed yet, and refreshes when the active collection changes.",
  "payload field / field schema": "The field to index and how to index it: keyword for exact values, text for full-text, integer, float, bool, geo, datetime, uuid. The choice decides which filter operators the Query tab will offer on that field. Creating is idempotent; on a field already indexed with another type the index is dropped and rebuilt with the new one.",
  "phrase matching": "Available on text only, and adds the exact-phrase operator on that field. To tell whether two words are adjacent the index must store each token's position, so it costs space in proportion to tokens per document — negligible on a title, real on a long body. Nothing else changes: contains-all-words and contains-any-word return the same results either way. Switching it later rebuilds the index.",
  "payload index field to delete": "Removes the index from a field, which also removes the ability to filter on it. Deleting an index that does not exist is reported as a readable 404 and changes nothing. The field keeps its payload values — only the index is dropped.",
  "06 Export Collection": "Downloads the active collection as one chunks file, walking every page server-side. Re-ingesting requires integer chunk ordinals and source identity; exported point_id is separate from the ingest id. This read stays available for incompatible collections. Empty collections produce a warning; a failed page or exceeded export cap yields no file.",
  "07 Delete": "DESTRUCTIVE. Removes the collection and every point in it, and cannot be undone. It acts on the name typed in its own field, not on the active collection, so read that field before confirming. Requires the exact name typed to confirm; afterwards the active collection falls back to another one from the list.",
};

// Same house style as PARAM_HELP/COLLECTIONS_HELP: two to four sentences,
// roughly 180-450 characters, opening with what the control does and closing
// on the constraint that actually bites. Entries follow the order the panel
// reads in, and every inline caption the page used to show lives here instead.
export const POINTS_HELP = {
  "collection_name": "All operations on this tab target the active collection selected in the connection bar; it cannot be changed from here. The Collections report and the sampled payload fields refresh automatically after every mutation, so a write's effect is visible without switching tabs.",
  "Ingest": "Uploading a chunks JSON file runs a preflight in the browser (counts, payload schema) and reveals Confirm & ingest when the file is valid. Point IDs are deterministic per source + chunk, so re-ingesting the same source overwrites the same points. source_id defaults to the file name without extension; ingest is always additive (no destructive replace).",
  "chunks JSON file": "File shaped as {\"chunks\": [{\"id\": int, \"text\": \"...\", ...}]}. Every chunk needs an integer id and a non-empty text; extra keys beyond the reserved ones are stored as payload verbatim. The browser preflight catches a missing or malformed field before any request reaches the server.",
  "Create payload indexes for typed fields after ingest": "Once every batch has landed, each mapping row typed to something other than — untyped is indexed under its final name, plus the chunk ordinal as integer, so the fields are filterable immediately. Only a fully successful ingest triggers it, and a field the collection already indexes is left alone — its type is locked in the table, and changing it stays a manual Collections → Payload Indexes operation.",
  "Upsert": "Insert or overwrite a single point by id with text and a JSON payload. The text is embedded server-side into dense and sparse vectors before the write, so a failed embedding call never touches stored data. update_mode below decides what happens when the id already exists.",
  "point_id": "UUID or integer identifier of the point — the only two id shapes Qdrant accepts. Upsert with an existing id overwrites that point by default; Get, Delete, and Payload all address one point by this same id, each on its own subtab.",
  "text": "Raw text of the point, embedded server-side into dense and sparse vectors at write time — there is no way to submit a vector directly, so the stored vectors always match this text exactly. The reranker also reads it back at query time to score candidates.",
  "payload JSON": "Arbitrary JSON metadata stored alongside the point's text and vectors. Create indexes for console filter fields in Collections → Payload Indexes. An unindexed API filter may scan the collection or be refused by its Qdrant strict-mode configuration.",
  "update_mode": "Chooses what an upsert does when a point_id already exists or doesn't. upsert always writes, silently overwriting an existing point; insert_only skips ids that already exist, so a repeat never overwrites by accident; update_only skips ids that don't exist, so a typo never creates a stray point. Existing ids are checked before embedding, so a skipped point costs no Smart Embedder call. Available on both Upsert and Batch.",
  "Batch": "Upsert many points at once from a JSON array of {point_id, text, payload} objects, embedded and written in a single Qdrant call. Duplicate point_id values within the same array are rejected before anything is sent. update_mode applies to the whole batch the same way it applies to a single Upsert.",
  "Get": "Fetch a point by id from the active collection and inspect its payload. Get many retrieves several ids at once (one per line or comma-separated); ids with no stored point are silently omitted rather than reported as errors.",
  "with_vectors": "Adds a summary of the point's stored vectors — names and shape (dense dimension, sparse nonzero count) — never the raw numbers. Useful to confirm which embedding generation a point currently carries, especially mid re-embed when old and new vector names briefly coexist. Available on both Get and Get many.",
  "Delete": "Remove one point by id, bulk-delete a list of ids, or delete every point matching a filter. All three DESTRUCTIVE variants require typing the collection name to confirm, and every input clears itself after a successful run.",
  "ids": "Bulk-delete list: one id per line or comma-separated, up to 10000 per request. A live counter above the confirm field shows how many ids were actually parsed, which catches a stray separator before it is sent.",
  "Delete by filter": "DESTRUCTIVE. Deletes every point matching the filter JSON, in the same shape the Query tab filters use — legacy {field: value} or logical must/should/must_not. Count matches previews the hit count without deleting anything; an empty filter is refused so the whole collection can never be wiped by accident.",
  "Payload": "Update a payload without re-embedding the point, since the stored vectors depend only on text. MERGE folds the given keys into the existing payload; REPLACE overwrites it entirely, though text and source_id are always carried forward regardless of mode.",
  "scope: POINT ID | FILTER": "Chooses what a payload write addresses: one point by id, or every point matching a filter — so \"set this field on every point where …\" is one call instead of one per point. Count matches previews the scope before writing. REPLACE stays point-scoped: a full replace carries that point's own text and source_id forward, which one write cannot do across many, so filter scope offers MERGE only. A match-all filter is refused, as in Delete by filter.",
  "Delete keys": "Remove named payload fields — one key per line or comma-separated — from one point or from every point matching the filter, depending on scope above. The protected fields text and source_id cannot be deleted this way, in either scope.",
  "Clear payload": "DESTRUCTIVE. Strips the payload down to the protected fields text and source_id, which are always kept so retrieval and reranking keep working — for one point, or for every point matching the filter. Requires typing the collection name to confirm. In filter scope the keys removed are those observed on a sample of the collection, so a field absent from that sample survives the clear.",
  "Count": "Read-only preflight for filters: how many points match, without retrieving any of them. Deliberately separate from Delete by filter, so asking the question never means visiting a destructive control. An empty filter counts the whole collection; exact off returns the fast approximate estimate instead of the precise figure.",
  "Facet": "Read-only: lists the distinct values stored in one payload field and how many points carry each, without fetching the points themselves — the answer to \"what values exist\" instead of one Count call per guessed value. Only keyword, integer, bool, and uuid indexes qualify (text, geo, float, and datetime are rejected by Qdrant for faceting), so the key selector lists only fields that do. exact off trades precision for a faster estimate per value.",
};

// Same house style as PARAM_HELP/COLLECTIONS_HELP/POINTS_HELP: two to four
// sentences, roughly 180-450 characters, opening with what the control shows
// and closing on the constraint that actually bites.
export const MONITORING_HELP = {
  "Auto-refresh": "Every panel on this tab refreshes automatically every 5 seconds while it stays open, so the badges, stats, chart, and per-endpoint table always reflect the latest background probe and Prometheus scrape. Refreshing stops the moment you leave the tab and resumes from scratch on return.",
  "livez": "Liveness probe: confirms only that the process is running and accepting connections, with no Qdrant or Smart Embedder I/O behind it. It stays UP even while both dependencies are unreachable — readyz and health are what actually report on those.",
  "readyz": "Readiness probe for load balancers and orchestrators: reads the same cached background health state as health rather than doing new I/O per request. UP means both Qdrant and the Smart Embedder answered the last background probe, taken every 15 seconds by default.",
  "health": "Aggregate health combining Qdrant and Smart Embedder reachability into one status. DOWN names exactly which dependency failed in its detail field, and the same cached probe backs the qdrant and embedder badges below, so all three move together.",
  "qdrant": "Qdrant database reachability, read from the manager's cached background probe rather than checked per request. A stale-but-recent UP can lag an actual outage by up to one probe interval, which is the trade-off for keeping readyz and health cheap to call.",
  "embedder": "Smart Embedder reachability for the active profile, from the same cached background probe as qdrant. DOWN blocks every Points and Query write and read, since embedding capacity is required for both.",
  "Embedder Models": "Active profile and the loaded dense, sparse, and reranker model names, shown as badges once the embedder probe reports reachable. They go blank rather than stale the moment the embedder drops, so what is displayed always matches the last successful probe.",
  "uptime": "How long the manager process has been running since its last start or restart, formatted as hours, minutes, and seconds. A value that resets unexpectedly is itself a signal the process restarted, independent of anything else on this tab.",
  "total_requests": "Cumulative requests served since the process started, plotted live on the chart below to show traffic over time. It only ever increases between restarts; Reset counters changes what is displayed here, not this underlying value.",
  "rejected_requests": "Requests blocked before reaching a route handler: a missing or wrong bearer token, a rate limit, an oversized body, or the shutdown guard once graceful shutdown begins. It never counts a request that failed inside a handler — that shows up in per-endpoint metrics instead.",
  "requests chart": "Total requests sampled every 5 seconds while this tab stays open, plotting the last 120 samples — about 10 minutes — as a rolling window. Hover any point for its exact value and timestamp; the series restarts from empty, like the counter it plots, when the manager restarts.",
  "Reset counters": "View-only: snapshots the current totals and shows every tile, the chart, and the per-endpoint table relative to that moment, so a fresh incident is easy to read without the running total in the way. Server-side Prometheus counters are untouched; p50/p95 latencies stay lifetime values regardless.",
  "Per-endpoint metrics": "Prometheus request counts, errors, and p50/p95 latency parsed per route from the same text the /metrics scraper reads. Only documented API endpoints — the ones Swagger lists — appear, and only once they have received at least one call since the last reset, so the table starts empty rather than listing every route at zero.",
};
