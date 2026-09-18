import assert from "node:assert/strict";
import test from "node:test";
import { validateIngestChunks, createRequestCoordinator, activeVectorNames, responseValidators } from "../dist/assets/models.js";

test("request revisions also guard completion timers after finish", () => {
  const c = createRequestCoordinator(), first = c.begin();
  c.finish(first);
  assert.equal(c.isCurrent(first), true);
  c.changed(); c.changed();
  assert.equal(c.isCurrent(first), false);
  const next = c.begin();
  c.finish(first);
  assert.equal(c.canCommit(next), true);
});

test("active vector names require the authoritative report names", () => {
  assert.deepEqual(activeVectorNames({}), []);
  assert.deepEqual(activeVectorNames({ embedding_compatibility: { active_vectors: { dense: "d_g2", sparse: "s_g2" } } }), ["d_g2", "s_g2"]);
  assert.deepEqual(activeVectorNames({ embedding_compatibility: { active_vectors: { dense: "d", sparse: null } } }), []);
});

test("ingest indexes literal dotted keys independently of nested indexes", () => {
  const result = buildIngestIndexRequests([
    { name: "a.b", type: "keyword" }, { name: "items[].label", type: "keyword" },
  ], { key: "source_id", type: "untyped" }, "", { "a.b": "keyword" });
  assert.deepEqual(result.requests.map((item) => item.field_name), ['"a.b"', '"items[].label"']);
  assert.deepEqual(result.alreadyIndexed, []);
});

test("response counts reject strings and booleans while keeping valid zero and skips", () => {
  const base = { ok: true, collection_name: "docs", upserted: 0, skipped: 1 };
  assert.equal(responseValidators.batch({ ...base, extra: true }), true);
  for (const value of [true, "0", -1, null, 0.5]) assert.equal(responseValidators.batch({ ...base, upserted: value }), false);
  assert.equal(responseValidators.upsert({ ok: true, collection_name: "docs", point_id: "7", skipped: true }), true);
  assert.equal(responseValidators.reindex({ collection_name: "docs", job: null, task_running: false }), true);
  assert.equal(responseValidators.indexCreate({ ok: false, collection_name: "docs", created: ["a"], updated: [], skipped: [], failed: ["b"] }), true);
  assert.equal(responseValidators.retrieve({ collection_name: "docs", returned: 0, results: [] }), true);
  assert.equal(responseValidators.retrieve({ collection_name: "docs", returned: 1, results: [] }), false);
});

import {
  applyIngestFieldMapping,
  buildCreatePayload,
  buildFilterCondition,
  buildFilterGroups,
  buildFilterTree,
  buildIngestChunksPayload,
  buildIngestIndexRequests,
  buildQueryParamHelp,
  buildRequestPayload,
  chunkArray,
  coercePayloadValue,
  collectionLockState,
  createQueryRequestCoordinator,
  defaultQueryState,
  downloadFilename,
  displayEmbedderProfile,
  embeddingCompatibilityGate,
  filterTreeStats,
  INGEST_CHUNK_ID_RESERVED_KEYS,
  INGEST_EXCLUDED_KEYS,
  INGEST_SOURCE_ID_RESERVED_KEYS,
  normalizeQueryConfig,
  parseNumericPointId,
  validateBatchPointIds,
  rrfWeightsFromBalance,
  summarizeIngestChunks,
  summarizeIngestIndexResult,
  validateQueryState,
} from "../dist/assets/models.js";

const COLLECTION = "documents";

test("batch and filters share numeric bounds without rewriting batch IDs", () => {
  assert.deepEqual(parseNumericPointId(0), [0, null]);
  assert.deepEqual(parseNumericPointId("18446744073709551615"), ["18446744073709551615", null]);
  assert.match(parseNumericPointId("18446744073709551616")[1], /unsigned 64-bit/);
  for (const point_id of [7, 9007199254740992, 1.5, -1]) {
    assert.match(validateBatchPointIds([{ point_id }]), /quoted ID string/);
  }
  const batch = [{ point_id: "a0000000000040008000000000000001", payload: { id: -1 } }];
  const original = JSON.stringify(batch);
  assert.equal(validateBatchPointIds(batch), null);
  assert.equal(JSON.stringify(batch), original);
});

test("download filename prefers the RFC 5987 Unicode parameter", () => {
  assert.equal(
    downloadFilename(
      'attachment; filename="audit-___export.json"; filename*=UTF-8\'\'audit-%C3%A0%E6%BC%A2_export.json',
      "fallback.json",
    ),
    "audit-à漢_export.json",
  );
  assert.equal(downloadFilename('attachment; filename="plain.json"', "fallback.json"), "plain.json");
  assert.equal(downloadFilename("attachment", "fallback.json"), "fallback.json");
});

test("embedding compatibility gate only opens for a verified match", () => {
  assert.deepEqual(
    embeddingCompatibilityGate({ status: "compatible" }),
    { blocked: false, status: "compatible", message: "" },
  );

  for (const status of ["incompatible"]) {
    const gate = embeddingCompatibilityGate({ status });
    assert.equal(gate.blocked, true);
    assert.equal(gate.status, status);
    assert.ok(gate.message.length > 0);
  }
});

test("embedding compatibility gate fails closed while loading or unknown", () => {
  assert.equal(
    embeddingCompatibilityGate(null, { loading: true }).status,
    "loading",
  );
  assert.equal(embeddingCompatibilityGate(null).blocked, true);
  assert.equal(
    embeddingCompatibilityGate(null, { hasCollection: false }).status,
    "no_collection",
  );
});

test("embedder profiles are limited to the two supported display values", () => {
  assert.equal(displayEmbedderProfile(" BASIC "), "basic");
  assert.equal(displayEmbedderProfile("Advanced"), "advanced");
  assert.equal(displayEmbedderProfile("custom"), "—");
  assert.equal(displayEmbedderProfile(null), "—");
});

function queryState(overrides = {}) {
  return {
    ...defaultQueryState(),
    query: "hybrid search",
    collectionName: COLLECTION,
    ...overrides,
  };
}

test("RRF balance is anchored to Qdrant's [1, 1] default", () => {
  assert.deepEqual(rrfWeightsFromBalance(0.5), [1, 1]);
  assert.deepEqual(rrfWeightsFromBalance(0.25), [0.5, 1.5]);
  assert.deepEqual(rrfWeightsFromBalance(0.75), [1.5, 0.5]);
  assert.deepEqual(rrfWeightsFromBalance(0), [0, 2]);
  assert.deepEqual(rrfWeightsFromBalance(1), [2, 0]);
  assert.deepEqual(rrfWeightsFromBalance(0.75, 2), [3, 1]);
  for (const balance of [0, 0.1, 0.25, 0.5, 0.8, 1]) {
    const weights = rrfWeightsFromBalance(balance);
    const mirrored = rrfWeightsFromBalance(1 - balance);
    assert.ok(Math.abs(weights[0] + weights[1] - 2) < 1e-9);
    assert.deepEqual(mirrored, [weights[1], weights[0]]);
  }
});

test("center mapping prevents the old [0.5, 0.5] ranking inversion", () => {
  const contribution = (rank, weight, k = 2) =>
    weight === 0 ? 0 : 1 / (k + ((rank + 1) / weight) - 1);
  const scores = ([denseWeight, sparseWeight]) => ({
    a: contribution(0, sparseWeight),
    b: contribution(1, denseWeight) + contribution(3, sparseWeight),
  });

  const current = scores(rrfWeightsFromBalance(0.5));
  const old = scores([0.5, 0.5]);
  assert.ok(current.b > current.a);
  assert.ok(old.a > old.b);
});

test("invalid RRF balances are rejected", () => {
  for (const value of [Number.NaN, "", false, -0.01, 1.01]) {
    assert.throws(() => rrfWeightsFromBalance(value), /between 0 and 1/);
  }
  for (const scale of [Number.NaN, "", false, 0, -1, 0.001, 1_000_001]) {
    assert.throws(() => rrfWeightsFromBalance(0.5, scale), /must be between/);
  }
});

test("request payload sends tuned RRF weights only for RRF", () => {
  const unset = buildRequestPayload(queryState());
  assert.equal("rrf_weights" in unset, false);

  const centered = buildRequestPayload(queryState({
    rrfBalance: 0.5,
    rrfWeightScale: 1,
  }));
  assert.deepEqual(centered.rrf_weights, [1, 1]);

  const absolute = buildRequestPayload(queryState({
    rrfBalance: 0.75,
    rrfWeightScale: 2,
  }));
  assert.deepEqual(absolute.rrf_weights, [3, 1]);

  const dbsf = buildRequestPayload(queryState({
    fusion: "dbsf",
    rrfK: 60,
    rrfBalance: 0.75,
  }));
  assert.equal("rrf_k" in dbsf, false);
  assert.equal("rrf_weights" in dbsf, false);
});

test("rerank always requests text without duplicating it", () => {
  const noneSelected = buildRequestPayload(queryState({ payloadFields: [] }));
  assert.deepEqual(noneSelected.with_payload, ["text"]);

  const titleSelected = buildRequestPayload(queryState({ payloadFields: ["title"] }));
  assert.deepEqual(titleSelected.with_payload, ["title", "text"]);

  const textSelected = buildRequestPayload(queryState({ payloadFields: ["text"] }));
  assert.deepEqual(textSelected.with_payload, ["text"]);

  const rerankOff = buildRequestPayload(queryState({
    rerank: false,
    payloadFields: [],
  }));
  assert.equal("with_payload" in rerankOff, false);
});

test("runtime query configuration controls defaults, limits, and help", () => {
  const config = normalizeQueryConfig({
    queryDefaultLimit: 7,
    queryMaxLimit: 80,
    queryDefaultCandidateLimit: 70,
    queryMaxCandidateLimit: 400,
    queryDefaultFusion: "dbsf",
    queryDefaultRerank: false,
    queryDefaultRerankTopK: 30,
    queryMaxRerankTopK: 300,
    queryRerankMaxChars: 2500,
  });
  const state = defaultQueryState(config);
  assert.equal(state.limit, 7);
  assert.equal(state.candidateLimit, 70);
  assert.equal(state.fusion, "dbsf");
  assert.equal(state.rerank, false);

  assert.deepEqual(
    validateQueryState(queryState({ limit: 51, candidateLimit: 70 }), {}, config),
    [],
  );
  assert.match(buildQueryParamHelp(config).limit, /1-80/);
  assert.match(buildQueryParamHelp(config).rerank_top_k, /1-300/);
  assert.match(buildQueryParamHelp(config).rerank, /2500/);
});

test("blank, zero, and fractional numeric inputs are not silently coerced", () => {
  const invalidValues = [Number.NaN, 0, 2.5];
  for (const limit of invalidValues) {
    assert.ok(validateQueryState(queryState({ limit })).some((message) =>
      message.startsWith("limit must be an integer")));
  }
  assert.ok(validateQueryState(queryState({ rrfK: 2.5 })).some((message) =>
    message.startsWith("rrf_k must be an integer")));
  assert.ok(validateQueryState(queryState({
    rrfBalance: 0.5,
    rrfWeightScale: Number.NaN,
  })).some((message) => message.startsWith("RRF weight scale")));
});

test("request coordinator rejects stale runs without unlocking a newer run", () => {
  const coordinator = createQueryRequestCoordinator();
  const runA = coordinator.begin();
  assert.equal(coordinator.isBusy(), true);
  assert.equal(coordinator.canCommit(runA), true);

  coordinator.changed();
  assert.equal(coordinator.isBusy(), false);
  assert.equal(coordinator.canCommit(runA), false);

  const runB = coordinator.begin();
  assert.equal(coordinator.isBusy(), true);
  coordinator.finish(runA);
  assert.equal(coordinator.isBusy(), true);
  assert.equal(coordinator.canCommit(runA), false);
  assert.equal(coordinator.canCommit(runB), true);

  coordinator.finish(runB);
  assert.equal(coordinator.isBusy(), false);
});

function filterRow(overrides = {}) {
  return {
    group: "must",
    fieldName: "status",
    op: "eq",
    rawValue: "active",
    rawValue2: "",
    rawValue3: "",
    rawValue4: "",
    ...overrides,
  };
}

test("an incomplete filter row is rejected instead of silently omitted", () => {
  const [condition, error] = buildFilterCondition(
    filterRow({ rawValue: "" }),
    "keyword",
  );

  assert.equal(condition, null);
  assert.match(error, /value is required/i);
});

test("same-element groups with distinct ids stay independent", () => {
  const rows = [
    filterRow({
      group: "must_nested",
      nestedGroupId: "items-a",
      fieldName: "items[].label",
      rawValue: "a",
    }),
    filterRow({
      group: "must_nested",
      nestedGroupId: "items-a",
      fieldName: "items[].qty",
      op: "gte",
      rawValue: "90",
    }),
    filterRow({
      group: "must_nested",
      nestedGroupId: "items-b",
      fieldName: "items[].label",
      rawValue: "b",
    }),
    filterRow({
      group: "must_nested",
      nestedGroupId: "items-b",
      fieldName: "items[].qty",
      op: "lte",
      rawValue: "10",
    }),
  ];

  const [filter, errors] = buildFilterGroups(rows, {
    "items[].label": "keyword",
    "items[].qty": "integer",
  });

  assert.deepEqual(errors, []);
  assert.equal(filter.must.length, 2);
  assert.ok(filter.must.every((condition) => condition.nested.key === "items"));
  assert.ok(filter.must.every((condition) => condition.nested.filter.must.length === 2));
});

test("polygon holes are serialized as Qdrant interior rings", () => {
  const [condition, error] = buildFilterCondition(filterRow({
    fieldName: "location",
    op: "geo_polygon",
    rawValue: "0 0; 4 0; 4 4; 0 0 | 1 1; 2 1; 1 2; 1 1",
  }), "geo");

  assert.equal(error, null);
  assert.equal(condition.geo_polygon.exterior.points.length, 4);
  assert.equal(condition.geo_polygon.interiors.length, 1);
});

test("point-id lists preserve uint64 values beyond JavaScript safe integers", () => {
  const [condition, error] = buildFilterCondition(filterRow({
    fieldName: "(point id)",
    op: "id_any",
    rawValue: '[1, "18446744073709551615"]',
  }), "point_id");

  assert.equal(error, null);
  assert.deepEqual(condition.has_id, [1, "18446744073709551615"]);
});

test("point-id JSON numbers cannot hide rounding behind string conversion", () => {
  for (const rawValue of ["[9007199254740993]", "[18446744073709551615]"]) {
    const [condition, error] = buildFilterCondition({
      fieldName: "(point id)", op: "id_any", rawValue,
    }, "point_id");
    assert.equal(condition, null);
    assert.match(error, /quoted ID string/);
  }
});

test("custom range and values-count preserve inclusive/exclusive bounds", () => {
  const [range, rangeError] = buildFilterCondition(filterRow({
    fieldName: "price",
    op: "range",
    lowerOp: "gt",
    upperOp: "lte",
    rawValue: "10.5",
    rawValue2: "20.5",
  }), "float");
  const [count, countError] = buildFilterCondition(filterRow({
    fieldName: "tags",
    op: "values_count",
    lowerOp: "gte",
    upperOp: "lt",
    rawValue: "2",
    rawValue2: "5",
  }), "keyword");

  assert.equal(rangeError, null);
  assert.deepEqual(range.range, { gt: 10.5, lte: 20.5 });
  assert.equal(countError, null);
  assert.deepEqual(count.values_count, { gte: 2, lt: 5 });
});

test("datetime ranges compare instants across timezone offsets", () => {
  const lower = "2026-01-01T10:00:00+02:00";
  const upper = "2026-01-01T09:00:00Z";
  const [condition, error] = buildFilterCondition(filterRow({
    fieldName: "created", op: "range", rawValue: lower, rawValue2: upper,
  }), "datetime");
  assert.equal(error, null);
  assert.deepEqual(condition.range, { gte: lower, lte: upper });
  const [reversed, reversedError] = buildFilterCondition(filterRow({
    fieldName: "created", op: "range", rawValue: upper, rawValue2: lower,
  }), "datetime");
  assert.equal(reversed, null);
  assert.match(reversedError, /cannot exceed/);
});

test("JSON list input keeps keyword values containing commas", () => {
  const [condition, error] = buildFilterCondition(filterRow({
    op: "any",
    rawValue: '["ACME, Inc.", "Other"]',
  }), "keyword");

  assert.equal(error, null);
  assert.deepEqual(condition.match.any, ["ACME, Inc.", "Other"]);
});

test("JSON keyword lists and numeric bounds reject lossy or non-finite values", () => {
  const [keywordCondition, keywordError] = buildFilterCondition(filterRow({
    op: "any",
    rawValue: '["valid", {"not": "a keyword"}]',
  }), "keyword");
  const [rangeCondition, rangeError] = buildFilterCondition(filterRow({
    fieldName: "score",
    op: "gte",
    rawValue: "1e999",
  }), "float");

  assert.equal(keywordCondition, null);
  assert.match(keywordError, /must all be strings/i);
  assert.equal(rangeCondition, null);
  assert.match(rangeError, /must be a number/i);
});

test("vector presence compiles to has_vector", () => {
  const [condition, error] = buildFilterCondition(filterRow({
    fieldName: "(vector presence)",
    op: "has_vector",
    rawValue: "sparse",
  }), "vector");

  assert.equal(error, null);
  assert.deepEqual(condition, { has_vector: "sparse" });
});

test("visual logical groups compile to recursive Qdrant filters", () => {
  const tree = {
    kind: "root",
    minShould: 1,
    children: [
      {
        kind: "group",
        clause: "must_not",
        label: "excluded pair",
        minShould: 1,
        children: [
          filterRow({ kind: "condition", clause: "must", fieldName: "city", rawValue: "London" }),
          filterRow({ kind: "condition", clause: "must", fieldName: "color", rawValue: "red" }),
        ],
      },
    ],
  };

  const [filter, errors] = buildFilterTree(tree, {
    city: "keyword",
    color: "keyword",
  });

  assert.deepEqual(errors, []);
  assert.equal(filter.must_not[0].must.length, 2);
  assert.equal(filter.must_not[0].must[0].key, "city");
});

test("same-element tree supports inner should, must-not, and min-should", () => {
  const tree = {
    kind: "root",
    minShould: 1,
    children: [
      {
        kind: "nested",
        clause: "must",
        root: "items",
        label: "items[]",
        minShould: 2,
        children: [
          filterRow({
            kind: "condition",
            clause: "should",
            fieldName: "items[].label",
            rawValue: "a",
          }),
          filterRow({
            kind: "condition",
            clause: "should",
            fieldName: "items[].qty",
            op: "gte",
            rawValue: "10",
          }),
          filterRow({
            kind: "condition",
            clause: "must_not",
            fieldName: "items[].status",
            rawValue: "deleted",
          }),
        ],
      },
    ],
  };

  const [filter, errors] = buildFilterTree(tree, {
    "items[].label": "keyword",
    "items[].qty": "integer",
    "items[].status": "keyword",
  });

  assert.deepEqual(errors, []);
  const nestedFilter = filter.must[0].nested.filter;
  assert.equal(nestedFilter.min_should.min_count, 2);
  assert.equal(nestedFilter.min_should.conditions.length, 2);
  assert.equal(nestedFilter.must_not[0].key, "status");
});

test("visual tree depth and condition limits mirror backend guards", () => {
  const tree = {
    kind: "root",
    minShould: 1,
    children: [{
      kind: "group",
      clause: "must",
      minShould: 1,
      children: [filterRow({ kind: "condition", clause: "must" })],
    }],
  };

  assert.deepEqual(filterTreeStats(tree), { conditions: 2, maxDepth: 4 });
  const errors = validateQueryState(queryState({ filterTree: tree }), { status: "keyword" }, {
    queryFilterMaxDepth: 3,
    queryFilterMaxConditions: 1,
  });
  assert.ok(errors.includes("filters exceed maximum depth 3."));
  assert.ok(errors.includes("filters exceed maximum condition count 1."));
});

test("collectionLockState locks incompatible collections only when services are up", () => {
  // Services down → readiness owns the lockdown, never lock here.
  assert.deepEqual(
    collectionLockState({ status: "incompatible", stored: {}, active: {} },
      { servicesReady: false }),
    { locked: false, recoverable: false },
  );

  // Compatible → never locked.
  assert.deepEqual(
    collectionLockState({ status: "compatible", stored: {}, active: {} },
      { servicesReady: true }),
    { locked: false, recoverable: false },
  );

  // Recoverable: app-native fingerprint drift (stored + active present).
  assert.deepEqual(
    collectionLockState(
      { status: "incompatible", stored: { fingerprint: "sha256:a" },
        active: { fingerprint: "sha256:b" } },
      { servicesReady: true }),
    { locked: true, recoverable: true },
  );

  // Foreign: no stored contract → locked but not recoverable.
  assert.deepEqual(
    collectionLockState({ status: "incompatible", stored: null, active: {} },
      { servicesReady: true }),
    { locked: true, recoverable: false },
  );

  // active == null → embedder-availability problem, not a lock (avoids wrong banner).
  assert.deepEqual(
    collectionLockState({ status: "incompatible", stored: {}, active: null },
      { servicesReady: true }),
    { locked: false, recoverable: false },
  );

  // Null compatibility (no report yet) → not locked.
  assert.deepEqual(
    collectionLockState(null, { servicesReady: true }),
    { locked: false, recoverable: false },
  );

  // Defaults: missing options object → servicesReady defaults false → not locked.
  assert.deepEqual(
    collectionLockState({ status: "incompatible", stored: {}, active: {} }),
    { locked: false, recoverable: false },
  );
});

const NO_TOGGLES = {
  denseOnDisk: null,
  quantization: null,
  sparseIdf: null,
  sparseOnDisk: null,
  payloadOnDisk: null,
};

test("buildCreatePayload carries recreate and its confirmation when not create-only", () => {
  assert.deepEqual(
    buildCreatePayload("docs", true, "docs", NO_TOGGLES),
    { collection_name: "docs", recreate: true, confirm_recreate: "docs" },
  );
  // Explicit create-only=false behaves exactly like the omitted options object.
  assert.deepEqual(
    buildCreatePayload("docs", true, "docs", NO_TOGGLES, { createOnly: false }),
    { collection_name: "docs", recreate: true, confirm_recreate: "docs" },
  );
});

test("buildCreatePayload strips recreate in create-only mode", () => {
  // The active collection is embedder-incompatible: the form may create a new
  // collection but must never be able to destroy the existing one, whatever the
  // checkbox state happens to be.
  assert.deepEqual(
    buildCreatePayload("docs", true, "docs", NO_TOGGLES, { createOnly: true }),
    { collection_name: "docs", recreate: false },
  );
  // Toggles still ride along — create-only only touches the destructive fields.
  assert.deepEqual(
    buildCreatePayload("docs", true, "docs", {
      ...NO_TOGGLES, denseOnDisk: true, quantization: false,
    }, { createOnly: true }),
    {
      collection_name: "docs",
      recreate: false,
      dense_on_disk: true,
      quantization: false,
    },
  );
});

test("summarizeIngestChunks counts valid chunks and derives payload fields", () => {
  const s = summarizeIngestChunks({
    chunks: [
      { id: 1, text: "a", source: "doc", page: 1 },
      { id: 2, text: "b", source: "doc", page: 2, tag: "intro" },
    ],
  });
  assert.equal(s.total, 2);
  assert.equal(s.valid, 2);
  assert.equal(s.invalid, 0);
  assert.deepEqual(s.errors, []);
  const byName = Object.fromEntries(s.fields.map((f) => [f.name, f]));
  assert.deepEqual(Object.keys(byName).sort(), ["page", "source", "tag"]);
  assert.equal(byName.source.type, "string");
  assert.equal(byName.source.coverage, "2/2");
  assert.equal(byName.source.sample, "doc");
  assert.equal(byName.page.type, "int");
  assert.equal(byName.tag.coverage, "1/2");
});

test("summarizeIngestChunks never lists id or text as payload fields", () => {
  // `id` has a synthetic row of its own (its payload key travels as
  // chunk_id_key), so it must not also appear among the ordinary fields.
  const s = summarizeIngestChunks({ chunks: [{ id: 1, text: "hello" }] });
  assert.deepEqual(s.fields, []);
});

test("summarizeIngestChunks flags mixed field types and floats", () => {
  const s = summarizeIngestChunks({
    chunks: [
      { id: 1, text: "a", score: 1 },
      { id: 2, text: "b", score: "high" },
      { id: 3, text: "c", ratio: 0.5 },
    ],
  });
  const byName = Object.fromEntries(s.fields.map((f) => [f.name, f]));
  assert.equal(byName.score.type, "mixed");
  assert.equal(byName.ratio.type, "float");
});

test("summarizeIngestChunks counts invalid chunks and surfaces errors", () => {
  const s = summarizeIngestChunks({
    chunks: [
      { id: 1, text: "ok" },
      { id: "x", text: "" },
    ],
  });
  assert.equal(s.total, 2);
  assert.equal(s.valid, 1);
  assert.equal(s.invalid, 1);
  assert.ok(s.errors.length >= 1);
});

test("summarizeIngestChunks handles a missing chunks array", () => {
  const s = summarizeIngestChunks({});
  assert.equal(s.total, 0);
  assert.equal(s.valid, 0);
  assert.equal(s.invalid, 0);
  assert.deepEqual(s.fields, []);
  assert.deepEqual(s.errors, ["Missing 'chunks' array."]);
});

test("summarizeIngestChunks truncates long samples", () => {
  const s = summarizeIngestChunks({
    chunks: [{ id: 1, text: "a", note: "x".repeat(80) }],
  });
  const note = s.fields.find((f) => f.name === "note");
  assert.ok(note.sample.length <= 40);
  assert.ok(note.sample.endsWith("…"));
});

test("coercePayloadValue casts ints from number, int-valued float, and string", () => {
  assert.deepEqual(coercePayloadValue(232, "integer"), { ok: true, value: 232 });
  assert.deepEqual(coercePayloadValue(232.0, "integer"), { ok: true, value: 232 });
  assert.deepEqual(coercePayloadValue("232", "integer"), { ok: true, value: 232 });
  assert.deepEqual(coercePayloadValue(0.5, "integer"), { ok: false });
  assert.deepEqual(coercePayloadValue("2.5", "integer"), { ok: false });
  assert.deepEqual(coercePayloadValue("abc", "integer"), { ok: false });
});

test("coercePayloadValue casts floats from number and numeric string", () => {
  assert.deepEqual(coercePayloadValue(0.5, "float"), { ok: true, value: 0.5 });
  assert.deepEqual(coercePayloadValue("1.25", "float"), { ok: true, value: 1.25 });
  assert.deepEqual(coercePayloadValue("", "float"), { ok: false });
  assert.deepEqual(coercePayloadValue("x", "float"), { ok: false });
});

test("coercePayloadValue casts bools from boolean, 0/1, and true/false strings", () => {
  assert.deepEqual(coercePayloadValue(true, "bool"), { ok: true, value: true });
  assert.deepEqual(coercePayloadValue(1, "bool"), { ok: true, value: true });
  assert.deepEqual(coercePayloadValue(0, "bool"), { ok: true, value: false });
  assert.deepEqual(coercePayloadValue("TRUE", "bool"), { ok: true, value: true });
  assert.deepEqual(coercePayloadValue("false", "bool"), { ok: true, value: false });
  assert.deepEqual(coercePayloadValue(2, "bool"), { ok: false });
  assert.deepEqual(coercePayloadValue("yes", "bool"), { ok: false });
});

test("coercePayloadValue stringifies scalars but refuses array/object", () => {
  assert.deepEqual(coercePayloadValue(232, "keyword"), { ok: true, value: "232" });
  assert.deepEqual(coercePayloadValue(true, "keyword"), { ok: true, value: "true" });
  assert.deepEqual(coercePayloadValue(null, "keyword"), { ok: true, value: "null" });
  assert.deepEqual(coercePayloadValue([1], "keyword"), { ok: false });
  assert.deepEqual(coercePayloadValue({ a: 1 }, "keyword"), { ok: false });
});

test("coercePayloadValue validates geo object without casting", () => {
  const obj = { a: 1 };
  assert.deepEqual(coercePayloadValue(obj, "geo"), { ok: true, value: obj });
  assert.deepEqual(coercePayloadValue([1], "geo"), { ok: false });
  assert.deepEqual(coercePayloadValue(null, "geo"), { ok: false });
});

test("applyIngestFieldMapping renames keys and coerces values, leaving id/text and unmapped keys", () => {
  const chunks = [
    { id: 1, text: "a", token_count: "232", source: "doc" },
    { id: 2, text: "b", token_count: "17", source: "doc" },
  ];
  const mapping = [
    { field: "token_count", name: "tokens", type: "integer" },
    { field: "source", name: "source", type: "keyword" },
  ];
  const { chunks: out, errors } = applyIngestFieldMapping(chunks, mapping);
  assert.deepEqual(errors, []);
  assert.deepEqual(out[0], { id: 1, text: "a", tokens: 232, source: "doc" });
  assert.deepEqual(out[1], { id: 2, text: "b", tokens: 17, source: "doc" });
});

test("applyIngestFieldMapping untyped passes values through unchanged under the new name", () => {
  const { chunks: out, errors } = applyIngestFieldMapping(
    [{ id: 1, text: "a", tag: { nested: true } }],
    [{ field: "tag", name: "label", type: "untyped" }],
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(out[0], { id: 1, text: "a", label: { nested: true } });
});

test("applyIngestFieldMapping skips a field absent from a chunk", () => {
  const { chunks: out, errors } = applyIngestFieldMapping(
    [{ id: 1, text: "a" }, { id: 2, text: "b", page: 3 }],
    [{ field: "page", name: "page", type: "integer" }],
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(out[0], { id: 1, text: "a" });
  assert.deepEqual(out[1], { id: 2, text: "b", page: 3 });
});

test("applyIngestFieldMapping reports empty, reserved, and duplicate names and blocks chunks", () => {
  const chunks = [{ id: 1, text: "a", f1: "x", f2: "y", f3: "z" }];
  const { chunks: out, errors } = applyIngestFieldMapping(chunks, [
    { field: "f1", name: "", type: "untyped" },
    { field: "f2", name: "id", type: "untyped" },
    { field: "f3", name: "dup", type: "untyped" },
    { field: "f4", name: "dup", type: "untyped" },
  ]);
  assert.equal(out.length, 0);
  assert.ok(errors.some((e) => e.includes("f1") && e.includes("required")));
  assert.ok(errors.some((e) => e.includes("f2") && e.includes("reserved")));
  assert.ok(errors.some((e) => e.includes("dup")));
});

test("applyIngestFieldMapping reports a coercion failure once per field and blocks chunks", () => {
  const chunks = [
    { id: 1, text: "a", source: "doc" },
    { id: 2, text: "b", source: "also-not-int" },
  ];
  const { chunks: out, errors } = applyIngestFieldMapping(chunks, [
    { field: "source", name: "source", type: "integer" },
  ]);
  assert.equal(out.length, 0);
  assert.equal(errors.length, 1);
  assert.ok(errors[0].includes("source") && errors[0].includes("integer"));
});

test("buildIngestChunksPayload defaults source_id key and type", () => {
  const body = buildIngestChunksPayload([{ id: 1, text: "a" }], "docs", 64, false, "doc_a");
  assert.equal(body.source_id, "doc_a");
  assert.equal(body.source_id_key, "source_id");
  assert.equal(body.source_id_type, "keyword");
});

test("buildIngestChunksPayload carries a custom source_id key and type", () => {
  const body = buildIngestChunksPayload(
    [{ id: 1, text: "a" }], "docs", 64, false, "12345", "doc_ref", "integer");
  assert.equal(body.source_id, "12345");
  assert.equal(body.source_id_key, "doc_ref");
  assert.equal(body.source_id_type, "integer");
});

test("coercePayloadValue keyword and text stringify scalars, refuse containers", () => {
  assert.deepEqual(coercePayloadValue(232, "keyword"), { ok: true, value: "232" });
  assert.deepEqual(coercePayloadValue(true, "text"), { ok: true, value: "true" });
  assert.deepEqual(coercePayloadValue([1], "keyword"), { ok: false });
  assert.deepEqual(coercePayloadValue({ a: 1 }, "text"), { ok: false });
});

test("coercePayloadValue float rejects non-finite numbers", () => {
  assert.deepEqual(coercePayloadValue(1.5, "float"), { ok: true, value: 1.5 });
  assert.deepEqual(coercePayloadValue(Infinity, "float"), { ok: false });
});

test("coercePayloadValue datetime accepts a parseable date, rejects a non-date", () => {
  assert.deepEqual(coercePayloadValue("2020-01-02", "datetime"), { ok: true, value: "2020-01-02" });
  assert.deepEqual(coercePayloadValue("not-a-date", "datetime"), { ok: false });
  assert.deepEqual(coercePayloadValue(5, "datetime"), { ok: false });
});

test("coercePayloadValue uuid accepts a valid uuid, rejects other strings", () => {
  assert.deepEqual(
    coercePayloadValue("12345678-1234-1234-1234-123456789abc", "uuid"),
    { ok: true, value: "12345678-1234-1234-1234-123456789abc" });
  assert.deepEqual(coercePayloadValue("doc_a", "uuid"), { ok: false });
});

test("coercePayloadValue geo accepts an object, rejects a scalar or array", () => {
  const g = { lat: 1, lon: 2 };
  assert.deepEqual(coercePayloadValue(g, "geo"), { ok: true, value: g });
  assert.deepEqual(coercePayloadValue("x", "geo"), { ok: false });
  assert.deepEqual(coercePayloadValue([1], "geo"), { ok: false });
});

test("chunkArray splits into exact and remainder groups", () => {
  assert.deepEqual(chunkArray([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);
  assert.deepEqual(chunkArray([1, 2, 3, 4], 2), [[1, 2], [3, 4]]);
});

test("chunkArray returns one group when size >= length", () => {
  assert.deepEqual(chunkArray([1, 2, 3], 10), [[1, 2, 3]]);
});

test("chunkArray guards a non-positive size to 1", () => {
  assert.deepEqual(chunkArray([1, 2], 0), [[1], [2]]);
  assert.deepEqual(chunkArray([1, 2], -3), [[1], [2]]);
});

test("chunkArray handles empty or non-array input", () => {
  assert.deepEqual(chunkArray([], 3), []);
  assert.deepEqual(chunkArray(null, 3), []);
});

test("INGEST_SOURCE_ID_RESERVED_KEYS mirrors the backend CHUNK_RESERVED_FIELDS plus text", () => {
  for (const k of ["id", "text", "overflow_details", "overlap_sentences_count", "split_metadata"]) {
    assert.ok(INGEST_SOURCE_ID_RESERVED_KEYS.has(k), `expected ${k} reserved`);
  }
});

test("the two reserved-key sets mirror their backend validators", () => {
  // id/text are what the middleware reads off every chunk, so neither may be
  // an ordinary field row nor a rename target.
  assert.deepEqual([...INGEST_EXCLUDED_KEYS].sort(), ["id", "text"]);
  // _validate_chunk_id_key: everything CHUNK_RESERVED_FIELDS holds except `id`,
  // which is the legitimate default, plus `text`.
  assert.deepEqual([...INGEST_CHUNK_ID_RESERVED_KEYS].sort(),
    ["overflow_details", "overlap_sentences_count", "split_metadata", "text"]);
});

test("summarizeIngestChunks lists chunker-internal keys while stripping is off", () => {
  const s = summarizeIngestChunks({
    chunks: [
      { id: 1, text: "a", source: "doc", overflow_details: {}, split_metadata: {}, overlap_sentences_count: 2 },
    ],
  });
  const names = s.fields.map((f) => f.name).sort();
  assert.deepEqual(names, ["overflow_details", "overlap_sentences_count", "source", "split_metadata"]);
});

test("applyIngestFieldMapping rejects renaming a field onto id or text", () => {
  for (const reserved of ["id", "text"]) {
    const { chunks, errors } = applyIngestFieldMapping(
      [{ id: 1, text: "a", tag: "x" }],
      [{ field: "tag", name: reserved, type: "untyped" }],
    );
    assert.equal(chunks.length, 0);
    assert.ok(errors.some((e) => e.includes(reserved) && e.includes("reserved")));
  }
});

test("applyIngestFieldMapping leaves the chunk ordinal on the chunk", () => {
  // points.py:_validate_chunks 400s on a chunk whose `id` is not an integer and
  // the point id derives from it, so the client never rewrites it. Its payload
  // key is chosen server-side via chunk_id_key instead.
  const chunks = [{ id: 256, text: "a", token_count: 261 }];
  const out = applyIngestFieldMapping(chunks, [
    { field: "token_count", name: "token", type: "integer" },
  ]);
  assert.deepEqual(out.errors, []);
  assert.deepEqual(out.chunks, [{ id: 256, text: "a", token: 261 }]);
});

test("buildIngestChunksPayload carries chunk_id_key, defaulting to id", () => {
  const chunks = [{ id: 1, text: "a" }];
  assert.equal(
    buildIngestChunksPayload(chunks, "docs", 64, false, "doc_a").chunk_id_key, "id");
  assert.equal(
    buildIngestChunksPayload(
      chunks, "docs", 64, false, "doc_a", "source_id", "keyword", "id_chunk",
    ).chunk_id_key,
    "id_chunk",
  );
});

test("applyIngestFieldMapping keeps a chunker-internal key mapped onto itself", () => {
  // While CHUNK_EXCLUDE_FIELDS is empty these are ordinary payload fields, so
  // the row's default NAME (the key itself) has to survive the mapping.
  const { chunks, errors } = applyIngestFieldMapping(
    [{ id: 1, text: "a", split_metadata: 2 }],
    [{ field: "split_metadata", name: "split_metadata", type: "integer" }],
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(chunks, [{ id: 1, text: "a", split_metadata: 2 }]);
});

// ── Auto-created payload indexes after ingest ────────────────────────────────

test("buildIngestIndexRequests indexes typed fields, source_id and the ordinal", () => {
  const { requests, alreadyIndexed } = buildIngestIndexRequests(
    [
      { field: "prezzo", name: "prezzo", type: "integer" },
      { field: "attivo", name: "attivo", type: "bool" },
      { field: "note", name: "note", type: "untyped" },
    ],
    { key: "source_id", type: "keyword" },
    "id",
    {},
  );
  assert.deepEqual(requests, [
    { field_name: "source_id", field_schema: "keyword" },
    { field_name: "id", field_schema: "integer" },
    { field_name: "prezzo", field_schema: "integer" },
    { field_name: "attivo", field_schema: "bool" },
  ]);
  assert.deepEqual(alreadyIndexed, []);
});

test("buildIngestIndexRequests never requests a field that is already indexed", () => {
  // The middleware REPLACES an index whose type differs (drop + recreate), so
  // an already-indexed field must never reach the request — whatever its type.
  const { requests, alreadyIndexed } = buildIngestIndexRequests(
    [
      { field: "prezzo", name: "prezzo", type: "integer" },
      { field: "autore", name: "autore", type: "keyword" },
      { field: "tag", name: "tag", type: "keyword" },
    ],
    { key: "source_id", type: "keyword" },
    "id",
    { source_id: "keyword", autore: "keyword", tag: "integer" },
  );
  assert.deepEqual(requests, [
    { field_name: "id", field_schema: "integer" },
    { field_name: "prezzo", field_schema: "integer" },
  ]);
  assert.deepEqual(alreadyIndexed, [
    { field_name: "source_id", field_schema: "keyword" },
    { field_name: "autore", field_schema: "keyword" },
    // reported with the EXISTING schema, not the one picked in the dropdown
    { field_name: "tag", field_schema: "integer" },
  ]);
});

test("buildIngestIndexRequests uses final names and skips an untyped source_id", () => {
  const { requests } = buildIngestIndexRequests(
    [{ field: "prezzo", name: "price", type: "float" }],
    { key: "origine", type: "untyped" },
    "chunk_no",
    {},
  );
  assert.deepEqual(requests, [
    { field_name: "chunk_no", field_schema: "integer" },
    { field_name: "price", field_schema: "float" },
  ]);
});

test("buildIngestIndexRequests drops empty names and duplicates", () => {
  const { requests } = buildIngestIndexRequests(
    [
      { field: "a", name: "  ", type: "keyword" },
      { field: "b", name: " tag ", type: "keyword" },
      { field: "c", name: "tag", type: "integer" },
    ],
    { key: "", type: "keyword" },
    "  ",
    {},
  );
  assert.deepEqual(requests, [{ field_name: "tag", field_schema: "keyword" }]);
});

test("summarizeIngestIndexResult merges the response with client-excluded fields", () => {
  const requests = [
    { field_name: "prezzo", field_schema: "integer" },
    { field_name: "attivo", field_schema: "bool" },
    { field_name: "tag", field_schema: "keyword" },
  ];
  const summary = summarizeIngestIndexResult(
    { created: ["prezzo", "attivo"], updated: ["tag"], skipped: [], failed: [] },
    requests,
    [{ field_name: "source_id", field_schema: "keyword" }],
  );
  assert.deepEqual(summary.counts,
    { created: 2, updated: 1, skipped: 1, failed: 0 });
  assert.deepEqual(summary.rows, [
    { field: "prezzo", schema: "integer", status: "created" },
    { field: "attivo", schema: "bool", status: "created" },
    { field: "tag", schema: "keyword", status: "updated" },
    { field: "source_id", schema: "keyword", status: "skipped" },
  ]);
  assert.equal(summary.replaced, true);
  assert.equal(summary.reason, "");
});

test("summarizeIngestIndexResult counts a server skip as already indexed", () => {
  const summary = summarizeIngestIndexResult(
    { created: [], updated: [], skipped: ["prezzo"], failed: [] },
    [{ field_name: "prezzo", field_schema: "integer" }],
    [],
  );
  assert.deepEqual(summary.counts,
    { created: 0, updated: 0, skipped: 1, failed: 0 });
  assert.deepEqual(summary.rows, [
    { field: "prezzo", schema: "integer", status: "skipped" },
  ]);
  assert.equal(summary.replaced, false);
});

test("summarizeIngestIndexResult fails every requested field when the call fails", () => {
  const summary = summarizeIngestIndexResult(
    null,
    [
      { field_name: "prezzo", field_schema: "integer" },
      { field_name: "attivo", field_schema: "bool" },
    ],
    [{ field_name: "source_id", field_schema: "keyword" }],
    // Production passes apiErrorText's output here: the server detail alone,
    // with no "HTTP <status>" prefix.
    "index schema mismatch",
  );
  assert.deepEqual(summary.counts,
    { created: 0, updated: 0, skipped: 1, failed: 2 });
  assert.deepEqual(summary.rows, [
    { field: "prezzo", schema: "integer", status: "failed" },
    { field: "attivo", schema: "bool", status: "failed" },
    { field: "source_id", schema: "keyword", status: "skipped" },
  ]);
  assert.equal(summary.reason, "index schema mismatch");
});

test("buildIngestIndexRequests never requests an index without a schema", () => {
  // The dropdowns always carry a value, so this is a guard against a caller
  // that forgets one — an index request with field_schema undefined would be
  // rejected by the manager as a 422 for the whole batch, taking the valid
  // fields down with it.
  const { requests } = buildIngestIndexRequests(
    [
      { field: "a", name: "a" },
      { field: "b", name: "b", type: "keyword" },
    ],
    { key: "source_id" },
    "id",
    {},
  );
  assert.deepEqual(requests, [
    { field_name: "id", field_schema: "integer" },
    { field_name: "b", field_schema: "keyword" },
  ]);
});

test("ingest preflight rejects duplicate and unsafe ordinals but accepts signed distinct IDs", () => {
  const chunks = (ids) => ({ chunks: ids.map((id) => ({ id, text: "text" })) });
  assert.deepEqual(validateIngestChunks(chunks([-1, 0, 1])), []);
  assert.match(validateIngestChunks(chunks([1, 1]))[0], /chunk\[1\].*duplicate/);
  assert.ok(validateIngestChunks(chunks([Number.MAX_SAFE_INTEGER + 1])).length);
});
