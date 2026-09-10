import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BASE_URL, normalizeBaseURL, validateOwnership, guardMutation, setupCall,
  requireResults, cleanupOwnedCollection } from './harness.mjs';

const owned = 'e2e_unit_owned';
const manifest = { version: 1, baseURL: BASE_URL, runId: 'unit',
  initialCollections: ['stellantis', 'e2e_unit_preexisting'], ownedCollections: [owned] };
const assertOwned = (name) => validateOwnership(manifest, name);

test('TB-HAR-001: exact ownership rejects blank, foreign and preexisting targets before transport', async () => {
  let calls = 0;
  for (const name of ['', ' ', 'cat', 'test', 'reindex_e2e_1', 'e2e_unit_unknown', 'e2e_unit_preexisting']) {
    await assert.rejects(setupCall('POST', '/collections', { collection_name: name }, {
      assertOwned, transport: async () => { calls++; },
    }), /not owned/);
  }
  assert.equal(calls, 0);
  assert.equal(assertOwned(owned), owned);
  assert.throws(() => validateOwnership({ ...manifest, ownedCollections: [...manifest.ownedCollections, 'e2e_unit_preexisting'] }, 'e2e_unit_preexisting'), /not owned/);
});

test('TB-HAR-001: guard checks every target, encoded paths and foreign origins', () => {
  guardMutation('POST', '/points/batch', { collection_name: owned }, assertOwned);
  guardMutation('DELETE', `/collections/${owned}`, undefined, assertOwned);
  assert.throws(() => guardMutation('PATCH', `/collections/${owned}`, { collection_name: 'stellantis' }, assertOwned), /not owned/);
  assert.throws(() => guardMutation('DELETE', '/collections/%20', undefined, assertOwned), /not owned/);
  assert.throws(() => guardMutation('POST', 'http://example.org/collections', { collection_name: owned }, assertOwned), /foreign origin/);
  assert.throws(() => guardMutation('POST', '/points/batch', {}, assertOwned), /missing collection/);
});

test('TB-HAR-002: normalize alternate port and trailing slash once', () => {
  for (const value of ['http://127.0.0.1:16337', 'http://127.0.0.1:16337/']) {
    const base = normalizeBaseURL(value);
    assert.equal(`${base}/collections`, 'http://127.0.0.1:16337/collections');
  }
  assert.throws(() => normalizeBaseURL('http://user:secret@localhost/'));
});

test('TB-HAR-005: setup stops at first failed response and never treats it as an empty oracle', async () => {
  let calls = 0;
  const transport = async () => { calls++; return { status: 500, json: () => ({ detail: 'failure' }) }; };
  await assert.rejects(async () => {
    await setupCall('POST', '/collections', { collection_name: owned }, { assertOwned, transport });
    await setupCall('POST', '/points/batch', { collection_name: owned }, { assertOwned, transport });
  }, /POST \/collections returned 500/);
  assert.equal(calls, 1);
  await assert.rejects(setupCall('GET', '/collections', undefined, { transport }), /returned 500/);
  assert.throws(() => requireResults({ detail: 'failure' }), /malformed/);
  assert.throws(() => requireResults({ results: {} }), /malformed/);
  assert.deepEqual(requireResults({ results: [] }), []);
});

test('TB-HAR-006: cleanup always addresses the owned target, regardless of UI selection', async () => {
  const calls = [];
  let cancelled = false;
  const call = async (method, path) => {
    calls.push([method, path]);
    if (method === 'POST') cancelled = true;
    return { status: 200, data: { job: { status: cancelled ? 'cancelled' : 'paused' }, task_running: false } };
  };
  await cleanupOwnedCollection(owned, { call, assertOwned, pause: async () => {} });
  assert.deepEqual(calls.filter(([method]) => method !== 'GET'), [
    ['POST', `/collections/${owned}/reindex/cancel`],
    ['DELETE', `/collections/${owned}?confirm_delete=${owned}`],
  ]);
  const previous = calls.length;
  await assert.rejects(cleanupOwnedCollection('stellantis', { call, assertOwned }), /not owned/);
  assert.equal(calls.length, previous);
});

test('TB-HAR-006: interrupted cleanup never proceeds to deletion', async () => {
  const methods = [];
  const call = async (method) => {
    methods.push(method);
    if (method === 'POST') throw new Error('interrupted');
    return { status: 200, data: { job: { status: 'paused' }, task_running: false } };
  };
  await assert.rejects(cleanupOwnedCollection(owned, { call, assertOwned }), /interrupted/);
  assert.ok(!methods.includes('DELETE'));
});

test('TB-HAR-006: cancelled status does not release cleanup while native work still runs', async () => {
  let running = true;
  const methods = [];
  const call = async (method) => {
    methods.push(method);
    if (method === 'DELETE') assert.equal(running, false);
    return { status: 200, data: { job: { status: 'cancelled' }, task_running: running } };
  };
  await cleanupOwnedCollection(owned, { call, assertOwned, pause: async () => { running = false; } });
  assert.deepEqual(methods, ['GET', 'GET', 'GET', 'DELETE']);
  await assert.rejects(cleanupOwnedCollection(owned, { assertOwned,
    call: async () => ({ status: 200, data: {} }) }), /Malformed/);
});
