// Exercise the actual seed/reset entrypoints against a disposable HTTP oracle.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { randomBytes } from 'node:crypto';
import { unlink } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const execute = promisify(execFile);
const frontend = fileURLToPath(new URL('../', import.meta.url));

test('impossible rate reservations reject promptly', async () => {
  for (const count of ['11', '-1', 'NaN', '1.5']) {
    await assert.rejects(execute(process.execPath, ['--input-type=module', '-e',
      `const {reserve} = await import('./tests/tb-live-helpers.js'); await reserve(${count});`],
    { cwd: frontend, timeout: 10000, env: { ...process.env, QHM_E2E_RATE_BUDGET: '10' } }),
    (error) => error.code === 1 && error.stderr.includes('Reservation must be'));
  }
  for (const budget of ['0', '-1', 'NaN', '1.5', ' ']) {
    await assert.rejects(execute(process.execPath, ['--input-type=module', '-e',
      "await import('./tests/tb-live-helpers.js');"],
    { cwd: frontend, timeout: 10000, env: { ...process.env, QHM_E2E_RATE_BUDGET: budget } }),
    (error) => error.code === 1 && error.stderr.includes('must be a positive integer'));
  }
});

test('book fixture helpers can be imported without a CLI filename or live setup', async () => {
  const { stdout } = await execute(process.execPath, ['--input-type=module', '-e',
    "const {filterPoint} = await import('./tests/tools/seed-book.mjs'); console.log(filterPoint(1).point_id);"],
  { cwd: frontend, env: { ...process.env, QHM_E2E_RUN_ID: '' } });
  assert.equal(stdout.trim(), '1');
  await assert.rejects(execute(process.execPath,
    ['tests/tools/seed-book.mjs', '--only', 'unknown-fixture'],
    { cwd: frontend, env: { ...process.env, QHM_E2E_RUN_ID: '' } }),
  (error) => error.code === 1 && error.stderr.includes('Unknown book fixture unknown-fixture'));
});

test('book verification rejects nested payload drift', async () => {
  const { verifyStoredPayload } = await import('./tools/seed-book.mjs');
  const point = { point_id: '7', text: 'text', payload: { geo: { lat: 45, lon: 9 }, nested: [{ value: 1 }] } };
  verifyStoredPayload('fixture', point, { text: point.text, ...point.payload });
  for (const change of [{ geo: { lat: 46, lon: 9 } }, { nested: [{ value: 2 }] }, { geo: {} }]) {
    assert.throws(() => verifyStoredPayload('fixture', point, { text: point.text, ...point.payload, ...change }), /payload differs/);
  }
  assert.throws(() => verifyStoredPayload('fixture', point, undefined), /payload differs/);
});

test('TB-HAR-001/002/005: real tools share the guarded origin and stop at failed prerequisites', async () => {
  const collections = new Map([['protected', new Set(['1'])]]);
  const calls = [];
  let failure = null;
  let partial = null;
  const server = createServer(async (request, response) => {
    let raw = '';
    for await (const chunk of request) raw += chunk;
    const body = raw ? JSON.parse(raw) : undefined;
    const path = new URL(request.url, 'http://localhost').pathname;
    calls.push({ method: request.method, path, body });
    response.setHeader('Content-Type', 'application/json');
    if (failure?.(request.method, path)) {
      response.statusCode = 500;
      return response.end(JSON.stringify({ detail: 'injected prerequisite failure' }));
    }
    let data = { ok: true };
    const name = path.split('/')[2];
    if (path === '/collections' && request.method === 'GET') {
      data = { collections: [...collections.keys()] };
    } else if (path === '/collections' && request.method === 'POST') {
      collections.set(body.collection_name, new Set());
    } else if (path === '/points/batch') {
      for (const point of body.points) collections.get(body.collection_name).add(point.point_id);
      data = { ok: true, upserted: body.points.length };
    } else if (path.endsWith('/payload-indexes')) {
      data = { indexes: [], failed: [] };
    } else if (path.endsWith('/reindex')) {
      data = { collection_name: name, job: null, task_running: false };
    } else if (request.method === 'DELETE') {
      collections.delete(name);
    } else if (request.method === 'GET') {
      if (!collections.has(name)) response.statusCode = 404;
      data = { points_count: collections.get(name)?.size ?? 0 };
    }
    if (partial?.(request.method, path)) data = { upserted: 0, failed: ['field'] };
    response.end(JSON.stringify(data));
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const manifests = [];
  const campaign = async (suffix = '') => {
    const runId = `unit_${randomBytes(6).toString('hex')}`;
    const env = { ...process.env, QHM_BASE_URL: base + suffix,
      QHM_E2E_RUN_ID: runId, QHM_E2E_TOKEN: '', QHM_TOKEN: '' };
    const run = (script) => execute(process.execPath, [`tests/tools/${script}.mjs`], { cwd: frontend, env });
    await run('init-run');
    manifests.push(fileURLToPath(new URL(`../test-results/ownership-${runId}.json`, import.meta.url)));
    return { run, runId, env };
  };
  try {
    for (const suffix of ['', '/']) {
      const { run, runId, env } = await campaign(suffix);
      const start = calls.length;
      for (const script of ['seed', 'seed-d', 'seed-big', 'seed-reg', 'reset-a', 'reset-c']) await run(script);
      for (const call of calls.slice(start)) {
        assert.ok(!call.path.startsWith('//'));
        if (call.method === 'GET') continue;
        const name = call.body?.collection_name || call.path.split('/')[2];
        assert.ok(name.startsWith(`e2e_${runId}_`), name);
      }
      assert.deepEqual([...collections.get('protected')], ['1']);
      const beforeOverrides = calls.length;
      for (const variable of ['QHM_REINDEX_COLL', 'QHM_NATIVE', 'QHM_FOREIGN', 'QHM_E7_COLL']) {
        for (const value of ['cat', 'test', 'reindex_e2e_1', ' ', `e2e_${runId}_unknown`]) {
          await assert.rejects(execute(process.execPath, ['--input-type=module', '-e',
            `import { fixtureOverride } from './tests/harness.mjs'; fixtureOverride('${variable}', 'e2e_z_reg');`],
          { cwd: frontend, env: { ...env, [variable]: value } }), /not owned/);
        }
      }
      assert.equal(calls.length, beforeOverrides);
    }
    for (const fail of [
      (method, path) => method === 'POST' && path === '/collections',
      (method, path) => method === 'POST' && path.endsWith('/payload-indexes'),
    ]) {
      const { run } = await campaign();
      failure = fail;
      await assert.rejects(run('seed'), /returned 500/);
      assert.ok(fail(calls.at(-1).method, calls.at(-1).path));
      failure = null;
    }
    for (const fail of [
      (method, path) => method === 'POST' && path === '/points/batch',
      (method, path) => method === 'POST' && path.endsWith('/payload-indexes'),
    ]) {
      const { run } = await campaign();
      partial = fail;
      await assert.rejects(run('seed'), /seed count mismatch|creation failed/);
      assert.ok(fail(calls.at(-1).method, calls.at(-1).path));
      partial = null;
    }
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
    for (const manifest of manifests) await unlink(manifest);
  }
});
