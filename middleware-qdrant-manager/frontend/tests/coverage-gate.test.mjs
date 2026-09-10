// NF-COV3: the book's qualified cases can only gain evidence, never lose it.
// Inputs are all tracked: the book, the contract and the actual collection of
// the browser and Node suites. Execution results are campaign evidence and are
// deliberately not consulted here.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { assignCases, collectNodeTitles, collectPlaywright, nodeCaseKeys, parseNodeTitles, readBook, readContract, summarize } from './coverage-gate.mjs';

const NODE_SUITES = ['harness.test.mjs', 'harness-tools.test.mjs', 'case-coverage.test.mjs', 'telemetry.test.mjs']
  .map((name) => fileURLToPath(new URL(`./${name}`, import.meta.url)));

const cases = readBook();
const contract = readContract();
const browser = collectPlaywright('playwright.config.js');
// The external specs annotate profile-dependent rows (E1 under BASIC, E3 under
// ADVANCED), so their inventory is collected once per profile configuration.
const external = [
  ...collectPlaywright('playwright.external.config.js', { QHM_PROFILE: 'basic' }),
  ...collectPlaywright('playwright.external.config.js', { QHM_PROFILE: 'advanced' }),
];
const node = nodeCaseKeys(collectNodeTitles(NODE_SUITES));
const assignment = assignCases(cases, { browser, external, node, contract });
const counts = summarize(assignment);

const destination = fileURLToPath(new URL('../test-results/coverage-map.json', import.meta.url));
mkdirSync(fileURLToPath(new URL('../test-results/', import.meta.url)), { recursive: true });
writeFileSync(destination, JSON.stringify({ kind: 'inventory', generatedAt: new Date().toISOString(), counts,
  ratchet: contract.ratchet, cases: assignment }, null, 2));

test('the collected inventories are real, not empty', () => {
  assert.ok(cases.length >= 500, `book yielded ${cases.length} qualified cases`);
  assert.ok(browser.length >= 400, `default collection yielded ${browser.length} tests`);
  assert.ok(external.length >= 10, `external collection yielded ${external.length} tests`);
  assert.ok(node.size >= 5, `node suites credited ${node.size} TB cases`);
});

test('NF-COV3: unassigned cases do not exceed the ratchet', () => {
  const unassigned = assignment.filter((item) => item.status === 'UNASSIGNED').map((item) => item.key).sort();
  assert.ok(unassigned.length <= contract.ratchet.maxUnassigned,
    `${unassigned.length} unassigned cases exceed the ratchet ${contract.ratchet.maxUnassigned}: ${unassigned.slice(0, 20).join(', ')}`);
});

test('an exclusion whose case gained evidence must be removed, not kept', () => {
  const stale = assignment.filter((item) => item.status === 'STALE_EXCLUSION').map((item) => item.key);
  assert.deepEqual(stale, []);
});

test('every excluded case carries its written reason and every owner is a known suite', () => {
  for (const item of assignment.filter((entry) => entry.status === 'EXCLUDED')) {
    assert.ok(item.reason.length >= 40, `${item.key} lacks a reason`);
  }
  assert.deepEqual(Object.keys(contract.owners).sort(), ['embedder', 'middleware']);
});

test('the zero-unassigned policy cannot be loosened even with a complete inventory', () => {
  assert.equal(contract.ratchet.maxUnassigned, 0);
});

test('inventory preserves partial evidence and never labels ownership as execution PASS', () => {
  const rows = [{ key: 'TB-HAR-004', id: 'TB-HAR-004', section: '0' }];
  const contract = { owners: {}, exclusions: [] };
  const browser = [{ id: 'partial', file: 'test.spec.js', title: 'partial scenario', partialCases: ['TB-HAR-004'] }];
  const [partial] = assignCases(rows, { browser, contract });
  assert.equal(partial.status, 'MAPPED');
  assert.equal(partial.browserCoverage, 'partial');
  const [owned] = assignCases(rows, { contract: { owners: { middleware: ['TB-HAR-004'] } } });
  assert.equal(owned.status, 'OWNED');
  assert.equal(owned.browserCoverage, 'none');
  const [full] = assignCases(rows, { browser: [{ ...browser[0], partialCases: [], cases: ['TB-HAR-004'] }], contract });
  assert.equal(full.browserCoverage, 'full');
});

test('skipped and TODO TAP titles cannot credit a Node case', () => {
  const titles = parseNodeTitles('ok 1 - TB-HAR-001 : executed\nok 2 - TB-HAR-002 : skipped # SKIP\nok 3 - TB-HAR-003 : pending # TODO\nnot ok 4 - TB-HAR-004 : failed');
  assert.deepEqual([...nodeCaseKeys(titles)], ['TB-HAR-001']);
});
