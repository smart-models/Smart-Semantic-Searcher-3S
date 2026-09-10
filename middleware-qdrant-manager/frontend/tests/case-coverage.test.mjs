import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bookCases, executionReport, executionSummary, requireCompleteCoverage, mapCollectedTests } from './case-coverage.mjs';

const cases = bookCases('### 6.2 C — create\n| C1 | create |\n### 8.2 QF — filters\n| C1 | invalid |\n| QF-A | group |\n### 9 M — monitoring\n| M-G | refresh |\n### 0.7 Harness\n| TB-HAR-004 | coverage |');

test('execution summary separates expected failures from passes and counts final retries once', () => {
  assert.deepEqual(executionSummary([
    { id: 'retry', status: 'failed', expectedStatus: 'passed' },
    { id: 'retry', status: 'passed', expectedStatus: 'passed' },
    { id: 'known-gap', status: 'failed', expectedStatus: 'failed' },
    { id: 'pending', status: 'skipped' },
  ]), { passed: 1, failed: 1, timedOut: 0, interrupted: 0, skipped: 1, expectedFailures: 1 });
});

test('TB-HAR-003/004: section-local IDs, digitless and generated titles remain distinct', () => {
  assert.equal(cases.length, 5);
  const collected = [
    { id: 'create', file: 'a-collections.spec.js', title: 'C1 — create' },
    { id: 'filter', file: 'd-filters.spec.js', title: 'QF-A — group' },
    { id: 'monitor', file: 'e-monitoring.spec.js', title: 'M-G — refresh' },
    { id: 'generated', file: 'harness.spec.js', title: `TB-HAR-${String(4).padStart(3, '0')} — generated`, cases: ['TB-HAR-004'] },
  ];
  const mapping = mapCollectedTests(cases, collected);
  const report = executionReport(cases, collected, collected.map(({ id }) => ({ id, status: 'passed' })), mapping);
  assert.equal(report.find((item) => item.key === 'book:6.2/C1').status, 'PASS');
  assert.equal(report.find((item) => item.key === 'book:8.2/C1').status, 'NOT_MAPPED');
  assert.equal(report.find((item) => item.key === 'book:9/M-G').status, 'PASS');
  assert.equal(report.find((item) => item.key === 'TB-HAR-004').status, 'PASS');
  assert.throws(() => requireCompleteCoverage(report), /8.2\/C1/);
});

test('TB-HAR-004: comment-only, excluded, missing, skipped and failed variants cannot count as PASS', () => {
  const mapping = { 'TB-HAR-004': ['variant'] };
  const status = (collected, results) => executionReport(cases, collected, results, mapping).at(-1).status;
  assert.equal(status([], []), 'NOT_RUN');
  assert.equal(status([{ id: 'variant' }], []), 'NOT_RUN');
  assert.equal(status([{ id: 'variant' }], [{ id: 'variant', status: 'skipped' }]), 'SKIPPED');
  assert.equal(status([{ id: 'variant' }], [{ id: 'variant', status: 'failed' }]), 'FAIL');
  assert.equal(executionReport(cases, [], [], {}).at(-1).status, 'NOT_MAPPED');
  assert.throws(() => requireCompleteCoverage(executionReport(cases, [], [], {})), /TB-HAR-004/);
  const partial = [{ id: 'browser-only', file: 'harness.spec.js', title: 'browser variant', partialCases: ['TB-HAR-004'] }];
  const partialReport = executionReport(cases, partial, [{ id: 'browser-only', status: 'passed' }], mapCollectedTests(cases, partial));
  assert.equal(partialReport.at(-1).status, 'PARTIAL');
  assert.throws(() => requireCompleteCoverage(partialReport), /TB-HAR-004/);
});

test('TB-HAR-003: combined rows expand into qualified IDs with named execution variants', () => {
  const rows = bookCases('### 13.4 Combined\n| NF-D3/D4 | threshold |\n### 5 RE\n| RE4/5/7 | lifecycle |');
  assert.deepEqual(rows.map(({ key }) => key), [
    'book:13.4/NF-D3', 'book:13.4/NF-D4', 'book:5/RE4', 'book:5/RE5', 'book:5/RE7',
  ]);
  const collected = [{ id: 'threshold', file: 'nf-newfeatures.spec.js', title: 'NF-D3/D4 — threshold' }];
  const mapped = mapCollectedTests(rows, collected);
  assert.equal(mapped['book:13.4/NF-D4'][0].label, 'nf-newfeatures.spec.js: NF-D3/D4 — threshold');
  const report = executionReport(rows, collected, [{ id: 'threshold', status: 'passed' }], mapped);
  assert.equal(report[1].status, 'PASS');
  assert.equal(report[2].status, 'NOT_MAPPED');
});
