// Static coverage gate: every qualified book case must be claimed by evidence
// that actually exists — a collected browser test (default or external
// configuration), a Node harness test title, a suite named as owner in the
// contract, or a written exclusion. Claims come from real collection output,
// never from spec source text.
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { basename } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { bookCases, contractIndex, mapCollectedTests } from './case-coverage.mjs';

const frontend = fileURLToPath(new URL('../', import.meta.url));
const require = createRequire(import.meta.url);

export function readBook() {
  return bookCases(readFileSync(new URL('./testbook/test-book-e2e.md', import.meta.url), 'utf8'));
}

export function readContract() {
  return JSON.parse(readFileSync(new URL('./testbook/coverage-contract.json', import.meta.url), 'utf8'));
}

function run(args, label, extraEnv = {}) {
  // A nested `node --test` inherits NODE_TEST_CONTEXT from the runner and would
  // speak the runner's binary protocol instead of TAP; clear it for the child.
  const { NODE_TEST_CONTEXT: _ignored, ...env } = process.env;
  const result = spawnSync(process.execPath, args, { cwd: frontend, encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024, env: { ...env, QHM_E2E_RUN_ID: '', PW_TEST_HTML_REPORT_OPEN: 'never', ...extraEnv } });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${label} failed (exit ${result.status}): ${result.stderr}`);
  return result.stdout;
}

// Playwright's JSON reporter for `--list` yields nested suites; static
// `annotation` declarations are present, runtime ones are not (by design).
export function collectPlaywright(config = 'playwright.config.js', env = {}) {
  const stdout = run([require.resolve('@playwright/test/cli'), 'test', '--list', '--reporter=json', '--config', config], `playwright --list ${config}`, env);
  const report = JSON.parse(stdout.slice(stdout.indexOf('{')));
  if (report.errors?.length) throw new Error(`Playwright collection reported ${report.errors.length} errors.`);
  const tests = [];
  const walk = (suite, file) => {
    const current = suite.file || file;
    for (const spec of suite.specs || []) {
      for (const test of spec.tests || []) {
        tests.push({ id: `${current}:${spec.line}:${spec.column}:${test.projectName || ''}`, file: basename(current),
          line: spec.line, title: spec.title,
          cases: (test.annotations || []).filter((a) => a.type === 'case').map((a) => a.description),
          partialCases: (test.annotations || []).filter((a) => a.type === 'case-partial').map((a) => a.description) });
      }
    }
    for (const child of suite.suites || []) walk(child, current);
  };
  for (const suite of report.suites || []) walk(suite, suite.file);
  return tests;
}

// Node harness tests are credited by their executed TAP titles, the same way
// browser tests are credited by collected titles. Failing tests are not evidence.
export function collectNodeTitles(files) {
  const stdout = run(['--test', '--test-reporter=tap', ...files], 'node --test');
  return parseNodeTitles(stdout);
}

export function parseNodeTitles(stdout) {
  const titles = [];
  for (const line of stdout.split(/\r?\n/)) {
    const match = /^(?:\s*)(ok|not ok) \d+ - (.+)$/.exec(line);
    if (match) titles.push({ passed: match[1] === 'ok' && !/ # (?:SKIP|TODO)\b/i.test(match[2]),
      title: match[2].replace(/ # .*$/, '') });
  }
  if (!titles.length) throw new Error('node --test produced no TAP test lines.');
  return titles;
}

export function nodeCaseKeys(titles) {
  const keys = new Set();
  for (const { passed, title } of titles) {
    if (!passed) continue;
    const head = title.split(/\s*[:—–]\s/, 1)[0].trim();
    for (const part of head.split('/').map((value) => value.trim())) {
      const id = /^TB-[A-Z]+-\d{3}$/.test(part) ? part : /^\d{3}$/.test(part) ? `${head.slice(0, head.indexOf('-', 3) + 1)}${part}` : null;
      if (id && /^TB-[A-Z]+-\d{3}$/.test(id)) keys.add(id);
    }
  }
  return keys;
}

export function assignCases(cases, { browser = [], external = [], node = new Set(), contract }) {
  const index = contractIndex(contract);
  const defaultMap = mapCollectedTests(cases, browser);
  const externalMap = mapCollectedTests(cases, external);
  const unknown = [...index.owners.keys(), ...index.exclusions.keys()].filter((key) => !cases.some((item) => item.key === key));
  if (unknown.length) throw new Error(`Contract names unknown cases: ${unknown.join(', ')}`);
  return cases.map((item) => {
    const evidence = [];
    for (const variant of defaultMap[item.key] || []) evidence.push({ layer: 'browser', ...variant });
    for (const variant of externalMap[item.key] || []) evidence.push({ layer: 'browser-external', ...variant });
    if (node.has(item.key)) evidence.push({ layer: 'node', label: 'node --test title' });
    if (index.owners.has(item.key)) evidence.push({ layer: 'owned', owner: index.owners.get(item.key) });
    const excluded = index.exclusions.get(item.key);
    if (excluded && evidence.length) return { ...item, status: 'STALE_EXCLUSION', evidence, reason: excluded };
    if (excluded) return { ...item, status: 'EXCLUDED', evidence, reason: excluded };
    if (!evidence.length) return { ...item, status: 'UNASSIGNED', evidence };
    const layers = new Set(evidence.map((entry) => entry.layer));
    const status = layers.has('browser') ? 'MAPPED' : layers.has('browser-external') ? 'EXTERNAL'
      : layers.has('node') ? 'NODE' : 'OWNED';
    const browserEvidence = evidence.filter((entry) => entry.layer.startsWith('browser'));
    const browserCoverage = !browserEvidence.length ? 'none'
      : browserEvidence.every((entry) => entry.partial) ? 'partial' : 'full';
    return { ...item, status, evidence, browserCoverage };
  });
}

export function summarize(assignment) {
  const counts = {};
  for (const item of assignment) counts[item.status] = (counts[item.status] || 0) + 1;
  return counts;
}
