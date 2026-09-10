// Collection and execution are distinct evidence. Never parse spec source text.
export function bookCases(markdown) {
  let section = '', fenced = false;
  const cases = new Map();
  for (const line of markdown.split(/\r?\n/)) {
    if (/^```/.test(line)) { fenced = !fenced; continue; }
    if (fenced) continue;
    const heading = /^#{2,4} (\d+(?:\.\d+[a-z]?)*)\.?\s/.exec(line);
    if (heading) section = heading[1];
    // Rows in the historical audit are observations, not new case definitions.
    if (!section || Number(section.split('.')[0]) > 13) continue;
    const row = /^\| ([A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*)\s+\|/.exec(line);
    if (!row || (!/\d/.test(row[1]) && !/^[A-Z]+-[A-Z]+$/.test(row[1]))) continue;
    for (const id of expandIds(row[1])) {
      const key = id.startsWith('TB-') ? id : `book:${section}/${id}`;
      if (!cases.has(key)) cases.set(key, { key, section, id });
    }
  }
  return [...cases.values()];
}

function expandIds(value) {
  const [first, ...rest] = value.split('/');
  const prefix = first.replace(/\d+$/, '');
  return [first, ...rest.map((part) => /^\d+$/.test(part) ? prefix + part
    : first.includes('-') && !part.includes('-')
      ? first.slice(0, first.lastIndexOf('-') + 1) + part : part)];
}

// The tracked disposition registry (testbook/coverage-contract.json). It never
// maps a case to a test: owners name the non-browser suite whose own gate must
// hold a marked test, exclusions carry a written reason. Shape errors throw.
export function contractIndex(contract) {
  const owners = new Map(), exclusions = new Map();
  if (!contract || typeof contract !== 'object') throw new Error('Malformed coverage contract.');
  for (const [runtime, keys] of Object.entries(contract.owners || {})) {
    if (!Array.isArray(keys)) throw new Error(`Owner ${runtime} must list case keys.`);
    for (const key of keys) {
      if (owners.has(key)) throw new Error(`Case ${key} has two owners.`);
      owners.set(key, runtime);
    }
  }
  for (const entry of contract.exclusions || []) {
    if (typeof entry?.id !== 'string' || (entry.reason || '').trim().length < 40) {
      throw new Error(`Exclusion ${entry?.id} needs a written reason of at least 40 characters.`);
    }
    if (owners.has(entry.id) || exclusions.has(entry.id)) throw new Error(`Case ${entry.id} is both owned and excluded.`);
    exclusions.set(entry.id, entry.reason.trim());
  }
  return { owners, exclusions };
}

export function executionReport(cases, collected, results, mapping, contract = null) {
  const inventory = new Set(collected.map((test) => test.id));
  const outcomes = new Map(results.map((result) => [result.id, result.status]));
  const index = contract ? contractIndex(contract) : { owners: new Map(), exclusions: new Map() };
  for (const key of Object.keys(mapping)) {
    if (!cases.some((item) => item.key === key)) throw new Error(`Unknown qualified case: ${key}`);
  }
  return cases.map((item) => {
    const variants = (mapping[item.key] || []).map((entry) => {
      const variant = typeof entry === 'string' ? { id: entry, label: entry } : entry;
      return { ...variant, status: inventory.has(variant.id)
        ? (outcomes.get(variant.id) || 'not-run') : 'not-collected' };
    });
    let status = 'NOT_MAPPED';
    if (variants.length) {
      if (variants.some((variant) => ['failed', 'timedOut', 'interrupted'].includes(variant.status))) status = 'FAIL';
      else if (variants.every((variant) => variant.status === 'passed')) {
        status = variants.some((variant) => variant.partial) ? 'PARTIAL' : 'PASS';
      }
      else if (variants.every((variant) => variant.status === 'skipped')) status = 'SKIPPED';
      else status = 'NOT_RUN';
    }
    else if (index.owners.has(item.key)) return { ...item, status: 'OWNED', owner: index.owners.get(item.key), variants };
    else if (index.exclusions.has(item.key)) return { ...item, status: 'EXCLUDED', reason: index.exclusions.get(item.key), variants };
    return { ...item, status, variants };
  });
}

// Browser acceptance accepts PASS only; a ledger merged with the other layers
// may accept OWNED/EXCLUDED explicitly. Nothing else is ever accepted.
export function requireCompleteCoverage(report, { accept = ['PASS'] } = {}) {
  const allowed = new Set(accept.filter((status) => ['PASS', 'OWNED', 'EXCLUDED'].includes(status)));
  const missing = report.filter((item) => !allowed.has(item.status));
  if (missing.length) throw new Error(`Incomplete execution coverage: ${missing.map((item) => item.key).join(', ')}`);
}

// Count final attempts, keeping expected failures distinct from actual passes.
export function executionSummary(results) {
  const counts = { passed: 0, failed: 0, timedOut: 0, interrupted: 0, skipped: 0, expectedFailures: 0 };
  for (const result of new Map(results.map((item) => [item.id, item])).values()) {
    if (Object.hasOwn(counts, result.status)) counts[result.status]++;
    if (result.status === 'failed' && result.expectedStatus === 'failed') counts.expectedFailures++;
  }
  return counts;
}

// Which book sections a spec file may claim by title prefix. A title "C1 — …"
// in a-collections.spec.js is book:6.2/C1; the same title in d-filters.spec.js
// is book:8.2/C1. Explicit `case` annotations bypass this table.
export const SPEC_SECTIONS = {
  'a-collections.spec.js': ['6.', '11'], 'b-ingest.spec.js': ['7.1', '11'],
  'b2-ingest-indexes.spec.js': ['7.1b', '11'], 'c-points.spec.js': ['7.2', '7.3', '7.4', '7.5', '7.6', '11'],
  'd-query.spec.js': ['8.1'], 'd-filters.spec.js': ['8.2'], 'e-apikey.spec.js': ['3'],
  'e-monitoring.spec.js': ['9'], 'e-nav.spec.js': ['10'], 'nf-newfeatures.spec.js': ['13'],
  'z1-reembed.spec.js': ['5'], 'z2-service.spec.js': ['2'], 'z3-compat-mocked.spec.js': ['4'],
  'z4-switch.spec.js': ['4', '9', '12'], 'z5-real-outage.spec.js': ['2'], 'z6-e7.spec.js': ['4'],
  'export-filename.e2e.spec.js': ['6.5'],
  'tb-filters-live.spec.js': ['8.2'], 'tb-filter-validation.spec.js': ['8.2'],
  'tb-monitoring.spec.js': ['9'], 'tb-ui.spec.js': ['11'],
};

export function mapCollectedTests(cases, collected) {
  const mapping = {};
  for (const test of collected) {
    // These are actual titles emitted by Playwright, including generated tests.
    const ids = expandIds(test.title.split(/\s+[—–]|\s+:/, 1)[0]);
    const explicit = [...(test.cases || []), ...(test.partialCases || [])];
    for (const key of explicit) {
      if (!cases.some((item) => item.key === key)) throw new Error(`Unknown qualified case: ${key}`);
    }
    // TB-* ids are globally unique (book §0.7), so any file may carry them by
    // title; legacy ids need the file's section to disambiguate `C1` and friends.
    const matched = cases.filter((item) => explicit.includes(item.key)
      || (ids.includes(item.id) && (item.key.startsWith('TB-')
        || (SPEC_SECTIONS[test.file] || []).some((section) => section.endsWith('.')
          ? item.section.startsWith(section) : item.section === section || item.section.startsWith(`${section}.`)))));
    for (const item of matched) (mapping[item.key] ||= []).push({ id: test.id,
      label: `${test.file}: ${test.title}`, file: test.file, title: test.title,
      partial: (test.partialCases || []).includes(item.key) });
  }
  return mapping;
}
