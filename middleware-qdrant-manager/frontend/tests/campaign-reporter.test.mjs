import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { syncBuiltinESMExports } from 'node:module';
import CampaignReporter from './campaign-reporter.mjs';

function reporterAt(destination) {
  const reporter = new CampaignReporter();
  Object.assign(reporter, {
    destination, startedAt: '2026-09-18T00:00:00Z',
    cases: [], collected: [], results: [], contract: {},
  });
  return reporter;
}

test('campaign publication preserves report contents and removes temporary files', () => {
  const directory = fs.mkdtempSync(join(tmpdir(), 'campaign-report-'));
  try {
    const destination = join(directory, 'campaign.json');
    fs.writeFileSync(destination, 'previous report');
    reporterAt(destination).write('passed');
    const result = JSON.parse(fs.readFileSync(destination, 'utf8'));
    assert.equal(result.status, 'passed');
    assert.deepEqual(result.collected, []);
    assert.deepEqual(result.results, []);
    assert.deepEqual(fs.readdirSync(directory), ['campaign.json']);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

for (const operation of ['writeFileSync', 'renameSync']) {
  test(`campaign ${operation} failure preserves the previous report`, (t) => {
    const directory = fs.mkdtempSync(join(tmpdir(), 'campaign-report-'));
    const destination = join(directory, 'campaign.json');
    fs.writeFileSync(destination, 'previous report');
    const originalWrite = fs.writeFileSync;
    const replacement = t.mock.method(fs, operation, (path) => {
      if (operation === 'writeFileSync') originalWrite(path, 'partial');
      throw new Error('injected disk failure');
    });
    syncBuiltinESMExports();
    try {
      assert.throws(() => reporterAt(destination).write('passed'), /injected disk failure/);
      assert.equal(fs.readFileSync(destination, 'utf8'), 'previous report');
      assert.deepEqual(fs.readdirSync(directory), ['campaign.json']);
    } finally {
      replacement.mock.restore();
      syncBuiltinESMExports();
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });
}
