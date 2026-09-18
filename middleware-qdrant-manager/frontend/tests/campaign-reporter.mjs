import { readFileSync, mkdirSync, writeFileSync, renameSync, rmSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { basename, resolve } from 'node:path';
import { bookCases, executionReport, executionSummary, mapCollectedTests } from './case-coverage.mjs';
import { redact } from './telemetry.mjs';
import { BASE_URL } from './harness.mjs';

export default class CampaignReporter {
  onBegin(config, suite) {
    this.startedAt = new Date().toISOString();
    this.destination = resolve(config.rootDir, '../test-results/campaign.json');
    this.cases = bookCases(readFileSync(new URL('./testbook/test-book-e2e.md', import.meta.url), 'utf8'));
    // Rows owned by another suite or excluded with a written reason are reported
    // as such, not as browser gaps; the coverage gate validates the contract.
    this.contract = JSON.parse(readFileSync(new URL('./testbook/coverage-contract.json', import.meta.url), 'utf8'));
    this.collected = suite.allTests().map((test) => ({ id: test.id, file: basename(test.location.file),
      line: test.location.line, title: test.title, titlePath: test.titlePath(),
      cases: test.annotations.filter((annotation) => annotation.type === 'case').map((annotation) => annotation.description),
      partialCases: test.annotations.filter((annotation) => annotation.type === 'case-partial').map((annotation) => annotation.description) }));
    this.results = [];
    this.write('collected');
  }

  onTestEnd(test, result) {
    const attachment = result.attachments.find((item) => item.name === 'campaign-telemetry');
    const body = attachment?.body || (attachment?.path ? readFileSync(attachment.path) : null);
    this.results.push({ id: test.id, status: result.status, expectedStatus: test.expectedStatus,
      retry: result.retry, duration: result.duration, errors: result.errors.map((error) => redact(error.message)),
      telemetry: body ? JSON.parse(body.toString()) : null });
    this.write('running');
  }

  onEnd(result) { this.write(result.status); }

  write(status) {
    mkdirSync(resolve(this.destination, '..'), { recursive: true });
    const mapping = mapCollectedTests(this.cases, this.collected);
    const payload = JSON.stringify({ status, startedAt: this.startedAt,
      updatedAt: new Date().toISOString(), baseURL: BASE_URL,
      runId: process.env.QHM_E2E_RUN_ID || null, collected: this.collected,
      results: this.results, outcomes: executionSummary(this.results),
      cases: executionReport(this.cases, this.collected, this.results, mapping, this.contract) }, null, 2);
    const temporary = `${this.destination}.${randomUUID()}.tmp`;
    try {
      writeFileSync(temporary, payload, { flag: 'wx' });
      renameSync(temporary, this.destination);
    } finally {
      rmSync(temporary, { force: true });
    }
  }
}
