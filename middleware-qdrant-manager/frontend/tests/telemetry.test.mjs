import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { BASE_URL } from './harness.mjs';
import { startTelemetry, redact } from './telemetry.mjs';

test('TB-HAR-005 / K6: an expected 404 cannot hide an unexpected 500 or page exception', () => {
  const page = new EventEmitter();
  const telemetry = startTelemetry(page);
  telemetry.expect('GET', '/points/missing', 404);
  const response = (status, method = 'GET', path = '/points/missing') => page.emit('response', {
    url: () => `${BASE_URL}${path}`, status: () => status, request: () => ({ method: () => method }),
  });
  response(404);
  page.emit('console', { type: () => 'error', location: () => ({ url: `${BASE_URL}/points/missing` }),
    text: () => 'Failed to load resource: the server responded with a status of 404 (Not Found)' });
  assert.deepEqual(telemetry.unexpected(), []);
  response(500);
  response(404, 'DELETE');
  response(404, 'GET', '/points/other');
  page.emit('pageerror', new Error('application exception'));
  assert.equal(telemetry.unexpected().length, 4);
  telemetry.dispose();
  assert.equal(page.eventNames().length, 0);
});

test('TB-HAR-006: telemetry disposal removes only its own listeners', () => {
  const page = new EventEmitter();
  const other = () => {};
  page.on('response', other);
  const telemetry = startTelemetry(page);
  assert.equal(startTelemetry(page), telemetry);
  telemetry.dispose();
  assert.deepEqual(page.listeners('response'), [other]);
  assert.equal(redact('Authorization: Bearer test-secret'), 'Authorization: Bearer [REDACTED]');
});

test('K6: intercepted console origin shares the per-test collector', () => {
  const page = new EventEmitter();
  const telemetry = startTelemetry(page);
  assert.equal(startTelemetry(page, { origin: 'http://application.test' }), telemetry);
  page.emit('response', {
    url: () => 'http://application.test/points/7', status: () => 500,
    request: () => ({ method: () => 'GET' }),
  });
  assert.deepEqual(telemetry.unexpected(), [
    { type: 'http', method: 'GET', path: '/points/7', status: 500 },
  ]);
  assert.equal(page.listenerCount('response'), 1);
  telemetry.dispose();
  assert.equal(page.eventNames().length, 0);
});
