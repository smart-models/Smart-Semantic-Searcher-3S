// Per-page, per-test observations. Expected failures match method/path/status.
import { BASE_URL, TOKEN, guardMutation } from './harness.mjs';

const pages = new WeakMap();
export function redact(value) {
  let text = String(value);
  if (TOKEN) text = text.split(TOKEN).join('[REDACTED]');
  return text.replace(/Bearer\s+[^\s"']+/gi, 'Bearer [REDACTED]');
}

export function startTelemetry(page, { origin = BASE_URL } = {}) {
  if (pages.has(page)) {
    const telemetry = pages.get(page);
    telemetry.origins.add(origin);
    return telemetry;
  }
  const origins = new Set([origin]);
  const events = [], expected = [], expectedNetwork = [], messages = [];
  const pathOf = (url) => { try { return new URL(url, BASE_URL).pathname; } catch { return ''; } };
  const onResponse = (response) => {
    if (!origins.has(new URL(response.url()).origin)) return;
    events.push({ type: 'http', method: response.request().method(),
      path: pathOf(response.url()), status: response.status() });
  };
  const onConsole = (message) => {
    if (message.type() !== 'error') return;
    events.push({ type: 'console', message: redact(message.text()), path: pathOf(message.location().url) });
    messages.push(redact(message.text()));
  };
  const onPageError = (error) => {
    events.push({ type: 'pageerror', message: redact(error.message) });
    messages.push(`pageerror: ${redact(error.message)}`);
  };
  const onRequestFailed = (request) => events.push({ type: 'network', method: request.method(),
    path: pathOf(request.url()), message: redact(request.failure()?.errorText || '') });
  page.on('response', onResponse);
  page.on('console', onConsole);
  page.on('pageerror', onPageError);
  page.on('requestfailed', onRequestFailed);
  const isExpected = (event) => expected.some((item) => item.method === event.method
    && item.path === event.path && item.status === event.status);
  const telemetry = {
    events, expected, expectedNetwork, messages, origins,
    expect(method, path, status) {
      if (!Number.isInteger(status) || status < 400) throw new Error('Expected HTTP failure requires an error status.');
      expected.push({ method: method.toUpperCase(), path: pathOf(path), status });
    },
    unexpected() {
      return events.filter((event) => {
        if (event.type === 'http') return event.status >= 400 && !isExpected(event);
        if (event.type === 'pageerror') return true;
        if (event.type === 'network') {
          // Normal navigation/fetch cancellation is recorded, not a server error.
          if (event.message === 'net::ERR_ABORTED' && event.method === 'GET') return false;
          return !expectedNetwork.some((item) => item.method === event.method && item.path === event.path);
        }
        if (/Failed to load resource: net::ERR_/.test(event.message)) {
          return !events.some((response) => response.type === 'network' && response.path === event.path
            && expectedNetwork.some((item) => item.method === response.method && item.path === response.path));
        }
        const match = /Failed to load resource: the server responded with a status of (\d{3})/.exec(event.message);
        if (!match) return true;
        return !events.some((response) => response.type === 'http'
          && response.path === event.path && response.status === Number(match[1]) && isExpected(response));
      });
    },
    dispose() {
      page.off('response', onResponse);
      page.off('console', onConsole);
      page.off('pageerror', onPageError);
      page.off('requestfailed', onRequestFailed);
      pages.delete(page);
    },
  };
  pages.set(page, telemetry);
  return telemetry;
}

export function expectHttpFailure(page, method, path, status) {
  startTelemetry(page).expect(method, path, status);
}

export async function fulfillResponse(route, options) {
  if (options.status >= 400) {
    const request = route.request();
    expectHttpFailure(request.frame().page(), request.method(), request.url(), options.status);
  }
  return route.fulfill(options);
}

export async function abortResponse(route) {
  const request = route.request();
  startTelemetry(request.frame().page()).expectedNetwork.push({
    method: request.method(), path: new URL(request.url()).pathname,
  });
  return route.abort();
}

export async function fetchResponse(route, options = {}) {
  const request = route.request();
  const method = options.method || request.method();
  let body = options.postData;
  if (body === undefined && request.postData()) body = request.postDataJSON();
  if (typeof body === 'string') body = JSON.parse(body);
  guardMutation(method, options.url || request.url(), body);
  return route.fetch(options);
}

export async function transformJsonResponse(route, transform) {
  const page = route.request().frame().page();
  try {
    const response = await fetchResponse(route);
    const json = transform(await response.json());
    await fulfillResponse(route, { response, json });
  } catch (error) {
    if (!page.isClosed()) throw error;
  }
}
