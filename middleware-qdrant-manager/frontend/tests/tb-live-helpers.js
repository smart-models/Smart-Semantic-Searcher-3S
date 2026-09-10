// Shared API-live helpers for the TB-* specs: owned writes through the
// campaign manifest, exact-status expectations and full-collection snapshots.
import { expect } from "@playwright/test";
import { BASE_URL, fixtureName, setupCall } from "./harness.mjs";
import { expectHttpFailure } from "./telemetry.mjs";

export const SCHEMA = { dense_on_disk: true, sparse_on_disk: true, on_disk_payload: true, quantization: true, sparse_idf: true };

// The middleware allows RATE_LIMIT_PER_MINUTE (300 by default) non-exempt
// requests per minute; a long page walk plus setup calls exceeds it in bursts.
// A single sliding window (workers: 1) counts every browser and API request
// and `reserve()` waits for room instead of letting a 429 fail an oracle.
const WINDOW_MS = 60_000;
const BUDGET = Number(process.env.QHM_E2E_RATE_BUDGET || 270);
const EXEMPT = new Set(["/", "/docs", "/health", "/livez", "/metrics", "/openapi.json", "/readyz", "/redoc", "/stats"]);
const stamps = [];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function counted(pathname) {
  return !EXEMPT.has(pathname) && !/^\/(assets|docs|redoc)\//.test(pathname);
}

export function trackRequests(page) {
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.origin === BASE_URL && counted(url.pathname)) stamps.push(Date.now());
  });
}

export async function reserve(count = 1) {
  for (;;) {
    const now = Date.now();
    while (stamps.length && stamps[0] < now - WINDOW_MS) stamps.shift();
    if (stamps.length + count <= BUDGET) return;
    await sleep(stamps[0] + WINDOW_MS - now + 100);
  }
}

// Guarded call with an exact expected status; anything else is a failed
// prerequisite (setup) or a failed oracle (negative case), never a skip.
export async function api(method, path, body, { expected = 200, page = null } = {}) {
  if (page && expected >= 400) expectHttpFailure(page, method, path, expected);
  await reserve(1);
  stamps.push(Date.now());
  const response = await setupCall(method, path, body, { expected: [expected] });
  return response;
}

// Read every page of a collection; reject repeated offsets; sort by id text.
export async function snapshot(collection) {
  const points = [];
  const offsets = new Set();
  let offset;
  do {
    await reserve(1);
    stamps.push(Date.now());
    const { data } = await setupCall("POST", "/points/scroll", {
      collection_name: collection, limit: 1000, ...(offset === undefined ? {} : { offset }),
    });
    points.push(...data.results);
    offset = data.next_page_offset;
    if (offset !== null && offset !== undefined) {
      expect(offsets.has(String(offset)), "repeated scroll offset").toBe(false);
      offsets.add(String(offset));
    }
  } while (offset !== null && offset !== undefined);
  return points.sort((a, b) => String(a.id).localeCompare(String(b.id)));
}

// Rebuild an owned collection from scratch (recreate needs its exact confirmation).
export async function fresh(logical) {
  const name = fixtureName(logical);
  await api("POST", "/collections", { collection_name: name, recreate: true, confirm_recreate: name, ...SCHEMA });
  return name;
}

export async function seedPoints(name, points) {
  for (let start = 0; start < points.length; start += 100) {
    const batch = points.slice(start, start + 100);
    const { data } = await api("POST", "/points/batch", { collection_name: name, points: batch });
    expect(data.upserted).toBe(batch.length);
  }
}

// Boot the console with request tracking and headroom for its startup calls.
export async function bootPaced(page, boot, options) {
  trackRequests(page);
  await reserve(10);
  return boot(page, options);
}
