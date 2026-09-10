// Shared live-test boundary. A prefix is a namespace, never proof of ownership.
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export function normalizeBaseURL(value = "http://127.0.0.1:6336") {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password
      || url.search || url.hash || url.pathname.replace(/\/$/, "") !== "") {
    throw new Error("QHM_BASE_URL must be an HTTP origin without credentials or a path.");
  }
  return url.origin;
}

export const BASE_URL = normalizeBaseURL(process.env.QHM_BASE_URL);
export const TOKEN = process.env.QHM_E2E_TOKEN || process.env.QHM_TOKEN || "";
const runId = process.env.QHM_E2E_RUN_ID || "";
if (runId && !/^[a-z0-9][a-z0-9_]{0,31}$/.test(runId)) {
  throw new Error("QHM_E2E_RUN_ID must contain 1–32 lowercase letters, digits or underscores.");
}
const frontend = fileURLToPath(new URL("../", import.meta.url));
export const manifestPath = resolve(frontend, "test-results", `ownership-${runId || 'missing'}.json`);

export function fixtureName(logicalName) {
  if (!/^e2e_[a-z0-9_]+$/.test(logicalName)) throw new Error("Invalid E2E fixture name.");
  // Collection-only discovery and intercepted tests need no live manifest.
  return runId ? `e2e_${runId}_${logicalName.slice(4)}` : logicalName;
}

export function validateOwnership(manifest, name, baseURL = BASE_URL) {
  if (manifest?.version !== 1 || manifest.baseURL !== baseURL
      || !/^[a-z0-9][a-z0-9_]{0,31}$/.test(manifest.runId || "")
      || typeof name !== "string" || !/^e2e_[a-z0-9_]+$/.test(name)
      || !name.startsWith(`e2e_${manifest.runId}_`)
      || !Array.isArray(manifest.initialCollections)
      || !Array.isArray(manifest.ownedCollections)
      || manifest.initialCollections.includes(name)
      || !manifest.ownedCollections.includes(name)) {
    throw new Error("E2E mutation refused: target is not owned by this campaign.");
  }
  return name;
}

export function ownedFixture(name) {
  if (!runId) throw new Error("Live writes require QHM_E2E_RUN_ID and an initialized manifest.");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (manifest.runId !== runId) throw new Error("E2E manifest belongs to another campaign.");
  return validateOwnership(manifest, name);
}

export function fixtureOverride(variable, fallback = "") {
  if (Object.hasOwn(process.env, variable)) return ownedFixture(process.env[variable]);
  return fallback ? fixtureName(fallback) : "";
}

export function guardMutation(method, path, body, assertOwned = ownedFixture) {
  const url = new URL(path, BASE_URL);
  if (["GET", "HEAD", "OPTIONS"].includes(method.toUpperCase())) return;
  if (url.origin !== BASE_URL) throw new Error("E2E mutation refused: foreign origin.");
  if (method.toUpperCase() === "POST" && (
    /^\/query(?:\/|$)/.test(url.pathname)
    || /^\/points\/(scroll|count|retrieve|facet)$/.test(url.pathname)
  )) return;
  const match = /^\/collections\/([^/]+)/.exec(url.pathname);
  const targets = [match && decodeURIComponent(match[1]), body?.collection_name,
    url.searchParams.get("collection_name")].filter((value) => value !== null && value !== undefined);
  if (!targets.length) throw new Error("E2E mutation refused: missing collection target.");
  for (const name of targets) assertOwned(name);
}

export function requireStatus(response, method, path, expected = [200]) {
  const status = typeof response.status === "function" ? response.status() : response.status;
  if (!expected.includes(status)) {
    throw new Error(`E2E prerequisite failed: ${method} ${new URL(path, BASE_URL).pathname} returned ${status}; expected ${expected.join('/')}.`);
  }
}

export async function setupCall(method, path, body, { expected = [200], transport = fetch,
  assertOwned = ownedFixture } = {}) {
  guardMutation(method, path, body, assertOwned);
  const response = await transport(new URL(path, `${BASE_URL}/`), {
    method, headers: { "Content-Type": "application/json", ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  requireStatus(response, method, path, expected);
  const data = await response.json();
  return { status: response.status, ok: response.ok, data };
}

export function requireResults(data) {
  if (!data || !Array.isArray(data.results)) throw new Error("E2E oracle returned malformed results.");
  return data.results;
}

export async function settleOwnedReindex(name, { call = setupCall,
  assertOwned = ownedFixture, pause = () => new Promise((resolve) => setTimeout(resolve, 500)),
  attempts = 240 } = {}) {
  assertOwned(name);
  const path = `/collections/${encodeURIComponent(name)}`;
  let cancelRequested = false;
  for (let index = 0; index < attempts; index++) {
    const response = await call('GET', `${path}/reindex`, undefined, { expected: [200, 404] });
    if (response.status === 404) return;
    const state = response.data;
    if (!state || !Object.hasOwn(state, 'job') || typeof state.task_running !== 'boolean') {
      throw new Error('Malformed reindex cleanup status; collection retained.');
    }
    const terminal = !state.job || ['completed', 'cancelled'].includes(state.job.status);
    if (terminal && !state.task_running && !state.metadata_invalid) return;
    if (!cancelRequested && (state.metadata_invalid
        || ['running', 'paused', 'interrupted', 'failed'].includes(state.job?.status))) {
      // The job can complete between observation and cancellation. Re-read the
      // exact target after a conflict; only a settled physical task permits cleanup.
      await call('POST', `${path}/reindex/cancel`, undefined, { expected: [200, 409] });
      cancelRequested = true;
    }
    await pause();
  }
  throw new Error('Owned reindex job did not settle; collection retained for inspection.');
}

export async function cleanupOwnedCollection(name, options = {}) {
  const { call = setupCall, assertOwned = ownedFixture } = options;
  assertOwned(name);
  const path = `/collections/${encodeURIComponent(name)}`;
  if ((await call('GET', path, undefined, { expected: [200, 404] })).status === 404) return;
  await settleOwnedReindex(name, options);
  await call('DELETE', `${path}?confirm_delete=${encodeURIComponent(name)}`, undefined, { expected: [200, 404] });
}

export function initializeManifest(initialCollections, names) {
  if (!runId) throw new Error("Set QHM_E2E_RUN_ID before initializing a campaign.");
  if (!Array.isArray(initialCollections) || initialCollections.some((name) => typeof name !== 'string')) {
    throw new Error("Malformed collection inventory.");
  }
  const manifest = { version: 1, baseURL: BASE_URL, runId, initialCollections,
    ownedCollections: [...new Set(names)] };
  for (const name of names) validateOwnership(manifest, name);
  mkdirSync(dirname(manifestPath), { recursive: true });
  // Refuse overwrite: resuming uses the original manifest, never a new claim.
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), { flag: "wx" });
  return manifest;
}
