// The three external-state scenarios the default runner excludes (see
// playwright.config.js): a real SMART_EMBEDDER_PROFILE switch (z4, z6) or the
// embedder container stopped by the operator (z5). Collecting them here keeps
// their case mapping visible to the coverage gate without executing them.
//
//   npx playwright test --config playwright.external.config.js --list
//   npx playwright test --config playwright.external.config.js tests/z4-switch.spec.js
//
// Run one file per prepared state — tests/README.md lists the prerequisites.
import { fileURLToPath } from "node:url";
import base from "./playwright.config.js";

export default {
  ...base,
  testMatch: /z[456]-(switch|real-outage|e7)\.spec\.js$/,
  testIgnore: [],
  outputDir: fileURLToPath(new URL("./test-results/external-artifacts", import.meta.url)),
};
