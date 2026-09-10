import { fulfillResponse } from "./telemetry.mjs";
import { fixtureName } from "./harness.mjs";
import fs from "node:fs/promises";
import { expect, test } from "./fixtures.js";

const collection = fixtureName("e2e_export_filename");
const body = JSON.stringify({ chunks: [{ text: 'à漢', point_id: '1' }],
  metadata: { n_chunks: 1, collection_name: collection } }, null, 2) + '\n';

for (const { label, disposition, filename } of [
  {
    label: "prefers the UTF-8 filename",
    disposition: "attachment; filename=\"ascii_export.json\"; filename*=UTF-8''%C3%A0%E6%BC%A2_export.json",
    filename: "à漢_export.json",
  },
  {
    label: "uses the fallback for malformed UTF-8 encoding",
    disposition: "attachment; filename=\"ascii_export.json\"; filename*=UTF-8''bad%ZZ.json",
    filename: "ascii_export.json",
  },
]) {
  test(`X6 — export ${label} and preserves the response bytes`, async ({ page }) => {
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.addInitScript(name => {
      localStorage.setItem("qhm_active_collection", name);
      sessionStorage.removeItem("qhm_api_key");
    }, collection);
    // Only static assets reach the server. Every API response is synthetic;
    // this browser regression creates no collections and needs no credentials.
    await page.route("**/*", async route => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname === "/assets/config.js") {
        return fulfillResponse(route, {
          contentType: "text/javascript",
          body: "window.QHM_CONFIG = {authRequired: false};",
        });
      }
      if (pathname === "/" || pathname.startsWith("/assets/")) {
        return route.fallback();
      }
      if (pathname === "/points/export") {
        expect(route.request().postDataJSON()).toEqual({
          collection_name: collection, destination: "body",
        });
        return fulfillResponse(route, {
          contentType: "application/json",
          headers: { "content-disposition": disposition },
          body,
        });
      }
      let json = {};
      if (pathname === "/health") {
        json = {
          status: "healthy", qdrant: "reachable", embedder: "reachable",
          embedder_profile: "basic", collections: 1,
        };
      } else if (pathname === "/collections") {
        json = { collections: [collection] };
      } else if (pathname === `/collections/${collection}`) {
        json = {
          collection_name: collection, status: "green", points_count: 1,
          dense: { size: 1024 }, sparse: {},
          embedding_compatibility: {
            status: "compatible", stored: { profile: "basic" },
            active: { profile: "basic" }, message: "Compatible",
          },
        };
      }
      return fulfillResponse(route, { json });
    });

    await page.goto("/#collections");
    await expect(page.locator("#conn-health-status")).toHaveText("HEALTHY");
    await expect(page.locator("#conn-collection")).toHaveValue(collection);
    const [download] = await Promise.all([
      page.waitForEvent("download"), page.locator("#ce-export").click(),
    ]);
    expect(download.suggestedFilename()).toBe(filename);
    expect(await fs.readFile(await download.path(), "utf8")).toBe(body);
    await expect(page.locator("#ce-result .alert-success")).toContainText("Exported 1 points");
    expect(errors).toEqual([]);
  });
}
