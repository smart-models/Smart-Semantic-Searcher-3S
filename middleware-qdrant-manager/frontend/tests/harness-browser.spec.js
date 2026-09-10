// Fully intercepted browser acceptance for the real shared boot/URL helper.
import { test, expect } from './fixtures.js';
import { boot, BASE_URL } from './helpers.js';
import { openConsole, A } from './application-fixtures.js';

test('TB-HAR-002 — shared boot and relative console assets use the configured origin', {
  annotation: { type: 'case-partial', description: 'TB-HAR-002' },
}, async ({ page }) => {
  const origins = [], paths = [];
  page.on('request', (request) => {
    const url = new URL(request.url());
    origins.push(url.origin);
    paths.push(url.pathname);
  });
  const mock = await openConsole(page, { origin: BASE_URL });
  await boot(page, { active: A });
  await expect(page.locator('#conn-collection')).toHaveValue(A);
  expect([...new Set(origins)]).toEqual([BASE_URL]);
  expect(paths).toContain('/assets/app.js');
  expect(paths).toContain('/collections');
  expect(paths.some((path) => path.startsWith('//'))).toBe(false);
  expect(mock.unknown).toEqual([]);
  expect(mock.errors).toEqual([]);
});
