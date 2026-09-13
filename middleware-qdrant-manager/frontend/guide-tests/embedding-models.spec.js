import { fileURLToPath } from 'node:url';
import { chromium, test, expect } from '@playwright/test';

const guide = new URL('../../../guide/embedding-models.html', import.meta.url).href;

test.beforeEach(async ({ context, page }) => {
  // No live endpoints or external assets may be contacted by this guide.
  await context.route(/^https?:/, route => {
    throw new Error(`Unexpected network request: ${route.request().url()}`);
  });
  page.on('pageerror', error => { throw error; });
  page.on('console', message => {
    if (message.type() === 'error') throw new Error(message.text());
  });
  await page.goto(guide);
  await page.evaluate(() => document.fonts.ready);
});

for (const [profile, device, dense, sparse, reranker] of [
  ['basic', 'cpu', 2048, 2048, 8192],
  ['basic', 'gpu', 8192, 8192, 8192],
  ['advanced', 'cpu', 2048, 8192, 2048],
  ['advanced', 'gpu', 32768, 8192, 32768],
]) {
  test(`${profile}/${device}: chooser and token windows use the displayed defaults`, async ({ page }) => {
    await page.getByLabel('Model profile').selectOption(profile);
    await page.getByLabel('Hardware', { exact: true }).selectOption(device);
    await expect(page.locator(`#choice-summary`)).toHaveText(
      `${profile} / ${device.toUpperCase()} · Dense ${dense} · Sparse ${sparse} · Reranker ${reranker} tokens.`,
    );
    await expect(page.locator(`[data-roster="${profile}"]`)).toBeVisible();
    await expect(page.locator('[data-roster]:visible')).toHaveCount(1);
    const sparseInput = profile === 'basic' ? 6000 : 7200;
    await expect(page.locator('#dense-window-text')).toContainText(`${Math.max(0, 6000 - dense)} beyond`);
    await expect(page.locator('#sparse-window-text')).toContainText(`${Math.max(0, sparseInput - sparse)} beyond`);
    await expect(page.locator('#reranker-window-text')).toContainText(`${Math.max(0, 3000 - reranker)} beyond`);
    await page.getByLabel('Illustrative input size').selectOption('short');
    for (const role of ['dense', 'sparse', 'reranker']) {
      await expect(page.locator(`#${role}-window-text`)).toContainText('0 beyond');
    }
  });
}

test('character slicing can remove a remedy while preserving the authored full text', async ({ page }) => {
  await page.locator('#character-example > summary').click();
  const original = await page.locator('#character-source').textContent();
  const kept = await page.locator('#character-kept').textContent();
  const removed = await page.locator('#character-removed').textContent();
  expect(Array.from(kept)).toHaveLength(1000);
  expect(kept.length).toBeGreaterThan(Array.from(kept).length);
  expect(kept + removed).toBe(original);
  expect(removed).toContain('Remedy:');
  await expect(page.locator('#character-status')).toContainText('Remedy removed');
  await page.getByLabel('Remedy position').selectOption('start');
  await expect(page.locator('#character-kept')).toContainText('Remedy:');
  await expect(page.locator('#character-status')).toContainText('Remedy retained');
  await page.getByLabel('Remedy position').selectOption('end');
  await expect(page.locator('#character-source')).toHaveText(original);
});

test('evidence control is keyboard operable and announces its teaching focus', async ({ page }) => {
  const control = page.getByLabel('Emphasize an evidence type');
  await control.focus();
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await expect(control).toHaveValue('meaning');
  await expect(page.locator('#evidence-summary')).toContainText('Focus B:');
  await expect(page.locator('[data-evidence][data-active]')).toHaveAttribute('data-evidence', 'meaning');
  await expect(page.locator('[data-evidence]:visible')).toHaveCount(3);
  await expect(page.locator('#evidence-summary')).toHaveAttribute('role', 'status');
});

for (const width of [320, 390, 768, 1440]) {
  test(`readable at ${width}px, including expanded details and text spacing`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.evaluate(() => document.querySelectorAll('details').forEach(item => { item.open = true; }));
    await page.addStyleTag({ content: '* { line-height: 1.5 !important; letter-spacing: .12em !important; word-spacing: .16em !important; } p { margin-bottom: 2em !important; }' });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
    for (const id of ['profile-choice', 'hardware-choice', 'evidence-choice', 'window-choice', 'remedy-position']) {
      const box = await page.locator(`#${id}`).boundingBox();
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(width + 1);
      expect(box.height).toBeGreaterThanOrEqual(44);
    }
    await expect(page.getByRole('columnheader', { name: 'Basic · BGE' })).toHaveCount(1);
  });
}

test('stable disclosure fragments open on direct navigation and history', async ({ page }) => {
  await page.goto(`${guide}#help-remote-code`);
  await expect(page.locator('#help-remote-code')).toHaveAttribute('open', '');
  await page.goto(`${guide}#spec-milco-650m`);
  await expect(page.locator('#spec-milco-650m')).toHaveAttribute('open', '');
  await page.goto(`${guide}#contract`);
  await expect(page.locator('#contract')).toContainText('Re-embed is the migration path');
  await expect(page.locator('#help-dims')).toContainText('three ways forward');
  await page.goBack();
  await expect(page.locator('#spec-milco-650m')).toHaveAttribute('open', '');
});

test('repeated real PDF printing reveals sources and restores controls and disclosures', async ({ page }) => {
  await page.getByLabel('Model profile').selectOption('advanced');
  await page.getByLabel('Hardware', { exact: true }).selectOption('gpu');
  await page.locator('#spec-bge-m3 > summary').click();
  const state = () => page.evaluate(() => ({
    details: [...document.querySelectorAll('details')].map(item => [item.id, item.open]),
    rosters: [...document.querySelectorAll('[data-roster]')].map(item => item.hidden),
    controls: [...document.querySelectorAll('select')].map(item => item.value),
  }));
  const before = await state();
  await page.evaluate(() => {
    window.printChecks = [];
    addEventListener('beforeprint', () => {
      window.printChecks.push([...document.querySelectorAll('details')].every(item => item.open)
        && [...document.querySelectorAll('[data-roster]')].every(item => !item.hidden));
    });
  });
  for (let i = 0; i < 2; i += 1) {
    const pdf = await page.pdf({ preferCSSPageSize: true, printBackground: true });
    expect(pdf.length).toBeGreaterThan(10000);
    expect(await state()).toEqual(before);
  }
  expect(await page.evaluate(() => window.printChecks)).toEqual([true, true]);
});

test('no JavaScript still exposes both rosters, reference limits, examples and sources', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
  try {
    const page = await context.newPage();
    const external = [];
    page.on('request', request => { if (/^https?:/.test(request.url())) external.push(request.url()); });
    await page.goto(guide);
    await expect(page.locator('[data-roster]:visible')).toHaveCount(2);
    await expect(page.locator('#roster-controls')).toBeHidden();
    await expect(page.locator('[data-setting-profile]')).toHaveCount(4);
    await expect(page.locator('[data-evidence]:visible')).toHaveCount(3);
    await page.locator('#spec-milco-650m > summary').click();
    await expect(page.locator('#spec-milco-650m [data-model-source]')).toHaveCount(2);
    await page.locator('#character-example > summary').click();
    await expect(page.locator('#character-source')).toBeVisible();
    expect(external).toEqual([]);
  } finally {
    await context.close();
  }
});

test('skip link, landmarks, contrast modes and embedded fonts retain usable semantics', async ({ page }) => {
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('main')).toBeFocused();
  await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1);
  expect(await page.evaluate(() => document.fonts.check('600 32px "Barlow Condensed"'))).toBe(true);
  expect(await page.evaluate(() => document.fonts.check('400 12px "JetBrains Mono"'))).toBe(true);
  await page.emulateMedia({ forcedColors: 'active', reducedMotion: 'reduce' });
  await page.getByLabel('Model profile').selectOption('advanced');
  await expect(page.locator('#choice-summary')).toContainText('advanced / CPU');
  await expect(page.getByLabel('Hardware', { exact: true })).toBeVisible();
});

test('actual 200% browser zoom preserves controls and reflow', async ({}, testInfo) => {
  const extension = fileURLToPath(new URL('./fixtures/zoom-extension', import.meta.url));
  const context = await chromium.launchPersistentContext(testInfo.outputPath('zoom-profile'), {
    channel: 'chromium', headless: true, viewport: { width: 1440, height: 960 },
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
  });
  try {
    await context.route(/^https?:/, route => {
      throw new Error(`Unexpected network request: ${route.request().url()}`);
    });
    const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker');
    const page = await context.newPage();
    await page.goto(guide);
    const before = await page.evaluate(() => ({ width: innerWidth, dpr: devicePixelRatio }));
    const factor = await worker.evaluate(async target => {
      const tab = (await chrome.tabs.query({})).find(tab => tab.url === target);
      await chrome.tabs.setZoom(tab.id, 2);
      return chrome.tabs.getZoom(tab.id);
    }, guide);
    expect(factor).toBe(2);
    await expect.poll(() => page.evaluate(() => innerWidth)).toBe(before.width / 2);
    expect(await page.evaluate(() => devicePixelRatio)).toBe(before.dpr * 2);
    await page.getByLabel('Model profile').selectOption('advanced');
    await page.getByLabel('Hardware', { exact: true }).selectOption('gpu');
    await expect(page.locator('#choice-summary')).toContainText('advanced / GPU');
    await page.evaluate(() => document.querySelectorAll('details').forEach(item => { item.open = true; }));
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
    const snapshot = await page.locator('#runtime-limits').ariaSnapshot();
    expect(snapshot).toContain('table');
    expect(snapshot).toContain('Dense tokens');
    expect(snapshot).toContain('advanced');
  } finally {
    await context.close();
  }
});
