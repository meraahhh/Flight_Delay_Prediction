/* Real-browser regression check against a running local Streamlit server.
 * Install Playwright outside the project, then run with its node_modules on
 * NODE_PATH and its installed browser directory on PLAYWRIGHT_BROWSERS_PATH.
 * Example: NODE_PATH=/tmp/flight-browser/node_modules \
 *   PLAYWRIGHT_BROWSERS_PATH=/tmp/flight-browser/browsers node tests/browser_analysis.cjs
 * Override the default localhost URL with FLIGHT_APP_URL if needed.
 */
const assert = require('node:assert/strict');
const { chromium, webkit } = require('playwright');
const {verifyChartSections} = require('./browser_helpers.cjs');

(async () => {
  for (const [name, engine] of Object.entries({ webkit, chromium })) {
    const browser = await engine.launch({ headless: true });
    try {
      const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.stack));
      page.on('console', message => {
        if (message.type() === 'error') errors.push(message.text());
      });
      await page.goto(process.env.FLIGHT_APP_URL || 'http://127.0.0.1:8765');
      await page.getByRole('button', { name: 'View Analytics', exact: true }).click();
      const charts = page.getByTestId('stVegaLiteChart');
      async function verify(label, flights) {
        await page.getByTestId('stMetricValue').first().filter({ hasText: flights }).waitFor();
        await verifyChartSections(page);
        await page.waitForTimeout(700);
        assert.equal(await charts.count(), 7);
        assert.equal(await page.getByTestId('stException').count(), 0);
        assert.deepEqual(errors, [], `${name} ${label}: browser errors`);
        console.log(`${name} ${label}: 7 charts expand/collapse individually, correct flight count, no browser errors`);
      }
      await verify('all months', '6,982,766');
      await page.getByText('Filters', {exact:true}).click();
      const months = page.getByTestId('stMultiSelect').filter({ hasText: 'Months' });
      const input = months.getByRole('combobox');
      await input.focus();
      await input.press('Backspace'); // Remove the initial All selection.
      await input.fill('January');
      await page.getByRole('option', { name: 'January', exact: true }).click();
      await input.press('Escape');
      await verify('January', '526,882');
      await page.getByRole('button', { name: 'Reset filters', exact: true }).click();
      await verify('reset to all months', '6,982,766');
    } finally {
      await browser.close();
    }
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
