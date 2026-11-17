const assert = require('node:assert/strict');
const chartSections = ['Monthly delays', 'Departure hours', 'Departure airports', 'Airline delays',
 'Weekday delays', 'Destination airports', 'Weekday–hour heatmap'];

async function verifyChartSections(page) {
 await page.waitForFunction(() => document.querySelectorAll('[data-testid="stVegaLiteChart"]').length === 7);
 assert.equal(await page.locator('[data-testid="stVegaLiteChart"]:visible').count(), 0);
 for (const title of chartSections) {
  const toggle = page.getByTestId('stExpander').locator('summary')
   .filter({has:page.getByText(title,{exact:true})});
  const section = toggle.locator('..');
  await toggle.click();
  const chart = section.getByTestId('stVegaLiteChart');
  await chart.waitFor({state:'visible'});
  await chart.locator('canvas, svg.marks').first().waitFor({state:'visible'});
  assert.equal(await page.locator('[data-testid="stVegaLiteChart"]:visible').count(), 1);
  const box = await chart.boundingBox();
  assert(box && box.width > 0 && box.height > 0, `${title}: zero-size chart`);
  await toggle.click();
  await chart.waitFor({state:'hidden'});
 }
 assert.equal(await page.locator('[data-testid="stVegaLiteChart"]:visible').count(),0);
}
module.exports = {verifyChartSections};
