const assert = require('node:assert/strict');
const chartGroups = [
 ['Monthly & Time Analysis',4], ['Airline Performance',1], ['Airport Performance',2],
];

async function verifyChartSections(page) {
 await page.waitForFunction(() => document.querySelectorAll('[data-testid="stVegaLiteChart"]').length === 7);
 assert.equal(await page.locator('[data-testid="stVegaLiteChart"]:visible').count(),0);
 for (const [title,count] of chartGroups) {
  const toggle=page.getByTestId('stExpander').locator('summary')
   .filter({has:page.getByText(title,{exact:true})});
  const section=toggle.locator('..');
  await toggle.click();
  const charts=section.getByTestId('stVegaLiteChart');
  assert.equal(await charts.count(),count);
  for(const chart of await charts.all()) {
   await chart.waitFor({state:'visible'});
   await chart.locator('canvas, svg.marks').first().waitFor({state:'visible'});
   const box=await chart.boundingBox();
   assert(box&&box.width>0&&box.height>0,`${title}: zero-size chart`);
  }
  assert.equal(await page.locator('[data-testid="stVegaLiteChart"]:visible').count(),count);
  await toggle.click();
  await charts.first().waitFor({state:'hidden'});
 }
 assert.equal(await page.locator('[data-testid="stVegaLiteChart"]:visible').count(),0);
}
module.exports={verifyChartSections};
