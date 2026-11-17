/* Complete local website checks; use the Playwright environment documented in browser_analysis.cjs. */
const assert = require('node:assert/strict');
const {webkit,chromium} = require('playwright');
async function downloadText(download) {
 const stream=await download.createReadStream();let text='';
 for await (const chunk of stream) text+=chunk.toString();
 return text;
}
(async()=>{
 for(const [name,engine] of Object.entries({webkit,chromium})){
  const browser=await engine.launch();
  try{
   for(const width of [1440,390]){
    const page=await browser.newPage({viewport:{width,height:1000},acceptDownloads:true});
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
    // Prove the website and hero do not require externally hosted assets.
    await page.route(/^https?:\/\//,route=>new URL(route.request().url()).hostname==='127.0.0.1'?route.continue():route.abort());
    await page.goto(process.env.FLIGHT_APP_URL||'http://127.0.0.1:8765');
    await page.locator('.hero-photo img').waitFor();
    assert(await page.locator('.hero-photo img').evaluate(img=>img.complete&&img.naturalWidth===1920));
    assert.equal((await page.locator('.hero-title').innerText()).replace(/\s+/g,' '),'Fly Smarter. Travel with Confidence.');
    await page.screenshot({path:`/tmp/airline-home-${name}-${width}.png`,fullPage:true});
    await page.getByRole('button',{name:'Predict Flight',exact:true}).click();
    const submit=page.getByRole('button',{name:'Predict Flight Delay',exact:true});
    await submit.waitFor();
    await page.getByTestId('stForm').getByRole('combobox').first().waitFor();
    await page.waitForLoadState('networkidle');
    await page.screenshot({path:`/tmp/airline-prediction-${name}-${width}.png`,fullPage:true});
    await submit.click();
    await page.getByText('Estimated delay probability',{exact:true}).waitFor();
    assert.equal(await page.getByTestId('stMetricValue').count(),3);
    assert.equal(await page.getByTestId('stException').count(),0);
    const nav=page.locator('.st-key-site_navigation');
    await nav.getByText('About',{exact:true}).click();
    await page.getByRole('heading',{name:'About the system',exact:true}).waitFor();
    assert.deepEqual(await page.getByTestId('stMetricValue').allTextContents(),['72.56%','27.04%','37.89%','31.56%']);
    await nav.getByText('Flight Analytics',{exact:true}).click();
    await page.waitForFunction(()=>document.querySelectorAll('[data-testid="stVegaLiteChart"] canvas, [data-testid="stVegaLiteChart"] svg.marks').length>=7);
    assert.equal(await page.getByTestId('stVegaLiteChart').count(),7);
    assert.equal(await page.getByTestId('stMetricValue').first().innerText(),'6,982,766');
    await page.screenshot({path:`/tmp/airline-analytics-${name}-${width}.png`,fullPage:true});
    await page.getByText('Explore and download summaries',{exact:true}).click();
    await page.waitForLoadState('networkidle');
    let ready=page.waitForEvent('download');
    await page.getByRole('button',{name:'Download filtered summary (CSV)',exact:true}).click();
    const csv=await downloadText(await ready);
    assert(csv.includes('delay_rate_pct')&&csv.includes('526882'));
    await page.getByText('Data source and calculation notes',{exact:true}).click();
    await page.waitForLoadState('networkidle');
    ready=page.waitForEvent('download');
    await page.getByRole('button',{name:'Download data metadata (JSON)',exact:true}).click();
    assert.equal(JSON.parse(await downloadText(await ready)).totals.flights,6982766);
    assert.deepEqual(errors,[]);
    console.log(`${name} ${width}px: bundled hero offline, Home CTA, prediction, About metrics, 7 analytics charts, CSV and JSON downloads passed`);
    await page.close();
   }
  }finally{await browser.close();}
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
