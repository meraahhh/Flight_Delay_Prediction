/* Complete local website checks; use the Playwright environment documented in browser_analysis.cjs. */
const assert = require('node:assert/strict');
const {webkit,chromium} = require('playwright');
const {verifyChartSections} = require('./browser_helpers.cjs');
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
    const airportBoxes=page.getByTestId('stForm').getByTestId('stSelectbox');
    await page.waitForFunction(()=>{
     const inputs=[...document.querySelectorAll('[data-testid="stForm"] [data-testid="stSelectbox"] input[role="combobox"]')].slice(0,2);
     return inputs.length===2&&inputs.every(input=>/^[A-Z]{3} — .+/.test(input.value));
    });
    assert.match(await airportBoxes.nth(0).getByRole('combobox').inputValue(), /^[A-Z]{3} — .+/);
    assert.match(await airportBoxes.nth(1).getByRole('combobox').inputValue(), /^[A-Z]{3} — .+/);
    assert.equal(await page.getByRole('heading').count(),1);
    assert.equal(await page.getByTestId('stForm').getByRole('combobox').count(),3);
    await page.screenshot({path:`/tmp/airline-prediction-${name}-${width}.png`,fullPage:true});
    await submit.click();
    await page.getByText('Delay Probability',{exact:true}).waitFor();
    assert.equal(await page.getByTestId('stMetricValue').count(),3);
    assert.deepEqual(await page.getByTestId('stMetricLabel').locator('p').allTextContents(),['Predicted Status','Delay Probability','Risk Level']);
    assert.equal(await page.getByTestId('stCaptionContainer').filter({hasText:'not live flight status'}).count(),1);
    assert.equal(await page.getByTestId('stProgress').count(),0);
    const originText=await airportBoxes.nth(0).getByRole('combobox').inputValue();
    const originCode=originText.match(/\b[A-Z]{3} —/)[0].slice(0,3);
    const destination=airportBoxes.nth(1).getByRole('combobox');
    await destination.fill(originCode);
    await page.getByRole('option',{name:new RegExp(`^${originCode} —`)}).click();
    await submit.click();
    await page.getByText('Departure and destination airports cannot be the same.',{exact:true}).waitFor();
    await page.waitForFunction(()=>document.querySelectorAll('[data-testid="stMetricValue"]').length===0);
    assert.equal(await page.getByTestId('stMetricValue').count(),0);
    assert.equal(await page.getByTestId('stException').count(),0);
    const nav=page.locator('.st-key-site_navigation');
    await nav.getByText('About',{exact:true}).click();
    await page.getByRole('heading',{name:'About Aviation Signature',exact:true}).waitFor();
    for(const title of ['How It Works','Model Performance','Data & Limitations']) await page.getByRole('heading',{name:title,exact:true}).waitFor();
    const aboutText=await page.getByTestId('stMain').innerText();
    assert(aboutText.includes('Aviation Signature uses machine learning and historical flight data to estimate departure delay risk.'));
    assert(!/university|student|coursework/i.test(aboutText));
    assert(aboutText.includes('not live flight status'));
    assert.deepEqual(await page.getByTestId('stMetricValue').allTextContents(),['72.56%','27.04%','37.89%','31.56%']);
    await nav.getByText('Flight Analytics',{exact:true}).click();
    await page.getByTestId('stMetricValue').first().filter({hasText:'6,982,766'}).waitFor();
    // Wait for Streamlit to remove the previous About section's metric nodes.
    await page.waitForFunction(()=>document.querySelectorAll('[data-testid="stMetric"]').length===4);
    assert.equal(await page.locator('[data-testid="stExpander"] details[open]').count(),0);
    assert.equal(await page.getByTestId('stExpander').count(),4);
    await page.getByTestId('stMultiSelect').filter({hasText:'Months'}).getByRole('combobox').waitFor({state:'visible'});
    assert.equal(await page.locator('[data-testid="stMetric"]:visible').count(),4);
    assert.equal(await page.locator('[data-testid="stDataFrame"]:visible').count(),0);
    assert(await page.evaluate(()=>{
     const kpis=document.querySelector('.st-key-analysis_kpis').getBoundingClientRect();
     const filters=document.querySelector('.st-key-analysis_filters').getBoundingClientRect();
     return kpis.bottom<=filters.top;
    }), 'KPI cards must appear above the visible filters');
    await verifyChartSections(page);
    assert.equal(await page.getByTestId('stVegaLiteChart').count(),7);
    assert.equal(await page.getByTestId('stMetricValue').first().innerText(),'6,982,766');
    await page.screenshot({path:`/tmp/airline-analytics-${name}-${width}.png`,fullPage:true});
    await page.getByText('Detailed Data',{exact:true}).click();
    await page.waitForLoadState('networkidle');
    let ready=page.waitForEvent('download');
    await page.getByRole('button',{name:'Download filtered summary (CSV)',exact:true}).click();
    const csv=await downloadText(await ready);
    assert(csv.includes('delay_rate_pct')&&csv.includes('526882'));
    await page.getByText('Detailed Data',{exact:true}).click();
    await page.getByTestId('stDataFrame').waitFor({state:'hidden'});
    await page.getByText('Detailed Data',{exact:true}).click();
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
