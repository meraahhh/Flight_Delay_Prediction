/* Final cross-page visual/accessibility checks; same Playwright setup as browser_analysis.cjs. */
const assert=require('node:assert/strict');
const{webkit,chromium}=require('playwright');
(async()=>{
 for(const[name,engine]of Object.entries({webkit,chromium})){
  const browser=await engine.launch();
  try{for(const width of[1440,834,390,320]){
   const page=await browser.newPage({viewport:{width,height:1000}});const errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.goto(process.env.FLIGHT_APP_URL||'http://127.0.0.1:8765');
   for(const section of['Home','Flight Prediction','Flight Analytics','About']){
    if(section!=='Home')await page.locator('.st-key-site_navigation').getByText(section,{exact:true}).click();
    const heading=section==='Home'?'Fly Smarter. Travel with Confidence.':section==='About'?'About Aviation Signature':section;
    await page.getByRole('heading',{name:heading,exact:true}).waitFor();
    if(section==='Flight Prediction'){
     await page.getByLabel('Departure Airport',{exact:true}).waitFor();
     await page.getByTestId('stCaptionContainer').filter({hasText:'not live flight status'}).waitFor();
    }
    if(section==='Flight Analytics'){
     await page.getByTestId('stMetricValue').first().filter({hasText:'6,982,766'}).waitFor();
     await page.getByTestId('stCaptionContainer').filter({hasText:'Historical 2024 US flights · Not live flight status.'}).waitFor();
    }
    if(section==='About'){
     await page.getByRole('heading',{name:'Data & Limitations',exact:true}).waitFor();
     await page.getByText('The system uses historical 2024 US flight data.',{exact:false}).waitFor();
    }
    const layout=await page.evaluate(()=>{
     const header=document.querySelector('[data-testid="stHeader"]').getBoundingClientRect();
     const heading=document.querySelector('h1,h3');const h=heading.getBoundingClientRect();
     const fields=[...document.querySelectorAll('[data-testid="stForm"] input, [data-testid="stForm"] [role="spinbutton"], [data-testid="stForm"] [data-testid="stWidgetLabel"]')]
      .filter(e=>!e.closest('[aria-hidden="true"]')&&e.getBoundingClientRect().height>0).map(e=>{const r=e.getBoundingClientRect();return{left:r.left,right:r.right}});
     return{overflow:document.documentElement.scrollWidth>innerWidth,headingClear:h.top>=header.bottom,
      clippedFields:fields.some(r=>r.left<0||r.right>innerWidth+1)};
    });
    assert.equal(layout.overflow,false,`${name} ${width} ${section}: overflow`);
    assert(layout.headingClear,`${name} ${width} ${section}: heading under toolbar`);
    assert.equal(layout.clippedFields,false,`${name} ${width} ${section}: clipped form controls`);
   }
   assert.deepEqual(errors,[]);
   console.log(`${name} ${width}px: all four pages, visible historical notes, no clipping/toolbar overlap or browser exceptions`);
   await page.close();
  }}finally{await browser.close();}
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
