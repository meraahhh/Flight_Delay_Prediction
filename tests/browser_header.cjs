/* Local regression test. Uses the same temporary Playwright setup as browser_analysis.cjs. */
const assert = require('node:assert/strict');
const {webkit, chromium} = require('playwright');
(async () => {
 for (const [engineName, engine] of Object.entries({webkit, chromium})) {
  const browser = await engine.launch();
  try {
   for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({viewport:{width,height:1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.FLIGHT_APP_URL || 'http://127.0.0.1:8765');
    await page.getByRole('button',{name:'Predict Flight',exact:true}).waitFor();
    await page.locator('.brand').waitFor();
    assert((await page.locator('.brand').innerText()).includes('AVIATION SIGNATURE'));
    await page.locator('.st-key-site_navigation [role="radiogroup"] label').first().waitFor();
    await page.locator('.hero-title').waitFor();
    assert.equal((await page.locator('.hero-title').innerText()).replace(/\s+/g,' '), 'Fly Smarter. Travel with Confidence.');
    assert.equal(await page.locator('.hero-copy').innerText(),'Predict flight delays and explore travel trends.');
    assert.equal(await page.locator('.feature-card').count(),0);
    assert.equal(await page.locator('.st-key-home_hero button').count(),2);
    assert.equal(await page.locator('.site-footer').innerText(),'AVIATION SIGNATURE');
    assert(await page.locator('.hero-photo img').evaluate(img=>img.complete&&img.naturalWidth===1920&&img.src.startsWith('data:image/jpeg;base64,'))); 
    const result = await page.evaluate(() => {
     const rgb = color => color.match(/[\d.]+/g).slice(0,3).map(Number);
     const luminance = color => rgb(color).map(v => v/255).map(v => v<=.04045 ? v/12.92 : ((v+.055)/1.055)**2.4).reduce((s,v,i)=>s+v*[.2126,.7152,.0722][i],0);
     const contrast = (a,b) => (Math.max(luminance(a),luminance(b))+.05)/(Math.min(luminance(a),luminance(b))+.05);
     const bg = 'rgb(245,240,233)'; // Warm ivory surface, including selected navigation.
     const headerBottom = document.querySelector('[data-testid="stHeader"]').getBoundingClientRect().bottom;
     const selectors = ['.brand','.brand small','.st-key-site_navigation [role="radiogroup"] label p','.hero-title','.hero-accent','.hero-copy','.eyebrow','.st-key-home_hero [data-testid="stCaptionContainer"] p','.st-key-home_hero button p'];
     const brandBottom=document.querySelector(".brand").getBoundingClientRect().bottom;
     const navTop=document.querySelector('.st-key-site_navigation [role="radiogroup"] label').getBoundingClientRect().top;
     return {brandNavClearance:navTop-brandBottom,overflow:document.documentElement.scrollWidth>innerWidth,
      elements:selectors.flatMap(selector=>[...document.querySelectorAll(selector)].map(el=>{
       const rect=el.getBoundingClientRect(),style=getComputedStyle(el);
       const hit=document.elementFromPoint(rect.x+Math.min(rect.width/2,100),rect.y+rect.height/2);
       const button=el.closest('button');
       return {selector,text:el.innerText,top:rect.top,bottom:rect.bottom,left:rect.left,right:rect.right,
        belowToolbar:rect.top>=headerBottom,
        unblocked:!!hit&&(hit===el||el.contains(hit)||hit.contains(el)),
        contrast:contrast(style.color,button?getComputedStyle(button).backgroundColor:bg),
        visible:style.visibility==='visible'&&style.opacity!=='0'&&rect.width>0&&rect.height>0};
      }))};
    });
    if(width<=760) assert(result.brandNavClearance>=4,`Mobile brand and links overlap: ${result.brandNavClearance}`);
    assert.equal(result.overflow,false,`${engineName} ${width}: horizontal overflow`);
    for (const element of result.elements) {
     assert(element.visible,`${element.text}: hidden`);
     assert(element.belowToolbar,`${element.text}: toolbar overlap`);
     assert(element.left>=0&&element.right<=width+1,`${element.text}: clipped horizontally`);
     assert(element.contrast>=4.5,`${element.text}: contrast ${element.contrast.toFixed(2)}`);
     // The Home text and action labels all fit within this viewport.
     if(element.bottom<1000) assert(element.unblocked,`${element.text}: covered by another element`);
    }
    const nav=page.locator('.st-key-site_navigation');
    for(const section of ['About','Flight Prediction','Flight Analytics','Home']){
     await nav.getByText(section,{exact:true}).click(); // No force; fails if covered.
     if(section!=='Home') await page.getByRole('heading',{name:section==='About'?'About Aviation Signature':section,exact:true}).waitFor();
     else await page.getByRole('button',{name:'Predict Flight',exact:true}).waitFor();
    }
    for(const name of ['Predict Flight','View Analytics']){
     const button=page.getByRole('button',{name,exact:true});
     await button.hover();
     await page.waitForTimeout(220);
     const colors=await button.evaluate(el=>({label:getComputedStyle(el.querySelector('p')).color,background:getComputedStyle(el).backgroundColor}));
     assert(colors.label!==colors.background,`${name}: invisible hover label`);
    }
    assert.deepEqual(errors,[]);
    await page.screenshot({path:`/tmp/flight-header-${engineName}-${width}.png`,fullPage:true});
    console.log(`${engineName} ${width}px: readable text (contrast ≥4.5), no toolbar overlap or clipping, all navigation links clickable`);
    await page.close();
   }
  } finally {await browser.close();}
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
