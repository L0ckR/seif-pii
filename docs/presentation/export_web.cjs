/* Render the offline vector deck and inspect actual browser text bounds. */
const fs = require('fs');
const path = require('path');
const deps = process.env.SEIF_DOCUMENT_NODE_MODULES || '/mnt/c/Users/doly2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { chromium } = require(path.join(deps, 'playwright'));
const root = __dirname;
const qa = path.resolve(root, '../..', 'output/playwright/seif-presentation');
fs.mkdirSync(qa, { recursive: true });

(async () => {
  const browser = await chromium.launch({executablePath: process.env.SEIF_CHROMIUM || '/home/lockr/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome', headless: true});
  const page = await browser.newPage({viewport:{width:1600,height:1024}, deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror', e=>errors.push(String(e)));
  await page.goto('http://127.0.0.1:8791/', {waitUntil:'networkidle'});
  await page.evaluate(()=>document.fonts.ready);
  const count=await page.locator('.slide').count();
  if(count!==16) throw Error(`Expected16 slides, got${count}`);
  const bounds=[];
  for(let i=0;i<count;i++) {
    await page.selectOption('#select',String(i));
    const local=await page.locator('.slide.is-active').evaluate((slide)=>[...slide.querySelectorAll('text')].map(t=>{
      const b=t.getBBox();const mat=t.getCTM();
      const p=new DOMPoint(b.x,b.y).matrixTransform(mat);
      const max=Number(t.getAttribute('data-max-width')||0);
      return {text:t.textContent, x:p.x,y:p.y,w:b.width*mat.a,h:b.height*mat.d,
        overflow:b.x<0||b.y<0||b.x+b.width>1602||b.y+b.height>902||(max>0&&b.width>max+1),
        maxWidth:max,rawWidth:b.width};
    }).filter(x=>x.overflow));
    if(local.length) bounds.push({slide:i+1,findings:local});
    await page.locator('.slide.is-active').screenshot({path:path.join(qa,`${String(i+1).padStart(2,'0')}.png`)});
  }
  // Basic real interactions, not an implementation-mirroring unit suite.
  await page.selectOption('#select','0');
  await page.locator('#next').click();
  if(await page.locator('#select').inputValue()!=='1')throw Error('Next navigation failed');
  await page.keyboard.press('ArrowLeft');
  if(await page.locator('#select').inputValue()!=='0')throw Error('Keyboard navigation failed');
  await page.locator('#notesButton').click();
  if(!await page.locator('#notes').isVisible())throw Error('Notes unavailable');
  await page.locator('#notesButton').click();
  await page.locator('#grid').click();
  await page.locator('.slide').nth(9).click();
  if(await page.locator('#select').inputValue()!=='9')throw Error('Grid selection failed');
  await page.evaluate(()=>{document.title='СЕЙФ — архитектура и эксперименты';});
  await page.emulateMedia({media:'print'});
  await page.pdf({path:path.join(root,'seif-architecture.pdf'), printBackground:true, preferCSSPageSize:true});
  await page.emulateMedia({media:'screen'});
  const thumbs=Array.from({length:count},(_,i)=>`<div><img src="http://127.0.0.1:8791/slides/${String(i+1).padStart(2,'0')}.svg"><b>${String(i+1).padStart(2,'0')}</b></div>`).join('');
  await page.setViewportSize({width:1600,height:1030});
  await page.setContent(`<html><body style="margin:0;background:#0c140f;color:#93a397;font:14px Consolas;padding:24px"><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px">${thumbs}</div><style>img{display:block;width:100%;border:1px solid #334138}b{display:block;padding:6px 0 9px}</style></body></html>`);
  await page.waitForFunction(()=>[...document.images].every(x=>x.complete));
  await page.screenshot({path:path.join(root,'overview.png'),fullPage:true});
  fs.writeFileSync(path.join(qa,'browser-checks.json'),JSON.stringify({slides:count,errors,textBounds:bounds,navigation:'PASS',pdf:'exported'},null,2));
  console.log(JSON.stringify({slides:count,errors,textBounds:bounds,output:path.join(root,'seif-architecture.pdf')},null,2));
  await browser.close();
  if(errors.length||bounds.length) process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1});
