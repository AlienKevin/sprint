// Opt-in Chromium CDP touch emulation, not an iOS/Android hardware-device test.
// CDP_PORT=<port> BENCH_BASE=http://127.0.0.1:59453 MODE=all run-heavy node this-file.mjs
// ISOLATED=1 uses a controlled same-origin iframe for the collision case only,
// avoiding unrelated homepage WebGL contexts on software-GPU test machines.
import fs from 'node:fs';
const port=Number(process.env.CDP_PORT||9222),base=process.env.BENCH_BASE||'http://127.0.0.1:59453',mode=process.env.MODE||'all';
const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json(),target=targets.find(x=>x.type==='page'&&x.url.startsWith(base))||targets.find(x=>x.type==='page');
const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let id=0;const pending=new Map(),events=[],report={base,mode,checks:[]};
ws.addEventListener('message',e=>{const d=JSON.parse(e.data);if(d.id){const p=pending.get(d.id);if(!p)return;pending.delete(d.id);clearTimeout(p.timer);d.error?p.reject(d.error):p.resolve(d.result);}else events.push(d);});
const send=(method,params={})=>new Promise((resolve,reject)=>{const n=++id,timer=setTimeout(()=>{pending.delete(n);reject(Error('Timeout '+method));},25000);pending.set(n,{resolve,reject,timer});ws.send(JSON.stringify({id:n,method,params}));});
const ev=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
const wait=async expression=>{for(let i=0;i<160;i++){if(await ev(expression))return;await new Promise(r=>setTimeout(r,100));}throw Error('Readiness timeout');};
const check=(name,pass,detail)=>{report.checks.push({name,pass,detail});console.log(JSON.stringify({name,pass,detail}));fs.writeFileSync(`/tmp/replay-touch-${mode}.json`,JSON.stringify(report,null,2));};
const viewport=width=>send('Emulation.setDeviceMetricsOverride',{width,height:844,deviceScaleFactor:1,mobile:width<=720});
const shot=async name=>{const d=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(`/tmp/replay-touch-${name}.png`,Buffer.from(d.data,'base64'));};
await send('Page.enable');await send('Runtime.enable');await send('Network.enable');await send('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:5});
if(mode==='all'||mode==='css'){
 // CSS-only runs need no live WebGL contexts; leave gesture runs untouched.
 if(mode==='css')await send('Network.setBlockedURLs',{urls:['*/replay/*']});
 await viewport(1280);await send('Page.navigate',{url:base+'/'});await wait(`document.querySelectorAll('#budget-results-table .budget-row').length===3`);
 await ev(`document.querySelectorAll('iframe').forEach(f=>{try{f.contentWindow.__G1_REPLAY__?.seek(0)}catch{}})`);
 const hint=await ev(`(()=>{const p=document.querySelector('.chart-interaction-hint');return {text:p?.textContent,previous:p?.previousElementSibling?.tagName,heading:p?.previousElementSibling?.textContent}})()`);check('chart helper directly follows performance heading',hint.text==='Click on any dot to replay that policy.'&&hint.previous==='H2'&&hint.heading==='Performance vs cost',hint);
 for(const width of [1280,390,678]){await viewport(width);
 const labelFonts=await ev(`['.hero-score-head strong','.hero-score-row span strong','.experiment-model strong'].map(s=>getComputedStyle(document.querySelector(s)).fontSize)`);check(`data label fonts ${width}px`,labelFonts.every(x=>x==='12px'),labelFonts);
 const branding=await ev(`(()=>{const nav=document.querySelector('.home-nav .brand'),footer=document.querySelector('footer .brand'),h2=document.querySelector('.section-head h2'),intro=document.querySelector('.hero-copy .intro');return {h1Count:document.querySelectorAll('h1').length,h1InNav:nav.tagName==='H1',heroH1:document.querySelectorAll('.hero-copy h1').length,fontSizes:[nav,footer,h2].map(x=>getComputedStyle(x).fontSize),introGap:intro.getBoundingClientRect().top-nav.getBoundingClientRect().bottom,footerWidth:document.querySelector('footer').scrollWidth,editorialDatePresent:!!document.querySelector('.footer-updated')||document.querySelector('footer').textContent.includes('Updated on')}})()`);
 check(`branding ${width}px`,branding.h1Count===1&&branding.h1InNav&&branding.heroH1===0&&new Set(branding.fontSizes).size===1&&branding.introGap>=0&&branding.introGap<100&&branding.footerWidth<=width&&!branding.editorialDatePresent,branding);
 for(const expanded of [false,true]){
  await ev(`(()=>{const b=document.querySelector('#resource-bars .experiment-table-toggle');if((b.getAttribute('aria-expanded')==='true')!==${expanded})b.click()})()`);
  const d=await ev(`(()=>{const shown=x=>Boolean(x.getClientRects().length)&&getComputedStyle(x).display!=='none',rows=[...document.querySelectorAll('#budget-results-table .budget-row')];return {width:innerWidth,overflow:document.documentElement.scrollWidth>innerWidth,rows:rows.length,visiblePrices:[...document.querySelectorAll('.budget-row .budget-price')].filter(shown).length,visibleMobilePrices:[...document.querySelectorAll('.budget-price-mobile')].filter(shown).length,priceHeaders:[...document.querySelectorAll('#budget-results-table thead th:nth-child(n+5)')].filter(shown).length,costs:rows.map(r=>[...r.querySelectorAll('.budget-heat')].filter(shown).map(x=>x.dataset.label)),footnotes:shown(document.querySelector('.budget-price-sources'))}})()`);
  const mobile=width<=720;check(`cost columns ${width}px ${expanded?'all':'best'} trials`,d.rows===(expanded?15:3)&&d.visiblePrices===(mobile?0:6)&&d.priceHeaders===(mobile?0:2)&&d.visibleMobilePrices===0&&d.costs.every(x=>x.join(',')==='Model API,GPU,CPU')&&d.footnotes&&!d.overflow,d);
  if(width===390&&!expanded){await ev(`document.querySelector('#cost-breakdown').scrollIntoView()`);await shot('cost-mobile');}
 }}
 for(const width of [1280,374,320]){await viewport(width);
  const cards=await ev(`(()=>{const rect=x=>{const r=x.getBoundingClientRect();return {left:r.left,right:r.right,top:r.top,bottom:r.bottom}};return [...document.querySelectorAll('.dq-card-head')].map(h=>({title:h.querySelector('h3').textContent,heading:rect(h.querySelector('h3')),link:rect(h.querySelector('a')),head:rect(h),href:h.querySelector('a').getAttribute('href')}))})()`);
  check(`demo title/link alignment ${width}px`,cards.length===2&&cards.every(c=>c.link.left>c.heading.right&&Math.abs((c.link.top+c.link.bottom-c.heading.top-c.heading.bottom)/2)<1&&Math.abs(c.link.right-c.head.right)<1)&&cards[0].href==='/trajectory?run=s10-vexp-r123-20260828-luna-4&policies=20&focus=20&step=a1-s651'&&cards[1].href==='/trajectory?run=s10-vexp-r123-20260828-luna-5&policies=1&focus=1&step=a1-s183',cards);
  if(width===374||width===320){await ev(`document.querySelector('.dq-cards').scrollIntoView({behavior:'instant',block:'start'})`);await wait(`Math.abs(document.querySelector('.dq-cards').getBoundingClientRect().top)<2`);await shot('demo-head-'+width);}
 }
 const axis=await ev(`document.querySelector('#cost-chart')?.textContent.includes('Cost ($)')`);check('cost chart concise axis label',axis);
 if(mode==='css')await send('Network.setBlockedURLs',{urls:[]});
}
if(mode==='all'||mode==='gestures'){
 await viewport(390);
 for(const [name,path,frame] of [
  ['hero','/','.model-race iframe'],
  ['collision',process.env.ISOLATED?'/version.json':'/',process.env.ISOLATED?'#touch-test-frame':'.dq-card:nth-of-type(2) iframe'],
  ['trajectory','/trajectory?run=s10-vexp-r123-20260828-luna-2&policies=9&focus=9','#trajectory-policy-replay-frame'],
 ]){
  if(process.env.ONLY&&process.env.ONLY!==name)continue;
  await viewport(390);await send('Page.bringToFront');await send('Page.navigate',{url:base+path});
  if(name==='collision'&&process.env.ISOLATED)await ev(`document.open();document.write('<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0;background:#000}iframe{display:block;width:320px;height:250px;margin:40px 35px;border:0}</style><iframe id="touch-test-frame" src="/replay/frontier-bca4f7ab8e3c?example=1&autoplay=0"></iframe>');document.close()`);
  const w=frame?`document.querySelector('${frame}').contentWindow`:'window',api=`${w}.__G1_REPLAY__`;
  if(frame){await wait(`Boolean(document.querySelector('${frame}'))`);await ev(`document.querySelector('${frame}').scrollIntoView({block:'center',behavior:'instant'})`);}
  await wait(frame?`Boolean(document.querySelector('${frame}')?.contentWindow?.__G1_REPLAY__)&&!document.querySelector('#trajectory-policy-replay')?.classList.contains('is-loading')`:'Boolean(window.__G1_REPLAY__)');
  await ev(`document.querySelectorAll('iframe').forEach(f=>{try{f.contentWindow.__G1_REPLAY__?.seek(0)}catch{}})`);
  if(name==='trajectory'){await ev(`window.__touchOriginal=${api};document.querySelector('.policy-reference[data-policy-label^="Policy #8,"]').click()`);await wait(`(${w}).document.querySelectorAll('.lc').length===2&&!document.querySelector('#trajectory-policy-replay').classList.contains('is-loading')`);check('trajectory retains renderer before touch tests',await ev(`${api}===__touchOriginal`));}
  await ev(`(()=>{const w=${w};w.__G1_REPLAY__.seek(0);w.__nativeTouch=[];['pointerdown','pointerup','pointercancel','touchstart','touchend','touchcancel'].forEach(type=>w.document.addEventListener(type,e=>w.__nativeTouch.push({type,trust:e.isTrusted,pointer:e.pointerType,touches:e.touches?.length}),true))})()`);
  await ev('document.fonts.ready');if(name!=='trajectory')await ev(`document.querySelector('${frame}').scrollIntoView({block:'center',behavior:'instant'})`);await shot(name+'-before');
  await ev('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
  const rect=await ev(`(()=>{const w=${w},r=w.document.querySelector('canvas').getBoundingClientRect(),f=${frame?`document.querySelector('${frame}').getBoundingClientRect()`:'{x:0,y:0}'};return {x:r.x+f.x,y:r.y+f.y,width:r.width,height:r.height}})()`);
  const cx=Math.round(rect.x+rect.width*.5),cy=Math.round(rect.y+rect.height*.62),point=(id,x,y=cy)=>({id,x,y,radiusX:4,radiusY:4,force:1});
  const hit=await ev(`({target:document.elementFromPoint(${cx},${cy})?.tagName,viewport:innerWidth,scrollY})`);check(`${name}: native touch targets visible mobile iframe`,hit.target==='IFRAME'&&hit.viewport===390&&cy>0&&cy<844,{hit,rect,cx,cy});
  const touch=async(type,touchPoints=[])=>{await send('Input.dispatchTouchEvent',{type,touchPoints});await new Promise(r=>setTimeout(r,150));await ev(`new Promise(r=>{${w}.requestAnimationFrame(r);${w}.setTimeout(r,200)})`);};
  const camera=()=>ev(`${api}.camera()`),equalOrbit=(a,b)=>Math.abs(a.view.az-b.view.az)<1e-9&&Math.abs(a.view.el-b.view.el)<1e-9;
  let a=await camera();await touch('touchStart',[point(1,cx)]);await touch('touchMove',[point(1,cx+30,cy+8)]);await touch('touchEnd');let b=await camera();check(`${name}: one finger rotates`,!equalOrbit(a,b),{before:a.view,after:b.view,rect});
  a=b;await touch('touchStart',[point(1,cx-30),point(2,cx+30)]);await touch('touchMove',[point(1,cx-65),point(2,cx+65)]);b=await camera();check(`${name}: spread zooms in without rotation`,b.view.dist<a.view.dist&&equalOrbit(a,b),{before:a.view,after:b.view});
  a=b;await touch('touchMove',[point(1,cx-20),point(2,cx+20)]);b=await camera();check(`${name}: close zooms out without rotation`,b.view.dist>a.view.dist&&equalOrbit(a,b),{before:a.view,after:b.view});
  // Chromium supports ending the named contact while retaining the other one.
  a=b;await touch('touchEnd',[point(2,cx+20)]);b=await camera();check(`${name}: two-to-one has no jump or pick`,equalOrbit(a,b)&&a.view.dist===b.view.dist&&a.follow===b.follow,{before:a.view,after:b.view});
  await touch('touchMove',[point(1,cx-10)]);let c=await camera();check(`${name}: remaining finger resumes small orbit`,Math.abs(c.view.az-b.view.az)>0&&Math.abs(c.view.az-b.view.az)<.3,{before:b.view,after:c.view});await touch('touchEnd');
  a=await camera();await touch('touchStart',[point(1,cx-30),point(2,cx+30)]);await touch('touchCancel');b=await camera();check(`${name}: cancel leaves camera unchanged`,equalOrbit(a,b)&&a.view.dist===b.view.dist&&a.follow===b.follow,{before:a.view,after:b.view});
  check(`${name}: mobile zoom buttons stay hidden`,await ev(`getComputedStyle(${w}.document.querySelector('.camera-ctl')).display==='none'`));
  a=b;await ev(`${w}.document.querySelector('#camera-zoom-in').click()`);b=await camera();await ev(`${w}.document.querySelector('#camera-zoom-out').click()`);c=await camera();check(`${name}: retained zoom button handlers still work`,b.view.dist<a.view.dist&&c.view.dist>b.view.dist,{before:a.view.dist,plus:b.view.dist,minus:c.view.dist});
  const native=await ev(`${w}.__nativeTouch`);check(`${name}: trusted touch events reached renderer`,native.some(x=>x.type==='touchstart'&&x.touches===2&&x.trust)&&native.some(x=>x.type==='pointercancel'&&x.trust),native);
  check(`${name}: gestures preserve paused playback`,await ev(`${api}.playback().time===0&&!${api}.playback().playing`));await shot(name);
 }
}
check('no uncaught browser exceptions',!events.some(x=>x.method==='Runtime.exceptionThrown'),events.filter(x=>x.method==='Runtime.exceptionThrown').map(x=>x.params.exceptionDetails.text));
ws.close();if(report.checks.some(x=>!x.pass))process.exitCode=1;
