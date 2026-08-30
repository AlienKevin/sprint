// Real-browser functional/churn checks after priority 3. Run after timing.
import fs from 'node:fs';
const port=Number(process.env.CDP_PORT||9222),base=process.env.BENCH_BASE||'http://127.0.0.1:59454';
const target=(await(await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t=>t.type==='page');
const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let id=0;const pending=new Map(),events=[];
ws.addEventListener('message',e=>{const d=JSON.parse(e.data);if(d.id){const p=pending.get(d.id);pending.delete(d.id);d.error?p.reject(d.error):p.resolve(d.result);}else events.push(d);});
const send=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
const ev=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
const report={checks:[],churn:[]};
const save=()=>fs.writeFileSync('/tmp/replay-priority3-correctness.json',JSON.stringify(report,null,2));
const check=(name,pass,detail)=>{report.checks.push({name,pass,detail});console.log(JSON.stringify({name,pass}));save();};
const frame="document.querySelector('#trajectory-policy-replay-frame')";
const win=`${frame}.contentWindow`;
const api=`${win}.__G1_REPLAY__`;
const snapshot=()=>ev(`(()=>{const f=${frame},w=f.contentWindow,a=w.__G1_REPLAY__;return {sameFrame:window.__originalFrame===f,sameApi:window.__originalApi===a,sameCanvas:window.__originalCanvas===w.document.querySelector('canvas'),url:location.href,lanes:[...w.document.querySelectorAll('.lc')].map(x=>x.getAttribute('aria-label')),playback:a?.playback(),camera:a?.camera(),diagnostics:a?.diagnostics?.(),cache:w.__G1_COMPARISON_CACHE__?.stats?.(),failure:w.document.querySelector('#failure-note')?.textContent,parentFailure:document.querySelector('#trajectory-policy-replay-loading')?.textContent}})()`);
const ready=async numbers=>{const end=Date.now()+45000;while(Date.now()<end){if(await ev(`(()=>{const f=${frame},w=f?.contentWindow;return Boolean(w?.__G1_REPLAY__&&!document.querySelector('#trajectory-policy-replay').classList.contains('is-loading')&&JSON.stringify([...w.document.querySelectorAll('.lc')].map(x=>Number(x.getAttribute('aria-label')?.match(/[0-9]+/)?.[0])))===${JSON.stringify(JSON.stringify(numbers))})})()`))return;await new Promise(r=>setTimeout(r,100));}throw Error('Ready timeout '+numbers);};
const click=number=>ev(`document.querySelector('.policy-reference[data-policy-label^="Policy #${number},"]').click()`);
const settleCamera=async()=>{const end=Date.now()+20000;while(Date.now()<end){if(await ev(`!${api}.camera().transitioning`))return;await new Promise(r=>setTimeout(r,100));}throw Error('Camera transition did not settle');};
const heap=async()=>{await send('HeapProfiler.collectGarbage');return send('Runtime.getHeapUsage');};
await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
await send('Network.emulateNetworkConditions',{offline:false,latency:0,downloadThroughput:-1,uploadThroughput:-1});
await send('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
await send('Page.navigate',{url:base+'/trajectory?run=s10-vexp-r123-20260828-luna-2&policies=9&focus=9'});
await ready([9]);
await ev(`window.__originalFrame=${frame};window.__originalApi=${api};window.__originalCanvas=${win}.document.querySelector('canvas');${api}.seek(1)`);
const initial=await snapshot();check('initial policy 9 paused at 1 second',initial.playback.time===1&&!initial.playback.playing&&initial.playback.end===3.06,initial);
await click(8);await ready([9,8]);
let state=await snapshot();check('add preserves iframe/API/canvas and resets paused at zero',state.sameFrame&&state.sameApi&&state.sameCanvas&&state.playback.time===0&&!state.playback.playing&&state.playback.end===3.12,state);
check('unchanged actor and shared geometry are reused',Boolean(initial.diagnostics?.actors?.[0]?.uuid)&&initial.diagnostics.actors[0].uuid===state.diagnostics?.actors?.[0]?.uuid&&JSON.stringify(initial.diagnostics.geometryIds)===JSON.stringify(state.diagnostics?.geometryIds),{before:initial.diagnostics,after:state.diagnostics});
const pairUrl=state.url;
await ev('history.back()');await ready([9]);state=await snapshot();check('Back restores prior policy set without frame replacement',state.sameFrame&&state.playback.time===0,new URL(state.url).searchParams.get('policies'));
await ev('history.forward()');await ready([9,8]);state=await snapshot();check('Forward restores pair without frame replacement',state.sameFrame&&state.playback.time===0&&new URL(state.url).searchParams.get('policies')==='9,8',{url:state.url,expected:pairUrl});
await ev(`${api}.seek(1)`);
const focusStart=events.length;await ev(`${win}.document.querySelector('[aria-label="Follow Policy #9"]').click()`);await settleCamera();state=await snapshot();
const focusRequests=events.slice(focusStart).filter(x=>x.method==='Network.requestWillBeSent');check('focus switch preserves paused time and performs no network requests',state.sameApi&&state.playback.time===1&&focusRequests.length===0,{state,requests:focusRequests.length});
await ev(`${win}.document.querySelector('#replay').click()`);await click(8);await ready([9]);state=await snapshot();check('changing a playing policy set stops and resets playback',state.sameApi&&state.playback.time===0&&!state.playback.playing,state);
await click(8);await ready([9,8]);
// Policy 7 has not been loaded yet: verify failed requests are retryable.
await send('Network.setBlockedURLs',{urls:['*/captures/frontier-ff5380a59e4a.json']});await click(7);
for(let i=0;i<100;i++){state=await snapshot();if(/Unable|failed|load/i.test(state.parentFailure)&&!/Loading selected policies/.test(state.parentFailure))break;await new Promise(r=>setTimeout(r,100));}
check('failed new capture is visible without losing the old renderer',state.sameApi&&/Unable|failed/i.test(state.parentFailure),state);
await click(7);await ready([9,8]);await send('Network.setBlockedURLs',{urls:[]});await click(7);await ready([9,8,7]);state=await snapshot();check('failed capture can be retried successfully',state.sameApi&&state.playback.time===0&&!state.playback.playing,state);
await click(7);await ready([9,8]);
// Delay uncached captures, then supersede a request before it resolves.
await send('Network.emulateNetworkConditions',{offline:false,latency:200,downloadThroughput:500000,uploadThroughput:500000});
await ev(`[1,1,2].forEach(n=>document.querySelector('.policy-reference[data-policy-label^="Policy #'+n+',"]').click())`);await ready([9,8,2]);
await new Promise(r=>setTimeout(r,1200));state=await snapshot();check('stale rapid selection cannot overwrite the latest set',state.sameApi&&state.lanes.join(',')==='Policy #9,Policy #8,Policy #2',state);
await send('Network.emulateNetworkConditions',{offline:false,latency:0,downloadThroughput:-1,uploadThroughput:-1});
await ev(`[1,3,4,6,7].forEach(n=>document.querySelector('.policy-reference[data-policy-label^="Policy #'+n+',"]').click())`);await ready([9,8,2,1,3,4,6,7]);state=await snapshot();
check('eight selected policies keep identity/order and all-policy stall end',state.sameApi&&state.playback.time===0&&!state.playback.playing&&state.playback.end===3.12&&state.playback.recordedEnd===60,state);
await ev(`${api}.seek(3.12)`);const button=await ev(`${win}.document.querySelector('#replay').textContent`);check('all-policy playback end presents Replay',/Replay/.test(button),button);
const restarted=await ev(`(()=>{${win}.document.querySelector('#replay').click();return ${api}.playback()})()`);check('Replay restarts playback',restarted.playing&&restarted.time===0,restarted);await ev(`${api}.seek(1)`);
await ev(`[2,1,3,4,6,7].forEach(n=>document.querySelector('.policy-reference[data-policy-label^="Policy #'+n+',"]').click())`);await ready([9,8]);
await settleCamera();report.beforeChurn=await snapshot();report.heapBeforeChurn=await heap();const churnStart=events.length;
for(let cycle=0;cycle<20;cycle++){
 await click(8);await ready([9]);await click(8);await ready([9,8]);
 if([0,4,9,19].includes(cycle)){await settleCamera();state=await snapshot();report.churn.push({cycle:cycle+1,heap:await heap(),state});console.log(JSON.stringify({cycle:cycle+1,diagnostics:state.diagnostics,cache:state.cache}));save();}
}
report.heapAfterChurn=await heap();state=await snapshot();const churnRequests=events.slice(churnStart).filter(x=>x.method==='Network.requestWillBeSent');
check('20 warm remove/re-add cycles retain one renderer and use zero HTTP',state.sameFrame&&state.sameApi&&state.sameCanvas&&churnRequests.length===0,{state,requests:churnRequests.length});
check('churn has bounded cache, one scene, and no retained GPU-resource growth',state.cache?.entries<=state.cache?.limitEntries&&state.cache?.bytes<=state.cache?.limitBytes&&state.cache?.sceneBuilds===1&&state.diagnostics?.activeActors===2&&JSON.stringify(state.diagnostics?.gpu)===JSON.stringify(report.beforeChurn.diagnostics?.gpu),{before:report.beforeChurn,after:state,heapBefore:report.heapBeforeChurn,heapAfter:report.heapAfterChurn});
const exceptions=events.filter(x=>x.method==='Runtime.exceptionThrown').map(x=>x.params.exceptionDetails);check('no uncaught browser exceptions',exceptions.length===0,exceptions);
report.finished=new Date().toISOString();save();await send('Network.setBlockedURLs',{urls:[]});ws.close();if(report.checks.some(x=>!x.pass))process.exitCode=1;
