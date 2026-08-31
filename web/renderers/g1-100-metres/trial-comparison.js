const TRIAL_STORAGE='g1:trial-comparison-state/v1';
// The document nonce stays fixed; selection generations may advance only via
// this document's same-origin parent. Async replies retain their own version.
const DOCUMENT_GENERATION=new URLSearchParams(location.search).get('replayDocumentGeneration')||new URLSearchParams(location.search).get('replayGeneration')||'0';
let REPLAY_GENERATION=new URLSearchParams(location.search).get('replayGeneration')||'0';
let activeState=null,rendererReady=false,requestSerial=0,pendingKey=null,disposed=false,sceneBuilds=0,sceneLayout=null,applyingSelection=false;
const lanesEl=document.querySelector('.lanes'),noteEl=document.getElementById('failure-note');
let noteTranslation=null;
function replayMessage(key,params={},fallback=key){return window.ReplayI18n?.t(key,params)||fallback.replace(/\{(\w+)\}/g,(match,name)=>params[name]??match);}
function showLocalized(key,params={},fallback=key,kind='incomplete'){const message=replayMessage(key,params,fallback);show(message,kind);noteTranslation={key,params,fallback,kind};return message;}
function localizedError(key,params,fallback){return Object.assign(new Error(replayMessage(key,params,fallback)),{i18nKey:key,i18nParams:params,i18nFallback:fallback});}
function showError(error){if(error.i18nKey)showLocalized(error.i18nKey,error.i18nParams,error.i18nFallback);else show(error.message);}
function refreshComparisonLanguage(){if(!disposed&&noteTranslation&&!noteEl.hidden){const {key,params,fallback,kind}=noteTranslation;showLocalized(key,params,fallback,kind);}}
lanesEl.innerHTML='';
function tell(type,detail={},generation=REPLAY_GENERATION){parent.postMessage({type,...detail,replayGeneration:generation,replayDocumentGeneration:DOCUMENT_GENERATION},location.origin);}
function remember(state){try{sessionStorage.setItem(TRIAL_STORAGE,JSON.stringify(state));}catch{}}
function show(message,kind='incomplete'){noteTranslation=null;if(!noteEl)return;noteEl.hidden=false;noteEl.dataset.kind=kind;noteEl.textContent=message;}
function captureId(value){const id=String(value||'');return /^frontier-[a-f0-9]{12}$/.test(id)?id:null;}
function colorHex(value){
  const raw=String(value||'').trim();
  let match=raw.match(/^#([a-f0-9]{3}|[a-f0-9]{6}|[a-f0-9]{8})$/i);
  if(match){let hex=match[1];if(hex.length===3)hex=[...hex].map(x=>x+x).join('');return `#${hex.slice(0,6).toUpperCase()}`;}
  match=raw.match(/^rgba?[(][ ]*([0-9.]+)[ ]*,[ ]*([0-9.]+)[ ]*,[ ]*([0-9.]+)/i);
  if(!match)return null;
  const channel=value=>Math.max(0,Math.min(255,Math.round(Number(value)||0))).toString(16).padStart(2,'0');
  return `#${channel(match[1])}${channel(match[2])}${channel(match[3])}`.toUpperCase();
}
function normalize(raw){
  const seen=new Set(),policies=[];
  for(const item of raw?.policies||[]){
    const id=captureId(item?.captureId);if(!id||seen.has(id))continue;seen.add(id);
    const fallback=TRIAL_BOOT.registry[id]||{};
    const policyNumber=Number(item.policyNumber??fallback.policyNumber);
    if(!Number.isInteger(policyNumber)||policyNumber<1)continue;
    let url=String(item.url||fallback.url||`/captures/${id}.json`);
    try{const parsed=new URL(url,location.href);if(parsed.origin!==location.origin)continue;url=parsed.pathname;}catch{continue;}
    // Registered captures always use the canonical model identity accent. The
    // parent supplies a computed CSS rgb() value, so normalize that too for
    // unregistered/development captures instead of feeding NaN to Three.js.
    const color=colorHex(fallback.color)||colorHex(item.color)||'#6E97C4';
    policies.push({...fallback,...item,captureId:id,url,policyNumber,label:String(item.label||fallback.label||`Policy #${policyNumber}`),color});
  }
  if(policies.length>8)throw localizedError('maxPolicies',{},'At most 8 policies can be compared.');
  const emphasizedCaptureId=captureId(raw?.emphasizedCaptureId);
  return {policies,emphasizedCaptureId:policies.some(x=>x.captureId===emphasizedCaptureId)?emphasizedCaptureId:(policies.at(-1)?.captureId||null)};
}
function queryIds(){return (new URLSearchParams(location.search).get('policies')||'').split(',').map(captureId).filter(Boolean);}
function initialState(){
  const ids=queryIds();let stored=null;try{stored=JSON.parse(sessionStorage.getItem(TRIAL_STORAGE)||'null');}catch{}
  const storedById=new Map((stored?.policies||[]).map(x=>[x.captureId,x]));
  const missing=ids.filter(id=>!storedById.has(id)&&!TRIAL_BOOT.registry[id]);
  if(missing.length){const error=localizedError(missing.length===1?'unknownOne':'unknownMany',{ids:missing.join(', ')},`Unknown policy capture${missing.length===1?'':'s'}: {ids}.`);error.failedCaptureIds=missing;throw error;}
  const policies=ids.map(id=>storedById.get(id)||TRIAL_BOOT.registry[id]);
  const emphasis=new URLSearchParams(location.search).get('emphasis');
  return normalize({policies,emphasizedCaptureId:emphasis||stored?.emphasizedCaptureId});
}
function canonicalUrl(state){const url=new URL(location.href);url.searchParams.set('policies',state.policies.map(x=>x.captureId).join(','));if(state.emphasizedCaptureId)url.searchParams.set('emphasis',state.emphasizedCaptureId);else url.searchParams.delete('emphasis');url.searchParams.set('replayGeneration',REPLAY_GENERATION);url.searchParams.set('replayDocumentGeneration',DOCUMENT_GENERATION);return url;}
window.addEventListener('g1:policy-focused',event=>{
  if(applyingSelection)return;
  const id=captureId(event.detail?.captureId);if(!activeState?.policies.some(x=>x.captureId===id))return;
  activeState={...activeState,emphasizedCaptureId:id};
  remember(activeState);history.replaceState(null,'',canonicalUrl(activeState));
  tell('g1:policy-focused',{captureId:id});
});
window.addEventListener('g1:policy-remove',event=>{
  const id=captureId(event.detail?.captureId);if(!activeState?.policies.some(x=>x.captureId===id))return;
  tell('g1:policy-remove',{captureId:id});
  if(parent===window){
    const next=normalize({policies:activeState.policies.filter(x=>x.captureId!==id),emphasizedCaptureId:activeState.emphasizedCaptureId});
    setSelection(next,String(Number(REPLAY_GENERATION)+1));
  }
});
window.addEventListener('message',event=>{
  if(event.origin!==location.origin||event.source!==parent||event.data?.type!=='g1:set-policies'||String(event.data.replayDocumentGeneration??'')!==DOCUMENT_GENERATION)return;
  const generation=String(event.data.replayGeneration??'');
  if(!/^\d+$/.test(generation)||Number(generation)<Number(REPLAY_GENERATION))return;
  try{setSelection(normalize(event.data),generation);
  }catch(error){showError(error);tell('g1:policies-error',{message:error.message,failedCaptureIds:[]});}
});
function terminal(raw,frames,run){
  if(run?.valid===true)return {time:null,reason:null};
  let time=run?.first_disqualification_time_s??run?.dq_time??run?.stop_time_s??run?.duration_s??frames.at(-1)?.[0]??0;
  let reason=run?.first_disqualification_gate??run?.dq_reason??run?.termination_reason;
  if(!reason||reason==='finished')reason='did_not_finish';return {time:Number(time),reason:String(reason)};
}
function packCapture(raw,item,links,names){
  if(Number(raw.schema_version)!==2)throw new Error(`${item.captureId} uses unsupported capture schema.`);
  const frameSet=(raw.frames||[])[Number(raw.representative_lane)||0]||raw.frames?.[0];
  if(!frameSet?.length)throw new Error(`${item.captureId} has no pose frames.`);
  const run=(raw.runs||[])[Number(raw.representative_lane)||0]||raw.runs?.[0]||{};
  const ridx=links.map(name=>names.indexOf(name)),fps=Number(raw.fps)||50,torso=Math.max(0,names.indexOf('torso_link'));
  let finish=null;if(run.finish!=null)finish=Number(run.finish);
  if(finish===null){const o=1+torso*7;for(const row of frameSet)if(row[o]>=100){finish=Number(row[0]);break;}}
  if(finish===null)finish=Number(frameSet.at(-1)[0]);
  const term=terminal(raw,frameSet,run),timedOut=run.valid===false&&['timeout','time_limit'].includes(term.reason);
  const clipT=timedOut?Number(frameSet.at(-1)[0]):((run.valid===false?term.time:finish)+1.5);
  const frames=frameSet.filter(row=>Number(row[0])<=clipT).map(row=>{const packed=[Number(row[0])];for(const body of ridx){const o=1+body*7;for(let j=0;j<7;j++)packed.push(Number(row[o+j]));}return packed;});
  const laneCheck=(run.checks||[]).find(x=>x.name==='in_lane');
  const policy={label:item.label,policy_number:item.policyNumber,lane_number:item.policyNumber,finish,frames,max_lateral_m:Number(laneCheck?.value)||0,valid:run.valid,terminal_time:term.time,terminal_reason:term.reason,timed_out:timedOut,disqualified:['in_lane','self_collision'].includes(term.reason),effective_speed_mps:Number(run.effective_speed_mps)||0,identity:item.identity||null,source_run_id:item.runId,color:item.renderColor,model_color:item.color,emphasized:item.emphasized,capture_id:item.captureId};
  if(run.valid===true){delete policy.terminal_time;delete policy.terminal_reason;}
  return policy;
}
function laneMarkup(policy,index){const color=policy.color,name=replayMessage('policy',{number:policy.policy_number},'Policy #{number}'),escape=value=>String(value).replace(/[&"<>]/g,char=>({'&':'&amp;','"':'&quot;','<':'&lt;','>':'&gt;'}[char])),follow=escape(replayMessage('follow',{name},'Follow {name}')),remove=escape(replayMessage('remove',{name},'Remove {name}'));return `<div class="lc${policy.emphasized?' emphasized':''}" data-capture-id="${policy.capture_id}" style="--emphasis:${color}" id="lane${index}" role="group" aria-label="${escape(name)}"><button type="button" class="policy-follow" aria-pressed="false" aria-label="${follow}"><span class="sw" style="background:${color}"></span><span class="nm">#${policy.policy_number}</span><span class="tm" style="color:${color}">0.00s</span><span class="d">0.0 m</span></button><button type="button" class="policy-remove" aria-label="${remove}" title="${remove}">×</button></div>`;}
function renderLaneCards(policies){
  const previous=new Map([...lanesEl.children].map(card=>[card.dataset.captureId,card]));
  policies.forEach((policy,index)=>{
    let card=previous.get(policy.capture_id);
    if(!card){const template=document.createElement('template');template.innerHTML=laneMarkup(policy,index);card=template.content.firstElementChild;}
    previous.delete(policy.capture_id);card.id='lane'+index;lanesEl.append(card);
  });
  for(const card of previous.values())card.remove();
}

// Packed pose arrays are immutable in this cache; actors receive shallow policy
// copies. The LRU budget bounds cached data, not the separately selected actors
// or their GPU textures. Active entries are reused even if evicted from the LRU.
const CACHE_LIMIT_BYTES=64*1024*1024,CACHE_LIMIT_ENTRIES=12;
const decodedCache=new Map(),pendingCaptures=new Map();
let activeEntries=new Map(),cacheBytes=0,cacheHits=0,fetchCount=0,packCount=0;
const captureKey=item=>`${item.captureId}:${item.url}`;
function cachePut(key,entry){
  const previous=decodedCache.get(key);if(previous){cacheBytes-=previous.bytes;decodedCache.delete(key);}
  if(entry.bytes>CACHE_LIMIT_BYTES)return;
  decodedCache.set(key,entry);cacheBytes+=entry.bytes;
  while(cacheBytes>CACHE_LIMIT_BYTES||decodedCache.size>CACHE_LIMIT_ENTRIES){const oldest=decodedCache.keys().next().value;cacheBytes-=decodedCache.get(oldest).bytes;decodedCache.delete(oldest);}
}
function loadCapture(item){
  const key=captureKey(item),cached=activeEntries.get(key)||decodedCache.get(key);
  if(cached){cacheHits++;cachePut(key,cached);return Promise.resolve(cached);}
  if(pendingCaptures.has(key))return pendingCaptures.get(key).promise;
  const controller=new AbortController(),task={controller,promise:null};
  task.promise=(async()=>{
    fetchCount++;
    const response=await fetch(item.url,{cache:'force-cache',signal:controller.signal});
    if(!response.ok)throw new Error(`${item.captureId}: ${response.status}`);
    const raw=await response.json();
    if(controller.signal.aborted||disposed)throw new Error('Replay load cancelled.');
    const names=raw.body_names,fps=Number(raw.fps)||50;
    if(!Array.isArray(names)||!names.includes('pelvis')||fps<=0||!Number.isFinite(fps))throw new Error(`${item.captureId} has invalid pose metadata.`);
    const links=TRIAL_BOOT.preferred.filter(name=>names.includes(name)&&TRIAL_BOOT.hq[name]);
    const policy=packCapture(raw,item,links,names);packCount++;
    if(!policy.frames.length||policy.frames.some(row=>row.some(value=>!Number.isFinite(value))))throw new Error(`${item.captureId} has invalid pose frames.`);
    const bytes=policy.frames.reduce((total,row)=>total+row.length*8+32,4096);
    const entry={policy,links,fps,layout:JSON.stringify({names,fps,links}),bytes};
    if(!controller.signal.aborted&&!disposed)cachePut(key,entry);
    return entry;
  })().finally(()=>{if(pendingCaptures.get(key)===task)pendingCaptures.delete(key);});
  pendingCaptures.set(key,task);return task.promise;
}
function cancelUnusedLoads(keys){
  for(const [key,task] of pendingCaptures)if(!keys.has(key)){pendingCaptures.delete(key);task.controller.abort();}
}
function acknowledge(state,generation,serial){
  requestAnimationFrame(()=>{
    if(disposed||serial!==requestSerial||generation!==REPLAY_GENERATION)return;
    rendererReady=Boolean(window.__G1_REPLAY__);pendingKey=null;
    tell('g1:policies-state',{...state,pausedAt:window.__G1_REPLAY__?.playback().time||0,playing:window.__G1_REPLAY__?.playback().playing||false,failedCaptureIds:[]},generation);
  });
}
async function setSelection(state,generation){
  if(disposed)return;
  if(window.__G1_REPLAY_ASSET_ERROR__){
    REPLAY_GENERATION=generation;window.__G1_REPLAY_GENERATION__=generation;
    const message=noteEl?.textContent||'Unable to load replay assets. Please reload to retry.';
    show(message);tell('g1:policies-error',{message,failedCaptureIds:[]},generation);return;
  }
  const key=JSON.stringify(state),sameGeneration=generation===REPLAY_GENERATION;
  if(sameGeneration&&pendingKey===key)return;
  REPLAY_GENERATION=generation;window.__G1_REPLAY_GENERATION__=generation;
  const serial=++requestSerial;pendingKey=key;
  remember(state);history.replaceState(null,'',canonicalUrl(state));
  cancelUnusedLoads(new Set(state.policies.map(captureKey)));
  if(!state.policies.length){
    // Parent viewers dispose the frame when removing the final policy. Direct
    // standalone shells retain a paused, hidden scene and can safely start anew.
    window.__G1_REPLAY__?.seek(0);document.getElementById('stage').hidden=true;
    lanesEl.innerHTML='';activeEntries.clear();activeState=state;
    showLocalized('empty',{},'Select up to 8 policies from one trial to compare.');acknowledge(state,generation,serial);return;
  }
  if(rendererReady&&JSON.stringify(activeState?.policies)===JSON.stringify(state.policies)){
    activeState=state;document.getElementById('stage').hidden=false;noteEl.hidden=true;
    const index=state.policies.findIndex(policy=>policy.captureId===state.emphasizedCaptureId);
    if(index>=0&&window.__G1_REPLAY__.camera().follow!==index)window.__G1_REPLAY__.follow(index);
    acknowledge(state,generation,serial);return;
  }
  showLocalized(state.policies.length===1?'loadingOne':'loadingMany',{count:state.policies.length},`Loading {count} polic${state.policies.length===1?'y':'ies'}…`);
  const settled=await Promise.allSettled(state.policies.map(loadCapture));
  if(disposed||serial!==requestSerial||generation!==REPLAY_GENERATION)return;
  const failed=settled.map((result,index)=>result.status==='rejected'?state.policies[index].captureId:null).filter(Boolean);
  if(failed.length){pendingKey=null;const message=showLocalized('unavailable',{ids:failed.join(', ')},'Unable to load {ids}. Remove the unavailable policy or retry.');tell('g1:policies-error',{message,failedCaptureIds:failed},generation);return;}
  const entries=settled.map(result=>result.value),{links,fps,layout}=entries[0];
  try{
    if(entries.some(entry=>entry.layout!==layout)||(sceneLayout&&sceneLayout!==layout))throw localizedError('incompatible',{},'Selected policies have incompatible pose layouts.');
    const emphasis=state.emphasizedCaptureId,items=state.policies.map(item=>({...item,identity:item.identity||TRIAL_BOOT.registry[item.captureId]?.identity||null,runId:item.runId||TRIAL_BOOT.registry[item.captureId]?.runId||null,emphasized:item.captureId===emphasis,renderColor:item.captureId===emphasis?item.color:'#FFFFFF'}));
    const policies=entries.map((entry,index)=>({...entry.policy,label:items[index].label,identity:items[index].identity,source_run_id:items[index].runId,policy_number:items[index].policyNumber,lane_number:items[index].policyNumber,color:items[index].renderColor,model_color:items[index].color,emphasized:items[index].emphasized}));
    renderLaneCards(policies);document.getElementById('stage').hidden=false;noteEl.hidden=true;applyingSelection=true;
    if(window.__G1_REPLAY__){window.__G1_REPLAY__.updatePolicies(policies,{emphasizedCaptureId:emphasis});}
    else{
      const data={fps,links,parents:Object.fromEntries(links.filter(x=>links.includes(TRIAL_BOOT.parents[x])).map(x=>[x,TRIAL_BOOT.parents[x]])),rest:{},hq:Object.fromEntries(links.map(x=>[x,TRIAL_BOOT.hq[x]])),policies,colors:policies.map(x=>parseInt(x.color.slice(1),16)),lane_indices:policies.map((_,i)=>i),lane_labels:Array(8).fill(null),meta:{comparison:true,trajectory_comparison:true,track_lanes:8,lane_half_width_m:.61,start_paused:true,interpolation:'adjacent_authoritative_position_lerp_quaternion_slerp'}};
      new Function('DATA',TRIAL_BOOT.scene)(data);sceneBuilds++;sceneLayout=layout;
    }
    applyingSelection=false;activeState=state;
    activeEntries=new Map(entries.map((entry,index)=>[captureKey(state.policies[index]),entry]));
    acknowledge(state,generation,serial);
  }catch(error){applyingSelection=false;pendingKey=null;showError(error);tell('g1:policies-error',{message:error.message,failedCaptureIds:state.policies.map(x=>x.captureId)},generation);}
}
function disposeComparison(){
  if(disposed)return;disposed=true;requestSerial++;cancelUnusedLoads(new Set());
  decodedCache.clear();activeEntries.clear();cacheBytes=0;window.__G1_REPLAY__?.dispose();
  window.removeEventListener('site:languagechange',refreshComparisonLanguage);
}
window.__G1_COMPARISON_CACHE__={stats(){return {entries:decodedCache.size,bytes:cacheBytes,limitBytes:CACHE_LIMIT_BYTES,limitEntries:CACHE_LIMIT_ENTRIES,inflight:pendingCaptures.size,hits:cacheHits,fetches:fetchCount,packs:packCount,activeEntries:activeEntries.size,activeBytes:[...activeEntries.values()].reduce((total,entry)=>total+entry.bytes,0),sceneBuilds,generation:REPLAY_GENERATION,documentGeneration:DOCUMENT_GENERATION,disposed};}};
window.__G1_TRIAL__={dispose:disposeComparison};
window.addEventListener('pagehide',event=>{if(!event.persisted)disposeComparison();});
async function boot(){
  tell('g1:policies-ready',{maxPolicies:8,path:'/replay/trial-comparison.html'});
  let state;try{state=initialState();}catch(error){showError(error);tell('g1:policies-error',{message:error.message,failedCaptureIds:error.failedCaptureIds||[]});return;}
  await setSelection(state,REPLAY_GENERATION);
}
window.addEventListener('site:languagechange',refreshComparisonLanguage);
boot();
