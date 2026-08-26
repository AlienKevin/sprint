(() => {
  let observedVersion = null;
  const $ = selector => document.querySelector(selector);
  const MODEL = {
    deepseek: {label:'DeepSeek V4 Flash Vision Exp', color:'#7C54CD', cls:'deepseek'},
    'flash-baidu': {label:'DeepSeek V4 Flash 0731 · Baidu', color:'#7C54CD', cls:'flash-baidu'},
    'pro-alibaba': {label:'DeepSeek V4 Pro 0813 · Alibaba', color:'#ff9f43', cls:'pro-alibaba'},
    luna: {label:'GPT‑5.6 Luna', color:'#66D693', cls:'luna'},
    sol: {label:'GPT‑5.6 Sol', color:'#2279DC', cls:'sol'}
  };
  const DISPLAY_FAMILIES = ['deepseek','flash-baidu','pro-alibaba','luna','sol'];
  const state = {runs:[], performance:null, batch:null, timelineUpdatedAt:null, refreshing:false};
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const groupBy = (rows,key) => rows.reduce((out,row)=>{const value=key(row);(out[value]??=[]).push(row);return out},{});
  const family = model => {const value=String(model||'').toLowerCase();return value.includes('deepseek-v4-pro-0813')?'pro-alibaba':value.includes('deepseek-v4-flash-0731')?'flash-baidu':value.includes('deepseek')?'deepseek':value.includes('luna')?'luna':value.includes('gpt-5.6-sol')?'sol':null};
  const failureLabel = name => ({finished:'did not finish',in_lane:'left the lane',lane:'left the lane',self_collision:'self-collision','self-collision':'self-collision'})[name]||String(name||'invalid result').replaceAll('_',' ');
  const fmtMoney = value => finite(value) ? `$${value.toFixed(value < 10 ? 2 : 0)}` : '—';
  const fmtTime = ms => finite(ms) ? (ms/3600000).toFixed(1)+' h' : '—';
  const fmtScore = value => finite(value) ? (value === 0 ? '0' : value.toFixed(value < .01 ? 4 : 3)) : '—';
  const fmtTokens = value => {const n=Number(value)||0;if(n>=1e9)return `${(n/1e9).toFixed(2)}B`;if(n>=1e6)return `${(n/1e6).toFixed(n>=1e8?0:1)}M`;if(n>=1e3)return `${(n/1e3).toFixed(n>=1e5?0:1)}K`;return String(n)};
  const fmtDuration = ms => {if(!finite(ms)||ms<0)return '—';const total=Math.floor(ms/1000),hours=Math.floor(total/3600),minutes=Math.floor(total%3600/60),seconds=total%60;return hours?`${hours}h ${String(minutes).padStart(2,'0')}m`:`${minutes}m ${String(seconds).padStart(2,'0')}s`};
  const elapsedSince = value => {const time=Date.parse(value||'');if(!finite(time))return 'waiting';const seconds=Math.max(0,Math.floor((Date.now()-time)/1000));if(seconds<60)return `${seconds}s ago`;if(seconds<3600)return `${Math.floor(seconds/60)}m ago`;return `${Math.floor(seconds/3600)}h ago`};
  const esc = value => String(value??'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  async function json(path){const separator=path.includes('?')?'&':'?';const response=await fetch(`${path}${separator}_=${Date.now()}`,{cache:'no-store'});if(!response.ok)throw Error(`${path}: ${response.status}`);return response.json()}
  function svg(tag,attrs={}){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);return node}
  function continuousChart(target, runs, xKey, xLabel, aucCap=null){
    const el=$(target),width=1000,height=500,p={l:78,r:24,t:28,b:58},selections=[];el.replaceChildren();el.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const plottable=point=>finite(point[xKey])&&finite(point.continuous_score_mps);
    const points=runs.flatMap(run=>(run.points||[]).filter(plottable).map(point=>({...point,model:run.model,color:MODEL[family(run.model)]?.color})));
    const xmax=finite(aucCap)?aucCap:Math.max(1,...points.map(point=>point[xKey])),rawMax=Math.max(.001,...points.map(point=>point.continuous_score_mps)),ymax=Math.ceil(rawMax*10)/10;
    const x=value=>p.l+Math.max(0,Math.min(xmax,value))/xmax*(width-p.l-p.r),y=value=>p.t+(ymax-Math.max(0,Math.min(ymax,value)))/ymax*(height-p.b-p.t);
    for(let i=0;i<=5;i++){const value=i*ymax/5,yy=y(value);el.append(svg('line',{x1:p.l,x2:width-p.r,y1:yy,y2:yy,class:'gridline'}));const label=svg('text',{x:p.l-10,y:yy+4,'text-anchor':'end',class:'chart-label'});label.textContent=fmtScore(value);el.append(label)}
    for(let i=0;i<=5;i++){const xx=p.l+i*(width-p.l-p.r)/5,value=i*xmax/5;const label=svg('text',{x:xx,y:height-22,'text-anchor':'middle',class:'chart-label'});label.textContent=xKey==='cumulative_agent_cost_usd'?fmtMoney(value):`${value.toFixed(1)}h`;el.append(label)}
    const xTitle=svg('text',{x:width/2,y:height-3,'text-anchor':'middle',class:'chart-label'});xTitle.textContent=xLabel;el.append(xTitle);
    const yTitle=svg('text',{x:15,y:height/2,'text-anchor':'middle',class:'chart-label',transform:`rotate(-90 15 ${height/2})`});yTitle.textContent='effective speed (m/s) · higher is better';el.append(yTitle);
    for(const run of runs){
      const rows=(run.points||[]).filter(plottable).sort((a,b)=>a[xKey]-b[xKey]),eligible=rows.filter(row=>!finite(aucCap)||row[xKey]<=aucCap);let best=0,path=`M${x(0)},${y(0)} `;const frontier=new Set();
      for(const row of eligible){path+=`L${x(row[xKey])},${y(best)} `;if(row.continuous_score_mps>best){best=row.continuous_score_mps;frontier.add(row.policy_sha256);path+=`L${x(row[xKey])},${y(best)} `}}
      if(path&&finite(aucCap)&&(!eligible.length||eligible.at(-1)[xKey]<aucCap))path+=`L${x(aucCap)},${y(best)} `;
      const color=MODEL[family(run.model)]?.color;if(path)el.append(svg('path',{d:path,class:'best-line',stroke:color}));
      for(const row of rows){const onFrontier=frontier.has(row.policy_sha256),replayable=Boolean(row.replay_url),label=`Open details for ${MODEL[family(run.model)]?.label}, trial ${row.source_trial}, submission ${row.submission_index}`,cx=x(row[xKey]),cy=y(row.continuous_score_mps),target=svg('g',{class:'submission-target'}),dot=svg('circle',{cx,cy,r:onFrontier?9.5:8,fill:color,class:`submission${onFrontier?' frontier':''}${replayable?' replayable':''}`,tabindex:'0',role:'button','aria-label':label}),hit=svg('circle',{cx,cy,r:18,class:'submission-hit',role:'button','aria-label':label}),title=svg('title');const dq=row.first_disqualification_gate?` · first DQ: ${failureLabel(row.first_disqualification_gate)}`:'',capNote=finite(aucCap)&&row[xKey]>aucCap?` · plotted at ${fmtMoney(aucCap)} cap`:'';title.textContent=`${MODEL[family(run.model)]?.label} · trial ${row.source_trial} · submission ${row.submission_index} · effective speed ${fmtScore(row.continuous_score_mps)} m/s · ${row.max_legal_distance_m.toFixed(3)}m legal in ${row.time_to_max_legal_distance_s.toFixed(2)}s${dq} · trial cost ${fmtMoney(row.cumulative_agent_cost_usd)}${capNote} · ${row.hours_since_agent_launch.toFixed(2)}h`;dot.addEventListener('keydown',event=>{if(!['Enter',' '].includes(event.key))return;event.preventDefault();showReadout(row,run.model)});selections.push({cx,cy,row,model:run.model});target.append(dot,hit,title);el.append(target)}
    }
    el.onclick=event=>{const matrix=el.getScreenCTM();if(!matrix)return;const cursor=el.createSVGPoint();cursor.x=event.clientX;cursor.y=event.clientY;const local=cursor.matrixTransform(matrix.inverse()),nearest=selections.map(item=>({...item,distance:(item.cx-local.x)**2+(item.cy-local.y)**2})).sort((a,b)=>a.distance-b.distance)[0];if(nearest&&nearest.distance<=22**2)showReadout(nearest.row,nearest.model)};
    if(!points.length){const label=svg('text',{x:width/2,y:height/2,'text-anchor':'middle',class:'chart-label'});label.textContent='No reconstructed readouts available';el.append(label)}
  }
  function renderPerformanceScores(target,dimension){
    const data=state.performance;if(!data){$(target).innerHTML='';return}
    const isCost=dimension==='cost',cap=isCost?fmtMoney(data.cost.common_auc_cap_usd):`${data.time.common_auc_cap_hours.toFixed(2)}h`,key=isCost?'cost_auc_mps_at_common_cap':'time_auc_mps_at_common_cap';
    const models=data.models||[],max=Math.max(.000001,...models.map(model=>Number(model.summary?.[key])||0));
    const rows=models.map(model=>{const s=model.summary||{},missing=s.missing_pose_capture_count||0,f=family(model.model),count=isCost?s.cost_readout_count_at_cap:s.time_readout_count_at_cap,value=Number(s[key])||0,width=100*value/max;return `<div class="auc-bar-row ${f}"><span class="auc-bar-label"><strong>${MODEL[f].label}</strong><small>${count} readouts${missing?` · ${missing} captures missing`:''}</small></span><div class="auc-bar-track" role="img" aria-label="${MODEL[f].label}: ${dimension} efficiency score ${fmtScore(value)} at ${cap}"><i style="width:${width}%;background:${MODEL[f].color}"></i></div><b>${fmtScore(value)}</b></div>`}).join('');
    const name=isCost?'Cost-Adjusted Effective Speed':'Time-Adjusted Effective Speed';
    const live=data.snapshot_status==='active_provisional'?' · live provisional':'';
    $(target).innerHTML=`<div class="auc-bar-chart"><div class="auc-bar-head"><strong>${name}</strong><span>normalized AUC at ${cap}${live} · higher is better</span></div>${rows}</div>`;
  }
  function renderCards(){const groups=groupBy(state.runs,r=>family(r.model)),performance=Object.fromEntries((state.performance?.models||[]).map(model=>[family(model.model),model])),keys=DISPLAY_FAMILIES.filter(key=>(groups[key]||[]).length),score=key=>Math.max(0,...(performance[key]?.points||[]).map(point=>Number(point.continuous_score_mps)||0)),max=Math.max(.000001,...keys.map(score));$('#model-cards').innerHTML=`<div class="hero-score-head"><strong>Effective Speed</strong></div>${keys.map(key=>{const value=score(key),width=100*value/max;return `<div class="hero-score-row ${key}"><span><strong>${MODEL[key].label}</strong></span><div class="hero-score-track" role="img" aria-label="${MODEL[key].label}: highest Effective Speed ${fmtScore(value)} metres per second across all trials"><i style="width:${width}%;background:${MODEL[key].color}"></i></div><b>${fmtScore(value)} m/s</b></div>`}).join('')}`}
  function showReadout(point,model){
    const detail=$('#readout-detail'),f=family(model),dq=point.first_disqualification_gate?failureLabel(point.first_disqualification_gate):point.valid_run?'finished legally':'none';
    $('#readout-title').textContent=`${MODEL[f].label} · trial ${point.source_trial} · policy ${point.submission_index}`;
    $('#readout-subtitle').textContent=point.valid_run?'Officially valid finish':'Officially disqualified or unfinished';
    $('#readout-stats').innerHTML=`<div><span>Effective Speed</span><b>${fmtScore(point.continuous_score_mps)} m/s</b></div><div><span>Legal distance</span><b>${Number(point.max_legal_distance_m).toFixed(3)} m</b></div><div><span>Time to that point</span><b>${Number(point.time_to_max_legal_distance_s).toFixed(2)} s</b></div><div><span>First stop reason</span><b>${esc(dq)}</b></div><div><span>Trial cost</span><b>${fmtMoney(point.cumulative_agent_cost_usd)}</b></div><div><span>Elapsed race time</span><b>${Number(point.hours_since_agent_launch).toFixed(2)} h</b></div>`;
    const replay=$('#readout-replay');if(point.replay_url){replay.removeAttribute('srcdoc');replay.src=point.replay_url}else{replay.removeAttribute('src');replay.srcdoc='<!doctype html><html><body style="margin:0;display:grid;place-items:center;height:100vh;background:#090909;color:#aaa;font:16px monospace;text-align:center"><p>This policy has no archived website replay.<br>Its verifier statistics are shown above.</p></body></html>'}
    $('#readout-timeline').src=`/timeline?run=${encodeURIComponent(point.source_run_id)}&embed=1`;
    detail.hidden=false;detail.scrollIntoView({behavior:'smooth',block:'start'});
  }
  function trialNumber(run){const match=String(run.run_id||'').match(/-(\d+)$/);return match?Number(match[1]):null}
  function modalRoleCost(run,role){
    const resources=run.timeline?.resource_usage_summary||{},billing=resources.modal_provider_billing||{};
    const billed=Number(billing.by_role_usd?.[role]);
    if(billing.provider_complete===true&&finite(billed))return {value:billed,basis:'exact Modal pre-credit billing'};
    const estimated=Number(resources.modal_estimate?.by_role?.[role]?.estimated_cost_usd);
    return {value:finite(estimated)?estimated:0,basis:'Modal tariff estimate'};
  }
  function renderResources(){
    const order=['deepseek','flash-baidu','pro-alibaba','luna','sol'],rows=state.runs.filter(run=>order.includes(family(run.model))).map(run=>{
      const api=Number(run.timeline?.comparison_summary?.final_api_cost_usd),cpu=modalRoleCost(run,'cpu_agent'),training=modalRoleCost(run,'training_gpu');
      const apiBasis=(run.timeline?.usage_summary?.calculated_api_usage_cost_basis||[]).includes('openrouter_reported_cost')?'OpenRouter reported request cost':'reconstructed at published list price';
      const parts=[{key:'api',label:'Model API',value:finite(api)?api:0,basis:apiBasis},{key:'cpu',label:'CPU agent',...cpu},{key:'training',label:'Training sandbox',...training}];
      return {run,family:family(run.model),trial:trialNumber(run),effort:run.reasoning_effort||'',parts,total:parts.reduce((sum,part)=>sum+part.value,0)};
    });
    if(!rows.length){$('#resource-bars').innerHTML='<p class="empty">Costs appear as trials start.</p>';return}
    const budget=10,effortOrder={medium:0,high:1,max:2},grouped=groupBy(rows,row=>row.family);
    const heat=value=>`${Math.round(Math.max(0,Math.min(1,value/budget))*44)}%`;
    const costCell=part=>`<td class="budget-heat budget-${part.key}" style="--heat:${heat(part.value)}" title="${esc(part.label)}: ${part.value.toFixed(2)} USD · ${esc(part.basis)}"><strong>${fmtMoney(part.value)}</strong><small>${Math.round(100*part.value/budget)}%</small></td>`;
    const bodies=order.filter(key=>grouped[key]?.length).map(key=>{
      const familyRows=grouped[key].sort((a,b)=>(effortOrder[a.effort]??99)-(effortOrder[b.effort]??99)||(a.trial||0)-(b.trial||0));
      return `<tbody class="budget-group ${esc(key)}">${familyRows.map((row,index)=>{
        const model=MODEL[key]?.label||key,trial=row.trial??index+1,parts=Object.fromEntries(row.parts.map(part=>[part.key,part])),unspent=Math.max(0,budget-row.total),href=`/trajectory?run=${encodeURIComponent(row.run.run_id)}`;
        return `<tr class="budget-row ${esc(key)}">${index===0?`<th class="budget-model" scope="rowgroup" rowspan="${familyRows.length}"><span class="trial-dot"></span><strong>${esc(model)}</strong></th>`:''}<td>${esc(row.effort||'—')}</td><td><a href="${href}" aria-label="Open trial ${esc(trial)} trace">${esc(trial)}</a></td>${costCell(parts.api)}${costCell(parts.cpu)}${costCell(parts.training)}${costCell({key:'unspent',label:'Unspent budget',value:unspent,basis:'$10 trial budget minus recorded agent-side cost'})}</tr>`;
      }).join('')}</tbody>`;
    }).join('');
    $('#resource-bars').innerHTML=`<div class="budget-table-wrap"><table class="budget-table"><caption>Cell shading and percentages use the shared $10 trial budget. Modal charges are pre-credit; verifier infrastructure is excluded.</caption><colgroup><col class="budget-col-model"><col class="budget-col-effort"><col class="budget-col-trial"><col span="4" class="budget-col-cost"></colgroup><thead><tr><th scope="col">Model</th><th scope="col">Effort</th><th scope="col">Trial</th><th scope="col">Model API</th><th scope="col">CPU agent</th><th scope="col">Training</th><th scope="col">Unspent</th></tr></thead>${bodies}</table></div>`;
  }
  function snapshotUpdatedAt(batch=state.batch){const candidates=[batch?.updated_at,state.timelineUpdatedAt,state.performance?.generated_at].map(value=>Date.parse(value||'')).filter(finite);return candidates.length?new Date(Math.max(...candidates)).toISOString():null}
  function batchIsLive(batch=state.batch){const updated=Date.parse(snapshotUpdatedAt(batch)||''),declared=['running','stopping','finalizing_site'].includes(batch?.status);return declared&&finite(updated)&&Date.now()-updated<30*60*1000}
  function experimentPhase(arm,run){const ledger=arm.ledger||{},submitted=Math.max(Number(ledger.submitted)||0,Number(run?.policy_count)||0),rendered=Number(run?.replay_count)||0;if(arm.status==='invalid_infrastructure'||arm.benchmark_valid===false)return 'Invalid · rerun required';if(arm.status==='finalized')return 'Complete';if(arm.status==='stopping'||arm.stop_requested_at)return 'Stopping';if((Number(ledger.running)||0)>0)return 'Scoring policy';if((Number(ledger.queued)||0)>0)return 'Policy queued';if(submitted>rendered&&batchIsLive())return 'Rendering replay';if(arm.harbor_alive===true)return submitted?'Agent iterating':'Agent exploring';if(arm.harbor_alive===false)return 'Agent stopped';if(arm.status==='planned')return 'Waiting to launch';if(arm.status==='launch_error'||arm.status==='missing_run_state')return 'Needs attention';return arm.status||'Waiting for telemetry'}
  function experimentElapsed(arm,run){const start=Date.parse(arm.launched_at||run?.created_at||'');if(!finite(start))return null;const terminal=['finalized','invalid_infrastructure'].includes(arm.status)||['complete','complete_with_invalid_trials'].includes(state.batch?.status)||!batchIsLive();const declaredEnd=Date.parse(arm.finalized_at||'');const timelineEnd=Number(run?.timeline?.clock?.end_epoch_ms);const snapshotEnd=Date.parse(snapshotUpdatedAt()||'');const end=terminal?(finite(declaredEnd)?declaredEnd:finite(timelineEnd)?timelineEnd:finite(snapshotEnd)?snapshotEnd:Date.now()):Date.now();return Math.max(0,end-start)}
  function renderExperimentTracker(){
    const target=$('#experiment-tracker'),batch=state.batch;
    if(!target)return;
    if(!batch){target.innerHTML='<p class="empty">No experiment batch has been published yet.</p>';return}
    const byId=Object.fromEntries(state.runs.map(run=>[run.run_id,run]));
    const performanceById=Object.fromEntries((state.performance?.runs||[]).map(run=>[run.run_id,run]));
    const rows=(batch.arms||[]).map(arm=>{
      const run=byId[arm.run_id]||{},usage=run.timeline?.usage_summary||{},ledger=arm.ledger||{},performance=performanceById[arm.run_id]||{};
      const submitted=Math.max(Number(ledger.submitted)||0,Number(run.policy_count)||0),rendered=Number(run.replay_count)||0,totalCost=Number(run.timeline?.comparison_summary?.final_agent_total_cost_usd);
      const bestScore=Math.max(0,Number(performance.summary?.best_continuous_score_mps)||0,...(performance.points||[]).map(point=>Number(point.continuous_score_mps)||0));
      return {arm,run,usage,submitted,rendered,totalCost,bestScore,elapsed:experimentElapsed(arm,run),phase:experimentPhase(arm,run),family:family(arm.model)||arm.family||'deepseek',effort:arm.reasoning_effort||run.reasoning_effort||state.batch.reasoning_effort||''};
    });
    const grouped=groupBy(rows,row=>row.family),familyOrder=[...DISPLAY_FAMILIES,...Object.keys(grouped).filter(key=>!DISPLAY_FAMILIES.includes(key))];
    const body=familyOrder.filter(key=>grouped[key]?.length).map(key=>{
      const effortOrder={medium:0,high:1,max:2};
      const familyRows=grouped[key].sort((a,b)=>(effortOrder[a.effort]??99)-(effortOrder[b.effort]??99)||(Number(a.arm.trial)||0)-(Number(b.arm.trial)||0));
      const familyBest=Math.max(-Infinity,...familyRows.map(row=>row.bestScore));
      return familyRows.map((row,index)=>{
        const label=MODEL[key]?.label||row.arm.model,href=`/trajectory?run=${encodeURIComponent(row.arm.run_id)}`,isBest=row.bestScore===familyBest,bestLabel=isBest?', highest Effective Speed for this model across all efforts':'',speed=`${fmtScore(row.bestScore)} m/s`;
        return `<tr class="experiment-row ${esc(key)}${isBest?' experiment-best':''}" data-family="${esc(key)}" data-run-href="${href}" tabindex="0" aria-label="Open ${esc(label)} ${esc(row.effort)} trial ${esc(row.arm.trial)} trace${bestLabel}">${index===0?`<th class="experiment-model" data-family="${esc(key)}" scope="rowgroup" rowspan="${familyRows.length}"><span class="trial-dot"></span><strong>${esc(label)}</strong></th>`:''}<td>${isBest?`<strong>${esc(row.effort||'—')}</strong>`:esc(row.effort||'—')}</td><td><a href="${href}" aria-label="Open trial ${esc(row.arm.trial)} trace">${esc(row.arm.trial)}</a></td><td>${isBest?`<b>${speed}</b>`:speed}</td><td data-experiment-elapsed="${esc(row.arm.run_id)}">${fmtDuration(row.elapsed)}</td><td>${fmtMoney(row.totalCost)}</td><td>${row.submitted}</td></tr>`;
      }).join('');
    }).join('');
    target.innerHTML=`<div class="experiment-table-wrap"><table class="experiment-table"><thead><tr><th scope="col">Model</th><th scope="col">Effort</th><th scope="col">Trial</th><th scope="col">Effective Speed</th><th scope="col">Elapsed</th><th scope="col">Total cost</th><th scope="col">Policies submitted</th></tr></thead><tbody>${body||'<tr><td colspan="7" class="empty">Trials are waiting to launch.</td></tr>'}</tbody></table></div>`;
    for(const row of target.querySelectorAll('tr[data-run-href]')){
      const open=()=>{window.location.href=row.dataset.runHref};
      row.addEventListener('click',event=>{if(!event.target.closest('a'))open()});
      row.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();open()}});
    }
    const clearModelHover=()=>target.querySelectorAll('.experiment-model-hover').forEach(node=>node.classList.remove('experiment-model-hover'));
    target.onpointerover=event=>{
      clearModelHover();
      const row=event.target.closest('tr[data-run-href]');
      if(!row||event.target.closest('.experiment-model'))return;
      target.querySelector(`.experiment-model[data-family="${row.dataset.family}"]`)?.classList.add('experiment-model-hover');
    };
    target.onpointerleave=clearModelHover;
  }
  function updateExperimentClocks(){const target=$('#experiment-tracker'),batch=state.batch;if(!target||!batch)return;const arms=Object.fromEntries((batch.arms||[]).map(arm=>[arm.run_id,arm])),runs=Object.fromEntries(state.runs.map(run=>[run.run_id,run]));for(const node of target.querySelectorAll('[data-experiment-elapsed]')){const runId=node.dataset.experimentElapsed,arm=arms[runId];if(!arm)continue;node.textContent=fmtDuration(experimentElapsed(arm,runs[runId]||{}))}}
  function render(){renderExperimentTracker();renderCards();const performanceModels=state.performance?.models||[];continuousChart('#cost-chart',performanceModels,'cumulative_agent_cost_usd','cost within each independent trial (API + CPU + training; verifier excluded)',state.performance?.cost?.common_auc_cap_usd);continuousChart('#time-chart',performanceModels,'hours_since_agent_launch','hours since agent launch',state.performance?.time?.common_auc_cap_hours);const legend=performanceModels.map(model=>{const f=family(model.model);return `<span><i style="background:${MODEL[f].color}"></i>${MODEL[f].label}</span>`}).join('');$('#cost-legend').innerHTML=legend;$('#time-legend').innerHTML=legend;renderPerformanceScores('#time-scores','time');renderResources();const updated=snapshotUpdatedAt();$('#updated').textContent=updated?`Updated ${new Date(updated).toLocaleString()}`:'No race data deployed yet'}
  async function loadSnapshot(){const [pIndex,tIndex,performance,batch]=await Promise.all([json('/data/policies/index.json').catch(()=>({runs:[]})),json('/data/timelines/index.json').catch(()=>({runs:[]})),json('/data/performance/current.json').catch(()=>null),json('/data/batches/current.json').catch(()=>null)]);const batchIds=new Set(batch?.arms?.map(arm=>arm.run_id)||[]),performanceIds=new Set(performance?.runs?.map(run=>run.run_id)||[]),activeIds=batchIds.size?batchIds:performanceIds,selected=tIndex.runs.filter(row=>activeIds.size?activeIds.has(row.run_id):family(row.model)).sort((a,b)=>`${b.created_at||''}:${b.run_id||''}`.localeCompare(`${a.created_at||''}:${a.run_id||''}`));state.performance=performance;state.timelineUpdatedAt=tIndex.updated_at||null;state.runs=selected.map(tMeta=>{const pMeta=pIndex.runs.find(row=>row.run_id===tMeta.run_id),timeline={coverage:{ready:Boolean(tMeta.ready)},usage_summary:tMeta.usage_summary||{},comparison_summary:tMeta.comparison_summary||{},resource_usage_summary:tMeta.resource_usage_summary||{},clock:{origin_epoch_ms:tMeta.origin_epoch_ms,end_epoch_ms:tMeta.end_epoch_ms},artifacts:tMeta.dashboard_artifacts||[]};return {...tMeta,...pMeta,timeline}});state.batch=batch||{batch_id:'latest-published-runs',status:performance?.snapshot_status==='active_provisional'?'running':'complete',updated_at:tIndex.updated_at,arms:state.runs.map(run=>({run_id:run.run_id,model:run.model,family:family(run.model),trial:trialNumber(run),status:'finalized',launched_at:run.created_at,finalized_at:run.updated_at,ledger:{submitted:run.policy_count||run.timeline.comparison_summary.submission_count||0,scored:run.timeline.comparison_summary.submission_count||0}}))};render()}
  async function init(){try{await loadSnapshot()}catch(error){$('#updated').textContent=`Data error: ${error.message}`;console.error(error)}}
  async function refresh(){if(state.refreshing||document.hidden)return;state.refreshing=true;try{await loadSnapshot()}catch(error){console.warn('Race refresh failed',error)}finally{state.refreshing=false}}
  async function refreshVersion(){try{const deployed=await json('/version.json');if(!deployed.version)return;if(observedVersion===null){observedVersion=deployed.version;return}if(deployed.version!==observedVersion)window.location.reload()}catch(error){console.warn('Dashboard version check failed',error)}}
  $('#readout-close').addEventListener('click',()=>{const detail=$('#readout-detail');detail.hidden=true;$('#readout-replay').removeAttribute('src');$('#readout-timeline').removeAttribute('src')});init();refreshVersion();setInterval(refresh,30000);setInterval(refreshVersion,30000);setInterval(updateExperimentClocks,1000);document.addEventListener('visibilitychange',()=>{if(!document.hidden){refresh();refreshVersion()}});window.addEventListener('focus',()=>{refresh();refreshVersion()});window.addEventListener('pageshow',()=>{refresh();refreshVersion()});
})();
