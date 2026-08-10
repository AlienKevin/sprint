(() => {
  const $ = selector => document.querySelector(selector);
  const MODEL = {
    deepseek: {label:'DeepSeek V4 Flash 0731', color:'#4D6BFF', cls:'deepseek'},
    luna: {label:'GPT‑5.6 Luna', color:'#66D693', cls:'luna'},
    sol: {label:'GPT‑5.6 Sol', color:'#f1c35b', cls:'sol'}
  };
  const DISPLAY_FAMILIES = ['deepseek','luna','sol'];
  const state = {runs:[], policies:[], timelines:new Map(), performance:null, filter:'all'};
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const groupBy = (rows,key) => rows.reduce((out,row)=>{const value=key(row);(out[value]??=[]).push(row);return out},{});
  const family = model => {const value=String(model||'').toLowerCase();return value.includes('deepseek')?'deepseek':value.includes('luna')?'luna':value.includes('gpt-5.6-sol')?'sol':null};
  const failureLabel = name => ({finished:'did not finish',in_lane:'left the lane',lane:'left the lane',self_collision:'self-collision','self-collision':'self-collision'})[name]||String(name||'invalid result').replaceAll('_',' ');
  const fmtMoney = value => finite(value) ? `$${value.toFixed(value < 10 ? 2 : 0)}` : '—';
  const fmtTime = ms => finite(ms) ? (ms/3600000).toFixed(1)+' h' : '—';
  const fmtScore = value => finite(value) ? value.toFixed(value < .01 ? 4 : 3) : '—';
  const esc = value => String(value??'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  async function json(path){const response=await fetch(path,{cache:'no-store'});if(!response.ok)throw Error(`${path}: ${response.status}`);return response.json()}
  function svg(tag,attrs={}){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);return node}
  function continuousChart(target, runs, xKey, xLabel, aucCap=null){
    const el=$(target),width=1000,height=500,p={l:78,r:24,t:28,b:58};el.replaceChildren();el.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const visible=point=>finite(point[xKey])&&finite(point.continuous_score_mps)&&(!finite(aucCap)||point[xKey]<=aucCap);
    const points=runs.flatMap(run=>(run.points||[]).filter(visible).map(point=>({...point,model:run.model,color:MODEL[family(run.model)]?.color})));
    const xmax=finite(aucCap)?aucCap:Math.max(1,...points.map(point=>point[xKey])),rawMax=Math.max(.001,...points.map(point=>point.continuous_score_mps)),ymax=Math.ceil(rawMax*10)/10;
    const x=value=>p.l+Math.max(0,value)/xmax*(width-p.l-p.r),y=value=>p.t+(ymax-Math.max(0,Math.min(ymax,value)))/ymax*(height-p.b-p.t);
    if(finite(aucCap)){const cutoffX=x(aucCap);el.append(svg('line',{x1:cutoffX,x2:cutoffX,y1:p.t,y2:height-p.b,class:'auc-cap-line'}));const cutoff=svg('text',{x:cutoffX-7,y:p.t+14,'text-anchor':'end',class:'chart-label auc-cap-label'});cutoff.textContent=`AUC cutoff ${xKey==='cumulative_agent_cost_usd'?fmtMoney(aucCap):aucCap.toFixed(2)+'h'}`;el.append(cutoff)}
    for(let i=0;i<=5;i++){const value=i*ymax/5,yy=y(value);el.append(svg('line',{x1:p.l,x2:width-p.r,y1:yy,y2:yy,class:'gridline'}));const label=svg('text',{x:p.l-10,y:yy+4,'text-anchor':'end',class:'chart-label'});label.textContent=fmtScore(value);el.append(label)}
    for(let i=0;i<=5;i++){const xx=p.l+i*(width-p.l-p.r)/5,value=i*xmax/5;const label=svg('text',{x:xx,y:height-22,'text-anchor':'middle',class:'chart-label'});label.textContent=xKey==='cumulative_agent_cost_usd'?fmtMoney(value):`${value.toFixed(1)}h`;el.append(label)}
    const xTitle=svg('text',{x:width/2,y:height-3,'text-anchor':'middle',class:'chart-label'});xTitle.textContent=xLabel;el.append(xTitle);
    const yTitle=svg('text',{x:15,y:height/2,'text-anchor':'middle',class:'chart-label',transform:`rotate(-90 15 ${height/2})`});yTitle.textContent='completion-adjusted legal speed (m/s) · higher is better';el.append(yTitle);
    for(const run of runs){
      const rows=(run.points||[]).filter(visible).sort((a,b)=>a[xKey]-b[xKey]);let best=0,path='';
      for(const row of rows){best=Math.max(best,row.continuous_score_mps);path+=`${path?'L':'M'}${x(row[xKey])},${y(best)} `}
      if(path&&finite(aucCap))path+=`L${x(aucCap)},${y(best)} `;
      const color=MODEL[family(run.model)]?.color;if(path)el.append(svg('path',{d:path,class:'best-line',stroke:color}));
      for(const row of rows){const dot=svg('circle',{cx:x(row[xKey]),cy:y(row.continuous_score_mps),r:4.1,fill:color,class:'submission'});const title=svg('title');const dq=row.first_disqualification_gate?` · first DQ: ${failureLabel(row.first_disqualification_gate)}`:'';title.textContent=`${MODEL[family(run.model)]?.label} · trial ${row.source_trial} · submission ${row.submission_index} · score ${fmtScore(row.continuous_score_mps)} m/s · ${row.max_legal_distance_m.toFixed(3)}m legal in ${row.time_to_max_legal_distance_s.toFixed(2)}s${dq} · aggregate model cost ${fmtMoney(row.cumulative_agent_cost_usd)} · ${row.hours_since_agent_launch.toFixed(2)}h`;dot.append(title);el.append(dot)}
    }
    if(!points.length){const label=svg('text',{x:width/2,y:height/2,'text-anchor':'middle',class:'chart-label'});label.textContent='No reconstructed readouts available';el.append(label)}
  }
  function renderPerformanceScores(target,dimension){
    const data=state.performance;if(!data){$(target).innerHTML='';return}
    const isCost=dimension==='cost',cap=isCost?fmtMoney(data.cost.common_auc_cap_usd):`${data.time.common_auc_cap_hours.toFixed(2)}h`,key=isCost?'cost_auc_mps_at_common_cap':'time_auc_mps_at_common_cap';
    const models=data.models||[],max=Math.max(.000001,...models.map(model=>Number(model.summary?.[key])||0));
    const rows=models.map(model=>{const s=model.summary||{},missing=s.missing_pose_capture_count||0,f=family(model.model),count=isCost?s.cost_readout_count_at_cap:s.time_readout_count_at_cap,value=Number(s[key])||0,width=100*value/max;return `<div class="auc-bar-row ${f}"><span class="auc-bar-label"><strong>${MODEL[f].label}</strong><small>best of 3 trials · ${count} readouts${missing?` · ${missing} captures missing`:''}</small></span><div class="auc-bar-track" role="img" aria-label="${MODEL[f].label}: normalized ${dimension} AUC ${fmtScore(value)} at ${cap}"><i style="width:${width}%;background:${MODEL[f].color}"></i></div><b>${fmtScore(value)}</b></div>`}).join('');
    $(target).innerHTML=`<div class="auc-bar-chart"><div class="auc-bar-head"><strong>Normalized ${dimension} AUC at ${cap}</strong><span>higher is better</span></div>${rows}</div>`;
  }
  function renderCards(){const groups=groupBy(state.runs,r=>family(r.model)),performance=Object.fromEntries((state.performance?.models||[]).map(model=>[family(model.model),model]));$('#model-cards').innerHTML=DISPLAY_FAMILIES.filter(key=>(groups[key]||[]).length).map(key=>{const runs=groups[key]||[],best=performance[key]?.summary?.best_continuous_score_mps,complete=runs.filter(r=>r.timeline?.coverage?.ready).length,efforts=[...new Set(runs.map(r=>r.reasoning_effort).filter(Boolean))].join(', ');return `<article class="model-card ${key}"><div><div class="model-name">${MODEL[key].label}</div><div class="model-meta">best of ${runs.length} trial${runs.length===1?'':'s'} · ${complete} finalized<br>best completion-adjusted legal speed${efforts?' · '+esc(efforts):''}</div></div><div class="model-stat">${finite(best)?fmtScore(best)+' m/s':'—'}</div></article>`}).join('')}
  function renderPolicies(){let rows=[...state.policies];if(state.filter==='failures')rows=rows.filter(p=>!p.valid_run);if(state.filter==='valid')rows=rows.filter(p=>p.valid_run);if(state.filter==='story'){const firstFailure=new Map(), selected=[];for(const p of rows){const key=`${p.run_id}:${(p.failed_gates||[]).join(',')||'no-finish'}`;if(!p.valid_run&&!firstFailure.has(key)){firstFailure.set(key,true);selected.push(p)}else if(p.valid_run&&(p.on_frontier||p.replay_ready))selected.push(p)}rows=selected}rows.sort((a,b)=>String(b.finished_at||'').localeCompare(String(a.finished_at||'')));$('#policy-grid').innerHTML=rows.length?rows.slice(0,state.filter==='all'?400:80).map(p=>{const f=family(p.model),result=p.valid_run?`${Number(p.best_100m_s).toFixed(3)} s`:'DQ / DNF',legal=p.max_distance_semantics==='legal_prefix_until_first_disqualification',detail=p.valid_run?`${finite(p.peak_speed_mps)?p.peak_speed_mps.toFixed(2)+' m/s peak':''}${p.on_frontier?' · Pareto frontier':''}`:`${(p.failed_gates||['finished']).map(failureLabel).join(' · ')}${finite(p.max_distance_m)?` · ${p.max_distance_m.toFixed(1)} m${legal?' legal':''}`:''}`;return `<article class="policy-card ${p.valid_run?'valid':'failure'}"><div class="policy-top"><span>${MODEL[f]?.label||esc(p.model)}</span><span>${esc(p.run_id.split('-').slice(-1)[0])} / #${p.submission_index}</span></div><div class="policy-result">${result}</div><div class="policy-detail">${esc(detail)}</div><div class="policy-actions"><span class="tag">website replay archived</span><span class="tag">${p.valid_run?'valid':'failure'}</span></div></article>`}).join(''):'<p class="empty">Waiting for the first scored policy…</p>'}
  function trialNumber(run){const match=String(run.run_id||'').match(/-(\d+)$/);return match?Number(match[1]):null}
  function modalRoleCost(run,role){
    const resources=run.timeline?.resource_usage_summary||{},billing=resources.modal_provider_billing||{};
    const billed=Number(billing.by_role_usd?.[role]);
    if(billing.provider_complete===true&&finite(billed))return {value:billed,basis:'exact Modal pre-credit billing'};
    const estimated=Number(resources.modal_estimate?.by_role?.[role]?.estimated_cost_usd);
    return {value:finite(estimated)?estimated:0,basis:'Modal tariff estimate'};
  }
  function renderResources(){
    const rows=state.runs.filter(run=>['deepseek','luna'].includes(family(run.model))).map(run=>{
      const api=Number(run.timeline?.comparison_summary?.final_api_cost_usd),cpu=modalRoleCost(run,'cpu_agent'),training=modalRoleCost(run,'training_gpu');
      const parts=[{key:'api',label:'Model API',value:finite(api)?api:0,basis:'reconstructed at published list price'},{key:'cpu',label:'CPU agent',...cpu},{key:'training',label:'Training sandbox',...training}];
      return {run,family:family(run.model),trial:trialNumber(run),parts,total:parts.reduce((sum,part)=>sum+part.value,0)};
    }).sort((a,b)=>['deepseek','luna'].indexOf(a.family)-['deepseek','luna'].indexOf(b.family)||(a.trial||0)-(b.trial||0));
    if(!rows.length){$('#resource-bars').innerHTML='<p class="empty">Costs appear as trials start.</p>';return}
    const max=Math.max(.01,...rows.map(row=>row.total));
    const legend='<div class="cost-legend"><span><i class="api"></i>Model API</span><span><i class="cpu"></i>CPU agent</span><span><i class="training"></i>Training sandbox</span></div><p class="cost-note">Six trial totals · verifier sandbox excluded · Modal charges are pre-credit</p>';
    const bars=rows.map((row,index)=>{
      const model=row.family==='deepseek'?'DeepSeek':'Luna',trial=row.trial??index+1;
      const segments=row.parts.map(part=>`<i class="cost-segment ${part.key}" style="width:${100*part.value/max}%" title="${esc(part.label)}: ${part.value.toFixed(2)} USD · ${esc(part.basis)}"></i>`).join('');
      const breakdown=row.parts.map(part=>`${part.label} $${part.value.toFixed(2)}`).join(', ');
      return `<div class="cost-trial-row ${row.family}"><span class="cost-trial-label"><strong>${model} ${trial}</strong><small>${esc(row.run.run_id)}</small></span><div class="cost-stack" role="img" aria-label="${model} trial ${trial}: ${esc(breakdown)}; verifier sandbox excluded">${segments}</div><b>$${row.total.toFixed(2)}</b></div>`;
    }).join('');
    $('#resource-bars').innerHTML=legend+`<div class="cost-trial-list">${bars}</div>`;
  }
  function renderRuns(){const links=state.runs.map(run=>{const t=run.timeline,summary=t?.comparison_summary||{};return `<a class="run-link" href="/timeline?run=${encodeURIComponent(run.run_id)}"><b>${esc(run.run_id)}</b><span>${MODEL[family(run.model)]?.label||esc(run.model)} · ${t?.coverage?.ready?'finalized':'live / incomplete'}<br>${fmtTime(summary.wall_duration_ms)} · ${summary.tool_call_count??0} tool calls · ${fmtMoney(summary.final_total_cost_usd)}</span></a>`});$('#run-links').innerHTML=links.join('')||'<p class="empty">Trials have not started.</p>'}
  function render(){renderCards();const performanceModels=state.performance?.models||[];continuousChart('#cost-chart',performanceModels,'cumulative_agent_cost_usd','aggregate cost across 3 trials (API + CPU + training; verifier excluded)',state.performance?.cost?.common_auc_cap_usd);continuousChart('#time-chart',performanceModels,'hours_since_agent_launch','hours since agent launch',state.performance?.time?.common_auc_cap_hours);const legend=performanceModels.map(model=>{const f=family(model.model);return `<span><i style="background:${MODEL[f].color}"></i>${MODEL[f].label} · best of 3 trials</span>`}).join('');$('#cost-legend').innerHTML=legend;$('#time-legend').innerHTML=legend;renderPerformanceScores('#cost-scores','cost');renderPerformanceScores('#time-scores','time');renderResources();renderPolicies();renderRuns();const updated=state.performance?.generated_at||state.runs.map(r=>r.updated_at).filter(Boolean).sort().at(-1);$('#updated').textContent=updated?`Updated ${new Date(updated).toLocaleString()}`:'No experiment data deployed yet';$('#live').textContent=state.runs.some(r=>!r.timeline?.coverage?.ready)?'● trials live':'experiment archive';$('#live').classList.toggle('on',state.runs.some(r=>!r.timeline?.coverage?.ready))}
  async function init(){try{const [pIndex,tIndex,performance]=await Promise.all([json('/data/policies/index.json').catch(()=>({runs:[]})),json('/data/timelines/index.json').catch(()=>({runs:[]})),json('/data/performance/r8-continuous.json').catch(()=>null)]);state.performance=performance;const selected=pIndex.runs.filter(r=>family(r.model)).sort((a,b)=>`${b.created_at||''}:${b.run_id||''}`.localeCompare(`${a.created_at||''}:${a.run_id||''}`)).slice(0,6);for(const meta of selected){const p=await json(meta.path);const tMeta=tIndex.runs.find(row=>row.run_id===meta.run_id);const timeline=tMeta?{coverage:{ready:Boolean(tMeta.ready)},comparison_summary:tMeta.comparison_summary||{},resource_usage_summary:tMeta.resource_usage_summary||{},clock:{origin_epoch_ms:tMeta.origin_epoch_ms,end_epoch_ms:tMeta.end_epoch_ms},artifacts:tMeta.dashboard_artifacts||[]}:null;const run={...meta,timeline};state.runs.push(run);for(const policy of p.policies||[])state.policies.push({...policy,run_id:p.run_id,model:p.model})}render()}catch(error){$('#updated').textContent=`Data error: ${error.message}`;console.error(error)}}
  $('#filters').addEventListener('click',event=>{const button=event.target.closest('[data-filter]');if(!button)return;state.filter=button.dataset.filter;for(const item of $('#filters').querySelectorAll('button'))item.classList.toggle('active',item===button);renderPolicies()});init();
})();
