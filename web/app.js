(() => {
  const t=(key,params={})=>window.SiteI18n?.t('home.'+key,params,key)??key.replace(/\{(\w+)\}/g,(match,name)=>params[name]??match);
  let activeReadout = null;
  const trajectoryHref=(runId,options={})=>window.TrajectoryURL?.build({runId,...options})||`/trajectory?run=${encodeURIComponent(runId)}`;
  let observedVersion = null;
  let showAllTrials = false;
  let showAllBudgetTrials = false;
  const $ = selector => document.querySelector(selector);
  function initCitation(){
    const button=$('#citation-copy'),code=$('#citation-bibtex'),status=$('#citation-copy-status');
    if(!button||!code||!status)return;
    button.addEventListener('click',async()=>{
      if(button.disabled)return;
      button.disabled=true;button.textContent=t('Copying…');status.textContent='';
      try{
        if(typeof navigator.clipboard?.writeText!=='function')throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(code.textContent);
        button.textContent=t('Copied');status.textContent=t('Citation copied to clipboard.');
      }catch{
        button.textContent=t('Copy');status.textContent=t('Could not copy automatically. Select the citation text and copy it manually.');
      }finally{button.disabled=false;}
    });
  }
  initCitation();
  const replayFrames=[$('.model-race-frame iframe'),$('#readout-replay')].filter(Boolean);
  const clearReplayHeight=frame=>{frame.parentElement?.style.removeProperty('height');frame.parentElement?.style.removeProperty('aspect-ratio');};
  window.addEventListener('message',event=>{
    if(event.origin!==location.origin||event.data?.type!=='g1:replay-layout')return;
    const frame=replayFrames.find(candidate=>event.source===candidate.contentWindow);
    if(!frame)return;
    const {mobile,height}=event.data;
    if(window.innerWidth>720||mobile!==true){clearReplayHeight(frame);return;}
    if(!Number.isFinite(height)||height<64||height>2000)return;
    frame.parentElement.style.height=`${Math.ceil(height)}px`;frame.parentElement.style.aspectRatio='auto';
  });
  window.addEventListener('resize',()=>{if(window.innerWidth>720)replayFrames.forEach(clearReplayHeight);});
  const MODEL = {
    deepseek: {label:'DeepSeek-V4-Flash', color:'#7C54CD', cls:'deepseek'},
    'flash-baidu': {label:'DeepSeek V4 Flash 0731 · Baidu', color:'#7C54CD', cls:'flash-baidu'},
    'pro-alibaba': {label:'DeepSeek V4 Pro 0813 · Alibaba', color:'#ff9f43', cls:'pro-alibaba'},
    luna: {label:'GPT‑5.6 Luna', color:'#66D693', cls:'luna'},
    sol: {label:'GPT‑5.6 Sol', color:'#2279DC', cls:'sol'},
    opus: {label:'Claude Opus 5', color:'#D97757', cls:'opus'},
    glm: {label:'GLM‑5.3‑Flash', color:'#39B8B2', cls:'glm'}
  };
  const DISPLAY_FAMILIES = ['deepseek','luna','glm','flash-baidu','pro-alibaba','sol','opus'];
  const state = {runs:[], performance:null, batch:null, pricing:null, timelineUpdatedAt:null, refreshing:false};
  const chartSelections = new Map();
  let selectedReadoutKey = null;
  const readoutKey = (point,model) => `${model}:${point.source_run_id||point.run_id||point.source_trial}:${point.submission_index}`;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const groupBy = (rows,key) => rows.reduce((out,row)=>{const value=key(row);(out[value]??=[]).push(row);return out},{});
  const family = model => {const value=String(model||'').toLowerCase();return value.includes('deepseek-v4-pro-0813')?'pro-alibaba':value.includes('deepseek-v4-flash-0731')?'flash-baidu':value.includes('deepseek')?'deepseek':value.includes('luna')?'luna':value.includes('gpt-5.6-sol')?'sol':value.includes('claude-opus-5')?'opus':value.includes('glm-5.3-flash')?'glm':null};
  const modelDisplayRank = model => {const rank=DISPLAY_FAMILIES.indexOf(family(model));return rank<0?DISPLAY_FAMILIES.length:rank};
  const orderedModels = models => [...models].sort((a,b)=>modelDisplayRank(a.model)-modelDisplayRank(b.model));
  const stopLabel = name => t(({finished:'FINISHED',timeout:'TIMEOUT',in_lane:'LANE DRIFT',lane:'LANE DRIFT',lane_exit:'LANE DRIFT',lane_drift:'LANE DRIFT','left lane':'LANE DRIFT',self_collision:'COLLISION','self collision':'COLLISION',collision:'COLLISION',collide:'COLLISION'})[String(name||'').toLowerCase().replaceAll('-','_')]||'STOPPED');
  const fmtMoney = value => finite(value) ? `$${value.toFixed(value < 10 ? 2 : 0)}` : '—';
  const fmtTokenPrice = value => finite(value)&&value>=0 ? `$${value.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:6,useGrouping:false})}` : '—';
  const fmtTime = ms => finite(ms) ? t('{hours} h',{hours:(ms/3600000).toFixed(1)}) : '—';
  const fmtScore = value => finite(value) ? value.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2,useGrouping:false}) : '—';
  const fmtTokens = value => {const n=Number(value)||0;if(n>=1e9)return `${(n/1e9).toFixed(2)}B`;if(n>=1e6)return `${(n/1e6).toFixed(n>=1e8?0:1)}M`;if(n>=1e3)return `${(n/1e3).toFixed(n>=1e5?0:1)}K`;return String(n)};
  const fmtDuration = ms => {if(!finite(ms)||ms<0)return '—';const total=Math.floor(ms/1000),hours=Math.floor(total/3600),minutes=Math.floor(total%3600/60),seconds=total%60;return hours?t('{hours}h {minutes}m',{hours,minutes:String(minutes).padStart(2,'0')}):t('{minutes}m {seconds}s',{minutes,seconds:String(seconds).padStart(2,'0')})};
  const elapsedSince = value => {const time=Date.parse(value||'');if(!finite(time))return t('waiting');const seconds=Math.max(0,Math.floor((Date.now()-time)/1000));if(seconds<60)return t('{seconds}s ago',{seconds});if(seconds<3600)return t('{minutes}m ago',{minutes:Math.floor(seconds/60)});return t('{hours}h ago',{hours:Math.floor(seconds/3600)})};
  const esc = value => String(value??'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  async function json(path){const separator=path.includes('?')?'&':'?';const response=await fetch(`${path}${separator}_=${Date.now()}`,{cache:'no-store'});if(!response.ok)throw Error(`${path}: ${response.status}`);return response.json()}
  function svg(tag,attrs={}){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);return node}
  function bestTrialPerformance(models){
    return models.map(model=>{
      const trials=Object.values(groupBy(model.points||[],point=>point.source_run_id||`trial-${point.source_trial}`));
      const points=trials.sort((a,b)=>Math.max(0,...b.map(point=>Number(point.continuous_score_mps)||0))-Math.max(0,...a.map(point=>Number(point.continuous_score_mps)||0))||(Number(a[0]?.source_trial)||0)-(Number(b[0]?.source_trial)||0))[0]||[];
      return {...model,points};
    });
  }
  function integerTicks(maximum,targetCount=5){
    const rough=Math.max(1,maximum/targetCount),magnitude=10**Math.floor(Math.log10(rough)),normalized=rough/magnitude;
    const step=(normalized<1.5?1:normalized<3?2:normalized<7?5:10)*magnitude;
    return Array.from({length:Math.floor(maximum/step)+1},(_,index)=>index*step);
  }
  function syncChartSelection(){
    for(const chart of chartSelections.values()){
      chart.overlay?.remove();
      chart.overlay=null;
      let selected=null;
      for(const item of chart.selections){
        const active=readoutKey(item.row,item.model)===selectedReadoutKey;
        item.dot.setAttribute('aria-pressed',String(active));
        item.hit.setAttribute('aria-pressed',String(active));
        if(active)selected=item;
      }
      if(!selected)continue;
      const {cx,cy,model}=selected;
      // A separate last-painted layer stays visible over overlapping dots,
      // without moving data or intercepting the original hit targets.
      const overlay=svg('g',{class:'chart-selection','aria-hidden':'true','data-selection-key':selectedReadoutKey});
      overlay.append(
        svg('circle',{cx,cy,r:13,class:'chart-selection-halo'}),
        svg('circle',{cx,cy,r:13,fill:MODEL[family(model)]?.color,class:'chart-selection-dot'}),
        svg('path',{d:`M${cx-25},${cy}h5 M${cx+20},${cy}h5 M${cx},${cy-25}v5 M${cx},${cy+20}v5`,class:'chart-selection-crosshair'})
      );
      chart.el.append(overlay);
      chart.overlay=overlay;
    }
  }
  function continuousChart(target, runs, xKey, xLabel, aucCap=null){
    const el=$(target),compact=window.matchMedia('(max-width: 720px)').matches,width=el.clientWidth||1000,height=el.clientHeight||500,p={l:58,r:18,t:18,b:50},selections=[];el.replaceChildren();el.setAttribute('viewBox',`0 0 ${width} ${height}`);
    // SVG text scales with its viewBox; compensate so its visible size matches
    // the shared table/legend label size at every viewport width.
    const chartScale=el.getScreenCTM()?.a||1;el.style.setProperty('--chart-font-scale',String(1/chartScale));
    const plottable=point=>finite(point[xKey])&&finite(point.continuous_score_mps);
    const points=runs.flatMap(run=>(run.points||[]).filter(plottable).map(point=>({...point,model:run.model,color:MODEL[family(run.model)]?.color})));
    const xmax=finite(aucCap)?aucCap:Math.max(1,...points.map(point=>point[xKey])),rawMax=Math.max(.001,...points.map(point=>point.continuous_score_mps)),ymax=Math.ceil(rawMax*10)/10;
    const x=value=>p.l+Math.max(0,Math.min(xmax,value))/xmax*(width-p.l-p.r),y=value=>p.t+(ymax-Math.max(0,Math.min(ymax,value)))/ymax*(height-p.b-p.t);
    for(const value of integerTicks(ymax)){const yy=y(value);el.append(svg('line',{x1:p.l,x2:width-p.r,y1:yy,y2:yy,class:'gridline'}));const label=svg('text',{x:p.l-10,y:yy+4,'text-anchor':'end',class:'chart-label'});label.textContent=fmtScore(value);el.append(label)}
    const xTickValues=xKey==='cumulative_agent_cost_usd'?integerTicks(xmax,compact?3:5):Array.from({length:6},(_,index)=>index*xmax/5);
    for(const value of xTickValues){const xx=x(value);const label=svg('text',{x:xx,y:height-27,'text-anchor':'middle',class:'chart-label'});label.textContent=xKey==='cumulative_agent_cost_usd'?`$${value}`:t('{hours}h',{hours:value.toFixed(1)});el.append(label)}
    const xTitle=svg('text',{x:(p.l+width-p.r)/2,y:height-7,'text-anchor':'middle',class:'chart-label'});xTitle.textContent=xLabel;el.append(xTitle);
    const yTitleY=(p.t+height-p.b)/2,yTitle=svg('text',{x:14,y:yTitleY,'text-anchor':'middle',class:'chart-label',transform:`rotate(-90 14 ${yTitleY})`});yTitle.textContent=t('Effective speed (m/s)');el.append(yTitle);
    for(const run of runs){
      const rows=(run.points||[]).filter(plottable).sort((a,b)=>a[xKey]-b[xKey]),eligible=rows;let best=0,path=`M${x(0)},${y(0)} `;const frontier=new Set();
      for(const row of eligible){path+=`L${x(row[xKey])},${y(best)} `;if(row.continuous_score_mps>best){best=row.continuous_score_mps;frontier.add(row.policy_sha256);path+=`L${x(row[xKey])},${y(best)} `}}
      if(path&&finite(aucCap)&&(!eligible.length||eligible.at(-1)[xKey]<aucCap))path+=`L${x(aucCap)},${y(best)} `;
      const color=MODEL[family(run.model)]?.color;if(path)el.append(svg('path',{d:path,class:'best-line',stroke:color}));
      for(const row of rows){const onFrontier=frontier.has(row.policy_sha256),replayable=Boolean(row.replay_url),label=t('Open details for {model}, trial {trial}, submission {policy}',{model:MODEL[family(run.model)]?.label,trial:row.source_trial,policy:row.submission_index}),cx=x(row[xKey]),cy=y(row.continuous_score_mps),target=svg('g',{class:'submission-target'}),dot=svg('circle',{cx,cy,r:onFrontier?9.5:8,fill:color,class:`submission${onFrontier?' frontier':''}${replayable?' replayable':''}`,tabindex:'0',role:'button','aria-label':label}),hit=svg('circle',{cx,cy,r:18,class:'submission-hit',role:'button','aria-label':label}),title=svg('title');const stop=row.termination_reason?t(' · stopped by {reason}',{reason:stopLabel(row.termination_reason)}):'',capNote=finite(aucCap)&&row[xKey]>aucCap?t(' · plotted at {cost} cap',{cost:fmtMoney(aucCap)}):'';const queueCost=finite(row.cost_at_queue_usd)?fmtMoney(row.cost_at_queue_usd):'—';title.textContent=t('{model} · trial {trial} · submission {policy} · effective speed {speed} m/s · {distance}m legal in {time}s{stop} · cost at queue {cost}{cap} · {hours}h',{model:MODEL[family(run.model)]?.label,trial:row.source_trial,policy:row.submission_index,speed:fmtScore(row.continuous_score_mps),distance:row.max_legal_distance_m.toFixed(3),time:row.time_to_max_legal_distance_s.toFixed(2),stop,cost:queueCost,cap:capNote,hours:row.hours_since_agent_launch.toFixed(2)});dot.addEventListener('keydown',event=>{if(!['Enter',' '].includes(event.key))return;event.preventDefault();showReadout(row,run.model)});selections.push({cx,cy,row,model:run.model,dot,hit});target.append(dot,hit,title);el.append(target)}
    }
    el.onclick=event=>{const matrix=el.getScreenCTM();if(!matrix)return;const cursor=el.createSVGPoint();cursor.x=event.clientX;cursor.y=event.clientY;const local=cursor.matrixTransform(matrix.inverse()),nearest=selections.map(item=>({...item,distance:(item.cx-local.x)**2+(item.cy-local.y)**2})).sort((a,b)=>a.distance-b.distance)[0];if(nearest&&nearest.distance<=22**2)showReadout(nearest.row,nearest.model)};
    if(!points.length){const label=svg('text',{x:width/2,y:height/2,'text-anchor':'middle',class:'chart-label'});label.textContent=t('No reconstructed readouts available');el.append(label)}
    chartSelections.set(target,{el,selections,overlay:null});
    syncChartSelection();
  }
  function renderPerformanceScores(target,dimension){
    const data=state.performance;if(!data){$(target).innerHTML='';return}
    const isCost=dimension==='cost',cap=isCost?fmtMoney(data.cost.common_auc_cap_usd):t('{hours}h',{hours:data.time.common_auc_cap_hours.toFixed(2)}),key=isCost?'cost_auc_mps_at_common_cap':'time_auc_mps_at_common_cap';
    const models=orderedModels(data.models||[]),max=Math.max(.000001,...models.map(model=>Number(model.summary?.[key])||0));
    const rows=models.map(model=>{const s=model.summary||{},missing=s.missing_pose_capture_count||0,f=family(model.model),count=isCost?s.cost_readout_count_at_cap:s.time_readout_count_at_cap,value=Number(s[key])||0,width=100*value/max;return `<div class="auc-bar-row ${f}"><span class="auc-bar-label"><strong>${MODEL[f].label}</strong><small>${t('{count} readouts',{count})}${missing?t(' · {count} captures missing',{count:missing}):''}</small></span><div class="auc-bar-track" role="img" aria-label="${esc(t('{model}: {dimension} efficiency score {score} at {cap}',{model:MODEL[f].label,dimension:t(dimension),score:fmtScore(value),cap}))}"><i style="width:${width}%;background:${MODEL[f].color}"></i></div><b>${fmtScore(value)}</b></div>`}).join('');
    const name=t(isCost?'Cost-Adjusted Effective Speed':'Time-Adjusted Effective Speed');
    const live=data.snapshot_status==='active_provisional'?t(' · live provisional'):'';
    $(target).innerHTML=`<div class="auc-bar-chart"><div class="auc-bar-head"><strong>${name}</strong><span>${t('normalized AUC at {cap}{live} · higher is better',{cap,live})}</span></div>${rows}</div>`;
  }
  function renderCards(){
    const groups=groupBy(experimentTrialRows(),row=>row.family),keys=DISPLAY_FAMILIES.filter(key=>groups[key]?.length);
    const winners=Object.fromEntries(keys.map(key=>[key,rankedExperimentTrials(groups[key])[0]])),max=Math.max(.000001,...keys.map(key=>winners[key].bestScore));
    $('#model-cards').innerHTML=`<div class="hero-score-head"><strong>${t('Effective Speed')}</strong></div>${keys.map(key=>{
      const winner=winners[key],value=winner.bestScore,width=100*value/max,href=trajectoryHref(winner.arm.run_id);
      return `<a class="hero-score-row ${key}" href="${esc(href)}" aria-label="${esc(t('Open {model} best trial trajectory, Effective Speed {speed} metres per second',{model:MODEL[key].label,speed:fmtScore(value)}))}"><span><strong>${MODEL[key].label}</strong></span><div class="hero-score-track" role="img" aria-label="${esc(t('{model}: highest Effective Speed {speed} metres per second across all trials',{model:MODEL[key].label,speed:fmtScore(value)}))}"><i style="width:${width}%;background:${MODEL[key].color}"></i></div><b>${fmtScore(value)} m/s</b></a>`;
    }).join('')}`;
  }
  function showReadout(point,model,refreshTextOnly=false){
    activeReadout={point,model};
    selectedReadoutKey=readoutKey(point,model);
    syncChartSelection();
    const detail=$('#readout-detail'),f=family(model);
    $('#readout-title').textContent=t('{model} · trial {trial} · policy {policy}',{model:MODEL[f].label,trial:point.source_trial,policy:point.submission_index});
    const queueCost=finite(point.cost_at_queue_usd)?fmtMoney(point.cost_at_queue_usd):'—';
    const runId=point.source_run_id||point.run_id,trialLink=runId?trajectoryHref(runId,{policies:[point.submission_index],focus:point.submission_index,step:point.queue_source_step_id||'',turn:point.queue_source_public_step_id??null}):null;
    $('#readout-stats').title=t('Cumulative API and estimated compute cost at first enqueue.');
    $('#readout-stats').innerHTML=`<div><span>${t('Effective Speed')}</span><b>${fmtScore(point.continuous_score_mps)} m/s</b></div><div><span>${t('Legal distance')}</span><b>${Number(point.max_legal_distance_m).toFixed(3)} m</b></div><div><span>${t('Cost so far')}</span><b>${queueCost}</b></div><div class="readout-trial">${trialLink?`<a class="text-link" href="${esc(trialLink)}" aria-label="${esc(t('Open trial with policy {policy} selected at its submission turn',{policy:point.submission_index}))}">${t('Open trial')}</a>`:`<span>${t('Trial unavailable')}</span>`}</div>`;
    const replay=$('#readout-replay');
    if(!refreshTextOnly){if(point.replay_url){replay.removeAttribute('srcdoc');replay.src=point.replay_url}else{replay.removeAttribute('src');replay.srcdoc=`<!doctype html><html lang="${window.SiteI18n?.language||'en'}"><body style="margin:0;display:grid;place-items:center;height:100vh;background:#090909;color:#aaa;font:16px monospace;text-align:center"><p>${t('This policy has no archived website replay.')}<br>${t('Its verifier statistics are shown above.')}</p></body></html>`}}
    else if(!point.replay_url){try{const p=replay.contentDocument.querySelector('p');if(p)p.replaceChildren(document.createTextNode(t('This policy has no archived website replay.')),document.createElement('br'),document.createTextNode(t('Its verifier statistics are shown above.')))}catch{}}
    detail.hidden=false;if(!refreshTextOnly)detail.scrollIntoView({behavior:'smooth',block:'start'});
  }
  function closeReadout(){
    activeReadout=null;
    $('#readout-detail').hidden=true;
    $('#readout-replay').removeAttribute('src');
    selectedReadoutKey=null;
    syncChartSelection();
  }
  function trialNumber(run){const match=String(run.run_id||'').match(/-(\d+)$/);return match?Number(match[1]):null}
  function modalRoleCost(run,role){
    const resources=run.timeline?.resource_usage_summary||{},billing=resources.modal_provider_billing||{};
    const billed=Number(billing.by_role_usd?.[role]);
    if(billing.provider_complete===true&&finite(billed))return {value:billed,basis:t('exact Modal pre-credit billing')};
    const estimated=Number(resources.modal_estimate?.by_role?.[role]?.estimated_cost_usd);
    return {value:finite(estimated)?estimated:0,basis:t('Modal tariff estimate')};
  }
  function renderResources(){
    const order=DISPLAY_FAMILIES,armById=Object.fromEntries((state.batch?.arms||[]).map(arm=>[arm.run_id,arm])),performanceById=Object.fromEntries((state.performance?.runs||[]).map(run=>[run.run_id,run])),rows=state.runs.filter(run=>order.includes(family(run.model))).map(run=>{
      const api=Number(run.timeline?.comparison_summary?.final_api_cost_usd),cpu=modalRoleCost(run,'cpu_agent'),training=modalRoleCost(run,'training_gpu');
      const apiBasis=t((run.timeline?.usage_summary?.calculated_api_usage_cost_basis||[]).includes('openrouter_reported_cost')?'OpenRouter reported request cost':'reconstructed at published list price');
      const parts=[{key:'api',label:t('Model API'),value:finite(api)?api:0,basis:apiBasis},{key:'cpu',label:t('CPU agent'),...cpu},{key:'training',label:t('Training sandbox'),...training}];
      const performance=performanceById[run.run_id]||{},bestScore=Math.max(0,Number(performance.summary?.best_continuous_score_mps)||0,...(performance.points||[]).map(point=>Number(point.continuous_score_mps)||0));
      return {run,family:family(run.model),trial:armById[run.run_id]?.trial??trialNumber(run),parts,total:parts.reduce((sum,part)=>sum+part.value,0),bestScore};
    });
    if(!rows.length){$('#resource-bars').innerHTML=`<p class="empty">${t('Costs appear as trials start.')}</p>`;return}
    const budget=10,grouped=groupBy(rows,row=>row.family);
    const heat=value=>`${Math.round(Math.max(0,Math.min(1,value/budget))*44)}%`;
    const costLabels={api:t('Model API'),training:t('GPU'),cpu:t('CPU')};
    const costCell=part=>`<td class="budget-heat budget-${part.key}" data-label="${esc(costLabels[part.key])}" style="--heat:${heat(part.value)}" title="${esc(part.label)}: ${part.value.toFixed(2)} USD · ${esc(part.basis)}"><strong>${fmtMoney(part.value)}</strong></td>`;
    const pricingModels=state.pricing?.basis==='undiscounted_list_price'?Object.fromEntries(Object.entries(state.pricing.models||{}).filter(([key,pricing])=>grouped[key]?.every(row=>row.run.model===pricing.model))):{};
    const rateColumns=[['cached_input',t('Cached input')],['output',t('Output')]];
    const rateCells=(key,rowspan,mobile=false)=>rateColumns.map(([field,label])=>{
      const pricing=pricingModels[key],title=pricing?t('{label}: {price} USD per 1M tokens · {provider} undiscounted list price · {date}{note}',{label,price:fmtTokenPrice(pricing[field]),provider:pricing.provider,date:state.pricing.as_of,note:pricing.note?` · ${t(pricing.note)}`:''}):t('Undiscounted list price unavailable');
      return `<td class="${mobile?'budget-price-mobile-cell':'budget-price'}"${mobile?'':` rowspan="${rowspan}"`} data-label="${label}" title="${esc(title)}">${mobile?`<span class="budget-price-label">${label}</span>`:''}<strong>${fmtTokenPrice(pricing?.[field])}</strong></td>`;
    }).join('');
    const bodies=order.filter(key=>grouped[key]?.length).map(key=>{
      const trialOrder=(a,b)=>(Number(a.trial)||0)-(Number(b.trial)||0),rankedRows=[...grouped[key]].sort((a,b)=>b.bestScore-a.bestScore||trialOrder(a,b)),winner=rankedRows[0],familyRows=[winner,...rankedRows.slice(1).sort(trialOrder)],visibleRows=showAllBudgetTrials?familyRows:[winner];
      return `<tbody class="budget-group ${esc(key)}">${visibleRows.map((row,index)=>{
        const model=MODEL[key]?.label||key,displayTrial=index+1,parts=Object.fromEntries(row.parts.map(part=>[part.key,part])),href=trajectoryHref(row.run.run_id);
        return `<tr class="budget-row ${esc(key)}" data-href="${href}" data-trial-label="${t('Trial {trial}',{trial:displayTrial})}" tabindex="0" aria-label="${esc(t('Open {model} trial {trial} trace{best}',{model,trial:displayTrial,best:''}))}">${index===0?`<th class="budget-model" scope="rowgroup" rowspan="${visibleRows.length}"><span class="trial-dot"></span><strong>${esc(model)}</strong></th>`:''}${costCell(parts.api)}${costCell(parts.training)}${costCell(parts.cpu)}${index===0?rateCells(key,visibleRows.length):''}</tr>`;
      }).join('')}<tr class="budget-price-mobile" aria-label="${esc(t('{model} token list prices in USD per 1M tokens',{model:MODEL[key]?.label||key}))}">${rateCells(key,1,true)}</tr></tbody>`;
    }).join('');
    const modelCount=order.filter(key=>grouped[key]?.length).length,hasAdditionalTrials=rows.length>modelCount;
    const lunaTier=pricingModels.luna?.long_context,lunaNote=grouped.luna?.length&&lunaTier?`<span>${t('Luna above {threshold} input tokens: {input} cached input / {output} output per 1M tokens.',{threshold:Number(lunaTier.threshold_input_tokens).toLocaleString('en-US'),input:fmtTokenPrice(lunaTier.cached_input),output:fmtTokenPrice(lunaTier.output)})}</span>`:'';
    const pricingSources=order.filter(key=>grouped[key]?.length&&pricingModels[key]?.source_url?.startsWith('https://')).map(key=>`<a class="text-link" href="${esc(pricingModels[key].source_url)}">${esc(MODEL[key]?.label||key)}</a>`).join(' ');
    $('#resource-bars').innerHTML=`${hasAdditionalTrials?`<div class="budget-table-actions"><button type="button" class="experiment-table-toggle" aria-controls="budget-results-table" aria-expanded="${showAllBudgetTrials}">${t(showAllBudgetTrials?'Show best trials':'Show all trials')}</button></div>`:''}<div class="budget-table-wrap" data-expanded="${showAllBudgetTrials}"><table class="budget-table" id="budget-results-table"><colgroup><col class="budget-col-model"><col span="3" class="budget-col-cost"><col span="2" class="budget-col-price"></colgroup><thead><tr><th scope="col">${t('Model')}</th><th scope="col">${t('Model API')}</th><th scope="col">${t('GPU')}</th><th scope="col">${t('CPU')}</th><th scope="col" class="budget-price-heading">${t('Cached input')}</th><th scope="col">${t('Output')}</th></tr></thead>${bodies}</table></div>${pricingSources?`<div class="budget-price-sources"><span>${t('List prices as of {date} without OpenRouter discounts:',{date:esc(state.pricing.as_of)})} ${pricingSources}</span>${lunaNote}</div>`:''}`;
    $('#resource-bars .experiment-table-toggle')?.addEventListener('click',()=>{showAllBudgetTrials=!showAllBudgetTrials;renderResources()});
    document.querySelectorAll('#resource-bars .budget-row').forEach(row=>{const open=event=>{if(event?.target.closest('.budget-price'))return;if(row.dataset.href)location.href=row.dataset.href};row.addEventListener('click',open);row.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();open(event)}})});
  }
  function snapshotUpdatedAt(batch=state.batch){const candidates=[batch?.updated_at,state.timelineUpdatedAt,state.performance?.generated_at].map(value=>Date.parse(value||'')).filter(finite);return candidates.length?new Date(Math.max(...candidates)).toISOString():null}
  function batchIsLive(batch=state.batch){const updated=Date.parse(snapshotUpdatedAt(batch)||''),declared=['running','stopping','finalizing_site'].includes(batch?.status);return declared&&finite(updated)&&Date.now()-updated<30*60*1000}
  function experimentPhase(arm,run){const ledger=arm.ledger||{},submitted=Math.max(Number(ledger.submitted)||0,Number(run?.policy_count)||0),rendered=Number(run?.replay_count)||0;if(arm.status==='invalid_infrastructure'||arm.benchmark_valid===false)return t('Invalid · rerun required');if(arm.status==='finalized')return t('Complete');if(arm.status==='stopping'||arm.stop_requested_at)return t('Stopping');if((Number(ledger.running)||0)>0)return t('Scoring policy');if((Number(ledger.queued)||0)>0)return t('Policy queued');if(submitted>rendered&&batchIsLive())return t('Rendering replay');if(arm.harbor_alive===true)return t(submitted?'Agent iterating':'Agent exploring');if(arm.harbor_alive===false)return t('Agent stopped');if(arm.status==='planned')return t('Waiting to launch');if(arm.status==='launch_error'||arm.status==='missing_run_state')return t('Needs attention');return arm.status||t('Waiting for telemetry')}
  function experimentElapsed(arm,run){const timelineStart=Number(run?.timeline?.clock?.origin_epoch_ms),timelineEnd=Number(run?.timeline?.clock?.end_epoch_ms),declaredStart=Date.parse(arm.launched_at||run?.created_at||''),start=finite(timelineStart)?timelineStart:declaredStart;if(!finite(start))return null;const terminal=['finalized','invalid_infrastructure'].includes(arm.status)||['complete','complete_with_invalid_trials'].includes(state.batch?.status)||!batchIsLive();if(terminal&&finite(timelineEnd))return Math.max(0,timelineEnd-start);const declaredEnd=Date.parse(arm.finalized_at||''),snapshotEnd=Date.parse(snapshotUpdatedAt()||''),end=terminal?(finite(declaredEnd)?declaredEnd:finite(snapshotEnd)?snapshotEnd:Date.now()):Date.now();return Math.max(0,end-start)}
  function experimentTrialRows(){
    const arms=state.batch?.arms||state.runs.map(run=>({...run,trial:trialNumber(run)}));
    const byId=Object.fromEntries(state.runs.map(run=>[run.run_id,run]));
    const performanceById=Object.fromEntries((state.performance?.runs||[]).map(run=>[run.run_id,run]));
    return arms.map(arm=>{
      const run=byId[arm.run_id]||{},usage=run.timeline?.usage_summary||{},ledger=arm.ledger||{},performance=performanceById[arm.run_id]||{};
      const submitted=Math.max(Number(ledger.submitted)||0,Number(run.policy_count)||0),rendered=Number(run.replay_count)||0,totalCost=Number(run.timeline?.comparison_summary?.final_agent_total_cost_usd);
      const bestScore=Math.max(0,Number(performance.summary?.best_continuous_score_mps)||0,...(performance.points||[]).map(point=>Number(point.continuous_score_mps)||0));
      return {arm,run,usage,submitted,rendered,totalCost,bestScore,elapsed:experimentElapsed(arm,run),phase:experimentPhase(arm,run),family:family(arm.model)||arm.family||'deepseek',effort:arm.reasoning_effort||run.reasoning_effort||state.batch?.reasoning_effort||''};
    });
  }
  function rankedExperimentTrials(rows){return [...rows].sort((a,b)=>b.bestScore-a.bestScore||(Number(a.arm.trial)||0)-(Number(b.arm.trial)||0))}
  function renderExperimentTracker(){
    const target=$('#experiment-tracker'),batch=state.batch;
    if(!target)return;
    if(!batch){target.innerHTML=`<p class="empty">${t('No experiment batch has been published yet.')}</p>`;return}
    const rows=experimentTrialRows();
    const grouped=groupBy(rows,row=>row.family),familyOrder=[...DISPLAY_FAMILIES,...Object.keys(grouped).filter(key=>!DISPLAY_FAMILIES.includes(key))];
    const trialCounts=[...new Set(Object.values(grouped).map(group=>group.length))];
    const bestTrialLabel=trialCounts.length===1?(trialCounts[0]===5?t('Best of Five'):t('Best of {count}',{count:trialCounts[0]})):t('Best trials');
    const body=familyOrder.filter(key=>grouped[key]?.length).map(key=>{
      const trialOrder=(a,b)=>(Number(a.arm.trial)||0)-(Number(b.arm.trial)||0);
      const rankedRows=rankedExperimentTrials(grouped[key]);
      const winner=rankedRows[0];
      const familyRows=[winner,...rankedRows.slice(1).sort(trialOrder)];
      const visibleRows=showAllTrials?familyRows:[winner];
      return `<tbody class="experiment-group ${esc(key)}">${visibleRows.map((row,index)=>{
        const label=MODEL[key]?.label||row.arm.model,href=trajectoryHref(row.arm.run_id),displayTrial=index+1,isBest=row===winner,bestLabel=isBest?t(', highest Effective Speed for this model across all efforts'):'',speed=`${fmtScore(row.bestScore)} m/s`;
        return `<tr class="experiment-row ${esc(key)}${isBest?' experiment-best':''}" data-family="${esc(key)}" data-run-href="${href}" data-trial-label="${t('Trial {trial}',{trial:displayTrial})}" tabindex="0" aria-label="${esc(t('Open {model} trial {trial} trace{best}',{model:label,trial:displayTrial,best:bestLabel}))}">${index===0?`<th class="experiment-model" data-family="${esc(key)}" scope="rowgroup" rowspan="${visibleRows.length}"><span class="trial-dot"></span><strong>${esc(label)}</strong></th>`:''}<td data-label="${t('Effective Speed')}">${isBest?`<b>${speed}</b>`:speed}</td><td data-label="${t('Elapsed')}" data-experiment-elapsed="${esc(row.arm.run_id)}">${fmtDuration(row.elapsed)}</td><td class="experiment-total-cost" data-label="${t('Total cost')}">${fmtMoney(row.totalCost)}</td><td data-label="${t('Policies submitted')}">${row.submitted}</td></tr>`;
      }).join('')}</tbody>`;
    }).join('');
    const modelCount=familyOrder.filter(key=>grouped[key]?.length).length,hasAdditionalTrials=rows.length>modelCount;
    target.innerHTML=`${hasAdditionalTrials?`<div class="experiment-table-actions"><span class="experiment-best-label" aria-hidden="${showAllTrials}">${bestTrialLabel}</span><button type="button" class="experiment-table-toggle" aria-controls="experiment-results-table" aria-expanded="${showAllTrials}">${t(showAllTrials?'Show best trials':'Show all trials')}</button></div>`:''}<div class="experiment-table-wrap" data-expanded="${showAllTrials}"><table class="experiment-table" id="experiment-results-table"><thead><tr><th scope="col">${t('Model')}</th><th scope="col">${t('Effective Speed')}</th><th scope="col">${t('Elapsed')}</th><th scope="col">${t('Total cost')}</th><th scope="col">${t('Policies submitted')}</th></tr></thead>${body||`<tbody><tr><td colspan="5" class="empty">${t('Trials are waiting to launch.')}</td></tr></tbody>`}</table></div><p class="experiment-model-note">${t('DeepSeek-V4-Flash corresponds to DeepSeek V4 Flash Vision Exp.')}</p>`;
    target.querySelector('.experiment-table-toggle')?.addEventListener('click',()=>{showAllTrials=!showAllTrials;renderExperimentTracker()});
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
  function renderCharts(){const performanceModels=orderedModels(state.performance?.models||[]),bestPerformanceModels=bestTrialPerformance(performanceModels);continuousChart('#cost-chart',bestPerformanceModels,'cumulative_agent_cost_usd',t('Cost ($)'),state.performance?.cost?.common_auc_cap_usd);continuousChart('#time-chart',bestPerformanceModels,'hours_since_agent_launch',t('hours since agent launch'),state.performance?.time?.common_auc_cap_hours)}
  function render(){renderExperimentTracker();renderCards();renderCharts();const performanceModels=orderedModels(state.performance?.models||[]);const legend=performanceModels.map(model=>{const f=family(model.model);return `<span><i style="background:${MODEL[f].color}"></i><span>${MODEL[f].label}</span></span>`}).join('');$('#cost-legend').innerHTML=legend;$('#time-legend').innerHTML=legend;renderPerformanceScores('#time-scores','time');renderResources();const updated=snapshotUpdatedAt();$('#updated').textContent=updated?t('Updated {date}',{date:new Date(updated).toLocaleDateString(window.SiteI18n?.language||'en')}):t('No race data deployed yet')}
  let chartResizeFrame;
  window.addEventListener('resize',()=>{cancelAnimationFrame(chartResizeFrame);chartResizeFrame=requestAnimationFrame(renderCharts)});
  async function loadSnapshot(){const [pIndex,tIndex,performance,batch,pricing]=await Promise.all([json('/data/policies/index.json').catch(()=>({runs:[]})),json('/data/timelines/index.json').catch(()=>({runs:[]})),json('/data/performance/current.json').catch(()=>null),json('/data/batches/current.json').catch(()=>null),json('/assets/token-pricing.json').catch(()=>null)]);const batchIds=new Set(batch?.arms?.map(arm=>arm.run_id)||[]),performanceIds=new Set(performance?.runs?.map(run=>run.run_id)||[]),activeIds=batchIds.size?batchIds:performanceIds,selected=tIndex.runs.filter(row=>activeIds.size?activeIds.has(row.run_id):family(row.model)).sort((a,b)=>`${b.created_at||''}:${b.run_id||''}`.localeCompare(`${a.created_at||''}:${a.run_id||''}`));state.performance=performance;state.pricing=pricing;state.timelineUpdatedAt=tIndex.updated_at||null;state.runs=selected.map(tMeta=>{const pMeta=pIndex.runs.find(row=>row.run_id===tMeta.run_id),timeline={coverage:{ready:Boolean(tMeta.ready)},usage_summary:tMeta.usage_summary||{},comparison_summary:tMeta.comparison_summary||{},resource_usage_summary:tMeta.resource_usage_summary||{},clock:{origin_epoch_ms:tMeta.origin_epoch_ms,end_epoch_ms:tMeta.end_epoch_ms},artifacts:tMeta.dashboard_artifacts||[]};return {...tMeta,...pMeta,timeline}});state.batch=batch||{batch_id:'latest-published-runs',status:performance?.snapshot_status==='active_provisional'?'running':'complete',updated_at:tIndex.updated_at,arms:state.runs.map(run=>({run_id:run.run_id,model:run.model,family:family(run.model),trial:trialNumber(run),status:'finalized',launched_at:run.created_at,finalized_at:run.updated_at,ledger:{submitted:run.policy_count||run.timeline.comparison_summary.submission_count||0,scored:run.timeline.comparison_summary.submission_count||0}}))};render()}
  async function init(){try{await loadSnapshot()}catch(error){$('#updated').textContent=t('Data error: {message}',{message:error.message});console.error(error)}}
  async function refresh(){if(state.refreshing||document.hidden)return;state.refreshing=true;try{await loadSnapshot()}catch(error){console.warn('Race refresh failed',error)}finally{state.refreshing=false}}
  async function refreshVersion(){try{const deployed=await json('/version.json');if(!deployed.version)return;if(observedVersion===null){observedVersion=deployed.version;return}if(deployed.version!==observedVersion)window.location.reload()}catch(error){console.warn('Dashboard version check failed',error)}}
  window.addEventListener('site:languagechange',()=>{render();if(activeReadout)showReadout(activeReadout.point,activeReadout.model,true);$('#citation-copy-status').textContent='';});
  $('#readout-close').addEventListener('click',closeReadout);init();refreshVersion();setInterval(refresh,30000);setInterval(refreshVersion,30000);setInterval(updateExperimentClocks,1000);document.addEventListener('visibilitychange',()=>{if(!document.hidden){refresh();refreshVersion()}});window.addEventListener('focus',()=>{refresh();refreshVersion()});window.addEventListener('pageshow',()=>{refresh();refreshVersion()});
})();
