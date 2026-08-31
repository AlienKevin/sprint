(() => {
  const I=window.SiteI18n;
  const words={
    'FINISHED':'已完赛','TIMEOUT':'超时','LANE DRIFT':'越出跑道','COLLISION':'自身碰撞','STOPPED':'已停止',
    '{seconds}s':'{seconds}秒','{minutes}m':'{minutes}分','{hours}h':'{hours}小时',
    'Policy {number}':'策略 {number}','Finished':'已完赛','Evaluation ended before the finish':'评测在到达终点前结束',
    'Effective speed':'有效速度','Agent cost':'智能体费用','Legal distance':'合法距离','Stop reason · supplementary':'停止原因 · 补充信息',
    'This replay is still rendering.':'该回放仍在生成中。',
    '{count} rendered {policies} · {submitted} submitted':'已生成 {count} 个回放 · 已提交 {submitted} 个策略',
    'policy':'policy','policies':'policies',
    'trial agent cost (API + CPU + training; verifier excluded)':'本次试验的智能体费用（API + CPU + 训练，不含评分器）',
    'effective speed (m/s) · higher is better':'有效速度（米/秒）· 越高越好',
    'Open rendered policy {number}':'打开策略 {number} 的回放',
    'Policy {number} · {speed} m/s · {cost}':'策略 {number} · {speed} 米/秒 · {cost}',
    'Submitted policies are awaiting rendered readouts':'已提交策略正在等待生成回放结果',
    'No policies submitted by this trial':'本次试验尚未提交策略',
    'CPU AGENT · %':'智能体 CPU · %','TRAINING GPU · %':'训练 GPU · %','VERIFIER GPU · %':'评分 GPU · %',
    'TOOLS / BUCKET':'工具调用 / 时间段','TRACE EVENTS':'记录事件','ALLOC / PREEMPT':'分配 / 抢占','SUBMISSIONS':'提交策略',
    'Loading…':'加载中…','complete':'完整','incomplete':'不完整',
    '{status} · training {trainingCovered}/{trainingTotal} · verifier {verifierCovered}/{verifierTotal} · {artifacts} artifacts':'{status} · 训练 {trainingCovered}/{trainingTotal} · 评分 {verifierCovered}/{verifierTotal} · {artifacts} 个产物',
    'No timeline index has been deployed yet':'尚未发布时间线索引','No experiment timelines found':'未找到试验时间线','No timeline data':'暂无时间线数据'
  };
  I.register('en',Object.fromEntries(Object.keys(words).map(key=>['timeline.'+key,key])));
  I.register('zh-CN',Object.fromEntries(Object.entries(words).map(([key,value])=>['timeline.'+key,value])));
  const bindings=[
    ['title','title','text',"Agents' 100m · 运行监控"],
    ['.timeline-heading h1','heading','text',"Agents' 100m · 运行监控"],
    ['.lede','lede','html','所有参赛模型使用同一个 UTC 时钟。智能体训练和封存策略的评分分别归属、单独呈现。<a id="trajectory-link" href="/trajectory">查看智能体过程记录 →</a>'],
    ['#run option','loading','text','正在加载试验…'],['#reset','reset','text','重置缩放'],
    ['#policy-cost-title','performance','text','性能与费用'],
    ['.policy-economics-head p','policiesIntro','text','仅显示本次试验提交的封存策略。点击数据点即可查看其回放。'],
    ['#policy-cost-chart','chartLabel','aria-label','本次试验的有效速度与智能体费用'],
    ['#policy-replay-title','selectedPolicy','text','选中的策略'],['#policy-replay-close','closeReplay','text','关闭回放'],
    ['#policy-replay-frame','replayTitle','title','选中策略的回放'],
    ['.help','help','text','滚动以缩放，拖动以平移，悬停以查看精确的 UTC 时间。记录标记只显示事件类型和工具名称；原始提示、参数和输出不在此公开。']
  ];
  const layerNames={cpu:'CPU',trainingGpu:'训练 GPU',trainingMem:'训练显存',verifierGpu:'评分 GPU',verifierMem:'评分显存',pipeline:'硬件流水线',tools:'工具调用',trace:'过程记录',infra:'资源分配',artifacts:'产物'};
  const legend=['CPU 使用率','训练 GPU 使用率','训练 GPU 显存','评分 GPU 使用率','评分 GPU 显存','SM 活跃度','SM 占用率','Tensor 流水线','FP32 FMA 流水线','FP16 指令','DRAM 吞吐量','工具调用 / 分钟','已被抢占','已分配 / 恢复'];
  function bind(){
    const en={},zh={};
    for(const [selector,name,mode,value]of bindings)for(const node of document.querySelectorAll(selector)){
      const key='timeline.static.'+name;
      if(mode==='html'){en[key]=node.innerHTML;node.dataset.i18nHtml=key}
      else if(mode==='text'){en[key]=node.textContent;node.dataset.i18n=key}
      else{en[key]=node.getAttribute(mode)||'';node.dataset.i18nAttr=mode+':'+key}
      zh[key]=value;
    }
    // Keep checkbox elements and telemetry markers intact; bind only their authored labels.
    for(const input of document.querySelectorAll('[data-layer]')){
      const label=input.parentElement,text=[...label.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join(''),span=document.createElement('span'),key='timeline.layer.'+input.dataset.layer;
      for(const n of [...label.childNodes])if(n.nodeType===3)n.remove();span.dataset.i18n=key;label.append(' ',span);en[key]=text.trim();zh[key]=layerNames[input.dataset.layer];
    }
    [...document.querySelectorAll('.legend > span')].forEach((node,index)=>{const text=[...node.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join(''),span=document.createElement('span'),key='timeline.legend.'+index;for(const n of [...node.childNodes])if(n.nodeType===3)n.remove();span.dataset.i18n=key;node.append(span);en[key]=text.trim();zh[key]=legend[index]});
    I.register('en',en);I.register('zh-CN',zh);I.apply();
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',bind,{once:true});else bind();
})();
