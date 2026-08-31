/* Hand-authored interface strings only; recordings and model identities stay unchanged. */
(() => {
  const strings = {
    play: ['Play', '播放'], pause: ['Pause', '暂停'], replay: ['Replay', '重播'],
    reset: ['Reset', '重置'], finished: ['FINISHED', '已完赛'], timeout: ['TIMEOUT', '超时'],
    drift: ['LANE DRIFT', '越出跑道'], collision: ['COLLISION', '自身碰撞'], fell: ['FELL', '跌倒'], dnf: ['DNF', '未完赛'],
    camera: ['Camera controls', '视角控制'], zoomIn: ['Zoom in', '放大'], zoomOut: ['Zoom out', '缩小'],
    speed: ['Playback speed', '播放速度'], selected: ['Selected policies', '已选策略'],
    policy: ['Policy #{number}', '策略 #{number}'], lane: ['Lane {number}', '第 {number} 跑道'],
    follow: ['Follow {name}', '跟随{name}'], remove: ['Remove {name}', '移除{name}'],
    title: ["Agents' 100m · Policy replay", "Agents' 100m · 策略回放"],
    empty: ['Select up to 8 policies from one trial to compare.', '选择同一试验中的最多 8 个策略进行对比。'],
    loadingOne: ['Loading {count} policy…', '正在加载 {count} 个策略…'],
    loadingMany: ['Loading {count} policies…', '正在加载 {count} 个策略…'],
    unavailable: ['Unable to load {ids}. Remove the unavailable policy or retry.', '无法加载 {ids}。请移除不可用的策略或重试。'],
    assetError: ['Unable to load replay {asset}. Please reload to retry.', '无法加载回放{asset}。请刷新重试。'],
    engine: ['engine', '引擎'], meshes: ['meshes', '模型资源'], assets: ['assets', '资源'],
    maxPolicies: ['At most 8 policies can be compared.', '最多可对比 8 个策略。'],
    incompatible: ['Selected policies have incompatible pose layouts.', '所选策略的姿态数据格式不兼容。'],
    cancelled: ['Replay load cancelled.', '已取消加载回放。'],
    runner: ['Runner', '选手'],
    unknownOne: ['Unknown policy capture: {ids}.', '未知策略记录：{ids}。'],
    unknownMany: ['Unknown policy captures: {ids}.', '未知策略记录：{ids}。'],
    bestTitle: ["Agents' 100m · Best-policy race", "Agents' 100m · 最佳策略竞赛"],
    bestEyebrow: ['BEST-POLICY RACE', '最佳策略竞赛'],
    bestHeadline: ['The fastest published policy from each model', '各模型已发布的最快策略'],
    bestLede: ['Three policies start together on the same 100-metre clock.', '三个策略在同一百米赛道上同时起跑。'],
    bestCaption: ['Best replayable policy per model from the current published cohort.', '当前公开试验中，各模型可回放的最佳策略。'],
    bestStory: ['The replay uses the verifier-authored pose capture for every runner.', '每位选手的回放均使用验证器记录的姿态数据。'],
    bestAccessible: ['Best published policies racing together in lanes one through three.', '已发布的最佳策略在第一至第三跑道上同场竞赛。'],
    policyEyebrow: ['POLICY REPLAY', '策略回放'],
    policyAccessible: ['Unitree G1 policy replay on the sprint course.', '宇树 G1 在短跑赛道上的策略回放。'],
    legacyTitle: ['The Race to AGI4ALL · Policy replay', 'The Race to AGI4ALL · 策略回放'],
    unfinished: ['Did not finish', '未完赛'],
    shortPolicy: ['Policy {number}', '策略 {number}'],
    corridorAccessible: ['{result}: Unitree G1 policy replay on a ±0.61 m corridor.', '{result}：宇树 G1 在 ±0.61 m 跑道内的策略回放。'],
    attempt100: ['G1 100 metres attempt #{number} - {result}', 'G1 百米尝试 #{number} - {result}'],
    attemptSprint: ['G1 Sprint attempt #{number} - {result}', 'G1 短跑尝试 #{number} - {result}'],
    readout: ['Policy readout.', '策略回放。'],
    recordReadout: ['Record-setting policy readout.', '创纪录策略回放。'],
    trialPolicy: ['Trial {trial} · policy {policy}', '试验 {trial} · 策略 {policy}'],
    effectiveSpeed: ['Effective Speed {value} m/s', '有效速度 {value} m/s'],
    back: ['← All runs and policies', '← 所有试验与策略'],
    comparison: ['Policy comparison', '策略对比'],
    laneRule: ['Whole body must stay between the ±0.61 m vertical lane planes; scoring stops at the verifier event', '全身必须保持在 ±0.61 m 的跑道垂直边界内；计分在验证器判定事件发生时停止'],
    laneRuleLegacy: ['Whole body must stay between the ±0.61 m vertical lane planes; distance freezes at the first stop', '全身必须保持在 ±0.61 m 的跑道垂直边界内；距离在首次停止时定格'],
    gestureHelp: ['Drag to orbit, scroll to zoom, space to pause', '拖动旋转视角，滚轮缩放，空格暂停'],
    settle: ['Distance freezes at the first stop; a short passive visual settle may follow.', '距离在首次停止时定格；画面随后可能短暂继续显示自然落定的动作。']
  };
  const byEnglish = new Map();
  for (const [key, [en, zh]] of Object.entries(strings)) {
    byEnglish.set(en, key);
    window.SiteI18n?.register('en', {['replay.' + key]: en});
    window.SiteI18n?.register('zh-CN', {['replay.' + key]: zh});
  }
  function t(key, params = {}) {
    const fallback = strings[key]?.[0] || key;
    return window.SiteI18n?.t('replay.' + key, params, fallback) || fallback.replace(/\{(\w+)\}/g, (_, name) => params[name] ?? '{' + name + '}');
  }
  function text(english) { return byEnglish.has(english) ? t(byEnglish.get(english)) : english; }
  function staticText(source) {
    if(byEnglish.has(source))return text(source);
    let match;
    if((match=source.match(/^Lane (\d+)(.*)$/)))return t('lane',{number:match[1]})+match[2];
    if((match=source.match(/^Trial (\d+) · policy (\d+)$/)))return t('trialPolicy',{trial:match[1],policy:match[2]});
    if((match=source.match(/^Policy (\d+)$/)))return t('shortPolicy',{number:match[1]});
    if((match=source.match(/^Effective Speed ([\d.]+) m\/s$/)))return t('effectiveSpeed',{value:match[1]});
    if((match=source.match(/^(.*): Unitree G1 policy replay on a ±0\.61 m corridor\.$/)))return t('corridorAccessible',{result:staticText(match[1])??match[1]});
    if((match=source.match(/^G1 (100 metres|Sprint) attempt #(\d+) - (.*)$/)))return t(match[1]==='Sprint'?'attemptSprint':'attempt100',{number:match[2],result:staticText(match[3])??match[3]});
  }
  function refreshStatic() {
    // Only renderer-authored presentation nodes; never recordings/model text.
    for (const node of document.querySelectorAll('.sr-only,.eyebrow,h1,.lede,.cap>span,.story>p,.polsel>a,.lc .nm')) {
      const source=node.dataset.replaySource??node.textContent;
      const translated=staticText(source);
      if(translated!==undefined){node.dataset.replaySource=source;if(node.textContent!==translated)node.textContent=translated;}
    }
    for(const card of document.querySelectorAll('.lanes .lc')){
      const source=card.dataset.replayInitialAria??card.getAttribute('aria-label')??'',match=source.match(/^Follow Lane (\d+)(.*)$/);
      if(match){card.dataset.replayInitialAria=source;card.setAttribute('aria-label',t('follow',{name:t('lane',{number:match[1]})+match[2]}));}
    }
    document.querySelector('.polsel')?.setAttribute('aria-label',t('comparison'));
    const title=document.documentElement?.dataset.replayTitle??document.title;
    document.title=staticText(title)??title;
  }
  function refresh() {
    if(document.documentElement&&!document.documentElement.dataset.replayTitle)document.documentElement.dataset.replayTitle=document.title;
    refreshStatic();
    const attrs = [['.camera-ctl', 'camera'], ['#camera-zoom-in', 'zoomIn'], ['#camera-zoom-out', 'zoomOut'], ['.seg', 'speed']];
    for (const [selector, key] of attrs) document.querySelector(selector)?.setAttribute('aria-label', t(key));
    document.querySelector('#camera-reset')?.replaceChildren(document.createTextNode(t('reset')));
    document.querySelector('.lanes')?.setAttribute('aria-label', t('selected'));
    const button = document.getElementById('replay');
    if (button && !window.__G1_REPLAY__) button.textContent = '▶ ' + t('play');
    if (document.title === strings.title[0] || document.title === strings.title[1]) document.title = t('title');
    if (window.__G1_REPLAY_ASSET_ERROR__) {
      const note=document.getElementById('failure-note');
      if(note)note.textContent=t('assetError',{asset:t(window.__G1_REPLAY_ASSET_NAME__||'assets')});
    }
  }
  window.ReplayI18n = {t, text, refresh};
  window.addEventListener('site:languagechange', refresh);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', refresh, {once: true});
    // External Three/HQ scripts block DOMContentLoaded. Translate the already
    // parsed controls before those large assets finish, then stop observing.
    if(typeof MutationObserver!=='undefined'){
      const observer=new MutationObserver(()=>{
        if(document.getElementById('replay')){observer.disconnect();refresh();}
      });
      observer.observe(document.documentElement,{childList:true,subtree:true});
    }
    if(document.getElementById('replay'))refresh();
  } else refresh();
})();
