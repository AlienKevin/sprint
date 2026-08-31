/* Hand-authored homepage UI and editorial translations. Source traces and BibTeX stay verbatim. */
(() => {
  const I = window.SiteI18n;
  const dynamic = {
    'Copy':'复制', 'Copying…':'正在复制…', 'Copied':'已复制',
    'Citation copied to clipboard.':'引用已复制到剪贴板。',
    'Could not copy automatically. Select the citation text and copy it manually.':'无法自动复制。请选中引用文本并手动复制。',
    'FINISHED':'已完赛', 'TIMEOUT':'超时', 'LANE DRIFT':'越出跑道', 'COLLISION':'自身碰撞', 'STOPPED':'已停止',
    'Effective Speed':'有效速度', 'Effective speed (m/s)':'有效速度（米/秒）', 'Cost ($)':'费用（美元）',
    'hours since agent launch':'智能体启动后的小时数', 'No reconstructed readouts available':'暂无可显示的成绩',
    'Open details for {model}, trial {trial}, submission {policy}':'查看 {model} 第 {trial} 次试验、第 {policy} 次提交的详情',
    '{model} · trial {trial} · submission {policy} · effective speed {speed} m/s · {distance}m legal in {time}s{stop} · cost at queue {cost}{cap} · {hours}h':'{model} · 第 {trial} 次试验 · 第 {policy} 次提交 · 有效速度 {speed} 米/秒 · {time} 秒内合法前进 {distance} 米{stop} · 提交时费用 {cost}{cap} · {hours} 小时',
    ' · stopped by {reason}':' · 停止原因：{reason}', ' · plotted at {cost} cap':' · 按 {cost} 上限绘制',
    '{count} readouts':'{count} 次成绩', ' · {count} captures missing':' · 缺少 {count} 份动作记录',
    '{model}: {dimension} efficiency score {score} at {cap}':'{model}：在 {cap} 上限下的{dimension}效率得分为 {score}',
    'cost':'费用', 'time':'时间', 'Cost-Adjusted Effective Speed':'费用调整后的有效速度', 'Time-Adjusted Effective Speed':'时间调整后的有效速度',
    ' · live provisional':' · 实时暂定结果', 'normalized AUC at {cap}{live} · higher is better':'在 {cap} 上限下的归一化曲线下面积{live} · 越高越好',
    'Open {model} best trial trajectory, Effective Speed {speed} metres per second':'查看 {model} 最佳试验的过程，有效速度为 {speed} 米/秒',
    '{model}: highest Effective Speed {speed} metres per second across all trials':'{model}：所有试验中的最高有效速度为 {speed} 米/秒',
    '{model} · trial {trial} · policy {policy}':'{model} · 第 {trial} 次试验 · 策略 {policy}',
    'Cumulative API and estimated compute cost at first enqueue.':'首次提交时累计的 API 费用与估算计算费用。',
    'Legal distance':'合法距离', 'Cost so far':'累计费用', 'Open trial':'查看试验', 'Trial unavailable':'试验不可用',
    'Open trial with policy {policy} selected at its submission turn':'查看试验，选中策略 {policy} 并定位到提交该策略的轮次',
    'This policy has no archived website replay.':'该策略没有已存档的网页回放。', 'Its verifier statistics are shown above.':'上方显示的是其评分统计。',
    'exact Modal pre-credit billing':'Modal 抵扣前的实际账单', 'Modal tariff estimate':'按 Modal 费率估算',
    'OpenRouter reported request cost':'OpenRouter 报告的请求费用', 'reconstructed at published list price':'按公开标价还原',
    'Model API':'模型 API', 'CPU agent':'智能体 CPU', 'Training sandbox':'训练沙盒', 'CPU':'CPU', 'GPU':'GPU',
    'Costs appear as trials start.':'试验开始后将显示费用。', 'Cached input':'缓存输入', 'Output':'输出',
    '{label}: {price} USD per 1M tokens · {provider} undiscounted list price · {date}{note}':'{label}：每百万 token {price} 美元 · {provider} 未折扣标价 · {date}{note}',
    'Undiscounted list price unavailable':'暂无未折扣标价', 'Trial {trial}':'第 {trial} 次试验',
    'Open {model} trial {trial} trace{best}':'查看 {model} 第 {trial} 次试验的记录{best}',
    '{model} token list prices in USD per 1M tokens':'{model} 的 token 标价（美元/百万 token）',
    'Luna above {threshold} input tokens: {input} cached input / {output} output per 1M tokens.':'Luna 输入超过 {threshold} token 时：每百万 token 的缓存输入为 {input}，输出为 {output}。',
    'Show best trials':'显示最佳试验', 'Show all trials':'显示全部试验', 'Model':'模型',
    'List prices as of {date} without OpenRouter discounts:':'截至 {date} 的标价（不含 OpenRouter 折扣）：',
    'Invalid · rerun required':'无效 · 需重新运行', 'Complete':'已完成', 'Stopping':'正在停止', 'Scoring policy':'正在评分',
    'Policy queued':'策略已排队', 'Rendering replay':'正在生成回放', 'Agent iterating':'智能体正在迭代', 'Agent exploring':'智能体正在探索',
    'Agent stopped':'智能体已停止', 'Waiting to launch':'等待启动', 'Needs attention':'需要检查', 'Waiting for telemetry':'等待运行数据',
    'No experiment batch has been published yet.':'尚未发布任何试验批次。', 'Best of Five':'五次试验最快', 'Best of {count}':'{count}次试验最快', 'Best trials':'最佳试验',
    ', highest Effective Speed for this model across all efforts':'，该模型所有推理强度设置中的最高有效速度',
    'Elapsed':'用时', 'Total cost':'总费用', 'Policies submitted':'提交策略数', 'Trials are waiting to launch.':'试验正在等待启动。',
    'DeepSeek-V4-Flash corresponds to DeepSeek V4 Flash Vision Exp.':'DeepSeek-V4-Flash 对应 DeepSeek V4 Flash Vision Exp。',
    'Updated {date}':'更新于 {date}', 'No race data deployed yet':'尚未发布比赛数据', 'Data error: {message}':'数据错误：{message}',
    'waiting':'等待中', '{seconds}s ago':'{seconds} 秒前', '{minutes}m ago':'{minutes} 分钟前', '{hours}h ago':'{hours} 小时前',
    '{hours}h {minutes}m':'{hours}小时 {minutes}分', '{minutes}m {seconds}s':'{minutes}分 {seconds}秒',
    '{hours} h':'{hours} 小时', '{hours}h':'{hours}小时',
    'Prices shown for context lengths up to 272,000 input tokens.':'所示价格适用于输入不超过 272,000 token 的上下文。',
    'DeepSeek peak rates; off-peak discounts excluded.':'DeepSeek 高峰时段费率，不含非高峰折扣。',
    'Base rates. Above 272,000 input tokens: $0.04 cached input / $1.80 output per 1M tokens.':'基础费率。输入超过 272,000 token 时：每百万 token 的缓存输入为 $0.04，输出为 $1.80。',
    'Z.ai list rates before the 50% promotion.':'Z.ai 五折促销前的标价。'
  };
  I.register('en', Object.fromEntries(Object.keys(dynamic).map(key => ['home.' + key, key])));
  I.register('zh-CN', Object.fromEntries(Object.entries(dynamic).map(([key,value]) => ['home.' + key, value])));
  // Each binding names a specific editorial element; no general text replacement runs on the DOM.
  const bindings = [
    ['meta[name="description"]','description','content','AI 智能体竞相训练速度更快、成本更低的人形机器人短跑选手。'],
    ['.hero-copy .intro','intro','text','智能体能训练人形机器人跑步吗？'],
    ['.hero-copy .intro-detail','introDetail','text','我们为每个智能体提供一张 A10G GPU 和 10 美元预算，让它训练出自己最快的跑者。'],
    ['.model-race','raceLabel','aria-label','各模型公开的最佳人形机器人策略同场竞速'],
    ['.model-race iframe','raceTitle','title','各模型公开的最佳人形机器人策略同场竞速'],
    ['#experiment-tracker .empty','loading','text','正在加载试验结果…'],
    ['#cost-performance h2','costPerformance','text','性能与费用'],
    ['.chart-interaction-hint','chartHint','text','点击任意数据点，即可回放该策略。'],
    ['#cost-chart','costChart','aria-label','独立试验中最佳策略的性能与费用'],
    ['#time-performance .kicker','raceProgress','text','比赛进展'],
    ['#time-performance h2','timePerformance','text','性能随时间的变化'],
    ['#time-performance .section-head > p','timeDescription','text','每个参赛模型的试验采用相同的已用时间范围。连线表示截至该时刻产出的最佳封存策略。'],
    ['#time-chart','timeChart','aria-label','性能随试验时间的连续变化'],
    ['#time-performance .metric-note','timeNote','text','时间调整后的有效速度在共同的已用时间窗口内比较各参赛模型。'],
    ['#readout-title','selectedPolicy','text','选中的策略'],
    ['#readout-close','close','text','关闭'], ['#readout-close','closeLabel','aria-label','关闭选中的策略'],
    ['#readout-stats','queueCost','title','首次提交时累计的 API 费用与估算计算费用。'],
    ['#readout-replay','replayTitle','title','选中策略的回放'],
    ['#setup h2','setup','text','实验设置'],
    ['.setup-copy > p','setupDescription','text','每个智能体都具备训练和测试人形机器人跑者所需的基本条件：PyTorch、Isaac Lab、G1 机器人资源，以及赛道和评分代码。它需要在 10 美元预算内自行构建控制器，预算涵盖模型调用与 CPU/GPU 运行时间。'],
    ['#harnesses-heading','harnesses','text','智能体框架'],
    ['.harnesses-copy > p:nth-of-type(1)','harnessIntro','text','我们根据各模型提供方的官方基准测试设置选择智能体框架，再将其适配到统一的 10 美元机器人训练任务。'],
    ['.harnesses-copy > p:nth-of-type(2)','deepseekHarness','html','<strong>DeepSeek-V4-Flash</strong> 使用 DeepSeek Harness 0.1.1-rc.2 的 Minimal Mode，遵循<a class="text-link" href="https://api-docs.deepseek.com/news/news260821/">DeepSeek 官方基准测试设置</a>。'],
    ['.harnesses-copy > p:nth-of-type(3)','lunaHarness','html','<strong>GPT-5.6 Luna</strong> 使用 Codex 0.149.1 和<a class="text-link" href="https://github.com/openai/codex/blob/rust-v0.149.1/codex-rs/models-manager/models.json">固定版本的官方 Luna 模型目录</a>。'],
    ['.harnesses-copy > p:nth-of-type(4)','glmHarness','html','<strong>GLM-5.3-Flash</strong> 使用 Claude Code 2.1.248。Z.ai 也在 <a class="text-link" href="https://z.ai/blog/glm-5.3-flash">Terminal-Bench 2.1 和 Agents’ Last Exam</a> 中使用 Claude Code。'],
    ['.setup-copy > h3','integrity','text','实验公正性'],
    ['.setup-copy > ol > li:nth-child(1)','network','html','<strong>限制网络访问。</strong>训练与官方评分进程无法访问互联网。智能体只能通过 OpenRouter 访问模型 API，不能浏览网页或下载外部资源。'],
    ['.setup-copy > ol > li:nth-child(2)','apiKeys','html','<strong>固定 API 密钥。</strong>每次试验使用独立且设有预算上限的 OpenRouter 密钥，仅限访问被测模型及提供方。代理强制执行该路由，并禁止回退到其他模型或提供方。'],
    ['.setup-copy > ol > li:nth-child(3)','blindGrading','html','<strong>盲评。</strong>提交策略后只返回回执，不返回成绩，避免由独立预算支持的评分器成为免费反馈来源。智能体可以使用所提供的评分代码、模拟器和机器人资源，在自身预算内测试和调试策略，并记录性能与费用。'],
    ['#cost-breakdown h2','costBreakdown','text','费用明细'],
    ['.cost-insights > p:nth-child(1)','costInsight','text','随着推理引擎和模型架构不断创新，Flash 模型的费用持续下降。对 DeepSeek 和 GLM 而言，模型 API 费用已低于运行智能体的 CPU 费用，同样的预算因此能支持更多 GPU 实验。'],
    ['.cost-insights > p:nth-child(2)','harnessInsight','text','框架设计与 token 价格同样重要。DeepSeek 的缓存输入 token 单价不到 GLM 的一半，但在各自的最佳试验中，DeepSeek 的 token 支出却接近 GLM 的两倍。DeepSeek 使用 DeepSeek Harness，随着上下文增长，在最佳试验中累计重读了 7400 万缓存 token。GLM 使用的 Claude Code 在约 168K token 时压缩上下文，仅重读了 1800 万 token。巧妙的上下文压缩可能比更便宜的 token 更省钱。'],
    ['#scoring-guide h2','scoring','text','评分方式'],
    ['.scoring-copy > p:nth-child(1)','scoringIntro','text','100 米赛跑的评分很简单：越快越好。我们与正式人类比赛的唯一区别，是将未到终点的部分进度也计入成绩。'],
    ['.scoring-copy > p:nth-child(2)','scoringFormula','html','我们称之为<strong>有效速度</strong>：平均速度乘以已完成赛程的比例。公式为 <strong>(d / 100 m) × (d / t)</strong>，其中 <strong>d</strong> 是达到的最大合法距离（不超过 100 米），<strong>t</strong> 是首次达到该距离的时间。完整跑完的得分为 100 / t。'],
    ['.dq-rule','stopRule','text','计算时，跑过终点、达到 60 秒时限，或首次出现越界或碰撞，就停止累计距离。在此之前的进度仍计入得分。'],
    ['.dq-card:nth-child(1) h3','laneDrift','text','越界'], ['.dq-card:nth-child(2) h3','collision','text','碰撞'],
    ['.dq-card-head a','openTrial','text','查看试验'],
    ['.dq-card:nth-child(1) iframe','laneReplay','title','Luna 策略 20 越界回放'],
    ['.dq-card:nth-child(2) iframe','collisionReplay','title','Luna 策略 1 碰撞回放'],
    ['.dq-card:nth-child(1) .dq-caption','laneCaption','text','机器人的右手越过了跑道右侧边界。'],
    ['.dq-card:nth-child(2) .dq-caption','collisionCaption','text','机器人摔倒时，两只脚撞在了一起。'],
    ['#observations h2','observations','text','观察与发现'],
    ['.observation-cards > article:nth-child(1) p','deepseekObservation','text','DeepSeek 的所有试验均未完成 100 米。在最佳试验中，它先尝试强化学习，但机器人总是停下、越界或发生碰撞。随后它改用脚本式爬行动作，搜索更好的关节角度、时序和转向方式。本回放展示了最终结果：一个沿着跑道爬行约 78 米后超时的机器人。'],
    ['.observation-cards > article:nth-child(2) p','lunaObservation','text','Luna 在五次试验中有一次完成了 100 米，有效速度为 3.64 米/秒。成功的试验从简化训练机器人逐步切换到赛道所用的完整碰撞形状，并加入转向修正以保持在跑道内。其余四次试验难以学会持续移动，主要依靠脚本动作前进几米。例如，这个策略迈出一大步后，就脸朝下摔倒了。'],
    ['.observation-cards > article:nth-child(3) p','glmObservation','text','GLM 最快的模拟 100 米成绩为 9.90 秒，比尤塞恩·博尔特的 9.58 秒世界纪录仅慢 0.32 秒，而且采用站立起跑、没有起跑器。它使用 PPO 自行构建强化学习训练器，并依据跑道边界和碰撞规则测试保存的控制器。这个早期尝试说明了测试的重要性：机器人跪着挪动，随后偏出了跑道。'],
    ['.observation-cards > article:nth-child(1) iframe','deepseekReplay','title','DeepSeek 最佳策略沿跑道爬行'],
    ['.observation-cards > article:nth-child(2) iframe','lunaReplay','title','Luna 第 2 次试验的策略 9 迈出一大步后摔倒'],
    ['.observation-cards > article:nth-child(3) iframe','glmReplay','title','GLM 早期策略跪着移动后越界'],
    ['#inspiration h2','inspirations','text','灵感来源'],
    ['.inspiration-card:nth-child(1) img','qwopAlt','alt','LYiHub 训练项目网站上的 QWOP 跑者'],
    ['.inspiration-card:nth-child(2) img','gamesAlt','alt','2026 世界人形机器人运动会上的机器人短跑选手'],
    ['.inspiration-card:nth-child(3) img','posttrainAlt','alt','展示语言模型训练基准的 PostTrainBench 网站'],
    ['.inspiration-card:nth-child(1) h3','qwopTitle','text','QWOP 游戏'],
    ['.inspiration-card:nth-child(2) h3','gamesTitle','text','机器人赛跑'],
    ['.inspiration-card:nth-child(1) p','qwopDescription','text','QWOP 是一款古怪的小游戏，只用四个按键就能控制跑者。LYi 用强化学习训练出了世界上最快的 QWOP 策略。受此启发，我们让智能体在更真实的环境中完全自主地训练跑者：使用完整的 Unitree G1 人形机器人，以及 Isaac Lab 的三维物理模拟。'],
    ['.inspiration-card:nth-child(2) p','gamesDescription','text','2026 世界人形机器人运动会既有破纪录的短跑，也有各种出人意料的步态，包括 400 米项目中那位“害羞”的机器人。它启发我们同时探索速度，以及机器人学会移动的奇妙方式。'],
    ['.inspiration-card:nth-child(3) p','posttrainDescription','text','PostTrainBench 通过智能体能训练出什么来评估它们：用一张 H100 和十小时来改进预训练语言模型。我们将这一思路用于从零构建、没有参考策略的机器人控制器。不固定 GPU 时间，而是让智能体自行将 10 美元预算分配给模型调用、CPU 和 GPU 训练。'],
    ['.inspiration-card:nth-child(1) .inspiration-links a:nth-child(2)','lyiVideo','text','LYi 的视频'],
    ['.inspiration-card:nth-child(2) .inspiration-links a:nth-child(1)','gamesLink','text','2026 运动会'],
    ['.inspiration-card:nth-child(2) .inspiration-links a:nth-child(2)','sprintLink','text','100 米比赛'],
    ['.inspiration-card:nth-child(2) .inspiration-links a:nth-child(3)','shyLink','text','400 米“害羞”跑者'],
    ['#citation-heading','citation','text','引用'],
    ['#citation > p','citationIntro','text','如果 Agents’ 100m 对您的研究有帮助，欢迎引用：'],
    ['#citation-copy','copy','text','复制'], ['#citation-copy','copyLabel','aria-label','复制 BibTeX 引用'],
    ['#updated','notDeployed','text','尚未发布比赛数据']
  ];
  let bound = false;
  function bind() {
    if (bound) return;
    bound = true;
    const originals = {}, translated = {};
    for (const [selector,name,mode,zh] of bindings) {
      for (const node of document.querySelectorAll(selector)) {
        const key = 'home.static.' + name;
        if (mode === 'html') { originals[key] = node.innerHTML; node.dataset.i18nHtml = key; }
        else if (mode === 'text') { originals[key] = node.textContent; node.dataset.i18n = key; }
        else { originals[key] = node.getAttribute(mode) || ''; node.dataset.i18nAttr = [node.dataset.i18nAttr, mode + ':' + key].filter(Boolean).join(';'); }
        translated[key] = zh;
      }
    }
    I.register('en', originals); I.register('zh-CN', translated); I.apply();
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind, {once:true});
    // Bind once the final authored element has been parsed, without waiting for
    // the bottom application script to download. Capture English before applying Chinese.
    if (typeof MutationObserver === 'function') {
      const observer = new MutationObserver(() => {
        if (!document.querySelector('footer #updated')) return;
        observer.disconnect(); bind();
      });
      observer.observe(document.documentElement, {childList:true, subtree:true});
      document.addEventListener('DOMContentLoaded', () => observer.disconnect(), {once:true});
    }
  } else bind();
})();
