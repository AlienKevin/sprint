(() => {
  const canvas = document.querySelector('#utilization-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const stage = canvas.parentElement;
  const tip = document.querySelector('#utilization-tip');
  const status = document.querySelector('#utilization-status');
  const dockSentinel = document.querySelector('#pulse-dock-sentinel');
  const pulse = document.querySelector('.utilization-overview');
  const trace = document.querySelector('.trace-column');
  const nav = document.querySelector('.viewer-nav');
  const replayPanel = document.querySelector('#trajectory-policy-replay');
  const replayFrame = document.querySelector('#trajectory-policy-replay-frame');
  const replayTitle = document.querySelector('#trajectory-policy-replay-title');
  const replayMeta = document.querySelector('#trajectory-policy-replay-meta');
  const replayClose = document.querySelector('#trajectory-policy-replay-close');
  const narrowViewport = window.matchMedia('(max-width: 850px)');
  const colors = {
    line: '#242a31',
    muted: '#7d838c',
    text: '#f4f4f2',
    cpu: '#4fc3f7',
    training: '#9ccc65',
    verifier: '#b388ff',
    tools: '#e5b45c',
  };
  const compactLanes = [
    {key: 'training', label: 'GPU', color: colors.training},
    {key: 'cpu', label: 'CPU', color: colors.cpu},
  ];
  const state = {
    timeline: null,
    trajectory: null,
    series: null,
    cursorEpoch: null,
    hoverEpoch: null,
    scrollFrame: null,
    scrollSyncLocked: false,
    scrollUnlockTimer: null,
    docked: false,
    selectedStepId: null,
    runId: null,
    policies: [],
    policyHits: [],
    selectedPolicyHash: null,
  };
  let replayResizeObserver = null;
  let replayHudObserver = null;
  const requestedRunId = new URLSearchParams(location.search).get('run');
  const settleTimeline = promise => promise.then(data => ({data}), error => ({error}));
  const prefetchedTimeline = requestedRunId
    ? settleTimeline(fetchTimeline(requestedRunId))
    : null;
  const prefetchedPolicies = requestedRunId
    ? settleTimeline(fetchPolicies(requestedRunId))
    : null;

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const fmtDuration = ms => {
    const seconds = Math.max(0, Math.round(ms / 1000));
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ${String(seconds % 60).padStart(2, '0')}s`;
  };
  const chartHeight = () => state.docked ? 94 : 126;
  const bounds = () => ({x0: 82, x1: Math.max(100, canvas.clientWidth - 12)});
  const xFor = epoch => {
    const clock = state.timeline.clock;
    const {x0, x1} = bounds();
    return x0 + ((epoch - clock.origin_epoch_ms) / Math.max(1, clock.end_epoch_ms - clock.origin_epoch_ms)) * (x1 - x0);
  };
  const epochFor = px => {
    const clock = state.timeline.clock;
    const {x0, x1} = bounds();
    return clock.origin_epoch_ms + clamp((px - x0) / Math.max(1, x1 - x0), 0, 1) * (clock.end_epoch_ms - clock.origin_epoch_ms);
  };

  function buildSeries(timeline) {
    const series = {cpu: [], training: [], trainingMemory: [], verifier: [], verifierMemory: [], infrastructure: []};
    for (const event of timeline.events || []) {
      if (event.category === 'infrastructure' && /allocated|preempt|lost|stop_requested/.test(event.kind || '')) {
        series.infrastructure.push(event);
      }
      if (event.category !== 'metrics') continue;
      if (event.role === 'cpu-agent' && event.metrics?.cpu_util_pct != null) {
        series.cpu.push({epoch: event.epoch_ms, value: Number(event.metrics.cpu_util_pct)});
      }
      const key = event.role === 'training-gpu' ? 'training' : event.role === 'verifier-gpu' ? 'verifier' : null;
      if (!key) continue;
      for (const gpu of event.metrics?.gpus || []) {
        if (gpu.util_gpu_pct != null) series[key].push({epoch: event.epoch_ms, value: Number(gpu.util_gpu_pct)});
        if (gpu.mem_used_mib != null && Number(gpu.mem_total_mib) > 0) {
          series[`${key}Memory`].push({epoch: event.epoch_ms, value: Number(gpu.mem_used_mib) / Number(gpu.mem_total_mib) * 100});
        }
      }
    }
    return series;
  }

  async function fetchTimeline(runId) {
    const overview = await fetch(`/data/timeline-overviews/${encodeURIComponent(runId)}.json`, {cache: 'no-store'});
    if (overview.ok) return overview.json();
    if (overview.status !== 404) throw Error(`${overview.status} ${overview.statusText}`);
    const full = await fetch(`/data/timelines/${encodeURIComponent(runId)}.json`, {cache: 'no-store'});
    if (!full.ok) throw Error(`${full.status} ${full.statusText}`);
    return full.json();
  }

  async function fetchPolicies(runId) {
    const response = await fetch(`/data/policies/${encodeURIComponent(runId)}.json`, {cache: 'no-store'});
    if (response.status === 404) return {policies: []};
    if (!response.ok) throw Error(`${response.status} ${response.statusText}`);
    return response.json();
  }

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(320, stage.clientWidth);
    const height = chartHeight();
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.height = `${height}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }

  function drawLine(points, y, height, color) {
    if (!points.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let started = false;
    let previous = null;
    for (const point of points) {
      const px = xFor(point.epoch);
      const py = y + height - clamp(point.value / 100, 0, 1) * (height - 5);
      if (!started || (previous != null && point.epoch - previous > 45000)) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
      started = true;
      previous = point.epoch;
    }
    ctx.stroke();
  }

  function accentColor() {
    return getComputedStyle(document.querySelector('.model-dot') || document.documentElement).backgroundColor || '#2279dc';
  }

  function policyScore(policy) {
    const score = Number(policy.effective_speed_mps);
    return Number.isFinite(score) && score > 0 ? score : 0;
  }

  function policyFinished(policy) {
    const finish = Number(policy.best_100m_s);
    return policy.termination_reason === 'finished' || (Number.isFinite(finish) && finish > 0);
  }

  function drawDiamond(x, y, filled, color) {
    ctx.beginPath();
    ctx.moveTo(x, y - 3.3);
    ctx.lineTo(x + 3.3, y);
    ctx.lineTo(x, y + 3.3);
    ctx.lineTo(x - 3.3, y);
    ctx.closePath();
    ctx.fillStyle = filled ? color : '#07090c';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  function drawStar(x, y, color) {
    ctx.beginPath();
    for (let index = 0; index < 10; index += 1) {
      const radius = index % 2 ? 2.1 : 4.2;
      const angle = -Math.PI / 2 + index * Math.PI / 5;
      const px = x + Math.cos(angle) * radius;
      const py = y + Math.sin(angle) * radius;
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = colors.text;
    ctx.lineWidth = .8;
    ctx.stroke();
  }

  function drawPolicyLegend(x1, color) {
    const labels = [
      {label: 'Finished', kind: 'finished'},
      {label: 'Did not finish', kind: 'failed'},
      {label: 'Best', kind: 'best'},
    ];
    ctx.font = '7px ui-monospace, monospace';
    const widths = labels.map(item => 9 + ctx.measureText(item.label).width);
    const total = widths.reduce((sum, width) => sum + width, 0) + 10 * (labels.length - 1);
    let x = Math.max(bounds().x0, x1 - total);
    for (const [index, item] of labels.entries()) {
      const markerX = x + 3;
      if (item.kind === 'best') drawStar(markerX, 8, color);
      else drawDiamond(markerX, 8, item.kind === 'finished', color);
      ctx.fillStyle = colors.muted;
      ctx.fillText(item.label, x + 8, 10.5);
      x += widths[index] + 10;
    }
  }

  function drawPolicies(y, laneHeight, color) {
    state.policyHits = [];
    const policies = state.policies
      .map(policy => ({...policy, epoch: Date.parse(policy.enqueued_at || policy.submitted_at || ''), score: policyScore(policy)}))
      .filter(policy => Number.isFinite(policy.epoch))
      .sort((left, right) => left.epoch - right.epoch || Number(left.submission_index) - Number(right.submission_index));
    if (!policies.length) return;

    const top = y + 3;
    const bottom = y + laneHeight - 3;
    const maxScore = Math.max(Number.EPSILON, ...policies.map(policy => policy.score));
    const bestScore = Math.max(0, ...policies.map(policy => policy.score));
    const failedSlots = new Map();
    const plotted = policies.map(policy => {
      const {x0, x1} = bounds();
      const rawX = clamp(xFor(policy.epoch), x0 + 4, x1 - 4);
      let markerY;
      if (policy.score > 0) {
        markerY = bottom - policy.score / maxScore * Math.max(5, bottom - top);
      } else {
        const bucket = Math.round(rawX / 7);
        const slot = failedSlots.get(bucket) || 0;
        failedSlots.set(bucket, slot + 1);
        markerY = bottom - (slot % 3) * 5;
      }
      return {...policy, x: rawX, y: markerY};
    });

    const frontier = [];
    let runningBest = 0;
    for (const policy of plotted) {
      if (policy.score <= runningBest) continue;
      if (!frontier.length) frontier.push({x: policy.x, y: bottom});
      else frontier.push({x: policy.x, y: frontier.at(-1).y});
      runningBest = policy.score;
      frontier.push({x: policy.x, y: policy.y});
    }
    if (frontier.length) {
      frontier.push({x: bounds().x1, y: frontier.at(-1).y});
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      frontier.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
      ctx.stroke();
    }

    const orderedPolicies = plotted
      .map(policy => ({
        ...policy,
        isBest: Boolean(policy.on_frontier) || (bestScore > 0 && Math.abs(policy.score - bestScore) < 1e-9),
      }))
      .sort((left, right) => Number(left.isBest) - Number(right.isBest));
    for (const policy of orderedPolicies) {
      const isBest = policy.isBest;
      if (isBest) drawStar(policy.x, policy.y, color);
      else drawDiamond(policy.x, policy.y, policyFinished(policy), color);
    }
    const selected = orderedPolicies.find(policy => state.selectedPolicyHash === policy.policy_sha256);
    if (selected) {
      ctx.strokeStyle = colors.text;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(selected.x, selected.y, 6, 0, Math.PI * 2);
      ctx.stroke();
    }
    state.policyHits = [...orderedPolicies]
      .reverse()
      .map(policy => ({x: policy.x, y: policy.y, policy}));
  }

  function draw() {
    const width = canvas.clientWidth;
    const height = chartHeight();
    const lanes = compactLanes;
    ctx.clearRect(0, 0, width, height);
    if (!state.timeline || !state.series) return;
    const {x0, x1} = bounds();
    const laneTop = state.docked ? 15 : 18;
    const laneHeight = state.docked ? 15 : 22;
    const accent = accentColor();

    drawPolicyLegend(x1, accent);

    const drawResourceLane = (lane, y) => {
      ctx.fillStyle = colors.muted;
      ctx.font = '8px ui-monospace, monospace';
      ctx.fillText(lane.label, 10, y + laneHeight - 5);
      ctx.strokeStyle = colors.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, y + laneHeight - 2);
      ctx.lineTo(x1, y + laneHeight - 2);
      ctx.stroke();
      drawLine(state.series[lane.key], y + 3, laneHeight - 8, lane.color);
    };

    drawResourceLane(lanes[0], laneTop);

    const eventY = laneTop + laneHeight;
    ctx.fillStyle = colors.muted;
    ctx.font = '8px ui-monospace, monospace';
    ctx.fillText('TOOL CALLS', 10, eventY + laneHeight - 5);
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, eventY + laneHeight - 2);
    ctx.lineTo(x1, eventY + laneHeight - 2);
    ctx.stroke();
    const buckets = state.timeline.tool_call_buckets?.buckets || [];
    const maxTools = Math.max(1, ...buckets.map(bucket => Number(bucket.total) || 0));
    for (const bucket of buckets) {
      const start = xFor(bucket.start_epoch_ms);
      const end = xFor(bucket.start_epoch_ms + (state.timeline.tool_call_buckets?.width_ms || 60000));
      const barHeight = Math.max(1, laneHeight - 6) * (Number(bucket.total) || 0) / maxTools;
      ctx.fillStyle = colors.tools;
      ctx.globalAlpha = .68;
      ctx.fillRect(start, eventY + laneHeight - 2 - barHeight, Math.max(1, end - start - 1), barHeight);
    }
    ctx.globalAlpha = 1;

    const cpuY = eventY + laneHeight;
    drawResourceLane(lanes[1], cpuY);

    const policyY = cpuY + laneHeight;
    ctx.fillStyle = colors.muted;
    ctx.font = '8px ui-monospace, monospace';
    ctx.fillText('SUBMISSIONS', 10, policyY + laneHeight - 5);
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, policyY + laneHeight - 2);
    ctx.lineTo(x1, policyY + laneHeight - 2);
    ctx.stroke();
    drawPolicies(policyY, laneHeight, accent);

    ctx.fillStyle = colors.muted;
    ctx.font = '8px ui-monospace, monospace';
    const duration = state.timeline.clock.end_epoch_ms - state.timeline.clock.origin_epoch_ms;
    const tickCount = 1;
    for (let index = 0; index <= tickCount; index += 1) {
      const label = index === 0 ? '0m' : fmtDuration(duration * index / tickCount);
      const px = x0 + (x1 - x0) * index / tickCount;
      const labelWidth = ctx.measureText(label).width;
      ctx.fillText(label, clamp(px - (index ? labelWidth / 2 : 0), x0, x1 - labelWidth), height - 6);
    }

    const markerEpoch = state.hoverEpoch ?? state.cursorEpoch;
    if (markerEpoch != null) {
      const px = xFor(markerEpoch);
      ctx.strokeStyle = state.hoverEpoch != null ? '#ffffffaa' : '#ffffff70';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, 5);
      ctx.lineTo(px, height - 18);
      ctx.stroke();
      ctx.fillStyle = colors.text;
      ctx.beginPath();
      ctx.arc(px, 5, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function policyAt(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let nearest = null;
    let distance = Infinity;
    for (const hit of state.policyHits) {
      const next = Math.hypot(hit.x - x, hit.y - y);
      if (next < distance) {
        nearest = hit.policy;
        distance = next;
      }
    }
    return distance <= 8 ? nearest : null;
  }

  function replayPolicy(policy) {
    if (!policy?.replay_ready || !policy.replay_url) return false;
    const isBest = Boolean(policy.on_frontier) || policyScore(policy) === Math.max(0, ...state.policies.map(policyScore));
    state.selectedPolicyHash = policy.policy_sha256;
    replayTitle.textContent = `Policy #${policy.submission_index}`;
    replayMeta.textContent = `${isBest ? 'Best · ' : ''}${policyScore(policy).toFixed(3)} m/s · ${policyFinished(policy) ? 'Finished' : 'Did not finish'}`;
    replayPanel.hidden = false;
    if (replayFrame.getAttribute('src') !== policy.replay_url) replayFrame.src = policy.replay_url;
    draw();
    requestAnimationFrame(() => replayPanel.scrollIntoView({behavior: 'auto', block: 'start'}));
    return true;
  }

  function fitReplayFrame() {
    const stageWrap = replayFrame?.contentDocument?.querySelector('.stagewrap');
    if (!stageWrap) return;
    const height = Math.ceil(stageWrap.getBoundingClientRect().height);
    if (height > 0) replayFrame.style.height = `${height}px`;
  }

  function watchReplaySize() {
    replayResizeObserver?.disconnect();
    replayHudObserver?.disconnect();
    const replayDocument = replayFrame?.contentDocument;
    replayDocument?.querySelectorAll('.lc .nm').forEach(label => label.remove());
    const replayLanes = replayDocument?.querySelectorAll('.lc') || [];
    const singleReplayLane = replayLanes.length === 1 ? replayLanes[0] : null;
    const replayClock = replayDocument?.querySelector('#clock');
    const replayClockWrap = replayClock?.parentElement;
    let replayClockStatus = replayDocument?.querySelector('#clock-status');
    if (singleReplayLane) {
      singleReplayLane.parentElement.style.display = 'none';
      if (!replayClockStatus && replayClockWrap) {
        replayClockStatus = replayDocument.createElement('span');
        replayClockStatus.id = 'clock-status';
        replayClockStatus.className = 'clock-status';
        replayClockStatus.style.cssText = 'font-size:14px;color:#B6F24E;font-weight:800;margin-left:8px';
        replayClockWrap.append(replayClockStatus);
      }
    }
    const scoredTime = singleReplayLane?.querySelector('.tm')?.textContent.match(/([\d.]+)s/)?.[1] || '';
    const scrubReplayHud = () => replayDocument?.querySelectorAll('.lc .d').forEach(label => {
      const clean = label.textContent
        .replace(/\s*·\s*body≤[\d.]+m/g, '')
        .replace(/^FINISH\s+[\d.]+s$/, 'FINISHED');
      if (clean !== label.textContent) label.textContent = clean;
      const finished = singleReplayLane?.classList.contains('fin') || clean === 'FINISHED';
      const statusText = finished ? 'FINISHED' : '';
      if (singleReplayLane && replayClockStatus && replayClockStatus.textContent !== statusText) replayClockStatus.textContent = statusText;
      if (finished && scoredTime && replayClock && replayClock.textContent !== scoredTime) replayClock.textContent = scoredTime;
    });
    scrubReplayHud();
    const replayHud = replayDocument?.querySelector('.hud');
    if (replayHud) {
      replayHudObserver = new MutationObserver(scrubReplayHud);
      replayHudObserver.observe(replayHud, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['class']});
    }
    const stageWrap = replayDocument?.querySelector('.stagewrap');
    if (!stageWrap) return;
    replayResizeObserver = new ResizeObserver(() => fitReplayFrame());
    replayResizeObserver.observe(stageWrap);
    requestAnimationFrame(fitReplayFrame);
  }

  function closeReplay() {
    replayResizeObserver?.disconnect();
    replayResizeObserver = null;
    replayHudObserver?.disconnect();
    replayHudObserver = null;
    replayPanel.hidden = true;
    replayFrame.removeAttribute('src');
    replayFrame.style.removeProperty('height');
    state.selectedPolicyHash = null;
    draw();
  }

  function nearestValue(points, epoch) {
    let best = null;
    let distance = Infinity;
    for (const point of points) {
      const next = Math.abs(point.epoch - epoch);
      if (next < distance) {
        distance = next;
        best = point;
      }
    }
    return distance <= 60000 ? best?.value : null;
  }

  function moveTip(event) {
    if (!state.timeline) return;
    const rect = canvas.getBoundingClientRect();
    const policy = policyAt(event.clientX, event.clientY);
    if (policy) {
      state.hoverEpoch = null;
      const isBest = Boolean(policy.on_frontier) || policyScore(policy) === Math.max(0, ...state.policies.map(policyScore));
      const result = policyFinished(policy) ? 'Finished' : 'Did not finish';
      tip.textContent = `Policy #${policy.submission_index} · ${isBest ? 'Best · ' : ''}${policyScore(policy).toFixed(3)} m/s · ${result}${policy.replay_ready ? ' · Click to replay' : ' · Replay unavailable'}`;
      tip.hidden = false;
      tip.style.left = `${clamp(event.clientX - rect.left + 12, 8, rect.width - tip.offsetWidth - 8)}px`;
      canvas.style.cursor = policy.replay_ready ? 'pointer' : 'default';
      draw();
      return;
    }
    canvas.style.cursor = 'crosshair';
    state.hoverEpoch = epochFor(event.clientX - rect.left);
    const values = compactLanes.map(lane => [lane.label, nearestValue(state.series[lane.key], state.hoverEpoch)]);
    tip.textContent = `${fmtDuration(state.hoverEpoch - state.timeline.clock.origin_epoch_ms)} · ${values.map(([label, value]) => `${label} ${value == null ? 'idle' : `${Math.round(value)}%`}`).join(' · ')}`;
    tip.hidden = false;
    tip.style.left = `${clamp(event.clientX - rect.left + 12, 8, rect.width - tip.offsetWidth - 8)}px`;
    draw();
  }

  function nearestStep(epoch) {
    let best = null;
    let distance = Infinity;
    for (const step of state.trajectory?.steps || []) {
      const stepEpoch = Date.parse(step.timestamp || '');
      const next = Math.abs(stepEpoch - epoch);
      if (Number.isFinite(stepEpoch) && next < distance) {
        best = step;
        distance = next;
      }
    }
    return best;
  }

  function renderedStepNode(step) {
    if (!step) return null;
    return document.querySelector(`[data-step-ids~="${CSS.escape(step.step_id)}"]`);
  }

  function scrollTargetForStep(target) {
    const divider = target?.previousElementSibling;
    return divider?.classList.contains('chapter-divider') ? divider : target;
  }

  function traceAnchor() {
    const controlsRect = pulse?.getBoundingClientRect();
    const traceRect = trace?.getBoundingClientRect();
    const overlaps = controlsRect && traceRect && controlsRect.right > traceRect.left && controlsRect.left < traceRect.right;
    return (overlaps ? controlsRect.bottom : nav?.getBoundingClientRect().bottom || 0) + 20;
  }

  function jumpToEpoch(epoch, forceScroll = false) {
    const step = nearestStep(epoch);
    const target = renderedStepNode(step);
    if (!target) return;
    const stepEpoch = Date.parse(step.timestamp || '');
    if (Number.isFinite(stepEpoch)) state.cursorEpoch = stepEpoch;
    if (state.selectedStepId === step.step_id && !forceScroll) {
      draw();
      return;
    }
    document.querySelector('.step.timeline-selected')?.classList.remove('timeline-selected');
    state.selectedStepId = step.step_id;
    target.classList.add('timeline-selected');
    const number = step.public_step_id ?? step.attempt_step_id;
    status.textContent = `Step #${number} · ${fmtDuration(stepEpoch - state.timeline.clock.origin_epoch_ms)}`;
    const previousBehavior = document.documentElement.style.scrollBehavior;
    state.scrollSyncLocked = true;
    clearTimeout(state.scrollUnlockTimer);
    document.documentElement.style.scrollBehavior = 'auto';
    scrollTargetForStep(target).scrollIntoView({behavior: 'auto', block: 'start'});
    document.documentElement.style.scrollBehavior = previousBehavior;
    state.scrollUnlockTimer = setTimeout(() => {
      state.scrollSyncLocked = false;
    }, 180);
    draw();
  }

  function syncFromScroll() {
    state.scrollFrame = null;
    if (state.scrollSyncLocked || !state.timeline || !state.trajectory) return;
    const anchor = traceAnchor();
    let current = state.trajectory.steps?.[0];
    for (const step of state.trajectory.steps || []) {
      const node = renderedStepNode(step);
      if (!node) continue;
      if (node.getBoundingClientRect().top <= anchor) current = step;
      else break;
    }
    const epoch = Date.parse(current?.timestamp || '');
    if (!Number.isFinite(epoch) || epoch === state.cursorEpoch) return;
    state.cursorEpoch = epoch;
    const number = current.public_step_id ?? current.attempt_step_id;
    status.textContent = `Step #${number} · ${fmtDuration(epoch - state.timeline.clock.origin_epoch_ms)}`;
    draw();
  }

  async function loadOverview(trajectory) {
    state.trajectory = trajectory;
    state.runId = trajectory.run?.run_id || '';
    state.timeline = null;
    state.series = null;
    state.policies = [];
    state.policyHits = [];
    status.textContent = 'Loading utilization…';
    draw();
    const runId = state.runId;
    try {
      const timelinePromise = prefetchedTimeline && requestedRunId === runId
        ? prefetchedTimeline
        : settleTimeline(fetchTimeline(runId));
      const policyPromise = prefetchedPolicies && requestedRunId === runId
        ? prefetchedPolicies
        : settleTimeline(fetchPolicies(runId));
      const result = await timelinePromise;
      if (result.error) throw result.error;
      if (state.runId !== runId) return;
      state.timeline = result.data;
      state.series = buildSeries(state.timeline);
      state.cursorEpoch = Date.parse(trajectory.steps?.[0]?.timestamp || '') || state.timeline.clock.origin_epoch_ms;
      status.textContent = 'Scroll the trace or select the chart';
      resize();
      syncFromScroll();
      policyPromise.then(policyResult => {
        if (state.runId !== runId) return;
        state.policies = policyResult.error ? [] : policyResult.data?.policies || [];
        draw();
      });
    } catch (error) {
      status.textContent = 'Utilization unavailable';
      canvas.setAttribute('aria-label', `Utilization unavailable: ${error.message}`);
    }
  }

  window.addEventListener('trajectory:loaded', event => loadOverview(event.detail.data));
  window.addEventListener('scroll', () => {
    if (state.scrollSyncLocked) {
      clearTimeout(state.scrollUnlockTimer);
      state.scrollUnlockTimer = setTimeout(() => {
        state.scrollSyncLocked = false;
      }, 180);
      return;
    }
    if (state.scrollFrame == null) state.scrollFrame = requestAnimationFrame(syncFromScroll);
  }, {passive: true});
  canvas.addEventListener('pointermove', moveTip);
  canvas.addEventListener('pointerleave', () => {
    state.hoverEpoch = null;
    tip.hidden = true;
    canvas.style.cursor = 'crosshair';
    draw();
  });
  canvas.addEventListener('click', event => {
    const policy = policyAt(event.clientX, event.clientY);
    if (policy && replayPolicy(policy)) return;
    const rect = canvas.getBoundingClientRect();
    state.hoverEpoch = null;
    tip.hidden = true;
    jumpToEpoch(epochFor(event.clientX - rect.left), true);
  });
  canvas.addEventListener('keydown', event => {
    if (['Enter', ' '].includes(event.key) && state.policies.length) {
      const nearest = state.policies
        .filter(policy => Number.isFinite(Date.parse(policy.enqueued_at || policy.submitted_at || '')))
        .sort((left, right) => (
          Math.abs(Date.parse(left.enqueued_at || left.submitted_at) - state.cursorEpoch)
          - Math.abs(Date.parse(right.enqueued_at || right.submitted_at) - state.cursorEpoch)
        ))[0];
      if (nearest?.replay_ready) {
        event.preventDefault();
        replayPolicy(nearest);
      }
      return;
    }
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key) || !state.timeline) return;
    event.preventDefault();
    const steps = state.trajectory?.steps || [];
    const current = nearestStep(state.cursorEpoch)?.step_id;
    const index = Math.max(0, steps.findIndex(step => step.step_id === current));
    const next = steps[clamp(index + (event.key === 'ArrowRight' ? 1 : -1), 0, steps.length - 1)];
    if (next) jumpToEpoch(Date.parse(next.timestamp));
  });
  replayFrame?.addEventListener('load', watchReplaySize);
  replayClose?.addEventListener('click', closeReplay);
  function setDocked(docked) {
    const next = narrowViewport.matches && docked;
    if (next === state.docked) return;
    state.docked = next;
    document.body.classList.toggle('pulse-docked', next);
    resize();
  }
  const dockObserver = new IntersectionObserver(entries => {
    const entry = entries[0];
    setDocked(!entry.isIntersecting && entry.boundingClientRect.top < 58);
  }, {rootMargin: '-58px 0px 0px'});
  dockObserver.observe(dockSentinel);
  narrowViewport.addEventListener('change', () => {
    if (!narrowViewport.matches) setDocked(false);
  });
  new ResizeObserver(resize).observe(stage);
})();
