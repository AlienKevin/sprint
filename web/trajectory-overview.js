(() => {
  const canvas = document.querySelector('#utilization-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const stage = canvas.parentElement;
  const tip = document.querySelector('#utilization-tip');
  const status = document.querySelector('#utilization-status');
  const toggle = document.querySelector('#utilization-toggle');
  const mode = document.querySelector('#utilization-mode');
  const overview = document.querySelector('.utilization-overview');
  const dockSentinel = document.querySelector('#pulse-dock-sentinel');
  const dock = document.querySelector('.trajectory-controls');
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
    {key: 'cpu', label: 'CPU', color: colors.cpu},
    {key: 'training', label: 'TRAIN GPU', color: colors.training},
  ];
  const detailedLanes = [
    ...compactLanes,
    {key: 'trainingMemory', label: 'TRAIN MEM', color: '#d5efa9'},
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
    expanded: false,
    docked: false,
    selectedStepId: null,
  };

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const fmtDuration = ms => {
    const seconds = Math.max(0, Math.round(ms / 1000));
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ${String(seconds % 60).padStart(2, '0')}s`;
  };
  const activeLanes = () => state.expanded ? detailedLanes : compactLanes;
  const chartHeight = () => state.expanded ? 215 : state.docked ? 94 : 126;
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

  function draw() {
    const width = canvas.clientWidth;
    const height = chartHeight();
    const lanes = activeLanes();
    ctx.clearRect(0, 0, width, height);
    if (!state.timeline || !state.series) return;
    const {x0, x1} = bounds();
    const laneTop = state.docked ? 3 : 8;
    const laneHeight = state.docked ? 22 : state.expanded ? 47 : 31;

    lanes.forEach((lane, index) => {
      const y = laneTop + index * laneHeight;
      ctx.fillStyle = colors.muted;
      ctx.font = '8px ui-monospace, monospace';
      ctx.fillText(lane.label, 10, y + 18);
      ctx.strokeStyle = colors.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, y + laneHeight - 2);
      ctx.lineTo(x1, y + laneHeight - 2);
      ctx.stroke();
      if (state.expanded) {
        ctx.strokeStyle = '#242a3188';
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        ctx.moveTo(x0, y + laneHeight / 2);
        ctx.lineTo(x1, y + laneHeight / 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = colors.muted;
        ctx.fillText('100', x0 - 23, y + 8);
        ctx.fillText('0', x0 - 10, y + laneHeight - 5);
      }
      drawLine(state.series[lane.key], y + 3, laneHeight - 8, lane.color);
    });

    const eventY = laneTop + lanes.length * laneHeight + 8;
    ctx.fillStyle = colors.muted;
    ctx.font = '8px ui-monospace, monospace';
    ctx.fillText('TOOL CALLS', 10, eventY + 11);
    const buckets = state.timeline.tool_call_buckets?.buckets || [];
    const maxTools = Math.max(1, ...buckets.map(bucket => Number(bucket.total) || 0));
    for (const bucket of buckets) {
      const start = xFor(bucket.start_epoch_ms);
      const end = xFor(bucket.start_epoch_ms + (state.timeline.tool_call_buckets?.width_ms || 60000));
      const barHeight = 14 * (Number(bucket.total) || 0) / maxTools;
      ctx.fillStyle = colors.tools;
      ctx.globalAlpha = .68;
      ctx.fillRect(start, eventY + 17 - barHeight, Math.max(1, end - start - 1), barHeight);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = colors.muted;
    ctx.font = '8px ui-monospace, monospace';
    const duration = state.timeline.clock.end_epoch_ms - state.timeline.clock.origin_epoch_ms;
    const tickCount = state.expanded && !state.docked ? 4 : 1;
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
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, 5);
      ctx.lineTo(px, height - 18);
      ctx.stroke();
      ctx.fillStyle = colors.text;
      ctx.beginPath();
      ctx.arc(px, 5, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
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
    state.hoverEpoch = epochFor(event.clientX - rect.left);
    const values = activeLanes().map(lane => [lane.label, nearestValue(state.series[lane.key], state.hoverEpoch)]);
    tip.textContent = `${fmtDuration(state.hoverEpoch - state.timeline.clock.origin_epoch_ms)} · ${values.map(([label, value]) => `${label} ${value == null ? 'idle' : `${Math.round(value)}%`}`).join(' · ')}`;
    tip.hidden = false;
    tip.style.left = `${clamp(event.clientX - rect.left + 12, 8, rect.width - tip.offsetWidth - 8)}px`;
    tip.style.top = `${clamp(event.clientY - rect.top - 34, 6, chartHeight() - 36)}px`;
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

  function jumpToEpoch(epoch, forceScroll = false) {
    const step = nearestStep(epoch);
    const target = step && document.getElementById(step.step_id);
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
    target.scrollIntoView({behavior: 'auto', block: 'start'});
    document.documentElement.style.scrollBehavior = previousBehavior;
    state.scrollUnlockTimer = setTimeout(() => {
      state.scrollSyncLocked = false;
    }, 180);
    draw();
  }

  function syncFromScroll() {
    state.scrollFrame = null;
    if (state.scrollSyncLocked || !state.timeline || !state.trajectory) return;
    const anchor = (dock?.getBoundingClientRect().bottom || 0) + 20;
    let current = state.trajectory.steps?.[0];
    for (const step of state.trajectory.steps || []) {
      const node = document.getElementById(step.step_id);
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
    state.timeline = null;
    state.series = null;
    status.textContent = 'Loading utilization…';
    toggle.disabled = true;
    draw();
    const runId = trajectory.run?.run_id || '';
    try {
      const response = await fetch(`/data/timelines/${encodeURIComponent(runId)}.json`, {cache: 'no-store'});
      if (!response.ok) throw Error(`${response.status} ${response.statusText}`);
      state.timeline = await response.json();
      state.series = buildSeries(state.timeline);
      state.cursorEpoch = Date.parse(trajectory.steps?.[0]?.timestamp || '') || state.timeline.clock.origin_epoch_ms;
      status.textContent = 'Scroll the trace or select the chart';
      toggle.disabled = false;
      resize();
      syncFromScroll();
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
    draw();
  });
  canvas.addEventListener('click', event => {
    const rect = canvas.getBoundingClientRect();
    state.hoverEpoch = null;
    tip.hidden = true;
    jumpToEpoch(epochFor(event.clientX - rect.left), true);
  });
  canvas.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key) || !state.timeline) return;
    event.preventDefault();
    const steps = state.trajectory?.steps || [];
    const current = nearestStep(state.cursorEpoch)?.step_id;
    const index = Math.max(0, steps.findIndex(step => step.step_id === current));
    const next = steps[clamp(index + (event.key === 'ArrowRight' ? 1 : -1), 0, steps.length - 1)];
    if (next) jumpToEpoch(Date.parse(next.timestamp));
  });
  function setExpanded(expanded) {
    state.expanded = expanded;
    toggle.setAttribute('aria-expanded', String(state.expanded));
    toggle.setAttribute('aria-label', state.expanded ? 'Collapse resource details' : 'Expand resource details');
    toggle.title = state.expanded ? 'Collapse resource details' : 'Expand resource details';
    overview.classList.toggle('is-expanded', state.expanded);
    document.body.classList.toggle('pulse-expanded', state.expanded);
    mode.textContent = state.expanded ? 'full resource detail' : 'hover, then click';
    canvas.setAttribute('aria-label', state.expanded
      ? 'Detailed agent CPU, training GPU, training memory, and tool calls over the run. Hover to preview and click to select the nearest agent step.'
      : 'Agent CPU, training GPU, and tool calls over the run. Hover to preview and click to select the nearest agent step.');
    resize();
  }
  function setDocked(docked) {
    const next = narrowViewport.matches && docked;
    if (next === state.docked) return;
    state.docked = next;
    document.body.classList.toggle('pulse-docked', next);
    if (next && state.expanded) setExpanded(false);
    else resize();
  }
  toggle.addEventListener('click', () => setExpanded(!state.expanded));
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
