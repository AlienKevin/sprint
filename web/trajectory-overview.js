(() => {
  const canvas = document.querySelector('#utilization-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const stage = canvas.parentElement;
  const tip = document.querySelector('#utilization-tip');
  const status = document.querySelector('#utilization-status');
  const detailLink = document.querySelector('#utilization-detail-link');
  const dock = document.querySelector('.trajectory-controls');
  const colors = {
    line: '#242a31',
    muted: '#7d838c',
    text: '#f4f4f2',
    cpu: '#4fc3f7',
    training: '#9ccc65',
    verifier: '#b388ff',
    tools: '#e5b45c',
    event: '#e8ebef',
  };
  const lanes = [
    {key: 'cpu', label: 'CPU', color: colors.cpu},
    {key: 'training', label: 'TRAIN GPU', color: colors.training},
    {key: 'verifier', label: 'VERIFY GPU', color: colors.verifier},
  ];
  const state = {
    timeline: null,
    trajectory: null,
    series: null,
    cursorEpoch: null,
    hoverEpoch: null,
    scrollFrame: null,
  };

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const fmtDuration = ms => {
    const seconds = Math.max(0, Math.round(ms / 1000));
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ${String(seconds % 60).padStart(2, '0')}s`;
  };
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
    const series = {cpu: [], training: [], verifier: [], infrastructure: []};
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
      }
    }
    return series;
  }

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(320, stage.clientWidth);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(154 * ratio);
    canvas.style.height = '154px';
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
    ctx.clearRect(0, 0, width, 154);
    if (!state.timeline || !state.series) return;
    const {x0, x1} = bounds();
    const laneTop = 8;
    const laneHeight = 31;

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
      drawLine(state.series[lane.key], y + 3, laneHeight - 8, lane.color);
    });

    const eventY = laneTop + lanes.length * laneHeight + 8;
    ctx.fillStyle = colors.muted;
    ctx.font = '8px ui-monospace, monospace';
    ctx.fillText('ACTIVITY', 10, eventY + 11);
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
    for (const event of state.series.infrastructure) {
      const px = xFor(event.epoch_ms);
      ctx.fillStyle = /preempt|lost/.test(event.kind || '') ? '#ff6b6b' : colors.event;
      ctx.beginPath();
      ctx.arc(px, eventY + 13, 2.4, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = colors.muted;
    ctx.font = '8px ui-monospace, monospace';
    ctx.fillText('0m', x0, 148);
    const endLabel = fmtDuration(state.timeline.clock.end_epoch_ms - state.timeline.clock.origin_epoch_ms);
    const endWidth = ctx.measureText(endLabel).width;
    ctx.fillText(endLabel, x1 - endWidth, 148);

    const markerEpoch = state.hoverEpoch ?? state.cursorEpoch;
    if (markerEpoch != null) {
      const px = xFor(markerEpoch);
      ctx.strokeStyle = state.hoverEpoch != null ? '#ffffffaa' : '#ffffff70';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, 5);
      ctx.lineTo(px, 136);
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
    const values = lanes.map(lane => [lane.label, nearestValue(state.series[lane.key], state.hoverEpoch)]);
    tip.textContent = `${fmtDuration(state.hoverEpoch - state.timeline.clock.origin_epoch_ms)} · ${values.map(([label, value]) => `${label} ${value == null ? 'idle' : `${Math.round(value)}%`}`).join(' · ')}`;
    tip.hidden = false;
    tip.style.left = `${clamp(event.clientX - rect.left + 12, 8, rect.width - tip.offsetWidth - 8)}px`;
    tip.style.top = `${clamp(event.clientY - rect.top - 34, 6, 118)}px`;
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

  function jumpToEpoch(epoch) {
    const step = nearestStep(epoch);
    const target = step && document.getElementById(step.step_id);
    if (!target) return;
    target.scrollIntoView({behavior: 'smooth', block: 'start'});
    target.classList.add('jump-flash');
    setTimeout(() => target.classList.remove('jump-flash'), 1200);
  }

  function syncFromScroll() {
    state.scrollFrame = null;
    if (!state.timeline || !state.trajectory) return;
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
    draw();
    const runId = trajectory.run?.run_id || '';
    detailLink.href = `/timeline?run=${encodeURIComponent(runId)}`;
    try {
      const response = await fetch(`/data/timelines/${encodeURIComponent(runId)}.json`, {cache: 'no-store'});
      if (!response.ok) throw Error(`${response.status} ${response.statusText}`);
      state.timeline = await response.json();
      state.series = buildSeries(state.timeline);
      state.cursorEpoch = Date.parse(trajectory.steps?.[0]?.timestamp || '') || state.timeline.clock.origin_epoch_ms;
      status.textContent = 'Scroll the trace or select the chart';
      resize();
      syncFromScroll();
    } catch (error) {
      status.textContent = 'Utilization unavailable';
      canvas.setAttribute('aria-label', `Utilization unavailable: ${error.message}`);
    }
  }

  window.addEventListener('trajectory:loaded', event => loadOverview(event.detail.data));
  window.addEventListener('scroll', () => {
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
    jumpToEpoch(epochFor(event.clientX - rect.left));
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
  new ResizeObserver(resize).observe(stage);
})();
