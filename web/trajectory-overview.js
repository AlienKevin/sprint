(() => {
  const canvas = document.querySelector('#utilization-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const stage = canvas.parentElement;
  const tip = document.querySelector('#utilization-tip');
  const fmtSpeed = value => value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2, useGrouping: false});
  const status = document.querySelector('#utilization-status');
  const dockSentinel = document.querySelector('#pulse-dock-sentinel');
  const pulse = document.querySelector('.utilization-overview');
  const trace = document.querySelector('.trace-column');
  const stepsTarget = document.querySelector('#steps');
  const unmappedPolicies = document.querySelector('#unmapped-policies');
  const unmappedPolicyList = document.querySelector('#unmapped-policy-list');
  const nav = document.querySelector('.viewer-nav');
  const replayPanel = document.querySelector('#trajectory-policy-replay');
  let replayFrame = document.querySelector('#trajectory-policy-replay-frame');
  const replayLoading = document.querySelector('#trajectory-policy-replay-loading');
  const replayTitle = document.querySelector('#trajectory-policy-replay-title');
  const replayClose = document.querySelector('#trajectory-policy-replay-close');
  const mobileReplayActions = document.querySelector('#mobile-replay-actions');
  const mobileReplayLabel = document.querySelector('#mobile-replay-label');
  const policySelectionStatuses = document.querySelectorAll('.policy-selection-status');
  const mobileReplayClose = document.querySelector('#mobile-replay-close');
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
    selectedPolicies: [],
    emphasizedPolicyHash: null,
    focusedStepId: null,
    ready: false,
    restoringUrl: false,
    pendingRestore: null,
    restoreLockUntil: 0,
  };
  const MAX_SELECTED_POLICIES = 8;
  const COMPARISON_REPLAY_PATH = '/replay/trial-comparison';
  let selectionStatusTimer = null;
  let replayHudObserver = null;
  let replayGeneration = 0;
  let overviewGeneration = 0;
  let urlRestoreGeneration = 0;
  let urlScrollTimer = null;
  history.scrollRestoration = 'manual';
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

  async function resolvePolicyReplay(policy) {
    if (policy?.replay_ready && policy.replay_url) return policy;
    const sha = String(policy?.policy_sha256 || '');
    if (sha.length < 12) return policy;
    const slug = `frontier-${sha.slice(0, 12)}`;
    try {
      const response = await fetch(`/captures/${slug}.json`, {method: 'HEAD', cache: 'no-store'});
      return response.ok ? {...policy, replay_ready: true, replay_url: `/replay/${slug}`} : policy;
    } catch {
      return policy;
    }
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

  function policyBest(policy) {
    const score = policyScore(policy);
    const bestScore = Math.max(0, ...state.policies.map(policyScore));
    return Boolean(policy.on_frontier) || (bestScore > 0 && Math.abs(score - bestScore) < 1e-9);
  }

  function policyKind(policy) {
    return policyBest(policy) ? 'best' : policyFinished(policy) ? 'finished' : 'failed';
  }

  function policyTerminalLabel(policy) {
    if (policyFinished(policy)) return 'FINISHED';
    const reason = String(policy?.termination_reason || '').toLowerCase().replaceAll('-', '_');
    if (reason === 'self_collision' || reason === 'collision') return 'COLLISION';
    if (['in_lane', 'lane', 'lane_exit', 'lane_drift'].includes(reason)) return 'LANE DRIFT';
    if (reason === 'timeout') return 'TIMEOUT';
    const failedGates = Array.isArray(policy?.failed_gates) ? policy.failed_gates : [];
    if (failedGates.includes('self_collision')) return 'COLLISION';
    if (failedGates.includes('in_lane')) return 'LANE DRIFT';
    return 'STOPPED';
  }

  function policyNumber(policy) {
    return Number(policy.submission_index);
  }

  function policyQueueStep(policy) {
    // Admission/bridge observations can arrive hours after the agent queued a
    // policy. Only the exporter's proven source step may place it in the trace.
    const provenBases = new Set([
      'gpu_submission_result', 'gpu_submission_artifact', 'submission_bridge_job',
      'direct_submission_response', 'gpu_explicit_path_request_interval',
    ]);
    if (!provenBases.has(policy?.queue_source_basis)) return null;
    const steps = state.trajectory?.steps || [];
    const sourceId = policy.queue_source_step_id;
    const publicId = policy.queue_source_public_step_id;
    if (sourceId) {
      const step = steps.find(item => item.step_id === sourceId);
      if (!step || (publicId != null && Number(step.public_step_id ?? step.attempt_step_id) !== Number(publicId))) return null;
      return step;
    }
    if (publicId == null || !Number.isInteger(Number(publicId)) || Number(publicId) < 1) return null;
    const matches = steps.filter(step => Number(step.public_step_id ?? step.attempt_step_id) === Number(publicId));
    return matches.length === 1 ? matches[0] : null;
  }

  function policyEpoch(policy) {
    // No submitted_at/artifact_observed_at fallback: an unresolved policy must
    // not be clamped onto the last turn or onto the end of the chart.
    return policyQueueStep(policy) ? Date.parse(policy.enqueued_at || '') : NaN;
  }

  function orderPolicies(policies) {
    return [...policies]
      .sort((left, right) => (Number.isFinite(policyEpoch(left)) ? policyEpoch(left) : Infinity)
        - (Number.isFinite(policyEpoch(right)) ? policyEpoch(right) : Infinity)
        || Number(left.submission_index) - Number(right.submission_index));
  }

  function restoreOutlineScroll(outline, scrollTop) {
    if (outline) outline.scrollTop = scrollTop;
  }

  function chapterIdForStep(step) {
    const number = Number(step?.public_step_id ?? step?.attempt_step_id);
    if (!Number.isFinite(number)) return null;
    const item = [...document.querySelectorAll('.chapter-item')]
      .find(chapter => number >= Number(chapter.dataset.startStep) && number <= Number(chapter.dataset.endStep));
    return item?.dataset.chapterId || null;
  }

  function policyChapterId(policy) {
    return chapterIdForStep(policyQueueStep(policy));
  }

  function releasePolicyChapter() {
    window.dispatchEvent(new CustomEvent('trajectory:policy-navigation-cleared'));
  }

  function policyHash(policy) {
    return String(policy?.policy_sha256 || '');
  }

  function selectedPolicyIndex(policy) {
    const hash = policyHash(policy);
    return state.selectedPolicies.findIndex(selected => policyHash(selected) === hash);
  }

  function captureIdForPolicy(policy) {
    try {
      if (policy?.replay_url) {
        const parts = new URL(policy.replay_url, location.origin).pathname.split('/').filter(Boolean);
        const candidate = (parts.at(-1) || '').replace(/\.html$/, '');
        if (/^frontier-[a-f0-9]{12}$/.test(candidate)) return candidate;
      }
    } catch {}
    const hash = policyHash(policy);
    return /^[a-f0-9]{64}$/.test(hash) ? `frontier-${hash.slice(0, 12)}` : '';
  }

  function rendererPolicy(policy) {
    const captureId = captureIdForPolicy(policy);
    return {
      captureId,
      url: `/captures/${encodeURIComponent(captureId)}.json`,
      policyNumber: policyNumber(policy),
      label: `Policy #${policyNumber(policy)}`,
      color: accentColor(),
    };
  }

  function announceSelection(message) {
    if (!policySelectionStatuses.length) return;
    clearTimeout(selectionStatusTimer);
    policySelectionStatuses.forEach(statusNode => {
      statusNode.textContent = message;
    });
    if (message) selectionStatusTimer = window.setTimeout(() => {
      policySelectionStatuses.forEach(statusNode => {
        statusNode.textContent = '';
      });
    }, 5000);
  }

  function renderPolicyChips(container) {
    if (!container) return;
    container.replaceChildren();
    state.selectedPolicies.forEach((policy, index) => {
      const chip = document.createElement('span');
      const emphasized = policyHash(policy) === state.emphasizedPolicyHash;
      chip.className = `policy-selection-chip${emphasized ? ' is-emphasized' : ''}`;
      chip.style.setProperty('--policy-color', accentColor());
      chip.title = `Policy #${policyNumber(policy)}, lane ${index + 1}${emphasized ? ', emphasized' : ''}`;
      const symbol = document.createElement('span');
      symbol.className = `policy-selection-chip-symbol policy-reference-${policyKind(policy)}`;
      symbol.setAttribute('aria-hidden', 'true');
      symbol.textContent = policyKind(policy) === 'best' ? '★' : policyFinished(policy) ? '◆' : '◇';
      const label = document.createElement('span');
      label.textContent = String(policyNumber(policy));
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = '×';
      remove.setAttribute('aria-label', `Remove Policy #${policyNumber(policy)} from lane ${index + 1}`);
      remove.title = `Remove Policy #${policyNumber(policy)} from lane ${index + 1}`;
      remove.addEventListener('click', event => {
        event.preventDefault();
        event.stopPropagation();
        removePolicySelection(policy, {navigate: emphasized, openReplay: !replayPanel.hidden});
      });
      chip.append(symbol, label, remove);
      container.append(chip);
    });
  }

  function syncPolicyMarkers() {
    const selectedIndexes = new Map(state.selectedPolicies.map((policy, index) => [policyHash(policy), index]));
    for (const marker of document.querySelectorAll('.policy-reference')) {
      if (marker.disabled) continue;
      const index = selectedIndexes.get(marker.dataset.policyHash);
      const selected = index != null;
      const emphasized = selected && marker.dataset.policyHash === state.emphasizedPolicyHash;
      marker.classList.toggle('is-selected', selected);
      marker.classList.toggle('is-emphasized', emphasized);
      marker.setAttribute('aria-pressed', String(selected));
      marker.setAttribute('aria-label', `${marker.dataset.policyLabel}${selected ? `, selected in lane ${index + 1}` : ', not selected'}`);
      marker.title = `${marker.dataset.policyTitle} · ${selected ? 'Click to remove from comparison' : 'Click to add to comparison'}`;
    }
  }

  function syncSelectionUI() {
    const hasSelection = state.selectedPolicies.length > 0;
    document.body.classList.toggle('policy-selection-active', hasSelection);
    if (mobileReplayActions) mobileReplayActions.hidden = !hasSelection;
    if (mobileReplayLabel) mobileReplayLabel.textContent = `${state.selectedPolicies.length} selected`;
    if (mobileReplayClose) mobileReplayClose.hidden = replayPanel.hidden;
    syncPolicyMarkers();
    draw();
  }

  function beginUrlInteraction() {
    urlRestoreGeneration += 1;
    state.restoringUrl = false;
    state.pendingRestore = null;
    state.restoreLockUntil = 0;
    clearTimeout(urlScrollTimer);
  }

  function commitUrl(mode = 'push') {
    if (!state.ready || state.restoringUrl) return;
    clearTimeout(urlScrollTimer);
    const focused = state.selectedPolicies.find(policy => policyHash(policy) === state.emphasizedPolicyHash);
    const url = window.TrajectoryURL.build({
      runId: state.runId,
      policies: state.selectedPolicies.map(policyNumber),
      focus: focused ? policyNumber(focused) : null,
      step: state.focusedStepId || '',
      replayVisible: !replayPanel.hidden,
    });
    if (url !== location.pathname + location.search + location.hash) {
      history[mode === 'replace' ? 'replaceState' : 'pushState']({trajectory: true}, '', url);
    }
  }

  function finishUrlRestore() {
    const pending = state.pendingRestore;
    if (pending && pending.generation !== replayGeneration) return;
    state.pendingRestore = null;
    if (pending?.step) jumpToStep(pending.step, true, true, true, null);
    state.restoringUrl = false;
    state.restoreLockUntil = performance.now() + 700;
    commitUrl('replace');
  }

  function restoreUrlState() {
    if (!state.ready) return;
    const restoreGeneration = ++urlRestoreGeneration;
    clearTimeout(urlScrollTimer);
    const request = window.TrajectoryURL.parse(location.href);
    if (request.runId && request.runId !== state.runId) return;
    const resolved = window.TrajectoryURL.resolve(request, {
      runId: state.runId, policies: state.policies, steps: state.trajectory.steps || [],
    });
    state.restoringUrl = true;
    state.selectedPolicies = resolved.policies;
    state.emphasizedPolicyHash = policyHash(resolved.focus) || null;
    // An explicit turn is independent of the selected policies. Only links
    // without a turn default to the emphasized policy's proven queue step.
    const step = resolved.step || (!(request.step || request.turn != null) ? policyQueueStep(resolved.focus) : null)
      || state.trajectory.steps?.[0] || null;
    state.focusedStepId = step?.step_id || null;
    if (resolved.replayVisible) replayPolicy(resolved.focus, {scrollToReplay: false});
    else hideReplay({clearFrame: true});
    state.pendingRestore = {step, generation: replayGeneration};
    syncSelectionUI();
    window.dispatchEvent(new CustomEvent('trajectory:policy-selected', {detail: {chapterId: chapterIdForStep(step)}}));
    requestAnimationFrame(() => {
      if (!state.restoringUrl || restoreGeneration !== urlRestoreGeneration) return;
      if (step) jumpToStep(step, true, true, true, null);
      if (!resolved.replayVisible) finishUrlRestore();
    });
    if (resolved.errors.length) announceSelection(resolved.errors.join(' '));
  }

  function comparisonPayload() {
    return {
      type: 'g1:set-policies',
      replayGeneration: replayGeneration,
      policies: state.selectedPolicies.map(rendererPolicy),
      emphasizedCaptureId: captureIdForPolicy(state.selectedPolicies.find(policy => policyHash(policy) === state.emphasizedPolicyHash)),
    };
  }

  function postRendererSelection() {
    if (!replayFrame?.contentWindow || !state.selectedPolicies.length) return;
    replayFrame.contentWindow.postMessage(comparisonPayload(), location.origin);
  }

  function comparisonReplaySrc() {
    const payload = comparisonPayload();
    const params = new URLSearchParams({
      scene: '20260830-7',
      policies: payload.policies.map(policy => policy.captureId).join(','),
      emphasis: payload.emphasizedCaptureId || '',
      replayGeneration: String(replayGeneration),
    });
    return `${COMPARISON_REPLAY_PATH}?${params}`;
  }

  function navigateToPolicy(policy, declaredChapterId = null) {
    const step = policyQueueStep(policy);
    const outline = document.querySelector('#chapter-nav');
    const outlineScroll = outline?.scrollTop || 0;
    window.dispatchEvent(new CustomEvent('trajectory:policy-selected', {detail: {chapterId: chapterIdForStep(step)}}));
    window.dispatchEvent(new CustomEvent('trajectory:outline-lock', {detail: {duration: 2500, scrollTop: outlineScroll}}));
    requestAnimationFrame(() => {
      if (step) jumpToStep(step, true, true, true);
      else {
        announceSelection(`Policy #${policyNumber(policy)} has no matched queue turn. Its replay is still available.`);
        commitUrl();
      }
      restoreOutlineScroll(outline, outlineScroll);
      requestAnimationFrame(() => restoreOutlineScroll(outline, outlineScroll));
    });
    return true;
  }

  function removePolicySelection(policy, {navigate = true, openReplay = true} = {}) {
    const index = selectedPolicyIndex(policy);
    if (index < 0) return false;
    beginUrlInteraction();
    state.selectedPolicies.splice(index, 1);
    const emphasized = state.selectedPolicies.at(-1) || null;
    state.emphasizedPolicyHash = policyHash(emphasized) || null;
    if (!emphasized) {
      hideReplay({clearFrame: true});
      syncSelectionUI();
      commitUrl();
      return true;
    }
    if (openReplay) replayPolicy(emphasized, {scrollToReplay: false});
    else syncSelectionUI();
    if (openReplay && navigate) navigateToPolicy(emphasized, policyChapterId(emphasized));
    syncSelectionUI();
    if (!openReplay || !navigate) commitUrl();
    return true;
  }

  function selectPolicy(policy, declaredChapterId = null) {
    const selectedIndex = selectedPolicyIndex(policy);
    if (selectedIndex >= 0) return removePolicySelection(policy);
    if (!policy?.replay_ready || !policy.replay_url) {
      announceSelection(`Replay unavailable for Policy #${policyNumber(policy)}.`);
      return false;
    }
    if (state.selectedPolicies.length >= MAX_SELECTED_POLICIES) {
      announceSelection(`You can compare up to ${MAX_SELECTED_POLICIES} policies. Remove one before adding Policy #${policyNumber(policy)}.`);
      return false;
    }
    beginUrlInteraction();
    state.selectedPolicies.push(policy);
    state.emphasizedPolicyHash = policyHash(policy);
    replayPolicy(policy, {scrollToReplay: false});
    navigateToPolicy(policy, declaredChapterId);
    syncSelectionUI();
    return true;
  }

  function policyMarker(policy, context, declaredChapterId = null) {
    const kind = policyKind(policy);
    const numberValue = policyNumber(policy);
    const statusText = `${kind === 'best' ? 'best policy, ' : ''}${policyTerminalLabel(policy)}`;
    const marker = document.createElement('button');
    marker.type = 'button';
    marker.className = `policy-reference policy-reference-${kind}`;
    marker.dataset.policyHash = policy.policy_sha256 || '';
    marker.dataset.policyLabel = `Policy #${numberValue}, ${statusText}, ${context}`;
    const queueStep = policyQueueStep(policy);
    if (queueStep) {
      marker.dataset.queueStepId = queueStep.step_id;
      marker.dataset.queuePublicStep = queueStep.public_step_id ?? queueStep.attempt_step_id;
    }
    marker.dataset.policyTitle = `Policy #${numberValue} · ${kind === 'best' ? 'Best · ' : ''}${policyTerminalLabel(policy)}`;
    marker.setAttribute('aria-pressed', 'false');
    marker.setAttribute('aria-label', `${marker.dataset.policyLabel}, not selected`);
    marker.title = `${marker.dataset.policyTitle} · Click to add to comparison`;
    if (!policy.replay_ready || !policy.replay_url) {
      marker.disabled = true;
      marker.setAttribute('aria-label', `${marker.dataset.policyLabel}, replay unavailable`);
      marker.title = `${marker.dataset.policyTitle} · Replay unavailable`;
    }
    const symbol = document.createElement('span');
    symbol.className = 'policy-reference-symbol';
    symbol.textContent = kind === 'best' ? '★' : kind === 'finished' ? '◆' : '◇';
    const number = document.createElement('span');
    number.textContent = String(numberValue);
    marker.append(symbol, number);
    marker.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      selectPolicy(policy, declaredChapterId);
    });
    return marker;
  }

  function renderPolicyReferences() {
    document.querySelectorAll('.step-policy-list').forEach(node => node.remove());
    document.querySelectorAll('.chapter-policy-list').forEach(node => node.replaceChildren());
    unmappedPolicyList?.replaceChildren();
    if (unmappedPolicies) unmappedPolicies.hidden = true;
    if (!state.policies.length || !state.trajectory) return;
    const mapped = state.policies
      .map(policy => {
        const step = policyQueueStep(policy);
        const target = renderedStepNode(step);
        if (!target) {
          unmappedPolicyList?.append(policyMarker(policy, 'queue turn not matched in this trace'));
          if (unmappedPolicies) unmappedPolicies.hidden = false;
        }
        return target ? {policy, step, target} : null;
      })
      .filter(Boolean)
      .sort((left, right) => policyEpoch(left.policy) - policyEpoch(right.policy) || policyNumber(left.policy) - policyNumber(right.policy));
    const stepGroups = new Map();
    for (const entry of mapped) {
      const target = entry.target;
      if (!stepGroups.has(target)) stepGroups.set(target, []);
      stepGroups.get(target).push(entry);
    }
    for (const [target, entries] of stepGroups) {
      const list = document.createElement('div');
      list.className = 'step-policy-list';
      list.setAttribute('aria-label', 'Policies queued at this turn');
      for (const {policy, step} of entries) {
        const number = step.public_step_id ?? step.attempt_step_id;
        list.append(policyMarker(policy, `queued at turn ${number}`, chapterIdForStep(step)));
      }
      target.querySelector(':scope > summary > .step-meta')?.append(list);
    }
    for (const item of document.querySelectorAll('.chapter-item')) {
      const start = Number(item.dataset.startStep);
      const end = Number(item.dataset.endStep);
      const list = item.querySelector('.chapter-policy-list');
      const policies = mapped
        .filter(({step}) => {
          const number = Number(step.public_step_id ?? step.attempt_step_id);
          return number >= start && number <= end;
        })
        .map(({policy}) => policy);
      if (!policies.length || !list) continue;
      list.setAttribute('aria-label', 'Policies queued during this chapter');
      const label = document.createElement('span');
      label.className = 'chapter-policy-label';
      label.textContent = 'Queued';
      list.append(label);
      for (const policy of policies) {
        const step = policyQueueStep(policy);
        list.append(policyMarker(policy, `queued at turn ${step.public_step_id ?? step.attempt_step_id}`, item.dataset.chapterId || null));
      }
    }
    syncPolicyMarkers();
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
      .map(policy => ({...policy, epoch: policyEpoch(policy), score: policyScore(policy)}))
      .filter(policy => Number.isFinite(policy.epoch))
      .sort((left, right) => left.epoch - right.epoch || policyNumber(left) - policyNumber(right));
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
    const selectedHashes = new Set(state.selectedPolicies.map(policyHash));
    for (const selected of orderedPolicies.filter(policy => selectedHashes.has(policyHash(policy)))) {
      const emphasized = policyHash(selected) === state.emphasizedPolicyHash;
      ctx.strokeStyle = emphasized ? color : colors.text;
      ctx.lineWidth = emphasized ? 2.5 : 1.2;
      ctx.beginPath();
      ctx.arc(selected.x, selected.y, emphasized ? 7.5 : 6, 0, Math.PI * 2);
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
      const markerRadius = 3;
      const markerY = laneTop + markerRadius;
      ctx.strokeStyle = state.hoverEpoch != null ? '#ffffffaa' : '#ffffff70';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, markerY);
      ctx.lineTo(px, height - 18);
      ctx.stroke();
      ctx.fillStyle = colors.text;
      ctx.beginPath();
      ctx.arc(px, markerY, markerRadius, 0, Math.PI * 2);
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

  function replayPolicy(policy, {scrollToReplay = true} = {}) {
    if (!state.selectedPolicies.length || !policy?.replay_ready || !policy.replay_url) return false;
    const count = state.selectedPolicies.length;
    replayTitle.textContent = count === 1 ? `Policy #${policyNumber(policy)}` : `${count} policies selected`;
    replayPanel.hidden = false;
    document.body.classList.add('policy-replay-active');
    const currentReplay = replayFrame.getAttribute('src') || '';
    const startsNewLoad = !currentReplay.startsWith(COMPARISON_REPLAY_PATH)
      || new URL(currentReplay, location.href).searchParams.get('policies') !== state.selectedPolicies.map(selected => captureIdForPolicy(selected)).join(',');
    if (startsNewLoad) replayGeneration += 1;
    const generation = replayGeneration;
    const replaySrc = comparisonReplaySrc();
    if (startsNewLoad && replayLoading) replayLoading.textContent = 'Loading selected policies…';
    if (startsNewLoad) {
      replayHudObserver?.disconnect();
      replayHudObserver = null;
      replayPanel.classList.add('is-loading');
      replaceReplayFrame();
      requestAnimationFrame(() => {
        if (generation !== replayGeneration) return;
        replaceReplayFrame(replaySrc);
      });
    } else {
      replayPanel.classList.add('is-loading');
      postRendererSelection();
    }
    syncSelectionUI();
    if (scrollToReplay) requestAnimationFrame(() => replayPanel.scrollIntoView({behavior: 'auto', block: 'start'}));
    return true;
  }

  function onReplayFrameLoad() {
    watchReplaySize();
    postRendererSelection();
  }

  function replaceReplayFrame(src = null) {
    // Navigating a surviving iframe creates joint browser-history entries.
    // An initial navigation on a fresh frame does not steal Back/Forward from
    // the parent trial URL. Keep the same DOM identity and loading contract.
    const replacement = replayFrame.cloneNode(false);
    replayFrame.parentElement.style.removeProperty('height');
    replayFrame.parentElement.style.removeProperty('aspect-ratio');
    replacement.removeAttribute('src');
    if (src) replacement.src = src;
    replacement.addEventListener('load', onReplayFrameLoad);
    replayFrame.replaceWith(replacement);
    replayFrame = replacement;
  }

  function watchReplaySize() {
    replayHudObserver?.disconnect();
    const replayDocument = replayFrame?.contentDocument;
    const isComparisonReplay = replayFrame?.getAttribute('src')?.startsWith(COMPARISON_REPLAY_PATH);
    const replayStage = replayDocument?.querySelector('#stage');
    if (replayStage) {
      replayStage.style.aspectRatio = '16 / 9';
      replayStage.style.minHeight = '0';
    }
    // The comparison renderer owns all lane rows, names, close controls, and
    // its clock. The legacy single-policy HUD scrubber below would otherwise
    // hide the only lane and rewrite its status while the iframe is loading.
    if (isComparisonReplay) return;
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
  }

  function hideReplay({clearFrame = true} = {}) {
    replayGeneration += 1;
    replayHudObserver?.disconnect();
    replayHudObserver = null;
    replayPanel.hidden = true;
    replayPanel.classList.remove('is-loading');
    if (clearFrame) replaceReplayFrame();
    document.body.classList.remove('policy-replay-active');
    releasePolicyChapter();
    syncSelectionUI();
  }

  function closeReplay() {
    beginUrlInteraction();
    hideReplay({clearFrame: true});
    commitUrl();
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
      const isBest = policyBest(policy);
      const result = policyTerminalLabel(policy);
      const action = selectedPolicyIndex(policy) >= 0 ? 'Click to remove from comparison' : 'Click to add to comparison';
      tip.textContent = `Policy #${policyNumber(policy)} · ${isBest ? 'Best · ' : ''}${fmtSpeed(policyScore(policy))} m/s · ${result}${policy.replay_ready ? ` · ${action}` : ' · Replay unavailable'}`;
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
    if (narrowViewport.matches) {
      // The overview scrolls out of the mobile viewport; its negative bottom
      // must not become the reading position. Only visible chrome can obscure
      // the trace here (the sticky toolbar and, when open, replay panel).
      const visibleBottom = element => {
        const rect = element?.getBoundingClientRect();
        return rect && rect.bottom > 0 && rect.top < window.innerHeight ? rect.bottom : 0;
      };
      return Math.max(0, visibleBottom(nav),
        visibleBottom(document.querySelector('#mobile-trace-toolbar')),
        replayPanel.hidden ? 0 : visibleBottom(replayPanel)) + 20;
    }
    const controlsRect = pulse?.getBoundingClientRect();
    const traceRect = trace?.getBoundingClientRect();
    const overlaps = controlsRect && traceRect && controlsRect.right > traceRect.left && controlsRect.left < traceRect.right;
    return (overlaps ? controlsRect.bottom : nav?.getBoundingClientRect().bottom || 0) + 20;
  }

  function traceScrollRoot() {
    if (!stepsTarget || narrowViewport.matches) return null;
    return /^(auto|scroll)$/.test(getComputedStyle(stepsTarget).overflowY) ? stepsTarget : null;
  }

  function scrollTraceTarget(target, block) {
    const root = traceScrollRoot();
    if (!root) {
      target.scrollIntoView({behavior: 'auto', block});
      if (narrowViewport.matches) {
        const toolbarBottom = document.querySelector('#mobile-trace-toolbar')?.getBoundingClientRect().bottom || 0;
        const replayBottom = replayPanel.hidden ? 0 : replayPanel.getBoundingClientRect().bottom;
        const visibleTop = Math.max(toolbarBottom, replayBottom) + 10;
        const visibleBottom = window.innerHeight - 12;
        const targetRect = target.getBoundingClientRect();
        if (visibleBottom > visibleTop && (targetRect.top < visibleTop || targetRect.bottom > visibleBottom)) {
          const targetCenter = targetRect.top + targetRect.height / 2;
          window.scrollBy({top: targetCenter - (visibleTop + visibleBottom) / 2, behavior: 'auto'});
        }
      }
      return;
    }
    const rootRect = root.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const targetTop = root.scrollTop + targetRect.top - rootRect.top;
    const desired = block === 'center'
      ? targetTop - (root.clientHeight - targetRect.height) / 2
      : targetTop;
    root.scrollTo({top: clamp(desired, 0, root.scrollHeight - root.clientHeight), behavior: 'auto'});
  }

  function jumpToEpoch(epoch, forceScroll = false, exactStep = false, shouldScroll = true) {
    return jumpToStep(nearestStep(epoch), forceScroll, exactStep, shouldScroll);
  }

  function jumpToStep(step, forceScroll = false, exactStep = false, shouldScroll = true, historyMode = 'push') {
    const target = renderedStepNode(step);
    if (!target) return;
    if (historyMode) beginUrlInteraction();
    state.focusedStepId = step.step_id;
    const stepEpoch = Date.parse(step.timestamp || '');
    if (Number.isFinite(stepEpoch)) state.cursorEpoch = stepEpoch;
    if (state.selectedStepId === step.step_id && !forceScroll) {
      draw();
      if (historyMode) commitUrl(historyMode);
      return;
    }
    document.querySelector('.step.timeline-selected')?.classList.remove('timeline-selected');
    state.selectedStepId = step.step_id;
    target.classList.add('timeline-selected');
    const number = step.public_step_id ?? step.attempt_step_id;
    const origin = state.timeline?.clock?.origin_epoch_ms ?? Date.parse(state.trajectory?.steps?.[0]?.timestamp || '');
    status.textContent = `Step #${number} · ${fmtDuration(stepEpoch - origin)}`;
    if (shouldScroll) {
      const previousBehavior = document.documentElement.style.scrollBehavior;
      state.scrollSyncLocked = true;
      clearTimeout(state.scrollUnlockTimer);
      document.documentElement.style.scrollBehavior = 'auto';
      scrollTraceTarget(exactStep ? target : scrollTargetForStep(target), exactStep ? 'center' : 'start');
      document.documentElement.style.scrollBehavior = previousBehavior;
      state.scrollUnlockTimer = setTimeout(() => {
        state.scrollSyncLocked = false;
      }, 180);
    }
    draw();
    if (historyMode) commitUrl(historyMode);
  }

  function syncFromScroll() {
    state.scrollFrame = null;
    if (state.scrollSyncLocked || state.restoringUrl || performance.now() < state.restoreLockUntil || !state.timeline || !state.trajectory) return;
    const root = traceScrollRoot();
    const anchor = root ? root.getBoundingClientRect().top + 20 : traceAnchor();
    let current = state.trajectory.steps?.[0];
    const visitedNodes = new Set();
    for (const step of state.trajectory.steps || []) {
      const node = renderedStepNode(step);
      if (!node || visitedNodes.has(node)) continue;
      visitedNodes.add(node);
      // A grouped tool turn has one visible card. Passive reading follows its
      // first/displayed step; explicit URL restores can still select any raw ID.
      if (node.getBoundingClientRect().top <= anchor) current = step;
      else break;
    }
    const epoch = Date.parse(current?.timestamp || '');
    if (!Number.isFinite(epoch) || epoch === state.cursorEpoch) return;
    state.cursorEpoch = epoch;
    state.focusedStepId = current.step_id;
    const number = current.public_step_id ?? current.attempt_step_id;
    status.textContent = `Step #${number} · ${fmtDuration(epoch - state.timeline.clock.origin_epoch_ms)}`;
    draw();
    if (state.ready) {
      clearTimeout(urlScrollTimer);
      urlScrollTimer = setTimeout(() => commitUrl('replace'), 300);
    }
  }

  async function loadOverview(trajectory) {
    const generation = ++overviewGeneration;
    clearTimeout(urlScrollTimer);
    state.ready = false;
    state.restoringUrl = false;
    state.pendingRestore = null;
    state.focusedStepId = null;
    state.selectedStepId = null;
    state.trajectory = trajectory;
    state.runId = trajectory.run?.run_id || '';
    state.timeline = null;
    state.series = null;
    state.policies = [];
    state.policyHits = [];
    state.selectedPolicies = [];
    state.emphasizedPolicyHash = null;
    unmappedPolicyList?.replaceChildren();
    if (unmappedPolicies) unmappedPolicies.hidden = true;
    replayPanel.hidden = true;
    replaceReplayFrame();
    document.body.classList.remove('policy-replay-active', 'policy-selection-active');
    syncSelectionUI();
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
      const [result, policyResult] = await Promise.all([timelinePromise, policyPromise]);
      if (state.runId !== runId || generation !== overviewGeneration) return;
      if (result.error) {
        status.textContent = 'Utilization unavailable';
        canvas.setAttribute('aria-label', `Utilization unavailable: ${result.error.message}`);
      } else {
        state.timeline = result.data;
        state.series = buildSeries(state.timeline);
        status.textContent = 'Scroll the trace or select the chart';
      }
      state.cursorEpoch = Date.parse(trajectory.steps?.[0]?.timestamp || '') || state.timeline?.clock?.origin_epoch_ms;
      resize();
      const policies = policyResult.error ? [] : await Promise.all((policyResult.data?.policies || []).map(resolvePolicyReplay));
      if (state.runId !== runId || generation !== overviewGeneration) return;
      state.policies = orderPolicies(policies);
      renderPolicyReferences();
      state.ready = true;
      restoreUrlState();
      draw();
    } catch (error) {
      status.textContent = 'Utilization unavailable';
      canvas.setAttribute('aria-label', `Utilization unavailable: ${error.message}`);
    }
  }

  window.addEventListener('trajectory:loaded', event => loadOverview(event.detail.data));
  window.addEventListener('trajectory:loading', () => {
    overviewGeneration += 1;
    state.ready = false;
    state.restoringUrl = false;
    state.pendingRestore = null;
    clearTimeout(urlScrollTimer);
    hideReplay({clearFrame: true});
  });
  const restoreHistory = () => {
    const run = new URLSearchParams(location.search).get('run');
    if (run && state.runId && run !== state.runId) { location.reload(); return; }
    restoreUrlState();
  };
  window.addEventListener('popstate', restoreHistory);
  window.addEventListener('hashchange', restoreHistory);
  stepsTarget?.addEventListener('click', event => {
    if (event.target.closest('button,a,input,select,textarea')) return;
    const target = event.target.closest('.step');
    if (!target || !event.target.closest('summary')) return;
    const step = state.trajectory?.steps?.find(item => item.step_id === target.dataset.stepIds?.split(' ')[0]);
    if (step) jumpToStep(step, true, true, false);
  });
  document.querySelector('#chapter-nav')?.addEventListener('click', event => {
    const item = event.target.closest('.chapter-link')?.closest('.chapter-item');
    if (!item) return;
    const step = state.trajectory?.steps?.find(step => Number(step.public_step_id ?? step.attempt_step_id) === Number(item.dataset.startStep));
    if (step) jumpToStep(step, true, true, false);
  });
  const scheduleScrollSync = () => {
    if (state.scrollSyncLocked) {
      clearTimeout(state.scrollUnlockTimer);
      state.scrollUnlockTimer = setTimeout(() => {
        state.scrollSyncLocked = false;
      }, 180);
      return;
    }
    if (state.scrollFrame == null) state.scrollFrame = requestAnimationFrame(syncFromScroll);
  };
  window.addEventListener('scroll', scheduleScrollSync, {passive: true});
  stepsTarget?.addEventListener('scroll', scheduleScrollSync, {passive: true});
  canvas.addEventListener('pointermove', moveTip);
  canvas.addEventListener('pointerleave', () => {
    state.hoverEpoch = null;
    tip.hidden = true;
    canvas.style.cursor = 'crosshair';
    draw();
  });
  canvas.addEventListener('click', event => {
    const policy = policyAt(event.clientX, event.clientY);
    if (policy) {
      selectPolicy(policy);
      return;
    }
    const rect = canvas.getBoundingClientRect();
    state.hoverEpoch = null;
    tip.hidden = true;
    releasePolicyChapter();
    jumpToEpoch(epochFor(event.clientX - rect.left), true);
  });
  canvas.addEventListener('keydown', event => {
    if (['Enter', ' '].includes(event.key) && state.policies.length) {
      const nearest = state.policies
        .filter(policy => Number.isFinite(policyEpoch(policy)))
        .sort((left, right) => (
          Math.abs(policyEpoch(left) - state.cursorEpoch)
          - Math.abs(policyEpoch(right) - state.cursorEpoch)
        ))[0];
      if (nearest?.replay_ready) {
        event.preventDefault();
        selectPolicy(nearest);
      }
      return;
    }
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key) || !state.timeline) return;
    event.preventDefault();
    releasePolicyChapter();
    const steps = state.trajectory?.steps || [];
    const current = nearestStep(state.cursorEpoch)?.step_id;
    const index = Math.max(0, steps.findIndex(step => step.step_id === current));
    const next = steps[clamp(index + (event.key === 'ArrowRight' ? 1 : -1), 0, steps.length - 1)];
    if (next) jumpToEpoch(Date.parse(next.timestamp));
  });
  replayFrame?.addEventListener('load', onReplayFrameLoad);
  window.addEventListener('message', event => {
    if (event.source !== replayFrame?.contentWindow || event.origin !== location.origin) return;
    const messageGeneration = String(event.data?.replayGeneration ?? '');
    if (event.data?.type === 'g1:policies-ready' && messageGeneration === String(replayGeneration)) {
      postRendererSelection();
      return;
    }
    if (messageGeneration !== String(replayGeneration)) return;
    if (event.data?.type === 'g1:replay-layout') {
      const shell=replayFrame.parentElement,{mobile,height}=event.data;
      if(window.innerWidth>720||mobile!==true){shell.style.removeProperty('height');shell.style.removeProperty('aspect-ratio');return;}
      if(!Number.isFinite(height)||height<64||height>2000)return;
      shell.style.height=`${Math.ceil(height)}px`;shell.style.aspectRatio='auto';
      return;
    }
    if (event.data?.type === 'g1:policy-focused') {
      const focused = state.selectedPolicies.find(policy => captureIdForPolicy(policy) === event.data.captureId);
      if (focused) {
        const changed = state.emphasizedPolicyHash !== policyHash(focused);
        state.emphasizedPolicyHash = policyHash(focused);
        syncSelectionUI();
        if (changed && !state.restoringUrl) { beginUrlInteraction(); commitUrl(); }
      }
      return;
    }
    if (event.data?.type === 'g1:policy-remove') {
      const removed = state.selectedPolicies.find(policy => captureIdForPolicy(policy) === event.data.captureId);
      if (removed) removePolicySelection(removed, {navigate: false, openReplay: true});
      return;
    }
    if (event.data?.type === 'g1:policies-state') {
      const expected = state.selectedPolicies.map(policy => captureIdForPolicy(policy));
      const actual = (event.data.policies || []).map(policy => policy.captureId).filter(Boolean);
      if (!replayPanel.hidden && actual.length === expected.length && actual.every((id, index) => id === expected[index])) {
        watchReplaySize();
        replayPanel.classList.remove('is-loading');
        if (state.restoringUrl) finishUrlRestore();
      }
    }
    if (event.data?.type === 'g1:policies-error') {
      const message = event.data.message || 'The selected replay policies could not be loaded.';
      if (!replayPanel.hidden) {
        if (replayLoading) replayLoading.textContent = message;
      }
      announceSelection(message);
    }
  });
  replayClose?.addEventListener('click', closeReplay);
  mobileReplayClose?.addEventListener('click', closeReplay);
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
