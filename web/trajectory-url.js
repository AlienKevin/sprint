/* Run-scoped, shareable trajectory state. Also imported by Node regression tests. */
((root) => {
  const MAX_POLICIES = 8;
  const positiveInteger = value => /^\d+$/.test(String(value ?? '')) && Number.isSafeInteger(Number(value)) && Number(value) > 0 ? Number(value) : null;
  function parse(value) {
    const url = new URL(value, 'http://localhost');
    const errors = [], policies = [];
    for (const token of (url.searchParams.get('policies') || '').split(',').filter(Boolean)) {
      const number = positiveInteger(token);
      if (number === null) { errors.push(`Invalid policy ${token}.`); continue; }
      if (policies.includes(number)) continue;
      if (policies.length === MAX_POLICIES) { errors.push('Only the first 8 policies can be compared.'); continue; }
      policies.push(number);
    }
    let anchor = '';
    try { anchor = decodeURIComponent(url.hash.slice(1)); } catch { errors.push('Invalid turn anchor.'); }
    return {
      runId: url.searchParams.get('run') || '', policies,
      focus: positiveInteger(url.searchParams.get('focus')),
      step: url.searchParams.get('step') || (anchor.startsWith('a') ? anchor : ''),
      turn: positiveInteger(url.searchParams.get('turn')),
      replayVisible: url.searchParams.get('replay') !== '0', errors,
    };
  }
  function build({runId, policies = [], focus = null, step = '', turn = null, replayVisible = true}) {
    const params = new URLSearchParams();
    if (runId) params.set('run', String(runId));
    const selected = [...new Set(policies.map(positiveInteger).filter(Boolean))].slice(0, MAX_POLICIES);
    if (selected.length) params.set('policies', selected.join(','));
    if (selected.includes(positiveInteger(focus))) params.set('focus', String(Number(focus)));
    if (step) params.set('step', String(step));
    else if (positiveInteger(turn)) params.set('turn', String(Number(turn)));
    if (!replayVisible) params.set('replay', '0');
    return `/trajectory${params.size ? `?${params}` : ''}`;
  }
  function resolve(request, {runId, policies = [], steps = []}) {
    const errors = [...request.errors];
    if (request.runId && request.runId !== runId) return {policies: [], focus: null, step: null, replayVisible: false, errors: ['This link refers to a different trial.']};
    const selected = [];
    for (const number of request.policies) {
      const matches = policies.filter(policy => Number(policy.submission_index) === number);
      if (matches.length !== 1 || !matches[0].replay_ready || !matches[0].replay_url) {
        errors.push(`Policy #${number} is unavailable in this trial.`); continue;
      }
      selected.push(matches[0]);
    }
    let step = null;
    if (request.step) step = steps.find(item => item.step_id === request.step) || null;
    else if (request.turn != null) step = steps.find(item => Number(item.public_step_id ?? item.attempt_step_id) === request.turn) || null;
    if ((request.step || request.turn != null) && !step) errors.push('The requested turn is unavailable in this trial.');
    const focus = selected.find(policy => Number(policy.submission_index) === request.focus) || selected.at(-1) || null;
    if (request.focus != null && !selected.some(policy => Number(policy.submission_index) === request.focus)) errors.push('The focused policy is not part of the available selection.');
    return {policies: selected, focus, step, replayVisible: request.replayVisible && selected.length > 0, errors};
  }
  const api = {parse, build, resolve, MAX_POLICIES};
  root.TrajectoryURL = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
