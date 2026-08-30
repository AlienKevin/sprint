// Compact raw benchmark output for review; excludes large request/resource payloads.
// Usage: node replay_performance_summarize.mjs <output.json> <stage> [stage...]
import fs from 'node:fs';

const [output, ...stages] = process.argv.slice(2);
if (!output || !stages.length) throw new Error('Provide output JSON path and stage names.');
const read = name => JSON.parse(fs.readFileSync(`/tmp/replay-${name}.json`, 'utf8'));
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const result = {
  schema: 2,
  environment: 'Chromium152 realWebGL ANGLE SwiftShader, shared VM',
  viewport: {width: 1280, height: 900, dpr: 1},
  encoding: 'Brotli quality5',
  stages: stages.map(stage => {
    const raw = read(`${stage}-metrics`);
    const samples = raw.samples.map(row => {
      if (!Number.isFinite(row.readyMs)) throw new Error(`Missing readiness: ${stage}/${row.name}`);
      return {
        name: row.name,
        readyMs: Number(row.readyMs.toFixed(1)),
        wireBytes: row.wireBytes,
        requests: row.requests.length,
        zeroWireRequests: row.requests.filter(request => !request.encodedDataLength).length,
        iframeSame: row.iframeSame,
        apiSame: row.apiSame,
        canvasSame: row.canvasSame,
        playback: row.playback,
        lanes: row.lanes,
        errors: row.errors,
        ...(row.frameErrors ? {frameErrors: row.frameErrors} : {}),
        ...(row.cache ? {cache: row.cache} : {}),
        ...(row.diagnostics ? {renderer: {
          activeActors: row.diagnostics.activeActors,
          disposedActors: row.diagnostics.disposedActors,
          gpu: row.diagnostics.gpu,
        }} : {}),
      };
    });
    const medians = {};
    for (const row of samples) {
      const [scenario, , action] = row.name.split('/');
      const key = `${scenario}/${action}`;
      if (medians[key]) continue;
      const group = samples.filter(sample => {
        const [s, , a] = sample.name.split('/');
        return s === scenario && a === action;
      });
      medians[key] = {
        readyMs: median(group.map(sample => sample.readyMs)),
        wireBytes: median(group.map(sample => sample.wireBytes)),
        requests: median(group.map(sample => sample.requests)),
      };
    }
    return {stage, network: raw.network, started: raw.started, finished: raw.finished, medians, samples};
  }),
};
if (fs.existsSync('/tmp/replay-priority3-correctness.json')) {
  const raw = read('priority3-correctness');
  result.persistentCorrectness = {
    checks: raw.checks.map(({name, pass}) => ({name, pass})),
    heapBeforeChurn: raw.heapBeforeChurn,
    heapAfterChurn: raw.heapAfterChurn,
    churn: raw.churn.map(({cycle, heap, state}) => ({
      cycle, heap, sameFrame: state.sameFrame, sameApi: state.sameApi,
      sameCanvas: state.sameCanvas, gpu: state.diagnostics.gpu,
      activeActors: state.diagnostics.activeActors, cache: state.cache,
    })),
  };
}
if (fs.existsSync('/tmp/replay-baseline-quality.json')) {
  const baseline = read('baseline-quality');
  result.fixedSeekQuality = [...new Set(stages.map(stage => stage.replace(/-4mbps$/, '')))]
    .filter(stage => fs.existsSync(`/tmp/replay-${stage}-quality.json`))
    .map(stage => {
      const quality = read(`${stage}-quality`);
      return {
        stage,
        samples: quality.samples.map((sample, index) => ({
          name: sample.name,
          time: sample.time,
          signatureEqualsBaseline: JSON.stringify(sample.detail) === JSON.stringify(baseline.samples[index]?.detail),
        })),
      };
    });
}
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
for (const stage of result.stages) console.log(JSON.stringify({stage: stage.stage, medians: stage.medians}));
console.log(output);
