"""Hero scores use native links to the same best trials as the results table."""

import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "web/app.js").read_text()


@pytest.mark.parametrize("use_url_builder", [True, False])
def test_whole_hero_score_links_to_matching_best_trial(use_url_builder: bool) -> None:
    functions = APP[APP.index("  const MODEL ="):APP.index("  let chartResizeFrame")]
    script = "const assert=require('node:assert/strict');\n"
    script += f"const api=require({json.dumps(str(ROOT / 'web/trajectory-url.js'))});\n"
    script += f"const window={{TrajectoryURL:{'api' if use_url_builder else 'null'}}};\n"
    # Select helpers by declaration, not source line position: localization adds
    # an independent helper before the URL builder.
    script += "\n".join(line for line in APP.splitlines()
                        if line.strip().startswith(("const t=", "const trajectoryHref="))) + "\n"
    script += """
let showAllTrials=false,showAllBudgetTrials=false;
const nodes=new Map();
const $=selector=>{
  if(!nodes.has(selector))nodes.set(selector,{innerHTML:'',querySelector(){return null},querySelectorAll(){return []}});
  return nodes.get(selector);
};
""" + functions + r"""
const models={deepseek:'DeepSeek-V4-Flash',luna:'GPT-5.6 Luna',glm:'GLM-5.3-Flash'};
const makeRun=(key,trial,score,summaryScore=score)=>({
  model:models[key],run_id:`${key} trial/${trial}&viewer`,trial,
  points:[{continuous_score_mps:score}],summary:{best_continuous_score_mps:summaryScore},
  timeline:{clock:{origin_epoch_ms:1000,end_epoch_ms:2000}}
});
// Tiny raw differences must choose trial 2 even though both display as 3.64.
// Equal scores use the lower original trial number, not fixture order.
// Summary-only maxima count too; no fixture includes an archived replay.
state.runs=[makeRun('deepseek',1,3.6411),makeRun('deepseek',2,3.6419),
  makeRun('luna',4,5),makeRun('luna',2,5),makeRun('glm',1,1),makeRun('glm',2,0,6)];
state.performance={runs:state.runs,models:Object.entries(models).map(([key,model])=>({model,points:[{continuous_score_mps:99}]}))};
state.batch={status:'complete',arms:state.runs.map(run=>({...run,status:'finalized'}))};
const original=JSON.stringify(state);
const expected={deepseek:state.runs[1],luna:state.runs[3],glm:state.runs[5]};
let initialLinks;
for(const expanded of [false,true,false]){
  showAllTrials=expanded;
  renderExperimentTracker();renderCards();
  const cards=$('#model-cards').innerHTML,table=$('#experiment-tracker').innerHTML;
  const links=[...cards.matchAll(/<a class="hero-score-row ([^"]+)" href="([^"]+)"([^>]*)>([\s\S]*?)<\/a>/g)];
  assert.equal(links.length,3);
  assert.deepEqual(links.map(match=>match[1]),['deepseek','luna','glm']);
  for(const [,key,href,attributes,contents] of links){
    const tableHref=table.match(new RegExp(`<tr class="experiment-row ${key} experiment-best"[^>]*data-run-href="([^"]+)"`))[1];
    assert.equal(href,tableHref);
    assert.equal(api.parse(href).runId,expected[key].run_id);
    assert.deepEqual(api.parse(href).policies,[]);
    assert.match(attributes,/aria-label="Open .+ best trial trajectory, Effective Speed/);
    assert.doesNotMatch(attributes,/onclick|role="button"|tabindex="-1"/);
    assert.match(contents,/<span><strong>.+<\/strong><\/span>/);
    assert.match(contents,/<div class="hero-score-track"/);
    assert.match(contents,/<i style="width:/);
    assert.match(contents,/<b>[0-9.]+ m\/s<\/b>/);
    assert.doesNotMatch(contents,/<a\b|<button\b/);
  }
  assert.match(cards,/>3\.64 m\/s<\/b>/);
  assert.match(cards,/>6\.00 m\/s<\/b>/);
  assert.doesNotMatch(cards,/>99\.00 m\/s<\/b>/);
  const currentLinks=links.map(match=>match[2]);
  if(initialLinks)assert.deepEqual(currentLinks,initialLinks);
  initialLinks=currentLinks;
}
assert.equal(JSON.stringify(state),original);
// Without performance or replay data, use the table's zero-score trial fallback.
state.performance=null;
renderExperimentTracker();renderCards();
assert.equal(api.parse($('#model-cards').innerHTML.match(/href="([^"]+)"/)[1]).runId,state.runs[0].run_id);
assert.match($('#model-cards').innerHTML,/>0\.00 m\/s<\/b>/);
// The no-batch path still has a valid run-only link; empty data has none.
state.batch=null;state.runs=[{model:models.deepseek,run_id:'fallback-trial-1'}];
renderCards();assert.match($('#model-cards').innerHTML,/href="\/trajectory\?run=fallback-trial-1"/);
state.runs=[];renderCards();assert.doesNotMatch($('#model-cards').innerHTML,/<a\b/);
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
