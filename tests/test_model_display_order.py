"""Model order is editorial, never determined by score or incoming data order."""

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_static_model_sections_follow_deepseek_luna_glm() -> None:
    page = (ROOT / "web/index.html").read_text()
    harnesses = re.search(r'<div class="harnesses-copy"[^>]*>([\s\S]*?)</div>', page).group(1)
    assert re.findall(r'<strong>([^<]+)</strong>', harnesses) == [
        'DeepSeek-V4-Flash', 'GPT-5.6 Luna', 'GLM-5.3-Flash',
    ]
    observations = re.search(r'<section[^>]+id="observations">([\s\S]*?)</section>', page).group(1)
    cards = re.findall(r'<article>([\s\S]*?)</article>', observations)
    assert [re.search(r'<h3>([^<]+)</h3>', card).group(1) for card in cards] == ['DeepSeek', 'Luna', 'GLM']
    assert [re.search(r'<iframe src="([^"]+)"', card).group(1) for card in cards] == [
        '/replay/frontier-76d7d31f8c51?example=1&amp;autoplay=0',
        '/replay/frontier-dd28af63bf84?example=1&amp;autoplay=0',
        '/replay/frontier-5eb6fee88ef0?example=1&amp;autoplay=0',
    ]


def test_homepage_groups_and_legends_ignore_input_order_and_model_scores() -> None:
    app = (ROOT / 'web/app.js').read_text()
    functions = app[app.index('  const MODEL ='):app.index('  let chartResizeFrame')]
    script = "const assert=require('node:assert/strict');\n"
    script += "\n".join(line for line in app.splitlines()
                        if line.strip().startswith(("const t=", "const trajectoryHref="))) + "\n"
    script += """
const window={},document={querySelectorAll(){return []}};
let showAllTrials=false,showAllBudgetTrials=false;
const nodes=new Map(),$=selector=>{
  if(!nodes.has(selector))nodes.set(selector,{innerHTML:'',textContent:'',addEventListener(){},querySelector(){return null},querySelectorAll(){return []}});
  return nodes.get(selector);
};
""" + functions + r"""
const expected=['deepseek','luna','glm'];
const makeRun=(key,trial,score)=>({run_id:`${key}-${trial}`,model:{deepseek:'deepseek/deepseek-v4-flash',luna:'openai/gpt-5.6-luna',glm:'z-ai/glm-5.3-flash'}[key],points:[{continuous_score_mps:score}],summary:{best_continuous_score_mps:score},timeline:{comparison_summary:{final_api_cost_usd:1},clock:{origin_epoch_ms:1000,end_epoch_ms:2000}}});
state.runs=[makeRun('glm',1,10),makeRun('luna',1,3),makeRun('deepseek',1,1),makeRun('glm',2,11)];
state.batch={status:'complete',arms:state.runs.map(run=>({...run,trial:Number(run.run_id.slice(-1)),status:'finalized'}))};
state.performance={runs:state.runs,models:[state.runs[0],state.runs[1],state.runs[2]],cost:{common_auc_cap_usd:10},time:{common_auc_cap_hours:10}};
state.pricing={basis:'undiscounted_list_price',as_of:'2026-08-30',models:Object.fromEntries(state.runs.map(run=>[family(run.model),{model:run.model,cached_input:.1,output:1,source_url:`https://example.com/${family(run.model)}`}]))};
assert.deepEqual(orderedModels(state.performance.models).map(row=>family(row.model)),expected);
assert.deepEqual(state.performance.models.map(row=>family(row.model)),['glm','luna','deepseek']);
renderCharts=()=>{};
for(const expanded of [false,true]){
  showAllTrials=expanded;showAllBudgetTrials=expanded;render();
  assert.deepEqual([...$('#model-cards').innerHTML.matchAll(/class="hero-score-row ([^"]+)"/g)].map(match=>match[1]),expected);
  assert.deepEqual([...$('#experiment-tracker').innerHTML.matchAll(/class="experiment-group ([^"]+)"/g)].map(match=>match[1]),expected);
  assert.deepEqual([...$('#resource-bars').innerHTML.matchAll(/class="budget-group ([^"]+)"/g)].map(match=>match[1]),expected);
  assert.deepEqual([...$('#resource-bars').innerHTML.matchAll(/href="https:\/\/example.com\/([^"]+)"/g)].map(match=>match[1]),expected);
  assert.deepEqual([...$('#time-scores').innerHTML.matchAll(/class="auc-bar-row ([^"]+)"/g)].map(match=>match[1]),expected);
  for(const target of ['#cost-legend','#time-legend']){
    const legend=$(target).innerHTML;
    assert.ok(legend.indexOf('DeepSeek')<legend.indexOf('Luna'));
    assert.ok(legend.indexOf('Luna')<legend.indexOf('GLM'));
  }
}
"""
    result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
