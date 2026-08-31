"""Presentation precision and model ordering never alter underlying scores."""

from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "web/app.js").read_text()


def run_js(script: str) -> None:
    # Exercise the real English fallback with no i18n catalog installed. These
    # numerical/stop-code fixtures do not need to construct the whole page.
    helpers = "const window={};\n"
    helpers += next(line for line in APP.splitlines() if line.strip().startswith("const t=")) + "\n"
    helpers += "const ui=t;\n"
    result = subprocess.run(["node", "-e", helpers + script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("filename,name", [
    ("app.js", "fmtScore"),
    ("timeline.js", "fmtScore"),
    ("trajectory-overview.js", "fmtSpeed"),
])
def test_speed_formatters_round_to_two_decimal_places(filename: str, name: str) -> None:
    source = (ROOT / "web" / filename).read_text()
    formatter = re.search(rf"  const {name} = [^\n]+", source).group(0)
    run_js("const assert=require('node:assert/strict');\n"
           "const finite=value=>typeof value==='number'&&Number.isFinite(value);\n"
           + formatter + "\n"
           + f"const fmt={name};\n" + """
for(const [value,expected] of [[0,'0.00'],[.0049,'0.00'],[.005,'0.01'],[1.005,'1.01'],[1.23456,'1.23'],[3.641019,'3.64'],[10.101341,'10.10'],[10.075,'10.08']]){
  assert.equal(fmt(value),expected);
}
""")


def test_homepage_model_order_is_shared_by_tables_cards_and_charts() -> None:
    # Execute real rendering functions against small DOM stubs, including both
    # table modes and raw scores that round to the same display value.
    functions = APP[APP.index("  const MODEL ="):APP.index("  let chartResizeFrame")]
    run_js("const assert=require('node:assert/strict');\n" + """
let showAllTrials=false,showAllBudgetTrials=false;
const nodes=new Map();
const $=selector=>{if(!nodes.has(selector))nodes.set(selector,{innerHTML:'',addEventListener(){},querySelector(){return null},querySelectorAll(){return []}});return nodes.get(selector)};
const document={querySelectorAll(){return []}};
const trajectoryHref=run=>`/trajectory?run=${run}`;
""" + functions + r"""
const models=['GLM-5.3-Flash','DeepSeek-V4-Flash','GPT-5.6 Luna'];
const families=['glm','deepseek','luna'];
state.runs=models.flatMap((model,i)=>[1,2].map(trial=>({model,run_id:`${families[i]}-${trial}`,timeline:{comparison_summary:{final_api_cost_usd:.5,final_agent_total_cost_usd:10},clock:{origin_epoch_ms:1000,end_epoch_ms:2000}}})));
const runs=state.runs.map(run=>({...run,points:[{continuous_score_mps:run.run_id.endsWith('-2')?3.6419:3.6411}],summary:{}}));
state.performance={runs,models:models.map(model=>({model,points:runs.filter(run=>run.model===model).flatMap(run=>run.points),summary:{time_auc_mps_at_common_cap:1.2345}})),time:{common_auc_cap_hours:1}};
state.batch={status:'complete',arms:state.runs.map(run=>({model:run.model,run_id:run.run_id,trial:Number(run.run_id.slice(-1)),status:'finalized'}))};
const original=JSON.stringify(state.performance);
const expected=['DeepSeek-V4-Flash','GPT‑5.6 Luna','GLM‑5.3‑Flash'];
const checkOrder=html=>{const positions=expected.map(label=>html.indexOf(label));assert.ok(positions.every(value=>value>=0));assert.ok(positions[0]<positions[1]&&positions[1]<positions[2]);};
assert.deepEqual(orderedModels(state.performance.models).map(row=>family(row.model)),['deepseek','luna','glm']);
for(const expanded of [false,true]){
  showAllTrials=expanded;showAllBudgetTrials=expanded;
  renderCards();renderResources();renderExperimentTracker();renderPerformanceScores('#time-scores','time');
  for(const id of ['#model-cards','#resource-bars','#experiment-tracker','#time-scores'])checkOrder($(id).innerHTML);
  assert.match($('#model-cards').innerHTML,/3\.64 m\/s/);
  assert.match($('#model-cards').innerHTML,/3\.64 metres per second/);
  assert.match($('#experiment-tracker').innerHTML,/3\.64 m\/s/);
  // Both raw scores display as3.64; trial2 must still win on its raw value.
  assert.ok($('#experiment-tracker').innerHTML.indexOf('deepseek-2') < (expanded?$('#experiment-tracker').innerHTML.indexOf('deepseek-1'):Infinity));
}
assert.equal(JSON.stringify(state.performance),original);
""")


def test_speed_axes_and_tooltips_use_display_formatter() -> None:
    assert "label.textContent=fmtScore(value)" in APP
    assert "effective speed {speed} m/s" in APP
    assert "speed:fmtScore(row.continuous_score_mps)" in APP
    assert "${fmtSpeed(policyScore(policy))} m/s" in (ROOT / "web/trajectory-overview.js").read_text()
    assert "const performanceModels=orderedModels(state.performance?.models||[]);const legend=" in APP


@pytest.mark.parametrize("filename", ["app.js", "timeline.js"])
def test_frontend_translates_stop_codes_without_changing_them(filename: str) -> None:
    source = (ROOT / "web" / filename).read_text()
    formatter = re.search(r"  const stopLabel = [^\n]+", source).group(0)
    run_js("const assert=require('node:assert/strict');\n" + formatter + "\n" + """
for(const code of ['in_lane','lane','lane_exit','left lane'])assert.equal(stopLabel(code),'LANE DRIFT');
for(const code of ['self_collision','self-collision','self collision','collide'])assert.equal(stopLabel(code),'COLLISION');
assert.equal(stopLabel('timeout'),'TIMEOUT');
assert.equal(stopLabel('disqualified'),'STOPPED');
assert.equal(stopLabel('DQ'),'STOPPED');
assert.equal(stopLabel('unknown_reason'),'STOPPED');
""")
    assert "stopLabel(point.termination_reason" in source or "stopLabel(row.termination_reason)" in source


def test_trajectory_terminal_labels_describe_stops_without_disqualification() -> None:
    source = (ROOT / "web/trajectory-overview.js").read_text()
    function = source[source.index("  function policyTerminalLabel("):source.index("  function policyNumber(")]
    run_js("const assert=require('node:assert/strict');\n"
           "const policyFinished=policy=>policy.termination_reason==='finished';\n"
           + function + """
for(const [policy,label] of [
  [{termination_reason:'finished'},'FINISHED'],
  [{termination_reason:'in_lane'},'LANE DRIFT'],
  [{termination_reason:'self-collision'},'COLLISION'],
  [{termination_reason:'timeout',failed_gates:['self_collision']},'TIMEOUT'],
  [{failed_gates:['self_collision']},'COLLISION'],
  [{failed_gates:['in_lane']},'LANE DRIFT'],
  [{termination_reason:'disqualified'},'STOPPED']
])assert.equal(policyTerminalLabel(policy),label);
""")
    assert "Disqualified" not in source
    assert "const result = policyTerminalLabel(policy);" in source
