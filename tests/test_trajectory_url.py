from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trajectory_url_contract_round_trips_independent_turn_and_selection() -> None:
    script = "const assert=require('node:assert/strict');\n"
    script += f"const api=require({json.dumps(str(ROOT / 'web/trajectory-url.js'))});\n"
    script += """
const ready=n=>({submission_index:n,policy_sha256:String(n),replay_ready:true,replay_url:`/replay/capture${n}`});
const policies=[ready(13),ready(32),{...ready(10),replay_ready:false}];
const steps=[{step_id:'a1-s593',public_step_id:590},{step_id:'a1-s651',public_step_id:648}];
const data={runId:'luna-trial',policies,steps};
const link=api.build({runId:data.runId,policies:[13,32],focus:13,step:'a1-s651'});
const parsed=api.parse(link);
assert.deepEqual(parsed.policies,[13,32]);assert.equal(parsed.focus,13);assert.equal(parsed.step,'a1-s651');
let resolved=api.resolve(parsed,data);
assert.deepEqual(resolved.policies,policies.slice(0,2));assert.equal(resolved.focus,policies[0]);assert.equal(resolved.step,steps[1]);assert.equal(resolved.replayVisible,true);
for(const suffix of ['&turn=648','#a1-s651','&step=a1-s651']){
  resolved=api.resolve(api.parse('/trajectory?run=luna-trial&policies=13'+suffix),data);
  assert.equal(resolved.step,steps[1]);
}
resolved=api.resolve(api.parse(api.build({runId:data.runId,policies:[13,32],focus:32,step:'a1-s651',replayVisible:false})),data);
assert.equal(resolved.replayVisible,false);assert.equal(resolved.policies.length,2);assert.equal(resolved.focus,policies[1]);
resolved=api.resolve(api.parse('/trajectory?run=luna-trial&step=a1-s651'),data);
assert.equal(resolved.replayVisible,false);assert.equal(resolved.policies.length,0);assert.equal(resolved.step,steps[1]);
resolved=api.resolve(api.parse('/trajectory?run=other-run&policies=13'),data);
assert.equal(resolved.policies.length,0);assert.equal(resolved.step,null);assert.match(resolved.errors[0],/different trial/);
resolved=api.resolve(api.parse('/trajectory?run=luna-trial&policies=999,10,13,13,frontier-deadbeef0000&focus=32&step=absent'),data);
assert.deepEqual(resolved.policies,[policies[0]]);assert.equal(resolved.step,null);assert.ok(resolved.errors.length>=4);
const many=api.parse('/trajectory?policies=1,2,3,4,5,6,7,8,9,10,NaN,-1,0,9007199254740992');
assert.deepEqual(many.policies,[1,2,3,4,5,6,7,8]);assert.ok(many.errors.length);
assert.equal(api.parse('/trajectory#%zz').step,'');
assert.equal(api.parse(api.build({runId:'run with spaces/strange?value',policies:[13]})).runId,'run with spaces/strange?value');
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_async_trial_load_never_publishes_a_stale_or_wrong_run_response() -> None:
    source = (ROOT / "web/trajectory.js").read_text()
    function = source[source.index("  async function loadRun("):source.index("  async function init(")]
    script = """
const assert=require('node:assert/strict');
const state={};let runLoadGeneration=0;
const pending=new Map(),published=[];
const fetch=path=>new Promise(resolve=>pending.set(path,resolve));
const stepsTarget={replaceChildren(){}};const el=()=>({});
const window={dispatchEvent(event){if(event.type==='trajectory:loaded')published.push(event.detail.data.run.run_id)}};
const CustomEvent=class{constructor(type,{detail}){this.type=type;this.detail=detail}};
const renderSummary=()=>{};const sha256=async()=>'';
const response=run=>({ok:true,text:async()=>JSON.stringify({run:{run_id:run}})});
const emptyOutline={ok:false};
""" + function + """
(async()=>{
  const old=loadRun('/old','old'),fresh=loadRun('/fresh','fresh');
  pending.get('/fresh')(response('fresh'));pending.get('/data/trajectories/fresh.outline.json')(emptyOutline);
  await fresh;
  pending.get('/old')(response('old'));pending.get('/data/trajectories/old.outline.json')(emptyOutline);
  await old;
  assert.deepEqual(published,['fresh']);assert.equal(state.data.run.run_id,'fresh');
  const wrong=loadRun('/wrong','requested');
  pending.get('/wrong')(response('other'));pending.get('/data/trajectories/requested.outline.json')(emptyOutline);
  await assert.rejects(wrong,/does not match/);assert.deepEqual(published,['fresh']);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
