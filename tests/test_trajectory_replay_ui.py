from __future__ import annotations

import subprocess
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "web/trajectory.html").read_text()
OVERVIEW = (ROOT / "web/trajectory-overview.js").read_text()


def test_viewer_navigation_has_no_redundant_slash() -> None:
    nav = HTML.split('<header class="viewer-nav">', 1)[1].split('</header>', 1)[0]
    assert '<a class="brand" href="/">Agents\' <em>100m</em></a>' in nav
    assert 'id="viewer-trial-model">Agent trajectory</span>' in nav
    assert 'id="viewer-trial-number"' in nav
    assert 'nav-slash' not in nav


def overview_function(name: str) -> str:
    start = OVERVIEW.index(f"  function {name}(")
    following = re.search(r"\n  (?:async )?function ", OVERVIEW[start + 1:])
    assert following is not None
    return OVERVIEW[start:start + 1 + following.start()]


def run_node(script: str) -> None:
    script = "const ui=(source,params={})=>source.replace(/\\{(\\w+)\\}/g,(match,key)=>params[key]??match);const setUI=(node,source,params)=>{node.textContent=ui(source,params)};\n" + script
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_playhead_cap_starts_at_gpu_row_boundary_in_both_chart_layouts() -> None:
    script = """
const assert=require('node:assert/strict');
const canvas={clientWidth:500},colors={muted:'muted',line:'line',tools:'tools',text:'text'};
const compactLanes=[{key:'training',label:'GPU',color:'green'},{key:'cpu',label:'CPU',color:'blue'}];
const state={docked:false,timeline:{clock:{origin_epoch_ms:0,end_epoch_ms:1000}},
  series:{training:[],cpu:[]},hoverEpoch:null,cursorEpoch:400};
const chartHeight=()=>state.docked?94:126,bounds=()=>({x0:82,x1:488});
const xFor=epoch=>82+epoch/1000*406,accentColor=()=>'',fmtDuration=()=>'',
  clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
let paths=[],path=[],arcs=[],resourceLines=[],labels=[];
const ctx={clearRect(){},beginPath(){path=[]},moveTo(x,y){path.push(['move',x,y])},
  lineTo(x,y){path.push(['line',x,y])},stroke(){paths.push({color:this.strokeStyle,path:[...path]})},
  arc(...args){arcs.push(args)},fill(){},fillRect(){},
  fillText(text,x,y){labels.push({text,y})},measureText(){return {width:10}}};
const drawPolicyLegend=()=>{},drawPolicies=()=>{};
const drawLine=(series,y,height,color)=>resourceLines.push({y,height,color});
""" + overview_function("draw") + """
for(const docked of [false,true]){
  state.docked=docked;
  for(const hover of [false,true]){
    state.hoverEpoch=hover?700:null;
    paths=[];arcs=[];resourceLines=[];labels=[];
    draw();
    const marker=paths.find(item=>item.color.startsWith('#ffffff'));
    assert.ok(marker,'both hover and selected-turn cursors are drawn');
    assert.equal(arcs.length,1);
    const [x,centerY,radius]=arcs[0];
    const gpuRowTop=resourceLines[0].y-3;
    assert.equal(centerY-radius,gpuRowTop,'white cap must not extend into the legend');
    assert.equal(gpuRowTop,docked?15:18);
    assert.deepEqual(marker.path,[['move',x,centerY],['line',x,chartHeight()-18]],
      'line connects to the cap and retains its existing bottom');
    assert.equal(x,xFor(hover?700:400),'hover/selected epoch mapping is unchanged');
    assert.equal(marker.color,hover?'#ffffffaa':'#ffffff70');
    assert.deepEqual(resourceLines.map(line=>line.y),docked?[18,48]:[21,65]);
    assert.deepEqual(labels.slice(0,4).map(label=>label.text),['GPU','TOOL CALLS','CPU','SUBMISSIONS']);
  }
}
state.hoverEpoch=null;state.cursorEpoch=null;arcs=[];
draw();assert.deepEqual(arcs,[],'no marker without a hover or selected turn');
"""
    run_node(script)


def test_passive_scroll_debounces_replace_without_adding_history_entries() -> None:
    script = "const assert=require('node:assert/strict');\n"
    script += f"const urlApi=require({json.dumps(str(ROOT / 'web/trajectory-url.js'))});\n"
    script += """
const steps=[1,2,3].map(n=>({step_id:`a1-s${n}`,public_step_id:n,timestamp:`2026-08-28T00:00:0${n}Z`}));
const state={ready:true,restoringUrl:false,restoreLockUntil:0,runId:'trial',selectedPolicies:[],
  trajectory:{steps},timeline:{clock:{origin_epoch_ms:0}},cursorEpoch:0};
const window={TrajectoryURL:urlApi},replayPanel={hidden:true},status={};
const location={pathname:'/trajectory',search:'?run=trial',hash:''};
const events=[];
const update=(kind,url)=>{events.push(kind);const next=new URL(url,'http://localhost');location.search=next.search;location.hash=next.hash};
const history={replaceState:(s,t,url)=>update('replace',url),pushState:(s,t,url)=>update('push',url)};
const policyHash=p=>p.hash,policyNumber=p=>p.submission_index;
let urlScrollTimer=null,nextTimer=0,activeIndex=1;
const timers=new Map();
const clearTimeout=id=>timers.delete(id);
const setTimeout=(callback,delay)=>{assert.equal(delay,300);timers.set(++nextTimer,callback);return nextTimer};
const flush=()=>{for(const [id,callback] of [...timers]){timers.delete(id);callback()}};
const performance={now:()=>1000},traceScrollRoot=()=>null,traceAnchor=()=>20;
const renderedStepNode=step=>({getBoundingClientRect:()=>({top:steps.indexOf(step)<=activeIndex?10:100})});
const fmtDuration=()=>'',draw=()=>{};
""" + overview_function("commitUrl") + overview_function("syncFromScroll") + """
syncFromScroll();activeIndex=2;syncFromScroll();
assert.equal(timers.size,1);assert.deepEqual(events,[]);
flush();assert.deepEqual(events,['replace']);assert.match(location.search,/step=a1-s3/);
commitUrl('replace');assert.deepEqual(events,['replace'],'identical state adds no history');
state.restoringUrl=true;activeIndex=0;syncFromScroll();commitUrl();
assert.equal(state.focusedStepId,'a1-s3');assert.deepEqual(events,['replace']);
state.restoringUrl=false;state.restoreLockUntil=2000;syncFromScroll();
assert.equal(state.focusedStepId,'a1-s3');assert.equal(timers.size,0);
state.restoreLockUntil=0;state.focusedStepId='a1-s1';commitUrl();
assert.deepEqual(events,['replace','push'],'explicit navigation gets one meaningful entry');
"""
    run_node(script)


def test_mobile_scroll_anchor_ignores_offscreen_overview_and_tracks_visible_turn() -> None:
    script = """
const assert=require('node:assert/strict');
const rect=(top,bottom)=>({top,bottom,left:12,right:293});
const pulse={getBoundingClientRect:()=>rect(-4437,-4215)};
const trace={getBoundingClientRect:()=>rect(-4100,10000)};
const nav={getBoundingClientRect:()=>rect(0,58)};
const toolbar={getBoundingClientRect:()=>rect(0,46)};
const replayPanel={hidden:false,getBoundingClientRect:()=>rect(46,252.9375)};
const narrowViewport={matches:true},window={innerHeight:844};
const document={querySelector:()=>toolbar};
""" + overview_function("traceAnchor") + """
assert.equal(traceAnchor(),272.9375);
const steps=[73,75,76,77,78,79].map(n=>({step_id:`a1-s${n}`,public_step_id:n,timestamp:`2026-08-28T00:${n-60}:00Z`}));
const tops=[-100,241.65,339,419];
const nodes=tops.map((_,index)=>({getBoundingClientRect:()=>({top:tops[index]})}));
const nodeIndices=[0,1,2,2,2,3];
const state={ready:true,restoringUrl:false,restoreLockUntil:0,trajectory:{steps},timeline:{clock:{origin_epoch_ms:0}},cursorEpoch:Date.parse(steps[0].timestamp)};
const performance={now:()=>1000},traceScrollRoot=()=>null;
const renderedStepNode=step=>nodes[nodeIndices[steps.indexOf(step)]];
const status={},fmtDuration=()=>'',draw=()=>{};
let urlScrollTimer=null,committed=null;
const clearTimeout=()=>{},setTimeout=callback=>{callback();return 1};
const commitUrl=mode=>{committed={mode,step:state.focusedStepId}};
""" + overview_function("syncFromScroll") + """
syncFromScroll();assert.deepEqual(committed,{mode:'replace',step:'a1-s75'});
assert.match(status.textContent,/Step #75/);
tops[2]=250;syncFromScroll();
assert.deepEqual(committed,{mode:'replace',step:'a1-s76'},'grouped card follows visible primary step, not hidden tool step78');
replayPanel.hidden=true;assert.equal(traceAnchor(),78);
narrowViewport.matches=false;
assert.equal(traceAnchor(),-4195,'desktop legacy anchor remains unchanged; desktop uses internal scroll root');
"""
    run_node(script)


def test_replay_header_has_single_title_and_close_without_duplicate_selection_row() -> None:
    assert 'id="trajectory-policy-replay-title"' in HTML
    assert 'id="trajectory-policy-replay-close"' in HTML
    assert 'id="replay-policy-selection"' not in HTML
    assert 'id="trajectory-policy-replay-meta"' not in HTML


def test_mobile_outline_toolbar_does_not_repeat_chapter_title() -> None:
    assert 'id="mobile-outline-toggle"' in HTML
    assert 'id="mobile-active-chapter"' not in HTML


def test_replay_loader_waits_for_matching_renderer_state() -> None:
    assert "replayPanel.classList.add('is-loading')" in OVERVIEW
    assert "replayPanel.classList.remove('is-loading')" in OVERVIEW
    assert "g1:policies-state" in OVERVIEW
    assert "replayGeneration: replayGeneration" in OVERVIEW
    assert "actual.every((id, index) => id === expected[index])" in OVERVIEW
    assert "replayGeneration += 1" in OVERVIEW


def test_comparison_replay_keeps_renderer_owned_lane_hud() -> None:
    assert "const isComparisonReplay = replayFrame?.getAttribute('src')?.startsWith(COMPARISON_REPLAY_PATH);" in OVERVIEW
    guard = "if (isComparisonReplay) return;"
    assert guard in OVERVIEW
    assert OVERVIEW.index(guard) < OVERVIEW.index("querySelectorAll('.lc .nm')")


def test_trajectory_chips_and_main_badge_use_outcome_only_reasons() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    terminal = scene.split("function terminalStatus(policy,done){", 1)[1].split("\n}\n", 1)[0]
    compact = scene.split("function compactPolicyFailure(policy){", 1)[1].split("\n}\n", 1)[0]
    hud = scene.split("function hud(t,xs,dones,focusedPolicy){", 1)[1].split("\nfunction resize", 1)[0]
    script = "const assert=require('node:assert/strict');\n"
    script += "function terminalStatus(policy,done){" + terminal + "\n}\n"
    script += "function compactPolicyFailure(policy){" + compact + "\n}\n"
    script += "function hud(t,xs,dones,focusedPolicy){" + hud + "\n"
    script += """
let IS_TRAJECTORY_COMPARISON=true;
const POL=[
  {failed:true,disqualified:true,terminal:{reason:'in_lane'}},
  {failed:true,disqualified:true,terminal:{reason:'self_collision'}},
  {failed:true,timedOut:true,eventT:60},
];
const cards=POL.map(()=>{
  const elements={'.d':{textContent:''},'.tm':{textContent:''}};
  return {querySelector:selector=>elements[selector],classList:{toggle:()=>{}}};
});
const clockEl={},clockStatusEl={dataset:{}},singlePolicy=false;
hud(60.5,[2,3,80],[true,true,true],1);
assert.deepEqual(cards.map(card=>card.querySelector('.d').textContent),['LANE DRIFT','COLLISION','TIMEOUT']);
assert.equal(clockStatusEl.textContent,'COLLISION');
assert.equal(clockStatusEl.dataset.kind,'incomplete');
IS_TRAJECTORY_COMPARISON=false;
hud(60.5,[2,3,80],[true,true,true],0);
assert.deepEqual(cards.map(card=>card.querySelector('.d').textContent),['LANE DRIFT','COLLISION','TIMEOUT']);
assert.equal(clockStatusEl.textContent,'LANE DRIFT');
assert.equal(clockStatusEl.dataset.kind,'incomplete');
assert.equal(cards[2].querySelector('.tm').textContent,'60.00s');
for(const reason of ['in_lane','self_collision','body_height','fell','fall','unrecognized_very_long_failure','']){
  const label=compactPolicyFailure({disqualified:true,terminal:{reason}});
  assert.ok(label.length<=(label==='LANE DRIFT'?10:9),`too long: ${label}`);
}
assert.equal(compactPolicyFailure({failed:true}),'DNF');
assert.equal(compactPolicyFailure({disqualified:true,terminal:{reason:'unrecognized'}}),'DNF');
assert.ok(['LANE DRIFT','COLLISION','TIMEOUT','FINISHED','DNF'].every(label=>label.length<=(label==='LANE DRIFT'?10:9)&&label===label.toUpperCase()));
hud(0,[0,0,0],[false,false,false],1);
assert.deepEqual(cards.map(card=>card.querySelector('.d').textContent),['0.0 m','0.0 m','0.0 m']);
assert.equal(clockStatusEl.textContent,'');
IS_TRAJECTORY_COMPARISON=false;
hud(60.5,[2,3,80],[true,true,true],0);
assert.equal(cards[0].querySelector('.d').textContent,'LANE DRIFT');
"""
    run_node(script)


def test_trial_shell_keeps_policy_list_at_top_and_shortens_visible_labels() -> None:
    shell = (ROOT / "web/renderers/g1-100-metres/render_trial_comparison.py").read_text()
    # This shell-only rule comes after the shared template's narrow-screen rule.
    assert '.hud:has(.clock-status:not(:empty)) .lanes{top:12px}' in shell
    runtime = (ROOT / "web/renderers/g1-100-metres/trial-comparison.js").read_text()
    assert '<span class="nm">#${policy.policy_number}</span>' in runtime
    assert "replayMessage('follow',{name},'Follow {name}')" in runtime
    assert 'aria-label="${follow}"' in runtime
    assert "replayMessage('remove',{name},'Remove {name}')" in runtime
    assert 'aria-label="${remove}"' in runtime


def test_mobile_comparison_results_use_flow_layout_and_report_actual_height() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    template = (ROOT / "web/replay-template.html").read_text().split("<script>const DATA=", 1)[0]
    app = (ROOT / "web/app.js").read_text()
    assert 'html.replay-mobile .hud{display:contents}' in template
    assert 'html.replay-mobile .lanes{position:static;order:2;' in template
    assert 'html.replay-mobile .replay-controls{order:1;' in template
    assert "if(!IS_COMPARISON)return false" in scene
    assert "width=window.parent.innerWidth" in scene
    assert "layoutObserver=new ResizeObserver(publishLayout);layoutObserver.observe(stageWrap)" in scene
    assert "if(key===lastLayout)return" in scene
    assert "const replayFrames=[$('.model-race-frame iframe'),$('#readout-replay')].filter(Boolean)" in app
    assert "event.source===candidate.contentWindow" in app
    assert "event.origin!==location.origin" in app
    assert "event.data?.type!=='g1:replay-layout'" in app
    assert "!Number.isFinite(height)||height<64||height>2000" in app
    assert "!Number.isFinite(height)||height<64||height>2000" in OVERVIEW
    layout = OVERVIEW.index("if (event.data?.type === 'g1:replay-layout')")
    assert OVERVIEW.index("if (messageGeneration !== String(replayGeneration)) return;") < layout


def test_replay_messages_require_current_generation() -> None:
    assert "const messageGeneration = String(event.data?.replayGeneration ?? '');" in OVERVIEW
    assert "messageGeneration !== String(replayGeneration)" in OVERVIEW
    assert "String(event.data.replayDocumentGeneration ?? '') === String(replayDocumentGeneration)" in OVERVIEW


def test_trajectory_overview_javascript_parses() -> None:
    result = subprocess.run(
        ["node", "--check", str(ROOT / "web/trajectory-overview.js")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_url_restore_waits_for_ready_scene_and_does_not_add_iframe_history_entries() -> None:
    assert HTML.index('src="/trajectory-url.js') < HTML.index('src="/trajectory.js')
    assert "if (!state.ready) return;" in overview_function("restoreUrlState")
    assert "jumpToStep(pending.step, true, true, true, null)" in overview_function("finishUrlRestore")
    assert "if (state.restoringUrl) finishUrlRestore()" in OVERVIEW
    assert "window.addEventListener('popstate', restoreHistory)" in OVERVIEW
    assert "window.addEventListener('hashchange', restoreHistory)" in OVERVIEW
    assert "setTimeout(() => commitUrl('replace'), 300)" in overview_function("syncFromScroll")
    frame = overview_function("replaceReplayFrame")
    assert frame.index("replacement.src = src") < frame.index("replayFrame.replaceWith(replacement)")
    assert "replayFrame = replacement" in frame


def test_url_restore_preserves_explicit_turn_and_discards_old_animation_callbacks() -> None:
    functions = overview_function("restoreUrlState") + overview_function("finishUrlRestore")
    script = "const assert=require('node:assert/strict');\n"
    script += f"const TrajectoryURL=require({json.dumps(str(ROOT / 'web/trajectory-url.js'))});\n"
    script += """
const queue={step_id:'a1-s593',public_step_id:590},explicit={step_id:'a1-s651',public_step_id:648};
const policy={submission_index:13,policy_sha256:'hash',replay_ready:true,replay_url:'/replay/capture'};
const state={ready:true,runId:'trial',policies:[policy],trajectory:{steps:[queue,explicit]}};
const location={href:'http://localhost/trajectory?run=trial&policies=13&step=a1-s651'};
const window={TrajectoryURL,dispatchEvent(){}};
const CustomEvent=class{constructor(){}};
let urlRestoreGeneration=0,replayGeneration=0,urlScrollTimer=null;
const callbacks=[],jumps=[],commits=[];
const requestAnimationFrame=callback=>callbacks.push(callback);
const policyHash=policy=>policy?.policy_sha256||'';
const policyQueueStep=policy=>policy?queue:null;
const replayPolicy=()=>replayGeneration++;
const hideReplay=()=>replayGeneration++;
const syncSelectionUI=()=>{};
const chapterIdForStep=step=>step?.step_id||null;
const jumpToStep=(step,...args)=>jumps.push({step,args});
const commitUrl=mode=>commits.push(mode);
const announceSelection=()=>{};
""" + functions + """
restoreUrlState();
assert.equal(state.focusedStepId,explicit.step_id);assert.equal(jumps.length,0);
callbacks.shift()();assert.equal(jumps.at(-1).step,explicit);
finishUrlRestore();assert.equal(jumps.at(-1).step,explicit);
assert.deepEqual(commits,['replace']);assert.equal(state.restoringUrl,false);
restoreUrlState();const obsolete=callbacks.shift();
location.href='http://localhost/trajectory?run=trial&turn=590&replay=0';
restoreUrlState();const latest=callbacks.shift();const count=jumps.length;
obsolete();assert.equal(jumps.length,count);
latest();assert.equal(state.focusedStepId,queue.step_id);assert.equal(jumps.at(-1).step,queue);
assert.equal(state.selectedPolicies.length,0);assert.equal(state.restoringUrl,false);
assert.ok(jumps.every(jump=>jump.args.at(-1)===null));
"""
    run_node(script)


def test_capture_ids_use_policy_hash_when_replay_metadata_is_missing() -> None:
    start = OVERVIEW.index("  function captureIdForPolicy(policy) {")
    end = OVERVIEW.index("\n  function rendererPolicy", start)
    script = """
const assert = require('node:assert/strict');
const location = {origin: 'http://localhost'};
const policyHash = policy => String(policy?.policy_sha256 || '');
""" + OVERVIEW[start:end] + """
const hash = 'e614d4450daba9158253271e6a76b34b80d56f8e08d237cfa08cb5081a0d10d0';
const expected = 'frontier-e614d4450dab';
for (const replay_url of [null, undefined, '', '/replay/trial-comparison.html', '/null']) {
  assert.equal(captureIdForPolicy({replay_url, policy_sha256: hash}), expected);
}
for (const replay_url of ['/replay/' + expected, '/replay/' + expected + '.html']) {
  assert.equal(captureIdForPolicy({replay_url}), expected);
}
assert.equal(captureIdForPolicy({policy_sha256: 'not-a-valid-policy-hash'}), '');
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_policy_placement_uses_exported_exact_queue_steps_across_published_models() -> None:
    script = """
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const state = {trajectory: null};
""" + f"const web = {json.dumps(str(ROOT / 'web'))};\n" + overview_function("policyQueueStep") + overview_function("policyEpoch") + """
let checked = 0;
const bases = new Set();
const published = JSON.parse(fs.readFileSync(path.join(web, 'data/policies/index.json'))).runs;
for (const run of published) {
  const name = `${run.run_id}.json`;
  const policies = JSON.parse(fs.readFileSync(path.join(web, 'data/policies', name))).policies || [];
  const tracePath = path.join(web, 'data/trajectories', name);
  state.trajectory = JSON.parse(fs.readFileSync(tracePath));
  for (const policy of policies) {
    const step = policyQueueStep(policy);
    assert.ok(step, `${name} policy ${policy.submission_index} must resolve to an exact source step`);
    assert.equal(step.step_id, policy.queue_source_step_id);
    assert.equal(step.public_step_id, policy.queue_source_public_step_id);
    assert.equal(policyEpoch(policy), Date.parse(policy.enqueued_at));
    bases.add(policy.queue_source_basis);
    checked++;
  }
  if (name === 's10-vexp-r123-20260828-luna-4.json') {
    for (const number of [13,14,15,16,17,30,31,32]) {
      const policy = policies.find(item => item.submission_index === number);
      const expected = number < 16 ? 590 : 596;
      assert.equal(policyQueueStep(policy).public_step_id, expected);
      assert.ok(policyEpoch(policy) < Date.parse(policy.artifact_observed_at));
    }
    assert.ok(policies.every(policy => ![744,758].includes(policyQueueStep(policy).public_step_id)));
  }
}
assert.ok(checked >= 219, `Checked only ${checked} published policies`);
assert.ok(bases.has('direct_submission_response'));
assert.ok(bases.has('gpu_submission_artifact'));
assert.ok(bases.has('gpu_submission_result'));
"""
    run_node(script)


def test_grouped_queue_badges_keep_exact_turn_and_unresolved_policies_remain_accessible() -> None:
    functions = "\n".join(overview_function(name) for name in [
        "policyQueueStep", "policyEpoch", "renderedStepNode", "chapterIdForStep",
        "policyChapterId", "renderPolicyReferences", "navigateToPolicy",
    ])
    script = """
const assert = require('node:assert/strict');
const exact = {step_id:'raw-queue', public_step_id:590, timestamp:'2026-08-28T20:36:08Z'};
const final = {step_id:'raw-final', public_step_id:758, timestamp:'2026-08-28T21:25:20Z'};
const state = {trajectory:{steps:[exact,final]},policies:[]};
const policy = {submission_index:13,queue_source_step_id:exact.step_id,queue_source_public_step_id:590,
  queue_source_basis:'gpu_submission_artifact',enqueued_at:exact.timestamp,submitted_at:final.timestamp};
const unresolved = {...policy,submission_index:32,queue_source_step_id:null,queue_source_public_step_id:null,
  queue_source_basis:'unresolved',enqueued_at:final.timestamp};
const element = () => ({children:[],attributes:{},dataset:{},append(...items){this.children.push(...items)},
  replaceChildren(){this.children=[]},setAttribute(k,v){this.attributes[k]=v}});
const meta=element(),chapterList=element(),unmappedPolicyList=element(),unmappedPolicies={hidden:true};
const grouped={dataset:{publicStep:'589'},querySelector(){return meta}};
const chapter={dataset:{startStep:'580',endStep:'610',chapterId:'queue-chapter'},querySelector(){return chapterList}};
const outline={scrollTop:100};
let urlRestoreGeneration=1;
const anchorMobileNavigation=()=>{};
const document={
  querySelectorAll(selector){return selector==='.chapter-item'?[chapter]:selector==='.chapter-policy-list'?[chapterList]:[]},
  querySelector(selector){return selector==='#chapter-nav'?outline:selector.includes('raw-queue')?grouped:null},
  createElement:element,
};
const CSS={escape:value=>value};
const policyNumber=policy=>policy.submission_index;
const policyMarker=(policy,context,chapterId)=>({policy,context,chapterId});
const syncPolicyMarkers=()=>{};
const restoreOutlineScroll=(node,value)=>node.scrollTop=value;
const requestAnimationFrame=callback=>callback();
const events=[];
const window={dispatchEvent:event=>events.push(event)};
const CustomEvent=class{constructor(type,options){this.type=type;this.detail=options.detail}};
let jumped=null,announcement='';
const jumpToStep=step=>jumped=step;
const announceSelection=message=>announcement=message;
const commitUrl=()=>{};
const nearestStep=()=>{throw Error('Policy navigation must never use nearest timestamp')};
""" + functions + """
state.policies=[policy,unresolved];
renderPolicyReferences();
assert.equal(meta.children[0].children.length,1);
assert.equal(meta.children[0].children[0].context,'queued at turn 590');
assert.equal(meta.children[0].children[0].chapterId,'queue-chapter');
assert.equal(chapterList.children[1].policy,policy);
assert.equal(unmappedPolicies.hidden,false);
assert.equal(unmappedPolicyList.children[0].policy,unresolved);
assert.ok(Number.isNaN(policyEpoch(unresolved)));
assert.equal(policyChapterId(unresolved),null);
navigateToPolicy(policy,'wrong-late-chapter');
assert.equal(jumped,exact);
assert.equal(events.find(event=>event.type==='trajectory:policy-selected').detail.chapterId,'queue-chapter');
jumped=null;
navigateToPolicy(unresolved,'wrong-late-chapter');
assert.equal(jumped,null);
assert.match(announcement,/no matched queue turn/);
for (const bad of [
  {...policy,queue_source_step_id:'absent'},
  {...policy,queue_source_public_step_id:758},
  {...policy,queue_source_basis:'gpu_job_unmapped'},
  {...policy,queue_source_basis:undefined},
  {...policy,queue_source_step_id:null,queue_source_public_step_id:null},
]) assert.equal(policyQueueStep(bad),null);
assert.equal(policyQueueStep({...policy,queue_source_step_id:null}),exact);
assert.equal(policyQueueStep({...policy,queue_source_basis:'direct_submission_response'}),exact);
assert.equal(policyQueueStep({...policy,queue_source_basis:'gpu_explicit_path_request_interval'}),exact);
"""
    run_node(script)
    assert 'id="unmapped-policies"' in HTML
    assert "epoch: policyEpoch(policy)" in overview_function("drawPolicies")
    assert "policy.submitted_at" not in overview_function("policyEpoch")
