"""Mobile policy navigation keeps the submitted turn below the sticky replay."""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERVIEW = (ROOT / "web/trajectory-overview.js").read_text()
TRACE = (ROOT / "web/trajectory.js").read_text()


def function(name, source=OVERVIEW):
    start = source.index(f"  function {name}(")
    following = re.search(r"\n  (?:async )?function ", source[start + 1:])
    assert following
    return source[start:start + 1 + following.start()]


def node(script):
    script = "const ui=(source,params={})=>source.replace(/\\{(\\w+)\\}/g,(match,key)=>params[key]??match);const setUI=(node,source,params)=>{node.textContent=ui(source,params)};\n" + script
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_mobile_alignment_waits_for_replay_height_and_discards_stale_intents():
    node("""
const assert=require('node:assert/strict');
let targetTop=5000,replayHeight=300,loading=true,scrolls=0,desktopRoot=null;
const window={scrollY:100,innerHeight:844,scrollTo({top,behavior}){assert.equal(behavior,'instant');this.scrollY=top;scrolls++;}};
const target={getBoundingClientRect:()=>({top:targetTop-window.scrollY,height:900})};
const toolbar={getBoundingClientRect:()=>({height:46})},nav={getBoundingClientRect:()=>({height:58})};
const replayPanel={hidden:false,classList:{contains:()=>loading},getBoundingClientRect:()=>({top:500,height:replayHeight})};
const document={querySelector:()=>toolbar},getComputedStyle=node=>({top:node===toolbar?'58px':'104px'});
const narrowViewport={matches:true},traceScrollRoot=()=>desktopRoot;
const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
const step={step_id:'actual-submit'},state={runId:'trial',focusedStepId:step.step_id};
let urlRestoreGeneration=1,replayGeneration=1,frames=[];
const requestAnimationFrame=callback=>{frames.push(callback);return frames.length};
const frame=()=>{const batch=frames;frames=[];batch.forEach(callback=>callback())};
const performance={now:()=>1000},renderedStepNode=()=>target;
""" + function("scrollTraceTarget") + function("anchorMobileNavigation")
         + function("realignMobileNavigation") + function("cancelMobileNavigation") + """
scrollTraceTarget(target,'center');
assert.equal(window.scrollY,4580);assert.equal(targetTop-window.scrollY,420,
 'a tall turn starts below the replay; its middle is not centered behind it');
anchorMobileNavigation(step);frame();assert.ok(state.mobileNavigation);
// Mobile iframe sizing can shift both document content and browser scroll anchoring.
replayHeight=500;targetTop+=200;window.scrollY+=200;
realignMobileNavigation();frame();assert.equal(window.scrollY,4580);
assert.equal(targetTop-window.scrollY,620);
loading=false;realignMobileNavigation(true);frame();frame();
assert.equal(state.mobileNavigation,null);assert.equal(state.restoreLockUntil,1700);
for(const change of [()=>urlRestoreGeneration++,()=>replayGeneration++,()=>state.runId='other',
 ()=>state.focusedStepId='other',()=>cancelMobileNavigation()]){
 state.runId='trial';state.focusedStepId=step.step_id;loading=true;anchorMobileNavigation(step);
 const before=scrolls;change();frame();assert.equal(scrolls,before,'stale/manual intent cannot scroll');
}
// The existing desktop scroll container remains independent of the outer page.
narrowViewport.matches=false;
desktopRoot={scrollTop:200,clientHeight:500,scrollHeight:3000,getBoundingClientRect:()=>({top:100}),scrollTo({top}){this.scrollTop=top}};
targetTop=window.scrollY+500;const outer=window.scrollY;
scrollTraceTarget(target,'start');assert.equal(desktopRoot.scrollTop,600);assert.equal(window.scrollY,outer);
""")


def test_policy_navigation_uses_proven_submission_and_latest_async_choice():
    node("""
const assert=require('node:assert/strict');
const first={step_id:'first',public_step_id:1},submitted={step_id:'submit',public_step_id:832},later={step_id:'later',public_step_id:900};
const state={runId:'trial',trajectory:{steps:[first,submitted,later]}};
const policy={submission_index:8,queue_source_basis:'gpu_submission_artifact',queue_source_step_id:'submit',queue_source_public_step_id:832};
let urlRestoreGeneration=1,frames=[],jumps=[],anchors=[],messages=[];
const requestAnimationFrame=callback=>frames.push(callback),frame=()=>{const batch=frames;frames=[];batch.forEach(callback=>callback())};
const outline={scrollTop:45},document={querySelector:()=>outline};
const window={dispatchEvent(){}},CustomEvent=class{};
const chapterIdForStep=step=>step?.step_id;
const jumpToStep=step=>{jumps.push(step);urlRestoreGeneration++;};
const anchorMobileNavigation=step=>anchors.push(step),restoreOutlineScroll=(node,top)=>node.scrollTop=top;
const announceSelection=message=>messages.push(message),policyNumber=policy=>policy.submission_index,commitUrl=()=>{};
""" + function("policyQueueStep") + function("navigateToPolicy") + """
navigateToPolicy(policy,'wrong-chapter');frame();assert.deepEqual(jumps,[submitted]);assert.deepEqual(anchors,[submitted]);frame();
jumps=[];anchors=[];navigateToPolicy(policy);urlRestoreGeneration++;navigateToPolicy({...policy,queue_source_step_id:'later',queue_source_public_step_id:900});
frame();assert.deepEqual(jumps,[later]);assert.deepEqual(anchors,[later]);frame();
jumps=[];navigateToPolicy({...policy,queue_source_basis:'unresolved'});frame();
assert.deepEqual(jumps,[]);assert.match(messages.at(-1),/no matched queue turn/);
""")


def test_selected_count_reopens_same_policies_at_current_turn():
    node("""
const assert=require('node:assert/strict');
const policy={hash:'a'},second={hash:'b'},queue={step_id:'submitted'},reading={step_id:'currently-reading'};
const state={selectedPolicies:[policy,second],emphasizedPolicyHash:'a',trajectory:{steps:[queue,reading]},focusedStepId:reading.step_id};
const narrowViewport={matches:true};
const replayPanel={hidden:false,classList:{remove(){}}};
const document={body:{classList:{remove(){}}}};
let replayGeneration=1,rendererSelectionKey='a,b',replayHudObserver=null,urlRestoreGeneration=1,urlScrollTimer=null;
let framesDisposed=0,commits=0,jumps=[],anchors=[],loaded=null;
const replaceReplayFrame=()=>framesDisposed++,releasePolicyChapter=()=>{},syncSelectionUI=()=>{},commitUrl=()=>commits++;
const policyHash=policy=>policy.hash,policyQueueStep=()=>queue;
const replayPolicy=(policy,options)=>{assert.equal(options.scrollToReplay,false);loaded=policy;replayPanel.hidden=false;return true};
const jumpToStep=(step,force,exact,scroll,historyMode)=>{jumps.push(step);if(historyMode!==null)commits++},anchorMobileNavigation=step=>anchors.push(step);
""" + function("beginUrlInteraction") + function("hideReplay") + function("closeReplay") + function("reopenSelectedReplay") + """
closeReplay();assert.equal(replayPanel.hidden,true);assert.equal(framesDisposed,1);
assert.deepEqual(state.selectedPolicies,[policy,second]);assert.equal(state.emphasizedPolicyHash,'a');
assert.equal(state.closedReplayStepId,reading.step_id);jumps=[];
state.focusedStepId=queue.step_id; // Retain the chosen reading turn despite any late pre-close scroll callback.
reopenSelectedReplay();assert.equal(replayPanel.hidden,false);assert.equal(loaded,policy);
assert.deepEqual(jumps,[reading]);assert.deepEqual(anchors,[reading]);assert.equal(commits,2);
reopenSelectedReplay();assert.equal(commits,2,'clicking an already open replay does not add history');
closeReplay();state.selectedPolicies=[];reopenSelectedReplay();assert.equal(replayPanel.hidden,true);
""")


def test_drawer_focus_and_unlock_do_not_start_competing_page_scrolls():
    node("""
const assert=require('node:assert/strict');let timer,focuses=[],scrolls=[];
const focus=options=>{assert.equal(options.preventScroll,true);focuses.push(options)};
const state={},classes=new Set(),mobileViewport={matches:true};
const outline={hidden:false,setAttribute(){},removeAttribute(){},focus};
const mobileOutlineToggle={setAttribute(){}},mobileOutlineBackdrop={};
const chapterNav={querySelector:()=>({focus})};
const document={activeElement:{focus},body:{style:{removeProperty(){}},classList:{add:value=>classes.add(value),remove:value=>classes.delete(value),contains:value=>classes.has(value)}}};
const window={scrollY:4321,setTimeout(callback){timer=callback;},scrollTo(options){scrolls.push(options);}};
""" + function("openOutlineDrawer", TRACE) + function("closeOutlineDrawer", TRACE) + """
openOutlineDrawer();timer();assert.equal(focuses.length,2);
closeOutlineDrawer(false);assert.deepEqual(scrolls,[{top:4321,behavior:'instant'}]);assert.equal(focuses.length,2);
timer();assert.equal(focuses.length,2,'late drawer focus is ignored after a policy selection closes it');
openOutlineDrawer();closeOutlineDrawer();assert.equal(focuses.length,4);
""")


def test_history_restore_and_trial_switch_clear_closed_reading_guard():
    node("""
const assert=require('node:assert/strict');
const restored={step_id:'new-history-step'},policy={hash:'policy'};
const state={ready:true,runId:'trial',trajectory:{steps:[restored]},policies:[policy],closedReplayStepId:'old-closed-step'};
let urlRestoreGeneration=0,replayGeneration=0,rendererSelectionKey='',replayHudObserver=null,urlScrollTimer=null;
const replayPanel={hidden:true,classList:{remove(){}}},document={body:{classList:{remove(){}}}};
let closedGuardAtReplay=null;
const window={TrajectoryURL:{parse:()=>({runId:'trial',step:'new-history-step'}),resolve:()=>({policies:[policy],focus:policy,step:restored,replayVisible:false,errors:[]})},dispatchEvent(){}};
const location={href:'http://localhost/trajectory?run=trial&step=new-history-step&replay=0'};
const policyHash=policy=>policy?.hash||'',policyQueueStep=()=>null,replaceReplayFrame=()=>{},releasePolicyChapter=()=>{},syncSelectionUI=()=>{};
const replayPolicy=()=>{closedGuardAtReplay=state.closedReplayStepId;replayGeneration++;};
const chapterIdForStep=step=>step?.step_id,CustomEvent=class{},announceSelection=()=>{};
const jumpToStep=()=>{},commitUrl=()=>{},performance={now:()=>1000};
const requestAnimationFrame=callback=>callback();
""" + function("hideReplay") + function("restoreUrlState") + function("finishUrlRestore")
         + function("cancelMobileNavigation") + """
restoreUrlState();assert.equal(state.closedReplayStepId,null);assert.equal(state.focusedStepId,restored.step_id);
state.closedReplayStepId='old-closed-step';
window.TrajectoryURL.resolve=()=>({policies:[policy],focus:policy,step:restored,replayVisible:true,errors:[]});
restoreUrlState();assert.equal(closedGuardAtReplay,null,'visible history restoration also clears before showing its replay');
state.closedReplayStepId='old-trial-step';hideReplay();assert.equal(state.closedReplayStepId,null,'trial loading calls this same hide path');
const button={closest:()=>({})};
state.closedReplayStepId='keep-for-count-click';cancelMobileNavigation({type:'pointerdown',target:button});
assert.equal(state.closedReplayStepId,'keep-for-count-click');
cancelMobileNavigation({type:'wheel',target:button});assert.equal(state.closedReplayStepId,null,'wheel scrolling is manual reading even over a button');
state.closedReplayStepId='clear-on-tab';cancelMobileNavigation();assert.equal(state.closedReplayStepId,null,'scroll/navigation key handler passes no event');
""")


def test_mobile_count_is_an_accessible_button_and_matches_outline_font_size():
    html = (ROOT / "web/trajectory.html").read_text()
    css = (ROOT / "web/trajectory.css").read_text()
    assert re.search(r'<button id="mobile-replay-label"[^>]*type="button"[^>]*aria-controls="trajectory-policy-replay"', html)
    assert "mobileReplayLabel?.addEventListener('click', reopenSelectedReplay)" in OVERVIEW
    assert "mobileReplayLabel.disabled = !hasSelection" in function("syncSelectionUI")
    assert "mobileReplayLabel.setAttribute('aria-expanded', String(!replayPanel.hidden))" in function("syncSelectionUI")
    toolbar = re.search(r'\.mobile-trace-toolbar button \{([^}]+)', css).group(1)
    label = re.search(r'\.mobile-replay-label \{([^}]+)', css).group(1)
    assert "font-size: 9px" in label and "font-size: 9px" in toolbar


def test_header_and_timeline_match_current_homepage_trial_mapping():
    batch = json.loads((ROOT / "web/data/batches/current.json").read_text())
    snapshot = json.loads((ROOT / "web/data/performance/current.json").read_text())
    performance = {"runs": [{"run_id": run["run_id"], "summary": {"best_continuous_score_mps": run["summary"].get("best_continuous_score_mps")},
                             "points": [{"continuous_score_mps": point.get("continuous_score_mps")} for point in run.get("points", [])]}
                            for run in snapshot["runs"]]}
    model_functions = "\n".join(line for line in TRACE.splitlines() if line.strip().startswith(("const modelFamily=", "const modelLabel=")))
    node("const assert=require('node:assert/strict');\n" + model_functions
         + f"\nconst batch={json.dumps(batch)},snapshot={json.dumps(performance)};\n"
         + """
const state={batch,displayTrials:{},index:{runs:[]}},nodes=new Map();
const $=selector=>{if(!nodes.has(selector))nodes.set(selector,{attributes:{},setAttribute(key,value){this.attributes[key]=value}});return nodes.get(selector)};
const document={};
""" + function("buildTrialDisplayNumbers", TRACE) + function("trialNumberForRun", TRACE)
         + function("renderTrialIdentity", TRACE) + """
state.displayTrials=buildTrialDisplayNumbers(batch,snapshot);
const expected={
 's10-vexp-r120-20260828-deepseek-1':1,
 's10-vexp-r123-20260828-luna-4':1,
 's10-vexp-r123-20260828-luna-2':2,
 'claude-goalfix2-20260828-1750-glm-2':1,
 'claude-goal-20260828-0237-main-glm-2':2,
};
for(const arm of batch.arms){
 const trial=state.displayTrials[arm.run_id];assert.ok(Number.isInteger(trial)&&trial>=1&&trial<=5);
 if(expected[arm.run_id])assert.equal(trial,expected[arm.run_id]);
 renderTrialIdentity(arm);const model=modelLabel(arm.model);
 assert.equal($('#viewer-trial-model').textContent,model);
 assert.equal($('#viewer-trial-number').textContent,`#${trial}`);
 assert.equal($('#viewer-trial-title').attributes['aria-label'],`${model} #${trial}`);
 assert.equal($('#run-model').textContent,`${model} (Trial #${trial})`);
 assert.equal(document.title,`${model} #${trial} · Agent trajectory`);
}
const original=JSON.stringify(state.displayTrials);
assert.equal(JSON.stringify(buildTrialDisplayNumbers(batch,{runs:[...snapshot.runs].reverse()})),original,'source array order is not a trial rank');
state.displayTrials={};assert.equal(trialNumberForRun(batch.arms[0]),null,'never flash the wrong launch-order trial while mapping is loading');
assert.equal(trialNumberForRun({run_id:'archived-luna-7',model:'openai/gpt-5.6-luna'}),7);
state.batch=null;assert.equal(trialNumberForRun({run_id:'unmapped-luna-2'}),null,'unavailable metadata does not invent a current trial number');
""")
    assert "void loadTrialDisplayNumbers()" in TRACE
    assert "await loadTrialDisplayNumbers()" not in TRACE
    css = (ROOT / "web/trajectory.css").read_text()
    assert "#viewer-trial-number { flex: 0 0 auto;" in css


def test_fixed_header_is_reserved_and_above_mobile_drawer_and_replay():
    css = (ROOT / "web/trajectory.css").read_text()
    header = re.search(r"\.viewer-nav \{([^}]+)", css).group(1)
    assert "position: fixed" in header and "inset: 0 0 auto" in header
    assert "z-index: 120" in header
    assert "padding-top: var(--viewer-nav-height)" in css
    mobile = css.split("@media (max-width: 850px)", 1)[1]
    toolbar = re.search(r"\.mobile-trace-toolbar \{([^}]+)", mobile).group(1)
    assert "top: var(--viewer-nav-height)" in toolbar
    assert "top: calc(var(--viewer-nav-height) + var(--mobile-toolbar-height))" in mobile
    assert "inset: var(--viewer-nav-height) auto 0 0" in mobile
    assert "inset: var(--viewer-nav-height) 0 0" in mobile
    assert "z-index: 100" in mobile and "z-index: 90" in mobile


def test_desktop_page_fits_viewport_and_keeps_its_panels_below_fixed_header():
    css = (ROOT / "web/trajectory.css").read_text()
    desktop = css.split("@media (min-width: 851px)", 1)[1].split("@media", 1)[0]
    assert ".viewer-layout { padding-bottom: 20px; }" in desktop
    assert "padding: 24px 0 80px" in css, "mobile keeps its normal document bottom gutter"
    assert "--viewer-nav-height: 58px" in css
    column = re.search(r"\.trace-column \{([^}]+)", desktop).group(1)
    assert "top: 82px" in column and "height: calc(100vh - 102px)" in column
    assert "overflow-y: auto" in re.search(r"\.steps \{([^}]+)", desktop).group(1)
    # Header + layout top + panel + bottom fill the viewport exactly. A larger
    # outer page lets the sticky grid's bottom bound push the replay under nav.
    assert 58 + 24 - 102 + 20 == 0


def test_chapter_and_policy_jumps_use_the_same_fixed_header_stack():
    overview_scroll = function("scrollTraceTarget").replace("function scrollTraceTarget(", "function scrollPolicy(", 1)
    chapter_scroll = function("scrollTraceTarget", TRACE).replace("function scrollTraceTarget(", "function scrollChapter(", 1)
    node("""
const assert=require('node:assert/strict');
const nav={getBoundingClientRect:()=>({top:0,bottom:58,height:58})};
const toolbar={getBoundingClientRect:()=>({top:58,bottom:104,height:46})};
const replayPanel={hidden:false,getBoundingClientRect:()=>({top:104,bottom:404,height:300})};
const document={querySelector:selector=>selector==='.viewer-nav'?nav:selector==='#mobile-trace-toolbar'?toolbar:replayPanel};
const getComputedStyle=node=>({top:node===toolbar?'58px':'104px'});
const traceScrollRoot=()=>null,narrowViewport={matches:true},mobileViewport=narrowViewport;
const window={scrollY:100,innerHeight:844,scrollTo({top,behavior}){assert.equal(behavior,'instant');this.scrollY=top}};
const target={getBoundingClientRect:()=>({top:1000-window.scrollY,height:200}),scrollIntoView(){throw Error('mobile jumps must target the page once with full sticky offsets')}};
""" + overview_scroll + chapter_scroll + function("traceAnchor", TRACE) + """
for(const scroll of [scrollPolicy,scrollChapter]){
 replayPanel.hidden=false;window.scrollY=100;scroll(target,'center');assert.equal(window.scrollY,580);
 assert.equal(target.getBoundingClientRect().top,replayPanel.getBoundingClientRect().bottom+16);
 replayPanel.hidden=true;window.scrollY=100;scroll(target,'center');assert.equal(window.scrollY,880);
 assert.equal(target.getBoundingClientRect().top,toolbar.getBoundingClientRect().bottom+16);
}
assert.equal(traceAnchor(),128);replayPanel.hidden=false;assert.equal(traceAnchor(),428);
""")
