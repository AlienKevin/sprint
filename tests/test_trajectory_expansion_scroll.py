"""Turn disclosures reveal content without moving the reader's viewport."""

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "web/trajectory.js").read_text()
OVERVIEW = (ROOT / "web/trajectory-overview.js").read_text()


def function(name):
    start = SOURCE.index(f"  function {name}(")
    following = re.search(r"\n  (?:(?:async )?function |const )", SOURCE[start + 1:])
    return SOURCE[start:start + 1 + following.start()]


def test_expansion_keeps_earlier_turns_open_and_opens_only_its_own_tools():
    script = r"""
const assert=require('node:assert/strict');
class Element {
 constructor(tag,className='',content=''){this.tagName=tag;this.className=className;this.content=content;this.children=[];this.dataset={};this.listeners={};this.open=false;this.classList={add(){}}}
 append(...nodes){this.children.push(...nodes)}
 addEventListener(name,callback){this.listeners[name]=callback}
 querySelectorAll(selector){
  assert.equal(selector,':scope > .step-body > details.activity.tool');
  return this.children.find(node=>node.className==='step-body').children.filter(node=>node.className==='activity tool');
 }
}
const el=(...args)=>new Element(...args),elapsed=()=>'',text=String,label=value=>value,
 stepPreview=()=>({label:'Agent message',value:'preview'});
const clipped=[],markClippedOutputs=step=>clipped.push(step),steps=[];
const stepsTarget={scrollTop:1800,querySelectorAll:()=>steps.filter(step=>step.open)};
const window={scrollY:2100,scrollTo(){throw Error('Expansion must not scroll')}},
 requestAnimationFrame=()=>{throw Error('Expansion must not schedule a delayed scroll')};
function appendStepActivities(body){body.append(new Element('details','activity tool'),new Element('details','activity observation'))}
""" + function("openStepTools") + function("renderStep") + r"""
for(const number of [23,80,90]){
 const primary={step_id:`a1-s${number}`,public_step_id:number,timestamp:'2026-08-28T00:00:00Z'};
 steps.push(renderStep({primary,steps:[primary]},primary.timestamp));
}
for(const step of steps){
 step.open=true;step.listeners.toggle();
 assert.equal(step.querySelectorAll(':scope > .step-body > details.activity.tool')[0].open,true);
 assert.equal(step.children.at(-1).children[1].open,false,'non-tool disclosures stay unchanged');
}
assert.deepEqual(steps.map(step=>step.open),[true,true,true],
 'Opening a later turn must not remove expanded height above the reading position');
assert.deepEqual(clipped,steps,'output clipping is still measured on each opened turn');
steps[1].open=false;steps[1].listeners.toggle();
assert.deepEqual(steps.map(step=>step.open),[true,false,true],'users can still close individual turns');
assert.equal(clipped.length,3,'closing must not reopen tools or schedule work');
assert.equal(stepsTarget.scrollTop,1800);assert.equal(window.scrollY,2100);
"""
    result = subprocess.run(["node", "-"], input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_trace_disables_browser_scroll_anchoring_at_all_viewport_sizes():
    css = (ROOT / "web/trajectory.css").read_text()
    declarations = re.search(r"^\.steps \{([^}]+)\}", css, re.M).group(1)
    assert "overflow-anchor: none" in declarations


def test_direct_turn_clicks_still_select_without_scrolling_but_navigation_can_scroll():
    click = OVERVIEW.split("stepsTarget?.addEventListener('click', event => {", 1)[1].split("\n  });", 1)[0]
    assert "jumpToStep(step, true, true, false)" in click
    assert "scrollTraceTarget(" not in click
    assert "scrollIntoView(" not in click
    assert "scrollTraceTarget(target,'center')" in function("jumpToChapter")
    assert "if (shouldScroll) {" in OVERVIEW
    assert "scrollTraceTarget(exactStep ? target : scrollTargetForStep(target)" in OVERVIEW
