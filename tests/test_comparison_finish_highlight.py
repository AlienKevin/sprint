from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "web/renderers/g1-100-metres"


def renderer():
    sys.path.insert(0, str(RENDERER))
    try:
        spec = importlib.util.spec_from_file_location(
            "comparison_finish_highlight", RENDERER / "render_comparison.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(RENDERER))


def js_assertions(assertions: str) -> None:
    stub = """
const assert=require('node:assert/strict');
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-9,`${actual} != ${expected}`);
const DATA={meta:{finish_highlights:[
  {policy_index:1,start_s:9.4,end_s:9.9,speed:.1,focus_hold_s:.5},
  {policy_index:2,start_s:27.28,end_s:27.48,speed:.1,focus_hold_s:.5},
  {policy_index:0,start_s:37.25,end_s:37.75,speed:.1,focus_hold_s:0}
],camera_orbits:[
  {policy_index:2,start_s:12.48,rotate_s:5,hold_s:5,return_s:5,side_azimuth_degrees:-90},
  {policy_index:0,start_s:30,rotate_s:5,hold_s:5,return_s:5,side_azimuth_degrees:-90}
]}};
const FOLLOW_VIEW={az:-.25,el:.13,dist:2.541};
let playT=0,speed=1,playing=false;
function button(value){return {dataset:{s:String(value)},handlers:[],pressed:null,
  classList:{values:new Set(),toggle(name,on){if(on)this.values.add(name);else this.values.delete(name)},contains(name){return this.values.has(name)}},
  addEventListener(kind,fn){this.handlers.push(fn)},
  setAttribute(name,value){this.pressed=value},
  click(){this.handlers.forEach(fn=>fn())}};}
const buttons=[.1,.5,1,2].map(button),cameraResetBtn=button(0);
const document={querySelectorAll(){return buttons}};
"""
    # Register the shared player's speed handler after the injected hook, as in
    # the generated HTML. The highlight must respect this user's new choice.
    shared = """
buttons.forEach(b=>b.addEventListener('click',()=>{speed=Number(b.dataset.s)}));
"""
    result = subprocess.run(
        ["node", "-"],
        input=stub + renderer().FINISH_HIGHLIGHT_SCRIPT + renderer().CAMERA_ORBIT_SCRIPT + shared
        + "\n(async()=>{\n" + assertions + "\n})().catch(e=>{console.error(e);process.exitCode=1});",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_glm_last_half_second_takes_five_wall_seconds_and_restores_base_speed():
    js_assertions("""
near(advanceFinishHighlight(8.9,.5,1),9.4);
near(advanceFinishHighlight(9.4,5,1),9.9);
near(advanceFinishHighlight(9.4,5,2),9.9);
near(advanceFinishHighlight(9.9,2,2),13.9);
// A single large frame crosses both boundaries without spending its entire
// wall-time delta at either the old rate or the highlight rate.
near(advanceFinishHighlight(8.5,6.45,2),11.9);
speed=2;playT=9.5;syncFinishHighlight(playT);
assert.equal(finishHighlightState().effectiveSpeed,.1);
assert.equal(buttons[0].pressed,'true');
playT=9.9;syncFinishHighlight(playT);
assert.equal(finishHighlightState().effectiveSpeed,2);
assert.equal(buttons[3].pressed,'true');
assert.equal(speed,2);
""")


def test_pause_does_not_consume_highlight_and_manual_speed_choice_wins():
    js_assertions("""
playT=9.5;syncFinishHighlight(playT);
for(let i=0;i<100;i++)syncFinishHighlight(playT);
near(playT,9.5);assert.equal(finishHighlightState().playing,false);
buttons[2].click();await Promise.resolve();
assert.equal(finishHighlightState().overridden,true);
assert.equal(finishHighlightState().effectiveSpeed,1);
assert.equal(buttons[2].pressed,'true');
near(advanceFinishHighlight(playT,1,speed),10.5);
// Overriding GLM does not suppress Luna's later finish highlight.
playT=27.3;syncFinishHighlight(playT);
assert.equal(finishHighlightState().policy,2);
assert.equal(finishHighlightState().overridden,false);
assert.equal(finishHighlightState().effectiveSpeed,.1);
""")


def test_auto_camera_holds_glm_through_crossing_after_normal_speed_returns():
    js_assertions("""
assert.equal(finishHighlightFocus(9.39),false);
assert.equal(finishHighlightFocus(9.4),true);
assert.equal(finishHighlightCamera(9.4).policy_index,1);
assert.equal(finishHighlightFocus(9.9),true);
assert.equal(finishHighlightFocus(10.1),true);
assert.equal(finishHighlightFocus(10.4),false);
assert.equal(finishHighlightActive(9.9),false);
playT=10.1;syncFinishHighlight(playT);
assert.equal(finishHighlightState().effectiveSpeed,1);
""")


def test_backward_seek_replay_and_reset_rearm_the_highlight():
    js_assertions("""
playT=9.5;syncFinishHighlight(playT);buttons[2].click();await Promise.resolve();
assert.equal(finishHighlightState().overridden,true);
cameraResetBtn.click();
assert.equal(finishHighlightState().active,true);
buttons[2].click();await Promise.resolve();
playT=0;syncFinishHighlight(playT);
assert.equal(finishHighlightState().overridden,false);
near(advanceFinishHighlight(9.4,5,speed),9.9);
// Replay starts at zero without requiring a separate initial draw.
playT=60;syncFinishHighlight(playT);finishHighlightOverrides.add(1);finishHighlightOverrides.add(2);
near(advanceFinishHighlight(0,14.4,speed),9.9);
assert.equal(finishHighlightOverrides.size,0);
""")


def test_luna_last_point_two_seconds_takes_two_wall_seconds_and_focuses_luna():
    js_assertions("""
near(advanceFinishHighlight(27.28,2,1),27.48);
near(advanceFinishHighlight(27.28,2,.5),27.48);
near(advanceFinishHighlight(27.48,1,.5),27.98);
assert.equal(finishHighlightFocus(27.279),false);
assert.equal(finishHighlightCamera(27.28).policy_index,2);
assert.equal(finishHighlightCamera(27.6).policy_index,2);
assert.equal(finishHighlightFocus(27.98),false);
playT=27.3;syncFinishHighlight(playT);
assert.equal(finishHighlightState().effectiveSpeed,.1);
buttons[1].click();await Promise.resolve();
assert.equal(finishHighlightState().overridden,true);
assert.equal(finishHighlightState().effectiveSpeed,.5);
// One large clock delta can cross both independently scheduled highlights.
finishHighlightOverrides.clear();
near(advanceFinishHighlight(0,33.78,1),27.48);
near(advanceFinishHighlight(0,34.78,1),28.48);
""")


def test_injection_is_homepage_only_and_fails_fast_if_shared_clock_changes():
    module = renderer()
    scene = (RENDERER / "scene.js").read_text()
    assert module.add_finish_highlight(scene, {"meta": {}}) == scene
    data = {"meta": {"finish_highlights": [{"start_s": 9.4}]}}
    output = module.add_finish_highlight(scene, data)
    assert "playT = advanceFinishHighlight(playT,(now-lastNow)/1000,speed);" in output
    assert "syncFinishHighlight(t);" in output
    assert "finishHighlight:finishHighlightState" in output
    # The hook does not force a camera reset or overwrite manual selection.
    assert "userFollowPolicy=" not in module.FINISH_HIGHLIGHT_SCRIPT
    for changed in (scene.replace("function draw(t){", "function draw(time){"), scene + "function draw(t){"):
        with pytest.raises(RuntimeError, match="anchor occurs"):
            module.add_finish_highlight(changed, data)


@pytest.mark.parametrize("luna_valid", [False, True])
@pytest.mark.parametrize("presentation", [False, True])
def test_schedule_uses_valid_official_finishes(tmp_path, monkeypatch, luna_valid, presentation):
    module = renderer()
    performance = tmp_path / "performance.json"
    hq = tmp_path / "hq.json"
    performance.write_text("{}")
    hq.write_text(json.dumps({"meshes": []}))
    monkeypatch.setattr(module, "_best_point", lambda *args: {"replay_url": "/replay/p.html"})
    monkeypatch.setattr(module, "_load_comparison_capture", lambda *args: {
        "body_names": [], "fps": 50, "frames": [[]], "runs": [{}],
        "presentation_extension": {
            "authoritative": presentation, "scoring_unchanged": presentation,
            "post_timeout_physical_terminal": {"reason": "finished", "distance_m": 100},
        },
    })
    def data(*args, **kwargs):
        return {"policies": [
            {"finish": 60, "valid": False},
            {"finish": 9.9, "valid": True},
            {"finish": 27.48, "valid": luna_valid},
        ], "meta": {}}
    monkeypatch.setattr(module, "capture_to_data", data)
    payload = module.build_comparison(performance, tmp_path, hq)
    expected = [{
        "policy_index": 1, "start_s": 9.4, "end_s": 9.9, "speed": 0.1,
        "focus_hold_s": 0.5,
    }]
    if luna_valid:
        expected.append({
            "policy_index": 2, "start_s": 27.28, "end_s": 27.48, "speed": 0.1,
            "focus_hold_s": 0.5,
        })
    expected.append({
        "policy_index": 0, "start_s": 37.25, "end_s": 37.75, "speed": 0.1,
        "focus_hold_s": 0.0,
    })
    assert payload["meta"]["finish_highlights"] == expected
    assert [policy["finish"] for policy in payload["policies"]] == [60, 9.9, 27.48]
    assert [policy["valid"] for policy in payload["policies"]] == [False, True, luna_valid]
    assert all(policy.get("presentation_result_distance_m") == (100 if presentation else None)
               for policy in payload["policies"])
    expected_orbits = [{
        "policy_index": 0, "start_s": 30.0, "rotate_s": 5.0,
        "hold_s": 5.0, "return_s": 5.0, "side_azimuth_degrees": -90,
    }]
    if luna_valid:
        expected_orbits.insert(0, {
            "policy_index": 2, "start_s": 12.48, "rotate_s": 5.0,
            "hold_s": 5.0, "return_s": 5.0, "side_azimuth_degrees": -90,
        })
    assert payload["meta"]["camera_orbits"] == expected_orbits


def test_camera_tours_rotate_uniformly_hold_and_return_without_accumulating():
    js_assertions("""
const sideOffset=-Math.PI/2-FOLLOW_VIEW.az;
for(const [policy,start] of [[2,12.48],[0,30]]){
  for(const [elapsed,fraction] of [[-1,0],[0,0],[1,.2],[2.5,.5],
    [5,1],[7.5,1],[10,1],[12.5,.5],[14,.2],[15,0],[16,0]]){
    near(comparisonOrbitOffset(start+elapsed,policy),sideOffset*fraction);
  }
  // Direct seeks and repeated paused draws cannot accumulate rotation.
  for(let i=0;i<5;i++)near(comparisonAutomaticView(start+7,policy).az,-Math.PI/2);
  near(comparisonAutomaticView(start,policy).az,-.25);
}
near(comparisonOrbitOffset(18,1),0); // GLM is never rotated.
near(comparisonOrbitOffset(40,2),0); // DeepSeek's tour never rotates Luna.
near(FOLLOW_VIEW.az,-.25);
// Luna's last 0.2 seconds still return smoothly while the clock slows down.
near(comparisonOrbitOffset(27.28,2),sideOffset*.04);
near(comparisonOrbitOffset(advanceFinishHighlight(27.28,2,1),2),0);
""")


def test_camera_injection_preserves_manual_controls_and_shared_replays():
    module = renderer()
    scene = (RENDERER / "scene.js").read_text()
    assert module.add_camera_orbits(scene, {"meta": {}}) == scene
    data = {"meta": {"camera_orbits": [{"policy_index": 0, "start_s": 30}]}}
    output = module.add_camera_orbits(scene, data)
    assert "automaticRunnerFollow?comparisonAutomaticView(t,runnerPolicyIndex):VIEW" in output
    # Drag and zoom inherit the current angle before disabling automation.
    assert "Object.assign(VIEW,comparisonAutomaticView(playT,currentAutoFollowPolicy));" in output
    assert "comparisonAutoCamera=false;" in output
    # Reset still re-enables the original auto camera without mutating its baseline.
    assert "Object.assign(VIEW,DEFAULT_VIEW);" in output
    for changed in (scene.replace("const activeView=", "const otherView="), scene + "const btn=document.getElementById('replay');"):
        with pytest.raises(RuntimeError, match="anchor occurs"):
            module.add_camera_orbits(changed, data)


def test_mobile_closeup_is_homepage_only_and_preserves_desktop_and_manual_zoom():
    module = renderer()
    scene = (RENDERER / "scene.js").read_text()
    output = module.add_mobile_closeup(scene)
    assert "runnerFollowComposition&&comparisonMobileLayout()?0.95:1" in output
    assert "responsiveScale:comparisonMobileLayout()?0.95:1" in output
    assert "followViewportWidth()<560?1.9:1" not in output
    # The same closeup covers the complete mobile layout, without a 560px jump.
    assert "return width<=720;" in output
    assert "VIEW.dist*factor" in output
    assert "Object.assign(VIEW,DEFAULT_VIEW);" in output
    assert 1.9 / 0.95 == 2
    for changed in (scene.replace("const compactFollowScale=", "const otherScale="), scene + scene):
        with pytest.raises(RuntimeError, match="anchor occurs"):
            module.add_mobile_closeup(changed)


def test_deepseek_side_view_middle_half_second_slows_without_changing_camera():
    js_assertions("""
near(advanceFinishHighlight(37.24,.01,1),37.25);
near(advanceFinishHighlight(37.25,5,1),37.75);
near(advanceFinishHighlight(37.75,.25,1),38);
for(const time of [37.25,37.5,37.75])near(comparisonAutomaticView(time,0).az,-Math.PI/2);
playT=37.5;syncFinishHighlight(playT);
assert.equal(finishHighlightState().policy,0);
assert.equal(finishHighlightState().effectiveSpeed,.1);
buttons[2].click();await Promise.resolve();
assert.equal(finishHighlightState().overridden,true);
near(advanceFinishHighlight(37.5,1,1),38.5);
""")


def test_each_automatic_slow_motion_window_highlights_button_and_clears_on_exit():
    js_assertions("""
for(const [start,end] of [[9.4,9.9],[27.28,27.48],[37.25,37.75]]){
  playT=start-.01;syncFinishHighlight(playT);
  assert.equal(buttons[0].classList.contains('automatic-slow-motion'),false);
  playT=start;syncFinishHighlight(playT);
  assert.equal(buttons[0].classList.contains('automatic-slow-motion'),true);
  assert.equal(buttons[0].pressed,'true');
  assert.equal(buttons[2].classList.contains('automatic-slow-motion'),false);
  syncFinishHighlight(playT);
  assert.equal(buttons[0].classList.contains('automatic-slow-motion'),true);
  playT=end;syncFinishHighlight(playT);
  assert.equal(buttons[0].classList.contains('automatic-slow-motion'),false);
}
playT=37.5;syncFinishHighlight(playT);buttons[2].click();await Promise.resolve();
assert.equal(buttons[0].classList.contains('automatic-slow-motion'),false);
cameraResetBtn.click();
assert.equal(buttons[0].classList.contains('automatic-slow-motion'),true);
""")
    css = renderer().SLOW_MOTION_CSS
    assert "animation:slow-motion-entry 1s ease-out 1" in css
    assert "@media(prefers-reduced-motion:reduce)" in css
    assert "animation:none" in css


def test_presentation_plaque_placement_does_not_change_result_labels_or_trial_scene():
    module = renderer()
    scene = (RENDERER / "scene.js").read_text()
    output = module.add_presentation_result_placement(scene)
    assert "Number.isFinite(p.presentation_result_distance_m)?p.presentation_result_distance_m:failureFrame(p)[1]-p.startX" in output
    assert "const label=policyPlaqueLabel(p);" in output
    assert "presentation_result_distance_m" not in scene
    assert "p.failed?failureFrame(p)[1]-p.startX+1.6:101.6" in scene
    with pytest.raises(RuntimeError, match="anchor occurs"):
        module.add_presentation_result_placement("changed scene")
