from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_TESTS = ROOT / "events/g1-100-metres/tests"
sys.path.insert(0, str(TASK_TESTS))

from verifier.replay import (  # noqa: E402
    DEFAULT_FPS,
    PoseRecorder,
    atomic_write,
    failure_modes,
    representative_index,
)


def load_web_renderer():
    path = ROOT / "web/renderers/g1-100-metres/render.py"
    spec = importlib.util.spec_from_file_location("g1_web_replay_renderer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_shell_declares_utf8_before_unicode_controls() -> None:
    template = (ROOT / "web/replay-template.html").read_text()
    assert template.startswith('<meta charset="utf-8">')


def test_model_hud_order_preserves_policy_and_physical_lane_identity() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    start = scene.index("function orderModelHudCards(){")
    helper = scene[start:scene.index("\norderModelHudCards();", start)]
    script = "const assert=require('node:assert/strict');\n" + helper + """
let IS_COMPARISON=true,IS_TRAJECTORY_COMPARISON=false;
const POL=[{identity:{model:'DeepSeek-V4-Flash'}},{label:'GLM-5.3-Flash'},{label:'GPT-5.6 Luna'}];
const cards=[{id:'lane0'},{id:'lane1'},{id:'lane2'}],appended=[];
const document={querySelector:()=>({append:card=>appended.push(card.id)})};
orderModelHudCards();
assert.deepEqual(appended,['lane0','lane2','lane1']);
assert.deepEqual(cards.map(card=>card.id),['lane0','lane1','lane2']);
assert.equal(POL[1].label,'GLM-5.3-Flash','physical policy indices stay unchanged');
appended.length=0;IS_TRAJECTORY_COMPARISON=true;orderModelHudCards();
assert.deepEqual(appended,[],'within-trial policy numbering remains unchanged');
IS_TRAJECTORY_COMPARISON=false;IS_COMPARISON=false;orderModelHudCards();
assert.deepEqual(appended,[],'standalone policy HUD is unchanged');
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def load_comparison_renderer():
    renderer_path = ROOT / "web/renderers/g1-100-metres"
    sys.path.insert(0, str(renderer_path))
    try:
        path = renderer_path / "render_comparison.py"
        spec = importlib.util.spec_from_file_location("g1_comparison_renderer", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(renderer_path))


def load_trial_comparison_renderer():
    renderer_path = ROOT / "web/renderers/g1-100-metres"
    sys.path.insert(0, str(renderer_path))
    try:
        path = renderer_path / "render_trial_comparison.py"
        spec = importlib.util.spec_from_file_location("g1_trial_comparison_renderer", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(renderer_path))


def lane(
    *,
    valid: bool,
    distance: float,
    finish: float | None = None,
    effective_speed: float = 0.0,
    peak: float = 0.0,
    failed: str | None = None,
) -> dict:
    checks = []
    if failed:
        checks.append({"name": failed, "passed": False})
    return {
        "valid": valid,
        "distance_m": distance,
        "max_distance_m": distance,
        "finish_time_s": finish,
        "effective_speed_mps": effective_speed,
        "termination_reason": failed or ("finished" if valid else "timeout"),
        "peak_speed_mps": peak,
        "checks": checks,
    }


def test_representative_prefers_highest_effective_speed_lane() -> None:
    rows = [
        lane(valid=False, distance=99.0, peak=9.0, effective_speed=10.6),
        lane(valid=True, distance=100.0, finish=12.0, effective_speed=8.333333),
        lane(valid=True, distance=100.0, finish=9.5, effective_speed=10.526316),
    ]
    assert representative_index(rows) == 0


def test_representative_failure_is_furthest_then_fastest() -> None:
    rows = [
        lane(valid=False, distance=25.0, peak=4.0, effective_speed=0.0),
        lane(
            valid=False,
            distance=40.0,
            peak=3.0,
            effective_speed=0.0,
            failed="in_lane",
        ),
        lane(
            valid=False,
            distance=40.0,
            peak=5.0,
            effective_speed=0.0,
            failed="self_collision",
        ),
    ]
    assert representative_index(rows) == 2
    assert failure_modes(rows[2]) == ["self_collision"]


def test_failure_without_explicit_gate_is_classified() -> None:
    assert failure_modes(lane(valid=False, distance=0.0)) == ["timeout"]


def test_replay_records_every_50_hz_policy_state() -> None:
    class Robot:
        body_names = ["pelvis"]

    class Origins:
        shape = (3, 3)

    recorder = PoseRecorder(Robot(), Origins(), control_hz=50.0)
    assert DEFAULT_FPS == 50.0
    assert recorder.sample_every == 1
    assert recorder.fps == 50.0


def test_web_replay_smoothly_interpolates_adjacent_authoritative_samples() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    renderer = (ROOT / "web/renderers/g1-100-metres/render.py").read_text()

    assert "authoritative captured-state playback" in scene
    assert "const i=Math.floor(frame),j=Math.min(last,i+1),alpha=" in scene
    assert "node.quaternion.copy(qA).slerp(qB,alpha)" in scene
    assert "A[o]+(B[o]-A[o])*alpha-sx" in scene
    assert "grp.add(node)" in scene
    assert "verifierTerminal" in scene
    assert "status=compactPolicyFailure(p)" in scene
    assert "'DISQUALIFIED'" not in scene
    assert "Disqualified:" not in scene
    assert "'TIMEOUT'" in scene
    assert "p.poseEndT=p.timedOut?captureEnd" in scene
    assert "visualDones[i]=p.timedOut?t>=p.freezeT:done" in scene
    assert "filter(i=>!visualDones[i])" in scene
    assert "new THREE.PlaneGeometry(4.8,.62)" in scene
    assert "wash.rotation.x=Math.PI/2" in scene
    assert "const CAMERA_FOLLOW={x:null,y:null,z:null,bodyOffsetX:null,span:null,t:null}" in scene
    assert "if(Math.abs(rawPackX-CAMERA_FOLLOW.x)>.018)" in scene
    assert "CAMERA_FOLLOW.y=rawPackY" in scene
    assert "if(Math.abs(desiredTargetZ-CAMERA_FOLLOW.z)>.012)" in scene
    assert "resetFollowDamping();startWall=null; draw(playT)" in scene
    assert "const activeIndices=xs.map((_,i)=>i).filter(i=>!visualDones[i])" in scene
    assert "camera.position.copy(camP)" in scene
    assert "g.computeBoundingBox()" in scene
    assert "groundRobotToTrack(rb)" in scene
    assert "rb.grp.position.z-=.12*ease" not in scene
    assert "DATA.colors" in scene
    assert "DATA.lane_indices" in scene
    assert "IS_COMPARISON" in scene
    assert "const IS_TRAJECTORY_COMPARISON=Boolean(DATA.meta&&DATA.meta.trajectory_comparison)" in scene
    assert "const REQUESTED_TRACK_LANES=Number(DATA.meta&&DATA.meta.track_lanes)" in scene
    assert "const TRACK_LANES=Number.isInteger(REQUESTED_TRACK_LANES)" in scene
    assert "{az:-0.82,el:0.34,dist:8.6}" in scene
    assert "DEFAULT_VIEW.dist+CAMERA_FOLLOW.span*.67" in scene
    assert "if(!IS_COMPARISON&&!IS_SCORING_EXAMPLE)" in scene
    assert "function laneNumberTexture(label,color='rgba(255,255,255,.97)')" in scene
    assert "map:laneNumberTexture(label)" in scene
    assert "const EXPLICIT_LANE_LABELS=Array.isArray(DATA.lane_labels)?DATA.lane_labels:null" in scene
    assert "const label=EXPLICIT_LANE_LABELS?EXPLICIT_LANE_LABELS[lane]:lane+1" in scene
    assert "if(label===null||label===undefined||label==='')continue" in scene
    assert "new THREE.PlaneGeometry(.62,1.18)" in scene
    assert "numeral.position.set(-3.2" in scene
    assert "numeral.rotation.z=" not in scene
    assert "TORSO_COLLAPSE" not in scene
    assert "lat>LANE_HALF" not in scene
    assert "runner_chest_bib" in scene
    assert "runner_chest_logo" in scene
    assert "runner_chest_cover" in scene
    assert "runner_chest_policy_number" in scene
    assert "policy.policy_number??policy.lane_number" in scene
    assert "IS_TRAJECTORY_COMPARISON?null:chestLogo(identity)" in scene
    assert "IS_TRAJECTORY_COMPARISON?chestPolicyNumber(p,identity):chestBib(identity,hex)" in scene
    assert "const emphasized=IS_TRAJECTORY_COMPARISON&&policy?.emphasized" in scene
    assert "emissive:emphasized?c:0x000000" in scene
    assert "metalness:emphasized?0.04:0.32" in scene
    assert "g.fillStyle='#fff';g.fillRect(0,0,c.width,c.height)" in scene
    assert "function policyNumberTexture(policy,identity)" in scene
    assert "material.toneMapped=false" in scene
    assert "curvedChestPlateGeometry(.228,.076,.117,.103,.040,10)" in scene
    assert "contouredChestGeometry(.234,.122,.226,.011,12,6)" in scene
    assert "contouredChestGeometry(.228,.116,.226,.012,12,6)" in scene
    assert "contouredChestGeometry(.274,.139,.226,.012,14,7)" in scene
    assert "identity.brand==='openai'?CHEST_OPENAI_LOGO_GEO" in scene
    assert "flattenMoldedChestBranding" not in scene
    assert "torsoChestSurfaceX(y,z)" in scene
    assert "const maxW=480,maxH=224" in scene
    assert "transparent:true,alphaTest:.02,depthWrite:false" in scene
    assert "grp.userData.policyIndex=ci" in scene
    assert "const raycaster=new THREE.Raycaster()" in scene
    assert "raycaster.intersectObjects(ROBOTS.map(robot=>robot.grp),true)" in scene
    assert "Math.hypot(e.clientX-startX,e.clientY-startY)>5" in scene
    assert "const focusedPolicy=validationPolicy??userFollowPolicy??(IS_TRAJECTORY_COMPARISON?null:currentAutoFollowPolicy)" in scene
    assert "const FOLLOW_VIEW={az:-0.25,el:0.13,dist:2.541,targetZ:0.63,targetXOffset:0}" in scene
    assert "const DEFAULT_VIEW=IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON" in scene
    assert "?{az:-0.82,el:0.34,dist:8.6}:{...FOLLOW_VIEW,az:IS_SIDE_EXAMPLE?-Math.PI/2:FOLLOW_VIEW.az}" in scene
    assert "let DEFAULT_FOLLOW_POLICY=defaultFollowPolicy()" in scene
    assert "let userFollowPolicy=DEFAULT_FOLLOW_POLICY" in scene
    assert "Object.assign(VIEW,{az:FOLLOW_VIEW.az,el:FOLLOW_VIEW.el,dist:FOLLOW_VIEW.dist})" in scene
    assert "const runnerFollowComposition=validationPolicy===null&&runnerPolicyIndex!==null" in scene
    assert "runnerFollowComposition?FOLLOW_VIEW.targetXOffset" in scene
    assert "const AUTO_FOLLOW_ORDER=POL.map((p,i)=>i).sort" in scene
    assert "const next=AUTO_FOLLOW_ORDER.find(i=>!visualDones[i])" in scene
    assert "IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON&&validationPolicy===null" in scene
    assert "IS_TRAJECTORY_COMPARISON?null:currentAutoFollowPolicy" in scene
    assert "comparisonFocusIndices=xs.map((_,i)=>i)" in scene
    assert "Math.max(focusMax-focusMin,occupiedLaneSpan)" in scene
    assert "automaticRunnerFollow?FOLLOW_VIEW:VIEW" in scene
    assert "runnerFollowComposition?FOLLOW_VIEW.targetZ" in scene
    assert "new THREE.Box3().setFromObject(ROBOTS[runnerPolicyIndex].grp)" in scene
    assert "Math.min(FOLLOW_VIEW.targetZ,Math.max(.10,center.z-.10))" in scene
    assert "if(bounds.max.z<1.25)desiredBodyOffsetX=center.x-xs[runnerPolicyIndex]" in scene
    assert "if(runnerFollowComposition)CAMERA_FOLLOW.x=cameraRootX(POL[runnerPolicyIndex],t)" in scene
    assert "camT.set(packX+CAMERA_FOLLOW.bodyOffsetX,packY,targetZ)" in scene
    assert "CAMERA_FOLLOW.bodyX" not in scene
    assert "desiredTargetZ=cameraPostureTargetZ(POL[runnerPolicyIndex],t,desiredTargetZ)" in scene
    assert "const compactFollowScale=IS_SCORING_EXAMPLE?1:(runnerFollowComposition&&followViewportWidth()<560\n    ?1.9:1)" in scene
    assert "responsiveScale:IS_SCORING_EXAMPLE?1:(followViewportWidth()<560?1.9:1)" in scene
    assert "if(singlePolicy&&!IS_TRAJECTORY_COMPARISON)" in scene
    assert "card.querySelector('.policy-remove')?.addEventListener('click'" in scene
    assert "event.preventDefault();event.stopPropagation();" in scene
    assert "if(compactFollowScale>1||IS_SCORING_EXAMPLE)desiredTargetZ+=.06" in scene
    assert ":VIEW.dist))*compactFollowScale" in scene
    assert "const followedPolicy=userFollowPolicy??(IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON?currentAutoFollowPolicy:null)" in scene
    assert "card.setAttribute('aria-current','true')" in scene
    assert "cameraModeEl.textContent='';cameraModeEl.hidden=true" in scene
    assert "Math.min(Math.max(0,t),p.eventT).toFixed(2)+'s'" in scene
    assert "VIEW.dist=Math.min(48,Math.max(1.6,VIEW.dist*factor))" in scene
    assert "zoomCamera(Math.exp(e.deltaY*.001))" in scene
    assert "follow(policy){return followPolicy(policy);}" in scene
    assert "orbit(deltaAz,deltaEl=0){return orbitCamera(deltaAz,deltaEl);}" in scene
    assert "zoom(factor){return zoomCamera(factor);}" in scene
    assert "position:camera.position.toArray(),target:camT.toArray()" in scene
    assert "resetCamera," in scene
    assert "cameraResetBtn.textContent='Reset'" in scene
    assert "Reset camera" not in scene
    assert "'Reset</button>'" in renderer
    assert '"lane_indices": [1 for _ in policies]' in renderer
    assert '"track_lanes": 3' in renderer
    assert "button,[role=button],a,input,select,textarea" in scene
    assert "if(REPLAY_AUTOPLAY&&!DATA.meta?.start_paused" in scene
    assert "playing=true; lastNow=null;startWall=null" in scene
    assert "window.addEventListener('resize',()=>{resize();if(!playing)draw(playT);})" in scene
    template = (ROOT / "web/replay-template.html").read_text().split(
        "<script>const DATA=", 1
    )[0]
    assert ".hud{position:absolute;z-index:2" in template
    assert 'id="camera-zoom-in"' in template
    assert 'id="camera-zoom-out"' in template
    assert 'id="camera-reset"' in template
    assert 'id="camera-mode" aria-live="polite"' in template
    assert 'data-s="0.5" aria-pressed="false">0.5&times;</button>' in template
    assert 'data-s="0.6"' not in template
    assert "pointer-events:auto" in template
    assert (
        '"interpolation": '
        '"adjacent_authoritative_position_lerp_quaternion_slerp"'
        in renderer
    )


def test_scoring_example_layout_is_opt_in_and_uses_available_stage_height() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    template = (ROOT / "web/replay-template.html").read_text()
    mode = next(line for line in scene.splitlines() if line.startswith("const IS_SCORING_EXAMPLE="))
    resize = next(line for line in scene.splitlines() if line.startswith("function resize(){"))
    script = "const assert=require('node:assert/strict');\n"
    script += "function dimensions(comparison,search,height){\nconst IS_COMPARISON=comparison,location={search};\n"
    script += mode + "\n"
    script += "const stage={clientWidth:235,clientHeight:height},sizes=[];\n"
    script += "const document={documentElement:{classList:{toggle(){}}}},replayMobileLayout=()=>false;\n"
    script += "const renderer={setSize:(w,h)=>sizes.push([w,h])},camera={updateProjectionMatrix(){}};\n"
    script += resize + "\nresize();return {sizes,aspect:camera.aspect};}\n"
    script += """
assert.deepEqual(dimensions(false,'?example=1',149).sizes,[[235,149]]);
assert.equal(dimensions(false,'?example=1',149).aspect,235/149);
assert.deepEqual(dimensions(false,'',149).sizes,[[235,132]]);
assert.deepEqual(dimensions(true,'?example=1',149).sizes,[[235,132]]);
assert.deepEqual(dimensions(false,'?example=0',149).sizes,[[235,132]]);
assert.deepEqual(dimensions(false,'?example=1',0).sizes,[[235,1]]);
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '<style id="scoring-example-layout">' in template
    assert ".replay-example .stagewrap{height:100%;display:grid;grid-template-rows:40px minmax(0,1fr) 68px}" in template
    assert ".replay-example #stage canvas{height:100%!important}" in template
    assert ".replay-example .ctl{bottom:36px" in template
    assert ".replay-example .camera-ctl{bottom:4px" in template
    assert "@media(min-width:340px)" in template
    assert "document.documentElement.classList.toggle('replay-example',IS_SCORING_EXAMPLE)" in scene
    assert "if(!IS_COMPARISON&&!IS_SCORING_EXAMPLE)" in scene
    assert "viewport:{width:stage.clientWidth,height:stage.clientHeight,aspect:camera.aspect}" in scene


def test_scoring_examples_omit_track_numerals_without_changing_regular_replays() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    track_tex = scene.split("function trackTex(){", 1)[1].split("\nconst track=", 1)[0]
    script = "const assert=require('node:assert/strict');\n"
    script += """
function labels(example,comparison=false){
  const IS_SCORING_EXAMPLE=example,IS_COMPARISON=comparison,TRACK_LANES=3,TRACK_LANE_WIDTH=1.22,TRACK_HALF_WIDTH=1.83;
  const painted=[];
  const g={fillText:text=>painted.push(text),fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){}};
  const document={createElement:()=>({getContext:()=>g})};
  const THREE={CanvasTexture:class{},sRGBEncoding:1};
"""
    script += "function trackTex(){" + track_tex + "\ntrackTex();return painted;}\n"
    script += "assert.deepEqual(labels(true),[]);assert.deepEqual(labels(false),['1','2','3']);assert.deepEqual(labels(false,true),[]);"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_web_replay_maps_model_identity_to_chest_bib() -> None:
    renderer = load_web_renderer()
    assert renderer.model_identity("openai/gpt-5.6-sol") == {
        "brand": "openai",
        "company": "OpenAI",
        "model": "GPT‑5.6 Sol",
        "color": "#2279DC",
        "logo": "/assets/model-logos/openai.svg",
    }
    assert renderer.model_identity("deepseek/deepseek-v4-flash-vision-exp") == {
        "brand": "deepseek",
        "company": "DeepSeek",
        "model": "DeepSeek-V4-Flash",
        "color": "#7C54CD",
        "logo": "/assets/model-logos/deepseek.svg",
    }


def test_homepage_comparison_uses_current_three_model_cohort() -> None:
    comparison = (
        ROOT / "web/renderers/g1-100-metres/render_comparison.py"
    ).read_text()
    assert '("deepseek-v4-flash-vision-exp", "DeepSeek-V4-Flash"' in comparison
    assert '("glm-5.3-flash", "GLM‑5.3‑Flash"' in comparison
    assert '("gpt-5.6-luna", "GPT‑5.6 Luna"' in comparison
    assert '"gpt-5.6-sol"' not in comparison
    assert '"claude-opus-5"' not in comparison
    assert '"identity": model_identity(competitor_needle)' in comparison
    assert 'data["lane_indices"] = list(range(len(COMPETITORS)))' in comparison
    assert '"track_lanes": len(COMPETITORS)' in comparison


def test_web_replay_never_infers_self_collision_from_a_fallen_pose() -> None:
    renderer = load_web_renderer()
    # A historical capture can contain a very low torso while the verifier's
    # self-collision gate explicitly passed. That posture is not a diagnosis.
    frames = [[0.0, 0.0, 0.0, 0.2], [60.0, 0.8, 0.0, 0.1]]
    run = {
        "valid": False,
        "duration_s": 60.0,
        "checks": [
            {"name": "finished", "passed": False},
            {"name": "self_collision", "passed": True, "value": 0.0},
        ],
    }
    assert renderer.compute_terminal_event(
        frames,
        ["torso_link"],
        run,
        failure_modes=["finished"],
    ) == (60.0, "did_not_finish")


def test_web_replay_uses_explicit_verifier_disqualification() -> None:
    renderer = load_web_renderer()
    run = {
        "valid": False,
        "first_disqualification_time_s": 0.774,
        "first_disqualification_gate": "self_collision",
    }
    assert renderer.compute_terminal_event([[0.0]], [], run) == (
        0.774,
        "self_collision",
    )


def test_timeout_keeps_recorded_visual_tail_without_changing_terminal_time() -> None:
    renderer = load_web_renderer()
    frames = [
        [time, time, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        for time in (0.0, 60.0, 61.0, 63.0)
    ]
    capture = {
        "body_names": ["pelvis"],
        "fps": 1.0,
        "frames": [frames],
        "runs": [
            {
                "valid": False,
                "stop_time_s": 60.0,
                "duration_s": 60.0,
                "termination_reason": "timeout",
            }
        ],
    }
    data = renderer.capture_to_data(capture, {"pelvis": {}}, "timeout policy")
    policy = data["policies"][0]
    assert policy["terminal_time"] == 60.0
    assert policy["terminal_reason"] == "timeout"
    assert policy["timed_out"] is True
    assert policy["frames"][-1][0] == 63.0


def test_comparison_accepts_only_exact_authoritative_presentation_tail(tmp_path: Path) -> None:
    comparison = load_comparison_renderer()
    replay_name = "frontier-policy"
    official = {
        "policy_sha256": "abc123",
        "runs": [{"valid": False, "termination_reason": "timeout", "stop_time_s": 60.0}],
        "frames": [[[0.0, 1.0], [60.0, 2.0]]],
    }
    presentation = {
        **official,
        "frames": [[*official["frames"][0], [61.0, 3.0]]],
        "presentation_extension": {
            "authoritative": True,
            "scoring_unchanged": True,
        },
    }
    official_path = tmp_path / f"{replay_name}.json"
    official_path.write_text(json.dumps(official))
    presentation["presentation_extension"]["official_capture_sha256"] = hashlib.sha256(
        official_path.read_bytes()
    ).hexdigest()
    (tmp_path / f"{replay_name}.presentation.json").write_text(json.dumps(presentation))
    loaded = comparison._load_comparison_capture(
        tmp_path, replay_name, {"policy_sha256": "abc123"}
    )
    assert loaded["frames"][0][-1][0] == 61.0

    presentation["frames"][0][1][1] = 99.0
    (tmp_path / f"{replay_name}.presentation.json").write_text(json.dumps(presentation))
    try:
        comparison._load_comparison_capture(
            tmp_path, replay_name, {"policy_sha256": "abc123"}
        )
    except RuntimeError as exc:
        assert "does not preserve the official prefix" in str(exc)
    else:
        raise AssertionError("a changed official pose prefix must be rejected")


def test_atomic_replay_write_replaces_complete_json(tmp_path: Path) -> None:
    target = tmp_path / "replay.json"
    atomic_write(target, {"schema_version": 1, "frames": [[[0.0]]]})
    assert json.loads(target.read_text())["frames"] == [[[0.0]]]
    assert not list(tmp_path.glob(".replay.json.*"))


def test_trial_comparison_registry_uses_true_submission_numbers() -> None:
    renderer = load_trial_comparison_renderer()
    registry = renderer.load_registry(renderer.POLICY_INDEX_DEFAULT)
    assert len(registry) >= 208
    assert registry["frontier-67698114effc"]["policyNumber"] == 3
    assert registry["frontier-5d98be719669"]["policyNumber"] == 10
    assert registry["frontier-8ee6d478cb79"]["policyNumber"] == 3
    assert registry["frontier-8ee6d478cb79"]["color"] == "#7C54CD"
    assert registry["frontier-e614d4450dab"]["policyNumber"] == 2
    assert registry["frontier-e614d4450dab"]["runId"] == "s10-vexp-r123-20260828-luna-5"
    assert all(item["captureId"].startswith("frontier-") for item in registry.values())
    assert all((ROOT / "web" / item["url"].lstrip("/")).is_file() for item in registry.values())


def test_trial_registry_recovers_only_exact_available_artifacts(tmp_path, monkeypatch) -> None:
    renderer = load_trial_comparison_renderer()
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    captures = tmp_path / "web/captures"
    captures.mkdir(parents=True)
    valid_sha, missing_sha, wrong_sha = "a" * 64, "b" * 64, "c" * 64
    policies = [
        {"submission_index": index, "policy_sha256": sha, "replay_ready": False, "replay_url": None}
        for index, sha in enumerate([valid_sha, missing_sha, wrong_sha, "invalid"], 1)
    ]
    (tmp_path / "web/policies.json").write_text(json.dumps({"run_id": "original-run", "model": "luna", "policies": policies}))
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"runs": [{"path": "policies.json"}]}))
    capture = {"schema_version": 2, "body_names": ["pelvis"], "frames": [[[0, 0, 0, 0, 0, 0, 0, 1]]], "policy_sha256": valid_sha}
    (captures / f"frontier-{valid_sha[:12]}.json").write_text(json.dumps(capture))
    (captures / f"frontier-{wrong_sha[:12]}.json").write_text(json.dumps(capture))
    registry = renderer.load_registry(index_path)
    assert list(registry) == [f"frontier-{valid_sha[:12]}"]
    assert registry[f"frontier-{valid_sha[:12]}"]["policyNumber"] == 1
    assert registry[f"frontier-{valid_sha[:12]}"]["runId"] == "original-run"


def test_camera_root_filter_tracks_live_speed_without_phase_lag() -> None:
    """Execute the actual JS camera filter at live frame cadences, not a seek."""
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    body = scene.split("function cameraRootX(p,t){", 1)[1].split("\n}\n", 1)[0]
    script = "const assert=require('node:assert/strict');const FPS=50;\n"
    script += "function cameraRootX(p,t){" + body + "\n}\n"
    script += r"""
const linear={startX:0,poseEndT:10,frames:Array.from({length:501},(_,i)=>[i/FPS,13*i/FPS])};
for(const hz of [30,60,120]){
  for(let frame=1;frame<hz*9;frame++){
    const t=frame/hz,expected=13*t,actual=cameraRootX(linear,t);
    if(t>.1)assert.ok(Math.abs(actual-expected)<1e-9,`live ${hz}Hz t=${t}: lag ${expected-actual}`);
    assert.ok(Math.abs(actual-expected)<=.180000001);
  }
}
for(const t of [0,.01,.05,9.95,9.99,10,10.1,36.8]){
  assert.ok(Math.abs(cameraRootX(linear,t)-13*Math.min(10,t))<=.180000001);
}
assert.equal(cameraRootX(linear,36.8),130); // A manually followed finisher stays centred.
const gait={...linear,frames:linear.frames.map(([t,x])=>[t,x+.07*Math.sin(t*Math.PI*16)])};
let rawError=0,filteredError=0;
for(let i=10;i<490;i++){
  const t=i/FPS;rawError+=(gait.frames[i][1]-13*t)**2;
  filteredError+=(cameraRootX(gait,t)-13*t)**2;
}
assert.ok(filteredError<rawError*.2,'centred filter should remove gait shake');
"""
    result = subprocess.run(["node", "-e", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_camera_posture_stabilizes_crawl_and_preserves_upright_framing() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    body = scene.split("function cameraPostureTargetZ(p,t,uprightTargetZ){", 1)[1].split("\n}\n", 1)[0]
    policies = {}
    for label, capture_id in {
        "crawl": "frontier-76d7d31f8c51",
        "glm": "frontier-0f0462090cee",
        "luna": "frontier-7a933a7feef9",
    }.items():
        capture = json.loads((ROOT / f"web/captures/{capture_id}.json").read_text())
        offset = 3 + 7 * capture["body_names"].index("pelvis")
        frames = [[row[0], 0, 0, row[offset]] for row in capture["frames"][0]]
        policies[label] = {"frames": frames, "poseEndT": frames[-1][0]}
    script = "const assert=require('node:assert/strict');const FPS=50,PELVIS_I=0;const LO=i=>1+7*i;\n"
    script += "function cameraPostureTargetZ(p,t,uprightTargetZ){" + body + "\n}\n"
    script += "const policies=" + json.dumps(policies) + ";\n"
    script += r"""
const crawlTargets=[];
for(let i=250;i<policies.crawl.frames.length;i++){
  const t=i/FPS;
  // Simulate the changing bounds of raised arms/feet using the real low torso.
  crawlTargets.push(cameraPostureTargetZ(policies.crawl,t,.18+.10*Math.sin(t*8)));
}
assert.ok(crawlTargets.length>2000);
assert.equal(Math.max(...crawlTargets)-Math.min(...crawlTargets),0,'sustained crawl camera must not bob');
assert.equal(crawlTargets[0],.18);
const groundY=z=>-z*Math.cos(.13)/(2.541+z*Math.sin(.13))/Math.tan(Math.PI/8);
assert.equal(Math.max(...crawlTargets.map(groundY))-Math.min(...crawlTargets.map(groundY)),0,'ground plane must stay fixed');
for(const name of ['glm','luna']){
  for(let i=0;i<policies[name].frames.length;i++){
    const t=i/FPS,existing=.53+.06*Math.sin(t*6);
    assert.ok(Math.abs(cameraPostureTargetZ(policies[name],t,existing)-existing)<1e-12,`${name} upright framing changed`);
  }
}
const fall={poseEndT:4,frames:Array.from({length:201},(_,i)=>{
  const t=i/FPS,z=.74-.52*Math.max(0,Math.min(1,t-1));return [t,0,0,z];
})};
let last=.6,maxStep=0;
for(let i=0;i<=240;i++){
  const target=cameraPostureTargetZ(fall,i/60,.6);
  assert.ok(target<=last+1e-12,'standing-to-crawl camera transition must be monotonic');
  maxStep=Math.max(maxStep,last-target);last=target;
}
assert.ok(maxStep<.035,'posture transition must not snap');
assert.equal(last,.18);
"""
    result = subprocess.run(["node"], input=script, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_trajectory_camera_defaults_to_emphasis_and_reset_restores_it() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    functions = []
    for signature in ["defaultFollowPolicy()", "followPolicy(index)", "resetCamera()"]:
        body = scene.split(f"function {signature}{{", 1)[1].split("\n}\n", 1)[0]
        functions.append(f"function {signature}{{" + body + "\n}\n")
    script = "const assert=require('node:assert/strict');\n"
    script += "function check(POL,IS_TRAJECTORY_COMPARISON,expected){\n"
    script += "const IS_COMPARISON=true,FOLLOW_VIEW={az:-.25,el:.13,dist:2.541};\n"
    script += "const DEFAULT_VIEW={...FOLLOW_VIEW},VIEW={...DEFAULT_VIEW};let playing=true,playT=0;\n"
    script += "const resetFollowDamping=()=>{},updateCameraControls=()=>{},startCameraSwitch=()=>{},cancelCameraSwitch=()=>{};\n"
    script += "\n".join(functions)
    script += r"""
let DEFAULT_FOLLOW_POLICY=defaultFollowPolicy();
let userFollowPolicy=DEFAULT_FOLLOW_POLICY,comparisonAutoCamera=IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON;
const emphasizePolicy=index=>{if(IS_TRAJECTORY_COMPARISON)DEFAULT_FOLLOW_POLICY=index;};
assert.equal(userFollowPolicy,expected);
assert.equal(followPolicy(0),true);
VIEW.dist=9;VIEW.az=1;
const reset=resetCamera();
assert.equal(reset.follow,IS_TRAJECTORY_COMPARISON?0:expected);
assert.equal(reset.auto,!IS_TRAJECTORY_COMPARISON);
assert.deepEqual(reset.view,FOLLOW_VIEW);
}
check([{emphasized:true}],true,0);
check([{emphasized:false},{emphasized:true},{emphasized:false}],true,1);
check([{emphasized:true},{emphasized:false}],true,0);
check([{emphasized:false},{emphasized:false}],true,1);
check([{emphasized:true},{emphasized:false}],false,null);
"""
    result = subprocess.run(["node"], input=script, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_trajectory_focus_recolors_robot_and_uses_parent_viewport() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    script = "const assert=require('node:assert/strict');\n"
    for signature in ["emphasizePolicy(index)", "followViewportWidth()"]:
        body = scene.split(f"function {signature}{{", 1)[1].split("\n}\n", 1)[0]
        script += f"function {signature}{{" + body + "\n}\n"
    script += r"""
const IS_TRAJECTORY_COMPARISON=true,stage={clientWidth:534};
const events=[],window={parent:{innerWidth:932},dispatchEvent:e=>events.push(e)};
class CustomEvent{constructor(type,options){this.type=type;this.detail=options.detail;}}
assert.equal(followViewportWidth(),932);window.parent.innerWidth=390;assert.equal(followViewportWidth(),390);
window.parent=window;assert.equal(followViewportWidth(),534);
const POL=[{capture_id:'first',model_color:'#7C54CD'},{capture_id:'second',model_color:'#7C54CD'}];
const COL=[0xffffff,0x7c54cd],cards=[null,null];let DEFAULT_FOLLOW_POLICY=1;
const ROBOTS=POL.map(()=>({material:{color:null,copy(m){this.color=m.color;}}}));
const shellMat=i=>({color:COL[i],dispose(){}});
let disposedTextures=0;
const plaque=(label,color)=>({userData:{label,color},dispose(){disposedTextures++}});
const policyPlaqueLabel=()=> 'TIMEOUT';
const PLAQUES=POL.map(()=>({material:{map:plaque('TIMEOUT','#FFFFFF')}}));
emphasizePolicy(0);
assert.equal(DEFAULT_FOLLOW_POLICY,0);assert.equal(ROBOTS[0].material.color,0x7c54cd);assert.equal(ROBOTS[1].material.color,0xffffff);
assert.deepEqual(POL.map(p=>p.emphasized),[true,false]);
assert.equal(events.at(-1).type,'g1:policy-focused');assert.equal(events.at(-1).detail.captureId,'first');
emphasizePolicy(1);
assert.equal(ROBOTS[0].material.color,0xffffff);assert.equal(ROBOTS[1].material.color,0x7c54cd);
assert.deepEqual(POL.map(p=>p.capture_id),['first','second']);
assert.equal(PLAQUES[0].material.map.userData.color,'#FFFFFF');
assert.equal(PLAQUES[1].material.map.userData.color,'#7C54CD');
assert.equal(disposedTextures,3,'replaced color textures are disposed');
"""
    result = subprocess.run(["node"], input=script, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_trial_comparison_shell_contract_and_compact_eight_lanes() -> None:
    renderer = load_trial_comparison_renderer()
    registry = {
        f"frontier-{index:012x}": {
            "captureId": f"frontier-{index:012x}",
            "url": f"/captures/frontier-{index:012x}.json",
            "policyNumber": number,
            "label": f"Policy #{number}",
            "color": "#66D693",
        }
        for index, number in enumerate((3, 10, 11, 12, 13, 14, 15, 16), start=1)
    }
    shell = renderer.build_shell(registry=registry, hq={"pelvis": {}})
    assert "g1:set-policies" in shell
    assert "g1:policies-ready" in shell
    assert "g1:policies-state" in shell
    assert "g1:policies-error" in shell
    assert "g1:policy-focused" in shell
    assert "g1:policy-remove" in shell
    assert 'class="policy-remove" aria-label="Remove ${name}"' in shell
    assert "model_color:item.color" in shell
    assert "history.replaceState(null,'',canonicalUrl(activeState))" in shell
    assert "activeState=null,rendererReady=false" in shell
    assert "function acknowledge(state,generation,serial)" in shell
    assert "serial!==requestSerial||generation!==REPLAY_GENERATION" in shell
    assert "failedCaptureIds" in shell
    assert "lanesEl.innerHTML=''" in shell
    assert "At most 8 policies can be compared." in shell
    assert "track_lanes:8" in shell
    assert "lane_indices:policies.map((_,i)=>i)" in shell
    assert "lane_labels:Array(8).fill(null)" in shell
    assert "start_paused:true" in shell
    assert "Number(raw.schema_version)!==2" in shell
    assert "window.__G1_REPLAY__.updatePolicies(policies" in shell
    assert "location.replace(next)" not in shell
    assert "sessionStorage.setItem(TRIAL_STORAGE" in shell
    assert "function colorHex(value)" in shell
    assert "colorHex(fallback.color)||colorHex(item.color)||'#6E97C4'" in shell
    assert "const color=colorHex(fallback.color)" in shell
    assert "renderColor:item.captureId===emphasis?item.color:'#FFFFFF'" in shell
    assert "Policy #3" in shell and "Policy #10" in shell
    assert "Unknown policy capture" in shell
    assert "error.failedCaptureIds=missing" in shell
    assert "let state;try{state=initialState();}" in shell


def test_trial_comparison_latches_document_nonce_but_accepts_new_selections() -> None:
    shell = load_trial_comparison_renderer().build_shell(registry={}, hq={})
    bootstrap = shell.split("const TRIAL_STORAGE=", 1)[1].split("function terminal(", 1)[0]
    script = r"""
const assert=require('node:assert/strict'),messages=[],listeners={},calls=[];
const parent={postMessage(message){messages.push(message)}};
const window={addEventListener(type,handler){listeners[type]=handler}};
const document={querySelector(){return {}},getElementById(){return {}}};
const location={origin:'https://example.test',href:'https://example.test/replay?replayGeneration=41',get search(){return new URL(this.href).search}};
const sessionStorage={setItem(){}},history={replaceState(){}};
const TRIAL_BOOT={registry:{'frontier-000000000001':{policyNumber:1,color:'#7C54CD'}}};
function setSelection(state,generation){calls.push(generation);REPLAY_GENERATION=generation;activeState=state;}
"""
    script += "const TRIAL_STORAGE=" + bootstrap
    script += r"""
const selection={type:'g1:set-policies',replayGeneration:41,replayDocumentGeneration:41,policies:[{captureId:'frontier-000000000001',policyNumber:1}]};
const send=(data,source=parent)=>listeners.message({origin:location.origin,source,data});
send({...selection,replayDocumentGeneration:40});send({...selection,replayGeneration:undefined});send(selection,{});
assert.equal(calls.length,0);
send(selection);send({...selection,replayGeneration:42});
assert.deepEqual(calls,['41','42']);
send(selection);assert.equal(calls.length,2,'stale selection version rejected');
location.href='https://example.test/replay?replayGeneration=99&replayDocumentGeneration=99';
assert.equal(DOCUMENT_GENERATION,'41');
for(const type of ['g1:policies-ready','g1:policies-state','g1:policies-error','g1:policy-focused','g1:policy-remove']){
  tell(type,{replayGeneration:'forged',replayDocumentGeneration:'forged'});
  assert.equal(messages.at(-1).replayGeneration,'42');
  assert.equal(messages.at(-1).replayDocumentGeneration,'41');
}
assert.equal(canonicalUrl(activeState).searchParams.get('replayDocumentGeneration'),'41');
"""
    result = subprocess.run(["node"], input=script, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_terminal_reason_is_a_focused_clock_badge_not_a_notice() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    template_head = (ROOT / "web/replay-template.html").read_text().split("<script>const DATA=", 1)[0]
    body = scene.split("function terminalStatus(policy,done){", 1)[1].split("\n}\n", 1)[0]
    script = "const assert=require('node:assert/strict');function terminalStatus(policy,done){" + body + "\n}\n"
    compact = scene.split("function compactPolicyFailure(policy){", 1)[1].split("\n}\n", 1)[0]
    script += "function compactPolicyFailure(policy){" + compact + "\n}\n"
    script += r"""
const dq=reason=>({failed:true,disqualified:true,terminal:{reason}});
assert.deepEqual(terminalStatus(dq('self_collision'),false),{kind:'',text:''});
assert.deepEqual(terminalStatus(null,true),{kind:'',text:''});
assert.deepEqual(terminalStatus(dq('self_collision'),true),{kind:'incomplete',text:'COLLISION'});
assert.deepEqual(terminalStatus(dq('in_lane'),true),{kind:'incomplete',text:'LANE DRIFT'});
assert.deepEqual(terminalStatus({failed:true,terminal:{reason:'body_height'}},true),{kind:'incomplete',text:'FELL'});
assert.equal(terminalStatus({failed:true,timedOut:true},true).text,'TIMEOUT');
assert.equal(terminalStatus({failed:false},true).text,'FINISHED');
assert.deepEqual(terminalStatus({failed:true},true),{kind:'incomplete',text:'DNF'});
"""
    result = subprocess.run(["node"], input=script, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "hud(t,xs,dones,runnerPolicyIndex)" in scene
    assert "const index=singlePolicy?0:focusedPolicy" in scene
    assert "failureNoteEl" not in scene  # Reserved solely for loading/error notices.
    assert "failurePresentation" not in scene  # No stale call when constructing DQ markers.
    assert "if(reason==='in_lane') return laneFailureMarker(p,ci)" in scene
    assert "if(reason==='self_collision') return selfCollisionMarker(p,ci)" in scene
    assert '.hud:has(.clock-status:not(:empty)) .lanes{top:88px}' in template_head
    assert '.failure-note{position:absolute;top:50%;left:50%' in template_head
    assert '.clock-status[data-kind="disqualified"]' not in template_head
    assert '.clock-status[data-kind="incomplete"]{border-color:rgba(255,209,102,.62);background:rgba(58,39,5,.88);color:#ffd166;' in template_head


def test_track_result_plaques_show_outcomes_and_fit_the_shared_finish_style() -> None:
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    script = "const assert=require('node:assert/strict');\n"
    for signature in ["compactPolicyFailure(policy)", "policyPlaqueLabel(policy)", "plaque(txt,hex)"]:
        body = scene.split(f"function {signature}{{", 1)[1].split("\n}\n", 1)[0]
        script += f"function {signature}{{" + body + "\n}\n"
    script += """
const drawn=[];
const document={createElement:()=>({getContext:()=>({
  beginPath(){},roundRect(){},fill(){},stroke(){},
  measureText(text){return {width:parseInt(this.font.match(/\\d+/)[0])*text.length*.6}},
  fillText(text){drawn.push({text,font:this.font,width:this.measureText(text).width})},
})})};
const THREE={CanvasTexture:class{constructor(image){this.image=image}},sRGBEncoding:3001};
const cases=[
  [{failed:false,finish:9.9},'9.90s'],
  [{failed:true,timedOut:true},'TIMEOUT'],
  [{failed:true,disqualified:true,terminal:{reason:'self_collision'}},'COLLISION'],
  [{failed:true,disqualified:true,terminal:{reason:'in_lane'}},'LANE DRIFT'],
  [{failed:true},'DNF'],
];
for(const [policy,expected] of cases){
  assert.equal(policyPlaqueLabel(policy),expected);
  const texture=plaque(expected,'#abcdef');
  assert.equal(texture.userData.label,expected);
  assert.match(texture.userData.font,/^bold \\d+px -apple-system,Segoe UI,sans-serif$/);
}
assert.ok(drawn.every(row=>row.width<=456),'outcome text must fit without clipping or horizontal compression');
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "m.position.set(p.failed?failureFrame(p)[1]-p.startX+1.6:101.6" in scene
    assert "if(p.failed){\n      pl.visible=done;" in scene
    assert "results(){return PLAQUES.map" in scene


def test_replay_speed_captions_use_two_decimals_without_rounding_capture_data() -> None:
    renderer = load_web_renderer()
    data = {"policies":[{"finish":9.9,"effective_speed_mps":10.101234}],"raw_speed":10.101234}
    html = renderer.assemble_html(
        data,title="Replay",eyebrow="REPLAY",headline="10.101 m/s",
        lede="A 3.641 m/s runner",cap="Effective Speed 0.12345 m/s",
        story="Replay",sr_only="Replay at 0 m/s",active="test",
    )
    head,payload=html.split("<script>const DATA=",1)
    assert "10.10 m/s" in head
    assert "3.64 m/s" in head
    assert "0.12 m/s" in head
    assert "0.00 m/s" in head
    assert '"raw_speed":10.101234' in payload
    assert '"effective_speed_mps":10.101234' in payload


def test_replay_stopped_policy_keeps_raw_verifier_state_without_a_dq_time_prefix() -> None:
    renderer = load_web_renderer()
    data = {"policies": [{"valid": False, "disqualified": True,
                         "terminal_reason": "in_lane", "terminal_time": 3.27}]}
    html = renderer.assemble_html(
        data, title="Replay", eyebrow="REPLAY", headline="Replay",
        lede="Replay", cap="Effective Speed 0.18 m/s", story="Distance freezes at the first stop.",
        sr_only="Replay", active="test",
    )
    head, payload = html.split("<script>const DATA=", 1)
    assert ">3.27s</span>" in head
    assert "DQ " not in head
    assert "disqualif" not in head.lower()
    assert "distance freezes at the first stop" in head
    assert '"disqualified":true' in payload
    assert '"terminal_reason":"in_lane"' in payload
