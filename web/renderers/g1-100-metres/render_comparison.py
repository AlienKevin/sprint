#!/usr/bin/env python3
"""Build the homepage race from the best replayable policy per model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from render import assemble_html, capture_to_data, model_identity, shared_asset_html


ROOT = Path(__file__).resolve().parents[3]
HQ_DEFAULT = Path(__file__).with_name("g1_hq.json")
COMPETITORS = (
    ("deepseek-v4-flash-vision-exp", "DeepSeek-V4-Flash", "#7C54CD"),
    ("glm-5.3-flash", "GLM‑5.3‑Flash", "#39B8B2"),
    ("gpt-5.6-luna", "GPT‑5.6 Luna", "#5EDC9A"),
)


FINISH_HIGHLIGHT_SCRIPT = r"""
// Homepage-only finish highlights. The base speed remains the user's choice;
// piecewise wall-time integration changes the one shared simulation clock.
const FINISH_HIGHLIGHTS=DATA.meta.finish_highlights;
const finishHighlightOverrides=new Set();
let finishHighlightLastTime=0;
function finishHighlightAt(t){
  return FINISH_HIGHLIGHTS.find(h=>t>=h.start_s&&t<h.end_s);
}
function finishHighlightActive(t){
  const highlight=finishHighlightAt(t);
  return !!highlight&&!finishHighlightOverrides.has(highlight.policy_index);
}
function finishHighlightCamera(t){
  return finishHighlightAt(t)||FINISH_HIGHLIGHTS.find(h=>t>=h.start_s&&t<h.end_s+h.focus_hold_s);
}
function finishHighlightFocus(t){
  return !!finishHighlightCamera(t);
}
function syncFinishHighlight(t){
  if(t<finishHighlightLastTime-1e-9)finishHighlightOverrides.clear();
  finishHighlightLastTime=t;
  const actual=finishHighlightActive(t)?finishHighlightAt(t).speed:speed;
  document.querySelectorAll('.seg button').forEach(button=>{
    button.setAttribute('aria-pressed',String(Number(button.dataset.s)===actual));
    // Adding the class once per automatic window starts one gentle pulse.
    // It remains selected after the pulse and is cleared on exit or override.
    button.classList.toggle('automatic-slow-motion',finishHighlightActive(t)&&Number(button.dataset.s)===actual);
  });
}
function advanceFinishHighlight(t,wallSeconds,baseSpeed){
  if(t<finishHighlightLastTime-1e-9)finishHighlightOverrides.clear();
  let remaining=Math.max(0,wallSeconds),cursor=t;
  while(remaining>0){
    const highlight=FINISH_HIGHLIGHTS.find(h=>cursor<h.end_s&&!finishHighlightOverrides.has(h.policy_index));
    if(!highlight)return cursor+remaining*baseSpeed;
    const before=cursor<highlight.start_s;
    const boundary=before?highlight.start_s:highlight.end_s;
    const rate=before?baseSpeed:highlight.speed;
    const wallToBoundary=(boundary-cursor)/rate;
    if(remaining<wallToBoundary)return cursor+remaining*rate;
    cursor=boundary;remaining-=wallToBoundary;
  }
  return cursor;
}
function finishHighlightState(){
  const highlight=finishHighlightCamera(playT)||FINISH_HIGHLIGHTS.find(h=>playT<h.end_s)||FINISH_HIGHLIGHTS.at(-1);
  return {time:playT,playing,policy:highlight.policy_index,
    start:highlight.start_s,end:highlight.end_s,highlights:FINISH_HIGHLIGHTS,
    active:finishHighlightActive(playT),overridden:finishHighlightOverrides.has(highlight.policy_index),
    baseSpeed:speed,effectiveSpeed:finishHighlightActive(playT)?highlight.speed:speed};
}
document.querySelectorAll('.seg button').forEach(button=>button.addEventListener('click',()=>{
  const highlight=finishHighlightAt(playT);
  if(highlight)finishHighlightOverrides.add(highlight.policy_index);
  // The shared player's listener applies the newly selected base speed first.
  queueMicrotask(()=>syncFinishHighlight(playT));
}));
cameraResetBtn?.addEventListener('click',()=>{
  finishHighlightOverrides.clear();syncFinishHighlight(playT);
});
"""


def add_finish_highlight(html: str, data: dict) -> str:
    """Install the comparison-only clock hook without changing shared players."""
    if not data.get("meta", {}).get("finish_highlights"):
        return html
    replacements = (
        ("const btn=document.getElementById('replay');", FINISH_HIGHLIGHT_SCRIPT + "\nconst btn=document.getElementById('replay');"),
        ("playT += (now-lastNow)/1000*speed;", "playT = advanceFinishHighlight(playT,(now-lastNow)/1000,speed);"),
        ("function draw(t){", "function draw(t){\n  syncFinishHighlight(t);"),
        ("const next=AUTO_FOLLOW_ORDER.find(i=>!visualDones[i]);", "const next=finishHighlightCamera(t)?.policy_index??AUTO_FOLLOW_ORDER.find(i=>!visualDones[i]);"),
        ("window.__G1_REPLAY__={", "window.__G1_REPLAY__={\n  finishHighlight:finishHighlightState,"),
    )
    for anchor, replacement in replacements:
        count = html.count(anchor)
        if count != 1:
            raise RuntimeError(f"finish-highlight renderer anchor occurs {count} times: {anchor}")
        html = html.replace(anchor, replacement, 1)
    return html


CAMERA_ORBIT_SCRIPT = r"""
// Uniform, simulation-time camera tours. Pausing/seeking is deterministic;
// manual camera controls opt out until Reset restores automatic following.
const COMPARISON_ORBITS=DATA.meta.camera_orbits;
function comparisonOrbitOffset(t,policy){
  const orbit=COMPARISON_ORBITS.find(o=>o.policy_index===policy&&t>=o.start_s&&t<o.start_s+o.rotate_s+o.hold_s+o.return_s);
  if(!orbit)return 0;
  const elapsed=t-orbit.start_s;
  const fraction=elapsed<orbit.rotate_s?elapsed/orbit.rotate_s:
    elapsed<orbit.rotate_s+orbit.hold_s?1:
    1-(elapsed-orbit.rotate_s-orbit.hold_s)/orbit.return_s;
  return (orbit.side_azimuth_degrees*Math.PI/180-FOLLOW_VIEW.az)*fraction;
}
function comparisonAutomaticView(t,policy){
  return {...FOLLOW_VIEW,az:FOLLOW_VIEW.az+comparisonOrbitOffset(t,policy)};
}
"""


def add_camera_orbits(html: str, data: dict) -> str:
    if not data.get("meta", {}).get("camera_orbits"):
        return html
    replacements = (
        ("const btn=document.getElementById('replay');", CAMERA_ORBIT_SCRIPT + "\nconst btn=document.getElementById('replay');"),
        ("const activeView=automaticRunnerFollow?FOLLOW_VIEW:VIEW;",
         "const activeView=automaticRunnerFollow?comparisonAutomaticView(t,runnerPolicyIndex):VIEW;"),
        ("Object.assign(VIEW,{az:FOLLOW_VIEW.az,el:FOLLOW_VIEW.el,dist:FOLLOW_VIEW.dist});\n  comparisonAutoCamera=false;",
         "Object.assign(VIEW,comparisonAutomaticView(playT,currentAutoFollowPolicy));\n  comparisonAutoCamera=false;"),
        ("view:{...(IS_COMPARISON&&!IS_TRAJECTORY_COMPARISON&&userFollowPolicy===null?FOLLOW_VIEW:VIEW)},",
         "view:{...(comparisonAutoCamera&&validationPolicy===null&&userFollowPolicy===null?comparisonAutomaticView(playT,currentAutoFollowPolicy):VIEW)},"),
    )
    for anchor, replacement in replacements:
        count = html.count(anchor)
        if count != 1:
            raise RuntimeError(f"camera-orbit renderer anchor occurs {count} times: {anchor}")
        html = html.replace(anchor, replacement, 1)
    return html


def add_mobile_closeup(html: str) -> str:
    """Give mobile comparison replays a close follow view, not a wide shot."""
    replacements = (
        ("const compactFollowScale=IS_SCORING_EXAMPLE?1:(runnerFollowComposition&&followViewportWidth()<560\n    ?1.9:1);",
         "// Mobile comparison stages have controls and results below the canvas.\n"
         "  // Use that unobstructed frame for a 2x closer shot than the old 1.9 scale.\n"
         "  const compactFollowScale=runnerFollowComposition&&comparisonMobileLayout()?0.95:1;"),
        ("responsiveScale:IS_SCORING_EXAMPLE?1:(followViewportWidth()<560?1.9:1),",
         "responsiveScale:comparisonMobileLayout()?0.95:1,"),
    )
    for anchor, replacement in replacements:
        count = html.count(anchor)
        if count != 1:
            raise RuntimeError(f"mobile-closeup renderer anchor occurs {count} times: {anchor}")
        html = html.replace(anchor, replacement, 1)
    return html


def add_presentation_result_placement(html: str) -> str:
    """Keep an official failure label at the verified visual finish, if any."""
    anchor = "p.failed?failureFrame(p)[1]-p.startX+1.6:101.6"
    if html.count(anchor) != 1:
        raise RuntimeError(f"presentation-result renderer anchor occurs {html.count(anchor)} times: {anchor}")
    return html.replace(
        anchor,
        "p.failed?(Number.isFinite(p.presentation_result_distance_m)?p.presentation_result_distance_m:failureFrame(p)[1]-p.startX)+1.6:101.6",
        1,
    )


SLOW_MOTION_CSS = """
.seg:has(button.automatic-slow-motion){overflow:visible}
.seg button.automatic-slow-motion{position:relative;z-index:1;border-radius:8px;color:#101806;background:#baff3b;box-shadow:inset 0 0 0 2px #efffc9;animation:slow-motion-entry 1s ease-out 1}
@keyframes slow-motion-entry{0%,100%{box-shadow:inset 0 0 0 2px #efffc9,0 0 0 0 rgba(186,255,59,0)}35%{box-shadow:inset 0 0 0 2px #fff,0 0 0 6px rgba(186,255,59,.6)}65%{box-shadow:inset 0 0 0 2px #fff,0 0 18px 3px rgba(186,255,59,.45)}}
@media(prefers-reduced-motion:reduce){.seg button.automatic-slow-motion{animation:none;outline:2px solid #efffc9;outline-offset:2px}}
"""


def _best_point(performance: dict, needle: str) -> dict:
    model = next(
        row
        for row in performance.get("models", [])
        if needle in str(row.get("model") or "").lower()
    )
    candidates = [point for point in model.get("points", []) if point.get("replay_url")]
    if not candidates:
        raise RuntimeError(f"no replayable policy found for {needle}")
    return max(candidates, key=lambda point: float(point.get("continuous_score_mps") or 0))


def _load_comparison_capture(captures: Path, replay_name: str, point: dict) -> dict:
    """Prefer a verified presentation-only continuation when one is present.

    The ordinary capture remains the source of scoring metadata.  An extended
    capture is accepted only when it identifies the same immutable policy,
    carries the exact official run row, and reproduces every official pose
    sample byte-for-byte before adding later simulator samples.
    """
    official_path = captures / f"{replay_name}.json"
    if not official_path.is_file():
        raise RuntimeError(f"missing comparison capture: {official_path}")
    official_bytes = official_path.read_bytes()
    official = json.loads(official_bytes)

    presentation_path = captures / f"{replay_name}.presentation.json"
    if not presentation_path.is_file():
        return official
    presentation = json.loads(presentation_path.read_text())
    provenance = presentation.get("presentation_extension") or {}
    policy_sha = str(point.get("policy_sha256") or official.get("policy_sha256") or "")
    if provenance.get("authoritative") is not True:
        raise RuntimeError(f"unverified presentation capture: {presentation_path}")
    if provenance.get("scoring_unchanged") is not True or provenance.get(
        "official_capture_sha256"
    ) != hashlib.sha256(official_bytes).hexdigest():
        raise RuntimeError(
            "presentation provenance does not match official capture: "
            f"{presentation_path}"
        )
    if not policy_sha or any(
        str(capture.get("policy_sha256") or "") != policy_sha
        for capture in (official, presentation)
    ):
        raise RuntimeError(f"presentation policy does not match published point: {presentation_path}")
    if presentation.get("runs") != official.get("runs"):
        raise RuntimeError(f"presentation capture changed official scoring data: {presentation_path}")
    official_frames = (official.get("frames") or [[]])[0]
    presentation_frames = (presentation.get("frames") or [[]])[0]
    if presentation_frames[: len(official_frames)] != official_frames:
        raise RuntimeError(f"presentation capture does not preserve the official prefix: {presentation_path}")
    return presentation


def build_comparison(performance_path: Path, captures: Path, hq_path: Path) -> dict:
    performance = json.loads(performance_path.read_text())
    source_captures: list[dict] = []
    points: list[dict] = []
    for needle, _, _ in COMPETITORS:
        point = _best_point(performance, needle)
        replay_name = Path(urlsplit(point["replay_url"]).path).stem
        source_captures.append(_load_comparison_capture(captures, replay_name, point))
        points.append(point)

    names = source_captures[0]["body_names"]
    fps = float(source_captures[0].get("fps") or 50)
    for capture in source_captures[1:]:
        if capture["body_names"] != names or float(capture.get("fps") or 50) != fps:
            raise RuntimeError("comparison captures use incompatible body layouts or frame rates")

    merged = {
        "schema_version": 2,
        "body_names": names,
        "fps": fps,
        "frames": [capture["frames"][0] for capture in source_captures],
        "runs": [(capture.get("runs") or [{}])[0] for capture in source_captures],
        "failure_modes": [],
    }
    hq = json.loads(hq_path.read_text())["meshes"]
    data = capture_to_data(merged, hq, "best published policy by model", pad=1.5)
    data["colors"] = [int(color.removeprefix("#"), 16) for _, _, color in COMPETITORS]
    data["lane_indices"] = list(range(len(COMPETITORS)))
    data["meta"].update(
        {
            "comparison": True,
            "track_lanes": len(COMPETITORS),
            "source_performance": str(performance_path),
            "presentation_extensions": [
                capture.get("presentation_extension")
                for capture in source_captures
                if capture.get("presentation_extension")
            ],
        }
    )
    highlights = []
    orbits = []
    for lane, (policy, point, (competitor_needle, label, color)) in enumerate(
        zip(data["policies"], points, COMPETITORS, strict=True), start=1
    ):
        policy.update(
            {
                "label": label,
                "identity": model_identity(competitor_needle),
                "lane_number": lane,
                "color": color,
                "effective_speed_mps": point.get("continuous_score_mps"),
                "source_run_id": point.get("source_run_id"),
                "submission_index": point.get("submission_index"),
            }
        )
        presentation = source_captures[lane - 1].get("presentation_extension") or {}
        visual_terminal = presentation.get("post_timeout_physical_terminal") or {}
        visual_distance = visual_terminal.get("distance_m")
        if (
            presentation.get("authoritative") is True
            and presentation.get("scoring_unchanged") is True
            and visual_terminal.get("reason") == "finished"
            and isinstance(visual_distance, (int, float))
            and math.isfinite(visual_distance)
        ):
            # This moves only the track label. The official timeout, score,
            # event time, legal distance and HUD are left untouched.
            policy["presentation_result_distance_m"] = visual_distance
        finish = policy.get("finish")
        highlight_seconds = {"glm-5.3-flash": 0.5, "gpt-5.6-luna": 0.2}.get(competitor_needle)
        has_valid_finish = (
            policy.get("valid") is True
            and isinstance(finish, (int, float))
            and math.isfinite(finish)
        )
        if highlight_seconds is not None and has_valid_finish and finish >= highlight_seconds:
            highlights.append({
                "policy_index": lane - 1,
                "start_s": round(finish - highlight_seconds, 3),
                "end_s": finish,
                "speed": 0.1,
                "focus_hold_s": 0.5,
            })
        orbit_start = None
        if competitor_needle == "deepseek-v4-flash-vision-exp":
            orbit_start = 30.0
            # Highlight the middle 0.5 seconds of the five-second side view.
            highlights.append({
                "policy_index": lane - 1,
                "start_s": orbit_start + 7.25, "end_s": orbit_start + 7.75,
                "speed": 0.1, "focus_hold_s": 0.0,
            })
        elif competitor_needle == "gpt-5.6-luna" and has_valid_finish and finish >= 15:
            orbit_start = round(finish - 15.0, 3)
        if orbit_start is not None:
            orbits.append({
                "policy_index": lane - 1, "start_s": orbit_start,
                "rotate_s": 5.0, "hold_s": 5.0, "return_s": 5.0,
                "side_azimuth_degrees": -90,
            })
    data["meta"]["finish_highlights"] = sorted(highlights, key=lambda highlight: highlight["start_s"])
    data["meta"]["camera_orbits"] = sorted(orbits, key=lambda orbit: orbit["start_s"])
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--captures", type=Path, default=ROOT / "web/captures")
    parser.add_argument("--hq", type=Path, default=HQ_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shared-assets", type=Path, help="Website shared asset directory; omit for standalone HTML")
    args = parser.parse_args()
    data = build_comparison(args.performance, args.captures, args.hq)
    html = assemble_html(
        data,
        title="Agents' 100m · Best-policy race",
        eyebrow="BEST-POLICY RACE",
        headline="The fastest published policy from each model",
        lede="Three policies start together on the same 100-metre clock.",
        cap="Best replayable policy per model from the current published cohort.",
        story="The replay uses the verifier-authored pose capture for every runner.",
        sr_only="Best published policies racing together in lanes one through three.",
        active="comparison",
        show_policy_labels=True,
    )
    html = add_finish_highlight(html, data)
    html = add_camera_orbits(html, data)
    html = add_mobile_closeup(html)
    html = add_presentation_result_placement(html)
    comparison_css = (
        ".lanes{top:10px;right:12px;gap:4px}.lc{min-width:330px;padding:4px 9px}"
        ".lc .nm{max-width:138px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ".lc .tm{font-size:11px}.lc .d{min-width:86px;font-size:11px}"
        "@media(max-width:720px){.lanes{right:8px}.lc{min-width:250px}.lc .nm{max-width:100px}}"
    )
    html = html.replace("</style>", comparison_css + SLOW_MOTION_CSS + "</style>", 1)
    if args.shared_assets is not None:
        html = shared_asset_html(html, args.shared_assets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "lanes": [
                    {
                        "lane": policy["lane_number"],
                        "model": policy["label"],
                        "effective_speed_mps": policy["effective_speed_mps"],
                        "source_run_id": policy["source_run_id"],
                        "submission_index": policy["submission_index"],
                    }
                    for policy in data["policies"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
