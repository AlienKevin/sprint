"""A side-view example opts in without changing normal replay cameras."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCENE = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()


def test_side_view_requires_example_and_preserves_zoom_orbit_and_reset() -> None:
    modes = "\n".join(line for line in SCENE.splitlines() if line.startswith(("const IS_SCORING_EXAMPLE=", "const IS_SIDE_EXAMPLE=")))
    start = SCENE.index("const FOLLOW_VIEW=")
    views = SCENE[start:SCENE.index("let validationPolicy=", start)]
    helpers = []
    for begin, end in [
        ("function beginManualCamera(){", "function policyCameraLabel("),
        ("function resetCamera(){", "(function(){"),
    ]:
        helpers.append(SCENE[SCENE.index(begin):SCENE.index(end, SCENE.index(begin))])
    script = "const assert=require('node:assert/strict');\n"
    script += "function check(search,comparison=false){const location={search},IS_COMPARISON=comparison,IS_TRAJECTORY_COMPARISON=false;\n"
    script += modes + "\n" + views + "\n" + "\n".join(helpers)
    script += """
let comparisonAutoCamera=false,userFollowPolicy=null,DEFAULT_FOLLOW_POLICY=null,playing=true,playT=0;
const resetFollowDamping=()=>{},updateCameraControls=()=>{},cancelCameraSwitch=()=>{};
return {side:IS_SIDE_EXAMPLE,initial:{...VIEW},orbit:orbitCamera,zoom:zoomCamera,reset:resetCamera};}
const side=check('?example=1&view=side');
assert.equal(side.side,true);assert.equal(side.initial.az,-Math.PI/2);
assert(Math.abs(Math.cos(side.initial.az))<1e-12,'view direction is perpendicular to down-track X');
assert.equal(side.orbit(.2).az,-Math.PI/2+.2,'manual orbit stays available');
assert.equal(side.zoom(1.22),2.541*1.22,'manual zoom stays available');
assert.equal(side.reset().view.az,-Math.PI/2,'Reset returns to the requested side view');
assert.equal(side.reset().view.dist,2.541);
assert.equal(check('?example=1').initial.az,-.25);
assert.equal(check('?view=side').initial.az,-.25,'ordinary replay ignores side option');
assert.equal(check('?example=0&view=side').initial.az,-.25);
assert.equal(check('?example=1&view=side',true).initial.az,-.82,'comparison camera is unchanged');
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_side_example_option_reaches_shared_shell_and_selected_capture() -> None:
    for relative in ["web/replay-template.html", "web/replay/frontier-3c89dd3bbca5.html"]:
        html = (ROOT / relative).read_text()
        assert "const IS_SIDE_EXAMPLE=IS_SCORING_EXAMPLE&&" in html
        assert "az:IS_SIDE_EXAMPLE?-Math.PI/2:FOLLOW_VIEW.az" in html
