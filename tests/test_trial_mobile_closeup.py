from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "web/renderers/g1-100-metres"


def trial_scene() -> str:
    sys.path.insert(0, str(RENDERER))
    try:
        spec = importlib.util.spec_from_file_location(
            "trial_mobile_closeup", RENDERER / "render_trial_comparison.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        shell = module.build_shell(registry={}, hq={})
    finally:
        sys.path.remove(str(RENDERER))
    return json.loads(shell.split("  scene:", 1)[1].split("\n};", 1)[0])


def test_trial_shell_reuses_mobile_closeup_but_not_homepage_choreography():
    scene = trial_scene()
    assert "runnerFollowComposition&&comparisonMobileLayout()?0.95:1" in scene
    assert "responsiveScale:comparisonMobileLayout()?0.95:1" in scene
    assert "finishHighlight" not in scene
    assert "comparisonAutomaticView" not in scene
    assert "presentation_result_distance_m" not in scene
    assert "userFollowPolicy=DEFAULT_FOLLOW_POLICY" in scene
    assert "VIEW.dist*factor" in scene
    assert "p.failed?failureFrame(p)[1]-p.startX+1.6:101.6" in scene


def test_mobile_closeup_follows_parent_breakpoint_without_affecting_desktop():
    scene = trial_scene()
    helpers = scene.split("function replayMobileLayout(){", 1)[1].split("\nfunction resize", 1)[0]
    script = """
const assert=require('node:assert/strict');
let IS_COMPARISON=true;
const window={innerWidth:400,parent:{innerWidth:491}};
function replayMobileLayout(){HELPERS
for(const width of [320,491,559,560,599,720]){
  window.parent.innerWidth=width;
  assert.equal(comparisonMobileLayout(),true);
  assert.equal(comparisonMobileLayout()?.95:1,.95);
}
window.parent.innerWidth=953;
assert.equal(comparisonMobileLayout(),false);
assert.equal(comparisonMobileLayout()?.95:1,1);
window.parent=window;window.innerWidth=491;
assert.equal(comparisonMobileLayout(),true);
assert.equal(1.9/.95,2);
IS_COMPARISON=false;
assert.equal(replayMobileLayout(),true);
assert.equal(comparisonMobileLayout(),false); // Single-player chrome, not comparison camera.
""".replace("HELPERS", helpers)
    result = subprocess.run(["node", "-"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
