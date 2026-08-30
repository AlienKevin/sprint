"""Switch selected trajectory runners without a camera teleport."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCENE = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()


def test_camera_switch_uses_wall_time_and_current_pose_without_advancing_replay() -> None:
    start = SCENE.index("const CAMERA_SWITCH_MS=")
    helper = SCENE[start:SCENE.index("const cameraModeEl=", start)]
    script = "const assert=require('node:assert/strict');\n" + """
class V {
  constructor(x=0,y=0,z=0){this.set(x,y,z);}
  set(x,y,z){this.x=x;this.y=y;this.z=z;return this;}
  clone(){return new V(this.x,this.y,this.z);}
  copy(v){return this.set(v.x,v.y,v.z);}
  lerpVectors(a,b,t){return this.set(a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,a.z+(b.z-a.z)*t);}
  array(){return [this.x,this.y,this.z];}
}
let now=0,reduced=false,IS_TRAJECTORY_COMPARISON=true,playing=false,playT=17.3;
let nextId=0,draws=0;const pending=new Map();
const performance={now:()=>now},matchMedia=()=>({matches:reduced});
const requestAnimationFrame=fn=>{pending.set(++nextId,fn);return nextId;};
const cancelAnimationFrame=id=>pending.delete(id);
const camera={position:new V(2,0,1)},camP=new V(),camT=new V(0,0,.6);
let desiredPosition=new V(2,10,1),desiredTarget=new V(0,10,.6);
function draw(t){assert.equal(t,17.3);draws++;camP.copy(desiredPosition);camT.copy(desiredTarget);applyCameraSwitch();camera.position.copy(camP);}
function frame(t){now=t;const callbacks=[...pending.values()];pending.clear();callbacks.forEach(fn=>fn(t));}
""" + helper + """
assert.equal(cameraSwitch,null,'initial load never animates');
startCameraSwitch();draw(playT);
assert.deepEqual(camera.position.array(),[2,0,1],'first frame keeps displayed camera');
frame(250);assert.equal(camera.position.y,5);assert.equal(camT.y,5);
frame(500);assert.equal(camera.position.y,10);assert.equal(camT.y,10);
assert.equal(cameraSwitch,null);assert.equal(pending.size,0);assert.equal(playT,17.3);
assert(draws>=3,'paused camera redraws until transition completes');

// Re-selection midway starts at the visible midpoint, not the former target.
now=1000;desiredPosition.y=20;desiredTarget.y=20;startCameraSwitch();frame(1250);
assert.equal(camera.position.y,15);
desiredPosition.y=-10;desiredTarget.y=-10;startCameraSwitch();draw(playT);
assert.equal(camera.position.y,15);assert.equal(camT.y,15);
frame(1500);assert.equal(camera.position.y,2.5);
frame(1750);assert.equal(camera.position.y,-10);assert.equal(pending.size,0);

// A running replay draws in its own loop; camera RAF must not draw twice.
playing=true;now=2000;desiredPosition.y=0;desiredTarget.y=0;startCameraSwitch();
const before=draws;frame(2250);assert.equal(draws,before);draw(playT);
assert.equal(camera.position.y,-5);now=2500;draw(playT);
assert.equal(camera.position.y,0);assert.equal(cameraSwitch,null);
assert.equal(playT,17.3,'playback speed/time are not read or changed by camera travel');

reduced=true;startCameraSwitch();assert.equal(cameraSwitch,null);assert.equal(pending.size,0);
reduced=false;IS_TRAJECTORY_COMPARISON=false;startCameraSwitch();assert.equal(cameraSwitch,null);
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_only_explicit_policy_changes_start_camera_transition() -> None:
    follow = SCENE.split("function followPolicy(index){", 1)[1].split("\n}\n", 1)[0]
    assert "if(next!==userFollowPolicy)startCameraSwitch();" in follow
    assert follow.index("startCameraSwitch();") < follow.index("userFollowPolicy=next")
    assert "emphasizePolicy(next);" in follow
    assert "if(!playing)draw(playT)" in follow
    assert "applyCameraSwitch();\n  camera.position.copy(camP);" in SCENE
    assert "transitioning:cameraSwitch!==null" in SCENE
