"""Pointer-gesture regressions for the shared replay camera controls."""

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "web/renderers/g1-100-metres/scene.js"


def test_replay_pinch_and_pointer_lifecycle():
    scene = SCENE.read_text()
    orbit = scene.split("function orbitCamera(", 1)[1].split("function policyCameraLabel", 1)[0]
    controls = scene.split("function zoomCamera(", 1)[1].split("function draw(t){", 1)[0]
    buttons = "\n".join(
        line for line in scene.splitlines()
        if "?.addEventListener('click',()=>zoomCamera(" in line
    )
    script = r"""
const assert=require('node:assert/strict'),vm=require('node:vm');
class Target {
  constructor(){this.listeners=new Map();this.style={};this.captured=new Set();this.releases=[];}
  addEventListener(type,fn,options){if(!this.listeners.has(type))this.listeners.set(type,[]);this.listeners.get(type).push({fn,options});}
  removeEventListener(type,fn){this.listeners.set(type,(this.listeners.get(type)||[]).filter(item=>item.fn!==fn));}
  emit(type,event={}){for(const {fn} of [...(this.listeners.get(type)||[])])fn(event);}
  setPointerCapture(id){this.captured.add(id);}
  hasPointerCapture(id){return this.captured.has(id);}
  releasePointerCapture(id){this.captured.delete(id);this.releases.push(id);this.emit('lostpointercapture',{pointerId:id});}
  getBoundingClientRect(){return {left:0,top:0,width:400,height:300};}
}
const el=new Target(),win=new Target(),plus=new Target(),minus=new Target();
let picks=[],draws=0,manual=0;
const context={renderer:{domElement:el},window:win,VIEW:{az:0,el:.5,dist:8},
  beginManualCamera(){manual++;},playing:false,playT:1,draw(){draws++;},
  IS_COMPARISON:true,camera:{},ROBOTS:[{grp:{}}],followPolicy(index){picks.push(index);},
  THREE:{Vector2:class{set(){}},Raycaster:class{
    setFromCamera(){} intersectObjects(){return [{object:{userData:{policyIndex:0}}}];}
  }},document:{getElementById(id){return id==='camera-zoom-in'?plus:minus;}}};
vm.createContext(context);
vm.runInContext(SOURCE,context);
const view=context.VIEW;
const point=(type,id,x=0,y=0)=>el.emit(type,{pointerId:id,clientX:x,clientY:y});
const down=(id,x,y)=>point('pointerdown',id,x,y),move=(id,x,y)=>point('pointermove',id,x,y),up=(id,x,y)=>point('pointerup',id,x,y);
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const reset=()=>{win.emit('blur');view.az=0;view.el=.5;view.dist=8;picks=[];};
assert.equal(el.style.touchAction,'none');

// Spreading zooms in; closing zooms out, without orbiting.
down(1,0,0);down(2,100,0);move(2,200,0);near(view.dist,4);
move(2,50,0);near(view.dist,16);near(view.az,0);near(view.el,.5);
up(2,50,0);up(1,0,0);assert.deepEqual(picks,[]);
assert.equal(el.captured.size,0);assert.equal(el.style.cursor,'grab');

// Camera distance clamps are shared with the existing buttons/wheel.
reset();down(1,0,0);down(2,100,0);move(2,10000,0);near(view.dist,1.6);
move(2,2,0);near(view.dist,48);win.emit('blur');
assert.equal(el.captured.size,0);assert.deepEqual(picks,[]);

// A third finger cannot orbit or zoom, and changing the tracked pair rebases.
reset();down(1,0,0);down(2,100,0);move(2,200,0);near(view.dist,4);
down(3,900,0);move(3,1200,0);near(view.dist,4);
up(1,0,0);move(3,1200,0);near(view.dist,4);
move(3,2200,0);near(view.dist,2);near(view.az,0);
up(3,2200,0);move(2,200,0);near(view.az,0);
move(2,210,5);near(view.az,-.05);near(view.el,.52);near(view.dist,2);
up(2,210,5);assert.deepEqual(picks,[]);

// Existing one-pointer drag threshold, direction and normal robot tap survive.
reset();down(1,10,10);move(1,13,12);near(view.az,0);
move(1,20,20);near(view.az,-.05);near(view.el,.54);
move(1,22,21);near(view.az,-.06);near(view.el,.544);
up(1,22,21);assert.deepEqual(picks,[]);
down(1,30,30);up(1,30,30);assert.deepEqual(picks,[0]);

// Cancellation/lost capture never picks a robot, including two-to-one moves.
for(const cancellation of ['pointercancel','lostpointercapture']){
  reset();down(1,0,0);point(cancellation,1);assert.deepEqual(picks,[]);
  move(1,200,0);near(view.az,0);assert.equal(el.captured.size,0);
  down(1,0,0);down(2,100,0);move(2,200,0);near(view.dist,4);
  point(cancellation,2);move(1,0,0);near(view.az,0);near(view.dist,4);
  up(1,0,0);assert.deepEqual(picks,[]);
  down(1,50,50);up(1,50,50);assert.deepEqual(picks,[0]);
}

// Coincident contacts establish a baseline without dividing by zero.
reset();down(1,0,0);down(2,0,0);move(2,100,0);near(view.dist,8);
move(2,200,0);near(view.dist,4);win.emit('blur');assert.deepEqual(picks,[]);
move(2,500,0);near(view.dist,4);

// Desktop wheel and +/- buttons keep their current factors and prevention.
reset();let prevented=false;
el.emit('wheel',{deltaY:100,preventDefault(){prevented=true;}});
near(view.dist,8*Math.exp(.1));assert.equal(prevented,true);
assert.equal(el.listeners.get('wheel')[0].options.passive,false);
view.dist=8;plus.emit('click');near(view.dist,8/1.22);minus.emit('click');near(view.dist,8);

// Playing uses its existing render loop; gesture controls don't change time/state.
reset();context.playing=true;let previousDraws=draws;
down(1,0,0);down(2,100,0);move(2,200,0);
near(view.dist,4);assert.equal(draws,previousDraws);assert.equal(context.playT,1);
assert.equal(context.playing,true);context.playing=false;
win.emit('blur');

// Teardown releases all captures and stops any further touch gestures.
down(1,0,0);down(2,100,0);vm.runInContext('disposeCameraPointers()',context);
assert.equal(el.captured.size,0);assert.equal(win.listeners.get('blur').length,0);
const finalDistance=view.dist;down(1,0,0);down(2,100,0);move(2,200,0);
near(view.dist,finalDistance);assert.deepEqual(picks,[]);assert.ok(manual>0);
console.log('pinch and pointer lifecycle passed');
"""
    script = script.replace("SOURCE", json.dumps("function orbitCamera(" + orbit + "function zoomCamera(" + controls + buttons))
    result = subprocess.run(["node", "-e", script], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pinch and pointer lifecycle passed" in result.stdout


def test_replay_disposal_clears_pointer_capture():
    scene = SCENE.read_text()
    disposal = scene.split("function disposeReplay(){", 1)[1].split("window.__G1_REPLAY__", 1)[0]
    assert "disposeCameraPointers();" in disposal
