"""Mobile replay height messages apply only to the sending, same-origin player."""

from pathlib import Path
import subprocess


def test_mobile_height_messages_are_scoped_validated_and_reset_on_desktop():
    app = (Path(__file__).resolve().parents[1] / "web/app.js").read_text()
    listener = app[app.index("  const replayFrames="):app.index("  const MODEL =")]
    script = """
const assert=require('node:assert/strict');
const frame=()=>({contentWindow:{},parentElement:{style:{removeProperty(key){delete this[key==='aspect-ratio'?'aspectRatio':key]}}}});
const hero=frame(),readout=frame();
const nodes={'.model-race-frame iframe':hero,'#readout-replay':readout};
const $=selector=>nodes[selector],listeners={};
const location={origin:'http://localhost:59453'};
const window={innerWidth:600,addEventListener(type,fn){listeners[type]=fn}};
""" + listener + """
const send=(frame,height,extra={})=>listeners.message({origin:location.origin,source:frame.contentWindow,data:{type:'g1:replay-layout',mobile:true,height},...extra});
send(readout,380.2);assert.equal(readout.parentElement.style.height,'381px');
assert.equal(readout.parentElement.style.aspectRatio,'auto');
assert.equal(hero.parentElement.style.height,undefined);
send(hero,420);assert.equal(hero.parentElement.style.height,'420px');
for(const height of [null,'500',NaN,Infinity,0,63,2001])send(readout,height);
send(readout,900,{origin:'https://untrusted.example'});
send(readout,900,{source:{}});
send(readout,900,{data:{type:'other',height:900,mobile:true}});
assert.equal(readout.parentElement.style.height,'381px');
send(readout,400,{data:{type:'g1:replay-layout',mobile:false}});
assert.equal(readout.parentElement.style.height,undefined);
assert.equal(readout.parentElement.style.aspectRatio,undefined);
assert.equal(hero.parentElement.style.height,'420px');
send(readout,390);window.innerWidth=721;listeners.resize();
for(const item of [hero,readout]){
  assert.equal(item.parentElement.style.height,undefined);
  assert.equal(item.parentElement.style.aspectRatio,undefined);
}
send(readout,390);assert.equal(readout.parentElement.style.height,undefined);
"""
    result = subprocess.run(["node", "-e", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
