"""Observations opt out of automatic playback without disabling user controls."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCENE = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()


def test_autoplay_opt_out_never_schedules_play_but_manual_play_and_replay_work() -> None:
    option = next(line for line in SCENE.splitlines() if line.startswith("const REPLAY_AUTOPLAY="))
    initializer = SCENE.splitlines()[-1]
    play = SCENE[SCENE.index("function play(){"):SCENE.index("function pause(){")]
    script = "const assert=require('node:assert/strict');\n"
    script += "function player(search,startPaused=false,reduced=false){const location={search},DATA={meta:{start_paused:startPaused}};\n"
    script += option + "\n" + play + """
const timers=[],frames=[],matchMedia=()=>({matches:reduced});
const setTimeout=(fn,ms)=>timers.push({fn,ms}),requestAnimationFrame=fn=>{frames.push(fn);return frames.length;},cancelAnimationFrame=()=>{};
let playT=0,playing=false,lastNow=null,startWall=null,raf=null,T_END=8;
const loop=()=>{},setBtn=()=>{},updateCameraControls=()=>{},localizeReplayScene=()=>{};
""" + initializer + """
return {timers,play,state:()=>({playT,playing}),end:()=>{playT=T_END;playing=false;}};}
for(let visit=0;visit<2;visit++){
  const paused=player('?example=1&autoplay=0');
  assert.equal(paused.timers.length,0,'no delayed autoplay on load or revisit');
  assert.deepEqual(paused.state(),{playT:0,playing:false});
  paused.play();assert.deepEqual(paused.state(),{playT:0,playing:true});
  paused.end();paused.play();assert.deepEqual(paused.state(),{playT:0,playing:true},'Replay restarts at zero');
}
for(const query of ['', '?example=1', '?example=1&view=side', '?autoplay=1']){
  const automatic=player(query);assert.equal(automatic.timers.length,1);assert.equal(automatic.timers[0].ms,500);
  automatic.timers[0].fn();assert.equal(automatic.state().playing,true);
}
assert.equal(player('',true).timers.length,0,'existing start-paused metadata remains respected');
assert.equal(player('',false,true).timers.length,0,'reduced motion remains respected');
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_opt_out_is_present_in_all_three_generated_observation_replays() -> None:
    for capture in ['76d7d31f8c51', 'dd28af63bf84', '5eb6fee88ef0']:
        html = (ROOT / f'web/replay/frontier-{capture}.html').read_text()
        assert "const REPLAY_AUTOPLAY=" in html
        assert "if(REPLAY_AUTOPLAY&&!DATA.meta?.start_paused" in html
