"""Trailing stalls shorten playback, never scores or authoritative captures."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCENE = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()


def run_node(body: str) -> None:
    start = SCENE.index("const STALL_MIN_SPEED_MPS=")
    end = SCENE.index("\nPOL.forEach", start)
    script = "const assert=require('node:assert/strict');const PELVIS_I=0,LO=i=>1+i*7;\n"
    script += SCENE[start:end] + "\n" + body
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_stalls_ignore_vertical_limb_and_in_place_motion() -> None:
    run_node("""
const policy=(x)=>({failed:true,freezeT:60,frames:Array.from({length:601},(_,i)=>{
  const t=i/10;return [t,x(t),Math.sin(t)*2,Math.sin(t)*3,0,0,0,1,Math.sin(t)*9];
})});
assert.equal(policyPlaybackEnd(policy(()=>0)),1);
assert.equal(policyPlaybackEnd(policy(t=>.009*Math.sin(t*12))),1);
assert.equal(policyPlaybackEnd(policy(t=>t*.001)),1,'sub-threshold root drift is not locomotion');
const burst=policy(t=>Math.min(t,2));
const cutoff=policyPlaybackEnd(burst);
assert.equal(cutoff,3,'exactly one second after the final advancing frame');
assert.equal(burst.freezeT,60,'scoring/recorded bounds are not rewritten');
assert.equal(burst.frames.at(-1)[0],60,'capture remains complete');
const rocking=policy(t=>t<2?t:1.5+.5*Math.cos((t-2)*3));
assert.equal(policyPlaybackEnd(rocking),3,'oscillation below prior progress is not advancement');
const lateJitter=policy(t=>Math.min(t,2)+(t>25?.009*Math.sin(t*12):0));
assert.equal(policyPlaybackEnd(lateJitter),3,'an isolated tiny later excursion does not restart the clock');
for(const speed of [.001,.01])for(const quantized of [false,true]){
  const creeping={failed:true,freezeT:60,frames:Array.from({length:3001},(_,i)=>{
    const t=i/50,x=t<=2?t:2+(t-2)*speed;
    return [t,quantized?Math.round(x*1000)/1000:x,0,.7];
  })};
  assert.equal(policyPlaybackEnd(creeping),3,`early launch must not mask later ${speed}m/s creep (rounded=${quantized})`);
}
""")


def test_delayed_starts_recoveries_slow_crawls_and_finishes_are_kept() -> None:
    run_node("""
const policy=(x)=>({failed:true,freezeT:60,frames:Array.from({length:601},(_,i)=>[i/10,x(i/10),0,.7])});
assert.equal(policyPlaybackEnd(policy(t=>Math.max(0,t-35)*.2)),60);
assert.equal(policyPlaybackEnd(policy(t=>Math.min(t,2)+Math.max(0,t-40)*.2)),60);
assert.equal(policyPlaybackEnd(policy(t=>t*.03)),60,'slow forward crawl survives');
const rounded=speed=>({failed:true,freezeT:60,frames:Array.from({length:3001},(_,i)=>[i/50,Math.round(i/50*speed*1000)/1000,0,.7])});
const crawl=rounded(.03);
assert(crawl.frames.some((frame,i)=>i>0&&frame[1]===crawl.frames[i-1][1]),'fixture contains quantized zero-motion samples');
assert.equal(policyPlaybackEnd(crawl),60,'50Hz millimetre-rounded slow crawl survives repeated samples');
assert.equal(policyPlaybackEnd(rounded(.01)),1,'quantized jumps do not inflate sub-threshold average speed');
assert.equal(policyPlaybackEnd(policy(t=>Math.max(0,t-59)*.03)),60,'late slow advance remains visible');
assert.equal(policyPlaybackEnd(policy(t=>Math.min(t,2)+Math.min(.6,Math.max(0,t-58)*.5))),60,'near-end recovery is found by full-recording lookahead');
assert.equal(policyPlaybackEnd(policy(t=>Math.min(t,2)+(t>=58.5?.1:0))),59.5,'an isolated late forward jump must be found');
assert.equal(policyPlaybackEnd({...policy(()=>0),failed:false,freezeT:27.48}),27.48);
assert.equal(policyPlaybackEnd({...policy(t=>t),freezeT:1.2}),1.2,'short classified clips are not extended');
""")


def test_comparison_ends_after_last_selected_policy_and_replay_restarts() -> None:
    run_node("""
const make=(x,freezeT=60,failed=true)=>({failed,freezeT,frames:Array.from({length:601},(_,i)=>[i/10,x(i/10),0,.7])});
const stalled=make(t=>Math.min(t,2)),moving=make(t=>t*.1),finished=make(t=>t,9.9,false);
const end=selected=>Math.max(...selected.map(policyPlaybackEnd));
assert.equal(end([stalled,moving,finished]),60,'nonfocused moving selection keeps playing');
assert.equal(end([stalled,finished]),9.9,'removal excludes the former moving selection');
assert.equal(end([stalled]),3);
assert.equal(end([stalled,make(t=>Math.min(t,4))]),5,'all selected runners must have stopped');
assert.equal(end([make(()=>0),make(()=>0)]),1,'entirely stationary selections stop at one second');
""")
    assert "const T_END=Math.max(...POL.map(p=>p.playbackEndT));" in SCENE
    assert "const T_CAPTURE_END=Math.max(...POL.map(p=>p.freezeT));" in SCENE
    assert "STALL_WINDOW_S" not in SCENE
    assert "playT>=T_END ? '&#9654; Replay'" in SCENE
    assert "if(playT>=T_END) playT=0; playing=true;" in SCENE
    assert "playT=T_END; draw(playT); playing=false; raf=null; setBtn(); return;" in SCENE


def test_irregular_frame_times_use_seconds_not_frame_numbers() -> None:
    run_node("""
const p={failed:true,freezeT:60,frames:[[0,0],[.3,.2],[1.4,1],[2,2],[4,2],[8,2],[60,2]]};
assert.equal(policyPlaybackEnd(p),3);
""")


def test_terminal_poses_and_extended_recordings_are_not_changed() -> None:
    run_node("""
const frames=Array.from({length:4001},(_,i)=>[i/50,i/50*.1,0,.7]);
const extended={failed:true,freezeT:77.36,frames:frames.filter(f=>f[0]<=77.36)};
assert.equal(policyPlaybackEnd(extended),77.36,'never silently clip hero footage at 60s');
const collision={failed:true,freezeT:2.45,frames:Array.from({length:61},(_,i)=>[i,i*.1])};
assert.equal(policyPlaybackEnd(collision),2.45,'later captured poses cannot unfreeze an official terminal visual');
const finish={failed:false,freezeT:9.9,frames:[[0,0],[9.9,100]]};
assert.equal(policyPlaybackEnd(finish),9.9,'a finished policy never waits for hypothetical future motion');
assert.equal(collision.freezeT,2.45);assert.equal(extended.frames.at(-1)[0],77.36);
""")
