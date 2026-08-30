#!/usr/bin/env python3
"""Build the static, query-driven multi-policy trajectory replay shell."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from render import G1_PARENT, PREFERRED, model_identity
from render_comparison import add_mobile_closeup


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "web/replay-template.html"
SCENE = Path(__file__).with_name("scene.js")
HQ_DEFAULT = Path(__file__).with_name("g1_hq.json")
POLICY_INDEX_DEFAULT = ROOT / "web/data/policies/index.json"


def load_registry(index_path: Path) -> dict[str, dict]:
    """Return every public policy whose authoritative capture is available."""

    registry: dict[str, dict] = {}
    index = json.loads(index_path.read_text())
    for run in index.get("runs", []):
        policy_path = ROOT / "web" / str(run["path"]).lstrip("/")
        payload = json.loads(policy_path.read_text())
        identity = model_identity(payload.get("model")) or {}
        for policy in payload.get("policies", []):
            replay_url = str(policy.get("replay_url") or "")
            capture_id = Path(replay_url).stem
            # Some captures were published after the policy export's readiness
            # snapshot. Recover only the exact submitted artifact, never an
            # arbitrary capture that happens to share its short URL prefix.
            needs_validation = not policy.get("replay_ready") or not replay_url
            policy_sha = str(policy.get("policy_sha256") or "").lower()
            if needs_validation:
                if not re.fullmatch(r"[a-f0-9]{64}", policy_sha):
                    continue
                capture_id = f"frontier-{policy_sha[:12]}"
            capture_path = ROOT / "web/captures" / f"{capture_id}.json"
            if (
                not capture_id.startswith("frontier-")
                or not capture_path.is_file()
            ):
                continue
            if needs_validation:
                try:
                    capture = json.loads(capture_path.read_text())
                except (OSError, ValueError):
                    continue
                if (
                    capture.get("policy_sha256") != policy_sha
                    or capture.get("schema_version") != 2
                    or not capture.get("body_names")
                    or not any(capture.get("frames") or [])
                ):
                    continue
            registry[capture_id] = {
                "captureId": capture_id,
                "url": f"/captures/{capture_id}.json",
                "policyNumber": int(policy["submission_index"]),
                "label": f"Policy #{int(policy['submission_index'])}",
                "color": identity.get("color") or "#6E97C4",
                "identity": identity,
                "runId": payload.get("run_id"),
                "model": payload.get("model"),
            }
    return registry


def _safe_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def build_shell(*, registry: dict, hq: dict) -> str:
    marker = "<script>const DATA="
    head = TEMPLATE.read_text().split(marker, 1)[0]
    head = head.replace(
        '<div class="lanes">',
        '<div class="lanes" aria-label="Selected policies">',
        1,
    )
    bootstrap = f"""
<script>
const TRIAL_BOOT={{
  registry:{_safe_json(registry)},
  hq:{_safe_json(hq)},
  preferred:{_safe_json(PREFERRED)},
  parents:{_safe_json(G1_PARENT)},
  scene:{_safe_json(add_mobile_closeup(SCENE.read_text()))}
}};
const TRIAL_STORAGE='g1:trial-comparison-state/v1';
// A document owns one parent load generation. Never adopt a newer token from
// a message: an old scene must not acknowledge readiness for a replacement.
const REPLAY_GENERATION=new URLSearchParams(location.search).get('replayGeneration')||'';
let activeState=null,rendererReady=false;
const lanesEl=document.querySelector('.lanes'),noteEl=document.getElementById('failure-note');
lanesEl.innerHTML='';
function tell(type,detail={{}}){{parent.postMessage({{type,...detail,replayGeneration:REPLAY_GENERATION}},location.origin);}}
function show(message,kind='incomplete'){{if(!noteEl)return;noteEl.hidden=false;noteEl.dataset.kind=kind;noteEl.textContent=message;}}
function captureId(value){{const id=String(value||'');return /^frontier-[a-f0-9]{{12}}$/.test(id)?id:null;}}
function colorHex(value){{
  const raw=String(value||'').trim();
  let match=raw.match(/^#([a-f0-9]{{3}}|[a-f0-9]{{6}}|[a-f0-9]{{8}})$/i);
  if(match){{let hex=match[1];if(hex.length===3)hex=[...hex].map(x=>x+x).join('');return `#${{hex.slice(0,6).toUpperCase()}}`;}}
  match=raw.match(/^rgba?[(][ ]*([0-9.]+)[ ]*,[ ]*([0-9.]+)[ ]*,[ ]*([0-9.]+)/i);
  if(!match)return null;
  const channel=value=>Math.max(0,Math.min(255,Math.round(Number(value)||0))).toString(16).padStart(2,'0');
  return `#${{channel(match[1])}}${{channel(match[2])}}${{channel(match[3])}}`.toUpperCase();
}}
function normalize(raw){{
  const seen=new Set(),policies=[];
  for(const item of raw?.policies||[]){{
    const id=captureId(item?.captureId);if(!id||seen.has(id))continue;seen.add(id);
    const fallback=TRIAL_BOOT.registry[id]||{{}};
    const policyNumber=Number(item.policyNumber??fallback.policyNumber);
    if(!Number.isInteger(policyNumber)||policyNumber<1)continue;
    let url=String(item.url||fallback.url||`/captures/${{id}}.json`);
    try{{const parsed=new URL(url,location.href);if(parsed.origin!==location.origin)continue;url=parsed.pathname;}}catch{{continue;}}
    // Registered captures always use the canonical model identity accent. The
    // parent supplies a computed CSS rgb() value, so normalize that too for
    // unregistered/development captures instead of feeding NaN to Three.js.
    const color=colorHex(fallback.color)||colorHex(item.color)||'#6E97C4';
    policies.push({{...fallback,...item,captureId:id,url,policyNumber,label:String(item.label||fallback.label||`Policy #${{policyNumber}}`),color}});
  }}
  if(policies.length>8)throw new Error('At most 8 policies can be compared.');
  const emphasizedCaptureId=captureId(raw?.emphasizedCaptureId);
  return {{policies,emphasizedCaptureId:policies.some(x=>x.captureId===emphasizedCaptureId)?emphasizedCaptureId:(policies.at(-1)?.captureId||null)}};
}}
function queryIds(){{return (new URLSearchParams(location.search).get('policies')||'').split(',').map(captureId).filter(Boolean);}}
function initialState(){{
  const ids=queryIds();let stored=null;try{{stored=JSON.parse(sessionStorage.getItem(TRIAL_STORAGE)||'null');}}catch{{}}
  const storedById=new Map((stored?.policies||[]).map(x=>[x.captureId,x]));
  const missing=ids.filter(id=>!storedById.has(id)&&!TRIAL_BOOT.registry[id]);
  if(missing.length){{const error=new Error(`Unknown policy capture${{missing.length===1?'':'s'}}: ${{missing.join(', ')}}.`);error.failedCaptureIds=missing;throw error;}}
  const policies=ids.map(id=>storedById.get(id)||TRIAL_BOOT.registry[id]);
  const emphasis=new URLSearchParams(location.search).get('emphasis');
  return normalize({{policies,emphasizedCaptureId:emphasis||stored?.emphasizedCaptureId}});
}}
function canonicalUrl(state){{const url=new URL(location.href);url.searchParams.set('policies',state.policies.map(x=>x.captureId).join(','));if(state.emphasizedCaptureId)url.searchParams.set('emphasis',state.emphasizedCaptureId);else url.searchParams.delete('emphasis');if(REPLAY_GENERATION)url.searchParams.set('replayGeneration',REPLAY_GENERATION);else url.searchParams.delete('replayGeneration');return url;}}
window.addEventListener('g1:policy-focused',event=>{{
  const id=captureId(event.detail?.captureId);if(!activeState?.policies.some(x=>x.captureId===id))return;
  activeState={{...activeState,emphasizedCaptureId:id}};
  sessionStorage.setItem(TRIAL_STORAGE,JSON.stringify(activeState));history.replaceState(null,'',canonicalUrl(activeState));
  tell('g1:policy-focused',{{captureId:id}});
}});
window.addEventListener('g1:policy-remove',event=>{{
  const id=captureId(event.detail?.captureId);if(!activeState?.policies.some(x=>x.captureId===id))return;
  tell('g1:policy-remove',{{captureId:id}});
  if(parent===window){{
    const next=normalize({{policies:activeState.policies.filter(x=>x.captureId!==id),emphasizedCaptureId:activeState.emphasizedCaptureId}});
    sessionStorage.setItem(TRIAL_STORAGE,JSON.stringify(next));location.replace(canonicalUrl(next));
  }}
}});
window.addEventListener('message',event=>{{
  if(event.origin!==location.origin||event.source!==parent||event.data?.type!=='g1:set-policies'||String(event.data.replayGeneration??'')!==REPLAY_GENERATION)return;
  try{{const state=normalize(event.data);sessionStorage.setItem(TRIAL_STORAGE,JSON.stringify(state));const next=canonicalUrl(state);
    if(next.search!==location.search)location.replace(next);else{{activeState=state;if(rendererReady)tell('g1:policies-state',{{...state,pausedAt:0}});}}
  }}catch(error){{show(error.message);tell('g1:policies-error',{{message:error.message,failedCaptureIds:[]}});}}
}});
function terminal(raw,frames,run){{
  if(run?.valid===true)return {{time:null,reason:null}};
  let time=run?.first_disqualification_time_s??run?.dq_time??run?.stop_time_s??run?.duration_s??frames.at(-1)?.[0]??0;
  let reason=run?.first_disqualification_gate??run?.dq_reason??run?.termination_reason;
  if(!reason||reason==='finished')reason='did_not_finish';return {{time:Number(time),reason:String(reason)}};
}}
function packCapture(raw,item,links,names){{
  if(Number(raw.schema_version)!==2)throw new Error(`${{item.captureId}} uses unsupported capture schema.`);
  const frameSet=(raw.frames||[])[Number(raw.representative_lane)||0]||raw.frames?.[0];
  if(!frameSet?.length)throw new Error(`${{item.captureId}} has no pose frames.`);
  const run=(raw.runs||[])[Number(raw.representative_lane)||0]||raw.runs?.[0]||{{}};
  const ridx=links.map(name=>names.indexOf(name)),fps=Number(raw.fps)||50,torso=Math.max(0,names.indexOf('torso_link'));
  let finish=null;if(run.finish!=null)finish=Number(run.finish);
  if(finish===null){{const o=1+torso*7;for(const row of frameSet)if(row[o]>=100){{finish=Number(row[0]);break;}}}}
  if(finish===null)finish=Number(frameSet.at(-1)[0]);
  const term=terminal(raw,frameSet,run),timedOut=run.valid===false&&['timeout','time_limit'].includes(term.reason);
  const clipT=timedOut?Number(frameSet.at(-1)[0]):((run.valid===false?term.time:finish)+1.5);
  const frames=frameSet.filter(row=>Number(row[0])<=clipT).map(row=>{{const packed=[Number(row[0])];for(const body of ridx){{const o=1+body*7;for(let j=0;j<7;j++)packed.push(Number(row[o+j]));}}return packed;}});
  const laneCheck=(run.checks||[]).find(x=>x.name==='in_lane');
  const policy={{label:item.label,policy_number:item.policyNumber,lane_number:item.policyNumber,finish,frames,max_lateral_m:Number(laneCheck?.value)||0,valid:run.valid,terminal_time:term.time,terminal_reason:term.reason,timed_out:timedOut,disqualified:['in_lane','self_collision'].includes(term.reason),effective_speed_mps:Number(run.effective_speed_mps)||0,identity:item.identity||null,source_run_id:item.runId,color:item.renderColor,model_color:item.color,emphasized:item.emphasized,capture_id:item.captureId}};
  if(run.valid===true){{delete policy.terminal_time;delete policy.terminal_reason;}}
  return policy;
}}
function laneMarkup(policy,index){{const color=policy.color,name=`Policy #${{policy.policy_number}}`;return `<div class="lc${{policy.emphasized?' emphasized':''}}" style="--emphasis:${{color}}" id="lane${{index}}" role="group" aria-label="${{name}}"><button type="button" class="policy-follow" aria-pressed="false" aria-label="Follow ${{name}}"><span class="sw" style="background:${{color}}"></span><span class="nm">#${{policy.policy_number}}</span><span class="tm" style="color:${{color}}">0.00s</span><span class="d">0.0 m</span></button><button type="button" class="policy-remove" aria-label="Remove ${{name}}" title="Remove ${{name}}">×</button></div>`;}}
async function boot(){{
  tell('g1:policies-ready',{{maxPolicies:8,path:'/replay/trial-comparison.html'}});
  let state;try{{state=initialState();}}catch(error){{show(error.message);tell('g1:policies-error',{{message:error.message,failedCaptureIds:error.failedCaptureIds||[]}});return;}}
  activeState=state;
  if(!state.policies.length){{show('Select up to 8 policies from one trial to compare.');tell('g1:policies-state',{{...state,pausedAt:0}});return;}}
  show(`Loading ${{state.policies.length}} polic${{state.policies.length===1?'y':'ies'}}…`);
  const settled=await Promise.allSettled(state.policies.map(async item=>{{const response=await fetch(item.url,{{cache:'force-cache'}});if(!response.ok)throw new Error(`${{item.captureId}}: ${{response.status}}`);return response.json();}}));
  const failed=settled.map((result,index)=>result.status==='rejected'?state.policies[index].captureId:null).filter(Boolean);
  if(failed.length){{const message=`Unable to load ${{failed.join(', ')}}.`;show(message);tell('g1:policies-error',{{message,failedCaptureIds:failed}});return;}}
  const raws=settled.map(x=>x.value),names=raws[0].body_names,fps=Number(raws[0].fps)||50;
  try{{for(let i=0;i<raws.length;i++)if(JSON.stringify(raws[i].body_names)!==JSON.stringify(names)||(Number(raws[i].fps)||50)!==fps)throw new Error(`${{state.policies[i].captureId}} is incompatible with this comparison.`);
    const links=TRIAL_BOOT.preferred.filter(name=>names.includes(name)&&TRIAL_BOOT.hq[name]);
    const emphasis=state.emphasizedCaptureId,items=state.policies.map(item=>({{...item,identity:item.identity||TRIAL_BOOT.registry[item.captureId]?.identity||null,runId:item.runId||TRIAL_BOOT.registry[item.captureId]?.runId||null,emphasized:item.captureId===emphasis,renderColor:item.captureId===emphasis?item.color:'#FFFFFF'}}));
    const policies=raws.map((raw,index)=>packCapture(raw,items[index],links,names));
    lanesEl.innerHTML=policies.map(laneMarkup).join('');noteEl.hidden=true;
    const data={{fps,links,parents:Object.fromEntries(links.filter(x=>links.includes(TRIAL_BOOT.parents[x])).map(x=>[x,TRIAL_BOOT.parents[x]])),rest:{{}},hq:Object.fromEntries(links.map(x=>[x,TRIAL_BOOT.hq[x]])),policies,colors:policies.map(x=>parseInt(x.color.slice(1),16)),lane_indices:policies.map((_,i)=>i),lane_labels:Array(8).fill(null),meta:{{comparison:true,trajectory_comparison:true,track_lanes:8,lane_half_width_m:.61,start_paused:true,interpolation:'adjacent_authoritative_position_lerp_quaternion_slerp'}}}};
    new Function('DATA',TRIAL_BOOT.scene)(data);
    requestAnimationFrame(()=>{{rendererReady=true;tell('g1:policies-state',{{...activeState,pausedAt:0,failedCaptureIds:[]}});}});
  }}catch(error){{show(error.message);tell('g1:policies-error',{{message:error.message,failedCaptureIds:state.policies.map(x=>x.captureId)}});}}
}}
boot();
</script>
"""
    shell_css = """
<style>
html,body{margin:0;background:#070908}.wrap{max-width:none;padding:0}.wrap>.eyebrow,.wrap>h1,.wrap>.lede,.polsel,.cap,.story{display:none}.stagewrap{margin:0;border:0;border-radius:0;box-shadow:none}
/* The result under the timer must not move the independent policy list. */
.hud:has(.clock-status:not(:empty)) .lanes{top:12px}
.lc.emphasized{box-shadow:inset 3px 0 0 var(--emphasis,#fff);border-color:var(--emphasis,#fff);background:color-mix(in srgb,var(--emphasis,#fff) 20%,#0a1118)}
.lc .policy-follow{display:flex;align-items:center;gap:8px;flex:1;min-width:0;border:0;background:none;padding:0;color:inherit;font:inherit;text-align:left;cursor:pointer}
.lc .policy-remove{flex:none;width:24px;height:24px;border:0;border-radius:4px;background:rgba(255,255,255,.08);color:inherit;font:700 18px/1 sans-serif;cursor:pointer}
.lc .policy-remove:hover{background:rgba(255,255,255,.2)}.lc button:focus-visible{outline:2px solid #fff;outline-offset:3px}
</style>
"""
    return head.replace('<div class="wrap">', shell_css + '<div class="wrap">', 1) + bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-index", type=Path, default=POLICY_INDEX_DEFAULT)
    parser.add_argument("--hq", type=Path, default=HQ_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registry = load_registry(args.policy_index)
    hq = json.loads(args.hq.read_text())["meshes"]
    html = build_shell(registry=registry, hq=hq)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(json.dumps({"output": str(args.out), "usableCaptures": len(registry)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
