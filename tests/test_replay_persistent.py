"""Persistent comparison lifecycle: versioned requests, bounded cache, recovery."""
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (ROOT / "web/renderers/g1-100-metres/trial-comparison.js").read_text()
SCENE = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()


def node(script):
    completed = subprocess.run(["node", "-"], input=script, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_runtime_reuses_renderer_cache_and_rejects_stale_requests():
    stub = r"""
const assert=require('node:assert/strict'),window=globalThis,listeners={},messages=[],rafs=[],requests=[];
window.addEventListener=(type,fn)=>(listeners[type]??=[]).push(fn);
window.removeEventListener=(type,fn)=>{listeners[type]=(listeners[type]||[]).filter(listener=>listener!==fn)};
const emit=(type,event)=>{for(const fn of listeners[type]||[])fn(event)};
const location={origin:'https://example.test',href:'https://example.test/replay/trial-comparison?policies=frontier-000000000001&replayGeneration=41&replayDocumentGeneration=41',get search(){return new URL(this.href).search}};
const parent={postMessage(message){messages.push(message)}},history={replaceState(_a,_b,url){location.href=String(url)}};
// Privacy settings may block writes: rendering must remain functional.
const sessionStorage={getItem(){throw Error('blocked')},setItem(){throw Error('blocked')}};
const note={hidden:true,dataset:{},textContent:''},stage={hidden:false};
const lanes={children:[],set innerHTML(v){assert.equal(v,'');this.children=[]},append(card){this.children=this.children.filter(x=>x!==card);this.children.push(card)}};
const document={querySelector(){return lanes},getElementById(id){return id==='stage'?stage:note},createElement(){return {content:{},set innerHTML(html){
  const card={dataset:{captureId:html.match(/data-capture-id="([^"]+)"/)[1]},id:'',remove(){lanes.children=lanes.children.filter(x=>x!==card)}};this.content.firstElementChild=card;
}}}};
const requestAnimationFrame=fn=>{rafs.push(fn);return rafs.length};
const id=n=>'frontier-'+n.toString(16).padStart(12,'0');
const item=n=>({captureId:id(n),url:`/captures/${id(n)}.json`,policyNumber:n,color:'#66D693',runId:'one-trial'});
const TRIAL_BOOT={registry:Object.fromEntries(Array.from({length:20},(_,i)=>[id(i+1),item(i+1)])),preferred:['pelvis'],hq:{pelvis:{}},parents:{},scene:'unused'};
const failures=new Set(),deferred=new Set(),waiting=new Map();
const raw=n=>({schema_version:2,body_names:['pelvis'],fps:50,frames:[[[0,0,0,1,0,0,0,1],[.02,n*.01,0,1,0,0,0,1]]],runs:[{valid:false,termination_reason:'timeout',duration_s:.02}]});
const fetch=(url,{signal})=>{const n=parseInt(url.match(/frontier-([a-f0-9]+)/)[1],16);requests.push(n);
  if(failures.delete(n))return Promise.resolve({ok:false,status:503});
  const response={ok:true,json:async()=>raw(n)};
  if(deferred.has(n))return new Promise(resolve=>waiting.set(n,()=>resolve(response)));
  return Promise.resolve(response);
};
let builds=0,updates=0,throwUpdate=false,playing=false,time=0,followIndex=0,currentIds=[];
function build(data){builds++;currentIds=data.policies.map(p=>p.capture_id);window.__G1_REPLAY__={
  updatePolicies(policies,{emphasizedCaptureId}){updates++;if(throwUpdate){throwUpdate=false;throw Error('temporary renderer failure')};currentIds=policies.map(p=>p.capture_id);time=0;playing=false;followIndex=currentIds.indexOf(emphasizedCaptureId)},
  playback(){return {time,playing}},camera(){return {follow:followIndex}},follow(index){followIndex=index},seek(t){time=t;playing=false},dispose(){playing=false},
};}
const Function=function(){return build};
const selection=(numbers,generation,focus=numbers.at(-1))=>({type:'g1:set-policies',replayGeneration:generation,replayDocumentGeneration:41,policies:numbers.map(item),emphasizedCaptureId:id(focus||1)});
const send=(payload,source=parent,origin=location.origin)=>emit('message',{data:payload,source,origin});
async function settle(){for(let i=0;i<14;i++)await Promise.resolve();const pending=rafs.splice(0);pending.forEach(fn=>fn());for(let i=0;i<5;i++)await Promise.resolve()}
"""
    checks = r"""
(async()=>{
  await settle();assert.equal(builds,1);const api=window.__G1_REPLAY__;
  assert.equal(requests.length,1);assert.equal(messages.at(-1).replayGeneration,'41');
  time=1;send(selection([1],41));await settle();assert.equal(time,1,'same selection does not reset');
  send(selection([1,2],42));send(selection([1,2],42));await settle();
  assert.equal(requests.filter(n=>n===2).length,1,'duplicate parent load message deduplicates');
  assert.equal(builds,1);assert.strictEqual(window.__G1_REPLAY__,api);assert.equal(time,0);
  time=1;playing=true;send(selection([1,2],42,1));await settle();assert.equal(time,1);assert.equal(playing,true,'focus-only preserves playing');assert.equal(requests.length,2);
  send(selection([1],43));await settle();assert.equal(time,0);assert.equal(playing,false,'set changes preserve prior fresh-player behavior');
  send(selection([1,2],44));await settle();assert.equal(requests.length,2,'decoded cache survives removing/re-adding');
  deferred.add(3);send(selection([1,3],45));await settle();send(selection([1],46));await settle();
  waiting.get(3)();await settle();assert.equal(messages.at(-1).replayGeneration,'46');assert.deepEqual(currentIds,[id(1)]);
  assert.equal(window.__G1_COMPARISON_CACHE__.stats().inflight,0);
  failures.add(4);send(selection([1,4],47));await settle();assert.equal(messages.at(-1).type,'g1:policies-error');assert.deepEqual(currentIds,[id(1)]);
  send(selection([1,4],48));await settle();assert.equal(requests.filter(n=>n===4).length,2,'failed promises do not poison retry');assert.deepEqual(currentIds,[id(1),id(4)]);
  const count=requests.length;send(selection([2],47));send({...selection([2],49),replayDocumentGeneration:40});send(selection([2],49),{});send(selection([2],49),parent,'https://other.test');await settle();assert.equal(requests.length,count);assert.equal(messages.at(-1).replayGeneration,'48');
  throwUpdate=true;send(selection([1,2],49));await settle();assert.equal(messages.at(-1).type,'g1:policies-error');
  const failedUpdates=updates;send(selection([1,2],49));await settle();assert.equal(updates,failedUpdates+1,'renderer retry must update, not acknowledge a failed state');assert.deepEqual(currentIds,[id(1),id(2)]);assert.equal(requests.length,count);
  send(selection([1,2,4,5,6,7,8,9],50));await settle();assert.equal(currentIds.length,8);assert.equal(window.__G1_COMPARISON_CACHE__.stats().activeEntries,8);assert.equal(builds,1);
  time=1;send(selection([],51));await settle();assert.equal(stage.hidden,true);assert.equal(time,0);
  send(selection([1],52));await settle();assert.equal(stage.hidden,false);assert.equal(builds,1);
  cachePut('large',{bytes:CACHE_LIMIT_BYTES});assert.equal(cacheBytes,CACHE_LIMIT_BYTES);assert.equal(decodedCache.size,1);
  cachePut('oversize',{bytes:CACHE_LIMIT_BYTES+1});assert.equal(cacheBytes,CACHE_LIMIT_BYTES);assert.equal(decodedCache.has('oversize'),false);
  for(let i=0;i<30;i++)cachePut('small'+i,{bytes:10});assert.equal(decodedCache.size,12);assert(cacheBytes<=CACHE_LIMIT_BYTES);
  const beforeAsset=requests.length;window.__G1_REPLAY_ASSET_ERROR__=true;note.textContent='Unable to load replay meshes. Please reload to retry.';
  send(selection([20],53));await settle();assert.equal(requests.length,beforeAsset);assert.equal(messages.at(-1).replayGeneration,'53');assert.match(messages.at(-1).message,/replay meshes/);
  window.__G1_REPLAY_ASSET_ERROR__=false;send(selection([20],54));await settle();assert.deepEqual(currentIds,[id(20)]);
  emit('pagehide',{persisted:true});assert.equal(window.__G1_COMPARISON_CACHE__.stats().disposed,false,'bfcache is not destructive teardown');
  emit('pagehide',{persisted:false});assert.equal(window.__G1_COMPARISON_CACHE__.stats().disposed,true);assert.equal(cacheBytes,0);assert.equal(decodedCache.size,0);
  const finalRequests=requests.length;send(selection([1],55));await settle();assert.equal(requests.length,finalRequests);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    node(stub + RUNTIME + checks)


def test_actor_disposal_preserves_shared_geometry_and_materials():
    function = SCENE.split("function disposeActorObject(object){", 1)[1].split("\n}\n", 1)[0]
    node("const assert=require('node:assert/strict');\nfunction disposeActorObject(object){" + function + r"""
}
const make=()=>({count:0,dispose(){this.count++}}),sharedGeometry=make(),ownGeometry=make(),texture={...make(),isTexture:true};
const jointMat=make(),footMat=make(),ownMaterial={...make(),map:texture};
const SHARED_GEOMETRIES=new Set([sharedGeometry]),removed=[],scene={remove(object){removed.push(object)}};
const object={traverse(fn){for(const node of [{geometry:sharedGeometry,material:jointMat},{geometry:sharedGeometry,material:ownMaterial},{geometry:ownGeometry,material:ownMaterial},{material:footMat}])fn(node)}};
disposeActorObject(object);assert.equal(sharedGeometry.count,0);assert.equal(jointMat.count,0);assert.equal(footMat.count,0);
assert.equal(ownGeometry.count,1);assert.equal(ownMaterial.count,1);assert.equal(texture.count,1);assert.equal(removed.length,1);
""")


def test_persistent_parent_tracks_document_and_selection_separately():
    source = (ROOT / "web/trajectory-overview.js").read_text()
    body = source.split("function replayPolicy(", 1)[1].split("function onReplayFrameLoad", 1)[0]
    assert "rendererSelectionKey !== selectionKey" in body
    assert "if (needsNewFrame)" in body
    assert "postRendererSelection();" in body
    assert "new URL(currentReplay" not in body
    assert "replayDocumentGeneration: replayDocumentGeneration" in source
    assert "__G1_TRIAL__?.dispose()" in source
    assert "scene: '20260830-11'" in source
