"""Replay chrome can change language without changing recordings or playback."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "web/renderers/g1-100-metres"
SCENE = (RENDERER / "scene.js").read_text()
CATALOG = (ROOT / "web/locales/replay.js").read_text()


def module(name):
    sys.path.insert(0, str(RENDERER))
    try:
        spec = importlib.util.spec_from_file_location(name, RENDERER / f"{name}.py")
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.remove(str(RENDERER))


def node(script):
    result = subprocess.run(["node", "-"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_inline_runtime_is_idempotent_offline_and_before_renderer():
    renderer = module("render")
    source = '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width">\n<script>const DATA={};</script>'
    html = renderer.replay_i18n_html(source)
    assert html == renderer.replay_i18n_html(html)
    assert html.startswith('<meta charset="utf-8">\n<meta name="viewport"')
    assert html.index('replay-i18n-core') < html.index('replay-i18n-catalog') < html.index('const DATA=')
    assert '<script src=' not in html
    for part, path in [('core', 'i18n.js'), ('catalog', 'locales/replay.js')]:
        digest = hashlib.sha256((ROOT / 'web' / path).read_bytes()).hexdigest()
        assert f'id="replay-i18n-{part}" data-source-sha256="{digest}"' in html


def test_raw_json_fidelity_and_comparison_metadata_survive_refresh():
    migration = module("refresh_replay_chrome")
    raw = '{"fps":50.0,"meta":{"label":"</script> \\u4e2d"},"policies":[{"frames":[[0.0,1.2300],[77.36,74.2]]}]}'
    original = '<meta charset="utf-8">\n<script>const DATA=' + raw + ';\n' + SCENE + '\n</script>'
    refreshed = migration.refresh_offline(original)
    assert migration.data_text(refreshed)[0] == raw
    assert migration.refresh_offline(refreshed) == refreshed
    registry = '{"frontier-000000000001":{"label":"Policy #1","float":1.2300}}'
    shell = '<meta charset="utf-8">\n<script>\nconst TRIAL_BOOT={\n  registry:' + registry
    shell += ',\n  hq:{},\n  preferred:[],\n  parents:{},\n  scene:"old"\n};\nconst TRIAL_STORAGE="old";\n</script>'
    refreshed = migration.refresh_offline(shell)
    assert migration.boot_field(refreshed, 'registry')[0] == registry
    assert migration.refresh_offline(refreshed) == refreshed
    assert 'html.replay-mobile .camera-ctl{display:none}' in (ROOT / 'web/replay-template.html').read_text()


def test_scene_language_refresh_preserves_state_and_disposes_only_old_textures():
    localize = SCENE[SCENE.index('function localizeReplayScene(){'):SCENE.index("window.addEventListener('site:languagechange',localizeReplayScene);")]
    button = next(line for line in SCENE.splitlines() if line.startswith('function setBtn(){'))
    node(r"""
const assert=require('node:assert/strict'),window=globalThis,listeners={},catalogs={};let language='en';
window.addEventListener=(name,fn)=>(listeners[name]??=[]).push(fn);
window.SiteI18n={register(lang,entries){Object.assign(catalogs[lang]??={},entries)},t(key,params={},fallback=key){return(catalogs[language]?.[key]||fallback).replace(/\{(\w+)\}/g,(_,name)=>params[name]??'{'+name+'}')}};
const btn={textContent:''},document={readyState:'loading',addEventListener(){},querySelectorAll(){return[]},querySelector(){return null},getElementById(){return btn},title:'',documentElement:{dataset:{}}};
""" + CATALOG + r"""
const card={attrs:{},setAttribute(key,value){this.attrs[key]=value},querySelector(){return null}},cards=[card];
const POL=[{capture_id:'frontier-000000000004',policy_number:4}],IS_TRAJECTORY_COMPARISON=true;
let texturesDisposed=0,draws=[],layouts=0,sceneDisposed=false,playing=false,playT=2.39,T_END=2.39;
function plaque(label,color){return {userData:{label:ReplayI18n.text(label),color},dispose(){texturesDisposed++}}}
const PLAQUES=[{material:{map:plaque('COLLISION',123)},userData:{}}],initialPlaque=PLAQUES[0];
const policyPlaqueLabel=()=> 'COLLISION',draw=t=>draws.push(t),publishReplayLayout=()=>layouts++;
const camera={az:.7,dist:4},original=JSON.stringify(POL),initialTexture=PLAQUES[0].material.map;
window.__G1_REPLAY__={camera};
""" + button + localize + r"""
localizeReplayScene();assert.equal(btn.textContent,'▶ Replay');assert.equal(texturesDisposed,0);
language='zh-CN';localizeReplayScene();assert.equal(btn.textContent,'▶ 重播');assert.equal(card.attrs['aria-label'],'策略 #4');
assert.equal(PLAQUES[0].userData.resultLabel,'自身碰撞');assert.equal(texturesDisposed,1);
assert.strictEqual(PLAQUES[0],initialPlaque);assert.notStrictEqual(PLAQUES[0].material.map,initialTexture);
assert.equal(playT,2.39);assert.equal(playing,false);assert.deepEqual(camera,{az:.7,dist:4});assert.equal(JSON.stringify(POL),original);
assert.deepEqual(draws,[2.39,2.39]);localizeReplayScene();assert.equal(texturesDisposed,1,'same-language refresh reuses texture');
playing=true;playT=1;language='en';localizeReplayScene();assert.equal(btn.textContent,'❙❙ Pause');assert.equal(playT,1);assert.equal(playing,true);
sceneDisposed=true;const old=draws.length;localizeReplayScene();assert.equal(draws.length,old,'disposed scenes ignore language events');
""")


def test_comparison_loading_and_error_copy_updates_without_selection_work():
    runtime = (RENDERER / 'trial-comparison.js').read_text()
    helpers = runtime[runtime.index('let noteTranslation=null;'):runtime.index("lanesEl.innerHTML='';")]
    show = next(line for line in runtime.splitlines() if line.startswith('function show('))
    lane_markup = next(line for line in runtime.splitlines() if line.startswith('function laneMarkup('))
    node(r"""
const assert=require('node:assert/strict'),window=globalThis,listeners={},catalogs={};let language='en',disposed=false;
window.addEventListener=(name,fn)=>(listeners[name]??=[]).push(fn);
window.SiteI18n={register(lang,entries){Object.assign(catalogs[lang]??={},entries)},t(key,params={},fallback=key){return(catalogs[language]?.[key]||fallback).replace(/\{(\w+)\}/g,(_,name)=>params[name]??'{'+name+'}')}};
const noteEl={hidden:true,textContent:'',dataset:{}},document={readyState:'loading',addEventListener(){},getElementById(){return null}};
""" + CATALOG + helpers + show + lane_markup + r"""
const policy={policy_number:4,capture_id:'frontier-000000000004',color:'#66D693'};
assert(laneMarkup(policy,0).includes('aria-label="Follow Policy #4"'));
assert(laneMarkup(policy,0).includes('aria-label="Remove Policy #4"'));
showLocalized('loadingMany',{count:3},'Loading {count} policies…');assert.equal(noteEl.textContent,'Loading 3 policies…');
language='zh-CN';refreshComparisonLanguage();assert.equal(noteEl.textContent,'正在加载 3 个策略…');
assert(laneMarkup(policy,0).includes('aria-label="跟随策略 #4"'));
assert(laneMarkup(policy,0).includes('aria-label="移除策略 #4"'));
showError(localizedError('unavailable',{ids:'frontier-abc'},'Unable to load {ids}. Remove the unavailable policy or retry.'));
assert.equal(noteEl.textContent,'无法加载 frontier-abc。请移除不可用的策略或重试。');
language='en';refreshComparisonLanguage();assert.equal(noteEl.textContent,'Unable to load frontier-abc. Remove the unavailable policy or retry.');
noteEl.hidden=true;language='zh-CN';refreshComparisonLanguage();assert.equal(noteEl.hidden,true);
""")


def test_controls_translate_before_heavy_assets_finish_loading():
    node(r"""
const assert=require('node:assert/strict'),window=globalThis,catalogs={};let callback,button=null,disconnected=0,language='zh-CN';
window.addEventListener=()=>{};
window.SiteI18n={register(lang,entries){Object.assign(catalogs[lang]??={},entries)},t(key,params={},fallback=key){return(catalogs[language]?.[key]||fallback).replace(/\{(\w+)\}/g,(_,name)=>params[name]??'{'+name+'}')}};
const sources=['Unitree G1 policy replay on the sprint course.','Policy 4: Unitree G1 policy replay on a ±0.61 m corridor.','Lane 3 · GPT-5.6 Luna','17.663 s'];
const nodes=sources.map(textContent=>({textContent,dataset:{}}));
const document={readyState:'loading',title:"Agents' 100m · Best-policy race",documentElement:{dataset:{}},
  addEventListener(){},querySelectorAll(selector){return selector==='.lanes .lc'?[]:nodes},querySelector(){return null},getElementById(){return button}};
const MutationObserver=class{constructor(fn){callback=fn}observe(){}disconnect(){disconnected++}};
""" + CATALOG + r"""
assert(callback,'early parser observer installed without waiting for DOMContentLoaded');
button={textContent:'▶ Replay'};callback();assert.equal(button.textContent,'▶ 播放');assert.equal(disconnected,1);
assert.equal(document.title,"Agents' 100m · 最佳策略竞赛");
assert.equal(nodes[0].textContent,'宇树 G1 在短跑赛道上的策略回放。');
assert.equal(nodes[1].textContent,'策略 4：宇树 G1 在 ±0.61 m 跑道内的策略回放。');
assert.equal(nodes[2].textContent,'第 3 跑道 · GPT-5.6 Luna');assert.equal(nodes[3].textContent,'17.663 s');
language='en';ReplayI18n.refresh();assert.deepEqual(nodes.map(node=>node.textContent),sources);
assert.equal(document.title,"Agents' 100m · Best-policy race");
""")
