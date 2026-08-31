"""Language changes are explicit, local, reversible and safe without browser storage."""

import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run_node(script: str) -> None:
    result = subprocess.run(["node", "-e", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_core_storage_fallback_interpolation_and_iframe_propagation() -> None:
    source = (ROOT / "web/i18n.js").read_text()
    script = "const assert=require('node:assert/strict'),vm=require('node:vm');\n"
    script += f"const source={json.dumps(source)};\n"
    script += r"""
function setup({stored,blocked=false,query='',parentLanguage=null}={}){
  const listeners={},messages=[],events=[],saved=[];
  const frame={contentWindow:{postMessage(data,origin){messages.push({data,origin})}}};
  const doc={readyState:'loading',documentElement:{lang:''},querySelectorAll(s){return s==='iframe'?[frame]:[]},addEventListener(){}};
  const location={href:'https://agents100m.test/'+query,origin:'https://agents100m.test'};
  const window={addEventListener(name,fn){listeners[name]=fn},dispatchEvent(event){events.push(event)},parent:null};
  window.parent=parentLanguage?{SiteI18n:{language:parentLanguage},postMessage(data,origin){messages.push({data,origin})}}:window;
  const localStorage={getItem(){if(blocked)throw Error('denied');return stored},setItem(key,value){if(blocked)throw Error('denied');saved.push({key,value})}};
  const history={state:null,replaceState(state,title,url){location.href=String(url)}};
  vm.runInNewContext(source,{window,document:doc,location,localStorage,history,URL,CustomEvent:class{constructor(type,options){this.type=type;this.detail=options.detail}}});
  return {api:window.SiteI18n,doc,listeners,messages,events,saved,frame,location};
}
const a=setup({blocked:true});assert.equal(a.api.language,'en');
a.api.register('en',{'test.message':'Hello {name}','test.onlyEn':'English fallback'});
a.api.register('zh-CN',{'test.message':'你好，{name}'});
a.api.setLanguage('zh-CN');assert.equal(a.doc.documentElement.lang,'zh-CN');assert.equal(a.api.t('test.message',{name:'跑者'}),'你好，跑者');
assert.equal(a.api.t('test.onlyEn'),'English fallback');assert.equal(a.api.t('missing',{},'Fallback'),'Fallback');
assert.equal(a.events.length,1);assert.equal(a.events[0].type,'site:languagechange');assert.equal(a.events[0].detail.language,'zh-CN');
assert.equal(a.messages.length,1);assert.equal(a.messages[0].origin,'https://agents100m.test');
a.api.setLanguage('zh-CN');assert.equal(a.events.length,1);assert.equal(a.messages.length,1); // no event loop
a.listeners.message({origin:'https://evil.test',source:a.frame.contentWindow,data:{type:'site:language',language:'en'}});assert.equal(a.api.language,'zh-CN');
a.listeners.message({origin:'https://agents100m.test',source:{},data:{type:'site:language',language:'en'}});assert.equal(a.api.language,'zh-CN');
a.listeners.message({origin:'https://agents100m.test',source:a.frame.contentWindow,data:{type:'site:language',language:'en'}});assert.equal(a.api.language,'en');
assert.equal(setup({stored:'zh-CN'}).api.language,'zh-CN');
assert.equal(setup({stored:'invalid'}).api.language,'en');
assert.equal(setup({query:'?lang=zh-CN'}).api.language,'zh-CN');
assert.equal(setup({query:'?lang=en',parentLanguage:'zh-CN'}).api.language,'zh-CN');
const b=setup({query:'?lang=en'});b.api.setLanguage('zh-CN');assert.equal(new URL(b.location.href).searchParams.get('lang'),'zh-CN');
assert.equal(b.saved.at(-1).key,'agents100m.language');
"""
    run_node(script)


def test_home_dynamic_catalog_covers_explicit_translated_strings() -> None:
    app = (ROOT / "web/app.js").read_text()
    keys = set(re.findall(r"(?<![\w.])t\('((?:[^'\\]|\\.)*)'", app))
    # Argument strings are deliberately literal UI keys, not trace contents.
    script = "const assert=require('node:assert/strict'),vm=require('node:vm');\n"
    script += f"const keys={json.dumps(sorted(keys))},source={json.dumps((ROOT / 'web/locales/home.js').read_text())};\n"
    script += r"""
const catalogs={};
vm.runInNewContext(source,{window:{SiteI18n:{register(lang,values){Object.assign(catalogs[lang]??={},values)}}},document:{readyState:'loading',addEventListener(){}}});
for(const key of keys){assert.ok(catalogs.en['home.'+key],key);assert.ok(catalogs['zh-CN']['home.'+key],key);}
assert.equal(catalogs['zh-CN']['home.LANE DRIFT'],'越出跑道');
assert.equal(catalogs['zh-CN']['home.COLLISION'],'自身碰撞');
"""
    run_node(script)


def test_rapid_language_changes_converge_across_nested_frames_and_storage() -> None:
    source = (ROOT / "web/i18n.js").read_text()
    script = "const assert=require('node:assert/strict'),vm=require('node:vm');\n"
    script += f"const source={json.dumps(source)};\n"
    script += r"""
const origin='https://agents100m.test',queue=[],contexts=[];let writes=0,stored='en';
function page(){
  const listeners={},doc={readyState:'loading',documentElement:{lang:''},frames:[],querySelectorAll(s){return s==='iframe'?this.frames:[]},addEventListener(name,fn){listeners['document:'+name]=fn}};
  const window={addEventListener(name,fn){listeners[name]=fn},dispatchEvent(){}};window.parent=window;
  const context={window,document:doc,listeners,location:{href:origin+'/',origin},history:{state:null,replaceState(){}},URL,CustomEvent:class{constructor(type,options){this.type=type;this.detail=options.detail}}};
  context.localStorage={getItem(){return stored},setItem(key,newValue){writes++;if(stored===newValue)return;stored=newValue;for(const other of contexts)if(other!==context)queue.push(()=>other.listeners.storage?.({key,newValue}));}};
  contexts.push(context);return context;
}
function connect(parent,child){
  const childProxy={postMessage(data){queue.push(()=>child.listeners.message?.({origin,source:parentProxy,data}));}};
  const parentProxy={postMessage(data){queue.push(()=>parent.listeners.message?.({origin,source:childProxy,data}));}};
  parent.document.frames.push({contentWindow:childProxy});child.window.parent=parentProxy;
}
const top=page(),child=page(),nested=page(),sibling=page();connect(top,child);connect(child,nested);connect(top,sibling);
for(const context of contexts)vm.runInNewContext(source,context);
function drain(){let count=0;while(queue.length&&count++<1000)queue.shift()();assert.equal(queue.length,0,'language messages must converge, not echo');assert.ok(count<30,'bounded tree relay');}
top.window.SiteI18n.setLanguage('zh-CN');
// The child finishes loading with stale English before queued parent updates
// arrive. Its readiness handshake must request, not overwrite, the parent's choice.
child.listeners['document:DOMContentLoaded']();drain();
for(const context of contexts)assert.equal(context.window.SiteI18n.language,'zh-CN');
const before=writes;top.window.SiteI18n.setLanguage('en');top.window.SiteI18n.setLanguage('zh-CN');drain();
for(const context of contexts)assert.equal(context.window.SiteI18n.language,'zh-CN');
assert.equal(writes-before,2,'received language changes must not rewrite storage');
nested.window.SiteI18n.setLanguage('en');drain();for(const context of contexts)assert.equal(context.window.SiteI18n.language,'en');
// PostMessage must reach nested frames even if storage updated the middle frame first.
child.listeners.storage({key:'agents100m.language',newValue:'zh-CN'});
const parentSource=child.window.parent;child.listeners.message({origin,source:parentSource,data:{type:'site:language',language:'zh-CN'}});drain();
assert.equal(nested.window.SiteI18n.language,'zh-CN');assert.equal(top.window.SiteI18n.language,'en','no reflection to the sender');
"""
    run_node(script)


def test_home_catalog_keeps_scientific_record_and_translates_named_editorial_nodes() -> None:
    html = (ROOT / "web/index.html").read_text()
    catalog = (ROOT / "web/locales/home.js").read_text()
    core = (ROOT / "web/i18n.js").read_text()
    assert html.index('/i18n.js?') < html.index('/locales/home.js?') < html.index('/app.js?')
    assert 'data-language-switcher' in html
    assert "['#citation-bibtex'" not in catalog
    for selector in [".cost-insights > p:nth-child(2)", ".observation-cards > article:nth-child(3) p", ".setup-copy > ol > li:nth-child(3)", ".inspiration-card:nth-child(3) p"]:
        assert selector in catalog
    assert "TreeWalker" not in core
    assert "fetch(" not in core
    assert "document.readyState === 'loading'" in catalog
    assert "if (bound) return" in catalog
    assert "document.querySelector('footer #updated')" in catalog
    assert "observer.disconnect(); bind()" in catalog


def test_language_switch_does_not_replace_selected_replay() -> None:
    app = (ROOT / "web/app.js").read_text()
    assert "showReadout(activeReadout.point,activeReadout.model,true)" in app
    assert "if(!refreshTextOnly){if(point.replay_url)" in app
    assert "if(!refreshTextOnly)detail.scrollIntoView" in app


def test_mobile_column_layout_does_not_depend_on_translated_labels() -> None:
    css = (ROOT / "web/styles.css").read_text()
    app = (ROOT / "web/app.js").read_text()
    assert '.experiment-row td.experiment-total-cost { display: none; }' in css
    assert '<td class="experiment-total-cost"' in app
    assert '[data-label="Total cost"]' not in css


def test_legacy_timeline_localizes_ui_without_resetting_replay_or_telemetry() -> None:
    app = (ROOT / "web/timeline.js").read_text()
    html = (ROOT / "web/timeline.html").read_text()
    assert html.index('/i18n.js?') < html.index('/locales/timeline.js?') < html.index('/timeline.js?')
    assert 'data-language-switcher' in html
    assert "if(activePolicy)showPolicy(activePolicy,true)" in app
    handler = app.split("window.addEventListener('site:languagechange'", 1)[1].split("addEventListener('resize'", 1)[0]
    assert "loadRun(" not in handler
    assert "view=" not in handler
    assert "ctx.fillText(e.kind.replace('gpu_','')" in app
    script = "const assert=require('node:assert/strict'),vm=require('node:vm');\n"
    script += f"const source={json.dumps((ROOT / 'web/locales/timeline.js').read_text())},keys={json.dumps(sorted(set(re.findall(r'(?<![\w.])t\(\x27([^\x27]+)\x27', app))))};\n"
    script += "const catalogs={}; vm.runInNewContext(source,{window:{SiteI18n:{register(lang,values){Object.assign(catalogs[lang]??={},values)}}},document:{readyState:'loading',addEventListener(){}}}); for(const key of keys)assert.ok(catalogs['zh-CN']['timeline.'+key],key);"
    run_node(script)
