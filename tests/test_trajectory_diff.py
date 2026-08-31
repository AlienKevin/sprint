"""Unified tool diffs are lazy, bounded, safe, and faithful to raw arguments."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def node(body):
    script = r"""
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element {
 constructor(tag){this.tagName=tag;this.className='';this.childNodes=[];this.dataset={};this.attributes={};this.listeners={};this.value='';}
 append(...nodes){this.childNodes.push(...nodes)}
 set textContent(value){this.value=String(value);this.childNodes=[]}
 get textContent(){return this.value+this.childNodes.map(n=>n.textContent??String(n)).join('')}
 set innerHTML(value){throw Error('HTML interpretation is forbidden')}
 replaceChildren(...nodes){this.value='';this.childNodes=nodes}
 setAttribute(key,value){this.attributes[key]=value}
 addEventListener(name,handler){this.listeners[name]=handler}
 closest(){return null}
 querySelectorAll(selector){return all(this,n=>n!==this&&n.className.split(' ').includes(selector.slice(1)))}
 replaceWith(value){this.replacement=value}
}
const all=(node,match)=>[...(match(node)?[node]:[]),...(node.childNodes||[]).flatMap(child=>all(child,match))];
const css=(node,name)=>all(node,n=>n.className?.split(' ').includes(name));
let observations=[],unobserved=[];
const document={createElement:tag=>new Element(tag),createDocumentFragment:()=>new Element('#fragment'),listeners:{},addEventListener(name,handler){this.listeners[name]=handler}};
class IntersectionObserver {constructor(handler){this.handler=handler}observe(node){observations.push(node)}unobserve(node){unobserved.push(node)}}
const context={document,performance,IntersectionObserver};context.window=context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('web/vendor/jsdiff-9.0.0.min.js','utf8'),context);
vm.runInContext(fs.readFileSync('web/trajectory-diff.js','utf8'),context);
const api=context.TrajectoryDiff;
const reconstruct=(model,side)=>model.rows.filter(r=>r.kind!==(side==='old'?'add':'remove')).map(r=>r.raw).join('');
const hydrate=node=>{const parent=new Element('div');parent.append(node);api.renderWithin(parent);return node};
""" + body
    result = subprocess.run(["node", "-"], input=script, text=True, capture_output=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_line_and_inline_diff_for_real_edit():
    data=json.loads((ROOT / 'web/data/trajectories/claude-goalfix2-20260828-1750-glm-2.json').read_text())
    # The displayed turn a1-s129 groups several sequential tool-only steps;
    # its third Edit (the annotated block) comes from raw step a1-s131.
    edit=next(call for step in data['steps'] if step['step_id']=='a1-s131' for call in step['tool_calls'] if call['function_name']=='Edit')
    node('const input='+json.dumps(edit['arguments'])+';\n'+r"""
const model=api.compute(input.old_string,input.new_string);
assert.equal(model.status,'ready');assert.equal(reconstruct(model,'old'),input.old_string);assert.equal(reconstruct(model,'new'),input.new_string);
assert.ok(model.added>0&&model.removed>0);assert.ok(model.rows.some(r=>r.segments?.some(s=>s.changed)));
const rendered=hydrate(api.create(input.old_string,input.new_string));
assert.ok(css(rendered,'diff-add').length);assert.ok(css(rendered,'diff-remove').length);assert.ok(css(rendered,'diff-inline-change').length);
assert.ok(rendered.textContent.includes('Line numbers refer to the edited excerpt.'));
assert.ok(css(rendered,'diff-sign').some(n=>n.textContent==='+'));
""")


def test_unicode_whitespace_blank_lines_insert_delete_and_crlf_are_lossless():
    node(r"""
for(const [before,after]of [
 ['', 'hello\n'],['hello\n',''],['\n\n','\n \n'],
 ['const 你好 = "😀";\n','const 你好 = "😃";\n'],
 ['\tbefore\r\nlast\r\n','\tafter\r\nlast\r\n'],
 ['same\r\n','same\n'],['a\r','a\n'],['same','same\n'],['\r\n','\n']]){
 const model=api.compute(before,after);assert.equal(model.status,'ready');
 assert.equal(reconstruct(model,'old'),before);assert.equal(reconstruct(model,'new'),after);
 for(const row of model.rows)if(row.segments)assert.equal(row.segments.map(x=>x.text).join(''),row.raw.replace(/(?:\r\n|\r|\n)$/,''));
}
assert.equal(api.compute('unchanged\r\n','unchanged\r\n').status,'unchanged');
assert.equal(api.compute('','').status,'unchanged');
const endings=hydrate(api.create('same\r\n','same\n'));
assert.ok(css(endings,'diff-line-ending').some(x=>x.textContent==='CRLF'));
assert.ok(css(endings,'diff-line-ending').some(x=>x.textContent==='LF'));
assert.ok(hydrate(api.create('same','same\n')).textContent.includes('No newline'));
const unicode=api.compute('emoji="😀"\n','emoji="😃"\n');
assert.equal(unicode.rows[0].segments.filter(x=>x.changed).map(x=>x.text).join(''),'😀');
""")


def test_rendering_is_safe_and_diff_is_lazy():
    node(r"""
let computations=0;const original=context.Diff.diffArrays;context.Diff.diffArrays=(...args)=>{computations++;return original(...args)};
const source='<img src=x onerror="globalThis.PWNED=true">\n</script>\n';
const rendered=api.create(source,source+'<b>still text</b>\n');
assert.equal(computations,0);assert.equal(rendered.dataset.diffState,'pending');assert.equal(observations.length,1);
hydrate(rendered);assert.equal(computations,1);assert.equal(rendered.dataset.diffState,'ready');
hydrate(rendered);assert.equal(computations,1,'render is cached per element');
assert.equal(all(rendered,n=>['img','script','b'].includes(n.tagName)).length,0);
assert.ok(rendered.textContent.includes('<b>still text</b>'));assert.equal(context.PWNED,undefined);
assert.ok(unobserved.includes(rendered));
const closed=api.create('old','new');closed.closest=()=>({open:false});hydrate(closed);assert.equal(closed.dataset.diffState,'pending');
closed.closest=()=>null;const parent=new Element('details');parent.open=true;parent.append(closed);
document.listeners.toggle({target:parent});assert.equal(closed.dataset.diffState,'ready');
""")


def test_context_is_compact_and_expandable_without_recomputing():
    node(r"""
const before=Array.from({length:50},(_,n)=>`line ${n}\n`).join(''),after=before.replace('line 25','changed 25');
const root=hydrate(api.create(before,after)),gaps=css(root,'diff-context-gap');assert.equal(gaps.length,2);
assert.ok(css(root,'diff-line').length<15);const gap=gaps[0],button=css(gap,'diff-context-toggle')[0];
assert.equal(button.type,'button');let stopped=false;button.listeners.click({stopPropagation(){stopped=true}});
assert.ok(stopped);assert.ok(gap.replacement);assert.ok(css(gap.replacement,'diff-line').length>10);
""")


def test_limits_fail_gracefully_and_keep_raw_arguments_external():
    node(r"""
assert.equal(api.compute('a'.repeat(api.limits.characters+1),'b').status,'limited');
assert.equal(api.compute('a\n'.repeat(api.limits.lines+1),'b').status,'limited');
assert.equal(api.compute(null,'b').status,'unavailable');
assert.equal(hydrate(api.create('a'.repeat(api.limits.characters+1),'b')).dataset.diffState,'limited');
assert.ok(hydrate(api.create('a'.repeat(api.limits.characters+1),'b')).textContent.includes('raw arguments below'));
const original=context.Diff.diffArrays;let settings;context.Diff.diffArrays=(a,b,options)=>{settings=options;return undefined};
assert.equal(api.compute('a','b').status,'limited');assert.equal(settings.timeout,25);assert.equal(settings.maxEditLength,1200);
context.Diff.diffArrays=original;context.Diff=null;assert.equal(api.compute('a','b').status,'unavailable');
""")


def test_chinese_labels_never_change_source_nodes():
    node(r"""
const catalogs={};context.SiteI18n={language:'en',register(lang,values){catalogs[lang]=values},t(key,params,fallback){return (catalogs[this.language][key]??fallback).replace(/\{(\w+)\}/g,(m,k)=>params[k]??m)}};
vm.runInContext(fs.readFileSync('web/locales/trajectory.js','utf8'),context);
const root=hydrate(api.create('value = 1\n','value = 2\n')),codes=css(root,'diff-code'),before=codes.map(n=>n.textContent);
context.SiteI18n.language='zh-CN';for(const node of all(root,n=>n.dataset?.trajectoryI18n))node.textContent=context.SiteI18n.t('trajectory.'+node.dataset.trajectoryI18n,JSON.parse(node.dataset.trajectoryParams),node.dataset.trajectoryI18n);
assert.ok(root.textContent.includes('改动'));assert.deepEqual(css(root,'diff-code'),codes);assert.deepEqual(codes.map(n=>n.textContent),before);
""")


def test_vendor_provenance_and_responsive_non_html_presentation():
    readme=(ROOT/'web/vendor/README.md').read_text()
    assert 'diff@9.0.0' in readme and 'BSD-3-Clause' in readme
    assert 'Copyright' in (ROOT/'web/vendor/jsdiff-9.0.0.LICENSE').read_text()
    source=(ROOT/'web/trajectory-diff.js').read_text()
    assert '.innerHTML' not in source and 'eval(' not in source
    css=(ROOT/'web/trajectory-diff.css').read_text()
    assert 'minmax(0,1fr)' in css and 'overflow-wrap: anywhere' in css
    assert 'white-space: pre-wrap' in css
