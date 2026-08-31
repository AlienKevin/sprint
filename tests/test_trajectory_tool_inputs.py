"""Readable tool inputs preserve exact data without interpreting model content."""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "web/trajectory.js").read_text()


def function(name):
    start = SOURCE.index(f"  function {name}(")
    following = re.search(r"\n  (?:(?:async )?function |const )", SOURCE[start + 1:])
    return SOURCE[start:start + 1 + following.start()]


def node(body):
    helpers = "\n".join(line for line in SOURCE.splitlines() if line.strip().startswith(
        ("const text=", "const compact=", "const preview=", "const el=")))
    helpers += "\n" + "\n".join(function(name) for name in [
        "ui", "label", "bindLabel",
        "commandValue", "toolArguments", "toolPreview", "toolInputFields", "toolInputSource",
        "structuredToolInput", "toolActivity", "outputPanel", "appendStepActivities",
    ])
    script = """
const assert=require('node:assert/strict');
class Element {
 constructor(tag){this.tagName=tag;this.className='';this.childNodes=[];this.listeners={};this.attributes={};this.dataset={};this.value='';}
 append(...nodes){this.childNodes.push(...nodes)}
 set textContent(value){this.value=String(value);this.childNodes=[]}
 get textContent(){return this.value+this.childNodes.map(n=>n.textContent??String(n)).join('')}
 set innerHTML(value){throw Error('Untrusted input must never use innerHTML')}
 addEventListener(name,handler){this.listeners[name]=handler}
 setAttribute(key,value){this.attributes[key]=value}
}
const document={createElement:tag=>new Element(tag)};
const window={};
const execCallSummary=()=>null,execActivity=()=>({codex:true}),observationActivity=result=>({unmatched:result});
const all=(node,match)=>[...(match(node)?[node]:[]),...(node.childNodes||[]).flatMap(child=>all(child,match))];
const css=(node,name)=>all(node,n=>n.className?.split(' ').includes(name));
const pre=node=>all(node,n=>n.tagName==='pre');
""" + helpers + "\n" + body
    result = subprocess.run(["node", "-"], input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_diff_integration_keeps_exact_inputs_and_raw_argument_fallback():
    node(r"""
const captured=[];window.TrajectoryDiff={create(before,after){captured.push([before,after]);const node=new Element('section');node.className='trajectory-diff';return node}};
const args={file_path:'/app/test.py',old_string:'old\r\n',new_string:'new\r\n',replace_all:false};
const root=structuredToolInput({function_name:'Edit',arguments:args});
assert.deepEqual(captured,[[args.old_string,args.new_string]]);assert.equal(css(root,'trajectory-diff').length,1);
assert.equal(css(root,'tool-input-before').length,0);assert.equal(css(root,'tool-input-after').length,0);
assert.equal(pre(css(root,'tool-input-raw')[0])[0].textContent,JSON.stringify(args,null,2));
const multi=structuredToolInput({function_name:'MultiEdit',arguments:{file_path:'/x',edits:[{old_string:'',new_string:'a'},{old_string:'b',new_string:''}]}});
assert.equal(css(multi,'trajectory-diff').length,2);assert.deepEqual(captured.slice(1),[['','a'],['b','']]);
assert.ok(multi.textContent.includes('Edit 1'));assert.ok(multi.textContent.includes('Edit 2'));
""")


def test_actual_write_and_edit_are_readable_and_verbatim():
    trace = json.loads((ROOT / "web/data/trajectories/claude-goalfix2-20260828-1750-glm-2.json").read_text())
    write = next(step for step in trace["steps"] if step["step_id"] == "a1-s23")["tool_calls"][0]
    edit = next(call for step in trace["steps"] for call in step.get("tool_calls", []) if call["function_name"] == "Edit")
    node(f"const write={json.dumps(write)},edit={json.dumps(edit)};\n" + r"""
const w=toolActivity(write,[]),e=toolActivity(edit,[]);
assert.equal(css(w,'tool-input-path')[0].textContent,'/app/sprint_train.py');
assert.equal(pre(css(w,'tool-input-source')[0])[0].textContent,write.arguments.content);
assert.ok(pre(w)[0].textContent.includes('\nimport argparse\n'));
assert.equal(css(w,'terminal-prompt').length,0,'file writes are not shell commands');
assert.equal(pre(css(w,'tool-input-raw')[0])[0].textContent,JSON.stringify(write.arguments,null,2));
assert.equal(pre(css(e,'tool-input-before')[0])[0].textContent,edit.arguments.old_string);
assert.equal(pre(css(e,'tool-input-after')[0])[0].textContent,edit.arguments.new_string);
assert.equal(css(e,'terminal-prompt').length,0);
assert.ok(toolPreview(edit).startsWith('file_path: '),'replace_all must not obscure the file preview');
""")


def test_injection_like_code_newlines_quotes_and_empty_replacements_are_exact():
    node(r"""
const content='</script><img src=x onerror="globalThis.PWNED=true">\n literal \\n '+String.fromCharCode(10)+'"quotes"\t';
const call={function_name:'Write',arguments:{file_path:'<b>not HTML</b>',content}};
const root=toolActivity(call,[]);
assert.equal(pre(root)[0].textContent,content);assert.equal(globalThis.PWNED,undefined);
assert.equal(all(root,n=>['script','img','b'].includes(n.tagName)).length,0);
const edit=structuredToolInput({function_name:'MultiEdit',arguments:{file_path:'/x',edits:[
 {old_string:'',new_string:content,replace_all:false},{old_string:content,new_string:''}]}});
assert.equal(css(edit,'tool-input-edit').length,2);
assert.deepEqual(css(edit,'tool-input-before').map(x=>pre(x)[0].textContent),['',content]);
assert.deepEqual(css(edit,'tool-input-after').map(x=>pre(x)[0].textContent),[content,'']);
assert.equal(css(edit,'tool-input-empty').length,2);
assert.ok(edit.textContent.includes('false'));
""")
    tool_source = function("structuredToolInput") + function("toolInputSource") + function("toolInputFields")
    assert "innerHTML" not in tool_source and "eval(" not in tool_source


def test_bash_metadata_preserves_false_zero_and_linked_outputs():
    node(r"""
const command='printf "%s\\n" "<script>text</script>"\nprintf done';
const call={function_name:'Bash',arguments:{command,description:'Run a check',timeout:0,run_in_background:false}};
const root=toolActivity(call,[{content:'raw output\nwith another line'}]);
assert.equal(css(root,'terminal-prompt')[0].textContent,'$');
assert.equal(pre(css(root,'terminal-command')[0])[0].textContent,command);
const fields=css(root,'tool-input-fields')[0];assert.ok(fields.textContent.includes('0'));assert.ok(fields.textContent.includes('false'));
assert.equal(pre(css(root,'terminal-output')[0])[0].textContent,'raw output\nwith another line');
let stopped=false;css(root,'tool-input-raw')[0].childNodes[0].listeners.click({stopPropagation(){stopped=true}});assert.equal(stopped,true);
assert.equal(css(root,'tool-input-raw')[0].open,undefined,'raw arguments remain collapsed');
""")


def test_common_tools_have_meaningful_structured_inputs():
    node("""
const cases=[
 ['Read',{file_path:'/x',offset:0,limit:20},['/x','Offset','0','Limit','20']],
 ['Glob',{path:'/src',pattern:'**/*.py'},['/src','Pattern','**/*.py']],
 ['Grep',{path:'/src',pattern:'a\\nb',output_mode:'content','-n':false},['Pattern','a\\nb','Output mode','content','false']],
 ['NotebookEdit',{notebook_path:'/x.ipynb',cell_id:'a',new_source:'print(1)\\n',edit_mode:'replace'},['/x.ipynb','Cell source','print(1)\\n']],
 ['WebFetch',{url:'https://example.com',prompt:'Summarize\\nthis'},['https://example.com','Prompt','Summarize\\nthis']],
 ['WebSearch',{query:'two words',allowed_domains:['example.com']},['Query','two words','example.com']],
 ['TaskCreate',{subject:'Train',description:'Try this\\nthen that',activeForm:'Training'},['Task','Train','Description','Active form']],
 ['TaskUpdate',{taskId:'1',status:'in_progress'},['Task ID','1','in_progress']],
 ['TaskGet',{taskId:'2'},['Task ID','2']],['TaskList',{},['View raw arguments']],
 ['TaskOutput',{task_id:'7',block:false,timeout:0},['Task ID','7','false','0']],
 ['KillShell',{shell_id:'s'},['Shell ID','s']],
 ['Task',{description:'Inspect',prompt:'Do not execute me',subagent_type:'Explore'},['Prompt','Do not execute me','Agent type']],
 ['TodoWrite',{todos:[{content:'Read file',status:'pending',activeForm:'Reading'}]},['Read file','pending','Reading']],
];
for(const [function_name,args,expected]of cases){const root=structuredToolInput({function_name,arguments:args});assert.ok(root,function_name);for(const text of expected)assert.ok(root.textContent.includes(text),function_name+' '+text);assert.equal(css(root,'terminal-prompt').length,0)}
""")


def test_unknown_or_malformed_arguments_fall_back_without_throwing_or_losing_data():
    node("""
for(const [function_name,args]of [['Unknown',{content:'line1\\nline2',other:false}],['Write',{file_path:'/x',content:12}],['MultiEdit',{file_path:'/x',edits:[null]}],['Write','{bad JSON'],['Read',[]],['Bash',null]]){
 const root=toolActivity({function_name,arguments:args},[]);assert.equal(css(root,'tool-input-raw').length,0);
 assert.equal(pre(root)[0].textContent,typeof args==='string'?args:JSON.stringify(args??{},null,2));
 assert.equal(css(root,'terminal-prompt').length,0);
}
const args={file_path:'/x',content:'line1\\nline2'};
assert.equal(pre(structuredToolInput({function_name:'Write',arguments:JSON.stringify(args)}))[0].textContent,args.content);
assert.deepEqual(toolActivity({function_name:'exec',arguments:{input:'tools.exec_command({cmd:"x"})'}},[]),{codex:true});
assert.equal(pre(toolActivity({function_name:'run',arguments:{cmd:'printf test'}},[]))[0].textContent,'printf test');
""")


def test_multiple_tools_keep_individual_matching_results():
    node("""
const body=new Element('div');
appendStepActivities(body,{tool_calls:[
 {tool_call_id:'a',function_name:'Write',arguments:{file_path:'/a',content:'first'}},
 {tool_call_id:'b',function_name:'Edit',arguments:{file_path:'/a',old_string:'first',new_string:'second'}}],
 observation:{results:[{source_call_id:'b',content:'edited'},{source_call_id:'a',content:'written'},{source_call_id:'other',content:'unmatched'}]}});
assert.equal(body.childNodes.length,3);
assert.equal(css(body.childNodes[0],'terminal-output')[0].textContent,'Outputwritten');
assert.equal(css(body.childNodes[1],'terminal-output')[0].textContent,'Outputedited');
assert.equal(body.childNodes[2].unmatched.content,'unmatched');
""")


def test_code_and_paths_wrap_inside_mobile_column_without_truncating_content():
    css = (ROOT / "web/trajectory.css").read_text()
    assert "overflow-wrap: anywhere" in re.search(r"\.tool-input-path \{([^}]+)", css).group(1)
    assert "minmax(0, 1fr)" in re.search(r"\.tool-input-fields > div \{([^}]+)", css).group(1)
    source = re.search(r"\.tool-input-source > pre \{([^}]+)", css).group(1)
    assert "max-height: 46vh" in source
    assert "white-space: pre-wrap" in re.search(r"\npre \{([^}]+)", css).group(1)


def test_language_switch_updates_only_explicit_ui_nodes_in_place():
    catalog = (ROOT / "web/locales/trajectory.js").read_text()
    node("""
const dictionaries={},state={data:null},stepsTarget={scrollTop:51},chapterNav={scrollTop:29};
window.SiteI18n={language:'en',register(lang,map){dictionaries[lang]=map},t(key,params={},fallback){return (dictionaries[this.language]?.[key]??fallback??key).replace(/\\{(\\w+)\\}/g,(match,key)=>params[key]??match)}};
""" + catalog + "\n" + function("refreshTrajectoryLanguage") + r"""
const content='Content\nBefore\n<script>not executed</script>\nliteral \\n';
const root=toolActivity({function_name:'Write',arguments:{file_path:'/Content.py',content}},[{content:'Output\nraw model reply'}]);
root.open=true;const raw=css(root,'tool-input-raw')[0];raw.open=true;
const originals=pre(root).map(node=>({node,value:node.textContent}));
const path=css(root,'tool-input-path')[0];
document.querySelectorAll=selector=>all(root,node=>node.dataset?.trajectoryI18n!==undefined);
window.SiteI18n.language='zh-CN';refreshTrajectoryLanguage();
assert.equal(css(root,'tool-input-heading')[0].textContent,'内容');
assert.equal(css(root,'tool-output-label')[0].textContent,'输出');
assert.equal(raw.childNodes[0].textContent,'查看原始参数');
assert.equal(root.open,true);assert.equal(raw.open,true);assert.equal(path.textContent,'/Content.py');
for(const {node,value}of originals){assert.ok(pre(root).includes(node));assert.equal(node.textContent,value)}
assert.equal(stepsTarget.scrollTop,51);assert.equal(chapterNav.scrollTop,29);
window.SiteI18n.language='en';refreshTrajectoryLanguage();
assert.equal(css(root,'tool-input-heading')[0].textContent,'Content');
for(const {node,value}of originals)assert.equal(node.textContent,value);
""")


def test_trajectory_catalog_covers_bound_static_ui_and_script_order():
    catalog = (ROOT / "web/locales/trajectory.js").read_text()
    html = (ROOT / "web/trajectory.html").read_text()
    keys = re.findall(r'data-i18n="([^"]+)"', html)
    keys += [pair.split(":", 1)[1] for value in re.findall(r'data-i18n-attr="([^"]+)"', html)
             for pair in value.split(";")]
    node("const dictionaries={};window.SiteI18n={register:(lang,map)=>dictionaries[lang]=map};\n"
         + catalog + f"\nfor(const key of {json.dumps(keys)})assert.ok(dictionaries['zh-CN'][key],key);\n")
    scripts = re.findall(r'<script src="([^?\"]+)', html)
    assert scripts.index("/i18n.js") < scripts.index("/locales/trajectory.js") < scripts.index("/trajectory.js")
    assert scripts.index("/locales/toc.js") < scripts.index("/trajectory.js")
    assert 'data-language-switcher' in html
    refresh = function("refreshTrajectoryLanguage")
    assert "renderSteps(" not in refresh and "replaceChildren(" not in refresh
    assert "bindLabel(node,node.dataset.trajectoryI18n" in refresh
    assert "stepsTarget.scrollTop=traceTop" in refresh
