"""Run selectors group models without changing trial membership or selection."""

import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('page', ['trajectory', 'timeline'])
@pytest.mark.parametrize('requested', ['', 'luna-new', 'deep-old'])
def test_selector_order_preserves_within_model_order_and_selected_run(page, requested) -> None:
    source = (ROOT / f'web/{page}.js').read_text()
    helpers = '\n'.join(line for line in source.splitlines() if line.startswith(('  const modelOrder=', '  const orderedRunOptions=')))
    init = next(line for line in source.splitlines() if line.startswith('  async function init()'))
    script = "const assert=require('node:assert/strict');\n"
    script += f"const page={json.dumps(page)},requested={json.dumps(requested)};\n"
    script += """
const rows=[
  ['glm-new','z-ai/glm-5.3-flash','2026-08-30'],
  ['luna-new','openai/gpt-5.6-luna','2026-08-29'],
  ['deep-new','deepseek/deepseek-v4-flash','2026-08-28'],
  ['glm-old','z-ai/glm-5.3-flash','2026-08-27'],
  ['deep-old','deepseek/deepseek-v4-flash','2026-08-26'],
  ['luna-old','openai/gpt-5.6-luna','2026-08-25'],
].map(([run_id,model,created_at])=>({run_id,model,created_at,path:`/${run_id}.json`,ready:true}));
const original=JSON.stringify(rows),options=[];
const runSelect={options,replaceChildren(){options.length=0},append(o){options.push(o)},get value(){return (options.find(o=>o.selected)||options[0])?.value}};
const document={createElement(){return {dataset:{}}}},el=()=>({dataset:{}}),stepsTarget={replaceChildren(){throw new Error('unexpected error UI')}},status={};
const location={search:requested?'?run='+requested:''},state={};let index,performance,loaded;
const fetch=async()=>({ok:true,json:async()=>({runs:rows})});
const setAccent=()=>{},modelLabel=value=>value,loadRun=async(path)=>{loaded=path};
""" + helpers + '\n' + init + """
(async()=>{
  await init();
  assert.deepEqual(options.map(o=>o.dataset.runId),['deep-new','deep-old','luna-new','luna-old','glm-new','glm-old']);
  assert.equal(loaded,`/${requested||'glm-new'}.json`);
  assert.equal(runSelect.value,loaded);
  assert.equal(JSON.stringify(rows),original);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
