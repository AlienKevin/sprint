"""The selected chart policy opens the exact trial, replay selection and queue turn."""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_every_published_chart_point_links_to_its_selected_policy_and_queue_turn() -> None:
    app = (ROOT / "web/app.js").read_text()
    functions = app[app.index("  const MODEL ="):app.index("  let chartResizeFrame")]
    script = "const assert=require('node:assert/strict');\n"
    script += f"const api=require({json.dumps(str(ROOT / 'web/trajectory-url.js'))});\n"
    script += f"const performance=require({json.dumps(str(ROOT / 'web/data/performance/current.json'))});\n"
    script += "const window={TrajectoryURL:api};\n" + app.splitlines()[1] + "\n"
    script += """
const nodes=new Map();
const $=selector=>{
  if(!nodes.has(selector))nodes.set(selector,{innerHTML:'',removeAttribute(){},scrollIntoView(){}});
  return nodes.get(selector);
};
""" + functions + r"""
let count=0;
for(const model of performance.models){
  for(const point of model.points){
    showReadout(point,model.model);
    const html=$('#readout-stats').innerHTML;
    assert.equal((html.match(/<div/g)||[]).length,4);
    const href=html.match(/<a class="text-link" href="([^"]+)"/)[1].replaceAll('&amp;','&');
    const parsed=api.parse(href);
    assert.equal(parsed.runId,point.source_run_id);
    assert.deepEqual(parsed.policies,[point.submission_index]);
    assert.equal(parsed.focus,point.submission_index);
    assert.equal(parsed.step,point.queue_source_step_id||'');
    if(!point.queue_source_step_id)assert.equal(parsed.turn,point.queue_source_public_step_id??null);
    assert.match(html,/>Open trial<\/a>/);
    assert.equal($('#readout-detail').hidden,false);
    count++;
  }
}
assert.ok(count>=145);
const point=performance.models[0].points[0];
showReadout({...point,queue_source_step_id:null,queue_source_public_step_id:42},performance.models[0].model);
assert.match($('#readout-stats').innerHTML,/turn=42/);
showReadout({...point,queue_source_step_id:null,queue_source_public_step_id:null},performance.models[0].model);
assert.doesNotMatch($('#readout-stats').innerHTML,/&amp;(step|turn)=/);
showReadout({...point,source_run_id:null,run_id:null},performance.models[0].model);
assert.match($('#readout-stats').innerHTML,/Trial unavailable/);
assert.doesNotMatch($('#readout-stats').innerHTML,/<a /);
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_readout_uses_four_desktop_cells_and_two_by_two_on_mobile() -> None:
    css = (ROOT / "web/styles.css").read_text()
    assert "grid-template-columns: repeat(4, minmax(0, 1fr));" in css
    mobile = css.split("@media (max-width: 720px)", 1)[1]
    assert ".readout-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }" in mobile
    assert ".readout-trial a { white-space: nowrap; }" in css
