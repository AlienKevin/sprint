"""Chart selection stays obvious without changing coordinates or hit targets."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "web/app.js").read_text()


def test_selection_survives_renders_switches_cleanly_and_keeps_keyboard_actions() -> None:
    functions = APP[APP.index("  const MODEL ="):APP.index("  let chartResizeFrame")]
    script = r"""
const assert=require('node:assert/strict');
class Node {
  constructor(tag='div'){this.tag=tag;this.attrs={};this.children=[];this.events={};this.clientWidth=800;this.clientHeight=520;this.style={setProperty(){}};}
  setAttribute(name,value){this.attrs[name]=String(value)}
  removeAttribute(name){delete this.attrs[name]}
  addEventListener(name,handler){this.events[name]=handler}
  append(...nodes){for(const node of nodes){node.parent=this;this.children.push(node)}}
  replaceChildren(){this.children=[]}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(node=>node!==this)}
  getScreenCTM(){return {a:1,inverse(){return this}}}
  createSVGPoint(){return {x:0,y:0,matrixTransform(){return {x:this.x,y:this.y}}}}
  scrollIntoView(){}
}
const document={createElementNS:(_,tag)=>new Node(tag)};
const window={matchMedia:()=>({matches:false})};
const nodes=new Map();
const $=selector=>{if(!nodes.has(selector))nodes.set(selector,new Node());return nodes.get(selector)};
const trajectoryHref=run=>`/trajectory?run=${run}`;
""" + next(line for line in APP.splitlines() if line.strip().startswith("const t=")) + "\n" + functions + r"""
const point=(index,cost)=>({source_run_id:'glm-2',source_trial:2,submission_index:index,
  policy_sha256:`policy-${index}`,cumulative_agent_cost_usd:cost,continuous_score_mps:10,
  replay_url:'/replay/example',max_legal_distance_m:100,time_to_max_legal_distance_s:10,
  cost_at_queue_usd:cost,hours_since_agent_launch:2});
const model='z-ai/glm-5.3-flash',rows=[point(6,5),point(7,5.04)];
const draw=()=>continuousChart('#cost-chart',[{model,points:rows}],'cumulative_agent_cost_usd','Cost',10);
draw();
let chart=chartSelections.get('#cost-chart');
const coords=chart.selections.map(({cx,cy})=>[cx,cy]);
assert.ok(Math.abs(coords[0][0]-coords[1][0])<18); // Overlapping hit circles.
assert.equal(chart.overlay,null);
let prevented=false;
chart.selections[0].dot.events.keydown({key:'Enter',preventDefault(){prevented=true}});
assert.ok(prevented);
assert.equal(chart.selections[0].dot.attrs['aria-pressed'],'true');
assert.equal(chart.selections[1].dot.attrs['aria-pressed'],'false');
assert.equal(chart.el.children.at(-1),chart.overlay);
assert.equal(chart.overlay.attrs['aria-hidden'],'true');
assert.equal(chart.overlay.children[1].attrs.r,'13');
assert.equal(chart.overlay.children[1].attrs.fill,MODEL.glm.color);
assert.ok(chart.overlay.children.some(node=>node.attrs.class==='chart-selection-crosshair'));
assert.deepEqual(chart.selections.map(({cx,cy})=>[cx,cy]),coords);
assert.ok(chart.selections.every(item=>item.hit.attrs.r==='18'));
const oldOverlay=chart.overlay;
chart.el.onclick({clientX:coords[1][0],clientY:coords[1][1]});
assert.equal(chart.selections[0].dot.attrs['aria-pressed'],'false');
assert.equal(chart.selections[1].dot.attrs['aria-pressed'],'true');
assert.ok(!chart.el.children.includes(oldOverlay));
assert.match($('#readout-title').textContent,/policy 7/);
assert.equal($('#readout-detail').hidden,false);
draw(); // Periodic refresh and resize preserve selection by stable policy key.
chart=chartSelections.get('#cost-chart');
assert.equal(chart.selections[1].dot.attrs['aria-pressed'],'true');
assert.equal(chart.el.children.at(-1),chart.overlay);
assert.equal(chart.el.children.filter(node=>node.attrs.class==='chart-selection').length,1);
closeReadout();
assert.equal(chart.overlay,null);
assert.equal(selectedReadoutKey,null);
assert.ok(chart.selections.every(item=>item.dot.attrs['aria-pressed']==='false'));
assert.equal($('#readout-detail').hidden,true);
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_selection_has_high_contrast_without_intercepting_neighbor_clicks() -> None:
    css = (ROOT / "web/styles.css").read_text()
    assert ".chart-selection { pointer-events: none; }" in css
    assert ".chart-selection-halo { fill: none; stroke: var(--bg); stroke-width: 9px; }" in css
    assert ".chart-selection-dot { stroke: var(--text); stroke-width: 3.5px; }" in css
    assert "$('#readout-close').addEventListener('click',closeReadout)" in APP
    assert "Selection ticks may extend into the card's padding" in css
    assert "overflow: visible" in css.split(".chart-card svg {", 1)[1].split("}", 1)[0]
