"""Replay chrome must be mobile-correct before renderer assets finish loading."""
import importlib.util
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ControlsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.controls = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "div":
            classes = attrs.get("class", "").split()
            if "ctl" in classes or "camera-ctl" in classes:
                self.controls.append((classes, "replay-controls" in self.stack))
            self.stack.append("replay-controls" if "replay-controls" in classes else "")

    def handle_endtag(self, tag):
        if tag == "div" and self.stack:
            self.stack.pop()


def test_initial_classes_and_static_controls_precede_heavy_assets():
    for name in ["replay-template.html", "model-race.html", "replay/frontier-bca4f7ab8e3c.html", "replay/trial-comparison.html"]:
        html = (ROOT / "web" / name).read_text()
        boot = html.index('<script id="replay-initial-layout">')
        assert html.index('<meta name="viewport"') < boot, name
        assert boot < html.index("<style>"), name
        if 'data-replay-asset="three"' in html:
            assert boot < html.index('data-replay-asset="three"'), name
        parser = ControlsParser()
        parser.feed(html.split('<script>const DATA=', 1)[0])
        assert len(parser.controls) == 2, name
        assert all(in_wrapper for _, in_wrapper in parser.controls), name
    scene = (ROOT / "web/renderers/g1-100-metres/scene.js").read_text()
    assert "if(!document.querySelector('.replay-controls')){" in scene


def test_initial_layout_uses_outer_viewport_without_renderer_dependencies():
    html = (ROOT / "web/replay-template.html").read_text()
    script = re.search(r'<script id="replay-initial-layout">([\s\S]*?)</script>', html)[1]
    assert "DATA" not in script and "THREE" not in script
    harness = """
const assert=require('node:assert/strict');
const classes=new Set();
const window=globalThis,parent={innerWidth:390,postMessage(){}};
const innerWidth=600,location={search:'?example=1',origin:'https://example.test'};
const document={documentElement:{classList:{toggle(k,on){on?classes.add(k):classes.delete(k)},contains(k){return classes.has(k)}}}};
const listeners={};function addEventListener(k,fn){listeners[k]=fn}
class MutationObserver{observe(){}disconnect(){}}
"""
    assertions = """
assert(classes.has('replay-mobile'));assert(classes.has('replay-embedded'));assert(classes.has('replay-example'));
parent.innerWidth=1280;listeners.resize();assert(!classes.has('replay-mobile'));
"""
    result = subprocess.run(["node", "-e", harness + script + assertions], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_generator_keeps_static_wrapper_and_display_order_without_changing_data():
    path = ROOT / "web/renderers/g1-100-metres/render.py"
    spec = importlib.util.spec_from_file_location("initial_layout_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = {"meta": {"comparison": True}, "hq": {}, "policies": [
        {"label": label, "finish": 1, "frames": [[0, 1.2300], [77.36, 2]]}
        for label in ["DeepSeek-V4-Flash", "GLM-5.3-Flash", "GPT-5.6 Luna"]
    ]}
    html = module.assemble_html(data, title="Test", eyebrow="Test", headline="Test", lede="Test", cap="Test", story="Test", sr_only="Test", active="Test", show_policy_labels=True)
    assert html.index('id="lane0"') < html.index('id="lane2"') < html.index('id="lane1"')
    parser = ControlsParser();parser.feed(html.split('<script>const DATA=', 1)[0])
    assert len(parser.controls) == 2 and all(in_wrapper for _, in_wrapper in parser.controls)
    payload = html.split('<script>const DATA=', 1)[1]
    assert json.JSONDecoder().raw_decode(payload)[0] == data


def test_mobile_hero_reserves_intrinsic_stage_and_controls_height():
    css = (ROOT / "web/styles.css").read_text()
    assert ".model-race-frame::before { content: ''; display: block; aspect-ratio: 16 / 9; }" in css
    assert ".model-race-frame::after { content: ''; display: block; height: 156px; }" in css
    assert ".model-race-frame iframe { position: absolute; inset: 0; }" in css
