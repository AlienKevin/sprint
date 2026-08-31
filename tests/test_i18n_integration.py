"""Cross-page localization contracts; model-authored content is not a catalog."""

import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def catalogs():
    script = r"""
const fs=require('node:fs'),vm=require('node:vm');
const catalogs={en:{},'zh-CN':{}};
const document={readyState:'loading',addEventListener(){},getElementById(){return null}};
const window={SiteI18n:{register(lang,values){Object.assign(catalogs[lang],values)}},addEventListener(){}};
for(const file of ['home','trajectory','timeline','replay'])
  vm.runInNewContext(fs.readFileSync('web/locales/'+file+'.js','utf8'),{window,document});
console.log(JSON.stringify(catalogs));
"""
    result = subprocess.run(["node", "-e", script], cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_dynamic_catalogs_have_matching_keys_and_interpolation_parameters():
    data = catalogs()
    assert data["en"].keys() == data["zh-CN"].keys()
    for key, english in data["en"].items():
        chinese = data["zh-CN"][key]
        assert chinese, key
        params = lambda value: set(re.findall(r"\{([\w.]+)\}", value))
        # English singular/plural wording is supplied as a separate fragment;
        # Chinese expresses this directly. Counts and other data must remain.
        grammatical = {"policies"} if key == "timeline.{count} rendered {policies} · {submitted} submitted" else set()
        assert params(english) - grammatical == params(chinese), key


def test_termination_names_are_consistent_across_all_interfaces():
    data = catalogs()["zh-CN"]
    for english, replay_key, translated in [
        ("FINISHED", "finished", "已完赛"),
        ("LANE DRIFT", "drift", "越出跑道"),
        ("COLLISION", "collision", "自身碰撞"),
        ("TIMEOUT", "timeout", "超时"),
    ]:
        for namespace in ("home", "trajectory", "timeline"):
            assert data[f"{namespace}.{english}"] == translated
        assert data[f"replay.{replay_key}"] == translated


def test_chinese_keeps_token_in_english():
    data = catalogs()
    for key, english in data["en"].items():
        if re.search(r"\btokens?\b", english, re.IGNORECASE):
            assert re.search(r"\btoken\b", data["zh-CN"][key]), key
    # Include static editorial bindings and every TOC summary as well.
    for path in (ROOT / "web" / "locales").glob("*.js"):
        assert "词元" not in path.read_text(), path.name


def test_trajectory_literal_ui_labels_are_all_translated():
    data = catalogs()["zh-CN"]
    for file in ("trajectory.js", "trajectory-overview.js"):
        source = (ROOT / "web" / file).read_text()
        keys = re.findall(r"(?<![\w.])(?:ui|label|announceSelection)\('((?:[^'\\]|\\.)*)'", source)
        missing = [key for key in keys if "trajectory." + key.replace("\\'", "'") not in data]
        assert not missing, (file, missing)


def test_pages_load_core_before_catalogs_and_offer_header_switcher():
    for page, namespace in (("index.html", "home"), ("trajectory.html", "trajectory"), ("timeline.html", "timeline")):
        html = (ROOT / "web" / page).read_text()
        assert html.index("/i18n.js?") < html.index(f"/locales/{namespace}.js?")
        assert "data-language-switcher" in html
        assert "/i18n.css?" in html
