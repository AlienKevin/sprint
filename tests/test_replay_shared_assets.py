from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "web/renderers/g1-100-metres"


def renderer():
    spec = importlib.util.spec_from_file_location("shared_asset_renderer", RENDERER / "render.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example(mesh='{"pelvis":{"p":"abc","f":"def"}}', three="/* Three.js Authors */window.THREE={};"):
    # Deliberately noncanonical formatting/float spellings: publication must
    # preserve the exact original JSON, not just equivalent parsed values.
    return ('<div id="failure-note" hidden></div><script>' + three + '</script>\n'
            '<script>const DATA={"fps":50.0,"hq":' + mesh + ',"policies":[{"frames":[[0.0,1.2300],[77.36,74.2]]}]};\n'
            'window.sceneRuns=(window.sceneRuns||0)+1;\n</script>')


def test_publication_is_lossless_and_offline_restorable(tmp_path):
    module = renderer()
    original = example()
    published = module.shared_asset_html(original, tmp_path)
    assert module.restore_shared_html(published, tmp_path) == original
    assert module.shared_asset_html(published, tmp_path) == published
    assert '"frames":[[0.0,1.2300],[77.36,74.2]]' in published
    assert '"hq":null' in published
    assert "Three.js Authors" not in published
    assert 'DATA.hq=window.__G1_REPLAY_HQ__' in published
    assert len(list(tmp_path.glob("*.js"))) == 2
    for path in tmp_path.iterdir():
        assert path.stem.endswith(hashlib.sha256(path.read_bytes()).hexdigest())


def test_independent_content_hashes_change_only_modified_asset(tmp_path):
    module = renderer()
    one = module.shared_asset_html(example(), tmp_path)
    two = module.shared_asset_html(example(mesh='{"pelvis":{"p":"new","f":"def"}}'), tmp_path)
    three = module.shared_asset_html(example(three="/* Three.js Authors */window.THREE={new:true};"), tmp_path)
    urls = lambda html: dict(re.findall(r'data-replay-asset="(.*?)" src="(.*?)"', html))
    assert urls(one)["three"] == urls(two)["three"]
    assert urls(one)["hq"] != urls(two)["hq"]
    assert urls(one)["hq"] == urls(three)["hq"]
    assert urls(one)["three"] != urls(three)["three"]


def test_shared_assets_load_in_order_and_failure_does_not_start_scene(tmp_path):
    module = renderer()
    published = module.shared_asset_html(example(), tmp_path)
    assert published.index('data-replay-asset="three"') < published.index('data-replay-asset="hq"') < published.index("const DATA=")
    scripts = re.findall(r"<script>([\s\S]*?)</script>", published)
    failure = scripts[0]
    boot = scripts[-1]
    script = f"""
const assert=require('node:assert/strict');
const window=globalThis,note={{hidden:true,dataset:{{}},textContent:''}},button={{disabled:false}},messages=[];
const parent={{postMessage(data){{messages.push(data)}}}},location={{search:'?replayGeneration=3',origin:'https://example.test'}};
const document={{getElementById(id){{return id==='replay'?button:note}}}};
{failure}
g1ReplayAssetFailed('meshes');
{boot}
assert.equal(window.sceneRuns,undefined);assert.equal(note.hidden,false);assert.equal(button.disabled,true);
assert.match(note.textContent,/Unable to load replay meshes/);assert.equal(messages[0].replayGeneration,'3');
"""
    completed = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_trial_shell_can_share_and_restore_assets(tmp_path):
    sys.path.insert(0, str(RENDERER))
    try:
        spec = importlib.util.spec_from_file_location("shared_trial_shell", RENDERER / "render_trial_comparison.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = module.build_shell(registry={}, hq={"pelvis": {"p": "abc"}})
        published = module.build_shell(registry={}, hq={"pelvis": {"p": "abc"}}, shared_assets_dir=tmp_path)
    finally:
        sys.path.remove(str(RENDERER))
    assert renderer().restore_shared_html(published, tmp_path) == original
    assert "hq:window.__G1_REPLAY_HQ__" in published
    assert "if(window.__G1_REPLAY_ASSET_ERROR__)return;" in published


def test_default_renderer_stays_self_contained(tmp_path):
    module = renderer()
    arguments = dict(data={"hq": {"pelvis": {}}, "policies": [{"finish": 1}]},
                     title="Replay", eyebrow="Replay", headline="Replay", lede="Replay", cap="Replay", story="Replay", sr_only="Replay", active="test")
    original = module.assemble_html(**arguments)
    published = module.assemble_html(**arguments, shared_assets_dir=tmp_path)
    assert "Three.js Authors" in original
    assert 'data-replay-asset=' not in original
    assert module.restore_shared_html(published, tmp_path) == original


def test_tampered_asset_fails_roundtrip_validation(tmp_path):
    module = renderer()
    published = module.shared_asset_html(example(), tmp_path)
    asset = next(tmp_path.glob("g1-hq-*.js"))
    asset.write_text("tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        module.restore_shared_html(published, tmp_path)


def test_missing_asset_fails_roundtrip_validation(tmp_path):
    module = renderer()
    published = module.shared_asset_html(example(), tmp_path)
    next(tmp_path.glob("g1-hq-*.js")).unlink()
    with pytest.raises(FileNotFoundError):
        module.restore_shared_html(published, tmp_path)


def test_shared_asset_is_published_atomically(tmp_path, monkeypatch):
    module = renderer()
    replace = module.os.replace
    observed = []

    def check_staging(source, target):
        source, target = Path(source), Path(target)
        assert source.name.startswith(".") and source.name.endswith(".tmp")
        assert not target.exists()
        assert target.stem.endswith(hashlib.sha256(source.read_bytes()).hexdigest())
        observed.append(target.name)
        replace(source, target)

    monkeypatch.setattr(module.os, "replace", check_staging)
    module.shared_asset_html(example(), tmp_path)
    assert len(observed) == 2
    assert not list(tmp_path.glob(".*.tmp"))


def test_only_shared_asset_namespace_has_immutable_cache_policy():
    config = json.loads((ROOT / "web/vercel.json").read_text())
    immutable = [entry for entry in config["headers"] if any("immutable" in header["value"] for header in entry["headers"])]
    assert [entry["source"] for entry in immutable] == ["/assets/replay/(.*)"]
    assert "shared_assets_dir=WEB / \"assets/replay\"" in (ROOT / "event_runtime/export/performance.py").read_text()
