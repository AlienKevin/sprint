"""Shared mobile replay chrome stays outside the scene and on one compact row."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "web/replay-template.html").read_text().split("<script>", 1)[0]


def rule(selector: str) -> str:
    return re.search(r"(?:^|\n)\s*" + re.escape(selector) + r"\{([^}]+)\}", TEMPLATE).group(1)


def test_mobile_playback_and_camera_controls_share_one_flow_row() -> None:
    controls = rule("html.replay-mobile .replay-controls")
    assert "order:1" in controls
    assert "display:flex" in controls
    assert "flex-wrap:nowrap" in controls
    assert "justify-content:space-between" in controls
    groups = rule("html.replay-mobile .ctl,html.replay-mobile .camera-ctl")
    assert "position:static" in groups
    assert "flex:none" in groups
    assert "margin-left:auto" in rule("html.replay-mobile .camera-ctl")
    assert "order:0" in rule("html.replay-mobile #stage")
    assert "order:2" in rule("html.replay-mobile .lanes")


def test_mobile_hides_reset_but_preserves_desktop_camera_controls() -> None:
    assert rule("html.replay-mobile .camera-mode,html.replay-mobile .camera-reset") == "display:none"
    assert 'id="camera-zoom-in"' in TEMPLATE
    assert 'id="camera-zoom-out"' in TEMPLATE
    assert 'id="camera-reset"' in TEMPLATE
    assert "position:absolute" in rule(".camera-ctl")
    assert rule(".replay-controls") == "display:contents"


def test_mobile_examples_reserve_separate_clock_scene_and_controls_rows() -> None:
    layout = rule("html.replay-mobile.replay-example .stagewrap")
    assert "display:grid" in layout
    assert "grid-template-rows:40px minmax(0,1fr) 44px" in layout
    assert "grid-row:3" in rule("html.replay-mobile.replay-example .replay-controls")
    # Small cards at a320px page viewport can have an iframe narrower than300px.
    narrow = TEMPLATE.split("@media(max-width:300px)", 1)[1]
    assert "html.replay-mobile .seg button{padding:6px 5px;font-size:10px}" in narrow
    assert "html.replay-mobile .camera-ctl .btn{min-width:28px;padding:6px}" in narrow
