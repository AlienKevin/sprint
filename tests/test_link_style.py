"""Editorial links match without restyling navigation controls or metric links."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_editorial_links_share_muted_bold_monospace_underlines() -> None:
    css = (ROOT / "web/fonts.css").read_text()
    rule = re.search(r":is\(\.text-link,[^{}]+\)\s*\{([^}]+)\}", css).group(1)
    assert "color: var(--text-link-color)" in rule
    assert "--text-link-color: #9d9d99" in css
    assert "font: 700 12px/1.5 var(--mono," in rule
    assert "text-decoration: underline" in rule
    assert "text-underline-offset: 2px" in rule
    for selector in [".observations-copy a", ".dq-card-head a", ".step-markdown a", ".lede a", ".help a"]:
        assert selector in css


def test_text_links_have_clear_hover_and_keyboard_focus() -> None:
    css = (ROOT / "web/fonts.css").read_text()
    assert ":is(:hover, :focus-visible)" in css
    assert "text-decoration-thickness: 2px" in css
    assert "outline: 2px solid var(--text, #f4f4f1)" in css
    assert "outline-offset: 3px" in css


def test_special_links_are_not_in_the_shared_text_link_selector() -> None:
    css = (ROOT / "web/fonts.css").read_text()
    selector = re.search(r"(:is\(\.text-link,[^{}]+\))\s*\{", css).group(1)
    for special in [".brand", ".experiment-table", ".budget-table", ".run-link", ".timeline-cta", ".policy-actions", ".chapter-link"]:
        assert special not in selector
    homepage = (ROOT / "web/styles.css").read_text()
    assert ".observation-cards a {" not in homepage
    assert ".dq-card-head a { white-space: nowrap; }" in homepage
    trajectory = (ROOT / "web/trajectory.css").read_text()
    assert ".step-markdown a {" not in trajectory
