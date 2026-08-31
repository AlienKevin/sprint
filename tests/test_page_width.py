"""The homepage should stop growing without changing its small-screen gutters."""

from pathlib import Path
import re


CSS = (Path(__file__).resolve().parents[1] / "web" / "styles.css").read_text()


def test_homepage_shells_share_a_centered_maximum_width() -> None:
    assert "--page-max-width: 1280px" in CSS
    rule = re.search(
        r"body > \.hero,\s*body > main,\s*body > footer\s*\{([^}]+)\}", CSS
    ).group(1)
    assert "width: 100%" in rule
    assert "max-width: var(--page-max-width)" in rule
    assert "margin-inline: auto" in rule


def test_shared_gutters_stop_growing_with_the_column() -> None:
    assert "--page-gutter: clamp(24px, 5vw, 64px)" in CSS
    assert "padding: 14px var(--page-gutter)" in CSS
    assert "padding: 0 var(--page-gutter) 38px" in CSS
    assert "padding: 0 var(--page-gutter)" in CSS
    assert "padding: 30px var(--page-gutter)" in CSS
    assert "max-width: 1680px" not in CSS
    mobile = CSS.split("@media (max-width: 720px)", 1)[1]
    assert "padding-left: max(16px, env(safe-area-inset-left))" in mobile
    assert "padding-right: max(16px, env(safe-area-inset-right))" in mobile


def test_retired_budget_caption_has_no_orphan_styles() -> None:
    assert ".budget-caption" not in CSS


def test_footer_has_no_editorial_date_or_orphan_date_styles() -> None:
    page = (Path(__file__).resolve().parents[1] / "web/index.html").read_text()
    footer = page.split("<footer>", 1)[1].split("</footer>", 1)[0]
    assert 'class="brand"' in footer
    assert "footer-updated" not in footer
    assert "Updated on" not in footer
    assert ".footer-updated" not in CSS


def test_demo_titles_and_trial_links_share_one_row_on_mobile_and_desktop() -> None:
    rule = re.search(r"\.dq-card-head\s*\{([^}]+)\}", CSS).group(1)
    assert "display: flex" in rule
    assert "flex-wrap: nowrap" in rule
    assert "justify-content: space-between" in rule
    assert "align-items: center" in rule
    assert ".dq-card-head { display: block; }" not in CSS
    assert ".dq-card-head a { display: inline-block; margin-top: 12px; }" not in CSS
