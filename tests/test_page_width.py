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
    assert "padding: 14px var(--page-gutter) 38px" in CSS
    assert "padding: 0 var(--page-gutter)" in CSS
    assert "padding: 30px var(--page-gutter)" in CSS
    assert "max-width: 1680px" not in CSS
    mobile = CSS.split("@media (max-width: 720px)", 1)[1]
    assert "padding-left: max(16px, env(safe-area-inset-left))" in mobile
    assert "padding-right: max(16px, env(safe-area-inset-right))" in mobile


def test_retired_budget_caption_has_no_orphan_styles() -> None:
    assert ".budget-caption" not in CSS


def test_footer_shows_fixed_editorial_date_on_the_right() -> None:
    page = (Path(__file__).resolve().parents[1] / "web/index.html").read_text()
    footer = page.split("<footer>", 1)[1].split("</footer>", 1)[0]
    assert '<time class="footer-updated" datetime="2026-08-30">Updated on August 30, 2026</time>' in footer
    assert footer.index('class="brand"') < footer.index('class="footer-updated"')
    rule = re.search(r"\.footer-updated\s*\{([^}]+)\}", CSS).group(1)
    assert "text-align: right" in rule
    assert "min-width: 0" in rule
    mobile = CSS.split("@media (max-width: 720px)", 1)[1]
    assert re.search(r"footer\s*\{\s*display: flex;", mobile)
