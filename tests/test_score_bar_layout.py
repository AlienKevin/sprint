"""Keep score values readable while the chart bars shrink on narrow screens."""

from pathlib import Path
import re


CSS = (Path(__file__).resolve().parents[1] / "web" / "styles.css").read_text()


def test_score_tracks_can_shrink_at_tablet_widths() -> None:
    rule = re.search(r"\.hero-score-row\s*\{([^}]+)\}", CSS).group(1)
    assert "minmax(0, 2fr)" in rule


def test_mobile_score_values_get_their_full_text_width() -> None:
    mobile = CSS.split("@media (max-width: 720px)", 1)[1]
    rule = re.search(r"\.hero-score-row > b\s*\{([^}]+)\}", mobile).group(1)
    # The longest displayed value, "10.101 m/s", is ten monospace characters.
    assert "min-width: 10ch" in rule
    assert "white-space: nowrap" in rule
    narrow = CSS.split("@media (max-width: 560px)", 1)[1]
    rule = re.search(r"\.hero-score-row\s*\{([^}]+)\}", narrow).group(1)
    assert "grid-template-columns: minmax(0, 1fr) max-content" in rule


def test_best_of_five_sits_above_desktop_headers_and_centers_on_mobile() -> None:
    rule = next(rule for rule in re.findall(r"\.experiment-best-label\s*\{([^}]+)\}", CSS) if "position: absolute" in rule)
    assert "position: absolute" in rule
    assert "bottom: -4px" in rule
    assert "top: 50%" not in rule
    mobile = CSS.split("@media (max-width: 720px)", 1)[1]
    rule = re.search(r"\.experiment-best-label\s*\{([^}]+)\}", mobile).group(1)
    assert "top: 50%" in rule
    assert "bottom: auto" in rule
    assert "transform: translateY(-50%)" in rule


def test_expanded_mobile_tables_show_each_column_label_only_once_per_model() -> None:
    mobile = CSS.split("@media (max-width: 720px)", 1)[1]
    assert "content: attr(data-trial-label)" not in mobile
    repeated_labels = re.search(
        r"\.experiment-row:not\(:first-child\) td::before,\s*"
        r"\.budget-row:not\(:first-child\) td::before\s*\{([^}]+)\}", mobile
    ).group(1)
    assert "content: none" in repeated_labels
    assert "display: none" in repeated_labels
    numeric_rows = re.search(
        r"\.experiment-row:not\(:first-child\) td,\s*"
        r"\.budget-row:not\(:first-child\) td\s*\{([^}]+)\}", mobile
    ).group(1)
    assert "min-height: 36px" in numeric_rows


def test_mobile_column_headers_keep_the_same_font_and_band_when_toggled() -> None:
    mobile = CSS.split("@media (max-width: 720px)", 1)[1]
    header = re.search(
        r"\.experiment-row td::before,\s*\.budget-row td::before\s*\{([^}]+)\}", mobile
    ).group(1)
    assert "font: 500 var(--data-label-size)/1.25 var(--mono)" in header
    assert "min-height: 50px" in header
    assert "box-sizing: border-box" in header
    assert "border-bottom: 1px solid var(--line)" in header
    assert 'data-expanded="true"' not in mobile
