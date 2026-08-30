"""The athletic display face is self-hosted and limited to editorial headings."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_display_face_is_a_self_hosted_extrabold_italic_with_license() -> None:
    css = (ROOT / "web/fonts.css").read_text()
    font = ROOT / "web/assets/fonts/BarlowCondensed-ExtraBoldItalic.woff2"
    assert font.read_bytes().startswith(b"wOF2")
    assert "font-style: italic" in css
    assert "font-weight: 800" in css
    assert "font-display: swap" in css
    assert "url('/assets/fonts/BarlowCondensed-ExtraBoldItalic.woff2')" in css
    assert "https://" not in css
    license_text = (font.parent / "OFL.txt").read_text()
    assert "SIL OPEN FONT LICENSE Version 1.1" in license_text
    assert "Copyright 2017 The Barlow Project Authors" in license_text


def test_homepage_headings_use_display_type_without_changing_metric_fonts() -> None:
    css = (ROOT / "web/styles.css").read_text()
    for selector in ["h1", ".section-head h2", ".readout-head h2"]:
        rule = re.search(re.escape(selector) + r"(?:,\s*[^{}]+)?\s*\{([^}]+)\}", css).group(1)
        assert "var(--display)" in rule
        assert "font-weight: 800" in rule
        assert "font-style: italic" in rule
    hero_highlight = re.search(r"h1 em\s*\{([^}]+)\}", css).group(1)
    assert "font-style: inherit" in hero_highlight
    assert "padding: 0.015em 0.2em 0.055em 0.1em" in hero_highlight
    metric = re.search(r"\.hero-score-row > b\s*\{([^}]+)\}", css).group(1)
    assert "var(--mono)" in metric
    assert "var(--display)" not in metric


def test_trajectory_brand_and_outline_use_display_type_not_transcripts() -> None:
    css = (ROOT / "web/trajectory.css").read_text()
    for selector in [".brand", ".rollout-outline h2", ".trajectory-policy-replay > header strong"]:
        rule = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css).group(1)
        assert "italic 800" in rule
        assert "var(--display)" in rule
    transcript = re.search(r"\.step-markdown h1[^{}]+\{([^}]+)\}", css).group(1)
    assert "var(--display)" not in transcript


def test_hero_highlight_matches_italic_slant_without_distorting_letters() -> None:
    css = (ROOT / "web/styles.css").read_text()
    face_css = (ROOT / "web/fonts.css").read_text()
    assert "--display-slant: -7deg" in face_css
    highlight = re.search(r"h1 em::before\s*\{([^}]+)\}", css).group(1)
    assert "transform: skewX(var(--display-slant))" in highlight
    assert "background: var(--text)" in highlight
    letters = re.search(r"h1 em\s*\{([^}]+)\}", css).group(1)
    assert "transform:" not in letters
    assert "isolation: isolate" in letters


def test_footer_uses_the_same_brand_style_and_hides_the_update_stamp() -> None:
    html = (ROOT / "web/index.html").read_text()
    footer = re.search(r"<footer>([\s\S]+?)</footer>", html).group(1)
    assert 'class="brand"' in footer
    assert 'id="updated" hidden' in footer
    css = (ROOT / "web/styles.css").read_text()
    brand = re.search(r"\.brand\s*\{([^}]+)\}", css).group(1)
    assert "color: var(--text)" in brand
    assert "italic 800" in brand
    assert "footer [hidden] { display: none; }" in css


def test_title_highlights_are_content_sized_and_exclude_data_and_transcripts() -> None:
    css = (ROOT / "web/fonts.css").read_text()
    assert "width: max-content" in css
    assert "max-width: none" in css
    assert "color: var(--bg, #050505)" in css
    assert "background: var(--text, #f4f4f1)" in css
    assert "transform: skewX(var(--display-slant))" in css
    for selector in [".brand em", ".section-head h2", ".readout-head h2", ".dq-card h3", ".observations-copy h3", ".setup-copy h3", ".rollout-outline h2"]:
        assert selector in css
    highlight_selector = re.search(r"(:is\(\.display-title,[^{}]+\))\s*\{", css).group(1)
    for selector in [".hero-score-row", ".experiment-model", ".step-markdown", ".chapter-title"]:
        assert selector not in highlight_selector


def test_editorial_titles_default_to_single_lines_without_truncation() -> None:
    css = (ROOT / "web/fonts.css").read_text()
    title_rule = re.search(r":is\(\.display-title,[^{}]+\)\s*\{([^}]+)\}", css).group(1)
    assert "white-space: nowrap" in title_rule
    assert "overflow-wrap: normal" in title_rule
    assert "width: max-content" in title_rule
    assert "max-width: none" in title_rule
    assert "text-overflow:" not in title_rule
    homepage = (ROOT / "web/styles.css").read_text()
    readout = re.search(r"\.readout-head\s*\{([^}]+)\}", homepage).group(1)
    assert "flex-wrap: wrap" in readout
    assert "font-size: clamp(18px, 5.3vw, 24px)" in homepage
    narrow_hero = homepage.split("@media (max-width: 560px)", 1)[1].split("}", 1)[0]
    assert "white-space: nowrap" in narrow_hero


def test_brand_highlights_only_the_100m_suffix() -> None:
    css = (ROOT / "web/fonts.css").read_text()
    assert ".brand em," in css
    assert ".brand," not in css
    for page in ["index.html", "trajectory.html"]:
        html = (ROOT / "web" / page).read_text()
        brands = re.findall(r'<(?:span|a|div)[^>]+class="brand"[^>]*>(.*?)</(?:span|a|div)>', html)
        assert brands
        assert all("<em>100m</em>" in brand for brand in brands)


def test_setup_copy_matches_scoring_spacing() -> None:
    css = (ROOT / "web/styles.css").read_text()
    assert ".scoring-copy,\n.setup-copy { max-width: 1080px; }" in css
    assert ".scoring-copy p,\n.setup-copy p {" in css
    assert ".setup-copy h3 { margin: 24px 0 8px; }" in css
    assert ".setup-copy li { padding-left: 6px; margin-bottom: 16px; }" in css
    assert ".setup-copy li strong,\n.harnesses-copy strong { color: var(--text); }" in css


def test_demo_card_titles_cannot_wrap_inside_their_highlights() -> None:
    css = (ROOT / "web/styles.css").read_text()
    title = re.search(r"\.dq-card \.dq-card-head h3\s*\{([^}]+)\}", css).group(1)
    assert "max-width: none" in title
    assert "white-space: nowrap" in title
    head = re.search(r"\.dq-card-head\s*\{([^}]+)\}", css).group(1)
    assert "flex-wrap: wrap" in head


def test_short_trajectory_titles_keep_their_intrinsic_single_line_width() -> None:
    css = (ROOT / "web/trajectory.css").read_text()
    titles = re.search(
        r"#rollout-outline-title,\s*#trajectory-policy-replay-title\s*\{([^}]+)\}", css
    ).group(1)
    assert "max-width: none" in titles
    assert "white-space: nowrap" in titles
    assert "flex-shrink: 0" in titles
