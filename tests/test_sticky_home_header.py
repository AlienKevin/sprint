"""Homepage navigation remains available beyond the hero at every viewport size."""

from html.parser import HTMLParser
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_navigation_is_body_level_so_hero_does_not_limit_stickiness():
    class Structure(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.nav_parents = []
            self.hero_parents = []

        def handle_starttag(self, tag, attrs):
            classes = dict(attrs).get("class", "").split()
            if "home-nav" in classes:
                self.nav_parents.append(self.stack.copy())
            if "hero" in classes:
                self.hero_parents.append(self.stack.copy())
            if tag not in {"meta", "link", "img", "input", "br", "hr", "source"}:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in self.stack:
                self.stack = self.stack[:len(self.stack) - self.stack[::-1].index(tag) - 1]

    html = (ROOT / "web/index.html").read_text()
    structure = Structure()
    structure.feed(html)
    assert structure.nav_parents == [["html", "body"]]
    assert structure.hero_parents == [["html", "body"]]
    assert html.index('<nav class="home-nav">') < html.index('<header class="hero">')
    nav = html.split('<nav class="home-nav">', 1)[1].split("</nav>", 1)[0]
    assert '<h1 class="brand">Agents\' <em>100m</em></h1>' in nav
    assert "data-language-switcher" in nav


def test_sticky_header_has_opaque_background_and_consistent_mobile_gutters():
    css = (ROOT / "web/styles.css").read_text()
    nav = re.search(r"^\.home-nav \{([^}]+)\}", css, re.M).group(1)
    assert "position: sticky" in nav
    assert "top: 0" in nav
    assert "z-index: 100" in nav
    assert "background: var(--bg)" in nav
    assert "padding: 14px var(--page-gutter)" in nav
    assert "body > .home-nav," in css
    mobile = css.split("@media (max-width: 720px)", 1)[1]
    gutters = re.search(r"\.home-nav,\s*\.hero,\s*main,\s*footer \{([^}]+)\}", mobile).group(1)
    assert "max(16px, env(safe-area-inset-left))" in gutters
    assert "max(16px, env(safe-area-inset-right))" in gutters
    assert "position:" not in mobile.split(".home-nav", 1)[1].split("}", 1)[0]


def test_requested_chinese_table_labels_are_exact_and_english_is_unchanged():
    path = str(ROOT / "web/locales/home.js")
    script = r"""
const assert=require('node:assert/strict');
const dictionaries={};
global.window={SiteI18n:{register(locale,values){dictionaries[locale]={...dictionaries[locale],...values}},apply(){}}};
global.document={readyState:'loading',addEventListener(){}};
""" + f"require({json.dumps(path)});\n" + r"""
assert.equal(dictionaries['zh-CN']['home.Best of Five'],'五次试验最快');
assert.equal(dictionaries['zh-CN']['home.Best of {count}'].replace('{count}','3'),'3次试验最快');
assert.equal(dictionaries['zh-CN']['home.Policies submitted'],'提交策略数');
assert.equal(dictionaries.en['home.Best of Five'],'Best of Five');
assert.equal(dictionaries.en['home.Policies submitted'],'Policies submitted');
"""
    result = subprocess.run(["node", "-"], input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
