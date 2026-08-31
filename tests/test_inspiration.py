"""Inspiration cards have genuine local artwork and identifiable source links."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def inspiration() -> str:
    page = (ROOT / "web/index.html").read_text()
    return re.search(r'<section[^>]+id="inspiration">([\s\S]*?)</section>', page).group(1)


def test_inspiration_follows_observations_and_precedes_footer() -> None:
    page = (ROOT / "web/index.html").read_text()
    assert page.index('id="observations"') < page.index('id="inspiration"') < page.index('<footer>')
    section = inspiration()
    assert '<h2>Inspirations</h2>' in section
    assert re.findall(r'<h3 class="display-title">(.*?)</h3>', section) == ["QWOP Game", "Robot Races", "PostTrainBench"]
    assert "Frontierbench" not in page
    assert "design-credit" not in page


def test_inspiration_images_are_local_linked_and_credited() -> None:
    section = inspiration()
    images = re.findall(r'<a class="inspiration-image" href="([^"]+)"><img ([^>]+)></a>', section)
    assert len(images) == 3
    for (_, attributes), filename in zip(images, ["lyi-qwop.webp", "robot-games.webp", "posttrainbench.webp"]):
        assert f'src="/assets/inspiration/{filename}"' in attributes
        assert 'width="1280" height="720"' in attributes
        assert 'loading="lazy"' in attributes
        assert re.search(r'alt="[^"]+"', attributes)
        assert (ROOT / "web/assets/inspiration" / filename).is_file()
    assert "<figcaption" not in section
    assert section.count('alt="') == 3


def test_inspiration_sources_and_distinctions_are_explicit() -> None:
    section = inspiration()
    links = re.findall(r'<a class="text-link" href="([^"]+)">([^<]+)</a>', section)
    assert {url for url, _ in links} == {
        "https://run.lyihub.com/",
        "https://www.youtube.com/watch?v=NgneWNGmq7g",
        "https://www.whrgoc.com/",
        "https://www.youtube.com/watch?v=KyFZuBgPBsg",
        "https://www.youtube.com/watch?v=CDI-Myx087U",
        "https://posttrainbench.com/",
    }
    for text in ["fully autonomously", "Unitree G1", "Isaac Lab", "3D physics",
                 "record-breaking", "400m event", "pretrained language models",
                 "one H100 and ten hours", "built from scratch", "no reference policy", "$10 budget"]:
        assert text in section
    prose = re.findall(r'<p>(.*?)</p>', section)
    assert 120 <= len(" ".join(prose).split()) <= 170
    assert len(prose[2].split()) <= 1.25 * max(len(paragraph.split()) for paragraph in prose[:2])
    assert "—" not in section


def test_inspiration_cards_stack_on_mobile_and_preserve_image_ratio() -> None:
    css = (ROOT / "web/styles.css").read_text()
    assert ".inspiration-cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));" in css
    mobile = css.split("@media (max-width: 1000px)", 1)[1]
    assert ".inspiration-cards { grid-template-columns: 1fr; gap: 18px; }" in mobile
    assert ".inspiration-card img { display: block; width: 100%; height: auto; aspect-ratio: 16 / 9;" in css
    assert ".inspiration-image:focus-visible { outline: 2px solid var(--text);" in css


def test_clickable_hero_scores_keep_chart_styling_and_keyboard_focus() -> None:
    css = (ROOT / "web/styles.css").read_text()
    row = re.search(r'\.hero-score-row \{([^}]+)\}', css).group(1)
    assert "color: var(--text)" in row
    assert "text-decoration: none" in row
    assert ".hero-score-row:focus-visible { outline: 2px solid var(--text);" in css
