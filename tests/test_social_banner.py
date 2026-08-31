"""Share previews use the supplied race screenshot without replacing the replay."""

from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import shutil
import struct

from event_runtime.export.site_bundle import build_site_bundle


ROOT = Path(__file__).resolve().parents[1]
ASSET = Path("assets/agents100m-race-banner.png")
IMAGE_URL = f"https://agents100m.com/{ASSET.as_posix()}"
ORIGINAL_SHA256 = "ab5682ec1ac3f39e8f456e11e508444b2e475bab575cd81ac41ddb20ec06b202"


class PageMetadata(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.iframes: list[str] = []
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta":
            key = attributes.get("property") or attributes.get("name")
            if key:
                assert key not in self.meta, f"Duplicate metadata: {key}"
                self.meta[key] = attributes.get("content", "")
        elif tag == "iframe":
            self.iframes.append(attributes.get("src", ""))
        elif tag == "img":
            self.images.append(attributes.get("src", ""))


def page_metadata() -> PageMetadata:
    page = PageMetadata()
    page.feed((ROOT / "web/index.html").read_text())
    return page


def test_static_share_metadata_uses_large_race_screenshot() -> None:
    meta = page_metadata().meta
    assert meta["og:title"] == meta["twitter:title"] == "Agents' 100m"
    assert meta["og:description"] == meta["twitter:description"] == meta["description"]
    assert meta["og:type"] == "website"
    assert meta["og:url"] == "https://agents100m.com/"
    assert meta["og:site_name"] == "Agents' 100m"
    assert meta["twitter:card"] == "summary_large_image"
    assert meta["og:image"] == meta["twitter:image"] == IMAGE_URL
    assert meta["og:image:alt"] == meta["twitter:image:alt"]
    assert "humanoid robots" in meta["og:image:alt"]


def test_banner_is_exact_original_png_with_matching_dimensions() -> None:
    image = (ROOT / "web" / ASSET).read_bytes()
    assert sha256(image).hexdigest() == ORIGINAL_SHA256
    assert image[:8] == b"\x89PNG\r\n\x1a\n"
    assert image[12:16] == b"IHDR"
    dimensions = struct.unpack(">II", image[16:24])
    assert dimensions == (1312, 735)
    meta = page_metadata().meta
    assert (int(meta["og:image:width"]), int(meta["og:image:height"])) == dimensions
    assert meta["og:image:type"] == "image/png"


def test_social_banner_does_not_replace_interactive_hero() -> None:
    page = page_metadata()
    assert "/model-race" in page.iframes
    assert not any("agents100m-race-banner.png" in source for source in page.images)


def test_site_bundle_preserves_share_banner(tmp_path: Path) -> None:
    web = tmp_path / "web"
    (web / ASSET.parent).mkdir(parents=True)
    shutil.copyfile(ROOT / "web/index.html", web / "index.html")
    shutil.copyfile(ROOT / "web" / ASSET, web / ASSET)
    bundle = tmp_path / "bundle"

    report = build_site_bundle(web, bundle)

    assert report["mode"] == "full_tree_without_current_batch"
    assert (bundle / "index.html").read_bytes() == (web / "index.html").read_bytes()
    assert sha256((bundle / ASSET).read_bytes()).hexdigest() == ORIGINAL_SHA256
