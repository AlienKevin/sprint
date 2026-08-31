#!/usr/bin/env python3
"""Refresh published replay UI without reserializing any recorded data.

Run without --write first to audit the complete publication. Each replacement
is atomic so previously frozen hard-linked deployment snapshots stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from render import REPOSITORY_ROOT, SCENE_SOURCE, replay_i18n_html, restore_shared_html, shared_asset_html
from render_comparison import add_camera_orbits, add_finish_highlight, add_mobile_closeup, add_presentation_result_placement

WEB = REPOSITORY_ROOT / "web"
DATA_MARKER = "<script>const DATA="
DECODER = json.JSONDecoder()


def data_text(html: str) -> tuple[str, dict, int]:
    start = html.index(DATA_MARKER) + len(DATA_MARKER)
    data, end = DECODER.raw_decode(html, start)
    return html[start:end], data, end


def boot_field(html: str, name: str) -> tuple[str, int, int]:
    marker = f"\n  {name}:"
    start = html.index(marker, html.index("const TRIAL_BOOT={")) + len(marker)
    _, end = DECODER.raw_decode(html, start)
    return html[start:end], start, end


def refresh_offline(html: str) -> str:
    scene = SCENE_SOURCE.read_text()
    if "const TRIAL_BOOT={" in html:
        _, start, end = boot_field(html, "scene")
        encoded = json.dumps(add_mobile_closeup(scene), separators=(",", ":")).replace("</", "<\\/")
        html = html[:start] + encoded + html[end:]
        runtime = html.index("const TRIAL_STORAGE=", start)
        html = html[:runtime] + SCENE_SOURCE.with_name("trial-comparison.js").read_text() + "\n</script>\n"
    else:
        _, data, end = data_text(html)
        previous_scene = html[end:]
        if "const FINISH_HIGHLIGHTS=DATA.meta.finish_highlights;" in previous_scene:
            scene = add_finish_highlight(scene, data)
        if "const COMPARISON_ORBITS=DATA.meta.camera_orbits;" in previous_scene:
            scene = add_camera_orbits(scene, data)
        if "const compactFollowScale=runnerFollowComposition&&comparisonMobileLayout()?0.95:1;" in previous_scene:
            scene = add_mobile_closeup(scene)
        if "Number.isFinite(p.presentation_result_distance_m)?p.presentation_result_distance_m:" in previous_scene:
            scene = add_presentation_result_placement(scene)
        html = html[:end] + ";\n" + scene + "\n</script>"
    html = html.replace("html.replay-mobile .camera-ctl{margin-left:auto}", "html.replay-mobile .camera-ctl{display:none}")
    return replay_i18n_html(html)


def atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".replay-chrome-", suffix=".tmp", delete=False) as stream:
        staging = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.chmod(staging, path.stat().st_mode & 0o777)
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)


def refresh(paths: list[Path], assets: Path, *, write: bool = False) -> dict:
    records = []
    for path in paths:
        before = path.read_text()
        offline = restore_shared_html(before, assets)
        trial = "const TRIAL_BOOT={" in offline
        raw = boot_field(offline, "registry")[0] if trial else data_text(offline)[0]
        updated = refresh_offline(offline)
        new_raw = boot_field(updated, "registry")[0] if trial else data_text(updated)[0]
        if raw != new_raw:
            raise AssertionError(f"recorded payload changed: {path}")
        if trial:
            for field in ("hq", "preferred", "parents"):
                if boot_field(offline, field)[0] != boot_field(updated, field)[0]:
                    raise AssertionError(f"comparison {field} changed: {path}")
        after = shared_asset_html(updated, assets) if "data-replay-asset=" in before else updated
        if restore_shared_html(after, assets) != updated:
            raise AssertionError(f"shared asset restoration differs: {path}")
        asset_urls = lambda value: re.findall(r'data-replay-asset="[^"]+" src="([^"]+)"', value)
        if asset_urls(before) != asset_urls(after):
            raise AssertionError(f"shared asset identity changed: {path}")
        if write and before != after:
            atomic_write(path, after)
        records.append({"path": str(path.relative_to(WEB)), "kind": "registry" if trial else "DATA",
                        "payloadSha256": hashlib.sha256(raw.encode()).hexdigest(), "changed": before != after,
                        "bytes": len(after.encode())})
    return {"documents": len(records), "dataDocuments": sum(item["kind"] == "DATA" for item in records),
            "registryDocuments": sum(item["kind"] == "registry" for item in records),
            "changed": sum(item["changed"] for item in records), "dataAndAssetsUnchanged": True, "records": records}


def refresh_bootstrap(paths: list[Path], *, write: bool = False) -> dict:
    """Update only the bounded inline language runtime, preserving all else."""
    pattern = r"<!-- replay-i18n:start -->[\s\S]*?<!-- replay-i18n:end -->\n?"
    records = []
    for path in paths:
        before = path.read_text()
        after = replay_i18n_html(before)
        untouched = re.sub(pattern, "", before, count=1)
        if re.sub(pattern, "", after, count=1) != untouched:
            raise AssertionError(f"content outside localization bootstrap changed: {path}")
        if write and before != after:
            atomic_write(path, after)
        records.append({"path": str(path.relative_to(WEB)), "changed": before != after,
                        "unchangedContentSha256": hashlib.sha256(untouched.encode()).hexdigest()})
    return {"documents": len(records), "changed": sum(item["changed"] for item in records),
            "dataAndAssetsUnchanged": True, "allContentOutsideBootstrapUnchanged": True, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    paths = [WEB / "replay-template.html", WEB / "model-race.html", *sorted((WEB / "replay").glob("*.html"))]
    report = refresh_bootstrap(paths, write=args.write) if args.bootstrap_only else refresh(paths, WEB / "assets/replay", write=args.write)
    if args.audit:
        args.audit.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()
