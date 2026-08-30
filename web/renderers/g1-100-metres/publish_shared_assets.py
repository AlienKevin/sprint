#!/usr/bin/env python3
"""Losslessly publish existing replays with shared assets (or restore offline).

No captures are regenerated. Each changed document is round-trip checked,
including the original DATA bytes and homepage presentation-only continuation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from render import restore_shared_html, shared_asset_html


def publish(paths: list[Path], assets: Path, *, restore: bool = False) -> list[dict]:
    records = []
    for path in paths:
        before = path.read_text()
        offline = restore_shared_html(before, assets)
        after = offline if restore else shared_asset_html(before, assets)
        if restore_shared_html(after, assets) != offline:
            raise RuntimeError(f"lossless replay round-trip failed: {path}")
        if before != after:
            path.write_text(after)
        records.append({"path": str(path), "beforeBytes": len(before.encode()),
                        "afterBytes": len(after.encode()),
                        "standaloneSha256": hashlib.sha256(offline.encode()).hexdigest()})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--restore", action="store_true", help="Restore self-contained HTML instead")
    args = parser.parse_args()
    records = publish(args.paths, args.assets, restore=args.restore)
    print(json.dumps({"documents": len(records), "beforeBytes": sum(x["beforeBytes"] for x in records),
                      "afterBytes": sum(x["afterBytes"] for x in records), "roundTripVerified": True}, indent=2))


if __name__ == "__main__":
    main()
