#!/usr/bin/env python3
"""Warm-up proof that stock G1 and debug markers resolve without network."""

from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args()).app

from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG  # noqa: E402
from isaaclab_assets import G1_CFG  # noqa: E402


def assert_local(path: str) -> None:
    if not path.startswith("/opt/assets/") or not os.path.exists(path):
        raise RuntimeError(f"asset did not resolve to the sealed mirror: {path}")


def main() -> int:
    assert_local(G1_CFG.spawn.usd_path)
    for marker in BLUE_ARROW_X_MARKER_CFG.markers.values():
        usd_path = getattr(marker, "usd_path", None)
        if usd_path:
            assert_local(usd_path)
    print(f"LOCAL_G1={G1_CFG.spawn.usd_path}")
    print("LOCAL_DEBUG_MARKERS=ok")
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)
