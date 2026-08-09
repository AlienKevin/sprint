#!/usr/bin/env python3
"""Run an agent-authored Isaac script against the sealed local asset mirror."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, "/opt")

from sprint_assets import (
    LOCAL_ASSET_ROOT,
    install_runtime_asset_redirect,
    use_local_assets,
)


def install_app_launcher_hook() -> None:
    from isaaclab.app import AppLauncher

    original = AppLauncher.__init__

    def local_asset_init(self, *args, **kwargs) -> None:
        original(self, *args, **kwargs)
        install_runtime_asset_redirect()

    AppLauncher.__init__ = local_asset_init


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: sprint-isaac-bootstrap.py SCRIPT [ARGS...]", file=sys.stderr)
        return 2
    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        print(f"Isaac script not found: {script}", file=sys.stderr)
        return 2
    os.environ["SPRINT_LOCAL_ASSET_ROOT"] = LOCAL_ASSET_ROOT
    # Some Kit builds expose carb settings before SimulationApp exists. Apply
    # eagerly when possible, then apply again after AppLauncher initializes.
    try:
        use_local_assets()
    except Exception:  # noqa: BLE001
        pass
    install_app_launcher_hook()
    sys.argv = [str(script), *sys.argv[2:]]
    # Match ``python3 /path/to/script.py``: runpy does not automatically put
    # the script directory at sys.path[0], but agent training scripts commonly
    # import sibling packages from their workspace.
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.path.pop(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
