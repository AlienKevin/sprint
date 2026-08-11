#!/usr/bin/env python3
"""Run an agent-authored Isaac script against the sealed local asset mirror."""

from __future__ import annotations

import json
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


APP_LAUNCHER_STATE_ENV = "SPRINT_APP_LAUNCHER_STATE_FILE"


def _write_app_launcher_state(state: str, **detail: object) -> None:
    raw_path = os.environ.get(APP_LAUNCHER_STATE_ENV, "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    payload = {"schema_version": 1, "state": state, **detail}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        os.replace(temporary, path)
    except OSError:
        # This marker only improves provider recovery. Never mask the agent's
        # actual launcher result because a best-effort /tmp write failed.
        return


def install_app_launcher_hook() -> None:
    from isaaclab.app import AppLauncher

    original = AppLauncher.__init__

    def local_asset_init(self, *args, **kwargs) -> None:
        _write_app_launcher_state("starting")
        try:
            original(self, *args, **kwargs)
        except SystemExit as exc:
            _write_app_launcher_state("system_exit", exit_code=exc.code)
            raise
        except BaseException as exc:
            _write_app_launcher_state("exception", error_type=type(exc).__name__)
            raise
        _write_app_launcher_state("completed")
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
    # Match normal workspace execution while being slightly more permissive:
    # runpy does not add either location automatically, and agent-authored
    # scripts may import a sibling module (script directory) or a workspace
    # package such as ``train.robot`` (working directory).
    import_roots = list(dict.fromkeys((str(script.parent), str(Path.cwd().resolve()))))
    for path in reversed(import_roots):
        sys.path.insert(0, path)
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        del sys.path[: len(import_roots)]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
