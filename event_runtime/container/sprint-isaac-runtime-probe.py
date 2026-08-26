#!/usr/bin/env python3
"""Trusted admission probe for one exact Isaac training allocation."""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab_assets import G1_CFG  # noqa: E402
from sprint_assets import localize_asset_path  # noqa: E402


def assert_local(path: str) -> None:
    if not path.startswith("/opt/assets/") or not os.path.exists(path):
        raise RuntimeError(f"asset did not resolve to the sealed mirror: {path}")


def main() -> int:
    # AppLauncher can return after a provider-side Vulkan failure.  One empty
    # CUDA physics step proves that this allocation can execute the event's
    # runtime before a benchmark lane is admitted.
    simulation = SimulationContext(SimulationCfg(device=str(args.device)))
    simulation.reset()
    simulation.step()
    print("LOCAL_SIMULATION=ok")

    local_g1 = localize_asset_path(G1_CFG.spawn.usd_path)
    assert_local(local_g1)
    for marker in BLUE_ARROW_X_MARKER_CFG.markers.values():
        usd_path = getattr(marker, "usd_path", None)
        if usd_path:
            assert_local(localize_asset_path(usd_path))
    print(f"LOCAL_G1={local_g1}")
    print("LOCAL_DEBUG_MARKERS=ok")
    return 0


if __name__ == "__main__":
    code = main()
    # The probe process is disposable; avoid Isaac's headless teardown path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
