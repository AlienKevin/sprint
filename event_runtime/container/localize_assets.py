#!/usr/bin/env python3
"""Mirror every stock Isaac asset class used by Sprint training scenes."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/opt")

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher({"device": "cpu", "headless": True}).app

import omni.client  # noqa: E402

from sprint_assets import LOCAL_ASSET_ROOT, remote_asset_root  # noqa: E402

MIRRORED = (
    "Isaac/IsaacLab/Robots/Unitree/G1",
    "Isaac/Props/UIElements",
    "Isaac/Environments/Grid",
    "Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned",
    "Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
)


def main() -> int:
    root = remote_asset_root()
    if not root:
        print("ERROR: Isaac reports no remote asset root")
        return 1
    for relative in MIRRORED:
        source = f"{root}/{relative}"
        target = os.path.join(LOCAL_ASSET_ROOT, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        result = omni.client.copy(
            source, target, behavior=omni.client.CopyBehavior.OVERWRITE
        )
        if result != omni.client.Result.OK:
            print(f"ERROR: copying {source} -> {target} returned {result}")
            return 1
        size = (
            os.path.getsize(target)
            if os.path.isfile(target)
            else sum(
                os.path.getsize(os.path.join(directory, filename))
                for directory, _, filenames in os.walk(target)
                for filename in filenames
            )
        )
        if size <= 0:
            print(f"ERROR: {target} is empty")
            return 1
        print(f"mirrored {relative} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    code = main()
    # Isaac Sim can segfault while tearing down a CPU-only build container
    # after every requested asset has already been copied successfully.  Image
    # construction has no live simulator state to preserve, so flush the
    # observable result and exit without invoking the unstable Kit destructor.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
