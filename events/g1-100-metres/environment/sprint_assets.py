#!/usr/bin/env python3
"""Redirect Isaac's nucleus asset root to the image-local offline mirror."""

from __future__ import annotations

import os
import sys

LOCAL_ASSET_ROOT = "/opt/assets"
ASSET_ROOT_SETTING = "/persistent/isaac/asset_root/cloud"


def remote_asset_root() -> str:
    import carb

    return str(carb.settings.get_settings().get(ASSET_ROOT_SETTING) or "")


def use_local_assets(root: str = LOCAL_ASSET_ROOT) -> bool:
    if not os.path.isdir(root):
        return False
    import carb

    carb.settings.get_settings().set(ASSET_ROOT_SETTING, root)
    return True


def localize_asset_path(path: str, root: str = LOCAL_ASSET_ROOT) -> str:
    """Map an official Isaac asset URL to the mirror when that file exists."""
    for marker in ("/Isaac/", "/NVIDIA/"):
        index = path.rfind(marker)
        if index < 0:
            continue
        candidate = root.rstrip("/") + path[index:]
        if os.path.exists(candidate):
            return candidate
    return path


def install_runtime_asset_redirect(root: str = LOCAL_ASSET_ROOT) -> None:
    """Patch both future config construction and already-built USD configs."""
    if not use_local_assets(root):
        raise RuntimeError(f"missing sealed Isaac asset mirror: {root}")

    import isaaclab.utils.assets as assets

    values = {
        "NUCLEUS_ASSET_ROOT_DIR": root,
        "NVIDIA_NUCLEUS_DIR": f"{root}/NVIDIA",
        "ISAAC_NUCLEUS_DIR": f"{root}/Isaac",
        "ISAACLAB_NUCLEUS_DIR": f"{root}/Isaac/IsaacLab",
    }
    for name, value in values.items():
        setattr(assets, name, value)
    # AppLauncher may already have imported modules that copied these constants
    # with ``from isaaclab.utils.assets import ...``. Refresh those globals too.
    for module in tuple(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not namespace:
            continue
        for name, value in values.items():
            if name in namespace:
                namespace[name] = value

    # Config objects created before the refresh retain their original URL.
    # Every UsdFileCfg ultimately calls this helper, so localize at the final
    # trusted spawn boundary as a fail-safe without mutating agent code.
    from isaaclab.sim.spawners.from_files import from_files

    original = from_files._spawn_from_usd_file
    if not getattr(original, "_sprint_local_asset_redirect", False):

        def local_spawn(prim_path, usd_path, *args, **kwargs):
            return original(
                prim_path,
                localize_asset_path(str(usd_path), root),
                *args,
                **kwargs,
            )

        local_spawn._sprint_local_asset_redirect = True
        from_files._spawn_from_usd_file = local_spawn

    # GroundPlaneCfg and some agent-authored scenes bypass UsdFileCfg and call
    # add_usd_reference directly. Patch that lower boundary as well, then
    # replace any copies imported into already-loaded modules.
    from isaaclab.sim.utils import prims

    original_reference = prims.add_usd_reference
    if getattr(original_reference, "_sprint_local_asset_redirect", False):
        return

    def local_reference(prim_path, usd_path, *args, **kwargs):
        return original_reference(
            prim_path,
            localize_asset_path(str(usd_path), root),
            *args,
            **kwargs,
        )

    local_reference._sprint_local_asset_redirect = True
    prims.add_usd_reference = local_reference
    for module in tuple(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if namespace and namespace.get("add_usd_reference") is original_reference:
            namespace["add_usd_reference"] = local_reference
