#!/usr/bin/env python3
# Copyright (c) 2026 QWOP-bench contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Dump each body's collision geometry in its own frame.

Checking that nothing penetrates the floor needs the lowest point of a body's
collision surface, not the height of its origin.  Those differ by whatever the
geometry happens to be — the G1's ankle origin sits 3.4 cm above its sole — and
the difference changes with orientation, so it cannot be a per-body constant.

What *is* constant is the geometry in the body's own frame.  This walks the
robot prim, collects the collision meshes' vertices per body, reduces each to a
convex hull so the per-frame check stays cheap, and writes them out.  Then the
lowest point of body *i* at any pose is

    min over hull vertices v of  (R_i · v + p_i)_z

which is exact for convex shapes and conservative for the concave ones, in the
direction that reports penetration rather than hiding it.

    python scripts/extract_collision_geometry.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/results/collision_geometry.json")
parser.add_argument("--max-vertices", type=int, default=64,
                    help="hull vertices kept per body")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sprintbench.tasks  # noqa: F401,E402
from sprintbench.sprint_env_cfg import G1Sprint100mEnvCfg  # noqa: E402


def local_transform(prim, root) -> np.ndarray:
    """4x4 transform of ``prim`` expressed in ``root``'s frame."""
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_p = np.array(cache.GetLocalToWorldTransform(prim)).T
    world_r = np.array(cache.GetLocalToWorldTransform(root)).T
    return np.linalg.inv(world_r) @ world_p


def shape_points(prim):
    """Represent a collision shape as points carrying a radius.

    The lowest point of a shape at pose (R, t) is then

        min over i of  (R · p_i + t)_z − r_i

    which is exact for spheres, capsules and meshes, and over-approximates a
    cylinder by its circumscribing capsule — an error in the direction that
    reports penetration rather than hiding it.

    Most of the G1's collision shapes are primitives rather than meshes; walking
    only ``UsdGeom.Mesh`` found two of them, both feet, and missed the knees,
    head and fingertips that the contact audit had already shown to collide.
    """
    if UsdGeom.Mesh(prim):
        pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], dtype=np.float64)
        return (pts, 0.0) if len(pts) else None

    if UsdGeom.Sphere(prim):
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 0.0)
        return np.zeros((1, 3)), r

    if UsdGeom.Capsule(prim):
        cap = UsdGeom.Capsule(prim)
        r = float(cap.GetRadiusAttr().Get() or 0.0)
        h = float(cap.GetHeightAttr().Get() or 0.0)
        axis = {"X": 0, "Y": 1, "Z": 2}[str(cap.GetAxisAttr().Get() or "Z")]
        ends = np.zeros((2, 3))
        ends[0, axis], ends[1, axis] = -h / 2.0, h / 2.0
        return ends, r

    if UsdGeom.Cylinder(prim):
        cyl = UsdGeom.Cylinder(prim)
        r = float(cyl.GetRadiusAttr().Get() or 0.0)
        h = float(cyl.GetHeightAttr().Get() or 0.0)
        axis = {"X": 0, "Y": 1, "Z": 2}[str(cyl.GetAxisAttr().Get() or "Z")]
        ends = np.zeros((2, 3))
        ends[0, axis], ends[1, axis] = -h / 2.0, h / 2.0
        return ends, r          # circumscribing capsule: conservative

    if UsdGeom.Cube(prim):
        s = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 0.0) / 2.0
        corners = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)])
        return corners, 0.0

    return None


def hull(points: np.ndarray, budget: int) -> np.ndarray:
    """Reduce a vertex cloud, keeping the extremes that decide the lowest point."""
    if len(points) <= budget:
        return points
    try:
        from scipy.spatial import ConvexHull
        points = points[ConvexHull(points).vertices]
    except Exception:  # noqa: BLE001  — scipy absent or degenerate geometry
        pass
    if len(points) <= budget:
        return points
    # Keep the extreme points along a spread of directions: whichever vertex is
    # lowest at some orientation is extreme along that direction, so sampling
    # directions retains the ones that can ever matter.
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(budget, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    keep = {int(np.argmin(points @ d)) for d in dirs}
    keep |= {int(np.argmin(points[:, 2])), int(np.argmax(points[:, 2]))}
    return points[sorted(keep)]


def main() -> int:
    cfg = G1Sprint100mEnvCfg()
    cfg.scene.num_envs = 1
    cfg.commands.base_velocity.speeds = (0.0,)
    env = gym.make("Isaac-Sprint-100m-G1-v0", cfg=cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    stage = env.unwrapped.sim.stage
    # By the time the scene is built the configured path has already been
    # resolved to its regex form (``/World/envs/env_.*/Robot``), which is not a
    # valid SdfPath.  Ask the scene for the concrete one rather than trying to
    # reconstruct it.
    env_root = env.unwrapped.scene.env_prim_paths[0]
    root_path = f"{env_root}/{robot.cfg.prim_path.rstrip('/').rsplit('/', 1)[-1]}"
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise SystemExit(f"robot prim not found at {root_path} "
                         f"(cfg said {robot.cfg.prim_path})")

    # map every collision mesh to the rigid body that owns it
    bodies: dict[str, list[tuple[np.ndarray, float]]] = {n: [] for n in robot.body_names}
    unsupported: set = set()
    # Isaac Lab clones environments with replicate_physics, so the robot is an
    # instanced prim and a default PrimRange stops at the instance boundary
    # without ever entering the prototype where the collision shapes live.  That
    # is why a mesh-only walk found two shapes on a robot the contact audit had
    # already shown collides on seven bodies.
    traverse = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

    body_prims = {}
    for prim in Usd.PrimRange(root, traverse):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_prims[prim.GetName()] = prim
    print(f"found {len(body_prims)} rigid-body prims under {root_path}")

    n_meshes = 0
    for prim in Usd.PrimRange(root, traverse):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        shape = shape_points(prim)
        if shape is None:
            unsupported.add(prim.GetTypeName())
            continue
        pts, radius = shape
        owner = None
        walk = prim
        while walk and walk.IsValid():
            if walk.GetName() in body_prims:
                owner = walk.GetName()
                break
            walk = walk.GetParent()
        if owner is None or owner not in bodies:
            continue
        T = local_transform(prim, body_prims[owner])
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        bodies[owner].append(((pts_h @ T.T)[:, :3], radius))
        n_meshes += 1

    doc, empty = {}, []
    for name, chunks in bodies.items():
        if not chunks:
            empty.append(name)
            continue
        pts = np.vstack([c[0] for c in chunks])
        rad = np.concatenate([np.full(len(c[0]), c[1]) for c in chunks])
        if len(pts) > args.max_vertices:
            # keep the widest radius per retained point: shrinking a sphere
            # would under-report how low it can reach
            keep = np.argsort(-(rad - pts[:, 2]))[:args.max_vertices]
            pts, rad = pts[keep], rad[keep]
        doc[name] = {"points": [[round(float(c), 5) for c in p] for p in pts],
                     "radii": [round(float(r), 5) for r in rad]}

    print(f"{n_meshes} collision shapes over {len(doc)} bodies"
          + (f"; unsupported prim types skipped: {sorted(unsupported)}" if unsupported else ""))
    print(f"{len(empty)} bodies with no collision geometry: {empty}")

    def reach(entry):
        return min(p[2] - r for p, r in zip(entry["points"], entry["radii"]))

    for name, v in sorted(doc.items(), key=lambda kv: reach(kv[1]))[:8]:
        print(f"  {name:<28} {len(v['points']):>3} pts, lowest local z {reach(v):+.4f}")

    out = {"bodies": doc, "no_collision": empty}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    sys.exit(code)
