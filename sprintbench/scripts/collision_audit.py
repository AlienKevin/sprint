#!/usr/bin/env python3
# Copyright (c) 2026 QWOP-bench contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Which parts of this robot can actually touch the ground?

``G1_MINIMAL_CFG`` is documented as removing "most collision meshes to speed up
simulation", and the sprint environment inherits it from Isaac Lab's velocity
task.  That is a reasonable trade for a walking policy whose feet are the only
things expected to touch anything.  It is not obviously safe for a benchmark
that permits any gait, because a body with no collision geometry does not rest
on the floor — it passes through it.

This drives the robot into the ground and records, per body, the lowest height
it reached and the largest contact force it ever registered.  A body that goes
below the plane while never registering a contact has no collision geometry, and
anything a policy does with that body is unphysical.

    python scripts/collision_audit.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--out", default="/results/collision_audit.json")
parser.add_argument("--asset", default="minimal", choices=["minimal", "full"],
                    help="g1_minimal.usd (Isaac Lab default) or g1.usd")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sprintbench.tasks  # noqa: F401,E402
from sprintbench.sprint_env_cfg import G1Sprint100mEnvCfg  # noqa: E402


def main() -> int:
    cfg = G1Sprint100mEnvCfg()
    if args.asset == 'full':
        from isaaclab_assets import G1_CFG
        cfg.scene.robot = G1_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    cfg.scene.num_envs = 1
    cfg.commands.base_velocity.speeds = (0.0,)
    cfg.episode_length_s = args.seconds + 5.0
    env = gym.make("Isaac-Sprint-100m-G1-v0", cfg=cfg)

    robot = env.unwrapped.scene["robot"]
    contacts = env.unwrapped.scene["contact_forces"]
    origins = env.unwrapped.scene.env_origins
    names = list(robot.body_names)
    # The contact sensor keeps its own body ordering, which is not the
    # articulation's.  Indexing one array with the other's order silently
    # attributes every force to the wrong link — it reported zero load on the
    # feet, which carry the entire robot.
    sensor_names = list(contacts.body_names)
    sensor_index = [sensor_names.index(n) if n in sensor_names else -1 for n in names]
    missing = [n for n in names if n not in sensor_names]
    print(f"{len(names)} articulation bodies, {len(sensor_names)} sensed"
          + (f"; not sensed: {missing}" if missing else ""))

    env.reset()
    n_act = robot.num_joints
    device = robot.device

    lowest = torch.full((len(names),), float("inf"), device=device)
    peak_force = torch.zeros(len(names), device=device)

    # Go limp.  With zero position targets the PD pulls every joint to its
    # default pose and the robot topples, which is exactly the state that puts
    # knees, elbows and torso against the floor.
    steps = int(args.seconds / env.unwrapped.step_dt)
    for _ in range(steps):
        with torch.inference_mode():
            env.step(torch.zeros((1, n_act), device=device))
        z = robot.data.body_pos_w[0, :, 2] - origins[0, 2]
        lowest = torch.minimum(lowest, z)
        sensed = contacts.data.net_forces_w[0].norm(dim=-1)
        f = torch.stack([sensed[i] if i >= 0 else torch.zeros((), device=device)
                         for i in sensor_index])
        peak_force = torch.maximum(peak_force, f)

    lo = lowest.tolist()
    pf = peak_force.tolist()
    rows = sorted(zip(names, lo, pf), key=lambda r: r[1])

    colliding = [r for r in rows if r[2] > 1.0]
    ghosts = [r for r in rows if r[2] <= 1.0 and r[1] < 0.0]

    print(f"{'body':<34}{'lowest z (m)':>14}{'peak contact (N)':>18}")
    print("-" * 66)
    for name, z, f in rows:
        flag = "  <- through the floor, never collided" if (f <= 1.0 and z < 0.0) else ""
        print(f"{name:<34}{z:>14.4f}{f:>18.1f}{flag}")

    print(f"\n{len(colliding)} of {len(names)} bodies ever registered contact")
    print(f"{len(ghosts)} bodies passed below the ground plane without colliding")

    doc = {
        "bodies": [{"name": n, "lowest_z_m": round(z, 5), "peak_contact_n": round(f, 2)}
                   for n, z, f in rows],
        "colliding": [r[0] for r in colliding],
        "ghosts": [r[0] for r in ghosts],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(doc, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    sys.exit(code)
