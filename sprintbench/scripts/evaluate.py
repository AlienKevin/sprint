#!/usr/bin/env python3
# Copyright (c) 2026 QWOP-bench contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Run a policy over the 100 m course and score it.

    python scripts/evaluate.py --checkpoint checkpoints/Isaac-Velocity-Flat-G1-v0.pt \\
        --speeds 0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0 --headless

The policy interface is deliberately narrow: a callable mapping the task's
123-dimensional observation to 37 joint-position targets.  An RSL-RL checkpoint
is rebuilt from its own tensor shapes rather than through the training runner,
so anything that can be saved as a state dict or TorchScript can be entered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Isaac Sim has to be launched before anything from isaaclab is imported, so
# argument parsing happens up here and the real imports come after.
parser = argparse.ArgumentParser(description="G1 100 m sprint benchmark")
parser.add_argument("--checkpoint", required=True, help="policy to evaluate")
parser.add_argument("--speeds", default="0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0",
                    help="comma-separated forward speed commands, one lane each")
parser.add_argument("--distance", type=float, default=100.0, help="course length in metres")
parser.add_argument("--max-seconds", type=float, default=200.0, help="wall clock per attempt")
parser.add_argument("--settle-seconds", type=float, default=1.0,
                    help="stand on the line under a zero command before the clock starts")
parser.add_argument("--out", default="results/run.json")
parser.add_argument("--label", default=None, help="name for this entry")
parser.add_argument("--video", action="store_true", help="record the run (needs cameras)")
parser.add_argument("--video-length", type=int, default=1200, help="video length in control steps")
parser.add_argument("--video-lane", type=int, default=0, help="which lane the camera follows")
parser.add_argument("--trace-out", default=None,
                    help="dump one lane's root pose and joint angles per frame, for offline rendering")
parser.add_argument("--trace-lane", type=int, default=0, help="which lane to trace")
parser.add_argument("--geometry", default="results/collision_geometry.json",
                    help="per-body collision hulls for the non-penetration check")
parser.add_argument("--soft-physics", action="store_true",
                    help="diagnostic: restore the stock solver settings, to measure\n                          what the hardening is actually buying")
parser.add_argument("--show-fall", action="store_true",
                    help="footage only: keep simulating after a fall instead of "
                         "terminating, so the collapse can be watched. Scores from a "
                         "run with this set are not comparable and are not recorded.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- everything below needs the app running ---------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402  (registers the stock task ids)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sprintbench.tasks  # noqa: F401,E402  (registers Isaac-Sprint-100m-G1-v0)
from sprintbench.metrics import STANDING_HOLD_S, evaluate_run, format_table  # noqa: E402
from sprintbench.policy import load_policy  # noqa: E402
from sprintbench.sprint_env_cfg import GATES_M, G1Sprint100mEnvCfg  # noqa: E402


def lowest_point(robot, origins, pts, rad, body):
    """Lowest point of the robot's collision geometry, per environment.

    Falls back to the base height when no geometry was supplied, which keeps the
    column present rather than silently changing the snapshot layout.
    """
    if pts is None:
        return (robot.data.root_pos_w[:, 2] - origins[:, 2]).unsqueeze(1)
    q = robot.data.body_quat_w[:, body]                 # (n, M, 4)
    t = robot.data.body_pos_w[:, body]                  # (n, M, 3)
    world = math_utils.quat_apply(q, pts.expand(q.shape[0], -1, -1)) + t
    z = world[..., 2] - origins[:, 2:3] - rad
    return z.min(dim=1).values.unsqueeze(1)


def main() -> int:
    speeds = tuple(float(s) for s in args.speeds.split(","))
    device = args.device if hasattr(args, "device") else "cuda:0"

    cfg = G1Sprint100mEnvCfg()
    cfg.commands.base_velocity.speeds = speeds
    cfg.scene.num_envs = len(speeds)
    cfg.sim.device = device
    cfg.episode_length_s = args.max_seconds + STANDING_HOLD_S + 5.0

    if args.soft_physics:
        # The control for the hardening.  Without it, a penetration reading of
        # zero is equally consistent with "the physics holds" and "the check
        # stopped measuring".
        cfg.scene.robot.spawn.collision_props = None
        cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity = 1.0
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        print("SOFT-PHYSICS: stock solver settings restored")

    if args.show_fall:
        # The harness normally ends a run the moment the robot goes down —
        # distance at the fall is the metric, and anything after it is a body
        # sliding.  For footage that is exactly the wrong cut, so this removes
        # the fall terminations and lets the collapse finish.  The resulting
        # scores mean nothing and the caller is expected to discard them.
        cfg.terminations.base_contact = None
        cfg.terminations.collapsed = None
        print("SHOW-FALL: fall terminations disabled, scores from this run are void")

    if args.video:
        # A chase camera locked to one lane's robot: a fixed world camera loses
        # the runner within a couple of seconds over a 100 m course.
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.env_index = args.video_lane
        cfg.viewer.eye = (-4.0, -3.0, 1.8)
        cfg.viewer.lookat = (0.0, 0.0, 0.6)
        cfg.viewer.resolution = (1280, 720)

    env = gym.make("Isaac-Sprint-100m-G1-v0", cfg=cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        video_dir = os.path.join(os.path.dirname(args.out) or ".", "videos")
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_dir, step_trigger=lambda s: s == 0,
            video_length=args.video_length, disable_logger=True,
        )

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contacts = unwrapped.scene["contact_forces"]
    command = unwrapped.command_manager.get_term("base_velocity")
    dt = unwrapped.step_dt
    n = cfg.scene.num_envs

    # --- exact non-penetration, over every body ------------------------------
    # The lowest point of body i at pose (R, t) is min over its collision points
    # of (R·p + t)_z − r.  Loading the geometry once and flattening it into a
    # single point cloud makes the per-step check one batched rotation, so the
    # whole robot costs about as much as the two-foot check it replaces.
    geom_pts = geom_rad = geom_body = None
    if args.geometry and os.path.exists(args.geometry):
        geom = json.load(open(args.geometry))
        pts, rads, bidx = [], [], []
        for name, entry in geom["bodies"].items():
            bid = robot.body_names.index(name) if name in robot.body_names else None
            if bid is None:
                continue
            pts += entry["points"]
            rads += entry["radii"]
            bidx += [bid] * len(entry["points"])
        geom_pts = torch.tensor(pts, dtype=torch.float32, device=device)
        geom_rad = torch.tensor(rads, dtype=torch.float32, device=device)
        geom_body = torch.tensor(bidx, dtype=torch.long, device=device)
        print(f"non-penetration: {len(geom['bodies'])} bodies, {len(pts)} collision points"
              f"; no geometry for {geom['no_collision']}")
    else:
        print("WARNING: no collision geometry; penetration will be checked on the feet only")

    foot_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
    contact_foot_ids, _ = contacts.find_bodies(".*_ankle_roll_link")
    # the snapshot below packs fixed columns, so a different foot count would
    # silently shift every channel
    assert len(foot_ids) == 2 and len(contact_foot_ids) == 2, f"expected two feet, got {foot_names}"
    print(f"feet: {foot_names}  control dt: {dt:.4f}s  envs: {n}")

    obs, _ = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    policy = load_policy(args.checkpoint, obs_t.shape[-1], robot.num_joints, device)
    print(f"policy: {args.checkpoint}  obs {obs_t.shape[-1]} -> act {robot.num_joints}")

    origins = unwrapped.scene.env_origins
    all_ids = torch.arange(n, device=device)

    # The robot is spawned at a nominal standing pose a centimetre or two clear
    # of the plane and drops onto it.  Timing from that instant charges the
    # policy for the drop, and — worse — makes the settle look like the feet
    # sinking through the floor, because foot heights are scored against where
    # they started.  Stand on the line under a zero command first, then start
    # the clock: on the line, at rest, settled.
    # The settle also calibrates the foot-height reference used to score ground
    # penetration.  A single frame will not do: a foot that happens to be
    # tilted at that instant reads high, and every later frame where it lies
    # flat then scores as the foot sinking.  Both feet share one geometry, so
    # take the lowest height either foot reaches across the whole settle — that
    # is the height of the link when its sole is flat on the plane.
    command.hold(all_ids)
    foot_ref = torch.full((n,), float("inf"), device=device)
    for _ in range(int(args.settle_seconds / dt)):
        with torch.inference_mode():
            obs, *_ = env.step(policy(obs_t))
        obs_t = obs["policy"] if isinstance(obs, dict) else obs
        heights = robot.data.body_pos_w[:, foot_ids, 2] - origins[:, 2:3]
        foot_ref = torch.minimum(foot_ref, heights.min(dim=1).values)
    command.release(all_ids)
    start_x = (robot.data.root_pos_w - origins)[:, 0].clone()
    foot_ref_l = foot_ref.tolist()
    print(f"settled after {args.settle_seconds:g}s; "
          f"base height {robot.data.root_pos_w[:, 2].mean() - origins[:, 2].mean():.4f} m; "
          f"foot reference {min(foot_ref_l):.4f}–{max(foot_ref_l):.4f} m")

    # per-step traces, kept on the CPU as plain lists so scoring never has to
    # think about which device a number came from
    trace: dict[str, list] = {k: [] for k in ("t", "x", "y", "z", "vx", "tilt", "fz", "fc", "low")}
    # Kit cannot create a render device in this container, so the rollout is
    # recorded as state and drawn afterwards.  Kept on the GPU and transferred
    # once at the end rather than every step.
    pose_rows: list[torch.Tensor] = []
    lane = args.trace_lane
    finish_time = [None] * n
    fell_at = [None] * n
    stood = [None] * n
    resolved = [False] * n

    t = 0.0
    steps = int((args.max_seconds + STANDING_HOLD_S + 2.0) / dt)
    for _ in range(steps):
        if not simulation_app.is_running():
            break

        # Read the state the policy is about to act on.  Isaac Lab resets a
        # terminated environment inside step(), so anything read afterwards can
        # be the reset pose rather than the pose that ended the run.
        pos = robot.data.root_pos_w - origins
        pos[:, 0] -= start_x            # distance from the line, not from spawn
        grav_z = robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
        snapshot = torch.cat([
            pos,                                                        # x y z
            robot.data.root_lin_vel_b[:, :1],                           # forward speed
            torch.rad2deg(torch.acos(-grav_z)).unsqueeze(1),            # torso lean
            robot.data.body_pos_w[:, foot_ids, 2] - origins[:, 2:3],    # foot heights
            contacts.data.net_forces_w[:, contact_foot_ids, :].norm(dim=-1),
            lowest_point(robot, origins, geom_pts, geom_rad, geom_body),
        ], dim=1)
        # one device-to-host transfer per control step: doing it per channel
        # costs six synchronisation points and dominates the run time
        rows = snapshot.tolist()
        if args.trace_out:
            pose_rows.append(torch.cat([
                pos[lane], robot.data.root_quat_w[lane], robot.data.joint_pos[lane],
                # foot heights and contact, so penetration can be judged inside
                # this simulator rather than against a re-modelled robot whose
                # link frames need not agree with this one's
                snapshot[lane, 5:9],
            ]).clone())

        trace["t"].append(t)
        trace["x"].append([r[0] for r in rows])
        trace["y"].append([r[1] for r in rows])
        trace["z"].append([r[2] for r in rows])
        trace["vx"].append([r[3] for r in rows])
        trace["tilt"].append([r[4] for r in rows])
        trace["fz"].append([(r[5], r[6]) for r in rows])
        trace["fc"].append([(r[7] > 1.0, r[8] > 1.0) for r in rows])
        trace["low"].append([r[9] for r in rows])

        with torch.inference_mode():
            action = policy(obs_t)
            obs, _, terminated, truncated, _ = env.step(action)
        obs_t = obs["policy"] if isinstance(obs, dict) else obs
        t += dt

        xs, zs = trace["x"][-1], trace["z"][-1]
        for i in range(n):
            if resolved[i]:
                continue
            if finish_time[i] is None and xs[i] >= args.distance:
                finish_time[i] = t
                command.hold(torch.tensor([i], device=device))
            # Going down is now an observation, not an outcome.  The environment
            # no longer terminates on it — a gait that keeps its base low is
            # allowed — so read it from the recorded height rather than from a
            # termination that will never fire.
            if fell_at[i] is None and zs[i] < 0.4:
                fell_at[i] = t
            if finish_time[i] is not None and t - finish_time[i] >= STANDING_HOLD_S:
                stood[i] = zs[i] >= 0.4
                resolved[i] = True

        if all(resolved):
            print(f"all lanes resolved at {t:.1f}s")
            break

    # ---- score -------------------------------------------------------------
    results = []
    for i in range(n):
        results.append(evaluate_run(
            env_id=i,
            commanded_speed=speeds[i],
            t=trace["t"],
            x=[row[i] for row in trace["x"]],
            y=[row[i] for row in trace["y"]],
            vx=[row[i] for row in trace["vx"]],
            tilt_deg=[row[i] for row in trace["tilt"]],
            foot_z=[(row[i][0], row[i][1]) for row in trace["fz"]],
            foot_contact=[(row[i][0], row[i][1]) for row in trace["fc"]],
            gates=GATES_M,
            fell_at_s=fell_at[i],
            stood_after_finish=stood[i],
            finish_distance_m=args.distance,
            foot_reference_m=foot_ref_l[i],
            lowest_point_m=[row[i] for row in trace["low"]],
        ))

    print()
    print(format_table(results))
    print()
    for r in results:
        failed = [c for c in r.checks if not c.passed]
        if failed:
            print(f"  cmd {r.commanded_speed:.2f} m/s failed: "
                  + "; ".join(f"{c.name} ({c.detail})" for c in failed))

    doc = {
        "label": args.label or os.path.basename(args.checkpoint),
        "checkpoint": args.checkpoint,
        "task": "Isaac-Sprint-100m-G1-v0",
        "distance_m": args.distance,
        "control_hz": round(1.0 / dt, 2),
        "sim_dt": cfg.sim.dt,
        "decimation": cfg.decimation,
        "num_joints": robot.num_joints,
        "obs_dim": int(obs_t.shape[-1]),
        "runs": [r.to_dict() for r in results],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nwrote {args.out}")

    if args.trace_out and pose_rows:
        frames = torch.stack(pose_rows).cpu()
        trace_doc = {
            "source": f"{doc['label']} lane {lane}",
            "commanded_speed": speeds[lane],
            "fps": round(1.0 / dt, 3),
            "joint_names": list(robot.joint_names),
            "layout": (["x", "y", "z", "qw", "qx", "qy", "qz"] + list(robot.joint_names)
                       + ["foot_z_l", "foot_z_r", "foot_f_l", "foot_f_r"]),
            "frames": [[round(v, 5) for v in row] for row in frames.tolist()],
        }
        os.makedirs(os.path.dirname(args.trace_out) or ".", exist_ok=True)
        with open(args.trace_out, "w") as f:
            json.dump(trace_doc, f, separators=(",", ":"))
        print(f"wrote {args.trace_out}: {len(trace_doc['frames'])} frames, "
              f"{len(robot.joint_names)} joints")

    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
