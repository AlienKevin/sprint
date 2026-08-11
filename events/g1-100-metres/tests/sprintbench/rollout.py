# Copyright (c) 2026 Sprint contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Run one time trial and score it.

This is the single implementation of the trial, shared by the local harness and
by the challenge verifier.  Keeping one copy is not tidiness: the verifier's
whole claim is that it measures the same thing the agent measured, and two
rollout loops that drift apart quietly break that.

The protocol, in order:

1. **Settle.**  Stand on the line under a zero command.  The G1 spawns a couple
   of centimetres clear of the plane and drops; timing from that instant charges
   the policy for simulator initialization.
2. **Release.**  Hand back the commanded speed and start the clock.  Distance is
   measured from here, not from the spawn.
3. **Record.**  One compact device-to-host transfer per control step containing
   only forward position, lateral position, speed, and the self-collision gate.
"""

from __future__ import annotations

import torch

from .metrics import evaluate_run

# Unitree G1 kinematic parents for bodies that carry collision geometry.  Used
# only to skip ancestor/descendant pairs within SELF_COLLISION_ANCESTRY hops so
# a hip capsule resting against its own knee is not scored as self-intersection.
# Cross-branch pairs (left vs right, arm vs torso, foot vs opposite thigh) stay.
G1_BODY_PARENT: dict[str, str | None] = {
    "pelvis": None,
    "pelvis_contour_link": "pelvis",
    "imu_link": "torso_link",
    "left_hip_pitch_link": "pelvis",
    "right_hip_pitch_link": "pelvis",
    "torso_link": "pelvis",
    "left_hip_roll_link": "left_hip_pitch_link",
    "right_hip_roll_link": "right_hip_pitch_link",
    "head_link": "torso_link",
    "logo_link": "torso_link",
    "left_shoulder_pitch_link": "torso_link",
    "right_shoulder_pitch_link": "torso_link",
    "left_hip_yaw_link": "left_hip_roll_link",
    "right_hip_yaw_link": "right_hip_roll_link",
    "left_shoulder_roll_link": "left_shoulder_pitch_link",
    "right_shoulder_roll_link": "right_shoulder_pitch_link",
    "left_knee_link": "left_hip_yaw_link",
    "right_knee_link": "right_hip_yaw_link",
    "left_shoulder_yaw_link": "left_shoulder_roll_link",
    "right_shoulder_yaw_link": "right_shoulder_roll_link",
    "left_ankle_pitch_link": "left_knee_link",
    "right_ankle_pitch_link": "right_knee_link",
    "left_elbow_pitch_link": "left_shoulder_yaw_link",
    "right_elbow_pitch_link": "right_shoulder_yaw_link",
    "left_ankle_roll_link": "left_ankle_pitch_link",
    "right_ankle_roll_link": "right_ankle_pitch_link",
    "left_elbow_roll_link": "left_elbow_pitch_link",
    "right_elbow_roll_link": "right_elbow_pitch_link",
    "left_palm_link": "left_elbow_roll_link",
    "right_palm_link": "right_elbow_roll_link",
    "left_zero_link": "left_palm_link",
    "right_zero_link": "right_palm_link",
    "left_one_link": "left_palm_link",
    "right_one_link": "right_palm_link",
    "left_two_link": "left_one_link",
    "right_two_link": "right_one_link",
    "left_three_link": "left_palm_link",
    "right_three_link": "right_palm_link",
    "left_four_link": "left_three_link",
    "right_four_link": "right_three_link",
    "left_five_link": "left_palm_link",
    "right_five_link": "right_palm_link",
    "left_six_link": "left_five_link",
    "right_six_link": "right_five_link",
}
SELF_COLLISION_ANCESTRY = 3
# Sphere pairs closer than this (metres of bounding-sphere overlap margin) get a
# point-wise check.  Keeps the per-step cost off the O(bodies² × points²) path.
SELF_COLLISION_SPHERE_MARGIN_M = 0.02
# Stock collision primitives are thinner than the visual limbs (knee capsules
# are 1.5 cm radius).  Without a pad the gate only fires on co-located limbs
# and misses ankle-through-torso near-misses that still look like body
# intersection.  2 cm was calibrated on representative lane captures: a valid
# 17.9 s gait stays under the 1 cm DQ threshold; the 6.9–7.4 s tight gaits do
# not.  3 cm false-DQs the clean gait.
SELF_COLLISION_RADIUS_PAD_M = 0.02


def _is_chain_neighbor(a: str, b: str, max_dist: int = SELF_COLLISION_ANCESTRY) -> bool:
    for start, other in ((a, b), (b, a)):
        x, d = start, 0
        while x is not None and d <= max_dist:
            if x == other:
                return True
            x = G1_BODY_PARENT.get(x)
            d += 1
    return False


def _is_sibling(a: str, b: str) -> bool:
    pa, pb = G1_BODY_PARENT.get(a), G1_BODY_PARENT.get(b)
    return pa is not None and pa == pb


def _is_digit_link(name: str) -> bool:
    # G1 finger links are noisy near the hip/thigh during arm swing; the palm
    # still represents the hand for self-collision.  Digits are excluded so a
    # grazing fingertip cannot DQ an otherwise clean gait.
    return any(f"_{d}_link" in name for d in
               ("zero", "one", "two", "three", "four", "five", "six"))


def load_geometry(path: str, robot, device: str):
    """Flatten the per-body collision hulls into one point cloud.

    Returns ``(points, radii, body_index)`` or ``(None, None, None)``.  The
    lowest point of the robot is then a single batched rotation rather than a
    loop over bodies.
    """
    import json
    import os

    if not path or not os.path.exists(path):
        return None, None, None
    geom = json.load(open(path))
    pts, rads, bidx = [], [], []
    for name, entry in geom["bodies"].items():
        if name not in robot.body_names:
            continue
        bid = robot.body_names.index(name)
        pts += entry["points"]
        rads += entry["radii"]
        bidx += [bid] * len(entry["points"])
    if not pts:
        return None, None, None
    return (torch.tensor(pts, dtype=torch.float32, device=device),
            torch.tensor(rads, dtype=torch.float32, device=device),
            torch.tensor(bidx, dtype=torch.long, device=device))


def prepare_self_collision(pts, rad, body, body_names, device: str):
    """Build the pair list and per-body local bounding spheres for the gate."""
    if pts is None:
        return None
    # Inflate radii for the self-collision check only (ground penetration keeps
    # the raw geometry).  See SELF_COLLISION_RADIUS_PAD_M.
    rad = rad + SELF_COLLISION_RADIUS_PAD_M
    present = sorted({int(b) for b in body.tolist()})
    id_to_name = {i: body_names[i] for i in present}
    pairs = []
    for i, bi in enumerate(present):
        for bj in present[i + 1:]:
            na, nb = id_to_name[bi], id_to_name[bj]
            if _is_digit_link(na) or _is_digit_link(nb):
                continue
            if na in G1_BODY_PARENT and nb in G1_BODY_PARENT:
                if _is_chain_neighbor(na, nb) or _is_sibling(na, nb):
                    continue
            pairs.append((bi, bj))
    if not pairs:
        return None

    local_c, local_r, slices = [], [], []
    for bid in present:
        mask = body == bid
        p = pts[mask]
        r = rad[mask]
        c = p.mean(dim=0)
        extent = (p - c).norm(dim=-1) + r
        local_c.append(c)
        local_r.append(extent.max().reshape(()))
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        slices.append((bid, idx))

    bid_to_slice = {bid: idx for bid, idx in slices}
    pair_a = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    pair_b = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    compact_ids = torch.tensor(present, dtype=torch.long, device=device)
    id_to_row = {bid: row for row, bid in enumerate(present)}
    row_a = torch.tensor([id_to_row[p[0]] for p in pairs], dtype=torch.long, device=device)
    row_b = torch.tensor([id_to_row[p[1]] for p in pairs], dtype=torch.long, device=device)
    return {
        "pair_a": pair_a,
        "pair_b": pair_b,
        "row_a": row_a,
        "row_b": row_b,
        "compact_ids": compact_ids,
        "local_c": torch.stack(local_c, dim=0),
        "local_r": torch.stack(local_r, dim=0),
        "bid_to_slice": bid_to_slice,
        "pts": pts,
        "rad": rad,
    }


def max_self_penetration(robot, prep) -> torch.Tensor:
    """Deepest non-adjacent collision-shape overlap, metres, per environment."""
    import isaaclab.utils.math as math_utils

    n_env = robot.data.body_pos_w.shape[0]
    if prep is None:
        return torch.zeros(n_env, 1, device=robot.device)

    compact = prep["compact_ids"]
    q = robot.data.body_quat_w[:, compact]
    t = robot.data.body_pos_w[:, compact]
    world_c = math_utils.quat_apply(q, prep["local_c"].expand(n_env, -1, -1)) + t
    world_r = prep["local_r"]
    row_a, row_b = prep["row_a"], prep["row_b"]
    gap = (world_c[:, row_a] - world_c[:, row_b]).norm(dim=-1) - (world_r[row_a] + world_r[row_b])
    near = gap < SELF_COLLISION_SPHERE_MARGIN_M
    worst = torch.zeros(n_env, device=robot.device)
    if not near.any():
        return worst.unsqueeze(1)

    pts, rad = prep["pts"], prep["rad"]
    bid_to_slice = prep["bid_to_slice"]
    env_idx, pair_idx = torch.where(near)
    max_checks = 64 * n_env
    if env_idx.numel() > max_checks:
        score = -gap[env_idx, pair_idx]
        keep = score.topk(max_checks).indices
        env_idx, pair_idx = env_idx[keep], pair_idx[keep]

    pair_a, pair_b = prep["pair_a"], prep["pair_b"]
    for e, p in zip(env_idx.tolist(), pair_idx.tolist()):
        ba, bb = int(pair_a[p]), int(pair_b[p])
        ia, ib = bid_to_slice[ba], bid_to_slice[bb]
        qa = robot.data.body_quat_w[e, ba].unsqueeze(0).expand(ia.numel(), -1)
        qb = robot.data.body_quat_w[e, bb].unsqueeze(0).expand(ib.numel(), -1)
        wa = math_utils.quat_apply(qa, pts[ia]) + robot.data.body_pos_w[e, ba]
        wb = math_utils.quat_apply(qb, pts[ib]) + robot.data.body_pos_w[e, bb]
        sep = torch.cdist(wa.unsqueeze(0), wb.unsqueeze(0)).squeeze(0)
        sep = sep - rad[ia][:, None] - rad[ib][None, :]
        pen = (-sep.min()).clamp(min=0.0)
        if pen > worst[e]:
            worst[e] = pen
    return worst.unsqueeze(1)


def run_trial(env, policy, speeds, *, gates, distance=100.0, max_seconds=200.0,
              settle_seconds=1.0, geometry=None, device="cuda:0", on_step=None):
    """Roll one policy over the course and return a RunResult per lane."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    command = unwrapped.command_manager.get_term("base_velocity")
    origins = unwrapped.scene.env_origins
    dt = unwrapped.step_dt
    n = len(speeds)

    geom_pts, geom_rad, geom_body = load_geometry(geometry, robot, device)
    self_prep = prepare_self_collision(
        geom_pts, geom_rad, geom_body, list(robot.body_names), device)
    if self_prep is None:
        raise RuntimeError("official self-collision geometry is unavailable")

    # World Athletics torso rule: the finish (and every split) is judged by the
    # forward-most point of the torso against the line, not the pelvis centre,
    # so a dip at the tape pays off exactly as it does for a sprinter.  The
    # torso's collision hull is transformed by its live pose each step and the
    # leading point taken; lateral position and height stay on the base, which
    # is what the lane and fall checks care about.
    import isaaclab.utils.math as _torso_math
    torso_id = (robot.body_names.index("torso_link")
                if "torso_link" in robot.body_names else None)
    if geom_pts is not None and torso_id is not None:
        _tmask = geom_body == torso_id
        torso_pts, torso_rad = geom_pts[_tmask], geom_rad[_tmask]
        if torso_pts.shape[0] == 0:
            torso_pts = torso_rad = None
    else:
        torso_pts = torso_rad = None

    def torso_forward():
        if torso_pts is None:
            return (robot.data.root_pos_w - origins)[:, 0]
        # quat_apply pairs one quaternion with one vector, so the torso's single
        # orientation is expanded to one per hull point (this is what the
        # penetration check gets for free, since its quaternion is already
        # indexed per point).
        n_pts = torso_pts.shape[0]
        q = robot.data.body_quat_w[:, torso_id].unsqueeze(1).expand(-1, n_pts, -1)
        tp = robot.data.body_pos_w[:, torso_id].unsqueeze(1)
        world = _torso_math.quat_apply(q, torso_pts.expand(q.shape[0], -1, -1)) + tp
        return ((world[..., 0] - origins[:, 0:1]) + torso_rad).max(dim=1).values

    obs, _ = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    all_ids = torch.arange(n, device=device)

    # 1. settle under zero command; the scored clock has not started.
    command.hold(all_ids)
    for _ in range(int(settle_seconds / dt)):
        with torch.inference_mode():
            obs, *_ = env.step(policy(obs_t))
        obs_t = obs["policy"] if isinstance(obs, dict) else obs

    # 2. release
    command.release(all_ids)
    start_x = torso_forward().clone()
    trace: dict[str, list] = {key: [] for key in ("t", "x", "y", "vx", "self")}
    resolved = [False] * n
    t = 0.0

    # Include the sample at exactly max_seconds, but never accept a crossing
    # after the public time window.
    steps = int(round(max_seconds / dt))
    for step in range(steps + 1):
        pos = robot.data.root_pos_w - origins
        pos[:, 0] = torso_forward() - start_x
        snapshot = torch.cat([
            pos[:, :2],
            robot.data.root_lin_vel_b[:, :1],
            max_self_penetration(robot, self_prep),
        ], dim=1)
        rows = snapshot.tolist()

        trace["t"].append(t)
        trace["x"].append([r[0] for r in rows])
        trace["y"].append([r[1] for r in rows])
        trace["vx"].append([r[2] for r in rows])
        trace["self"].append([r[3] for r in rows])

        xs = trace["x"][-1]
        for i in range(n):
            if not resolved[i] and xs[i] >= distance:
                command.hold(torch.tensor([i], device=device))
                resolved[i] = True

        if on_step is not None:
            on_step(t, trace, resolved)
        if all(resolved):
            break
        if step == steps:
            break
        with torch.inference_mode():
            obs, *_ = env.step(policy(obs_t))
        obs_t = obs["policy"] if isinstance(obs, dict) else obs
        t += dt

    return [
        evaluate_run(
            env_id=i,
            commanded_speed=speeds[i],
            t=trace["t"],
            x=[row[i] for row in trace["x"]],
            y=[row[i] for row in trace["y"]],
            vx=[row[i] for row in trace["vx"]],
            gates=gates,
            finish_distance_m=distance,
            self_penetration_m=[row[i] for row in trace["self"]],
        )
        for i in range(n)
    ], {"collision_points": 0 if geom_pts is None else int(geom_pts.shape[0]),
        "self_collision_pairs": 0 if self_prep is None else int(self_prep["pair_a"].numel()),
        "control_hz": round(1.0 / dt, 2)}
