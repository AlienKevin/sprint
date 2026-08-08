#!/usr/bin/env python3
"""Convert a sealed verifier replay and G1 meshes to self-contained 3D HTML."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import statistics
from pathlib import Path

SP = Path(__file__).resolve().parent / "_lane_chrome"
HQ_DEFAULT = Path(__file__).resolve().parent / "g1_hq.json"
COLS = ["#6E97C4", "#E0A43B", "#B6F24E", "#F2704E"]
LANE_HALF_WIDTH_M = 0.61
SCORED_TIMEOUT_S = 60.0
# Display-only proxy when capture JSON has no self-penetration trace.
TORSO_COLLAPSE_Z_M = 0.35

PREFERRED = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "pelvis_contour_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_roll_link",
    "left_palm_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_roll_link",
    "right_palm_link",
]

G1_PARENT = {
    "pelvis": None,
    "pelvis_contour_link": "pelvis",
    "left_hip_pitch_link": "pelvis",
    "left_hip_roll_link": "left_hip_pitch_link",
    "left_hip_yaw_link": "left_hip_roll_link",
    "left_knee_link": "left_hip_yaw_link",
    "left_ankle_pitch_link": "left_knee_link",
    "left_ankle_roll_link": "left_ankle_pitch_link",
    "right_hip_pitch_link": "pelvis",
    "right_hip_roll_link": "right_hip_pitch_link",
    "right_hip_yaw_link": "right_hip_roll_link",
    "right_knee_link": "right_hip_yaw_link",
    "right_ankle_pitch_link": "right_knee_link",
    "right_ankle_roll_link": "right_ankle_pitch_link",
    "torso_link": "pelvis",
    "head_link": "torso_link",
    "left_shoulder_pitch_link": "torso_link",
    "left_shoulder_roll_link": "left_shoulder_pitch_link",
    "left_shoulder_yaw_link": "left_shoulder_roll_link",
    "left_elbow_pitch_link": "left_shoulder_yaw_link",
    "left_elbow_roll_link": "left_elbow_pitch_link",
    "left_palm_link": "left_elbow_roll_link",
    "right_shoulder_pitch_link": "torso_link",
    "right_shoulder_roll_link": "right_shoulder_pitch_link",
    "right_shoulder_yaw_link": "right_shoulder_roll_link",
    "right_elbow_pitch_link": "right_shoulder_yaw_link",
    "right_elbow_roll_link": "right_elbow_pitch_link",
    "right_palm_link": "right_elbow_roll_link",
}


def _rotate_inverse_xyzw(q: list[float], v: list[float]) -> list[float]:
    """Rotate ``v`` by the inverse of an xyzw quaternion."""
    x, y, z, w = (float(a) for a in q)
    mag = math.sqrt(x * x + y * y + z * z + w * w)
    if mag == 0:
        raise ValueError("zero-length capture quaternion")
    x, y, z, w = -x / mag, -y / mag, -z / mag, w / mag
    vx, vy, vz = (float(a) for a in v)
    uv = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    uuv = (
        y * uv[2] - z * uv[1],
        z * uv[0] - x * uv[2],
        x * uv[1] - y * uv[0],
    )
    return [
        vx + 2 * (w * uv[0] + uuv[0]),
        vy + 2 * (w * uv[1] + uuv[1]),
        vz + 2 * (w * uv[2] + uuv[2]),
    ]


def _rest_offsets(cap: dict, names: list[str], links: list[str]) -> dict[str, list[float]]:
    """Recover fixed child origins in each captured parent link frame.

    Capture rows store independently rounded world poses. Taking a median over
    every frame removes that sub-millimetre noise and gives the fixed
    articulation offsets from the exact asset used for the rollout.
    """
    offsets: dict[str, list[list[float]]] = {}
    for child in links:
        parent = G1_PARENT.get(child)
        if parent not in links:
            continue
        ci, pi = names.index(child), names.index(parent)
        samples = offsets.setdefault(child, [])
        for run in cap["frames"]:
            for row in run:
                co, po = 1 + ci * 7, 1 + pi * 7
                delta = [row[co + axis] - row[po + axis] for axis in range(3)]
                samples.append(_rotate_inverse_xyzw(row[po + 3 : po + 7], delta))
    return {
        child: [round(statistics.median(axis), 6) for axis in zip(*samples)]
        for child, samples in offsets.items()
    }


def compute_dq_event(
    frames: list,
    names: list[str],
    run: dict | None,
    *,
    lane_half: float = LANE_HALF_WIDTH_M,
) -> tuple[float | None, str | None]:
    """Website-only DQ instant for replay freeze (does not affect scoring).

    Prefer an explicit ``dq_time`` on the run record. Otherwise take the earliest
    of: first |lateral| > lane half-width, first torso collapse (self-collision
    proxy when penetration traces are absent), or end of scored segment when the
    finished gate fails.
    """
    run = run or {}
    if run.get("valid") is True:
        return None, None
    if run.get("dq_time") is not None:
        return float(run["dq_time"]), str(run.get("dq_reason") or "disqualified")

    if not frames:
        return None, None

    ti = names.index("torso_link") if "torso_link" in names else 0
    o_tor = 1 + ti * 7
    sy = float(frames[0][2])
    sx = float(frames[0][1])
    t_lane = None
    t_collapse = None
    t_cross = None
    last_t = float(frames[-1][0])

    for i, fr in enumerate(frames):
        t = float(fr[0])
        lat = abs(float(fr[2]) - sy)
        dist = float(fr[o_tor]) - sx
        z = float(fr[o_tor + 2])
        if t_lane is None and lat > lane_half:
            if i > 0:
                prev = frames[i - 1]
                lat0 = abs(float(prev[2]) - sy)
                dt = t - float(prev[0])
                if lat0 <= lane_half < lat and lat != lat0 and dt > 0:
                    t_lane = float(prev[0]) + (lane_half - lat0) / (lat - lat0) * dt
                else:
                    t_lane = t
            else:
                t_lane = t
        if t_collapse is None and z < TORSO_COLLAPSE_Z_M:
            t_collapse = t
        if t_cross is None and dist >= 100.0:
            t_cross = t

    cands: list[tuple[float, str]] = []
    if t_lane is not None:
        cands.append((t_lane, "in_lane"))
    if t_collapse is not None:
        cands.append((t_collapse, "self_collision"))
    finish = run.get("finish")
    if t_cross is None and finish is None:
        cands.append((min(last_t, SCORED_TIMEOUT_S), "finished"))
    if not cands:
        t = t_cross if t_cross is not None else (float(finish) if finish is not None else last_t)
        return t, str(run.get("dq_reason") or "disqualified")
    cands.sort(key=lambda x: x[0])
    return cands[0][0], cands[0][1]


def capture_to_data(cap: dict, hq_all: dict, meta_policy: str, pad: float = 0.6) -> dict:
    names = cap["body_names"]
    links = [n for n in PREFERRED if n in hq_all and n in names]
    ridx = [names.index(n) for n in links]
    parents = {n: G1_PARENT[n] for n in links if G1_PARENT.get(n) in links}
    rest = _rest_offsets(cap, names, links)
    runs = cap.get("runs") or []
    finishes = []
    dq_events: list[tuple[float | None, str | None]] = []
    for i, fr in enumerate(cap["frames"]):
        run = runs[i] if i < len(runs) else {}
        fin = None
        if run.get("finish") is not None:
            fin = float(run["finish"])
        else:
            ti = names.index("torso_link") if "torso_link" in names else 0
            o = 1 + ti * 7
            for row in fr:
                if row[o] >= 100.0:
                    fin = float(row[0])
                    break
            if fin is None and fr:
                fin = float(fr[-1][0])
        finishes.append(fin if fin is not None else float(fr[-1][0] if fr else 0.0))
        dq_events.append(compute_dq_event(fr, names, run))

    src_fps = float(cap.get("fps") or 25.0)
    out_fps = src_fps

    def pack(fr):
        row = [round(fr[0], 3)]
        for b in ridx:
            o = 1 + b * 7
            row += [
                round(fr[o], 3),
                round(fr[o + 1], 3),
                round(fr[o + 2], 3),
                round(fr[o + 3], 4),
                round(fr[o + 4], 4),
                round(fr[o + 5], 4),
                round(fr[o + 6], 4),
            ]
        return row

    policies = []
    for i, fr in enumerate(cap["frames"]):
        fin = finishes[i]
        run = runs[i] if i < len(runs) else {}
        valid = bool(run["valid"]) if i < len(runs) and "valid" in run else None
        dq_time, dq_reason = dq_events[i]
        # Clip each seed at its own freeze horizon so DQ pages stay small.
        clip_t = (dq_time if (valid is False and dq_time is not None) else fin) + pad
        kept = [pack(f) for f in fr if f[0] <= clip_t]
        sy = kept[0][2] if kept else 0.0
        ys = [abs(f[2] - sy) for f in kept]
        pol = {
            "label": f"Seed {i + 1}",
            "finish": round(fin, 3),
            "frames": kept,
            "max_lateral_m": round(max(ys) if ys else 0.0, 3),
            "valid": valid,
        }
        if valid is False and dq_time is not None:
            pol["dq_time"] = round(float(dq_time), 3)
            pol["dq_reason"] = dq_reason
        policies.append(pol)

    return {
        "fps": out_fps,
        "links": links,
        "parents": parents,
        "rest": rest,
        "hq": {n: hq_all[n] for n in links},
        "policies": policies,
        "meta": {
            "policy": meta_policy,
            "lane_half_width_m": LANE_HALF_WIDTH_M,
            "source": cap.get("policy"),
            "frame_space": "world_link",
            "position_unit": "m",
            "quaternion_order": "xyzw",
            "visual_origin": "link_frame",
            "interpolation": "local_hierarchy",
            "dq_display": "freeze_at_first_gate_failure",
        },
    }


def policy_nav_html(_active: str) -> str:
    """Return to the live comparison; policy navigation lives there."""
    return (
        '<div class="polsel" role="navigation" aria-label="Sprint comparison">'
        '<a class="pol" href="/">← All runs and policies</a></div>'
    )




if __name__ == "__main__":
    main()
