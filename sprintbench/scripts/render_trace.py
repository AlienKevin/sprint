#!/usr/bin/env python3
# Copyright (c) 2026 QWOP-bench contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Draw a recorded Isaac Lab rollout as video.

Kit cannot create a render device inside the Modal GPU container — every
driver library is there, but only ``/dev/nvidia3`` is exposed without the
matching ``/proc/driver/nvidia`` entries, so ``vkCreateDevice`` fails.  PhysX is
unaffected, because it goes through CUDA rather than the render device, so the
simulation is exactly the one that would have been filmed; only the camera is
missing.

So the harness records state and this draws it.  Nothing is stepped here: the
root pose and joint angles from the trace are written straight into ``qpos``
and the model is posed by forward kinematics.  A second simulation would drift
away from the run being inspected; this cannot.

The G1 is the same robot in both places, and the twelve leg joints carry the
same names, so the gait transfers exactly.  Isaac's model has 37 joints against
this one's 23 — the extra fourteen are finger joints, which do not move here.

    python scripts/render_trace.py results/trace-2.0.trace.json -o results/trace-2.0.mp4
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(HERE, "assets", "replay_scene.xml")

# Isaac Lab's G1 asset uses the older Unitree joint names for the torso and the
# forearm; the legs, which are what a gait is made of, match exactly.
ALIASES = {
    "torso_joint": "waist_yaw_joint",
    "left_elbow_pitch_joint": "left_elbow_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "left_elbow_roll_joint": "left_wrist_roll_joint",
    "right_elbow_roll_joint": "right_wrist_roll_joint",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=None, help="trim the clip")
    ap.add_argument("--view", default="chase", choices=["chase", "side", "front"])
    args = ap.parse_args()

    doc = json.load(open(args.trace))
    frames = np.asarray(doc["frames"], dtype=np.float64)
    names = doc["joint_names"]
    layout = doc.get("layout", [])
    src_fps = doc["fps"]
    print(f"{args.trace}: {len(frames)} frames @ {src_fps} Hz, "
          f"{len(names)} joints, cmd {doc['commanded_speed']} m/s")

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    # map each recorded joint onto this model's qpos, by name
    addr, matched, missing = {}, [], []
    for i, name in enumerate(names):
        target = ALIASES.get(name, name)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target)
        if jid < 0:
            missing.append(name)
            continue
        addr[i] = model.jnt_qposadr[jid]
        matched.append(name)
    print(f"posing {len(matched)} joints; not in this model: "
          f"{', '.join(missing) if missing else 'none'}")
    leg = [n for n in matched if "hip" in n or "knee" in n or "ankle" in n]
    assert len(leg) == 12, f"expected all twelve leg joints, matched {len(leg)}: {leg}"

    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
                   for s in ("left", "right")]
    # the four small spheres per foot that actually touch the ground
    foot_spheres = [g for g in range(model.ngeom)
                    if model.geom_bodyid[g] in foot_bodies
                    and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    assert len(foot_spheres) == 8, f"expected eight contact spheres, found {len(foot_spheres)}"

    def pose(row, lift: float = 0.0):
        data.qpos[:] = 0.0
        data.qpos[0:3] = row[0:3]
        data.qpos[2] += lift
        data.qpos[3:7] = row[3:7]           # w x y z, same convention as MuJoCo
        for i, a in addr.items():
            data.qpos[a] = row[7 + i]
        mujoco.mj_forward(model, data)

    # This model is not the one that was simulated.  Isaac runs the 37-DoF
    # g1_minimal USD; this is Unitree's 23-DoF MJCF, and their pelvis-to-ankle
    # geometry differs by about 8 cm.  Writing the recorded root height straight
    # into a model with longer legs buries the feet in the floor — which looks
    # exactly like the simulator letting them sink, and is not.
    #
    # The trace carries Isaac's own foot heights, so the correction is a
    # measured constant rather than a number tuned until it looked right: pose
    # the first frame, compare this model's ankle height against Isaac's, and
    # shift the root by the difference for every frame.
    # Calibrate on the sole, not on the ankle.  Matching the two models' ankle
    # origins still left the soles 9 mm under the floor, because the sole sits a
    # different distance below the ankle in each model and matching one does not
    # match the other.  What is actually known is physical: whenever a foot is
    # carrying load, its lowest contact point is on the plane.  So take the
    # deepest sole position across every loaded frame and lift by exactly that.
    lift = 0.0
    if "foot_f_l" in layout:
        gi = {n: i for i, n in enumerate(layout)}
        deepest = None
        for row in frames[::5]:
            if max(row[gi["foot_f_l"]], row[gi["foot_f_r"]]) <= 1.0:
                continue                       # airborne: says nothing about the floor
            pose(row)
            z = min(data.geom_xpos[g][2] - model.geom_size[g][0]
                    for g in foot_spheres)
            deepest = z if deepest is None else min(deepest, z)
        if deepest is not None:
            lift = -deepest
            print(f"vertical correction {lift * 1000:+.1f} mm "
                  f"(deepest loaded sole sat at {deepest:.4f} m)")
    else:
        print("WARNING: trace has no foot contact; the robot will render sunk "
              "into the floor by however much the two models' legs differ")

    step = max(1, int(round(src_fps / args.fps)))
    if args.seconds:
        frames = frames[: int(args.seconds * src_fps)]

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    if args.view == "chase":
        cam.distance, cam.elevation, cam.azimuth = 3.2, -10, 128
    elif args.view == "side":
        cam.distance, cam.elevation, cam.azimuth = 3.6, -8, 90
    else:
        cam.distance, cam.elevation, cam.azimuth = 3.4, -8, 180

    out = args.out or os.path.splitext(args.trace)[0].replace(".trace", "") + ".mp4"
    writer = imageio.get_writer(out, fps=args.fps, quality=8, macro_block_size=1)

    for k in range(0, len(frames), step):
        row = frames[k]
        pose(row, lift)
        cam.lookat[:] = [row[0], row[1], 0.75]
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())

    writer.close()
    dist = frames[-1][0] - frames[0][0]
    secs = len(frames) / src_fps
    print(f"wrote {out}: {len(range(0, len(frames), step))} frames, "
          f"{dist:.1f} m in {secs:.1f} s ({dist / secs:.2f} m/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
