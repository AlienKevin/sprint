# Copyright (c) 2026 Sprint contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Forward ready start shared by every policy rollout.

World Athletics requires blocks for the 100 metres.  Its standing-start rules
instead constrain contact with the ground and starting line without defining a
single whole-body pose.  This event therefore freezes the G1 ready pose below:
the torso pitched 40 degrees toward the finish, the right foot leading by
0.46 m, both feet planted on the track, hands clear of the ground, and every
velocity zero.  The entire robot starts behind the line.  It adds no blocks or
other track bodies.
"""

from __future__ import annotations

from typing import Any


CANONICAL_START_NAME = "forward-ready-start-v1"
FORWARD_LEAN_DEGREES = 40.0
ROOT_PITCH_DEGREES = 40.0
LEAD_LEG = "right"
ROOT_POSITION_M = (-0.55, 0.0, 0.696634)
ROOT_ORIENTATION_WXYZ = (0.939692621, 0.0, 0.342020143, 0.0)
ROOT_LINEAR_VELOCITY_M_S = (0.0, 0.0, 0.0)
ROOT_ANGULAR_VELOCITY_RAD_S = (0.0, 0.0, 0.0)

# Sagittal-leg angles were solved against the benchmark's single-torso-joint
# Isaac G1 and its effective joint limits.  Each foot has sole collision
# contact at z=0, with ankle-roll centres 0.46 m apart.  The rear and lead feet
# pitch by 3.33 and -1.76 degrees respectively: the closest planted solution
# that preserves the approved 40-degree silhouette and ankle-limit margin.
JOINT_POSITIONS_RAD = {
    "left_hip_pitch_joint": -0.80738942,
    "left_knee_joint": 0.83745486,
    "left_ankle_pitch_joint": -0.67,
    "right_hip_pitch_joint": -1.18079717,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.45187998,
    ".*_elbow_pitch_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": -0.65,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.95,
    "left_one_joint": 1.0,
    "right_one_joint": -1.0,
    "left_two_joint": 0.52,
    "right_two_joint": -0.52,
}
JOINT_VELOCITIES_RAD_S = {".*": 0.0}


def apply_canonical_standing_start(robot_cfg: Any) -> Any:
    """Apply the exact start, including the policy's joint-action reference."""
    state = robot_cfg.init_state
    state.pos = ROOT_POSITION_M
    state.rot = ROOT_ORIENTATION_WXYZ
    state.lin_vel = ROOT_LINEAR_VELOCITY_M_S
    state.ang_vel = ROOT_ANGULAR_VELOCITY_RAD_S
    state.joint_pos = dict(JOINT_POSITIONS_RAD)
    state.joint_vel = dict(JOINT_VELOCITIES_RAD_S)
    return robot_cfg
