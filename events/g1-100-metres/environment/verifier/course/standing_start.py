# Copyright (c) 2026 Sprint contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Canonical no-block standing start shared by every policy rollout.

World Athletics requires blocks for the 100 metres.  Its standing-start rules
instead constrain contact with the ground and starting line without defining a
single whole-body pose.  This event therefore freezes the stable G1 standing
pose below: upright and aligned behind the line, feet grounded, hands clear of
the ground, and every velocity zero.  It adds no blocks or other track bodies.
"""

from __future__ import annotations

from typing import Any


CANONICAL_START_NAME = "no-block-standing-start-v1"
ROOT_POSITION_M = (0.0, 0.0, 0.74)
ROOT_ORIENTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)
ROOT_LINEAR_VELOCITY_M_S = (0.0, 0.0, 0.0)
ROOT_ANGULAR_VELOCITY_RAD_S = (0.0, 0.0, 0.0)

# This is the Isaac Lab 2.3.2 G1_CFG standing pose, made explicit so an
# upstream asset-default change cannot silently alter the event start.
JOINT_POSITIONS_RAD = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_pitch_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
    "left_one_joint": 1.0,
    "right_one_joint": -1.0,
    "left_two_joint": 0.52,
    "right_two_joint": -0.52,
}
JOINT_VELOCITIES_RAD_S = {".*": 0.0}


def apply_canonical_standing_start(robot_cfg: Any) -> Any:
    """Apply the exact event start without changing its action reference pose."""
    state = robot_cfg.init_state
    state.pos = ROOT_POSITION_M
    state.rot = ROOT_ORIENTATION_WXYZ
    state.lin_vel = ROOT_LINEAR_VELOCITY_M_S
    state.ang_vel = ROOT_ANGULAR_VELOCITY_RAD_S
    state.joint_pos = dict(JOINT_POSITIONS_RAD)
    state.joint_vel = dict(JOINT_VELOCITIES_RAD_S)
    return robot_cfg
