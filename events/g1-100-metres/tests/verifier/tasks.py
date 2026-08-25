# Copyright (c) 2026 Sprint contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Gym registration for the course. Importing this needs a running Kit app."""

import gymnasium as gym

from . import environment

gym.register(
    id="Isaac-G1-100Metres-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{environment.__name__}:G1100MetresEnvCfg",
    },
)
