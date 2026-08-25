#!/usr/bin/env python3
"""Prove that the published rollout can be reused in one Isaac process."""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, "/opt/event-verifier")
from verifier.assets import use_local_assets  # noqa: E402

if not use_local_assets():
    raise RuntimeError("local asset mirror is unavailable")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import verifier.tasks  # noqa: F401,E402
from verifier.environment import G1100MetresEnvCfg  # noqa: E402
from verifier.rollout import run_trial  # noqa: E402


def main() -> int:
    cfg = G1100MetresEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.episode_length_s = 3.0
    env = gym.make("Isaac-G1-100Metres-v0", cfg=cfg)
    try:
        robot = env.unwrapped.scene["robot"]
        actions = torch.zeros((1, robot.num_joints), device=robot.device)

        def zero_policy(obs: torch.Tensor) -> torch.Tensor:
            return actions.expand(obs.shape[0], -1)

        # The second and third calls are the regression boundary: env.reset()
        # must be able to update every tensor retained by the previous step.
        for repeat in range(3):
            results, _ = run_trial(
                env,
                zero_policy,
                [0.0],
                gates=(1.0,),
                distance=1.0,
                max_seconds=0.04,
                settle_seconds=0.02,
                geometry="/opt/event-verifier/verifier/collision_geometry.json",
                device="cuda:0",
            )
            assert len(results) == 1
            print(f"repeat {repeat + 1}: ok", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    exit_code = main()
    # Isaac Sim 5.1 can hang indefinitely in SimulationApp.close() after the
    # environment has already released its resources. This is a disposable
    # canary process, so flush its evidence and let process teardown reclaim
    # the remaining Kit globals instead of entering that unbounded hook.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
