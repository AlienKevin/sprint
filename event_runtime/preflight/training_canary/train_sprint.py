#!/usr/bin/env python3
"""Deterministic Isaac Lab PPO workload for the functional launch canary.

This program is benchmark infrastructure, not an agent-produced candidate.  It
exists solely to prove that the immutable training image can run real PPO,
export the event's TorchScript policy ABI, and hand the resulting bytes to both
verifiers before a paid batch is allowed to launch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--max_iters", type=int, default=10)
parser.add_argument("--chunk_iters", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.exporter import export_policy_as_jit  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    G1FlatPPORunnerCfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import verifier.tasks  # noqa: F401,E402
from verifier.assets import use_local_assets  # noqa: E402
from verifier.environment import G1100MetresEnvCfg  # noqa: E402


SEED = 20260824


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    if args.chunk_iters <= 0 or args.chunk_iters > args.max_iters:
        raise ValueError("chunk_iters must be in [1, max_iters]")
    if not use_local_assets():
        raise RuntimeError("sealed Isaac Lab asset mirror is unavailable")

    checkpoint_dir = Path(
        os.environ.get("SPRINT_GPU_CHECKPOINT_DIR", "/app/checkpoints")
    )
    progress_file = Path(
        os.environ.get("SPRINT_GPU_PROGRESS_FILE", "/app/progress.json")
    )
    train_root = Path(os.environ.get("SPRINT_TRAIN_ROOT", "/app/logs"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        progress_file,
        {"schema_version": 1, "finished": False, "completed_iterations": 0},
    )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env_cfg = G1100MetresEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 4.0
    env_cfg.episode_length_s = 8.0
    env_cfg.seed = SEED
    env_cfg.sim.device = args.device or "cuda:0"
    env_cfg.export_io_descriptors = False
    env_cfg.log_dir = str(train_root)

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.seed = SEED
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.max_iterations = args.max_iters
    agent_cfg.num_steps_per_env = 24
    agent_cfg.save_interval = args.save_interval
    agent_cfg.experiment_name = "sprint_functional_canary"
    agent_cfg.run_name = "pinned"
    agent_cfg.logger = "tensorboard"

    env = gym.make("Isaac-G1-100Metres-v0", cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    try:
        runner = OnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=str(train_root),
            device=agent_cfg.device,
        )
        runner.learn(
            num_learning_iterations=args.max_iters,
            init_at_random_ep_len=False,
        )
        runner.save(str(checkpoint_dir / "runner_final.pt"))
        policy = runner.alg.policy
        normalizer = getattr(policy, "actor_obs_normalizer", None)
        export_policy_as_jit(
            policy,
            normalizer=normalizer,
            path=str(checkpoint_dir),
            filename="policy_final.pt",
        )
    finally:
        wrapped.close()

    atomic_json(
        progress_file,
        {
            "schema_version": 1,
            "finished": True,
            "completed_iterations": args.max_iters,
            "seed": SEED,
        },
    )
    print(f"Learning iteration {args.max_iters - 1}/{args.max_iters}", flush=True)
    print("[sprint] training complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
