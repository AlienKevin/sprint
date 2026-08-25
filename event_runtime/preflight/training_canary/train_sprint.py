#!/usr/bin/env python3
"""Method-neutral Isaac Lab optimization workload for the launch canary.

This program is benchmark infrastructure, not an agent-produced candidate.  It
proves that the immutable training image can construct and step the exact
environment, optimize a Torch module on the GPU, export the event's TorchScript
policy ABI, and hand the resulting bytes to both verifiers before a paid batch
is allowed to launch.  It deliberately avoids choosing a learning algorithm on
the entrant's behalf.
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

import verifier.tasks  # noqa: F401,E402
from verifier.assets import use_local_assets  # noqa: E402
from verifier.environment import G1100MetresEnvCfg  # noqa: E402
from train.spec import ACTION_DIM, OBSERVATION_DIM  # noqa: E402


SEED = 20260824


class CanaryPolicy(torch.nn.Module):
    """Small generic policy used only to exercise optimization and export."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_DIM, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, ACTION_DIM),
            torch.nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


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

    env = gym.make("Isaac-G1-100Metres-v0", cfg=env_cfg)
    policy = CanaryPolicy().to(env_cfg.sim.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    try:
        observations, _ = env.reset()
        for iteration in range(args.max_iters):
            policy_observations = observations["policy"].detach()
            actions = policy(policy_observations)
            loss = actions.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            observations, *_ = env.step(actions.detach())
            print(
                f"Optimization iteration {iteration}/{args.max_iters} "
                f"loss={loss.item():.6f}",
                flush=True,
            )

        torch.save(
            {
                "seed": SEED,
                "iterations": args.max_iters,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_dir / "optimizer_final.pt",
        )
        scripted = torch.jit.script(policy.to("cpu").eval())
        torch.jit.save(scripted, str(checkpoint_dir / "policy_final.pt"))
    finally:
        env.close()

    atomic_json(
        progress_file,
        {
            "schema_version": 1,
            "finished": True,
            "completed_iterations": args.max_iters,
            "seed": SEED,
        },
    )
    print(f"Optimization iteration {args.max_iters - 1}/{args.max_iters}", flush=True)
    print("[sprint] optimization canary complete", flush=True)


if __name__ == "__main__":
    main()
    # The functional canary runs in an ephemeral GPU sandbox. Its policy and
    # completion record are atomically committed above; waiting for Isaac/Kit
    # destruction has occasionally hung despite all useful work being done.
    # Exercise the same durable completion boundary as production workers.
    os._exit(0)
