#!/usr/bin/env python3
# Copyright (c) 2026 QWOP-bench contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Load the real checkpoint on the CPU, before spending a GPU minute on it.

Rebuilding a network from a state dict is exactly the kind of code that looks
right and silently is not, and finding out inside a running simulator costs an
Isaac Sim start-up per attempt.

    python tests/test_policy.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sprintbench.policy import load_policy  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(HERE, "checkpoints", "Isaac-Velocity-Flat-G1-v0.pt")


def check(name: str, condition, detail: str = "") -> bool:
    passed = bool(condition)
    print(f"  {'PASS' if passed else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    return passed


def main() -> int:
    ok = True

    print("zero baseline")
    zero = load_policy("zero", 123, 37, "cpu")
    out = zero(torch.zeros(5, 123))
    ok &= check("shape", out.shape == (5, 37), str(tuple(out.shape)))
    ok &= check("all zero", bool((out == 0).all()))

    if not os.path.exists(CKPT):
        print(f"\nSKIP: {CKPT} not present")
        return 0 if ok else 1

    print("\nofficial Isaac Lab G1 flat checkpoint")
    policy = load_policy(CKPT, 123, 37, "cpu")
    obs = torch.zeros(4, 123)
    out = policy(obs)
    ok &= check("shape", out.shape == (4, 37), str(tuple(out.shape)))
    ok &= check("finite", bool(torch.isfinite(out).all()))
    ok &= check("not identically zero", bool(out.abs().sum() > 0), f"|a| mean {out.abs().mean():.4f}")

    # The rebuilt actor must reproduce the saved network exactly, not merely
    # have the right shape: compare against a hand-rolled forward pass through
    # the checkpoint's own tensors.
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]
    x = torch.randn(3, 123)
    h = x
    for i, idx in enumerate([0, 2, 4, 6]):
        h = h @ sd[f"actor.{idx}.weight"].T + sd[f"actor.{idx}.bias"]
        if i < 3:
            h = torch.nn.functional.elu(h)
    ok &= check("matches a manual forward pass",
                torch.allclose(policy(x), h, atol=1e-6),
                f"max diff {(policy(x) - h).abs().max():.2e}")

    print("\nshape mismatches are refused")
    for obs_dim, act_dim, why in [(310, 37, "rough-terrain observation"), (123, 12, "12-DoF embodiment")]:
        try:
            load_policy(CKPT, obs_dim, act_dim, "cpu")
            ok &= check(why, False, "loaded anyway")
        except (ValueError, RuntimeError) as e:
            ok &= check(why, True, type(e).__name__)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
