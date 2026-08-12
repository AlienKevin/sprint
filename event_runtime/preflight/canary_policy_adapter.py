"""Normalize the functional canary policy to the published optional reset ABI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


class MaskedResetPolicy(torch.nn.Module):
    """Wrap a stateless trained policy with the event's masked reset method."""

    def __init__(self, policy: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy(observation)

    @torch.jit.export
    def reset(self, done_mask: torch.Tensor) -> None:
        # The wrapped policy is stateless. Consuming the mask keeps the method
        # signature explicit without changing the policy's trained output.
        _ = done_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    args = parser.parse_args()

    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    wrapped = torch.jit.script(MaskedResetPolicy(policy).eval())
    temporary = args.policy.with_suffix(args.policy.suffix + ".tmp")
    torch.jit.save(wrapped, str(temporary))
    temporary.replace(args.policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
