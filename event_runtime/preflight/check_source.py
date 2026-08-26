#!/usr/bin/env python3
"""Reject launches whose evaluation-affecting source is not committed."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# Keep this list narrow and explicit. Exporters, generated run data, and
# operator-only documentation cannot affect an evaluation and must not block a
# launch merely because another task is editing them in the same worktree.
EVALUATION_SOURCE_PATHS = (
    "event_runtime/agent",
    "event_runtime/container",
    "event_runtime/control",
    "event_runtime/models",
    "event_runtime/preflight",
    "event_runtime/event.py",
    "event_runtime/image.py",
    "event_runtime/pricing.py",
    "event_runtime/sync_verifier.py",
    "events/g1-100-metres",
    "harbor",
)


def dirty_evaluation_source() -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *EVALUATION_SOURCE_PATHS,
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [line for line in completed.stdout.splitlines() if line]


def main() -> int:
    dirty = dirty_evaluation_source()
    if dirty:
        print("evaluation-affecting source is not committed:")
        print("\n".join(dirty))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
