#!/usr/bin/env python3
"""Queue the published verifier on this trial's own training A10G.

Usage:
  event test POLICY.pt [--note TEXT]

The command prints a GPU job ID. Inspect it with ``event gpu status``,
``wait``, or ``logs``. This is agent-funded local debugging; official scoring
is separate and blind.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def build_command(worker_policy: Path, note: str) -> list[str]:
    """Build the exact public local-verifier GPU command."""
    script = (
        "export SPRINT_ISAAC_BOOTSTRAP=/opt/sprint-isaac-bootstrap.py; "
        "set -euo pipefail; "
        "export TESTS_DIR=/opt/event-verifier; "
        f"export SUBMISSION={shlex.quote(str(worker_policy))}; "
        "export LOGS_DIR=/durable/runs/$SPRINT_RUN_ID/local-verifier/"
        "$SPRINT_GPU_JOB_ID; "
        "exec bash /opt/event-verifier/test.sh"
    )
    return [
        "/usr/local/bin/event",
        "gpu",
        "--timeout",
        "900",
        "--job-kind",
        "verify",
        "--note",
        note,
        "--",
        "bash",
        "-lc",
        script,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", help="TorchScript policy below /app")
    parser.add_argument("--note", default="local verifier", help="GPU job annotation")
    args = parser.parse_args()

    policy = Path(args.policy).resolve()
    app = Path("/app").resolve()
    try:
        relative = policy.relative_to(app)
    except ValueError:
        parser.error("policy must be below /app so it is included in the GPU workspace")
    if not policy.is_file():
        parser.error(f"policy not found: {policy}")

    worker_policy = Path("/app") / relative
    command = build_command(worker_policy, args.note)
    completed = subprocess.run(command, env=os.environ.copy(), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
