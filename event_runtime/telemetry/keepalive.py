#!/usr/bin/env python3
"""Emit a Modal keepalive argv JSON that starts in-sandbox telemetry.

Used by durable and open Harbor runners so every future trial gets sampling
without host coupling. Safe when /opt/sprint-telemetry.sh is missing.
"""

from __future__ import annotations

import argparse
import json
import shlex


def keepalive_argv(
    *,
    run_id: str = "",
    interval_seconds: int = 20,
    then: list[str] | None = None,
) -> list[str]:
    run_export = f"SPRINT_RUN_ID={shlex.quote(run_id)} " if run_id else ""
    run_arg = f"--run-id {shlex.quote(run_id)} " if run_id else ""
    start = (
        "umask 077; mkdir -p /logs/artifacts/telemetry /run; "
        "if [ -x /opt/sprint-telemetry.sh ]; then "
        f"{run_export}/opt/sprint-telemetry.sh --role cpu-agent {run_arg}"
        f"--out-dir /logs/artifacts/telemetry "
        f"--interval-seconds {int(interval_seconds)} "
        "--pidfile /run/sprint-telemetry.pid || true; "
        "fi; "
    )
    if then:
        command = start + "exec " + " ".join(shlex.quote(part) for part in then)
    else:
        command = start + "exec sleep infinity"
    return ["sh", "-c", command]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--interval-seconds", type=int, default=20)
    parser.add_argument(
        "--then",
        nargs=argparse.REMAINDER,
        help="Optional command to exec after starting telemetry",
    )
    args = parser.parse_args()
    then = args.then
    if then and then[0] == "--":
        then = then[1:]
    print(
        json.dumps(
            keepalive_argv(
                run_id=args.run_id,
                interval_seconds=args.interval_seconds,
                then=then or None,
            ),
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
