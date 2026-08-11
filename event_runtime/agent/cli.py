"""Dispatch the agent-visible ``event`` command.

Commands:
  event gpu [ARGS...]       run or inspect this trial's GPU work
  event check POLICY.pt     validate the event-specific policy interface on CPU
  event test POLICY.pt      run the published checker on trial-local GPU compute
  event archive POLICY.pt   retain a candidate for blind official scoring
  event cost                print this trial's cumulative comparison cost as JSON
  event history             list this trial's candidate receipts
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from collections.abc import Callable

COMMANDS = {
    "gpu": "event_runtime.agent.gpu",
    "test": "event_runtime.agent.test_policy",
    "archive": "event_runtime.agent.archive",
    "cost": "event_runtime.agent.cost",
    "history": "event_runtime.agent.history",
}
CHECK_POLICY = "/opt/event/check_policy.py"


def resolve_command(action: str) -> Callable[[], int]:
    return import_module(COMMANDS[action]).main


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__.strip())
        return 0
    action = sys.argv[1]
    arguments = sys.argv[2:]
    if action == "check":
        os.execv(sys.executable, [sys.executable, CHECK_POLICY, *arguments])
        return 127
    if action not in COMMANDS:
        print(f"unknown event command: {action}", file=sys.stderr)
        print("run 'event --help' for available commands", file=sys.stderr)
        return 2
    sys.argv = [f"event {action}", *arguments]
    return resolve_command(action)()


if __name__ == "__main__":
    raise SystemExit(main())
