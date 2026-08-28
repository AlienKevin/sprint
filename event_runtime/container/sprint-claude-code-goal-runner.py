#!/usr/bin/env python3
"""Keep one Claude Code goal session alive until the operator stops it.

Claude Code's ``--print`` mode exits after a successful model turn, even when
the session has an unmet native ``/goal``.  Benchmark goals are host-owned and
remain active until the trusted budget/operator stop, so a normal ``end_turn``
must resume the same Claude session rather than terminate the CPU sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Sequence
import uuid


DEFAULT_CONTINUATION_PROMPT = (
    "Continue working toward the active operator-controlled goal. The operator "
    "has not stopped this run. Do not summarize or stop; continue improving and "
    "submitting candidate policies."
)


class GoalRunnerError(RuntimeError):
    """The host-owned Claude goal lifecycle could not be enforced."""


def write_json(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "updated_at_epoch_s": time.time(),
        **payload,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _prompt_index(arguments: Sequence[str]) -> int:
    if "--print" not in arguments and "-p" not in arguments:
        raise GoalRunnerError("Claude Code goal arguments require --print")
    if not arguments:
        raise GoalRunnerError("Claude Code goal arguments have no prompt")
    prompt_index = len(arguments) - 1
    if arguments[prompt_index].startswith("-"):
        raise GoalRunnerError("Claude Code goal arguments have no final prompt")
    return prompt_index


def _print_index(arguments: Sequence[str]) -> int:
    for option in ("--print", "-p"):
        try:
            return arguments.index(option)
        except ValueError:
            continue
    raise GoalRunnerError("Claude Code goal arguments require --print")


def initial_args(arguments: Sequence[str], session_id: str) -> list[str]:
    result = list(arguments)
    _prompt_index(result)
    if any(item in {"--session-id", "--resume", "-r", "--continue", "-c"} for item in result):
        raise GoalRunnerError("Claude Code goal runner requires a fresh session")
    print_index = _print_index(result)
    result[print_index:print_index] = ["--session-id", session_id]
    return result


def continuation_args(
    arguments: Sequence[str], session_id: str, prompt: str
) -> list[str]:
    result = list(arguments)
    prompt_index = _prompt_index(result)
    result[prompt_index] = prompt
    print_index = _print_index(result)
    result[print_index:print_index] = ["--resume", session_id]
    return result


class SignalRelay:
    def __init__(self) -> None:
        self.child: subprocess.Popen[bytes] | None = None
        self.signum: int | None = None

    def handle(self, signum: int, _frame: Any) -> None:
        self.signum = signum
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signum)


def run_turn(command: Sequence[str], relay: SignalRelay) -> int:
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    relay.child = child
    try:
        return child.wait()
    finally:
        relay.child = None


def stop_requested() -> bool:
    runtime = Path(os.environ.get("SPRINT_RUNTIME_DIR", "/run"))
    durable = Path(os.environ.get("SPRINT_DURABLE_DIR", "/durable"))
    run_id = os.environ.get("SPRINT_RUN_ID", "")
    markers = [runtime / "sprint-stop"]
    if run_id:
        markers.append(durable / "runs" / run_id / "BUDGET_STOP_REQUESTED.json")
    return any(path.exists() for path in markers)


def _write_interrupted(
    lifecycle_path: Path, *, session_id: str, turns: int, signum: int
) -> int:
    write_json(
        lifecycle_path,
        session_id=session_id,
        goal_owner="operator",
        goal_status="active",
        completed_turns=turns,
        runner_state="interrupted",
        signal=signum,
    )
    return 128 + signum


def run_goal_loop(
    *,
    claude_executable: str,
    claude_arguments: Sequence[str],
    bootstrap_path: Path,
    lifecycle_path: Path,
    continuation_prompt: str,
    relay: SignalRelay,
    session_id: str | None = None,
    turn_runner: Callable[[Sequence[str], SignalRelay], int] = run_turn,
) -> int:
    session_id = session_id or str(uuid.uuid4())
    first_arguments = initial_args(claude_arguments, session_id)
    later_arguments = continuation_args(
        claude_arguments, session_id, continuation_prompt
    )
    original_prompt = claude_arguments[_prompt_index(claude_arguments)]
    write_json(
        bootstrap_path,
        session_id=session_id,
        goal_owner="operator",
        goal_status="active",
        objective_sha256=hashlib.sha256(original_prompt.encode()).hexdigest(),
    )
    turns = 0
    write_json(
        lifecycle_path,
        session_id=session_id,
        goal_owner="operator",
        goal_status="active",
        completed_turns=turns,
        runner_state="running",
    )

    while True:
        if relay.signum is not None:
            return _write_interrupted(
                lifecycle_path,
                session_id=session_id,
                turns=turns,
                signum=relay.signum,
            )
        command_arguments = first_arguments if turns == 0 else later_arguments
        rc = turn_runner([claude_executable, *command_arguments], relay)
        turns += 1
        if relay.signum is not None:
            return _write_interrupted(
                lifecycle_path,
                session_id=session_id,
                turns=turns,
                signum=relay.signum,
            )
        if stop_requested():
            # The sealed API proxy writes the trusted stop marker before it
            # rejects a request that would exceed the benchmark budget. Claude
            # Code reports that intentional HTTP 402 as a non-zero turn exit.
            # Keep the operator-owned runner alive for the supervisor signal
            # instead of racing the stop path and misclassifying the run as an
            # infrastructure failure.
            lifecycle = {
                "session_id": session_id,
                "goal_owner": "operator",
                "goal_status": "active",
                "completed_turns": turns,
                "runner_state": "waiting_for_stop",
            }
            if rc != 0:
                lifecycle["turn_exit_code"] = rc
            write_json(lifecycle_path, **lifecycle)
            while stop_requested() and relay.signum is None:
                time.sleep(0.25)
            if relay.signum is not None:
                return _write_interrupted(
                    lifecycle_path,
                    session_id=session_id,
                    turns=turns,
                    signum=relay.signum,
                )
        if rc != 0:
            write_json(
                lifecycle_path,
                session_id=session_id,
                goal_owner="operator",
                goal_status="active",
                completed_turns=turns,
                runner_state="invalid_infrastructure",
                failure_code="goal_turn_failed",
                detail=f"Claude Code goal turn exited with status {rc}",
                turn_exit_code=rc,
            )
            return rc

        # A normal Claude end_turn never completes an operator-owned goal.
        # Preserve a signalable process until the supervisor delivers the stop.
        write_json(
            lifecycle_path,
            session_id=session_id,
            goal_owner="operator",
            goal_status="active",
            completed_turns=turns,
            runner_state="running",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claude-executable", required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument(
        "--continuation-prompt", default=DEFAULT_CONTINUATION_PROMPT
    )
    parser.add_argument("claude_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    claude_arguments = list(args.claude_arguments)
    if claude_arguments and claude_arguments[0] == "--":
        claude_arguments.pop(0)
    try:
        relay = SignalRelay()
        signal.signal(signal.SIGINT, relay.handle)
        signal.signal(signal.SIGTERM, relay.handle)
        return run_goal_loop(
            claude_executable=args.claude_executable,
            claude_arguments=claude_arguments,
            bootstrap_path=args.bootstrap,
            lifecycle_path=args.lifecycle,
            continuation_prompt=args.continuation_prompt,
            relay=relay,
        )
    except GoalRunnerError as exc:
        write_json(
            args.lifecycle,
            goal_owner="operator",
            goal_status="active",
            runner_state="invalid_infrastructure",
            failure_code="goal_runner_failed",
            detail=str(exc),
        )
        print(f"Claude Code goal runner failed: {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
