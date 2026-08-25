#!/usr/bin/env python3
"""Keep one Codex goal thread alive until the goal or operator stops it.

``codex exec resume`` performs one turn and exits successfully even when the
thread still has an active persistent goal.  The outer sandbox must remain the
same process allocation, so this runner executes successive turns inside one
stable process group and consults Codex app-server after every clean turn.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Sequence


DEFAULT_CONTINUATION_PROMPT = "Continue working toward the active goal."
DEFAULT_RPC_TIMEOUT_SECONDS = 30.0
TERMINAL_GOAL_STATUSES = {"complete", "blocked"}


class GoalRunnerError(RuntimeError):
    """The persistent goal lifecycle could not be verified."""


def write_lifecycle(path: Path, **payload: Any) -> None:
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


def _rpc_request(
    process: subprocess.Popen[str],
    method: str,
    params: dict[str, Any],
    *,
    request_id: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    if process.stdin is None or process.stdout is None:
        raise GoalRunnerError("app-server requires stdin and stdout pipes")
    process.stdin.write(
        json.dumps({"id": request_id, "method": method, "params": params}) + "\n"
    )
    process.stdin.flush()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GoalRunnerError(f"timed out waiting for {method}")
            if not selector.select(remaining):
                continue
            line = process.stdout.readline()
            if not line:
                raise GoalRunnerError(f"app-server exited while waiting for {method}")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") != request_id:
                continue
            if message.get("error") is not None:
                raise GoalRunnerError(f"app-server rejected {method}: {message['error']}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise GoalRunnerError(f"app-server returned no object for {method}")
            return result
    finally:
        selector.close()


def read_goal_status(
    *,
    codex_executable: str,
    app_server_args: Sequence[str],
    thread_id: str,
    receipt_path: Path,
    timeout_seconds: float = DEFAULT_RPC_TIMEOUT_SECONDS,
) -> str:
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GoalRunnerError(f"cannot read goal receipt: {exc}") from exc
    if receipt.get("thread_id") != thread_id:
        raise GoalRunnerError("goal receipt references a different thread")
    objective = receipt.get("objective")
    if not isinstance(objective, str) or not objective:
        raise GoalRunnerError("goal receipt has no objective")

    process = subprocess.Popen(
        [codex_executable, "app-server", "--stdio", *app_server_args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        _rpc_request(
            process,
            "initialize",
            {
                "clientInfo": {"name": "sprint-codex-goal-runner", "version": "1"},
                "capabilities": {"experimentalApi": True},
            },
            request_id=1,
            timeout_seconds=timeout_seconds,
        )
        if process.stdin is None:
            raise GoalRunnerError("app-server has no stdin")
        process.stdin.write(json.dumps({"method": "initialized", "params": {}}) + "\n")
        process.stdin.flush()
        result = _rpc_request(
            process,
            "thread/goal/get",
            {"threadId": thread_id},
            request_id=2,
            timeout_seconds=timeout_seconds,
        )
        goal = result.get("goal")
        if not isinstance(goal, dict):
            raise GoalRunnerError("goal lookup returned no goal")
        if goal.get("threadId") != thread_id or goal.get("objective") != objective:
            raise GoalRunnerError("goal lookup did not match the bootstrapped goal")
        status = goal.get("status")
        if status not in {"active", *TERMINAL_GOAL_STATUSES}:
            raise GoalRunnerError(f"unexpected goal status: {status!r}")
        return str(status)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def continuation_args(arguments: Sequence[str], prompt: str) -> list[str]:
    """Replace only the prompt following Codex's ``--`` separator."""
    try:
        separator = arguments.index("--")
    except ValueError as exc:
        raise GoalRunnerError("Codex exec arguments have no -- prompt separator") from exc
    return [*arguments[: separator + 1], prompt]


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


def run_goal_loop(
    *,
    codex_executable: str,
    codex_arguments: Sequence[str],
    thread_id: str,
    receipt_path: Path,
    lifecycle_path: Path,
    app_server_args: Sequence[str],
    continuation_prompt: str,
    relay: SignalRelay,
    turn_runner: Callable[[Sequence[str], SignalRelay], int] = run_turn,
    status_reader: Callable[..., str] = read_goal_status,
) -> int:
    arguments = list(codex_arguments)
    turns = 0
    write_lifecycle(
        lifecycle_path,
        thread_id=thread_id,
        goal_status="active",
        completed_turns=turns,
        runner_state="running",
    )
    while True:
        rc = turn_runner([codex_executable, *arguments], relay)
        turns += 1
        if relay.signum is not None:
            write_lifecycle(
                lifecycle_path,
                thread_id=thread_id,
                goal_status="active",
                completed_turns=turns,
                runner_state="interrupted",
                signal=relay.signum,
            )
            return 128 + relay.signum
        if rc != 0:
            write_lifecycle(
                lifecycle_path,
                thread_id=thread_id,
                goal_status="active",
                completed_turns=turns,
                runner_state="turn_failed",
                turn_exit_code=rc,
            )
            return rc

        status: str | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                status = status_reader(
                    codex_executable=codex_executable,
                    app_server_args=app_server_args,
                    thread_id=thread_id,
                    receipt_path=receipt_path,
                )
                break
            except GoalRunnerError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1)
        if status is None:
            raise GoalRunnerError(f"cannot verify persistent goal: {last_error}")
        write_lifecycle(
            lifecycle_path,
            thread_id=thread_id,
            goal_status=status,
            completed_turns=turns,
            runner_state="terminal" if status in TERMINAL_GOAL_STATUSES else "running",
        )
        if status in TERMINAL_GOAL_STATUSES:
            return 0

        # A durable stop is authoritative. Remain a live, signalable process
        # until the supervisor sends the expected interrupt and writes STOP_ACK.
        while stop_requested() and relay.signum is None:
            time.sleep(0.25)
        if relay.signum is not None:
            write_lifecycle(
                lifecycle_path,
                thread_id=thread_id,
                goal_status="active",
                completed_turns=turns,
                runner_state="interrupted",
                signal=relay.signum,
            )
            return 128 + relay.signum
        arguments = continuation_args(arguments, continuation_prompt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-executable", required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument(
        "--app-server-args-env", default="SPRINT_CODEX_APP_SERVER_ARGS_JSON"
    )
    parser.add_argument(
        "--continuation-prompt",
        default=DEFAULT_CONTINUATION_PROMPT,
    )
    parser.add_argument("codex_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    codex_arguments = list(args.codex_arguments)
    if codex_arguments and codex_arguments[0] == "--":
        codex_arguments.pop(0)
    try:
        raw_app_server_args = os.environ.get(args.app_server_args_env, "[]")
        app_server_args = json.loads(raw_app_server_args)
        if not isinstance(app_server_args, list) or not all(
            isinstance(item, str) for item in app_server_args
        ):
            raise GoalRunnerError("app-server args must be a JSON string array")
        relay = SignalRelay()
        signal.signal(signal.SIGINT, relay.handle)
        signal.signal(signal.SIGTERM, relay.handle)
        return run_goal_loop(
            codex_executable=args.codex_executable,
            codex_arguments=codex_arguments,
            thread_id=args.thread_id,
            receipt_path=args.receipt,
            lifecycle_path=args.lifecycle,
            app_server_args=app_server_args,
            continuation_prompt=args.continuation_prompt,
            relay=relay,
        )
    except (GoalRunnerError, json.JSONDecodeError) as exc:
        print(f"Codex goal runner failed: {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
