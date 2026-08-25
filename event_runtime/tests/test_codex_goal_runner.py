from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "event_runtime/container/sprint-codex-goal-runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("sprint_codex_goal_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_continuation_replaces_only_prompt() -> None:
    runner = load_runner()
    original = ["exec", "resume", "--model", "m", "thread", "--", "first task"]
    assert runner.continuation_args(original, "keep going") == [
        "exec",
        "resume",
        "--model",
        "m",
        "thread",
        "--",
        "keep going",
    ]


def test_goal_loop_resumes_active_thread_until_terminal(tmp_path: Path) -> None:
    runner = load_runner()
    receipt = tmp_path / "receipt.json"
    lifecycle = tmp_path / "lifecycle.json"
    receipt.write_text("{}")
    commands: list[list[str]] = []
    statuses = iter(["active", "active", "complete"])

    def turn(command: Sequence[str], _relay: Any) -> int:
        commands.append(list(command))
        return 0

    def status(**_kwargs: Any) -> str:
        return next(statuses)

    rc = runner.run_goal_loop(
        codex_executable="/bin/codex",
        codex_arguments=["exec", "resume", "thread", "--", "original"],
        thread_id="thread",
        receipt_path=receipt,
        lifecycle_path=lifecycle,
        app_server_args=[],
        continuation_prompt="continue",
        relay=runner.SignalRelay(),
        turn_runner=turn,
        status_reader=status,
    )
    assert rc == 0
    assert [command[-1] for command in commands] == [
        "original",
        "continue",
        "continue",
    ]
    assert runner.json.loads(lifecycle.read_text())["goal_status"] == "complete"


def test_goal_loop_propagates_failed_turn_without_resuming(tmp_path: Path) -> None:
    runner = load_runner()
    status_called = False

    def status(**_kwargs: Any) -> str:
        nonlocal status_called
        status_called = True
        return "active"

    rc = runner.run_goal_loop(
        codex_executable="/bin/codex",
        codex_arguments=["exec", "resume", "thread", "--", "original"],
        thread_id="thread",
        receipt_path=tmp_path / "receipt.json",
        lifecycle_path=tmp_path / "lifecycle.json",
        app_server_args=[],
        continuation_prompt="continue",
        relay=runner.SignalRelay(),
        turn_runner=lambda _command, _relay: 42,
        status_reader=status,
    )
    assert rc == 42
    assert not status_called


def test_continuation_requires_prompt_separator() -> None:
    runner = load_runner()
    with pytest.raises(runner.GoalRunnerError, match="prompt separator"):
        runner.continuation_args(["exec", "resume", "thread"], "continue")
