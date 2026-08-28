from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "event_runtime/container/sprint-claude-code-goal-runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("sprint_claude_goal_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_goal_loop_resumes_same_session_after_clean_end_turn(tmp_path: Path) -> None:
    runner = load_runner()
    commands: list[list[str]] = []
    relay = runner.SignalRelay()

    def turn(command: Sequence[str], _relay: Any) -> int:
        commands.append(list(command))
        if len(commands) == 2:
            relay.signum = 15
        return 0

    rc = runner.run_goal_loop(
        claude_executable="/bin/claude",
        claude_arguments=["--verbose", "--print", "/goal keep working"],
        bootstrap_path=tmp_path / "bootstrap.json",
        lifecycle_path=tmp_path / "lifecycle.json",
        continuation_prompt="continue",
        relay=relay,
        session_id="11111111-1111-4111-8111-111111111111",
        turn_runner=turn,
    )

    assert rc == 143
    assert commands == [
        [
            "/bin/claude",
            "--verbose",
            "--session-id",
            "11111111-1111-4111-8111-111111111111",
            "--print",
            "/goal keep working",
        ],
        [
            "/bin/claude",
            "--verbose",
            "--resume",
            "11111111-1111-4111-8111-111111111111",
            "--print",
            "continue",
        ],
    ]
    lifecycle = runner.json.loads((tmp_path / "lifecycle.json").read_text())
    assert lifecycle["goal_status"] == "active"
    assert lifecycle["runner_state"] == "interrupted"
    assert lifecycle["completed_turns"] == 2


def test_failed_turn_records_infrastructure_failure(tmp_path: Path) -> None:
    runner = load_runner()
    rc = runner.run_goal_loop(
        claude_executable="/bin/claude",
        claude_arguments=["--print", "/goal keep working"],
        bootstrap_path=tmp_path / "bootstrap.json",
        lifecycle_path=tmp_path / "lifecycle.json",
        continuation_prompt="continue",
        relay=runner.SignalRelay(),
        session_id="11111111-1111-4111-8111-111111111111",
        turn_runner=lambda _command, _relay: 42,
    )
    assert rc == 42
    lifecycle = runner.json.loads((tmp_path / "lifecycle.json").read_text())
    assert lifecycle["runner_state"] == "invalid_infrastructure"
    assert lifecycle["failure_code"] == "goal_turn_failed"


def test_goal_runner_rejects_resume_or_missing_print() -> None:
    runner = load_runner()
    with pytest.raises(runner.GoalRunnerError, match="fresh session"):
        runner.initial_args(
            ["--continue", "--print", "goal"],
            "11111111-1111-4111-8111-111111111111",
        )
    with pytest.raises(runner.GoalRunnerError, match="require --print"):
        runner.initial_args(
            ["goal"], "11111111-1111-4111-8111-111111111111"
        )


def test_wrapper_enables_stable_external_goal_runner() -> None:
    wrapper = (
        ROOT / "event_runtime/container/sprint-claude-code-exec-wrapper.sh"
    ).read_text()
    assert "SPRINT_CLAUDE_CODE_GOAL_MODE" in wrapper
    assert "sprint-claude-code-goal-runner.py" in wrapper
    assert '--bootstrap "$AGENT_LOG_DIR/goal-bootstrap.json"' in wrapper
    assert '--lifecycle "$AGENT_LOG_DIR/goal-lifecycle.json"' in wrapper
    assert 'setsid "${agent_command[@]}" &' in wrapper
