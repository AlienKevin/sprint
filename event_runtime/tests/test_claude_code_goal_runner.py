from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
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


def test_supervisor_recognizes_goal_runner_as_claude_process(tmp_path: Path) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    agent_logs = tmp_path / "agent"
    artifact_logs = tmp_path / "artifacts"
    for path in (durable, runtime, agent_logs, artifact_logs):
        path.mkdir()

    fake_claude = tmp_path / "fake-claude.py"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import os, signal, time\n"
        "signal.signal(signal.SIGINT, lambda *_: os._exit(0))\n"
        "signal.signal(signal.SIGTERM, lambda *_: os._exit(0))\n"
        "while True: time.sleep(1)\n"
    )
    fake_claude.chmod(0o755)

    runner = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--claude-executable",
            str(fake_claude),
            "--bootstrap",
            str(agent_logs / "goal-bootstrap.json"),
            "--lifecycle",
            str(agent_logs / "goal-lifecycle.json"),
            "--",
            "--print",
            "/goal keep working",
        ],
        env={
            **os.environ,
            "SPRINT_RUNTIME_DIR": str(runtime),
            "SPRINT_DURABLE_DIR": str(durable),
            "SPRINT_RUN_ID": "test-claude-goal",
        },
        preexec_fn=os.setsid,
    )
    process_dir = runtime / "sprint-agent"
    process_dir.mkdir()
    start_time = Path(f"/proc/{runner.pid}/stat").read_text().split()[21]
    (process_dir / "agent-process").write_text(
        f"{runner.pid} {os.getpgid(runner.pid)} {start_time}\n"
    )
    supervisor = subprocess.Popen(
        [
            "bash",
            str(ROOT / "event_runtime/container/sprint-agent-supervisor.sh"),
            "--run-id",
            "test-claude-goal",
            "--agent-kind",
            "claude-code",
            "--poll-seconds",
            "1",
            "--durable-dir",
            str(durable),
            "--runtime-dir",
            str(runtime),
            "--agent-log-dir",
            str(agent_logs),
            "--artifact-log-dir",
            str(artifact_logs),
            "--budget-watchdog-bin",
            "/bin/true",
        ]
    )
    try:
        first_seen = (
            durable
            / "runs/test-claude-goal/supervisor/first-claude-code-seen"
        )
        deadline = time.time() + 10
        while not first_seen.exists() and time.time() < deadline:
            time.sleep(0.1)
        assert first_seen.exists()
        heartbeat_path = (
            durable / "runs/test-claude-goal/supervisor/heartbeat.json"
        )
        heartbeat: dict[str, Any] = {}
        while time.time() < deadline:
            heartbeat = json.loads(heartbeat_path.read_text())
            if heartbeat.get("agent_seen") is True:
                break
            time.sleep(0.1)
        assert heartbeat["agent_seen"] is True
        assert heartbeat["agent_pid"] == runner.pid
    finally:
        supervisor.kill()
        supervisor.wait()
        if runner.poll() is None:
            os.killpg(os.getpgid(runner.pid), signal.SIGKILL)
        runner.wait()
