"""Deterministic Codex goal bootstrap tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.codex import Codex
from harbor.agents.installed.codex_goal_bootstrap import (
    AppServerError,
    _validated_goal,
)


def test_split_goal_instruction() -> None:
    objective = "Keep improving the policy until stopped.\n\nTask:\nRun fast."
    assert Codex._split_goal_instruction(f"/goal {objective}") == (
        objective,
        objective,
    )


def test_non_goal_instruction_is_unchanged() -> None:
    assert Codex._split_goal_instruction("/goalkeeper is a path") == (
        None,
        "/goalkeeper is a path",
    )
    assert Codex._split_goal_instruction("ordinary task") == (
        None,
        "ordinary task",
    )


def test_empty_goal_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty objective"):
        Codex._split_goal_instruction("/goal   ")


def test_validated_goal_requires_active_unbudgeted_exact_goal() -> None:
    expected = {
        "goal": {
            "threadId": "thread-1",
            "objective": "objective",
            "status": "active",
            "tokenBudget": None,
        }
    }
    assert _validated_goal(
        expected, thread_id="thread-1", objective="objective"
    ) == expected["goal"]

    for field, value in (
        ("threadId", "thread-2"),
        ("objective", "changed"),
        ("status", "complete"),
        ("tokenBudget", 1),
    ):
        malformed = json.loads(json.dumps(expected))
        malformed["goal"][field] = value
        with pytest.raises(AppServerError):
            _validated_goal(
                malformed, thread_id="thread-1", objective="objective"
            )


@pytest.mark.asyncio
async def test_first_goal_is_bootstrapped_before_codex_exec(
    temp_dir, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-sealed-test")
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_FORCE_AUTH_JSON", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    agent = Codex(
        logs_dir=temp_dir,
        model_name="openai/gpt-5.6-luna",
        reasoning_effort="max",
        extra_env={
            "SPRINT_OPENROUTER_LEDGER_REQUIRED": "1",
            "CODEX_GOAL_BOOTSTRAP_MODEL": "@preset/pinned-luna",
            "CODEX_GOAL_BOOTSTRAP_EXPECTED_PROVIDER": "sprint_openrouter",
            "CODEX_GOAL_BOOTSTRAP_PREPARE_SCRIPT": "/opt/apply-pinned-luna.sh",
        },
    )
    environment = AsyncMock()
    environment.default_user = "agent"
    environment.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")

    await agent.run("/goal deterministic objective", environment, AsyncMock())

    environment.upload_file.assert_called_once()
    uploaded_source, uploaded_destination = environment.upload_file.call_args.args
    assert uploaded_source.name == "codex_goal_bootstrap.py"
    assert uploaded_destination == "/tmp/codex-secrets/codex_goal_bootstrap.py"

    commands = [
        call.kwargs["command"]
        for call in environment.exec.call_args_list
        if "command" in call.kwargs
    ]
    execution = next(command for command in commands if "codex exec" in command)
    setup = next(command for command in commands if "/opt/apply-pinned-luna.sh" in command)
    assert "bash /opt/apply-pinned-luna.sh" in setup
    assert execution.index("codex_goal_bootstrap.py") < execution.index("codex exec")
    assert "--model @preset/pinned-luna" in execution
    assert "--expected-provider sprint_openrouter" in execution
    assert "codex exec resume " in execution
    assert '"$sprint_codex_thread_id"' in execution
    assert "SPRINT_CODEX_GOAL_PERSIST=1" in execution
    assert "SPRINT_CODEX_GOAL_THREAD_ID" in execution
    assert "SPRINT_CODEX_GOAL_RECEIPT" in execution
    assert "/logs/agent/goal-bootstrap.json" in execution
    assert "/goal deterministic objective" not in execution

    execution_call = next(
        call
        for call in environment.exec.call_args_list
        if "codex exec" in call.kwargs.get("command", "")
    )
    assert execution_call.kwargs["env"]["SPRINT_CODEX_GOAL_OBJECTIVE"] == (
        "deterministic objective"
    )
    assert execution_call.kwargs["env"]["OPENROUTER_API_KEY"] == (
        "sk-or-sealed-test"
    )
    assert json.loads(
        execution_call.kwargs["env"]["SPRINT_CODEX_APP_SERVER_ARGS_JSON"]
    ) == ["-c", "model_reasoning_effort=max"]


@pytest.mark.asyncio
async def test_openrouter_ledger_requires_sealed_key(temp_dir, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    agent = Codex(
        logs_dir=temp_dir,
        model_name="openai/gpt-5.6-luna",
        extra_env={"SPRINT_OPENROUTER_LEDGER_REQUIRED": "1"},
    )
    environment = AsyncMock()

    with pytest.raises(ValueError, match="requires OPENROUTER_API_KEY"):
        await agent.run("ordinary task", environment, AsyncMock())

    environment.exec.assert_not_called()


@pytest.mark.asyncio
async def test_resume_does_not_recreate_existing_goal(temp_dir, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_FORCE_AUTH_JSON", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    agent = Codex(logs_dir=temp_dir, model_name="openai/gpt-5.6-luna")
    environment = AsyncMock()
    environment.default_user = "agent"
    environment.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")

    await agent.resume("/goal persistent objective", environment, AsyncMock())

    environment.upload_file.assert_not_called()
    commands = "\n".join(
        call.kwargs["command"]
        for call in environment.exec.call_args_list
        if "command" in call.kwargs
    )
    assert "codex exec resume --last" in commands
    assert "codex_goal_bootstrap.py" not in commands
    assert "/goal persistent objective" not in commands
