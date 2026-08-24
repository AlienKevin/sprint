from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_runtime.export import trajectory


def write_fixture(state: Path) -> None:
    run_id = state.name
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": "openai/test-model",
                "agent_kind": "codex",
                "reasoning_effort": "high",
                "created_at": "2026-08-23T08:00:00Z",
            }
        )
    )
    (state / "telemetry").mkdir()
    (state / "telemetry" / "unified-timeline.json").write_text(
        json.dumps(
            {
                "usage_summary": {"input_tokens": 123, "output_tokens": 45},
                "comparison_summary": {"final_agent_total_cost_usd": 9.99},
                "coverage": {
                    "cpu_metric_coverage": {
                        "attempts": [
                            {
                                "cpu_attempt": 1,
                                "start_epoch_ms": 1787472000000,
                                "end_epoch_ms": 1787472060000,
                            },
                            {
                                "cpu_attempt": 2,
                                "start_epoch_ms": 1787472120000,
                                "end_epoch_ms": 1787472240000,
                            },
                        ]
                    }
                },
            }
        )
    )
    for attempt in (1, 2):
        target = (
            state
            / "trace"
            / "reconstructed"
            / f"cpu-attempt-{attempt:03d}"
            / f"session-{attempt}"
        )
        target.mkdir(parents=True)
        steps = [
            {
                "step_id": 0,
                "timestamp": f"2026-08-23T08:0{attempt}:00Z",
                "source": "system",
                "message": "private bootstrap OPENROUTER_API_KEY=sk-or-v1-private",
            },
            {
                "step_id": 1,
                "timestamp": f"2026-08-23T08:0{attempt}:01Z",
                "source": "user",
                "message": "<environment_context>private</environment_context>",
            },
            {
                "step_id": 2,
                "timestamp": f"2026-08-23T08:0{attempt}:02Z",
                "source": "user",
                "message": "build the fastest policy",
            },
            {
                "step_id": 3,
                "timestamp": f"2026-08-23T08:0{attempt}:03Z",
                "source": "agent",
                "model_name": "test-model",
                "reasoning_content": "I should inspect before acting.",
                "message": "Inspecting the workspace.",
                "tool_calls": [
                    {
                        "tool_call_id": "raw-call-id",
                        "function_name": "exec_command",
                        "arguments": {
                            "cmd": "printenv",
                            "api_key": "sk-or-v1-secret",
                        },
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "raw-call-id",
                            "content": "OPENROUTER_API_KEY=sk-or-v1-secret\nOK",
                        }
                    ]
                },
                "metrics": {
                    "cost_usd": 0.2,
                    "prompt_tokens": 10,
                    "cached_tokens": 8,
                    "completion_tokens": 2,
                    "extra": {"private": "omit"},
                },
            },
        ]
        (target / "trajectory.json").write_text(
            json.dumps({"schema_version": "1.0", "steps": steps})
        )


def test_build_public_trajectory_redacts_and_preserves_attempts(tmp_path: Path) -> None:
    state = tmp_path / "trajectory-fixture"
    web = tmp_path / "web"
    write_fixture(state)

    payload = trajectory.build_public_trajectory(state, web_dir=web)

    assert payload is not None
    assert payload["summary"]["attempt_count"] == 2
    assert payload["summary"]["step_count"] == 4
    assert payload["summary"]["omitted_bootstrap_steps"] == 4
    assert [row["attempt"] for row in payload["attempts"]] == [1, 2]
    assert payload["attempts"][0]["started_at"] == "2026-08-23T08:00:00.000Z"
    assert payload["attempts"][0]["ended_at"] == "2026-08-23T08:01:00.000Z"
    assert payload["steps"][0]["message"] == "build the fastest policy"
    assert payload["steps"][0]["public_step_id"] == 1
    assert payload["steps"][0]["attempt_step_id"] == 2
    assert payload["steps"][1]["public_step_id"] == 2
    agent_step = payload["steps"][1]
    assert agent_step["reasoning_content"] == "I should inspect before acting."
    assert agent_step["tool_calls"][0]["arguments"]["api_key"] == "[REDACTED]"
    assert agent_step["tool_calls"][0]["tool_call_id"] != "raw-call-id"
    assert agent_step["observation"]["results"][0]["source_call_id"] == agent_step["tool_calls"][0]["tool_call_id"]
    serialized = json.dumps(payload)
    assert "sk-or-v1" not in serialized
    assert "private bootstrap" not in serialized
    assert "<environment_context>" not in serialized
    assert payload["summary"]["final_agent_cost_usd"] == 9.99
    assert payload["summary"]["input_tokens"] == 123

    index = json.loads((web / "data" / "trajectories" / "index.json").read_text())
    assert index["runs"][0]["path"] == "/data/trajectories/trajectory-fixture.json"


def test_redact_handles_headers_urls_and_images() -> None:
    value = {
        "Authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
        "url": "https://example.test/?token=super-secret&ok=1",
        "parts": [{"type": "image", "data": "base64-secret"}],
    }

    result = trajectory.redact(value)

    assert result["Authorization"] == "[REDACTED]"
    assert "super-secret" not in result["url"]
    assert result["parts"][0]["content"] == "[IMAGE OMITTED FROM PUBLIC VIEW]"


def test_build_public_trajectory_from_deepseek_harness_events(tmp_path: Path) -> None:
    state = tmp_path / "deepseek-trajectory"
    web = tmp_path / "web"
    state.mkdir()
    (state / "run.json").write_text(
        json.dumps(
            {
                "run_id": state.name,
                "model": "deepseek/deepseek-v4-flash-vision-exp",
                "agent_kind": "deepseek-harness",
                "created_at": "2026-08-24T06:00:00Z",
            }
        )
    )
    (state / "telemetry").mkdir()
    (state / "telemetry" / "unified-timeline.json").write_text("{}")
    chunks = (
        state
        / "durable-trace"
        / "raw"
        / "cpu-attempt-001"
        / "deepseek-harness"
        / "source-one"
        / "chunks"
    )
    chunks.mkdir(parents=True)
    events = [
        {
            "type": "user/message",
            "seq": 5,
            "time": 1787551204669,
            "data": {"content": [{"type": "text", "text": "build fastest"}]},
        },
        {
            "type": "assistant/message",
            "seq": 10,
            "time": 1787551205000,
            "data": {
                "step": 1,
                "message": {
                    "content": [
                        {
                            "type": "reasoning",
                            "text": "I should inspect the policy contract first.",
                        },
                        {"type": "text", "text": "Inspecting."},
                        {
                            "type": "tool-call",
                            "id": "raw-call",
                            "name": "bash",
                            "arguments": '{"command":"pwd"}',
                        },
                    ],
                    "source": {
                        "kind": "model",
                        "model": "deepseek/deepseek-v4-flash-vision-exp",
                    },
                },
                "usage": {
                    "inputTokens": 3,
                    "cacheReadTokens": 7,
                    "outputTokens": 2,
                },
            },
        },
        {
            "type": "tool/result",
            "seq": 12,
            "time": 1787551205100,
            "data": {
                "step": 1,
                "message": {
                    "source": {"kind": "tool", "callId": "raw-call"},
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "raw-call",
                            "content": [{"type": "text", "text": "/app"}],
                        }
                    ],
                },
            },
        },
    ]
    rows = [
        {
            "schema_version": 1,
            "method": "session.event",
            "payload": {"event": event},
        }
        for event in events
    ]
    (chunks / "000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    payload = trajectory.build_public_trajectory(state, web_dir=web)

    assert payload is not None
    assert payload["summary"]["step_count"] == 2
    assert payload["steps"][0]["source"] == "user"
    assert payload["steps"][0]["message"] == "build fastest"
    agent = payload["steps"][1]
    assert agent["reasoning_content"] == (
        "I should inspect the policy contract first."
    )
    assert agent["message"] == "Inspecting."
    assert agent["metrics"] == {
        "prompt_tokens": 10,
        "cached_tokens": 7,
        "completion_tokens": 2,
    }
    assert agent["tool_calls"][0]["function_name"] == "bash"
    assert agent["tool_calls"][0]["arguments"] == {"command": "pwd"}
    assert agent["observation"]["results"][0]["content"] == "/app"
    assert agent["observation"]["results"][0]["source_call_id"] == agent["tool_calls"][0]["tool_call_id"]
