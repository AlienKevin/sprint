"""Unit tests for Codex ATIF trajectory conversion."""

import hashlib
import json

from harbor.agents.installed.codex import Codex
from harbor.models.agent.context import AgentContext


class TestCodexTrajectoryConversion:
    def test_tool_call_without_message_does_not_fabricate_assistant_text(
        self, temp_dir
    ):
        agent = Codex(logs_dir=temp_dir, model_name="openai/o3")

        step = agent._convert_event_to_step(
            {
                "kind": "tool_call",
                "timestamp": "2026-01-01T00:00:00Z",
                "call_id": "call_1",
                "tool_name": "shell",
                "arguments": {"command": "pwd"},
                "output": "/workspace",
            },
            step_id=1,
        )

        assert step.message == ""
        assert step.tool_calls is not None
        assert step.tool_calls[0].function_name == "shell"
        assert step.observation is not None
        assert step.observation.results[0].content == "/workspace"

    def test_converted_trajectory_emits_latest_atif_version(self, temp_dir):
        agent = Codex(logs_dir=temp_dir, model_name="openai/o3")
        session_dir = temp_dir / "codex-session"
        session_dir.mkdir()
        events = [
            {"type": "session_meta", "payload": {"id": "session-1"}},
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done."}],
                },
            },
        ]
        (session_dir / "session.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events) + "\n"
        )

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        assert trajectory.schema_version == "ATIF-v1.7"


class TestCodexApiCallGrouping:
    """ATIF v1.7: one step per model API call, bounded by token_count events."""

    def _write_session(self, temp_dir, events):
        session_dir = temp_dir / "codex-session"
        session_dir.mkdir(exist_ok=True)
        (session_dir / "session.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events) + "\n"
        )
        return session_dir

    def _token_count_event(self, prompt, completion, total):
        return {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": prompt,
                        "output_tokens": completion,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "reasoning_output_tokens": 0,
                        "total_tokens": total,
                    },
                    "total_token_usage": {
                        "input_tokens": prompt,
                        "output_tokens": completion,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "reasoning_output_tokens": 0,
                        "total_tokens": total,
                    },
                },
            },
        }

    def test_one_step_per_api_call_with_bundled_tool_calls(self, temp_dir):
        agent = Codex(logs_dir=temp_dir, model_name="openai/o3")
        events = [
            {"type": "session_meta", "payload": {"id": "session-1"}},
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:00Z",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Plan the fix."}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Inspecting."}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:02Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "arguments": json.dumps({"command": "ls"}),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:03Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "README.md",
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:04Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "shell",
                    "arguments": json.dumps({"command": "cat README.md"}),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:05Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "hello",
                },
            },
            self._token_count_event(100, 20, 120),
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:06Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done."}],
                },
            },
            self._token_count_event(150, 5, 155),
        ]
        session_dir = self._write_session(temp_dir, events)

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        agent_steps = [s for s in trajectory.steps if s.source == "agent"]
        # Two API calls -> exactly two agent steps, not 1 message + 2 tool steps.
        assert len(agent_steps) == 2

        first, second = agent_steps
        assert first.message == "Inspecting."
        assert first.reasoning_content == "Plan the fix."
        assert first.tool_calls is not None and len(first.tool_calls) == 2
        assert [tc.tool_call_id for tc in first.tool_calls] == ["call_1", "call_2"]
        assert first.tool_calls[0].extra == {
            "started_at": "2026-01-01T00:00:02Z",
            "finished_at": "2026-01-01T00:00:03Z",
            "duration_ms": 1000,
        }
        assert first.tool_calls[1].extra == {
            "started_at": "2026-01-01T00:00:04Z",
            "finished_at": "2026-01-01T00:00:05Z",
            "duration_ms": 1000,
        }
        assert first.observation is not None
        assert [r.content for r in first.observation.results] == [
            "README.md",
            "hello",
        ]
        assert first.llm_call_count == 1
        assert first.metrics is not None and first.metrics.prompt_tokens == 100
        assert first.extra is not None
        assert first.extra["api_call_id"] == "api_call_1"

        assert second.message == "Done."
        assert second.llm_call_count == 1
        assert second.metrics is not None and second.metrics.prompt_tokens == 150

    def test_user_messages_are_never_grouped(self, temp_dir):
        agent = Codex(logs_dir=temp_dir, model_name="openai/o3")
        events = [
            {"type": "session_meta", "payload": {"id": "session-2"}},
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Fix the bug."}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "On it."}],
                },
            },
            self._token_count_event(50, 10, 60),
        ]
        session_dir = self._write_session(temp_dir, events)

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        assert [s.source for s in trajectory.steps] == ["user", "agent"]
        user_step, agent_step = trajectory.steps
        assert user_step.llm_call_count is None
        assert agent_step.llm_call_count == 1

    def test_terra_preserves_per_request_cost_inputs_and_pricing(self, temp_dir):
        agent = Codex(logs_dir=temp_dir, model_name="openai/gpt-5.6-terra")
        first_usage = {
            "input_tokens": 130_000,
            "cached_input_tokens": 40_000,
            "cache_write_input_tokens": 30_000,
            "output_tokens": 10_000,
            "reasoning_output_tokens": 2_000,
            "total_tokens": 140_000,
        }
        events = [
            {"type": "session_meta", "payload": {"id": "terra-session"}},
            {
                "type": "turn_context",
                "timestamp": "2026-08-08T00:00:00Z",
                "payload": {
                    "turn_id": "turn-1",
                    "model": "gpt-5.6-terra",
                    "effort": "max",
                    "service_tier": "default",
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-08T00:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done."}],
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-08-08T00:00:02Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": first_usage,
                        "total_token_usage": first_usage,
                        "model_context_window": 1_050_000,
                    },
                },
            },
        ]
        session_dir = self._write_session(temp_dir, events)

        trajectory = agent._convert_events_to_trajectory(session_dir)

        assert trajectory is not None
        step = next(step for step in trajectory.steps if step.source == "agent")
        assert step.metrics is not None
        assert step.metrics.prompt_tokens == 130_000
        assert step.metrics.cached_tokens == 40_000
        assert step.metrics.cost_usd == 0.323
        assert step.metrics.extra is not None
        request_audit = step.metrics.extra["cost_audit"]
        assert request_audit["cache_write_input_tokens"] == 30_000
        assert request_audit["ordinary_uncached_input_tokens"] == 60_000
        assert request_audit["reasoning_effort"] == "max"
        assert request_audit["service_tier"] == "default"
        assert request_audit["cost_reconstruction_status"] == "complete"

        assert trajectory.final_metrics is not None
        assert trajectory.final_metrics.total_cost_usd == 0.323
        assert trajectory.final_metrics.extra is not None
        assert trajectory.final_metrics.extra["cache_write_tokens"] == 30_000
        summary = trajectory.final_metrics.extra["usage_audit"]
        assert summary["cost_reconstruction_complete"] is True
        assert summary["request_count"] == 1


def test_codex_pins_default_service_tier_for_reproducible_pricing(temp_dir):
    agent = Codex(
        logs_dir=temp_dir,
        model_name="openai/gpt-5.6-terra",
        service_tier="default",
    )

    assert agent._resolved_flags["service_tier"] == "default"
    assert "-c service_tier=default" in agent.build_cli_flags()


def test_populate_context_writes_checksummed_usage_audit(temp_dir):
    agent = Codex(logs_dir=temp_dir, model_name="openai/gpt-5.6-terra")
    session_dir = temp_dir / "sessions" / "2026" / "08" / "08"
    session_dir.mkdir(parents=True)
    raw_usage = {
        "input_tokens": 1_300,
        "cached_input_tokens": 400,
        "cache_write_input_tokens": 300,
        "output_tokens": 100,
        "reasoning_output_tokens": 20,
        "total_tokens": 1_400,
    }
    events = [
        {"type": "session_meta", "payload": {"id": "terra-session"}},
        {
            "type": "turn_context",
            "payload": {
                "model": "gpt-5.6-terra",
                "effort": "high",
                "service_tier": "default",
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-08-08T00:00:01Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        },
        {
            "type": "event_msg",
            "timestamp": "2026-08-08T00:00:02Z",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": raw_usage,
                    "total_token_usage": raw_usage,
                    "model_context_window": 1_050_000,
                },
            },
        },
    ]
    source_path = session_dir / "rollout.jsonl"
    source_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    context = AgentContext()

    agent.populate_context_post_run(context)

    audit = json.loads((temp_dir / "usage-audit.json").read_text())
    assert audit["cost_reconstruction_complete"] is True
    assert (
        audit["provenance"]["source_session_path"]
        == "usage-provenance/source-session.jsonl"
    )
    assert len(audit["provenance"]["source_session_sha256"]) == 64
    assert (
        audit["provenance"]["trajectory_path"]
        == "usage-provenance/trajectory.json"
    )
    assert len(audit["provenance"]["trajectory_sha256"]) == 64
    source_path.write_text("later mutable session tail\n")
    (temp_dir / "trajectory.json").write_text("later mutable trajectory\n")
    assert hashlib.sha256(
        (temp_dir / audit["provenance"]["source_session_path"]).read_bytes()
    ).hexdigest() == audit["provenance"]["source_session_sha256"]
    assert hashlib.sha256(
        (temp_dir / audit["provenance"]["trajectory_path"]).read_bytes()
    ).hexdigest() == audit["provenance"]["trajectory_sha256"]
    assert context.n_input_tokens == 1_300
    assert context.n_cache_tokens == 400
    assert context.n_output_tokens == 100
    assert context.cost_usd == audit["calculated_api_usage_usd"]
