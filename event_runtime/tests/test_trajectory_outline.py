from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from event_runtime.export import trajectory_outline


def payload() -> dict:
    return {
        "schema_version": 1,
        "source_fingerprint": "f" * 64,
        "run": {"run_id": "run-a", "model": "deepseek/test"},
        "summary": {"step_count": 5},
        "steps": [
            {
                "step_id": f"a1-s{number}",
                "public_step_id": number,
                "timestamp": f"2026-08-25T00:00:0{number}Z",
                "reasoning_content": f"Reasoning for phase {number}",
                "tool_calls": [],
            }
            for number in range(1, 6)
        ],
    }


def authored() -> dict:
    return {
        "synopsis": (
            "The agent audits the verifier, then trains and evaluates a concrete "
            "PPO policy before recording the measured outcome."
        ),
        "chapters": [
            {
                "id": "audit-verifier",
                "start_step_id": "a1-s1",
                "end_step_id": "a1-s2",
                "title": "Audit the verifier contract",
                "summary": (
                    "The agent reads the policy interface and scoring constraints "
                    "before selecting a training method."
                ),
            },
            {
                "id": "train-ppo",
                "start_step_id": "a1-s3",
                "end_step_id": "a1-s5",
                "title": "Train and evaluate PPO",
                "summary": (
                    "The agent trains a PPO controller, evaluates the resulting "
                    "checkpoint, and records the measured result."
                ),
            },
        ],
    }


def test_build_artifact_anchors_every_chapter_to_trace() -> None:
    result = trajectory_outline.build_artifact(
        authored(),
        payload(),
        trajectory_sha256="a" * 64,
        model="gpt-5.6-sol",
        reasoning_effort="high",
        generated_at="2026-08-25T12:00:00Z",
    )

    assert result["schema_version"] == "trajectory-outline/v1"
    assert result["trajectory"]["source_fingerprint"] == "f" * 64
    assert result["generator"]["prompt_version"] == "rollout-outline/v2"
    assert result["generator"]["source_access"] == "direct-public-trajectory/v1"
    assert result["chapters"][0]["start"] == {
        "step_id": "a1-s1",
        "public_step_id": 1,
    }
    assert result["chapters"][-1]["end"] == {
        "step_id": "a1-s5",
        "public_step_id": 5,
    }


def test_validation_rejects_gaps_between_model_chapters() -> None:
    broken = authored()
    broken["chapters"][1]["start_step_id"] = "a1-s4"

    with pytest.raises(ValueError, match="gap or overlaps"):
        trajectory_outline.validate_authored(broken, payload())


def test_validation_rejects_copy_that_cannot_fit_the_viewer() -> None:
    oversized_chapter = authored()
    oversized_chapter["chapters"][0]["summary"] = "x" * 151
    with pytest.raises(ValueError, match="30–150 characters"):
        trajectory_outline.validate_authored(oversized_chapter, payload())

    oversized_synopsis = authored()
    oversized_synopsis["synopsis"] = "x" * 361
    with pytest.raises(ValueError, match="40–360 characters"):
        trajectory_outline.validate_authored(oversized_synopsis, payload())


def test_prompt_points_codex_at_exact_public_trace(tmp_path: Path) -> None:
    trajectory = tmp_path / "web" / "data" / "trajectories" / "run-a.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(json.dumps(payload()))

    result = trajectory_outline.prompt(trajectory, repository=tmp_path)

    assert "web/data/trajectories/run-a.json" in result
    assert "do not ask for or\nrely on a preprocessed summary" in result
    assert "`public_step_id`" in result


def test_generate_reuses_valid_matching_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = tmp_path / "run-a.json"
    raw = json.dumps(payload(), separators=(",", ":")).encode()
    trajectory.write_bytes(raw)
    output = tmp_path / "run-a.outline.json"
    cached = trajectory_outline.build_artifact(
        authored(),
        payload(),
        trajectory_sha256=trajectory_outline.sha256_bytes(raw),
        model="gpt-5.6-sol",
        reasoning_effort="high",
    )
    output.write_text(json.dumps(cached))
    monkeypatch.setattr(
        trajectory_outline,
        "invoke_codex",
        lambda *args, **kwargs: pytest.fail("cached outline should skip Codex"),
    )

    result = trajectory_outline.generate(
        trajectory,
        output_path=output,
        repository=tmp_path,
    )

    assert result == cached


def test_invoke_codex_captures_event_stream_and_final_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = tmp_path / "run-a.json"
    trajectory.write_text(json.dumps(payload()))
    capture = tmp_path / "capture"

    @contextmanager
    def fake_isolation(*args: object, **kwargs: object):
        source = tmp_path / "isolated" / "trajectory.json"
        source.parent.mkdir()
        source.write_bytes(trajectory.read_bytes())
        schema = source.parent / "outline.schema.json"
        schema.write_text("{}")
        yield {
            "prefix": ["codex"],
            "repository": source.parent,
            "trajectory": source,
            "schema": schema,
            "output": source.parent / "final-response.json",
        }

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert "--json" in command
        assert "--ephemeral" in command
        assert "--ignore-user-config" in command
        assert "--dangerously-bypass-approvals-and-sandbox" in command
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(json.dumps(authored()))
        kwargs["stdout"].write('{"type":"thread.started","thread_id":"test"}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        trajectory_outline, "isolated_codex_environment", fake_isolation
    )
    monkeypatch.setattr(trajectory_outline.subprocess, "run", fake_run)
    result = trajectory_outline.invoke_codex(
        trajectory,
        repository=tmp_path,
        model="gpt-5.6-sol",
        reasoning_effort="high",
        codex_bin="codex",
        timeout_seconds=30,
        capture_dir=capture,
    )

    assert result == authored()
    assert json.loads((capture / "events.jsonl").read_text())["type"] == "thread.started"
    assert json.loads((capture / "final-response.json").read_text()) == authored()
    assert (capture / "stderr.log").is_file()
    assert (capture / "prompt.txt").is_file()
    assert json.loads((capture / "command.json").read_text())["isolation"] == (
        "unprivileged-user-read-only-source/v1"
    )


def test_generate_preserves_rejected_response_and_failure_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = tmp_path / "run-a.json"
    trajectory.write_text(json.dumps(payload()))
    capture = tmp_path / "capture"
    rejected = {
        "synopsis": authored()["synopsis"],
        "chapters": [
            {
                "id": "one-chapter",
                "start_step_id": "a1-s1",
                "end_step_id": "a1-s5",
                "title": "One oversized chapter",
                "summary": "This response deliberately collapses the complete trace into one chapter.",
            }
        ],
    }

    def fake_invoke(*args: object, **kwargs: object) -> dict:
        capture.mkdir()
        (capture / "events.jsonl").write_text('{"type":"turn.completed"}\n')
        (capture / "final-response.json").write_text(json.dumps(rejected))
        (capture / "stderr.log").write_text("")
        return rejected

    monkeypatch.setattr(trajectory_outline, "invoke_codex", fake_invoke)
    with pytest.raises(ValueError, match="expected 2–20 chapters, received 1"):
        trajectory_outline.generate(
            trajectory,
            output_path=tmp_path / "run-a.outline.json",
            repository=tmp_path,
            capture_dir=capture,
            force=True,
        )

    assert json.loads((capture / "final-response.json").read_text()) == rejected
    manifest = json.loads((capture / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "ValueError"


def test_generate_records_interrupted_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = tmp_path / "run-a.json"
    trajectory.write_text(json.dumps(payload()))
    capture = tmp_path / "capture"

    def interrupt(*args: object, **kwargs: object) -> dict:
        capture.mkdir()
        (capture / "events.jsonl").write_text('{"type":"turn.started"}\n')
        raise KeyboardInterrupt

    monkeypatch.setattr(trajectory_outline, "invoke_codex", interrupt)
    with pytest.raises(KeyboardInterrupt):
        trajectory_outline.generate(
            trajectory,
            output_path=tmp_path / "run-a.outline.json",
            repository=tmp_path,
            capture_dir=capture,
            force=True,
        )

    manifest = json.loads((capture / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["error"]["type"] == "KeyboardInterrupt"


def test_authored_response_can_be_recovered_from_jsonl(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                '{"type":"thread.started","thread_id":"test"}',
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": json.dumps(authored()),
                        },
                    }
                ),
            ]
        )
    )

    assert trajectory_outline.authored_from_event_stream(events) == authored()
