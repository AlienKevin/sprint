from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_runtime.export import harbor_job  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _job_fixture(tmp_path: Path, *, deepseek: bool = False) -> tuple[Path, Path]:
    state = tmp_path / "runs" / "ops" / "run-1"
    job = state / "harbor-jobs" / "run-1"
    trial = job / "rendered-task__abc"
    secret_dir = tmp_path / "secrets" / "run-1"
    secret_dir.mkdir(parents=True)
    (secret_dir / "harbor.env").write_text(
        "OPENROUTER_API_KEY=sk-or-v1-known-secret-value\n", encoding="utf-8"
    )
    _write_json(
        state / "run.json",
        {"run_id": "run-1", "secret_dir": str(secret_dir), "model": "test-model"},
    )
    for path, payload in (
        (job / "config.json", {"job_name": "run-1"}),
        (job / "lock.json", {"schema_version": 1}),
        (job / "result.json", {"id": "job-id"}),
        (trial / "config.json", {"trial_name": "trial-1"}),
        (trial / "lock.json", {"schema_version": 2}),
        (trial / "result.json", {"id": "trial-id"}),
    ):
        _write_json(path, payload)
    (job / "job.log").write_text("contact owner@example.com\n", encoding="utf-8")
    (trial / "trial.log").write_text(
        "OPENROUTER_API_KEY=sk-or-v1-known-secret-value from 10.1.2.3\n",
        encoding="utf-8",
    )
    agent = trial / "agent"
    agent.mkdir()
    (agent / "codex-state").mkdir()
    (agent / "codex-state" / "state_5.sqlite").write_bytes(b"private state")
    (agent / "unexpected.txt").write_text("not allowlisted", encoding="utf-8")
    if deepseek:
        event = {
            "schema_version": 1,
            "method": "session.event",
            "payload": {
                "event": {
                    "seq": 1,
                    "time": 1_787_472_000_000,
                    "type": "assistant/message",
                    "data": {
                        "message": {
                            "content": [
                                {"type": "reasoning", "text": "inspect first"},
                                {"type": "text", "text": "working"},
                            ],
                            "source": {"model": "deepseek/test"},
                        },
                        "usage": {"inputTokens": 2, "outputTokens": 1},
                    },
                }
            },
        }
        (agent / "deepseek-harness-events.jsonl").write_text(
            json.dumps(event) + "\n", encoding="utf-8"
        )
    else:
        _write_json(
            agent / "trajectory.json",
            {
                "schema_version": "1.0",
                "steps": [
                    {
                        "step_id": 1,
                        "source": "agent",
                        "message": "email me at owner@example.com",
                        "internal_chat_message_metadata_passthrough": {
                            "turn_id": "not-needed"
                        },
                        "tool_calls": [
                            {
                                "tool_call_id": "call-1",
                                "function_name": "bash",
                                "arguments": {
                                    "api_key": "sk-or-v1-known-secret-value"
                                },
                            }
                        ],
                    }
                ],
            },
        )
    artifacts = trial / "artifacts"
    _write_json(artifacts / "telemetry" / "latest.json", {"cpu": 4})
    (artifacts / "policy.pt").write_bytes(b"binary policy")
    return job, trial


def test_sanitize_job_stages_clean_harbor_job(tmp_path: Path) -> None:
    job, trial = _job_fixture(tmp_path)
    source_trajectory = trial / "agent" / "trajectory.json"
    source_before = source_trajectory.read_bytes()

    destination = harbor_job.sanitize_job(
        job, tmp_path / "staged", validate_harbor=False, run_kingfisher=False
    )

    assert source_trajectory.read_bytes() == source_before
    assert not (destination / trial.name / "agent" / "codex-state").exists()
    assert not (destination / trial.name / "agent" / "unexpected.txt").exists()
    assert not (destination / trial.name / "artifacts" / "policy.pt").exists()
    staged_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in destination.rglob("*")
        if path.is_file()
    )
    assert "sk-or-v1-known-secret-value" not in staged_text
    assert "owner@example.com" not in staged_text
    assert "10.1.2.3" not in staged_text
    assert "internal_chat_message_metadata_passthrough" not in staged_text
    report = json.loads((destination / "SANITIZATION_REPORT.json").read_text())
    assert report["status"] == "clean"
    assert report["findings"] == []
    excluded = {item["path"] for item in report["excluded"]}
    assert any(path.endswith("agent/codex-state") for path in excluded)
    assert any(path.endswith("artifacts/policy.pt") for path in excluded)


def test_sanitize_job_materializes_deepseek_trajectory_without_source_write(
    tmp_path: Path,
) -> None:
    job, trial = _job_fixture(tmp_path, deepseek=True)
    assert not (trial / "agent" / "trajectory.json").exists()

    destination = harbor_job.sanitize_job(
        job, tmp_path / "staged", validate_harbor=False, run_kingfisher=False
    )

    assert not (trial / "agent" / "trajectory.json").exists()
    payload = json.loads(
        (destination / trial.name / "agent" / "trajectory.json").read_text()
    )
    assert payload["steps"][0]["reasoning_content"] == "inspect first"
    assert payload["steps"][0]["message"] == "working"


def test_scan_tree_fails_closed_on_unsanitized_material(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    root.mkdir()
    (root / "leak.txt").write_text(
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz", encoding="utf-8"
    )

    findings = harbor_job.scan_tree(root, ())

    assert {finding["kind"] for finding in findings} == {"bearer_token"}


def test_home_redaction_does_not_match_library_subpaths_or_its_marker(
    tmp_path: Path,
) -> None:
    audit = harbor_job.Audit()
    sanitizer = harbor_job.Sanitizer({}, audit)
    text = sanitizer.text(
        "owner /home/alice/project; library /boost/spirit/home/support/header.hpp"
    )

    assert "/home/alice" not in text
    assert "/boost/spirit/home/support/header.hpp" in text
    path = tmp_path / "clean.txt"
    path.write_text(text, encoding="utf-8")
    assert harbor_job.scan_tree(tmp_path, ()) == []


def test_existing_destination_requires_force(tmp_path: Path) -> None:
    job, _ = _job_fixture(tmp_path)
    output = tmp_path / "staged"
    harbor_job.sanitize_job(
        job, output, validate_harbor=False, run_kingfisher=False
    )

    with pytest.raises(FileExistsError):
        harbor_job.sanitize_job(
            job, output, validate_harbor=False, run_kingfisher=False
        )


def test_kingfisher_failure_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, _ = _job_fixture(tmp_path)

    def finding(_target: Path, _report: Path) -> dict[str, object]:
        return {"version": harbor_job.KINGFISHER_VERSION, "findings": 1}

    monkeypatch.setattr(harbor_job, "_scan_with_kingfisher", finding)

    with pytest.raises(ValueError, match="sanitization blocked"):
        harbor_job.sanitize_job(job, tmp_path / "staged", validate_harbor=False)

    assert not (tmp_path / "staged" / job.name).exists()
    report = json.loads(
        (tmp_path / "staged" / f"{job.name}.SANITIZATION_FAILED.json").read_text()
    )
    assert report["findings"][-1] == {
        "count": 2,
        "kind": "kingfisher",
        "path": ".",
    }


def test_kingfisher_exit_200_is_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "stage"
    target.mkdir()
    report_path = tmp_path / "kingfisher.json"

    def run(command: list[str], **_kwargs: object) -> object:
        _write_json(
            Path(command[-1]),
            {
                "findings": [{"finding": {"snippet": "[REDACTED]"}}],
                "metadata": {"kingfisher_version": harbor_job.KINGFISHER_VERSION},
            },
        )
        return type("Completed", (), {"returncode": 200, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(harbor_job.shutil, "which", lambda _name: "/bin/kingfisher")
    monkeypatch.setattr(harbor_job.subprocess, "run", run)

    assert harbor_job._scan_with_kingfisher(target, report_path) == {
        "version": harbor_job.KINGFISHER_VERSION,
        "findings": 1,
    }
