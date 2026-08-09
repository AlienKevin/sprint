from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from harbor.agents.installed.codex import Codex
from harbor.models.agent.context import AgentContext


SCRIPT = Path(__file__).resolve().parents[1] / "reconstruct_codex_usage.py"


def write_session(state: Path, attempt: int, session_id: str, timestamp: str) -> None:
    chunk = (
        state
        / "durable-trace"
        / "raw"
        / f"cpu-attempt-{attempt:03d}"
        / "codex"
        / f"source-{attempt}"
        / "chunks"
        / "0000000000000000-0000000000001000-test.jsonl"
    )
    chunk.parent.mkdir(parents=True)
    usage = {
        "input_tokens": 1_000,
        "cached_input_tokens": 800,
        "cache_write_input_tokens": 0,
        "output_tokens": 100,
        "reasoning_output_tokens": 50,
        "total_tokens": 1_100,
    }
    rows = [
        {"type": "session_meta", "payload": {"id": session_id}},
        {
            "type": "turn_context",
            "timestamp": timestamp,
            "payload": {"model": "deepseek-v4-flash", "effort": "high"},
        },
        {
            "type": "response_item",
            "timestamp": timestamp,
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
        {
            "type": "event_msg",
            "timestamp": timestamp,
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": usage,
                    "total_token_usage": usage,
                    "model_context_window": 1_000_000,
                },
            },
        },
    ]
    chunk.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_reconstructs_all_cpu_attempts_and_aggregates_cost(tmp_path: Path) -> None:
    run = {
        "run_id": "restart-cost-fixture",
        "model": "deepseek/deepseek-v4-flash",
        "resolved_model_version": "DeepSeek-V4-Flash-0731",
        "reasoning_effort": "high",
        "cpu_launch_history": [{"attempt": 1}, {"attempt": 2}],
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    write_session(tmp_path, 1, "session-one", "2026-08-08T00:00:01Z")
    write_session(tmp_path, 2, "session-two", "2026-08-08T00:01:01Z")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--state-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    audit = json.loads((tmp_path / "usage" / "run-usage-audit.json").read_text())
    assert audit["request_count"] == 2
    assert audit["captured_cpu_attempts"] == [1, 2]
    assert audit["attempt_coverage_complete"] is True
    assert audit["cost_reconstruction_complete"] is True
    assert audit["calculated_api_usage_usd"] > 0
    trajectories = list((tmp_path / "trace" / "reconstructed").rglob("trajectory.json"))
    assert len(trajectories) == 2


def test_harbor_final_archive_supersedes_verified_durable_prefix(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "harbor-jobs/job/task__trial"
    session = trial / "agent/sessions/rollout.jsonl"
    trajectory = trial / "agent/trajectory.json"
    session.parent.mkdir(parents=True)
    run = {
        "run_id": "normal-exit-tail",
        "model": "deepseek/deepseek-v4-flash",
        "resolved_model_version": "DeepSeek-V4-Flash-0731",
        "reasoning_effort": "high",
        "cpu_launch_attempt": 1,
        "cpu_launch_history": [{"attempt": 1}],
        "trial_path": str(trial),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    write_session(tmp_path, 1, "session-one", "2026-08-08T00:00:01Z")
    durable = next((tmp_path / "durable-trace/raw").rglob("*.jsonl"))
    session.write_bytes(durable.read_bytes() + b'{"type":"final_tail"}\n')
    trajectory.write_text("{}\n")
    provenance = trial / "agent/usage-provenance"
    provenance.mkdir()
    source_snapshot = provenance / "source-session.jsonl"
    trajectory_snapshot = provenance / "trajectory.json"
    source_snapshot.write_bytes(session.read_bytes())
    trajectory_snapshot.write_bytes(trajectory.read_bytes())
    session_sha = hashlib.sha256(source_snapshot.read_bytes()).hexdigest()
    trajectory_sha = hashlib.sha256(trajectory_snapshot.read_bytes()).hexdigest()
    (trial / "agent/usage-audit.json").write_text(
        json.dumps(
            {
                "session_id": "session-one",
                "cost_reconstruction_complete": True,
                "calculated_api_usage_usd": 0.3,
                "requests": [
                    {"api_call_id": "api_call_1", "calculated_cost_usd": 0.1},
                    {"api_call_id": "api_call_2", "calculated_cost_usd": 0.2},
                ],
                "pricing_snapshots": [{"id": "test-pricing"}],
                "reconciliation_mismatches": {},
                "provenance": {
                    "source_session_path": "usage-provenance/source-session.jsonl",
                    "source_session_sha256": session_sha,
                    "trajectory_path": "usage-provenance/trajectory.json",
                    "trajectory_sha256": trajectory_sha,
                },
            }
        )
    )

    subprocess.run(
        [sys.executable, str(SCRIPT), "--state-dir", str(tmp_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    audit = json.loads((tmp_path / "usage/run-usage-audit.json").read_text())
    assert audit["request_count"] == 2
    assert abs(audit["calculated_api_usage_usd"] - 0.3) < 1e-12
    assert audit["source_sessions"][0]["origin"] == "harbor_final_archive"


def test_stale_mutable_harbor_provenance_is_replayed_and_attested(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "harbor-jobs/job/task__trial"
    session_dir = trial / "agent/sessions/2026/08/08"
    session_dir.mkdir(parents=True)
    run = {
        "run_id": "stale-provenance",
        "model": "deepseek/deepseek-v4-flash",
        "resolved_model_version": "DeepSeek-V4-Flash-0731",
        "reasoning_effort": "high",
        "cpu_launch_attempt": 1,
        "cpu_launch_history": [{"attempt": 1}],
        "trial_path": str(trial),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    write_session(tmp_path, 1, "session-one", "2026-08-08T00:00:01Z")
    durable = next((tmp_path / "durable-trace/raw").rglob("*.jsonl"))
    session = session_dir / "rollout.jsonl"
    session.write_bytes(durable.read_bytes())
    agent = Codex(
        logs_dir=trial / "agent",
        model_name=run["model"],
        reasoning_effort=run["reasoning_effort"],
    )
    agent.populate_context_post_run(AgentContext())

    audit_path = trial / "agent/usage-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["provenance"] = {
        "source_session_file": session.name,
        "source_session_sha256": hashlib.sha256(session.read_bytes()).hexdigest(),
        "trajectory_sha256": hashlib.sha256(
            (trial / "agent/trajectory.json").read_bytes()
        ).hexdigest(),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    session.write_text(session.read_text() + '{"type":"final_tail"}\n')

    subprocess.run(
        [sys.executable, str(SCRIPT), "--state-dir", str(tmp_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    repaired = json.loads(audit_path.read_text())
    assert repaired["provenance"]["source_session_path"] == (
        "usage-provenance/source-session.jsonl"
    )
    assert repaired["provenance"]["trajectory_path"] == (
        "usage-provenance/trajectory.json"
    )
    provenance = trial / "agent/usage-provenance"
    attestation = json.loads((provenance / "recovery-attestation.json").read_text())
    assert attestation["method"] == "deterministic_codex_session_replay"
    assert attestation["request_count"] == repaired["request_count"]
    assert list(provenance.glob("original-usage-audit.*.json"))
    run_audit = json.loads((tmp_path / "usage/run-usage-audit.json").read_text())
    assert run_audit["request_count"] == repaired["request_count"]
