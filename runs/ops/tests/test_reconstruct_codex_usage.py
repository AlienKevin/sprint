from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "reconstruct_codex_usage.py"
UV = "/home/ubuntu/.local/bin/uv"
HARBOR = str(ROOT / "harbor")


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
            UV,
            "run",
            "--project",
            HARBOR,
            "--frozen",
            "--extra",
            "modal",
            "python",
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
