from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from harbor.agents.installed.codex import Codex
from harbor.models.agent.context import AgentContext


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "runs/ops"
SCRIPT = ROOT / "event_runtime/cost/model_usage.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OPS))

from event_runtime.control import run as sprintctl  # noqa: E402
from event_runtime.cost.model_usage import (  # noqa: E402
    apply_provider_reported_costs,
    prefer_complete_session,
    provider_usage_records,
)


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


def write_zero_request_session(state: Path, attempt: int, session_id: str) -> None:
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
    rows = [
        {"type": "session_meta", "payload": {"id": session_id}},
        {
            "type": "turn_context",
            "timestamp": "2026-08-09T05:44:35Z",
            "payload": {"model": "gpt-5.6-luna", "effort": "max"},
        },
        {
            "type": "response_item",
            "timestamp": "2026-08-09T05:44:35Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "starting"}],
            },
        },
        {
            "type": "event_msg",
            "timestamp": "2026-08-09T05:44:36Z",
            "payload": {
                "type": "error",
                "message": "provider rejected request before completion",
            },
        },
    ]
    chunk.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_deepseek_harness_session(state: Path, attempt: int) -> None:
    chunk = (
        state
        / "durable-trace"
        / "raw"
        / f"cpu-attempt-{attempt:03d}"
        / "deepseek-harness"
        / f"harness-source-{attempt}"
        / "chunks"
        / "0000000000000000-0000000000001000-test.jsonl"
    )
    chunk.parent.mkdir(parents=True)
    rows = [
        {
            "schema_version": 1,
            "method": "session.event",
            "payload": {
                "event": {
                    "seq": 1,
                    "type": "user/message",
                    "time": 1_787_571_739_098,
                    "data": {"content": [{"type": "text", "text": "task"}]},
                }
            },
        },
        {
            "schema_version": 1,
            "method": "session.event",
            "payload": {
                "event": {
                    "seq": 2,
                    "type": "assistant/message",
                    "time": 1_787_571_740_098,
                    "data": {
                        "message": {
                            "content": [{"type": "text", "text": "working"}],
                            "source": {"model": "deepseek-v4-flash-vision-exp"},
                        },
                        "usage": {
                            "inputTokens": 200,
                            "cacheReadTokens": 800,
                            "outputTokens": 100,
                        },
                    },
                }
            },
        },
    ]
    chunk.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_reconstructs_deepseek_harness_trace_with_provider_billing(
    tmp_path: Path,
) -> None:
    run = {
        "run_id": "deepseek-harness-fixture",
        "model": "deepseek/deepseek-v4-flash-vision-exp",
        "resolved_model_version": "DeepSeek-V4-Flash-Vision-Exp",
        "reasoning_effort": "max",
        "cpu_launch_history": [{"attempt": 1}],
        "budget_enforcement": {
            "api_cost_source": "openrouter_reported_per_request",
            "api_budget_cost_basis": "openrouter_list_price_with_deepseek_peak_floor",
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    write_deepseek_harness_session(tmp_path, 1)
    record = tmp_path / "provider-api-usage/api-usage/requests/request.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": run["run_id"],
                "cpu_attempt": 1,
                "ledger_request_id": "ledger-deepseek-1",
                "generation_id": "generation-deepseek-1",
                "requested_at": "2026-08-24T04:20:44Z",
                "completed_at": "2026-08-24T04:20:45Z",
                "state": "complete",
                "requested_model": run["model"],
                "response_model": run["model"],
                "provider_reported_cost_usd": 0.1,
                "undiscounted_cost_usd": 0.2,
                "benchmark_cost_usd": 0.3,
                "cost_basis": "openrouter_list_price_with_deepseek_peak_floor",
                "promotion_snapshot": {"discount_fraction": 0.5},
                "usage": {
                    "prompt_tokens": 1000,
                    "prompt_tokens_details": {
                        "cached_tokens": 800,
                        "cache_write_tokens": 0,
                    },
                    "completion_tokens": 100,
                    "completion_tokens_details": {"reasoning_tokens": 50},
                    "total_tokens": 1100,
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
    assert audit["source"] == "durable_agent_session_chunks"
    assert audit["captured_cpu_attempts"] == [1]
    assert audit["attempt_coverage_complete"] is True
    assert audit["request_count"] == 1
    assert audit["calculated_api_usage_usd"] == 0.3
    assert audit["requests"][0]["provider_only_usage"] is True
    assert audit["requests"][0]["input_tokens"] == 1000
    assert audit["requests"][0]["cached_input_tokens"] == 800
    assert audit["requests"][0]["output_tokens"] == 100
    assert audit["requests"][0]["reasoning_output_tokens"] == 50
    assert audit["source_sessions"][0]["agent_kind"] == "deepseek-harness"
    assert audit["source_sessions"][0]["trajectory_sha256"]
    assert sprintctl.run_usage_audit_ready(tmp_path, run) == (True, [])


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


def test_zero_completed_requests_are_attested_as_exact_zero_cost(
    tmp_path: Path,
) -> None:
    run = {
        "run_id": "provider-rejected-before-response",
        "model": "openai/gpt-5.6-luna",
        "resolved_model_version": "gpt-5.6-luna",
        "reasoning_effort": "max",
        "cpu_launch_history": [{"attempt": 1}, {"attempt": 2}],
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    write_zero_request_session(tmp_path, 1, "session-one")
    write_zero_request_session(tmp_path, 2, "session-two")

    subprocess.run(
        [sys.executable, str(SCRIPT), "--state-dir", str(tmp_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    audit = json.loads((tmp_path / "usage/run-usage-audit.json").read_text())
    assert audit["request_count"] == 0
    assert audit["cost_reconstruction_complete"] is True
    assert audit["calculated_api_usage_usd"] == 0.0
    assert audit["attempt_coverage_complete"] is True
    assert audit["zero_request_reason"] == (
        "no completed model request was present in any captured CPU attempt"
    )
    assert all(row["trajectory_sha256"] for row in audit["source_sessions"])
    assert sprintctl.run_usage_audit_ready(tmp_path, run) == (True, [])


def test_openrouter_cost_is_bound_to_the_matching_codex_usage() -> None:
    request = {
        "cpu_attempt": 2,
        "input_tokens": 100,
        "cached_input_tokens": 80,
        "cache_write_input_tokens": 5,
        "output_tokens": 20,
        "reasoning_output_tokens": 7,
        "total_tokens": 120,
    }
    record = {
        "cpu_attempt": 2,
        "ledger_request_id": "ledger-1",
        "generation_id": "gen-1",
        "response_model": "openai/gpt-5.6-luna-20260709",
        "provider_reported_cost_usd": 0.123,
        "undiscounted_cost_usd": 0.246,
        "promotion_discount_fraction": 0.5,
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {
                "cached_tokens": 80,
                "cache_write_tokens": 5,
            },
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 7},
            "total_tokens": 120,
            "cost": 0.123,
            "cost_details": {"upstream_inference_cost": 0.123},
        },
    }

    apply_provider_reported_costs([request], [record])

    assert request["calculated_cost_usd"] == 0.246
    assert request["provider_reported_cost_usd"] == 0.123
    assert request["promotion_savings_usd"] == 0.123
    assert request["cost_basis"] == "openrouter_list_price_before_endpoint_discount"
    assert request["openrouter_generation_id"] == "gen-1"
    assert request["cost_components_usd"] == {"upstream_inference_cost": 0.123}


def test_deepseek_peak_benchmark_cost_overrides_endpoint_list_cost() -> None:
    request = {
        "cpu_attempt": 2,
        "input_tokens": 100,
        "cached_input_tokens": 80,
        "cache_write_input_tokens": 0,
        "output_tokens": 20,
        "reasoning_output_tokens": 7,
        "total_tokens": 120,
    }
    record = {
        "cpu_attempt": 2,
        "ledger_request_id": "ledger-peak",
        "generation_id": "gen-peak",
        "provider_reported_cost_usd": 0.1,
        "undiscounted_cost_usd": 0.2,
        "benchmark_cost_usd": 0.3,
        "promotion_adjustment_usd": 0.1,
        "deepseek_peak_adjustment_usd": 0.1,
        "cost_basis": "openrouter_list_price_with_deepseek_peak_floor",
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 80},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 7},
            "total_tokens": 120,
        },
    }

    apply_provider_reported_costs([request], [record])

    assert request["calculated_cost_usd"] == 0.3
    assert request["provider_reported_cost_usd"] == 0.1
    assert request["endpoint_list_cost_usd"] == 0.2
    assert request["promotion_adjustment_usd"] == 0.1
    assert request["deepseek_peak_adjustment_usd"] == 0.1
    assert request["benchmark_adjustment_usd"] == pytest.approx(0.2)
    assert request["promotion_savings_usd"] == pytest.approx(0.2)
    assert request["cost_basis"] == ("openrouter_list_price_with_deepseek_peak_floor")


def test_recovered_openrouter_cost_is_bound_in_serial_request_order(
    tmp_path: Path,
) -> None:
    request = {
        "cpu_attempt": 1,
        "input_tokens": 100,
        "cached_input_tokens": 80,
        "cache_write_input_tokens": 0,
        "output_tokens": 20,
        "reasoning_output_tokens": 7,
        "total_tokens": 120,
    }
    record_path = tmp_path / "provider-api-usage" / "requests" / "recovered.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "cpu_attempt": 1,
                "ledger_request_id": "ledger-recovered",
                "generation_id": "gen-recovered",
                "requested_at": "2026-08-19T00:00:00Z",
                "state": "recovered_complete",
                "provider_reported_cost_usd": 0.25,
                "generation_audit": {"total_cost": 0.25},
            }
        )
    )

    records = provider_usage_records(tmp_path, "run-1")
    apply_provider_reported_costs([request], records)

    assert request["calculated_cost_usd"] == 0.25
    assert request["openrouter_generation_id"] == "gen-recovered"
    assert request["cost_components_usd"] == {}


def test_pending_provider_summary_fails_reconstruction_closed(tmp_path: Path) -> None:
    root = tmp_path / "provider-api-usage" / "api-usage"
    root.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "run-1",
                "pending_request_count": 1,
                "in_flight_request_count": 1,
                "cost_recovery_required_count": 0,
            }
        )
    )

    with pytest.raises(SystemExit, match="ledger is not settled"):
        provider_usage_records(tmp_path, "run-1")


def test_openrouter_usage_mismatch_fails_closed() -> None:
    request = {
        "cpu_attempt": 1,
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 5,
        "reasoning_output_tokens": 0,
        "total_tokens": 15,
    }
    record = {
        "cpu_attempt": 1,
        "provider_reported_cost_usd": 0.1,
        "usage": {
            "input_tokens": 11,
            "output_tokens": 5,
            "total_tokens": 16,
        },
    }

    with pytest.raises(SystemExit, match="no matching OpenRouter"):
        apply_provider_reported_costs([request], [record])


def test_provider_only_billed_request_is_retained_after_interruption() -> None:
    requests: list[dict[str, object]] = []
    record = {
        "cpu_attempt": 3,
        "ledger_request_id": "ledger-provider-only",
        "generation_id": "gen-provider-only",
        "requested_model": "gpt-5.6-luna",
        "response_model": "openai/gpt-5.6-luna-20260709",
        "completed_at": "2026-08-19T23:00:00Z",
        "provider_reported_cost_usd": 0.123,
        "undiscounted_cost_usd": 0.246,
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {
                "cached_tokens": 80,
                "cache_write_tokens": 5,
            },
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 7},
            "total_tokens": 120,
            "cost_details": {"upstream_inference_cost": 0.123},
        },
    }

    apply_provider_reported_costs(requests, [record])

    assert len(requests) == 1
    request = requests[0]
    assert request["provider_only_usage"] is True
    assert request["cpu_attempt"] == 3
    assert request["ordinary_uncached_input_tokens"] == 15
    assert request["calculated_cost_usd"] == 0.246
    assert request["provider_reported_cost_usd"] == 0.123


def test_openrouter_run_audit_uses_reconciled_provider_cost_over_session_cost(
    tmp_path: Path,
) -> None:
    run = {
        "run_id": "openrouter-session-cost",
        "model": "deepseek/deepseek-v4-flash",
        "resolved_model_version": "DeepSeek-V4-Flash-0731",
        "reasoning_effort": "high",
        "cpu_launch_history": [{"attempt": 1}],
        "budget_enforcement": {
            "api_cost_source": "openrouter_reported_per_request",
            "api_budget_cost_basis": ("openrouter_list_price_before_endpoint_discount"),
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    write_session(tmp_path, 1, "session-one", "2026-08-20T00:00:00Z")
    record = tmp_path / "provider-api-usage/api-usage/requests/request.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": run["run_id"],
                "cpu_attempt": 1,
                "ledger_request_id": "ledger-1",
                "generation_id": "generation-1",
                "requested_at": "2026-08-20T00:00:00Z",
                "completed_at": "2026-08-20T00:00:01Z",
                "state": "complete",
                "requested_model": "deepseek-v4-flash",
                "response_model": "deepseek-v4-flash",
                "provider_reported_cost_usd": 0.1,
                "undiscounted_cost_usd": 0.2,
                "promotion_snapshot": {"discount_fraction": 0.5},
                "usage": {
                    "input_tokens": 1000,
                    "input_tokens_details": {
                        "cached_tokens": 800,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 50},
                    "total_tokens": 1100,
                },
            }
        )
    )
    provider_only = record.with_name("provider-only.json")
    provider_only.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": run["run_id"],
                "cpu_attempt": 1,
                "ledger_request_id": "ledger-provider-only",
                "generation_id": "generation-provider-only",
                "requested_at": "2026-08-20T00:00:02Z",
                "completed_at": "2026-08-20T00:00:03Z",
                "state": "complete",
                "requested_model": "deepseek-v4-flash",
                "response_model": "provider/deepseek-v4-flash-versioned",
                "provider_reported_cost_usd": 0.05,
                "undiscounted_cost_usd": 0.1,
                "promotion_snapshot": {"discount_fraction": 0.5},
                "usage": {
                    "input_tokens": 2000,
                    "input_tokens_details": {
                        "cached_tokens": 1900,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens": 50,
                    "output_tokens_details": {"reasoning_tokens": 20},
                    "total_tokens": 2050,
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
    audit_path = tmp_path / "usage/run-usage-audit.json"
    audit = json.loads(audit_path.read_text())
    provider_only_row = next(
        row for row in audit["requests"] if row.get("provider_only_usage") is True
    )
    assert provider_only_row["model"] == "deepseek-v4-flash"
    assert provider_only_row["reasoning_effort"] == "high"
    assert provider_only_row["openrouter_response_model"] == (
        "provider/deepseek-v4-flash-versioned"
    )
    audit["source_sessions"][0]["cost_reconstruction_complete"] = False
    audit_path.write_text(json.dumps(audit))

    assert sprintctl.run_usage_audit_ready(tmp_path, run) == (True, [])

    audit["provider_billing_reconciled"] = False
    audit_path.write_text(json.dumps(audit))
    ready, details = sprintctl.run_usage_audit_ready(tmp_path, run)
    assert ready is False
    assert "run usage source session cost is incomplete" in details


def test_dominating_harbor_final_supersedes_shifted_durable_ordinals(
    tmp_path: Path,
) -> None:
    durable_path = tmp_path / "durable.jsonl"
    final_path = tmp_path / "final.jsonl"
    durable_path.write_text("durable")
    final_path.write_text("final")
    durable = {
        "session_id": "same-session",
        "combined_session_sha256": "durable",
        "_session_path": str(durable_path),
        "requests": [
            {
                "run_api_call_id": "same-session:api_call_1",
                "input_tokens": 100,
                "cached_input_tokens": 80,
                "cache_write_input_tokens": 10,
                "output_tokens": 10,
                "reasoning_output_tokens": 5,
                "total_tokens": 110,
            }
        ],
    }
    final = {
        "session_id": "same-session",
        "origin": "harbor_final_archive",
        "combined_session_sha256": "final",
        "_session_path": str(final_path),
        "requests": [
            {
                "run_api_call_id": "same-session:api_call_1",
                "input_tokens": 90,
                "cached_input_tokens": 70,
                "cache_write_input_tokens": 10,
                "output_tokens": 10,
                "reasoning_output_tokens": 5,
                "total_tokens": 100,
            },
            {
                "run_api_call_id": "same-session:api_call_2",
                "input_tokens": 20,
                "cached_input_tokens": 20,
                "cache_write_input_tokens": 0,
                "output_tokens": 5,
                "reasoning_output_tokens": 1,
                "total_tokens": 25,
            },
        ],
    }

    assert prefer_complete_session(durable, final) is final


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


def test_post_scrub_provenance_snapshots_are_replayed_and_reattested(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "harbor-jobs/job/task__trial"
    session_dir = trial / "agent/sessions/2026/08/08"
    session_dir.mkdir(parents=True)
    run = {
        "run_id": "post-scrub-provenance",
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

    provenance = trial / "agent/usage-provenance"
    source_snapshot = provenance / "source-session.jsonl"
    trajectory_snapshot = provenance / "trajectory.json"
    source_snapshot.write_text(
        source_snapshot.read_text().replace("done", "[REDACTED]")
    )
    trajectory_snapshot.write_text(
        trajectory_snapshot.read_text().replace("done", "[REDACTED]")
    )
    audit_path = trial / "agent/usage-audit.json"
    stale = json.loads(audit_path.read_text())
    assert (
        stale["provenance"]["source_session_sha256"]
        != hashlib.sha256(source_snapshot.read_bytes()).hexdigest()
    )

    subprocess.run(
        [sys.executable, str(SCRIPT), "--state-dir", str(tmp_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    repaired = json.loads(audit_path.read_text())
    assert (
        repaired["provenance"]["source_session_sha256"]
        == hashlib.sha256(source_snapshot.read_bytes()).hexdigest()
    )
    assert (
        repaired["provenance"]["trajectory_sha256"]
        == hashlib.sha256(trajectory_snapshot.read_bytes()).hexdigest()
    )
    assert repaired["request_count"] == 1
    run_audit = json.loads((tmp_path / "usage/run-usage-audit.json").read_text())
    assert run_audit["request_count"] == 1
    assert run_audit["source_sessions"][0]["origin"] == "harbor_final_archive"
