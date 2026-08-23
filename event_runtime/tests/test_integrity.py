from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_runtime.control.integrity import build_integrity_report  # noqa: E402


def test_clean_single_attempt_run_is_eligible(tmp_path: Path) -> None:
    run = {"run_id": "clean-run", "cpu_launch_attempt": 1}
    (tmp_path / "STOP_ACK.json").write_text(
        json.dumps({"reason": "agent_cost_budget_exhausted"})
    )

    report = build_integrity_report(tmp_path, run)

    assert report["status"] == "clean"
    assert report["benchmark_valid"] is True
    assert report["replacement_required"] is False


def test_controller_relaunch_and_budget_telemetry_failure_are_invalid(
    tmp_path: Path,
) -> None:
    run = {"run_id": "bad-run", "cpu_launch_attempt": 3}
    (tmp_path / "STOP_ACK.json").write_text(
        json.dumps({"reason": "budget_telemetry_unavailable"})
    )

    report = build_integrity_report(tmp_path, run)

    assert report["status"] == "invalid_infrastructure"
    assert report["benchmark_valid"] is False
    assert report["replacement_required"] is True
    assert {reason["code"] for reason in report["reasons"]} == {
        "cpu_agent_relaunched",
        "budget_telemetry_unavailable",
    }


def test_gpu_retry_and_unconfirmed_termination_are_invalid(tmp_path: Path) -> None:
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    (registry / "job-1.json").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "attempt": 2,
                "retry_reason": "worker_lost",
                "terminate_error": "TimeoutError: termination did not settle",
            }
        )
    )

    report = build_integrity_report(
        tmp_path, {"run_id": "gpu-retry", "cpu_launch_attempt": 1}
    )

    assert {reason["code"] for reason in report["reasons"]} == {
        "gpu_worker_retried",
        "gpu_termination_unconfirmed",
    }


def test_expected_idempotent_teardown_error_is_not_an_integrity_failure(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    (registry / "job-1.json").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "attempt": 1,
                "terminate_error": "ConflictError: container is not running",
            }
        )
    )

    report = build_integrity_report(
        tmp_path, {"run_id": "expected-stop", "cpu_launch_attempt": 1}
    )

    assert report["benchmark_valid"] is True


def test_unattributed_gpu_termination_requires_replacement(tmp_path: Path) -> None:
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    (registry / "job-1.json").write_text(
        json.dumps({"job_id": "job-1", "attempt": 1, "status": "terminated"})
    )

    report = build_integrity_report(
        tmp_path, {"run_id": "unknown-stop", "cpu_launch_attempt": 1}
    )

    assert report["benchmark_valid"] is False
    assert report["reasons"][0]["code"] == "gpu_termination_unattributed"


def test_worker_budget_stop_is_attributed_without_legacy_reason_field(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    (registry / "job-1.json").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "attempt": 1,
                "status": "terminated",
                "error": "agent cost budget exhausted",
            }
        )
    )

    report = build_integrity_report(
        tmp_path, {"run_id": "budget-stop", "cpu_launch_attempt": 1}
    )

    assert report["benchmark_valid"] is True


def test_unacknowledged_control_request_requires_replacement(tmp_path: Path) -> None:
    (tmp_path / "control-events.jsonl").write_text(
        json.dumps(
            {
                "event": "cancel_requested",
                "request_id": "a" * 32,
                "job_id": "job-1",
            }
        )
        + "\n"
    )

    report = build_integrity_report(
        tmp_path, {"run_id": "lost-cancel", "cpu_launch_attempt": 1}
    )

    assert report["benchmark_valid"] is False
    assert report["reasons"][0]["code"] == "gpu_cancellation_unacknowledged"
