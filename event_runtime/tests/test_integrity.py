from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_runtime.control.integrity import build_integrity_report  # noqa: E402
from event_runtime.control.integrity import submission_bridge_reasons  # noqa: E402


def run_contract(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "cpu_execution_policy": "single_process_no_resume",
    }


def write_codex_goal_files(
    state_dir: Path, *, lifecycle: dict[str, object] | None
) -> None:
    agent_dir = state_dir / "harbor-jobs" / "run-1" / "task-1" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "goal-bootstrap.json").write_text(
        json.dumps({"thread_id": "thread-1", "objective": "finish the benchmark"})
    )
    if lifecycle is not None:
        (agent_dir / "goal-lifecycle.json").write_text(json.dumps(lifecycle))


def write_deepseek_goal_file(state_dir: Path, *, lifecycle: dict[str, object]) -> None:
    agent_dir = state_dir / "harbor-jobs" / "run-1" / "task-1" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "goal-lifecycle.json").write_text(json.dumps(lifecycle))


def write_clean_exit(path: Path, *, code: int = 0, requested: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "attempt": 1,
                "raw_exit_code": code,
                "stop_requested": requested,
                "execution_policy": "single_process_no_resume",
            }
        )
    )


def test_clean_single_attempt_run_is_eligible(tmp_path: Path) -> None:
    run = run_contract("clean-run")
    (tmp_path / "STOP_ACK.json").write_text(
        json.dumps({"reason": "agent_cost_budget_exhausted"})
    )
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")

    report = build_integrity_report(tmp_path, run)

    assert report["status"] == "clean"
    assert report["benchmark_valid"] is True
    assert report["replacement_required"] is False


def test_gpu_submission_drain_timeout_is_invalid(tmp_path: Path) -> None:
    run = run_contract("drain-timeout")
    (tmp_path / "STOP_ACK.json").write_text(
        json.dumps(
            {
                "reason": "agent_cost_budget_exhausted",
                "gpu_submission_drain_timed_out": True,
            }
        )
    )
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    (registry / "job-1.json").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "submission_paths": ["/app/policy.pt"],
                "progress": {
                    "submission_results": [
                        {"path": "/app/policy.pt", "state": "rejected"}
                    ]
                },
            }
        )
    )

    report = build_integrity_report(tmp_path, run)

    assert report["benchmark_valid"] is False
    assert {reason["code"] for reason in report["reasons"]} == {
        "gpu_submission_drain_timeout"
    }


def test_gpu_submission_drain_timeout_without_submissions_is_not_invalid(
    tmp_path: Path,
) -> None:
    run = run_contract("empty-drain-timeout")
    (tmp_path / "STOP_ACK.json").write_text(
        json.dumps(
            {
                "reason": "agent_cost_budget_exhausted",
                "gpu_submission_drain_timed_out": True,
            }
        )
    )
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")

    report = build_integrity_report(tmp_path, run)

    assert report["benchmark_valid"] is True


def test_submission_bridge_requires_terminal_results_and_harbor_forwarding(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    job = registry / "job-1.json"
    job.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "status": "succeeded",
                "submission_paths": ["/app/policy.pt"],
            }
        )
    )
    assert {reason["code"] for reason in submission_bridge_reasons(tmp_path)} == {
        "gpu_submission_results_missing"
    }

    job.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "status": "succeeded",
                "submission_paths": ["/app/policy.pt"],
                "progress": {
                    "submission_results": [
                        {"path": "/app/policy.pt", "state": "staged"}
                    ]
                },
            }
        )
    )
    assert {reason["code"] for reason in submission_bridge_reasons(tmp_path)} == {
        "gpu_submission_forwarding_incomplete"
    }

    bridge = tmp_path / "submission-bridge"
    bridge.mkdir()
    (bridge / "123456-abcd.json").write_text(
        json.dumps(
            {
                "submission_id": "123456-abcd",
                "gpu_job_id": "job-1",
                "queue_name": "123456-abcd.pt",
                "state": "forwarded",
            }
        )
    )
    ledger = (
        tmp_path
        / "harbor-jobs/run-1/task-1/artifacts/continuous/ledger.jsonl"
    )
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"name": "123456-abcd.pt"}) + "\n")
    assert submission_bridge_reasons(tmp_path) == []


def test_submission_bridge_waits_for_active_job_terminal_drain(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    for status in ("queued", "claiming", "dispatched", "running"):
        (registry / "job-1.json").write_text(
            json.dumps(
                {
                    "job_id": "job-1",
                    "status": status,
                    "submission_paths": ["/app/policy.pt"],
                }
            )
        )
        assert submission_bridge_reasons(tmp_path) == []


def test_submission_bridge_fails_closed_when_terminal_status_is_missing(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    (registry / "job-1.json").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "submission_paths": ["/app/policy.pt"],
            }
        )
    )

    assert {reason["code"] for reason in submission_bridge_reasons(tmp_path)} == {
        "gpu_submission_results_missing"
    }


def test_wrong_execution_policy_and_budget_telemetry_failure_are_invalid(
    tmp_path: Path,
) -> None:
    run = run_contract("bad-run")
    run["cpu_execution_policy"] = "restartable"
    (tmp_path / "STOP_ACK.json").write_text(
        json.dumps({"reason": "budget_telemetry_unavailable"})
    )
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")

    report = build_integrity_report(tmp_path, run)

    assert report["status"] == "invalid_infrastructure"
    assert report["benchmark_valid"] is False
    assert report["replacement_required"] is True
    assert {reason["code"] for reason in report["reasons"]} == {
        "cpu_execution_policy_mismatch",
        "budget_telemetry_unavailable",
    }


def test_codex_agent_exit_without_goal_lifecycle_is_invalid(tmp_path: Path) -> None:
    run = run_contract("missing-goal-lifecycle")
    run["agent_kind"] = "codex"
    (tmp_path / "STOP_ACK.json").write_text(json.dumps({"reason": "agent_exit"}))
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")
    write_codex_goal_files(tmp_path, lifecycle=None)

    report = build_integrity_report(tmp_path, run)

    assert report["benchmark_valid"] is False
    assert {reason["code"] for reason in report["reasons"]} == {
        "codex_goal_lifecycle_missing"
    }


def test_codex_agent_exit_with_active_goal_is_invalid(tmp_path: Path) -> None:
    run = run_contract("active-goal-exit")
    run["agent_kind"] = "codex"
    (tmp_path / "STOP_ACK.json").write_text(json.dumps({"reason": "agent_exit"}))
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")
    write_codex_goal_files(
        tmp_path,
        lifecycle={"goal_status": "active", "runner_state": "running"},
    )

    report = build_integrity_report(tmp_path, run)

    assert report["benchmark_valid"] is False
    assert {reason["code"] for reason in report["reasons"]} == {
        "codex_goal_active_at_agent_exit"
    }


def test_codex_agent_exit_after_terminal_goal_is_clean(tmp_path: Path) -> None:
    run = run_contract("terminal-goal-exit")
    run["agent_kind"] = "codex"
    (tmp_path / "STOP_ACK.json").write_text(json.dumps({"reason": "agent_exit"}))
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")
    write_codex_goal_files(
        tmp_path,
        lifecycle={"goal_status": "complete", "runner_state": "terminal"},
    )

    report = build_integrity_report(tmp_path, run)

    assert report["benchmark_valid"] is True


def test_deepseek_agent_exit_with_active_goal_is_invalid(tmp_path: Path) -> None:
    run = run_contract("deepseek-active-goal-exit")
    run["agent_kind"] = "deepseek-harness"
    (tmp_path / "STOP_ACK.json").write_text(json.dumps({"reason": "agent_exit"}))
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")
    write_deepseek_goal_file(
        tmp_path,
        lifecycle={"goal_status": "active", "runner_state": "running"},
    )

    report = build_integrity_report(tmp_path, run)

    assert report["benchmark_valid"] is False
    assert {reason["code"] for reason in report["reasons"]} == {
        "deepseek_goal_active_at_agent_exit"
    }


def test_deepseek_agent_exit_after_terminal_goal_is_clean(tmp_path: Path) -> None:
    run = run_contract("deepseek-terminal-goal-exit")
    run["agent_kind"] = "deepseek-harness"
    (tmp_path / "STOP_ACK.json").write_text(json.dumps({"reason": "agent_exit"}))
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")
    write_deepseek_goal_file(
        tmp_path,
        lifecycle={"goal_status": "blocked", "runner_state": "terminal"},
    )

    report = build_integrity_report(tmp_path, run)

    assert report["benchmark_valid"] is True


def test_terminal_goal_runner_failure_is_invalid_even_after_operator_stop(
    tmp_path: Path,
) -> None:
    run = run_contract("deepseek-goal-runner-failed")
    run["agent_kind"] = "deepseek-harness"
    (tmp_path / "STOP_ACK.json").write_text(
        json.dumps({"reason": "operator_batch_stop"})
    )
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")
    write_deepseek_goal_file(
        tmp_path,
        lifecycle={
            "goal_status": "active",
            "runner_state": "invalid_infrastructure",
            "failure_code": "goal_continuation_timeout",
            "detail": "goal-round driver did not start the next turn",
        },
    )

    report = build_integrity_report(tmp_path, run)

    assert report["benchmark_valid"] is False
    assert {reason["code"] for reason in report["reasons"]} == {
        "agent_goal_runner_invalid"
    }


def test_zero_submissions_and_low_spend_require_review_not_replacement(
    tmp_path: Path,
) -> None:
    run = run_contract("suspicious-clean-run")
    run["agent_cost_budget_usd"] = 10.0
    (tmp_path / "STOP_ACK.json").write_text(
        json.dumps({"reason": "operator_batch_stop"})
    )
    write_clean_exit(tmp_path / "CPU_TRIAL_EXIT.json")
    (tmp_path / "status.json").write_text(
        json.dumps({"ledger": {"accepted": 0}})
    )
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    (telemetry / "agent-cost.json").write_text(json.dumps({"total_usd": 7.5}))

    report = build_integrity_report(tmp_path, run)

    assert report["benchmark_valid"] is True
    assert report["replacement_required"] is False
    assert report["review_recommended"] is True
    assert {reason["code"] for reason in report["review_reasons"]} == {
        "zero_accepted_submissions",
        "low_budget_utilization",
    }
    assert report["observations"]["budget_utilization"] == 0.75


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

    report = build_integrity_report(tmp_path, run_contract("gpu-retry"))

    assert {reason["code"] for reason in report["reasons"]} == {
        "gpu_worker_retried",
        "gpu_termination_unconfirmed",
    }


def test_gpu_budget_feed_failure_is_invalid_even_when_attributed(
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
                "termination_reason": "budget_telemetry_unavailable",
            }
        )
    )

    report = build_integrity_report(tmp_path, run_contract("gpu-budget-feed"))

    assert report["benchmark_valid"] is False
    assert {reason["code"] for reason in report["reasons"]} == {
        "gpu_budget_telemetry_unavailable"
    }


def test_provider_gpu_initialization_failure_is_invalid(tmp_path: Path) -> None:
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    (registry / "job-1.json").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "attempt": 1,
                "status": "failed",
                "provider_terminal_error": (
                    "Isaac GPU/Vulkan initialization failed before required "
                    "outputs were produced"
                ),
            }
        )
    )

    report = build_integrity_report(tmp_path, run_contract("gpu-init"))

    assert report["benchmark_valid"] is False
    assert {reason["code"] for reason in report["reasons"]} == {
        "provider_gpu_initialization_failed"
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

    report = build_integrity_report(tmp_path, run_contract("expected-stop"))

    assert report["benchmark_valid"] is True


def test_unattributed_gpu_termination_requires_replacement(tmp_path: Path) -> None:
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    (registry / "job-1.json").write_text(
        json.dumps({"job_id": "job-1", "attempt": 1, "status": "terminated"})
    )

    report = build_integrity_report(tmp_path, run_contract("unknown-stop"))

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

    report = build_integrity_report(tmp_path, run_contract("budget-stop"))

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

    report = build_integrity_report(tmp_path, run_contract("lost-cancel"))

    assert report["benchmark_valid"] is False
    assert report["reasons"][0]["code"] == "gpu_cancellation_unacknowledged"


def test_terminal_cancel_registry_record_is_authoritative_acknowledgement(
    tmp_path: Path,
) -> None:
    request_id = "a" * 32
    (tmp_path / "control-events.jsonl").write_text(
        json.dumps(
            {
                "event": "cancel_requested",
                "request_id": request_id,
                "job_id": "job-1",
            }
        )
        + "\n"
    )
    registry = tmp_path / "gpu-job-registry"
    registry.mkdir()
    (registry / "job-1.json").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "status": "terminated",
                "termination_reason": "agent_cancelled",
                "cancel_request_id": request_id,
            }
        )
    )

    report = build_integrity_report(tmp_path, run_contract("cancelled"))

    assert report["benchmark_valid"] is True
    assert report["observations"]["cancel_requests_acknowledged"] == 1
