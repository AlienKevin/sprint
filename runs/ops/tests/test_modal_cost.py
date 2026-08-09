from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS))

import modal_cost  # noqa: E402
import sprintctl  # noqa: E402


CONTRACT = {
    "cpu_agent": {
        "physical_cpu_cores": 4,
        "vcpus_equivalent": 8,
        "memory_mb": 16384,
        "gpus": 0,
    },
    "training_worker": {
        "physical_cpu_cores": 8,
        "vcpus_equivalent": 16,
        "memory_mb": 32768,
        "gpu_count": 1,
        "gpu_type": "A10G",
    },
    "verifier": {
        "physical_cpu_cores": 8,
        "vcpus_equivalent": 16,
        "memory_mb": 32768,
        "gpu_count": 1,
        "gpu_type": "A10G",
    },
}


def run_payload() -> dict:
    return {
        "run_id": "cost-run",
        "created_at": "2026-08-08T04:10:00Z",
        "modal_profile": "test",
        "app_name": "sprint-cost-run",
        "training_app_name": "sprint-cost-run-training",
        "verifier_app_name": "sprint-cost-run-verifier",
        "volume_name": "sprint-cost-run-volume",
    }


def test_estimate_preserves_role_and_billing_categories() -> None:
    payload = modal_cost.estimate_cost(
        resource_contract=CONTRACT,
        allocated_ms_by_role={
            "cpu_agent": 3_600_000,
            "training_gpu": 1_800_000,
            "verifier_gpu": 60_000,
        },
    )

    assert payload["by_role"]["cpu_agent"]["quantities"] == {
        "CPU": 4.0,
        "Memory": 16.0,
        "A10G": 0.0,
    }
    assert payload["by_role"]["cpu_agent"]["estimated_cost_usd"] == 0.95184
    assert payload["by_role"]["training_gpu"]["estimated_cost_usd"] == 1.50264
    assert payload["by_role"]["verifier_gpu"]["estimated_cost_usd"] == 0.050088
    assert payload["estimated_cost_usd"] == 2.504568
    assert payload["by_category_usd"] == {
        "A10G": 0.56916,
        "CPU": 1.1542176,
        "Memory": 0.7811904,
    }


def test_provider_rows_are_filtered_and_split_by_role_and_category() -> None:
    rows = [
        {
            "object_id": "cpu",
            "description": "sprint-cost-run",
            "environment": "main",
            "interval_start": "2026-08-08T04:00:00",
            "resource": "CPU",
            "cost": "1.25",
        },
        {
            "object_id": "cpu",
            "description": "sprint-cost-run",
            "environment": "main",
            "interval_start": "2026-08-08T04:00:00",
            "resource": "Memory",
            "cost": "0",
        },
        {
            "object_id": "training",
            "description": "sprint-cost-run-training",
            "environment": "main",
            "interval_start": "2026-08-08T04:00:00",
            "resource": "A10G",
            "cost": "2.5",
        },
        {
            "object_id": "verifier",
            "description": "sprint-cost-run-verifier",
            "environment": "main",
            "interval_start": "2026-08-08T04:00:00",
            "resource": "Memory",
            "cost": "0.75",
        },
        {
            "object_id": "unrelated",
            "description": "someone-else",
            "resource": "CPU",
            "cost": "999",
        },
    ]

    payload = modal_cost.aggregate_billing_rows(rows, run=run_payload())

    assert payload["provider_cost_precredits_usd"] == 4.5
    assert payload["by_role_usd"] == {
        "cpu_agent": 1.25,
        "training_gpu": 2.5,
        "verifier_gpu": 0.75,
    }
    assert payload["by_category_usd"] == {
        "A10G": 2.5,
        "CPU": 1.25,
        "Memory": 0.75,
    }
    assert len(payload["items"]) == 4


def test_billing_objects_must_have_unique_descriptions() -> None:
    run = run_payload()
    run["volume_name"] = run["app_name"]
    try:
        modal_cost.billing_object_roles(run)
    except ValueError as exc:
        assert "shared by cpu_agent and volume" in str(exc)
    else:
        raise AssertionError("ambiguous Modal billing descriptions were accepted")


def test_continuous_verifier_result_does_not_end_running_cpu_billing(
    tmp_path: Path,
) -> None:
    state = tmp_path / "cost-run"
    job = state / "harbor-jobs" / "cost-run"
    continuous = job / "task" / "artifacts/continuous/attempts/0001"
    continuous.mkdir(parents=True)
    (continuous / "result.json").write_text(
        json.dumps({"finished_at": "2026-08-08T04:30:00Z"})
    )
    run = run_payload()
    run["job_path"] = str(job)
    start, stopped = modal_cost.run_bounds(state, run)
    assert start == dt.datetime(2026, 8, 8, 4, 10, tzinfo=dt.timezone.utc)
    assert stopped is None

    job.mkdir(parents=True, exist_ok=True)
    (job / "result.json").write_text(
        json.dumps({"finished_at": "2026-08-08T04:40:00Z"})
    )
    _, stopped = modal_cost.run_bounds(state, run)
    assert stopped == dt.datetime(2026, 8, 8, 4, 40, tzinfo=dt.timezone.utc)


def test_collection_waits_for_complete_hour_then_persists_provider_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "cost-run"
    state.mkdir()
    (state / "run.json").write_text(json.dumps(run_payload()))
    (state / "STOP_ACK.json").write_text(
        json.dumps({"acknowledged_at": "2026-08-08T04:50:00Z"})
    )

    pending = modal_cost.collect_provider_billing(
        state, now=dt.datetime(2026, 8, 8, 5, 4, tzinfo=dt.timezone.utc)
    )
    assert pending["status"] == "pending"
    assert pending["eligible_at"] == "2026-08-08T05:05:00Z"

    rows = [
        {
            "object_id": "cpu",
            "description": "sprint-cost-run",
            "environment": "main",
            "interval_start": "2026-08-08T04:00:00",
            "resource": "CPU",
            "cost": "0.50",
        },
        {
            "object_id": "cpu",
            "description": "sprint-cost-run",
            "environment": "main",
            "interval_start": "2026-08-08T04:00:00",
            "resource": "Memory",
            "cost": "0",
        },
    ]
    calls = []
    monkeypatch.setattr(modal_cost.shutil, "which", lambda name: "/tools/modal")

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(rows), "")

    complete = modal_cost.collect_provider_billing(
        state,
        now=dt.datetime(2026, 8, 8, 5, 6, tzinfo=dt.timezone.utc),
        runner=runner,
        volume_scanner=lambda run, duration_seconds: {
            "status": "captured",
            "logical_bytes_at_collection": 123,
            "duration_seconds": duration_seconds,
        },
    )
    assert complete["provider_complete"] is True
    assert complete["provider_cost_precredits_usd"] == 0.5
    assert complete["by_role_category_usd"] == {
        "cpu_agent": {"CPU": 0.5, "Memory": 0.0}
    }
    assert "--show-resources" in calls[0][0]
    assert calls[0][0][0] == "/tools/modal"
    assert calls[0][1]["env"]["MODAL_PROFILE"] == "test"
    assert len(complete["provider_report_sha256"]) == 64
    assert len(complete["selected_items_sha256"]) == 64
    assert complete["volume_storage"] == {
        "status": "captured",
        "logical_bytes_at_collection": 123,
        "duration_seconds": 2400.0,
    }
    persisted = json.loads((state / "telemetry" / "modal-cost.json").read_text())
    assert persisted == complete
    ready, details = sprintctl.modal_billing_ready(state, "cost-run")
    assert ready is True
    assert details == []


def test_modal_billing_readiness_rejects_pending_artifact(tmp_path: Path) -> None:
    state = tmp_path / "cost-run"
    (state / "telemetry").mkdir(parents=True)
    (state / "telemetry" / "modal-cost.json").write_text(
        json.dumps(
            {
                "schema_version": modal_cost.SCHEMA_VERSION,
                "run_id": "cost-run",
                "provider_complete": False,
                "pending_reason": "waiting",
                "by_role_usd": {},
                "by_category_usd": {},
                "by_role_category_usd": {},
                "provider_cost_precredits_usd": None,
            }
        )
    )
    ready, details = sprintctl.modal_billing_ready(state, "cost-run")
    assert ready is False
    assert any("not complete" in detail for detail in details)
    assert any("Volume storage snapshot" in detail for detail in details)


def test_collection_waits_when_expected_role_is_missing(tmp_path: Path) -> None:
    state = tmp_path / "cost-run"
    (state / "telemetry").mkdir(parents=True)
    (state / "run.json").write_text(json.dumps(run_payload()))
    (state / "STOP_ACK.json").write_text(
        json.dumps({"acknowledged_at": "2026-08-08T04:50:00Z"})
    )
    (state / "telemetry" / "unified-timeline.json").write_text(
        json.dumps(
            {
                "resource_usage_summary": {
                    "cpu_agent": {"allocated_ms": 1},
                    "training_gpu": {"allocated_ms": 1},
                    "verifier_gpu": {"allocated_ms": 0},
                }
            }
        )
    )

    rows = [
        {
            "object_id": "cpu",
            "description": "sprint-cost-run",
            "resource": "CPU",
            "cost": "0.5",
        },
        {
            "object_id": "cpu",
            "description": "sprint-cost-run",
            "resource": "Memory",
            "cost": "0",
        },
    ]

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, json.dumps(rows), "")

    payload = modal_cost.collect_provider_billing(
        state,
        now=dt.datetime(2026, 8, 8, 5, 6, tzinfo=dt.timezone.utc),
        runner=runner,
        volume_scanner=lambda run, duration_seconds: {},
    )
    assert payload["provider_complete"] is False
    assert payload["missing_billed_roles"] == ["training_gpu"]
    assert payload["pending_reason"] == "provider_report_missing_expected_role_rows"


def test_expected_roles_use_source_lifecycle_even_without_timeline(
    tmp_path: Path,
) -> None:
    state = tmp_path / "cost-run"
    write_path = state / "telemetry" / "gpu_timeline.jsonl"
    write_path.parent.mkdir(parents=True)
    write_path.write_text(json.dumps({"detail": {"event": "gpu_allocated"}}) + "\n")
    verifier = (
        state
        / "harbor-jobs"
        / "cost-run"
        / "task__abc"
        / "verifier"
        / "telemetry"
        / "lifecycle.json"
    )
    verifier.parent.mkdir(parents=True)
    verifier.write_text("{}")

    assert modal_cost._expected_billed_roles(state) == {
        "cpu_agent",
        "training_gpu",
        "verifier_gpu",
    }


def test_collection_waits_for_volume_storage_snapshot(tmp_path: Path) -> None:
    state = tmp_path / "cost-run"
    state.mkdir()
    (state / "run.json").write_text(json.dumps(run_payload()))
    (state / "STOP_ACK.json").write_text(
        json.dumps({"acknowledged_at": "2026-08-08T04:50:00Z"})
    )
    rows = [
        {
            "object_id": "cpu",
            "description": "sprint-cost-run",
            "resource": "CPU",
            "cost": "0.5",
        },
        {
            "object_id": "cpu",
            "description": "sprint-cost-run",
            "resource": "Memory",
            "cost": "0",
        },
    ]

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, json.dumps(rows), "")

    payload = modal_cost.collect_provider_billing(
        state,
        now=dt.datetime(2026, 8, 8, 5, 6, tzinfo=dt.timezone.utc),
        runner=runner,
        volume_scanner=lambda run, duration_seconds: {
            "status": "unavailable",
            "reason": "provider_volume_scan_failed",
        },
    )
    assert payload["provider_complete"] is False
    assert payload["pending_reason"] == "provider_volume_storage_snapshot_unavailable"
