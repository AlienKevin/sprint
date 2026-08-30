from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from event_runtime.export.performance import (
    policy_queue_cost_fields,
    position_point_at_queue,
    step_auc,
)


def ledger():
    return {
        "origin_epoch_ms": 0,
        "end_epoch_ms": 10000,
        "cpu_start_epoch_ms": 0,
        "cpu_intervals": [(0, 10000)],
        "cpu_usd_per_second": 1,
        "training_intervals": [(1000, 8000)],
        "training_usd_per_second": 10,
        "api_events": [(1000, 0.2), (3000, 0.3), (4000, 100)],
    }


def reconciliation():
    return {"basis": "provider_settled_modal_endpoint", "modal_scale": 0.5}


def test_cost_at_queue_excludes_future_usage_and_later_compute():
    policy = {
        "queue_source_step_id": "a1-s1",
        "queue_source_public_step_id": 1,
        "enqueued_at": "1970-01-01T00:00:03Z",
        "submitted_at": "1970-01-01T00:00:09Z",
        "finished_at": "1970-01-01T00:00:10Z",
    }
    original = deepcopy(policy)
    result = policy_queue_cost_fields(policy, ledger(), reconciliation())
    assert result["cost_at_queue_api_usd"] == pytest.approx(0.5)
    assert result["cost_at_queue_compute_usd"] == pytest.approx((3 + 2 * 10) * 0.5)
    assert result["cost_at_queue_usd"] == pytest.approx(12.0)
    assert result["queue_epoch_ms"] == 3000
    assert result["cost_at_queue_basis"] == "provider_settled_modal_endpoint_estimate"
    assert policy == original


@pytest.mark.parametrize("extra", [
    {},
    {"enqueued_at": "1970-01-01T00:00:03Z", "enqueued_at_basis": "gpu_output_observed"},
    {"queue_source_step_id": "a1-s1", "enqueued_at": "not-a-time"},
])
def test_unresolved_queue_never_falls_back_to_result_or_final_cost(extra):
    result = policy_queue_cost_fields({
        "submitted_at": "1970-01-01T00:00:09Z",
        "finished_at": "1970-01-01T00:00:10Z",
        **extra,
    }, ledger(), reconciliation())
    assert result["cost_at_queue_usd"] is None
    assert result["queue_epoch_ms"] is None


def test_active_run_uses_deterministic_curve_without_claiming_settlement():
    result = policy_queue_cost_fields({
        "queue_source_step_id": "a1-s1",
        "enqueued_at": "1970-01-01T00:00:03Z",
    }, ledger(), {"basis": "deterministic_tariff_curve", "modal_scale": 1.0})
    assert result["cost_at_queue_usd"] == pytest.approx(23.5)
    assert result["cost_at_queue_basis"] == "deterministic_tariff_curve"


def result_point():
    return {
        "submission_index": 1,
        "epoch_ms": 9000,
        "hours_since_agent_launch": 8 / 3600,
        "cumulative_agent_cost_usd": 9.0,
        "cumulative_api_cost_usd": 1.0,
        "cumulative_modal_cost_usd_reconciled": 8.0,
        "continuous_score_mps": 2.0,
        "finished_at": "1970-01-01T00:00:09Z",
    }


def test_chart_coordinates_and_auc_use_first_queue_preserving_result_provenance():
    point = {
        **result_point(),
        "queue_source_step_id": "a1-s1",
        "queue_epoch_ms": 3000,
        "cost_at_queue_usd": 3.0,
        "cost_at_queue_api_usd": 0.5,
        "cost_at_queue_compute_usd": 2.5,
    }
    original = deepcopy(point)
    positioned = position_point_at_queue(point, 1000)
    assert positioned["epoch_ms"] == 3000
    assert positioned["hours_since_agent_launch"] == round(2 / 3600, 6)
    assert positioned["cumulative_agent_cost_usd"] == 3.0
    assert positioned["cumulative_api_cost_usd"] == 0.5
    assert positioned["cumulative_modal_cost_usd_reconciled"] == 2.5
    assert positioned["result_epoch_ms"] == 9000
    assert positioned["result_hours_since_agent_launch"] == 8 / 3600
    assert positioned["cost_at_result_usd"] == 9.0
    assert positioned["cost_at_result_api_usd"] == 1.0
    assert positioned["cost_at_result_compute_usd"] == 8.0
    assert positioned["continuous_score_mps"] == 2.0
    assert positioned["finished_at"] == point["finished_at"]
    assert step_auc([positioned], "cumulative_agent_cost_usd", 10) == pytest.approx(1.4)
    assert step_auc([point], "cumulative_agent_cost_usd", 10) == pytest.approx(0.2)
    assert point == original


@pytest.mark.parametrize("queue", [
    {},
    {"queue_epoch_ms": 3000, "cost_at_queue_usd": 3.0},
    {"queue_source_step_id": "a1-s1", "queue_epoch_ms": 3000},
])
def test_unresolved_queue_keeps_score_but_has_no_chart_or_auc_coordinate(queue):
    positioned = position_point_at_queue({**result_point(), **queue}, 1000)
    assert positioned["continuous_score_mps"] == 2.0
    assert positioned["cost_at_result_usd"] == 9.0
    assert positioned["epoch_ms"] is None
    assert positioned["hours_since_agent_launch"] is None
    assert positioned["cumulative_agent_cost_usd"] is None
    assert positioned["cumulative_api_cost_usd"] is None
    assert positioned["cumulative_modal_cost_usd_reconciled"] is None
    assert step_auc([positioned], "cumulative_agent_cost_usd", 10) == 0.0


def test_live_chart_points_and_aucs_use_queue_cost_not_result_cost():
    path = Path(__file__).resolve().parents[2] / "web/data/performance/current.json"
    if not path.exists():
        pytest.skip("Local exported performance fixture is not available")
    payload = json.loads(path.read_text())
    assert payload["cost"]["coordinate_basis"] == "first_agent_queue"
    assert payload["time"]["coordinate_basis"] == "first_agent_queue"
    for collection in (payload["runs"], payload["models"]):
        for group in collection:
            for point in group["points"]:
                assert point["epoch_ms"] == point["queue_epoch_ms"]
                assert point["cumulative_agent_cost_usd"] == point["cost_at_queue_usd"]
                assert point["cumulative_api_cost_usd"] == point["cost_at_queue_api_usd"]
                assert point["cumulative_modal_cost_usd_reconciled"] == point["cost_at_queue_compute_usd"]
                if point["queue_epoch_ms"] is not None:
                    assert point["result_epoch_ms"] >= point["queue_epoch_ms"]
            assert group["summary"]["cost_auc_mps_at_common_cap"] == pytest.approx(
                step_auc(
                    group["points"],
                    "cost_at_queue_usd",
                    payload["cost"]["common_auc_cap_usd"],
                )
            )
