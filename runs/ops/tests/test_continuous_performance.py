from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "runs/build_continuous_performance.py"
SPEC = importlib.util.spec_from_file_location(
    "build_continuous_performance", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
continuous = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = continuous
SPEC.loader.exec_module(continuous)


@pytest.mark.parametrize(
    ("distance_m", "elapsed_s", "expected"),
    [
        (0.0, 10.0, 0.0),
        (10.0, 2.0, 0.5),
        (50.0, 10.0, 2.5),
        (100.0, 20.0, 5.0),
        (-1.0, 10.0, 0.0),
        (10.0, 0.0, 0.0),
    ],
)
def test_completion_adjusted_speed(
    distance_m: float, elapsed_s: float, expected: float
) -> None:
    assert continuous.completion_adjusted_speed(distance_m, elapsed_s) == pytest.approx(
        expected
    )


def test_step_auc_uses_best_so_far_and_common_cap() -> None:
    points = [
        {"cost": 2.0, "continuous_score_mps": 1.0},
        {"cost": 5.0, "continuous_score_mps": 0.5},
        {"cost": 8.0, "continuous_score_mps": 3.0},
        {"cost": 12.0, "continuous_score_mps": 100.0},
    ]
    # [0,2): 0; [2,8): 1; [8,10]: 3 => (0 + 6 + 6) / 10.
    assert continuous.step_auc(points, "cost", 10.0) == pytest.approx(1.2)


def test_cost_ledger_integrates_requests_and_allocation_intervals() -> None:
    timeline = {
        "clock": {"origin_epoch_ms": 0, "end_epoch_ms": 4000},
        "events": [
            {"kind": "cpu_allocated", "epoch_ms": 0},
            {"kind": "gpu_allocated", "epoch_ms": 1000, "lease_id": "lease"},
            {
                "kind": "model_request_usage",
                "epoch_ms": 2000,
                "calculated_cost_usd": 0.5,
            },
            {"kind": "gpu_released", "epoch_ms": 3000, "lease_id": "lease"},
        ],
        "resource_usage_summary": {
            "resource_contract": {
                "cpu_agent": {
                    "physical_cpu_cores": 1,
                    "memory_mb": 0,
                    "gpu_count": 0,
                },
                "training_worker": {
                    "physical_cpu_cores": 0,
                    "memory_mb": 0,
                    "gpu_count": 1,
                    "gpu_type": "A10G",
                },
            },
            "modal_estimate": {
                "pricing_snapshot": {
                    "rates_usd_per_second": {
                        "CPU": "1",
                        "Memory": "0",
                        "A10G": "10",
                    }
                },
                "by_role": {
                    "cpu_agent": {"estimated_cost_usd": 4},
                    "training_gpu": {"estimated_cost_usd": 20},
                },
            },
            "modal_provider_billing": {
                "by_role_usd": {"cpu_agent": 4, "training_gpu": 20}
            },
        },
    }
    ledger = continuous.build_cost_ledger(timeline)
    assert continuous.cumulative_cost_at_epoch(ledger, 500) == pytest.approx(0.5)
    assert continuous.cumulative_cost_at_epoch(ledger, 2500) == pytest.approx(18.0)
    assert continuous.cumulative_cost_at_epoch(ledger, 5000) == pytest.approx(24.5)


def test_dashboard_loads_continuous_readouts() -> None:
    app = (ROOT / "sprint-web/app.js").read_text()
    page = (ROOT / "sprint-web/index.html").read_text()
    assert "/data/performance/r8-continuous.json" in app
    assert "continuous_score_mps" in app
    assert 'id="cost-scores"' in page
    assert 'id="time-scores"' in page
    assert "auc-bar-row" in app
    assert page.index("Performance vs cost") < page.index("Performance over time")
    assert "representative-lane verifier captures" in page
