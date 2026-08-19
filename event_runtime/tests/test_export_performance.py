from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "event_runtime/export/performance.py"
SPEC = importlib.util.spec_from_file_location("event_performance", MODULE_PATH)
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


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek/deepseek-v4-flash", "deepseek"),
        ("openai/gpt-5.6-luna", "luna"),
        ("openai/gpt-5.6-sol", "sol"),
    ],
)
def test_model_family_supports_controlled_openrouter_models(
    model: str, expected: str
) -> None:
    assert continuous.model_family(model) == expected


def test_frontier_replays_are_union_of_cost_and_time_record_setters() -> None:
    points = [
        {
            "source_run_id": "run-1",
            "policy_sha256": "a" * 64,
            "submission_index": 1,
            "cumulative_agent_cost_usd": 2.0,
            "hours_since_agent_launch": 8.0,
            "continuous_score_mps": 1.0,
        },
        {
            "source_run_id": "run-1",
            "policy_sha256": "b" * 64,
            "submission_index": 2,
            "cumulative_agent_cost_usd": 8.0,
            "hours_since_agent_launch": 2.0,
            "continuous_score_mps": 2.0,
        },
        {
            "source_run_id": "run-1",
            "policy_sha256": "c" * 64,
            "submission_index": 3,
            "cumulative_agent_cost_usd": 9.0,
            "hours_since_agent_launch": 9.0,
            "continuous_score_mps": 0.5,
        },
    ]

    selected = continuous.policy_replay_points([{"points": points}])

    assert {point["policy_sha256"] for point in selected} == {
        "a" * 64,
        "b" * 64,
        "c" * 64,
    }


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
                # Provider reconciliation is audit-only and must not mutate the
                # deterministic comparison ledger or an agent's live cost.
                "by_role_usd": {"cpu_agent": 400, "training_gpu": 2000}
            },
        },
    }
    ledger = continuous.build_cost_ledger(timeline)
    assert continuous.cumulative_cost_at_epoch(ledger, 500) == pytest.approx(0.5)
    assert continuous.cumulative_cost_at_epoch(ledger, 2500) == pytest.approx(18.0)
    assert continuous.cumulative_cost_at_epoch(ledger, 5000) == pytest.approx(24.5)


def test_run_is_terminal_requires_durable_stop_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(continuous, "RUNS", tmp_path)
    run = tmp_path / "run-a"
    run.mkdir()

    assert not continuous.run_is_terminal("run-a")

    (run / "STOP_REQUESTED.json").write_text("{}\n")
    assert not continuous.run_is_terminal("run-a")

    (run / "STOP_ACK.json").write_text("{}\n")
    assert continuous.run_is_terminal("run-a")


def test_dashboard_loads_continuous_readouts() -> None:
    app = (ROOT / "web/app.js").read_text()
    page = (ROOT / "web/index.html").read_text()
    styles = (ROOT / "web/styles.css").read_text()
    timeline_page = (ROOT / "web/timeline.html").read_text()
    timeline_app = (ROOT / "web/timeline.js").read_text()
    assert "/data/performance/current.json" in app
    assert "continuous_score_mps" in app
    assert (
        "const plottable=point=>finite(point[xKey])&&finite(point.continuous_score_mps)"
        in app
    )
    assert "point[xKey]<=aucCap" not in app
    assert "submission${onFrontier?' frontier':''}" in app
    assert "class:'submission-hit'" in app
    assert "class:'submission-target'" in app
    assert "el.getScreenCTM()" in app
    assert "nearest.distance<=22**2" in app
    assert "app.js?v=20260819-5" in page
    assert "const APP_VERSION = '20260819-5'" in app
    assert "refreshVersion" in app
    assert "setInterval(refresh,30000)" in app
    assert "visibilitychange" in app
    assert "window.addEventListener('focus'" in app
    assert "window.addEventListener('pageshow'" in app
    assert "This policy has no archived website replay." in app
    assert 'id="cost-scores"' in page
    assert 'id="time-scores"' in page
    assert "auc-bar-row" in app
    assert "color:'#4D6BFF'" in app
    assert "color:'#66D693'" in app
    assert "color:'#f1c35b'" in app
    assert "--deep: #4D6BFF" in styles
    assert "--luna: #66D693" in styles
    assert "background: var(--cost-cpu)" in styles
    assert "background: var(--cost-training)" in styles
    assert "--deepseek:#4D6BFF" in timeline_page
    assert "--luna:#66D693" in timeline_page
    assert "--sol:#bdbdb9" in timeline_page
    assert 'id="policy-cost-chart"' in timeline_page
    assert 'id="policy-replay-frame"' in timeline_page
    assert "/data/performance/current.json" in timeline_app
    assert "renderPolicyChart" in timeline_app
    assert "showPolicy(point)" in timeline_app
    assert "Open rendered policy" in timeline_app
    assert "setModelAccent" in timeline_app
    assert "<title>The Race to AGI4ALL</title>" in page
    assert "The Race to<br><em>AGI4ALL.</em>" in page
    assert '<span class="brand">AGI4ALL</span>' in page
    assert "color: var(--bg)" in styles
    assert "background: var(--text)" in styles
    assert 'id="live"' not in page
    assert "$('#live')" not in app
    assert "AI agents compete to produce the fastest policy at the lowest cost." in page
    assert "The Race to AGI4ALL · Race control" in timeline_page
    assert page.index("Performance vs cost") < page.index("Performance over time")
    assert "The Time-Adjusted Effective Speed compares competitors" in page
    assert 'id="time-performance" hidden' in page
    assert 'id="readout-detail"' in page
    assert 'id="readout-replay"' in page
    assert 'id="readout-timeline"' in page
    assert 'id="policy-grid"' not in page
    assert 'id="run-links"' not in page
    assert "Cost-Adjusted Effective Speed" in app
    assert "Best Cost-Adjusted Effective Speed" not in app
    assert "cost_auc_mps_at_common_cap" in app
    assert "effective speed (m/s)" in app
    assert "showReadout" in app
    assert "active_provisional" in app
    assert "combined active-trial cost" in app
    assert "best of ${trials}" not in app
    assert "hours_since_agent_launch" in app
    assert "hours_since_agent launch" not in app
    assert "best of 3 trials" not in app


def test_trusted_pose_capture_index_recovers_renderer_failure_and_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-a"
    attempt = (
        tmp_path
        / run_id
        / "harbor-jobs"
        / "job"
        / "task"
        / "artifacts"
        / "continuous"
        / "attempts"
        / "0001-policy.pt"
    )
    verifier = attempt / "verifier"
    verifier.mkdir(parents=True)
    digest = "a" * 64
    (attempt / "result.json").write_text('{"artifact_sha256":"' + digest + '"}\n')
    (verifier / "replay.json").write_text('{"trusted":true}\n')
    monkeypatch.setattr(continuous, "RUNS", tmp_path)

    index = continuous.trusted_pose_capture_index([run_id])
    resolved = continuous.resolve_pose_capture(
        run_id=run_id, policy_hash=digest, trusted=index
    )

    assert resolved == (verifier / "replay.json", "trusted_verifier_replay")


def test_resolve_pose_capture_prefers_validated_website_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-a"
    digest = "b" * 64
    rendered = tmp_path / run_id / "captures" / f"frontier-{digest[:12]}.json"
    rendered.parent.mkdir(parents=True)
    rendered.write_text('{"rendered":true}\n')
    trusted = tmp_path / "trusted.json"
    trusted.write_text('{"trusted":true}\n')
    monkeypatch.setattr(continuous, "RUNS", tmp_path)

    assert continuous.resolve_pose_capture(
        run_id=run_id, policy_hash=digest, trusted={digest: trusted}
    ) == (rendered, "validated_website_capture")
