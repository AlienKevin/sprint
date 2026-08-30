from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime
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
def test_effective_speed(distance_m: float, elapsed_s: float, expected: float) -> None:
    assert continuous.effective_speed(distance_m, elapsed_s) == pytest.approx(expected)


def test_capture_scoring_includes_progress_at_timeout(tmp_path: Path) -> None:
    capture = tmp_path / "replay.json"
    capture.write_text(
        json.dumps(
            {
                "body_names": ["torso_link"],
                "frames": [
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        [2.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                "runs": [
                    {
                        "valid": False,
                        "termination_reason": "timeout",
                        "checks": [
                            {"name": "finished", "passed": False},
                            {"name": "in_lane", "passed": True},
                            {"name": "self_collision", "passed": True},
                        ],
                    }
                ],
                "fps": 50.0,
                "representative_lane": 0,
            }
        )
    )

    score = continuous.score_capture(capture)

    assert score["max_legal_distance_m"] == pytest.approx(10.0)
    assert score["time_to_max_legal_distance_s"] == pytest.approx(2.0)
    assert score["continuous_score_mps"] == pytest.approx(0.5)


def test_step_auc_uses_best_so_far_and_common_cap() -> None:
    points = [
        {"cost": 2.0, "continuous_score_mps": 1.0},
        {"cost": 5.0, "continuous_score_mps": 0.5},
        {"cost": 8.0, "continuous_score_mps": 3.0},
        {"cost": 12.0, "continuous_score_mps": 100.0},
    ]
    # [0,2): 0; [2,8): 1; [8,10]: 3 => (0 + 6 + 6) / 10.
    assert continuous.step_auc(points, "cost", 10.0) == pytest.approx(1.2)


def test_completed_comparison_uses_per_trial_cost_cap() -> None:
    runs = [
        {
            "run_id": "batch-luna-1",
            "model": "openai/gpt-5.6-luna",
            "points": [],
            "summary": {
                "final_agent_cost_usd": 9.9,
                "missing_readout_indices": [],
            },
        },
        {
            "run_id": "batch-sol-1",
            "model": "openai/gpt-5.6-sol",
            "points": [],
            "summary": {
                "final_agent_cost_usd": 10.0,
                "missing_readout_indices": [],
            },
        },
    ]
    ledgers = {
        "batch-luna-1": {"origin_epoch_ms": 0},
        "batch-sol-1": {"origin_epoch_ms": 0},
    }

    _, cap = continuous.aggregate_models(
        runs,
        common_time_cap=1.0,
        cost_ledgers=ledgers,
        per_trial_cost_cap=10.0,
    )

    assert cap == pytest.approx(10.0)


def test_comparison_supports_unequal_model_cohort_sizes() -> None:
    runs = [
        {
            "run_id": "batch-luna-1",
            "model": "openai/gpt-5.6-luna",
            "points": [],
            "summary": {
                "final_agent_cost_usd": 10.0,
                "missing_readout_indices": [],
            },
        },
        *[
            {
                "run_id": f"batch-sol-{trial}",
                "model": "openai/gpt-5.6-sol",
                "points": [],
                "summary": {
                    "final_agent_cost_usd": 10.0,
                    "missing_readout_indices": [],
                },
            }
            for trial in range(1, 4)
        ],
    ]
    ledgers = {run["run_id"]: {"origin_epoch_ms": 0} for run in runs}

    models, cap = continuous.aggregate_models(
        runs,
        common_time_cap=1.0,
        cost_ledgers=ledgers,
        per_trial_cost_cap=10.0,
    )

    assert {model["family"]: len(model["run_ids"]) for model in models} == {
        "luna": 1,
        "sol": 3,
    }
    assert cap == pytest.approx(10.0)


def test_completed_comparison_carries_frontier_to_declared_budget() -> None:
    runs = [
        {
            "run_id": "batch-luna-1",
            "model": "openai/gpt-5.6-luna",
            "points": [],
            "summary": {
                "final_agent_cost_usd": 9.9,
                "missing_readout_indices": [],
            },
        },
        {
            "run_id": "batch-sol-1",
            "model": "openai/gpt-5.6-sol",
            "points": [],
            "summary": {
                "final_agent_cost_usd": 10.0,
                "missing_readout_indices": [],
            },
        },
    ]
    ledgers = {
        "batch-luna-1": {"origin_epoch_ms": 0},
        "batch-sol-1": {"origin_epoch_ms": 0},
    }

    _, cap = continuous.aggregate_models(
        runs,
        common_time_cap=1.0,
        cost_ledgers=ledgers,
        per_trial_cost_cap=10.0,
    )

    assert cap == pytest.approx(10.0)


def test_single_family_snapshot_keeps_per_trial_cap() -> None:
    runs = [
        {
            "run_id": "batch-luna-1",
            "model": "openai/gpt-5.6-luna",
            "points": [],
            "summary": {
                "final_agent_cost_usd": 4.25,
                "missing_readout_indices": [],
            },
        },
        {
            "run_id": "batch-luna-2",
            "model": "openai/gpt-5.6-luna",
            "points": [],
            "summary": {
                "final_agent_cost_usd": 5.75,
                "missing_readout_indices": [],
            },
        },
    ]
    ledgers = {
        "batch-luna-1": {"origin_epoch_ms": 0},
        "batch-luna-2": {"origin_epoch_ms": 0},
    }

    models, cap = continuous.aggregate_models(
        runs,
        common_time_cap=1.0,
        cost_ledgers=ledgers,
        per_trial_cost_cap=10.0,
    )

    assert [model["family"] for model in models] == ["luna"]
    assert cap == pytest.approx(10.0)


def test_cli_prints_empty_single_family_snapshot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {
        "runs": [],
        "models": [
            {
                "family": "luna",
                "summary": {
                    "readout_count": 0,
                    "best_continuous_score_mps": None,
                    "cost_auc_mps_at_common_cap": 0.0,
                    "time_auc_mps_at_common_cap": 0.0,
                },
            }
        ],
    }
    monkeypatch.setattr(continuous, "build", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(sys, "argv", ["performance.py"])

    assert continuous.main() == 0
    assert "luna: 0 merged readouts, best=n/a" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek/deepseek-v4-flash", "flash-baidu"),
        ("deepseek/deepseek-v4-flash-0731", "flash-baidu"),
        ("deepseek/deepseek-v4-flash-vision-exp", "deepseek"),
        ("deepseek/deepseek-v4-pro-0813", "pro-alibaba"),
        ("openai/gpt-5.6-luna", "luna"),
        ("openai/gpt-5.6-sol", "sol"),
    ],
)
def test_model_family_supports_controlled_openrouter_models(
    model: str, expected: str
) -> None:
    assert continuous.model_family(model) == expected


def test_flash_and_pro_are_aggregated_as_separate_competitors() -> None:
    runs = [
        {
            "run_id": f"batch-{name}-{trial}",
            "model": model,
            "points": [],
            "summary": {
                "final_agent_cost_usd": cost,
                "missing_readout_indices": [],
            },
        }
        for name, model, cost in (
            ("flash", "deepseek/deepseek-v4-flash-0731", 7.5),
            ("pro", "deepseek/deepseek-v4-pro-0813", 9.5),
        )
        for trial in range(1, 4)
    ]
    ledgers = {run["run_id"]: {"origin_epoch_ms": 0} for run in runs}

    models, cap = continuous.aggregate_models(
        runs,
        common_time_cap=1.0,
        cost_ledgers=ledgers,
        per_trial_cost_cap=10.0,
    )

    assert [model["family"] for model in models] == [
        "flash-baidu",
        "pro-alibaba",
    ]
    assert [len(model["run_ids"]) for model in models] == [3, 3]
    assert cap == pytest.approx(10.0)


def test_model_curve_uses_each_policy_own_trial_cost() -> None:
    runs = [
        {
            "run_id": f"batch-luna-{trial}",
            "model": "openai/gpt-5.6-luna",
            "points": [
                {
                    "epoch_ms": 2000,
                    "hours_since_agent_launch": 0.5,
                    "submission_index": trial,
                    "policy_sha256": str(trial) * 64,
                    "continuous_score_mps": float(trial),
                    "cumulative_agent_cost_usd": cost,
                }
            ],
            "summary": {
                "final_agent_cost_usd": 9.9,
                "missing_readout_indices": [],
            },
        }
        for trial, cost in ((1, 2.0), (2, 4.0), (3, 6.0))
    ]
    # If costs were still aggregated at the shared event timestamp, all three
    # points would be placed at 12 USD.
    ledgers = {
        run["run_id"]: {
            "origin_epoch_ms": 0,
            "end_epoch_ms": 2000,
            "api_events": [(2000, float(index * 2))],
            "cpu_intervals": [],
            "training_intervals": [],
            "rates": {"cpu_per_s": 0.0, "training_per_s": 0.0},
        }
        for index, run in enumerate(runs, start=1)
    }

    models, cap = continuous.aggregate_models(
        runs,
        common_time_cap=1.0,
        cost_ledgers=ledgers,
        per_trial_cost_cap=10.0,
    )

    assert cap == pytest.approx(10.0)
    assert [point["trial_agent_cost_usd"] for point in models[0]["points"]] == [
        2.0,
        4.0,
        6.0,
    ]
    assert [point["cumulative_agent_cost_usd"] for point in models[0]["points"]] == [
        2.0,
        4.0,
        6.0,
    ]


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


def test_performance_curve_reconciles_modal_to_authoritative_settlement() -> None:
    timeline = {
        "clock": {"origin_epoch_ms": 0, "end_epoch_ms": 4000},
        "comparison_summary": {
            "final_agent_total_cost_usd": 12.5,
            "final_api_cost_usd": 0.5,
        },
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
        },
    }
    ledger = continuous.build_cost_ledger(timeline)
    reconciliation = continuous.settled_cost_reconciliation(timeline, ledger)

    assert reconciliation["basis"] == "provider_settled_modal_endpoint"
    assert reconciliation["modal_scale"] == pytest.approx(0.5)
    assert continuous.settled_cost_components_at_epoch(
        ledger, 2500, reconciliation
    ) == pytest.approx(
        {"model_api_usd": 0.5, "agent_modal_usd": 8.75, "total_usd": 9.25}
    )
    assert continuous.settled_cost_components_at_epoch(
        ledger, 5000, reconciliation
    )["total_usd"] == pytest.approx(12.5)


def test_deepseek_policy_submission_chapters_remain_authoritative() -> None:
    run_id = "s10-vexp-r120-20260828-deepseek-1"
    trajectory = json.loads(
        (ROOT / f"web/data/trajectories/{run_id}.json").read_text()
    )
    outline = json.loads(
        (ROOT / f"web/data/trajectories/{run_id}.outline.json").read_text()
    )
    policies = sorted(
        json.loads((ROOT / f"web/data/policies/{run_id}.json").read_text())[
            "policies"
        ],
        key=lambda policy: policy["enqueued_at"],
    )

    def epoch(value: str) -> float:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()

    mapped = []
    for policy_number in (4, 5, 7):
        policy = policies[policy_number - 1]
        # Queue timestamps can lag the actual trace by hours.  The exporter
        # now records the authoritative source step; test that provenance
        # rather than reintroducing a nearest-timestamp guess.
        step = next(
            item for item in trajectory["steps"]
            if item["step_id"] == policy["queue_source_step_id"]
        )
        step_number = step.get("public_step_id", step.get("attempt_step_id"))
        chapter = next(
            chapter
            for chapter in outline["chapters"]
            if chapter["start"]["public_step_id"]
            <= step_number
            <= chapter["end"]["public_step_id"]
        )
        mapped.append((policy_number, step_number, chapter["id"]))

    assert mapped == [
        (4, 283, "tuning-motion-frequency"),
        (5, 293, "validating-near-78-metre-runs"),
        (7, 313, "queuing-final-candidate"),
    ]


def test_glm_third_policy_has_recoverable_replay_and_chapter_mapping() -> None:
    run_id = "claude-goal-20260828-0237-main-glm-2"
    trajectory = json.loads(
        (ROOT / f"web/data/trajectories/{run_id}.json").read_text()
    )
    outline = json.loads(
        (ROOT / f"web/data/trajectories/{run_id}.outline.json").read_text()
    )
    policy = sorted(
        json.loads((ROOT / f"web/data/policies/{run_id}.json").read_text())[
            "policies"
        ],
        key=lambda item: item["enqueued_at"],
    )[2]
    epoch = lambda value: datetime.fromisoformat(  # noqa: E731
        value.replace("Z", "+00:00")
    ).timestamp()
    step = min(
        trajectory["steps"],
        key=lambda item: abs(epoch(item["timestamp"]) - epoch(policy["enqueued_at"])),
    )
    step_number = step.get("public_step_id", step.get("attempt_step_id"))
    chapter_index, chapter = next(
        (index, chapter)
        for index, chapter in enumerate(outline["chapters"], 1)
        if chapter["start"]["public_step_id"]
        <= step_number
        <= chapter["end"]["public_step_id"]
    )
    capture_slug = f"frontier-{policy['policy_sha256'][:12]}"

    assert (step_number, chapter_index, chapter["id"]) == (
            152,
        8,
        "launch-final-speed-push",
    )
    assert policy["replay_ready"] is False
    assert policy["replay_url"] is None
    assert (ROOT / f"web/captures/{capture_slug}.json").is_file()


def test_cost_ledger_stops_cpu_at_reconciled_allocation_end() -> None:
    timeline = {
        "clock": {"origin_epoch_ms": 0, "end_epoch_ms": 5000},
        "events": [{"kind": "cpu_allocated", "epoch_ms": 0}],
        "resource_usage_summary": {
            "cpu_agent": {
                "allocated_ms": 3000,
                "intervals": [{"start_epoch_ms": 0, "end_epoch_ms": 3000}],
            },
            "training_gpu": {"allocated_ms": 0, "intervals": []},
            "resource_contract": {
                "cpu_agent": {
                    "physical_cpu_cores": 1,
                    "memory_mb": 0,
                    "gpu_count": 0,
                },
                "training_worker": {
                    "physical_cpu_cores": 0,
                    "memory_mb": 0,
                    "gpu_count": 0,
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
                    "cpu_agent": {"estimated_cost_usd": 3},
                    "training_gpu": {"estimated_cost_usd": 0},
                },
            },
        },
    }

    ledger = continuous.build_cost_ledger(timeline)

    assert continuous.cumulative_cost_at_epoch(ledger, 2000) == pytest.approx(2.0)
    assert continuous.cumulative_cost_at_epoch(ledger, 5000) == pytest.approx(3.0)


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
    trajectory_app = (ROOT / "web/trajectory.js").read_text()
    trajectory_overview = (ROOT / "web/trajectory-overview.js").read_text()
    trajectory_page = (ROOT / "web/trajectory.html").read_text()
    trajectory_styles = (ROOT / "web/trajectory.css").read_text()
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
    assert "styles.css?v=20260830-charts34" in page
    assert '<section class="section observations-guide" id="observations">' in page
    assert "We reviewed all 15 trials—five per model." not in page
    assert "No trial finished 100m." in page
    assert "GLM’s fastest simulated 100m took 9.90 s" in page
    assert "one of five trials" in page
    assert ".observation-cards { display: grid; grid-template-columns: 1fr;" in styles
    assert ".observation-cards { grid-template-columns: 1fr; gap: 18px; }" in styles
    assert "app.js?v=20260830-charts24" in page
    assert '"version":"20260830-23"' in (ROOT / "web/version.json").read_text()
    assert "AUC cutoff" not in app
    assert "auc-cap-line" not in app
    assert ".auc-cap-line" not in styles
    assert "minimumFractionDigits:2,maximumFractionDigits:2" in app
    assert "AI AGENTS · ONE HUMANOID · ONE FINISH LINE" not in page
    assert "Can agents train a humanoid to run?" in page
    assert (
        "We give each agent an A10G GPU and a $10 budget to train their fastest runner."
        in page
    )
    assert ".intro-detail" in styles
    assert "font-size: clamp(17px, 2vw, 24px)" in styles
    assert 'class="experiment-table"' in app
    assert 'scope="rowgroup"' in app
    assert (
        '<table class="experiment-table" id="experiment-results-table"><thead><tr><th scope="col">Model</th>'
        '<th scope="col">Effective Speed</th><th scope="col">Elapsed</th>' in app
    )
    assert (
        '<table class="experiment-table" id="experiment-results-table"><thead><tr><th scope="col">Model</th>'
        '<th scope="col">Effort</th>' not in app
    )
    assert 'class="experiment-model-effort"' not in app
    assert 'colspan="5"' in app
    assert app.count('<th scope="col">Trial</th>') == 0
    assert "familyEfforts" not in app
    assert 'displayTrial=index+1' in app
    assert 'aria-label="Open trial ${displayTrial} trace">${displayTrial}</a>' not in app
    assert ">Trial ${esc(row.arm.trial)}</a>" not in app
    assert "let showAllTrials = false" in app
    assert "const rankedRows=rankedExperimentTrials(grouped[key])" in app
    assert "const familyRows=[winner,...rankedRows.slice(1).sort(trialOrder)]" in app
    assert "const visibleRows=showAllTrials?familyRows:[winner]" in app
    assert "isBest=row===winner" in app
    assert "Show all trials" in app
    assert "Show best trials" in app
    assert 'aria-expanded="${showAllTrials}"' in app
    assert "(best of 5)" not in app
    assert ".experiment-table-toggle" in styles
    assert ".experiment-heading-note" not in styles
    assert "DeepSeek-V4-Flash corresponds to DeepSeek V4 Flash Vision Exp." in app
    assert "Cell shading and percentages use the shared $10 trial budget." not in app
    assert "Math.round(100*part.value/budget)" not in app
    assert "document.querySelectorAll('#resource-bars .budget-row')" in app
    assert ".experiment-model-note" in styles
    assert "${isBest?`<b>${speed}</b>`:speed}" in app
    assert "highest Effective Speed for this model across all efforts" in app
    assert ".experiment-row.experiment-best" in styles
    assert ".experiment-row.experiment-best > td:first-of-type" in styles
    assert ".experiment-row:focus-visible > td:first-of-type" in styles
    assert "box-shadow: inset 4px 0 var(--model-accent)" in styles
    assert "box-shadow: inset -4px 0 var(--model-accent)" not in styles
    assert ".experiment-row:hover:not(:has(.experiment-model:hover))" in styles
    assert ".experiment-model.experiment-model-hover" in styles
    assert 'data-family="${esc(key)}"' in app
    assert "experiment-model-hover" in app
    assert "best_continuous_score_mps" in app
    assert "function bestTrialPerformance(models)" in app
    assert "bestPerformanceModels=bestTrialPerformance(performanceModels)" in app
    assert "continuousChart('#cost-chart',bestPerformanceModels" in app
    assert "continuousChart('#time-chart',bestPerformanceModels" in app
    assert 'class="budget-table"' in app
    assert 'scope="rowgroup"' in app
    assert "Cell shading and percentages use the shared $10 trial budget" not in app
    assert '<th scope="col">Total</th>' not in app
    assert 'class="budget-total"' not in app
    assert 'class="budget-col-model"' in app
    assert 'class="budget-col-effort"' not in app
    assert 'class="budget-model-effort"' not in app
    assert '<th scope="col">Effort</th>' not in app.split(
        'class="budget-table"', 1
    )[1]
    assert '<th scope="col">Unspent</th>' not in app
    assert '<th scope="col">CPU agent</th>' not in app
    assert '<th scope="col">Training</th>' not in app
    assert (
        '<th scope="col">Model API</th><th scope="col">GPU</th>'
        '<th scope="col">CPU</th>' in app
    )
    assert "${costCell(parts.api)}${costCell(parts.training)}${costCell(parts.cpu)}" in app
    assert "let showAllBudgetTrials = false" in app
    assert "visibleRows=showAllBudgetTrials?familyRows:[winner]" in app
    assert 'aria-controls="budget-results-table"' in app
    assert 'aria-expanded="${showAllBudgetTrials}"' in app
    assert "showAllBudgetTrials=!showAllBudgetTrials" in app
    assert ".budget-table-actions" in styles
    assert "table-layout: fixed" in styles
    assert ".budget-col-model" in styles
    assert ".budget-col-model {\n  width: 25%;" in styles
    assert ".budget-col-cost,\n.budget-col-price {\n  width: 15%;" in styles
    assert ".budget-col-effort" not in styles
    assert ".budget-group,\n.budget-row" not in styles
    assert "cost-stack" not in app
    assert ".budget-heat" in styles
    assert (
        "background: color-mix(in srgb, var(--text) var(--heat), var(--panel))"
        in styles
    )
    assert ".cost-trial-row" not in styles
    preview = trajectory_app.split("function stepPreview", 1)[1].split(
        "function matchesFilter", 1
    )[0]
    assert preview.index("if(step.reasoning_content)") < preview.index(
        "const calls=step.tool_calls"
    )
    assert "activity tool terminal" in trajectory_app
    assert "meta.append(el('b','',`#" in trajectory_app
    assert "fmtClock(step.timestamp)" not in trajectory_app
    assert 'class="right-rail"' not in trajectory_page
    assert "trajectory.css?v=20260830-8" in trajectory_page
    assert "trajectory.js?v=20260830-2" in trajectory_page
    assert "trajectory-overview.js?v=20260830-14" in trajectory_page
    assert "Number(policy.effective_speed_mps)" in trajectory_overview
    assert "100 / finish" not in trajectory_overview
    assert "function policyFinished(policy)" in trajectory_overview
    assert "drawDiamond(policy.x, policy.y, policyFinished(policy), color)" in trajectory_overview
    assert "gpu_output_observed" in (ROOT / "event_runtime/export/frontier.py").read_text()
    assert 'id="rollout-outline"' in trajectory_page
    assert '<h2 id="rollout-outline-title">Trial Outline</h2>' in trajectory_page
    assert 'aria-label="Trial outline chapters"' in trajectory_page
    assert ">Rollout outline</h2>" not in trajectory_page
    assert trajectory_page.index(
        'class="utilization-overview"'
    ) < trajectory_page.index('id="rollout-outline"')
    assert trajectory_page.index('id="rollout-outline"') < trajectory_page.index(
        'class="trace-column"'
    )
    assert 'class="utilization-footer"' not in trajectory_page
    assert 'id="summary-tools"' not in trajectory_page
    assert "grid-template-columns: repeat(3, 1fr)" in trajectory_styles
    assert 'class="trajectory-controls"' not in trajectory_page
    assert 'id="search"' not in trajectory_page
    assert 'id="jump-step"' not in trajectory_page
    assert ".trajectory-controls" not in trajectory_styles
    assert ".jump-control" not in trajectory_styles
    assert "gap: 12px" in trajectory_styles
    assert "flex: 0 0 auto" in trajectory_styles
    assert "function authoredChapters" in trajectory_app
    assert "Started at ${started} · Lasted for ${" in trajectory_app
    assert (
        "function chapterTarget(chapter){return document.getElementById(chapter.id)}"
        in trajectory_app
    )
    assert "function traceScrollRoot()" in trajectory_app
    assert "scrollTraceTarget(target,'center')" in trajectory_app
    assert "stepsTarget.addEventListener('scroll',scheduleChapterSync" in trajectory_app
    assert "state.outlineLockUntil=performance.now()+500" in trajectory_app
    assert "setActiveChapter(chapter,false)" in trajectory_app
    assert "step.classList.add('jump-flash')" in trajectory_app
    assert "function jumpToStepInput(input)" not in trajectory_app
    assert "$('#jump-step')" not in trajectory_app
    assert "$('#search')" not in trajectory_app
    assert "function hasPrimaryContent(group)" in trajectory_app
    assert (
        "const pulse=document.querySelector('.utilization-overview')" in trajectory_app
    )
    assert (
        "current.offsetTop-chapterNav.offsetTop-(chapterNav.clientHeight-current.offsetHeight)/2"
        in trajectory_app
    )
    assert "rootRect?rootRect.top+rootRect.height/2" in trajectory_app
    assert (
        "Math.abs(target.getBoundingClientRect().top+target.offsetHeight/2-anchor)"
        in trajectory_app
    )
    assert "current.scrollIntoView({block:'nearest'})" not in trajectory_app
    assert "el('span','chapter-divider-number',number)" in trajectory_app
    assert "function scrollTraceTarget(target, block)" in trajectory_overview
    assert "scrollTraceTarget(exactStep ? target : scrollTargetForStep(target)" in trajectory_overview
    assert "jumpToEpoch(epochFor(event.clientX - rect.left), true)" in trajectory_overview
    select_policy_source = trajectory_overview.split("function selectPolicy(policy,", 1)[1].split(
        "function policyMarker(policy, context,", 1
    )[0]
    navigate_policy_source = trajectory_overview.split("function navigateToPolicy(policy,", 1)[1].split(
        "function removePolicySelection(policy,", 1
    )[0]
    assert "replayPanel.scrollIntoView" not in select_policy_source
    assert "requestAnimationFrame(() => restoreOutlineScroll" in navigate_policy_source
    assert "duration: 2500, scrollTop: outlineScroll" in navigate_policy_source
    assert "outlineLockScrollTop" in trajectory_app
    assert "if (policy) {\n      selectPolicy(policy);\n      return;\n    }" in trajectory_overview
    assert "selectPolicy(nearest)" in trajectory_overview
    assert "stepsTarget?.addEventListener('scroll', scheduleScrollSync" in trajectory_overview
    assert "alignExactStepBelowReplay" not in trajectory_overview
    assert "position: sticky;" in trajectory_styles
    assert "height: calc(100vh - 102px);" in trajectory_styles
    assert "overscroll-behavior: contain;" in trajectory_styles
    assert 'id="mobile-outline-toggle"' in trajectory_page
    assert 'aria-controls="rollout-outline"' in trajectory_page
    assert 'id="mobile-outline-backdrop"' in trajectory_page
    assert 'id="mobile-replay-toggle"' not in trajectory_page
    assert "function openOutlineDrawer()" in trajectory_app
    assert "function closeOutlineDrawer(restoreFocus=true)" in trajectory_app
    assert "function trapOutlineFocus(event)" in trajectory_app
    assert "document.addEventListener('keydown',trapOutlineFocus)" in trajectory_app
    assert "body.outline-drawer-open .rollout-outline" in trajectory_styles
    assert ".trajectory-policy-replay.mobile-collapsed" not in trajectory_styles
    assert "function toggleMobileReplay()" not in trajectory_overview
    assert ".trajectory-policy-replay > header > button { display: none; }" in trajectory_styles
    assert "async function resolvePolicyReplay(policy)" in trajectory_overview
    assert "method: 'HEAD'" in trajectory_overview
    assert "await Promise.all((policyResult.data?.policies || []).map(resolvePolicyReplay))" in trajectory_overview
    assert "new CustomEvent('trajectory:policy-selected'," in trajectory_overview
    assert "window.addEventListener('trajectory:policy-selected'" in trajectory_app
    assert "function policyChapterId(policy)" in trajectory_overview
    assert "label.textContent = String(policyNumber(policy))" in trajectory_overview
    assert "label.textContent = 'Submissions'" not in trajectory_overview
    assert "Policies queued during this chapter" in trajectory_overview
    assert "trajectory:policy-navigation-cleared" in trajectory_overview
    assert "const MAX_SELECTED_POLICIES = 8" in trajectory_overview
    assert "selectedPolicies: []" in trajectory_overview
    assert "function removePolicySelection(policy," in trajectory_overview
    assert "state.selectedPolicies.push(policy)" in trajectory_overview
    assert "state.selectedPolicies.splice(index, 1)" in trajectory_overview
    assert "state.selectedPolicies.at(-1)" in trajectory_overview
    assert "marker.setAttribute('aria-pressed', 'false')" in trajectory_overview
    assert "marker.setAttribute('aria-pressed', String(selected))" in trajectory_overview
    assert "You can compare up to ${MAX_SELECTED_POLICIES} policies" in trajectory_overview
    assert "type: 'g1:set-policies'" in trajectory_overview
    assert "COMPARISON_REPLAY_PATH = '/replay/trial-comparison'" in trajectory_overview
    assert "event.data?.type === 'g1:policies-ready'" in trajectory_overview
    assert "return Number(policy.submission_index)" in trajectory_overview
    assert "display_submission_index" not in trajectory_overview
    assert 'id="mobile-policy-selection"' not in trajectory_page
    assert 'id="mobile-replay-label"' in trajectory_page
    assert 'id="replay-policy-selection"' not in trajectory_page
    assert 'id="trajectory-policy-replay-meta"' not in trajectory_page
    assert 'id="trajectory-policy-replay-title"' in trajectory_page
    assert 'id="trajectory-policy-replay-close"' in trajectory_page
    assert 'id="mobile-replay-close"' in trajectory_page
    assert "policy-selection-chip" in trajectory_styles
    assert "label.textContent = String(policyNumber(policy))" in trajectory_overview
    assert "label.textContent = `Policy #${policyNumber(policy)}`" not in trajectory_overview
    assert ".policy-reference.is-selected" in trajectory_styles
    assert ".policy-reference.is-emphasized" in trajectory_styles
    assert "if(state.policyChapterId)" in trajectory_app
    assert "setActiveChapter(policyChapter,false)" in trajectory_app
    assert "stepsTarget.addEventListener('click'" in trajectory_app
    assert "button.closest('.chapter-item')?.classList.toggle('active',selected)" in trajectory_app
    assert ".chapter-item.active { background: var(--accent-soft); }" in trajectory_styles
    assert ".chapter-item.active .chapter-policy-label" in trajectory_styles
    assert ".step-markdown p code, .step-markdown li code" in trajectory_styles
    assert "overflow-wrap: anywhere; word-break: break-word;" in trajectory_styles
    assert "max-width: min(100%, calc(97.78vh - 85.33px));" in trajectory_styles
    assert (
        "classList.contains('chapter-divider') ? divider : target"
        in trajectory_overview
    )
    assert "const anchor = root ? root.getBoundingClientRect().top + 20 : traceAnchor();" in trajectory_overview
    assert "{key: 'training', label: 'GPU'" in trajectory_overview
    assert "TRAIN GPU" not in trajectory_overview
    assert "const laneTop = state.docked ? 15 : 18" in trajectory_overview
    assert "ctx.fillText('SUBMISSIONS'" in trajectory_overview
    assert "drawPolicies(policyY, laneHeight, accent)" in trajectory_overview
    assert 'id="trajectory-policy-replay"' in trajectory_page
    assert "tip.style.top" not in trajectory_overview
    assert "trajectory-outline/v1" in trajectory_app
    assert "generator.model!=='gpt-5.6-sol'" in trajectory_app
    assert "outlinePath=`/data/trajectories/" in trajectory_app
    assert "Promise.all([fetch(path" in trajectory_app
    assert "function deriveChapters" not in trajectory_app
    assert "function chapterTopics" not in trajectory_app
    assert "phaseSignals" not in trajectory_app
    assert "let observedVersion = null" in app
    assert "state.timelineUpdatedAt=tIndex.updated_at||null" in app
    assert "Date.parse(snapshotUpdatedAt(batch)||'')" in app
    assert "refreshVersion" in app
    assert "if(observedVersion===null){observedVersion=deployed.version;return}" in app
    assert "setInterval(refresh,30000)" in app
    assert "setInterval(updateExperimentClocks,1000)" in app
    assert "setInterval(renderExperimentTracker,1000)" not in app
    assert "data-experiment-elapsed" in app
    assert "visibilitychange" in app
    assert "window.addEventListener('focus'" in app
    assert "window.addEventListener('pageshow'" in app
    assert "This policy has no archived website replay." in app
    assert 'id="cost-scores"' not in page
    assert 'id="time-scores"' in page
    assert "auc-bar-row" in app
    assert "color:'#7C54CD'" in app
    assert "color:'#66D693'" in app
    assert "color:'#2279DC'" in app
    assert "--deep: #7C54CD" in styles
    assert "--luna: #66D693" in styles
    assert "--deepseek: #7c54cd" in trajectory_styles
    assert "--sol: #2279dc" in trajectory_styles
    assert (
        "border-right: 2px solid color-mix(in srgb, var(--model-accent) 72%, var(--line))"
        in styles
    )
    assert "--cost-cpu" not in styles
    assert "--cost-training" not in styles
    assert "--deepseek:#7C54CD" in timeline_page
    assert "--luna:#66D693" in timeline_page
    assert "--sol:#2279DC" in timeline_page
    assert 'id="policy-cost-chart"' in timeline_page
    assert 'id="policy-replay-frame"' in timeline_page
    assert "/data/performance/current.json" in timeline_app
    assert "renderPolicyChart" in timeline_app
    assert "showPolicy(point)" in timeline_app
    assert "Open rendered policy" in timeline_app
    assert "setModelAccent" in timeline_app
    assert "<title>Agents' 100m</title>" in page
    assert "<h1>Agents' <em>100m</em></h1>" in page
    assert '<span class="brand">Agents\' <em>100m</em></span>' in page
    assert "color: var(--bg)" in styles
    assert "background: var(--text)" in styles
    assert 'id="live"' not in page
    assert "$('#live')" not in app
    assert "Can agents train a humanoid to run?" in page
    assert "Agents' 100m · Race control" in timeline_page
    assert page.index("Performance vs cost") < page.index("Performance over time")
    assert "The Time-Adjusted Effective Speed compares competitors" in page
    assert 'id="time-performance" hidden' in page
    assert 'id="readout-detail"' in page
    assert 'id="readout-replay"' in page
    assert '<span>Cost so far</span><b>${queueCost}</b>' in app
    readout_source = app.split("function showReadout(point,model){", 1)[1].split("function trialNumber(", 1)[0]
    assert "Stop reason" not in readout_source
    assert "Cost so far" in readout_source
    assert "cost_at_queue_usd" in readout_source
    assert "Trial cost" not in readout_source
    assert "Elapsed race time" not in readout_source
    assert "<span>Effective Speed</span>" in readout_source
    assert "<span>Legal distance</span>" in readout_source
    assert 'id="readout-timeline"' not in page
    assert "RUN TIMELINE" not in page
    assert "What the agent was doing" not in page
    assert "readout-timeline" not in app
    assert ".readout-timeline" not in styles
    assert 'id="policy-grid"' not in page
    assert 'id="run-links"' not in page
    assert "<strong>Effective Speed</strong>" in app
    assert "highest Effective Speed" in app
    assert "winners[key].bestScore" in app
    assert "rankedExperimentTrials(groups[key])[0]" in app
    assert "cost_auc_mps_at_common_cap" in app
    assert "path=`M${x(0)},${y(0)} `" in app
    assert "(run.points||[]).filter(plottable)" in app
    assert "const xmax=finite(aucCap)?aucCap" in app
    assert "Math.min(xmax,value)" in app
    assert "eligible=rows" in app
    assert "cost at queue ${queueCost}" in app
    assert "Effective speed (m/s)" in app
    assert "showReadout" in app
    assert "active_provisional" in app
    assert "'cumulative_agent_cost_usd','Cost at queue ($)'" in app
    assert "best of ${trials}" not in app
    assert "hours_since_agent_launch" in app
    assert "hours_since_agent launch" not in app
    assert "best of 3 trials" not in app


def test_homepage_chart_labels_match_table_text_and_remain_responsive() -> None:
    page = (ROOT / "web/index.html").read_text()
    app = (ROOT / "web/app.js").read_text()
    styles = (ROOT / "web/styles.css").read_text()

    assert "RACE ECONOMICS" not in page
    assert "Every dot is a submitted policy placed" not in page
    assert "The line follows the best sealed policy" not in page
    assert "<h2>Performance vs cost</h2>" in page
    assert '<section class="section" id="cost-performance">' in page
    assert '<section class="section" id="cost-breakdown">' in page
    assert '<section class="section scoring-guide" id="scoring-guide">' in page
    assert "<strong>(d / 100 m) × (d / t)</strong>" in page
    assert "Scoring is simple for a 100m race: the faster, the better." in page
    assert "average speed multiplied by the fraction of the course completed" in page
    assert ".scoring-copy p + p,\n.setup-copy p + p { margin-top: 14px; }" in styles
    assert "Cost so far" in app
    assert "point.cost_at_queue_usd" in app
    assert "For this calculation, we stop counting distance" in page
    assert "Progress made before that point still counts towards the score." in page
    assert "disqualify the run" not in page
    assert "a timeout is not itself a disqualification" not in page
    assert "the 60-second timeout, or the first lane drift or collision" in page
    assert "These are real rendered runs from the active trials" not in page
    assert "#scoring-guide,\n#setup { padding: 44px 0; }" in styles
    assert "#scoring-guide,\n  #setup,\n  #observations,\n  #inspiration { padding: 32px 0; }" in styles
    assert '<div class="dq-guide" id="disqualification-guide">' in page
    assert '<section class="section dq-guide" id="disqualification-guide">' not in page
    assert '<h2>Disqualification</h2>' not in page
    assert 'title="Luna Policy 20 lane-drift replay"' in page
    assert 'title="Luna Policy 1 collision replay"' in page
    assert 'href="/trajectory?run=s10-vexp-r123-20260828-luna-4&amp;policies=20&amp;focus=20&amp;step=a1-s651">Open trial</a>' in page
    assert 'href="/trajectory?run=s10-vexp-r123-20260828-luna-5&amp;policies=1&amp;focus=1&amp;step=a1-s183">Open trial</a>' in page
    assert "GLM-5.3-Flash · Policy #1" not in page
    assert "GPT-5.6 Luna · Policy #1" not in page
    assert 'src="/replay/frontier-8e2feee19eac?example=1"' in page
    assert 'src="/replay/frontier-bca4f7ab8e3c?example=1"' in page
    assert '<script src="/trajectory-url.js?v=20260830-1"></script>' in page
    assert ".dq-cards { display: grid; grid-template-columns: repeat(2" in styles
    assert ".dq-cards { grid-template-columns: 1fr; }" in styles
    assert "#cost-breakdown { padding: 44px 0; }" in styles
    assert "#cost-breakdown { padding: 32px 0; }" in styles
    assert "TEAM SPEND" not in page
    assert "Where the budget went" not in page
    assert "<h2>Cost breakdown</h2>" in page
    insights = page.split('<div class="cost-insights">', 1)[1].split('</div>', 1)[0]
    assert insights.count('<p>') == 2
    assert "<p>As the cost of flash models continues to drop due to inference engine and model architecture innovations, model API cost is now even lower than the CPU cost of running the agent for DeepSeek and GLM, allowing more GPU experiments within the same budget.</p>" in insights
    assert "DeepSeek used DeepSeek Harness, which reread 74 million cached tokens across its best trial as context grew." in insights
    assert "GLM’s Claude Code compacted near 168K tokens and read only 18 million." in insights
    assert "yet DeepSeek spent nearly 2× as much on tokens in their best trials." in insights
    assert "Clever compactions can outweigh cheaper tokens." in insights
    assert "—" not in insights
    assert "incomplete accounting" not in insights
    assert "Luna" not in insights
    assert ".cost-insights" in styles
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in styles
    assert "grid-template-columns: 1fr;" in styles
    assert "POLICY READOUT" not in page
    assert 'id="readout-subtitle"' not in page
    assert "$('#readout-subtitle')" not in app
    assert "Officially disqualified or unfinished" not in app
    assert "Officially valid finish" not in app
    assert 'id="readout-title"' in page
    assert 'id="readout-close"' in page
    assert 'id="readout-stats"' in page
    assert 'id="readout-replay"' in page
    assert "Compare how each independent trial allocated its $10 agent-side budget." not in page
    assert "'cumulative_agent_cost_usd','Cost at queue ($)'" in app
    assert "cost within each independent trial (API + CPU + training; verifier excluded)" not in app
    assert "--data-label-size: 12px" in styles
    assert "font-size: var(--data-label-size)" in styles
    assert "font: var(--data-label-size)/1.35 var(--mono)" in styles
    assert "font: calc(var(--data-label-size) * var(--chart-font-scale, 1)) var(--mono)" in styles
    assert "const chartScale=el.getScreenCTM()?.a||1" in app
    assert "el.style.setProperty('--chart-font-scale',String(1/chartScale))" in app
    cost_section = page.split("<h2>Performance vs cost</h2>", 1)[1].split("</section>", 1)[0]
    assert "metric-note" not in cost_section
    assert "counted distance ÷ 100m" not in page
    assert "includes model API, CPU agents, and training sandboxes" not in page
    assert ".legend > span" in styles
    assert "overflow-wrap: anywhere" in styles
    assert "width=el.clientWidth||1000,height=el.clientHeight||500" in app
    assert "integerTicks(xmax,compact?3:5)" in app
    assert "for(const value of integerTicks(ymax)){const yy=y(value)" in app
    assert "for(const value of xTickValues){const xx=x(value)" in app
    assert "p={l:58,r:18,t:18,b:50}" in app
    chart_source = app.split("function continuousChart(", 1)[1].split(
        "function renderPerformanceScores(", 1
    )[0]
    assert "yTitle.textContent='Effective speed (m/s)'" in chart_source
    assert "higher is better" not in chart_source
    assert "requestAnimationFrame(renderCharts)" in app


def test_mobile_trial_results_match_three_column_cost_breakdown_cards() -> None:
    styles = (ROOT / "web/styles.css").read_text()
    assert "min-width: 760px" not in styles
    assert "min-width: 700px" not in styles
    assert ".budget-table {\n  width: 100%;\n  min-width: 0;" in styles
    assert "padding: 14px clamp(8px, 1.2vw, 16px)" in styles
    assert ".experiment-table {\n  width: 100%;\n  min-width: 0;\n  table-layout: fixed;" in styles
    assert "padding: 16px clamp(8px, 1.2vw, 18px)" in styles
    assert ".experiment-table thead th:nth-child(5) { width: 22%; }" in styles
    mobile_styles = styles.split("@media (max-width: 720px)", 1)[1].split(
        "@media (max-width: 560px)", 1
    )[0]
    assert ".experiment-row,\n  .budget-row {\n    grid-template-columns: repeat(3, minmax(0, 1fr));" in mobile_styles
    assert '.experiment-row td[data-label="Total cost"] { display: none; }' in mobile_styles
    assert ".experiment-model,\n  .budget-model {\n    display: flex;\n    grid-column: 1 / -1;" in mobile_styles
    assert ".experiment-row td,\n  .budget-row td {" in mobile_styles
    assert ".experiment-row td::before,\n  .budget-row td::before" in mobile_styles
    assert ".experiment-row td:not(:last-child),\n  .budget-row td:not(:last-child)" in mobile_styles
    assert 'content: attr(data-trial-label)' not in mobile_styles
    assert '.experiment-row:not(:first-child) td::before,' in mobile_styles
    assert '.budget-row:not(:first-child) td::before' in mobile_styles
    app = (ROOT / "web/app.js").read_text()
    assert 'scope="rowgroup" rowspan="${visibleRows.length}"' in app
    assert "const costLabels={api:'Model API',training:'GPU',cpu:'CPU'}" in app
    assert 'data-label="${esc(costLabels[part.key])}"' in app
    assert 'title="${esc(part.label)}: ${part.value.toFixed(2)} USD · ${esc(part.basis)}"' in app


def test_best_trial_label_is_outside_stable_table_headers() -> None:
    app = (ROOT / "web/app.js").read_text()
    styles = (ROOT / "web/styles.css").read_text()
    assert "`Best of ${trialCounts[0]===5?'Five':trialCounts[0]}`" in app
    assert '<div class="experiment-table-actions"><span class="experiment-best-label" aria-hidden="${showAllTrials}">' in app
    assert '<td data-label="Effective Speed">' in app
    assert "(best of ${grouped[key].length})&#10;" not in app
    header = app.split('<table class="experiment-table"', 1)[1].split("</thead>", 1)[0]
    assert "bestTrialLabel" not in header
    assert "showAllTrials" not in header
    assert ".experiment-table thead th,\n.experiment-best-label {" in styles
    label_rule = styles.split(".experiment-best-label {", 1)[1].split("}", 1)[0]
    assert "font-family: var(--mono)" in label_rule
    assert '.experiment-best-label[aria-hidden="true"] { visibility: hidden; }' in styles


def test_performance_divider_bleeds_without_widening_content() -> None:
    styles = (ROOT / "web/styles.css").read_text()
    divider = styles.split("#cost-performance::after {", 1)[1].split("}", 1)[0]
    assert "height: 1px" in divider
    assert "box-shadow: 0 0 0 100vmax var(--line)" in divider
    assert "clip-path: inset(0 -100vmax)" in divider
    assert "width: 100vw" not in divider
    assert "pointer-events: none" in divider


def test_homepage_updated_footer_displays_date_without_time() -> None:
    app = (ROOT / "web/app.js").read_text()
    render_source = app.split("function render(){", 1)[1].split("let chartResizeFrame", 1)[0]
    assert "Updated ${new Date(updated).toLocaleDateString()}" in render_source
    assert "new Date(updated).toLocaleString()" not in render_source


def test_homepage_chart_ticks_are_integer_values_not_rounded_fractional_positions() -> None:
    app = (ROOT / "web/app.js").read_text()
    helper = "function integerTicks(" + app.split("function integerTicks(", 1)[1].split(
        "function continuousChart(", 1
    )[0]
    completed = subprocess.run(
        [
            "node",
            "-e",
            helper
            + "console.log(JSON.stringify([integerTicks(10.2), integerTicks(10,5),"
            + " integerTicks(10,3), integerTicks(1), integerTicks(100)]))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == [
        [0, 2, 4, 6, 8, 10],
        [0, 2, 4, 6, 8, 10],
        [0, 5, 10],
        [0, 1],
        [0, 20, 40, 60, 80, 100],
    ]


def test_trajectory_comparison_loading_is_neutral_and_preserves_replay_space() -> None:
    trajectory_page = (ROOT / "web/trajectory.html").read_text()
    trajectory_styles = (ROOT / "web/trajectory.css").read_text()
    trajectory_overview = (ROOT / "web/trajectory-overview.js").read_text()
    replay_shell = trajectory_page.split('id="trajectory-policy-replay-shell"', 1)[1].split(
        "</section>", 1
    )[0]

    # The selected lineup can contain 1–8 policies, so a fixed race poster is
    # misleading even when it is only visible while the iframe initializes.
    assert "replay-placeholder" not in trajectory_page
    assert "<img" not in replay_shell
    assert 'class="trajectory-policy-replay-loading" role="status"' in replay_shell
    assert "Loading selected policies…" in replay_shell
    assert 'id="trajectory-policy-replay-frame"' in replay_shell
    assert "aspect-ratio: 16 / 9" in trajectory_styles
    assert ".trajectory-policy-replay-shell > img" not in trajectory_styles
    assert ".trajectory-policy-replay-loading { position: absolute; inset: 0; display: none;" in trajectory_styles
    assert ".trajectory-policy-replay.is-loading .trajectory-policy-replay-loading { display: grid; }" in trajectory_styles
    assert ".trajectory-policy-replay.is-loading .trajectory-policy-replay-shell > iframe { opacity: 0; pointer-events: none; }" in trajectory_styles
    assert "replayPanel.classList.add('is-loading')" in trajectory_overview
    assert "replayPanel.classList.remove('is-loading')" in trajectory_overview


def test_trajectory_exec_parser_preserves_escaped_shell_quotes() -> None:
    trajectory_app = (ROOT / "web/trajectory.js").read_text()
    helpers = []
    for name in ("decodeJsEscapes", "jsStringProperty", "parseExecCalls"):
        prefix = f"  function {name}"
        helpers.append(
            next(
                line.strip()
                for line in trajectory_app.splitlines()
                if line.startswith(prefix)
            )
        )

    command = r"""date -u +%H:%M:%S; event gpu status fcc0bce2c548 | rg '\"provider_live_logs_checked_at\"|\"status\"' | head -n 8; event cost | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get(\"total_usd\"))'"""
    program = (
        "text((await tools.exec_command({"
        f'cmd:{json.dumps(command)},workdir:"/app",yield_time_ms:10000'
        "})).output);"
        'text(await tools.write_stdin({session_id:7,chars:""}));'
    )
    script = "\n".join(
        [
            *helpers,
            f"const program={json.dumps(program)};",
            "const calls=parseExecCalls(program);",
            "console.log(JSON.stringify({names:calls.map(call=>call.name),command:jsStringProperty(calls[0].body,'cmd')}));",
        ]
    )
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(completed.stdout)
    assert parsed == {"names": ["exec_command", "write_stdin"], "command": command}


def test_trajectory_exec_parser_hides_orchestration_separators() -> None:
    trajectory_app = (ROOT / "web/trajectory.js").read_text()
    helper = next(
        line.strip()
        for line in trajectory_app.splitlines()
        if line.startswith("  function isExecDivider")
    )
    script = "\n".join(
        [
            helper,
            "console.log(JSON.stringify([",
            "  isExecDivider('\\n---RESULT---\\n'),",
            "  isExecDivider('\\\\n---RESULT---\\\\n'),",
            "  isExecDivider('--- exit=0 ---'),",
            "  isExecDivider('real command output')",
            "]));",
        ]
    )
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == [True, True, True, False]


def test_trajectory_exec_pairs_parallel_outputs_with_command_cards() -> None:
    trajectory_app = (ROOT / "web/trajectory.js").read_text()

    assert "outputs.length===operationCards.length" in trajectory_app
    assert "operationCards[index].append(panel)" in trajectory_app


def test_trajectory_exec_parser_distinguishes_wrapper_status() -> None:
    trajectory_app = (ROOT / "web/trajectory.js").read_text()
    helper = next(
        line.strip()
        for line in trajectory_app.splitlines()
        if line.startswith("  function normalizeExecOutput")
    )
    script = "\n".join(
        [
            helper,
            "console.log(JSON.stringify([",
            "  normalizeExecOutput('Script completed\\nWall time 0.2 seconds\\nOutput:\\n'),",
            "  normalizeExecOutput('Script failed\\nWall time 1.4 seconds\\nOutput:\\n'),",
            "  normalizeExecOutput('Script running with cell ID abc123\\n')",
            "]));",
        ]
    )
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == [
        {"status": "Completed · 0.2s", "content": ""},
        {"status": "Failed · 1.4s", "content": ""},
        {"status": "Running · cell abc123", "content": ""},
    ]


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
