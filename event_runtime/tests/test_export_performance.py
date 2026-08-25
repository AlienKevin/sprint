from __future__ import annotations

import importlib.util
import json
import subprocess
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
    trajectory_page = (ROOT / "web/trajectory.html").read_text()
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
    assert "app.js?v=20260825-5" in page
    assert '"version":"20260825-5"' in (ROOT / "web/version.json").read_text()
    assert 'class="experiment-table"' in app
    assert 'scope="rowgroup"' in app
    assert '<th scope="col">Effort</th>' in app
    assert "best_continuous_score_mps" in app
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
    assert "trajectory.js?v=20260825-16" in trajectory_page
    assert 'id="rollout-outline"' in trajectory_page
    assert 'class="utilization-footer"' not in trajectory_page
    assert "function authoredChapters" in trajectory_app
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
    assert 'id="cost-scores"' in page
    assert 'id="time-scores"' in page
    assert "auc-bar-row" in app
    assert "color:'#4D6BFF'" in app
    assert "color:'#66D693'" in app
    assert "color:'#239057'" in app
    assert "--deep: #4D6BFF" in styles
    assert "--luna: #66D693" in styles
    assert "border-right: 2px solid color-mix(in srgb, var(--model-accent) 72%, var(--line))" in styles
    assert "background: var(--cost-cpu)" in styles
    assert "background: var(--cost-training)" in styles
    assert "--deepseek:#4D6BFF" in timeline_page
    assert "--luna:#66D693" in timeline_page
    assert "--sol:#239057" in timeline_page
    assert 'id="policy-cost-chart"' in timeline_page
    assert 'id="policy-replay-frame"' in timeline_page
    assert "/data/performance/current.json" in timeline_app
    assert "renderPolicyChart" in timeline_app
    assert "showPolicy(point)" in timeline_app
    assert "Open rendered policy" in timeline_app
    assert "setModelAccent" in timeline_app
    assert "<title>Agents' 100m</title>" in page
    assert "<h1>Agents' <em>100m</em></h1>" in page
    assert '<span class="brand">Agents\' 100m</span>' in page
    assert "color: var(--bg)" in styles
    assert "background: var(--text)" in styles
    assert 'id="live"' not in page
    assert "$('#live')" not in app
    assert "Which agent runs the fastest?" in page
    assert "Agents' 100m · Race control" in timeline_page
    assert page.index("Performance vs cost") < page.index("Performance over time")
    assert "The Time-Adjusted Effective Speed compares competitors" in page
    assert 'id="time-performance" hidden' in page
    assert 'id="readout-detail"' in page
    assert 'id="readout-replay"' in page
    assert 'id="readout-timeline"' in page
    assert 'id="policy-grid"' not in page
    assert 'id="run-links"' not in page
    assert "<strong>Effective Speed</strong>" in app
    assert "highest Effective Speed" in app
    assert "Math.max(0,...(performance[key]?.points||[])" in app
    assert "cost_auc_mps_at_common_cap" in app
    assert "path=`M${x(0)},${y(0)} `" in app
    assert "(run.points||[]).filter(plottable)" in app
    assert "const xmax=finite(aucCap)?aucCap" in app
    assert "Math.min(xmax,value)" in app
    assert "eligible=rows.filter" in app
    assert "trial cost ${fmtMoney(row.cumulative_agent_cost_usd)}" in app
    assert "effective speed (m/s)" in app
    assert "showReadout" in app
    assert "active_provisional" in app
    assert "cost within each independent trial" in app
    assert "best of ${trials}" not in app
    assert "hours_since_agent_launch" in app
    assert "hours_since_agent launch" not in app
    assert "best of 3 trials" not in app


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
        f"cmd:{json.dumps(command)},workdir:\"/app\",yield_time_ms:10000"
        "})).output);"
        "text(await tools.write_stdin({session_id:7,chars:\"\"}));"
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
