from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "event_runtime/container/sprint-budget-watchdog.py"
PRICING = ROOT / "harbor/src/harbor/agents/installed/codex_cost.py"

spec = importlib.util.spec_from_file_location("sprint_budget_watchdog", SCRIPT)
assert spec and spec.loader
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


def write_run(
    durable: Path,
    run_id: str,
    *,
    budget: float = 10.0,
    model: str = "deepseek/deepseek-v4-flash",
    service_tier: str | None = None,
    reserve: float | None = None,
    api_cost_source: str | None = None,
) -> Path:
    root = durable / "runs" / run_id
    state = root / "state"
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "agent_kind": "codex",
                "model": model,
                "service_tier": service_tier,
                "reasoning_effort": "high",
                "usage_audit_required": True,
                "standing_gpu_worker": False,
                "agent_cost_budget_usd": budget,
                "cpu_launch_attempt": 1,
                "budget_enforcement": {
                    **(
                        {"api_cost_source": api_cost_source}
                        if api_cost_source is not None
                        else {}
                    ),
                    "shutdown_reserve_usd": (reserve if reserve is not None else 0.1),
                    "minimum_safe_shutdown_reserve_usd": 0.1,
                },
            }
        )
    )
    return root


def write_openrouter_cost(
    root: Path, *, cost: float, undiscounted_cost: float | None = None
) -> None:
    path = root / "api-usage" / "requests" / "request.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2 if undiscounted_cost is not None else 1,
                "ledger_request_id": "request",
                "run_id": "unit",
                "cpu_attempt": 1,
                "state": "complete",
                "generation_id": "gen-test",
                "provider_reported_cost_usd": cost,
                **(
                    {
                        "undiscounted_cost_usd": undiscounted_cost,
                        "promotion_discount_fraction": 1 - cost / undiscounted_cost,
                    }
                    if undiscounted_cost is not None
                    else {}
                ),
                "usage": {
                    "cost": cost,
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 2},
                    "output_tokens": 3,
                    "output_tokens_details": {"reasoning_tokens": 1},
                    "total_tokens": 13,
                },
            }
        )
    )


def write_codex_request(codex_home: Path, *, model: str = "deepseek-v4-flash") -> None:
    session = codex_home / "sessions/2026/08/18/rollout.jsonl"
    session.parent.mkdir(parents=True)
    usage = {
        "input_tokens": 1_000,
        "cached_input_tokens": 800,
        "cache_write_input_tokens": 0,
        "output_tokens": 100,
        "reasoning_output_tokens": 50,
        "total_tokens": 1_100,
    }
    rows = [
        {
            "type": "turn_context",
            "timestamp": "2026-08-18T00:00:00Z",
            "payload": {"model": model, "effort": "high"},
        },
        {
            "type": "response_item",
            "timestamp": "2026-08-18T00:00:01Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            },
        },
        {
            "type": "event_msg",
            "timestamp": "2026-08-18T00:00:02Z",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": usage,
                    "model_context_window": 1_000_000,
                },
            },
        },
    ]
    session.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_host_cost_mirror(
    runtime: Path,
    *,
    checked_at: float,
    api: float = 0.0,
    cpu: float = 0.02,
    training: float = 9.89,
) -> Path:
    path = runtime / "sprint-gpu-mirror/cost.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "unit",
                "checked_at_epoch_s": checked_at,
                "budget_usd": 10.0,
                "stop_threshold_usd": 9.9,
                "status": "stop_requested",
                "total_usd": api + cpu + training,
                "components": {
                    "model_api": {
                        "cost_usd": api,
                        "request_count": 0,
                        "pending_request_count": 0,
                    },
                    "cpu_agent": {
                        "cost_usd": cpu,
                        "allocated_seconds": 100.0,
                    },
                    "training_sandboxes": {
                        "cost_usd": training,
                        "allocated_seconds": 15_000.0,
                    },
                },
            }
        )
        + "\n"
    )
    return path


def test_live_watchdog_prices_api_cpu_and_gpu(tmp_path: Path, monkeypatch) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    codex_home = tmp_path / "codex"
    root = write_run(durable, "unit")
    write_codex_request(codex_home)
    monkeypatch.setenv("SPRINT_CPU_LAUNCH_ATTEMPT", "1")
    watchdog.ensure_cpu_start(root, 1, 1_000)
    events = root / "telemetry/gpu_timeline/events"
    events.mkdir(parents=True)
    (events / "one.json").write_text(
        json.dumps(
            {
                "event_id": "one",
                "epoch_s": 1_010,
                "lease_id": "lease-1",
                "phase": "gpu_lifecycle",
                "action": "instant",
                "detail": {"event": "gpu_allocated"},
            }
        )
    )
    (events / "two.json").write_text(
        json.dumps(
            {
                "event_id": "two",
                "epoch_s": 1_020,
                "lease_id": "lease-1",
                "phase": "gpu_lifecycle",
                "action": "instant",
                "detail": {"event": "gpu_released"},
            }
        )
    )

    payload = watchdog.check_once(
        run_id="unit",
        durable_dir=durable,
        runtime_dir=runtime,
        codex_home=codex_home,
        pricing_path=PRICING,
        now=1_100,
    )

    assert payload["status"] == "within_budget"
    assert payload["request_count"] == 1
    assert payload["cpu_allocated_seconds"] == 100
    assert payload["training_allocated_seconds"] == 10
    assert payload["components"]["model_api"]["cost_usd"] > 0
    assert not (root / "BUDGET_STOP_REQUESTED.json").exists()


def test_gpu_cost_ignores_late_allocation_after_same_lease_terminal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    events = root / "telemetry/gpu_timeline/events"
    events.mkdir(parents=True)
    rows = [
        {
            "event_id": "terminal",
            "epoch_s": 1_020,
            "lease_id": "lease-race",
            "phase": "gpu_lifecycle",
            "action": "instant",
            "detail": {"event": "gpu_released"},
        },
        {
            "event_id": "late-allocation",
            "epoch_s": 1_023,
            "lease_id": "lease-race",
            "phase": "gpu_lifecycle",
            "action": "instant",
            "detail": {"event": "gpu_reallocated"},
        },
    ]
    for row in rows:
        (events / f"{row['event_id']}.json").write_text(json.dumps(row))

    assert watchdog.gpu_allocated_seconds(
        root, 2_000, standing=False, cpu_seconds=0
    ) == 0


def test_gpu_cost_closes_missing_release_from_terminal_attempt_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    events = root / "telemetry/gpu_timeline/events"
    events.mkdir(parents=True)
    (events / "allocated.json").write_text(
        json.dumps(
            {
                "event_id": "allocated",
                "epoch_s": 1_000,
                "lease_id": "lease-stopped",
                "phase": "gpu_lifecycle",
                "action": "instant",
                "detail": {"event": "gpu_allocated"},
            }
        )
    )
    attempt = root / "gpu-jobs/attempts/job-stopped/1.json"
    attempt.parent.mkdir(parents=True)
    attempt.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "job_id": "job-stopped",
                "attempt": 1,
                "lease_id": "lease-stopped",
                "status": "terminated",
                "finished_at_epoch_s": 1_020,
            }
        )
    )

    assert watchdog.gpu_allocated_seconds(
        root, 2_000, standing=False, cpu_seconds=0
    ) == 20


def test_live_watchdog_merges_fresh_host_training_cost_and_stops(
    tmp_path: Path, monkeypatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    codex_home = tmp_path / "codex"
    root = write_run(durable, "unit")
    monkeypatch.setenv("SPRINT_CPU_LAUNCH_ATTEMPT", "1")
    watchdog.ensure_cpu_start(root, 1, 1_000)
    write_host_cost_mirror(runtime, checked_at=1_099)

    payload = watchdog.check_once(
        run_id="unit",
        durable_dir=durable,
        runtime_dir=runtime,
        codex_home=codex_home,
        pricing_path=PRICING,
        now=1_100,
    )

    assert payload["components"]["training_sandboxes"]["cost_usd"] == pytest.approx(
        9.89
    )
    assert payload["total_usd"] >= 9.9
    assert payload["status"] == "stop_requested"
    assert payload["host_cost_mirror_checked_at_epoch_s"] == 1_099
    assert (runtime / "sprint-stop").read_text() == "agent_cost_budget_exhausted\n"


def test_fresh_host_mirror_is_authoritative_over_idle_gap_fallbacks(
    tmp_path: Path, monkeypatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    codex_home = tmp_path / "codex"
    root = write_run(durable, "unit")
    monkeypatch.setenv("SPRINT_CPU_LAUNCH_ATTEMPT", "1")
    watchdog.ensure_cpu_start(root, 1, 1_000)
    events = root / "telemetry/gpu_timeline/events"
    events.mkdir(parents=True)
    (events / "open.json").write_text(
        json.dumps(
            {
                "event_id": "open",
                "epoch_s": 1_000,
                "lease_id": "stale-open-lease",
                "phase": "gpu_lifecycle",
                "detail": {"event": "gpu_allocated"},
            }
        )
        + "\n"
    )
    write_host_cost_mirror(
        runtime,
        checked_at=9_999,
        cpu=0.02,
        training=0.03,
    )

    payload = watchdog.check_once(
        run_id="unit",
        durable_dir=durable,
        runtime_dir=runtime,
        codex_home=codex_home,
        pricing_path=PRICING,
        now=10_000,
    )

    assert payload["components"]["cpu_agent"]["cost_usd"] == pytest.approx(0.02)
    assert payload["components"]["training_sandboxes"]["cost_usd"] == pytest.approx(
        0.03
    )
    assert payload["cpu_allocated_seconds"] == 100
    assert payload["training_allocated_seconds"] == 15_000
    assert payload["component_snapshot_sources"]["cpu_agent"] == "host_cost_mirror"
    assert payload["component_snapshot_sources"]["training_sandboxes"] == (
        "host_cost_mirror"
    )


def test_live_watchdog_fails_closed_when_host_cost_mirror_is_stale(
    tmp_path: Path, monkeypatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    codex_home = tmp_path / "codex"
    root = write_run(durable, "unit")
    monkeypatch.setenv("SPRINT_CPU_LAUNCH_ATTEMPT", "1")
    watchdog.ensure_cpu_start(root, 1, 1_000)
    write_host_cost_mirror(runtime, checked_at=1_000)

    with pytest.raises(watchdog.BudgetTelemetryError, match="host cost mirror is stale"):
        watchdog.check_once(
            run_id="unit",
            durable_dir=durable,
            runtime_dir=runtime,
            codex_home=codex_home,
            pricing_path=PRICING,
            now=1_061,
        )


def test_live_watchdog_prices_pinned_luna_default_tier(
    tmp_path: Path, monkeypatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    codex_home = tmp_path / "codex"
    root = write_run(
        durable,
        "unit",
        model="openai/gpt-5.6-luna",
        service_tier="default",
    )
    write_codex_request(codex_home, model="gpt-5.6-luna")
    monkeypatch.setenv("SPRINT_CPU_LAUNCH_ATTEMPT", "1")
    watchdog.ensure_cpu_start(root, 1, 1_000)

    payload = watchdog.check_once(
        run_id="unit",
        durable_dir=durable,
        runtime_dir=runtime,
        codex_home=codex_home,
        pricing_path=PRICING,
        now=1_100,
    )

    assert payload["status"] == "within_budget"
    assert payload["request_count"] == 1
    assert payload["components"]["model_api"]["cost_usd"] > 0


def test_live_watchdog_uses_openrouter_undiscounted_cost_for_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    root = write_run(
        durable,
        "unit",
        api_cost_source="openrouter_reported_per_request",
    )
    write_openrouter_cost(root, cost=0.75, undiscounted_cost=1.5)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openrouter-key-long-enough")
    watchdog.ensure_cpu_start(root, 1, 1_000)

    payload = watchdog.check_once(
        run_id="unit",
        durable_dir=durable,
        runtime_dir=runtime,
        codex_home=tmp_path / "codex",
        pricing_path=PRICING,
        now=1_000,
    )

    assert payload["components"]["model_api"] == {
        "cost_usd": 1.5,
        "provider_billed_cost_usd": 0.75,
        "promotion_savings_usd": 0.75,
        "request_count": 1,
        "pending_request_count": 0,
        "cost_source": "openrouter_list_price_with_deepseek_peak_floor",
        "provider_reported": True,
    }
    assert payload["total_usd"] == pytest.approx(1.5)
    assert payload["budget_remaining_usd"] == pytest.approx(8.5)


def test_openrouter_watchdog_bootstraps_without_agent_scoped_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    root = write_run(
        durable,
        "unit",
        api_cost_source="openrouter_reported_per_request",
    )
    write_openrouter_cost(root, cost=0.75)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    watchdog.ensure_cpu_start(root, 1, 1_000)

    payload = watchdog.check_once(
        run_id="unit",
        durable_dir=durable,
        runtime_dir=runtime,
        codex_home=tmp_path / "codex",
        pricing_path=PRICING,
        now=1_000,
    )

    assert payload["status"] == "within_budget"
    assert payload["components"]["model_api"]["cost_usd"] == 0.75


def test_openrouter_watchdog_fails_closed_on_unrecoverable_charge_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    root = write_run(
        durable,
        "unit",
        api_cost_source="openrouter_reported_per_request",
    )
    record = root / "api-usage/requests/request.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "run_id": "unit",
                "state": "cost_recovery_required",
                "generation_id": "gen-recover",
                "provider_reported_cost_usd": None,
            }
        )
    )
    process = runtime / "sprint-agent/codex-process"
    process.parent.mkdir(parents=True)
    process.write_text(f"{os.getpid()} {os.getpid()} 1\n")
    (process.parent / "openrouter-proxy.pid").write_text(f"{os.getpid()}\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(
        watchdog.BudgetTelemetryError,
        match="requires controller credentials",
    ):
        watchdog.check_once(
            run_id="unit",
            durable_dir=durable,
            runtime_dir=runtime,
            codex_home=tmp_path / "codex",
            pricing_path=PRICING,
            now=1_000,
        )


def test_openrouter_watchdog_allows_only_bounded_pre_agent_proxy_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    root = write_run(
        durable,
        "unit",
        api_cost_source="openrouter_reported_per_request",
    )
    request_id = "a" * 32
    record = root / f"api-usage/requests/{request_id}.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "ledger_request_id": request_id,
                "run_id": "unit",
                "state": "cost_recovery_required",
                "generation_id": "gen-recover",
                "provider_reported_cost_usd": None,
            }
        )
    )
    proxy_pid = runtime / "sprint-agent/openrouter-proxy.pid"
    proxy_pid.parent.mkdir(parents=True)
    proxy_pid.write_text(f"{os.getpid()}\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SPRINT_CPU_LAUNCH_ATTEMPT", "1")
    watchdog.ensure_cpu_start(root, 1, 1_000)

    payload = watchdog.check_once(
        run_id="unit",
        durable_dir=durable,
        runtime_dir=runtime,
        codex_home=tmp_path / "codex",
        pricing_path=PRICING,
        now=1_001,
    )

    assert payload["status"] == "within_budget"
    assert payload["pending_request_count"] == 1
    assert payload["components"]["model_api"]["telemetry_state"] == (
        "awaiting_proxy_recovery"
    )
    assert not (root / "BUDGET_STOP_REQUESTED.json").exists()

    with pytest.raises(
        watchdog.BudgetTelemetryError,
        match="requires controller credentials",
    ):
        watchdog.check_once(
            run_id="unit",
            durable_dir=durable,
            runtime_dir=runtime,
            codex_home=tmp_path / "codex",
            pricing_path=PRICING,
            now=1_301,
        )


def test_live_watchdog_fails_closed_if_codex_outlives_cost_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    root = write_run(
        durable,
        "unit",
        api_cost_source="openrouter_reported_per_request",
    )
    process = runtime / "sprint-agent/codex-process"
    process.parent.mkdir(parents=True)
    process.write_text("123 123 1\n")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openrouter-key-long-enough")
    watchdog.ensure_cpu_start(root, 1, 1_000)

    with pytest.raises(
        watchdog.BudgetTelemetryError,
        match="without its OpenRouter cost ledger proxy",
    ):
        watchdog.check_once(
            run_id="unit",
            durable_dir=durable,
            runtime_dir=runtime,
            codex_home=tmp_path / "codex",
            pricing_path=PRICING,
            now=1_000,
        )


def test_interrupted_openrouter_stream_recovers_exact_generation_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    path = root / "api-usage/requests/request.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "run_id": "unit",
                "cpu_attempt": 1,
                "state": "cost_recovery_required",
                "generation_id": "gen-recover",
                "provider_reported_cost_usd": None,
            }
        )
    )
    monkeypatch.setattr(
        watchdog,
        "recover_openrouter_generation",
        lambda generation_id, api_key: {
            "id": generation_id,
            "total_cost": 0.456,
        },
    )

    cost, provider_cost, complete, pending = watchdog.openrouter_api_cost(
        root, run_id="unit", api_key="test-key"
    )

    assert (cost, provider_cost, complete, pending) == (0.456, 0.456, 1, 0)
    recovered = json.loads(path.read_text())
    assert recovered["state"] == "recovered_complete"
    assert recovered["provider_reported_cost_usd"] == 0.456


def test_interrupted_discounted_stream_recovers_undiscounted_budget_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    path = root / "api-usage/requests/request.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "unit",
                "cpu_attempt": 1,
                "state": "cost_recovery_required",
                "generation_id": "gen-discounted",
                "provider_reported_cost_usd": None,
                "undiscounted_cost_usd": None,
                "promotion_snapshot": {"discount_fraction": 0.5},
            }
        )
    )
    monkeypatch.setattr(
        watchdog,
        "recover_openrouter_generation",
        lambda generation_id, api_key: {
            "id": generation_id,
            "total_cost": 0.4,
        },
    )

    cost, provider_cost, complete, pending = watchdog.openrouter_api_cost(
        root, run_id="unit", api_key="test-key"
    )

    assert (cost, provider_cost, complete, pending) == (0.8, 0.4, 1, 0)
    recovered = json.loads(path.read_text())
    assert recovered["undiscounted_cost_usd"] == 0.8
    assert recovered["promotion_discount_fraction"] == 0.5


def test_interrupted_deepseek_stream_recovers_peak_normalized_budget_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    path = root / "api-usage/requests/request.json"
    path.parent.mkdir(parents=True)
    peak_rates = {
        "uncached_input": 0.44 / 1_000_000,
        "cached_input": 0.014 / 1_000_000,
        "output": 1.32 / 1_000_000,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": "unit",
                "cpu_attempt": 1,
                "state": "cost_recovery_required",
                "generation_id": "gen-deepseek",
                "provider_reported_cost_usd": None,
                "promotion_snapshot": {
                    "discount_fraction": 0,
                    "deepseek_peak_pricing_usd_per_token": peak_rates,
                    "cost_basis": "openrouter_list_price_with_deepseek_peak_floor",
                },
            }
        )
    )
    monkeypatch.setattr(
        watchdog,
        "recover_openrouter_generation",
        lambda generation_id, api_key: {
            "id": generation_id,
            "total_cost": 0.0001156,
            "native_tokens_prompt": 1_000,
            "native_tokens_cached": 800,
            "native_tokens_completion": 100,
        },
    )

    cost, provider_cost, complete, pending = watchdog.openrouter_api_cost(
        root, run_id="unit", api_key="test-key"
    )

    assert (cost, provider_cost, complete, pending) == pytest.approx(
        (0.0002312, 0.0001156, 1, 0)
    )
    recovered = json.loads(path.read_text())
    assert recovered["benchmark_cost_usd"] == pytest.approx(0.0002312)
    assert recovered["undiscounted_cost_usd"] == pytest.approx(0.0001156)


def test_openrouter_recovery_reads_only_named_pending_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    requests = root / "api-usage/requests"
    requests.mkdir(parents=True)
    request_id = "d" * 32
    path = requests / f"{request_id}.json"
    path.write_text(
        json.dumps(
            {
                "ledger_request_id": request_id,
                "run_id": "unit",
                "state": "cost_recovery_required",
                "generation_id": "gen-recover",
                "provider_reported_cost_usd": None,
            }
        )
    )
    (requests / "historical-malformed.json").write_text("{not-json}\n")
    (requests.parent / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "unit",
                "model_api_usd": 1.25,
                "completed_request_count": 2_000,
                "pending_request_count": 1,
                "in_flight_request_count": 0,
                "cost_recovery_required_count": 1,
                "in_flight_request_ids": [],
                "cost_recovery_required_request_ids": [request_id],
            }
        )
    )
    monkeypatch.setattr(
        watchdog,
        "recover_openrouter_generation",
        lambda generation_id, api_key: {
            "id": generation_id,
            "total_cost": 0.456,
        },
    )

    cost, provider_cost, complete, pending = watchdog.openrouter_api_cost(
        root, run_id="unit", api_key="test-key"
    )
    assert cost == pytest.approx(1.706)
    assert provider_cost == pytest.approx(1.706)
    assert (complete, pending) == (2_001, 0)
    summary = json.loads((requests.parent / "summary.json").read_text())
    assert summary["in_flight_request_ids"] == []
    assert summary["cost_recovery_required_request_ids"] == []


def test_openrouter_watchdog_uses_exact_rollup_without_rescanning_shards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    requests = root / "api-usage/requests"
    requests.mkdir(parents=True)
    # A malformed historical shard proves the steady-state watchdog did not
    # touch the growing audit directory once the proxy published its rollup.
    (requests / "historical.json").write_text("{not-json}\n")
    (requests.parent / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "unit",
                "updated_at": "2026-08-19T00:00:00Z",
                "model_api_usd": 1.25,
                "completed_request_count": 2_000,
                "pending_request_count": 0,
                "in_flight_request_count": 0,
                "cost_recovery_required_count": 0,
            }
        )
        + "\n"
    )

    assert watchdog.openrouter_api_cost(root, run_id="unit", api_key=None) == (
        1.25,
        1.25,
        2_000,
        0,
    )


def test_live_watchdog_stops_before_cap_using_shutdown_reserve(
    tmp_path: Path, monkeypatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    codex_home = tmp_path / "codex"
    root = write_run(durable, "unit")
    monkeypatch.setenv("SPRINT_CPU_LAUNCH_ATTEMPT", "1")
    watchdog.ensure_cpu_start(root, 1, 1_000)
    (root / "BUDGET_STOP_REQUESTED.json").write_text(
        '{"reason":"agent_cost_budget_exhausted"}\n'
    )
    elapsed = 9.95 / watchdog.CPU_USD_PER_SECOND

    payload = watchdog.check_once(
        run_id="unit",
        durable_dir=durable,
        runtime_dir=runtime,
        codex_home=codex_home,
        pricing_path=PRICING,
        now=1_001 + elapsed,
    )

    assert payload["status"] == "stop_requested"
    marker = json.loads((root / "BUDGET_STOP_REQUESTED.json").read_text())
    assert marker["reason"] == "agent_cost_budget_exhausted"
    assert marker["schema_version"] == 2
    assert marker["status"] == "stop_requested"
    assert marker["total_usd"] >= marker["stop_threshold_usd"]
    assert (runtime / "sprint-stop").read_text() == "agent_cost_budget_exhausted\n"


def test_invalid_complete_usage_record_is_fail_closed(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "bad.jsonl").write_text("{not-json}\n")
    pricing = watchdog.load_pricing_module(PRICING)
    with pytest.raises(watchdog.BudgetTelemetryError):
        watchdog.codex_api_cost(
            sessions,
            default_model="deepseek-v4-flash",
            default_service_tier=None,
            default_effort="high",
            pricing_module=pricing,
        )


def test_gpu_cost_ignores_release_for_never_allocated_queued_attempt(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    events = run_root / "telemetry/gpu_timeline/events"
    events.mkdir(parents=True)
    (events / "release.json").write_text(
        json.dumps(
            {
                "event_id": "release",
                "phase": "gpu_lifecycle",
                "attempt": 0,
                "lease_id": None,
                "epoch_s": 1_000,
                "detail": {"event": "gpu_released", "reason": "operator_stop"},
            }
        )
        + "\n"
    )

    assert watchdog.gpu_allocated_seconds(
        run_root, 1_100, standing=False, cpu_seconds=0
    ) == 0


def load_gpu_worker():
    worker_spec = importlib.util.spec_from_file_location(
        "sprint_gpu_worker_budget_snapshot",
        ROOT / "event_runtime/container/sprint-gpu-worker-run.py",
    )
    assert worker_spec and worker_spec.loader
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)
    return worker


def write_budget_snapshot(
    path: Path, *, checked_at: float, total: float = 1.0, threshold: float = 9.9
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "unit",
                "checked_at_epoch_s": checked_at,
                "total_usd": total,
                "stop_threshold_usd": threshold,
                "status": "stop_requested" if total >= threshold else "within_budget",
            }
        )
        + "\n"
    )


def test_gpu_worker_accepts_fresh_budget_snapshot(tmp_path: Path) -> None:
    worker = load_gpu_worker()
    runtime_snapshot = tmp_path / "runtime-budget.json"
    write_budget_snapshot(runtime_snapshot, checked_at=1_000)

    assert worker.refresh_budget_stop(
        "unit", str(tmp_path), now=1_010, runtime_snapshot=runtime_snapshot
    ) is None
    assert not (tmp_path / "runs/unit/BUDGET_STOP_REQUESTED.json").exists()


def test_gpu_worker_ignores_agent_writable_durable_stop_and_snapshot(
    tmp_path: Path,
) -> None:
    worker = load_gpu_worker()
    runtime_snapshot = tmp_path / "runtime-budget.json"
    write_budget_snapshot(runtime_snapshot, checked_at=1_000)
    durable_snapshot = tmp_path / "runs/unit/budget/watchdog.json"
    write_budget_snapshot(durable_snapshot, checked_at=9_999, total=9.9)
    marker = tmp_path / "runs/unit/BUDGET_STOP_REQUESTED.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "unit",
                "reason": "agent_cost_budget_exhausted",
                "status": "stop_requested",
                "stop_threshold_usd": 9.9,
                "total_usd": 9.9,
            }
        )
    )

    assert worker.refresh_budget_stop(
        "unit", str(tmp_path), now=1_010, runtime_snapshot=runtime_snapshot
    ) is None
    # Within-budget verification does not need to mutate the audit artifact.
    assert json.loads(marker.read_text())["total_usd"] == pytest.approx(9.9)


def test_gpu_worker_prefers_fresh_host_mirror_over_stale_volume_snapshot(
    tmp_path: Path,
) -> None:
    worker = load_gpu_worker()
    durable_snapshot = tmp_path / "runs/unit/budget/watchdog.json"
    write_budget_snapshot(durable_snapshot, checked_at=9_999, total=9.9)
    runtime_snapshot = tmp_path / "runtime-budget.json"
    write_budget_snapshot(runtime_snapshot, checked_at=1_190, total=1.5)

    assert not worker.refresh_budget_stop(
        "unit",
        str(tmp_path),
        now=1_200,
        runtime_snapshot=runtime_snapshot,
    )
    assert not (tmp_path / "runs/unit/BUDGET_STOP_REQUESTED.json").exists()


def test_gpu_worker_fails_closed_on_stale_budget_snapshot(tmp_path: Path) -> None:
    worker = load_gpu_worker()
    runtime_snapshot = tmp_path / "runtime-budget.json"
    write_budget_snapshot(runtime_snapshot, checked_at=1_000)

    assert worker.refresh_budget_stop(
        "unit", str(tmp_path), now=1_121, runtime_snapshot=runtime_snapshot
    ) == "budget_telemetry_unavailable"
    marker = json.loads(
        (tmp_path / "runs/unit/BUDGET_STOP_REQUESTED.json").read_text()
    )
    assert marker["reason"] == "budget_telemetry_unavailable"
    assert "stale" in marker["error"]


def test_gpu_worker_propagates_fresh_budget_stop(tmp_path: Path) -> None:
    worker = load_gpu_worker()
    runtime_snapshot = tmp_path / "runtime-budget.json"
    write_budget_snapshot(runtime_snapshot, checked_at=1_000, total=9.9)

    assert worker.refresh_budget_stop(
        "unit", str(tmp_path), now=1_010, runtime_snapshot=runtime_snapshot
    ) == "agent_cost_budget_exhausted"
    marker = json.loads(
        (tmp_path / "runs/unit/BUDGET_STOP_REQUESTED.json").read_text()
    )
    assert marker["reason"] == "agent_cost_budget_exhausted"
    assert marker["total_usd"] == pytest.approx(9.9)
