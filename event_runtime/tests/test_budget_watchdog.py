from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "event_runtime/container/sprint-budget-watchdog.py"
PRICING = ROOT / "harbor/src/harbor/agents/installed/codex_cost.py"

spec = importlib.util.spec_from_file_location("sprint_budget_watchdog", SCRIPT)
assert spec and spec.loader
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


def write_run(durable: Path, run_id: str, *, budget: float = 10.0) -> Path:
    root = durable / "runs" / run_id
    state = root / "state"
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "agent_kind": "codex",
                "model": "deepseek/deepseek-v4-flash",
                "reasoning_effort": "high",
                "service_tier": None,
                "usage_audit_required": True,
                "standing_gpu_worker": False,
                "agent_cost_budget_usd": budget,
                "cpu_launch_attempt": 1,
                "budget_enforcement": {"shutdown_reserve_usd": 0.6},
            }
        )
    )
    return root


def write_deepseek_request(codex_home: Path) -> None:
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
            "payload": {"model": "deepseek-v4-flash", "effort": "high"},
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


def test_live_watchdog_prices_api_cpu_and_gpu(tmp_path: Path, monkeypatch) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    codex_home = tmp_path / "codex"
    root = write_run(durable, "unit")
    write_deepseek_request(codex_home)
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
    assert payload["components"]["model_api_usd"] > 0
    assert not (root / "BUDGET_STOP_REQUESTED.json").exists()


def test_live_watchdog_stops_before_cap_using_shutdown_reserve(
    tmp_path: Path, monkeypatch
) -> None:
    durable = tmp_path / "durable"
    runtime = tmp_path / "run"
    codex_home = tmp_path / "codex"
    root = write_run(durable, "unit")
    monkeypatch.setenv("SPRINT_CPU_LAUNCH_ATTEMPT", "1")
    watchdog.ensure_cpu_start(root, 1, 1_000)
    elapsed = 9.4 / watchdog.CPU_USD_PER_SECOND

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


def test_gpu_worker_observes_durable_budget_marker(tmp_path: Path) -> None:
    marker = tmp_path / "runs/unit/BUDGET_STOP_REQUESTED.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n")
    worker_spec = importlib.util.spec_from_file_location(
        "sprint_gpu_worker_budget",
        ROOT / "event_runtime/container/sprint-gpu-worker-run.py",
    )
    assert worker_spec and worker_spec.loader
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)
    assert worker.budget_stop_requested("unit", str(tmp_path))
