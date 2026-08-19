from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "event_runtime/container/sprint-openrouter-ledger-proxy.py"

spec = importlib.util.spec_from_file_location("sprint_openrouter_ledger_proxy", SCRIPT)
assert spec and spec.loader
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
REQUEST_1 = "a" * 32
REQUEST_2 = "b" * 32
PENDING_REQUEST = "c" * 32


def test_terminal_response_event_exposes_exact_usage_cost() -> None:
    usage, response = proxy.usage_from_event(
        {
            "type": "response.completed",
            "response": {
                "id": "gen-1",
                "model": "deepseek/deepseek-v4-flash-0731",
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                    "cost": 0.000123,
                    "cost_details": {"upstream_inference_cost": 0.000123},
                },
            },
        }
    )

    assert usage is not None
    assert usage["cost"] == 0.000123
    assert response["id"] == "gen-1"


def test_generic_proxy_seals_provider_endpoint_and_quantization() -> None:
    body, payload = proxy.pin_provider_route(
        json.dumps(
            {
                "model": "deepseek/deepseek-v4-flash-0731",
                "input": "hello",
                "provider": {"sort": "price", "allow_fallbacks": True},
            }
        ).encode(),
        provider_endpoint="baidu/fp8",
        quantization="fp8",
    )

    assert json.loads(body) == payload
    assert payload["provider"] == {
        "only": ["baidu/fp8"],
        "order": ["baidu/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "quantizations": ["fp8"],
    }


def test_endpoint_promotion_is_reversed_without_changing_cache_skus() -> None:
    # Exercise the pure parser used by the live fetcher. Cache-read pricing is
    # deliberately irrelevant here: the endpoint promotion is a single factor
    # that OpenRouter applies to every priced SKU.
    parsed = proxy.sys.modules[
        "sprint_openrouter_pricing"
    ].parse_endpoint_discount_snapshot(
        {
            "data": {
                "endpoints": [
                    {
                        "provider_name": "OpenAI",
                        "tag": "openai",
                        "quantization": "unknown",
                        "pricing": {
                            "prompt": "0.0000025",
                            "input_cache_read": "0.00000025",
                            "completion": "0.000015",
                            "discount": 0.5,
                        },
                    }
                ]
            }
        },
        model="openai/gpt-5.6-sol",
        provider_tag="openai",
        captured_at="2026-08-19T00:00:00Z",
    )

    assert parsed["discount_fraction"] == 0.5
    assert parsed["endpoints"][0]["effective_pricing"]["input_cache_read"] == (
        "0.00000025"
    )
    assert proxy.undiscounted_cost_usd(0.25, parsed) == pytest.approx(0.5)


def test_unpinned_route_with_different_discounts_fails_closed() -> None:
    parser = proxy.sys.modules[
        "sprint_openrouter_pricing"
    ].parse_endpoint_discount_snapshot
    with pytest.raises(proxy.OpenRouterPricingError, match="pin one provider"):
        parser(
            {
                "data": {
                    "endpoints": [
                        {"tag": "one", "pricing": {"discount": 0.5}},
                        {"tag": "two", "pricing": {"discount": 0}},
                    ]
                }
            },
            model="vendor/model",
            provider_tag=None,
        )


def test_atomic_ledger_record_is_private_and_complete(tmp_path: Path) -> None:
    path = tmp_path / "requests/request.json"
    proxy.atomic_json(path, {"state": "complete", "cost": 0.5})

    assert json.loads(path.read_text()) == {"state": "complete", "cost": 0.5}
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(path.parent.glob(".*.tmp"))


@pytest.mark.parametrize(
    "upstream",
    [
        "http://openrouter.ai/api/v1",
        "https://example.com/api/v1",
        "https://openrouter.ai/other",
    ],
)
def test_proxy_refuses_any_non_openrouter_upstream(
    tmp_path: Path, upstream: str
) -> None:
    with pytest.raises(ValueError, match="upstream must"):
        proxy.LedgerProxyServer(
            ("127.0.0.1", 0),
            upstream=upstream,
            ledger_root=tmp_path,
            run_id="run-1",
            cpu_attempt=1,
        )


def test_generic_openrouter_budget_gate_blocks_a_second_paid_request(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/run-1"
    state = run_root / "state"
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps({"run_id": "run-1", "agent_cost_budget_usd": 10.0})
    )
    watchdog = run_root / "budget/watchdog.json"
    watchdog.parent.mkdir(parents=True)
    watchdog.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "run-1",
                "status": "within_budget",
                "components": {
                    "cpu_agent": {"cost_usd": 0.04},
                    "training_sandboxes": {"cost_usd": 0.06},
                },
            }
        )
    )
    record = run_root / f"api-usage/requests/{REQUEST_1}.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "ledger_request_id": REQUEST_1,
                "run_id": "run-1",
                "state": "complete",
                "provider_reported_cost_usd": 9.95,
            }
        )
    )
    runtime = tmp_path / "runtime"
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        runtime_dir=runtime,
    )
    try:
        allowed, snapshot = server.budget_snapshot()
        assert allowed is False
        assert snapshot["total_usd"] == pytest.approx(10.05)
        server.write_stop(snapshot)
    finally:
        server.server_close()

    assert (runtime / "sprint-stop").read_text() == "agent_cost_budget_exhausted\n"
    marker = json.loads((run_root / "BUDGET_STOP_REQUESTED.json").read_text())
    assert marker["status"] == "stop_requested"


def test_live_proxy_observes_watchdog_recovered_pending_cost(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/run-1"
    state = run_root / "state"
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps({"run_id": "run-1", "agent_cost_budget_usd": 10.0})
    )
    record = run_root / f"api-usage/requests/{PENDING_REQUEST}.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "ledger_request_id": PENDING_REQUEST,
                "run_id": "run-1",
                "state": "cost_recovery_required",
                "provider_reported_cost_usd": None,
            }
        )
    )
    (record.parent.parent / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run-1",
                "model_api_usd": 1.0,
                "completed_request_count": 10,
                "pending_request_count": 1,
                "in_flight_request_count": 0,
                "cost_recovery_required_count": 1,
                "in_flight_request_ids": [],
                "cost_recovery_required_request_ids": [PENDING_REQUEST],
            }
        )
    )
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=2,
        runtime_dir=tmp_path / "runtime",
    )
    try:
        recovered = json.loads(record.read_text())
        recovered.update(
            {
                "state": "recovered_complete",
                "provider_reported_cost_usd": 0.5,
            }
        )
        record.write_text(json.dumps(recovered))

        allowed, snapshot = server.budget_snapshot()
        assert allowed is True
        assert snapshot["component_totals_usd"]["model_api_usd"] == 1.5
        assert server.completed_request_count == 11
        assert not server.cost_recovery_required_request_ids
    finally:
        server.server_close()


def test_generic_openrouter_budget_gate_fails_closed_on_unknown_prior_charge(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/run-1"
    state = run_root / "state"
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps({"run_id": "run-1", "agent_cost_budget_usd": 10.0})
    )
    record = run_root / f"api-usage/requests/{REQUEST_1}.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "ledger_request_id": REQUEST_1,
                "run_id": "run-1",
                "state": "cost_recovery_required",
                "provider_reported_cost_usd": None,
            }
        )
    )
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        runtime_dir=tmp_path / "runtime",
    )
    try:
        allowed, snapshot = server.budget_snapshot()
    finally:
        server.server_close()

    assert allowed is False
    assert snapshot["status"] == "fail_closed"
    assert snapshot["reason"] == "budget_telemetry_unavailable"


def test_proxy_maintains_constant_size_exact_cost_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/run-1"
    state = run_root / "state"
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps({"run_id": "run-1", "agent_cost_budget_usd": 10.0})
    )
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        runtime_dir=tmp_path / "runtime",
    )
    try:
        server.begin_request(REQUEST_1)
        in_flight = json.loads(server.summary_path.read_text())
        assert in_flight["pending_request_count"] == 1
        assert in_flight["in_flight_request_count"] == 1
        assert server.budget_snapshot()[0] is False

        server.complete_request(REQUEST_1, 0.125)
        complete = json.loads(server.summary_path.read_text())
        assert complete["model_api_usd"] == pytest.approx(0.125)
        assert complete["completed_request_count"] == 1
        assert complete["pending_request_count"] == 0
        assert server.budget_snapshot()[0] is True
        server.begin_request(REQUEST_2)
        with pytest.raises(ValueError, match="invalid provider-reported cost"):
            server.complete_request(REQUEST_2, float("nan"))
    finally:
        server.server_close()


def test_proxy_restart_trusts_completed_rollup_without_scanning_history(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/run-1"
    state = run_root / "state"
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps({"run_id": "run-1", "agent_cost_budget_usd": 10.0})
    )
    requests = run_root / "api-usage/requests"
    requests.mkdir(parents=True)
    (requests / "historical.json").write_text("{not-json}\n")
    (requests.parent / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run-1",
                "model_api_usd": 1.5,
                "completed_request_count": 2_000,
                "pending_request_count": 0,
                "in_flight_request_count": 0,
                "cost_recovery_required_count": 0,
                "in_flight_request_ids": [],
                "cost_recovery_required_request_ids": [],
            }
        )
        + "\n"
    )

    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=2,
    )
    try:
        assert server.api_cost_usd == pytest.approx(1.5)
        assert server.completed_request_count == 2_000
    finally:
        server.server_close()


def test_proxy_restart_reconciles_only_named_pending_request(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/run-1"
    state = run_root / "state"
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps({"run_id": "run-1", "agent_cost_budget_usd": 10.0})
    )
    requests = run_root / "api-usage/requests"
    requests.mkdir(parents=True)
    (requests / "historical.json").write_text("{not-json}\n")
    (requests / f"{PENDING_REQUEST}.json").write_text(
        json.dumps(
            {
                "ledger_request_id": PENDING_REQUEST,
                "run_id": "run-1",
                "state": "complete",
                "provider_reported_cost_usd": 0.25,
            }
        )
        + "\n"
    )
    (requests.parent / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run-1",
                "model_api_usd": 1.0,
                "completed_request_count": 100,
                "pending_request_count": 1,
                "in_flight_request_count": 1,
                "cost_recovery_required_count": 0,
                "in_flight_request_ids": [PENDING_REQUEST],
                "cost_recovery_required_request_ids": [],
            }
        )
        + "\n"
    )

    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=2,
    )
    try:
        assert server.api_cost_usd == pytest.approx(1.25)
        assert server.completed_request_count == 101
        assert not server.in_flight_request_ids
        summary = json.loads(server.summary_path.read_text())
        assert summary["pending_request_count"] == 0
    finally:
        server.server_close()
