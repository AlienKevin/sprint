from __future__ import annotations

import importlib.util
import http.client
import json
from pathlib import Path
import threading

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
                "parallel_tool_calls": False,
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
    assert "parallel_tool_calls" not in payload


def test_chat_completions_contract_is_sealed_and_usage_is_forced() -> None:
    body, payload = proxy.pin_provider_route(
        json.dumps(
            {
                "model": "caller/model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "temperature": 0,
                "top_p": 0.1,
                "max_tokens": 5,
                "reasoning_effort": "low",
                "stream_options": {"include_usage": False},
            }
        ).encode(),
        provider_endpoint="deepseek",
        quantization=None,
        request_contract={
            "model": "deepseek/deepseek-v4-flash-vision-exp",
            "stream": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 384_000,
            "reasoning_effort": "max",
        },
    )

    assert json.loads(body) == payload
    assert payload["model"] == "deepseek/deepseek-v4-flash-vision-exp"
    assert payload["provider"]["only"] == ["deepseek"]
    assert payload["provider"]["allow_fallbacks"] is False
    assert payload["stream"] is True
    assert payload["stream_options"]["include_usage"] is True
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["max_tokens"] == 384_000
    assert payload["reasoning_effort"] == "max"


def test_chat_completion_usage_event_exposes_exact_usage_cost() -> None:
    usage, response = proxy.usage_from_event(
        {
            "id": "gen-chat-1",
            "model": "deepseek/deepseek-v4-flash-vision-exp",
            "choices": [],
            "usage": {
                "prompt_tokens": 1_000,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens": 100,
                "completion_tokens_details": {"reasoning_tokens": 75},
                "total_tokens": 1_100,
                "cost": 0.0001156,
            },
        }
    )

    assert usage is not None
    assert usage["cost"] == 0.0001156
    assert usage["prompt_tokens_details"]["cached_tokens"] == 800
    assert response["id"] == "gen-chat-1"


def test_generic_proxy_preserves_requested_parallel_tool_calls() -> None:
    _body, payload = proxy.pin_provider_route(
        json.dumps(
            {
                "model": "example/model",
                "input": "hello",
                "parallel_tool_calls": True,
            }
        ).encode(),
        provider_endpoint="example",
        quantization=None,
    )

    assert payload["parallel_tool_calls"] is True


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


def test_baidu_promotion_cannot_extend_the_budget() -> None:
    parsed = proxy.sys.modules[
        "sprint_openrouter_pricing"
    ].parse_endpoint_discount_snapshot(
        {
            "data": {
                "endpoints": [
                    {
                        "provider_name": "Baidu",
                        "tag": "baidu/fp8",
                        "quantization": "fp8",
                        "pricing": {
                            "prompt": "0.0000000658",
                            "input_cache_read": "0.00000001316",
                            "completion": "0.0000001316",
                            "discount": 0.53,
                        },
                    }
                ]
            }
        },
        model="deepseek/deepseek-v4-flash-0731",
        provider_tag="baidu/fp8",
    )

    assert parsed["discount_fraction"] == 0.53
    assert parsed["gross_up_multiplier"] == pytest.approx(1 / 0.47)
    assert proxy.undiscounted_cost_usd(4.70, parsed) == pytest.approx(10.0)
    assert proxy.benchmark_cost_usd(
        4.70,
        parsed,
        {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
        },
    ) == pytest.approx(10.0)


def test_deepseek_off_peak_charge_is_normalized_to_official_peak_rates() -> None:
    parsed = proxy.sys.modules[
        "sprint_openrouter_pricing"
    ].parse_endpoint_discount_snapshot(
        {
            "data": {
                "endpoints": [
                    {
                        "provider_name": "DeepSeek",
                        "tag": "deepseek",
                        "pricing": {
                            "prompt": "0.00000022",
                            "input_cache_read": "0.000000007",
                            "completion": "0.00000066",
                            "discount": 0,
                        },
                    }
                ]
            }
        },
        model="deepseek/deepseek-v4-flash-0731",
        provider_tag="deepseek",
    )
    usage = {
        "input_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 800},
        "output_tokens": 100,
    }

    # 200 cache misses at $0.44/M + 800 hits at $0.014/M +
    # 100 output at $1.32/M.
    peak_cost = 0.0002312
    assert parsed["cost_basis"] == proxy.BENCHMARK_COST_BASIS
    assert proxy.benchmark_cost_usd(peak_cost / 2, parsed, usage) == pytest.approx(
        peak_cost
    )


def test_deepseek_peak_floor_applies_to_any_pinned_openrouter_provider() -> None:
    parsed = proxy.sys.modules[
        "sprint_openrouter_pricing"
    ].parse_endpoint_discount_snapshot(
        {
            "data": {
                "endpoints": [
                    {
                        "provider_name": "Baidu",
                        "tag": "baidu/fp8",
                        "pricing": {"discount": 0.5},
                    }
                ]
            }
        },
        model="deepseek/deepseek-v4-flash-0731",
        provider_tag="baidu/fp8",
    )
    usage = {
        "input_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 800},
        "output_tokens": 100,
    }

    assert proxy.benchmark_cost_usd(0.00005, parsed, usage) == pytest.approx(
        0.0002312
    )


def test_non_deepseek_route_keeps_undiscounted_openrouter_cost() -> None:
    parsed = proxy.sys.modules[
        "sprint_openrouter_pricing"
    ].parse_endpoint_discount_snapshot(
        {
            "data": {
                "endpoints": [
                    {
                        "provider_name": "OpenAI",
                        "tag": "openai",
                        "pricing": {"discount": 0.5},
                    }
                ]
            }
        },
        model="openai/gpt-5.6-luna",
        provider_tag="openai",
    )

    assert proxy.benchmark_cost_usd(0.25, parsed, {}) == pytest.approx(0.5)


def test_deepseek_peak_normalization_fails_closed_without_token_usage() -> None:
    parsed = proxy.sys.modules[
        "sprint_openrouter_pricing"
    ].parse_endpoint_discount_snapshot(
        {
            "data": {
                "endpoints": [
                    {"provider_name": "DeepSeek", "tag": "deepseek", "pricing": {}}
                ]
            }
        },
        model="deepseek/deepseek-v4-flash-0731",
        provider_tag="deepseek",
    )

    with pytest.raises(proxy.OpenRouterPricingError, match="requires token usage"):
        proxy.benchmark_cost_usd(0.1, parsed, None)


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


def test_proxy_rejects_unknown_request_contract_fields(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()
    (tmp_path / "state/run.json").write_text(
        json.dumps({"run_id": "run-1", "model": "vendor/model"})
    )
    with pytest.raises(ValueError, match="invalid request contract fields"):
        proxy.LedgerProxyServer(
            ("127.0.0.1", 0),
            upstream="https://openrouter.ai/api/v1",
            ledger_root=tmp_path / "api-usage",
            run_id="run-1",
            cpu_attempt=1,
            provider_endpoint="provider",
            request_contract={"unsealed": True},
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
    (run_root / "BUDGET_STOP_REQUESTED.json").write_text(
        '{"reason":"agent_cost_budget_exhausted"}\n'
    )
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
    assert marker["schema_version"] == 2
    assert marker["status"] == "stop_requested"
    assert marker["total_usd"] == pytest.approx(10.05)


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


def test_streaming_chat_completions_is_sealed_metered_and_peak_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs/run-1"
    state = run_root / "state"
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "model": "deepseek/deepseek-v4-flash-vision-exp",
                "agent_cost_budget_usd": 0.0002,
            }
        )
    )
    promotion = proxy.sys.modules[
        "sprint_openrouter_pricing"
    ].parse_endpoint_discount_snapshot(
        {
            "data": {
                "endpoints": [
                    {
                        "provider_name": "DeepSeek",
                        "tag": "deepseek",
                        "pricing": {"discount": 0},
                    }
                ]
            }
        },
        model="deepseek/deepseek-v4-flash-vision-exp",
        provider_tag="deepseek",
    )
    monkeypatch.setattr(proxy, "capture_endpoint_discount_snapshot", lambda **_: promotion)

    class FakeResponse:
        status = 200
        reason = "OK"

        def __init__(self) -> None:
            event = {
                "id": "gen-chat",
                "model": "deepseek/deepseek-v4-flash-vision-exp",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1_000,
                    "prompt_tokens_details": {"cached_tokens": 800},
                    "completion_tokens": 100,
                    "total_tokens": 1_100,
                    "cost": 0.0001156,
                },
            }
            self.chunks = [
                f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode(),
                b"",
            ]

        def getheader(self, name: str) -> str | None:
            return {
                "Content-Type": "text/event-stream",
                "X-Generation-Id": "gen-chat",
            }.get(name)

        def getheaders(self) -> list[tuple[str, str]]:
            return [("Content-Type", "text/event-stream")]

        def read(self, _size: int) -> bytes:
            return self.chunks.pop(0)

    class FakeConnection:
        sent_body: dict[str, object] | None = None

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request(
            self,
            _method: str,
            _path: str,
            *,
            body: bytes,
            headers: dict[str, str],
        ) -> None:
            FakeConnection.sent_body = json.loads(body)
            assert headers["X-OpenRouter-Metadata"] == "enabled"

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(proxy.http.client, "HTTPSConnection", FakeConnection)
    contract = {
        "model": "deepseek/deepseek-v4-flash-vision-exp",
        "stream": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 384_000,
        "reasoning_effort": "max",
    }
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        runtime_dir=tmp_path / "runtime",
        provider_endpoint="deepseek",
        request_contract=contract,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        client.request(
            "POST",
            "/api/v1/chat/completions",
            body=json.dumps(
                {
                    "model": "caller/model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                }
            ).encode(),
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
        )
        response = client.getresponse()
        assert response.status == 200
        assert b"gen-chat" in response.read()
        client.close()

        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        client.request(
            "POST",
            "/api/v1/chat/completions",
            body=b'{"messages":[]}',
            headers={"Content-Type": "application/json"},
        )
        response = client.getresponse()
        assert response.status == 402
        assert b"budget exhausted" in response.read()
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    sent = FakeConnection.sent_body
    assert sent is not None
    assert sent["model"] == contract["model"]
    assert sent["provider"] == {
        "only": ["deepseek"],
        "order": ["deepseek"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["temperature"] == 1.0
    assert sent["top_p"] == 0.95
    assert sent["max_tokens"] == 384_000
    summary = json.loads((run_root / "api-usage/summary.json").read_text())
    assert summary["provider_billed_model_api_usd"] == pytest.approx(0.0001156)
    assert summary["model_api_usd"] == pytest.approx(0.0002312)
    records = list((run_root / "api-usage/requests").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["api_path"] == "/api/v1/chat/completions"
    assert record["state"] == "complete"
    assert record["usage"]["prompt_tokens_details"]["cached_tokens"] == 800
    marker = json.loads((run_root / "BUDGET_STOP_REQUESTED.json").read_text())
    assert marker["status"] == "stop_requested"
    assert marker["total_usd"] == pytest.approx(0.0002312)
    assert (tmp_path / "runtime/sprint-stop").read_text().strip() == (
        "agent_cost_budget_exhausted"
    )


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
