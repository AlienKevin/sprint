from __future__ import annotations

import importlib.util
import http.client
import json
from pathlib import Path
import threading

import pytest

from event_runtime.container.sprint_openrouter_usage import empty_token_usage


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "event_runtime/container/sprint-openrouter-ledger-proxy.py"

spec = importlib.util.spec_from_file_location("sprint_openrouter_ledger_proxy", SCRIPT)
assert spec and spec.loader
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
REQUEST_1 = "a" * 32
REQUEST_2 = "b" * 32
PENDING_REQUEST = "c" * 32
UNDISCOUNTED_BASIS = "openrouter_list_price_before_endpoint_discount"
DEEPSEEK_PEAK_BASIS = "openrouter_list_price_with_deepseek_peak_floor"


def write_run_contract(
    run_root: Path,
    *,
    run_id: str = "run-1",
    model: str = "vendor/model",
    budget: float = 10.0,
) -> None:
    pricing = proxy.sys.modules["sprint_openrouter_pricing"]
    basis = pricing.benchmark_cost_basis_for_model(model)
    provider = model.split("/", 1)[0]
    endpoint_snapshot = pricing.parse_endpoint_discount_snapshot(
        {
            "data": {
                "endpoints": [
                    {
                        "provider_name": provider,
                        "tag": provider,
                        "pricing": {"discount": 0},
                    }
                ]
            }
        },
        model=model,
        provider_tag=provider,
        captured_at="2026-08-26T00:00:00Z",
        source_url="https://openrouter.ai/test-fixture",
    )
    state = run_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": model,
                "agent_cost_budget_usd": budget,
                "api_pricing_snapshot": {"cost_basis": basis},
                "openrouter_pricing_snapshot": endpoint_snapshot,
                "budget_enforcement": {"api_budget_cost_basis": basis},
            }
        )
    )


def test_proxy_does_not_timeout_long_reasoning_requests() -> None:
    assert proxy.UPSTREAM_SOCKET_TIMEOUT_SECONDS is None


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


def test_openrouter_child_key_usage_recovery_uses_provider_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"data":{"usage":0.03448014}}'

    def urlopen(request: object, timeout: float):
        assert request.full_url == "https://openrouter.ai/api/v1/key"
        assert request.headers["Authorization"] == "Bearer isolated-key"
        assert timeout == 5
        return Response()

    monkeypatch.setattr(proxy.urllib.request, "urlopen", urlopen)

    assert proxy.recover_openrouter_key_usage_usd(
        "Bearer isolated-key"
    ) == pytest.approx(0.03448014)


@pytest.mark.parametrize("value", [None, True, -0.01, "0.1"])
def test_openrouter_child_key_usage_recovery_rejects_invalid_totals(
    value: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"data": {"usage": value}}).encode()

    monkeypatch.setattr(
        proxy.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )

    assert proxy.recover_openrouter_key_usage_usd("Bearer key") is None


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


@pytest.mark.parametrize(
    "tool",
    [
        {
            "type": "function",
            "name": "create_goal",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string"},
                    "token_budget": {"type": "integer"},
                },
                "required": ["objective", "token_budget"],
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_goal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "token_budget": {"type": "integer"},
                    },
                    "required": ["objective"],
                },
            },
        },
    ],
)
def test_proxy_removes_model_controlled_goal_token_budget(tool: dict) -> None:
    _body, payload = proxy.pin_provider_route(
        json.dumps(
            {
                "model": "example/model",
                "input": "/goal keep working",
                "tools": [
                    tool,
                    {
                        "type": "function",
                        "name": "unrelated",
                        "parameters": {
                            "type": "object",
                            "properties": {"token_budget": {"type": "integer"}},
                        },
                    },
                ],
            }
        ).encode(),
        provider_endpoint="example",
        quantization=None,
    )

    goal = payload["tools"][0]
    schema = goal.get("parameters") or goal["function"]["parameters"]
    assert "token_budget" not in schema["properties"]
    assert "token_budget" not in schema.get("required", [])
    assert schema["additionalProperties"] is False
    assert payload["tools"][1]["parameters"]["properties"]["token_budget"] == {
        "type": "integer"
    }


@pytest.mark.parametrize(
    ("blocked_tool", "blocked_choice"),
    [
        (
            {"type": "function", "name": "request_user_input"},
            {"type": "function", "name": "request_user_input"},
        ),
        (
            {
                "type": "function",
                "function": {"name": "request_user_input"},
            },
            {
                "type": "function",
                "function": {"name": "request_user_input"},
            },
        ),
    ],
)
def test_proxy_removes_operator_input_tool_from_unattended_runs(
    blocked_tool: dict, blocked_choice: dict
) -> None:
    _body, payload = proxy.pin_provider_route(
        json.dumps(
            {
                "model": "example/model",
                "input": "keep working autonomously",
                "tools": [
                    blocked_tool,
                    {"type": "function", "name": "exec_command"},
                ],
                "tool_choice": blocked_choice,
            }
        ).encode(),
        provider_endpoint="example",
        quantization=None,
    )

    assert [proxy.tool_name(tool) for tool in payload["tools"]] == ["exec_command"]
    assert "tool_choice" not in payload


def test_proxy_strips_hallucinated_goal_budget_from_streamed_tool_call() -> None:
    sanitizer = proxy.GoalToolStreamSanitizer()
    events = [
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "fc_goal",
                "type": "function_call",
                "name": "create_goal",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "output_index": 0,
            "item_id": "fc_goal",
            "delta": '{"objective":"keep working",',
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 3,
            "output_index": 0,
            "item_id": "fc_goal",
            "delta": '"token_budget":1}',
        },
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": 4,
            "output_index": 0,
            "item_id": "fc_goal",
            "arguments": '{"objective":"keep working","token_budget":1}',
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 5,
            "output_index": 0,
            "item": {
                "id": "fc_goal",
                "type": "function_call",
                "name": "create_goal",
                "arguments": '{"objective":"keep working","token_budget":1}',
            },
        },
    ]

    rewritten = []
    for event in events:
        rewritten.extend(sanitizer.rewrite_event(event))

    deltas = [
        event["delta"]
        for event in rewritten
        if event.get("type") == "response.function_call_arguments.delta"
    ]
    assert deltas == ['{"objective":"keep working"}']
    terminal_arguments = [
        event["arguments"]
        for event in rewritten
        if event.get("type") == "response.function_call_arguments.done"
    ]
    terminal_arguments.extend(
        event["item"]["arguments"]
        for event in rewritten
        if event.get("type") == "response.output_item.done"
    )
    assert terminal_arguments == [
        '{"objective":"keep working"}',
        '{"objective":"keep working"}',
    ]
    assert all("token_budget" not in value for value in terminal_arguments)


def test_proxy_strips_goal_budget_from_nonstream_response() -> None:
    payload = {
        "id": "response-1",
        "output": [
            {
                "type": "function_call",
                "name": "create_goal",
                "arguments": '{"objective":"continue","token_budget":1}',
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"token_budget":1}',
            },
        ],
    }

    proxy.sanitize_goal_tool_calls(payload)

    assert payload["output"][0]["arguments"] == '{"objective":"continue"}'
    assert payload["output"][1]["arguments"] == '{"token_budget":1}'


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


def test_deepseek_reasoning_is_aliased_to_native_harness_field() -> None:
    payload = {
        "choices": [
            {
                "delta": {
                    "reasoning": "inspect the task",
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "inspect the task"}
                    ],
                }
            }
        ]
    }

    proxy.expose_deepseek_reasoning_content(payload)

    delta = payload["choices"][0]["delta"]
    assert delta["reasoning"] == "inspect the task"
    assert delta["reasoning_content"] == "inspect the task"
    assert delta["reasoning_details"] == [
        {"type": "reasoning.text", "text": "inspect the task"}
    ]


def test_deepseek_native_reasoning_content_is_not_overwritten() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "reasoning": "normalized",
                    "reasoning_content": "native",
                }
            }
        ]
    }

    proxy.expose_deepseek_reasoning_content(payload)

    assert payload["choices"][0]["message"]["reasoning_content"] == "native"


def test_proxy_sanitizes_binary_terminal_text_before_provider_request() -> None:
    body, payload = proxy.pin_provider_route(
        json.dumps(
            {
                "model": "deepseek/deepseek-v4-flash-vision-exp",
                "messages": [
                    {
                        "role": "tool",
                        "content": "binary:\udf8b\u0000\u001b[31m\ufffd",
                    }
                ],
            }
        ).encode(),
        provider_endpoint="deepseek",
        quantization=None,
    )

    assert json.loads(body) == payload
    content = payload["messages"][0]["content"]
    assert content == "binary:\ufffd\\x00\\x1b[31m\ufffd"
    assert not any(0xD800 <= ord(character) <= 0xDFFF for character in content)
    assert all(ord(character) >= 0x20 or character in "\t\n\r" for character in content)


def test_responses_contract_seals_luna_benchmark_parameters() -> None:
    body, payload = proxy.pin_provider_route(
        json.dumps(
            {
                "model": "caller/other-model",
                "input": "hello",
                "temperature": 0.2,
                "top_p": 0.2,
                "max_output_tokens": 32,
                "reasoning": {"effort": "low"},
                "service_tier": "flex",
            }
        ).encode(),
        provider_endpoint="openai",
        quantization=None,
        request_contract={
            "model": "openai/gpt-5.6-luna",
            "max_output_tokens": 128_000,
            "reasoning": {"effort": "max", "summary": "auto"},
            "service_tier": "default",
        },
    )

    assert json.loads(body) == payload
    assert payload["model"] == "openai/gpt-5.6-luna"
    assert payload["provider"]["only"] == ["openai"]
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.2
    assert payload["max_output_tokens"] == 128_000
    assert payload["reasoning"] == {"effort": "max", "summary": "auto"}
    assert payload["service_tier"] == "default"


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


def test_anthropic_message_usage_is_canonicalized_and_merged() -> None:
    start_usage, response = proxy.usage_from_event(
        {
            "type": "message_start",
            "message": {
                "id": "gen-anthropic-1",
                "model": "anthropic/claude-opus-5",
                "usage": {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 80,
                    "output_tokens": 1,
                },
            },
        }
    )
    end_usage, _ = proxy.usage_from_event(
        {"type": "message_delta", "usage": {"output_tokens": 30}}
    )

    assert start_usage is not None and end_usage is not None
    usage = proxy.merge_stream_usage(start_usage, end_usage)
    assert usage["input_tokens"] == 200
    assert usage["input_tokens_details"] == {
        "cached_tokens": 80,
        "cache_write_tokens": 20,
    }
    assert usage["output_tokens"] == 30
    assert usage["total_tokens"] == 230
    assert response["id"] == "gen-anthropic-1"


def test_glm_messages_translates_claude_effort_without_openai_stream_options() -> None:
    _body, payload = proxy.pin_provider_route(
        json.dumps(
            {
                "model": "caller-alias",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "output_config": {"effort": "low"},
            }
        ).encode(),
        provider_endpoint="z-ai/fp8",
        quantization="fp8",
        request_contract={
            "model": "z-ai/glm-5.3-flash",
            "max_tokens": 131_072,
            "reasoning_effort": "max",
            "stream": True,
        },
        inference_path="messages",
    )

    assert payload["provider"] == {
        "only": ["z-ai/fp8"],
        "order": ["z-ai/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "quantizations": ["fp8"],
    }
    assert payload["reasoning_effort"] == "max"
    assert "output_config" not in payload
    assert "stream_options" not in payload


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


def test_official_openai_route_removes_unsupported_codex_verbosity_hint() -> None:
    _body, payload = proxy.pin_provider_route(
        json.dumps(
            {
                "model": "openai/gpt-5.6-luna",
                "input": "hello",
                "text": {"verbosity": "low"},
            }
        ).encode(),
        provider_endpoint="openai",
        quantization=None,
    )

    assert "text" not in payload


def test_proxy_preserves_text_configuration_outside_narrow_openai_shim() -> None:
    rich_text = {"verbosity": "low", "format": {"type": "text"}}
    for provider_endpoint, text in (
        ("example", {"verbosity": "low"}),
        ("openai", rich_text),
    ):
        _body, payload = proxy.pin_provider_route(
            json.dumps(
                {
                    "model": "example/model",
                    "input": "hello",
                    "text": text,
                }
            ).encode(),
            provider_endpoint=provider_endpoint,
            quantization=None,
        )

        assert payload["text"] == text


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


@pytest.mark.parametrize(
    ("model", "provider_tag", "provider_name", "quantization", "discount"),
    (
        ("anthropic/claude-opus-5", "anthropic", "Anthropic", "unknown", 0.25),
        ("z-ai/glm-5.3-flash", "z-ai/fp8", "Z.AI", "fp8", 0.5),
    ),
)
def test_claude_code_models_reverse_openrouter_endpoint_discounts(
    model: str,
    provider_tag: str,
    provider_name: str,
    quantization: str,
    discount: float,
) -> None:
    pricing = proxy.sys.modules["sprint_openrouter_pricing"]
    parsed = pricing.parse_endpoint_discount_snapshot(
        {
            "data": {
                "endpoints": [
                    {
                        "provider_name": provider_name,
                        "tag": provider_tag,
                        "quantization": quantization,
                        "pricing": {"discount": discount},
                    }
                ]
            }
        },
        model=model,
        provider_tag=provider_tag,
        captured_at="2026-08-28T00:00:00Z",
        source_url=f"https://openrouter.ai/api/v1/models/{model}/endpoints",
    )

    multiplier = 1.0 / (1.0 - discount)
    assert parsed["cost_basis"] == "openrouter_list_price_before_endpoint_discount"
    assert parsed["discount_fraction"] == discount
    assert parsed["gross_up_multiplier"] == pytest.approx(multiplier)
    assert pricing.undiscounted_cost_usd(0.25, parsed) == pytest.approx(
        0.25 * multiplier
    )
    assert pricing.benchmark_cost_usd(0.25, parsed, {}) == pytest.approx(
        0.25 * multiplier
    )


def test_sol_removes_only_openrouter_endpoint_discount() -> None:
    parsed = proxy.sys.modules[
        "sprint_openrouter_pricing"
    ].parse_endpoint_discount_snapshot(
        {
            "data": {
                "endpoints": [
                    {
                        "provider_name": "OpenAI",
                        "tag": "openai",
                        "pricing": {"discount": 0.25},
                    }
                ]
            }
        },
        model="openai/gpt-5.6-sol",
        provider_tag="openai",
    )
    usage = {
        "input_tokens": 1_000_000,
        "input_tokens_details": {
            "cached_tokens": 800_000,
            "cache_write_tokens": 100_000,
        },
        "output_tokens": 100_000,
    }

    # The endpoint discount first grosses $3.00 up to $4.00. The official
    # promotional long-context schedule is higher: $0.80 uncached input +
    # $0.64 cached input + $1.00 cache write + $3.00 output = $5.44.
    assert proxy.benchmark_cost_usd(3.0, parsed, usage) == pytest.approx(5.44)
    assert parsed["cost_basis"] == (
        "openai_sol_official_promotional_list_price_after_openrouter_discount_reversal"
    )


def test_sol_promotional_short_context_schedule() -> None:
    parsed = proxy.sys.modules[
        "sprint_openrouter_pricing"
    ].parse_endpoint_discount_snapshot(
        {"data": {"endpoints": [{"tag": "openai", "pricing": {"discount": 0}}]}},
        model="openai/gpt-5.6-sol",
        provider_tag="openai",
    )
    usage = {
        "input_tokens": 200_000,
        "input_tokens_details": {
            "cached_tokens": 100_000,
            "cache_write_tokens": 50_000,
        },
        "output_tokens": 10_000,
    }

    # $0.20 uncached + $0.04 cached + $0.25 cache write + $0.20 output.
    assert proxy.benchmark_cost_usd(0.1, parsed, usage) == pytest.approx(0.69)


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
    assert parsed["cost_basis"] == DEEPSEEK_PEAK_BASIS
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

    assert proxy.benchmark_cost_usd(0.00005, parsed, usage) == pytest.approx(0.0002312)


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
    write_run_contract(tmp_path)
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


def test_responses_only_proxy_rejects_chat_and_other_post_paths(tmp_path: Path) -> None:
    write_run_contract(tmp_path, model="openai/gpt-5.6-luna")
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=tmp_path / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        allowed_inference_path="responses",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for path in ("/api/v1/chat/completions", "/api/v1/embeddings"):
            client = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=5
            )
            client.request("POST", path, body=b"{}")
            response = client.getresponse()
            assert response.status == 405
            response.read()
            client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_proxy_rejects_run_model_cost_basis_mismatch(tmp_path: Path) -> None:
    write_run_contract(tmp_path, model="openai/gpt-5.6-luna")
    run_path = tmp_path / "state/run.json"
    run = json.loads(run_path.read_text())
    run["budget_enforcement"]["api_budget_cost_basis"] = DEEPSEEK_PEAK_BASIS
    run["api_pricing_snapshot"]["cost_basis"] = DEEPSEEK_PEAK_BASIS
    run_path.write_text(json.dumps(run))

    with pytest.raises(ValueError, match="model and cost basis mismatch"):
        proxy.LedgerProxyServer(
            ("127.0.0.1", 0),
            upstream="https://openrouter.ai/api/v1",
            ledger_root=tmp_path / "api-usage",
            run_id="run-1",
            cpu_attempt=1,
        )


def test_proxy_requires_a_valid_launch_pricing_snapshot(tmp_path: Path) -> None:
    write_run_contract(tmp_path)
    run_path = tmp_path / "state/run.json"
    run = json.loads(run_path.read_text())
    run.pop("openrouter_pricing_snapshot")
    run_path.write_text(json.dumps(run))

    with pytest.raises(ValueError, match="sealed OpenRouter pricing snapshot"):
        proxy.LedgerProxyServer(
            ("127.0.0.1", 0),
            upstream="https://openrouter.ai/api/v1",
            ledger_root=tmp_path / "api-usage",
            run_id="run-1",
            cpu_attempt=1,
        )


def test_generic_openrouter_budget_gate_blocks_a_second_paid_request(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(run_root)
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
                "benchmark_cost_usd": 9.95,
                "cost_basis": UNDISCOUNTED_BASIS,
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
    write_run_contract(run_root)
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
                "schema_version": 3,
                "run_id": "run-1",
                "model_api_usd": 1.0,
                "model_api_cost_basis": UNDISCOUNTED_BASIS,
                "completed_request_count": 10,
                "pending_request_count": 1,
                "in_flight_request_count": 0,
                "cost_recovery_required_count": 1,
                "in_flight_request_ids": [],
                "cost_recovery_required_request_ids": [PENDING_REQUEST],
                "token_usage": empty_token_usage(),
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
                "benchmark_cost_usd": 0.5,
                "cost_basis": UNDISCOUNTED_BASIS,
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
    write_run_contract(run_root)
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


def test_proxy_health_blocks_startup_while_prior_charge_is_unknown(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(run_root)
    record = run_root / f"api-usage/requests/{PENDING_REQUEST}.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "ledger_request_id": PENDING_REQUEST,
                "run_id": "run-1",
                "state": "cost_recovery_required",
                "generation_id": None,
                "provider_reported_cost_usd": None,
            }
        )
    )
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=2,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        client.request("GET", "/healthz")
        response = client.getresponse()
        assert response.status == 503
        health = json.loads(response.read())
        client.close()
        assert health == {
            "status": "reconciling",
            "ready": False,
            "pending_request_count": 1,
            "in_flight_request_count": 0,
            "cost_recovery_required_count": 1,
        }

        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        client.request("GET", "/ledger-status")
        response = client.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["pending_request_count"] == 1
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_proxy_restart_recovers_exact_generation_before_becoming_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs/run-1"
    model = "deepseek/deepseek-v4-flash-vision-exp"
    write_run_contract(run_root, model=model)
    pricing = {
        "uncached_input": 0.44 / 1_000_000,
        "cached_input": 0.014 / 1_000_000,
        "output": 1.32 / 1_000_000,
    }
    record = run_root / f"api-usage/requests/{PENDING_REQUEST}.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "ledger_request_id": PENDING_REQUEST,
                "run_id": "run-1",
                "state": "cost_recovery_required",
                "generation_id": "gen-recover-me",
                "provider_reported_cost_usd": None,
                "promotion_snapshot": {
                    "model": model,
                    "provider_tag": "deepseek",
                    "discount_fraction": 0.0,
                    "deepseek_peak_pricing_usd_per_token": pricing,
                    "cost_basis": "openrouter_list_price_with_deepseek_peak_floor",
                },
            }
        )
    )
    (record.parent.parent / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": "run-1",
                "model_api_usd": 1.0,
                "model_api_cost_basis": DEEPSEEK_PEAK_BASIS,
                "provider_billed_model_api_usd": 0.75,
                "completed_request_count": 4,
                "pending_request_count": 1,
                "in_flight_request_count": 0,
                "cost_recovery_required_count": 1,
                "in_flight_request_ids": [],
                "cost_recovery_required_request_ids": [PENDING_REQUEST],
                "token_usage": empty_token_usage(),
            }
        )
    )
    monkeypatch.setattr(
        proxy,
        "recover_openrouter_generation",
        lambda generation_id, authorization: {
            "id": generation_id,
            "total_cost": 0.10,
            "native_tokens_prompt": 1_000_000,
            "native_tokens_cached": 0,
            "native_tokens_completion": 1_000_000,
            "authorization_seen": authorization,
        },
    )

    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=2,
        upstream_api_key="sealed-child-key-123456",
    )
    try:
        assert server.reconciliation_status()["ready"] is True
        assert server.api_cost_usd == pytest.approx(2.76)
        assert server.provider_billed_api_cost_usd == pytest.approx(0.85)
        assert server.completed_request_count == 5
        assert server.token_usage == {
            "input_tokens": 1_000_000,
            "ordinary_uncached_input_tokens": 1_000_000,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": 1_000_000,
            "reasoning_output_tokens": 0,
            "total_tokens": 2_000_000,
        }
        recovered = json.loads(record.read_text())
        assert recovered["state"] == "recovered_complete"
        assert recovered["benchmark_cost_usd"] == pytest.approx(1.76)
        assert recovered["provider_reported_cost_usd"] == pytest.approx(0.10)
        assert recovered["recovered_after_proxy_restart"] is True
    finally:
        server.server_close()


def test_codex_wrapper_drains_proxy_and_persists_unrecoverable_stop() -> None:
    source = (ROOT / "event_runtime/container/sprint-codex-exec-wrapper.sh").read_text()

    assert 'url = f"{base.scheme}://{base.netloc}/ledger-status"' in source
    assert "OPENROUTER_PROXY_DRAIN_TIMEOUT_SECONDS" in source
    assert "OPENROUTER_PROXY_RECOVERY_TIMEOUT_SECONDS" in source
    assert "fail_closed_openrouter_recovery" in source
    assert '"reason": "budget_telemetry_unavailable"' in source


def test_proxy_maintains_constant_size_exact_cost_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(run_root)
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


def test_proxy_waits_for_exact_generation_recovery_before_releasing_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(run_root)
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        runtime_dir=tmp_path / "runtime",
    )
    server.cost_recovery_required_request_ids.add(REQUEST_1)
    calls = 0

    def recover(*, include_in_flight: bool) -> None:
        nonlocal calls
        assert include_in_flight is False
        calls += 1
        if calls == 2:
            server.cost_recovery_required_request_ids.remove(REQUEST_1)

    monkeypatch.setattr(server, "_reconcile_pending_requests", recover)
    monkeypatch.setattr(proxy.time, "sleep", lambda _seconds: None)
    try:
        assert server.recover_request_until(REQUEST_1, timeout_seconds=1.0) is True
        assert calls == 2
    finally:
        server.server_close()


def test_proxy_generation_recovery_has_hard_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(run_root)
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        runtime_dir=tmp_path / "runtime",
    )
    server.cost_recovery_required_request_ids.add(REQUEST_1)
    monotonic = iter((0.0, 2.0))
    monkeypatch.setattr(server, "_reconcile_pending_requests", lambda **_: None)
    monkeypatch.setattr(proxy.time, "monotonic", lambda: next(monotonic))
    try:
        assert server.recover_request_until(REQUEST_1, timeout_seconds=1.0) is False
    finally:
        server.server_close()


def test_proxy_ambiguous_charge_recovery_stops_only_after_both_checks_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(run_root)
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        runtime_dir=tmp_path / "runtime",
    )
    server.cost_recovery_required_request_ids.add(REQUEST_1)
    calls: list[str] = []
    monkeypatch.setattr(
        server,
        "recover_request_until",
        lambda request_id, *, timeout_seconds: (
            calls.append(f"generation:{request_id}:{timeout_seconds}") or False
        ),
    )
    monkeypatch.setattr(
        server,
        "resolve_unbilled_request_from_key_usage",
        lambda request_id: calls.append(f"key:{request_id}") or False,
    )
    try:
        assert server.recover_request_charge_or_stop(REQUEST_1) is False
        assert calls == [
            f"generation:{REQUEST_1}:120.0",
            f"key:{REQUEST_1}",
        ]
        marker = json.loads((run_root / "BUDGET_STOP_REQUESTED.json").read_text())
        assert marker == {
            "schema_version": 2,
            "run_id": "run-1",
            "reason": "budget_telemetry_unavailable",
            "status": "fail_closed",
        }
    finally:
        server.server_close()


@pytest.mark.parametrize("generation_recovered", [True, False])
def test_proxy_ambiguous_charge_recovery_does_not_stop_after_exact_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_recovered: bool,
) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(run_root)
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        runtime_dir=tmp_path / "runtime",
    )
    server.cost_recovery_required_request_ids.add(REQUEST_1)
    key_calls = 0
    monkeypatch.setattr(
        server,
        "recover_request_until",
        lambda _request_id, *, timeout_seconds: generation_recovered,
    )

    def recover_from_key(_request_id: str) -> bool:
        nonlocal key_calls
        key_calls += 1
        return True

    monkeypatch.setattr(
        server, "resolve_unbilled_request_from_key_usage", recover_from_key
    )
    try:
        assert server.recover_request_charge_or_stop(REQUEST_1) is True
        assert key_calls == (0 if generation_recovered else 1)
        assert not (run_root / "BUDGET_STOP_REQUESTED.json").exists()
        assert not (tmp_path / "runtime/sprint-stop").exists()
    finally:
        server.server_close()


def test_proxy_resolves_missing_generation_only_when_child_key_proves_unbilled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(run_root)
    record = run_root / f"api-usage/requests/{PENDING_REQUEST}.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "ledger_request_id": PENDING_REQUEST,
                "run_id": "run-1",
                "state": "cost_recovery_required",
                "generation_id": "gen-never-audited",
                "provider_reported_cost_usd": None,
                "promotion_snapshot": {
                    "cost_basis": UNDISCOUNTED_BASIS,
                    "discount_fraction": 0.0,
                },
            }
        )
    )
    monkeypatch.setattr(proxy, "recover_openrouter_generation", lambda *_args: None)
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        upstream_api_key="isolated-child-key",
    )
    server.cost_recovery_required_request_ids.add(PENDING_REQUEST)
    monkeypatch.setattr(
        proxy,
        "recover_openrouter_key_usage_usd",
        lambda authorization: server.provider_billed_api_cost_usd,
    )
    try:
        assert server.resolve_unbilled_request_from_key_usage(PENDING_REQUEST) is True
        assert server.completed_request_count == 1
        assert not server.cost_recovery_required_request_ids
        recovered = json.loads(record.read_text())
        assert recovered["state"] == "recovered_not_billed"
        assert recovered["provider_reported_cost_usd"] == 0.0
        assert recovered["benchmark_cost_usd"] == 0.0
        assert recovered["aggregate_key_usage_at_recovery_usd"] == 0.0
        assert recovered["recovered_after_missing_generation"] is True
    finally:
        server.server_close()


def test_proxy_keeps_missing_generation_fail_closed_when_child_key_usage_grew(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(run_root)
    record = run_root / f"api-usage/requests/{PENDING_REQUEST}.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "ledger_request_id": PENDING_REQUEST,
                "run_id": "run-1",
                "state": "cost_recovery_required",
                "provider_reported_cost_usd": None,
                "promotion_snapshot": {"cost_basis": UNDISCOUNTED_BASIS},
            }
        )
    )
    monkeypatch.setattr(proxy, "recover_openrouter_generation", lambda *_args: None)
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        upstream_api_key="isolated-child-key",
    )
    server.cost_recovery_required_request_ids.add(PENDING_REQUEST)
    monkeypatch.setattr(
        proxy, "recover_openrouter_key_usage_usd", lambda _authorization: 0.01
    )
    try:
        assert server.resolve_unbilled_request_from_key_usage(PENDING_REQUEST) is False
        assert server.completed_request_count == 0
        assert server.cost_recovery_required_request_ids == {PENDING_REQUEST}
        assert json.loads(record.read_text())["state"] == "cost_recovery_required"
    finally:
        server.server_close()


def test_upstream_incomplete_read_waits_for_charge_recovery_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(run_root, model="openai/gpt-5.6-luna")
    monkeypatch.setattr(
        proxy,
        "capture_endpoint_discount_snapshot",
        lambda **_: (_ for _ in ()).throw(
            proxy.OpenRouterPricingError("transient metadata outage")
        ),
    )

    class IncompleteResponse:
        status = 200
        reason = "OK"

        def getheader(self, name: str) -> str | None:
            return {
                "Content-Type": "text/event-stream",
                "X-Generation-Id": "gen-incomplete",
            }.get(name)

        def getheaders(self) -> list[tuple[str, str]]:
            return [("Content-Type", "text/event-stream")]

        def read(self, _size: int) -> bytes:
            raise http.client.IncompleteRead(b"", 1)

    class IncompleteConnection:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request(self, *_args: object, **_kwargs: object) -> None:
            pass

        def getresponse(self) -> IncompleteResponse:
            return IncompleteResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(proxy.http.client, "HTTPSConnection", IncompleteConnection)
    server = proxy.LedgerProxyServer(
        ("127.0.0.1", 0),
        upstream="https://openrouter.ai/api/v1",
        ledger_root=run_root / "api-usage",
        run_id="run-1",
        cpu_attempt=1,
        runtime_dir=tmp_path / "runtime",
        provider_endpoint="openai",
        request_contract={
            "model": "openai/gpt-5.6-luna",
            "max_output_tokens": 128_000,
            "reasoning": {"effort": "max", "summary": "auto"},
            "service_tier": "default",
        },
        allowed_inference_path="responses",
        upstream_api_key="sealed-child-key-123456",
    )
    recovered: list[str] = []

    def recover(request_id: str) -> bool:
        assert server.cost_recovery_required_request_ids == {request_id}
        record = json.loads((server.requests_dir / f"{request_id}.json").read_text())
        assert record["state"] == "cost_recovery_required"
        assert record["generation_id"] == "gen-incomplete"
        assert record["proxy_error_type"] == "IncompleteRead"
        server.cost_recovery_required_request_ids.remove(request_id)
        server.write_summary()
        recovered.append(request_id)
        return True

    monkeypatch.setattr(server, "recover_request_charge_or_stop", recover)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        client.request(
            "POST",
            "/api/v1/responses",
            body=json.dumps({"model": "caller/model", "input": "hello"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = client.getresponse()
        assert response.status == 200
        assert response.read() == b""
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert len(recovered) == 1
    summary = json.loads((run_root / "api-usage/summary.json").read_text())
    assert summary["pending_request_count"] == 0
    assert not (run_root / "BUDGET_STOP_REQUESTED.json").exists()
    assert not (tmp_path / "runtime/sprint-stop").exists()


def test_streaming_chat_completions_is_sealed_metered_and_peak_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs/run-1"
    write_run_contract(
        run_root,
        model="deepseek/deepseek-v4-flash-vision-exp",
        budget=0.0002,
    )
    monkeypatch.setattr(
        proxy,
        "capture_endpoint_discount_snapshot",
        lambda **_: (_ for _ in ()).throw(
            proxy.OpenRouterPricingError("transient metadata outage")
        ),
    )

    class FakeResponse:
        status = 200
        reason = "OK"

        def __init__(self) -> None:
            event = {
                "id": "gen-chat",
                "model": "deepseek/deepseek-v4-flash-vision-exp",
                "choices": [
                    {
                        "delta": {
                            "reasoning": "inspect the task",
                            "reasoning_details": [
                                {
                                    "type": "reasoning.text",
                                    "text": "inspect the task",
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
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
            assert headers["Authorization"] == "Bearer sealed-child-key-123456"

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
        upstream_api_key="sealed-child-key-123456",
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
        response_body = response.read()
        assert b"gen-chat" in response_body
        streamed = json.loads(response_body.split(b"\n", 1)[0][6:])
        delta = streamed["choices"][0]["delta"]
        assert delta["reasoning"] == "inspect the task"
        assert delta["reasoning_content"] == "inspect the task"
        assert delta["reasoning_details"][0]["text"] == "inspect the task"
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
    assert summary["schema_version"] == 3
    assert summary["token_usage"] == {
        "input_tokens": 1_000,
        "ordinary_uncached_input_tokens": 200,
        "cached_input_tokens": 800,
        "cache_write_input_tokens": 0,
        "output_tokens": 100,
        "reasoning_output_tokens": 0,
        "total_tokens": 1_100,
    }
    records = list((run_root / "api-usage/requests").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["api_path"] == "/api/v1/chat/completions"
    assert record["state"] == "complete"
    assert record["promotion_snapshot"]["selection_source"] == (
        "latest_valid_snapshot_after_refresh_failure"
    )
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
    write_run_contract(run_root)
    requests = run_root / "api-usage/requests"
    requests.mkdir(parents=True)
    (requests / "historical.json").write_text("{not-json}\n")
    (requests.parent / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": "run-1",
                "model_api_usd": 1.5,
                "model_api_cost_basis": UNDISCOUNTED_BASIS,
                "completed_request_count": 2_000,
                "pending_request_count": 0,
                "in_flight_request_count": 0,
                "cost_recovery_required_count": 0,
                "in_flight_request_ids": [],
                "cost_recovery_required_request_ids": [],
                "token_usage": empty_token_usage(),
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
    write_run_contract(run_root)
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
                "benchmark_cost_usd": 0.25,
                "cost_basis": UNDISCOUNTED_BASIS,
            }
        )
        + "\n"
    )
    (requests.parent / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": "run-1",
                "model_api_usd": 1.0,
                "model_api_cost_basis": UNDISCOUNTED_BASIS,
                "completed_request_count": 100,
                "pending_request_count": 1,
                "in_flight_request_count": 1,
                "cost_recovery_required_count": 0,
                "in_flight_request_ids": [PENDING_REQUEST],
                "cost_recovery_required_request_ids": [],
                "token_usage": empty_token_usage(),
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
