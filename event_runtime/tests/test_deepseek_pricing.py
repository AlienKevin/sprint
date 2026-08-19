from __future__ import annotations

import datetime as dt
import json

import pytest

from event_runtime.control import deepseek_pricing


PRICING = b"""
<table>
<tr><td>MODEL VERSION</td><td>DeepSeek-V4-Flash-0731</td><td>DeepSeek-V4-Pro-0813</td></tr>
<tr><td rowspan="2">1M INPUT TOKENS (CACHE HIT)</td><td>OFF-PEAK</td><td>$0.007</td><td>$0.022</td></tr>
<tr><td>PEAK</td><td>$0.014</td><td>$0.044</td></tr>
<tr><td rowspan="2">1M INPUT TOKENS (CACHE MISS)</td><td>OFF-PEAK</td><td>$0.22</td><td>$0.66</td></tr>
<tr><td>PEAK</td><td>$0.44</td><td>$1.32</td></tr>
<tr><td rowspan="2">1M OUTPUT TOKENS</td><td>OFF-PEAK</td><td>$0.66</td><td>$1.98</td></tr>
<tr><td>PEAK</td><td>$1.32</td><td>$3.96</td></tr>
<tr><td>Concurrency Limit</td><td>2500</td></tr>
</table>
<p>Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC (all other hours are off-peak).</p>
"""
ANNOUNCEMENT = b"<p>New pricing takes effect at 16:00 UTC, Aug 16, 2026</p>"


def openrouter_fixtures() -> tuple[bytes, bytes, bytes]:
    endpoint = {
        "data": {
            "endpoints": [
                {
                    "name": "DeepSeek | deepseek/deepseek-v4-flash-20260731",
                    "model_id": "deepseek/deepseek-v4-flash-0731",
                    "provider_name": "DeepSeek",
                    "tag": "deepseek",
                    "quantization": "unknown",
                    "context_length": 1_048_576,
                    "max_completion_tokens": 384_000,
                    "supported_parameters": [
                        "reasoning",
                        "reasoning_effort",
                        "tools",
                        "temperature",
                        "top_p",
                    ],
                    "pricing": {
                        "prompt": "0.00000022",
                        "completion": "0.00000066",
                        "input_cache_read": "0.000000007",
                        "overrides": [
                            {
                                "utc_start": 100,
                                "utc_end": 400,
                                "prompt": "0.00000044",
                                "completion": "0.00000132",
                                "input_cache_read": "0.000000014",
                            },
                            {
                                "utc_start": 600,
                                "utc_end": 1000,
                                "prompt": "0.00000044",
                                "completion": "0.00000132",
                                "input_cache_read": "0.000000014",
                            },
                        ],
                    },
                }
            ]
        }
    }
    zdr = {"data": []}
    preset = {
        "data": {
            "id": "preset-fixture",
            "slug": "sprint-deepseek-v4-flash-0731-official",
            "status": "active",
            "designated_version": {
                "version": 2,
                "config": {
                    "model": "deepseek/deepseek-v4-flash-0731",
                    "temperature": 1,
                    "top_p": 0.95,
                    "provider": {
                        "only": ["deepseek"],
                        "allow_fallbacks": False,
                        "data_collection": "allow",
                        "require_parameters": False,
                        "zdr": False,
                    },
                },
            },
        }
    }
    return tuple(json.dumps(value).encode() for value in (endpoint, zdr, preset))


def test_parse_first_party_deepseek_tariff() -> None:
    snapshot = deepseek_pricing.parse_snapshot(
        PRICING,
        ANNOUNCEMENT,
        captured_at=dt.datetime(2026, 8, 18, 18, tzinfo=dt.timezone.utc),
    )

    assert snapshot["model_version"] == "DeepSeek-V4-Flash-0731"
    assert snapshot["effective_from"] == "2026-08-16T16:00:00Z"
    assert snapshot["peak_hours_utc"] == [
        {"start_hour": 1, "end_hour": 4},
        {"start_hour": 6, "end_hour": 10},
    ]
    assert snapshot["rates_usd_per_million_tokens"] == {
        "off_peak": {
            "cache_hit_input": "0.007",
            "cache_miss_input": "0.22",
            "output": "0.66",
        },
        "peak": {
            "cache_hit_input": "0.014",
            "cache_miss_input": "0.44",
            "output": "1.32",
        },
    }
    assert len(snapshot["source_sha256"]) == 64


def test_parse_fails_closed_when_schedule_disappears() -> None:
    with pytest.raises(RuntimeError, match="UTC peak schedule"):
        deepseek_pricing.parse_snapshot(
            PRICING.replace(b"Peak hours are", b"Hours are"), ANNOUNCEMENT
        )


def test_parse_openrouter_official_deepseek_contract() -> None:
    snapshot = deepseek_pricing.parse_openrouter_snapshot(
        *openrouter_fixtures(),
        captured_at=dt.datetime(2026, 8, 18, 19, tzinfo=dt.timezone.utc),
    )

    assert snapshot["provider"] == "openrouter"
    assert snapshot["upstream_provider"] == "DeepSeek"
    assert snapshot["endpoint_tag"] == "deepseek"
    assert snapshot["resolved_model"] == "deepseek/deepseek-v4-flash-20260731"
    assert snapshot["quantization"] == "unknown"
    assert snapshot["context_length"] == 1_048_576
    assert snapshot["max_completion_tokens"] == 384_000
    assert snapshot["zdr"] is False
    assert snapshot["preset_version"] == 2
    assert snapshot["peak_hours_utc"] == [
        {"start_hour": 1, "end_hour": 4},
        {"start_hour": 6, "end_hour": 10},
    ]
    assert snapshot["rates_usd_per_million_tokens"] == {
        "off_peak": {
            "cache_hit_input": "0.007",
            "cache_miss_input": "0.22",
            "output": "0.66",
        },
        "peak": {
            "cache_hit_input": "0.014",
            "cache_miss_input": "0.44",
            "output": "1.32",
        },
    }


def test_parse_openrouter_contract_while_peak_tariff_is_active() -> None:
    endpoints_raw, zdr, preset = openrouter_fixtures()
    endpoints = json.loads(endpoints_raw)
    pricing = endpoints["data"]["endpoints"][0]["pricing"]
    pricing.update(
        {
            "prompt": "0.00000044",
            "completion": "0.00000132",
            "input_cache_read": "0.000000014",
            "overrides": [
                {
                    "utc_start": 1000,
                    "utc_end": 100,
                    "prompt": "0.00000022",
                    "completion": "0.00000066",
                    "input_cache_read": "0.000000007",
                },
                *pricing["overrides"],
                {
                    "utc_start": 400,
                    "utc_end": 600,
                    "prompt": "0.00000022",
                    "completion": "0.00000066",
                    "input_cache_read": "0.000000007",
                },
            ],
        }
    )

    snapshot = deepseek_pricing.parse_openrouter_snapshot(
        json.dumps(endpoints).encode(), zdr, preset
    )

    assert snapshot["peak_hours_utc"] == [
        {"start_hour": 1, "end_hour": 4},
        {"start_hour": 6, "end_hour": 10},
    ]
    assert snapshot["rates_usd_per_million_tokens"]["off_peak"]["output"] == "0.66"
    assert snapshot["rates_usd_per_million_tokens"]["peak"]["output"] == "1.32"


def test_parse_openrouter_fails_closed_on_preset_fallback_drift() -> None:
    endpoints, zdr, preset_raw = openrouter_fixtures()
    preset = json.loads(preset_raw)
    preset["data"]["designated_version"]["config"]["provider"]["allow_fallbacks"] = True

    with pytest.raises(RuntimeError, match="preset drifted"):
        deepseek_pricing.parse_openrouter_snapshot(
            endpoints, zdr, json.dumps(preset).encode()
        )
