from typing import Any

import pytest

from harbor.agents.installed.codex_cost import (
    DEEPSEEK_V4_FLASH_PRICING,
    GPT_5_6_LUNA_PRICING,
    GPT_5_6_TERRA_PRICING,
    build_request_usage_record,
    build_usage_audit,
)


def usage(
    *,
    ordinary: int,
    cached: int,
    cache_write: int,
    output: int,
    reasoning: int = 0,
) -> dict[str, int]:
    input_tokens = ordinary + cached + cache_write
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "output_tokens": output,
        "reasoning_output_tokens": reasoning,
        "total_tokens": input_tokens + output,
    }


def record(
    raw_usage: dict[str, int], *, api_call_id: str = "api_call_1"
) -> dict[str, Any]:
    return build_request_usage_record(
        api_call_id=api_call_id,
        model="openai/gpt-5.6-terra",
        service_tier="default",
        reasoning_effort="high",
        usage=raw_usage,
        usage_reported_at="2026-08-08T00:00:00Z",
        model_context_window=1_050_000,
    )


def test_terra_cost_separates_cache_reads_writes_and_ordinary_input() -> None:
    result = record(
        usage(ordinary=60_000, cached=40_000, cache_write=30_000, output=10_000)
    )

    assert result["ordinary_uncached_input_tokens"] == 60_000
    assert result["cache_write_input_tokens"] == 30_000
    assert result["long_context_pricing_applied"] is False
    assert result["cost_reconstruction_status"] == "complete"
    # 60k*$2/M + 40k*$0.20/M + 30k*$2.50/M + 10k*$12/M
    assert result["calculated_cost_usd"] == pytest.approx(0.323)


def test_long_context_multiplier_is_applied_per_request() -> None:
    short = record(
        usage(ordinary=270_000, cached=0, cache_write=0, output=10_000),
        api_call_id="api_call_1",
    )
    long = record(
        usage(ordinary=273_000, cached=0, cache_write=0, output=10_000),
        api_call_id="api_call_2",
    )

    assert short["long_context_pricing_applied"] is False
    assert short["calculated_cost_usd"] == pytest.approx(0.66)
    assert long["long_context_pricing_applied"] is True
    assert long["calculated_cost_usd"] == pytest.approx(1.272)


def test_luna_cost_uses_pinned_standard_tariff_and_cache_write_rate() -> None:
    result = build_request_usage_record(
        api_call_id="api_call_luna",
        model="openai/gpt-5.6-luna",
        service_tier="default",
        reasoning_effort="max",
        usage=usage(
            ordinary=60_000,
            cached=40_000,
            cache_write=30_000,
            output=10_000,
        ),
        usage_reported_at="2026-08-08T00:00:00Z",
        model_context_window=1_050_000,
    )

    assert result["pricing_snapshot_id"] == GPT_5_6_LUNA_PRICING["id"]
    assert result["cost_reconstruction_status"] == "complete"
    # 60k*$1/M + 40k*$0.10/M + 30k*$1.25/M + 10k*$6/M
    assert result["calculated_cost_usd"] == pytest.approx(0.1615)


def test_luna_long_context_multiplier_applies_to_entire_request() -> None:
    result = build_request_usage_record(
        api_call_id="api_call_luna_long",
        model="gpt-5.6-luna",
        service_tier="default",
        reasoning_effort="max",
        usage=usage(
            ordinary=273_000,
            cached=0,
            cache_write=0,
            output=10_000,
        ),
        usage_reported_at="2026-08-08T00:00:00Z",
        model_context_window=1_050_000,
    )

    assert result["long_context_pricing_applied"] is True
    assert result["calculated_cost_usd"] == pytest.approx(0.636)


def test_invalid_usage_fails_closed_instead_of_guessing() -> None:
    raw = usage(ordinary=1, cached=10, cache_write=20, output=5)
    raw["input_tokens"] = 25

    result = record(raw)

    assert result["cost_reconstruction_status"] == "incomplete"
    assert result["calculated_cost_usd"] is None
    assert "cached_plus_cache_write_exceeds_input" in result["incomplete_reasons"]


def test_non_default_service_tier_is_not_mispriced() -> None:
    result = build_request_usage_record(
        api_call_id="api_call_1",
        model="gpt-5.6-terra",
        service_tier="priority",
        reasoning_effort="max",
        usage=usage(ordinary=10, cached=0, cache_write=0, output=2),
        usage_reported_at=None,
        model_context_window=None,
    )

    assert result["cost_reconstruction_status"] == "incomplete"
    assert result["calculated_cost_usd"] is None
    assert result["incomplete_reasons"] == [
        "unsupported_or_unreported_service_tier:priority"
    ]


def test_deepseek_cost_separates_cache_hits_and_misses() -> None:
    result = build_request_usage_record(
        api_call_id="api_call_1",
        model="deepseek/deepseek-v4-flash",
        service_tier=None,
        reasoning_effort="max",
        usage=usage(
            ordinary=1_000_000,
            cached=2_000_000,
            cache_write=0,
            output=500_000,
            reasoning=300_000,
        ),
        usage_reported_at="2026-08-08T00:00:00Z",
        model_context_window=1_000_000,
    )

    assert result["ordinary_uncached_input_tokens"] == 1_000_000
    assert result["pricing_snapshot_id"] == DEEPSEEK_V4_FLASH_PRICING["id"]
    assert result["cost_reconstruction_status"] == "complete"
    # 1M miss*$0.14/M + 2M hit*$0.0028/M + 0.5M output*$0.28/M
    assert result["calculated_cost_usd"] == pytest.approx(0.2856)


def test_usage_audit_reconciles_every_billable_bucket() -> None:
    first_usage = usage(
        ordinary=100, cached=20, cache_write=30, output=40, reasoning=10
    )
    second_usage = usage(ordinary=50, cached=70, cache_write=10, output=20, reasoning=5)
    records = [
        record(first_usage, api_call_id="api_call_1"),
        record(second_usage, api_call_id="api_call_2"),
    ]
    total = {field: first_usage[field] + second_usage[field] for field in first_usage}

    audit = build_usage_audit(
        session_id="session-1", records=records, total_usage=total
    )

    assert audit["cost_reconstruction_complete"] is True
    assert audit["reconciliation_mismatches"] == {}
    assert audit["reported_total_usage"]["cache_write_input_tokens"] == 40
    assert audit["pricing_snapshots"] == [GPT_5_6_TERRA_PRICING]
    assert audit["calculated_api_usage_usd"] == pytest.approx(
        sum(item["calculated_cost_usd"] for item in records)
    )


def test_usage_audit_rejects_replayed_or_dropped_totals() -> None:
    raw = usage(ordinary=100, cached=20, cache_write=30, output=40)
    audit = build_usage_audit(
        session_id="session-1",
        records=[record(raw)],
        total_usage={
            **raw,
            "output_tokens": 41,
            "total_tokens": raw["total_tokens"] + 1,
        },
    )

    assert audit["cost_reconstruction_complete"] is False
    assert set(audit["reconciliation_mismatches"]) == {
        "output_tokens",
        "total_tokens",
    }
    assert audit["calculated_api_usage_usd"] is None
