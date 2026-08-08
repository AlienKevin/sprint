"""Lossless Codex usage capture and reproducible API cost calculation."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


USAGE_AUDIT_SCHEMA_VERSION = 1

# This is deliberately pinned rather than delegated to LiteLLM. Cost reports must
# remain reproducible after a dependency updates its mutable pricing table.
# Refresh the snapshot only after checking the linked first-party model page.
GPT_5_6_TERRA_PRICING: dict[str, Any] = {
    "id": "openai-gpt-5.6-terra-default-2026-08-08",
    "provider": "openai",
    "model": "gpt-5.6-terra",
    "service_tier": "default",
    "currency": "USD",
    "captured_at": "2026-08-08",
    "source_url": "https://developers.openai.com/api/docs/models/gpt-5.6-terra",
    "unit_tokens": 1_000_000,
    "rates_usd_per_million_tokens": {
        "uncached_input": "2.00",
        "cached_input": "0.20",
        "cache_write_input": "2.50",
        "output": "12.00",
    },
    "cache_write_multiplier": "1.25",
    "long_context": {
        "threshold_input_tokens": 272_000,
        "input_multiplier": "2.0",
        "output_multiplier": "1.5",
        "scope": "full_request",
    },
}

# DeepSeek publishes cache-hit, cache-miss, and output prices. Keep the dated
# model version in the snapshot because the public API name is a moving alias.
DEEPSEEK_V4_FLASH_PRICING: dict[str, Any] = {
    "id": "deepseek-v4-flash-0731-2026-08-08",
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "model_version": "DeepSeek-V4-Flash-0731",
    "currency": "USD",
    "captured_at": "2026-08-08",
    "source_url": "https://api-docs.deepseek.com/quick_start/pricing",
    "unit_tokens": 1_000_000,
    "rates_usd_per_million_tokens": {
        "cache_hit_input": "0.0028",
        "cache_miss_input": "0.14",
        "output": "0.28",
    },
}

_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def canonical_model_name(model: str | None) -> str | None:
    """Strip an optional provider prefix from a model identifier."""
    if not model:
        return None
    return model.split("/", 1)[-1]


def _token_count(usage: dict[str, Any], name: str) -> int | None:
    value = usage.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _usd(tokens: int, rate_per_million: str, multiplier: Decimal) -> Decimal:
    return Decimal(tokens) * Decimal(rate_per_million) * multiplier / Decimal(1_000_000)


def build_request_usage_record(
    *,
    api_call_id: str,
    model: str | None,
    service_tier: str | None,
    reasoning_effort: str | None,
    usage: dict[str, Any],
    usage_reported_at: str | None,
    model_context_window: int | None,
) -> dict[str, Any]:
    """Preserve one Codex token event and calculate its published token charge.

    Codex reports cache reads and writes as subsets of ``input_tokens``. The
    ordinary uncached bucket is therefore ``input - reads - writes``. Reasoning
    tokens are a subset of output tokens and are retained for audit, not billed
    a second time.
    """
    normalized_model = canonical_model_name(model)
    counts = {name: _token_count(usage, name) for name in _USAGE_FIELDS}
    record: dict[str, Any] = {
        "api_call_id": api_call_id,
        "usage_reported_at": usage_reported_at,
        "model": normalized_model,
        "service_tier": service_tier,
        "reasoning_effort": reasoning_effort,
        "model_context_window": model_context_window,
        **counts,
        "ordinary_uncached_input_tokens": None,
        "long_context_pricing_applied": None,
        "pricing_snapshot_id": None,
        "calculated_cost_usd": None,
        "cost_components_usd": None,
        "cost_reconstruction_status": "unsupported_model",
        "incomplete_reasons": [],
    }

    supported = {
        GPT_5_6_TERRA_PRICING["model"],
        DEEPSEEK_V4_FLASH_PRICING["model"],
    }
    if normalized_model not in supported:
        return record

    if normalized_model == DEEPSEEK_V4_FLASH_PRICING["model"]:
        record["pricing_snapshot_id"] = DEEPSEEK_V4_FLASH_PRICING["id"]
        missing = [
            name
            for name in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "total_tokens",
            )
            if counts[name] is None
        ]
        reasons = (
            ["missing_or_invalid_usage_fields:" + ",".join(missing)] if missing else []
        )
        if reasons:
            record["cost_reconstruction_status"] = "incomplete"
            record["incomplete_reasons"] = reasons
            return record
        input_tokens = counts["input_tokens"]
        cached_tokens = counts["cached_input_tokens"]
        output_tokens = counts["output_tokens"]
        reasoning_tokens = counts["reasoning_output_tokens"]
        total_tokens = counts["total_tokens"]
        if any(
            value is None
            for value in (
                input_tokens,
                cached_tokens,
                output_tokens,
                reasoning_tokens,
                total_tokens,
            )
        ):
            raise RuntimeError("validated DeepSeek usage unexpectedly contains None")
        assert isinstance(input_tokens, int)
        assert isinstance(cached_tokens, int)
        assert isinstance(output_tokens, int)
        assert isinstance(reasoning_tokens, int)
        assert isinstance(total_tokens, int)
        ordinary_tokens = input_tokens - cached_tokens
        if ordinary_tokens < 0:
            reasons.append("cached_input_exceeds_input")
        if reasoning_tokens > output_tokens:
            reasons.append("reasoning_output_exceeds_output")
        if total_tokens != input_tokens + output_tokens:
            reasons.append("total_tokens_does_not_equal_input_plus_output")
        if reasons:
            record["cost_reconstruction_status"] = "incomplete"
            record["incomplete_reasons"] = reasons
            return record
        rates = DEEPSEEK_V4_FLASH_PRICING["rates_usd_per_million_tokens"]
        components = {
            "cache_miss_input": _usd(
                ordinary_tokens, rates["cache_miss_input"], Decimal("1")
            ),
            "cache_hit_input": _usd(
                cached_tokens, rates["cache_hit_input"], Decimal("1")
            ),
            "output": _usd(output_tokens, rates["output"], Decimal("1")),
        }
        cost = sum(components.values(), Decimal("0"))
        record.update(
            {
                "ordinary_uncached_input_tokens": ordinary_tokens,
                "long_context_pricing_applied": False,
                "calculated_cost_usd": float(cost),
                "cost_components_usd": {
                    name: float(value) for name, value in components.items()
                },
                "cost_reconstruction_status": "complete",
            }
        )
        return record

    record["pricing_snapshot_id"] = GPT_5_6_TERRA_PRICING["id"]
    missing = [
        name
        for name in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        )
        if counts[name] is None
    ]
    reasons: list[str] = []
    if missing:
        reasons.append("missing_or_invalid_usage_fields:" + ",".join(missing))

    if service_tier != GPT_5_6_TERRA_PRICING["service_tier"]:
        reasons.append(f"unsupported_or_unreported_service_tier:{service_tier}")

    if reasons:
        record["cost_reconstruction_status"] = "incomplete"
        record["incomplete_reasons"] = reasons
        return record

    input_tokens = counts["input_tokens"]
    cached_tokens = counts["cached_input_tokens"]
    cache_write_tokens = counts["cache_write_input_tokens"]
    output_tokens = counts["output_tokens"]
    reasoning_tokens = counts["reasoning_output_tokens"]
    total_tokens = counts["total_tokens"]
    if (
        input_tokens is None
        or cached_tokens is None
        or cache_write_tokens is None
        or output_tokens is None
        or reasoning_tokens is None
        or total_tokens is None
    ):
        raise RuntimeError("validated usage unexpectedly contains None")

    ordinary_tokens = input_tokens - cached_tokens - cache_write_tokens
    if ordinary_tokens < 0:
        reasons.append("cached_plus_cache_write_exceeds_input")
    if reasoning_tokens > output_tokens:
        reasons.append("reasoning_output_exceeds_output")
    if total_tokens != input_tokens + output_tokens:
        reasons.append("total_tokens_does_not_equal_input_plus_output")
    if reasons:
        record["cost_reconstruction_status"] = "incomplete"
        record["incomplete_reasons"] = reasons
        return record

    threshold = GPT_5_6_TERRA_PRICING["long_context"]["threshold_input_tokens"]
    long_context = input_tokens > threshold
    input_multiplier = Decimal(
        GPT_5_6_TERRA_PRICING["long_context"]["input_multiplier"]
        if long_context
        else "1"
    )
    output_multiplier = Decimal(
        GPT_5_6_TERRA_PRICING["long_context"]["output_multiplier"]
        if long_context
        else "1"
    )
    rates = GPT_5_6_TERRA_PRICING["rates_usd_per_million_tokens"]
    components = {
        "ordinary_uncached_input": _usd(
            ordinary_tokens, rates["uncached_input"], input_multiplier
        ),
        "cached_input": _usd(cached_tokens, rates["cached_input"], input_multiplier),
        "cache_write_input": _usd(
            cache_write_tokens, rates["cache_write_input"], input_multiplier
        ),
        "output": _usd(output_tokens, rates["output"], output_multiplier),
    }
    cost = sum(components.values(), Decimal("0"))
    record.update(
        {
            "ordinary_uncached_input_tokens": ordinary_tokens,
            "long_context_pricing_applied": long_context,
            "calculated_cost_usd": float(cost),
            "cost_components_usd": {
                name: float(value) for name, value in components.items()
            },
            "cost_reconstruction_status": "complete",
        }
    )
    return record


def build_usage_audit(
    *,
    session_id: str,
    records: list[dict[str, Any]],
    total_usage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reconcile per-request usage against Codex's cumulative counters."""
    summed: dict[str, int] = {}
    for field in _USAGE_FIELDS:
        values = [record.get(field) for record in records]
        if all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            summed[field] = sum(values)

    reported = {
        field: _token_count(total_usage or {}, field) for field in _USAGE_FIELDS
    }
    mismatches = {
        field: {"summed_requests": summed.get(field), "reported_total": reported[field]}
        for field in _USAGE_FIELDS
        if summed.get(field) != reported[field]
    }
    incomplete_requests = [
        record["api_call_id"]
        for record in records
        if record.get("cost_reconstruction_status") == "incomplete"
    ]
    supported_records = [
        record
        for record in records
        if record.get("cost_reconstruction_status") != "unsupported_model"
    ]
    complete = bool(supported_records) and not mismatches and not incomplete_requests
    calculated_cost = (
        sum(float(record["calculated_cost_usd"]) for record in supported_records)
        if complete
        else None
    )
    snapshot_ids = sorted(
        {
            str(record["pricing_snapshot_id"])
            for record in supported_records
            if record.get("pricing_snapshot_id")
        }
    )
    snapshots = [
        snapshot
        for snapshot_id in snapshot_ids
        for snapshot in (GPT_5_6_TERRA_PRICING, DEEPSEEK_V4_FLASH_PRICING)
        if snapshot_id == snapshot["id"]
    ]
    return {
        "schema_version": USAGE_AUDIT_SCHEMA_VERSION,
        "session_id": session_id,
        "source": "codex_session_jsonl_token_count",
        "request_count": len(records),
        "requests": records,
        "reported_total_usage": reported,
        "summed_request_usage": summed,
        "reconciliation_mismatches": mismatches,
        "incomplete_request_ids": incomplete_requests,
        "pricing_snapshots": snapshots,
        "cost_reconstruction_complete": complete,
        "calculated_api_usage_usd": calculated_cost,
    }
