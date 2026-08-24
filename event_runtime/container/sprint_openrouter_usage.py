"""Canonical OpenRouter token accounting shared by proxy and controller."""

from __future__ import annotations

import math
from typing import Any, Mapping


TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "ordinary_uncached_input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def empty_token_usage() -> dict[str, int]:
    return {field: 0 for field in TOKEN_USAGE_FIELDS}


def _token_count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid OpenRouter {field}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError(f"invalid OpenRouter {field}")
    return int(numeric)


def normalize_token_usage(usage: object) -> dict[str, int]:
    """Normalize Responses or Chat Completions usage into one exact rollup."""
    if not isinstance(usage, Mapping):
        raise ValueError("OpenRouter response did not include usage")
    if "prompt_tokens" in usage:
        input_key = "prompt_tokens"
        output_key = "completion_tokens"
        input_details_key = "prompt_tokens_details"
        output_details_key = "completion_tokens_details"
    else:
        input_key = "input_tokens"
        output_key = "output_tokens"
        input_details_key = "input_tokens_details"
        output_details_key = "output_tokens_details"

    input_tokens = _token_count(usage.get(input_key, 0), field=input_key)
    output_tokens = _token_count(usage.get(output_key, 0), field=output_key)
    input_details = usage.get(input_details_key) or {}
    output_details = usage.get(output_details_key) or {}
    if not isinstance(input_details, Mapping) or not isinstance(
        output_details, Mapping
    ):
        raise ValueError("invalid OpenRouter token details")
    cached_tokens = _token_count(
        input_details.get("cached_tokens", 0), field="cached_tokens"
    )
    cache_write_tokens = _token_count(
        input_details.get("cache_write_tokens", 0), field="cache_write_tokens"
    )
    reasoning_tokens = _token_count(
        output_details.get("reasoning_tokens", 0), field="reasoning_tokens"
    )
    if cached_tokens + cache_write_tokens > input_tokens:
        raise ValueError("OpenRouter cache tokens exceed input tokens")
    if reasoning_tokens > output_tokens:
        raise ValueError("OpenRouter reasoning tokens exceed output tokens")
    return {
        "input_tokens": input_tokens,
        "ordinary_uncached_input_tokens": (
            input_tokens - cached_tokens - cache_write_tokens
        ),
        "cached_input_tokens": cached_tokens,
        "cache_write_input_tokens": cache_write_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        # The raw provider record retains any provider-reported total. The
        # canonical rollup derives it so the invariant is exact across routes.
        "total_tokens": input_tokens + output_tokens,
    }


def validate_token_usage_totals(usage: object) -> dict[str, int]:
    if not isinstance(usage, Mapping):
        raise ValueError("OpenRouter summary has no token usage")
    totals = {
        field: _token_count(usage.get(field, 0), field=field)
        for field in TOKEN_USAGE_FIELDS
    }
    if (
        totals["ordinary_uncached_input_tokens"]
        + totals["cached_input_tokens"]
        + totals["cache_write_input_tokens"]
        != totals["input_tokens"]
        or totals["reasoning_output_tokens"] > totals["output_tokens"]
        or totals["total_tokens"]
        != totals["input_tokens"] + totals["output_tokens"]
    ):
        raise ValueError("inconsistent OpenRouter summary token usage")
    return totals


def add_token_usage(
    totals: dict[str, int], usage: object, *, normalized: bool = False
) -> None:
    addition = (
        validate_token_usage_totals(usage)
        if normalized
        else normalize_token_usage(usage)
    )
    for field in TOKEN_USAGE_FIELDS:
        totals[field] += addition[field]


def generation_usage_payload(generation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert OpenRouter's generation audit fields to Responses usage."""
    input_tokens = _token_count(
        generation.get("native_tokens_prompt", 0), field="native_tokens_prompt"
    )
    cached_tokens = _token_count(
        generation.get("native_tokens_cached", 0), field="native_tokens_cached"
    )
    output_tokens = _token_count(
        generation.get("native_tokens_completion", 0),
        field="native_tokens_completion",
    )
    if cached_tokens > input_tokens:
        raise ValueError("OpenRouter cached tokens exceed prompt tokens")
    payload: dict[str, Any] = {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": cached_tokens,
            "cache_write_tokens": 0,
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }
    total_cost = generation.get("total_cost")
    if isinstance(total_cost, (int, float)) and not isinstance(total_cost, bool):
        payload["cost"] = float(total_cost)
        payload["cost_details"] = {
            "upstream_inference_cost": float(total_cost)
        }
    return payload
