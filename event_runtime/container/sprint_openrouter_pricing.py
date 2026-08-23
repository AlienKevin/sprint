"""OpenRouter benchmark accounting shared by the live circuit breakers."""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any, Callable
import urllib.parse
import urllib.request


UNDISCOUNTED_COST_BASIS = "openrouter_list_price_before_endpoint_discount"
BENCHMARK_COST_BASIS = "openrouter_list_price_with_deepseek_peak_floor"
PROVIDER_COST_BASIS = "openrouter_reported_per_request"

# Official DeepSeek peak rates, in USD per token.  The benchmark deliberately
# uses these fixed rates even when OpenRouter reports an off-peak charge.  Keep
# the exact model aliases here so a newly introduced DeepSeek model cannot be
# silently priced with the wrong schedule.
DEEPSEEK_PEAK_PRICING: dict[str, dict[str, float]] = {
    "deepseek/deepseek-v4-flash": {
        "uncached_input": 0.44 / 1_000_000,
        "cached_input": 0.014 / 1_000_000,
        "output": 1.32 / 1_000_000,
    },
    "deepseek/deepseek-v4-flash-0731": {
        "uncached_input": 0.44 / 1_000_000,
        "cached_input": 0.014 / 1_000_000,
        "output": 1.32 / 1_000_000,
    },
    "deepseek/deepseek-v4-flash-20260731": {
        "uncached_input": 0.44 / 1_000_000,
        "cached_input": 0.014 / 1_000_000,
        "output": 1.32 / 1_000_000,
    },
    "deepseek/deepseek-v4-flash-vision-exp": {
        "uncached_input": 0.44 / 1_000_000,
        "cached_input": 0.014 / 1_000_000,
        "output": 1.32 / 1_000_000,
    },
    "deepseek/deepseek-v4-pro-0813": {
        "uncached_input": 1.32 / 1_000_000,
        "cached_input": 0.044 / 1_000_000,
        "output": 3.96 / 1_000_000,
    },
    "deepseek/deepseek-v4-pro": {
        "uncached_input": 1.32 / 1_000_000,
        "cached_input": 0.044 / 1_000_000,
        "output": 3.96 / 1_000_000,
    },
    "deepseek/deepseek-v4-pro-20260813": {
        "uncached_input": 1.32 / 1_000_000,
        "cached_input": 0.044 / 1_000_000,
        "output": 3.96 / 1_000_000,
    },
}


class OpenRouterPricingError(RuntimeError):
    """Live endpoint metadata cannot produce one auditable list-price factor."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _fetch_json(
    url: str,
    *,
    authorization: str | None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    request = urllib.request.Request(url, headers=headers)
    try:
        with opener(request, timeout=10) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 - converted to fail-closed telemetry
        raise OpenRouterPricingError(
            f"cannot fetch OpenRouter pricing metadata: {url}"
        ) from exc
    if not isinstance(payload, dict):
        raise OpenRouterPricingError("OpenRouter pricing metadata is not an object")
    return payload


def _preset_contract(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    data = payload.get("data")
    designated = data.get("designated_version") if isinstance(data, dict) else None
    config = designated.get("config") if isinstance(designated, dict) else None
    if not isinstance(config, dict):
        raise OpenRouterPricingError(
            "OpenRouter preset has no designated configuration"
        )
    model = config.get("model")
    provider = config.get("provider") or {}
    only = provider.get("only") if isinstance(provider, dict) else None
    if provider.get("allow_fallbacks") is not False:
        raise OpenRouterPricingError(
            "OpenRouter preset must disable provider fallbacks"
        )
    if not isinstance(model, str) or "/" not in model:
        raise OpenRouterPricingError("OpenRouter preset has no canonical model")
    if not isinstance(only, list) or len(only) != 1 or not isinstance(only[0], str):
        raise OpenRouterPricingError(
            "OpenRouter preset must select exactly one provider"
        )
    return model, only[0]


def parse_endpoint_discount_snapshot(
    payload: dict[str, Any],
    *,
    model: str,
    provider_tag: str | None,
    captured_at: str | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    data = payload.get("data")
    rows = data.get("endpoints") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise OpenRouterPricingError(
            "OpenRouter endpoint metadata has no endpoint list"
        )
    candidates = [row for row in rows if isinstance(row, dict)]
    if provider_tag:
        candidates = [row for row in candidates if row.get("tag") == provider_tag]
    if not candidates:
        raise OpenRouterPricingError(
            "OpenRouter endpoint selection has no pricing candidate"
        )

    discounts: set[float] = set()
    selected: list[dict[str, Any]] = []
    for endpoint in candidates:
        pricing = endpoint.get("pricing")
        raw_discount = pricing.get("discount") if isinstance(pricing, dict) else None
        try:
            discount = float(raw_discount or 0.0)
        except (TypeError, ValueError) as exc:
            raise OpenRouterPricingError(
                "OpenRouter endpoint discount is invalid"
            ) from exc
        if not math.isfinite(discount) or not 0 <= discount < 1:
            raise OpenRouterPricingError(
                "OpenRouter endpoint discount is outside [0, 1)"
            )
        discounts.add(discount)
        selected.append(
            {
                "provider_name": endpoint.get("provider_name"),
                "tag": endpoint.get("tag"),
                "quantization": endpoint.get("quantization"),
                "discount_fraction": discount,
                "effective_pricing": pricing,
            }
        )
    if len(discounts) != 1:
        raise OpenRouterPricingError(
            "OpenRouter route has multiple promotion discounts; pin one provider"
        )
    discount = discounts.pop()
    peak_pricing = DEEPSEEK_PEAK_PRICING.get(model)
    return {
        "schema_version": 2,
        "captured_at": captured_at or _utc_now(),
        "source_url": source_url,
        "model": model,
        "provider_tag": provider_tag,
        "endpoints": selected,
        "discount_fraction": discount,
        "gross_up_multiplier": 1.0 / (1.0 - discount),
        "deepseek_peak_pricing_usd_per_token": peak_pricing,
        "cost_basis": (
            BENCHMARK_COST_BASIS if peak_pricing else UNDISCOUNTED_COST_BASIS
        ),
    }


def capture_endpoint_discount_snapshot(
    *,
    canonical_model: str,
    requested_model: object,
    request_provider: object,
    authorization: str | None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Resolve the request's exact endpoint and freeze its live promotion."""
    model = canonical_model
    provider_tag: str | None = None
    if isinstance(request_provider, dict):
        only = request_provider.get("only")
        if isinstance(only, list) and len(only) == 1 and isinstance(only[0], str):
            provider_tag = only[0]
    if isinstance(requested_model, str) and requested_model.startswith("@preset/"):
        slug = requested_model.removeprefix("@preset/")
        preset_url = "https://openrouter.ai/api/v1/presets/" + urllib.parse.quote(
            slug, safe=""
        )
        model, provider_tag = _preset_contract(
            _fetch_json(preset_url, authorization=authorization, opener=opener)
        )
    if not isinstance(model, str) or "/" not in model:
        raise OpenRouterPricingError("run contract has no canonical OpenRouter model")
    author, slug = model.split("/", 1)
    endpoint_url = (
        "https://openrouter.ai/api/v1/models/"
        + urllib.parse.quote(author, safe="")
        + "/"
        + urllib.parse.quote(slug, safe="")
        + "/endpoints"
    )
    return parse_endpoint_discount_snapshot(
        _fetch_json(endpoint_url, authorization=authorization, opener=opener),
        model=model,
        provider_tag=provider_tag,
        source_url=endpoint_url,
    )


def undiscounted_cost_usd(charged_cost_usd: object, snapshot: object) -> float:
    try:
        charged = float(charged_cost_usd)
        discount = float(snapshot["discount_fraction"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise OpenRouterPricingError("request has no valid promotion snapshot") from exc
    if not math.isfinite(charged) or charged < 0:
        raise OpenRouterPricingError("OpenRouter charged cost is invalid")
    if not math.isfinite(discount) or not 0 <= discount < 1:
        raise OpenRouterPricingError("request promotion discount is invalid")
    return charged / (1.0 - discount)


def _usage_token_counts(usage: object) -> tuple[float, float, float]:
    """Return uncached input, cached input, and output tokens.

    Supports both a Responses API usage object and OpenRouter's generation
    audit shape used to recover a streamed request after a proxy restart.
    """
    if not isinstance(usage, dict):
        raise OpenRouterPricingError("DeepSeek peak pricing requires token usage")

    if "native_tokens_prompt" in usage:
        input_tokens = usage.get("native_tokens_prompt")
        cached_tokens = usage.get("native_tokens_cached", 0)
        output_tokens = usage.get("native_tokens_completion")
    elif "prompt_tokens" in usage:
        input_tokens = usage.get("prompt_tokens")
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = (
            details.get("cached_tokens", 0) if isinstance(details, dict) else 0
        )
        output_tokens = usage.get("completion_tokens")
    else:
        input_tokens = usage.get("input_tokens")
        details = usage.get("input_tokens_details") or {}
        cached_tokens = (
            details.get("cached_tokens", 0) if isinstance(details, dict) else 0
        )
        output_tokens = usage.get("output_tokens")

    values: list[float] = []
    for name, raw in (
        ("input", input_tokens),
        ("cached input", cached_tokens),
        ("output", output_tokens),
    ):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise OpenRouterPricingError(
                f"DeepSeek peak pricing has no valid {name} token count"
            ) from exc
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise OpenRouterPricingError(
                f"DeepSeek peak pricing has an invalid {name} token count"
            )
        values.append(value)
    input_value, cached_value, output_value = values
    if cached_value > input_value:
        raise OpenRouterPricingError(
            "DeepSeek cached token count exceeds total input tokens"
        )
    return input_value - cached_value, cached_value, output_value


def benchmark_cost_usd(
    charged_cost_usd: object, snapshot: object, usage: object
) -> float:
    """Return the cost used by the agent, cutoff, telemetry, and scoring.

    Every route is first grossed up to its undiscounted OpenRouter endpoint
    price. DeepSeek additionally gets a fixed official-peak reconstruction from
    token usage. The greater value wins, so neither a promotion nor an
    off-peak/cheaper endpoint can increase the amount of work bought by a run.
    """
    list_cost = undiscounted_cost_usd(charged_cost_usd, snapshot)
    try:
        pricing = snapshot.get("deepseek_peak_pricing_usd_per_token")  # type: ignore[union-attr]
    except AttributeError as exc:
        raise OpenRouterPricingError("request has no valid pricing snapshot") from exc
    if pricing is None:
        return list_cost
    if not isinstance(pricing, dict):
        raise OpenRouterPricingError("DeepSeek peak pricing snapshot is invalid")
    uncached, cached, output = _usage_token_counts(usage)
    try:
        peak_cost = (
            uncached * float(pricing["uncached_input"])
            + cached * float(pricing["cached_input"])
            + output * float(pricing["output"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OpenRouterPricingError("DeepSeek peak rates are invalid") from exc
    if not math.isfinite(peak_cost) or peak_cost < 0:
        raise OpenRouterPricingError("DeepSeek peak cost is invalid")
    return max(list_cost, peak_cost)
