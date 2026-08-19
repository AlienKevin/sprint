"""OpenRouter endpoint-promotion accounting shared by the live circuit breakers."""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any, Callable
import urllib.parse
import urllib.request


UNDISCOUNTED_COST_BASIS = "openrouter_list_price_before_endpoint_discount"
PROVIDER_COST_BASIS = "openrouter_reported_per_request"


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
    return {
        "schema_version": 1,
        "captured_at": captured_at or _utc_now(),
        "source_url": source_url,
        "model": model,
        "provider_tag": provider_tag,
        "endpoints": selected,
        "discount_fraction": discount,
        "gross_up_multiplier": 1.0 / (1.0 - discount),
        "cost_basis": UNDISCOUNTED_COST_BASIS,
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
