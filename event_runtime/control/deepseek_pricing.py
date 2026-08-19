"""Fetch reproducible DeepSeek V4 Flash pricing snapshots."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
import hashlib
import json
import re
import urllib.request
from typing import Any


PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing"
ANNOUNCEMENT_URL = "https://api-docs.deepseek.com/news/news260813"
OPENROUTER_MODEL = "deepseek/deepseek-v4-flash-0731"
OPENROUTER_RESOLVED_MODEL = "deepseek/deepseek-v4-flash-20260731"
OPENROUTER_PROVIDER_TAG = "deepseek"
OPENROUTER_PRESET_SLUG = "sprint-deepseek-v4-flash-0731-official"
OPENROUTER_ENDPOINTS_URL = (
    "https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints"
)
OPENROUTER_ZDR_URL = "https://openrouter.ai/api/v1/endpoints/zdr"
OPENROUTER_PRESET_URL = "https://openrouter.ai/api/v1/presets/" + OPENROUTER_PRESET_SLUG


def _fetch(url: str, *, api_key: str | None = None) -> bytes:
    headers = {"Accept": "application/json", "User-Agent": "sprint-benchmark-pricing/1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _one(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match is None:
        raise RuntimeError(f"DeepSeek pricing page is missing {label}")
    return match.group(1)


def _rates(text: str, bucket: str) -> dict[str, str]:
    section = _one(
        rf"(1M {bucket}.*?)(?=1M (?:INPUT|OUTPUT) TOKENS|Concurrency Limit)",
        text,
        bucket.lower(),
    )
    off_peak = _one(r"<td>OFF-PEAK</td><td>\$([0-9.]+)", section, "off-peak rate")
    peak = _one(r"<td>PEAK</td><td>\$([0-9.]+)", section, "peak rate")
    return {"off_peak": off_peak, "peak": peak}


def parse_snapshot(
    pricing_html: bytes,
    announcement_html: bytes,
    *,
    captured_at: dt.datetime | None = None,
) -> dict[str, Any]:
    pricing = pricing_html.decode("utf-8", errors="strict")
    announcement = announcement_html.decode("utf-8", errors="strict")
    version = _one(
        r"MODEL VERSION</td><td>(DeepSeek-V4-Flash-[0-9]+)",
        pricing,
        "V4 Flash model version",
    )
    cache_hit = _rates(pricing, r"INPUT TOKENS \(CACHE HIT\)")
    cache_miss = _rates(pricing, r"INPUT TOKENS \(CACHE MISS\)")
    output = _rates(pricing, "OUTPUT")
    hours = re.search(
        r"Peak hours are\s+(\d{2}):00\s*-\s*(\d{2}):00\s+and\s+"
        r"(\d{2}):00\s*-\s*(\d{2}):00\s+UTC",
        pricing,
        re.IGNORECASE,
    )
    if hours is None:
        raise RuntimeError("DeepSeek pricing page is missing its UTC peak schedule")
    effective = re.search(
        r"takes effect at\s+(\d{2}):00 UTC,\s+([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})",
        announcement,
        re.IGNORECASE,
    )
    if effective is None:
        raise RuntimeError("DeepSeek announcement is missing the tariff effective time")
    hour, month, day, year = effective.groups()
    effective_text = f"{year}-{month}-{day} {hour}:00"
    for month_format in ("%B", "%b"):
        try:
            effective_at = dt.datetime.strptime(
                effective_text, f"%Y-{month_format}-%d %H:%M"
            ).replace(tzinfo=dt.timezone.utc)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("DeepSeek announcement has an invalid effective month")
    observed = captured_at or dt.datetime.now(dt.timezone.utc)
    if observed.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    source_hash = hashlib.sha256(pricing_html + b"\0" + announcement_html).hexdigest()
    return {
        "schema_version": 1,
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "model_version": version,
        "currency": "USD",
        "unit_tokens": 1_000_000,
        "captured_at": observed.astimezone(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "effective_from": effective_at.isoformat().replace("+00:00", "Z"),
        "source_url": PRICING_URL,
        "announcement_url": ANNOUNCEMENT_URL,
        "source_sha256": source_hash,
        "peak_hours_utc": [
            {"start_hour": int(hours.group(1)), "end_hour": int(hours.group(2))},
            {"start_hour": int(hours.group(3)), "end_hour": int(hours.group(4))},
        ],
        "rates_usd_per_million_tokens": {
            "off_peak": {
                "cache_hit_input": cache_hit["off_peak"],
                "cache_miss_input": cache_miss["off_peak"],
                "output": output["off_peak"],
            },
            "peak": {
                "cache_hit_input": cache_hit["peak"],
                "cache_miss_input": cache_miss["peak"],
                "output": output["peak"],
            },
        },
    }


def fetch_snapshot() -> dict[str, Any]:
    """Fetch both official pages and return one validated launch-time snapshot."""
    return parse_snapshot(_fetch(PRICING_URL), _fetch(ANNOUNCEMENT_URL))


def _per_million(raw: object, label: str) -> str:
    try:
        value = Decimal(str(raw)) * Decimal(1_000_000)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"OpenRouter endpoint has invalid {label} pricing") from exc
    if value <= 0:
        raise RuntimeError(f"OpenRouter endpoint has non-positive {label} pricing")
    return format(value.normalize(), "f")


def _openrouter_rates(pricing: dict[str, Any]) -> dict[str, str]:
    return {
        "cache_hit_input": _per_million(
            pricing.get("input_cache_read"), "cached input"
        ),
        "cache_miss_input": _per_million(pricing.get("prompt"), "input"),
        "output": _per_million(pricing.get("completion"), "output"),
    }


def _utc_hour(raw: object, label: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"OpenRouter endpoint has invalid {label}") from exc
    hour, minute = divmod(value, 100)
    if minute != 0 or not 0 <= hour <= 24:
        raise RuntimeError(f"OpenRouter endpoint has invalid {label}")
    return hour


def parse_openrouter_snapshot(
    endpoints_json: bytes,
    zdr_json: bytes,
    preset_json: bytes,
    *,
    captured_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Validate and freeze the controlled OpenRouter DeepSeek endpoint."""
    try:
        endpoints = json.loads(endpoints_json)
        zdr = json.loads(zdr_json)
        rows = endpoints["data"]["endpoints"]
        zdr_rows = zdr["data"]
        preset = json.loads(preset_json)["data"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "OpenRouter endpoint metadata response is malformed"
        ) from exc
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("tag") == OPENROUTER_PROVIDER_TAG
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "OpenRouter official DeepSeek endpoint is missing or ambiguous"
        )
    endpoint = matches[0]
    if endpoint.get("provider_name") != "DeepSeek":
        raise RuntimeError("OpenRouter endpoint is not operated by DeepSeek")
    if endpoint.get("model_id") != OPENROUTER_MODEL:
        raise RuntimeError("OpenRouter endpoint resolved to the wrong model revision")
    if endpoint.get("name") != f"DeepSeek | {OPENROUTER_RESOLVED_MODEL}":
        raise RuntimeError("OpenRouter endpoint concrete model revision changed")
    if endpoint.get("context_length") != 1_048_576:
        raise RuntimeError("OpenRouter DeepSeek endpoint context length changed")
    if endpoint.get("max_completion_tokens") != 384_000:
        raise RuntimeError("OpenRouter DeepSeek maximum output length changed")
    supported = set(endpoint.get("supported_parameters") or [])
    required = {"reasoning", "reasoning_effort", "tools", "temperature", "top_p"}
    if not required.issubset(supported):
        missing = ", ".join(sorted(required - supported))
        raise RuntimeError(f"OpenRouter DeepSeek endpoint lost parameters: {missing}")
    is_zdr = any(
        isinstance(row, dict)
        and row.get("model_id") == OPENROUTER_MODEL
        and row.get("tag") == OPENROUTER_PROVIDER_TAG
        for row in zdr_rows
    )
    if preset.get("slug") != OPENROUTER_PRESET_SLUG or preset.get("status") != "active":
        raise RuntimeError("OpenRouter controlled preset is missing or inactive")
    designated = preset.get("designated_version")
    if not isinstance(designated, dict):
        raise RuntimeError("OpenRouter controlled preset has no designated version")
    config = designated.get("config")
    provider = config.get("provider") if isinstance(config, dict) else None
    if not isinstance(config, dict) or not isinstance(provider, dict):
        raise RuntimeError("OpenRouter controlled preset configuration is malformed")
    expected_provider = {
        "only": [OPENROUTER_PROVIDER_TAG],
        "allow_fallbacks": False,
        "require_parameters": False,
        "data_collection": "allow",
        "zdr": False,
    }
    if (
        config.get("model") != OPENROUTER_MODEL
        or Decimal(str(config.get("temperature"))) != Decimal("1")
        or Decimal(str(config.get("top_p"))) != Decimal("0.95")
        or any(provider.get(key) != value for key, value in expected_provider.items())
    ):
        raise RuntimeError(
            "OpenRouter controlled preset drifted from the audited config"
        )
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, dict):
        raise RuntimeError("OpenRouter DeepSeek endpoint has no pricing object")
    off_peak_rates = _openrouter_rates(pricing)
    overrides = pricing.get("overrides")
    if not isinstance(overrides, list) or not overrides:
        raise RuntimeError("OpenRouter DeepSeek endpoint lost its pricing schedule")
    override_rows: list[tuple[int, int, dict[str, str]]] = []
    for index, override in enumerate(overrides):
        if not isinstance(override, dict):
            raise RuntimeError("OpenRouter DeepSeek pricing override is malformed")
        override_rows.append(
            (
                _utc_hour(override.get("utc_start"), f"override {index} start"),
                _utc_hour(override.get("utc_end"), f"override {index} end"),
                _openrouter_rates(override),
            )
        )
    distinct_rates = {
        tuple(sorted(rates.items()))
        for _start, _end, rates in override_rows
        if rates != off_peak_rates
    }
    if len(distinct_rates) != 1:
        raise RuntimeError("OpenRouter DeepSeek peak tariff is missing or ambiguous")
    peak_rates = dict(next(iter(distinct_rates)))
    if any(
        Decimal(peak_rates[key]) < Decimal(off_peak_rates[key]) for key in peak_rates
    ):
        raise RuntimeError(
            "OpenRouter DeepSeek peak tariff is below its off-peak tariff"
        )
    peak_hours = [
        {"start_hour": start, "end_hour": end}
        for start, end, rates in override_rows
        if rates == peak_rates
    ]
    if not peak_hours or any(
        window["start_hour"] >= window["end_hour"] for window in peak_hours
    ):
        raise RuntimeError("OpenRouter DeepSeek peak schedule is invalid")
    observed = captured_at or dt.datetime.now(dt.timezone.utc)
    if observed.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    source_hash = hashlib.sha256(
        endpoints_json + b"\0" + zdr_json + b"\0" + preset_json
    ).hexdigest()
    return {
        "schema_version": 2,
        "provider": "openrouter",
        "upstream_provider": "DeepSeek",
        "endpoint_tag": OPENROUTER_PROVIDER_TAG,
        "model": "deepseek-v4-flash",
        "openrouter_model": OPENROUTER_MODEL,
        "resolved_model": OPENROUTER_RESOLVED_MODEL,
        "model_version": "DeepSeek-V4-Flash-0731",
        "quantization": str(endpoint.get("quantization") or "unknown"),
        "context_length": 1_048_576,
        "max_completion_tokens": 384_000,
        "zdr": is_zdr,
        "preset_slug": OPENROUTER_PRESET_SLUG,
        "preset_id": str(preset["id"]),
        "preset_version": int(designated["version"]),
        "currency": "USD",
        "unit_tokens": 1_000_000,
        "captured_at": observed.astimezone(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_url": OPENROUTER_ENDPOINTS_URL,
        "zdr_source_url": OPENROUTER_ZDR_URL,
        "preset_source_url": OPENROUTER_PRESET_URL,
        "source_sha256": source_hash,
        "peak_hours_utc": peak_hours,
        "rates_usd_per_million_tokens": {
            "off_peak": off_peak_rates,
            "peak": peak_rates,
        },
    }


def fetch_openrouter_snapshot(api_key: str) -> dict[str, Any]:
    """Fetch and validate the live official DeepSeek route and tariff."""
    if not api_key:
        raise ValueError(
            "OpenRouter API key is required to audit the controlled preset"
        )
    return parse_openrouter_snapshot(
        _fetch(OPENROUTER_ENDPOINTS_URL),
        _fetch(OPENROUTER_ZDR_URL),
        _fetch(OPENROUTER_PRESET_URL, api_key=api_key),
    )
