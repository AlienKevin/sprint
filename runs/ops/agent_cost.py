#!/usr/bin/env python3
"""Deterministic agent-cost ledger shared by live agents and the website.

Benchmark comparison cost is published API list price plus the pinned Modal
tariff applied to requested CPU, memory, and A10G allocation.  Official
verification, observability, website hosting, storage, credits, and taxes are
excluded.  Provider billing remains a separate audit and never changes this
comparison ledger retroactively.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
AGENT_ROLES = ("cpu_agent", "training_gpu")


def _role_rate(
    contract: dict[str, Any], contract_key: str, rates: dict[str, Any]
) -> float:
    spec = contract.get(contract_key) or {}
    rate = float(spec.get("physical_cpu_cores") or spec.get("vcpus") or 0) * float(
        rates["CPU"]
    ) + (float(spec.get("memory_mb") or 0) / 1024.0) * float(rates["Memory"])
    if int(spec.get("gpu_count") or 0):
        rate += int(spec["gpu_count"]) * float(rates[str(spec["gpu_type"])])
    return rate


def build_cost_ledger(timeline: dict[str, Any]) -> dict[str, Any]:
    """Build the shared time-indexed comparison ledger from one timeline."""
    resources = timeline["resource_usage_summary"]
    estimate = resources["modal_estimate"]
    rates = estimate["pricing_snapshot"]["rates_usd_per_second"]
    contract = resources["resource_contract"]
    events = timeline["events"]
    end_epoch_ms = int(timeline["clock"]["end_epoch_ms"])

    cpu_starts = [
        int(event["epoch_ms"])
        for event in events
        if event.get("kind") == "cpu_allocated"
    ]
    if not cpu_starts:
        raise RuntimeError("CPU allocation lifecycle is missing")
    starts = {
        str(event["lease_id"]): int(event["epoch_ms"])
        for event in events
        if event.get("kind") in {"gpu_allocated", "gpu_reallocated"}
        and event.get("lease_id")
    }
    ends = {
        str(event["lease_id"]): int(event["epoch_ms"])
        for event in events
        if event.get("kind") in {"gpu_released", "gpu_preempted"}
        and event.get("lease_id")
    }
    training_intervals = sorted(
        (start, max(start, ends.get(lease_id, end_epoch_ms)))
        for lease_id, start in starts.items()
    )
    return {
        "origin_epoch_ms": int(timeline["clock"]["origin_epoch_ms"]),
        "end_epoch_ms": end_epoch_ms,
        "cpu_start_epoch_ms": min(cpu_starts),
        "cpu_usd_per_second": _role_rate(contract, "cpu_agent", rates),
        "training_intervals": training_intervals,
        "training_usd_per_second": _role_rate(contract, "training_worker", rates),
        "api_events": sorted(
            (
                int(event["epoch_ms"]),
                float(event.get("calculated_cost_usd") or 0.0),
            )
            for event in events
            if event.get("kind") == "model_request_usage"
            and isinstance(event.get("calculated_cost_usd"), (int, float))
            and not isinstance(event.get("calculated_cost_usd"), bool)
        ),
    }


def cumulative_cost_components_at_epoch(
    ledger: dict[str, Any], epoch_ms: int
) -> dict[str, float]:
    """Integrate API and allocation cost through one wall-clock point."""
    cutoff = min(int(epoch_ms), int(ledger["end_epoch_ms"]))
    api_cost = sum(
        cost for event_ms, cost in ledger["api_events"] if event_ms <= cutoff
    )
    cpu_ms = max(0, cutoff - int(ledger["cpu_start_epoch_ms"]))
    training_ms = sum(
        max(0, min(cutoff, end_ms) - start_ms)
        for start_ms, end_ms in ledger["training_intervals"]
        if cutoff > start_ms
    )
    cpu_cost = cpu_ms / 1000.0 * float(ledger["cpu_usd_per_second"])
    training_cost = training_ms / 1000.0 * float(ledger["training_usd_per_second"])
    return {
        "model_api_usd": api_cost,
        "cpu_agent_usd": cpu_cost,
        "training_sandboxes_usd": training_cost,
        "total_usd": api_cost + cpu_cost + training_cost,
    }


def cumulative_cost_at_epoch(ledger: dict[str, Any], epoch_ms: int) -> float:
    return cumulative_cost_components_at_epoch(ledger, epoch_ms)["total_usd"]


def _pricing_snapshots(state_dir: Path | None) -> list[dict[str, Any]]:
    if state_dir is None:
        return []
    try:
        audit = json.loads((state_dir / "usage" / "run-usage-audit.json").read_text())
    except (OSError, json.JSONDecodeError):
        audit = {}
    snapshots = [
        item for item in audit.get("pricing_snapshots") or [] if isinstance(item, dict)
    ]
    if snapshots:
        return snapshots

    # Before the first response arrives there is no usage audit yet. Load the
    # exact pricing module from this run's frozen Harbor source rather than
    # maintaining a second mutable tariff table in Sprint.
    try:
        run = json.loads((state_dir / "run.json").read_text())
        source = (
            Path(str(run["harbor_path"])) / "src/harbor/agents/installed/codex_cost.py"
        )
        spec = importlib.util.spec_from_file_location(
            f"sprint_cost_rates_{run['run_id']}", source
        )
        if spec is None or spec.loader is None:
            return []
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        snapshot = module.pricing_snapshot_for_request(
            run.get("model"), run.get("created_at")
        )
        return [snapshot] if isinstance(snapshot, dict) else []
    except (KeyError, OSError, json.JSONDecodeError):
        return []


def _sum_api_components(events: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for event in events:
        if event.get("kind") != "model_request_usage":
            continue
        for name, value in (event.get("cost_components_usd") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[str(name)] = totals.get(str(name), 0.0) + float(value)
    return {name: round(value, 12) for name, value in sorted(totals.items())}


def build_snapshot(
    timeline: dict[str, Any], *, state_dir: Path | None = None
) -> dict[str, Any]:
    """Return the sole agent-facing JSON cost document."""
    resources = timeline["resource_usage_summary"]
    estimate = resources["modal_estimate"]
    pricing = estimate["pricing_snapshot"]
    roles = estimate["by_role"]
    usage = timeline["usage_summary"]
    events = timeline["events"]
    priced_requests = sum(
        1
        for event in events
        if event.get("kind") == "model_request_usage"
        and isinstance(event.get("calculated_cost_usd"), (int, float))
        and not isinstance(event.get("calculated_cost_usd"), bool)
    )
    request_count = int(usage.get("request_count") or 0)
    api_complete = priced_requests == request_count
    api_cost = sum(
        float(event["calculated_cost_usd"])
        for event in events
        if event.get("kind") == "model_request_usage"
        and isinstance(event.get("calculated_cost_usd"), (int, float))
        and not isinstance(event.get("calculated_cost_usd"), bool)
    )

    def role_payload(role: str) -> dict[str, Any]:
        value = roles.get(role) or {}
        return {
            "allocated_seconds": round(float(value.get("allocated_ms") or 0) / 1000, 3),
            "requested_resources": value.get("quantities") or {},
            "cost_components_usd": value.get("cost_components_usd") or {},
            "cost_usd": float(value.get("estimated_cost_usd") or 0.0),
        }

    cpu = role_payload("cpu_agent")
    training = role_payload("training_gpu")
    modal_cost_usd = cpu["cost_usd"] + training["cost_usd"]
    total = api_cost + modal_cost_usd if api_complete else None
    rates = pricing["rates_usd_per_second"]
    contract = resources["resource_contract"]
    cpu_spec = contract.get("cpu_agent") or {}
    training_spec = contract.get("training_worker") or {}
    snapshots = _pricing_snapshots(state_dir)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": timeline["run"]["run_id"],
        "model": timeline["run"].get("model"),
        "currency": "USD",
        "as_of": timeline["generated_at"],
        "as_of_epoch_ms": int(timeline["clock"]["end_epoch_ms"]),
        "status": "complete" if api_complete else "incomplete_api_usage",
        "total_usd": round(total, 12) if total is not None else None,
        "components": {
            "model_api": {
                "request_count": request_count,
                "priced_request_count": priced_requests,
                "cost_reconstruction_complete": api_complete,
                "tokens": {
                    key: int(usage.get(key) or 0)
                    for key in (
                        "ordinary_uncached_input_tokens",
                        "cached_input_tokens",
                        "cache_write_input_tokens",
                        "output_tokens",
                        "reasoning_output_tokens",
                    )
                },
                "cost_components_usd": _sum_api_components(events),
                "cost_usd": round(api_cost, 12),
                "pricing_snapshots": snapshots,
            },
            "cpu_agent": cpu,
            "training_sandboxes": training,
        },
        "equation": {
            "total": "C(t) = C_api(t) + C_cpu_agent(t) + C_training(t)",
            "modal_role": (
                "C_role(t) = allocated_seconds(t) * "
                "(physical_cpu_cores * CPU_rate + memory_gib * Memory_rate "
                "+ a10g_count * A10G_rate)"
            ),
            "model_api": (
                "C_api(t) = sum over completed requests and token classes of "
                "tokens_class * published_rate_class / 1,000,000; reasoning "
                "tokens are included in output_tokens and are not billed twice"
            ),
            "openai_long_context": (
                "when request input_tokens > 272000, multiply all input token "
                "rates by 2.0 and the output token rate by 1.5"
            ),
        },
        "constants": {
            "modal_pricing_snapshot": pricing,
            "rates_usd_per_hour": {
                "physical_cpu_core": round(float(rates["CPU"]) * 3600, 9),
                "memory_gib": round(float(rates["Memory"]) * 3600, 9),
                "a10g_gpu": round(float(rates["A10G"]) * 3600, 9),
            },
            "role_rates_usd_per_hour": {
                "cpu_agent": round(_role_rate(contract, "cpu_agent", rates) * 3600, 9),
                "training_sandbox": round(
                    _role_rate(contract, "training_worker", rates) * 3600, 9
                ),
            },
            "cpu_agent_request": cpu_spec,
            "training_sandbox_request": training_spec,
            "a10g_memory_gib": 24,
            "a10g_memory_billing": "included_in_a10g_rate_not_separately_billed",
            "api_pricing_snapshots": snapshots,
        },
        "included": [
            "model_api_published_list_price",
            "persistent_cpu_agent_requested_cpu_and_memory",
            "training_sandbox_requested_cpu_memory_and_a10g",
        ],
        "excluded": [
            "official_verifier",
            "observability_and_telemetry",
            "website_hosting",
            "volume_storage",
            "provider_credits_discounts_taxes_and_invoice_adjustments",
        ],
        "cost_basis": "published_api_list_price_plus_pinned_modal_requested_resource_tariff",
        "invoice_exact": False,
    }
