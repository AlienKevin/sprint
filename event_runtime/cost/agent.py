#!/usr/bin/env python3
"""Deterministic agent-cost ledger shared by live agents and the website.

OpenRouter runs use its per-response reported charge for the API component;
Modal components use live allocated seconds at the pinned requested-resource
tariff. Official verification, observability, website hosting, storage,
credits, and credit-purchase fees are excluded.
"""

from __future__ import annotations

import importlib.util
import json
import os
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
    # maintaining a second mutable tariff table in the event runtime.
    try:
        run = json.loads((state_dir / "run.json").read_text())
        pricing_snapshot = run.get("api_pricing_snapshot")
        if isinstance(pricing_snapshot, dict):
            os.environ["SPRINT_DEEPSEEK_PRICING_SNAPSHOT"] = json.dumps(
                pricing_snapshot, separators=(",", ":"), sort_keys=True
            )
        else:
            os.environ.pop("SPRINT_DEEPSEEK_PRICING_SNAPSHOT", None)
        source = (
            Path(str(run["harbor_path"])) / "src/harbor/agents/installed/codex_cost.py"
        )
        spec = importlib.util.spec_from_file_location(
            f"event_cost_rates_{run['run_id']}", source
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
    canonical: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None
    if state_dir is not None:
        canonical_path = state_dir / "telemetry" / "budget-watchdog.json"
        try:
            candidate = json.loads(canonical_path.read_text())
        except (OSError, json.JSONDecodeError):
            candidate = None
        if (
            isinstance(candidate, dict)
            and candidate.get("schema_version") == 2
            and candidate.get("run_id") == timeline["run"]["run_id"]
            and isinstance(candidate.get("total_usd"), (int, float))
            and not isinstance(candidate.get("total_usd"), bool)
        ):
            canonical = candidate
        previous_path = state_dir / "telemetry" / "agent-cost.json"
        try:
            candidate = json.loads(previous_path.read_text())
        except (OSError, json.JSONDecodeError):
            candidate = None
        if (
            isinstance(candidate, dict)
            and candidate.get("run_id") == timeline["run"]["run_id"]
            and isinstance(candidate.get("total_usd"), (int, float))
            and not isinstance(candidate.get("total_usd"), bool)
        ):
            previous = candidate
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
    snapshot = {
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
    if canonical is None:
        return snapshot

    # The in-sandbox watchdog sees API requests and CPU lifetime immediately,
    # but a long-lived Modal Volume mount does not see host-written GPU
    # lifecycle shards without a reload.  The host timeline has those GPU
    # allocation events.  Merge component-wise using the greater cumulative
    # value so a stale view can never lower spend, then mirror this exact
    # document back to the agent and active GPU workers.
    canonical_components = canonical.get("components") or {}
    previous_components = (previous or {}).get("components") or {}
    host_components = snapshot["components"]

    def cumulative_numeric_map(*values: Any) -> dict[str, float | int]:
        maps = [value for value in values if isinstance(value, dict)]
        keys = {str(key) for value in maps for key in value}
        result: dict[str, float | int] = {}
        for key in keys:
            candidates = [
                value.get(key)
                for value in maps
                if isinstance(value.get(key), (int, float))
                and not isinstance(value.get(key), bool)
            ]
            if candidates:
                result[key] = max(candidates)
        return result

    def merged_component(name: str) -> dict[str, Any]:
        host = dict(host_components.get(name) or {})
        prior = dict(previous_components.get(name) or {})
        remote = dict(canonical_components.get(name) or {})
        merged = {**host, **prior, **remote}
        merged["cost_usd"] = max(
            float(host.get("cost_usd") or 0.0),
            float(prior.get("cost_usd") or 0.0),
            float(remote.get("cost_usd") or 0.0),
        )
        merged["cost_components_usd"] = cumulative_numeric_map(
            host.get("cost_components_usd"),
            prior.get("cost_components_usd"),
            remote.get("cost_components_usd"),
        )
        if name != "model_api":
            merged["allocated_seconds"] = max(
                float(host.get("allocated_seconds") or 0.0),
                float(prior.get("allocated_seconds") or 0.0),
                float(remote.get("allocated_seconds") or 0.0),
            )
        else:
            merged["tokens"] = cumulative_numeric_map(
                host.get("tokens"), prior.get("tokens"), remote.get("tokens")
            )
            merged["request_count"] = max(
                int(host.get("request_count") or 0),
                int(prior.get("request_count") or 0),
                int(remote.get("request_count") or 0),
            )
            merged["priced_request_count"] = max(
                int(host.get("priced_request_count") or 0),
                int(prior.get("priced_request_count") or 0),
                int(remote.get("priced_request_count") or 0),
            )
        return merged

    components = {
        name: merged_component(name)
        for name in ("model_api", "cpu_agent", "training_sandboxes")
    }
    totals = {
        "model_api_usd": components["model_api"]["cost_usd"],
        "cpu_agent_usd": components["cpu_agent"]["cost_usd"],
        "training_sandboxes_usd": components["training_sandboxes"]["cost_usd"],
    }
    total = sum(totals.values())
    # The previous merged document is a high-water mark, not a source of
    # current metadata.  Prefer the fresh in-sandbox watchdog for status and
    # budget fields while preserving cumulative component maxima above.
    merged = {**snapshot, **(previous or {}), **canonical}
    merged.update(
        {
            "as_of": snapshot.get("as_of"),
            "as_of_epoch_ms": snapshot.get("as_of_epoch_ms"),
            "checked_at_epoch_s": float(snapshot["as_of_epoch_ms"]) / 1000.0,
            "components": components,
            "component_totals_usd": totals,
            "total_usd": total,
            "estimated_total_usd": total,
            "cpu_allocated_seconds": components["cpu_agent"][
                "allocated_seconds"
            ],
            "training_allocated_seconds": components["training_sandboxes"][
                "allocated_seconds"
            ],
            "request_count": components["model_api"]["request_count"],
            "pending_request_count": int(
                (canonical_components.get("model_api") or {}).get(
                    "pending_request_count"
                )
                or canonical.get("pending_request_count")
                or 0
            ),
            "component_snapshot_sources": {
                "model_api": "max(openrouter_watchdog,host_provider_usage)",
                "cpu_agent": "max(in_sandbox_lifecycle,host_timeline)",
                "training_sandboxes": "max(in_sandbox_lifecycle,host_timeline)",
            },
        }
    )
    budget = canonical.get("budget_usd")
    if isinstance(budget, (int, float)) and not isinstance(budget, bool):
        merged["budget_remaining_usd"] = max(0.0, float(budget) - total)
    threshold = canonical.get("stop_threshold_usd")
    if canonical.get("status") == "stop_requested" or (
        previous is not None and previous.get("status") == "stop_requested"
    ) or (
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and total >= float(threshold)
    ):
        merged["status"] = "stop_requested"
    else:
        merged["status"] = "within_budget"
    return merged
