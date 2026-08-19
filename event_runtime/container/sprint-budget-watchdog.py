#!/usr/bin/env python3
"""In-sandbox, fail-closed enforcement of the per-run agent-cost budget.

The controller also reconstructs the authoritative cost ledger.  This smaller
ledger deliberately runs in the CPU sandbox so a controller outage cannot
remove the circuit breaker. OpenRouter runs consume the proxy's exact
per-response charge; other supported routes use the pinned pricing module.
Modal uses the pinned tariff. One durable stop marker is visible to every
sandbox.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


CPU_USD_PER_SECOND = 2 * 0.00003942 + 8 * 0.00000667
TRAINING_USD_PER_SECOND = 6 * 0.00003942 + 12 * 0.00000667 + 0.000306
STOP_REASON = "agent_cost_budget_exhausted"


class BudgetTelemetryError(RuntimeError):
    """A condition that makes safe live cost reconstruction impossible."""


def valid_ledger_request_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def recover_openrouter_generation(
    generation_id: str, api_key: str
) -> dict[str, Any] | None:
    url = "https://openrouter.ai/api/v1/generation?" + urllib.parse.urlencode(
        {"id": generation_id}
    )
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, dict) else None


def openrouter_api_cost(
    run_root: Path, *, run_id: str, api_key: str | None
) -> tuple[float, int, int]:
    """Sum exact OpenRouter charges, recovering interrupted streams by ID."""
    requests_dir = run_root / "api-usage" / "requests"
    summary_path = requests_dir.parent / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
            if summary.get("schema_version") != 1 or summary.get("run_id") != run_id:
                raise BudgetTelemetryError("OpenRouter ledger summary identity mismatch")
            total = float(summary["model_api_usd"])
            completed = int(summary["completed_request_count"])
            pending = int(summary["pending_request_count"])
            in_flight = int(summary["in_flight_request_count"])
            recovery_required = int(summary["cost_recovery_required_count"])
            in_flight_ids_raw = summary.get("in_flight_request_ids")
            recovery_ids_raw = summary.get("cost_recovery_required_request_ids")
            has_pending_ids = isinstance(in_flight_ids_raw, list) and isinstance(
                recovery_ids_raw, list
            )
            if has_pending_ids:
                if not all(
                    valid_ledger_request_id(item)
                    for item in in_flight_ids_raw + recovery_ids_raw
                ):
                    raise BudgetTelemetryError(
                        "invalid OpenRouter pending request identity"
                    )
                in_flight_ids = set(in_flight_ids_raw)
                recovery_ids = set(recovery_ids_raw)
            else:
                in_flight_ids = set()
                recovery_ids = set()
            if not (
                math.isfinite(total)
                and total >= 0
                and completed >= 0
                and pending >= 0
                and in_flight >= 0
                and recovery_required >= 0
                and pending == in_flight + recovery_required
                and (
                    not has_pending_ids
                    or (
                        len(in_flight_ids) == in_flight
                        and len(recovery_ids) == recovery_required
                        and in_flight_ids.isdisjoint(recovery_ids)
                    )
                )
            ):
                raise BudgetTelemetryError("invalid OpenRouter ledger summary")
            if recovery_required == 0 or not api_key:
                if recovery_required and not api_key:
                    raise BudgetTelemetryError(
                        "OpenRouter charge recovery requires controller credentials"
                    )
                return total, completed, pending
            if has_pending_ids:
                remaining_recovery_ids: set[str] = set()
                for request_id in sorted(recovery_ids):
                    path = requests_dir / f"{request_id}.json"
                    try:
                        record = json.loads(path.read_text())
                    except (OSError, json.JSONDecodeError) as exc:
                        raise BudgetTelemetryError(
                            f"invalid OpenRouter ledger record: {path}"
                        ) from exc
                    if (
                        not isinstance(record, dict)
                        or record.get("run_id") != run_id
                        or record.get("ledger_request_id") != request_id
                    ):
                        raise BudgetTelemetryError(
                            f"OpenRouter ledger identity mismatch: {path}"
                        )
                    cost = record.get("provider_reported_cost_usd")
                    if (
                        isinstance(cost, (int, float))
                        and not isinstance(cost, bool)
                        and math.isfinite(float(cost))
                        and float(cost) >= 0
                    ):
                        recovered_cost = float(cost)
                    else:
                        generation_id = record.get("generation_id")
                        recovered = (
                            recover_openrouter_generation(str(generation_id), api_key)
                            if generation_id
                            and record.get("state") == "cost_recovery_required"
                            else None
                        )
                        candidate = recovered.get("total_cost") if recovered else None
                        if not (
                            isinstance(candidate, (int, float))
                            and not isinstance(candidate, bool)
                            and math.isfinite(float(candidate))
                            and float(candidate) >= 0
                        ):
                            remaining_recovery_ids.add(request_id)
                            continue
                        recovered_cost = float(candidate)
                        record.update(
                            {
                                "state": "recovered_complete",
                                "completed_at": dt.datetime.now(dt.timezone.utc)
                                .isoformat()
                                .replace("+00:00", "Z"),
                                "provider_reported_cost_usd": recovered_cost,
                                "generation_audit": recovered,
                            }
                        )
                        atomic_json(path, record)
                    total += recovered_cost
                    completed += 1
                pending = len(in_flight_ids) + len(remaining_recovery_ids)
                atomic_json(
                    summary_path,
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "updated_at": dt.datetime.now(dt.timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "model_api_usd": total,
                        "completed_request_count": completed,
                        "pending_request_count": pending,
                        "in_flight_request_count": len(in_flight_ids),
                        "cost_recovery_required_count": len(
                            remaining_recovery_ids
                        ),
                        "in_flight_request_ids": sorted(in_flight_ids),
                        "cost_recovery_required_request_ids": sorted(
                            remaining_recovery_ids
                        ),
                    },
                )
                return total, completed, pending
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise BudgetTelemetryError("invalid OpenRouter ledger summary") from exc

    total = 0.0
    completed = 0
    pending = 0
    in_flight = 0
    recovery_required = 0
    in_flight_ids: set[str] = set()
    recovery_ids: set[str] = set()
    for path in sorted(requests_dir.glob("*.json")) if requests_dir.is_dir() else ():
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BudgetTelemetryError(
                f"invalid OpenRouter ledger record: {path}"
            ) from exc
        if not isinstance(record, dict) or record.get("run_id") != run_id:
            raise BudgetTelemetryError(f"OpenRouter ledger identity mismatch: {path}")
        cost = record.get("provider_reported_cost_usd")
        if (
            isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(float(cost))
            and float(cost) >= 0
        ):
            total += float(cost)
            completed += 1
            continue
        generation_id = record.get("generation_id")
        state = record.get("state")
        if state == "cost_recovery_required" and not api_key:
            raise BudgetTelemetryError(
                "OpenRouter charge recovery requires controller credentials"
            )
        recovered = (
            recover_openrouter_generation(str(generation_id), api_key)
            if generation_id and state == "cost_recovery_required" and api_key
            else None
        )
        recovered_cost = recovered.get("total_cost") if recovered else None
        if (
            isinstance(recovered_cost, (int, float))
            and not isinstance(recovered_cost, bool)
            and math.isfinite(float(recovered_cost))
            and float(recovered_cost) >= 0
        ):
            record.update(
                {
                    "state": "recovered_complete",
                    "completed_at": dt.datetime.now(dt.timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "provider_reported_cost_usd": float(recovered_cost),
                    "generation_audit": recovered,
                }
            )
            atomic_json(path, record)
            total += float(recovered_cost)
            completed += 1
            continue
        if state not in {"in_flight", "cost_recovery_required"}:
            raise BudgetTelemetryError(f"invalid OpenRouter ledger state: {path}")
        pending += 1
        request_id = record.get("ledger_request_id") or path.stem
        if not valid_ledger_request_id(request_id):
            raise BudgetTelemetryError(
                f"invalid OpenRouter pending request identity: {path}"
            )
        if state == "in_flight":
            in_flight += 1
            in_flight_ids.add(request_id)
        else:
            recovery_required += 1
            recovery_ids.add(request_id)
    atomic_json(
        summary_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "updated_at": dt.datetime.now(dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "model_api_usd": total,
            "completed_request_count": completed,
            "pending_request_count": pending,
            "in_flight_request_count": in_flight,
            "cost_recovery_required_count": recovery_required,
            "in_flight_request_ids": sorted(in_flight_ids),
            "cost_recovery_required_request_ids": sorted(recovery_ids),
        },
    )
    return total, completed, pending


def require_live_openrouter_proxy(runtime_dir: Path) -> None:
    process_path = runtime_dir / "sprint-agent" / "codex-process"
    if not process_path.is_file():
        return
    proxy_path = runtime_dir / "sprint-agent" / "openrouter-proxy.pid"
    try:
        proxy_pid = int(proxy_path.read_text().strip())
        os.kill(proxy_pid, 0)
    except (OSError, ValueError) as exc:
        raise BudgetTelemetryError(
            "Codex is running without its OpenRouter cost ledger proxy"
        ) from exc


def load_pricing_module(path: Path):
    spec = importlib.util.spec_from_file_location("sprint_live_codex_cost", path)
    if spec is None or spec.loader is None:
        raise BudgetTelemetryError(f"cannot load API pricing module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _complete_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise BudgetTelemetryError(f"cannot read Codex session {path}: {exc}") from exc
    # An active writer can leave one partial final record.  Every earlier record
    # must remain valid or live accounting is no longer trustworthy.
    complete = data[: data.rfind(b"\n") + 1] if b"\n" in data else b""
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(complete.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BudgetTelemetryError(
                f"invalid complete JSONL record {path}:{number}: {exc}"
            ) from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def codex_api_cost(
    sessions: Path,
    *,
    default_model: str,
    default_service_tier: str | None,
    default_effort: str | None,
    pricing_module,
) -> tuple[float, int]:
    streams = [
        [path]
        for path in (sorted(sessions.rglob("*.jsonl")) if sessions.is_dir() else ())
    ]
    return codex_api_cost_streams(
        streams,
        default_model=default_model,
        default_service_tier=default_service_tier,
        default_effort=default_effort,
        pricing_module=pricing_module,
    )


def codex_api_cost_streams(
    streams: list[list[Path]],
    *,
    default_model: str,
    default_service_tier: str | None,
    default_effort: str | None,
    pricing_module,
) -> tuple[float, int]:
    total = 0.0
    request_count = 0
    for paths in streams:
        model = default_model
        service_tier = default_service_tier
        effort = default_effort
        saw_model_output = False
        rows = [row for path in paths for row in _complete_jsonl(path)]
        for row in rows:
            payload = row.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            if row.get("type") == "turn_context":
                if isinstance(payload.get("model"), str):
                    model = payload["model"]
                if isinstance(payload.get("service_tier"), str):
                    service_tier = payload["service_tier"]
                if isinstance(payload.get("effort"), str):
                    effort = payload["effort"]
                continue
            if row.get("type") == "response_item":
                kind = payload.get("type")
                if kind == "message" and payload.get("role") == "assistant":
                    saw_model_output = True
                elif kind in {"web_search_call", "function_call", "custom_tool_call"}:
                    saw_model_output = True
                continue
            if (
                row.get("type") != "event_msg"
                or payload.get("type") != "token_count"
                or not saw_model_output
            ):
                continue
            info = payload.get("info")
            usage = info.get("last_token_usage") if isinstance(info, dict) else None
            if not isinstance(usage, dict):
                raise BudgetTelemetryError(
                    f"model response has no complete token usage in {paths[-1]}"
                )
            record = pricing_module.build_request_usage_record(
                api_call_id=f"live:{paths[-1].name}:{request_count + 1}",
                model=model,
                service_tier=service_tier,
                reasoning_effort=effort,
                usage=usage,
                usage_reported_at=row.get("timestamp"),
                model_context_window=(
                    info.get("model_context_window")
                    if isinstance(info.get("model_context_window"), int)
                    else None
                ),
            )
            if record.get("cost_reconstruction_status") != "complete":
                raise BudgetTelemetryError(
                    "API usage cannot be priced safely: "
                    + ",".join(record.get("incomplete_reasons") or [])
                    + f" ({record.get('cost_reconstruction_status')})"
                )
            total += float(record["calculated_cost_usd"])
            request_count += 1
            saw_model_output = False
    return total, request_count


def durable_codex_streams(run_root: Path) -> list[list[Path]]:
    roots = run_root / "durable-trace" / "raw"
    source_dirs = sorted(roots.glob("cpu-attempt-*/codex/*")) if roots.is_dir() else []
    return [sorted(path.glob("chunks/*.jsonl")) for path in source_dirs]


def ensure_cpu_start(run_root: Path, attempt: int, now: float) -> float:
    path = run_root / "budget" / "cpu-attempts" / f"{attempt:03d}.json"
    try:
        payload = json.loads(path.read_text())
        start = float(payload["started_at_epoch_s"])
        if math.isfinite(start) and start > 0:
            return start
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    atomic_json(
        path,
        {
            "schema_version": 1,
            "attempt": attempt,
            "started_at_epoch_s": now,
            "started_at": dt.datetime.fromtimestamp(now, dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        },
    )
    return now


def cpu_allocated_seconds(run_root: Path, attempt: int, now: float) -> float:
    total = 0.0
    starts: list[tuple[int, float]] = []
    for path in sorted((run_root / "budget" / "cpu-attempts").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            starts.append(
                (int(payload["attempt"]), float(payload["started_at_epoch_s"]))
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            raise BudgetTelemetryError(f"invalid CPU allocation marker: {path}")
    for index, (number, start) in enumerate(starts):
        if number > attempt or start > now:
            raise BudgetTelemetryError("invalid CPU allocation chronology")
        end = starts[index + 1][1] if index + 1 < len(starts) else now
        total += max(0.0, end - start)
    return total


def _timeline_events(run_root: Path) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    shards = run_root / "telemetry" / "gpu_timeline" / "events"
    for path in sorted(shards.glob("*.json")) if shards.is_dir() else ():
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BudgetTelemetryError(
                f"invalid GPU lifecycle event {path}: {exc}"
            ) from exc
        if isinstance(row, dict):
            identity = str(row.get("event_id") or path.name)
            events[identity] = row
    return sorted(events.values(), key=lambda row: float(row.get("epoch_s") or 0))


def gpu_allocated_seconds(
    run_root: Path, now: float, *, standing: bool, cpu_seconds: float
) -> float:
    if standing:
        return cpu_seconds
    starts: dict[str, float] = {}
    total = 0.0
    for event in _timeline_events(run_root):
        if event.get("phase") != "gpu_lifecycle":
            continue
        detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
        kind = detail.get("event")
        lease = str(event.get("lease_id") or "")
        epoch = float(event.get("epoch_s") or 0)
        if not lease or epoch <= 0:
            raise BudgetTelemetryError("GPU lifecycle event lacks lease or timestamp")
        if kind in {"gpu_allocated", "gpu_reallocated"}:
            starts.setdefault(lease, epoch)
        elif kind in {"gpu_released", "gpu_preempted"} and lease in starts:
            total += max(0.0, epoch - starts.pop(lease))
    total += sum(max(0.0, now - start) for start in starts.values())
    return total


def write_stop(run_root: Path, runtime_dir: Path, payload: dict[str, Any]) -> None:
    marker = run_root / "BUDGET_STOP_REQUESTED.json"
    if not marker.exists():
        atomic_json(marker, payload)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stop = runtime_dir / "sprint-stop"
    temporary = stop.with_name(f".{stop.name}.{os.getpid()}.tmp")
    temporary.write_text(str(payload.get("reason") or STOP_REASON) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, stop)


def check_once(
    *,
    run_id: str,
    durable_dir: Path,
    runtime_dir: Path,
    codex_home: Path,
    pricing_path: Path,
    now: float | None = None,
) -> dict[str, Any]:
    ref = time.time() if now is None else float(now)
    run_root = durable_dir / "runs" / run_id
    run_path = run_root / "state" / "run.json"
    try:
        run = json.loads(run_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BudgetTelemetryError(f"cannot read durable run contract: {exc}") from exc
    budget = float(run["agent_cost_budget_usd"])
    enforcement = run.get("budget_enforcement") or {}
    reserve = float(enforcement.get("shutdown_reserve_usd") or 0)
    threshold = budget - reserve
    if not (math.isfinite(budget) and budget > 0 and 0 <= reserve < budget):
        raise BudgetTelemetryError("invalid budget or shutdown reserve")
    minimum_reserve = float(enforcement.get("minimum_safe_shutdown_reserve_usd") or 0)
    if not math.isfinite(minimum_reserve) or minimum_reserve <= 0:
        raise BudgetTelemetryError("run contract has no valid minimum safe reserve")
    if reserve < minimum_reserve:
        canonical_model = str(run.get("model") or "").split("/", 1)[-1]
        raise BudgetTelemetryError(
            f"shutdown reserve ${reserve:g} is below the ${minimum_reserve:g} "
            f"hard-cap minimum for {canonical_model}"
        )
    attempt = int(
        os.environ.get("SPRINT_CPU_LAUNCH_ATTEMPT")
        or run.get("cpu_launch_attempt")
        or 1
    )
    ensure_cpu_start(run_root, attempt, ref)
    cpu_seconds = cpu_allocated_seconds(run_root, attempt, ref)
    gpu_seconds = gpu_allocated_seconds(
        run_root,
        ref,
        standing=bool(run.get("standing_gpu_worker")),
        cpu_seconds=cpu_seconds,
    )
    if run.get("agent_kind") != "codex" or not run.get("usage_audit_required"):
        raise BudgetTelemetryError("live API pricing is unsupported for this run")
    if enforcement.get("api_cost_source") == "openrouter_reported_per_request":
        require_live_openrouter_proxy(runtime_dir)
        # The API key is deliberately scoped to the agent exec and controller,
        # not the sandbox keepalive. Completed response charges are already in
        # the durable proxy ledger and need no credential. An interrupted
        # response still fails closed here until the credentialed controller
        # can recover its exact OpenRouter generation charge.
        raw_api_key = os.environ.get("OPENAI_API_KEY", "")
        api_key = raw_api_key if len(raw_api_key) >= 16 else None
        api_usd, requests, pending_requests = openrouter_api_cost(
            run_root, run_id=run_id, api_key=api_key
        )
    else:
        pricing = load_pricing_module(pricing_path)
        api_kwargs = {
            "default_model": str(run.get("model") or ""),
            "default_service_tier": run.get("service_tier"),
            "default_effort": run.get("reasoning_effort"),
            "pricing_module": pricing,
        }
        local_sessions = codex_home / "sessions"
        if local_sessions.is_dir() and any(local_sessions.rglob("*.jsonl")):
            api_usd, requests = codex_api_cost(local_sessions, **api_kwargs)
        else:
            api_usd, requests = codex_api_cost_streams(
                durable_codex_streams(run_root), **api_kwargs
            )
        pending_requests = 0
    cpu_usd = cpu_seconds * CPU_USD_PER_SECOND
    training_usd = gpu_seconds * TRAINING_USD_PER_SECOND
    component_totals = {
        "model_api_usd": api_usd,
        "cpu_agent_usd": cpu_usd,
        "training_sandboxes_usd": training_usd,
    }
    total = sum(component_totals.values())
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "model": run.get("model"),
        "currency": "USD",
        "checked_at_epoch_s": ref,
        "as_of_epoch_ms": round(ref * 1000),
        "budget_usd": budget,
        "budget_remaining_usd": max(0.0, budget - total),
        "shutdown_reserve_usd": reserve,
        "minimum_safe_shutdown_reserve_usd": minimum_reserve,
        "stop_threshold_usd": threshold,
        "total_usd": total,
        "estimated_total_usd": total,
        "components": {
            "model_api": {
                "cost_usd": api_usd,
                "request_count": requests,
                "pending_request_count": pending_requests,
                "cost_source": enforcement.get(
                    "api_cost_source", "token_rate_reconstruction"
                ),
                "provider_reported": enforcement.get("api_cost_source")
                == "openrouter_reported_per_request",
            },
            "cpu_agent": {
                "cost_usd": cpu_usd,
                "allocated_seconds": cpu_seconds,
                "cost_source": "live_allocated_seconds_x_pinned_modal_tariff",
            },
            "training_sandboxes": {
                "cost_usd": training_usd,
                "allocated_seconds": gpu_seconds,
                "cost_source": "live_allocated_seconds_x_pinned_modal_tariff",
            },
        },
        "component_totals_usd": component_totals,
        "request_count": requests,
        "pending_request_count": pending_requests,
        "cpu_allocated_seconds": cpu_seconds,
        "training_allocated_seconds": gpu_seconds,
        "status": "stop_requested" if total >= threshold else "within_budget",
        "cost_basis": (
            "openrouter_reported_per_request_plus_pinned_modal_requested_resource_tariff"
            if enforcement.get("api_cost_source") == "openrouter_reported_per_request"
            else "published_api_list_price_plus_pinned_modal_requested_resource_tariff"
        ),
        "excluded_costs": ["openrouter_credit_purchase_fee"],
        "equation": {
            "total": "C(t) = C_openrouter_reported(t) + C_cpu_agent(t) + C_training(t)",
            "model_api": "sum OpenRouter usage.cost over completed requests",
            "modal_role": "allocated_seconds * pinned requested-resource rate",
        },
    }
    atomic_json(run_root / "budget" / "watchdog.json", payload)
    if total >= threshold:
        write_stop(run_root, runtime_dir, {**payload, "reason": STOP_REASON})
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--durable-dir", type=Path, default=Path("/durable"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("/run"))
    parser.add_argument("--codex-home", type=Path, default=Path("/tmp/codex-home"))
    parser.add_argument(
        "--pricing-module", type=Path, default=Path("/opt/sprint-codex-cost.py")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.durable_dir / "runs" / args.run_id
    try:
        payload = check_once(
            run_id=args.run_id,
            durable_dir=args.durable_dir,
            runtime_dir=args.runtime_dir,
            codex_home=args.codex_home,
            pricing_path=args.pricing_module,
        )
    except Exception as exc:  # noqa: BLE001 - accounting uncertainty fails closed
        payload = {
            "schema_version": 1,
            "run_id": args.run_id,
            "checked_at_epoch_s": time.time(),
            "status": "fail_closed",
            "reason": "budget_telemetry_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(run_root / "budget" / "watchdog.json", payload)
        write_stop(run_root, args.runtime_dir, payload)
        return 20
    return 10 if payload["status"] == "stop_requested" else 0


if __name__ == "__main__":
    raise SystemExit(main())
