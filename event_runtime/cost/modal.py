#!/usr/bin/env python3
"""Estimate and reconcile per-run Modal infrastructure cost.

The estimate is reproducible from lifecycle intervals plus a pinned public
tariff.  The provider reconciliation is authoritative pre-credit spend from
``modal billing report`` and is intentionally retained by billing category.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from typing import Any


SCHEMA_VERSION = 1
BILLING_COLLECTION_BUFFER_SECONDS = 5 * 60
BILLING_REPORT_CACHE_SECONDS = 60
MODAL_SANDBOX_PRICING: dict[str, Any] = {
    "id": "modal-sandbox-public-2026-08-08",
    "provider": "modal",
    "currency": "USD",
    "captured_at": "2026-08-08",
    "source_url": "https://modal.com/pricing",
    "billing_basis": "per_second_max_of_request_or_actual_cpu_memory",
    "rates_usd_per_second": {
        "CPU": "0.00003942",  # one physical core
        "Memory": "0.00000667",  # one GiB
        "A10G": "0.000306",  # one GPU (pricing page calls this A10)
    },
    "volume_storage": {
        "rate_usd_per_gib_month": "0.09",
        "workspace_included_gib_month": "1024",
        "attribution": "workspace_level_not_run_level",
    },
}

ROLE_CONTRACT_KEYS = {
    "cpu_agent": "cpu_agent",
    "training_gpu": "training_worker",
    "verifier_gpu": "verifier",
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def iso_z(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def floor_hour(value: dt.datetime) -> dt.datetime:
    return value.astimezone(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)


def ceil_hour(value: dt.datetime) -> dt.datetime:
    floored = floor_hour(value)
    return floored if value == floored else floored + dt.timedelta(hours=1)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value or "0"))


def _contract_quantity(contract: dict[str, Any], name: str) -> Decimal:
    if name == "physical_cpu_cores":
        value = contract.get(name, contract.get("vcpus", 0))
    elif name == "memory_gib":
        value = _decimal(contract.get("memory_mb")) / Decimal(1024)
        return value
    elif name == "gpu_count":
        value = contract.get(name, 0)
    else:
        value = 0
    return _decimal(value)


def estimate_cost(
    *, resource_contract: dict[str, Any], allocated_ms_by_role: dict[str, int]
) -> dict[str, Any]:
    """Return a pinned-tariff estimate by role and provider billing category."""
    rates = MODAL_SANDBOX_PRICING["rates_usd_per_second"]
    by_role: dict[str, Any] = {}
    category_totals: dict[str, Decimal] = {}
    total = Decimal(0)
    for role, contract_key in ROLE_CONTRACT_KEYS.items():
        contract = resource_contract.get(contract_key) or {}
        seconds = Decimal(max(0, int(allocated_ms_by_role.get(role, 0)))) / Decimal(
            1000
        )
        quantities = {
            "CPU": _contract_quantity(contract, "physical_cpu_cores"),
            "Memory": _contract_quantity(contract, "memory_gib"),
            "A10G": _contract_quantity(contract, "gpu_count"),
        }
        components: dict[str, float] = {}
        role_total = Decimal(0)
        for category, quantity in quantities.items():
            if not quantity:
                continue
            value = seconds * quantity * _decimal(rates[category])
            components[category] = float(value)
            category_totals[category] = (
                category_totals.get(category, Decimal(0)) + value
            )
            role_total += value
        by_role[role] = {
            "allocated_ms": int(seconds * 1000),
            "quantities": {key: float(value) for key, value in quantities.items()},
            "cost_components_usd": components,
            "estimated_cost_usd": float(role_total),
        }
        total += role_total
    return {
        "pricing_snapshot": MODAL_SANDBOX_PRICING,
        "by_role": by_role,
        "by_category_usd": {
            key: float(value) for key, value in sorted(category_totals.items())
        },
        "estimated_cost_usd": float(total),
        "estimate_kind": "requested_resource_floor",
        "estimate_note": (
            "CPU and memory are estimated from requests. Modal bills max(request, "
            "actual), so provider reconciliation can be higher when a sandbox bursts."
        ),
    }


def billing_object_roles(run: dict[str, Any]) -> dict[str, str]:
    pairs = {
        "cpu_agent": run.get("app_name"),
        "training_gpu": run.get("training_app_name"),
        "verifier_gpu": run.get("verifier_app_name"),
        "volume": run.get("volume_name"),
    }
    descriptions: dict[str, str] = {}
    for role, value in pairs.items():
        if not value:
            continue
        description = str(value)
        if description in descriptions:
            raise ValueError(
                f"Modal billing object description {description!r} is shared by "
                f"{descriptions[description]} and {role}"
            )
        descriptions[description] = role
    return descriptions


def aggregate_billing_rows(
    rows: list[dict[str, Any]], *, run: dict[str, Any]
) -> dict[str, Any]:
    """Filter exact run-owned Modal objects and retain every billed category."""
    descriptions = billing_object_roles(run)
    selected: list[dict[str, Any]] = []
    by_role: dict[str, Decimal] = {}
    by_category: dict[str, Decimal] = {}
    by_role_category: dict[str, dict[str, Decimal]] = {}
    for row in rows:
        description = str(row.get("description") or "")
        role = descriptions.get(description)
        if role is None:
            continue
        category = str(row.get("resource") or "Uncategorized")
        cost = _decimal(row.get("cost"))
        selected.append(
            {
                "object_id": row.get("object_id"),
                "description": description,
                "environment": row.get("environment"),
                "interval_start": row.get("interval_start"),
                "role": role,
                "category": category,
                "cost_usd": float(cost),
            }
        )
        by_role[role] = by_role.get(role, Decimal(0)) + cost
        by_category[category] = by_category.get(category, Decimal(0)) + cost
        role_categories = by_role_category.setdefault(role, {})
        role_categories[category] = role_categories.get(category, Decimal(0)) + cost
    total = sum(by_role.values(), Decimal(0))
    return {
        "items": selected,
        "by_role_usd": {key: float(value) for key, value in sorted(by_role.items())},
        "by_category_usd": {
            key: float(value) for key, value in sorted(by_category.items())
        },
        "by_role_category_usd": {
            role: {
                category: float(value) for category, value in sorted(categories.items())
            }
            for role, categories in sorted(by_role_category.items())
        },
        "provider_cost_precredits_usd": float(total),
    }


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _expected_billed_roles(state_dir: pathlib.Path) -> set[str]:
    """Return roles whose lifecycle evidence says Modal must report spend."""
    expected = {"cpu_agent"}
    for path in state_dir.rglob("gpu_timeline.jsonl"):
        try:
            contents = path.read_text()
        except OSError:
            continue
        if '"gpu_allocated"' in contents or '"gpu_reallocated"' in contents:
            expected.add("training_gpu")
            break
    if any(state_dir.rglob("verifier/telemetry/lifecycle.json")):
        expected.add("verifier_gpu")
    timeline = _read_json(state_dir / "telemetry" / "unified-timeline.json")
    summary = timeline.get("resource_usage_summary")
    if not isinstance(summary, dict):
        return expected
    for role in ("training_gpu", "verifier_gpu"):
        role_summary = summary.get(role)
        if (
            isinstance(role_summary, dict)
            and int(role_summary.get("allocated_ms") or 0) > 0
        ):
            expected.add(role)
    return expected


def _expected_role_categories(roles: set[str]) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {}
    for role in roles:
        categories = {"CPU", "Memory"}
        if role in {"training_gpu", "verifier_gpu"}:
            categories.add("A10G")
        expected[role] = categories
    return expected


def scan_volume_storage(
    run: dict[str, Any], *, duration_seconds: float
) -> dict[str, Any]:
    """Capture final logical Volume bytes and a non-invoiced nominal estimate."""
    volume_name = str(run.get("volume_name") or "")
    if not volume_name:
        return {"status": "unavailable", "reason": "missing_volume_name"}
    old_profile = os.environ.get("MODAL_PROFILE")
    if run.get("modal_profile"):
        os.environ["MODAL_PROFILE"] = str(run["modal_profile"])
    try:
        import modal

        volume = modal.Volume.from_name(volume_name, create_if_missing=False)
        entries = list(volume.iterdir("/", recursive=True))
    except Exception:  # noqa: BLE001 - provider read failure is retained as status
        return {"status": "unavailable", "reason": "provider_volume_scan_failed"}
    finally:
        if old_profile is None:
            os.environ.pop("MODAL_PROFILE", None)
        else:
            os.environ["MODAL_PROFILE"] = old_profile
    files = [
        entry
        for entry in entries
        if getattr(getattr(entry, "type", None), "name", "") == "FILE"
    ]
    logical_bytes = sum(max(0, int(getattr(entry, "size", 0) or 0)) for entry in files)
    gib = Decimal(logical_bytes) / Decimal(1024**3)
    seconds_per_month = Decimal("365.2425") * Decimal(86400) / Decimal(12)
    nominal = (
        gib
        * Decimal(str(max(0.0, duration_seconds)))
        / seconds_per_month
        * _decimal(MODAL_SANDBOX_PRICING["volume_storage"]["rate_usd_per_gib_month"])
    )
    return {
        "status": "captured",
        "volume_name": volume_name,
        "logical_bytes_at_collection": logical_bytes,
        "file_count_at_collection": len(files),
        "nominal_cost_usd_assuming_final_size_for_full_run": float(nominal),
        "included_in_run_total": False,
        "exclusion_reason": (
            "Modal's included Volume allowance and storage billing are workspace-level, "
            "and an endpoint byte count is not a time-weighted invoice measurement."
        ),
    }


def run_bounds(
    state_dir: pathlib.Path, run: dict[str, Any]
) -> tuple[dt.datetime, dt.datetime | None]:
    start = parse_time(run.get("created_at"))
    if start is None:
        raise ValueError("run.json is missing a valid created_at")
    end_candidates: list[dt.datetime] = []
    # STOP_ACK ends the agent process, not necessarily every run-owned Modal
    # allocation. Conversely, FINALIZED is a host-side evidence timestamp and
    # must never extend the provider billing window: doing so makes a complete
    # report reopen into the next hour merely because finalization ran later.
    # Only the current Harbor job result is a run terminal boundary. Continuous
    # blind-verifier attempts also contain result.json files, but the CPU agent
    # keeps running after those scores and they must not truncate its billing
    # interval.
    job_path = run.get("job_path") or run.get("expected_job_path")
    job_finished_at: dt.datetime | None = None
    if isinstance(job_path, str) and job_path:
        job = pathlib.Path(job_path)
        job_finished_at = parse_time(_read_json(job / "result.json").get("finished_at"))
        if job_finished_at is not None:
            end_candidates.append(job_finished_at)
        # Central blind scoring can drain accepted policies after the CPU
        # agent exits. A trusted attempt result is written only after its
        # sealed verifier GPU has stopped, so it is a safe terminal bound once
        # the CPU agent itself is terminal. While the agent is running, an
        # earlier verifier result must not make the run appear stopped.
        if job_finished_at is not None:
            for result_path in job.glob(
                "*/artifacts/continuous/attempts/*/result.json"
            ):
                value = parse_time(_read_json(result_path).get("finished_at"))
                if value is not None:
                    end_candidates.append(value)

    # Training workers can outlive the CPU process briefly while the host
    # fences a lease. Preserve their authoritative lifecycle boundary without
    # depending on the derived unified timeline.
    for name in (
        "cpu_lifecycle.jsonl",
        "gpu_timeline.jsonl",
        "durable-gpu-timeline.jsonl",
    ):
        path = state_dir / "telemetry" / name
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for raw in lines:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            event = row.get("event")
            detail = row.get("detail")
            semantic = detail.get("event") if isinstance(detail, dict) else None
            terminal = (
                event == "cpu_launch_exited"
                or semantic in {"gpu_preempted", "gpu_released"}
                or (row.get("phase") == "active" and row.get("action") == "exit")
            )
            if not terminal:
                continue
            value = parse_time(row.get("ts_utc") or row.get("at"))
            if value is None and isinstance(row.get("epoch_s"), (int, float)):
                value = dt.datetime.fromtimestamp(
                    float(row["epoch_s"]), tz=dt.timezone.utc
                )
            if value is not None:
                end_candidates.append(value)
    return start, max(end_candidates, default=None)


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(raw, 0o600)
        os.replace(raw, path)
    finally:
        pathlib.Path(raw).unlink(missing_ok=True)


def _cached_billing_report(
    state_dir: pathlib.Path,
    *,
    command: list[str],
    env: dict[str, str],
    profile: str,
    current: dt.datetime,
    runner: Any,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """Serialize and briefly share identical workspace billing queries."""
    identity = hashlib.sha256(
        json.dumps(
            {"command": command, "profile": profile},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    root = state_dir.parent / ".modal-billing-report-cache"
    cache_path = root / f"{identity}.json"
    lock_path = root / f"{identity}.lock"
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cached = _read_json(cache_path)
        captured_at = cached.get("captured_at_epoch_s")
        if (
            isinstance(captured_at, (int, float))
            and not isinstance(captured_at, bool)
            and 0 <= current.timestamp() - float(captured_at) <= BILLING_REPORT_CACHE_SECONDS
            and isinstance(cached.get("stdout"), str)
        ):
            return (
                subprocess.CompletedProcess(command, 0, cached["stdout"], ""),
                True,
            )
        completed = runner(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        if completed.returncode == 0:
            _atomic_json(
                cache_path,
                {
                    "schema_version": 1,
                    "captured_at_epoch_s": current.timestamp(),
                    "stdout": completed.stdout,
                },
            )
        return completed, False


def collect_provider_billing(
    state_dir: pathlib.Path,
    *,
    now: dt.datetime | None = None,
    runner: Any = subprocess.run,
    volume_scanner: Any = scan_volume_storage,
) -> dict[str, Any]:
    """Collect a complete full-hour Modal billing report for one stopped run."""
    output_path = state_dir / "telemetry" / "modal-cost.json"
    existing = _read_json(output_path)
    run = _read_json(state_dir / "run.json")
    if not run:
        raise ValueError(f"missing run.json under {state_dir}")
    start, stopped_at = run_bounds(state_dir, run)
    if stopped_at is not None and existing.get("provider_complete") is True:
        expected_query_end = iso_z(ceil_hour(stopped_at))
        if existing.get("query_end") == expected_query_end:
            return existing
    current = (now or utc_now()).astimezone(dt.timezone.utc)
    query_start = floor_hour(start)
    expected_roles = _expected_billed_roles(state_dir)
    expected_categories = _expected_role_categories(expected_roles)
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.get("run_id") or state_dir.name,
        "generated_at": iso_z(current),
        "source": "modal_billing_report",
        "billing_basis": "provider_report_precredits",
        "billing_objects": billing_object_roles(run),
        "expected_billed_roles": sorted(expected_roles),
        "expected_role_categories": {
            role: sorted(categories)
            for role, categories in sorted(expected_categories.items())
        },
        "provider_complete": False,
        "status": "pending",
        "items": [],
        "by_role_usd": {},
        "by_category_usd": {},
        "by_role_category_usd": {},
        "provider_cost_precredits_usd": None,
    }
    if stopped_at is None:
        base["pending_reason"] = "run_has_no_terminal_timestamp"
        _atomic_json(output_path, base)
        return base
    query_end = ceil_hour(stopped_at)
    eligible_at = query_end + dt.timedelta(seconds=BILLING_COLLECTION_BUFFER_SECONDS)
    base.update(
        {
            "run_started_at": iso_z(start),
            "run_stopped_at": iso_z(stopped_at),
            "query_start": iso_z(query_start),
            "query_end": iso_z(query_end),
            "eligible_at": iso_z(eligible_at),
        }
    )
    if current < eligible_at:
        base["pending_reason"] = "waiting_for_full_hour_and_modal_collection_buffer"
        _atomic_json(output_path, base)
        return base

    modal_cli = os.environ.get("MODAL_CLI") or shutil.which("modal")
    cmd = [
        *([modal_cli] if modal_cli else [sys.executable, "-m", "modal"]),
        "billing",
        "report",
        "--start",
        query_start.strftime("%Y-%m-%dT%H:%M:%S"),
        "--end",
        query_end.strftime("%Y-%m-%dT%H:%M:%S"),
        "--resolution",
        "h",
        "--show-resources",
        "--json",
    ]
    env = os.environ.copy()
    if run.get("modal_profile"):
        env["MODAL_PROFILE"] = str(run["modal_profile"])
    completed, cache_hit = _cached_billing_report(
        state_dir,
        command=cmd,
        env=env,
        profile=str(run.get("modal_profile") or ""),
        current=current,
        runner=runner,
    )
    base["billing_report_cache_hit"] = cache_hit
    if completed.returncode != 0:
        base["status"] = "error"
        base["error"] = (
            completed.stderr or completed.stdout or "billing command failed"
        )[-2000:]
        _atomic_json(output_path, base)
        return base
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        base["status"] = "error"
        base["error"] = f"invalid Modal billing JSON: {exc}"
        _atomic_json(output_path, base)
        return base
    if not isinstance(rows, list):
        base["status"] = "error"
        base["error"] = "Modal billing JSON was not a list"
        _atomic_json(output_path, base)
        return base
    aggregated = aggregate_billing_rows(rows, run=run)
    base.update(aggregated)
    base["provider_report_sha256"] = hashlib.sha256(
        completed.stdout.encode("utf-8")
    ).hexdigest()
    base["selected_items_sha256"] = hashlib.sha256(
        json.dumps(aggregated["items"], sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    missing_roles = sorted(
        set(base["expected_billed_roles"]) - set(aggregated["by_role_usd"])
    )
    if missing_roles:
        base["pending_reason"] = "provider_report_missing_expected_role_rows"
        base["missing_billed_roles"] = missing_roles
        _atomic_json(output_path, base)
        return base
    missing_categories = {
        role: sorted(categories - set(aggregated["by_role_category_usd"].get(role, {})))
        for role, categories in expected_categories.items()
        if categories - set(aggregated["by_role_category_usd"].get(role, {}))
    }
    if missing_categories:
        base["pending_reason"] = "provider_report_missing_expected_category_rows"
        base["missing_role_categories"] = missing_categories
        _atomic_json(output_path, base)
        return base
    base["volume_storage"] = volume_scanner(
        run, duration_seconds=(stopped_at - start).total_seconds()
    )
    if base["volume_storage"].get("status") != "captured":
        base["pending_reason"] = "provider_volume_storage_snapshot_unavailable"
        _atomic_json(output_path, base)
        return base
    base["provider_complete"] = True
    base["status"] = "complete"
    base["invoice_adjustments_included"] = False
    base["invoice_note"] = (
        "Provider report is exact pre-credit resource spend for run-owned Modal "
        "objects; credits, reservations, subscription fees, and taxes are invoice-level."
    )
    _atomic_json(output_path, base)
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    payload = collect_provider_billing(args.state_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
