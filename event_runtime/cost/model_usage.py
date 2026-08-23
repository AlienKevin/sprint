#!/usr/bin/env python3
"""Rebuild per-attempt ATIF and one run-level usage ledger from durable Codex logs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from harbor.agents.installed.codex import Codex
from harbor.agents.installed.codex_cost import (
    USAGE_AUDIT_SCHEMA_VERSION,
    pricing_snapshot_for_request,
)
from harbor.utils.trajectory_utils import format_trajectory_json


ATTEMPT_RE = re.compile(r"cpu-attempt-(\d+)")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def resolve_agent_provenance(agent_dir: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    candidate = (agent_dir / relative).resolve()
    try:
        candidate.relative_to(agent_dir.resolve())
    except ValueError:
        raise SystemExit("usage provenance path escapes the Harbor agent directory")
    return candidate if candidate.is_file() else None


def recover_harbor_provenance(
    *, trial: Path, run: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild stale mutable-path provenance into immutable signed snapshots."""
    agent_dir = trial / "agent"
    provenance = audit.get("provenance") or {}
    source_path = resolve_agent_provenance(
        agent_dir, provenance.get("source_session_path")
    )
    if source_path is None:
        source_name = provenance.get("source_session_file")
        if not isinstance(source_name, str) or not source_name:
            raise SystemExit("Harbor usage audit has no recoverable source session")
        matches = list((agent_dir / "sessions").rglob(source_name))
        matches.extend((agent_dir / "codex-state" / "sessions").rglob(source_name))
        matches = [path.resolve() for path in matches if path.is_file()]
        if not matches:
            raise SystemExit("Harbor usage audit source session is missing")
        actual_hashes = {sha256_file(path) for path in matches}
        if len(actual_hashes) != 1:
            raise SystemExit("Harbor final source session copies disagree")
        source_path = matches[0]

    kwargs: dict[str, Any] = {
        "logs_dir": agent_dir,
        "model_name": str(run["model"]),
        "reasoning_effort": str(run["reasoning_effort"]),
    }
    if str(run["model"]).split("/", 1)[-1] in {
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    }:
        kwargs["service_tier"] = "default"
    agent = Codex(**kwargs)
    trajectory = agent._convert_events_to_trajectory(source_path.parent)
    recovered = getattr(agent, "_last_usage_audit", None)
    if trajectory is None or not isinstance(recovered, dict):
        raise SystemExit("could not reconstruct Harbor final usage provenance")

    def request_signature(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [
            (
                row.get("api_call_id"),
                row.get("model"),
                row.get("service_tier"),
                row.get("input_tokens"),
                row.get("cached_input_tokens"),
                row.get("cache_write_input_tokens"),
                row.get("output_tokens"),
                row.get("reasoning_output_tokens"),
                row.get("total_tokens"),
                row.get("usage_reported_at"),
            )
            for row in payload.get("requests") or []
            if isinstance(row, dict)
        ]

    if request_signature(recovered) != request_signature(audit):
        raise SystemExit("reconstructed Harbor request ledger differs from its audit")
    recovered_cost = recovered.get("calculated_api_usage_usd")
    previous_cost = audit.get("calculated_api_usage_usd")

    trajectory_text = format_trajectory_json(trajectory.to_json_dict())
    final_metrics = json.loads(trajectory_text).get("final_metrics") or {}
    if final_metrics.get("total_cost_usd") != recovered_cost:
        raise SystemExit("reconstructed Harbor ATIF cost differs from its audit")

    provenance_dir = agent_dir / "usage-provenance"
    source_snapshot = provenance_dir / "source-session.jsonl"
    trajectory_snapshot = provenance_dir / "trajectory.json"
    atomic_bytes(source_snapshot, source_path.read_bytes())
    atomic_text(trajectory_snapshot, trajectory_text)

    original_bytes = (agent_dir / "usage-audit.json").read_bytes()
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    original_path = provenance_dir / f"original-usage-audit.{original_sha}.json"
    if not original_path.exists():
        atomic_bytes(original_path, original_bytes)
    recovered["provider_reported_total_cost_usd"] = audit.get(
        "provider_reported_total_cost_usd"
    )
    recovered["selected_total_cost_usd"] = (
        recovered["provider_reported_total_cost_usd"]
        if isinstance(recovered["provider_reported_total_cost_usd"], (int, float))
        else recovered_cost
    )
    if previous_cost != recovered_cost:
        recovered["pricing_correction"] = {
            "reason": "pricing_snapshot_refresh",
            "previous_calculated_api_usage_usd": previous_cost,
            "corrected_calculated_api_usage_usd": recovered_cost,
            "previous_pricing_snapshot_ids": sorted(
                str(row["id"])
                for row in audit.get("pricing_snapshots") or []
                if isinstance(row, dict) and row.get("id")
            ),
            "corrected_pricing_snapshot_ids": sorted(
                str(row["id"])
                for row in recovered.get("pricing_snapshots") or []
                if isinstance(row, dict) and row.get("id")
            ),
        }
    recovered["provenance"] = {
        "source_session_path": str(source_snapshot.relative_to(agent_dir)),
        "source_session_sha256": sha256_file(source_snapshot),
        "trajectory_path": str(trajectory_snapshot.relative_to(agent_dir)),
        "trajectory_sha256": sha256_file(trajectory_snapshot),
    }
    recovered_text = json.dumps(recovered, indent=2, sort_keys=True) + "\n"
    recovered_sha = hashlib.sha256(recovered_text.encode()).hexdigest()
    atomic_text(agent_dir / "usage-audit.json", recovered_text)
    atomic_text(
        provenance_dir / "recovery-attestation.json",
        json.dumps(
            {
                "schema_version": 1,
                "recovered_at": dt.datetime.now(dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "method": "deterministic_codex_session_replay",
                "original_usage_audit_path": str(original_path.relative_to(agent_dir)),
                "original_usage_audit_sha256": original_sha,
                "recovered_usage_audit_sha256": recovered_sha,
                "request_count": len(recovered.get("requests") or []),
                "calculated_api_usage_usd": recovered_cost,
                "previous_calculated_api_usage_usd": previous_cost,
                "source_session_sha256": sha256_file(source_snapshot),
                "trajectory_sha256": sha256_file(trajectory_snapshot),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return recovered


def source_groups(state_dir: Path) -> list[tuple[int, str, list[Path]]]:
    groups: dict[tuple[int, str], list[Path]] = {}
    roots = [state_dir / "durable-trace" / "raw", state_dir / "trace" / "raw"]
    roots.extend(state_dir.glob("recovery/*/*/trace/raw"))
    for root in roots:
        for chunks in root.glob("cpu-attempt-*/codex/*/chunks"):
            match = ATTEMPT_RE.fullmatch(chunks.parents[2].name)
            if not match:
                continue
            attempt = int(match.group(1))
            source_id = chunks.parent.name
            files = sorted(chunks.glob("*.jsonl"))
            if not files:
                continue
            key = (attempt, source_id)
            existing = groups.get(key)
            if existing is None or sum(path.stat().st_size for path in files) > sum(
                path.stat().st_size for path in existing
            ):
                # Recovery snapshots may contain an older prefix of the same
                # immutable stream. Use the most complete copy.
                groups[key] = files
    return [
        (attempt, source_id, groups[(attempt, source_id)])
        for attempt, source_id in sorted(groups)
    ]


def provider_usage_records(state_dir: Path, run_id: str) -> list[dict[str, Any]]:
    """Load completed OpenRouter records from the synced durable ledger."""
    rows: list[dict[str, Any]] = []
    root = state_dir / "provider-api-usage"
    for path in sorted(root.glob("**/requests/*.json")) if root.is_dir() else ():
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid provider usage record: {path}") from exc
        if not isinstance(record, dict) or record.get("run_id") != run_id:
            raise SystemExit(f"provider usage identity mismatch: {path}")
        cost = record.get("provider_reported_cost_usd")
        benchmark_cost = record.get("benchmark_cost_usd")
        if benchmark_cost is None:
            benchmark_cost = record.get("undiscounted_cost_usd")
        if benchmark_cost is None and record.get("schema_version") in {None, 1}:
            benchmark_cost = cost
        if (
            isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(float(cost))
            and float(cost) >= 0
            and isinstance(benchmark_cost, (int, float))
            and not isinstance(benchmark_cost, bool)
            and math.isfinite(float(benchmark_cost))
            and float(benchmark_cost) >= float(cost)
            and record.get("state") in {"complete", "recovered_complete"}
        ):
            rows.append(record)
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("cpu_attempt") or 0),
            str(row.get("requested_at") or ""),
            str(row.get("ledger_request_id") or ""),
        ),
    )


def _provider_usage_signature(record: dict[str, Any]) -> tuple[int, ...] | None:
    usage = record.get("usage")
    if not isinstance(usage, dict):
        return None
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return (
        int(usage.get("input_tokens") or 0),
        int(input_details.get("cached_tokens") or 0),
        int(input_details.get("cache_write_tokens") or 0),
        int(usage.get("output_tokens") or 0),
        int(output_details.get("reasoning_tokens") or 0),
        int(usage.get("total_tokens") or 0),
    )


def _codex_usage_signature(request: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(request.get(field) or 0)
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        )
    )


def apply_provider_reported_costs(
    requests: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    """Bind every Codex usage event to exactly one OpenRouter charge."""
    def bind(request: dict[str, Any], record: dict[str, Any]) -> None:
        provider_cost = float(record["provider_reported_cost_usd"])
        cost = float(
            record.get(
                "benchmark_cost_usd",
                record.get("undiscounted_cost_usd", provider_cost),
            )
        )
        usage = record.get("usage")
        details = (usage.get("cost_details") or {}) if isinstance(usage, dict) else {}
        request.update(
            {
                "pricing_snapshot_id": None,
                "calculated_cost_usd": cost,
                "provider_reported_cost_usd": provider_cost,
                "endpoint_list_cost_usd": float(
                    record.get("undiscounted_cost_usd", provider_cost)
                ),
                "promotion_savings_usd": cost - provider_cost,
                "promotion_adjustment_usd": float(
                    record.get("promotion_adjustment_usd", 0.0)
                ),
                "deepseek_peak_adjustment_usd": float(
                    record.get("deepseek_peak_adjustment_usd", 0.0)
                ),
                "benchmark_adjustment_usd": cost - provider_cost,
                "promotion_discount_fraction": record.get(
                    "promotion_discount_fraction"
                ),
                "promotion_snapshot": record.get("promotion_snapshot"),
                "cost_components_usd": {
                    str(name): float(value)
                    for name, value in details.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                },
                "cost_reconstruction_status": "complete",
                "cost_basis": record.get(
                    "cost_basis", "openrouter_list_price_before_endpoint_discount"
                ),
                "provider_cost_basis": "openrouter_reported_per_request",
                "openrouter_generation_id": record.get("generation_id"),
                "openrouter_ledger_request_id": record.get("ledger_request_id"),
                "openrouter_response_model": record.get("response_model"),
            }
        )

    remaining = list(records)
    for request in requests:
        attempt = int(request.get("cpu_attempt") or 0)
        signature = _codex_usage_signature(request)
        match_index = next(
            (
                index
                for index, record in enumerate(remaining)
                if int(record.get("cpu_attempt") or 0) == attempt
                and _provider_usage_signature(record) == signature
            ),
            None,
        )
        if match_index is None:
            # If a stream was interrupted after OpenRouter assigned a generation
            # ID, the generation endpoint can recover the exact billed cost but
            # may not return the terminal Responses-API usage object. Codex is
            # serial within an attempt, so bind those recovery-only records in
            # durable request order after all exact token-signature matches.
            match_index = next(
                (
                    index
                    for index, record in enumerate(remaining)
                    if int(record.get("cpu_attempt") or 0) == attempt
                    and _provider_usage_signature(record) is None
                    and record.get("state") == "recovered_complete"
                ),
                None,
            )
        if match_index is None:
            raise SystemExit(
                "Codex usage has no matching OpenRouter per-request cost: "
                f"attempt={attempt} usage={signature}"
            )
        record = remaining.pop(match_index)
        bind(request, record)

    # A request may finish and be billed immediately before the Codex turn is
    # interrupted, leaving no terminal token_count in the local session log.
    # The provider ledger is authoritative for both its cost and usage, so keep
    # such calls as explicit provider-only audit rows instead of dropping spend
    # or failing the archived run rebuild.
    for record in remaining:
        usage = record.get("usage") or {}
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        cached_tokens = int(input_details.get("cached_tokens") or 0)
        cache_write_tokens = int(input_details.get("cache_write_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)
        ledger_id = str(record.get("ledger_request_id") or "unknown")
        request = {
            "api_call_id": f"openrouter_only_{ledger_id}",
            "run_api_call_id": f"openrouter_only:{ledger_id}",
            "cpu_attempt": int(record.get("cpu_attempt") or 0),
            "usage_reported_at": record.get("completed_at")
            or record.get("requested_at"),
            "model": record.get("response_model") or record.get("requested_model"),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "cache_write_input_tokens": cache_write_tokens,
            "ordinary_uncached_input_tokens": max(
                0, input_tokens - cached_tokens - cache_write_tokens
            ),
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "provider_only_usage": True,
        }
        bind(request, record)
        requests.append(request)


def reconstruct_group(
    *,
    state_dir: Path,
    run: dict[str, Any],
    attempt: int,
    source_id: str,
    chunks: list[Path],
) -> dict[str, Any]:
    out = (
        state_dir / "trace" / "reconstructed" / f"cpu-attempt-{attempt:03d}" / source_id
    )
    sessions = out / "sessions"
    combined = b"".join(path.read_bytes() for path in chunks)
    session_path = sessions / "rollout.jsonl"
    atomic_text(session_path, combined.decode("utf-8", errors="replace"))

    kwargs: dict[str, Any] = {
        "logs_dir": out,
        "model_name": str(run["model"]),
        "reasoning_effort": str(run["reasoning_effort"]),
    }
    if str(run["model"]).split("/", 1)[-1] in {
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    }:
        kwargs["service_tier"] = "default"
    agent = Codex(**kwargs)
    trajectory = agent._convert_events_to_trajectory(sessions)
    audit = getattr(agent, "_last_usage_audit", None)
    if trajectory is not None:
        trajectory_path = out / "trajectory.json"
        atomic_text(trajectory_path, format_trajectory_json(trajectory.to_json_dict()))
    else:
        trajectory_path = None

    source = {
        "cpu_attempt": attempt,
        "source_id": source_id,
        "combined_session_sha256": hashlib.sha256(combined).hexdigest(),
        "chunks": [
            {
                "path": str(path.relative_to(state_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in chunks
        ],
        "trajectory_path": (
            str(trajectory_path.relative_to(state_dir)) if trajectory_path else None
        ),
        "trajectory_sha256": sha256_file(trajectory_path) if trajectory_path else None,
        "_session_path": str(session_path),
    }
    if not isinstance(audit, dict):
        source.update(
            {
                "session_id": None,
                "request_count": 0,
                "cost_reconstruction_complete": True,
                "calculated_api_usage_usd": 0.0,
                "requests": [],
                "pricing_snapshots": [],
                "note": "no completed model request was present",
            }
        )
        return source

    session_id = str(audit.get("session_id") or source_id)
    requests = []
    for request in audit.get("requests") or []:
        if not isinstance(request, dict):
            continue
        requests.append(
            {
                **request,
                "session_id": session_id,
                "cpu_attempt": attempt,
                "run_api_call_id": f"{session_id}:{request.get('api_call_id')}",
            }
        )
    source.update(
        {
            "session_id": session_id,
            "request_count": len(requests),
            "cost_reconstruction_complete": audit.get("cost_reconstruction_complete"),
            "calculated_api_usage_usd": audit.get("calculated_api_usage_usd"),
            "calculated_api_usage_cost_basis": audit.get(
                "calculated_api_usage_cost_basis"
            ),
            "requests": requests,
            "pricing_snapshots": audit.get("pricing_snapshots") or [],
            "reconciliation_mismatches": audit.get("reconciliation_mismatches") or {},
        }
    )
    atomic_text(
        out / "usage-audit.json", json.dumps(source, indent=2, sort_keys=True) + "\n"
    )
    return source


def harbor_final_source(
    *, state_dir: Path, run: dict[str, Any]
) -> dict[str, Any] | None:
    """Load the complete normally-exited session Harbor archived locally.

    The durable mirror remains authoritative for abruptly killed CPU attempts.
    On a normal exit, Harbor's own archive may contain a final tail written
    after the last mirror interval; merge that signed superset instead of
    silently dropping the last model requests.
    """
    raw_trial = run.get("trial_path")
    if not raw_trial:
        return None
    trial = Path(str(raw_trial)).resolve()
    try:
        trial.relative_to(state_dir)
    except ValueError as exc:
        raise SystemExit("Harbor trial path escapes the run state directory") from exc
    audit_path = trial / "agent" / "usage-audit.json"
    try:
        audit = json.loads(audit_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    agent_dir = trial / "agent"
    provenance = audit.get("provenance") or {}
    session_path = resolve_agent_provenance(
        agent_dir, provenance.get("source_session_path")
    )
    provenance_trajectory = resolve_agent_provenance(
        agent_dir, provenance.get("trajectory_path")
    )
    source_hash = provenance.get("source_session_sha256")
    stale_luna_pricing = any(
        isinstance(request, dict)
        and (
            expected := pricing_snapshot_for_request(
                request.get("model"), request.get("usage_reported_at")
            )
        )
        is not None
        and request.get("pricing_snapshot_id") != expected.get("id")
        for request in audit.get("requests") or []
    )
    if (
        session_path is None
        or provenance_trajectory is None
        or source_hash != sha256_file(session_path)
        or provenance.get("trajectory_sha256") != sha256_file(provenance_trajectory)
        or stale_luna_pricing
    ):
        audit = recover_harbor_provenance(trial=trial, run=run, audit=audit)
        provenance = audit["provenance"]
        session_path = resolve_agent_provenance(
            agent_dir, provenance["source_session_path"]
        )
        provenance_trajectory = resolve_agent_provenance(
            agent_dir, provenance["trajectory_path"]
        )
        source_hash = provenance["source_session_sha256"]
    if session_path is None or provenance_trajectory is None:
        raise SystemExit("Harbor usage audit immutable provenance is missing")
    attempt = int(run.get("cpu_launch_attempt") or 1)
    session_id = str(audit.get("session_id") or source_hash[:20])
    requests = [
        {
            **request,
            "session_id": session_id,
            "cpu_attempt": attempt,
            "run_api_call_id": f"{session_id}:{request.get('api_call_id')}",
        }
        for request in audit.get("requests") or []
        if isinstance(request, dict)
    ]
    return {
        "cpu_attempt": attempt,
        "source_id": f"harbor-final-{source_hash[:20]}",
        "combined_session_sha256": source_hash,
        "chunks": [
            {
                "path": str(session_path.relative_to(state_dir)),
                "sha256": source_hash,
                "bytes": session_path.stat().st_size,
            }
        ],
        "trajectory_path": str(provenance_trajectory.relative_to(state_dir)),
        "trajectory_sha256": sha256_file(provenance_trajectory),
        "_session_path": str(session_path),
        "session_id": session_id,
        "request_count": len(requests),
        "cost_reconstruction_complete": audit.get("cost_reconstruction_complete"),
        "calculated_api_usage_usd": audit.get("calculated_api_usage_usd"),
        "calculated_api_usage_cost_basis": audit.get("calculated_api_usage_cost_basis"),
        "requests": requests,
        "pricing_snapshots": audit.get("pricing_snapshots") or [],
        "reconciliation_mismatches": audit.get("reconciliation_mismatches") or {},
        "origin": "harbor_final_archive",
    }


def prefer_complete_session(
    previous: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Choose a byte prefix or a request-ledger-verified semantic superset."""
    if previous["combined_session_sha256"] == candidate["combined_session_sha256"]:
        return max(
            (previous, candidate),
            key=lambda source: (
                int(source.get("request_count") or 0),
                len(Path(source["_session_path"]).read_bytes()),
            ),
        )
    previous_bytes = Path(previous["_session_path"]).read_bytes()
    candidate_bytes = Path(candidate["_session_path"]).read_bytes()
    shorter, longer = sorted(
        ((previous_bytes, previous), (candidate_bytes, candidate)),
        key=lambda item: len(item[0]),
    )
    if longer[0].startswith(shorter[0]):
        return longer[1]

    # Harbor scrubs secrets and run-specific paths from its final archive, so
    # that trusted copy need not be byte-prefix comparable to the raw durable
    # mirror.  Request IDs plus every billing field remain stable across that
    # transformation.  Accept only a strict/equal request-ledger superset with
    # byte-for-byte-equivalent shared request objects; any divergence still
    # fails closed.
    def request_map(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(row["run_api_call_id"]): row
            for row in source.get("requests") or []
            if isinstance(row, dict) and row.get("run_api_call_id")
        }

    previous_requests = request_map(previous)
    candidate_requests = request_map(candidate)

    def dominating_harbor_final(final: dict[str, Any], durable: dict[str, Any]) -> bool:
        if final.get("origin") != "harbor_final_archive":
            return False
        if final.get("session_id") != durable.get("session_id"):
            return False
        final_rows = [
            row for row in final.get("requests") or [] if isinstance(row, dict)
        ]
        durable_rows = [
            row for row in durable.get("requests") or [] if isinstance(row, dict)
        ]
        if len(final_rows) < len(durable_rows):
            return False
        fields = (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        )
        return all(
            sum(int(row.get(field) or 0) for row in final_rows)
            >= sum(int(row.get(field) or 0) for row in durable_rows)
            for field in fields
        )

    # Codex's local api_call_N identifiers are parser ordinals, not provider
    # request IDs. A missing event in the durable mirror can shift every later
    # ordinal even when Harbor's immutable final archive is the strict semantic
    # superset. Accept only a same-session final archive that dominates the
    # durable mirror in request count and every cumulative billing bucket.
    if dominating_harbor_final(candidate, previous):
        return candidate
    if dominating_harbor_final(previous, candidate):
        return previous
    common = set(previous_requests) & set(candidate_requests)
    if any(previous_requests[key] != candidate_requests[key] for key in common):
        raise SystemExit(
            f"conflicting request records for Codex session {candidate.get('session_id')}"
        )
    previous_ids = set(previous_requests)
    candidate_ids = set(candidate_requests)
    if previous_ids <= candidate_ids:
        return candidate
    if candidate_ids <= previous_ids:
        return previous
    raise SystemExit(
        f"incomparable request ledgers for Codex session {candidate.get('session_id')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    state_dir = args.state_dir.resolve()
    run = json.loads((state_dir / "run.json").read_text())
    pricing_snapshot = run.get("api_pricing_snapshot")
    if isinstance(pricing_snapshot, dict):
        os.environ["SPRINT_DEEPSEEK_PRICING_SNAPSHOT"] = json.dumps(
            pricing_snapshot, separators=(",", ":"), sort_keys=True
        )
    groups = source_groups(state_dir)
    sources = [
        reconstruct_group(
            state_dir=state_dir,
            run=run,
            attempt=attempt,
            source_id=source_id,
            chunks=chunks,
        )
        for attempt, source_id, chunks in groups
    ]
    final_source = harbor_final_source(state_dir=state_dir, run=run)
    if final_source is not None:
        sources.append(final_source)
    if not sources:
        raise SystemExit("no durable or Harbor Codex session is available")
    unique_sessions: dict[str, dict[str, Any]] = {}
    anonymous = []
    for source in sources:
        session_id = source.get("session_id")
        if session_id:
            previous = unique_sessions.get(str(session_id))
            unique_sessions[str(session_id)] = (
                prefer_complete_session(previous, source) if previous else source
            )
        else:
            anonymous.append(source)
    sessions = list(unique_sessions.values()) + anonymous
    requests = [request for source in sessions for request in source["requests"]]
    provider_cost_source = (run.get("budget_enforcement") or {}).get(
        "api_cost_source"
    ) == "openrouter_reported_per_request"
    provider_records = (
        provider_usage_records(state_dir, str(run["run_id"]))
        if provider_cost_source
        else []
    )
    if provider_cost_source:
        apply_provider_reported_costs(requests, provider_records)
        # A provider request can finish after the Codex process is interrupted,
        # leaving no local token_count event to carry the configured identity.
        # Keep the provider response model separately, but attest the benchmark
        # model and reasoning configuration on the synthetic run-audit row.
        expected_model = str(run["model"]).split("/", 1)[-1]
        for request in requests:
            if request.get("provider_only_usage") is True:
                request["model"] = expected_model
                request["reasoning_effort"] = run["reasoning_effort"]
    request_ids = [request["run_api_call_id"] for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise SystemExit("duplicate run-level Codex API call IDs")
    snapshots: dict[str, dict[str, Any]] = {}
    for source in sessions:
        for snapshot in source["pricing_snapshots"]:
            if isinstance(snapshot, dict) and snapshot.get("id"):
                snapshots[str(snapshot["id"])] = snapshot
    complete = (
        True
        if provider_cost_source
        else all(source["cost_reconstruction_complete"] for source in sessions)
    )
    expected_attempts = {
        int(row["attempt"])
        for row in run.get("cpu_launch_history") or []
        if isinstance(row, dict) and row.get("attempt") is not None
    }
    captured_attempts = {int(source["cpu_attempt"]) for source in sessions}
    payload = {
        "schema_version": USAGE_AUDIT_SCHEMA_VERSION,
        "run_id": run["run_id"],
        "model": run["model"],
        "resolved_model_version": run.get("resolved_model_version"),
        "reasoning_effort": run["reasoning_effort"],
        "source": "durable_codex_session_chunks",
        "source_sessions": [
            {
                key: value
                for key, value in source.items()
                if key not in {"requests", "_session_path"}
            }
            for source in sessions
        ],
        "expected_cpu_attempts": sorted(expected_attempts),
        "captured_cpu_attempts": sorted(captured_attempts),
        "attempt_coverage_complete": expected_attempts <= captured_attempts,
        "request_count": len(requests),
        "requests": requests,
        "pricing_snapshots": list(snapshots.values()),
        "cost_reconstruction_complete": complete,
        "calculated_api_usage_usd": (
            sum(float(request["calculated_cost_usd"]) for request in requests)
            if complete
            else None
        ),
        "calculated_api_usage_cost_basis": (
            (run.get("budget_enforcement") or {}).get(
                "api_budget_cost_basis",
                "openrouter_list_price_before_endpoint_discount",
            )
            if provider_cost_source
            else next(
                iter(
                    {
                        str(source["calculated_api_usage_cost_basis"])
                        for source in sessions
                        if source.get("calculated_api_usage_cost_basis")
                    }
                ),
                None,
            )
        ),
        "provider_billed_api_usage_usd": (
            sum(
                float(record["provider_reported_cost_usd"])
                for record in provider_records
            )
            if provider_cost_source
            else None
        ),
        "promotion_savings_usd": (
            sum(
                float(
                    record.get(
                        "benchmark_cost_usd",
                        record.get(
                            "undiscounted_cost_usd",
                            record["provider_reported_cost_usd"],
                        ),
                    )
                )
                - float(record["provider_reported_cost_usd"])
                for record in provider_records
            )
            if provider_cost_source
            else None
        ),
        "provider_billing_reconciled": provider_cost_source,
        "invoice_exact": False,
    }
    if complete and not requests:
        payload["zero_request_reason"] = (
            "no completed model request was present in any captured CPU attempt"
        )
    atomic_text(
        state_dir / "usage" / "run-usage-audit.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    if not args.quiet:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["cost_reconstruction_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
