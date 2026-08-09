#!/usr/bin/env python3
"""Build a web-safe, single-clock experiment timeline from durable run data.

Raw agent traces remain private.  The exported trace lane contains one record
per timestamped native event plus tool name/call identifiers, never prompts,
tool arguments, tool output, environment variables, or message text.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

import modal_cost

SCHEMA_VERSION = 6
DEFAULT_BUCKET_SECONDS = 60
PUBLIC_RUN_LIMIT = 6
DEFAULT_GPU_MAX_GAP_SECONDS = 45
ISO_KEYS = ("timestamp", "ts_utc", "created_at", "at", "submitted_at")
RESOURCE_ROLES = {"cpu-agent", "training-gpu", "verifier-gpu", "host-controller"}
GPU_PIPELINE_FIELDS = (
    "sm_active_pct",
    "sm_occupancy_pct",
    "tensor_pipe_active_pct",
    "fp32_fma_pipe_active_pct",
    "fp16_instruction_pct_of_peak_active",
    "dram_throughput_pct",
)


def parse_epoch_ms(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        number = float(value)
        return int(number if number > 10_000_000_000 else number * 1000)
    if not isinstance(value, str) or not value:
        return None
    try:
        return int(
            dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000
        )
    except ValueError:
        return None


def iso_from_ms(epoch_ms: int) -> str:
    return (
        dt.datetime.fromtimestamp(epoch_ms / 1000, dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, payload: dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, separators=(",", ": "))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def read_jsonl(path: pathlib.Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return rows, 1
    with handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict):
                row["_line_number"] = line_number
                rows.append(row)
            else:
                malformed += 1
    return rows, malformed


def cpu_attempt_for(path: pathlib.Path, state_dir: pathlib.Path) -> int:
    try:
        relative = path.relative_to(state_dir)
    except ValueError:
        return 1
    parts = relative.parts
    if "cpu-attempts" in parts:
        index = parts.index("cpu-attempts")
        if index + 1 < len(parts) and parts[index + 1].isdigit():
            return int(parts[index + 1])
    match = re.search(r"cpu-attempt-(\d+)", str(relative))
    return int(match.group(1)) if match else 1


def trial_dirs(state_dir: pathlib.Path, run: dict[str, Any]) -> list[pathlib.Path]:
    roots: set[pathlib.Path] = {state_dir / "harbor-jobs"}
    roots.update(state_dir.glob("cpu-attempts/*/harbor-jobs"))
    for row in run.get("cpu_launch_history") or []:
        value = row.get("jobs_root") if isinstance(row, dict) else None
        if isinstance(value, str):
            roots.add(pathlib.Path(value))
    trials: set[pathlib.Path] = set()
    run_id = str(run.get("run_id") or state_dir.name)
    for root in roots:
        job = root / run_id
        if job.is_dir():
            trials.update(path for path in job.iterdir() if path.is_dir())
    return sorted(trials)


def safe_attempt_name(name: Any) -> str:
    """Match Harbor's stable continuous-attempt directory naming."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(name))


class Builder:
    def __init__(
        self, state_dir: pathlib.Path, run: dict[str, Any], bucket_seconds: int
    ) -> None:
        self.state_dir = state_dir
        self.run = run
        self.run_id = str(run.get("run_id") or state_dir.name)
        self.bucket_ms = bucket_seconds * 1000
        self.events: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.counts: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self._event_ids: set[str] = set()
        self._trace_record_hashes: set[str] = set()
        self._metric_keys: set[str] = set()
        self._artifact_ids: set[str] = set()
        self.final_verifier_rewards: dict[str, Any] = {}
        self.final_verifier_finished_at: str | None = None

    def relative(self, path: pathlib.Path) -> str:
        try:
            return str(path.resolve().relative_to(self.state_dir.resolve()))
        except ValueError:
            return path.name

    def add_event(
        self,
        *,
        epoch_ms: int | None,
        category: str,
        kind: str,
        source: str,
        identity: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        if epoch_ms is None:
            self.counts["events_without_timestamp"] += 1
            return
        event_id = hashlib.sha256(
            f"{self.run_id}\0{category}\0{kind}\0{source}\0{identity}".encode()
        ).hexdigest()[:20]
        if event_id in self._event_ids:
            return
        self._event_ids.add(event_id)
        event: dict[str, Any] = {
            "id": event_id,
            "epoch_ms": epoch_ms,
            "ts": iso_from_ms(epoch_ms),
            "category": category,
            "kind": kind,
            "source": source,
        }
        if data:
            event.update(data)
        self.events.append(event)
        self.counts[f"event:{category}"] += 1
        self.counts[f"kind:{kind}"] += 1

    def add_run_lifecycle(self) -> None:
        created = parse_epoch_ms(self.run.get("created_at"))
        self.add_event(
            epoch_ms=created,
            category="infrastructure",
            kind="run_created",
            source="run.json",
            identity="created",
            data={"cpu_attempt": 1},
        )
        for row in self.run.get("cpu_launch_history") or []:
            if not isinstance(row, dict):
                continue
            attempt = int(row.get("attempt") or 0)
            self.add_event(
                epoch_ms=parse_epoch_ms(row.get("launched_at")),
                category="infrastructure",
                kind="cpu_allocated" if attempt <= 1 else "cpu_reallocated",
                source="run.json",
                identity=f"cpu:{attempt}:{row.get('launched_at')}",
                data={"cpu_attempt": attempt},
            )
        for name, kind, key in (
            ("STOP_REQUESTED.json", "stop_requested", "requested_at"),
            ("STOP_ACK.json", "stop_acknowledged", "acknowledged_at"),
            ("FINALIZED.json", "run_finalized", "finalized_at"),
        ):
            path = self.state_dir / name
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            self.add_event(
                epoch_ms=parse_epoch_ms(payload.get(key)),
                category="infrastructure",
                kind=kind,
                source=name,
                identity=json.dumps(payload, sort_keys=True),
                data=(
                    {"cpu_attempt": int(self.run.get("cpu_launch_attempt") or 1)}
                    if kind == "stop_acknowledged"
                    else None
                ),
            )

    def add_cpu_lifecycle(self) -> None:
        paths = [self.state_dir / "telemetry" / "cpu_lifecycle.jsonl"]
        paths.extend(self.state_dir.glob("recovery/*/*/telemetry/cpu_lifecycle.jsonl"))
        for path in sorted({path for path in paths if path.is_file()}):
            rows, malformed = read_jsonl(path)
            self.counts["malformed_cpu_lifecycle"] += malformed
            self.source_counts["cpu_lifecycle_files"] += 1
            for row in rows:
                event = row.get("event")
                if event not in {"cpu_launch_started", "cpu_launch_exited"}:
                    self.counts["unknown_cpu_lifecycle_events"] += 1
                    continue
                self.add_event(
                    epoch_ms=parse_epoch_ms(row.get("at")),
                    category="infrastructure",
                    kind=(
                        "cpu_allocation_requested"
                        if event == "cpu_launch_started"
                        else "cpu_interrupted"
                    ),
                    source=self.relative(path),
                    identity=f"{row.get('attempt')}:{event}:{row.get('at')}",
                    data={
                        "cpu_attempt": row.get("attempt"),
                        "exit_code": row.get("exit_code"),
                    },
                )

    def telemetry_files(self, trials: Iterable[pathlib.Path]) -> list[pathlib.Path]:
        paths = [
            self.state_dir / "telemetry" / "host-samples.jsonl",
            self.state_dir / "telemetry" / "durable-samples.jsonl",
            self.state_dir / "telemetry" / "durable-gpu-samples.jsonl",
        ]
        paths.extend(self.state_dir.glob("telemetry/durable-by-job/*/samples.jsonl"))
        for trial in trials:
            root = trial / "artifacts" / "logs" / "artifacts" / "telemetry"
            paths.extend((root / "samples.jsonl", root / "host-samples.jsonl"))
            paths.append(trial / "verifier" / "telemetry" / "samples.jsonl")
            paths.extend(
                (trial / "artifacts" / "continuous" / "attempts").glob(
                    "*/verifier/telemetry/samples.jsonl"
                )
            )
        paths.extend(self.state_dir.glob("recovery/*/*/telemetry/samples.jsonl"))
        paths.extend(self.state_dir.glob("recovery/*/*/telemetry/host-samples.jsonl"))
        return sorted({path for path in paths if path.is_file()})

    def verifier_evaluation_id(
        self, path: pathlib.Path, trials: Iterable[pathlib.Path]
    ) -> str | None:
        for trial in trials:
            if path.is_relative_to(trial / "verifier" / "telemetry"):
                return f"final:{self.relative(trial)}"
            attempts = trial / "artifacts" / "continuous" / "attempts"
            if path.is_relative_to(attempts):
                relative = path.relative_to(attempts)
                if relative.parts:
                    return f"continuous:{self.relative(attempts / relative.parts[0])}"
        return None

    def add_telemetry(self, trials: list[pathlib.Path]) -> None:
        for path in self.telemetry_files(trials):
            rows, malformed = read_jsonl(path)
            self.counts["malformed_telemetry"] += malformed
            source = self.relative(path)
            evaluation_id = self.verifier_evaluation_id(path, trials)
            self.source_counts["telemetry_files"] += 1
            for row in rows:
                epoch_ms = parse_epoch_ms(row.get("epoch_s")) or parse_epoch_ms(
                    row.get("ts_utc")
                )
                source_role = str(row.get("role") or "unknown")
                role = source_role if source_role in RESOURCE_ROLES else "unknown"
                gpus = row.get("gpus") if isinstance(row.get("gpus"), list) else []
                reported_role = None
                if role == "unknown":
                    self.counts["unknown_metric_roles"] += 1
                # Never infer ownership from an impossible sample. Attribution
                # must come from the producer's explicit current-schema role.
                if (
                    role == "cpu-agent"
                    and int(self.run.get("agent_gpus") or 0) == 0
                    and gpus
                ):
                    reported_role = source_role
                    role = "unknown"
                    self.counts["impossible_cpu_agent_gpu_samples"] += 1
                key_data = {
                    "epoch_ms": epoch_ms,
                    "role": role,
                    "container_id": row.get("container_id"),
                    "hostname": row.get("hostname"),
                    "job_id": row.get("job_id"),
                    "cpu_attempt": row.get("cpu_attempt"),
                    "attempt": row.get("attempt"),
                    "lease_id": row.get("lease_id"),
                    "sample_index": row.get("sample_index"),
                }
                key = json.dumps(key_data, sort_keys=True)
                if key in self._metric_keys:
                    continue
                self._metric_keys.add(key)
                metrics: dict[str, Any] = {}
                for name in (
                    "resource_accounting_scope",
                    "cpu_requested_cores",
                    "cpu_limit_cores",
                    "cpu_usage_cores",
                    "cpu_usage_usec",
                    "cpu_user_usec",
                    "cpu_system_usec",
                    "cpu_nr_throttled",
                    "cpu_throttled_usec",
                    "cpu_util_pct",
                    "load1",
                    "load5",
                    "load15",
                    "mem_requested_kib",
                    "mem_limit_kib",
                    "mem_used_kib",
                    "mem_total_kib",
                    "mem_available_kib",
                    "memory_peak_kib",
                    "memory_oom_events",
                    "memory_oom_kill_events",
                    "net_rx_bytes",
                    "net_tx_bytes",
                    "disk_root_used_pct",
                    "pcie_rx_mib_s",
                    "pcie_tx_mib_s",
                ):
                    if row.get(name) is not None:
                        metrics[name] = row[name]
                compact_gpus = []
                for gpu in gpus:
                    if not isinstance(gpu, dict):
                        continue
                    compact_gpus.append(
                        {
                            name: gpu.get(name)
                            for name in (
                                "gpu_index",
                                "gpu_uuid",
                                "gpu_name",
                                "util_gpu_pct",
                                "util_mem_pct",
                                "mem_used_mib",
                                "mem_total_mib",
                                "power_draw_w",
                                "power_limit_w",
                                "temp_gpu_c",
                                "clock_sm_mhz",
                                "clock_mem_mhz",
                                "pipeline_metrics_source",
                                "pipeline_metrics_group",
                                "pipeline_metrics_status",
                                "pipeline_metrics_sample_count",
                                "pipeline_metrics_window_ms",
                                "pipeline_metrics_error",
                                *GPU_PIPELINE_FIELDS,
                            )
                            if gpu.get(name) is not None
                        }
                    )
                if compact_gpus:
                    metrics["gpus"] = compact_gpus
                self.add_event(
                    epoch_ms=epoch_ms,
                    category="metrics",
                    kind="resource_sample",
                    source=source,
                    identity=key,
                    data={
                        "role": role,
                        **({"reported_role": reported_role} if reported_role else {}),
                        "container_id": row.get("container_id"),
                        "hostname": row.get("hostname"),
                        "cpu_attempt": (
                            row.get("cpu_attempt")
                            if row.get("cpu_attempt") is not None
                            else cpu_attempt_for(path, self.state_dir)
                        ),
                        "gpu_job_id": row.get("job_id"),
                        "gpu_attempt": row.get("attempt"),
                        "lease_id": row.get("lease_id"),
                        "evaluation_id": evaluation_id,
                        "metrics": metrics,
                    },
                )
                self.counts[f"metric_role:{role}"] += 1
                if compact_gpus:
                    self.counts["gpu_metric_samples"] += 1
                    self.counts[f"gpu_metric_role:{role}"] += 1

    def add_verifier_gpu_lifecycle(self, trials: list[pathlib.Path]) -> None:
        """Read trusted sampler boundaries from each archived SCORE sandbox."""
        paths: list[pathlib.Path] = []
        for trial in trials:
            paths.append(trial / "verifier" / "telemetry" / "lifecycle.json")
            paths.extend(
                (trial / "artifacts" / "continuous" / "attempts").glob(
                    "*/verifier/telemetry/lifecycle.json"
                )
            )
        for path in paths:
            if not path.is_file():
                continue
            self.source_counts["verifier_gpu_lifecycle_files"] += 1
            evaluation_id = self.verifier_evaluation_id(path, trials)
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                self.counts["malformed_verifier_gpu_lifecycle"] += 1
                continue
            common = {
                "evaluation_id": evaluation_id,
                "evaluation_scope": (
                    "continuous"
                    if evaluation_id and evaluation_id.startswith("continuous:")
                    else "final"
                ),
            }
            self.add_event(
                epoch_ms=parse_epoch_ms(payload.get("started_at")),
                category="infrastructure",
                kind="verifier_gpu_started",
                source=self.relative(path),
                identity=f"{evaluation_id}:started:{payload.get('started_at')}",
                data=common,
            )
            self.add_event(
                epoch_ms=parse_epoch_ms(payload.get("finished_at")),
                category="infrastructure",
                kind="verifier_gpu_stopped",
                source=self.relative(path),
                identity=f"{evaluation_id}:stopped:{payload.get('finished_at')}",
                data={**common, "sampler_complete": bool(payload.get("complete"))},
            )

    def timeline_files(self, trials: Iterable[pathlib.Path]) -> list[pathlib.Path]:
        paths = [
            self.state_dir / "telemetry" / "gpu_timeline.jsonl",
            self.state_dir / "telemetry" / "durable-gpu-timeline.jsonl",
        ]
        for trial in trials:
            paths.append(
                trial
                / "artifacts"
                / "logs"
                / "artifacts"
                / "telemetry"
                / "gpu_timeline.jsonl"
            )
        return [path for path in paths if path.is_file()]

    def add_gpu_lifecycle(self, trials: list[pathlib.Path]) -> None:
        files = self.timeline_files(trials)
        self.source_counts["gpu_lifecycle_files"] = len(files)
        seen_event_ids: set[str] = set()
        for path in files:
            rows, malformed = read_jsonl(path)
            self.counts["malformed_gpu_lifecycle"] += malformed
            for row in rows:
                event_id = str(row.get("event_id") or "")
                if event_id and event_id in seen_event_ids:
                    continue
                if event_id:
                    seen_event_ids.add(event_id)
                detail = (
                    row.get("detail") if isinstance(row.get("detail"), dict) else {}
                )
                semantic = detail.get("event")
                phase = str(row.get("phase") or "gpu")
                action = str(row.get("action") or "event")
                kind = str(semantic or f"{phase}_{action}")
                self.add_event(
                    epoch_ms=parse_epoch_ms(row.get("epoch_s"))
                    or parse_epoch_ms(row.get("ts_utc")),
                    category="infrastructure",
                    kind=kind,
                    source=self.relative(path),
                    identity=str(event_id or json.dumps(row, sort_keys=True)),
                    data={
                        "gpu_job_id": row.get("job_id"),
                        "gpu_attempt": row.get("attempt"),
                        "lease_id": row.get("lease_id"),
                        "phase": phase,
                        "action": action,
                        "reason": detail.get("reason"),
                    },
                )

        self.add_gpu_attempt_lifecycle()
        self.add_gpu_registry_lifecycle()
        self.close_orphaned_gpu_lifecycle()

    def add_gpu_attempt_lifecycle(self) -> None:
        """Recover exact worker terminal boundaries from durable attempts."""
        terminal_kinds = {"gpu_preempted", "gpu_released"}
        existing = {
            (event.get("gpu_job_id"), event.get("gpu_attempt"))
            for event in self.events
            if event["kind"] in terminal_kinds
        }
        root = self.state_dir / "telemetry" / "durable-gpu-attempts"
        for path in sorted(root.glob("*/*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                self.counts["malformed_gpu_attempt_record"] += 1
                continue
            job_id = payload.get("job_id")
            attempt = payload.get("attempt")
            finished_at = payload.get("finished_at")
            if (
                not isinstance(job_id, str)
                or not job_id
                or not isinstance(attempt, int)
                or attempt <= 0
                or not finished_at
            ):
                self.counts["malformed_gpu_attempt_record"] += 1
                continue
            key = (job_id, attempt)
            if key in existing:
                continue
            status = str(payload.get("status") or "worker_terminal")
            kind = (
                "gpu_preempted"
                if status in {"interrupted", "lost", "preempted", "fenced"}
                else "gpu_released"
            )
            self.add_event(
                epoch_ms=parse_epoch_ms(finished_at),
                category="infrastructure",
                kind=kind,
                source=self.relative(path),
                identity=f"attempt:{job_id}:{attempt}:{finished_at}:{kind}",
                data={
                    "gpu_job_id": job_id,
                    "gpu_attempt": attempt,
                    "lease_id": payload.get("lease_id"),
                    "reason": status,
                    "exit_code": payload.get("exit_code"),
                    "lifecycle_recovered": True,
                    "lifecycle_recovery_source": "durable_worker_attempt",
                },
            )
            existing.add(key)
            self.counts["gpu_attempt_terminal_events"] += 1

    def add_gpu_registry_lifecycle(self) -> None:
        """Close allocations from the append-only host-owned job registry.

        The worker stream remains the primary lifecycle source.  The registry
        is an independent host record that survives agent cleanup and abrupt
        worker loss, so it is used only when the primary stream lacks a
        terminal event for an allocated attempt.
        """
        terminal_kinds = {"gpu_preempted", "gpu_released"}
        existing = {
            (event.get("gpu_job_id"), event.get("gpu_attempt"))
            for event in self.events
            if event["kind"] in terminal_kinds
        }
        registry_root = self.state_dir / "gpu-job-registry"
        for path in sorted(registry_root.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                self.counts["malformed_gpu_job_registry"] += 1
                continue
            job_id = payload.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                self.counts["malformed_gpu_job_registry"] += 1
                continue
            records = list(payload.get("attempt_history") or [])
            records.append(payload)
            for record in records:
                if not isinstance(record, dict):
                    continue
                attempt = record.get("attempt")
                finished_at = record.get("finished_at") or record.get("terminated_at")
                if not isinstance(attempt, int) or attempt <= 0 or not finished_at:
                    continue
                key = (job_id, attempt)
                if key in existing:
                    continue
                reason = str(
                    record.get("reason")
                    or record.get("termination_reason")
                    or record.get("status")
                    or "registry_terminal"
                )
                kind = (
                    "gpu_preempted"
                    if reason
                    in {
                        "graceful_preemption",
                        "worker_lost",
                        "lost",
                        "preempted",
                    }
                    else "gpu_released"
                )
                self.add_event(
                    epoch_ms=parse_epoch_ms(finished_at),
                    category="infrastructure",
                    kind=kind,
                    source=self.relative(path),
                    identity=f"registry:{job_id}:{attempt}:{finished_at}:{kind}",
                    data={
                        "gpu_job_id": job_id,
                        "gpu_attempt": attempt,
                        "lease_id": record.get("lease_id"),
                        "reason": reason,
                        "lifecycle_recovered": True,
                        "lifecycle_recovery_source": "host_job_registry",
                    },
                )
                existing.add(key)
                self.counts["gpu_registry_terminal_events"] += 1

    def close_orphaned_gpu_lifecycle(self) -> None:
        """Conservatively close legacy pre-registry allocation intervals.

        Older workers could lose their canonical release record. A matching
        worker-reported active exit is still an exact terminal boundary.
        """
        starts = sorted(
            (
                event
                for event in self.events
                if event["kind"] in {"gpu_allocated", "gpu_reallocated"}
            ),
            key=lambda event: event["epoch_ms"],
        )
        terminal_kinds = {"gpu_preempted", "gpu_released"}
        terminal_keys = {
            (event.get("gpu_job_id"), event.get("gpu_attempt"))
            for event in self.events
            if event["kind"] in terminal_kinds
        }
        active_exits: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
        for event in self.events:
            if event["kind"] == "gpu_active_exit":
                active_exits[
                    (event.get("gpu_job_id"), event.get("gpu_attempt"))
                ].append(event)
        for start in starts:
            key = (start.get("gpu_job_id"), start.get("gpu_attempt"))
            if key in terminal_keys:
                continue
            exits = [
                event
                for event in active_exits.get(key, [])
                if event["epoch_ms"] >= start["epoch_ms"]
            ]
            if exits:
                end = min(exits, key=lambda event: event["epoch_ms"])
                boundary = "worker_reported_active_exit"
            else:
                continue
            self.add_event(
                epoch_ms=end["epoch_ms"],
                category="infrastructure",
                kind="gpu_released",
                source=end["source"],
                identity=(
                    f"recovered-release:{key[0]}:{key[1]}:{end['epoch_ms']}:{boundary}"
                ),
                data={
                    "gpu_job_id": key[0],
                    "gpu_attempt": key[1],
                    "reason": "legacy_lifecycle_recovery",
                    "lifecycle_recovered": True,
                    "lifecycle_recovery_source": boundary,
                    "end_is_upper_bound": False,
                },
            )
            terminal_keys.add(key)
            self.counts["gpu_inferred_terminal_events"] += 1

    @staticmethod
    def _record_timestamp(record: dict[str, Any]) -> int | None:
        for key in ISO_KEYS:
            value = parse_epoch_ms(record.get(key))
            if value is not None:
                return value
        payload = record.get("payload")
        if isinstance(payload, dict):
            for key in ISO_KEYS:
                value = parse_epoch_ms(payload.get(key))
                if value is not None:
                    return value
        return None

    @staticmethod
    def _trace_shape(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        outer = str(record.get("type") or "event")
        payload = (
            record.get("payload") if isinstance(record.get("payload"), dict) else {}
        )
        subtype = str(payload.get("type") or outer)
        data: dict[str, Any] = {"trace_type": outer, "trace_subtype": subtype}
        call_types = {"custom_tool_call", "function_call", "tool_use"}
        output_types = {
            "custom_tool_call_output",
            "function_call_output",
            "tool_result",
        }
        if subtype in call_types:
            tool = str(payload.get("name") or "unknown")
            data["tool"] = tool
            data["call_id"] = payload.get("call_id") or payload.get("id")
            if tool == "write_stdin":
                raw_arguments = payload.get("arguments", payload.get("input"))
                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else raw_arguments
                    )
                except json.JSONDecodeError:
                    arguments = None
                if isinstance(arguments, dict):
                    session_id = arguments.get("session_id")
                    if isinstance(session_id, (str, int)):
                        data["shell_session_id"] = str(session_id)
            return "tool_call", data
        if subtype in output_types:
            data["call_id"] = payload.get("call_id") or payload.get("tool_use_id")
            output = payload.get("output")
            if isinstance(output, str):
                session_match = re.search(
                    r"Process running with session ID ([A-Za-z0-9_-]+)", output
                )
                if session_match:
                    data["shell_session_id"] = session_match.group(1)
                    data["shell_process_running"] = True
                exit_match = re.search(r"Process exited with code (-?\d+)", output)
                if exit_match:
                    data["shell_process_exit_code"] = int(exit_match.group(1))
                    data["shell_process_running"] = False
                wall_match = re.search(r"Wall time: ([0-9.]+) seconds", output)
                if wall_match:
                    data["reported_tool_wall_ms"] = round(
                        float(wall_match.group(1)) * 1000
                    )
                if output.startswith("aborted by user"):
                    data["shell_process_aborted"] = True
                    data["shell_process_running"] = False
            return "tool_result", data
        message = (
            record.get("message") if isinstance(record.get("message"), dict) else {}
        )
        content = (
            message.get("content") if isinstance(message.get("content"), list) else []
        )
        tools = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if tools:
            first = tools[0]
            data["tool"] = str(first.get("name") or "unknown")
            data["call_id"] = first.get("id")
            data["tool_calls_in_record"] = len(tools)
            return "tool_call", data
        role = message.get("role") or payload.get("role")
        if role:
            data["role"] = role
        return "agent_event", data

    def trace_files(self, trials: Iterable[pathlib.Path]) -> list[pathlib.Path]:
        candidates: list[pathlib.Path] = []
        for trial in trials:
            native = list((trial / "agent" / "sessions").rglob("*.jsonl"))
            native.extend(
                (trial / "agent" / "codex-state" / "sessions").rglob("*.jsonl")
            )
            if native:
                candidates.extend(native)
            else:
                # Stdout is a fallback. It often contains protocol/status
                # records without timestamps in addition to the native trace.
                for name in ("claude-code.txt", "codex.txt"):
                    path = trial / "agent" / name
                    if path.is_file():
                        candidates.append(path)
        for root in (
            self.state_dir / "durable-trace",
            self.state_dir / "trace" / "raw",
        ):
            if root.is_dir():
                candidates.extend(root.rglob("*.jsonl"))
        candidates.extend(self.state_dir.glob("recovery/*/*/trace/raw/**/*.jsonl"))
        unique: list[pathlib.Path] = []
        file_hashes: set[str] = set()
        for path in sorted(set(candidates)):
            try:
                digest = sha256_file(path)
            except OSError:
                continue
            if digest in file_hashes:
                continue
            file_hashes.add(digest)
            unique.append(path)
        return unique

    def add_traces(self, trials: list[pathlib.Path]) -> None:
        files = self.trace_files(trials)
        self.source_counts["trace_files"] = len(files)
        for path in files:
            rows, malformed = read_jsonl(path)
            self.counts["malformed_trace_lines"] += malformed
            source = self.relative(path)
            attempt = cpu_attempt_for(path, self.state_dir)
            for row in rows:
                line_number = row.pop("_line_number", None)
                native = json.dumps(row, sort_keys=True, separators=(",", ":"))
                record_hash = hashlib.sha256(native.encode()).hexdigest()
                if record_hash in self._trace_record_hashes:
                    continue
                self._trace_record_hashes.add(record_hash)
                kind, data = self._trace_shape(row)
                data.update(
                    {
                        "cpu_attempt": attempt,
                        "raw_ref": {
                            "path": source,
                            "line": line_number,
                            "sha256": record_hash,
                        },
                    }
                )
                self.add_event(
                    epoch_ms=self._record_timestamp(row),
                    category="trace",
                    kind=kind,
                    source=source,
                    identity=record_hash,
                    data=data,
                )
                self.counts["native_trace_records"] += 1

    def add_usage_audits(self, trials: list[pathlib.Path]) -> None:
        """Expose web-safe per-request token and cost data on the same clock."""
        request_ids: set[str] = set()
        snapshot_ids: set[str] = set()
        run_audit = self.state_dir / "usage" / "run-usage-audit.json"
        sources = (
            [(run_audit, None)]
            if run_audit.is_file()
            else [(trial / "agent" / "usage-audit.json", trial) for trial in trials]
        )
        for path, trial in sources:
            try:
                audit = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            self.source_counts["usage_audit_files"] += 1
            audit_requests = audit.get("requests")
            audit_request_count = audit.get("request_count")
            calculated = audit.get("calculated_api_usage_usd")
            audit_complete = bool(
                audit.get("cost_reconstruction_complete") is True
                and isinstance(audit_requests, list)
                and audit_request_count == len(audit_requests)
                and isinstance(calculated, (int, float))
                and not isinstance(calculated, bool)
            )
            if audit_complete:
                self.counts["complete_usage_audits"] += 1
            if (
                audit_complete
                and audit_request_count == 0
                and float(calculated) == 0.0
                and audit.get("zero_request_reason")
                == "no completed model request was present in any captured CPU attempt"
            ):
                self.counts["attested_zero_request_usage_audits"] += 1
            default_session_id = self.relative(trial) if trial else self.run_id
            session_id = str(audit.get("session_id") or default_session_id)
            for snapshot in audit.get("pricing_snapshots") or []:
                if isinstance(snapshot, dict) and snapshot.get("id"):
                    snapshot_ids.add(str(snapshot["id"]))
            for request in audit.get("requests") or []:
                if not isinstance(request, dict):
                    self.counts["malformed_usage_requests"] += 1
                    continue
                api_call_id = str(request.get("api_call_id") or "")
                request_session_id = str(request.get("session_id") or session_id)
                identity = str(
                    request.get("run_api_call_id")
                    or f"{request_session_id}:{api_call_id}"
                )
                if not api_call_id or identity in request_ids:
                    self.counts["duplicate_or_unidentified_usage_requests"] += 1
                    continue
                request_ids.add(identity)
                data = {
                    key: request.get(key)
                    for key in (
                        "api_call_id",
                        "model",
                        "service_tier",
                        "reasoning_effort",
                        "model_context_window",
                        "input_tokens",
                        "ordinary_uncached_input_tokens",
                        "cached_input_tokens",
                        "cache_write_input_tokens",
                        "output_tokens",
                        "reasoning_output_tokens",
                        "total_tokens",
                        "long_context_pricing_applied",
                        "pricing_snapshot_id",
                        "calculated_cost_usd",
                        "cost_components_usd",
                        "cost_reconstruction_status",
                    )
                }
                data.update(
                    {
                        "session_id": request_session_id,
                        "cpu_attempt": (
                            request.get("cpu_attempt")
                            if request.get("cpu_attempt") is not None
                            else cpu_attempt_for(path, self.state_dir)
                        ),
                    }
                )
                self.add_event(
                    epoch_ms=parse_epoch_ms(request.get("usage_reported_at")),
                    category="usage",
                    kind="model_request_usage",
                    source=self.relative(path),
                    identity=identity,
                    data=data,
                )
                self.counts["model_requests"] += 1
                if request.get("cost_reconstruction_status") != "complete":
                    self.counts["incomplete_model_request_costs"] += 1
        self.counts["pricing_snapshots"] = len(snapshot_ids)

    def add_submissions(self, trials: list[pathlib.Path]) -> None:
        for trial in trials:
            attempt = cpu_attempt_for(trial, self.state_dir)
            uses_frozen_final = (
                self.run.get("primary_score_policy") == "frozen_final_artifact"
            )
            final_policy = trial / "artifacts" / "app" / "submission" / "policy.pt"
            try:
                final_policy_digest = (
                    sha256_file(final_policy) if uses_frozen_final else None
                )
            except OSError:
                final_policy_digest = None
            ledger = trial / "artifacts" / "continuous" / "ledger.jsonl"
            rows, malformed = read_jsonl(ledger) if ledger.is_file() else ([], 0)
            self.counts["malformed_ledger_lines"] += malformed
            if ledger.is_file():
                self.source_counts["ledger_files"] += 1
            matching_final_rows = []
            for candidate in rows:
                relative = candidate.get("artifact_path")
                candidate_path = (
                    trial / "artifacts" / str(relative)
                    if isinstance(relative, str)
                    else None
                )
                if (
                    final_policy_digest
                    and candidate_path is not None
                    and candidate_path.is_file()
                    and sha256_file(candidate_path) == final_policy_digest
                    and isinstance(candidate.get("rewards"), dict)
                    and not candidate.get("error")
                ):
                    matching_final_rows.append(candidate)
            primary_row = (
                max(matching_final_rows, key=lambda row: int(row.get("index") or 0))
                if matching_final_rows
                else None
            )
            for row in rows:
                row.pop("_line_number", None)
                relative = row.get("artifact_path")
                artifact_path = (
                    trial / "artifacts" / str(relative)
                    if isinstance(relative, str)
                    else None
                )
                exists = bool(artifact_path and artifact_path.is_file())
                digest = (
                    sha256_file(artifact_path) if exists and artifact_path else None
                )
                primary_final = bool(
                    primary_row is row and digest == final_policy_digest
                )
                artifact_id = hashlib.sha256(
                    f"{self.relative(trial)}:{attempt}:{row.get('index')}:{row.get('name')}:{digest or 'missing'}".encode()
                ).hexdigest()
                if artifact_id not in self._artifact_ids:
                    self._artifact_ids.add(artifact_id)
                    artifact = {
                        "id": artifact_id,
                        "cpu_attempt": attempt,
                        "submission_index": row.get("index"),
                        "name": row.get("name"),
                        "sha256": digest,
                        "bytes": artifact_path.stat().st_size
                        if exists and artifact_path
                        else None,
                        "internal_path": self.relative(artifact_path)
                        if exists and artifact_path
                        else None,
                        "captured": exists,
                        "submitted_at": row.get("submitted_at"),
                        "finished_at": row.get("finished_at"),
                        "artifact_sha256_recorded": row.get("artifact_sha256"),
                        "evaluation_fingerprint": row.get("evaluation_fingerprint"),
                        "cache_hit": bool(row.get("cache_hit")),
                        "source_evaluation_id": row.get("source_evaluation_id"),
                        "verification_attempts": int(
                            row.get("verification_attempts") or 0
                        ),
                        "verification_retry_events": row.get(
                            "verification_retry_events"
                        )
                        if isinstance(row.get("verification_retry_events"), list)
                        else [],
                        "submission_origin": "agent_feedback_submit",
                        "rewards": row.get("rewards")
                        if isinstance(row.get("rewards"), dict)
                        else {},
                        "error": row.get("error"),
                    }
                    if uses_frozen_final:
                        artifact["primary_final"] = primary_final
                    self.artifacts.append(artifact)
                    self.counts["submission_artifacts"] += 1
                    if not exists:
                        self.counts["missing_submission_artifacts"] += 1
                common = {
                    "cpu_attempt": attempt,
                    "submission_index": row.get("index"),
                    "artifact_id": artifact_id,
                    "artifact_name": row.get("name"),
                    "evaluation_id": (
                        "continuous:"
                        + self.relative(
                            trial
                            / "artifacts"
                            / "continuous"
                            / "attempts"
                            / f"{int(row.get('index') or 0):04d}-{safe_attempt_name(row.get('name'))}"
                        )
                    ),
                    "evaluation_scope": "continuous",
                    "scoring_queue_key": self.run.get("scoring_queue_key")
                    or self.run_id,
                    "scheduler_wait_ms": (
                        round(float(row["scheduler_wait_sec"]) * 1000)
                        if isinstance(row.get("scheduler_wait_sec"), (int, float))
                        else None
                    ),
                    "cache_hit": bool(row.get("cache_hit")),
                    "evaluation_fingerprint": row.get("evaluation_fingerprint"),
                    "source_evaluation_id": row.get("source_evaluation_id"),
                    "verification_attempts": int(row.get("verification_attempts") or 0),
                }
                if uses_frozen_final:
                    common["primary_final"] = primary_final
                start_key = (
                    "scheduler_acquired_at"
                    if row.get("cache_hit")
                    else "verification_started_at"
                    if row.get("verification_started_at")
                    else "started_at"
                )
                start_kind = (
                    "evaluation_cache_hit"
                    if row.get("cache_hit")
                    else "evaluation_started"
                )
                for key, kind in (
                    ("submitted_at", "artifact_submitted"),
                    (start_key, start_kind),
                    ("finished_at", "evaluation_finished"),
                ):
                    data = dict(common)
                    if kind == "evaluation_finished":
                        data["rewards"] = (
                            row.get("rewards")
                            if isinstance(row.get("rewards"), dict)
                            else {}
                        )
                        data["error"] = row.get("error")
                    if kind in {"evaluation_started", "evaluation_cache_hit"}:
                        submitted_ms = parse_epoch_ms(row.get("submitted_at"))
                        started_ms = parse_epoch_ms(row.get(key))
                        data["queue_wait_ms"] = (
                            started_ms - submitted_ms
                            if submitted_ms is not None and started_ms is not None
                            else None
                        )
                    self.add_event(
                        epoch_ms=parse_epoch_ms(row.get(key)),
                        category="artifact",
                        kind=kind,
                        source=self.relative(ledger),
                        identity=f"{artifact_id}:{kind}:{row.get(key)}",
                        data=data,
                    )
                retry_events = row.get("verification_retry_events")
                if isinstance(retry_events, list):
                    for retry in retry_events:
                        if not isinstance(retry, dict):
                            continue
                        data = dict(common)
                        data.update(
                            {
                                "failed_attempt": retry.get("attempt"),
                                "error_type": retry.get("error_type"),
                                "error": retry.get("error"),
                            }
                        )
                        self.add_event(
                            epoch_ms=parse_epoch_ms(retry.get("failed_at")),
                            category="infrastructure",
                            kind="evaluation_retry",
                            source=self.relative(ledger),
                            identity=(
                                f"{artifact_id}:evaluation_retry:"
                                f"{retry.get('attempt')}:{retry.get('failed_at')}"
                            ),
                            data=data,
                        )
            if uses_frozen_final and final_policy_digest and primary_row is None:
                artifact_id = hashlib.sha256(
                    f"{self.relative(trial)}:{attempt}:host-final:{final_policy_digest}".encode()
                ).hexdigest()
                if artifact_id not in self._artifact_ids:
                    self._artifact_ids.add(artifact_id)
                    self.artifacts.append(
                        {
                            "id": artifact_id,
                            "cpu_attempt": attempt,
                            "submission_index": None,
                            "name": "policy.pt",
                            "sha256": final_policy_digest,
                            "bytes": final_policy.stat().st_size,
                            "internal_path": self.relative(final_policy),
                            "captured": True,
                            "submitted_at": None,
                            "finished_at": None,
                            "artifact_sha256_recorded": final_policy_digest,
                            "evaluation_fingerprint": None,
                            "cache_hit": False,
                            "source_evaluation_id": None,
                            "submission_origin": "host_frozen_final",
                            "primary_final": True,
                            "rewards": {},
                            "error": None,
                        }
                    )
                    self.counts["submission_artifacts"] += 1

    def add_final_verification(self, trials: list[pathlib.Path]) -> None:
        """Add the sealed final verifier as a distinct verifier-GPU interval."""
        for trial in trials:
            result_path = trial / "result.json"
            try:
                result = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            verifier = result.get("verifier")
            if not isinstance(verifier, dict):
                continue
            finished_at = verifier.get("finished_at")
            if isinstance(finished_at, str) and (
                self.final_verifier_finished_at is None
                or finished_at > self.final_verifier_finished_at
            ):
                verifier_result = result.get("verifier_result") or {}
                rewards = verifier_result.get("rewards")
                if isinstance(rewards, dict):
                    self.final_verifier_rewards = rewards
                    self.final_verifier_finished_at = finished_at
                    agent = result.get("agent") or {}
                    for artifact in self.artifacts:
                        if artifact.get("primary_final"):
                            artifact["rewards"] = rewards
                            artifact["finished_at"] = finished_at
                            if artifact.get("submitted_at") is None:
                                artifact["submitted_at"] = agent.get("finished_at")
            evaluation_id = f"final:{self.relative(trial)}"
            common = {
                "evaluation_id": evaluation_id,
                "evaluation_scope": "final",
                "cpu_attempt": cpu_attempt_for(trial, self.state_dir),
                "scoring_queue_key": self.run.get("scoring_queue_key") or self.run_id,
                "cache_hit": bool(
                    result.get("verifier_reused_continuous_evaluation_id")
                ),
                "source_evaluation_id": result.get(
                    "verifier_reused_continuous_evaluation_id"
                ),
            }
            self.add_event(
                epoch_ms=parse_epoch_ms(verifier.get("started_at")),
                category="infrastructure",
                kind="cpu_released",
                source=self.relative(result_path),
                identity=(
                    f"cpu:{common['cpu_attempt']}:released:{verifier.get('started_at')}"
                ),
                data={"cpu_attempt": common["cpu_attempt"]},
            )
            start_kind = (
                "evaluation_cache_hit" if common["cache_hit"] else "evaluation_started"
            )
            for key, kind in (
                ("started_at", start_kind),
                ("finished_at", "evaluation_finished"),
            ):
                self.add_event(
                    epoch_ms=parse_epoch_ms(verifier.get(key)),
                    category="infrastructure",
                    kind=kind,
                    source=self.relative(result_path),
                    identity=f"{evaluation_id}:{kind}:{verifier.get(key)}",
                    data=common,
                )

    @staticmethod
    def _paired_intervals(
        events: list[dict[str, Any]],
        *,
        start_kinds: set[str],
        end_kinds: set[str],
        key_fields: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        opened: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        intervals: list[dict[str, Any]] = []
        for event in events:
            key = tuple(event.get(field) for field in key_fields)
            if event["kind"] in start_kinds:
                opened[key].append(event)
            elif event["kind"] in end_kinds and opened.get(key):
                start = opened[key].pop(0)
                intervals.append(
                    {
                        "start_epoch_ms": start["epoch_ms"],
                        "end_epoch_ms": event["epoch_ms"],
                        **{field: start.get(field) for field in key_fields},
                    }
                )
        for key, starts in opened.items():
            for start in starts:
                intervals.append(
                    {
                        "start_epoch_ms": start["epoch_ms"],
                        "end_epoch_ms": None,
                        **{field: value for field, value in zip(key_fields, key)},
                    }
                )
        return intervals

    @staticmethod
    def _metric_coverage(
        intervals: list[dict[str, Any]],
        samples: list[dict[str, Any]],
        *,
        max_gap_ms: int,
        match_fields: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        coverage: list[dict[str, Any]] = []
        for interval in intervals:
            start = interval["start_epoch_ms"]
            end = interval["end_epoch_ms"]
            matching = []
            if end is not None:
                for sample in samples:
                    if not start <= sample["epoch_ms"] <= end:
                        continue
                    if any(
                        interval.get(field) is not None
                        and sample.get(field) != interval.get(field)
                        for field in match_fields
                    ):
                        continue
                    matching.append(sample["epoch_ms"])
            matching.sort()
            gaps = [] if end is None or not matching else [matching[0] - start]
            if matching:
                gaps.extend(b - a for a, b in zip(matching, matching[1:]))
                gaps.append(end - matching[-1])
            worst_gap = max(gaps) if gaps else None
            coverage.append(
                {
                    **interval,
                    "sample_count": len(matching),
                    "max_gap_ms": worst_gap,
                    "covered": bool(
                        end is not None
                        and matching
                        and worst_gap is not None
                        and worst_gap <= max_gap_ms
                    ),
                }
            )
        return coverage

    def finalize(self) -> dict[str, Any]:
        trace_order = {"tool_call": 0, "tool_result": 1}
        self.events.sort(
            key=lambda event: (
                event["epoch_ms"],
                trace_order.get(event["kind"], 0),
                event["category"],
                event["id"],
            )
        )
        pending_tools: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
        for event in self.events:
            call_id = event.get("call_id")
            key = (event.get("cpu_attempt"), call_id)
            if event["kind"] == "tool_call" and call_id:
                pending_tools[key].append(event)
            elif event["kind"] == "tool_result" and call_id and pending_tools.get(key):
                call = pending_tools[key].pop(0)
                duration_ms = max(0, event["epoch_ms"] - call["epoch_ms"])
                call["finished_epoch_ms"] = event["epoch_ms"]
                call["finished_ts"] = event["ts"]
                call["duration_ms"] = duration_ms
                event["duration_ms"] = duration_ms
                event["tool"] = call.get("tool")
                session_id = event.get("shell_session_id") or call.get(
                    "shell_session_id"
                )
                if session_id is not None:
                    call["shell_session_id"] = str(session_id)
                    event["shell_session_id"] = str(session_id)
                self.counts["timed_tool_calls"] += 1
        self.counts["unmatched_tool_calls"] = sum(
            len(calls) for calls in pending_tools.values()
        )
        if self.events:
            origin = self.events[0]["epoch_ms"]
            end = self.events[-1]["epoch_ms"]
            for event in self.events:
                event["elapsed_ms"] = event["epoch_ms"] - origin
        else:
            origin = end = None

        buckets: dict[int, Counter[str]] = defaultdict(Counter)
        for event in self.events:
            if event["kind"] != "tool_call":
                continue
            start = (event["epoch_ms"] // self.bucket_ms) * self.bucket_ms
            calls = int(event.get("tool_calls_in_record") or 1)
            buckets[start][str(event.get("tool") or "unknown")] += calls
        tool_buckets = [
            {
                "start_epoch_ms": start,
                "start_ts": iso_from_ms(start),
                "elapsed_ms": start - origin if origin is not None else None,
                "total": sum(counts.values()),
                "by_tool": dict(sorted(counts.items())),
            }
            for start, counts in sorted(buckets.items())
        ]
        tool_timings: dict[str, list[int]] = defaultdict(list)
        tool_call_counts: Counter[str] = Counter()
        for event in self.events:
            if event["kind"] != "tool_call":
                continue
            tool = str(event.get("tool") or "unknown")
            tool_call_counts[tool] += int(event.get("tool_calls_in_record") or 1)
            if isinstance(event.get("duration_ms"), int):
                tool_timings[tool].append(event["duration_ms"])

        def percentile(values: list[int], fraction: float) -> int | None:
            if not values:
                return None
            ordered = sorted(values)
            return ordered[round((len(ordered) - 1) * fraction)]

        tool_timing_summary = {
            tool: {
                "call_count": tool_call_counts[tool],
                "timed_call_count": len(tool_timings[tool]),
                "total_duration_ms": sum(tool_timings[tool]),
                "mean_duration_ms": (
                    round(sum(tool_timings[tool]) / len(tool_timings[tool]))
                    if tool_timings[tool]
                    else None
                ),
                "p50_duration_ms": percentile(tool_timings[tool], 0.50),
                "p95_duration_ms": percentile(tool_timings[tool], 0.95),
                "max_duration_ms": max(tool_timings[tool])
                if tool_timings[tool]
                else None,
            }
            for tool in sorted(tool_call_counts)
        }

        # A long shell command may span one exec_command call and several
        # write_stdin polls. Keep both the individual tool latency above and
        # the end-to-end subprocess lifetime here. Arguments/output remain
        # private; only the opaque session ID and timing/status are exported.
        shell_sessions: dict[tuple[Any, str], dict[str, Any]] = {}
        tool_results = {
            (event.get("cpu_attempt"), event.get("call_id")): event
            for event in self.events
            if event["kind"] == "tool_result" and event.get("call_id")
        }
        for event in self.events:
            if event["kind"] != "tool_call":
                continue
            result = tool_results.get((event.get("cpu_attempt"), event.get("call_id")))
            if not result:
                continue
            session_id = event.get("shell_session_id")
            if session_id is None:
                continue
            key = (event.get("cpu_attempt"), str(session_id))
            session = shell_sessions.get(key)
            if session is None:
                session = {
                    "cpu_attempt": event.get("cpu_attempt"),
                    "shell_session_id": str(session_id),
                    "started_epoch_ms": event["epoch_ms"],
                    "started_ts": event["ts"],
                    "finished_epoch_ms": None,
                    "finished_ts": None,
                    "duration_ms": None,
                    "tool_call_count": 0,
                    "timed_tool_call_ms": 0,
                    "exit_code": None,
                    "aborted": False,
                    "complete": False,
                }
                shell_sessions[key] = session
            session["tool_call_count"] += 1
            session["timed_tool_call_ms"] += int(event.get("duration_ms") or 0)
            still_running = result.get("shell_process_running") is True
            terminal_poll = event.get("tool") == "write_stdin" and not still_running
            explicit_terminal = (
                result.get("shell_process_exit_code") is not None
                or result.get("shell_process_aborted") is True
            )
            if terminal_poll or explicit_terminal:
                session["finished_epoch_ms"] = result["epoch_ms"]
                session["finished_ts"] = result["ts"]
                session["duration_ms"] = max(
                    0, result["epoch_ms"] - session["started_epoch_ms"]
                )
                session["exit_code"] = result.get("shell_process_exit_code")
                session["aborted"] = bool(result.get("shell_process_aborted"))
                session["complete"] = True
        shell_session_summary = sorted(
            shell_sessions.values(),
            key=lambda session: (
                session["started_epoch_ms"],
                session["shell_session_id"],
            ),
        )

        usage_events = [
            event for event in self.events if event["kind"] == "model_request_usage"
        ]
        token_fields = (
            "input_tokens",
            "ordinary_uncached_input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        )
        usage_totals = {
            field: sum(
                int(event.get(field) or 0)
                for event in usage_events
                if isinstance(event.get(field), int)
                and not isinstance(event.get(field), bool)
            )
            for field in token_fields
        }
        usage_summary = {
            "request_count": len(usage_events),
            **usage_totals,
            "calculated_api_usage_usd": (
                sum(
                    float(event["calculated_cost_usd"])
                    for event in usage_events
                    if isinstance(event.get("calculated_cost_usd"), (int, float))
                    and not isinstance(event.get("calculated_cost_usd"), bool)
                )
                if usage_events
                else (
                    0.0
                    if self.counts["attested_zero_request_usage_audits"]
                    == self.source_counts["usage_audit_files"]
                    and self.source_counts["usage_audit_files"] > 0
                    else None
                )
            ),
            "pricing_snapshot_ids": sorted(
                {
                    str(event["pricing_snapshot_id"])
                    for event in usage_events
                    if event.get("pricing_snapshot_id")
                }
            ),
        }

        max_gap_ms = (
            int(
                self.run.get("telemetry_gpu_max_gap_seconds")
                or DEFAULT_GPU_MAX_GAP_SECONDS
            )
            * 1000
        )
        cpu_max_gap_ms = (
            int(
                self.run.get("telemetry_cpu_max_gap_seconds")
                or DEFAULT_GPU_MAX_GAP_SECONDS
            )
            * 1000
        )
        cpu_intervals = self._paired_intervals(
            self.events,
            start_kinds={"cpu_allocated", "cpu_reallocated"},
            end_kinds={"cpu_interrupted", "cpu_released"},
            key_fields=("cpu_attempt",),
        )
        training_intervals = self._paired_intervals(
            self.events,
            start_kinds={"gpu_allocated", "gpu_reallocated"},
            end_kinds={"gpu_preempted", "gpu_released"},
            key_fields=("gpu_job_id", "gpu_attempt"),
        )
        verifier_evaluation_intervals = self._paired_intervals(
            self.events,
            start_kinds={"evaluation_started"},
            end_kinds={"evaluation_finished"},
            key_fields=("evaluation_id",),
        )
        verifier_intervals = self._paired_intervals(
            self.events,
            start_kinds={"verifier_gpu_started"},
            end_kinds={"verifier_gpu_stopped"},
            key_fields=("evaluation_id",),
        )
        training_samples = [
            event
            for event in self.events
            if event["kind"] == "resource_sample"
            and event.get("role") == "training-gpu"
            and (event.get("metrics") or {}).get("gpus")
        ]
        cpu_samples = [
            event
            for event in self.events
            if event["kind"] == "resource_sample" and event.get("role") == "cpu-agent"
        ]
        cpu_coverage = self._metric_coverage(
            cpu_intervals,
            cpu_samples,
            max_gap_ms=cpu_max_gap_ms,
            match_fields=("cpu_attempt",),
        )
        verifier_samples = [
            event
            for event in self.events
            if event["kind"] == "resource_sample"
            and event.get("role") == "verifier-gpu"
            and (event.get("metrics") or {}).get("gpus")
        ]
        training_coverage = self._metric_coverage(
            training_intervals,
            training_samples,
            max_gap_ms=max_gap_ms,
            match_fields=("gpu_job_id", "gpu_attempt"),
        )
        verifier_coverage = self._metric_coverage(
            verifier_intervals,
            verifier_samples,
            max_gap_ms=max_gap_ms,
            match_fields=("evaluation_id",),
        )
        pipeline_max_gap_ms = (
            int(self.run.get("telemetry_gpu_pipeline_max_gap_seconds") or 45) * 1000
        )

        def pipeline_coverage(
            intervals: list[dict[str, Any]],
            samples: list[dict[str, Any]],
            *,
            match_fields: tuple[str, ...],
        ) -> dict[str, list[dict[str, Any]]]:
            return {
                field: self._metric_coverage(
                    intervals,
                    [
                        sample
                        for sample in samples
                        if any(
                            gpu.get(field) is not None
                            for gpu in ((sample.get("metrics") or {}).get("gpus") or [])
                            if isinstance(gpu, dict)
                        )
                    ],
                    max_gap_ms=pipeline_max_gap_ms,
                    match_fields=match_fields,
                )
                for field in GPU_PIPELINE_FIELDS
            }

        training_pipeline_coverage = pipeline_coverage(
            training_intervals,
            training_samples,
            match_fields=("gpu_job_id", "gpu_attempt"),
        )
        verifier_pipeline_coverage = pipeline_coverage(
            verifier_intervals,
            verifier_samples,
            match_fields=("evaluation_id",),
        )
        training_expected = bool(training_intervals)
        verifier_expected = bool(verifier_evaluation_intervals)
        verifier_evaluation_ids = {
            interval.get("evaluation_id") for interval in verifier_evaluation_intervals
        }
        completed_verifier_ids = {
            interval.get("evaluation_id")
            for interval in verifier_intervals
            if interval.get("end_epoch_ms") is not None
        }
        self.counts["training_gpu_intervals"] = len(training_intervals)
        self.counts["cpu_intervals"] = len(cpu_intervals)
        self.counts["cpu_intervals_covered"] = sum(
            bool(item["covered"]) for item in cpu_coverage
        )
        self.counts["training_gpu_intervals_covered"] = sum(
            bool(item["covered"]) for item in training_coverage
        )
        self.counts["verifier_gpu_intervals"] = len(verifier_intervals)
        self.counts["verifier_evaluation_intervals"] = len(
            verifier_evaluation_intervals
        )
        self.counts["verifier_gpu_intervals_covered"] = sum(
            bool(item["covered"]) for item in verifier_coverage
        )

        requirements = {
            "timestamped_agent_trace": self.counts["event:trace"] > 0,
            "cpu_agent_lifecycle": (
                bool(cpu_intervals)
                and all(
                    interval["end_epoch_ms"] is not None for interval in cpu_intervals
                )
                if self.run.get("cpu_supervised")
                else True
            ),
            "cpu_agent_metrics": (
                bool(cpu_coverage) and all(item["covered"] for item in cpu_coverage)
                if self.run.get("cpu_supervised")
                else self.counts["metric_role:cpu-agent"] > 0
            ),
            "training_gpu_lifecycle": (
                all(
                    interval["end_epoch_ms"] is not None
                    for interval in training_intervals
                )
                if training_expected
                else True
            ),
            "training_gpu_metrics": (
                all(item["covered"] for item in training_coverage)
                if training_expected
                else True
            ),
            "verifier_gpu_lifecycle": (
                all(
                    interval["end_epoch_ms"] is not None
                    for interval in verifier_evaluation_intervals
                )
                and verifier_evaluation_ids <= completed_verifier_ids
                if verifier_expected
                else True
            ),
            "verifier_gpu_metrics": (
                bool(verifier_coverage)
                and verifier_evaluation_ids
                <= {
                    item.get("evaluation_id")
                    for item in verifier_coverage
                    if item["covered"]
                }
                if verifier_expected
                else True
            ),
            "all_submitted_artifacts_captured": self.counts[
                "missing_submission_artifacts"
            ]
            == 0,
            "submission_ledger": self.source_counts["ledger_files"] > 0,
            "model_usage_and_cost": (
                self.source_counts["usage_audit_files"] > 0
                and self.counts["complete_usage_audits"]
                == self.source_counts["usage_audit_files"]
                and self.counts["incomplete_model_request_costs"] == 0
                if self.run.get("usage_audit_required")
                else True
            ),
        }
        if self.run.get("gpu_pipeline_telemetry_required"):
            requirements["training_gpu_pipeline_metrics"] = (
                all(
                    all(item["covered"] for item in field_coverage)
                    for field_coverage in training_pipeline_coverage.values()
                )
                if training_expected
                else True
            )
            requirements["verifier_gpu_pipeline_metrics"] = (
                all(
                    all(item["covered"] for item in field_coverage)
                    for field_coverage in verifier_pipeline_coverage.values()
                )
                if verifier_expected
                else True
            )
        cgroup_samples = list(cpu_samples)
        if training_expected:
            cgroup_samples.extend(training_samples)
        if verifier_expected:
            cgroup_samples.extend(verifier_samples)
        self.counts["cgroup_scoped_resource_samples"] = sum(
            (event.get("metrics") or {}).get("resource_accounting_scope")
            in {"cgroup-v1", "cgroup-v2"}
            for event in cgroup_samples
        )
        if self.run.get("cgroup_telemetry_required"):
            requirements["cgroup_scoped_cpu_memory"] = bool(cgroup_samples) and all(
                (event.get("metrics") or {}).get("resource_accounting_scope")
                in {"cgroup-v1", "cgroup-v2"}
                for event in cgroup_samples
            )
        if self.counts["events_without_timestamp"]:
            self.warnings.append(
                f"{self.counts['events_without_timestamp']} records had no usable timestamp"
            )
        if self.counts["malformed_trace_lines"]:
            self.warnings.append(
                f"{self.counts['malformed_trace_lines']} malformed trace lines were retained only in raw data"
            )
        if self.counts["unknown_metric_roles"]:
            self.warnings.append(
                f"{self.counts['unknown_metric_roles']} metric samples used an unknown resource role"
            )
        if self.counts["impossible_cpu_agent_gpu_samples"]:
            self.warnings.append(
                f"{self.counts['impossible_cpu_agent_gpu_samples']} GPU samples claimed the CPU-only agent role and were not attributed"
            )
        if training_expected and not requirements["training_gpu_metrics"]:
            self.warnings.append(
                "training GPU telemetry does not cover every allocation interval"
            )
        if verifier_expected and not requirements["verifier_gpu_metrics"]:
            self.warnings.append(
                "verifier GPU telemetry does not cover every scoring interval"
            )
        if not requirements.get("cgroup_scoped_cpu_memory", True):
            self.warnings.append(
                "CPU or memory telemetry used host-wide fallback counters instead of cgroup-local counters"
            )
        coverage = {
            "ready": all(requirements.values()),
            "requirements": requirements,
            "gpu_metric_coverage": {
                "max_gap_ms": max_gap_ms,
                "training": training_coverage,
                "verifier": verifier_coverage,
            },
            "gpu_pipeline_metric_coverage": {
                "max_gap_ms": pipeline_max_gap_ms,
                "training": training_pipeline_coverage,
                "verifier": verifier_pipeline_coverage,
            },
            "cpu_metric_coverage": {
                "max_gap_ms": cpu_max_gap_ms,
                "attempts": cpu_coverage,
            },
            "counts": dict(sorted(self.counts.items())),
            "sources": dict(sorted(self.source_counts.items())),
            "warnings": self.warnings,
        }

        def allocated_ms(
            intervals: list[dict[str, Any]], *, cutoff_ms: int | None = None
        ) -> int:
            effective_end = cutoff_ms if cutoff_ms is not None else end
            if effective_end is None:
                return 0
            return sum(
                max(
                    0,
                    min(
                        int(interval["end_epoch_ms"])
                        if interval.get("end_epoch_ms") is not None
                        else effective_end,
                        effective_end,
                    )
                    - int(interval["start_epoch_ms"]),
                )
                for interval in intervals
                if int(interval["start_epoch_ms"]) < effective_end
            )

        allocated_by_role = {
            "cpu_agent": allocated_ms(cpu_intervals),
            "training_gpu": allocated_ms(training_intervals),
            "verifier_gpu": allocated_ms(verifier_intervals),
        }
        modal_estimate = modal_cost.estimate_cost(
            resource_contract=self.run.get("resource_contract") or {},
            allocated_ms_by_role=allocated_by_role,
        )
        provider_path = self.state_dir / "telemetry" / "modal-cost.json"
        try:
            modal_provider_raw = json.loads(provider_path.read_text())
        except (OSError, json.JSONDecodeError):
            modal_provider_raw = {
                "schema_version": modal_cost.SCHEMA_VERSION,
                "status": "unavailable",
                "provider_complete": False,
            }
        provider_fields = {
            "schema_version",
            "run_id",
            "generated_at",
            "source",
            "billing_basis",
            "billing_objects",
            "expected_billed_roles",
            "expected_role_categories",
            "provider_complete",
            "status",
            "pending_reason",
            "missing_billed_roles",
            "missing_role_categories",
            "run_started_at",
            "run_stopped_at",
            "query_start",
            "query_end",
            "eligible_at",
            "items",
            "by_role_usd",
            "by_category_usd",
            "by_role_category_usd",
            "provider_cost_precredits_usd",
            "provider_report_sha256",
            "selected_items_sha256",
            "invoice_adjustments_included",
            "invoice_note",
            "volume_storage",
        }
        modal_provider = {
            key: value
            for key, value in modal_provider_raw.items()
            if key in provider_fields
        }
        provider_complete = modal_provider_raw.get("provider_complete") is True
        selected_modal_cost = (
            modal_provider.get("provider_cost_precredits_usd")
            if provider_complete
            else modal_estimate["estimated_cost_usd"]
        )
        estimated_agent_modal_cost = sum(
            float(
                (modal_estimate["by_role"].get(role) or {}).get("estimated_cost_usd")
                or 0
            )
            for role in ("cpu_agent", "training_gpu")
        )
        estimated_verifier_cost = float(
            (modal_estimate["by_role"].get("verifier_gpu") or {}).get(
                "estimated_cost_usd"
            )
            or 0
        )
        if provider_complete:
            provider_by_role = modal_provider.get("by_role_usd") or {}
            selected_agent_modal_cost = sum(
                float(provider_by_role.get(role) or 0)
                for role in ("cpu_agent", "training_gpu")
            )
            selected_verifier_cost = float(provider_by_role.get("verifier_gpu") or 0)
        else:
            selected_agent_modal_cost = estimated_agent_modal_cost
            selected_verifier_cost = estimated_verifier_cost

        def pipeline_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
            summary: dict[str, Any] = {}
            for field in GPU_PIPELINE_FIELDS:
                weighted: list[tuple[float, float]] = []
                for sample in samples:
                    for gpu in (sample.get("metrics") or {}).get("gpus") or []:
                        if not isinstance(gpu, dict):
                            continue
                        value = gpu.get(field)
                        if not isinstance(value, (int, float)) or isinstance(
                            value, bool
                        ):
                            continue
                        weight = gpu.get("pipeline_metrics_window_ms")
                        weighted.append(
                            (
                                float(value),
                                float(weight)
                                if isinstance(weight, (int, float)) and weight > 0
                                else 1.0,
                            )
                        )
                if not weighted:
                    summary[field] = {
                        "sample_count": 0,
                        "window_weighted_mean_pct": None,
                        "p50_pct": None,
                        "p95_pct": None,
                        "max_pct": None,
                    }
                    continue
                values = sorted(value for value, _ in weighted)
                summary[field] = {
                    "sample_count": len(values),
                    "window_weighted_mean_pct": round(
                        sum(value * weight for value, weight in weighted)
                        / sum(weight for _, weight in weighted),
                        4,
                    ),
                    "p50_pct": round(values[round((len(values) - 1) * 0.50)], 4),
                    "p95_pct": round(values[round((len(values) - 1) * 0.95)], 4),
                    "max_pct": round(max(values), 4),
                }
            return {
                "collector": "cupti-pm-sampling",
                "metrics": summary,
            }

        resource_usage_summary = {
            "cpu_agent": {
                "allocation_count": len(cpu_intervals),
                "allocated_ms": allocated_by_role["cpu_agent"],
            },
            "training_gpu": {
                "allocation_count": len(training_intervals),
                "allocated_ms": allocated_by_role["training_gpu"],
            },
            "verifier_gpu": {
                "allocation_count": len(verifier_intervals),
                "allocated_ms": allocated_by_role["verifier_gpu"],
            },
            "gpu_pipeline": {
                "training_gpu": pipeline_summary(training_samples),
                "verifier_gpu": pipeline_summary(verifier_samples),
            },
            "resource_contract": self.run.get("resource_contract"),
            "modal_estimate": modal_estimate,
            "modal_provider_billing": modal_provider,
            "usd_cost": selected_modal_cost,
            "agent_compute_usd_cost": selected_agent_modal_cost,
            "verifier_measurement_overhead_usd_cost": selected_verifier_cost,
            "usd_cost_kind": (
                "provider_report_precredits"
                if provider_complete
                else "pinned_tariff_request_floor"
            ),
        }
        scored_artifacts = []
        for artifact in self.artifacts:
            rewards = artifact.get("rewards") or {}
            value = rewards.get("best_100m_s")
            valid = bool(rewards.get("valid_run"))
            if (
                valid
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                scored_artifacts.append((float(value), artifact))
        best_time = min((item[0] for item in scored_artifacts), default=None)
        best_artifact = (
            min(scored_artifacts, key=lambda item: item[0])[1]
            if scored_artifacts
            else None
        )
        best_submitted_ms = (
            parse_epoch_ms(best_artifact.get("submitted_at")) if best_artifact else None
        )
        best_result_ms = (
            parse_epoch_ms(best_artifact.get("finished_at")) if best_artifact else None
        ) or best_submitted_ms
        uses_frozen_final = (
            self.run.get("primary_score_policy") == "frozen_final_artifact"
        )
        evaluation_result_policy = self.run.get("evaluation_result_policy") or (
            "frozen_final_artifact" if uses_frozen_final else "all_feedback_submissions"
        )
        primary_artifacts = (
            [artifact for artifact in self.artifacts if artifact.get("primary_final")]
            if uses_frozen_final
            else []
        )
        primary_artifact = primary_artifacts[-1] if primary_artifacts else None
        primary_rewards = self.final_verifier_rewards
        primary_time_raw = primary_rewards.get("best_100m_s")
        primary_time = (
            float(primary_time_raw)
            if primary_rewards.get("valid_run")
            and isinstance(primary_time_raw, (int, float))
            and not isinstance(primary_time_raw, bool)
            else None
        )
        primary_submitted_ms = (
            parse_epoch_ms(primary_artifact.get("submitted_at"))
            if primary_artifact
            else None
        )
        primary_result_ms = parse_epoch_ms(self.final_verifier_finished_at)

        def costs_at(cutoff_ms: int | None) -> dict[str, Any] | None:
            if cutoff_ms is None:
                return None
            api_cost = (
                sum(
                    float(event["calculated_cost_usd"])
                    for event in usage_events
                    if event["epoch_ms"] <= cutoff_ms
                    and isinstance(event.get("calculated_cost_usd"), (int, float))
                    and not isinstance(event.get("calculated_cost_usd"), bool)
                )
                if usage_events
                else None
            )
            estimated_modal = modal_cost.estimate_cost(
                resource_contract=self.run.get("resource_contract") or {},
                allocated_ms_by_role={
                    "cpu_agent": allocated_ms(cpu_intervals, cutoff_ms=cutoff_ms),
                    "training_gpu": allocated_ms(
                        training_intervals, cutoff_ms=cutoff_ms
                    ),
                    "verifier_gpu": allocated_ms(
                        verifier_intervals, cutoff_ms=cutoff_ms
                    ),
                },
            )
            modal_value = estimated_modal["estimated_cost_usd"]
            agent_modal_value = sum(
                float(
                    (estimated_modal["by_role"].get(role) or {}).get(
                        "estimated_cost_usd"
                    )
                    or 0
                )
                for role in ("cpu_agent", "training_gpu")
            )
            verifier_modal_value = float(
                (estimated_modal["by_role"].get("verifier_gpu") or {}).get(
                    "estimated_cost_usd"
                )
                or 0
            )
            return {
                "epoch_ms": cutoff_ms,
                "api_calculated_usd": api_cost,
                "modal_tariff_estimated_usd": modal_value,
                "agent_modal_tariff_estimated_usd": agent_modal_value,
                "verifier_measurement_overhead_estimated_usd": verifier_modal_value,
                "total_estimated_usd": api_cost + agent_modal_value
                if api_cost is not None
                else None,
                "modal_estimate_kind": estimated_modal["estimate_kind"],
            }

        for artifact in self.artifacts:
            artifact["cost_at_submission"] = costs_at(
                parse_epoch_ms(artifact.get("submitted_at"))
            )
            artifact["cost_at_result"] = costs_at(
                parse_epoch_ms(artifact.get("finished_at"))
            )
        best_cost = costs_at(best_result_ms)
        primary_cost = costs_at(primary_submitted_ms) if uses_frozen_final else None
        modal_estimate_at_best = None
        api_cost_at_best = None
        if best_cost is not None:
            modal_estimate_at_best = best_cost["modal_tariff_estimated_usd"]
            api_cost_at_best = best_cost["api_calculated_usd"]
        comparison_summary = {
            "submission_count": len(self.artifacts),
            "valid_submission_count": len(scored_artifacts),
            "disqualified_submission_count": len(self.artifacts)
            - len(scored_artifacts),
            "evaluation_result_policy": evaluation_result_policy,
            "best_100m_s": primary_time if uses_frozen_final else best_time,
            "best_submission_epoch_ms": best_submitted_ms,
            "best_result_epoch_ms": best_result_ms,
            "time_to_best_ms": (
                best_result_ms - origin
                if best_result_ms is not None and origin is not None
                else None
            ),
            "api_cost_at_best_usd": api_cost_at_best,
            "modal_estimated_cost_at_best_usd": modal_estimate_at_best,
            "total_estimated_cost_at_best_usd": (
                api_cost_at_best + modal_estimate_at_best
                if api_cost_at_best is not None and modal_estimate_at_best is not None
                else None
            ),
            "final_api_cost_usd": usage_summary["calculated_api_usage_usd"],
            "final_modal_estimated_cost_usd": modal_estimate["estimated_cost_usd"],
            "final_modal_provider_cost_precredits_usd": modal_provider.get(
                "provider_cost_precredits_usd"
            )
            if provider_complete
            else None,
            "final_agent_total_cost_usd": (
                float(usage_summary["calculated_api_usage_usd"])
                + float(selected_agent_modal_cost)
                if usage_summary["calculated_api_usage_usd"] is not None
                and selected_agent_modal_cost is not None
                else None
            ),
            "final_verifier_measurement_overhead_usd": selected_verifier_cost,
            "final_total_cost_usd": (
                float(usage_summary["calculated_api_usage_usd"])
                + float(selected_agent_modal_cost)
                if usage_summary["calculated_api_usage_usd"] is not None
                and selected_agent_modal_cost is not None
                else None
            ),
            "final_total_cost_kind": (
                "api_calculated_plus_agent_modal_provider_precredits"
                if provider_complete
                else "api_calculated_plus_agent_modal_tariff_estimate"
            ),
            "wall_duration_ms": end - origin
            if origin is not None and end is not None
            else None,
            "tool_call_count": sum(tool_call_counts.values()),
            "timed_tool_call_count": self.counts["timed_tool_calls"],
        }
        if uses_frozen_final:
            comparison_summary.update(
                {
                    "primary_score_policy": self.run.get("primary_score_policy"),
                    "primary_final_100m_s": primary_time,
                    "primary_final_valid": bool(primary_rewards.get("valid_run")),
                    "primary_final_submission_epoch_ms": primary_submitted_ms,
                    "primary_final_result_epoch_ms": primary_result_ms,
                    "retrospective_best_100m_s": best_time,
                    "primary_final_agent_cost_at_submission_usd": (
                        primary_cost.get("total_estimated_usd")
                        if primary_cost is not None
                        else None
                    ),
                    "verifier_measurement_overhead_at_primary_submission_estimated_usd": (
                        primary_cost.get("verifier_measurement_overhead_estimated_usd")
                        if primary_cost is not None
                        else None
                    ),
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": dt.datetime.now(dt.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "run": {
                "run_id": self.run_id,
                "model": self.run.get("model"),
                "agent_kind": self.run.get("agent_kind"),
                "reasoning_effort": self.run.get("reasoning_effort"),
                "resolved_model_version": self.run.get("resolved_model_version"),
                "codex_version": self.run.get("codex_version"),
                "harbor_commit": self.run.get("harbor_commit"),
                "evaluation_provenance": self.run.get("evaluation_provenance"),
                "created_at": self.run.get("created_at"),
                "scoring_queue_scope": self.run.get("scoring_queue_scope"),
                "scoring_queue_key": self.run.get("scoring_queue_key") or self.run_id,
                "scoring_max_concurrent": self.run.get("scoring_max_concurrent"),
                "scoring_global_max_concurrent": self.run.get(
                    "scoring_global_max_concurrent"
                ),
                "scoring_feedback_policy": self.run.get("scoring_feedback_policy"),
                "evaluation_result_policy": evaluation_result_policy,
            },
            "resource_roles": {
                "cpu-agent": {"resource": "cpu", "trust": "agent"},
                "training-gpu": {"resource": "gpu", "trust": "agent"},
                "verifier-gpu": {"resource": "gpu", "trust": "benchmark"},
                "host-controller": {"resource": "cpu", "trust": "benchmark"},
            },
            "clock": {
                "unit": "epoch_ms",
                "origin_epoch_ms": origin,
                "end_epoch_ms": end,
                "duration_ms": end - origin
                if origin is not None and end is not None
                else None,
            },
            "events": self.events,
            "tool_call_buckets": {"width_ms": self.bucket_ms, "buckets": tool_buckets},
            "tool_timing_summary": tool_timing_summary,
            "shell_session_summary": shell_session_summary,
            "usage_summary": usage_summary,
            "resource_usage_summary": resource_usage_summary,
            "comparison_summary": comparison_summary,
            "artifacts": sorted(
                self.artifacts,
                key=lambda item: (
                    item["cpu_attempt"],
                    item.get("submission_index") or 0,
                ),
            ),
            "coverage": coverage,
        }


def build_timeline(
    state_dir: pathlib.Path,
    *,
    web_dir: pathlib.Path | None = None,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> dict[str, Any]:
    run = json.loads((state_dir / "run.json").read_text())
    builder = Builder(state_dir, run, bucket_seconds)
    trials = trial_dirs(state_dir, run)
    builder.add_run_lifecycle()
    builder.add_cpu_lifecycle()
    builder.add_telemetry(trials)
    builder.add_verifier_gpu_lifecycle(trials)
    builder.add_gpu_lifecycle(trials)
    builder.add_traces(trials)
    builder.add_usage_audits(trials)
    builder.add_submissions(trials)
    builder.add_final_verification(trials)
    payload = builder.finalize()
    internal = state_dir / "telemetry" / "unified-timeline.json"
    atomic_json(internal, payload, mode=0o600)
    if web_dir is not None:
        public = web_dir / "data" / "timelines" / f"{builder.run_id}.json"
        atomic_json(public, payload, mode=0o644)
        index_path = web_dir / "data" / "timelines" / "index.json"
        try:
            index = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError):
            index = {"schema_version": SCHEMA_VERSION, "runs": []}
        entries = {
            str(item.get("run_id")): item
            for item in index.get("runs", [])
            if isinstance(item, dict) and item.get("run_id")
        }
        entries[builder.run_id] = {
            "run_id": builder.run_id,
            "model": run.get("model"),
            "agent_kind": run.get("agent_kind"),
            "reasoning_effort": run.get("reasoning_effort"),
            "resolved_model_version": run.get("resolved_model_version"),
            "created_at": run.get("created_at"),
            "generated_at": payload["generated_at"],
            "path": f"/data/timelines/{builder.run_id}.json",
            "ready": payload["coverage"]["ready"],
            "origin_epoch_ms": payload["clock"]["origin_epoch_ms"],
            "end_epoch_ms": payload["clock"]["end_epoch_ms"],
            "artifact_count": len(payload["artifacts"]),
            "event_count": len(payload["events"]),
            "comparison_summary": payload["comparison_summary"],
            "resource_usage_summary": payload["resource_usage_summary"],
            "dashboard_artifacts": [
                {
                    "submission_index": artifact.get("submission_index"),
                    "finished_epoch_ms": parse_epoch_ms(artifact.get("finished_at")),
                    "rewards": {
                        "valid_run": (artifact.get("rewards") or {}).get("valid_run"),
                        "best_100m_s": (artifact.get("rewards") or {}).get(
                            "best_100m_s"
                        ),
                        "gate_finished": (artifact.get("rewards") or {}).get(
                            "gate_finished"
                        ),
                        "gate_in_lane": (artifact.get("rewards") or {}).get(
                            "gate_in_lane"
                        ),
                        "gate_self_collision": (artifact.get("rewards") or {}).get(
                            "gate_self_collision"
                        ),
                        "peak_speed_mps": (artifact.get("rewards") or {}).get(
                            "peak_speed_mps"
                        ),
                    },
                    "cost_at_result": artifact.get("cost_at_result"),
                }
                for artifact in payload["artifacts"]
            ],
        }
        index = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": payload["generated_at"],
            "runs": sorted(
                entries.values(),
                key=lambda item: (item.get("created_at") or "", item["run_id"]),
                reverse=True,
            )[:PUBLIC_RUN_LIMIT],
        }
        atomic_json(index_path, index, mode=0o644)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=pathlib.Path, required=True)
    parser.add_argument("--web-dir", type=pathlib.Path)
    parser.add_argument("--bucket-seconds", type=int, default=DEFAULT_BUCKET_SECONDS)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    if args.bucket_seconds <= 0:
        parser.error("--bucket-seconds must be positive")
    payload = build_timeline(
        args.state_dir, web_dir=args.web_dir, bucket_seconds=args.bucket_seconds
    )
    print(json.dumps(payload["coverage"], indent=2, sort_keys=True))
    return 0 if payload["coverage"]["ready"] or not args.require_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
