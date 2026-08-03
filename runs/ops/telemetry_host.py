#!/usr/bin/env python3
"""Host-side backup telemetry poller for durable Modal lane runs.

Uses ``modal container exec`` (via qwopctl helpers) to sample the agent
sandbox, and best-effort samples verifier sandboxes in the same app.
Writes CSV/JSONL under ``runs/ops/<run_id>/telemetry/`` and a flat
``runs/monitor-<run_id>-telemetry.csv`` mirror.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import re
import sys
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path("/data/qwop-bench")
RUNS = ROOT / "runs"
sys.path.insert(0, str(SCRIPT_DIR))

import qwopctl  # noqa: E402

SECRET_RE = re.compile(
    r"(?i)("
    r"(?:api[_-]?key|oauth(?:_token)?|token|password|secret|authorization)\s*[:=]\s*\S+"
    r"|bearer\s+\S+"
    r"|sk-(?:ant-|harbor-|or-)?[A-Za-z0-9._\-]{8,}"
    r"|wk-[A-Za-z0-9._\-]+\.ws-[A-Za-z0-9._\-]+"
    r")"
)

ONESHOT_SCRIPT = r"""
umask 077
OUT=/tmp/qwop-host-telemetry-once
mkdir -p "$OUT"
if [ -x /opt/qwop-telemetry.sh ]; then
  /opt/qwop-telemetry.sh --once --role __ROLE__ --run-id __RUN_ID__ \
    --out-dir "$OUT" --pidfile /tmp/qwop-host-telemetry.pid --force \
    >/tmp/qwop-host-telemetry-once.stdout 2>/tmp/qwop-host-telemetry-once.stderr || true
  if [ -f "$OUT/latest.json" ]; then
    cat "$OUT/latest.json"
  elif [ -f /logs/artifacts/telemetry/latest.json ]; then
    cat /logs/artifacts/telemetry/latest.json
  else
    printf '%s\n' '{"ok":false,"notes":"no_latest_json"}'
  fi
else
  # Minimal fallback when image lacks the sampler (pre-rebuild open jobs).
  python3 - <<'PY'
import json, os, time, pathlib, shutil
sample = {
  "schema_version": 1,
  "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "epoch_s": int(time.time()),
  "role": "__ROLE__",
  "run_id": "__RUN_ID__",
  "hostname": os.uname().nodename,
  "nproc": os.cpu_count(),
  "notes": "fallback_no_qwop_telemetry",
  "nvidia_smi_ok": bool(shutil.which("nvidia-smi")),
}
try:
    sample["uptime_s"] = float(pathlib.Path("/proc/uptime").read_text().split()[0])
except Exception:
    sample["uptime_s"] = None
print(json.dumps(sample, sort_keys=True))
PY
fi
"""


def _redact(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text)


def _shell_oneshot(role: str, run_id: str) -> str:
    return (
        ONESHOT_SCRIPT.replace("__ROLE__", role).replace("__RUN_ID__", run_id).strip()
    )


def _monitor_csv_path(run_id: str) -> pathlib.Path:
    return RUNS / f"monitor-{run_id}-telemetry.csv"


def _state_telemetry_dir(state_dir: pathlib.Path) -> pathlib.Path:
    path = state_dir / "telemetry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_jsonl(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _flatten_for_csv(sample: dict[str, Any], *, container_id: str) -> list[dict[str, Any]]:
    base = {
        "ts_utc": sample.get("ts_utc"),
        "epoch_s": sample.get("epoch_s"),
        "role": sample.get("role"),
        "run_id": sample.get("run_id"),
        "container_id": container_id,
        "hostname": sample.get("hostname"),
        "cpu_util_pct": sample.get("cpu_util_pct"),
        "load1": sample.get("load1"),
        "load5": sample.get("load5"),
        "load15": sample.get("load15"),
        "mem_used_kib": sample.get("mem_used_kib"),
        "mem_total_kib": sample.get("mem_total_kib"),
        "mem_available_kib": sample.get("mem_available_kib"),
        "swap_used_kib": sample.get("swap_used_kib"),
        "disk_root_used_pct": sample.get("disk_root_used_pct"),
        "disk_tmp_used_pct": sample.get("disk_tmp_used_pct"),
        "disk_app_used_pct": sample.get("disk_app_used_pct"),
        "net_rx_bytes": sample.get("net_rx_bytes"),
        "net_tx_bytes": sample.get("net_tx_bytes"),
        "nproc": sample.get("nproc"),
        "uptime_s": sample.get("uptime_s"),
        "gpu_count": sample.get("gpu_count"),
        "pcie_rx_mib_s": sample.get("pcie_rx_mib_s"),
        "pcie_tx_mib_s": sample.get("pcie_tx_mib_s"),
        "nvidia_smi_ok": sample.get("nvidia_smi_ok"),
        "notes": sample.get("notes"),
    }
    gpus = sample.get("gpus") or [None]
    rows = []
    for gpu in gpus:
        row = dict(base)
        if isinstance(gpu, dict):
            row.update(
                {
                    "gpu_index": gpu.get("gpu_index"),
                    "gpu_name": gpu.get("gpu_name"),
                    "util_gpu_pct": gpu.get("util_gpu_pct"),
                    "util_mem_pct": gpu.get("util_mem_pct"),
                    "mem_used_mib": gpu.get("mem_used_mib"),
                    "mem_total_mib": gpu.get("mem_total_mib"),
                    "mem_free_mib": gpu.get("mem_free_mib"),
                    "power_draw_w": gpu.get("power_draw_w"),
                    "power_limit_w": gpu.get("power_limit_w"),
                    "temp_gpu_c": gpu.get("temp_gpu_c"),
                    "clock_graphics_mhz": gpu.get("clock_graphics_mhz"),
                    "clock_sm_mhz": gpu.get("clock_sm_mhz"),
                    "clock_mem_mhz": gpu.get("clock_mem_mhz"),
                    "pcie_link_gen": gpu.get("pcie_link_gen"),
                    "pcie_link_width": gpu.get("pcie_link_width"),
                    "ecc_mode": gpu.get("ecc_mode"),
                    "driver_version": gpu.get("driver_version"),
                }
            )
        rows.append(row)
    return rows


CSV_FIELDS = [
    "ts_utc",
    "epoch_s",
    "role",
    "run_id",
    "container_id",
    "hostname",
    "cpu_util_pct",
    "load1",
    "load5",
    "load15",
    "mem_used_kib",
    "mem_total_kib",
    "mem_available_kib",
    "swap_used_kib",
    "disk_root_used_pct",
    "disk_tmp_used_pct",
    "disk_app_used_pct",
    "net_rx_bytes",
    "net_tx_bytes",
    "nproc",
    "uptime_s",
    "gpu_count",
    "pcie_rx_mib_s",
    "pcie_tx_mib_s",
    "nvidia_smi_ok",
    "gpu_index",
    "gpu_name",
    "util_gpu_pct",
    "util_mem_pct",
    "mem_used_mib",
    "mem_total_mib",
    "mem_free_mib",
    "power_draw_w",
    "power_limit_w",
    "temp_gpu_c",
    "clock_graphics_mhz",
    "clock_sm_mhz",
    "clock_mem_mhz",
    "pcie_link_gen",
    "pcie_link_width",
    "ecc_mode",
    "driver_version",
    "notes",
]


def _append_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _parse_json_payload(raw: str) -> dict[str, Any] | None:
    text = _redact(raw).strip()
    if not text:
        return None
    # container exec may wrap noise; take the last JSON object line/blob.
    candidates = [text]
    if "\n" in text:
        candidates = list(reversed(text.splitlines())) + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    # Try to find a JSON object substring.
    start = text.rfind("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
    return None


def sample_container(
    run: dict[str, Any],
    container_id: str,
    *,
    role: str,
    run_id: str,
) -> dict[str, Any] | None:
    result = qwopctl.exec_container(
        run,
        container_id,
        _shell_oneshot(role, run_id),
        check=False,
    )
    payload = _parse_json_payload(result.stdout)
    if payload is None:
        return {
            "schema_version": 1,
            "role": role,
            "run_id": run_id,
            "container_id": container_id,
            "nvidia_smi_ok": False,
            "notes": f"host_poll_parse_failed,rc={result.returncode}",
            "stderr_excerpt": _redact((result.stderr or "")[:300]),
        }
    payload.setdefault("role", role)
    payload.setdefault("run_id", run_id)
    payload["container_id"] = container_id
    return payload


def discover_verifier_containers(
    state_dir: pathlib.Path, run: dict[str, Any]
) -> list[str]:
    """Best-effort: app containers that are not the durable agent sandbox."""
    app_id = qwopctl.discover_app_id(state_dir, run)
    if not app_id:
        return []
    containers = qwopctl.containers_for_app(run, app_id)
    agent = qwopctl.discover_agent_container(state_dir, run)
    out: list[str] = []
    for container in containers:
        if agent and container == agent:
            continue
        # Skip anything that looks like the agent keepalive image.
        if qwopctl.is_agent_container(run, container):
            continue
        out.append(container)
    return out


def poll_once(run_id: str) -> dict[str, Any]:
    state_dir, run = qwopctl.load_run(run_id)
    pathlib.Path("/data/.keepalive").touch()
    out_dir = _state_telemetry_dir(state_dir)
    monitor_csv = _monitor_csv_path(run_id)
    state_csv = out_dir / "host-samples.csv"
    summary: dict[str, Any] = {
        "run_id": run_id,
        "polled_at": qwopctl.utc_now(),
        "agent_container_id": None,
        "verifier_container_ids": [],
        "samples": 0,
        "errors": [],
    }

    agent = qwopctl.discover_agent_container(state_dir, run)
    targets: list[tuple[str, str]] = []
    if agent:
        summary["agent_container_id"] = agent
        targets.append(("agent", agent))
    else:
        summary["errors"].append("agent_container_unavailable")

    try:
        verifiers = discover_verifier_containers(state_dir, run)
    except Exception as exc:  # noqa: BLE001
        verifiers = []
        summary["errors"].append(f"verifier_discover_failed:{type(exc).__name__}")
    summary["verifier_container_ids"] = verifiers
    for container in verifiers:
        targets.append(("verifier", container))

    for role, container_id in targets:
        try:
            sample = sample_container(run, container_id, role=role, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"{role}:{container_id}:{type(exc).__name__}")
            continue
        if not sample:
            summary["errors"].append(f"{role}:{container_id}:empty")
            continue
        _append_jsonl(out_dir / "host-samples.jsonl", sample)
        rows = _flatten_for_csv(sample, container_id=container_id)
        _append_csv(state_csv, rows)
        _append_csv(monitor_csv, rows)
        # Also mirror into trial artifacts when the job/trial is known.
        job, trial = qwopctl.discover_job_and_trial(state_dir, run)
        if trial is not None:
            trial_telem = trial / "artifacts" / "logs" / "artifacts" / "telemetry"
            trial_telem.mkdir(parents=True, exist_ok=True)
            _append_jsonl(trial_telem / "host-samples.jsonl", sample)
            _append_csv(trial_telem / "host-samples.csv", rows)
        summary["samples"] += 1

    qwopctl.atomic_write_json(out_dir / "host-latest.json", summary, mode=0o600)
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    payload = poll_once(args.run_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("samples", 0) or not payload.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
