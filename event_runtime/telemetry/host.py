#!/usr/bin/env python3
"""Host-side backup telemetry poller for durable Modal lane runs.

Uses ``modal container exec`` (via sprintctl helpers) to sample the agent
sandbox and discover sidecar GPU sandboxes. Verifiers also write authoritative
in-container telemetry into their archived output; this poller is a backup for
infrastructure diagnosis, not the source used to attribute SCORE usage.
Writes CSV/JSONL under ``runs/ops/<run_id>/telemetry/`` and a flat
``runs/monitor-<run_id>-telemetry.csv`` mirror.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[2]
OPS_DIR = ROOT / "runs/ops"
RUNS = ROOT / "runs"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OPS_DIR))

from event_runtime.control import run as sprintctl  # noqa: E402

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
OUT=/tmp/sprint-host-telemetry-once
mkdir -p "$OUT"
if [ -x /opt/sprint-telemetry.sh ]; then
  /opt/sprint-telemetry.sh --once --role __ROLE__ --run-id __RUN_ID__ \
    --out-dir "$OUT" --pidfile /tmp/sprint-host-telemetry.pid --force \
    >/tmp/sprint-host-telemetry-once.stdout 2>/tmp/sprint-host-telemetry-once.stderr || true
  if [ -f "$OUT/latest.json" ]; then
    cat "$OUT/latest.json"
  elif [ -f /logs/artifacts/telemetry/latest.json ]; then
    cat /logs/artifacts/telemetry/latest.json
  else
    printf '%s\n' '{"ok":false,"notes":"no_latest_json"}'
  fi
else
  # Self-contained fallback for sealed verifier images. They intentionally do
  # not contain the agent helper scripts, but the host still needs full CPU,
  # memory, and GPU utilization on the common experiment clock.
  python3 - <<'PY'
import json, os, time, pathlib, shutil, subprocess

def cpu_times():
    values = [int(v) for v in pathlib.Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle

def keyed(path):
    try:
        lines = pathlib.Path(path).read_text().splitlines()
    except OSError:
        return {}
    result = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return result

def scalar(path):
    try:
        value = pathlib.Path(path).read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None

def number(value):
    value = value.strip()
    if not value or value.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None

root = pathlib.Path("/sys/fs/cgroup")
try:
    for line in pathlib.Path("/proc/self/cgroup").read_text().splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            candidate = root / parts[2].lstrip("/")
            if (candidate / "cpu.stat").is_file():
                root = candidate
            break
except OSError:
    pass
cpu0 = keyed(root / "cpu.stat")
memory_current = scalar(root / "memory.current")
requested_cpu = {"training-gpu": 6.0, "verifier-gpu": 4.0}.get("__ROLE__", 2.0)
requested_memory_kib = {
    "training-gpu": 12288,
    "verifier-gpu": 10240,
}.get("__ROLE__", 8192) * 1024
resource = {}
if "usage_usec" in cpu0 and memory_current is not None:
    started_ns = time.monotonic_ns()
    time.sleep(0.1)
    cpu1 = keyed(root / "cpu.stat")
    elapsed_usec = max(1.0, (time.monotonic_ns() - started_ns) / 1000)
    used_cores = max(0, cpu1.get("usage_usec", 0) - cpu0["usage_usec"]) / elapsed_usec
    memory_limit = scalar(root / "memory.max")
    memory_limit_kib = memory_limit // 1024 if memory_limit else None
    memory_total_kib = requested_memory_kib or memory_limit_kib
    memory_used_kib = memory_current // 1024
    events = keyed(root / "memory.events")
    peak = scalar(root / "memory.peak")
    resource = {
      "resource_accounting_scope": "cgroup-v2",
      "cpu_requested_cores": requested_cpu,
      "cpu_usage_cores": round(used_cores, 4),
      "cpu_usage_usec": cpu1.get("usage_usec"),
      "cpu_user_usec": cpu1.get("user_usec"),
      "cpu_system_usec": cpu1.get("system_usec"),
      "cpu_nr_throttled": cpu1.get("nr_throttled"),
      "cpu_throttled_usec": cpu1.get("throttled_usec"),
      "cpu_util_pct": round(100.0 * used_cores / requested_cpu, 2),
      "mem_requested_kib": requested_memory_kib,
      "mem_limit_kib": memory_limit_kib,
      "mem_total_kib": memory_total_kib,
      "mem_used_kib": memory_used_kib,
      "mem_available_kib": max(0, memory_total_kib - memory_used_kib),
      "memory_peak_kib": peak // 1024 if peak is not None else None,
      "memory_oom_events": events.get("oom"),
      "memory_oom_kill_events": events.get("oom_kill"),
    }
else:
    cpuacct_root = pathlib.Path("/sys/fs/cgroup/cpuacct")
    cpu_root = pathlib.Path("/sys/fs/cgroup/cpu")
    memory_root = pathlib.Path("/sys/fs/cgroup/memory")
    usage0_ns = scalar(cpuacct_root / "cpuacct.usage")
    memory_current_v1 = scalar(memory_root / "memory.usage_in_bytes")
    if usage0_ns is not None and memory_current_v1 is not None:
        started_ns = time.monotonic_ns()
        time.sleep(0.1)
        usage1_ns = scalar(cpuacct_root / "cpuacct.usage")
        elapsed_usec = max(1.0, (time.monotonic_ns() - started_ns) / 1000)
        used_cores = max(0, (usage1_ns or usage0_ns) - usage0_ns) / 1000 / elapsed_usec
        memory_limit = scalar(memory_root / "memory.limit_in_bytes")
        if memory_limit is not None and memory_limit >= 1 << 60:
            memory_limit = None
        memory_limit_kib = memory_limit // 1024 if memory_limit else None
        memory_total_kib = requested_memory_kib or memory_limit_kib
        memory_used_kib = memory_current_v1 // 1024
        quota = scalar(cpu_root / "cpu.cfs_quota_us")
        period = scalar(cpu_root / "cpu.cfs_period_us")
        cpu_limit = quota / period if quota and quota > 0 and period and period > 0 else None
        cpuacct_stat = keyed(cpuacct_root / "cpuacct.stat")
        cpu_stat = keyed(cpu_root / "cpu.stat")
        try:
            tick_usec = 1000000 / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError):
            tick_usec = None
        peak = scalar(memory_root / "memory.max_usage_in_bytes")
        failures = scalar(memory_root / "memory.failcnt")
        resource = {
          "resource_accounting_scope": "cgroup-v1",
          "cpu_requested_cores": requested_cpu,
          "cpu_limit_cores": cpu_limit,
          "cpu_usage_cores": round(used_cores, 4),
          "cpu_usage_usec": (usage1_ns or usage0_ns) // 1000,
          "cpu_user_usec": round(cpuacct_stat["user"] * tick_usec) if tick_usec is not None and "user" in cpuacct_stat else None,
          "cpu_system_usec": round(cpuacct_stat["system"] * tick_usec) if tick_usec is not None and "system" in cpuacct_stat else None,
          "cpu_nr_throttled": cpu_stat.get("nr_throttled"),
          "cpu_throttled_usec": cpu_stat.get("throttled_time", 0) // 1000 if "throttled_time" in cpu_stat else None,
          "cpu_util_pct": round(100.0 * used_cores / requested_cpu, 2),
          "mem_requested_kib": requested_memory_kib,
          "mem_limit_kib": memory_limit_kib,
          "mem_total_kib": memory_total_kib,
          "mem_used_kib": memory_used_kib,
          "mem_available_kib": max(0, memory_total_kib - memory_used_kib),
          "memory_peak_kib": peak // 1024 if peak is not None else None,
          "memory_oom_events": failures,
          "memory_oom_kill_events": None,
        }
    else:
        total0, idle0 = cpu_times()
        time.sleep(0.1)
        total1, idle1 = cpu_times()
        delta = max(1, total1 - total0)
        mem = {}
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            try:
                mem[key] = int(value.strip().split()[0])
            except (ValueError, IndexError):
                pass
        resource = {
          "resource_accounting_scope": "host-proc-fallback",
          "cpu_util_pct": round(100.0 * (delta - (idle1 - idle0)) / delta, 2),
          "mem_total_kib": mem.get("MemTotal"),
          "mem_available_kib": mem.get("MemAvailable"),
          "mem_free_kib": mem.get("MemFree"),
          "mem_used_kib": ((mem.get("MemTotal") or 0) - (mem.get("MemAvailable") or 0)),
        }
sample = {
  "schema_version": 2,
  "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "epoch_s": int(time.time()),
  "role": "__ROLE__",
  "run_id": "__RUN_ID__",
  "hostname": os.uname().nodename,
  "nproc": len(os.sched_getaffinity(0)),
  "notes": "fallback_no_sprint_telemetry",
  **resource,
  "load1": os.getloadavg()[0],
  "load5": os.getloadavg()[1],
  "load15": os.getloadavg()[2],
  "gpus": [],
}
try:
    sample["uptime_s"] = float(pathlib.Path("/proc/uptime").read_text().split()[0])
except Exception:
    sample["uptime_s"] = None
query = [
    "index", "name", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "memory.free", "power.draw",
    "power.limit", "temperature.gpu", "clocks.gr", "clocks.sm", "clocks.mem",
]
binary = shutil.which("nvidia-smi")
gpu_result = subprocess.run(
    [binary, "--query-gpu=" + ",".join(query), "--format=csv,noheader,nounits"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
) if binary else None
sample["nvidia_smi_ok"] = bool(gpu_result and gpu_result.returncode == 0)
if sample["nvidia_smi_ok"]:
    keys = [
        "gpu_index", "gpu_name", "util_gpu_pct", "util_mem_pct",
        "mem_used_mib", "mem_total_mib", "mem_free_mib", "power_draw_w",
        "power_limit_w", "temp_gpu_c", "clock_graphics_mhz", "clock_sm_mhz",
        "clock_mem_mhz",
    ]
    for line in gpu_result.stdout.splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(keys):
            continue
        gpu = {keys[0]: int(number(values[0]) or 0), keys[1]: values[1]}
        gpu.update({key: number(value) for key, value in zip(keys[2:], values[2:])})
        sample["gpus"].append(gpu)
sample["gpu_count"] = len(sample["gpus"])
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


def _flatten_for_csv(
    sample: dict[str, Any], *, container_id: str
) -> list[dict[str, Any]]:
    base = {
        "ts_utc": sample.get("ts_utc"),
        "epoch_s": sample.get("epoch_s"),
        "role": sample.get("role"),
        "run_id": sample.get("run_id"),
        "container_id": container_id,
        "hostname": sample.get("hostname"),
        "resource_accounting_scope": sample.get("resource_accounting_scope"),
        "cpu_requested_cores": sample.get("cpu_requested_cores"),
        "cpu_limit_cores": sample.get("cpu_limit_cores"),
        "cpu_usage_cores": sample.get("cpu_usage_cores"),
        "cpu_usage_usec": sample.get("cpu_usage_usec"),
        "cpu_util_pct": sample.get("cpu_util_pct"),
        "load1": sample.get("load1"),
        "load5": sample.get("load5"),
        "load15": sample.get("load15"),
        "mem_requested_kib": sample.get("mem_requested_kib"),
        "mem_limit_kib": sample.get("mem_limit_kib"),
        "mem_used_kib": sample.get("mem_used_kib"),
        "mem_total_kib": sample.get("mem_total_kib"),
        "mem_available_kib": sample.get("mem_available_kib"),
        "memory_peak_kib": sample.get("memory_peak_kib"),
        "memory_oom_events": sample.get("memory_oom_events"),
        "memory_oom_kill_events": sample.get("memory_oom_kill_events"),
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
    "resource_accounting_scope",
    "cpu_requested_cores",
    "cpu_limit_cores",
    "cpu_usage_cores",
    "cpu_usage_usec",
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


def _exec_target(
    run: dict[str, Any],
    target_id: str,
    shell_command: str,
    *,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    """Exec in either a Harbor container (ta-) or agent GPU Sandbox (sb-).

    Modal's ``container exec`` CLI deliberately accepts only ``ta-`` IDs.
    Agent-controlled training workers are created through ``Sandbox.create``
    and are identified by ``sb-`` IDs, so polling them through that CLI always
    failed.  Use the Sandbox SDK for those workers while preserving the same
    CompletedProcess contract and per-run Modal profile.
    """
    if not target_id.startswith("sb-"):
        return sprintctl.exec_container(
            run,
            target_id,
            shell_command,
            check=False,
            timeout=timeout,
        )

    previous_profile = os.environ.get("MODAL_PROFILE")
    profile = run.get("modal_profile")
    if profile:
        os.environ["MODAL_PROFILE"] = str(profile)
    try:
        import modal

        process = modal.Sandbox.from_id(target_id).exec(
            "sh", "-c", shell_command, timeout=timeout
        )
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        returncode = int(process.wait())
    except Exception as exc:  # noqa: BLE001
        return subprocess.CompletedProcess(
            ["modal.Sandbox.exec", target_id],
            1,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if previous_profile is None:
            os.environ.pop("MODAL_PROFILE", None)
        else:
            os.environ["MODAL_PROFILE"] = previous_profile
    return subprocess.CompletedProcess(
        ["modal.Sandbox.exec", target_id],
        returncode,
        stdout=stdout,
        stderr=stderr,
    )


def sample_container(
    run: dict[str, Any],
    container_id: str,
    *,
    role: str,
    run_id: str,
) -> dict[str, Any] | None:
    result = _exec_target(
        run,
        container_id,
        _shell_oneshot(role, run_id),
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


def normalize_resource_contract(
    run: dict[str, Any], role: str, sample: dict[str, Any]
) -> None:
    """Normalize host-polled utilization to the run's requested resources.

    Long-lived immutable images can contain an older telemetry helper.  A
    one-shot host exec also does not inherit the sandbox launch environment,
    so that helper may report CPU utilization against Modal's larger cgroup
    burst ceiling and memory totals against the ceiling rather than the
    requested/billed baseline.  Preserve the observed ceiling fields, but use
    the trusted run contract for percentages and chart totals.
    """
    contract_key = {
        "cpu-agent": "cpu_agent",
        "training-gpu": "training_worker",
        "verifier-gpu": "verifier",
    }.get(role)
    contract = (run.get("resource_contract") or {}).get(contract_key or "") or {}

    requested_cpu = contract.get("physical_cpu_cores")
    if isinstance(requested_cpu, (int, float)) and requested_cpu > 0:
        requested_cpu = float(requested_cpu)
        sample["cpu_requested_cores"] = requested_cpu
        used_cpu = sample.get("cpu_usage_cores")
        if isinstance(used_cpu, (int, float)):
            sample["cpu_util_pct"] = round(100.0 * float(used_cpu) / requested_cpu, 2)

    requested_memory_mb = contract.get("memory_mb")
    if isinstance(requested_memory_mb, (int, float)) and requested_memory_mb > 0:
        requested_memory_kib = int(requested_memory_mb * 1024)
        sample["mem_requested_kib"] = requested_memory_kib
        sample["mem_total_kib"] = requested_memory_kib
        used_memory_kib = sample.get("mem_used_kib")
        if isinstance(used_memory_kib, (int, float)):
            sample["mem_available_kib"] = max(
                0, requested_memory_kib - int(used_memory_kib)
            )


def _container_role_probe(run: dict[str, Any], container_id: str) -> str:
    """Classify a sandbox into one unambiguous resource-accounting role."""
    result = _exec_target(
        run,
        container_id,
        "if [ -f /run/sprint-role ]; then cat /run/sprint-role; "
        "elif [ -x /opt/sprint-agent-supervisor.sh ]; then echo cpu-agent; "
        "elif command -v nvidia-smi >/dev/null 2>&1 && "
        "[ -f /opt/sprint-gpu-worker-run.py ]; then echo training-gpu; "
        "else echo verifier-gpu; fi",
        timeout=45,
    )
    text = ((result.stdout or "") + (result.stderr or "")).strip().splitlines()
    role = (text[-1] if text else "unknown").strip()
    if role in {"cpu-agent", "training-gpu", "verifier-gpu"}:
        return role
    return "unknown"


def active_training_container_ids(run: dict[str, Any]) -> set[str]:
    """Return only currently owned training sandboxes from durable job state.

    Historical job records deliberately retain their Sandbox IDs for
    provenance.  They are not telemetry targets after the job is terminal;
    repeatedly exec'ing every dead Sandbox makes monitor cycles grow with the
    number of jobs and delays terminal-policy reconciliation.
    """
    from event_runtime.compute import claim as gpu_claim
    from event_runtime.compute import worker as gpu_worker

    active: set[str] = set()
    # The append-only host registry is authoritative after claim and avoids a
    # Modal Volume list/get for every historical queue record on every poll.
    for job_id in gpu_worker.list_host_job_ids(run):
        job = gpu_worker.load_host_job(run, job_id) or {}
        if str(job.get("status") or "") not in gpu_claim.OWNED:
            continue
        sandbox_id = job.get("sandbox_id")
        if isinstance(sandbox_id, str) and sandbox_id.startswith(("sb-", "ta-")):
            active.add(sandbox_id)
    return active


def discover_sidecar_containers(
    state_dir: pathlib.Path, run: dict[str, Any]
) -> list[tuple[str, str]]:
    """GPU workers + verifiers in the same Modal app (not the CPU agent)."""
    app_id = sprintctl.discover_app_id(state_dir, run)
    if not app_id:
        return []
    containers = sprintctl.containers_for_app(run, app_id)
    agent = sprintctl.discover_agent_container(state_dir, run)
    # Also include sandboxes recorded on dispatched GPU jobs.
    known_gpu: set[str] = set()
    try:
        known_gpu = active_training_container_ids(run)
    except Exception:  # noqa: BLE001
        pass

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for container in list(containers) + sorted(known_gpu):
        if container in seen:
            continue
        seen.add(container)
        if agent and container == agent:
            continue
        try:
            role = _container_role_probe(run, container)
        except Exception:  # noqa: BLE001
            role = "verifier-gpu" if container.startswith("ta-") else "training-gpu"
        if role == "cpu-agent":
            continue
        if role == "unknown":
            role = "training-gpu" if container in known_gpu else "verifier-gpu"
        out.append((role, container))
    return out


def gpu_job_metadata(run: dict[str, Any], container_id: str) -> dict[str, Any]:
    try:
        from event_runtime.compute import worker as gpu_worker

        for job_id in gpu_worker.list_job_ids(run):
            job = gpu_worker.load_job(run, job_id) or {}
            if str(job.get("sandbox_id") or "") == container_id:
                return {
                    "job_id": job_id,
                    "attempt": int(job.get("attempt") or 0),
                    "lease_id": job.get("lease_id"),
                }
    except Exception:  # noqa: BLE001
        pass
    return {}


def poll_once(run_id: str) -> dict[str, Any]:
    state_dir, run = sprintctl.load_run(run_id)
    pathlib.Path("/data/.keepalive").touch()
    out_dir = _state_telemetry_dir(state_dir)
    monitor_csv = _monitor_csv_path(run_id)
    state_csv = out_dir / "host-samples.csv"
    summary: dict[str, Any] = {
        "run_id": run_id,
        "polled_at": sprintctl.utc_now(),
        "agent_container_id": None,
        "verifier_gpu_container_ids": [],
        "training_gpu_container_ids": [],
        "samples": 0,
        "errors": [],
    }

    agent = sprintctl.discover_agent_container(state_dir, run)
    targets: list[tuple[str, str]] = []
    if agent:
        summary["agent_container_id"] = agent
        targets.append(("cpu-agent", agent))
    else:
        summary["errors"].append("agent_container_unavailable")

    try:
        sidecars = discover_sidecar_containers(state_dir, run)
    except Exception as exc:  # noqa: BLE001
        sidecars = []
        summary["errors"].append(f"sidecar_discover_failed:{type(exc).__name__}")
    for role, container in sidecars:
        targets.append((role, container))
        if role == "verifier-gpu":
            summary["verifier_gpu_container_ids"].append(container)
        elif role == "training-gpu":
            summary["training_gpu_container_ids"].append(container)

    for role, container_id in targets:
        try:
            sample = sample_container(run, container_id, role=role, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"{role}:{container_id}:{type(exc).__name__}")
            continue
        if not sample:
            summary["errors"].append(f"{role}:{container_id}:empty")
            continue
        normalize_resource_contract(run, role, sample)
        if role == "training-gpu":
            for key, value in gpu_job_metadata(run, container_id).items():
                sample.setdefault(key, value)
        elif role == "cpu-agent":
            # Internal provenance partition; one authoritative CPU process is
            # the only supported execution model.
            sample["cpu_attempt"] = 1
        _append_jsonl(out_dir / "host-samples.jsonl", sample)
        rows = _flatten_for_csv(sample, container_id=container_id)
        _append_csv(state_csv, rows)
        _append_csv(monitor_csv, rows)
        # Also mirror into trial artifacts when the job/trial is known.
        job, trial = sprintctl.discover_job_and_trial(state_dir, run)
        if trial is not None:
            trial_telem = trial / "artifacts" / "logs" / "artifacts" / "telemetry"
            trial_telem.mkdir(parents=True, exist_ok=True)
            _append_jsonl(trial_telem / "host-samples.jsonl", sample)
            _append_csv(trial_telem / "host-samples.csv", rows)
        summary["samples"] += 1

    sprintctl.atomic_write_json(out_dir / "host-latest.json", summary, mode=0o600)
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
