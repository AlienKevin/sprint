#!/usr/bin/env python3
"""GPU + CPU/system telemetry sampler for G1 sprint Harbor sandboxes.

Writes continuously under /logs/artifacts/telemetry/ (Harbor convention mount)
and optionally mirrors under /durable/runs/<run_id>/telemetry/.

Never dumps environment variables or process command lines (secrets risk).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from sprint_gpu_pipeline import collector_capability, collect_pipeline_metrics

SCHEMA_VERSION = 3
DEFAULT_INTERVAL = 20
SECRET_NAME_RE = re.compile(
    r"(api[_-]?key|oauth|token|password|secret|authorization|bearer)",
    re.IGNORECASE,
)

GPU_QUERY = (
    "index,name,uuid,"
    "utilization.gpu,utilization.memory,"
    "memory.used,memory.total,memory.free,"
    "power.draw,power.limit,"
    "temperature.gpu,"
    "clocks.current.graphics,clocks.current.sm,clocks.current.memory,"
    "clocks.max.graphics,clocks.max.sm,clocks.max.memory,"
    "pcie.link.gen.current,pcie.link.width.current,"
    "ecc.mode.current,"
    "ecc.errors.corrected.volatile.total,"
    "ecc.errors.uncorrected.volatile.total,"
    "driver_version"
)

GPU_QUERY_FIELDS = [
    "gpu_index",
    "gpu_name",
    "gpu_uuid",
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
    "clock_max_graphics_mhz",
    "clock_max_sm_mhz",
    "clock_max_mem_mhz",
    "pcie_link_gen",
    "pcie_link_width",
    "ecc_mode",
    "ecc_corrected_volatile",
    "ecc_uncorrected_volatile",
    "driver_version",
]

GPU_CSV_FIELDS = GPU_QUERY_FIELDS + [
    "pipeline_metrics_source",
    "pipeline_metrics_group",
    "pipeline_metrics_status",
    "pipeline_metrics_sample_count",
    "pipeline_metrics_window_ms",
    "pipeline_metrics_error",
    "sm_active_pct",
    "sm_occupancy_pct",
    "tensor_pipe_active_pct",
    "fp32_fma_pipe_active_pct",
    "fp16_instruction_pct_of_peak_active",
    "dram_throughput_pct",
]

SAMPLE_BASE_FIELDS = [
    "ts_utc",
    "epoch_s",
    "role",
    "run_id",
    "job_id",
    "cpu_attempt",
    "attempt",
    "lease_id",
    "hostname",
    "sample_index",
    "uptime_s",
    "nproc",
    "load1",
    "load5",
    "load15",
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
    "cpu_per_core_pct",
    "mem_requested_kib",
    "mem_limit_kib",
    "mem_total_kib",
    "mem_used_kib",
    "mem_available_kib",
    "mem_free_kib",
    "swap_total_kib",
    "swap_used_kib",
    "swap_free_kib",
    "memory_peak_kib",
    "memory_oom_events",
    "memory_oom_kill_events",
    "disk_root_used_pct",
    "disk_root_used_kib",
    "disk_root_total_kib",
    "disk_tmp_used_pct",
    "disk_tmp_used_kib",
    "disk_tmp_total_kib",
    "disk_app_used_pct",
    "disk_app_used_kib",
    "disk_app_total_kib",
    "net_rx_bytes",
    "net_tx_bytes",
    "net_rx_packets",
    "net_tx_packets",
    "gpu_count",
    "pcie_rx_mib_s",
    "pcie_tx_mib_s",
    "nvidia_smi_ok",
    "notes",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path: pathlib.Path, text: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def run_cmd(args: list[str], timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def parse_num(value: str) -> float | int | str | None:
    text = (value or "").strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NA", "NONE"}:
        return None
    text = text.replace(",", "")
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def redact_name(name: str | None) -> str:
    if not name:
        return ""
    if SECRET_NAME_RE.search(name):
        return "[redacted]"
    # Never keep argv-looking strings; nvidia-smi usually returns basename only.
    base = name.split("\0", 1)[0].split()[0]
    return base[:256]


def read_uptime_s() -> float | None:
    try:
        return float(pathlib.Path("/proc/uptime").read_text().split()[0])
    except (OSError, IndexError, ValueError):
        return None


def read_loadavg() -> tuple[float | None, float | None, float | None]:
    try:
        parts = pathlib.Path("/proc/loadavg").read_text().split()
        return float(parts[0]), float(parts[1]), float(parts[2])
    except (OSError, IndexError, ValueError):
        return None, None, None


def read_meminfo() -> dict[str, int | None]:
    keys = {
        "MemTotal": "mem_total_kib",
        "MemAvailable": "mem_available_kib",
        "MemFree": "mem_free_kib",
        "SwapTotal": "swap_total_kib",
        "SwapFree": "swap_free_kib",
    }
    out: dict[str, int | None] = {v: None for v in keys.values()}
    try:
        text = pathlib.Path("/proc/meminfo").read_text()
    except OSError:
        return out
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        num = rest.strip().split()[0]
        try:
            values[key] = int(num)
        except ValueError:
            continue
    for src, dst in keys.items():
        out[dst] = values.get(src)
    total = values.get("MemTotal")
    avail = values.get("MemAvailable")
    free = values.get("MemFree")
    if total is not None and avail is not None:
        out["mem_used_kib"] = total - avail
    elif total is not None and free is not None:
        out["mem_used_kib"] = total - free
    else:
        out["mem_used_kib"] = None
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    if swap_total is not None and swap_free is not None:
        out["swap_used_kib"] = swap_total - swap_free
    else:
        out["swap_used_kib"] = None
    return out


def _read_int(path: pathlib.Path) -> int | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_keyed_ints(path: pathlib.Path) -> dict[str, int]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    values: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


def current_cgroup_v2_root(
    mount: pathlib.Path = pathlib.Path("/sys/fs/cgroup"),
    membership: pathlib.Path = pathlib.Path("/proc/self/cgroup"),
) -> pathlib.Path:
    """Resolve this process's cgroup-v2 directory inside or outside a namespace."""
    try:
        lines = membership.read_text().splitlines()
    except OSError:
        return mount
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            candidate = mount / parts[2].lstrip("/")
            if (candidate / "cpu.stat").is_file():
                return candidate
    return mount


def read_cgroup_v2(
    root: pathlib.Path | None = None, *, captured_ns: int | None = None
) -> tuple[dict[str, Any], dict[str, int | float] | None]:
    """Read cgroup-local CPU/memory counters used for Modal billing telemetry."""
    root = root or current_cgroup_v2_root()
    cpu_stat = _read_keyed_ints(root / "cpu.stat")
    memory_current = _read_int(root / "memory.current")
    if "usage_usec" not in cpu_stat or memory_current is None:
        return {}, None

    requested_cpu = None
    requested_cpu_raw = os.environ.get("SPRINT_REQUESTED_CPU_CORES")
    if requested_cpu_raw:
        try:
            requested_cpu = float(requested_cpu_raw)
        except ValueError:
            requested_cpu = None
    requested_memory_kib = None
    requested_memory_raw = os.environ.get("SPRINT_REQUESTED_MEMORY_MIB")
    if requested_memory_raw:
        try:
            requested_memory_kib = int(requested_memory_raw) * 1024
        except ValueError:
            requested_memory_kib = None

    cpu_limit = None
    try:
        quota_text, period_text = (root / "cpu.max").read_text().split()[:2]
        if quota_text != "max":
            cpu_limit = int(quota_text) / int(period_text)
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        cpu_limit = None
    memory_limit_bytes = _read_int(root / "memory.max")
    memory_limit_kib = (
        memory_limit_bytes // 1024 if memory_limit_bytes is not None else None
    )
    memory_total_kib = requested_memory_kib or memory_limit_kib
    memory_current_kib = memory_current // 1024
    memory_events = _read_keyed_ints(root / "memory.events")
    swap_current = _read_int(root / "memory.swap.current")
    swap_limit = _read_int(root / "memory.swap.max")
    memory_peak = _read_int(root / "memory.peak")
    denominator = requested_cpu or cpu_limit
    metrics: dict[str, Any] = {
        "resource_accounting_scope": "cgroup-v2",
        "cpu_requested_cores": requested_cpu,
        "cpu_limit_cores": cpu_limit,
        "cpu_usage_usec": cpu_stat["usage_usec"],
        "cpu_user_usec": cpu_stat.get("user_usec"),
        "cpu_system_usec": cpu_stat.get("system_usec"),
        "cpu_nr_throttled": cpu_stat.get("nr_throttled"),
        "cpu_throttled_usec": cpu_stat.get("throttled_usec"),
        "mem_requested_kib": requested_memory_kib,
        "mem_limit_kib": memory_limit_kib,
        "mem_total_kib": memory_total_kib,
        "mem_used_kib": memory_current_kib,
        "mem_available_kib": (
            max(0, memory_total_kib - memory_current_kib)
            if memory_total_kib is not None
            else None
        ),
        "mem_free_kib": None,
        "swap_total_kib": swap_limit // 1024 if swap_limit is not None else None,
        "swap_used_kib": swap_current // 1024 if swap_current is not None else None,
        "swap_free_kib": (
            max(0, (swap_limit - (swap_current or 0)) // 1024)
            if swap_limit is not None
            else None
        ),
        "memory_peak_kib": memory_peak // 1024 if memory_peak is not None else None,
        "memory_oom_events": memory_events.get("oom"),
        "memory_oom_kill_events": memory_events.get("oom_kill"),
    }
    snapshot: dict[str, int | float] = {
        "captured_ns": captured_ns if captured_ns is not None else time.monotonic_ns(),
        "usage_usec": cpu_stat["usage_usec"],
    }
    if denominator is not None:
        snapshot["denominator_cores"] = denominator
    return metrics, snapshot


def read_cgroup_v1(
    mount: pathlib.Path = pathlib.Path("/sys/fs/cgroup"),
    *,
    captured_ns: int | None = None,
) -> tuple[dict[str, Any], dict[str, int | float] | None]:
    """Read container-local counters from Modal's cgroup-v1 controller mounts."""
    cpuacct = mount / "cpuacct"
    cpu = mount / "cpu"
    memory = mount / "memory"
    usage_ns = _read_int(cpuacct / "cpuacct.usage")
    memory_current = _read_int(memory / "memory.usage_in_bytes")
    if usage_ns is None or memory_current is None:
        return {}, None

    requested_cpu = None
    try:
        requested_cpu = float(os.environ.get("SPRINT_REQUESTED_CPU_CORES", ""))
    except ValueError:
        pass
    requested_memory_kib = None
    try:
        requested_memory_kib = (
            int(os.environ.get("SPRINT_REQUESTED_MEMORY_MIB", "")) * 1024
        )
    except ValueError:
        pass

    quota = _read_int(cpu / "cpu.cfs_quota_us")
    period = _read_int(cpu / "cpu.cfs_period_us")
    cpu_limit = (
        quota / period
        if quota is not None and quota > 0 and period is not None and period > 0
        else None
    )
    memory_limit_bytes = _read_int(memory / "memory.limit_in_bytes")
    if memory_limit_bytes is not None and memory_limit_bytes >= 1 << 60:
        memory_limit_bytes = None
    memory_limit_kib = (
        memory_limit_bytes // 1024 if memory_limit_bytes is not None else None
    )
    memory_total_kib = requested_memory_kib or memory_limit_kib
    memory_current_kib = memory_current // 1024
    memory_peak = _read_int(memory / "memory.max_usage_in_bytes")
    memory_failures = _read_int(memory / "memory.failcnt")
    memsw_current = _read_int(memory / "memory.memsw.usage_in_bytes")
    memsw_limit = _read_int(memory / "memory.memsw.limit_in_bytes")
    cpuacct_stat = _read_keyed_ints(cpuacct / "cpuacct.stat")
    cpu_stat = _read_keyed_ints(cpu / "cpu.stat")
    try:
        tick_usec = 1_000_000 / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError):
        tick_usec = None
    usage_usec = usage_ns // 1000
    denominator = requested_cpu or cpu_limit
    metrics: dict[str, Any] = {
        "resource_accounting_scope": "cgroup-v1",
        "cpu_requested_cores": requested_cpu,
        "cpu_limit_cores": cpu_limit,
        "cpu_usage_usec": usage_usec,
        "cpu_user_usec": (
            round(cpuacct_stat["user"] * tick_usec)
            if tick_usec is not None and "user" in cpuacct_stat
            else None
        ),
        "cpu_system_usec": (
            round(cpuacct_stat["system"] * tick_usec)
            if tick_usec is not None and "system" in cpuacct_stat
            else None
        ),
        "cpu_nr_throttled": cpu_stat.get("nr_throttled"),
        "cpu_throttled_usec": (
            cpu_stat["throttled_time"] // 1000 if "throttled_time" in cpu_stat else None
        ),
        "mem_requested_kib": requested_memory_kib,
        "mem_limit_kib": memory_limit_kib,
        "mem_total_kib": memory_total_kib,
        "mem_used_kib": memory_current_kib,
        "mem_available_kib": (
            max(0, memory_total_kib - memory_current_kib)
            if memory_total_kib is not None
            else None
        ),
        "mem_free_kib": None,
        "swap_total_kib": (
            max(0, (memsw_limit - (memory_limit_bytes or 0)) // 1024)
            if memsw_limit is not None and memory_limit_bytes is not None
            else None
        ),
        "swap_used_kib": (
            max(0, (memsw_current - memory_current) // 1024)
            if memsw_current is not None
            else None
        ),
        "swap_free_kib": None,
        "memory_peak_kib": memory_peak // 1024 if memory_peak is not None else None,
        "memory_oom_events": memory_failures,
        "memory_oom_kill_events": None,
    }
    if metrics["swap_total_kib"] is not None and metrics["swap_used_kib"] is not None:
        metrics["swap_free_kib"] = max(
            0, metrics["swap_total_kib"] - metrics["swap_used_kib"]
        )
    snapshot: dict[str, int | float] = {
        "captured_ns": captured_ns if captured_ns is not None else time.monotonic_ns(),
        "usage_usec": usage_usec,
    }
    if denominator is not None:
        snapshot["denominator_cores"] = denominator
    return metrics, snapshot


def read_cgroup(
    *, captured_ns: int | None = None
) -> tuple[dict[str, Any], dict[str, int | float] | None]:
    metrics, snapshot = read_cgroup_v2(captured_ns=captured_ns)
    if metrics:
        return metrics, snapshot
    return read_cgroup_v1(captured_ns=captured_ns)


def cgroup_cpu_delta(
    previous: dict[str, int | float] | None,
    current: dict[str, int | float] | None,
) -> tuple[float | None, float | None]:
    """Return absolute used cores and percent of requested/quota CPU capacity."""
    if not previous or not current:
        return None, None
    elapsed_usec = (int(current["captured_ns"]) - int(previous["captured_ns"])) / 1000
    usage_delta = int(current["usage_usec"]) - int(previous["usage_usec"])
    if elapsed_usec <= 0 or usage_delta < 0:
        return None, None
    used_cores = usage_delta / elapsed_usec
    denominator = current.get("denominator_cores")
    percent = (
        100.0 * used_cores / float(denominator)
        if isinstance(denominator, (int, float)) and denominator > 0
        else None
    )
    return round(used_cores, 4), round(percent, 2) if percent is not None else None


def read_net_counters() -> dict[str, int]:
    rx_b = tx_b = rx_p = tx_p = 0
    try:
        lines = pathlib.Path("/proc/net/dev").read_text().splitlines()[2:]
    except OSError:
        return {
            "net_rx_bytes": 0,
            "net_tx_bytes": 0,
            "net_rx_packets": 0,
            "net_tx_packets": 0,
        }
    for line in lines:
        if ":" not in line:
            continue
        iface, rest = line.split(":", 1)
        name = iface.strip()
        if name == "lo":
            continue
        cols = rest.split()
        if len(cols) < 10:
            continue
        try:
            rx_b += int(cols[0])
            rx_p += int(cols[1])
            tx_b += int(cols[8])
            tx_p += int(cols[9])
        except ValueError:
            continue
    return {
        "net_rx_bytes": rx_b,
        "net_tx_bytes": tx_b,
        "net_rx_packets": rx_p,
        "net_tx_packets": tx_p,
    }


def disk_usage(path: str) -> tuple[float | None, int | None, int | None]:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None, None, None
    total_kib = usage.total // 1024
    used_kib = (usage.total - usage.free) // 1024
    pct = (100.0 * used_kib / total_kib) if total_kib else None
    return pct, used_kib, total_kib


def read_proc_stat() -> tuple[list[int], list[list[int]]]:
    total: list[int] = []
    cores: list[list[int]] = []
    try:
        lines = pathlib.Path("/proc/stat").read_text().splitlines()
    except OSError:
        return total, cores
    for line in lines:
        if not line.startswith("cpu"):
            break
        parts = line.split()
        try:
            nums = [int(x) for x in parts[1:]]
        except ValueError:
            continue
        if parts[0] == "cpu":
            total = nums
        else:
            cores.append(nums)
    return total, cores


def cpu_util_from_delta(prev: list[int] | None, cur: list[int]) -> float | None:
    if not prev or not cur or len(prev) < 4 or len(cur) < 4:
        return None
    prev_idle = prev[3] + (prev[4] if len(prev) > 4 else 0)
    cur_idle = cur[3] + (cur[4] if len(cur) > 4 else 0)
    prev_total = sum(prev)
    cur_total = sum(cur)
    total_d = cur_total - prev_total
    idle_d = cur_idle - prev_idle
    if total_d <= 0:
        return None
    return round(100.0 * (1.0 - (idle_d / total_d)), 2)


def per_core_util(
    prev_cores: list[list[int]] | None, cur_cores: list[list[int]]
) -> list[float | None]:
    if not prev_cores:
        return [None] * len(cur_cores)
    out: list[float | None] = []
    for idx, cur in enumerate(cur_cores):
        prev = prev_cores[idx] if idx < len(prev_cores) else None
        out.append(cpu_util_from_delta(prev, cur))
    return out


def nvidia_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def snapshot_gpu_static() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": utc_now(),
        "nvidia_smi": nvidia_available(),
        "driver_version": None,
        "cuda_version": None,
        "gpus": [],
        "gpu_pipeline": collector_capability(),
        "notes": [],
    }
    if not nvidia_available():
        payload["notes"].append("nvidia-smi_missing")
        return payload
    # CUDA version is reported by the tool banner / --query-gpu in some drivers;
    # also try nvcc as a fallback without leaking paths that look secret-bearing.
    q = run_cmd(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ]
    )
    if q.returncode != 0:
        payload["notes"].append(f"nvidia-smi_query_rc={q.returncode}")
        return payload
    gpus = []
    for line in q.stdout.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < len(GPU_QUERY_FIELDS):
            continue
        row = {field: parse_num(cols[i]) for i, field in enumerate(GPU_QUERY_FIELDS)}
        gpus.append(row)
        if payload["driver_version"] is None:
            payload["driver_version"] = row.get("driver_version")
    payload["gpus"] = gpus
    # Prefer nvidia-smi header for CUDA version (safe; no env dump).
    banner = run_cmd(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*([0-9.]+)", banner.stdout or "")
    if match:
        payload["cuda_version"] = match.group(1)
    elif shutil.which("nvcc"):
        nvcc = run_cmd(["nvcc", "--version"])
        match = re.search(r"release\s+([0-9.]+)", nvcc.stdout or "")
        if match:
            payload["cuda_version"] = match.group(1)
    return payload


def sample_gpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    notes: list[str] = []
    gpus: list[dict[str, Any]] = []
    procs: list[dict[str, Any]] = []
    extras: dict[str, Any] = {
        "pcie_rx_mib_s": None,
        "pcie_tx_mib_s": None,
        "nvidia_smi_ok": False,
    }
    if not nvidia_available():
        notes.append("nvidia-smi_missing")
        return gpus, procs, {**extras, "notes": notes}
    q = run_cmd(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ]
    )
    if q.returncode == 0:
        extras["nvidia_smi_ok"] = True
        for line in q.stdout.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < len(GPU_QUERY_FIELDS):
                continue
            gpus.append(
                {field: parse_num(cols[i]) for i, field in enumerate(GPU_QUERY_FIELDS)}
            )
    else:
        notes.append(f"nvidia-smi_query_rc={q.returncode}")

    if gpus:
        # Fairness contracts permit one GPU per worker.  CUPTI PM sampling is
        # device scoped and observes the separate training/verifier process.
        gpus[0].update(collect_pipeline_metrics())

    p = run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,gpu_bus_id,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if p.returncode == 0:
        for line in p.stdout.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 5:
                continue
            procs.append(
                {
                    "gpu_uuid": cols[0],
                    "gpu_bus_id": cols[1],
                    "pid": parse_num(cols[2]),
                    "process_name": redact_name(cols[3]),
                    "used_memory_mib": parse_num(cols[4]),
                }
            )
    else:
        notes.append(f"nvidia-smi_apps_rc={p.returncode}")

    # Best-effort PCIe throughput via one-shot dmon.
    dmon = run_cmd(["nvidia-smi", "dmon", "-c", "1", "-s", "t"], timeout=10.0)
    if dmon.returncode == 0:
        rx_vals: list[float] = []
        tx_vals: list[float] = []
        for line in dmon.stdout.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Typical: gpu pwr gtemp mtemp sm mem enc dec mclk pclk
            # With -s t: gpu rxpci txpci
            if len(parts) >= 3 and parts[0].isdigit():
                rx = parse_num(parts[1])
                tx = parse_num(parts[2])
                if isinstance(rx, (int, float)):
                    rx_vals.append(float(rx))
                if isinstance(tx, (int, float)):
                    tx_vals.append(float(tx))
        if rx_vals:
            extras["pcie_rx_mib_s"] = round(sum(rx_vals), 3)
        if tx_vals:
            extras["pcie_tx_mib_s"] = round(sum(tx_vals), 3)
    else:
        notes.append("pcie_dmon_unavailable")

    extras["notes"] = notes
    return gpus, procs, extras


def build_sample(
    *,
    role: str,
    run_id: str,
    sample_index: int,
    prev_cpu: list[int] | None,
    prev_cores: list[list[int]] | None,
    prev_cgroup_cpu: dict[str, int | float] | None,
) -> tuple[
    dict[str, Any],
    list[int],
    list[list[int]],
    dict[str, int | float] | None,
]:
    cur_cpu, cur_cores = read_proc_stat()
    load1, load5, load15 = read_loadavg()
    cgroup, cur_cgroup_cpu = read_cgroup()
    if cgroup:
        mem = cgroup
        cpu_usage_cores, cpu_util_pct = cgroup_cpu_delta(
            prev_cgroup_cpu, cur_cgroup_cpu
        )
        mem["cpu_usage_cores"] = cpu_usage_cores
        per_core: list[float | None] = []
    else:
        mem = {
            "resource_accounting_scope": "host-proc-fallback",
            **read_meminfo(),
        }
        cpu_util_pct = cpu_util_from_delta(prev_cpu, cur_cpu)
        per_core = per_core_util(prev_cores, cur_cores)
    net = read_net_counters()
    root_pct, root_used, root_total = disk_usage("/")
    tmp_pct, tmp_used, tmp_total = disk_usage("/tmp")
    app_pct, app_used, app_total = disk_usage("/app")
    gpus, procs, gpu_extra = sample_gpus()
    notes = list(gpu_extra.get("notes") or [])
    sample: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": utc_now(),
        "epoch_s": int(time.time()),
        "role": role,
        "run_id": run_id or None,
        "job_id": None,
        "cpu_attempt": (
            int(os.environ["SPRINT_CPU_LAUNCH_ATTEMPT"])
            if os.environ.get("SPRINT_CPU_LAUNCH_ATTEMPT", "").isdigit()
            else None
        ),
        "attempt": (
            int(os.environ["SPRINT_GPU_ATTEMPT"])
            if os.environ.get("SPRINT_GPU_ATTEMPT", "").isdigit()
            else None
        ),
        "lease_id": os.environ.get("SPRINT_GPU_LEASE_ID") or None,
        "hostname": os.uname().nodename,
        "sample_index": sample_index,
        "uptime_s": read_uptime_s(),
        "nproc": len(os.sched_getaffinity(0)),
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "cpu_util_pct": cpu_util_pct,
        "cpu_per_core_pct": per_core,
        **mem,
        "disk_root_used_pct": root_pct,
        "disk_root_used_kib": root_used,
        "disk_root_total_kib": root_total,
        "disk_tmp_used_pct": tmp_pct,
        "disk_tmp_used_kib": tmp_used,
        "disk_tmp_total_kib": tmp_total,
        "disk_app_used_pct": app_pct,
        "disk_app_used_kib": app_used,
        "disk_app_total_kib": app_total,
        **net,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "gpu_processes": procs,
        "pcie_rx_mib_s": gpu_extra.get("pcie_rx_mib_s"),
        "pcie_tx_mib_s": gpu_extra.get("pcie_tx_mib_s"),
        "nvidia_smi_ok": gpu_extra.get("nvidia_smi_ok"),
        "notes": ",".join(notes) if notes else "",
    }
    return sample, cur_cpu, cur_cores, cur_cgroup_cpu


def flatten_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """One CSV row per GPU (or a single row when no GPUs)."""
    base = {field: sample.get(field) for field in SAMPLE_BASE_FIELDS}
    base["cpu_per_core_pct"] = json.dumps(
        sample.get("cpu_per_core_pct") or [], separators=(",", ":")
    )
    gpus = sample.get("gpus") or []
    if not gpus:
        row = dict(base)
        for field in GPU_CSV_FIELDS:
            row[field] = None
        return [row]
    rows = []
    for gpu in gpus:
        row = dict(base)
        for field in GPU_CSV_FIELDS:
            row[field] = gpu.get(field)
        rows.append(row)
    return rows


def append_jsonl(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def ensure_csv_header(path: pathlib.Path, fields: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def append_csv_rows(
    path: pathlib.Path, fields: list[str], rows: list[dict[str, Any]]
) -> None:
    ensure_csv_header(path, fields)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def mirror_file(src: pathlib.Path, dest: pathlib.Path) -> None:
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    shutil.copy2(src, tmp)
    os.chmod(tmp, 0o600)
    os.replace(tmp, dest)


def flush_durable_mount(durable_dir: str) -> None:
    """Best-effort volume flush so a preemption loses at most one interval."""
    root = pathlib.Path(durable_dir)
    if not root.is_dir():
        return
    try:
        os.sync()
    except OSError:
        pass
    try:
        subprocess.run(
            ["sync", str(root)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        pass


class TelemetryWriter:
    def __init__(
        self,
        out_dirs: list[pathlib.Path],
        role: str,
        run_id: str,
        *,
        job_id: str = "",
        durable_dir: str = "/durable",
    ):
        self.out_dirs = out_dirs
        self.role = role
        self.run_id = run_id
        self.job_id = job_id
        self.durable_dir = durable_dir
        self.csv_fields = SAMPLE_BASE_FIELDS + GPU_CSV_FIELDS
        for directory in out_dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def write_snapshot(self, snapshot: dict[str, Any]) -> None:
        text = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
        for directory in self.out_dirs:
            atomic_write_text(directory / "snapshot.json", text)
        flush_durable_mount(self.durable_dir)

    def write_sample(self, sample: dict[str, Any]) -> None:
        if self.job_id and not sample.get("job_id"):
            sample["job_id"] = self.job_id
        rows = flatten_sample(sample)
        for directory in self.out_dirs:
            append_jsonl(directory / "samples.jsonl", sample)
            append_csv_rows(directory / "samples.csv", self.csv_fields, rows)
            for proc in sample.get("gpu_processes") or []:
                append_jsonl(
                    directory / "processes.jsonl",
                    {
                        "ts_utc": sample["ts_utc"],
                        "sample_index": sample["sample_index"],
                        "role": self.role,
                        "run_id": self.run_id or None,
                        "job_id": self.job_id or None,
                        "cpu_attempt": sample.get("cpu_attempt"),
                        "attempt": sample.get("attempt"),
                        "lease_id": sample.get("lease_id"),
                        **proc,
                    },
                )
            atomic_write_text(
                directory / "latest.json",
                json.dumps(sample, indent=2, sort_keys=True) + "\n",
            )
        # Live durable write: do not wait for restic (~5 min).
        flush_durable_mount(self.durable_dir)


def resolve_out_dirs(
    out_dir: str,
    run_id: str,
    durable_dir: str,
    *,
    role: str = "cpu-agent",
    job_id: str = "",
) -> list[pathlib.Path]:
    """Local artifacts + preemption-safe durable mirrors.

    The CPU agent owns the canonical flat durable stream. Training and
    verifier GPUs write independent role/job streams under the same run
    telemetry tree so neither can be confused with the other's compute.
    """
    dirs = [pathlib.Path(out_dir)]
    if run_id and pathlib.Path(durable_dir).exists():
        durable = pathlib.Path(durable_dir) / "runs" / run_id / "telemetry"
        if role == "cpu-agent":
            dirs.append(durable)
        else:
            dirs.append(durable / "by-role" / role)
            if job_id:
                dirs.append(durable / "by-job" / job_id)
            # Merged GPU stream (role-tagged lines), separate from the CPU-agent
            # root file to avoid concurrent append races on one inode.
            dirs.append(durable / "gpu-stream")
    seen: set[str] = set()
    unique: list[pathlib.Path] = []
    for path in dirs:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def write_pidfile(path: pathlib.Path) -> None:
    atomic_write_text(path, f"{os.getpid()}\n")


def already_running(pidfile: pathlib.Path) -> bool:
    if not pidfile.exists():
        return False
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return False
    if pid <= 1 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # Confirm cmdline looks like telemetry.
    try:
        cmdline = (
            pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
        )
    except OSError:
        return False
    return "sprint-telemetry" in cmdline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        default="cpu-agent",
        choices=["cpu-agent", "training-gpu", "verifier-gpu", "host-controller"],
    )
    parser.add_argument("--run-id", default=os.environ.get("SPRINT_RUN_ID", ""))
    parser.add_argument(
        "--job-id",
        default=os.environ.get("SPRINT_GPU_JOB_ID", ""),
        help="GPU training job id (tagged into samples + durable by-job path)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("SPRINT_TELEMETRY_DIR", "/logs/artifacts/telemetry"),
    )
    parser.add_argument("--durable-dir", default="/durable")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=int(os.environ.get("SPRINT_TELEMETRY_INTERVAL", DEFAULT_INTERVAL)),
    )
    parser.add_argument("--once", action="store_true", help="Take one sample and exit")
    parser.add_argument(
        "--pidfile",
        default="/run/sprint-telemetry.pid",
        help="Pidfile used for idempotent daemon start",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start even if pidfile points at a live telemetry process",
    )
    args = parser.parse_args()
    if args.interval_seconds < 5:
        print("interval-seconds must be >= 5", file=sys.stderr)
        return 2

    pidfile = pathlib.Path(args.pidfile)
    if not args.once and not args.force and already_running(pidfile):
        print("telemetry already running", flush=True)
        return 0

    out_dirs = resolve_out_dirs(
        args.out_dir,
        args.run_id,
        args.durable_dir,
        role=args.role,
        job_id=args.job_id,
    )
    writer = TelemetryWriter(
        out_dirs,
        args.role,
        args.run_id,
        job_id=args.job_id,
        durable_dir=args.durable_dir,
    )
    stop = False

    def _stop(signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    if not args.once:
        write_pidfile(pidfile)

    snapshot = snapshot_gpu_static()
    snapshot.update(
        {
            "role": args.role,
            "run_id": args.run_id or None,
            "job_id": args.job_id or None,
            "cpu_attempt": (
                int(os.environ["SPRINT_CPU_LAUNCH_ATTEMPT"])
                if os.environ.get("SPRINT_CPU_LAUNCH_ATTEMPT", "").isdigit()
                else None
            ),
            "attempt": (
                int(os.environ["SPRINT_GPU_ATTEMPT"])
                if os.environ.get("SPRINT_GPU_ATTEMPT", "").isdigit()
                else None
            ),
            "lease_id": os.environ.get("SPRINT_GPU_LEASE_ID") or None,
            "interval_seconds": args.interval_seconds,
            "out_dirs": [str(path) for path in out_dirs],
            "hostname": os.uname().nodename,
            "nproc": len(os.sched_getaffinity(0)),
            "uptime_s": read_uptime_s(),
        }
    )
    writer.write_snapshot(snapshot)

    prev_cpu: list[int] | None = None
    prev_cores: list[list[int]] | None = None
    prev_cgroup_cpu: dict[str, int | float] | None = None
    # Prime CPU counters so the first sample has a util delta.
    prev_cpu, prev_cores = read_proc_stat()
    _, prev_cgroup_cpu = read_cgroup()
    time.sleep(0.25)

    index = 0
    while not stop:
        sample, prev_cpu, prev_cores, prev_cgroup_cpu = build_sample(
            role=args.role,
            run_id=args.run_id,
            sample_index=index,
            prev_cpu=prev_cpu,
            prev_cores=prev_cores,
            prev_cgroup_cpu=prev_cgroup_cpu,
        )
        if args.job_id:
            sample["job_id"] = args.job_id
        writer.write_sample(sample)
        index += 1
        if args.once:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "sample_index": 0,
                        "out_dirs": [str(p) for p in out_dirs],
                    }
                )
            )
            return 0
        deadline = time.time() + args.interval_seconds
        while not stop and time.time() < deadline:
            time.sleep(min(1.0, max(0.05, deadline - time.time())))

    # Final sample on stop.
    sample, _, _, _ = build_sample(
        role=args.role,
        run_id=args.run_id,
        sample_index=index,
        prev_cpu=prev_cpu,
        prev_cores=prev_cores,
        prev_cgroup_cpu=prev_cgroup_cpu,
    )
    if args.job_id:
        sample["job_id"] = args.job_id
    sample["notes"] = ((sample.get("notes") or "") + ",final").strip(",")
    writer.write_sample(sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
