#!/usr/bin/env python3
"""Durably sample one sealed verifier GPU on the experiment UTC clock."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import signal
import subprocess
import time
from typing import Any

from sprint_gpu_pipeline import collect_pipeline_metrics

STOP = False
SCHEMA_VERSION = 3


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def number(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def cpu_times() -> tuple[int, int]:
    values = [
        int(value)
        for value in pathlib.Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    ]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def keyed_ints(path: pathlib.Path) -> dict[str, int]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    result: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            result[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return result


def scalar(path: pathlib.Path) -> int | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def cgroup_root() -> pathlib.Path:
    mount = pathlib.Path("/sys/fs/cgroup")
    try:
        lines = pathlib.Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return mount
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            candidate = mount / parts[2].lstrip("/")
            if (candidate / "cpu.stat").is_file():
                return candidate
    return mount


def cgroup_v1_sample(mount: pathlib.Path) -> dict[str, Any]:
    cpuacct = mount / "cpuacct"
    cpu = mount / "cpu"
    memory = mount / "memory"
    usage0 = scalar(cpuacct / "cpuacct.usage")
    memory_current = scalar(memory / "memory.usage_in_bytes")
    if usage0 is None or memory_current is None:
        return {}
    started_ns = time.monotonic_ns()
    time.sleep(0.1)
    usage1 = scalar(cpuacct / "cpuacct.usage")
    if usage1 is None:
        return {}
    elapsed_usec = max(1.0, (time.monotonic_ns() - started_ns) / 1000)
    used_cores = max(0, usage1 - usage0) / 1000 / elapsed_usec
    requested_cores = float(os.environ.get("SPRINT_REQUESTED_CPU_CORES", "4"))
    requested_memory_kib = (
        int(os.environ.get("SPRINT_REQUESTED_MEMORY_MIB", "10240")) * 1024
    )
    memory_limit = scalar(memory / "memory.limit_in_bytes")
    if memory_limit is not None and memory_limit >= 1 << 60:
        memory_limit = None
    memory_limit_kib = memory_limit // 1024 if memory_limit is not None else None
    memory_total_kib = requested_memory_kib or memory_limit_kib
    memory_used_kib = memory_current // 1024
    quota = scalar(cpu / "cpu.cfs_quota_us")
    period = scalar(cpu / "cpu.cfs_period_us")
    cpu_limit = (
        quota / period
        if quota is not None and quota > 0 and period is not None and period > 0
        else None
    )
    cpuacct_stat = keyed_ints(cpuacct / "cpuacct.stat")
    cpu_stat = keyed_ints(cpu / "cpu.stat")
    try:
        tick_usec = 1_000_000 / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError):
        tick_usec = None
    peak = scalar(memory / "memory.max_usage_in_bytes")
    failures = scalar(memory / "memory.failcnt")
    return {
        "resource_accounting_scope": "cgroup-v1",
        "cpu_requested_cores": requested_cores,
        "cpu_limit_cores": cpu_limit,
        "cpu_usage_cores": round(used_cores, 4),
        "cpu_usage_usec": usage1 // 1000,
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
        "cpu_util_pct": round(100.0 * used_cores / requested_cores, 2),
        "mem_requested_kib": requested_memory_kib,
        "mem_limit_kib": memory_limit_kib,
        "mem_total_kib": memory_total_kib,
        "mem_used_kib": memory_used_kib,
        "mem_available_kib": max(0, memory_total_kib - memory_used_kib),
        "memory_peak_kib": peak // 1024 if peak is not None else None,
        "memory_oom_events": failures,
        "memory_oom_kill_events": None,
    }


def cgroup_sample(root: pathlib.Path | None = None) -> dict[str, Any]:
    root = root or cgroup_root()
    cpu0 = keyed_ints(root / "cpu.stat")
    memory_current = scalar(root / "memory.current")
    if "usage_usec" not in cpu0 or memory_current is None:
        return cgroup_v1_sample(root)
    started_ns = time.monotonic_ns()
    time.sleep(0.1)
    cpu1 = keyed_ints(root / "cpu.stat")
    elapsed_usec = max(1.0, (time.monotonic_ns() - started_ns) / 1000)
    used_cores = max(0, cpu1.get("usage_usec", 0) - cpu0["usage_usec"]) / elapsed_usec
    requested_cores = float(os.environ.get("SPRINT_REQUESTED_CPU_CORES", "4"))
    requested_memory_kib = (
        int(os.environ.get("SPRINT_REQUESTED_MEMORY_MIB", "10240")) * 1024
    )
    memory_limit = scalar(root / "memory.max")
    memory_limit_kib = memory_limit // 1024 if memory_limit is not None else None
    memory_total_kib = requested_memory_kib or memory_limit_kib
    memory_used_kib = memory_current // 1024
    events = keyed_ints(root / "memory.events")
    peak = scalar(root / "memory.peak")
    return {
        "resource_accounting_scope": "cgroup-v2",
        "cpu_requested_cores": requested_cores,
        "cpu_usage_cores": round(used_cores, 4),
        "cpu_usage_usec": cpu1.get("usage_usec"),
        "cpu_user_usec": cpu1.get("user_usec"),
        "cpu_system_usec": cpu1.get("system_usec"),
        "cpu_nr_throttled": cpu1.get("nr_throttled"),
        "cpu_throttled_usec": cpu1.get("throttled_usec"),
        "cpu_util_pct": round(100.0 * used_cores / requested_cores, 2),
        "mem_requested_kib": requested_memory_kib,
        "mem_limit_kib": memory_limit_kib,
        "mem_total_kib": memory_total_kib,
        "mem_used_kib": memory_used_kib,
        "mem_available_kib": max(0, memory_total_kib - memory_used_kib),
        "memory_peak_kib": peak // 1024 if peak is not None else None,
        "memory_oom_events": events.get("oom"),
        "memory_oom_kill_events": events.get("oom_kill"),
    }


def sample(index: int) -> dict[str, Any]:
    cgroup = cgroup_sample()
    if not cgroup:
        total0, idle0 = cpu_times()
        time.sleep(0.1)
        total1, idle1 = cpu_times()
        delta = max(1, total1 - total0)
        cgroup = {
            "resource_accounting_scope": "host-proc-fallback",
            "cpu_util_pct": round(100.0 * (delta - (idle1 - idle0)) / delta, 2),
        }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": utc_now(),
        "epoch_s": time.time(),
        "role": "verifier-gpu",
        "hostname": os.uname().nodename,
        "sample_index": index,
        **cgroup,
        "gpus": [],
    }
    query = [
        "index",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "power.limit",
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
    ]
    binary = shutil.which("nvidia-smi")
    try:
        result = (
            subprocess.run(
                [
                    binary,
                    "--query-gpu=" + ",".join(query),
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=5,
            )
            if binary
            else None
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    payload["nvidia_smi_ok"] = bool(result and result.returncode == 0)
    if payload["nvidia_smi_ok"] and result is not None:
        keys = [
            "gpu_index",
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
        ]
        for line in result.stdout.splitlines():
            values = [item.strip() for item in line.split(",")]
            if len(values) != len(keys):
                continue
            gpu: dict[str, Any] = {
                "gpu_index": int(number(values[0]) or 0),
                "gpu_name": values[1],
            }
            gpu.update({key: number(value) for key, value in zip(keys[2:], values[2:])})
            payload["gpus"].append(gpu)
    if payload["gpus"]:
        payload["gpus"][0].update(collect_pipeline_metrics())
    payload["gpu_count"] = len(payload["gpus"])
    return payload


def stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.out_dir / "samples.jsonl"
    lifecycle_path = args.out_dir / "lifecycle.json"
    lifecycle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "role": "verifier-gpu",
        "started_at": utc_now(),
        "finished_at": None,
        "sample_count": 0,
        "complete": False,
    }
    atomic_json(lifecycle_path, lifecycle)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        while not STOP:
            payload = sample(lifecycle["sample_count"] + 1)
            with samples_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            lifecycle["sample_count"] += 1
            atomic_json(args.out_dir / "latest.json", payload)
            deadline = time.monotonic() + args.interval_seconds
            while not STOP and time.monotonic() < deadline:
                time.sleep(max(0.0, min(0.25, deadline - time.monotonic())))
    finally:
        lifecycle["finished_at"] = utc_now()
        lifecycle["complete"] = True
        atomic_json(lifecycle_path, lifecycle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
