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

SCHEMA_VERSION = 1
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

GPU_CSV_FIELDS = [
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

SAMPLE_BASE_FIELDS = [
    "ts_utc",
    "epoch_s",
    "role",
    "run_id",
    "hostname",
    "sample_index",
    "uptime_s",
    "nproc",
    "load1",
    "load5",
    "load15",
    "cpu_util_pct",
    "cpu_per_core_pct",
    "mem_total_kib",
    "mem_used_kib",
    "mem_available_kib",
    "mem_free_kib",
    "swap_total_kib",
    "swap_used_kib",
    "swap_free_kib",
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


def cpu_util_from_delta(
    prev: list[int] | None, cur: list[int]
) -> float | None:
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
        if len(cols) < len(GPU_CSV_FIELDS):
            continue
        row = {
            field: parse_num(cols[i]) for i, field in enumerate(GPU_CSV_FIELDS)
        }
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
            if len(cols) < len(GPU_CSV_FIELDS):
                continue
            gpus.append(
                {field: parse_num(cols[i]) for i, field in enumerate(GPU_CSV_FIELDS)}
            )
    else:
        notes.append(f"nvidia-smi_query_rc={q.returncode}")

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
) -> tuple[dict[str, Any], list[int], list[list[int]]]:
    cur_cpu, cur_cores = read_proc_stat()
    load1, load5, load15 = read_loadavg()
    mem = read_meminfo()
    net = read_net_counters()
    root_pct, root_used, root_total = disk_usage("/")
    tmp_pct, tmp_used, tmp_total = disk_usage("/tmp")
    app_pct, app_used, app_total = disk_usage("/app")
    gpus, procs, gpu_extra = sample_gpus()
    notes = list(gpu_extra.get("notes") or [])
    per_core = per_core_util(prev_cores, cur_cores)
    sample: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": utc_now(),
        "epoch_s": int(time.time()),
        "role": role,
        "run_id": run_id or None,
        "hostname": os.uname().nodename,
        "sample_index": sample_index,
        "uptime_s": read_uptime_s(),
        "nproc": os.cpu_count(),
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "cpu_util_pct": cpu_util_from_delta(prev_cpu, cur_cpu),
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
    return sample, cur_cpu, cur_cores


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


def append_csv_rows(path: pathlib.Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
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


class TelemetryWriter:
    def __init__(self, out_dirs: list[pathlib.Path], role: str, run_id: str):
        self.out_dirs = out_dirs
        self.role = role
        self.run_id = run_id
        self.csv_fields = SAMPLE_BASE_FIELDS + GPU_CSV_FIELDS
        for directory in out_dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def write_snapshot(self, snapshot: dict[str, Any]) -> None:
        text = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
        for directory in self.out_dirs:
            atomic_write_text(directory / "snapshot.json", text)

    def write_sample(self, sample: dict[str, Any]) -> None:
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
                        **proc,
                    },
                )
            atomic_write_text(
                directory / "latest.json",
                json.dumps(sample, indent=2, sort_keys=True) + "\n",
            )


def resolve_out_dirs(out_dir: str, run_id: str, durable_dir: str) -> list[pathlib.Path]:
    dirs = [pathlib.Path(out_dir)]
    if run_id:
        durable = pathlib.Path(durable_dir) / "runs" / run_id / "telemetry"
        if pathlib.Path(durable_dir).exists():
            dirs.append(durable)
    # Deduplicate while preserving order.
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
        cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
    except OSError:
        return False
    return "qwop-telemetry" in cmdline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", default="agent", choices=["agent", "verifier", "host"])
    parser.add_argument("--run-id", default=os.environ.get("QWOP_RUN_ID", ""))
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("QWOP_TELEMETRY_DIR", "/logs/artifacts/telemetry"),
    )
    parser.add_argument("--durable-dir", default="/durable")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=int(os.environ.get("QWOP_TELEMETRY_INTERVAL", DEFAULT_INTERVAL)),
    )
    parser.add_argument("--once", action="store_true", help="Take one sample and exit")
    parser.add_argument(
        "--pidfile",
        default="/run/qwop-telemetry.pid",
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

    out_dirs = resolve_out_dirs(args.out_dir, args.run_id, args.durable_dir)
    writer = TelemetryWriter(out_dirs, args.role, args.run_id)
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
            "interval_seconds": args.interval_seconds,
            "out_dirs": [str(path) for path in out_dirs],
            "hostname": os.uname().nodename,
            "nproc": os.cpu_count(),
            "uptime_s": read_uptime_s(),
        }
    )
    writer.write_snapshot(snapshot)

    prev_cpu: list[int] | None = None
    prev_cores: list[list[int]] | None = None
    # Prime CPU counters so the first sample has a util delta.
    prev_cpu, prev_cores = read_proc_stat()
    time.sleep(0.25)

    index = 0
    while not stop:
        sample, prev_cpu, prev_cores = build_sample(
            role=args.role,
            run_id=args.run_id,
            sample_index=index,
            prev_cpu=prev_cpu,
            prev_cores=prev_cores,
        )
        writer.write_sample(sample)
        index += 1
        if args.once:
            print(json.dumps({"ok": True, "sample_index": 0, "out_dirs": [str(p) for p in out_dirs]}))
            return 0
        deadline = time.time() + args.interval_seconds
        while not stop and time.time() < deadline:
            time.sleep(min(1.0, max(0.05, deadline - time.time())))

    # Final sample on stop.
    sample, _, _ = build_sample(
        role=args.role,
        run_id=args.run_id,
        sample_index=index,
        prev_cpu=prev_cpu,
        prev_cores=prev_cores,
    )
    sample["notes"] = ((sample.get("notes") or "") + ",final").strip(",")
    writer.write_sample(sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
