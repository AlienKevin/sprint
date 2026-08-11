#!/usr/bin/env python3
"""Low-overhead NVIDIA GPU hardware-pipeline telemetry.

The runtime collector invokes a small device-scoped CUPTI PM-sampling helper.
The helper observes all CUDA work on the sandbox GPU, so training and verifier
processes do not need to import this module or opt in.  Metrics are rotated in
single-pass-compatible groups because CUPTI PM sampling cannot replay an
arbitrary workload to satisfy multi-pass counter configurations.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Any

COLLECTOR_SCHEMA_VERSION = 1
DEFAULT_SAMPLER = "/opt/sprint-pm-sampling"
DEFAULT_DURATION_MS = 1000
DEFAULT_SAMPLING_INTERVAL_CYCLES = 75_000_000

# Each tuple is known to fit in one PM-sampling hardware pass on an NVIDIA A10
# (GA102, compute capability 8.6).  Field names distinguish exact counters from
# broad labels: FP32 is the Ampere FMA-heavy pipe and FP16 is instruction issue
# as a percentage of the active-cycle peak.  Raw names remain in provenance.
PIPELINE_METRIC_GROUPS: tuple[tuple[tuple[str, str], ...], ...] = (
    (
        (
            "sm_active_pct",
            "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
        ),
        (
            "fp32_fma_pipe_active_pct",
            "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed",
        ),
        (
            "dram_throughput_pct",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
    ),
    (
        (
            "sm_occupancy_pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ),
        (
            "tensor_pipe_active_pct",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        ),
        (
            "dram_throughput_pct",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
    ),
    (
        (
            "sm_active_pct",
            "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
        ),
        (
            "sm_occupancy_pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ),
        (
            "fp16_instruction_pct_of_peak_active",
            "sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active",
        ),
        (
            "dram_throughput_pct",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
    ),
)

_group_cursor = 0


def metric_provenance() -> dict[str, Any]:
    return {
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "collector": "cupti-pm-sampling",
        "sampler_path": os.environ.get("SPRINT_PM_SAMPLER", DEFAULT_SAMPLER),
        "duration_ms": int(
            os.environ.get("SPRINT_PM_DURATION_MS", str(DEFAULT_DURATION_MS))
        ),
        "sampling_interval_gpu_cycles": DEFAULT_SAMPLING_INTERVAL_CYCLES,
        "rotation_groups": [
            {field: metric for field, metric in group}
            for group in PIPELINE_METRIC_GROUPS
        ],
        "semantics": {
            "sm_active_pct": "SM active cycles / elapsed cycles",
            "sm_occupancy_pct": "active warps / peak active warps",
            "tensor_pipe_active_pct": "tensor-pipe active cycles / elapsed cycles",
            "fp32_fma_pipe_active_pct": (
                "Ampere FMA-heavy pipe active cycles / elapsed cycles; FP32 proxy"
            ),
            "fp16_instruction_pct_of_peak_active": (
                "FP16 FMA instructions / active-cycle peak"
            ),
            "dram_throughput_pct": "DRAM throughput / sustained peak",
        },
        "first_sample_discarded": True,
    }


def collector_capability() -> dict[str, Any]:
    sampler = pathlib.Path(os.environ.get("SPRINT_PM_SAMPLER", DEFAULT_SAMPLER))
    return {
        **metric_provenance(),
        "available": sampler.is_file() and os.access(sampler, os.X_OK),
    }


def _parse_metric_values(stdout: str, raw_metric: str) -> list[float]:
    values: list[float] = []
    prefix = raw_metric + " "
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        try:
            values.append(float(stripped.rsplit(maxsplit=1)[1]))
        except (IndexError, ValueError):
            continue
    # NVIDIA documents the first PM sample as potentially spanning an invalid
    # timestamp origin.  The live A10 probe exhibited exactly that outlier.
    return values[1:] if len(values) > 1 else []


def collect_pipeline_metrics() -> dict[str, Any]:
    """Collect one rotating single-pass counter group, failing open explicitly."""
    global _group_cursor

    group_index = _group_cursor % len(PIPELINE_METRIC_GROUPS)
    _group_cursor += 1
    group = PIPELINE_METRIC_GROUPS[group_index]
    sampler = pathlib.Path(os.environ.get("SPRINT_PM_SAMPLER", DEFAULT_SAMPLER))
    base: dict[str, Any] = {
        "pipeline_metrics_source": "cupti-pm-sampling",
        "pipeline_metrics_group": group_index,
        "pipeline_metrics_status": "unavailable",
        "pipeline_metrics_sample_count": 0,
    }
    if not sampler.is_file() or not os.access(sampler, os.X_OK):
        base["pipeline_metrics_error"] = "collector_missing"
        return base

    duration_ms = max(
        250,
        min(
            5000,
            int(os.environ.get("SPRINT_PM_DURATION_MS", str(DEFAULT_DURATION_MS))),
        ),
    )
    metrics = [raw for _, raw in group]
    env = os.environ.copy()
    env["SPRINT_PM_DURATION_MS"] = str(duration_ms)
    try:
        completed = subprocess.run(
            [
                str(sampler),
                "--samplingInterval",
                str(DEFAULT_SAMPLING_INTERVAL_CYCLES),
                "--maxsamples",
                "64",
                "--hardwareBufferSize",
                str(16 * 1024 * 1024),
                "--metrics",
                ",".join(metrics),
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=(duration_ms / 1000) + 8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        base["pipeline_metrics_error"] = type(exc).__name__
        return base
    if completed.returncode != 0:
        error = re.search(r"CUPTI_ERROR_[A-Z_]+", completed.stderr)
        base["pipeline_metrics_error"] = (
            error.group(0) if error else f"collector_rc_{completed.returncode}"
        )
        return base

    counts: list[int] = []
    for field, raw_metric in group:
        values = _parse_metric_values(completed.stdout, raw_metric)
        if not values:
            continue
        counts.append(len(values))
        base[field] = round(statistics.fmean(values), 4)
    if counts:
        base["pipeline_metrics_status"] = "ok"
        base["pipeline_metrics_sample_count"] = min(counts)
        base["pipeline_metrics_window_ms"] = duration_ms
    else:
        base["pipeline_metrics_error"] = "no_completed_samples"
    return base


def build_sampler(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Build a device-scoped sleep sampler from NVIDIA's installed sample."""
    required = ("Makefile", "pm_sampling.cu", "pm_sampling.h")
    if any(not (source / name).is_file() for name in required):
        raise RuntimeError(f"incomplete CUPTI PM sample source: {source}")
    with tempfile.TemporaryDirectory(prefix="sprint-pm-build-") as raw:
        build = pathlib.Path(raw)
        for name in required:
            shutil.copy2(source / name, build / name)
        code_path = build / "pm_sampling.cu"
        code = code_path.read_text(encoding="utf-8")
        replacements = (
            (
                "    VectorLaunchWorkLoad vectorWorkLoad;\n"
                "    vectorWorkLoad.SetUp();\n",
                "",
            ),
            (
                "    vectorWorkLoad.TearDown();\n",
                "",
            ),
        )
        for old, new in replacements:
            if old not in code:
                raise RuntimeError(f"CUPTI sample source changed; missing: {old!r}")
            code = code.replace(old, new, 1)
        start_marker = "    const size_t NUM_OF_ITERATIONS = 100000;"
        end_marker = "    // 5. Stop the PM sampling"
        start = code.find(start_marker)
        end = code.find(end_marker, start)
        if start < 0 or end < 0:
            raise RuntimeError("CUPTI workload markers changed")
        sleep_code = (
            '    const char* durationText = std::getenv("SPRINT_PM_DURATION_MS");\n'
            "    uint64_t durationMs = durationText ? std::stoull(durationText) : 1000;\n"
            "    std::this_thread::sleep_for(std::chrono::milliseconds(durationMs));\n\n"
        )
        code = code[:start] + sleep_code + code[end:]
        # The reference decoder intentionally spins.  A short pause preserves
        # the hardware buffer while keeping profiler CPU overhead negligible.
        decode_marker = "    while (!stopDecodeThread)\n    {"
        if decode_marker not in code:
            raise RuntimeError("CUPTI decode loop changed")
        code = code.replace(
            decode_marker,
            decode_marker
            + "\n        std::this_thread::sleep_for(std::chrono::milliseconds(50));",
            1,
        )
        code_path.write_text(code, encoding="utf-8")
        header_path = build / "pm_sampling.h"
        header = header_path.read_text(encoding="utf-8")
        sample_loop = "for(size_t sampleIndex = 0; sampleIndex < 50; ++sampleIndex)"
        if sample_loop not in header:
            raise RuntimeError("CUPTI sample print loop changed")
        header_path.write_text(
            header.replace(
                sample_loop,
                "for(size_t sampleIndex = 0; "
                "sampleIndex < m_samplerRanges.size(); ++sampleIndex)",
                1,
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["CUDA_INSTALL_PATH"] = "/usr/local/cuda"
        env["SMS"] = "86"
        subprocess.run(
            ["make"],
            cwd=build,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build / "pm_sampling", destination)
        destination.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.build:
        if args.output is None:
            parser.error("--output is required with --build")
        build_sampler(args.build, args.output)
        return 0
    parser.error("use --build SOURCE --output PATH")


if __name__ == "__main__":
    raise SystemExit(main())
