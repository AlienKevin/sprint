# G1 sprint GPU + CPU telemetry

For the joined agent/GPU/tool/submission view and its launch-readiness gate,
see [`UNIFIED_TIMELINE.md`](UNIFIED_TIMELINE.md). The synchronized website
viewer is `/timeline.html`.

Every future durable (and open-runner) Harbor trial starts an in-sandbox sampler
that records GPU and sandbox-local resource usage on a 15–30s interval (default
**20s**).

GPU workers additionally run NVIDIA CUPTI PM sampling against the sandbox GPU.
This is device-scoped, so it observes the separate agent training or sealed
verifier process without changing that process. The collector rotates three
single-hardware-pass counter groups: replaying an arbitrary training workload
to satisfy a multi-pass profile would distort the benchmark. Training and
sealed-verifier samplers run every 5s, so each rotated counter is normally
refreshed at least every 15s.

The primary hardware-pipeline fields are:

- `sm_active_pct` — SM active cycles divided by elapsed cycles;
- `sm_occupancy_pct` — active warps divided by peak active warps;
- `tensor_pipe_active_pct` — tensor-pipe active cycles divided by elapsed cycles;
- `fp32_fma_pipe_active_pct` — Ampere FMA-heavy pipe activity (the recorded
  FP32 proxy, named as such rather than presented as total FLOP utilization);
- `fp16_instruction_pct_of_peak_active` — issued FP16 FMA instructions relative
  to the active-cycle peak;
- `dram_throughput_pct` — DRAM throughput relative to sustained peak.

Each point retains collector source, counter group, capture window, completed
sample count, and an explicit status/error. The documented CUPTI first-sample
timestamp outlier is discarded. Unsupported or denied counters fail open for
the workload but fail the future-run timeline readiness gate; ordinary NVML
utilization, power, VRAM, and clocks remain as fallback evidence.

CPU and memory fields come from the sandbox's cgroup controller: Modal's
cgroup-v1 mounts (`cpuacct.usage`, `cpu.cfs_*`, and `memory.*_in_bytes`) or
cgroup v2 (`cpu.stat`, `cpu.max`, `memory.current`, `memory.max`,
`memory.peak`, and `memory.events`), not host-wide `/proc` totals.
`cpu_usage_cores` is absolute consumed cores over the
sample interval; `cpu_util_pct` is relative to the requested cores and may
exceed 100% if Modal permits a burst. Memory usage, peak, headroom, and OOM
events belong to the sandbox cgroup. `resource_accounting_scope` identifies
the source as `cgroup-v1` or `cgroup-v2`. A clearly labelled
`host-proc-fallback` is retained for diagnostics, but current-run timeline
readiness rejects it.

`mem_requested_kib` is the configured Modal reservation/billing baseline;
`mem_limit_kib` is the cgroup's actual enforcement ceiling. Modal can expose a
node-sized ceiling and permit use beyond the request, so `mem_total_kib` and
`mem_available_kib` intentionally use the requested amount for utilization and
cost comparisons while preserving the distinct ceiling.

## What starts it (default on)

| Path | How telemetry starts |
|---|---|
| `runs/run-lane-durable.sh` | Modal `keepalive` starts `/opt/sprint-telemetry.sh`, then `sprint-snapshot-loop` starts it again idempotently |
| `runs/run-luna.sh` / `run-deepseek.sh` | Same (they call the durable launcher) |
| `runs/run-luna.sh` / `runs/run-deepseek.sh` | Use the durable launcher path above |
| Host monitor (`sprintctl monitor`) | Backup poll via `modal container exec` every monitor tick |

Disable in-sandbox only by removing the binary from the image or overriding
keepalive. Interval: `SPRINT_TELEMETRY_INTERVAL` (seconds, min 5).

## Artifact paths

### In-sandbox (primary; Hub-collectable)

```
/logs/artifacts/telemetry/snapshot.json     # driver/CUDA + GPU static at start
/logs/artifacts/telemetry/samples.jsonl     # full samples (one JSON object/line)
/logs/artifacts/telemetry/samples.csv       # flat rows (one row per GPU)
/logs/artifacts/telemetry/processes.jsonl   # GPU compute apps (pid/name/mem)
/logs/artifacts/telemetry/latest.json       # most recent sample
/logs/artifacts/telemetry/telemetry-daemon.log
```

Harbor always collects the convention dir `/logs/artifacts`. On the host that
mirrors to:

```
<jobs>/<job>/<trial>/artifacts/logs/artifacts/telemetry/
```

Hub trial archives include the whole `artifacts/` tree
(`harbor/upload/uploader.py` `_TRIAL_ARCHIVE_INCLUDES`), so these files upload
with the job when Hub auth works.

### Durable live mirror (preemption-safe)

When `/durable` is mounted (durable launcher), each sample is written **live**
(fsync + best-effort `sync /durable`), not only via restic (~5 min):

```
/durable/runs/<run_id>/telemetry/                 # cpu-agent stream
/durable/runs/<run_id>/telemetry/by-role/training-gpu/
/durable/runs/<run_id>/telemetry/by-role/verifier-gpu/
/durable/runs/<run_id>/telemetry/by-job/<job_id>/
/durable/runs/<run_id>/telemetry/gpu-stream/      # merged non-agent GPU
/durable/runs/<run_id>/telemetry/gpu_timeline.jsonl
/durable/runs/<run_id>/telemetry/gpu_time_summary.json
```

Restic still snapshots `/logs/artifacts` and `/durable` periodically; live
writes mean a GPU preempt loses at most one sample interval.

Training GPUs start `/opt/sprint-telemetry.sh --role training-gpu` (interval
often 5s). Each sealed SCORE container starts `/tests/verifier_telemetry.py`
inside the trusted verifier and samples every 10s. Harbor archives its stream
and completed sampler boundary under:

```
<trial>/artifacts/continuous/attempts/<attempt>/verifier/telemetry/ # interim
<trial>/verifier/telemetry/                                        # final
```

These samples are tagged `verifier-gpu`, joined to exactly that evaluation by
the attempt path, and never merged into the agent's training budget.

At finalization the host re-imports every append-only training `by-job` stream
and durable worker-attempt record, then verifies them against the merged GPU
stream and host job registry. This prevents a concurrent Volume merge race or
an abrupt worker exit from silently dropping samples or lifecycle boundaries.

### Host backup (monitor)

```
runs/ops/<run_id>/telemetry/host-samples.csv
runs/ops/<run_id>/telemetry/host-samples.jsonl
runs/ops/<run_id>/telemetry/host-latest.json
runs/ops/<run_id>/telemetry/gpu_timeline.jsonl
runs/ops/<run_id>/telemetry/gpu_time_summary.json
runs/monitor-<run_id>-telemetry.csv
```

Host poller samples **cpu-agent**, **training-gpu**, and any discoverable
**verifier-gpu** containers each monitor tick. It is diagnostic backup only;
the archived in-verifier sampler is the authoritative SCORE usage source.

Timeline readiness is gated independently for the two GPU roles. Every closed
training allocation and every sealed verifier sampler interval must have
detailed GPU samples with no uncovered gap longer than 45 seconds, and every
started evaluation must have a completed sampler interval. A late host sample
cannot satisfy an archived verifier interval, and a training sample cannot
satisfy SCORE coverage.

### GPU-active time (fairness)

See `runs/CPU_GPU_SPLIT.md`. Timeline phases: `gpu_queue_wait`,
`gpu_worker_starting`, `isaac_starting`, `gpu_active`, `gpu_idle_assigned`.
Events pair by logical job, attempt, and phase. Duplicate phase enters within
one attempt are ignored. Dead-worker detection closes open phases at the last
durable heartbeat.

Summary:

- `wall_time_s` — global first→last event
- `gpu_wait_s` / `gpu_startup_s` / `isaac_startup_s` — summed intervals
- `gpu_active_interval_s` — sum of `gpu_active` intervals
- **`gpu_active_s`** — sum of closed `gpu_active` intervals across attempts
- `wall_minus_wait_s` — summed per-attempt wall minus wait/startup
- `per_attempt[]` / `per_job[]` — retry-aware breakdowns

Prefer `gpu_active_s` for cross-run compute comparisons. Samples carry
`role`, `run_id`, `job_id`, `attempt`, and `lease_id` so streams from retries
do not mix silently.

## Sample schema (CSV columns)

Base (every row):

`ts_utc`, `epoch_s`, `role`, `run_id`, `job_id`, `attempt`, `lease_id`,
`hostname`, `sample_index`,
`uptime_s`, `nproc`, `load1`, `load5`, `load15`,
`resource_accounting_scope`, `cpu_requested_cores`, `cpu_limit_cores`,
`cpu_usage_cores`, `cpu_usage_usec`, `cpu_user_usec`, `cpu_system_usec`,
`cpu_nr_throttled`, `cpu_throttled_usec`, `cpu_util_pct`,
`cpu_per_core_pct` (JSON array), `mem_total_kib`, `mem_used_kib`,
`mem_available_kib`, `mem_free_kib`, `swap_total_kib`, `swap_used_kib`,
`swap_free_kib`, `memory_peak_kib`, `memory_oom_events`,
`memory_oom_kill_events`, `disk_root_used_pct|used_kib|total_kib`,
`disk_tmp_*`, `disk_app_*`, `net_rx_bytes`, `net_tx_bytes`,
`net_rx_packets`, `net_tx_packets`, `gpu_count`, `pcie_rx_mib_s`,
`pcie_tx_mib_s`, `nvidia_smi_ok`, `notes`

Per-GPU fields (repeated when multiple GPUs):

`gpu_index`, `gpu_name`, `gpu_uuid`, `util_gpu_pct`, `util_mem_pct`,
`mem_used_mib`, `mem_total_mib`, `mem_free_mib`, `power_draw_w`,
`power_limit_w`, `temp_gpu_c`, `clock_graphics_mhz`, `clock_sm_mhz`,
`clock_mem_mhz`, `clock_max_*`, `pcie_link_gen`, `pcie_link_width`,
`ecc_mode`, `ecc_corrected_volatile`, `ecc_uncorrected_volatile`,
`driver_version`

CUPTI pipeline fields (rotated single-pass groups):

`pipeline_metrics_source`, `pipeline_metrics_group`,
`pipeline_metrics_status`, `pipeline_metrics_sample_count`,
`pipeline_metrics_window_ms`, `pipeline_metrics_error`, `sm_active_pct`,
`sm_occupancy_pct`, `tensor_pipe_active_pct`,
`fp32_fma_pipe_active_pct`, `fp16_instruction_pct_of_peak_active`,
`dram_throughput_pct`

`samples.jsonl` keeps nested `gpus[]` and `gpu_processes[]`. Process records
store `pid`, redacted `process_name`, and `used_memory_mib` only — never
command lines or environment variables.

## Secrets

Telemetry never dumps `environ`, API keys, OAuth tokens, or process argv.
Process names matching `api_key|token|password|secret|...` are replaced with
`[redacted]`. Host poller also redacts secret-shaped substrings in exec noise.

## Smoke (no Modal job)

```bash
# One-shot on the host (CPU-only is fine; notes=nvidia_smi_missing)
python3 challenge/g1-100-metres/environment/sprint-telemetry.py \
  --once --role host-controller --out-dir /tmp/sprint-telem-smoke --force \
  --pidfile /tmp/sprint-telem-smoke.pid

# Keepalive JSON shape
python3 runs/ops/telemetry_keepalive.py --run-id smoke | jq .
```
