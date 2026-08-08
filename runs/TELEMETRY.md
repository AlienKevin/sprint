# G1 sprint GPU + CPU telemetry

For the joined agent/GPU/tool/submission view and its launch-readiness gate,
see [`UNIFIED_TIMELINE.md`](UNIFIED_TIMELINE.md). The synchronized website
viewer is `/timeline.html`.

Every future durable (and open-runner) Harbor trial starts an in-sandbox sampler
that records GPU and sandbox-local resource usage on a 15–30s interval (default
**20s**).

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
| `runs/run-lane-durable.sh` / `run-lane.sh` | Modal `keepalive` starts `/opt/sprint-telemetry.sh`, then `sprint-snapshot-loop` starts it again idempotently |
| `runs/run-luna.sh` / `run-deepseek.sh` | Same (they call the durable launcher) |
| `runs/run-terra.sh` / `run-opus.sh` | Modal `keepalive` from `runs/ops/telemetry_keepalive.py` |
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

`samples.jsonl` keeps nested `gpus[]` and `gpu_processes[]`. Process records
store `pid`, redacted `process_name`, and `used_memory_mib` only — never
command lines or environment variables.

## Secrets

Telemetry never dumps `environ`, API keys, OAuth tokens, or process argv.
Process names matching `api_key|token|password|secret|...` are replaced with
`[redacted]`. Host poller also redacts secret-shaped substrings in exec noise.

## Hub upload caveat

As of 2026-08-02, many Hub uploads are blocked by `HARBOR_API_KEY` **HTTP 401**
(see `runs/HUB_JOBS.md`). Paths above are still correct for local trial dirs and
will be in the archive once a valid `sk-harbor-...` key is configured. Prefer
`/data/harbor-adapters-experiments/.env` for Hub keys.

Telemetry JSON/CSV must not contain API keys (sampler redacts process names).
Before any Hub upload, run the scrub + leak-gate path documented in
`runs/SECRETS_AND_HUB.md` (`python3 runs/hub_track_upload.py --once`).

## Smoke (no Modal job)

```bash
# One-shot on the host (CPU-only is fine; notes=nvidia_smi_missing)
python3 challenge/g1-sprint-100m-lane/environment/sprint-telemetry.py \
  --once --role host-controller --out-dir /tmp/sprint-telem-smoke --force \
  --pidfile /tmp/sprint-telem-smoke.pid

# Keepalive JSON shape
python3 runs/ops/telemetry_keepalive.py --run-id smoke | jq .
```
