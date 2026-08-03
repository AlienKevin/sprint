# G1 sprint GPU + CPU telemetry

Every future durable (and open-runner) Harbor trial starts an in-sandbox sampler
that records GPU and host resource usage on a 15–30s interval (default **20s**).

## What starts it (default on)

| Path | How telemetry starts |
|---|---|
| `runs/run-lane-durable.sh` / `run-lane.sh` | Modal `keepalive` starts `/opt/qwop-telemetry.sh`, then `qwop-snapshot-loop` starts it again idempotently |
| `runs/run-luna.sh` / `run-deepseek.sh` | Same (they call the durable launcher) |
| `runs/run-terra.sh` / `run-opus.sh` / `run-kimi-k3.sh` | Modal `keepalive` from `runs/ops/telemetry_keepalive.py` |
| Host monitor (`qwopctl monitor`) | Backup poll via `modal container exec` every monitor tick |

Disable in-sandbox only by removing the binary from the image or overriding
keepalive. Interval: `QWOP_TELEMETRY_INTERVAL` (seconds, min 5).

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

### Durable restic mirror

When `/durable` is mounted (durable launcher):

```
/durable/runs/<run_id>/telemetry/
```

Restic snapshots already include `/logs/artifacts`, so telemetry is in every
periodic/queue/final snapshot.

### Host backup (monitor)

```
runs/ops/<run_id>/telemetry/host-samples.csv
runs/ops/<run_id>/telemetry/host-samples.jsonl
runs/ops/<run_id>/telemetry/host-latest.json
runs/monitor-<run_id>-telemetry.csv
```

Host poller also appends into the trial convention path when the trial dir is
known, and best-effort samples **verifier** sandboxes in the same Modal app
(containers that are not the agent keepalive image).

## Sample schema (CSV columns)

Base (every row):

`ts_utc`, `epoch_s`, `role`, `run_id`, `hostname`, `sample_index`,
`uptime_s`, `nproc`, `load1`, `load5`, `load15`, `cpu_util_pct`,
`cpu_per_core_pct` (JSON array), `mem_total_kib`, `mem_used_kib`,
`mem_available_kib`, `mem_free_kib`, `swap_total_kib`, `swap_used_kib`,
`swap_free_kib`, `disk_root_used_pct|used_kib|total_kib`,
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
python3 challenge/g1-sprint-100m-lane/environment/qwop-telemetry.py \
  --once --role host --out-dir /tmp/qwop-telem-smoke --force \
  --pidfile /tmp/qwop-telem-smoke.pid

# Keepalive JSON shape
python3 runs/ops/telemetry_keepalive.py --run-id smoke | jq .
```
