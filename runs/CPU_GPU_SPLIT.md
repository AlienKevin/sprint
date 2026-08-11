# CPU agent + GPU worker split

Modal preempts **GPU Sandboxes** (and GPU Functions). CPU Sandboxes are not
preemptible. Luna’s bakeoff death (`exit 137`) was GPU-sandbox preemption of the
Codex harness — see `LUNA_SIGKILL_ROOTCAUSE.md`.

## Layout

| Role | Where | GPUs | Preemptible? |
|---|---|---|---|
| Codex / Claude harness | Harbor agent Modal Sandbox | **0** (`task.toml` `[environment]`) | No |
| Isaac training | Host-dispatched Modal Sandbox, same app + `/durable` volume | A10G | Yes |
| Continuous / final verifier | Harbor separate verifier sandboxes | 1× A10G (`[verifier.environment]`) | Yes (unchanged) |

Do **not** pass Harbor `--override-gpus 0`. That overrides the verifier env too.

## Contract

Inside the agent sandbox (prefer `python3`; host rewrites bare `python`):

```bash
sprint-gpu-train --timeout 3600 --max-attempts 3 -- python3 -u train_loop.py
sprint-gpu-train status
sprint-gpu-train wait JOB_ID
sprint-gpu-train logs JOB_ID
```

Job files:

```text
/durable/runs/<run_id>/gpu-jobs/
  queue/<job_id>.json      # enqueue snapshot + host-owned claim mirror
  status/<job_id>.json     # canonical logical-job state
  attempts/<job_id>/<n>.json
  heartbeats/<job_id>/<n>.json
  checkpoints/<job_id>/progress.json
  work/<job_id>/app.tar.gz
  out/<job_id>/attempt-<n>/worker.log
```

### Dispatch (idempotent claim-before-spawn)

Host monitor (`sprintctl monitor`) calls `gpu_worker.dispatch_once` each poll.

1. Host flock `runs/ops/<run_id>/gpu-dispatch.lock`
2. Persist `status=claiming` + `claim_id` + `claimed_at` **before**
   `Sandbox.create` / image build
3. Emit `gpu_worker_starting` enter only after a successful claim
4. Spawn A10G worker; write `dispatched` + `sandbox_id` only if claim still owned
5. The worker writes only attempt-scoped state. The host owns canonical
   `status/`, so a fenced worker cannot overwrite a replacement attempt.

### Automatic recovery

Each attempt has a random `lease_id` and increasing `fence_epoch`. The worker
writes a durable heartbeat every five seconds. The host checks both that lease
and `Sandbox.poll()`:

- a fresh heartbeat or live Sandbox keeps the lease;
- a dead Sandbox plus stale heartbeat starts a second observation window;
- after the dead grace, the host fences the lease, closes its timeline at the
  last heartbeat, and queues the same `job_id`;
- the next attempt starts after bounded exponential backoff;
- `max_attempts` ends a retry loop cleanly.

Defaults are 45 seconds for heartbeat expiry, 20 seconds for dead grace, 10–120
seconds for retry backoff, and three attempts. `sprint-gpu-train` accepts
`--heartbeat-timeout`, `--retry-backoff`, `--retry-backoff-max`, and
`--max-attempts`.

All attempts get the same durable checkpoint directory through
`SPRINT_GPU_CHECKPOINT_DIR` and `SPRINT_GPU_PROGRESS_FILE`. Replacement attempts
also get `SPRINT_GPU_RESUME=1` and `SPRINT_GPU_RESUME_CHECKPOINT`. If a training
program accepts a checkpoint flag, pass `--resume-arg FLAG`; the worker appends
`FLAG <latest-checkpoint>` on attempts after the first.

New jobs should use the checksummed atomic checkpoint protocol rather than
selecting raw model files by modification time. See `runs/PREEMPTIBLE_GPU_API.md`.
It defines lease fencing, interruption grace, replay cursors, idempotency
limits, retry exhaustion, cleanup, and the small provider adapter boundary.

Manual:

```bash
python3 -m event_runtime.control.run gpu-dispatch --run-id "$RUN_ID"
python3 -m event_runtime.control.run gpu-terminate --run-id "$RUN_ID" --job-id "$JOB_ID"
```

## GPU telemetry (preemption-safe)

Every training GPU samples util/mem/power/temp/processes via
`/opt/sprint-telemetry.sh`. Every sealed verifier starts its own trusted
`/tests/verifier_telemetry.py` sampler.

**Live durable writes** (not restic-only):

```text
/durable/runs/<run_id>/telemetry/          # agent (CPU) stream
  by-role/training-gpu/samples.jsonl       # agent-controlled training GPU
  by-job/<job_id>/samples.jsonl            # per job
  gpu-stream/samples.jsonl                 # merged non-agent GPU stream
```

Each training sample is `fsync`’d and best-effort `sync /durable`, tagged with
`role` / `run_id` / `job_id` / `attempt` / `lease_id`. Verifier samples and
their lifecycle boundary are downloaded into each attempt's `verifier/telemetry`
directory. The host monitor also keeps a best-effort diagnostic mirror under
`runs/ops/<run_id>/telemetry/` and `runs/monitor-<run_id>-telemetry.csv`.

On GPU worker death, the CPU harness stays up; the next worker appends to the
same run_id streams. At most one sample interval is lost.

## GPU-active time accounting

Timeline JSONL pairs phases by `(job_id, attempt, phase)`. Dead-worker
reconciliation writes synthetic exits at the last durable heartbeat, so a lost
attempt cannot inflate the replacement attempt:

```text
/durable/runs/<run_id>/telemetry/gpu_timeline.jsonl
/durable/runs/<run_id>/telemetry/gpu_timeline/events/*.json
/durable/runs/<run_id>/telemetry/gpu_time_summary.json
```

| Phase | Meaning | Metrics expected? |
|---|---|---|
| `gpu_queue_wait` | Queued / waiting for next GPU instance | No |
| `gpu_worker_starting` | Sandbox/function allocating | No |
| `isaac_starting` | Isaac / sim stack init | Maybe low util |
| `gpu_active` | Training/inference steps | Yes |
| `gpu_idle_assigned` | Optional: GPU held, not training | Low/zero util |

Summary fields:

- `wall_time_s` — global first→last timeline event (may span many jobs)
- `gpu_wait_s` / `gpu_startup_s` / `isaac_startup_s` — summed phase intervals
- `gpu_active_interval_s` — sum of `gpu_active` intervals
- **`gpu_active_s`** — sum of closed `gpu_active` intervals across attempts
- `wall_minus_wait_s` — sum of per-attempt wall minus wait/startup
- `per_attempt[]` and `per_job[]` — retry-aware breakdowns

Prefer **`gpu_active_s`** for bakeoff / cross-run compare.

```bash
python3 /opt/sprint-gpu-timeline.py emit --run-id "$SPRINT_RUN_ID" \
  --phase gpu_active --action enter --job-id "$JOB"
python3 runs/ops/gpu_timeline_host.py summarize --run-id "$RUN_ID"
```

## Supervise / relaunch (bakeoffs)

`runs/ops/supervise_lane.py` restarts a durable lane after Harbor/agent loss
unless an operator stop file exists. The same `sprint-$RUN_ID` volume is
remounted from a JSON argv vector; no shell command is evaluated. Each CPU
relaunch gets a new host jobs directory while keeping the logical run and
Volume. Bounded exponential backoff, `supervise.lock`, and a 50-restart default
prevent restart storms. All current model launchers run the supervisor inside a
user systemd service with `Restart=on-failure`, so the supervisor itself also
has a watchdog. A clean `STOP_REQUESTED.json` / `STOP_ACK` stays stopped;
`sprintctl stop` fences and terminates owned GPU workers before it signals the
agent. Unrecoverable volume loss and retry exhaustion write
`SUPERVISOR_FAILED.json` and stop rather than restart from scratch.

```bash
# dry-run / once (no paid bakeoff):
python3 runs/ops/supervise_lane.py --run-id "$RUN_ID" --dry-run --once
# real bakeoff only with explicit operator approval:
CONFIRM_LAUNCH=1 runs/run-deepseek.sh
```

Do **not** start a multi-hour Luna/DeepSeek bakeoff from this doc alone.

## Launch

```bash
export MODAL_PROFILE=kevinli020508
set -a; source runs/.secrets/deepseek.env; set +a
CONFIRM_LAUNCH=1 runs/run-deepseek.sh
# smoke:
CONFIRM_LAUNCH=1 runs/run-smoke-cpu-gpu-split.sh
```

Env in agent: `SPRINT_RUN_ID`, `SPRINT_GPU_JOBS_ROOT` (also
`/run/sprint-run-id`).

## Tests

```bash
run-heavy python3 -m unittest runs.ops.tests.test_cpu_gpu_split -v
```

Covers liveness grace, two-pass dead detection, lease fencing, flock
concurrency, retry limits, stop order, attempt-aware timeline sums, launcher
wiring, graceful signal forwarding, abrupt worker loss, atomic checkpoint
publication failure, checksum fallback, repeated preemptions, replay avoidance,
and checkpoint cleanup.

## Proven vs residual

**Proven live:** `smoke-gpu-recovery-20260803T032006Z`, report
`runs/SMOKE_GPU_RECOVERY_20260803T032006Z.md`. Attempt 1 wrote progress, was
terminated with exit 137, and the monitor detected it without a requeue.
Attempt 2 kept the logical job ID, loaded the durable checkpoint, and
succeeded. The CPU Codex harness stayed up. Telemetry and timeline contain both
attempts.

**Unit-tested only:** concurrent-host last-writer fencing and CPU supervisor
relaunch after a real host crash.

**Residual risk:** Modal Volume has no compare-and-swap primitive; the host lock,
lease confirmation, host-owned canonical status, and post-spawn ownership check
bound the race. A first image build can take minutes.
