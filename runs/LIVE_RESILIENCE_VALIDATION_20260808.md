# Sprint live resilience validation — 2026-08-08

**Recovery verdict: PASS. Telemetry-completeness verdict: SUPERSEDED.**

Schema-v2 reprocessing showed that the two DeepSeek runs had only one late
host-polled verifier sample each and no authoritative training-GPU samples.
They therefore now report `coverage.ready: false`; their schema-v1
`FINALIZED.json` files are historical and must not be treated as current
completeness evidence. New verifier images sample SCORE from inside every
sealed sandbox, and current finalization requires schema 2 with role-specific
coverage. The recovery and queue-isolation observations below remain valid.

The live tests used Modal CPU sandboxes, A10 GPUs, DeepSeek through the normal
agent launcher, sealed scoring, durable Modal volumes, restic snapshots, and
the production operations controller. Test compute is stopped. Successful
audit volumes are retained.

## Live scenarios

### GPU preemption and checkpoint continuation

- Run: `smoke-recovery-20260808T035927Z`
- Logical GPU job: `504f5050f34e`
- Attempt 1 sandbox: `sb-2RNjaZufl8zFgCvB3AHxoa`
- Forced termination exited `137` after progress step 1 was committed.
- The controller fenced the dead lease and classified the interruption as
  `graceful_preemption` without a manual requeue.
- Attempt 2 sandbox: `sb-MkeDWDd3xC8c00vM3IogpC`
- Attempt 2 loaded the step 1 checkpoint, recorded
  `resumed_from_attempt: 1`, completed step 2, and succeeded.
- The CPU agent sandbox remained alive throughout the GPU interruption.
- Both GPU attempts appear in telemetry and lifecycle accounting.

Detailed evidence is in `SMOKE_GPU_RECOVERY_20260808T035927Z.md` and
`ops/smoke-recovery-20260808T035927Z/gpu-recovery-smoke-result.json`.

### Abrupt CPU-agent loss and automatic Harbor restart

- Run: `cpu-restart3-20260808T0435Z`
- Attempt 1 CPU container: `ta-01KZFTEYNKAYD3F7SW3P73P8GS`
- Attempt 1 was stopped abruptly while its agent and snapshot watcher were
  healthy.
- The systemd supervisor launched attempt 2 automatically in an independent
  jobs root.
- Attempt 2 CPU container: `ta-01KZFTHFF64NNBBCRRXVF4VEAS`
- Durable trace data contains both `cpu-attempt-001` and `cpu-attempt-002`.
- Clean operator stop produced snapshot `e71d182d` and a valid stop ACK.
- The resulting unified timeline includes `cpu_allocated` and
  `cpu_reallocated` events.

This is process/workspace recovery, not register-level VM restoration: a new
Codex process starts against the same durable volume, snapshots, trace, and
task state.

### Two independent DeepSeek runs and scoring queues

- High run: `concurrent-high-20260808T0444Z`
- Max run: `concurrent-max-20260808T0444Z`
- The runs used distinct Modal apps, durable volumes, CPU containers, jobs
  roots, and scoring queue keys.
- Identical canary policies were submitted in the same second.
- The two sealed verifier containers overlapped in wall-clock time and both
  evaluations finished at `2026-08-08T04:48:06Z`; neither queue blocked the
  other.
- Each run recorded exactly one submitted, scored, terminal attempt with no
  malformed ledger entries or controller errors.
- Both canaries intentionally produced DNF; this test validates infrastructure,
  not policy quality.
- Both historical schema-v1 `FINALIZED.json` files reported `complete: true`;
  schema-v2 regeneration correctly invalidates that telemetry verdict.

## Timeline and archive evidence

The original export marked both concurrent runs ready. Schema-v2 regeneration
marks both `coverage.ready: false`, because one final-verifier sample cannot
cover both continuous and final evaluation intervals. The timelines still
contain:

- timestamped native agent events and tool-call/tool-result buckets;
- CPU utilization and memory samples;
- detailed NVIDIA A10 utilization, memory, power, temperature, and clock data;
- evaluation start and finish events;
- the submitted policy artifact name and SHA-256 identity;
- stop request, stop acknowledgment, and run-finalized events.

High contains 30 trace events, 3 tool calls, 32 resource samples, and one GPU
metric sample. Max contains 60 trace events, 9 tool calls, 32 resource samples,
and one GPU metric sample. Both contain one captured submission artifact.

Archive verification passed for both runs: one checksummed submission archive
per run and all 4/4 restic snapshots valid. The GPU-recovery run separately
passed restore into `/data/sprint-recovered/` and restored its Codex JSONL,
SQLite session state, agent log, workspace, and telemetry.

## Issues found and fixed

1. Modal dynamic networking could not enable a domain/CIDR dimension omitted
   when the sandbox was created. Dynamic sandboxes now start with fail-closed
   sentinel filters and can transition safely from no network to the one
   audited model API host and back to no network.
2. The CPU restart watcher reused stale first-seen markers and stop ACK state.
   Markers are now per launch attempt; only an operator-stop ACK suppresses a
   stop signal to a current replacement.
3. Monitor discovery could overwrite newer launch-attempt metadata and later
   misclassify a lone final verifier as the CPU agent. Run metadata updates are
   locked and field-scoped, and a verifier can no longer replace a known agent.
4. The independent keepalive/snapshot watcher did not receive the CPU launch
   attempt. It now receives the attempt in its environment and writes separate
   durable attempt state.
5. A sealed verifier image without `sprint-telemetry.sh` only reported GPU
   presence. The host fallback now collects CPU, memory, and detailed
   `nvidia-smi` GPU metrics itself.
6. Timeline readiness required a training lifecycle file even for score-only
   runs. A paired sealed-evaluation start/finish is now also a valid GPU
   lifecycle source.
7. Historical impossible GPU samples labeled as CPU-agent samples are safely
   reclassified as verifier/GPU-worker events while preserving the reported
   role for audit.

## Automated verification

- Sprint operations suite: **96/96 passed**.
- Focused telemetry suite: **7/7 passed**.
- Focused unified-timeline suite: **7/7 passed**.
- Focused durable-operations and telemetry suites after the discovery fix:
  **20/20 passed**.
- Harbor Modal networking unit suite: **123/123 passed**.
- Disposable live Modal dynamic-network transition: **1/1 passed**.
- Python compile checks: passed.
- Shell syntax checks: passed.
- Full ~20 GB Isaac/agent image build: passed (cold rebuild took about 474 s).

CPU simulations cover graceful preemption, abrupt loss/two-pass detection,
interruption during checkpoint publication, repeated preemptions, fallback to
the latest valid checksummed checkpoint, replay suppression, bounded retry
exhaustion, lease fencing, monotonic resume pointers, and cleanup of partial
checkpoint generations.

## Not exercised

- A provider-initiated Modal eviction was not available on demand. Forced
  termination exercised the same replacement, fencing, checkpoint, and resume
  path.
- A 24-hour soak was not run. The CPU sandbox lifetime is configured and tested
  as `86400` seconds.
- Repeated live GPU kills and a live kill during the checkpoint write window
  were not used because the deterministic CPU suite covers those destructive
  edge cases without extra GPU cost.

## Cleanup

All test apps are stopped and there are zero live test containers. Failed or
accidental test volumes were deleted. The following successful evidence volumes
remain: `sprint-smoke-recovery-20260808T035927Z`,
`sprint-cpu-restart3-20260808T0435Z`,
`sprint-concurrent-high-20260808T0444Z`, and
`sprint-concurrent-max-20260808T0444Z`.
