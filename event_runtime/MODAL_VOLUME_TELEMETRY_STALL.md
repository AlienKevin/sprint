# Modal Volume v1 telemetry stall

## Status

- Observed: 2026-08-28
- Affected run: `claude-goalfix2-20260828-1750-r4-glm-4`
- Affected GPU job: `48c153fac17c`
- Runtime: Modal client 1.5.1, Modal Volume v1
- Scope: shared runtime telemetry and GPU-worker liveness; not specific to
  Claude Code or GLM
- Benchmark impact: the trial completed, but its immutable telemetry coverage
  certificate is incomplete and the trial is infrastructure-invalid

No runtime change is part of this document. It records the evidence, current
diagnosis, and candidate remediation for a future change.

## Summary

One successful A10G training allocation stopped producing trusted GPU samples
for 78 seconds. The configured maximum interval is 45 seconds, so the unified
timeline correctly failed `training_gpu_metrics` and
`training_gpu_pipeline_metrics` coverage. Training continued, the command
exited zero, the declared policy was staged, and Harbor independently verified
the policy. The failure is therefore an observability failure, not evidence of
a training or scoring failure.

The highest-confidence explanation is that synchronous, process-wide
filesystem flushes blocked both the telemetry daemon and GPU-worker heartbeat
while Modal Volume v1 experienced transient persistence latency. This diagnosis
is strongly supported by independent clocks, but it was not captured under
`strace`; treat the exact syscall attribution as high-confidence rather than
directly proven.

## Evidence

The durable per-job stream contains consecutive samples with no missing sample
index:

| Sample | UTC timestamp | Sandbox uptime | Pipeline collector |
| --- | --- | ---: | --- |
| 0 | `2026-08-28T22:52:19Z` | 8.05 s | `ok` |
| 1 | `2026-08-28T22:53:37Z` | 85.88 s | `ok` |

The timestamp gap is 78 seconds and the independent sandbox-uptime delta is
77.83 seconds. Because the sample indices are consecutive and a forced final
sync found no additional records, this is not a host-import or archival gap.

The GPU-worker heartbeat independently paused for approximately 79.93 seconds,
from epoch `1787957536.063755` to `1787957615.9949892`. The heartbeat and
telemetry daemon are separate processes. A simultaneous pause therefore points
to a shared sandbox resource or blocking operation rather than a bug in sample
indexing.

Additional observations:

- All other 18 training-job streams in this run had maximum consecutive sample
  gaps of nine seconds or less.
- CUPTI PM sampling succeeded on both samples surrounding the gap, so CUPTI is
  not the cause of this particular pause. Separate explicit CUPTI availability
  failures elsewhere in the run remain recorded but do not explain the
  simultaneous heartbeat gap.
- Job `48c153fac17c` exited with code 0 and produced the declared policy artifact.
- The policy was independently scored, so the 4.102394 m/s result remains
  informative even though the trial is not benchmark-valid.

## Relevant implementation

The trial launcher explicitly creates a v1 Volume:

```bash
python3 -m modal volume create --version 1 "$VOLUME_NAME"
```

See `control/launch.sh`.

The telemetry daemon writes each sample to its local directory and as many as
three durable projections. Each JSONL and CSV write is individually `fsync`ed,
after which `flush_durable_mount()` calls unbounded `os.sync()` and then invokes
`sync /durable` with a five-second subprocess timeout. See
`container/sprint-telemetry.py`.

The GPU worker follows a similar path for every heartbeat: atomic rename,
per-file `fsync`, then unbounded `os.sync()`. See
`container/sprint-gpu-worker-run.py`.

`os.sync()` flushes all pending filesystem writes visible to the sandbox and
has no timeout. A slow Volume flush can consequently block unrelated producers.
The matching telemetry and heartbeat pauses are consistent with this failure
mode. The timeout around the later `sync /durable` subprocess does not bound the
preceding `os.sync()` call.

Modal documents that Sandbox Volume changes receive background commits every
few seconds and a final commit at termination. Mount-specific `sync` commits
are a Volume v2 feature, not a v1 commit API. See the official
[Modal Volume guide](https://modal.com/docs/guide/volumes) and
[Sandbox filesystem guide](https://modal.com/docs/guide/sandbox-files).

## Suggested remediation

### 1. Remove global flushes from hot paths

Do not call `os.sync()` from telemetry sampling, heartbeat publication, or
other latency-sensitive control loops. Preserve atomic writes and the minimum
file-level durability required by each record, but never make the next sample
or child-supervision iteration wait for a whole-sandbox flush.

This is the smallest high-value change and should be completed regardless of
the Volume version.

### 2. Separate observation from durable persistence

Make the sampler write one canonical local append-only stream on its fixed
schedule. A separate persistence process should mirror complete records to the
Volume in batches. The producer must communicate through a bounded local spool
or queue so a slow durable sink cannot delay sample timestamps.

The persistence process should record its own queue depth, oldest-uncommitted
record age, write latency, and errors. On graceful shutdown it should drain the
spool before the Sandbox terminates. Preemption safety should be defined as a
bounded amount of potentially uncommitted telemetry rather than an unbounded
pause in the trusted sampler.

### 3. Isolate heartbeat publication from child supervision

GPU child supervision must continue polling, forwarding signals, and enforcing
budgets even if durable heartbeat publication is slow. Publish the heartbeat
through an isolated writer and have the host combine heartbeat freshness with
provider-side Sandbox liveness before classifying a worker as dead.

This must not weaken budget fail-closed behavior. Budget enforcement needs an
independent trusted path and must remain authoritative if its own telemetry is
unavailable.

### 4. Store one canonical durable stream

Do not synchronously duplicate every training sample into local, by-role,
by-job, and merged streams. Persist one canonical per-job stream, then derive
the role and merged views during host synchronization or finalization. This
reduces small writes, metadata updates, and opportunities for commit contention
without changing exported data.

### 5. Evaluate Volume v2 separately

Volume v2 is a promising fit because it improves concurrent distinct-file
writes, random access, commit/reload latency, and large-file-count behavior. It
also permits an explicit mount-specific `sync` commit from a Sandbox. It is not,
however, a substitute for decoupling persistence from hot loops: an explicit
commit can still be slow and must run outside sampling and supervision.

Modal currently labels v2 as beta and warns that data loss remains possible.
Adopt it only after a paired canary; do not silently change existing benchmark
storage semantics.

## Validation plan

Before enabling a remediation for benchmark trials:

1. Add a unit test with a durable sink that blocks for 90 seconds. Assert that
   local five-second sampling and child polling continue throughout the stall.
2. Add a Sandbox integration canary that injects the same persistence delay.
   Assert that the host does not falsely terminate a provider-live worker.
3. Verify that the persistence queue drains after recovery with contiguous
   sequence numbers and no duplicated records.
4. Verify that finalization still fails closed when records are genuinely lost,
   rather than merely delayed.
5. Run paired v1 and v2 canaries with identical append workloads. Capture p50,
   p95, and p99 write and commit latency, maximum sample gap, queue depth, attach
   time, finalization time, and any missing records.
6. Run the full shared test suite and confirm that Codex, DeepSeek Harness, and
   Claude Code execution, stopping, scoring, and billing semantics are
   unchanged.

Acceptance criteria are:

- no sampling or heartbeat gap above the configured 45-second coverage limit
  during an injected 90-second persistence stall;
- no false worker termination while provider liveness is positive;
- no loss or duplication after the sink recovers;
- unchanged benchmark validity rules and cost accounting; and
- no model- or harness-specific exception for missing telemetry.

## Current handling

Do not retrospectively mark the affected trial clean. Its performance result
may be used as an explicitly qualified observation, but benchmark-valid
aggregates must exclude it. A replacement trial is required if a fourth clean
GLM measurement is needed.

Finalization now records immutable telemetry failures as
`invalid_infrastructure` instead of waiting forever for evidence that cannot
arrive after Sandbox termination.
