# Unified experiment timeline

Future durable lane runs produce one schema-v6 web-safe timeline at:

```
runs/ops/<run_id>/telemetry/unified-timeline.json
sprint-web/data/timelines/<run_id>.json
```

The website viewer is `/timeline.html`. Every record uses UTC `epoch_ms` and
also has `elapsed_ms` from the first run event. The file joins:

- CPU, memory, network, and disk samples from the agent sandbox;
- separate training-GPU and verifier-GPU utilization, memory, power,
  temperature, clocks, PCIe, SM activity/occupancy, tensor/FP32/FP16 pipeline,
  and DRAM-throughput samples;
- GPU allocation, preemption/loss, retry, reallocation, and release events;
- one summary row per timestamped native agent trace record;
- tool-call counts in fixed UTC buckets (60 seconds by default);
- every submission, evaluation start/finish, result, artifact hash, and path.

## Modal cost accounting

Each run uses distinct Modal billing identities for the CPU agent, training
GPU, verifier GPU, and Volume. This is required because Modal's billing export
is grouped by App/Volume description; Sandbox tags are not currently present
in billing rows.

The timeline exposes two explicitly labelled cost views:

- a live requested-resource floor from allocation intervals and a pinned
  public Modal tariff, split by role and CPU, memory, and A10G category;
- the post-run `modal billing report --show-resources` reconciliation, split by
  role, category, and hourly interval, with report and selected-row SHA-256
  provenance.

Each policy has `cost_at_submission` and `cost_at_result` on the same clock.
These use the live tariff estimate because Modal's authoritative report has
hourly resolution. Agent performance-versus-cost excludes `verifier_gpu` and
uses only model API, CPU-agent, and training-GPU spend. Verifier spend is
retained separately as `verifier_measurement_overhead_*`; it cannot influence
the agent; official score and gate feedback remains hidden during the run.

Provider rows include resource spend before credits. Credits, reservations,
subscription charges, taxes, and storage/build costs not owned by a run App
remain invoice-level rather than being guessed into an individual model run.
The final run Volume is scanned recursively so its logical bytes and file count
are retained, together with a nominal full-run storage estimate that is
explicitly excluded from the total because it is not time-weighted and Modal's
included storage allowance is workspace-level. Shared image warm-up is
benchmark overhead and is not charged to a model arm.

## Model API cost accounting

Model API request events preserve the provider-reported token buckets, model,
service tier, context size, and usage timestamp. Their calculated cost uses the
published standard list price selected at that request timestamp. This matters
for mutable aliases such as GPT-5.6 Luna, whose price changed by 5x on
2026-08-10. The timeline labels this value
`published_standard_list_price`; it is reproducible but is not an invoice.
Provider billing exports remain authoritative for contracted rates, credits,
adjustments, and requests billed without a locally completed response.

## Privacy boundary

Native Codex and Claude traces are private forensic artifacts. They may contain
prompts, tool arguments, tool output, and secrets, so they are never copied to
the Vercel tree. The public timeline exposes only timestamp, event type, role,
tool name, call ID, CPU attempt, and a private raw-record reference/hash.

Raw traces are preserved in three ways:

1. Harbor's `/logs/agent` collection;
2. periodic encrypted restic snapshots;
3. near-live immutable complete-line chunks under
   `/durable/runs/<run_id>/trace/raw/cpu-attempt-NNN/`.

The trace mirror publishes a chunk before advancing its cursor. Replaying a
chunk after interruption is safe and the exporter deduplicates native records
by content hash. GPU preemption cannot interrupt this capture because the
agent and trace mirror run in the separate CPU sandbox.

## Completeness gate

New runs set `unified_timeline_required=true`. Finalization refuses to mark a
run complete unless `coverage.ready` proves:

- timestamped agent trace records exist;
- `cpu-agent` samples exist;
- CPU and memory samples use sandbox-local cgroup-v1 or cgroup-v2 counters rather than
  host-wide `/proc` fallback values;
- every `training-gpu` allocation is closed and covered by detailed samples;
- every cache-miss SCORE evaluation has its own completed in-verifier sampler
  interval, covered separately from training; cache hits carry their canonical
  source evaluation ID and checksummed result fingerprint instead;
- no covered GPU interval has a boundary or sampling gap over 45 seconds;
- for runs requiring pipeline telemetry, every training and verifier interval
  is covered independently for SM, occupancy, tensor, FP32, FP16, and DRAM;
- a submission ledger exists;
- every ledger submission resolves to a captured, checksummed artifact;
- Modal's full-hour billing report contains every role that has an allocation.

Malformed or untimestamped raw records are counted in `coverage.warnings`; raw
data is retained even when a row cannot be rendered. This makes omissions
visible instead of silently producing a plausible-looking chart.

The public summary reports window-weighted mean, p50, p95, maximum, and sample
count for every CUPTI field, split between training and verifier GPUs under
`resource_usage_summary.gpu_pipeline`.

## Concurrency semantics

The agent sandbox is CPU-only and independent of training GPU workers and
scoring/verifier workers.

- A preempted training GPU is fenced and retried from checkpoint while the CPU
  agent remains alive. `event gpu wait` is a blocking tool call; agents
  should normally submit and poll status if they have useful CPU-side work.
- `event archive` returns immediately. Scoring does not pause the agent or stop
  an existing training worker; `event history` lists receipts only. Agents debug
  with the published verifier on their own training GPU, while official output
  remains in the trusted archive.

## Blind archival scoring

Every model run has its own immutable queue and trusted acceptance gate. All
accepted work feeds one batch-scoped, crash-safe verifier lease, so at most one
official verifier sandbox is active. Each policy executes in an isolated process.
Exact policy bytes under the same complete task fingerprint may reuse a
checksummed result only after the new request passes the same gate. The host
accepts at most one outstanding policy per trial and no more than one every 300
seconds; request-ID replay is idempotent. All accepted work drains after the
agent exits. Scores, gates, traces, queue progress, and completion timing are
never returned to the agent.

`run.json` records `scoring_queue_scope=shared_blind_archival_queue`, the
batch-scoped provenance key, per-trial concurrency one, the 300-second interval,
and the one-outstanding limit. Timeline events carry queue wait and cache/source
provenance for every accepted policy.
There is no host-frozen primary artifact. The result is the complete submitted
policy trajectory; `best_100m_s` is the best valid policy achieved by the fixed
agent stop, and the website derives the performance–time–cost Pareto
frontier from all scored submissions.

## Build and validate

```bash
python3 -m event_runtime.export.timeline \
  --state-dir runs/ops/<run_id> \
  --web-dir sprint-web \
  --require-ready
```

The host monitor performs this build continuously for new runs, syncs private
trace chunks first, and uploads the internal export to the run's Modal Volume.
`sprintctl finalize` accepts only timeline schema 6, so an older permissive
timeline cannot make a resumed run look complete.
