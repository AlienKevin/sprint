# Event runtime

This package contains mechanisms shared by every event: trusted cost
accounting, compute-job lifecycle, telemetry, archival scheduling, and
public-data export. Event rules, embodiments, policy interfaces, metrics, and
verifiers stay under `events/<event>/`.

The runtime must not reinterpret an event result. It transports trusted event
outputs and applies the comparison-cost inclusion policy used by both the agent
and the website.

The source tree is organized by responsibility:

- `agent/`: the trial-local `event` command;
- `container/`: files installed into every agent and training image;
- `compute/`: metered GPU jobs, checkpoints, and preemption handling;
- `control/`: batch launch, stop, supervision, and provider adapters;
- `cost/`: reproducible trial cost accounting;
- `preflight/`: image, fleet, and end-to-end launch gates;
- `models/`: provider-specific catalogs locked to one tool contract;
- `telemetry/` and `export/`: trusted observation and public artifacts.

Generated state is written under ignored `runs/ops/`; it is not source code.
Every launch must use `batch preflight` before `batch launch`. The gate verifies
credentials, exact immutable images, one fresh worker per trial, a real
training-to-policy-to-verifier canary, local/official verifier equivalence, and
agent-visible cost parity. A failed or stale gate blocks launch.

Run the shared tests with:

```bash
uv run --project harbor pytest -q event_runtime/tests tests
```

## Website trajectory outlines

`web/trajectory.js` derives the rollout table of contents from each run's own
public trace at render time. Keep that generator deterministic and
trace-local:

- split chapters from topic changes in the current trace;
- use a short action-and-topic phrase for each title, never a raw sentence or
  question from the model;
- synthesize chapter summaries separately from titles;
- synthesize the rollout synopsis from the ordered chapter activities;
- do not add model-, method-, or run-specific title/summary templates.

The website is an observer. Outline generation and publishing must never
control, restart, stop, or otherwise affect an experiment.
