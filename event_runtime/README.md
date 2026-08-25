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
