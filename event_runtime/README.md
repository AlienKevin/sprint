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

Rollout tables of contents are authored offline by a read-only Codex agent and
published beside the sanitized trajectory as
`web/data/trajectories/<run-id>.outline.json`. Run:

```bash
python -m event_runtime.export.trajectory_outline \
  --trajectory web/data/trajectories/<run-id>.json
```

Each Codex synthesis attempt is retained under
`.artifacts/trajectory-outlines/<run-id>/<timestamp>-<pid>/`. The directory
contains the complete `codex exec --json` event stream, the final structured
response, stderr, and a manifest recording success or the validation error.
These review traces remain local and are not included in the website bundle.

The command pins the authoring model to GPT-5.6 Sol and lets it browse the exact
already-public trajectory with read-only file-inspection tools. There is no
projection or preprocessing step. The response is schema-constrained JSON,
every chapter is validated against immutable step IDs, and the result is cached
by trace fingerprint and SHA-256.
Generate outlines after a trial is complete; never invoke the model
from the recurring timeline refresh path. A missing or stale outline simply
hides the table of contents—the browser must not fabricate replacement prose
with keyword or regex templates.

The website is an observer. Outline generation and publishing must never
control, restart, stop, or otherwise affect an experiment.
