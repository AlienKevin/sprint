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

## Runtime incident notes

- [Modal Volume v1 telemetry stall](MODAL_VOLUME_TELEMETRY_STALL.md): evidence,
  integrity impact, and proposed remediation for synchronous durable-I/O stalls
  observed in GPU telemetry and heartbeat publication.

## Website trajectory outlines

Rollout tables of contents are authored offline by a Codex agent running as an
unprivileged OS user over an exact, read-only copy of the public trajectory and
published beside the sanitized trajectory as
`web/data/trajectories/<run-id>.outline.json`. Run:

```bash
python -m event_runtime.export.trajectory_outline \
  --trajectory web/data/trajectories/<run-id>.json
```

Each Codex synthesis attempt is retained under
`.artifacts/trajectory-outlines/<run-id>/<timestamp>-<pid>/`. The directory
contains the complete `codex exec --json` event stream, exact prompt, invocation
metadata, final structured response, stderr, and a manifest recording success
or the validation error. These review traces remain local and are not included
in the website bundle.

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

## Local Harbor publication sanitization

Never upload a live job directory directly. Build a separate, Harbor-compatible
staging tree first:

```bash
uv run --project harbor python -m event_runtime.export.harbor_job \
  --job-dir runs/ops/<run-id>/harbor-jobs/<run-id> \
  --output-root .artifacts/harbor-sanitized \
  --secret-env-file /data/harbor-adapters-experiments/.env
```

Install the pinned local secret scanner once:

```bash
uv tool install kingfisher-bin==2.0.0
```

Repeat `--job-dir` to stage a batch. The command also discovers the run's
ephemeral `harbor.env` from `run.json`, if it still exists. It never modifies
the source run and it never uploads anything.

The sanitizer copies only Harbor metadata, raw/sanitized harness traces,
normalized ATIF, text telemetry, and submission-result metadata. Transient
Codex SQLite/WAL state, symlinks, policy binaries, unknown agent files, and
unknown binary artifacts are excluded. DeepSeek ATIF is materialized into the
staged job without writing derived files back into the source run. Claude Code
session logs mirrored during Opus 5 and GLM-5.3-Flash goal-mode trials are also
reconstructed into ATIF before staging.

Every staged job contains `SANITIZATION_REPORT.json` with hashes, exclusions,
redaction counts, and the audit-archive digest. The tree and a generated tarball
are both scanned by the built-in focused detector and Kingfisher 2.0.0. The
Kingfisher invocation disables live validation, update checks, and Git history;
its report never leaves the machine. Before publishing the stage, the command
also exercises the same job/result/lock parsers and per-trial archive builder
used by `harbor upload`. Any surviving credential or PII signature, malformed
Harbor metadata, or archive failure blocks publication and leaves a sibling
`*.SANITIZATION_FAILED.json` report instead of a staged job. Review a clean
report and the staged prose before running `harbor upload`; automated detectors
cannot prove that arbitrary natural-language text contains no semantic PII.
