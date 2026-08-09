# Sprint

Self-contained infrastructure for the G1 100 m agent evaluation behind
[g1-sprint.vercel.app](https://g1-sprint.vercel.app/). This repository contains
the task, our pinned Harbor fork, Modal launch and recovery code, tests, and the
timeline website source. Generated runs, model policies, and secrets are never
committed.

## What a run provides

- one durable CPU agent sandbox (4 physical cores, 16 GiB, no GPU);
- at most one active A10G training worker per model run, resumed from durable
  checksummed checkpoints after preemption;
- blind asynchronous policy submissions through one shared verifier-GPU queue;
- a host-frozen final policy after the agent exits;
- durable raw agent traces, ATIF trajectories, tool latency, tokens, model cost,
  resource telemetry, Modal allocation/cost reconciliation, and schema-v6
  unified timelines;
- no general agent internet access and no cloud control-plane credentials.

Concurrent model runs have independent CPU sandboxes, volumes, training queues,
and ledgers. They contend only for the intentionally shared verifier lease and
Modal GPU availability. Training and verifier GPU costs remain separate.

## One-time setup

Prerequisites: Linux, Docker with Buildx, Python 3.12+, `uv`, and a Modal account.

```bash
git clone git@github.com:AlienKevin/sprint.git
cd sprint
uv sync --project harbor --extra modal
uv run --project harbor modal setup
npx vercel link --project sprint
cp .env.example .env
```

Put credentials in `.env`, then load only the key needed by an arm:

```bash
set -a; source .env; set +a
export MODAL_PROFILE=your-modal-profile
```

Before a paid batch, warm and prove both images with a disposable policy:

```bash
uv run --project harbor python challenge/g1-sprint-100m-lane/tests/adversarial_policy.py \
  --mode slam --out /tmp/sprint-warmup.pt
uv run --project harbor python runs/ops/warm_modal_images.py \
  --policy /tmp/sprint-warmup.pt
```

## Run the six-trial evaluation

The batch controller fixes the comparison contract to three independent
DeepSeek V4 Flash 0731 trials and three independent GPT-5.6 Luna trials, all
through Codex 0.147.0 at max reasoning. The evaluation has no fixed deadline;
the 24-hour value is only the maximum lifetime of one Modal CPU sandbox, which
the host supervisor can replace from durable state. The controller starts a
restartable host monitor, publishes verifier-native success and failure replays
without an extra GPU, and centralizes website deployments.

```bash
BATCH_ID="sprint-$(date -u +%Y%m%d)"
python3 runs/ops/batch_eval.py preflight --batch-id "$BATCH_ID"
python3 runs/ops/batch_eval.py launch --batch-id "$BATCH_ID" --confirm
python3 runs/ops/batch_eval.py status --batch-id "$BATCH_ID"
```

To use a secrets file outside the clone, add `--env-file /absolute/path/.env`.

The systemd batch monitor survives shell disconnects and continuously checks
Modal state, CPU heartbeat, GPU dispatch, verifier/replay publishing, API rate
limits/auth/quota failures, unified timelines, and Vercel deployment. Operate a
single run without killing Harbor directly:

```bash
RUN_ID="${BATCH_ID}-luna-1"
python3 runs/ops/sprintctl.py status --run-id "$RUN_ID"
python3 runs/ops/sprintctl.py stop --run-id "$RUN_ID"
python3 runs/ops/sprintctl.py wait --run-id "$RUN_ID"
```

`stop` is idempotent and preserves traces/artifacts; `wait` succeeds only after
all accepted blind submissions drain, checksums reconcile, the timeline is
ready, and Modal billing data is available. See
[`runs/ops/RUNBOOK.md`](runs/ops/RUNBOOK.md) for recovery and failure handling.

Stop the entire batch safely with:

```bash
python3 runs/ops/batch_eval.py stop --batch-id "$BATCH_ID"
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `harbor/` | Vendored `AlienKevin/harbor` continuous-verification fork |
| `challenge/g1-sprint-100m-lane/` | Agent image, task contract, and sealed verifier |
| `runs/run-{deepseek,luna}.sh` | Safe model-specific launchers used by the fixed batch matrix |
| `runs/ops/` | Batch controller, supervisor, recovery, telemetry, billing, finalization, and tests |
| `sprint-web/` | Static site and schema-v6 timeline renderer |

The exact Harbor source revision is recorded in
`harbor/.sprint-upstream-commit` and verified by every launch.

## Verify locally

The operations suite is CPU-only:

```bash
run-heavy uv run --project harbor pytest -q runs/ops/tests
```

Never commit `.env`, API keys, OAuth tokens, run directories, policies, Modal
manifests, or telemetry. The checked-in ignore rules cover these paths; still
inspect staged changes before every push.
