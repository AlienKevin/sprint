# Sprint

Sprint benchmarks how efficiently autonomous agents turn model inference and
compute into embodied performance. In the current
[G1 100 metres event](events/g1-100-metres/README.md), independent agents train
Unitree G1 policies in Isaac Lab and are compared by Cost-Adjusted Effective
Speed. Results are published at [g1-sprint.vercel.app](https://g1-sprint.vercel.app/).

Agents begin with the [`/app/train` guide](events/g1-100-metres/environment/train/README.md).
Operators can use the [runbook](runs/ops/RUNBOOK.md) for recovery and detailed
infrastructure procedures.

## Setup

Requires Linux, Docker with Buildx, Python 3.12+, `uv`, and a Modal account.

```bash
git clone git@github.com:AlienKevin/sprint.git
cd sprint
uv sync --project harbor --extra modal
uv run --project harbor modal setup
cp .env.example .env  # add provider keys and MODAL_PROFILE
```

## Run the benchmark

Preflight verifies credentials, immutable images, the full training-to-verifier
path, and one fresh A10G worker per planned trial before any agents launch.

```bash
BATCH_ID="sprint-$(date -u +%Y%m%d)"
TRIALS_PER_MODEL=3

uv run --project harbor python runs/ops/batch_eval.py preflight \
  --batch-id "$BATCH_ID" --trials-per-model "$TRIALS_PER_MODEL"
uv run --project harbor python runs/ops/batch_eval.py launch \
  --batch-id "$BATCH_ID" --trials-per-model "$TRIALS_PER_MODEL" --confirm
uv run --project harbor python runs/ops/batch_eval.py status \
  --batch-id "$BATCH_ID"

# Safely stop every trial and preserve its artifacts.
uv run --project harbor python runs/ops/batch_eval.py stop \
  --batch-id "$BATCH_ID"
```
