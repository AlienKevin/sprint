# Sprint

Sprint benchmarks how well autonomous agents turn a fixed model-inference and
compute budget into embodied performance. In the current
[G1 100 metres event](events/g1-100-metres/README.md), independent agents train
Unitree G1 policies in Isaac Lab. Each trial receives the same agent-cost cap;
its result is the fastest valid policy explicitly submitted before that cap is
exhausted.
Results are published at [g1-sprint.vercel.app](https://g1-sprint.vercel.app/).

Agents begin with the [event environment guide](events/g1-100-metres/environment/README.md),
installed as `/app/train/README.md` inside each agent sandbox.
Shared orchestration and metered compute lifecycle live in
[`event_runtime`](event_runtime/README.md).

## Setup

Requires Linux, Docker with Buildx, Python 3.12+, `uv`, and a Modal account.

```bash
git clone git@github.com:AlienKevin/sprint.git
cd sprint
uv sync --project harbor --extra modal
uv run --project harbor modal setup
cp .env.example .env
```

Add both OpenRouter credentials to `.env`: `OPENROUTER_API_KEY` is the parent
inference key used for preflight credit, model, route, and provider probes;
`OPENROUTER_MANAGEMENT_KEY` creates and revokes the model/provider/budget-locked
key used by each trial. `MODAL_PROFILE` is optional when the active Modal CLI
profile already selects the intended account. Native OpenAI, DeepSeek, and
Claude credentials are not used by the current benchmark.

## Run the benchmark

Preflight verifies credentials, immutable images, the full training-to-verifier
path, and one fresh A10G worker per planned trial before any agents launch.

```bash
BATCH_ID="event-$(date -u +%Y%m%d)"
: "${TRIALS_PER_MODEL:?set the desired trials per model}"

uv run --project harbor python -m event_runtime.control.batch preflight \
  --batch-id "$BATCH_ID" --trials-per-model "$TRIALS_PER_MODEL"
uv run --project harbor python -m event_runtime.control.batch launch \
  --batch-id "$BATCH_ID" --trials-per-model "$TRIALS_PER_MODEL" --confirm
uv run --project harbor python -m event_runtime.control.batch status \
  --batch-id "$BATCH_ID"

# Safely stop every trial and preserve its artifacts.
uv run --project harbor python -m event_runtime.control.batch stop \
  --batch-id "$BATCH_ID"
```

Claude Code goal-mode comparisons use the pinned Claude Code 2.1.248 harness
through OpenRouter's Anthropic Messages API. Opus 5 is fixed to medium effort
and Anthropic's first-party route; GLM-5.3-Flash defaults to max effort and is
fixed to Z.AI's first-party FP8 route:

```bash
uv run --project harbor python -m event_runtime.control.batch preflight \
  --batch-id "$BATCH_ID-claude-code" --families opus glm \
  --trials-per-model "$TRIALS_PER_MODEL"
```

Use the same arguments with `batch launch --confirm` after preflight. The batch
default remains max; the Opus arm is sealed to medium independently.
