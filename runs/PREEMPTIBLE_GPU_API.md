# Preemptible GPU execution API

This is the agent-facing contract used by the G1 sprint experiments. A caller
submits one **logical job** and waits on that job ID. Provider instances are
replaceable attempts; preemption never requires the caller to submit again.

## Recommended use

```bash
job=$(sprint-gpu-train \
  --max-attempts 4 \
  --interruption-grace 20 \
  --resume-arg=--checkpoint \
  -- python3 -u train.py)
sprint-gpu-train wait "$job"
```

Inside `train.py`:

```python
import os
from pathlib import Path

import torch

from sprint_resilience import CheckpointStore, Interruption

checkpoints = CheckpointStore.from_env()
resume = checkpoints.latest_valid()
state = torch.load(resume.path) if resume else make_initial_state()

with Interruption() as interruption:
    while not finished(state):
        train_one_step(state)
        if should_checkpoint(state) or interruption.requested:
            staging = Path("/tmp/checkpoint.pt")
            torch.save(state, staging)
            checkpoints.commit(
                staging,
                sequence=state.step,
                replay_cursor=state.step,
                idempotency_key=f"train-step:{state.step}",
            )
        if interruption.requested:
            break
```

Shell-based jobs can publish the same format:

```bash
sprint-gpu-train checkpoint save /tmp/checkpoint.pt \
  --sequence "$step" --cursor "$step" --idempotency-key "train-step:$step"
sprint-gpu-train checkpoint latest --path-only
sprint-gpu-train checkpoint cleanup --keep 3
```

## Semantics

- **Lease and fencing:** every attempt has an increasing attempt number,
  `fence_epoch`, and random `lease_id`. A fenced attempt cannot publish through
  `CheckpointStore.from_env()`. The host writes the fence before terminating an
  old provider instance.
- **Checkpoint persistence:** a checkpoint is copied to an immutable generation,
  fsynced, checksummed, given a manifest, atomically renamed, and only then made
  latest. Hidden partial generations are never resumable. If the latest payload
  is invalid, recovery scans backward to the newest checksum-valid generation.
- **Interruption:** the worker converts SIGTERM/SIGINT into `checkpointing`,
  forwards the signal to the training process group, and allows the configured
  grace period. A graceful interruption is retried immediately. Abrupt death is
  detected from both provider state and a stale fenced heartbeat, with a second
  observation window to avoid false replacement during control-plane lag.
- **Resumption:** replacement attempts receive the same checkpoint directory and
  logical job ID plus `SPRINT_GPU_RESUME=1`, `SPRINT_GPU_RESUME_CHECKPOINT`,
  `SPRINT_GPU_RESUME_CHECKPOINT_ID`, `SPRINT_GPU_RESUME_SEQUENCE`,
  `SPRINT_GPU_REPLAY_CURSOR`, and `SPRINT_GPU_RESUME_MANIFEST`. `--resume-arg` appends
  a framework-specific checkpoint flag automatically.
- **Idempotency:** checkpointed state should include optimizer/RNG/dataloader
  state and a monotonic replay cursor. `CompletionJournal.run_once(key, fn)`
  prevents replay of results already published through the journal. Arbitrary
  external effects remain at-least-once if the process dies after the effect but
  before its completion marker; pass the same deterministic key downstream.
- **Retry limits:** `max_attempts` includes the initial attempt. Only provider
  loss or an explicit graceful interruption is retried automatically; an
  ordinary nonzero command exit is a terminal application failure. Backoff is
  bounded exponential. Exhaustion returns one terminal `failed` logical job.
- **Cleanup:** provider resources are fenced before termination. Checkpoint
  cleanup always retains at least one valid generation and removes abandoned
  partials and older generations; terminal job metadata and logs remain for
  audit.

## Provider boundary

`sprint_resilience.ExecutionProvider` is deliberately only three operations:

```text
start(job, lease) -> ProviderHandle
probe(handle) -> alive | exited | unknown
terminate(handle) -> optional error
```

Modal Sandbox mechanics implement that interface in
`event_runtime/compute/worker.py`.

## Storage tree

```text
gpu-jobs/checkpoints/<job_id>/
  .sprint-resilience/
    latest.json
    committed/<sequence>-<uuid>/
      manifest.json
      <checkpoint payload>
    completed/<sha256(work-key)>.json
```

Only committed `CheckpointStore` generations are resumable. Loose model files
are intentionally ignored because they provide neither atomic publication nor
checksum validation.
