# `/app/train` — scored-course G1 PhysX spawn

Stock Isaac Lab G1 locomotion leaves **self-collisions off** and often uses
`G1_MINIMAL_CFG`. The scored course uses **full `G1_CFG`**,
`enabled_self_collisions=True`, `contact_offset=0.04`, and
`max_depenetration_velocity=10.0`.

This package only ships that robot spawn. Rewards, terrain, and the geometric
self-collision DQ are still yours (the DQ is verifier-only).

## Policy interface

The executable interface contract is in `/app/train/spec.py`:

```python
from train.spec import (
    ACTION_DIM,
    ACTION_SCALE,
    CONTROL_FREQUENCY_HZ,
    OBSERVATION_DIM,
    OBSERVATION_FIELDS,
    OBSERVATION_SLICES,
)
```

Your TorchScript policy maps `(N, OBSERVATION_DIM)` to `(N, ACTION_DIM)` joint
position targets. Use `OBSERVATION_SLICES` instead of copying numeric offsets.
An optional `reset()` method clears policy state between verifier trials.

## Runtime and GPU jobs

The persistent agent sandbox has 4 physical CPU cores and 16 GiB RAM. Isaac
training runs on a separate A10G worker:

```bash
sprint-gpu-train -- python3 -u /app/train/YOUR_SCRIPT.py
sprint-gpu-train status
sprint-gpu-train logs
sprint-gpu-train wait
```

One training job is active per run; additional jobs queue. Workers may be
preempted and replaced. Write checkpoints under `$SPRINT_GPU_CHECKPOINT_DIR`
and inspect `$SPRINT_GPU_RESUME` and `$SPRINT_GPU_RESUME_CHECKPOINT` on startup.
For atomic publication, use `sprint-gpu-train checkpoint save` or the
`CheckpointStore` in `/opt/sprint_resilience.py`. A recovery checkpoint must be
full trainer state: model, optimizer, scheduler or scaler state when used, and
the completed iteration/cursor. Do not commit an exported TorchScript policy as
a recovery checkpoint; policies are inference/submission artifacts and cannot
resume optimization. A replacement attempt with `--resume-arg` fails closed if
no valid trainer-state checkpoint exists, so it never silently restarts work.
When a job finishes, the host mirrors the policy named by `progress.json` into
the CPU sandbox and reports its fresh path as `agent_policy_mirror_path` in
`sprint-gpu-train status`; submit that path even if the long-lived `/durable`
mount has not refreshed yet.

The sandbox has no general internet or cloud credentials. PyTorch, Isaac Lab,
and required assets are preinstalled. The trusted worker redirects stock Isaac
assets to `/opt/assets` after `AppLauncher` starts; do not restore NVIDIA remote
asset URLs. Workspace and `/durable/runs/$SPRINT_RUN_ID` survive CPU sandbox
recreation, but processes and RAM do not. On a relaunched CPU attempt, resume
from durable state.

Run `sprint-gpu-train --help` for the full job and checkpoint interface.

## Import

`/app` is on `PYTHONPATH` when you work from `/app`. Prefer:

```python
import sys
sys.path.insert(0, "/app")
from train.robot import make_scored_course_g1_cfg, apply_scored_course_g1
```

## Attach to a Velocity-derived env

```python
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)
from train.robot import apply_scored_course_g1

cfg = G1FlatEnvCfg()
apply_scored_course_g1(cfg)   # replaces scene.robot after cfg __post_init__
# Or: cfg.scene.robot = make_scored_course_g1_cfg()
```

Call **after** the env cfg finishes `__post_init__` so a parent that rebuilds
`scene.robot` does not wipe the settings.

## Constants

| setting | value |
|---|---|
| asset | `G1_CFG` (full meshes, not `G1_MINIMAL_CFG`) |
| `enabled_self_collisions` | `True` |
| `contact_offset` / `rest_offset` | `0.04` / `0.0` |
| `max_depenetration_velocity` | `10.0` |

See `sprint-check --rules` for gating; this starter does not implement the
geometry pad check.
