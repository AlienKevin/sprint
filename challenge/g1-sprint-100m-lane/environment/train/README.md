# `/app/train` — environment and policy contracts

The scored course uses full `G1_CFG`, `enabled_self_collisions=True`,
`contact_offset=0.04`, and `max_depenetration_velocity=10.0`.

This package contains the embodiment configuration and executable policy
interface. It does not provide an objective, controller, training loop,
algorithm, or reference policy. Any method that produces a compliant
TorchScript policy may be used. The official course and geometric gate
implementation remain verifier-only.

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

`spec.py` is authoritative for tensor shapes, observation fields, action scale,
control frequency, and the optional state-reset ABI. Use its named values rather
than copying offsets or signatures. `sprint-check POLICY.pt` validates the
exported TorchScript contract before submission.

## Runtime and GPU jobs

The persistent agent sandbox has 4 physical CPU cores and 16 GiB RAM. GPU work
runs on a separate A10G worker:

```bash
sprint-gpu-train -- python3 -u /app/train/YOUR_SCRIPT.py
sprint-gpu-train status
sprint-gpu-train logs
sprint-gpu-train wait
```

One GPU job is active per run; additional jobs queue. Workers may be
preempted and replaced. Write checkpoints under `$SPRINT_GPU_CHECKPOINT_DIR`
and inspect `$SPRINT_GPU_RESUME` and `$SPRINT_GPU_RESUME_CHECKPOINT` on startup.
For atomic publication, use `sprint-gpu-train checkpoint save` or the
`CheckpointStore` in `/opt/sprint_resilience.py`. A recovery checkpoint must
contain all mutable state needed to continue the chosen process, including its
completed cursor. Do not use an exported TorchScript candidate as recovery
state unless it is independently sufficient to continue that process. A
replacement attempt with `--resume-arg` fails closed if no valid resumable
checkpoint exists, so it never silently restarts work.
Atomically update `progress.json` with `{"policy_path": "/durable/.../policy.pt"}`
after each complete export. The host mirrors every new reported policy into the
CPU sandbox, including while training continues, and reports its fresh path as
`agent_policy_mirror_path` in `sprint-gpu-train status`; submit that path even
if the long-lived `/durable` mount has not refreshed yet.

The sandbox has no general internet or cloud credentials. PyTorch, Isaac Lab,
and required assets are preinstalled. The trusted worker redirects stock Isaac
assets to `/opt/assets` after `AppLauncher` starts; do not restore NVIDIA remote
asset URLs. Workspace and `/durable/runs/$SPRINT_RUN_ID` survive CPU sandbox
recreation, but processes and RAM do not. On a relaunched CPU attempt, resume
from durable state.

Run `sprint-gpu-train --help` for the full job and checkpoint interface.

## Import

`/app` is on `PYTHONPATH` when you work from `/app`:

```python
import sys
sys.path.insert(0, "/app")
from train.robot import make_scored_course_g1_cfg
```

## Embodiment configuration

```python
cfg.scene.robot = make_scored_course_g1_cfg()
```

Apply this after any configuration lifecycle that rebuilds `scene.robot`.

## Constants

| setting | value |
|---|---|
| asset | `G1_CFG` (full meshes, not `G1_MINIMAL_CFG`) |
| `enabled_self_collisions` | `True` |
| `contact_offset` / `rest_offset` | `0.04` / `0.0` |
| `max_depenetration_velocity` | `10.0` |

See `sprint-check --rules` for gating. This package does not implement the
geometry pad check.
