# `/app/train` — scored-course G1 PhysX spawn

Stock Isaac Lab G1 locomotion leaves **self-collisions off** and often uses
`G1_MINIMAL_CFG`. The scored course uses **full `G1_CFG`**,
`enabled_self_collisions=True`, `contact_offset=0.04`, and
`max_depenetration_velocity=10.0`.

This package only ships that robot spawn. Rewards, terrain, and the geometric
self-collision DQ are still yours (the DQ is verifier-only).

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
