# `/app/train` — environment and policy contracts

The scored course uses full `G1_CFG`, `enabled_self_collisions=True`,
`contact_offset=0.04`, and `max_depenetration_velocity=10.0`.

Every rollout begins from `no-block-standing-start-v1`, defined exactly in
`/app/train/standing_start.py`: the stable Isaac Lab 2.3.2 G1 standing pose,
upright and aligned behind the line, with zero root and joint velocity. The
100 metres normally uses starting blocks; this event deliberately adds no
blocks or other track bodies, so the existing ground and contact physics are
unchanged. Apply `apply_canonical_standing_start()` when constructing a custom
training scene so local starts match the verifier.

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

`spec.py` is authoritative for tensor shapes, observation fields, action scale,
control frequency, and the optional state-reset ABI. Use its named values rather
than copying offsets or signatures. `event check POLICY.pt` validates the
exported TorchScript contract before submission.

## Target metric

The benchmark has a policy metric and a model-level target metric.

For one simulator rollout, let `d` be the maximum forward distance from the
start reached before the earliest of finishing, leaving the lane, or exceeding
the self-collision limit. Let `t` be the elapsed time when that maximum is first
reached. The rollout's **Effective Speed** is

```text
E = (d / 100 m) × (d / t) = d² / (100 m × t)
```

`E` is zero when `d <= 0` or `t <= 0`. For a valid finish, `d = 100 m` and `t`
is the interpolated finish-crossing time, so `E = 100 m / t`. A submitted
policy's score is the greatest `E` across its three official rollouts.

Each model is evaluated by three independent agent trials. At any wall-clock
time, aggregate cost is the sum of the cumulative costs reported by
`event cost` for those three trials. At aggregate cost `c`, let `Q(c)` be the
greatest policy score recorded from any of the three trials by that point.
`Q(c)` is zero before the first scored policy and holds the best score so far
between readouts.

For a common budget cutoff `B`, the model's **Cost-Adjusted Effective Speed** is

```text
CAES(B) = (1 / B) × integral from 0 to B of Q(c) dc
```

This is the area under the best-so-far Effective Speed versus aggregate-cost
curve, divided by `B`. It has units of m/s; higher is better. `B` may be chosen
after the experiment, but the same `B` must be used for every compared model
and cannot exceed the aggregate cost observed for any of them. The cumulative
cost definition, included components, exclusions, and frozen rates are below.

## Runtime and GPU jobs

The persistent agent sandbox has 2 physical CPU cores and 8 GiB RAM. GPU work
runs on a separate A10G worker:

```bash
event gpu -- python3 -u /app/train/YOUR_SCRIPT.py
event gpu status
event gpu logs
event gpu wait
```

One GPU job is active per run; additional jobs queue. Workers may be
preempted and replaced. Write checkpoints under `$SPRINT_GPU_CHECKPOINT_DIR`
and inspect `$SPRINT_GPU_RESUME` and `$SPRINT_GPU_RESUME_CHECKPOINT` on startup.
For atomic publication, use `event gpu checkpoint save` or the
`CheckpointStore` in `/opt/sprint_resilience.py`. A recovery checkpoint must
contain all mutable state needed to continue the chosen process, including its
completed cursor. Do not use an exported TorchScript candidate as recovery
state unless it is independently sufficient to continue that process. A
replacement attempt with `--resume-arg` fails closed if no valid resumable
checkpoint exists, so it never silently restarts work.
Atomically update `progress.json` with `{"policy_path": "/durable/.../policy.pt"}`
after each complete export. The host mirrors every new reported policy into the
CPU sandbox, including while training continues, and reports its fresh path as
`agent_policy_mirror_path` in `event gpu status`; archive that path even
if the long-lived `/durable` mount has not refreshed yet.

The sandbox has no general internet or cloud credentials. PyTorch, Isaac Lab,
and required assets are preinstalled. The trusted worker redirects stock Isaac
assets to `/opt/assets` after `AppLauncher` starts; do not restore NVIDIA remote
asset URLs. Workspace and `/durable/runs/$SPRINT_RUN_ID` survive CPU sandbox
recreation, but processes and RAM do not. On a relaunched CPU attempt, resume
from durable state.

Run `event gpu --help` for the full job and checkpoint interface.

## Cumulative agent cost

Run `event cost` with no arguments. It prints one machine-readable JSON
document from the trusted host, refreshed on the normal monitor cadence. The
same deterministic ledger drives the website comparison:

```text
C(t) = C_api(t) + C_cpu_agent(t) + C_training(t)
C_role(t) = seconds(t) × (cores × CPU_rate + GiB × memory_rate + A10G × GPU_rate)
C_api(t) = Σ requests Σ token_classes tokens × class_rate / 1,000,000
```

Reasoning tokens are already included in output tokens and are not charged a
second time. For GPT-5.6 requests above 272,000 input tokens, all input rates
are multiplied by 2 and the output rate by 1.5. A10G VRAM is included in the
GPU price rather than billed separately.

The frozen published rates currently exposed in the JSON are:

| component | rate |
|---|---:|
| physical CPU core | `$0.00003942/s` = `$0.141912/h` |
| memory | `$0.00000667/GiB/s` = `$0.024012/GiB/h` |
| A10G (including 24 GiB VRAM) | `$0.000306/s` = `$1.1016/h` |
| CPU agent: 2 cores + 8 GiB | `$0.475920/h` |
| training sandbox: 6 cores + 12 GiB + A10G | `$2.241216/h` |
| GPT-5.6 Luna uncached / cached / cache-write / output | `$0.20 / $0.02 / $0.25 / $1.20` per 1M tokens |
| DeepSeek V4 Flash cache-miss / cache-hit / output | `$0.14 / $0.0028 / $0.28` per 1M tokens |

Official verifier, observability/telemetry, website, volume storage, credits,
discounts, taxes, and invoice adjustments are excluded. The JSON includes its
timestamp, token counts, allocated seconds, component costs, pricing snapshot
provenance, and the exact rates applicable to this run.

## Local verification

The exact nominal verifier source is published read-only at `/app/verifier`.
Queue it on this trial's training A10G with:

```bash
event test /app/policy.pt
event gpu logs JOB_ID
```

This local result is for your own debugging. `event archive` separately retains
immutable policy bytes for blind official scoring; the official verifier does
not return its result or trace during the run.

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

See `event check --rules` for gating. This package does not implement the
geometry pad check.
