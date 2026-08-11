# G1 100 metres

Build and export a TorchScript policy for the frozen Unitree G1 course. This
directory is the agent-facing source of truth for the start state, policy ABI,
score, cost, and available commands.

## Course and standing start

Every rollout begins from `no-block-standing-start-v1`, defined exactly in
`standing_start.py`. It follows the standing-start convention—stationary,
upright, aligned with the lane, and wholly behind the line—but freezes one exact
G1 root pose, joint pose, heading, and zero-velocity state because athletics
rules do not define robot joint angles. There are no starting blocks, pedals,
or extra track bodies, so ground and contact physics are unchanged. Apply
`apply_canonical_standing_start()` to custom training scenes.

The scored robot uses full `G1_CFG`, self-collisions enabled, a `0.04 m` contact
offset, zero rest offset, and `10 m/s` maximum depenetration velocity. Create it
with `make_scored_course_g1_cfg()` from `robot.py`.

This package only ships that robot spawn. Rewards, terrain, and the geometric
self-collision DQ are still yours (the DQ is verifier-only).

## Policy interface

`spec.py` is authoritative for observation/action shapes, named observation
fields, action scale, 50 Hz control, and the optional state-reset ABI. Import
its named constants instead of copying offsets. Export with `torch.jit.save`;
`event check POLICY.pt` validates the portable CPU contract.

## Score

For one rollout, let `d` be the maximum forward distance reached before the
earliest of finishing, any part of the robot crossing either vertical lane
boundary at `±0.61 m` from the lane centre, or exceeding `1 cm` of
non-adjacent padded-body overlap. Lane containment uses all 1,960 authoritative
collision samples across all 38 represented bodies—including their radii—so
arms, hands and fingers, legs and feet, torso, pelvis, and head must all remain
inside. Let `t` be the first time that distance is reached. **Effective Speed**
is

```text
E = (d / 100 m) × (d / t) = d² / (100 m × t)
```

It is zero when `d <= 0` or `t <= 0`. A valid finish has `d = 100 m`, so
`E = 100 m / t`. A policy score is the highest Effective Speed from three
official rollouts.

Models are compared with three independent agent trials. At combined agent cost
`c`, `Q(c)` is the highest policy score any of those trials has produced by that
spending point. For a shared post-experiment cost cutoff `B`, **Cost-Adjusted
Effective Speed** is

```text
CAES(B) = (1 / B) × integral from 0 to B of Q(c) dc
```

This is the mean best-so-far Effective Speed over the shared budget horizon. It
has units of `m/s`; higher is better. The same `B` is used for every model and
cannot exceed the combined cost observed for any compared model.

## Commands

```bash
event gpu -- python3 -u /app/train/YOUR_SCRIPT.py  # run GPU work
event gpu status                                   # job and policy mirror
event gpu logs JOB_ID                              # worker output
event gpu wait JOB_ID                              # wait for completion
event check POLICY.pt                              # validate TorchScript ABI
event test POLICY.pt                               # run local published verifier
event archive POLICY.pt --note "..."               # retain immutable candidate
event history                                      # list archive receipts
event cost                                         # this trial's cumulative cost JSON
```

Only one A10G job runs per trial; later jobs queue. The exact nominal verifier
is read-only at `/app/verifier` and `event test` runs it on this trial's own GPU
allocation. Official scoring separately evaluates archived bytes and does not
return results or traces during the run. At most one archive may be outstanding,
with a five-minute interval between accepted archives.

## Durable GPU work

The CPU agent has 2 physical cores and 8 GiB RAM. GPU jobs have 6 cores, 12 GiB
RAM, and one A10G. They may be preempted. Save complete mutable training state
under `$SPRINT_GPU_CHECKPOINT_DIR`; read `$SPRINT_GPU_RESUME` and
`$SPRINT_GPU_RESUME_CHECKPOINT` on startup. Use
`event gpu checkpoint save` or `/opt/sprint_resilience.py` for atomic
publication. A replacement fails closed without a valid resumable checkpoint,
so work never silently restarts.

Atomically update `progress.json` with a completed candidate path:

```json
{"policy_path": "/durable/.../policy.pt"}
```

The host mirrors each newly reported policy into the CPU sandbox. Read the
fresh `agent_policy_mirror_path` from `event gpu status` before archiving. The
workspace and durable run directory survive CPU recreation; processes and RAM
do not. The sandbox has no general internet or cloud credentials.

## Cost

`event cost` prints one trusted JSON document for the cost incurred by this one
agent trial. The same ledger drives the final website:

```text
C(t) = C_api(t) + C_cpu_agent(t) + C_training(t)
C_role(t) = seconds(t) × (cores × CPU_rate + GiB × memory_rate + A10G × GPU_rate)
C_api(t) = sum(tokens_by_class × class_rate / 1,000,000)
```

Reasoning tokens are included in output tokens and are not charged twice. For
GPT-5.6 requests above 272,000 input tokens, input rates are multiplied by 2
and output by 1.5. A10G VRAM is included in the GPU price.

| component | frozen rate |
|---|---:|
| physical CPU core | `$0.141912/h` |
| memory | `$0.024012/GiB/h` |
| A10G, including 24 GiB VRAM | `$1.1016/h` |
| CPU agent: 2 cores + 8 GiB | `$0.475920/h` |
| training: 6 cores + 12 GiB + A10G | `$2.241216/h` |
| GPT-5.6 Luna uncached / cached / cache-write / output | `$0.20 / $0.02 / $0.25 / $1.20` per 1M tokens |
| DeepSeek V4 Flash miss / hit / output | `$0.14 / $0.0028 / $0.28` per 1M tokens |

Official verifier, observability, website, storage, credits, discounts, taxes,
and invoice adjustments are excluded. The JSON contains its timestamp, exact
equation and rates, token counts, allocated seconds, component costs, and
pricing provenance. Run `event check --rules` for the complete gating contract.
