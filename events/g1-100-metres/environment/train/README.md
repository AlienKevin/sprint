# G1 100 metres

Build and export a TorchScript policy for the frozen Unitree G1 course. This
directory is the agent-facing source of truth for the start state, policy ABI,
score, cost, and available commands.

## Course

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

## Target metric

**Optimize Cost-Adjusted Effective Speed (CAES); higher is better.** Effective
Speed measures one policy, while CAES rewards producing better policies with
less cumulative agent cost.

Only movement before the finish, timeout, or first lane or self-collision
disqualification counts. Let `d` be the greatest forward distance reached
during that interval, capped at `100 m`, and `t` the time when `d` was first
reached. **Effective Speed** is

```text
E = (d / 100 m) × (d / t) = d² / (100 m × t)
```

For a valid finish, this simplifies to `100 m / t`. A policy keeps its highest
Effective Speed from the official rollouts.

Models are compared across independent agent trials. At combined agent cost
`c`, `Q(c)` is the highest policy score any of those trials has produced by that
spending point. For a shared post-experiment cost cutoff `B`, **Cost-Adjusted
Effective Speed** is

```text
CAES(B) = (1 / B) × integral from 0 to B of Q(c) dc
```

This is the mean best-so-far Effective Speed over the shared cost horizon. It
has units of `m/s`; higher is better.

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

`event cost` returns the trusted JSON snapshot of this trial's cumulative model
API, persistent CPU-agent, and training-sandbox cost. It includes the timestamp,
component totals, exact equation and frozen rates, usage quantities, and pricing
provenance.

Official verifier, observability, and storage costs are excluded, as are
credits, discounts, taxes, and invoice adjustments. Run
`event check --rules` for the complete gating contract.
