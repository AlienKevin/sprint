# G1 100 metres

Build and export a TorchScript policy for the frozen Unitree G1 course. This is
the agent-facing source of truth for the policy interface, target metric, and
available commands. The writable workspace is `/app`; this guide and the frozen
training helpers are installed in `/app/train`. The exact course implementation
is in `/app/verifier`.

## Policy interface

`spec.py` is authoritative for observation/action shapes, named observation
fields, action scale, 50 Hz control, and the optional state-reset ABI. Import
its named constants instead of copying offsets. Export with `torch.jit.save`;
`event check POLICY.pt` validates the portable CPU contract.

Every rollout starts from the versioned `forward-ready-start-v1` pose: at rest,
torso leaned 40 degrees toward the finish, right foot leading by 0.46 m, both
feet planted, and the full robot behind the starting line. The pose is also the
zero-action joint reference. Training helpers, the local verifier, and official
scoring use the same byte-identical definition.

## Target metric

**Optimize Effective Speed; higher is better.**

An official policy evaluation measures legal progress until the policy
finishes, times out, leaves the lane, or self-collides. Let `d` be the greatest
forward distance before that point, capped at `100 m`, and `t` the first time it
reaches `d`. **Effective Speed** is

```text
E = (d / 100 m) × (d / t) = d² / (100 m × t)
```

For a valid finish, this simplifies to `100 m / t`. When the configured budget
is exhausted, the final score is the highest Effective Speed from the policies
archived by that point.

## Commands

```bash
event gpu --output /app/policy.pt -- python3 -u /app/YOUR_SCRIPT.py
event gpu status                                   # job and policy mirror
event gpu logs JOB_ID                              # worker output
event gpu wait JOB_ID                              # wait for completion
event gpu get JOB_ID /app/policy.pt                # retrieve verified output
event gpu cancel JOB_ID                            # cancel an undispatched job
event check POLICY.pt                              # validate TorchScript ABI
event test POLICY.pt                               # run local published verifier
event archive POLICY.pt --note "..."               # durably stage candidate
event history                                      # inspect Harbor admission
event cost                                         # cumulative agent-cost JSON
```

Declare every file that must return from the isolated GPU sandbox with a
repeatable `--output /app/...` option. Declared files are required, bounded,
checksummed, and copied automatically; `.pt` and `.pth` outputs become available
through `event gpu get`. Do not encode model files into logs. Only one A10G job
runs at a time; later jobs run FIFO. `event gpu cancel` works only before a GPU
sandbox is allocated.

The worker automatically bootstraps Python scripts that use Isaac Lab; do not
wrap them in another launcher or pass wrapper-reserved device flags. `/app`,
`/opt`, and the published verifier are already on `PYTHONPATH`. The exact nominal
verifier is read-only at `/app/verifier` and `event test` runs it on the current
GPU allocation. Official scoring separately evaluates archived bytes and does
not return results or traces during the run. At most one archive may be
outstanding, with a five-minute interval between accepted archives.

## Checkpointing

GPU jobs may be preempted. The runtime restores only checkpoints that your
training program explicitly publishes; it cannot recover state left only in
RAM. Periodically save all state needed to resume with
`event gpu checkpoint save STATE --sequence N`. Run
`event gpu checkpoint --help` for details.

## Cost

`event cost` returns a JSON snapshot of cumulative model API, CPU, and training
cost, including the equation, rates, and component totals. Run
`event check --rules` for the complete gating contract.
