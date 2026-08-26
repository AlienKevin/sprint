# G1 100 metres

Build and export a TorchScript policy for the frozen Unitree G1 course. This is
the agent-facing source of truth for the policy interface, target metric, and
available commands. The writable workspace is `/app`; this guide and the frozen
training helpers are installed in `/app/train`. The exact course implementation
is in `/app/verifier`. General outbound internet access is unavailable; all
required runtime dependencies and assets are installed locally.

## Policy interface

`spec.py` is authoritative for observation/action shapes, named observation
fields, action scale, 50 Hz control, and the optional state-reset ABI. Import
its named constants instead of copying offsets. Export with `torch.jit.save`;
`event check POLICY.pt` validates the portable CPU contract.

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
event gpu --submit-output /app/policy.pt -- python3 -u /app/YOUR_SCRIPT.py
event gpu status                                   # job and policy mirror
event gpu logs JOB_ID                              # worker output
event gpu wait JOB_ID                              # wait for completion
event gpu get JOB_ID /app/policy.pt                # retrieve verified output
event gpu cancel JOB_ID                            # cancel a queued or running job
event check POLICY.pt                              # validate TorchScript ABI
event test POLICY.pt                               # run local published verifier
event history                                      # inspect Harbor admission
event cost                                         # cumulative agent-cost JSON
```

Declare files that must return from the isolated GPU sandbox with repeatable
`--output /app/...` options. Use `--submit-output /app/POLICY.pt` only for a
policy you intentionally want considered by the blind official verifier; it
also returns that file as an output. Intermediate checkpoints and ordinary
outputs are never inferred as submissions. A submitted policy must be a valid
TorchScript `.pt` file no larger than 32 MiB. Each trial may submit at most 32
unique policies that pass the structural interface check; invalid or duplicate
policies do not consume the allowance. Declared files are required, bounded,
checksummed, and copied automatically. Do not encode model files into logs.
Only one A10G job runs at a time; later jobs run FIFO. Cancellation is
asynchronous once a GPU sandbox has been allocated, so inspect
`event gpu status JOB_ID` for the terminal acknowledgement.

The worker automatically bootstraps Python scripts that use Isaac Lab; do not
wrap them in another launcher or pass wrapper-reserved device flags. `/app`,
`/opt`, and the published verifier are already on `PYTHONPATH`. The exact nominal
verifier is read-only at `/app/verifier` and `event test` runs it on the current
GPU allocation. Official scoring separately evaluates explicitly submitted
bytes and does not return results or traces during the run.

## Checkpointing

GPU jobs may be preempted. The runtime reports the job as preempted and does not
silently restart it. State left only in RAM is lost; use
`event gpu checkpoint save STATE --sequence N` when you want the option to
submit a later job that resumes from a durable checkpoint. Run
`event gpu checkpoint --help` for details.

## Cost

`event cost` returns a JSON snapshot of cumulative model API, CPU, and training
cost, including the equation, rates, and component totals. Run
`event check --rules` for the complete gating contract.
