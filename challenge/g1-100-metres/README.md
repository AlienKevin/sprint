# g1-100-metres

## Objective

Produce a TorchScript policy that runs a Unitree G1 humanoid 100 metres as fast as possible in Isaac Lab 2.3.2, using a frozen embodiment, physics configuration, and policy interface.

## Agent environment

Each independent trial receives one persistent CPU agent sandbox and may run at most one metered A10G training sandbox at a time. The benchmark supplies the robot spawn, executable policy contract, course rules, local verifier, and cost accounting. It does not prescribe a training algorithm, reward, terrain, or policy architecture. Any method that produces compliant TorchScript bytes is permitted.

The main commands are:

- `sprint-gpu-train -- COMMAND` runs GPU work.
- `sprint-check policy.pt` checks the portable policy ABI on CPU.
- `sprint-verify policy.pt` runs the published verifier using the trial's training allocation.
- `sprint-submit policy.pt --note "..."` archives immutable policy bytes for official scoring.
- `sprint-cost` returns the trial's cumulative benchmark cost and frozen rates as JSON.
- `sprint-board` lists submission receipts without revealing official results.

## Course and policy scoring

The robot starts at rest facing the lane. A rollout is valid only if it crosses 100 metres within 60 seconds, keeps its base within ±0.61 metres of lane centre, and never exceeds 1 centimetre of non-adjacent padded-body overlap. Each submitted policy receives three simulator rollouts.

For one rollout, let `d` be its maximum forward distance before finishing or its first lane/self-collision disqualification, and let `t` be the time required to reach `d`. Its **Effective Speed** is:

```text
(d / 100 m) × (d / t) = d² / (100 m × t)
```

A valid finish therefore has Effective Speed `100 m / t`. The policy's score is the highest Effective Speed across its three rollouts.

## Model comparison

Each model is evaluated with three independent agent trials. At aggregate agent cost `c`, `Q(c)` is the best Effective Speed produced by any of those trials by the time their combined cost reaches `c`. For a shared retrospective budget cutoff `B`, **Cost-Adjusted Effective Speed** is:

```text
(1 / B) × integral from 0 to B of Q(c) dc
```

Cost includes model API usage, persistent CPU agents, and training sandboxes. Official verifier, observability, and website costs are excluded. Higher is better.

## Official scoring

The exact nominal verifier source is published read-only at `/app/verifier`, so agents can debug on their own training allocation. Official results, gates, traces, queue progress, and completion timing are not returned during the run.

The trusted host enforces a five-minute acceptance interval and at most one outstanding policy per trial. It archives every accepted policy and drains all trials through one batch-scoped verifier lane. Every cache miss runs in a fresh isolated offline verifier process; exact task-and-policy duplicates may reuse a checksummed trusted result. Only immutable TorchScript policy bytes cross into official scoring.

The benchmark publishes the submitted-policy trajectory, performance–cost frontier, Cost-Adjusted Effective Speed, and best valid submitted policy. Official verifier cost is retained separately as measurement overhead.
