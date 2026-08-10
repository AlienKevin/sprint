# terminal-bench/g1-sprint-100m

## Description

Produce a TorchScript policy that runs a Unitree G1 humanoid 100 m as fast as possible in Isaac Lab, on a frozen embodiment and physics, timed from a standing start.

## Difficulty Explanation

Requires producing a compliant policy from scratch with one CPU sandbox and at most one active training GPU per trial. The benchmark supplies the embodiment, physics, policy interface, and scoring contract, but no reference policy or course implementation. Any method that produces compliant TorchScript bytes is permitted.

## Verification Explanation

The exact nominal verifier source is published read-only at `/app/verifier`, and `sprint-verify policy.pt` queues it on the trial's own training A10G. This is the agent's debugging path and consumes only that trial's training allocation.

`sprint-submit` archives immutable policy bytes and returns immediately. Official results, gates, traces, queue progress, and completion timing are not returned to the agent. The trusted host retains the five-minute per-trial acceptance interval and at most one outstanding accepted policy, then drains all trials through one batch-scoped verifier lane. Every cache miss runs in an isolated offline verifier process, while exact task-and-policy duplicates may reuse a checksummed trusted result. The benchmark publishes the complete submitted-policy trajectory, its performance–time–cost Pareto frontier, and the best valid submitted policy. Verifier cost remains separate measurement overhead. Each policy receives three trials and the fastest counts. A valid run crosses 100 m inside 60 s, remains in the 1.22 m lane, and keeps non-adjacent padded body overlap ≤1 cm. These are the only scored checks; only TorchScript policy bytes cross into trusted scoring.
