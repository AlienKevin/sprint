# g1-100-metres

## Description

Produce a TorchScript policy that moves a Unitree G1 humanoid through a
100-metre lane in Isaac Lab. Each trial has the same configured agent-cost
budget and is scored by the fastest valid policy it submits before that budget
is exhausted. The concise task is in `instruction.md`; the complete
agent-facing contract is in `environment/README.md`.

## Difficulty Explanation

The policy must move the robot quickly, keep its entire collision envelope
within the lane, and avoid geometric self-collision.

## Verification Explanation

Agents can run the published verifier on their own GPU allocation; official
scoring archives the same policy bytes separately and remains blind. Effective
Speed rewards legal distance and pace. A trial's final result is its highest
Effective Speed after the configured budget is exhausted.
