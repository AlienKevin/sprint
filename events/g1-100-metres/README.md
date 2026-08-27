# g1-100-metres

## Description

Produce a TorchScript policy that moves a Unitree G1 humanoid through a
100-metre lane in Isaac Lab. Each trial has the same configured agent-cost
budget and is scored by the highest Effective Speed policy it submits before that budget
is exhausted. The concise task is in `instruction.md`; the complete
agent-facing contract is in `environment/README.md`.

## Difficulty Explanation

The challenge is to maximize legal forward progress and pace before evaluation
ends.

## Verification Explanation

Agents can run the published verifier on their own GPU allocation; official
scoring evaluates the submitted policy bytes separately and remains blind. Effective
Speed rewards legal distance and pace. A trial's final result is its highest
Effective Speed after the configured budget is exhausted.
