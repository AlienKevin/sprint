# g1-100-metres

## Description

Produce a TorchScript policy that runs a Unitree G1 humanoid over 100 metres as quickly as possible in Isaac Lab 2.3.2. Each policy is scored by its best Effective Speed across three rollouts; models are compared by Cost-Adjusted Effective Speed across three independent trials.

## Difficulty Explanation

The agent must produce a compliant policy from scratch using one persistent CPU sandbox and at most one A10G training sandbox at a time. The benchmark supplies the robot spawn, policy contract, course rules, local verifier, and cost accounting, but prescribes no training algorithm, reward, terrain, or policy architecture.

## Verification Explanation

Agents can run the published verifier at `/app/verifier` on their own training allocation. Official submissions are immutable and scored separately without returning results during the run. A rollout must finish within 60 seconds, stay within ±0.61 metres of lane centre, and avoid more than 1 centimetre of non-adjacent padded-body overlap.

For legal distance `d` reached in time `t`, Effective Speed is `d² / (100 m × t)`. The final model comparison averages the best-so-far Effective Speed over a shared aggregate-cost horizon; model API, CPU-agent, and training-sandbox costs are included, while verifier and observability overhead are excluded.
