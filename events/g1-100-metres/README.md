# g1-100-metres

## Description

Produce a TorchScript policy that moves a Unitree G1 humanoid through a
100-metre lane quickly and at low cost in Isaac Lab 2.3.2. The concise task is
in `instruction.md`; the complete agent-facing contract is in
`environment/train/README.md`.

## Difficulty Explanation

The policy must learn locomotion, keep its entire collision envelope within the
lane, avoid geometric self-collision, and balance quality against the API and
compute cost required to discover it. No training method, reward, terrain, or
architecture is prescribed.

## Verification Explanation

Agents can run the published verifier on their own GPU allocation; official
scoring archives the same policy bytes separately and remains blind. Effective
Speed rewards legal distance and pace; Cost-Adjusted Effective Speed averages
the best score found over a shared combined-cost horizon.
