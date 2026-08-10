# terminal-bench/g1-sprint-100m

## Description

Produce a TorchScript policy that runs a Unitree G1 humanoid 100 m as fast as possible in Isaac Lab, on a frozen embodiment and physics, timed from a standing start.

## Difficulty Explanation

Requires producing a compliant policy from scratch with one CPU sandbox and at most one active training GPU per trial. The benchmark supplies the embodiment, physics, policy interface, and scoring contract, but no reference policy or course implementation. Any method that produces compliant TorchScript bytes is permitted.

## Verification Explanation

During the run, `sprint-submit` queues immutable policy bytes and returns immediately; `sprint-board` shows host acceptance and completed score/gate feedback. Each model trial owns an independent verifier queue with at most one outstanding accepted policy and a five-minute host-enforced acceptance interval. Every cache miss runs in a fresh ephemeral sealed container, while exact task-and-policy duplicates reuse a checksummed trusted result after passing the same acceptance gate. The queues drain after the agent exits. There is no privileged final artifact or post-agent verification: the benchmark publishes the complete submitted-policy trajectory, its performance–time–cost Pareto frontier, and the best valid submitted policy. Verifier cost remains separate measurement overhead. Each policy receives three trials and the fastest counts. A valid run crosses 100 m inside 60 s, remains in the 1.22 m lane, and keeps non-adjacent padded body overlap ≤1 cm. These are the only scored checks. Every scoring run is isolated and offline; only TorchScript policy bytes cross into the trusted verifier.
