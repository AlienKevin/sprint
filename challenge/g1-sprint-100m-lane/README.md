# terminal-bench/g1-sprint-100m

## Description

Train a control policy that runs a Unitree G1 humanoid 100 m as fast as possible in Isaac Lab, on a frozen embodiment and physics, submitted as a TorchScript policy and timed from a standing start

## Difficulty Explanation

Requires producing a running policy from scratch on one GPU: reward design, curriculum, algorithm and training budget are all the agent's own, and neither a reference policy nor a copy of the course is provided. The agent trains against whatever it builds and may submit immutable candidates for retrospective scoring, but verifier feedback stays sealed. Model behavior therefore depends on its own training evidence rather than the latency or frequency of official scoring. The observation exposes cross-track and heading error but no target velocity, so the policy must choose its own speed. For scale, Isaac Lab's published G1 checkpoint averages 1.78 m/s and takes 56 s, the fastest published result on this robot is SPRINT (Wei et al., arXiv:2605.28549) at a 6 m/s peak, and the Guinness 100 m record for a bipedal robot is Cassie's 24.73 s.

## Verification Explanation

During the run, `sprint-submit` queues immutable policy bytes and returns only a durable local receipt. Harbor archives and scores every accepted policy on the real course without returning scores, failures, queue depth, or completion timing to the agent. Concurrent model runs have independent queues sharing one trusted verifier lease; each cache miss runs in a fresh sealed container, while exact task-and-policy duplicates reuse a checksummed trusted result. The queues drain after the agent exits. There is no privileged final artifact or post-agent verification: the benchmark publishes the complete submitted-policy trajectory, its performance–time–cost Pareto frontier, and the best valid submitted policy. Verifier cost remains separate measurement overhead. Each policy receives three trials and the fastest counts. A valid run crosses 100 m inside 60 s, remains in the 1.22 m lane, and keeps non-adjacent padded body overlap ≤1 cm. Other gait and robustness checks are reported but do not disqualify. Every scoring run is isolated and offline; only TorchScript policy bytes cross into the trusted verifier.
