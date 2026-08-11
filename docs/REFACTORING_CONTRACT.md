# Refactoring compatibility contract

This refactor changes ownership and naming, not evaluation behavior. Line count,
file count, and the relative size of Harbor, the competition runtime, and an
event are not acceptance criteria. Code belongs where its responsibility is
most coherent, even when that produces a larger component.

The current `main` behavior is the baseline. A behavior may change only when it
is listed below as an intentional interface rename or is separately approved.

## Intentional interface changes

The proposed `event` command consolidates existing agent commands without
changing their behavior:

| Current interface | Proposed interface | Required semantic equivalence |
|---|---|---|
| `sprint-gpu-train -- COMMAND` | `event gpu -- COMMAND` | Same workspace pinning, resource contract, queue, logs, checkpoint, retry, and policy-mirror behavior |
| `sprint-gpu-train status/logs/wait/stop` | `event gpu status/logs/wait/stop` | Same job identity, status fields, output, and lifecycle |
| `sprint-check POLICY.pt` | `event check POLICY.pt` | Same CPU ABI validation, rules, and exit status |
| `sprint-verify POLICY.pt` | `event test POLICY.pt` | Same published checker source, training allocation, result, and artifacts |
| `sprint-submit POLICY.pt --note ...` | `event archive POLICY.pt --note ...` | Same immutable bytes, receipt, cooldown, queueing, and archival scoring |
| `sprint-cost` | `event cost` | Same trial-local JSON, quantities, rates, exclusions, and total |
| `sprint-board` | `event history` | Same trial-local receipts and no official result disclosure |

No compatibility aliases are required after the cutover. Until the cutover is
complete, the existing commands remain authoritative.

## Agent-visible invariants

The refactor must preserve all of the following:

1. The instruction remains method-neutral and does not recommend a training
   algorithm, reward, terrain, policy architecture, curriculum, or solution.
2. `/app/train/README.md` and `/app/train/spec.py` remain the authoritative
   human-readable and executable policy contracts.
3. The observation shape, action shape, observation ordering, action scaling,
   control frequency, reset ABI, and TorchScript portability requirements are
   byte-for-byte or semantically identical to the current contract.
4. The agent receives one persistent CPU sandbox with the same CPU, memory,
   storage, network, credential, and durability boundaries.
5. GPU commands continue to use separate metered A10G compute, with at most one
   active training job for the trial and queued additional jobs.
6. Submitted GPU work runs the exact workspace bytes pinned at claim time.
   Retries restore those bytes and fail closed on digest mismatch.
7. Preempted replacement attempts require a valid full resumable checkpoint;
   they never silently restart training.
8. The checkpoint command, resume variables, progress reporting, policy mirror,
   status, logs, wait, and stop behavior remain available.
9. The published checker remains readable and runnable by the agent using the
   trial's own training allocation.
10. Local checker output remains available to the agent. Official archival
    checker results and traces remain hidden during the run.
11. Candidate acceptance remains limited to one outstanding accepted candidate
    and one new acceptance per 300 seconds per trial.
12. Every accepted candidate is retained and scored, including accepted work
    draining after a safe stop.
13. Candidate request replay remains idempotent. New request identity with the
    same bytes remains a new candidate subject to the cooldown.
14. `event cost` exposes only the current trial. No agent command exposes another
    trial's cost, telemetry, candidates, results, files, or identifiers.
15. The cost JSON retains the exact equation, frozen rates, quantities,
    component breakdown, exclusions, provenance, freshness, and completeness
    needed to reproduce the website's per-trial cost.

## Event and checker invariants

The canonical checker source may move, but local and archival execution must be
built from the same source and produce equivalent nominal output for identical
policy bytes and rollout inputs.

The G1 100 metres contract remains:

- Isaac Lab 2.3.2 with the current full G1 embodiment and physics settings;
- the exact versioned no-block standing start, with fixed root/joint pose and
  zero initial velocity for every policy and rollout;
- finish when the forward-most torso point crosses 100 metres;
- a 60-second finish requirement;
- base within 0.61 metres of lane centre;
- non-adjacent padded body overlap no greater than 1 centimetre;
- three official nominal rollouts per candidate;
- finish time interpolated at the crossing;
- legal distance cut off at the first lane or self-collision disqualification;
- rollout Effective Speed `d^2 / (100 m * t)`;
- candidate score equal to the greatest Effective Speed across its three
  official rollouts.

The refactor must also preserve:

1. Exact immutable policy bytes and SHA-256 provenance.
2. Checker image and source provenance.
3. A separate trusted checker environment with no agent-writable checker code.
4. CUDA/physics fallback detection and existing fail-closed behavior.
5. Retry only for classified infrastructure failures, using fresh trusted
   execution, up to the configured limit.
6. No infrastructure retry for deterministic policy or checker failures.
7. Durable result, lane, replay, stdout, lifecycle, and telemetry artifacts.
8. Exact-result caching only under the current policy and checker fingerprint.
9. Existing robustness behavior and its distinction from nominal gates.

## Cost and model-comparison invariants

Harbor may own usage collection and rate application, while the competition
runtime owns which roles count toward comparison. That boundary must not change
the resulting numbers.

The comparison cost remains:

```text
C(t) = C_api(t) + C_cpu_agent(t) + C_training(t)
```

Official checker, observability, website, storage, credits, discounts, taxes,
and invoice adjustments remain excluded from the comparison cost. Token-class
handling, long-context multipliers, allocation-based CPU and memory charges,
and A10G pricing remain identical to the frozen rate card used by the website.

For three independent trials of one model, aggregate cost is the sum of their
trial-local costs. If `Q(c)` is the best candidate score any of the three has
produced by aggregate cost `c`, Cost-Adjusted Effective Speed remains:

```text
CAES(B) = (1 / B) * integral from 0 to B of Q(c) dc
```

The shared post-experiment cutoff rules and step-function integration semantics
remain unchanged.

## Website and public-data invariants

The website remains a consumer of trusted exported data rather than an
independent scoring authority. The refactor must preserve:

1. Per-trial and merged three-trial identities.
2. Every scored candidate, including invalid candidates.
3. Official gates, legal distance, Effective Speed, and valid finish time.
4. Best-so-far performance-versus-cost construction.
5. Cost-Adjusted Effective Speed and its shared cutoff.
6. The website cost matching the sum of the same trial-local ledgers exposed to
   the agents.
7. Replay/stat/timeline links where the trusted artifact exists.
8. Explicit missing-artifact status rather than silently omitting readouts.
9. Existing public JSON schema meaning, or a versioned migration with an
   equivalence proof for every consumer.
10. No use of presentation code to recompute or override official results.

## Required evidence before deleting old code

Each migrated slice must pass the narrow tests for that slice and the following
broader gates before its predecessor is removed:

1. **Agent surface golden tests:** command help, accepted arguments, exit codes,
   status fields, JSON schemas, file visibility, and trial isolation.
2. **Policy ABI tests:** all current valid, invalid, adversarial, reset, shape,
   finite-action, bounded-action, and reproducibility cases.
3. **Checker tests:** course rules, thresholds, interpolation, legal-prefix
   cutoff, three-rollout selection, deterministic failures, and infrastructure
   retries.
4. **Checker equivalence canary:** the same policy bytes run through the
   agent-visible and trusted paths with equivalent nominal results and captured
   provenance.
5. **Cost equivalence canary:** the agent-visible cumulative cost exactly
   matches the trusted host calculation and website export at the same cutoff
   timestamp.
6. **Compute recovery tests:** workspace pinning, digest fencing, checkpoint
   publication, preemption, retries, cancellation, orphan cleanup, and safe
   stop.
7. **Harbor unit and integration suites:** both focused changed-area tests and
   the broad suite, with unrelated failures identified rather than ignored.
8. **Immutable image gates:** source digest, image identity, build, warm-up, and
   fresh-sandbox probes.
9. **Full-path functional canary:** real A10G training updates, policy export,
   local check, archival check, replay, telemetry, cost, and cleanup.
10. **Planned-count fleet probe:** one fresh worker per planned trial, all
    reaching the required application-ready marker and all cleaned up.
11. **Website contract tests:** trusted export, all public endpoints, DOM
    rendering, charts, candidate selection, and cost/score numerical parity.
12. **Failure injection:** agent restart, controller restart, worker preemption,
    checker infrastructure loss, partial artifact download, and stop while
    accepted work is draining.

No paid experiment launches until all applicable gates pass against the final
frozen commit and exact immutable images.

## Placement rule

Code moves according to responsibility, never to satisfy a size target:

- Harbor owns mechanisms reusable by unrelated Harbor tasks.
- The competition runtime owns the `event` experience, cross-trial comparison,
  inclusion policy, campaign control, and public export.
- An event owns embodiment, policy ABI, rules, event metric, checker, and event
  semantic tests.
- The website owns presentation only.
- Generated state remains outside tracked source.

If a responsibility does not fit one of these owners cleanly, it stays in place
until the boundary is understood and tested.
