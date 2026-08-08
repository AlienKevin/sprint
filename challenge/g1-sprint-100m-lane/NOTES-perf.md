# Verifier performance: CUDA physics is active; Vulkan is unavailable

The old conclusion that headless A10G verification had fallen back to CPU was
incorrect.  Modal GPU Sandboxes run under gVisor and do not expose the Vulkan
device that Kit's graphics foundation probes, so startup prints:

```
[Error] [gpu.foundation.plugin] No device could be created.
[Warning] [omni.physx.foundation.plugin] PhysXFoundation: Unable to get
          IGpuFoundation, GpuDevices or Graphics!
```

That is a camera/RTX graphics-path failure.  The verifier is headless and has no
camera sensors.  It is not the message PhysX emits when simulation falls back:

```
PhysX warning: GPU solver pipeline failed, switching to software
PhysX warning: GPU Bp pipeline failed, switching to software
```

`tests/test.sh` now archives Isaac stdout and refuses to score if either real
software-fallback marker appears.

## Current evidence

One bounded live A10G smoke on 2026-08-08 used the current verifier image, one
lane, an existing policy, a 1 m finish, and a 2 s trial window:

| phase / signal | observed |
|---|---:|
| verifier process wall time | 30 s |
| Kit until graphics-foundation probe | 13.5 s |
| scene creation | 0.39 s |
| simulation start | 1.56 s |
| active-rollout A10 SM utilization | 29-43% |
| active VRAM | 2,751 MiB |
| PhysX software-fallback markers | none |

The sandbox was terminated after the smoke.  Its long outer elapsed time
(513.7 s) was almost entirely a cold verifier-image rebuild; it is not per-score
runtime once the immutable image is cached.

Fresh archived production verifiers agree:

| policy | simulated work | process wall time |
|---|---:|---:|
| collapsed policy, 3 lanes | full 64 s window | 130.6 s |
| DeepSeek policy reaching 84.6-87.1 m, 3 lanes | full 64 s window | 116.1 s |

An August 8 telemetry sample taken during the collapsed-policy verifier showed
2.75 GiB of VRAM and 53% A10 utilization, consistent with the dedicated live
smoke.  A collapsed policy is not cheap: falling is deliberately non-gating
because crawling or a hands-and-feet gait is allowed, so the verifier must still
give it the entire trial window unless it crosses the line.

## Remaining cost

The dominant steady-state work is not startup.  Every 50 Hz control step runs
four PhysX substeps and evaluates full-body ground/self-collision geometry
(1,960 points) for each lane.  A nominal three-lane trial is currently about two
minutes.  A valid policy then receives two separate one-lane held-out physics
trials.  Result JSON now records Kit startup, CUDA identity, requested/actual
Isaac device, per-trial scene creation, reset/policy initialization, rollout,
close, and total wall times so future regressions are measured rather than
inferred from Kit's Vulkan warning.
