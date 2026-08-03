# G1 Sprint — a 100 m time trial for humanoid policies on Isaac Lab

A humanoid locomotion policy is normally reported as an average reward, on an
environment whose commands, start pose and disturbances are all randomised.
That number is the right one for training and the wrong one for comparison: it
is not a time, it is not reproducible across runs, and it says nothing about
whether the gait it scores would be recognised as running.

This is the same physics with the training machinery taken out. One robot, one
lane, 100 m, from a standing start, timed.

## What is frozen

The benchmark is Isaac Lab's `Isaac-Velocity-Flat-G1-v0` with everything a
policy can see or feel left exactly as trained, because a policy is only valid
inside the environment it was trained in. Change the contract and the number
measures the change, not the policy.

| | |
|---|---|
| embodiment | Unitree G1, `G1_CFG` (full collision meshes), 37 actuated joints |
| physics | PhysX, 5 ms step, decimation 4 → 50 Hz control |
| ground | infinite plane, static friction 0.8 / dynamic 0.6 |
| observation | 123-D: base lin/ang velocity, projected gravity, command, joint pos/vel, last action |
| action | 37 joint-position targets, scale 0.5, offset from default pose |

## What changes

| term | training | evaluation |
|---|---|---|
| command | resampled every 10 s from a range | one fixed speed per lane, never resampled |
| start pose | ±0.5 m, yaw ±180° | on the line, facing the finish |
| start velocity | ±0.5 m/s, ±0.5 rad/s | at rest |
| observation noise | on | off |
| pushes / external force | on | off |
| episode length | 20 s | long enough to finish |
| start of trial | on reset | after a 1 s settle on the line |

The settle matters more than it looks. The G1 is spawned at a nominal standing
pose about 11 cm above where it actually stands and drops onto the plane;
timing from that instant charges the policy for the drop, and makes the settle
read as the feet sinking through the floor, since foot heights are scored
against where they started. One second under a zero command fixes both: on the
line, at rest, settled, then the clock starts.

Nothing stochastic remains, so a run is a deterministic replay rather than a
sample from a distribution. The heading target is pinned to +x — not as a
correction bolted onto the policy's output, but through the `heading_command`
interface it was trained with, which derives the yaw-rate element of its
command from heading error.

## What is measured

- **100 m time**, plus 10 m splits, interpolated between control steps so
  splits are not quantised to the 20 ms control period
- **achieved speed** against commanded speed, measured after a 2 s
  acceleration transient
- **peak speed**, **distance at fall** for runs that do not finish

## Validity

The task is to cross 100 m quickly. It is not to run, and the scoring is careful
about the difference: a G1 that sprints on hands and feet, bounds or hops is
answering the question, and disqualifying it would be prescribing the solution.

**Two checks gate a time:**

| check | rule | why |
|---|---|---|
| `finished` | crossed 100 m | a DNF is not a time |
| `no_ground_penetration` | feet stay within 1 cm of the height at which their soles rest on the plane | a solver artefact is not locomotion |

**Eight are measured and reported, and never disqualify:** `no_fall`, `in_lane`,
`upright_posture`, `alternating_gait`, `feet_leave_ground`, `foot_clearance`,
`steady_progress`, `returned_to_standing`. They are what make a result
interpretable — they are how we know the reference checkpoint jogs rather than
runs, and crabs 10 m sideways down the course — but they are reported next to
the time rather than subtracted from it.

Two caveats on the penetration gate, both open:

- it inspects the feet only. Widening it to every body needs a per-body
  calibration table — the lowest height each body can reach while loaded, which
  is a geometric constant of the asset — because a reference taken at the settle
  is wrong for limbs that legitimately swing low, and one taken from a body's own
  contact heights defines away what it is measuring.
- until that exists, the real defence against a policy exploiting the solver is a
  repeat at 2x fidelity (half the timestep, four times the position iterations),
  which is specified for the challenge but not yet part of this harness.

Every check reports its measured value next to its verdict, so a failure is
diagnosable and a threshold that turns out to be wrong is arguable against the
record rather than re-run from scratch.

The scoring is pure Python over recorded traces and is tested against synthetic
runs of each cheat above, with no simulator in the loop:

```bash
python tests/test_metrics.py
```

## Running it

Isaac Sim needs an RTX-capable GPU, so the harness runs on Modal:

```bash
MODAL_PROFILE=<profile> modal run modal_app.py
```

Directly, on a machine with Isaac Lab 2.3.2 installed:

```bash
python scripts/evaluate.py --checkpoint checkpoints/Isaac-Velocity-Flat-G1-v0.pt --headless
```

Each speed in `--speeds` gets its own lane, so one launch measures the whole
sweep. Environments are 6 m apart and Isaac Lab filters collisions between
them, so every lane covers the same 100 m of ground without interacting.

`--checkpoint zero` runs the do-nothing baseline — hold the default pose. It
should topple in about a second, and it is the cheapest end-to-end proof that
the harness is measuring the policy rather than itself.

## Video

Kit cannot create a render device on the Modal GPU container. Every NVIDIA
driver library is present and Vulkan enumerates the GPU, but the container
exposes `/dev/nvidia3` without the matching `/proc/driver/nvidia` entries, so
`vkCreateDevice` fails with `ERROR_INITIALIZATION_FAILED` and `--video`
produces nothing. PhysX is unaffected — it goes through CUDA rather than the
render device — so the simulation is exactly the one that would have been
filmed; only the camera is missing.

So the harness records state and draws it afterwards:

```bash
python scripts/render_trace.py results/isaaclab-g1-flat.trace.json --seconds 15
```

`--trace-out` writes the root pose and all 37 joint angles per frame for one
lane. `render_trace.py` writes those straight into `qpos` on the Unitree G1
MJCF and poses the model by forward kinematics — nothing is stepped, so unlike
a re-simulation it cannot drift away from the run being inspected. The twelve
leg joints carry identical names in both models; the fourteen Isaac joints with
no counterpart here are finger joints.

## First entry: Isaac Lab's own G1 checkpoint

The published `rsl_rl` checkpoint for `Isaac-Velocity-Flat-G1-v0`, fetched from
NVIDIA's asset store. It is `iter 1499` with 256/128/128 ELU layers, which is
exactly what `G1FlatPPORunnerCfg` specifies, so the published artifact matches
the published recipe.

| commanded | achieved | 100 m | valid |
|---|---|---|---|
| 0.25 | — | DNF | no |
| 0.50 | — | DNF | no |
| 0.75 | 0.62 | 160.74 s | **yes** |
| 1.00 | 0.85 | 117.55 s | **yes** |
| 1.25 | 1.08 | 92.45 s | **yes** |
| 1.50 | 1.32 | 76.03 s | **yes** |
| 2.00 | 1.78 | 56.12 s | **yes** |
| 2.50 | — | DNF | no |
| 3.00 | — | DNF | no |
| 4.00 | — | DNF | no |

Measured on the full-collision `G1_CFG`. Switching from Isaac Lab's
`G1_MINIMAL_CFG` changed these times by 0.000 s — this policy only ever
touches the ground with its feet, which collide in both assets — but the
minimal asset lets twenty other bodies pass through the floor, so a policy
using a knee or a hand would not be measured honestly on it.

Trained with `lin_vel_x` sampled from [0, 1] m/s, it extrapolates cleanly to
2.0 commanded — double its training range — at a consistent ~11% undershoot,
then cliffs rather than degrading. Its 10 m splits at 2.0 are metronomic: 6.58 s
for the first, including acceleration from rest, then nine splits between 5.49
and 5.51 s.

Five of the ten lanes produce a valid time. The reported descriptors that fire:

- **`in_lane`** at four speeds, up to 10.4 m of lateral drift over the 56 s run,
  with yaw held at zero throughout. It runs forward and crabs — a defect a
  velocity-tracking reward has no reason to penalise and a course does.
- **`foot_clearance`** below 1 m/s, where the gait becomes a shuffle.

For scale: the best published result on this robot is SPRINT (Wei et al.,
arXiv:2605.28549), a peak sprinting velocity of 6 m/s on a Unitree G1. The
Guinness 100 m record for a bipedal robot is Cassie's 24.73 s, 4.04 m/s. Usain
Bolt averaged 10.4 m/s. A velocity-tracking locomotion policy is not aimed at
speed, which is the gap this benchmark exists to measure.

## Entering a policy

The interface is a callable from the 123-D observation to 37 joint-position
targets. `sprintbench/policy.py` accepts an RSL-RL checkpoint, a bare actor
state dict, or a TorchScript module, and rebuilds the network from the shapes
in the checkpoint itself rather than through a training runner — so entering a
policy cannot drag in an environment that differs from the one being measured.
Width mismatches raise instead of degrading quietly.

## Layout

```
sprintbench/
  sprint_env_cfg.py   the evaluation environment
  sprint_command.py   constant-speed command with the heading pinned
  metrics.py          scoring and validity rules
  policy.py           framework-independent policy loading
  tasks.py            gym registration
scripts/evaluate.py   the harness
tests/test_metrics.py scoring tests, no GPU needed
modal_app.py          GPU image and entry points
```
