#!/usr/bin/env python3
# Copyright (c) 2026 QWOP-bench contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Build a self-contained results page.

The Artifact sandbox blocks every external host, so the clip and the chart are
inlined as data URIs rather than linked.  Written as a builder because a 1 MB
base64 blob does not belong in a hand-edited file.

    python scripts/build_page.py
"""

from __future__ import annotations

import base64
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = "/tmp/claude-1000/-data-qwop-bench/b30c9aaa-c092-4bc7-b4da-4d4719508d31/scratchpad"


def data_uri(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


SPLITS = [6.583, 5.488, 5.501, 5.498, 5.509, 5.504, 5.511, 5.507, 5.514, 5.506]

CSS = """
:root {
  /* pulled from the render itself: the simulator's ground plane and the G1's
     white shell, rather than a palette picked in the abstract */
  --plane: #3c5a78;
  --ink: #141a21;
  --paper: #e9ecef;
  --card: #ffffff;
  --clay: #b04a2c;
  --moss: #3f6b52;
  --mute: #6b7580;
  --rule: #ccd3da;
  --display: ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  --body: Charter, "Bitstream Charter", "Iowan Old Style", "Source Serif Pro", Georgia, serif;
  --data: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --plane: #7ea3c4; --ink: #dfe4e9; --paper: #0f1318; --card: #171d24;
    --clay: #d9704e; --moss: #6aa585; --mute: #8d97a2; --rule: #2a333c;
  }
}
:root[data-theme="dark"] {
  --plane: #7ea3c4; --ink: #dfe4e9; --paper: #0f1318; --card: #171d24;
  --clay: #d9704e; --moss: #6aa585; --mute: #8d97a2; --rule: #2a333c;
}
:root[data-theme="light"] {
  --plane: #3c5a78; --ink: #141a21; --paper: #e9ecef; --card: #ffffff;
  --clay: #b04a2c; --moss: #3f6b52; --mute: #6b7580; --rule: #ccd3da;
}

body {
  background: var(--paper); color: var(--ink);
  font-family: var(--body); font-size: 17px; line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 900px; margin: 0 auto; padding: 3rem 1.25rem 5rem; }
main { display: flex; flex-direction: column; gap: 3rem; }
section { display: flex; flex-direction: column; gap: 1rem; }

.eyebrow {
  font-family: var(--data); font-size: 0.7rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--mute);
}
h1 {
  font-family: var(--display); font-weight: 800; letter-spacing: -0.03em;
  font-size: clamp(2rem, 5.5vw, 3.1rem); line-height: 1.05; text-wrap: balance;
  margin: 0.4rem 0 0;
}
h2 {
  font-family: var(--display); font-weight: 700; letter-spacing: -0.02em;
  font-size: 1.3rem; margin: 0; text-wrap: balance;
}
.stand { font-size: 1.13rem; color: var(--ink); max-width: 62ch; margin: 0; }
p { margin: 0; max-width: 68ch; }
.note { color: var(--mute); font-size: 0.95rem; }
strong { font-weight: 600; }

/* the three numbers the page exists to deliver */
.band {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1px; background: var(--rule); border: 1px solid var(--rule);
  border-radius: 3px; overflow: hidden;
}
.band div { background: var(--card); padding: 1.1rem 1.2rem; }
.band .n {
  font-family: var(--data); font-variant-numeric: tabular-nums;
  font-size: 1.85rem; font-weight: 600; letter-spacing: -0.02em;
  display: block; line-height: 1.1;
}
.band .n.warn { color: var(--clay); }
.band .k {
  font-family: var(--data); font-size: 0.68rem; letter-spacing: 0.11em;
  text-transform: uppercase; color: var(--mute); display: block; margin-top: 0.4rem;
}

figure { margin: 0; }
video, figure img {
  width: 100%; display: block; border-radius: 3px;
  border: 1px solid var(--rule); background: var(--plane);
}
figcaption { font-size: 0.9rem; color: var(--mute); margin-top: 0.6rem; }

/* splits really are a sequence, so they get an ordered treatment */
.splits { display: flex; flex-direction: column; gap: 0.35rem; }
.split { display: grid; grid-template-columns: 3.4rem 1fr 4rem; align-items: center; gap: 0.7rem; }
.split .g {
  font-family: var(--data); font-size: 0.78rem; color: var(--mute);
  font-variant-numeric: tabular-nums; text-align: right;
}
.split .bar { height: 12px; background: var(--plane); border-radius: 1px; }
.split.first .bar { background: var(--clay); }
.split .t {
  font-family: var(--data); font-size: 0.85rem; font-variant-numeric: tabular-nums;
}

.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; min-width: 460px; }
th, td { padding: 0.5rem 0.7rem; text-align: right; border-bottom: 1px solid var(--rule); }
th {
  font-family: var(--data); font-size: 0.66rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--mute); font-weight: 500;
  border-bottom: 1px solid var(--ink);
}
th:last-child, td:last-child { text-align: left; }
td { font-family: var(--data); font-variant-numeric: tabular-nums; }
tr.best td { background: color-mix(in srgb, var(--plane) 12%, transparent); font-weight: 600; }
tr.gone td { color: var(--mute); }
.tag {
  font-family: var(--data); font-size: 0.68rem; letter-spacing: 0.06em;
  padding: 0.1rem 0.42rem; border-radius: 2px; text-transform: uppercase;
}
.tag.ok { color: var(--moss); border: 1px solid currentColor; }
.tag.no { color: var(--clay); border: 1px solid currentColor; }

.checks { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1px;
  background: var(--rule); border: 1px solid var(--rule); border-radius: 3px; overflow: hidden; }
.checks article { background: var(--card); padding: 1.1rem 1.2rem; display: flex;
  flex-direction: column; gap: 0.5rem; }
.checks h3 { font-family: var(--data); font-size: 0.82rem; margin: 0; font-weight: 600;
  letter-spacing: 0.01em; color: var(--clay); }
.checks p { font-size: 0.93rem; }

hr { border: 0; border-top: 1px solid var(--rule); margin: 0; }
footer { color: var(--mute); font-size: 0.88rem; }
code { font-family: var(--data); font-size: 0.88em; }
"""


def split_rows() -> str:
    widest = max(SPLITS)
    out = []
    for i, s in enumerate(SPLITS):
        gate = (i + 1) * 10
        cls = "split first" if i == 0 else "split"
        out.append(
            f'<div class="{cls}"><span class="g">{gate} m</span>'
            f'<span class="bar" style="width:{s / widest * 100:.1f}%"></span>'
            f'<span class="t">{s:.3f}s</span></div>'
        )
    return "\n".join(out)


def table_rows(runs: list[dict]) -> str:
    out = []
    for r in sorted(runs, key=lambda r: r["commanded_speed"]):
        cmd = r["commanded_speed"]
        fin = r["finish_time_s"]
        cls = "best" if cmd == 2.0 else ("gone" if fin is None else "")
        fell = next((c for c in r["checks"] if c["name"] == "no_fall"), None)
        if fin:
            outcome = '<span class="tag ok">finished</span>'
            speed = f'{r["mean_speed_mps"]:.2f}'
            time = f"{fin:.1f} s"
        elif fell and not fell["passed"]:
            outcome = f'<span class="tag no">fell at {fell["value"]:.2f} s</span>'
            speed = "—"
            time = f'{r["distance_m"]:.1f} m'
        else:
            outcome = '<span class="tag no">too slow</span>'
            speed = f'{r["mean_speed_mps"]:.2f}'
            time = f'{r["distance_m"]:.1f} m'
        out.append(
            f'<tr class="{cls}"><td>{cmd:.2f}</td><td>{speed}</td>'
            f"<td>{time}</td><td>{outcome}</td></tr>"
        )
    return "\n".join(out)


def main() -> None:
    doc = json.load(open(os.path.join(HERE, "results", "isaaclab-g1-flat-fullcollision.json")))
    clip = data_uri(os.path.join(SCRATCH, "clip.mp4"), "video/mp4")
    fall = data_uri(os.path.join(SCRATCH, "fall.mp4"), "video/mp4")
    chart = data_uri(os.path.join(HERE, "results", "isaaclab-g1-flat.png"), "image/png")

    html = f"""<title>G1 Sprint — Isaac Lab's own checkpoint over 100 m</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <span class="eyebrow">Isaac-Sprint-100m-G1-v0 · Unitree G1 · Isaac Lab 2.3.2</span>
  <h1>The reference humanoid policy jogs the&nbsp;100&nbsp;m in 56&nbsp;seconds</h1>
</header>

<main>
<section>
  <p class="stand">Isaac Lab's published locomotion checkpoint for the Unitree G1 is
  reported as an average reward on an environment that randomises its command every
  ten seconds. Put the same policy on a fixed 100&nbsp;m course, from a standing start,
  and time it, and a different picture emerges: it is competent, repeatable, and
  slower than the bipedal world record.</p>

  <div class="band">
    <div><span class="n">1.78<span style="font-size:0.55em"> m/s</span></span>
         <span class="k">peak sustained speed</span></div>
    <div><span class="n">56.1<span style="font-size:0.55em"> s</span></span>
         <span class="k">fastest valid 100 m</span></div>
    <div><span class="n warn">2.5<span style="font-size:0.55em"> m/s</span></span>
         <span class="k">command where it falls</span></div>
  </div>
</section>

<section>
  <h2>What it actually looks like</h2>
  <figure>
    <video controls loop muted playsinline preload="metadata" id="clip" src="{clip}"></video>
    <figcaption>The complete 100&nbsp;m at 2.0&nbsp;m/s commanded, chase camera, 56.1&nbsp;s
    plus the standing hold.
    This is a forward-kinematics replay of the recorded rollout — root pose and all
    37&nbsp;joint angles written straight into the Unitree G1 model — because Isaac's own
    renderer cannot create a device in the evaluation container. Nothing is re-simulated,
    so it cannot drift from the run being measured. The drawing model is the 23-DoF MJCF
    against Isaac's 37-DoF USD, whose legs differ by 77&nbsp;mm; the root is shifted by
    that measured amount so the soles meet the floor instead of sinking through it.</figcaption>
  </figure>
  <figure style="margin-top:1.6rem">
    <video controls loop muted playsinline preload="metadata" src="{fall}"></video>
    <figcaption>The same policy commanded to 2.5&nbsp;m/s. It is on the ground in
    1.48&nbsp;s. There is no gradual degradation between 2.0 and 2.5 — the gait
    simply has no answer above its ceiling. The environment no longer terminates on
    a fall, because ending an episode on torso contact assumes an upright biped and
    the task does not require one; a robot that goes down just stops covering ground
    and times out.</figcaption>
  </figure>
  <p>It is not running. The pelvis sits at 0.63&nbsp;m against a 0.74&nbsp;m nominal stance,
  so it is permanently crouched; the arms hang almost motionless instead of
  counter-rotating; the feet clear the ground by about 5&nbsp;cm and there is no flight
  phase. That is the gait you would predict from a reward that pays for velocity
  tracking and charges for joint torque, action rate and orientation deviation.
  Nothing in that objective asks for speed.</p>
</section>

<section>
  <h2>It is astonishingly consistent</h2>
  <p>Ten-metre splits from the 56.1&nbsp;s run. The first carries the acceleration from
  rest; the remaining nine never vary by more than 26&nbsp;milliseconds. It reaches
  steady state within 10&nbsp;m and holds it for the rest of the course.</p>
  <div class="splits">
{split_rows()}
  </div>
</section>

<section>
  <h2>Where the tracking gives out</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>commanded</th><th>achieved</th><th>100 m</th><th>outcome</th></tr></thead>
    <tbody>
{table_rows(doc["runs"])}
    </tbody>
  </table>
  </div>
  <p>The policy was trained with forward-velocity commands sampled from
  0–1&nbsp;m/s. It extrapolates cleanly to <strong>double</strong> that range at a
  consistent ~11% undershoot, then does not degrade so much as fall off a cliff:
  at 2.5&nbsp;m/s it is on the ground in a second and a half.</p>
  <figure>
    <img src="{chart}" alt="Left: achieved speed against commanded speed, tracking the
    ideal line to 2.0 m/s then collapsing to zero. Right: 100 m times falling from
    161 s to 56 s, with DNF bars at the extremes.">
  </figure>
</section>

<section>
  <h2>What counts, and what is merely observed</h2>
  <p>The task is to cross 100&nbsp;m quickly — not to run. Any gait the simulator permits
  counts, so only two checks gate a time: the base crossed the line, and nothing passed
  through the floor. Five of the ten lanes produce a valid time. Everything else is
  measured and reported next to it, never subtracted from it:</p>
  <div class="checks">
    <article>
      <h3>in_lane</h3>
      <p>The one that matters. Up to <strong>10.4&nbsp;m</strong> of lateral drift across
      the 56&nbsp;s run, with yaw held at zero throughout. It runs forward and crabs
      sideways — something a velocity-tracking reward has no reason to penalise, and a
      course does. This alone disqualifies four of the seven finishing runs.</p>
    </article>
    <article>
      <h3>foot_clearance</h3>
      <p>Median swing clearance of <strong>5.8&nbsp;cm</strong> at 0.25&nbsp;m/s, under the
      6&nbsp;cm floor. At the slowest commanded speed the gait becomes a shuffle.</p>
    </article>
    <article>
      <h3>collision geometry</h3>
      <p>Isaac Lab's velocity task uses an asset with most collision meshes stripped for
      speed: measured, <strong>three of forty-four bodies could collide</strong> and twenty
      passed through the floor — knees 4.7&nbsp;cm under, fingertips 13&nbsp;cm. Now switched
      to the full-collision asset, where nothing goes underground. It changed these times
      by <strong>0.000&nbsp;s</strong>, because this policy only ever touches the ground
      with its feet — but a policy using a knee or a hand would not have been measured
      honestly.</p>
    </article>
  </div>
</section>

<section>
  <h2>For scale</h2>
  <p>The best published result on this same robot is SPRINT, which reports a peak
  sprinting velocity of <strong>6&nbsp;m/s</strong> on a Unitree G1 with zero-shot
  sim-to-real transfer. The Guinness 100&nbsp;m record for a bipedal robot is Cassie's
  <strong>24.73&nbsp;s</strong>, an average of 4.04&nbsp;m/s from a standing start with a
  return to standing. Usain Bolt averaged 10.4&nbsp;m/s.</p>
  <p>So this checkpoint is under a third of the speed the same robot has been driven to
  in a published policy, and well under half the pace of the hardware record over the
  full distance. That is the entire point of measuring it this way: a velocity-tracking
  policy was never aimed at speed, and an average reward will never tell you by how much
  it misses.</p>
</section>

<hr>
<footer>
  <p>Evaluation environment derived from <code>Isaac-Velocity-Flat-G1-v0</code> with the
  training machinery removed: fixed command, deterministic start from rest, no
  observation noise, no pushes, seeded. Embodiment, physics, the 123-dimensional
  observation and the 37-joint action are unchanged from training, because a policy is
  only valid inside the setup it was trained in. Checkpoint: the published
  <code>rsl_rl</code> artifact, iteration 1499.</p>
</footer>
</main>
</div>
<script>
  // autoplay the loop unless the reader has asked for less motion
  var v = document.getElementById('clip');
  if (v && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {{
    v.autoplay = true;
    v.play().catch(function () {{}});
  }}
</script>
"""
    out = os.path.join(SCRATCH, "sprint_report.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
