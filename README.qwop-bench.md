# qwop-bench — 100 m replay viewer

A 3D, broadcast-style spectator view of a 100 m final where each lane is a
different agent. It replays recorded runs; it is not a playable game.

```bash
python3 tools/make_runs.py        # -> data/runs/demo.json  (synthetic sample field)
python3 tools/build_viewer.py     # -> dist/race.html       (single self-contained file)
node    tools/smoke.js            # headless checks: no-throw + shot framing
```

`dist/race.html` has zero external requests — it runs from `file://`, any static
host, or inside a strict-CSP embed. Raw WebGL, no libraries.

## The broadcast

The shot list follows World Athletics' own TV production doctrine for track
races rather than generic game-camera instincts. Three rules from it shaped the
design:

* **The cutting lives in the buildup.** Aerial, cablecam, crowd, walk-out and
  lane-by-lane introductions all cut; each finalist gets a close-up as they are
  introduced, because viewers need to be able to identify them later.
* **"Great moderation" after the "on your marks" cue** — no close-ups, no
  cutting, the frame settles and waits.
* **The race is one shot.** *"The race is filmed in one shot. All separate shots
  of the start should be dropped from the cutting patterns of host
  broadcasters"* — every finalist stays in continuous view. Reactions come
  before replays after the line.

| Cue | Race clock | Shot |
| --- | --- | --- |
| `open`    | −48.0 s | aerial / beauty pass over the bowl, title |
| `cable`   | −40.5 s | cablecam down the home straight to the start |
| `lights`  | −33.0 s | house lights down, crowd light show |
| `walkout` | −26.5 s | the finalists come out, one per ~0.55 s |
| `intros`  | −19.5 s | lane-by-lane introductions, ~1.6 s each, spotlight + lower third |
| `prep`    | −6.6 s  | lights up, strides and practice starts |
| `blocks`  | −3.4 s  | into the blocks |
| `marks`   | −1.75 s | "on your marks" — the race frame settles |
| `set`     | −0.60 s | "set" |
| gun       | 0.0 s   | **one continuous rail move to the line** |
| finish    | +0.45 s | winner reaction (0.45× slow motion), then the rest of the field |

Edit `CUE` in `viewer/race.template.html` to retime the whole opening; every
shot boundary is derived from it.

### The one-shot rule, adapted

A real 100 m field finishes inside ~2 m, so one frame trivially holds everyone.
A field of QWOP agents strings out over 60 m — the winner crosses the line while
someone is still face-down at 4 m — and no single lens holds both. So the rail
cam frames the **contending group** (upright, within `CONTEND_M` = 45 m of the
leader), solving for camera distance first and only widening the lens once the
rail runs out of room. Anyone dropped is named on the camera slate
(`· N OFF FRAME`), tracked in the field strip, and picked up by CAM 10 after the
finish. `tools/smoke.js` asserts this: every contender in frame for the whole
move, and the introduction close-ups hold their athlete.

## The venue

A real 400 m track: two 84.39 m straights, 36.5 m bends, eight 1.22 m lanes,
generated procedurally in the ground shader along with the infield, the mown
football pitch inside it and the concrete apron. Because the home straight is
only 84.39 m long, the 100 m start sits on a **straight extension** past the
bend — so the bowl has an opening at that end, exactly as a real stadium does.

Lane 1 is the inside lane, which puts the infield at negative z and the main
grandstand at positive z. Every camera therefore works from the infield, with
the crowd behind the athletes. This is not cosmetic: an earlier build had the
stands as two slabs beside a straight strip of track, and the race camera pulled
back *through* them as the field spread — the picture went black. `safeEye()`
now clamps any camera that would end up inside the structure, and `tools/smoke.js`
asserts the director never needs that clamp, that the bowl never overlaps the
track, and that the start corridor stays open.

The bowl is three seating rings plus a roof ring, each segment rotated to face
the infield with a crowd-shaded deck on its inner face (seats along the
segment's tangent, rows up its height, ~78% occupancy, and per-seat phone lights
during the pre-race light show). Also modelled: the starter on his rostrum
outside lane 8 — arm and pistol up through "set", muzzle flash on the gun, and
he stays there — the photo-finish tower on the line, the finish gantry, the
trackside boards, and the wind gauge at 50 m.

## Replay format

`data/runs/*.json`:

```jsonc
{
  "meta": {
    "event": "100m", "title": "...", "subtitle": "...",
    "track_m": 100, "fps": 20, "duration": 30.0,
    "fields": ["x", "lean", "bob", "roll", "hip_l", "knee_l", "hip_r", "knee_r",
               "sh_l", "el_l", "sh_r", "el_r", "fallen"]
  },
  "runners": [{
    "id": "ppo-baseline", "label": "ppo-baseline", "note": "PPO, 40M steps",
    "lane": 2, "kit": "#2f7de1",
    "status": "finished" | "fell" | "dnf",
    "finish_time": 16.75,          // null if they never got there
    "frames": [[x, lean, bob, ...], ...]   // one row per 1/fps second from the gun
  }]
}
```

Angles are radians; `x` is metres down the track; `bob` offsets pelvis height
from 0.92 m; `fallen` blends 0 → 1 into the face-plant pose. The viewer
interpolates between frames, so 20 fps is enough. Lanes are 1–8, centred on
z = 0, 1.22 m apart.

To wire this to real episodes, replace `PROFILES` + `simulate()` in
`tools/make_runs.py` with a loop over recorded agent runs that emits the same
rows. Nothing else needs to change; the camera, captions ("… DOWN AT 30m",
"WINNER — …"), and leaderboard are all derived from the data.

## Controls

`space` play/pause · `F` free cam · drag / wheel to orbit · `1`–`8` follow a
lane · `0` back to the director · click a row in the field strip to follow that
agent · **⏭ To gun** skips the buildup.

## Sources

* [World Athletics — TV coverage of major athletics events: track competition](https://worldathletics.org/news/news/tv-coverage-of-major-athletics-events-track-c)
* [Ross Video / Spidercam cable-cam systems at Paris 2024](https://www.newscaststudio.com/2024/08/14/ross-video-supports-paris-olympics-coverage-with-ar-aerial-cameras/)
* [Olympics.com — the Paris 2024 100 m final crowd light show and walk-out](https://www.olympics.com/en/news/noah-lyles-wins-paris-2024-athletics-olympic-mens-100m-gold)
