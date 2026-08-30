#!/usr/bin/env python3
"""Convert a sealed verifier replay and G1 meshes to self-contained 3D HTML."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path

RENDERER_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PLAYER_TEMPLATE = REPOSITORY_ROOT / "web/replay-template.html"
SCENE_SOURCE = RENDERER_ROOT / "scene.js"
HQ_DEFAULT = RENDERER_ROOT / "g1_hq.json"
COLS = ["#6E97C4", "#E0A43B", "#B6F24E", "#F2704E"]
LANE_HALF_WIDTH_M = 0.61

# The default renderer remains a portable, self-contained HTML export. Website
# publication can explicitly share these immutable, content-addressed assets.
SHARED_ASSET_PREFIX = "/assets/replay/"
SHARED_ASSET_ERROR = """<script>
window.g1ReplayAssetFailed=function(name){
  window.__G1_REPLAY_ASSET_ERROR__=true;
  const message='Unable to load replay '+name+'. Please reload to retry.';
  const note=document.getElementById('failure-note');
  if(note){note.hidden=false;note.dataset.kind='incomplete';note.textContent=message;}
  const play=document.getElementById('replay');if(play)play.disabled=true;
  if(parent!==window)parent.postMessage({type:'g1:policies-error',message,failedCaptureIds:[],
    replayGeneration:new URLSearchParams(location.search).get('replayGeneration')||''},location.origin);
};
</script>"""


def _shared_asset(directory: Path, name: str, content: str) -> str:
    payload = content.encode("utf-8")
    filename = f"{name}-{hashlib.sha256(payload).hexdigest()}.js"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"content-addressed replay asset differs: {path}")
    else:
        # A publisher may snapshot this directory concurrently. Never expose a
        # partial immutable URL; the site's bundler excludes atomic .*.tmp files.
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".", suffix=".tmp", delete=False) as stream:
            staging = Path(stream.name)
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                os.chmod(staging, 0o644)
                os.replace(staging, path)
            finally:
                staging.unlink(missing_ok=True)
    return SHARED_ASSET_PREFIX + filename


def shared_asset_html(html: str, directory: Path) -> str:
    """Publish shared meshes/Three.js without reserializing authoritative data.

    Replacing only the HQ value preserves every other byte of DATA, including
    the hero's extended capture. ``restore_shared_html`` reverses this exactly
    for auditing and for downloading a standalone/offline HTML artifact.
    """
    if 'data-replay-asset="three"' in html:
        # Idempotent migration still fails closed if an asset is missing.
        restore_shared_html(html, directory)
        return html
    scripts = list(re.finditer(r"<script>([\s\S]*?)</script>", html))
    three = next((match for match in scripts if "Three.js Authors" in match[1][:200]), None)
    if three is None:
        raise ValueError("self-contained replay is missing Three.js")
    is_trial = "const TRIAL_BOOT={" in html
    marker = "const TRIAL_BOOT={" if is_trial else "<script>const DATA="
    offset = html.index(marker) + len(marker)
    match = re.search(r"\bhq:" if is_trial else r'"hq"\s*:', html[offset:])
    if match is None:
        raise ValueError("replay is missing its HQ mesh payload")
    start = offset + match.end()
    while html[start].isspace():
        start += 1
    _, end = json.JSONDecoder().raw_decode(html, start)
    mesh_json = html[start:end]
    mesh_url = _shared_asset(directory, "g1-hq", "window.__G1_REPLAY_HQ__=" + mesh_json + ";\n")
    three_url = _shared_asset(directory, "three", three[1])
    shared_tags = SHARED_ASSET_ERROR + "\n" + "\n".join(
        f'<script data-replay-asset="{kind}" src="{url}" '
        f'onerror="g1ReplayAssetFailed(\'{label}\')"></script>'
        for kind, url, label in (("three", three_url, "engine"), ("hq", mesh_url, "meshes"))
    )
    # HQ is assigned synchronously before any scene code runs. Scripts use the
    # browser cache across independent iframes; no global parent state required.
    html = html[:start] + ("window.__G1_REPLAY_HQ__" if is_trial else "null") + html[end:]
    html = html[:three.start()] + shared_tags + html[three.end():]
    if is_trial:
        html = html.replace("async function boot(){", "async function boot(){\n  if(window.__G1_REPLAY_ASSET_ERROR__)return;", 1)
    else:
        payload_start = html.index("<script>const DATA=") + len("<script>const DATA=")
        _, payload_end = json.JSONDecoder().raw_decode(html, payload_start)
        prefix = ";\nDATA.hq=window.__G1_REPLAY_HQ__;\nif(!window.__G1_REPLAY_ASSET_ERROR__){"
        if html[payload_end:payload_end + 2] != ";\n":
            raise ValueError("unexpected replay DATA terminator")
        html = html[:payload_end] + prefix + html[payload_end + 1:]
        closing = html.rindex("</script>")
        html = html[:closing] + "}\n" + html[closing:]
    return html


def restore_shared_html(html: str, directory: Path) -> str:
    """Restore the byte-identical self-contained export from published assets."""
    if 'data-replay-asset="three"' not in html:
        return html
    contents = {}
    pattern = r'<script data-replay-asset="(three|hq)" src="(/assets/replay/[^"/]+)" onerror="[^"]*"></script>'
    tags = list(re.finditer(pattern, html))
    if len(tags) != 2:
        raise ValueError("invalid shared replay assets")
    for tag in tags:
        path = directory / Path(tag[2]).name
        payload = path.read_bytes()
        if not path.stem.endswith(hashlib.sha256(payload).hexdigest()):
            raise ValueError(f"shared replay asset digest mismatch: {path}")
        contents[tag[1]] = payload.decode("utf-8")
    mesh = contents["hq"].removeprefix("window.__G1_REPLAY_HQ__=").removesuffix(";\n")
    if "const TRIAL_BOOT={" in html:
        html = html.replace("hq:window.__G1_REPLAY_HQ__", "hq:" + mesh, 1)
        html = html.replace("async function boot(){\n  if(window.__G1_REPLAY_ASSET_ERROR__)return;", "async function boot(){", 1)
    else:
        offset = html.index("<script>const DATA=") + len("<script>const DATA=")
        match = re.search(r'"hq"\s*:null', html[offset:])
        if match is None:
            raise ValueError("shared replay is missing its HQ placeholder")
        pos = offset + match.end() - 4
        html = html[:pos] + mesh + html[pos + 4:]
        html = html.replace(";\nDATA.hq=window.__G1_REPLAY_HQ__;\nif(!window.__G1_REPLAY_ASSET_ERROR__){", ";", 1)
        closing = html.rindex("}\n</script>")
        html = html[:closing] + html[closing + 2:]
    # Locate anew because the HQ insertion changes subsequent offsets.
    tags = list(re.finditer(pattern, html))
    html = html[:tags[0].start()] + "<script>" + contents["three"] + "</script>" + html[tags[1].end():]
    return html.replace(SHARED_ASSET_ERROR + "\n", "", 1)

MODEL_IDENTITIES = (
    ("deepseek", {"brand": "deepseek", "company": "DeepSeek", "model": "DeepSeek-V4-Flash", "color": "#7C54CD", "logo": "/assets/model-logos/deepseek.svg"}),
    ("glm-5.3", {"brand": "zai", "company": "Z.ai", "model": "GLM‑5.3‑Flash", "color": "#39B8B2", "logo": "/assets/model-logos/zai.svg"}),
    ("gpt-5.6-sol", {"brand": "openai", "company": "OpenAI", "model": "GPT‑5.6 Sol", "color": "#2279DC", "logo": "/assets/model-logos/openai.svg"}),
    ("gpt-5.6-luna", {"brand": "openai", "company": "OpenAI", "model": "GPT‑5.6 Luna", "color": "#66D693", "logo": "/assets/model-logos/openai.svg"}),
    ("claude-opus-5", {"brand": "anthropic", "company": "Anthropic", "model": "Claude Opus 5", "color": "#D97757", "logo": "/assets/model-logos/anthropic.svg"}),
)


def model_identity(model: str | None) -> dict[str, str] | None:
    """Return the compact identity printed on a runner's chest bib."""
    key = str(model or "").lower()
    for needle, identity in MODEL_IDENTITIES:
        if needle in key:
            return dict(identity)
    return None

PREFERRED = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "pelvis_contour_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_roll_link",
    "left_palm_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_roll_link",
    "right_palm_link",
]

G1_PARENT = {
    "pelvis": None,
    "pelvis_contour_link": "pelvis",
    "left_hip_pitch_link": "pelvis",
    "left_hip_roll_link": "left_hip_pitch_link",
    "left_hip_yaw_link": "left_hip_roll_link",
    "left_knee_link": "left_hip_yaw_link",
    "left_ankle_pitch_link": "left_knee_link",
    "left_ankle_roll_link": "left_ankle_pitch_link",
    "right_hip_pitch_link": "pelvis",
    "right_hip_roll_link": "right_hip_pitch_link",
    "right_hip_yaw_link": "right_hip_roll_link",
    "right_knee_link": "right_hip_yaw_link",
    "right_ankle_pitch_link": "right_knee_link",
    "right_ankle_roll_link": "right_ankle_pitch_link",
    "torso_link": "pelvis",
    "head_link": "torso_link",
    "left_shoulder_pitch_link": "torso_link",
    "left_shoulder_roll_link": "left_shoulder_pitch_link",
    "left_shoulder_yaw_link": "left_shoulder_roll_link",
    "left_elbow_pitch_link": "left_shoulder_yaw_link",
    "left_elbow_roll_link": "left_elbow_pitch_link",
    "left_palm_link": "left_elbow_roll_link",
    "right_shoulder_pitch_link": "torso_link",
    "right_shoulder_roll_link": "right_shoulder_pitch_link",
    "right_shoulder_yaw_link": "right_shoulder_roll_link",
    "right_elbow_pitch_link": "right_shoulder_yaw_link",
    "right_elbow_roll_link": "right_elbow_pitch_link",
    "right_palm_link": "right_elbow_roll_link",
}


def _rotate_inverse_xyzw(q: list[float], v: list[float]) -> list[float]:
    """Rotate ``v`` by the inverse of an xyzw quaternion."""
    x, y, z, w = (float(a) for a in q)
    mag = math.sqrt(x * x + y * y + z * z + w * w)
    if mag == 0:
        raise ValueError("zero-length capture quaternion")
    x, y, z, w = -x / mag, -y / mag, -z / mag, w / mag
    vx, vy, vz = (float(a) for a in v)
    uv = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    uuv = (
        y * uv[2] - z * uv[1],
        z * uv[0] - x * uv[2],
        x * uv[1] - y * uv[0],
    )
    return [
        vx + 2 * (w * uv[0] + uuv[0]),
        vy + 2 * (w * uv[1] + uuv[1]),
        vz + 2 * (w * uv[2] + uuv[2]),
    ]


def _rest_offsets(
    cap: dict, names: list[str], links: list[str]
) -> dict[str, list[float]]:
    """Recover fixed child origins in each captured parent link frame.

    Capture rows store independently rounded world poses. Taking a median over
    every frame removes that sub-millimetre noise and gives the fixed
    articulation offsets from the exact asset used for the rollout.
    """
    offsets: dict[str, list[list[float]]] = {}
    for child in links:
        parent = G1_PARENT.get(child)
        if parent not in links:
            continue
        ci, pi = names.index(child), names.index(parent)
        samples = offsets.setdefault(child, [])
        for run in cap["frames"]:
            for row in run:
                co, po = 1 + ci * 7, 1 + pi * 7
                delta = [row[co + axis] - row[po + axis] for axis in range(3)]
                samples.append(_rotate_inverse_xyzw(row[po + 3 : po + 7], delta))
    return {
        child: [round(statistics.median(axis), 6) for axis in zip(*samples)]
        for child, samples in offsets.items()
    }


def compute_terminal_event(
    frames: list,
    names: list[str],
    run: dict | None,
    *,
    failure_modes: list[str] | None = None,
) -> tuple[float | None, str | None]:
    """Return only the terminal classification recorded by the verifier.

    A pose replay is not collision telemetry. In particular, a low torso does
    not prove self-collision and a pelvis position does not represent the
    verifier's whole-body lane envelope. Never infer a failure mode from the
    animation. Older captures without terminal provenance receive the neutral
    ``did_not_finish`` classification.
    """
    del names  # Kept in the signature for renderer API compatibility.
    run = run or {}
    if run.get("valid") is True:
        return None, None

    last_t = float(frames[-1][0]) if frames else None
    explicit_time = run.get("first_disqualification_time_s")
    explicit_reason = run.get("first_disqualification_gate")
    if explicit_time is None:
        explicit_time = run.get("dq_time")
    if explicit_reason is None:
        explicit_reason = run.get("dq_reason")
    reason = explicit_reason or run.get("termination_reason")
    if reason is None:
        modes = [str(value) for value in (failure_modes or []) if value]
        reason = modes[0] if len(modes) == 1 else None

    # Historical schema versions used the failed gate name ``finished`` where
    # newer verifier payloads use the terminal reason ``timeout``. Preserve the
    # fact (the policy did not finish) without claiming why it stopped.
    if reason in {None, "finished"}:
        reason = "did_not_finish"

    terminal_time = explicit_time
    if terminal_time is None:
        terminal_time = run.get("stop_time_s")
    if terminal_time is None:
        terminal_time = run.get("duration_s")
    if terminal_time is None:
        terminal_time = last_t
    return (
        None if terminal_time is None else float(terminal_time),
        str(reason),
    )


def capture_to_data(
    cap: dict, hq_all: dict, meta_policy: str, pad: float = 1.5
) -> dict:
    names = cap["body_names"]
    links = [n for n in PREFERRED if n in hq_all and n in names]
    ridx = [names.index(n) for n in links]
    parents = {n: G1_PARENT[n] for n in links if G1_PARENT.get(n) in links}
    rest = _rest_offsets(cap, names, links)
    runs = cap.get("runs") or []
    finishes = []
    terminal_events: list[tuple[float | None, str | None]] = []
    for i, fr in enumerate(cap["frames"]):
        run = runs[i] if i < len(runs) else {}
        fin = None
        if run.get("finish") is not None:
            fin = float(run["finish"])
        else:
            ti = names.index("torso_link") if "torso_link" in names else 0
            o = 1 + ti * 7
            for row in fr:
                if row[o] >= 100.0:
                    fin = float(row[0])
                    break
            if fin is None and fr:
                fin = float(fr[-1][0])
        finishes.append(fin if fin is not None else float(fr[-1][0] if fr else 0.0))
        terminal_events.append(
            compute_terminal_event(
                fr,
                names,
                run,
                failure_modes=cap.get("failure_modes"),
            )
        )

    src_fps = float(cap.get("fps") or 25.0)
    out_fps = src_fps

    def pack(fr):
        row = [round(fr[0], 3)]
        for b in ridx:
            o = 1 + b * 7
            row += [
                round(fr[o], 3),
                round(fr[o + 1], 3),
                round(fr[o + 2], 3),
                round(fr[o + 3], 4),
                round(fr[o + 4], 4),
                round(fr[o + 5], 4),
                round(fr[o + 6], 4),
            ]
        return row

    policies = []
    for i, fr in enumerate(cap["frames"]):
        fin = finishes[i]
        run = runs[i] if i < len(runs) else {}
        valid = bool(run["valid"]) if i < len(runs) and "valid" in run else None
        terminal_time, terminal_reason = terminal_events[i]
        timed_out = valid is False and terminal_reason in {"timeout", "time_limit"}
        # Clip each seed shortly after its verifier-authored terminal event.
        # A timeout ends scoring at 60 s, but is not a physical failure. Retain
        # every recorded pose after it so replay presentation can continue
        # without changing the official result. Other failures keep the short
        # terminal tail used for their passive visual settle.
        clip_t = (
            float(fr[-1][0])
            if timed_out and fr
            else (
                terminal_time
                if (valid is False and terminal_time is not None)
                else fin
            )
            + pad
        )
        kept = [pack(f) for f in fr if f[0] <= clip_t]
        sy = kept[0][2] if kept else 0.0
        ys = [abs(f[2] - sy) for f in kept]
        lane_check = next(
            (
                check
                for check in run.get("checks", [])
                if check.get("name") == "in_lane"
            ),
            None,
        )
        max_lateral = (
            float(lane_check["value"])
            if lane_check is not None and lane_check.get("value") is not None
            else (max(ys) if ys else 0.0)
        )
        pol = {
            "label": f"Seed {i + 1}",
            "finish": round(fin, 3),
            "frames": kept,
            "max_lateral_m": round(max_lateral, 3),
            "valid": valid,
        }
        if valid is False and terminal_time is not None:
            pol["terminal_time"] = round(float(terminal_time), 3)
            pol["terminal_reason"] = terminal_reason
            pol["disqualified"] = terminal_reason in {"in_lane", "self_collision"}
            pol["timed_out"] = timed_out
        policies.append(pol)

    return {
        "fps": out_fps,
        "links": links,
        "parents": parents,
        "rest": rest,
        "hq": {n: hq_all[n] for n in links},
        "policies": policies,
        # Single-policy pages use the same compact three-lane course as the
        # homepage race, with the runner centered in physical lane 2.
        "lane_indices": [1 for _ in policies],
        "meta": {
            "policy": meta_policy,
            "track_lanes": 3,
            "lane_half_width_m": LANE_HALF_WIDTH_M,
            "source": cap.get("policy"),
            "frame_space": "world_link",
            "position_unit": "m",
            "quaternion_order": "xyzw",
            "visual_origin": "link_frame",
            "interpolation": "adjacent_authoritative_position_lerp_quaternion_slerp",
            "terminal_display": "verifier_classification_with_passive_visual_settle",
            "timeout_visual_playback": "all_recorded_frames_after_scoring_terminal",
        },
    }


def policy_nav_html(_active: str) -> str:
    """Return to the live comparison; policy navigation lives there."""
    return (
        '<div class="polsel" role="navigation" aria-label="Policy comparison">'
        '<a class="pol" href="/">← All runs and policies</a></div>'
    )


def inject_policy_nav(head: str, active: str) -> str:
    """Add or replace the comparison link above the replay stage."""
    nav = policy_nav_html(active)
    css = (
        ".polsel{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px}"
        ".pol{display:inline-flex;align-items:center;padding:7px 12px;"
        "border-radius:999px;border:1px solid var(--line);color:var(--muted);"
        "text-decoration:none;font-size:13px;font-weight:700;letter-spacing:.01em;"
        "background:transparent}.pol.on{background:var(--ink);color:var(--bg);"
        "border-color:var(--ink)}.pol:hover{border-color:var(--ink);color:var(--ink)}"
        ".pol.on:hover{color:var(--bg)}"
    )
    if ".polsel{" not in head:
        head = head.replace("</style>", css + "\n</style>", 1)
    if 'class="polsel"' in head:
        return re.sub(
            r'<div class="polsel"[^>]*>.*?</div>',
            nav,
            head,
            count=1,
            flags=re.S,
        )
    return head.replace(
        '<div class="stagewrap">', nav + '\n      <div class="stagewrap">', 1
    )


def assemble_html(
    data: dict,
    *,
    title: str,
    eyebrow: str,
    headline: str,
    lede: str,
    cap: str,
    story: str,
    sr_only: str,
    active: str,
    show_policy_labels: bool = False,
    shared_assets_dir: Path | None = None,
) -> str:
    """Render one self-contained replay from the checked-in HTML/JS chrome."""
    scene = SCENE_SOURCE.read_text()
    scene = scene.replace(
        "let raf=null,startWall=null,speed=0.6;",
        "let raf=null,startWall=null,speed=1;",
    )
    template = PLAYER_TEMPLATE.read_text()
    marker = "<script>const DATA="
    if marker not in template:
        raise ValueError(f"replay shell is missing data marker: {PLAYER_TEMPLATE}")
    head = template.split(marker, 1)[0]

    labels: list[str] = []
    for index, policy in enumerate(data["policies"]):
        color = policy.get("color") or COLS[index % len(COLS)]
        if policy.get("timed_out"):
            time_label = f"{policy['terminal_time']:.2f}"
        elif policy.get("valid") is False and policy.get("terminal_time") is not None:
            time_label = f"{policy['terminal_time']:.2f}"
        else:
            time_label = f"{policy['finish']:.2f}"
        policy_name = str(policy.get("label") or f"Seed {index + 1}")
        lane_number = int(policy.get("lane_number") or index + 1)
        name = (
            f'<span class="nm">Lane {lane_number} · {policy_name}</span>'
            if show_policy_labels
            else ""
        )
        labels.append(
            f'<div class="lc" id="lane{index}" role="button" tabindex="0" '
            f'aria-pressed="false" aria-label="Follow Lane {lane_number}, {policy_name}"><span class="sw" '
            f'style="background:{color}"></span>{name}<span class="tm" style="color:{color}">'
            f'{time_label}s</span><span class="d">0.0 m</span></div>'
        )
    lanes_html = "\n".join(labels)
    head = re.sub(
        r'<div class="lanes">.*?</div>\s*<div class="ctl">',
        f'<div class="lanes">\n{lanes_html}\n</div>\n      <div class="ctl">',
        head,
        count=1,
        flags=re.S,
    )
    head = head.replace(
        'data-s="0.5" aria-pressed="true"',
        'data-s="0.5" aria-pressed="false"',
    )
    head = head.replace(
        'data-s="1" aria-pressed="false"',
        'data-s="1" aria-pressed="true"',
    )
    head = head.replace(
        'data-s="0.1" aria-pressed="true"',
        'data-s="0.1" aria-pressed="false"',
    )
    head = head.replace(
        '<button class="btn camera-reset" id="camera-reset" type="button">'
        'Reset camera</button>',
        '<button class="btn camera-reset" id="camera-reset" type="button">'
        'Reset</button>',
    )

    substitutions = (
        (r"<title>.*?</title>", f"<title>{title}</title>"),
        (r'<h2 class="sr-only">.*?</h2>', f'<h2 class="sr-only">{sr_only}</h2>'),
        (r'<p class="eyebrow">.*?</p>', f'<p class="eyebrow">{eyebrow}</p>'),
        (r"<h1>.*?</h1>", f"<h1>{headline}</h1>"),
        (r'<p class="lede">.*?</p>', f'<p class="lede">{lede}</p>'),
        (
            r'<div class="cap">.*?</div>',
            f'<div class="cap"><span>{cap}</span>'
            "<span>Whole body must stay between the <b>±0.61 m</b> vertical lane "
            "planes; distance freezes at the first stop</span><span><b>Drag</b> to orbit, <b>scroll</b> to zoom, "
            "<b>space</b> to pause</span></div>",
        ),
        (
            r'<div class="story">.*?</div>\s*</div>\s*<script>',
            f'<div class="story"><p>{story}</p></div>\n</div>\n<script>',
        ),
    )
    for pattern, replacement in substitutions:
        head = re.sub(pattern, replacement, head, count=1, flags=re.S)
    head = inject_policy_nav(head, active)
    # Presentation only: keep capture fields and scoring precision unchanged.
    head = re.sub(
        r"(?<![\w.])(-?\d+(?:\.\d+)?)(\s*m/s\b)",
        lambda match: f"{float(match[1]):.2f}{match[2]}",
        head,
    )
    payload = json.dumps(data, separators=(",", ":"))
    html = head + marker + payload + ";\n" + scene + "\n</script>"
    return shared_asset_html(html, shared_assets_dir) if shared_assets_dir is not None else html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--hq", default=HQ_DEFAULT, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--meta-policy", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--eyebrow", required=True)
    parser.add_argument("--headline", required=True)
    parser.add_argument("--lede", required=True)
    parser.add_argument("--cap", required=True)
    parser.add_argument("--story", required=True)
    parser.add_argument("--sr-only", default="")
    parser.add_argument("--active", required=True)
    parser.add_argument("--pad", type=float, default=1.5)
    parser.add_argument("--shared-assets", type=Path, help="Website mode: write shared content-addressed assets here; omit for standalone HTML")
    args = parser.parse_args()

    capture = json.loads(args.capture.read_text())
    hq_all = json.loads(args.hq.read_text())["meshes"]
    data = capture_to_data(
        capture,
        hq_all,
        args.meta_policy,
        pad=args.pad,
    )
    sr_only = args.sr_only or (
        f"{args.headline}: Unitree G1 policy replay on a ±0.61 m corridor."
    )
    html = assemble_html(
        data,
        title=args.title,
        eyebrow=args.eyebrow,
        headline=args.headline,
        lede=args.lede,
        cap=args.cap,
        story=args.story,
        sr_only=sr_only,
        active=args.active,
        shared_assets_dir=args.shared_assets,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    finishes = [policy["finish"] for policy in data["policies"]]
    disqualifications = [
        (policy.get("terminal_time"), policy.get("terminal_reason"))
        for policy in data["policies"]
    ]
    lateral = [policy["max_lateral_m"] for policy in data["policies"]]
    print(
        f"wrote {args.out} ({args.out.stat().st_size} bytes) "
        f"finishes={finishes} dq={disqualifications} max_lat={lateral}"
    )


if __name__ == "__main__":
    main()
