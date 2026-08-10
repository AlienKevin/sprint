#!/usr/bin/env python3
"""Reconstruct completion-adjusted legal speed for an archived Sprint batch.

The r8 verifier predates the legal-prefix diagnostics now emitted by the
official scorer.  Its trusted pose captures nevertheless retain the selected
representative lane at 10 Hz.  This tool reconstructs, without rerunning a
policy, the first lane/self-collision DQ, maximum legal forward distance, time
to that distance, and the continuous score::

    completion_adjusted_speed_mps = distance_m ** 2 / (100 * elapsed_s)

The result is explicitly labelled as replay-derived rather than an official
rescore.  Official validity and failed-gate fields remain unchanged.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "ops"
WEB = ROOT / "sprint-web"
COURSE_DISTANCE_M = 100.0
LANE_HALF_WIDTH_M = 0.61
SELF_COLLISION_THRESHOLD_M = 0.01


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def completion_adjusted_speed(distance_m: float, elapsed_s: float) -> float:
    """Legal-prefix average speed discounted by the uncompleted course fraction."""

    if distance_m <= 0.0 or elapsed_s <= 0.0:
        return 0.0
    return distance_m * distance_m / (COURSE_DISTANCE_M * elapsed_s)


def contract_constants() -> tuple[dict[str, str | None], int, float, float]:
    """Read collision constants from the scorer without importing Isaac/Torch."""

    source = (
        ROOT
        / "challenge/g1-sprint-100m-lane/environment/verifier/sprintbench/rollout.py"
    )
    tree = ast.parse(source.read_text())
    wanted = {
        "G1_BODY_PARENT",
        "SELF_COLLISION_ANCESTRY",
        "SELF_COLLISION_RADIUS_PAD_M",
        "SELF_COLLISION_SPHERE_MARGIN_M",
    }
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)
    missing = wanted - values.keys()
    if missing:
        raise RuntimeError(f"scorer collision constants missing: {sorted(missing)}")
    return (
        values["G1_BODY_PARENT"],
        int(values["SELF_COLLISION_ANCESTRY"]),
        float(values["SELF_COLLISION_RADIUS_PAD_M"]),
        float(values["SELF_COLLISION_SPHERE_MARGIN_M"]),
    )


def is_chain_neighbor(
    a: str, b: str, parents: dict[str, str | None], ancestry: int
) -> bool:
    for start, other in ((a, b), (b, a)):
        current: str | None = start
        distance = 0
        while current is not None and distance <= ancestry:
            if current == other:
                return True
            current = parents.get(current)
            distance += 1
    return False


def is_sibling(a: str, b: str, parents: dict[str, str | None]) -> bool:
    pa, pb = parents.get(a), parents.get(b)
    return pa is not None and pa == pb


def is_digit_link(name: str) -> bool:
    return any(
        f"_{digit}_link" in name
        for digit in ("zero", "one", "two", "three", "four", "five", "six")
    )


def rotate_xyzw(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors by normalized xyzw quaternions with NumPy broadcasting."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    magnitude = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = quaternion / np.where(magnitude == 0.0, 1.0, magnitude)
    qvec = quaternion[..., :3]
    scalar = quaternion[..., 3:4]
    first = np.cross(qvec, vectors)
    second = np.cross(qvec, first)
    return vectors + 2.0 * (scalar * first + second)


@dataclass(frozen=True)
class BodyGeometry:
    name: str
    capture_index: int
    points: np.ndarray
    radii: np.ndarray
    center: np.ndarray
    sphere_radius: float


@dataclass(frozen=True)
class CollisionModel:
    bodies: tuple[BodyGeometry, ...]
    pairs: tuple[tuple[int, int], ...]
    sphere_margin_m: float


def collision_model(names: list[str]) -> CollisionModel:
    parents, ancestry, radius_pad, sphere_margin = contract_constants()
    geometry_path = (
        ROOT
        / "challenge/g1-sprint-100m-lane/environment/verifier/collision_geometry.json"
    )
    geometry = load_json(geometry_path)
    bodies: list[BodyGeometry] = []
    for name, entry in geometry["bodies"].items():
        if name not in names:
            continue
        points = np.asarray(entry["points"], dtype=np.float64)
        radii = np.asarray(entry["radii"], dtype=np.float64) + radius_pad
        center = points.mean(axis=0)
        sphere_radius = float(np.max(np.linalg.norm(points - center, axis=1) + radii))
        bodies.append(
            BodyGeometry(
                name=name,
                capture_index=names.index(name),
                points=points,
                radii=radii,
                center=center,
                sphere_radius=sphere_radius,
            )
        )
    pairs: list[tuple[int, int]] = []
    for left, a in enumerate(bodies):
        for right in range(left + 1, len(bodies)):
            b = bodies[right]
            if is_digit_link(a.name) or is_digit_link(b.name):
                continue
            if a.name in parents and b.name in parents:
                if is_chain_neighbor(a.name, b.name, parents, ancestry) or is_sibling(
                    a.name, b.name, parents
                ):
                    continue
            pairs.append((left, right))
    if not bodies or not pairs:
        raise RuntimeError("capture has no usable official collision geometry")
    return CollisionModel(tuple(bodies), tuple(pairs), sphere_margin)


def exact_pair_penetration(
    body_a: BodyGeometry,
    body_b: BodyGeometry,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> float:
    qa, qb = quaternions[body_a.capture_index], quaternions[body_b.capture_index]
    pa, pb = positions[body_a.capture_index], positions[body_b.capture_index]
    world_a = (
        rotate_xyzw(np.broadcast_to(qa, (len(body_a.points), 4)), body_a.points) + pa
    )
    world_b = (
        rotate_xyzw(np.broadcast_to(qb, (len(body_b.points), 4)), body_b.points) + pb
    )
    minimum = math.inf
    # Block one point cloud to keep pathological collision hulls memory-bounded.
    for start in range(0, len(world_a), 96):
        chunk = world_a[start : start + 96]
        distance = np.linalg.norm(chunk[:, None, :] - world_b[None, :, :], axis=-1)
        separation = (
            distance
            - body_a.radii[start : start + len(chunk), None]
            - body_b.radii[None, :]
        )
        minimum = min(minimum, float(np.min(separation)))
    return max(0.0, -minimum)


def frame_poses(frames: list[list[float]], body_count: int) -> tuple[np.ndarray, ...]:
    array = np.asarray(frames, dtype=np.float64)
    expected = 1 + body_count * 7
    if array.ndim != 2 or array.shape[1] != expected:
        raise ValueError(f"capture frame width {array.shape} does not match {expected}")
    poses = array[:, 1:].reshape(len(array), body_count, 7)
    return array[:, 0], poses[..., :3], poses[..., 3:7]


def torso_forward_trace(
    names: list[str], positions: np.ndarray, quaternions: np.ndarray
) -> np.ndarray:
    geometry_path = (
        ROOT
        / "challenge/g1-sprint-100m-lane/environment/verifier/collision_geometry.json"
    )
    geometry = load_json(geometry_path)["bodies"]["torso_link"]
    body_index = names.index("torso_link")
    points = np.asarray(geometry["points"], dtype=np.float64)
    radii = np.asarray(geometry["radii"], dtype=np.float64)
    output = np.empty(len(positions), dtype=np.float64)
    for index in range(len(positions)):
        q = np.broadcast_to(quaternions[index, body_index], (len(points), 4))
        world = rotate_xyzw(q, points) + positions[index, body_index]
        output[index] = float(np.max(world[:, 0] + radii))
    return output - output[0]


def interpolated_crossing(
    values: np.ndarray, times: np.ndarray, threshold: float
) -> tuple[float, int, float] | None:
    if values[0] > threshold:
        return float(times[0]), 0, 0.0
    indices = np.flatnonzero(values > threshold)
    if not len(indices):
        return None
    index = int(indices[0])
    if index == 0:
        return float(times[0]), 0, 0.0
    previous, current = float(values[index - 1]), float(values[index])
    fraction = (
        0.0 if current == previous else (threshold - previous) / (current - previous)
    )
    fraction = min(max(fraction, 0.0), 1.0)
    time = float(times[index - 1] + fraction * (times[index] - times[index - 1]))
    return time, index, fraction


def first_self_collision(
    model: CollisionModel,
    times: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> tuple[float, int, float] | None:
    compact = np.asarray([body.capture_index for body in model.bodies], dtype=int)
    local_centers = np.asarray([body.center for body in model.bodies])
    sphere_radii = np.asarray([body.sphere_radius for body in model.bodies])
    q = quaternions[:, compact]
    p = positions[:, compact]
    centers = rotate_xyzw(q, np.broadcast_to(local_centers, q.shape[:-1] + (3,))) + p
    pair_a = np.asarray([pair[0] for pair in model.pairs], dtype=int)
    pair_b = np.asarray([pair[1] for pair in model.pairs], dtype=int)
    gaps = (
        np.linalg.norm(centers[:, pair_a] - centers[:, pair_b], axis=-1)
        - sphere_radii[pair_a]
        - sphere_radii[pair_b]
    )
    previous = 0.0
    for frame_index in range(len(times)):
        candidates = np.flatnonzero(gaps[frame_index] < model.sphere_margin_m)
        if len(candidates) > 64:
            candidates = candidates[np.argsort(gaps[frame_index, candidates])[:64]]
        deepest = 0.0
        for pair_index in candidates:
            left, right = model.pairs[int(pair_index)]
            deepest = max(
                deepest,
                exact_pair_penetration(
                    model.bodies[left],
                    model.bodies[right],
                    positions[frame_index],
                    quaternions[frame_index],
                ),
            )
        if deepest > SELF_COLLISION_THRESHOLD_M:
            if frame_index == 0 or deepest == previous:
                return float(times[frame_index]), frame_index, 0.0
            fraction = (SELF_COLLISION_THRESHOLD_M - previous) / (deepest - previous)
            fraction = min(max(fraction, 0.0), 1.0)
            time = float(
                times[frame_index - 1]
                + fraction * (times[frame_index] - times[frame_index - 1])
            )
            return time, frame_index, fraction
        previous = deepest
    return None


def interpolate(values: np.ndarray, index: int, fraction: float) -> float:
    if index <= 0:
        return float(values[0])
    return float(values[index - 1] + fraction * (values[index] - values[index - 1]))


def score_capture(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    capture = json.loads(raw)
    names = list(capture["body_names"])
    frames = capture.get("frames") or []
    if len(frames) != 1 or not frames[0]:
        raise ValueError("expected one nonempty representative verifier lane")
    run = (capture.get("runs") or [{}])[0]
    times, positions, quaternions = frame_poses(frames[0], len(names))
    forward = torso_forward_trace(names, positions, quaternions)
    pelvis_index = names.index("pelvis")
    lateral = np.abs(positions[:, pelvis_index, 1])
    finish = interpolated_crossing(forward, times, COURSE_DISTANCE_M)
    finish_time = None if finish is None else finish[0]
    lane = interpolated_crossing(lateral, times, LANE_HALF_WIDTH_M)

    failed = {
        str(check.get("name"))
        for check in run.get("checks", [])
        if isinstance(check, dict) and not check.get("passed")
    }
    self_collision = None
    if "self_collision" in failed:
        self_collision = first_self_collision(
            collision_model(names), times, positions, quaternions
        )

    candidates: list[tuple[float, str, int, float]] = []
    if lane is not None:
        candidates.append((lane[0], "in_lane", lane[1], lane[2]))
    if self_collision is not None:
        candidates.append(
            (
                self_collision[0],
                "self_collision",
                self_collision[1],
                self_collision[2],
            )
        )
    if finish_time is not None:
        candidates = [item for item in candidates if item[0] <= finish_time]
    first_dq = min(candidates, default=None)

    horizon = finish_time if finish_time is not None else float(times[-1])
    reason = None
    if first_dq is not None:
        horizon = first_dq[0]
        reason = first_dq[1]
    legal_times = list(times[times < horizon])
    legal_forward = list(forward[times < horizon])
    if first_dq is not None:
        legal_times.append(horizon)
        legal_forward.append(interpolate(forward, first_dq[2], first_dq[3]))
    elif finish is not None:
        legal_times.append(finish_time)
        legal_forward.append(COURSE_DISTANCE_M)
    if not legal_forward:
        legal_times = [float(times[0])]
        legal_forward = [float(forward[0])]

    distance = max(0.0, max(legal_forward))
    max_index = legal_forward.index(max(legal_forward))
    elapsed = float(legal_times[max_index])
    valid = bool(run.get("valid")) and finish_time is not None and reason is None
    if valid:
        distance = COURSE_DISTANCE_M
        elapsed = float(finish_time)
    score = completion_adjusted_speed(distance, elapsed)
    return {
        "continuous_score_mps": round(score, 6),
        "max_legal_distance_m": round(distance, 3),
        "time_to_max_legal_distance_s": round(elapsed, 3),
        "first_disqualification_gate": reason,
        "first_disqualification_time_s": (
            None if first_dq is None else round(first_dq[0], 3)
        ),
        "valid_run": valid,
        "capture_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_fps": capture.get("fps"),
        "representative_lane": capture.get("representative_lane"),
    }


def trusted_pose_capture_index(run_ids: Iterable[str]) -> dict[str, Path]:
    """Index sealed verifier trajectories by the exact policy bytes they graded.

    Website replay rendering is deliberately downstream of verification.  A
    renderer or HTML validation failure must therefore never make a trusted
    pose trajectory unavailable to performance reconstruction.  Cache hits are
    covered too: their policy hash resolves to the original fresh evaluation's
    verifier artifact, even when that evaluation belongs to another trial in
    the batch.
    """

    captures: dict[str, Path] = {}
    for run_id in sorted(set(run_ids)):
        attempts = RUNS / run_id / "harbor-jobs"
        for result_path in sorted(
            attempts.glob("**/artifacts/continuous/attempts/*/result.json")
        ):
            try:
                result = load_json(result_path)
            except (OSError, json.JSONDecodeError):
                continue
            policy_hash = str(result.get("artifact_sha256") or "")
            replay = result_path.parent / "verifier" / "replay.json"
            if len(policy_hash) == 64 and replay.is_file():
                captures.setdefault(policy_hash, replay)
    return captures


def resolve_pose_capture(
    *, run_id: str, policy_hash: str, trusted: dict[str, Path]
) -> tuple[Path, str] | None:
    """Resolve a scoreable trajectory without depending on website rendering."""

    rendered = RUNS / run_id / "captures" / f"frontier-{policy_hash[:12]}.json"
    if rendered.is_file():
        return rendered, "validated_website_capture"
    source = trusted.get(policy_hash)
    if source is not None and source.is_file():
        return source, "trusted_verifier_replay"
    return None


def step_auc(points: Iterable[dict[str, Any]], key: str, cap: float) -> float:
    rows = sorted(
        (
            (float(row[key]), float(row["continuous_score_mps"]))
            for row in points
            if finite_number(row.get(key)) is not None
            and finite_number(row.get("continuous_score_mps")) is not None
            and float(row[key]) <= cap
        ),
        key=lambda item: item[0],
    )
    cursor = 0.0
    best = 0.0
    area = 0.0
    for x_value, score in rows:
        x_value = max(cursor, min(x_value, cap))
        area += (x_value - cursor) * best
        best = max(best, score)
        cursor = x_value
    area += (cap - cursor) * best
    return area / cap if cap > 0.0 else 0.0


def frontier_replay_points(
    models: Iterable[dict[str, Any]], cost_cap: float, time_cap: float
) -> list[dict[str, Any]]:
    """Select the union of record-setting readouts on cost and time curves."""

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for model in models:
        for key, cap in (
            ("cumulative_agent_cost_usd", cost_cap),
            ("hours_since_agent_launch", time_cap),
        ):
            best = 0.0
            rows = sorted(
                (
                    point
                    for point in model.get("points", [])
                    if finite_number(point.get(key)) is not None
                    and float(point[key]) <= cap
                ),
                key=lambda point: float(point[key]),
            )
            for point in rows:
                score = float(point["continuous_score_mps"])
                if score <= best:
                    continue
                best = score
                selected[(point["source_run_id"], point["policy_sha256"])] = point
    return sorted(
        selected.values(),
        key=lambda point: (point["source_run_id"], point["submission_index"]),
    )


def publish_frontier_replays(
    *,
    models: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    trusted: dict[str, Path],
    cost_cap: float,
    time_cap: float,
) -> None:
    """Publish only record-setting replays and attach their stable URLs."""

    module_path = ROOT / "runs/build_lane_3d.py"
    spec = importlib.util.spec_from_file_location("sprint_frontier_replay", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load replay renderer: {module_path}")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    hq = load_json(ROOT / "runs/g1_hq.json")["meshes"]
    replay_dir = WEB / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    for stale in replay_dir.glob("readout-*.html"):
        stale.unlink()

    url_by_hash: dict[str, str] = {}
    for point in frontier_replay_points(models, cost_cap, time_cap):
        policy_hash = str(point["policy_sha256"])
        resolved = resolve_pose_capture(
            run_id=str(point["source_run_id"]),
            policy_hash=policy_hash,
            trusted=trusted,
        )
        if resolved is None:
            raise RuntimeError(f"frontier replay unavailable for {policy_hash}")
        capture_path, _ = resolved
        capture = load_json(capture_path)
        data = renderer.capture_to_data(
            capture,
            hq,
            f"trial {point['source_trial']} / policy {point['submission_index']}",
        )
        html = renderer.assemble_html(
            data,
            title="The Race to AGI4ALL · Policy replay",
            eyebrow="POLICY REPLAY",
            headline=f"Trial {point['source_trial']} · policy {point['submission_index']}",
            lede="Record-setting policy readout.",
            cap=f"Effective Speed {float(point['continuous_score_mps']):.3f} m/s",
            story="Replay freezes at the first disqualification or finish.",
            sr_only="Unitree G1 policy replay on the sprint course.",
            active=policy_hash[:12],
        )
        embed_css = (
            "<style>html,body{margin:0;background:#070908}.wrap{max-width:none;"
            "padding:0}.wrap>.eyebrow,.wrap>h1,.wrap>.lede,.polsel,.cap,.story{"
            "display:none}.stagewrap{margin:0;border:0;border-radius:0;box-shadow:none}"
            "</style>"
        )
        html = html.replace('<div class="wrap">', embed_css + '<div class="wrap">', 1)
        filename = f"readout-{policy_hash[:12]}.html"
        (replay_dir / filename).write_text(html)
        url_by_hash[policy_hash] = f"/replay/{filename}"

    for collection in (models, runs):
        for group in collection:
            for point in group.get("points", []):
                replay_url = url_by_hash.get(str(point["policy_sha256"]))
                if replay_url:
                    point["replay_url"] = replay_url


def model_family(model: str | None) -> str:
    value = (model or "").lower()
    if "deepseek" in value:
        return "deepseek"
    if "luna" in value:
        return "luna"
    raise ValueError(f"unsupported model family: {model!r}")


def build_cost_ledger(timeline: dict[str, Any]) -> dict[str, Any]:
    resources = timeline["resource_usage_summary"]
    estimate = resources["modal_estimate"]
    rates = {
        key: float(value)
        for key, value in estimate["pricing_snapshot"]["rates_usd_per_second"].items()
    }
    contract = resources["resource_contract"]
    provider = resources["modal_provider_billing"]["by_role_usd"]

    def role_rate(role: str, contract_key: str) -> float:
        spec = contract[contract_key]
        rate = (
            float(spec["physical_cpu_cores"]) * rates["CPU"]
            + (float(spec["memory_mb"]) / 1024.0) * rates["Memory"]
        )
        if int(spec.get("gpu_count") or 0):
            rate += rates[spec["gpu_type"]]
        estimated = float(estimate["by_role"][role]["estimated_cost_usd"])
        exact = float(provider[role])
        return rate * (exact / estimated if estimated else 1.0)

    events = timeline["events"]
    cpu_start = min(
        int(event["epoch_ms"])
        for event in events
        if event.get("kind") == "cpu_allocated"
    )
    starts = {
        event["lease_id"]: int(event["epoch_ms"])
        for event in events
        if event.get("kind") in {"gpu_allocated", "gpu_reallocated"}
        and event.get("lease_id")
    }
    ends = {
        event["lease_id"]: int(event["epoch_ms"])
        for event in events
        if event.get("kind") in {"gpu_released", "gpu_preempted"}
        and event.get("lease_id")
    }
    if set(starts) != set(ends):
        raise RuntimeError("training allocation lifecycle is incomplete")
    return {
        "origin_epoch_ms": int(timeline["clock"]["origin_epoch_ms"]),
        "end_epoch_ms": int(timeline["clock"]["end_epoch_ms"]),
        "cpu_start_epoch_ms": cpu_start,
        "cpu_usd_per_second": role_rate("cpu_agent", "cpu_agent"),
        "training_intervals": sorted((starts[key], ends[key]) for key in starts),
        "training_usd_per_second": role_rate("training_gpu", "training_worker"),
        "api_events": sorted(
            (
                int(event["epoch_ms"]),
                float(event.get("calculated_cost_usd") or 0.0),
            )
            for event in events
            if event.get("kind") == "model_request_usage"
        ),
    }


def cumulative_cost_at_epoch(ledger: dict[str, Any], epoch_ms: int) -> float:
    api_cost = sum(
        cost for event_ms, cost in ledger["api_events"] if event_ms <= epoch_ms
    )
    cpu_ms = max(
        0,
        min(epoch_ms, ledger["end_epoch_ms"]) - ledger["cpu_start_epoch_ms"],
    )
    training_ms = sum(
        max(0, min(epoch_ms, end_ms) - start_ms)
        for start_ms, end_ms in ledger["training_intervals"]
        if epoch_ms > start_ms
    )
    return (
        api_cost
        + (cpu_ms / 1000.0) * ledger["cpu_usd_per_second"]
        + (training_ms / 1000.0) * ledger["training_usd_per_second"]
    )


def aggregate_models(
    output_runs: list[dict[str, Any]],
    common_time_cap: float,
    cost_ledgers: dict[str, dict[str, Any]],
    requested_cost_cap: float,
) -> tuple[list[dict[str, Any]], float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in output_runs:
        grouped.setdefault(model_family(run.get("model")), []).append(run)
    if set(grouped) != {"deepseek", "luna"}:
        raise RuntimeError(f"expected DeepSeek and Luna runs, found {sorted(grouped)}")

    common_observed_cost = min(
        sum(float(run["summary"]["final_agent_cost_usd"]) for run in runs)
        for runs in grouped.values()
    )
    if requested_cost_cap > common_observed_cost:
        raise RuntimeError(
            f"requested ${requested_cost_cap:.2f} cap exceeds common observed "
            f"cost ${common_observed_cost:.2f}"
        )
    common_cost_cap = requested_cost_cap
    output_models: list[dict[str, Any]] = []
    for family, runs in sorted(grouped.items()):
        family_origin_ms = min(
            cost_ledgers[run["run_id"]]["origin_epoch_ms"] for run in runs
        )
        events = sorted(
            (
                int(point["epoch_ms"]),
                run,
                point,
            )
            for run in runs
            for point in run["points"]
            if finite_number(point.get("hours_since_agent_launch")) is not None
            and finite_number(point.get("epoch_ms")) is not None
        )
        points: list[dict[str, Any]] = []
        for epoch_ms, source_run, point in events:
            aggregate_cost = sum(
                cumulative_cost_at_epoch(cost_ledgers[run["run_id"]], epoch_ms)
                for run in runs
            )
            points.append(
                {
                    **point,
                    "source_run_id": source_run["run_id"],
                    "source_trial": int(source_run["run_id"].rsplit("-", 1)[-1]),
                    "hours_since_agent_launch": round(
                        (epoch_ms - family_origin_ms) / 3_600_000.0, 6
                    ),
                    "cumulative_agent_cost_usd": round(aggregate_cost, 6),
                }
            )
        best = max(points, key=lambda row: row["continuous_score_mps"])
        missing = sum(len(run["summary"]["missing_readout_indices"]) for run in runs)
        output_models.append(
            {
                "family": family,
                "model": runs[0]["model"],
                "run_ids": [run["run_id"] for run in runs],
                "points": points,
                "summary": {
                    "readout_count": len(points),
                    "cost_readout_count_at_cap": sum(
                        point["cumulative_agent_cost_usd"] <= common_cost_cap
                        for point in points
                    ),
                    "time_readout_count_at_cap": sum(
                        point["hours_since_agent_launch"] <= common_time_cap
                        for point in points
                    ),
                    "missing_pose_capture_count": missing,
                    "best_continuous_score_mps": best["continuous_score_mps"],
                    "best_source_run_id": best["source_run_id"],
                    "best_submission_index": best["submission_index"],
                    "cost_auc_mps_at_common_cap": step_auc(
                        points, "cumulative_agent_cost_usd", common_cost_cap
                    ),
                    "time_auc_mps_at_common_cap": step_auc(
                        points, "hours_since_agent_launch", common_time_cap
                    ),
                    "final_agent_cost_usd": sum(
                        float(run["summary"]["final_agent_cost_usd"]) for run in runs
                    ),
                },
            }
        )
    return output_models, common_cost_cap


def build(batch_prefix: str, output: Path, cost_cap: float = 80.0) -> dict[str, Any]:
    policy_index = load_json(WEB / "data/policies/index.json")
    timeline_index = load_json(WEB / "data/timelines/index.json")
    timeline_by_run = {row["run_id"]: row for row in timeline_index["runs"]}
    selected = sorted(
        (row for row in policy_index["runs"] if row["run_id"].startswith(batch_prefix)),
        key=lambda row: row["run_id"],
    )
    if len(selected) != 6:
        raise RuntimeError(f"expected six {batch_prefix!r} runs, found {len(selected)}")
    trusted_captures = trusted_pose_capture_index(
        str(row["run_id"]) for row in selected
    )

    cost_caps: list[float] = []
    time_caps: list[float] = []
    prepared: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    cost_ledgers: dict[str, dict[str, Any]] = {}
    for meta in selected:
        timeline_meta = timeline_by_run[meta["run_id"]]
        timeline = load_json(WEB / timeline_meta["path"].removeprefix("/"))
        summary = timeline.get("comparison_summary") or {}
        final_cost = finite_number(summary.get("final_agent_total_cost_usd"))
        wall_ms = finite_number(summary.get("wall_duration_ms"))
        if final_cost is None or wall_ms is None:
            raise RuntimeError(f"{meta['run_id']} lacks finalized cost/time summary")
        cost_caps.append(final_cost)
        time_caps.append(wall_ms / 3_600_000.0)
        ledger = build_cost_ledger(timeline)
        cost_ledgers[meta["run_id"]] = ledger
        prepared.append((meta, timeline, ledger))

    common_cost_cap = min(cost_caps)
    common_time_cap = min(time_caps)
    output_runs: list[dict[str, Any]] = []
    for meta, timeline, cost_ledger in prepared:
        run_id = meta["run_id"]
        public = load_json(WEB / meta["path"].removeprefix("/"))
        artifacts = {
            int(row["submission_index"]): row
            for row in timeline.get("artifacts", [])
            if row.get("submission_index") is not None
        }
        points: list[dict[str, Any]] = []
        missing: list[int] = []
        for policy in sorted(
            public.get("policies", []), key=lambda row: row["submission_index"]
        ):
            index = int(policy["submission_index"])
            artifact = artifacts.get(index)
            if artifact is None:
                missing.append(index)
                continue
            resolved_capture = resolve_pose_capture(
                run_id=run_id,
                policy_hash=str(policy["policy_sha256"]),
                trusted=trusted_captures,
            )
            if resolved_capture is None:
                missing.append(index)
                continue
            capture, capture_provenance = resolved_capture
            reconstruction = score_capture(capture)
            result_cost = artifact.get("cost_at_result") or {}
            api_cost = float(result_cost.get("api_calculated_usd") or 0.0)
            epoch_ms = finite_number(result_cost.get("epoch_ms"))
            origin_ms = finite_number(
                (timeline.get("clock") or {}).get("origin_epoch_ms")
            )
            hours = (
                None
                if epoch_ms is None or origin_ms is None
                else (epoch_ms - origin_ms) / 3_600_000.0
            )
            point = {
                "submission_index": index,
                "policy_sha256": policy["policy_sha256"],
                "finished_at": policy.get("finished_at"),
                "hours_since_agent_launch": None if hours is None else round(hours, 6),
                "epoch_ms": None if epoch_ms is None else int(epoch_ms),
                "cumulative_agent_cost_usd": (
                    None
                    if epoch_ms is None
                    else round(cumulative_cost_at_epoch(cost_ledger, int(epoch_ms)), 6)
                ),
                "cumulative_api_cost_usd": round(api_cost, 6),
                "cumulative_modal_cost_usd_reconciled": (
                    None
                    if epoch_ms is None
                    else round(
                        cumulative_cost_at_epoch(cost_ledger, int(epoch_ms)) - api_cost,
                        6,
                    )
                ),
                "failed_gates": policy.get("failed_gates") or [],
                "pose_capture_provenance": capture_provenance,
                **reconstruction,
            }
            points.append(point)
        if missing:
            raise RuntimeError(
                f"{run_id} lacks trusted pose trajectories for submissions "
                f"{missing}; continuous score publication is fail-closed"
            )
        if not points:
            raise RuntimeError(f"{run_id} has no reconstructable policy readouts")
        best = max(points, key=lambda row: row["continuous_score_mps"])
        output_runs.append(
            {
                "run_id": run_id,
                "model": meta.get("model"),
                "reasoning_effort": meta.get("reasoning_effort"),
                "created_at": meta.get("created_at"),
                "points": points,
                "summary": {
                    "readout_count": len(points),
                    "missing_readout_indices": missing,
                    "best_continuous_score_mps": best["continuous_score_mps"],
                    "best_submission_index": best["submission_index"],
                    "cost_auc_mps_at_common_cap": step_auc(
                        points, "cumulative_agent_cost_usd", common_cost_cap
                    ),
                    "time_auc_mps_at_common_cap": step_auc(
                        points, "hours_since_agent_launch", common_time_cap
                    ),
                    "final_agent_cost_usd": (
                        timeline.get("comparison_summary") or {}
                    ).get("final_agent_total_cost_usd"),
                    "wall_duration_hours": (
                        float(
                            (timeline.get("comparison_summary") or {}).get(
                                "wall_duration_ms"
                            )
                        )
                        / 3_600_000.0
                    ),
                },
            }
        )

    output_models, model_cost_cap = aggregate_models(
        output_runs, common_time_cap, cost_ledgers, cost_cap
    )
    publish_frontier_replays(
        models=output_models,
        runs=output_runs,
        trusted=trusted_captures,
        cost_cap=model_cost_cap,
        time_cap=common_time_cap,
    )
    payload = {
        "schema_version": 2,
        "generated_at": utc_now(),
        "batch_prefix": batch_prefix,
        "metric": {
            "name": "effective_speed",
            "technical_name": "completion_adjusted_legal_speed",
            "formula": "distance_m^2 / (100m * time_to_distance_s)",
            "unit": "m/s",
            "higher_is_better": True,
            "distance_semantics": "maximum forward distance before first reconstructed lane/self-collision DQ",
            "provenance": "offline reconstruction from trusted representative-lane 10 Hz verifier pose captures; not an official rescore",
            "coverage_policy": "fail closed unless every published policy has an exact trusted pose trajectory; website rendering is not a scoring dependency",
        },
        "cost": {
            "includes": ["model API", "CPU agent", "training sandbox"],
            "excludes": ["verifier sandbox", "website", "observability infrastructure"],
            "modal_method": "allocation intervals integrated to each readout timestamp and reconciled by role to exact final pre-credit Modal billing",
            "common_auc_cap_usd": model_cost_cap,
            "aggregation": "sum cumulative cost across three trials; take best policy quality produced by any trial",
        },
        "time": {
            "common_auc_cap_hours": common_time_cap,
            "aggregation": "align three trials by elapsed agent time; take best policy quality produced by any trial",
        },
        "models": output_models,
        "runs": output_runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-prefix", default="sprint-20260810-r8-")
    parser.add_argument("--cost-cap", type=float, default=80.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=WEB / "data/performance/r8-continuous.json",
    )
    args = parser.parse_args()
    payload = build(args.batch_prefix, args.output, args.cost_cap)
    for run in payload["runs"]:
        summary = run["summary"]
        print(
            f"{run['run_id']}: {summary['readout_count']} readouts, "
            f"best={summary['best_continuous_score_mps']:.4f} m/s, "
            f"cost-AUC={summary['cost_auc_mps_at_common_cap']:.4f} m/s, "
            f"time-AUC={summary['time_auc_mps_at_common_cap']:.4f} m/s"
        )
    for model in payload["models"]:
        summary = model["summary"]
        print(
            f"{model['family']}: {summary['readout_count']} merged readouts, "
            f"best={summary['best_continuous_score_mps']:.4f} m/s, "
            f"cost-AUC={summary['cost_auc_mps_at_common_cap']:.4f} m/s, "
            f"time-AUC={summary['time_auc_mps_at_common_cap']:.4f} m/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
