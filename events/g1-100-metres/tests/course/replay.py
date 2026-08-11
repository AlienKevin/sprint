"""Compact, trusted pose replays produced during a course rollout."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
DEFAULT_FPS = 10.0


def result_dict(result: Any) -> dict[str, Any]:
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def representative_index(results: Sequence[Any]) -> int:
    """Choose the fastest valid lane, otherwise the lane that got furthest."""
    rows = [result_dict(result) for result in results]
    if not rows:
        raise ValueError("a replay requires at least one lane result")
    valid = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("valid") and row.get("finish_time_s") is not None
    ]
    if valid:
        return min(valid, key=lambda item: float(item[1]["finish_time_s"]))[0]
    return max(
        enumerate(rows),
        key=lambda item: (
            float(item[1].get("distance_m") or 0.0),
            float(item[1].get("peak_speed_mps") or 0.0),
            -item[0],
        ),
    )[0]


def failure_modes(row: dict[str, Any]) -> list[str]:
    modes = [
        str(check.get("name"))
        for check in row.get("checks", [])
        if isinstance(check, dict)
        and check.get("gating")
        and not check.get("passed")
        and check.get("name")
    ]
    if not modes and not row.get("valid"):
        modes.append("no_valid_finish")
    return sorted(set(modes))


class PoseRecorder:
    """Record world-space G1 body poses at a web-friendly sample rate."""

    def __init__(
        self,
        robot: Any,
        origins: Any,
        *,
        control_hz: float,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.robot = robot
        self.origins = origins
        self.body_names = list(robot.body_names)
        self.sample_every = max(1, round(float(control_hz) / float(fps)))
        self.fps = float(control_hz) / self.sample_every
        self.frames: list[list[list[float]]] = [
            [] for _ in range(int(origins.shape[0]))
        ]
        self._step = 0

    def __call__(
        self, t: float, _trace: dict[str, list], _resolved: list[bool]
    ) -> None:
        if self._step % self.sample_every == 0:
            positions = (
                (self.robot.data.body_pos_w - self.origins.unsqueeze(1))
                .detach()
                .cpu()
                .tolist()
            )
            # Isaac stores quaternions wxyz. The renderer contract is xyzw.
            quaternions = self.robot.data.body_quat_w.detach().cpu().tolist()
            for env_index, (env_positions, env_quaternions) in enumerate(
                zip(positions, quaternions, strict=True)
            ):
                row = [round(float(t), 3)]
                for position, quaternion in zip(
                    env_positions, env_quaternions, strict=True
                ):
                    px, py, pz = position
                    w, x, y, z = quaternion
                    row.extend(
                        (
                            round(float(px), 3),
                            round(float(py), 3),
                            round(float(pz), 3),
                            round(float(x), 4),
                            round(float(y), 4),
                            round(float(z), 4),
                            round(float(w), 4),
                        )
                    )
                self.frames[env_index].append(row)
        self._step += 1

    def payload(self, results: Sequence[Any], *, policy_path: str) -> dict[str, Any]:
        rows = [result_dict(result) for result in results]
        selected = representative_index(rows)
        result = rows[selected]
        return {
            "schema_version": SCHEMA_VERSION,
            "body_names": self.body_names,
            "fps": round(self.fps, 6),
            "runs": [result],
            "frames": [self.frames[selected]],
            "representative_lane": selected,
            "failure_modes": failure_modes(result),
            "policy_sha256": hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
        }


def atomic_write(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
