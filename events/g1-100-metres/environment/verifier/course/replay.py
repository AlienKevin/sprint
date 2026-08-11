"""Compact, trusted pose replays produced during a course rollout."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 2
DEFAULT_FPS = 50.0


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
        self._times: list[float] = []
        self._positions: list[Any] = []
        self._quaternions: list[Any] = []
        self._step = 0

    def __call__(
        self, t: float, _trace: dict[str, list], _resolved: list[bool]
    ) -> None:
        if self._step % self.sample_every == 0:
            # Keep samples on-device during the rollout. At 50 Hz, forcing a
            # GPU-to-host synchronization here would add one stall per policy
            # step; payload() transfers only the selected representative lane.
            self._times.append(round(float(t), 3))
            self._positions.append(
                (self.robot.data.body_pos_w - self.origins.unsqueeze(1))
                .detach()
                .clone()
            )
            self._quaternions.append(
                self.robot.data.body_quat_w.detach().clone()
            )
        self._step += 1

    def payload(self, results: Sequence[Any], *, policy_path: str) -> dict[str, Any]:
        rows = [result_dict(result) for result in results]
        selected = representative_index(rows)
        result = rows[selected]
        frames: list[list[float]] = []
        if self._times:
            import torch

            positions = torch.stack(
                [sample[selected] for sample in self._positions], dim=0
            ).cpu().tolist()
            # Isaac stores quaternions wxyz. The renderer contract is xyzw.
            quaternions = torch.stack(
                [sample[selected] for sample in self._quaternions], dim=0
            ).cpu().tolist()
            for t, frame_positions, frame_quaternions in zip(
                self._times, positions, quaternions, strict=True
            ):
                row = [t]
                for position, quaternion in zip(
                    frame_positions, frame_quaternions, strict=True
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
                frames.append(row)
        return {
            "schema_version": SCHEMA_VERSION,
            "lane_gate": {
                "semantics": "whole_body_collision_envelope",
                "half_width_m": 0.61,
                "collision_samples": 1960,
            },
            "body_names": self.body_names,
            "fps": round(self.fps, 6),
            "runs": [result],
            "frames": [frames],
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
