"""Executable policy interface for the Sprint task.

Required TorchScript method::

    forward(observation: Tensor[N, 122]) -> Tensor[N, 37]

Stateful policies may additionally expose::

    reset(done_mask: Tensor[N, bool]) -> None

The verifier calls ``reset`` with a boolean mask for every parallel environment.
Stateless policies must omit the method rather than exporting a different
signature.
"""

from __future__ import annotations

from dataclasses import dataclass


OBSERVATION_DIM = 122
ACTION_DIM = 37
ACTION_SCALE = 0.5
CONTROL_FREQUENCY_HZ = 50
RESET_METHOD = "reset(done_mask: Tensor[N, bool]) -> None"
RESET_MASK_DESCRIPTION = "one boolean per parallel environment; true clears state"


@dataclass(frozen=True)
class ObservationField:
    name: str
    start: int
    stop: int
    description: str

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


OBSERVATION_FIELDS = (
    ObservationField("base_linear_velocity", 0, 3, "body frame"),
    ObservationField("base_angular_velocity", 3, 6, "body frame"),
    ObservationField("projected_gravity", 6, 9, "body frame"),
    ObservationField("joint_position", 9, 46, "relative to default pose"),
    ObservationField("joint_velocity", 46, 83, "joint-space velocity"),
    ObservationField("previous_action", 83, 120, "previous policy action"),
    ObservationField("cross_track_error", 120, 121, "metres; left is positive"),
    ObservationField("heading_error", 121, 122, "radians; zero faces finish"),
)
OBSERVATION_SLICES = {field.name: field.slice for field in OBSERVATION_FIELDS}


def validate() -> None:
    """Fail if the declared fields do not exactly tile the observation."""
    assert OBSERVATION_FIELDS[0].start == 0
    assert OBSERVATION_FIELDS[-1].stop == OBSERVATION_DIM
    assert all(
        left.stop == right.start
        for left, right in zip(OBSERVATION_FIELDS, OBSERVATION_FIELDS[1:])
    )


validate()
