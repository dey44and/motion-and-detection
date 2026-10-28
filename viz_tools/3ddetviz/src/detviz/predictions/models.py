"""Immutable canonical detections shared by importers, filters, and renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from visual_utils import Box3D


class PredictionError(RuntimeError):
    """A prediction source is unavailable or inconsistent."""


class PredictionCancelled(PredictionError):
    """The caller cancelled an import before its cache was published."""


def immutable_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PredictionError(f"{name} must contain finite numbers with shape {shape}.") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise PredictionError(f"{name} must contain finite numbers with shape {shape}.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Detection:
    box: Box3D
    score: float
    identifier: str
    velocity: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box3D):
            raise PredictionError("Detection box must be a Box3D.")
        try:
            score = float(self.score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PredictionError("Detection score must be finite and in [0, 1].") from exc
        if isinstance(self.score, (bool, np.bool_)) or not np.isfinite(score) or not 0 <= score <= 1:
            raise PredictionError("Detection score must be finite and in [0, 1].")
        if not isinstance(self.identifier, str) or not self.identifier:
            raise PredictionError("Detection identifier must be a nonempty string.")
        object.__setattr__(self, "score", score)
        if self.velocity is not None:
            object.__setattr__(self, "velocity", immutable_array(self.velocity, (3,), "Detection velocity"))


@dataclass(frozen=True)
class PredictionFrame:
    sample_token: str
    detections: tuple[Detection, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_token, str) or not self.sample_token:
            raise PredictionError("Prediction sample token must be a nonempty string.")
        detections = tuple(self.detections)
        if any(not isinstance(detection, Detection) for detection in detections):
            raise PredictionError("Prediction frames must contain Detection instances.")
        if len({detection.identifier for detection in detections}) != len(detections):
            raise PredictionError("Prediction identifiers must be unique within a frame.")
        object.__setattr__(self, "detections", detections)


@dataclass(frozen=True)
class PredictionInfo:
    name: str
    format: str
    coordinate_frame: str
    stage: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_frame, str) or self.coordinate_frame not in {"world", "lidar"}:
            raise PredictionError("Prediction coordinate frame must be 'world' or 'lidar'.")
        if not isinstance(self.stage, str) or self.stage not in {"pre_nms", "post_nms", "unknown"}:
            raise PredictionError("Prediction stage must be 'pre_nms', 'post_nms', or 'unknown'.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
