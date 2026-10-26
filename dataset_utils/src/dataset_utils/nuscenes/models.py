"""nuScenes camera records independent of any visualization application."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import BoundingBox3D, FrameInfo


@dataclass(frozen=True)
class NuScenesCamera:
    """RGB image and its optical-camera pose at the actual exposure time."""

    channel: str
    image: np.ndarray
    intrinsics: np.ndarray
    sensor_to_world: np.ndarray
    timestamp_us: int


@dataclass(frozen=True)
class NuScenesCameraFrame:
    """Keyframe camera observations with sample-time world-coordinate boxes."""

    info: FrameInfo
    cameras: tuple[NuScenesCamera, ...]
    boxes: tuple[BoundingBox3D, ...]
