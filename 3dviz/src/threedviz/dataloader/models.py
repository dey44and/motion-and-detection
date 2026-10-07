"""Dataset-independent data exchanged between adapters and the viewer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneInfo:
    token: str
    name: str
    description: str = ""
    frame_count: int = 0


@dataclass(frozen=True)
class FrameInfo:
    token: str
    scene_token: str
    index: int
    timestamp_us: int


@dataclass(frozen=True)
class BoundingBox3D:
    """A sensor-frame box, in metres, with local x/y/z = length/width/height.

    ``rotation`` maps local box coordinates to the point cloud's sensor frame.
    It is a full 3-by-3 rotation matrix, so pitch and roll are preserved.
    """

    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    class_name: str
    token: str = ""


@dataclass(frozen=True)
class FrameData:
    """One point cloud and its annotations in the same coordinate frame.

    Points have shape (N, 3) in the sweep reference sensor frame. Intensity,
    when supplied, has shape (N,) and preserves the original dataset values.

    Optional acquisition timestamps and uncompensated points retain the same
    point order. Each raw point is expressed in its own acquisition-time sensor
    frame; boxes remain in the sweep reference frame. RGB arrays have shape
    (N, 3) in [0, 1], with a boolean (N,) validity mask for camera coverage.
    """

    info: FrameInfo
    points: np.ndarray
    boxes: tuple[BoundingBox3D, ...]
    intensity: np.ndarray | None = None
    lidar_timestamp_us: int | None = None
    ring_indices: np.ndarray | None = None
    point_timestamps_us: np.ndarray | None = None
    raw_points: np.ndarray | None = None
    rgb_colors: np.ndarray | None = None
    rgb_valid_mask: np.ndarray | None = None
