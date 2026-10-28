"""Native Waymo values; transforms retain their original reference frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class NativeWaymoError(ValueError):
    """Native Waymo sensor data is missing, corrupt, or inconsistent."""


@dataclass(frozen=True)
class NativeSequence:
    context_name: str
    path: Path
    split: str


@dataclass(frozen=True)
class NativeFrameHeader:
    frame_index: int
    timestamp_us: int

    @property
    def index(self):
        return self.frame_index


@dataclass(frozen=True)
class NativeBox:
    center: np.ndarray
    size: np.ndarray
    heading: float
    class_name: str
    identifier: str
    most_visible_camera_name: str = ""
    camera_synced_box: NativeBox | None = None


@dataclass(frozen=True)
class NativeCamera:
    name: int
    image_rgb: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    vehicle_to_world: np.ndarray
    pose_timestamp_s: float
    velocity: np.ndarray
    shutter_s: float
    trigger_time_s: float
    readout_done_time_s: float
    rolling_shutter_direction: int


@dataclass(frozen=True)
class NativeLaserCalibration:
    name: int
    extrinsic: np.ndarray
    beam_inclinations: np.ndarray
    beam_inclination_min: float
    beam_inclination_max: float


@dataclass(frozen=True)
class NativeRangeImage:
    range_image_compressed: bytes
    camera_projection_compressed: bytes = b""
    range_image_pose_compressed: bytes = b""


@dataclass(frozen=True)
class NativeLaser:
    name: int
    returns: tuple[NativeRangeImage, NativeRangeImage]


@dataclass(frozen=True)
class NativeCameraLabels:
    name: int
    boxes: tuple[NativeBox, ...]


@dataclass(frozen=True)
class NativePoints:
    """Rows follow sorted laser ID, then return 0/1, then valid pixel order.

    ``points`` use the reference vehicle frame. ``raw_points`` use each beam's
    acquisition vehicle frame only where ``raw_valid_mask`` is true (TOP with
    per-pixel poses); other rows retain reference coordinates. Camera
    projections are the native per-return [camera,u,v,camera,u,v] associations.
    """

    points: np.ndarray
    intensity: np.ndarray
    elongation: np.ndarray
    nlz: np.ndarray
    lidar_id: np.ndarray
    return_id: np.ndarray
    camera_projections: np.ndarray
    raw_points: np.ndarray | None = None
    raw_valid_mask: np.ndarray | None = None


@dataclass(frozen=True)
class NativeFrame:
    context_name: str
    timestamp_us: int
    vehicle_to_world: np.ndarray
    cameras: tuple[NativeCamera, ...]
    boxes: tuple[NativeBox, ...]
    lasers: tuple[NativeLaser, ...] = ()
    laser_calibrations: tuple[NativeLaserCalibration, ...] = ()
    camera_labels: tuple[NativeCameraLabels, ...] = ()
    projected_lidar_labels: tuple[NativeCameraLabels, ...] = ()

    def points(self) -> NativePoints:
        from .range_image import decode_lidar

        return decode_lidar(self)
