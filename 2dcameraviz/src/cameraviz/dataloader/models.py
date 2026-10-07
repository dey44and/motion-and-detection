"""Dataset-neutral camera images and world-coordinate annotations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string.")


def _integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and finite values.")
    result.setflags(write=False)
    return result


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    result = _array(value, (3, 3), name)
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(result), 1, atol=1e-6):
        raise ValueError(f"{name} must be a proper rotation matrix.")
    return result


@dataclass(frozen=True)
class SceneInfo:
    token: str
    name: str
    description: str = ""
    frame_count: int = 0

    def __post_init__(self) -> None:
        _text(self.token, "Scene token")
        _text(self.name, "Scene name")
        _integer(self.frame_count, "Frame count")


@dataclass(frozen=True)
class FrameInfo:
    token: str
    scene_token: str
    index: int
    timestamp_us: int

    def __post_init__(self) -> None:
        _text(self.token, "Frame token")
        _text(self.scene_token, "Scene token")
        _integer(self.index, "Frame index")
        _integer(self.timestamp_us, "Frame timestamp")


@dataclass(frozen=True)
class CameraInfo:
    channel: str
    label: str
    row: int
    column: int

    def __post_init__(self) -> None:
        _text(self.channel, "Camera channel")
        _text(self.label, "Camera label")
        _integer(self.row, "Camera row")
        _integer(self.column, "Camera column")


@dataclass(frozen=True)
class BoundingBox3D:
    """World box in metres; local axes are length, width, height.

    ``rotation`` maps local box coordinates into the world coordinate system.
    Full orientation, including pitch and roll, is retained.
    """

    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    class_name: str
    token: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _array(self.center, (3,), "Box center"))
        size = _array(self.size, (3,), "Box size")
        if np.any(size <= 0):
            raise ValueError("Box dimensions must be positive.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "rotation", _rotation(self.rotation, "Box rotation"))
        _text(self.class_name, "Box class")


@dataclass(frozen=True)
class CameraFrame:
    """RGB image with a calibrated optical camera pose at image exposure time.

    Optical coordinates are x right, y down and z forward. ``sensor_to_world``
    maps those coordinates into the same world frame as the annotations.
    """

    info: CameraInfo
    image: np.ndarray
    intrinsics: np.ndarray
    sensor_to_world: np.ndarray
    timestamp_us: int

    def __post_init__(self) -> None:
        if not isinstance(self.info, CameraInfo):
            raise ValueError("Camera info must be a CameraInfo.")
        image = np.asarray(self.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
            raise ValueError("Camera image must be a nonempty uint8 RGB array with shape (H, W, 3).")
        # Decoded images in the adapter cache are already immutable. Preserve
        # them without copying several megabytes whenever a frame is revisited.
        if image.flags.writeable or not image.flags.c_contiguous:
            image = np.array(image, copy=True, order="C")
        image.setflags(write=False)
        object.__setattr__(self, "image", image)
        intrinsics = _array(self.intrinsics, (3, 3), "Camera intrinsics")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-9):
            raise ValueError("Camera intrinsics must have positive focal lengths and last row [0, 0, 1].")
        object.__setattr__(self, "intrinsics", intrinsics)
        transform = _array(self.sensor_to_world, (4, 4), "Camera pose")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("Camera pose must have homogeneous last row [0, 0, 0, 1].")
        _rotation(transform[:3, :3], "Camera pose rotation")
        object.__setattr__(self, "sensor_to_world", transform)
        _integer(self.timestamp_us, "Camera timestamp")


@dataclass(frozen=True)
class FrameData:
    info: FrameInfo
    cameras: tuple[CameraFrame, ...]
    boxes: tuple[BoundingBox3D, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.info, FrameInfo):
            raise ValueError("Frame info must be a FrameInfo.")
        cameras = tuple(self.cameras)
        boxes = tuple(self.boxes)
        if any(not isinstance(camera, CameraFrame) for camera in cameras):
            raise ValueError("Frame cameras must contain CameraFrame objects.")
        if any(not isinstance(box, BoundingBox3D) for box in boxes):
            raise ValueError("Frame boxes must contain BoundingBox3D objects.")
        channels = [camera.info.channel for camera in cameras]
        slots = [(camera.info.row, camera.info.column) for camera in cameras]
        if len(set(channels)) != len(channels) or len(set(slots)) != len(slots):
            raise ValueError("Frame cameras must have unique channels and grid positions.")
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "boxes", boxes)
