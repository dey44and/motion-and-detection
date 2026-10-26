"""Color world-space LiDAR returns from timestamped, calibrated camera images.

The camera contract is an undistorted, global-shutter pinhole camera with optical
coordinates x right, y down, and z forward. No dataset-specific code is needed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


__all__ = ["CameraObservation", "project_camera_rgb"]


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric array with shape {shape}.") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite numeric array with shape {shape}.")
    return array


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


@dataclass(frozen=True)
class CameraObservation:
    """One image and its actual exposure pose, calibration, and timestamp.

    ``sensor_to_world`` maps camera optical coordinates into world coordinates.
    ``intrinsics`` is the image's 3x3 pinhole matrix; image dimensions must match
    the decoded file. Arrays are copied and read-only after construction.
    """

    channel: str
    timestamp_us: int
    image_path: Path
    intrinsics: np.ndarray
    sensor_to_world: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError("Camera channel must be a nonempty string.")
        timestamp = _integer(self.timestamp_us, "Camera timestamp_us")
        width = _integer(self.width, "Camera width", minimum=1)
        height = _integer(self.height, "Camera height", minimum=1)
        try:
            path = Path(self.image_path)
        except TypeError as exc:
            raise ValueError("Camera image_path must be a filesystem path.") from exc
        intrinsics = _finite_array(self.intrinsics, (3, 3), "Camera intrinsics")
        if not np.allclose(intrinsics[2], [0, 0, 1], rtol=0, atol=1e-9):
            raise ValueError("Camera intrinsics must have last row [0, 0, 1].")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError("Camera intrinsics must have positive focal lengths.")
        if not np.isclose(intrinsics[1, 0], 0, rtol=0, atol=1e-9):
            raise ValueError("Camera intrinsics must use the standard pinhole matrix form.")
        transform = _finite_array(self.sensor_to_world, (4, 4), "Camera sensor_to_world")
        if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-9):
            raise ValueError("Camera sensor_to_world must have last row [0, 0, 0, 1].")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6):
            raise ValueError("Camera sensor_to_world rotation must be orthonormal.")
        if not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-6):
            raise ValueError("Camera sensor_to_world must contain a proper rotation.")
        intrinsics.setflags(write=False)
        transform.setflags(write=False)
        object.__setattr__(self, "timestamp_us", timestamp)
        object.__setattr__(self, "image_path", path)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "sensor_to_world", transform)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)


def _load_rgb(path: Path) -> np.ndarray:
    # Pillow stays optional until an image is actually needed.
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _image_for_camera(
    camera: CameraObservation, loader: Callable[[Path], np.ndarray]
) -> np.ndarray:
    context = f"camera {camera.channel} at {camera.timestamp_us} us: {camera.image_path}"
    try:
        image = np.asarray(loader(camera.image_path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing image for {context}. Check that camera images are installed under the dataset root."
        ) from exc
    except OSError as exc:
        raise OSError(f"Cannot read image for {context}: {exc}") from exc
    expected_shape = (camera.height, camera.width, 3)
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise ValueError(
            f"Image for {context} must be uint8 RGB with shape {expected_shape}; "
            f"received {image.dtype} with shape {image.shape}."
        )
    return image


def _bilinear_rgb(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    x0 = np.floor(uv[:, 0]).astype(np.int64)
    y0 = np.floor(uv[:, 1]).astype(np.int64)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    wx = (uv[:, 0] - x0)[:, None]
    wy = (uv[:, 1] - y0)[:, None]
    top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
    return (top * (1 - wy) + bottom * wy) / 255.0


def project_camera_rgb(
    points_world: np.ndarray,
    point_timestamps_us: np.ndarray,
    cameras: Sequence[CameraObservation],
    *,
    image_loader: Callable[[Path], np.ndarray] | None = None,
    max_time_delta_us: int = 100_000,
    occlusion_tolerance_m: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Project each return into its nearest image per camera channel.

    Image timestamps need not be keyframes. Equal timestamp distances choose
    the earlier image. The camera pose is taken at the selected image exposure;
    the caller supplies world points obtained from its acquisition-time poses.
    Images beyond ``max_time_delta_us`` are ignored.

    Visibility uses a sparse depth buffer with each projection rounded to its
    nearest pixel. A point may be at most ``occlusion_tolerance_m`` behind that
    pixel's closest return (camera z depth, meters). All temporally eligible
    cloud points contribute to this buffer, even when their nearest image is a
    different one. This approximates occlusion from the available LiDAR; it
    cannot detect surfaces absent from the cloud or correct moving objects.

    Among visible overlapping cameras, choose the greatest cosine to the
    optical axis, then the smallest time gap, then channel/timestamp order.
    Colors use bilinear sampling and are normalized RGB. Uncolored points have
    RGB [0.5, 0.5, 0.5] and a false validity mask. Files are loaded only when
    they can color at least one point; loaders may cache decoded images.
    """
    try:
        points = np.asarray(points_world, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("points_world must be a finite numeric array with shape (N, 3).") from exc
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_world must be a finite numeric array with shape (N, 3).")
    times = _finite_array(point_timestamps_us, (len(points),), "point_timestamps_us")
    if np.any(times < 0):
        raise ValueError("point_timestamps_us must be nonnegative.")
    max_gap = _integer(max_time_delta_us, "max_time_delta_us")
    try:
        tolerance = float(occlusion_tolerance_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("occlusion_tolerance_m must be finite and nonnegative.") from exc
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("occlusion_tolerance_m must be finite and nonnegative.")
    channels: dict[str, list[CameraObservation]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for camera in cameras:
        if not isinstance(camera, CameraObservation):
            raise ValueError("cameras must contain CameraObservation instances.")
        key = (camera.channel, camera.timestamp_us)
        if key in seen:
            raise ValueError(f"Duplicate camera observation for {camera.channel} at {camera.timestamp_us} us.")
        seen.add(key)
        channels[camera.channel].append(camera)

    colors = np.full((len(points), 3), 0.5, dtype=np.float64)
    valid = np.zeros(len(points), dtype=bool)
    if not len(points):
        return colors, valid
    best_cosine = np.full(len(points), -np.inf)
    best_gap = np.full(len(points), np.inf)
    loader = _load_rgb if image_loader is None else image_loader
    for channel in sorted(channels):
        observations = sorted(channels[channel], key=lambda camera: camera.timestamp_us)
        camera_times = np.array([camera.timestamp_us for camera in observations], dtype=np.float64)
        following = np.clip(np.searchsorted(camera_times, times), 0, len(observations) - 1)
        preceding = np.maximum(following - 1, 0)
        choose_following = np.abs(camera_times[following] - times) < np.abs(camera_times[preceding] - times)
        nearest = np.where(choose_following, following, preceding)
        for camera_index in np.unique(nearest):
            camera = observations[int(camera_index)]
            gaps = np.abs(times - camera.timestamp_us)
            eligible = gaps <= max_gap
            if not np.any(eligible & (nearest == camera_index)):
                continue
            # Invert the rigid world pose without a general matrix inversion.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                optical = (points - camera.sensor_to_world[:3, 3]) @ camera.sensor_to_world[:3, :3]
                projected = optical @ camera.intrinsics.T
                uv = projected[:, :2] / projected[:, 2, None]
            in_image = (
                eligible & (optical[:, 2] > 0) & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0) & (uv[:, 0] <= camera.width - 1)
                & (uv[:, 1] >= 0) & (uv[:, 1] <= camera.height - 1)
            )
            visible_indices = np.flatnonzero(in_image)
            if not len(visible_indices) or not np.any(nearest[visible_indices] == camera_index):
                continue
            pixels = np.floor(uv[visible_indices] + 0.5).astype(np.int64)
            pixel_ids = pixels[:, 1] * camera.width + pixels[:, 0]
            _, pixel_inverse = np.unique(pixel_ids, return_inverse=True)
            min_depth = np.full(int(pixel_inverse.max()) + 1, np.inf)
            np.minimum.at(min_depth, pixel_inverse, optical[visible_indices, 2])
            unoccluded = optical[visible_indices, 2] <= min_depth[pixel_inverse] + tolerance
            indices = visible_indices[unoccluded & (nearest[visible_indices] == camera_index)]
            if not len(indices):
                continue
            # hypot avoids overflow in the norm of large but finite coordinates.
            norm = np.hypot(np.hypot(optical[indices, 0], optical[indices, 1]), optical[indices, 2])
            cosine = optical[indices, 2] / norm
            better_angle = cosine > best_cosine[indices] + 1e-12
            tied_angle = np.abs(cosine - best_cosine[indices]) <= 1e-12
            improves = better_angle | (tied_angle & (gaps[indices] < best_gap[indices]))
            indices = indices[improves]
            if not len(indices):
                continue
            image = _image_for_camera(camera, loader)
            colors[indices] = _bilinear_rgb(image, uv[indices])
            best_cosine[indices] = cosine[improves]
            best_gap[indices] = gaps[indices]
            valid[indices] = True
    return colors, valid
