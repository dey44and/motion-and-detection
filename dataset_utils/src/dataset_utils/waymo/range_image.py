# Portions based on the Waymo range-image geometry, Copyright 2019 The Waymo
# Open Dataset Authors, licensed under the Apache License, Version 2.0.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0 .
# Unless required by applicable law or agreed to in writing, software under
# that license is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""NumPy range decoding and native return-specific camera RGB associations.

The polar/extrinsic/per-pixel-pose equations follow the Apache-2.0 Waymo
range_image_utils implementation (v1.6.1), also documented in NVIDIA NCore's
tools/data_converter/waymo/waymo_range_image_utils.py. This independent NumPy
implementation retains both returns and their own camera projections.

https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/utils/range_image_utils.py
"""

from __future__ import annotations

import numpy as np

from .models import NativePoints, NativeWaymoError
from .proto import decode_matrix


def rotation_rpy(roll, pitch, yaw):
    """Vehicle-to-world ZYX Euler rotation for scalar or array angles."""
    roll, pitch, yaw = np.broadcast_arrays(roll, pitch, yaw)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    matrix = np.empty(roll.shape + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = cy * cp
    matrix[..., 0, 1] = cy * sp * sr - sy * cr
    matrix[..., 0, 2] = cy * sp * cr + sy * sr
    matrix[..., 1, 0] = sy * cp
    matrix[..., 1, 1] = sy * sp * sr + cy * cr
    matrix[..., 1, 2] = sy * sp * cr - cy * sr
    matrix[..., 2, 0] = -sp
    matrix[..., 2, 1] = cp * sr
    matrix[..., 2, 2] = cp * cr
    return matrix


def range_to_vehicle(ranges, calibration):
    """Compute acquisition vehicle coordinates before TOP ego compensation."""
    height, width = ranges.shape
    if len(calibration.beam_inclinations):
        if len(calibration.beam_inclinations) != height:
            raise NativeWaymoError("Laser beam inclination count does not match its range image height.")
        inclination = calibration.beam_inclinations
    else:
        inclination = (np.arange(height, dtype=np.float64) + 0.5) / height
        inclination = inclination * (calibration.beam_inclination_max - calibration.beam_inclination_min) + calibration.beam_inclination_min
    inclination = np.asarray(inclination, dtype=np.float64)[::-1]
    extrinsic = calibration.extrinsic
    correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratio = (np.arange(width, 0, -1, dtype=np.float64) - 0.5) / width
    azimuth = (ratio * 2 - 1) * np.pi - correction
    cos_inclination = np.cos(inclination)[:, None]
    points = np.stack((np.cos(azimuth)[None, :] * cos_inclination * ranges,
                       np.sin(azimuth)[None, :] * cos_inclination * ranges,
                       np.broadcast_to(np.sin(inclination)[:, None], ranges.shape) * ranges), axis=-1)
    return points @ extrinsic[:3, :3].T + extrinsic[:3, 3]


def decode_lidar(frame) -> NativePoints:
    """Decode sorted lasers, return 0 then 1, retaining range>0 pixels only."""
    calibrations = {calibration.name: calibration for calibration in frame.laser_calibrations}
    arrays = [[] for _ in range(9)]
    for laser in sorted(frame.lasers, key=lambda item: item.name):
        calibration = calibrations.get(laser.name)
        if calibration is None:
            raise NativeWaymoError(f"Laser {laser.name} has no calibration.")
        top_pose = None
        if laser.name == 1:
            compressed_pose = next((item.range_image_pose_compressed for item in laser.returns if item.range_image_pose_compressed), b"")
            if compressed_pose:
                top_pose = decode_matrix(compressed_pose)
                if top_pose.ndim != 3 or top_pose.shape[2] != 6:
                    raise NativeWaymoError("TOP per-pixel vehicle poses must have shape (H,W,6).")
        for return_index, item in enumerate(laser.returns):
            if not item.range_image_compressed:
                continue
            ranges = decode_matrix(item.range_image_compressed)
            if ranges.ndim != 3 or ranges.shape[2] != 4:
                raise NativeWaymoError("Native range images must have shape (H,W,4).")
            valid = ranges[..., 0] > 0
            raw = range_to_vehicle(ranges[..., 0], calibration)[valid]
            points = raw
            has_raw = laser.name == 1 and top_pose is not None
            if has_raw:
                if top_pose.shape[:2] != ranges.shape[:2]:
                    raise NativeWaymoError("TOP pixel pose shape does not match the corresponding range image.")
                poses = top_pose[valid].astype(np.float64)
                rotations = rotation_rpy(poses[:, 0], poses[:, 1], poses[:, 2])
                world = np.einsum("nij,nj->ni", rotations, raw) + poses[:, 3:6]
                points = (world - frame.vehicle_to_world[:3, 3]) @ frame.vehicle_to_world[:3, :3]
            projections = np.zeros((len(raw), 6), dtype=np.int32)
            if item.camera_projection_compressed:
                projection_image = decode_matrix(item.camera_projection_compressed, integer=True)
                if projection_image.shape != ranges.shape[:2] + (6,):
                    raise NativeWaymoError("Camera projection shape does not match its corresponding lidar return.")
                projections = projection_image[valid]
                if np.any((projections[:, (0, 3)] < 0) | (projections[:, (0, 3)] > 5)):
                    raise NativeWaymoError("Camera projection IDs must be official IDs 0–5.")
            values = (points, ranges[..., 1][valid], ranges[..., 2][valid], ranges[..., 3][valid],
                      np.full(len(raw), laser.name, dtype=np.uint8), np.full(len(raw), return_index, dtype=np.uint8),
                      projections, raw, np.full(len(raw), has_raw, dtype=bool))
            for target, value in zip(arrays, values):
                target.append(value)
    if not any(arrays):
        return NativePoints(np.empty((0, 3), dtype=np.float32), *[np.empty(0, dtype=np.float32) for _ in range(3)],
            np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8), np.empty((0, 6), dtype=np.int32),
            np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool))
    joined = [np.concatenate(parts) for parts in arrays]
    if not np.isfinite(joined[0]).all():
        raise NativeWaymoError("Decoded native points contain non-finite coordinates.")
    with np.errstate(over="ignore", invalid="ignore"):
        joined[0] = np.ascontiguousarray(joined[0], dtype=np.float32)
        joined[7] = np.ascontiguousarray(joined[7], dtype=np.float32)
    if not np.isfinite(joined[0]).all() or not np.isfinite(joined[7]).all():
        raise NativeWaymoError("Decoded reference or raw points exceed the finite float32 coordinate range.")
    return NativePoints(*joined)


def colorize_points(frame, pointset):
    """Sample JPEG RGB from each return's native, rolling-shutter-aware links.

    Prefer the first valid association in official camera priority order, using
    the second when the first camera/pixel is unavailable. These are dataset
    correspondences, not a new pinhole approximation. They do not establish
    complete visibility or correct object motion. Unmatched points remain gray.
    """
    projections = np.asarray(pointset.camera_projections)
    if projections.shape != (len(pointset.points), 6) or projections.dtype.kind not in "iu":
        raise NativeWaymoError("Point camera projections must be an integer array with shape (N,6).")
    cameras = {camera.name: camera.image_rgb for camera in frame.cameras}
    if not cameras:
        raise NativeWaymoError("Native Frame contains no camera images for Camera RGB.")
    colors = np.full((len(pointset.points), 3), .5, dtype=np.float64)
    valid = np.zeros(len(colors), dtype=bool)
    for offset in (0, 3):
        for name, image in cameras.items():
            selected = np.flatnonzero(~valid & (projections[:, offset] == name))
            if not len(selected):
                continue
            u, v = projections[selected, offset + 1], projections[selected, offset + 2]
            inside = (u >= 0) & (v >= 0) & (u < image.shape[1]) & (v < image.shape[0])
            selected, u, v = selected[inside], u[inside], v[inside]
            colors[selected] = image[v, u, :3].astype(np.float64) / 255
            valid[selected] = True
    return colors, valid
