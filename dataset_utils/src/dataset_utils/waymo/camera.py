# Reference camera equations: Copyright (c) 2019 Waymo LLC.
# Upstream BSD terms and attribution are bundled in third_party/.
"""Native Waymo lens distortion and rolling-shutter camera projection.

This NumPy implementation follows the reference camera model's first-order
pose/velocity model, lever arm and normalized-coordinate shutter equation:
https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/wdl_limited/camera/camera_model.cc

World points are stationary during exposure. Camera-synchronized annotation
centers can be supplied by callers when available; object motion is otherwise
not inferred. Neither TensorFlow nor detector-framework imports are required.
"""

from __future__ import annotations

import numpy as np


# Waymo camera (forward, left, up) <- conventional optical (right, down, forward).
OPTICAL_TO_WAYMO = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])
OPTICAL_TO_WAYMO.setflags(write=False)


def _skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _array(value, shape, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result.setflags(write=False)
    return result


class WaymoCameraProjection:
    """Project world XYZ into an original, distorted Waymo camera JPEG.

    ``project_world`` returns native image coordinates and camera-forward
    depth. Behind-camera or failed rolling-shutter projections have NaN image
    coordinates. Points outside the image remain available for line clipping.
    Excessive radial distortion receives the reference model's outside-image
    fallback; those coordinates must not be interpreted as valid image pixels.
    """

    def __init__(self, camera):
        self.intrinsic = _array(camera.intrinsic, (9,), "Waymo camera intrinsic")
        if np.any(self.intrinsic[:2] <= 0):
            raise ValueError("Camera focal lengths must be positive.")
        self.extrinsic = _array(camera.extrinsic, (4, 4), "Waymo camera extrinsic")
        self.vehicle_to_world = _array(camera.vehicle_to_world, (4, 4), "Camera vehicle pose")
        for matrix in (self.extrinsic, self.vehicle_to_world):
            if (not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
                    or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0, atol=1e-6)
                    or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, rtol=0, atol=1e-6)):
                raise ValueError("Camera poses must be proper rigid transforms.")
        self.height, self.width = np.asarray(camera.image_rgb).shape[:2]
        if self.width < 1 or self.height < 1:
            raise ValueError("Camera image dimensions must be positive.")
        velocity = _array(camera.velocity, (6,), "Camera vehicle velocity")
        self.direction = int(camera.rolling_shutter_direction)
        if self.direction not in (1, 2, 3, 4, 5):
            raise ValueError("Waymo rolling shutter direction must be in [1, 5].")
        times = np.asarray([camera.pose_timestamp_s, camera.shutter_s, camera.trigger_time_s, camera.readout_done_time_s], dtype=np.float64)
        if not np.isfinite(times).all() or camera.shutter_s < 0:
            raise ValueError("Camera timing values must be finite with nonnegative exposure.")
        pose = self.vehicle_to_world @ self.extrinsic
        self._camera_position = pose[:3, 3]
        self._world_to_camera = pose[:3, :3].T
        body_omega = velocity[3:]
        world_omega = self.vehicle_to_world[:3, :3] @ body_omega
        camera_omega = self.extrinsic[:3, :3].T @ body_omega
        self._camera_velocity = velocity[:3] + np.cross(
            world_omega, self.vehicle_to_world[:3, :3] @ self.extrinsic[:3, 3]
        )
        self._rotation_derivative = -_skew(camera_omega) @ self._world_to_camera
        self._time_factor = 0.
        self._time_offset = 0.
        self._horizontal = self.direction in (2, 4)
        if self.direction != 5:
            readout = float(camera.readout_done_time_s - camera.trigger_time_s - camera.shutter_s)
            if readout < -1e-6:
                raise ValueError("Camera readout end must follow exposure and trigger.")
            readout = max(0., readout)
            fu, fv, cu, cv = self.intrinsic[:4]
            if self._horizontal:
                first = self._undistort(np.array([[0., self.height / 2]]))[0, 0]
                last = self._undistort(np.array([[self.width, self.height / 2]]))[0, 0]
                principal_fraction = cu / self.width
            else:
                first = self._undistort(np.array([[self.width / 2, 0.]]))[0, 1]
                last = self._undistort(np.array([[self.width / 2, self.height]]))[0, 1]
                principal_fraction = cv / self.height
            reverse = self.direction in (3, 4)
            if reverse:
                principal_fraction = 1 - principal_fraction
            normalized_range = last - first
            if not np.isfinite(normalized_range) or normalized_range <= 0:
                raise ValueError("Camera distortion does not define a usable shutter interval.")
            self._time_factor = (-1 if reverse else 1) * readout / normalized_range
            # Keep subtraction of epoch timestamps ahead of small offsets.
            self._time_offset = (
                float(camera.pose_timestamp_s - camera.trigger_time_s)
                - .5 * float(camera.shutter_s) - readout * principal_fraction
            )

    def _undistort(self, pixels):
        target = (pixels - self.intrinsic[2:4]) / self.intrinsic[:2]
        normalized = target.copy()
        k1, k2, p1, p2, k3 = self.intrinsic[4:]
        for _ in range(50):
            u, v = normalized.T
            r2 = u * u + v * v
            radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            tangential = np.column_stack((2 * p1 * u * v + p2 * (r2 + 2 * u * u), p1 * (r2 + 2 * v * v) + 2 * p2 * u * v))
            following = (target - tangential) / radial[:, None]
            if np.max(np.abs(following - normalized)) < 1e-12:
                return following
            normalized = following
        return normalized

    def _distort(self, normalized):
        u, v = normalized.T
        r2 = u * u + v * v
        k1, k2, p1, p2, k3 = self.intrinsic[4:]
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        distorted = np.column_stack((
            u * radial + 2 * p1 * u * v + p2 * (r2 + 2 * u * u),
            v * radial + p1 * (r2 + 2 * v * v) + 2 * p2 * u * v,
        ))
        pixels = distorted * self.intrinsic[:2] + self.intrinsic[2:4]
        excessive = (radial < .8) | (radial > 1.2)
        if np.any(excessive):
            directions = normalized[excessive]
            pixels[excessive] = (
                directions / np.linalg.norm(directions, axis=1)[:, None]
                * np.hypot(self.width, self.height) + self.intrinsic[2:4]
            )
        return pixels

    def project_world(self, points_world):
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("World points must be finite with shape (N, 3).")
        if not len(points):
            return np.empty((0, 2)), np.empty(0)
        relative = points - self._camera_position
        camera = relative @ self._world_to_camera.T
        depth = camera[:, 0].copy()
        valid = depth > 0
        normalized = np.zeros((len(points), 2))
        if self.direction == 5:
            normalized[valid] = -camera[valid, 1:3] / depth[valid, None]
        else:
            elapsed = np.zeros(len(points))
            coordinate = 0 if self._horizontal else 1
            for _ in range(8):
                matrices = self._world_to_camera + elapsed[:, None, None] * self._rotation_derivative
                offset = relative - elapsed[:, None] * self._camera_velocity
                camera = np.einsum("nij,nj->ni", matrices, offset)
                depth = camera[:, 0]
                valid &= np.isfinite(camera).all(axis=1) & (depth > 0)
                safe_depth = np.where(valid, depth, 1.)
                normalized = -camera[:, 1:3] / safe_depth[:, None]
                residual = elapsed - normalized[:, coordinate] * self._time_factor + self._time_offset
                derivative = offset @ self._rotation_derivative.T - np.einsum("nij,j->ni", matrices, self._camera_velocity)
                normalized_derivative = -(derivative[:, coordinate + 1] + normalized[:, coordinate] * derivative[:, 0]) / safe_depth
                jacobian = 1 - self._time_factor * normalized_derivative
                active = valid & (np.abs(residual) > 1e-9)
                valid &= np.isfinite(jacobian) & (np.abs(jacobian) > 1e-12)
                if not np.any(active & valid):
                    break
                elapsed[active & valid] -= residual[active & valid] / jacobian[active & valid]
            matrices = self._world_to_camera + elapsed[:, None, None] * self._rotation_derivative
            camera = np.einsum("nij,nj->ni", matrices, relative - elapsed[:, None] * self._camera_velocity)
            depth = camera[:, 0].copy()
            valid &= (depth > 0) & np.isfinite(camera).all(axis=1)
            normalized = -camera[:, 1:3] / np.where(valid, depth, 1.)[:, None]
            residual = elapsed - normalized[:, coordinate] * self._time_factor + self._time_offset
            valid &= np.isfinite(residual) & (np.abs(residual) < .1)
        pixels = np.full((len(points), 2), np.nan)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pixels[valid] = self._distort(normalized[valid])
        failed = ~np.isfinite(pixels).all(axis=1)
        depth[failed & (depth > 0)] = np.nan
        return pixels, depth


__all__ = ["OPTICAL_TO_WAYMO", "WaymoCameraProjection"]
