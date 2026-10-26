"""Reusable normalized pinhole projection for rectified camera observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ProjectionResult, _array, _transform
from ..errors import DatasetValidationError


@dataclass(frozen=True, eq=False)
class PinholeProjection:
    intrinsics: np.ndarray
    T_reference_from_camera: np.ndarray

    def __post_init__(self):
        matrix = _array(self.intrinsics, (3, 3), "intrinsics", dtype=np.float64)
        if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-9)):
            raise DatasetValidationError("Pinhole intrinsics must have positive focal lengths and final row [0,0,1].")
        object.__setattr__(self, "intrinsics", matrix)
        object.__setattr__(self, "T_reference_from_camera",
                           _transform(self.T_reference_from_camera, "T_reference_from_camera"))

    def project_reference(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise DatasetValidationError("Projection input must be finite (N,3) reference points.")
        pose = self.T_reference_from_camera
        camera = (points - pose[:3, 3]) @ pose[:3, :3]
        depth = camera[:, 2]
        uv = np.full((len(points), 2), np.nan)
        forward = depth > 0
        projected = camera[forward] @ self.intrinsics.T
        uv[forward] = projected[:, :2] / projected[:, 2, None]
        valid = forward & np.isfinite(uv).all(axis=1)
        return ProjectionResult(uv, depth, valid)
