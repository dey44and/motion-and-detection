"""Oriented bounding-box wireframes in the point cloud coordinate system."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class _Box(Protocol):
    """Minimal geometry contract, independent of a concrete dataset model."""

    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray


# Corner indices trace the lower face (0..3), upper face (4..7), then uprights.
BOX_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)
BOX_EDGES.setflags(write=False)

_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=np.float64,
)


def box_corners(box: _Box) -> np.ndarray:
    """Return eight oriented corners in sensor coordinates.

    ``size`` is ``(length, width, height)`` along the local x, y, z axes.
    ``rotation`` is a proper 3x3 rotation from box-local to sensor coordinates,
    and ``center`` is in sensor coordinates. Lower-face corners are indices
    0..3; upper-face corners 4..7 have the same x/y order. ``BOX_EDGES`` connects
    these corners into a wireframe. Invalid geometry raises ``ValueError``.
    """
    center = np.asarray(box.center, dtype=np.float64)
    size = np.asarray(box.size, dtype=np.float64)
    rotation = np.asarray(box.rotation, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("Box center must have shape (3,).")
    if size.shape != (3,):
        raise ValueError("Box size must have shape (3,) in length, width, height order.")
    if rotation.shape != (3, 3):
        raise ValueError("Box rotation must have shape (3, 3).")
    if not all(np.isfinite(values).all() for values in (center, size, rotation)):
        raise ValueError("Box geometry must contain only finite values.")
    if np.any(size <= 0):
        raise ValueError("Box dimensions must be positive.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=1e-5):
        raise ValueError("Box rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("Box rotation must be a proper rotation with determinant +1.")

    local_corners = _CORNER_SIGNS * (size / 2.0)
    return local_corners @ rotation.T + center
