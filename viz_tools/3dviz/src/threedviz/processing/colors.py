"""Colors for point clouds and annotation classes, without a renderer dependency."""

from __future__ import annotations

import colorsys
import hashlib

import numpy as np
from matplotlib import colormaps

COLOR_SCHEMES = ("gray", "jet", "turbo")

# A fixed palette for the nuScenes detection classes. Saturated, moderately dark
# colors keep thin box edges visible against the viewer's white background.
_CLASS_COLORS = {
    "car": (0.05, 0.35, 0.75),
    "truck": (0.75, 0.25, 0.05),
    "bus": (0.55, 0.20, 0.70),
    "trailer": (0.55, 0.35, 0.15),
    "construction_vehicle": (0.65, 0.45, 0.00),
    "pedestrian": (0.85, 0.10, 0.20),
    "motorcycle": (0.00, 0.50, 0.40),
    "bicycle": (0.10, 0.55, 0.10),
    "traffic_cone": (0.90, 0.35, 0.00),
    "barrier": (0.30, 0.35, 0.40),
}


def colorize_points(
    points: np.ndarray, scheme: str = "gray", max_distance: float = 80.0
) -> np.ndarray:
    """Return one RGB color per point, using its Euclidean sensor distance.

    ``points`` must be finite with shape ``(N, 3)``. Distance is normalized to
    ``max_distance`` and clipped there, so colors remain stable between frames.
    Gray increases from 0.12 to 0.72 instead of reaching white; jet and turbo use
    Matplotlib's colormaps over the full normalized distance range. Empty point
    clouds return an empty ``(0, 3)`` array. Invalid inputs raise ``ValueError``.
    """
    if scheme not in COLOR_SCHEMES:
        raise ValueError(f"Unknown color scheme {scheme!r}; choose from {COLOR_SCHEMES}.")
    try:
        distance_limit = float(max_distance)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_distance must be a positive finite number.") from exc
    if not np.isfinite(distance_limit) or distance_limit <= 0:
        raise ValueError("max_distance must be a positive finite number.")

    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if not np.isfinite(coordinates).all():
        raise ValueError("points must contain only finite coordinates.")

    # Capping each component first avoids overflow for very large finite input.
    # If any component reaches the limit, its distance color is already clipped.
    scaled = np.minimum(np.abs(coordinates), distance_limit) / distance_limit
    distances = np.minimum(np.linalg.norm(scaled, axis=1), 1.0)
    if scheme == "gray":
        gray = 0.12 + 0.60 * distances
        return np.repeat(gray[:, None], 3, axis=1)
    return np.asarray(colormaps[scheme](distances)[:, :3], dtype=np.float64)


def class_color(class_name: str) -> tuple[float, float, float]:
    """Return a repeatable RGB color for any dataset class name.

    Unknown classes derive a hue from SHA-256, so the same name produces the
    same color across runs, machines, and Python hash seeds. Class names are
    treated literally, allowing dataset adapters to choose their own taxonomy.
    """
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("class_name must be a nonempty string.")
    if class_name in _CLASS_COLORS:
        return _CLASS_COLORS[class_name]
    digest = hashlib.sha256(class_name.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / (2**32)
    return colorsys.hsv_to_rgb(hue, 0.75, 0.70)
