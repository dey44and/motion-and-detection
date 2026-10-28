"""Stable annotation colors shared with the 3D viewer's palette."""

from __future__ import annotations

import colorsys
import hashlib


_CLASS_COLORS = {
    "car": (0.05, 0.35, 0.75),
    "truck": (0.75, 0.25, 0.05),
    "construction_vehicle": (0.65, 0.45, 0.00),
    "bus": (0.55, 0.20, 0.70),
    "trailer": (0.55, 0.35, 0.15),
    "barrier": (0.30, 0.35, 0.40),
    "motorcycle": (0.00, 0.50, 0.40),
    "bicycle": (0.10, 0.55, 0.10),
    "pedestrian": (0.85, 0.10, 0.20),
    "traffic_cone": (0.90, 0.35, 0.00),
}


def class_color(class_name: str) -> str:
    """Return a reproducible ``#rrggbb`` color without importing a renderer."""

    if not isinstance(class_name, str) or not class_name:
        raise ValueError("class_name must be a nonempty string.")
    rgb = _CLASS_COLORS.get(class_name)
    if rgb is None:
        digest = hashlib.sha256(class_name.encode("utf-8")).digest()
        hue = int.from_bytes(digest[:4], "big") / (2**32)
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.70)
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)
