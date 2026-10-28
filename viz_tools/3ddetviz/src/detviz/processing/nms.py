"""Deterministic CPU nonmaximum suppression using oriented BEV footprints.

Footprints are convex hulls of all eight rotated cuboid corners projected onto
the reference LiDAR XY plane. Height does not affect overlap for upright boxes;
pitch and roll are retained when they change that footprint. This is BEV IoU,
not 3D-volume IoU. No detector framework or compiled extensions are imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Sequence

from visual_utils import Box3D

import numpy as np

if TYPE_CHECKING:
    from detviz.predictions.models import Detection


DEFAULT_NMS_GROUPS = (
    ("car",),
    ("truck", "construction_vehicle"),
    ("bus", "trailer"),
    ("barrier",),
    ("motorcycle", "bicycle"),
    ("pedestrian", "traffic_cone"),
)
NMS_MODES = ("off", "class", "agnostic", "groups")

_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)



@dataclass(frozen=True)
class NMSResult:
    """Indices into the supplied sequence, including suppression provenance."""

    kept_indices: tuple[int, ...]
    suppressed_by: dict[int, int]


def _validate_nms_options(mode: str, threshold: float, groups: object) -> tuple[str, float, tuple[tuple[str, ...], ...]]:
    if not isinstance(mode, str) or mode not in NMS_MODES:
        raise ValueError(f"nms_mode must be one of {', '.join(NMS_MODES)}.")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise ValueError("nms_iou_threshold must be a finite number in [0, 1].")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("nms_iou_threshold must be a finite number in [0, 1].")
    if isinstance(groups, (str, bytes)):
        raise ValueError("nms_groups must contain nonempty groups of distinct class names.")
    try:
        source_groups = tuple(groups)
    except TypeError as exc:
        raise ValueError("nms_groups must contain nonempty groups of distinct class names.") from exc
    normalized = []
    seen = set()
    for source in source_groups:
        if isinstance(source, (str, bytes)):
            raise ValueError("Each NMS group must be a nonempty sequence of class names.")
        try:
            group = tuple(source)
        except TypeError as exc:
            raise ValueError("Each NMS group must be a nonempty sequence of class names.") from exc
        if not group:
            raise ValueError("Each NMS group must be nonempty.")
        for name in group:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("NMS group class names must be nonempty strings.")
            if name in seen:
                raise ValueError(f"Class {name!r} occurs more than once in nms_groups.")
            seen.add(name)
        normalized.append(group)
    return mode, threshold, tuple(normalized)


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return counterclockwise vertices, excluding a repeated closing vertex."""

    ordered = np.unique(points, axis=0)
    if len(ordered) < 3:
        raise ValueError("Box BEV footprint must have positive area.")
    lower = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-1] - lower[-2], point - lower[-1]) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and _cross(upper[-1] - upper[-2], point - upper[-1]) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    # Subtract a local origin before the shoelace sum, preserving small areas
    # when coordinates use a large global map offset.
    relative = polygon - polygon[0]
    following = np.roll(relative, -1, axis=0)
    return abs(float(np.sum(relative[:, 0] * following[:, 1] - relative[:, 1] * following[:, 0]))) / 2


def _footprint(box: Box3D) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    center, size, rotation = box.center, box.size, box.rotation
    corners = (_SIGNS * (size / 2)) @ rotation.T + center
    if not np.isfinite(corners).all():
        raise ValueError("Box geometry produced a nonfinite BEV footprint.")
    polygon = _convex_hull(corners[:, :2])
    area = _area(polygon)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Box BEV footprint must have positive finite area.")
    return polygon, area, polygon.min(axis=0), polygon.max(axis=0)


def _intersection(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Clip a convex polygon against a counterclockwise convex polygon."""

    polygon = subject
    for index, first in enumerate(clipper):
        if not len(polygon):
            break
        edge = clipper[(index + 1) % len(clipper)] - first
        output = []
        previous = polygon[-1]
        previous_distance = _cross(edge, previous - first)
        for current in polygon:
            distance = _cross(edge, current - first)
            if (distance >= 0) != (previous_distance >= 0):
                fraction = previous_distance / (previous_distance - distance)
                output.append(previous + fraction * (current - previous))
            if distance >= 0:
                output.append(current)
            previous, previous_distance = current, distance
        polygon = np.asarray(output, dtype=np.float64).reshape(-1, 2)
    return polygon


def _footprint_iou(first, second) -> float:
    first_polygon, first_area, first_min, first_max = first
    second_polygon, second_area, second_min, second_max = second
    if np.any(np.minimum(first_max, second_max) <= np.maximum(first_min, second_min)):
        return 0.0
    origin = first_polygon[0]
    intersection = _area(_intersection(first_polygon - origin, second_polygon - origin))
    # Roundoff can put an identical polygon's intersection a few ulps above
    # its own area. Bounding the intersection keeps the resulting IoU in [0,1].
    intersection = min(max(intersection, 0.0), first_area, second_area)
    return float(intersection / (first_area + second_area - intersection))


def bev_iou(first: Box3D, second: Box3D) -> float:
    """Return IoU between full-rotation cuboid footprints in reference LiDAR XY."""

    return _footprint_iou(_footprint(first), _footprint(second))


def non_maximum_suppression(
    detections: Sequence[Detection], *, mode: str = "class", iou_threshold: float = 0.2,
    groups: tuple[tuple[str, ...], ...] = DEFAULT_NMS_GROUPS,
) -> NMSResult:
    """Greedily retain highest scores, suppressing only IoU > the threshold.

    Equal scores preserve input order. Indices always reference the supplied
    sequence. In ``groups`` mode, classes omitted from the configuration each
    retain their own singleton group. ``off`` still returns stable score order
    but performs no geometric suppression. Input detections are never changed.
    """

    mode, threshold, groups = _validate_nms_options(mode, iou_threshold, groups)
    order = sorted(range(len(detections)), key=lambda index: -detections[index].score)
    if mode == "off" or not order:
        return NMSResult(tuple(order), {})

    group_by_class = {name: index for index, group in enumerate(groups) for name in group}
    def group_key(index):
        name = detections[index].box.class_id
        if mode == "agnostic":
            return ("all",)
        if mode == "groups" and name in group_by_class:
            return ("group", group_by_class[name])
        return ("class", name)

    footprints = [_footprint(detection.box) for detection in detections]
    minima = np.array([footprint[2] for footprint in footprints])
    maxima = np.array([footprint[3] for footprint in footprints])
    kept = []
    kept_by_group: dict[tuple, list[int]] = {}
    suppressed_by = {}
    for index in order:
        key = group_key(index)
        previous_indices = np.asarray(kept_by_group.get(key, ()), dtype=np.int64)
        # Most pairs are spatially disjoint. Test their AABBs together instead
        # of making a Python/NumPy call for every pair in the greedy loop.
        if previous_indices.size:
            overlaps = np.all(
                (maxima[previous_indices] > minima[index])
                & (minima[previous_indices] < maxima[index]), axis=1,
            )
            previous_indices = previous_indices[overlaps]
        for previous in previous_indices:
            previous = int(previous)
            if _footprint_iou(footprints[index], footprints[previous]) > threshold:
                suppressed_by[index] = previous
                break
        else:
            kept.append(index)
            kept_by_group.setdefault(key, []).append(index)
    return NMSResult(tuple(kept), suppressed_by)


__all__ = ["DEFAULT_NMS_GROUPS", "NMS_MODES", "NMSResult", "bev_iou", "non_maximum_suppression"]
