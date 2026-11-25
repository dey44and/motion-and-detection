"""nuScenes LiDAR segmentation label order and evaluation mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..errors import DatasetValidationError


# The 16 challenge classes use the nuScenes devkit palette. Give ignored
# points their own visible swatch so they remain recognizable in dark views.
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/utils.py
EVALUATION_CLASSES = (
    ("ignored", (111, 120, 129)),
    ("barrier", (112, 128, 144)),
    ("bicycle", (220, 20, 60)),
    ("bus", (255, 127, 80)),
    ("car", (255, 158, 0)),
    ("construction_vehicle", (233, 150, 70)),
    ("motorcycle", (255, 61, 99)),
    ("pedestrian", (0, 0, 230)),
    ("traffic_cone", (47, 79, 79)),
    ("trailer", (255, 140, 0)),
    ("truck", (255, 99, 71)),
    ("driveable_surface", (0, 207, 191)),
    ("other_flat", (175, 0, 75)),
    ("sidewalk", (75, 0, 75)),
    ("terrain", (112, 180, 60)),
    ("manmade", (222, 184, 135)),
    ("vegetation", (0, 175, 0)),
)

# Native ID order in the official lidarseg category table. The fallback
# handles installs that keep the detection category.json with no index field.
_NATIVE_CLASSES = (
    ("noise", 0),
    ("animal", 0),
    ("human.pedestrian.adult", 7),
    ("human.pedestrian.child", 7),
    ("human.pedestrian.construction_worker", 7),
    ("human.pedestrian.personal_mobility", 0),
    ("human.pedestrian.police_officer", 7),
    ("human.pedestrian.stroller", 0),
    ("human.pedestrian.wheelchair", 0),
    ("movable_object.barrier", 1),
    ("movable_object.debris", 0),
    ("movable_object.pushable_pullable", 0),
    ("movable_object.trafficcone", 8),
    ("static_object.bicycle_rack", 0),
    ("vehicle.bicycle", 2),
    ("vehicle.bus.bendy", 3),
    ("vehicle.bus.rigid", 3),
    ("vehicle.car", 4),
    ("vehicle.construction", 5),
    ("vehicle.emergency.ambulance", 0),
    ("vehicle.emergency.police", 0),
    ("vehicle.motorcycle", 6),
    ("vehicle.trailer", 9),
    ("vehicle.truck", 10),
    ("flat.driveable_surface", 11),
    ("flat.other", 12),
    ("flat.sidewalk", 13),
    ("flat.terrain", 14),
    ("static.manmade", 15),
    ("static.other", 0),
    ("static.vegetation", 16),
    ("vehicle.ego", 0),
)


def native_to_evaluation(categories: tuple[dict, ...]) -> np.ndarray:
    mapping = np.array([label for _, label in _NATIVE_CLASSES], dtype=np.uint8)
    known = dict(_NATIVE_CLASSES)
    seen_ids, seen_names = set(), set()
    for category in categories:
        if "index" not in category:
            continue
        identifier, name = category["index"], category["name"]
        if (type(identifier) is not int or not 0 <= identifier < len(mapping)
                or identifier in seen_ids or name not in known or name in seen_names):
            raise DatasetValidationError(f"Invalid nuScenes lidarseg category {name!r} at index {identifier!r}.")
        mapping[identifier] = known[name]
        seen_ids.add(identifier)
        seen_names.add(name)
    if seen_ids and len(seen_ids) != len(mapping):
        raise DatasetValidationError("nuScenes lidarseg category indices must cover all 32 native classes.")
    return mapping


def validate_labels(values, point_count: int, *, context: str, minimum: int = 0, maximum: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.shape != (point_count,):
        raise DatasetValidationError(
            f"{context} has label shape {labels.shape}; expected one label for each of {point_count:,} original points."
        )
    if labels.dtype.kind not in "iu" or np.any((labels < minimum) | (labels > maximum)):
        raise DatasetValidationError(f"{context} requires integer class IDs in {minimum}–{maximum}.")
    return np.frombuffer(labels.astype(np.uint8, copy=False).tobytes(), dtype=np.uint8)


def read_label_file(path: Path, point_count: int) -> np.ndarray:
    if path.stat().st_size != point_count:
        raise DatasetValidationError(f"{path.name} must contain {point_count:,} uint8 labels in the original scan order.")
    with path.open("rb") as stream:
        return np.frombuffer(stream.read(point_count + 1), dtype=np.uint8)
