"""Pure prediction filtering; cached source data and ground truth stay intact."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from detviz.predictions.models import Detection, PredictionFrame
from .nms import DEFAULT_NMS_GROUPS, _validate_nms_options, non_maximum_suppression


def _number(value: object, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return result


def _limit(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be None or a nonnegative integer.")
    return int(value)


@dataclass(frozen=True)
class FilterConfig:
    """Filtering policy, independent of display class/layer visibility.

    Distance is horizontal radius of the box center in reference LiDAR XY;
    height is its center's LiDAR z coordinate. All thresholds are inclusive.
    Score overrides replace the global threshold for their named class.
    Limits are global across classes and permit zero. NMS suppresses only
    overlaps strictly greater than its IoU threshold.
    """

    score_threshold: float = 0.1
    per_class_scores: Mapping[str, float] | None = None
    min_distance: float = 0.0
    max_distance: float | None = None
    min_height: float | None = None
    max_height: float | None = None
    pre_nms_limit: int | None = None
    max_detections: int | None = None
    nms_mode: str = "off"
    nms_iou_threshold: float = 0.2
    nms_groups: tuple[tuple[str, ...], ...] = DEFAULT_NMS_GROUPS

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_threshold", _number(self.score_threshold, "score_threshold", minimum=0, maximum=1))
        if self.per_class_scores is not None and not isinstance(self.per_class_scores, Mapping):
            raise ValueError("per_class_scores must be a mapping of class names to thresholds.")
        overrides = {}
        for name, threshold in (self.per_class_scores or {}).items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("per_class_scores keys must be nonempty class names.")
            overrides[name] = _number(threshold, f"per_class_scores[{name!r}]", minimum=0, maximum=1)
        object.__setattr__(self, "per_class_scores", MappingProxyType(overrides))
        minimum = _number(self.min_distance, "min_distance", minimum=0)
        maximum = None if self.max_distance is None else _number(self.max_distance, "max_distance", minimum=0)
        if maximum is not None and maximum < minimum:
            raise ValueError("max_distance must be >= min_distance.")
        object.__setattr__(self, "min_distance", minimum)
        object.__setattr__(self, "max_distance", maximum)
        lower = None if self.min_height is None else _number(self.min_height, "min_height")
        upper = None if self.max_height is None else _number(self.max_height, "max_height")
        if lower is not None and upper is not None and upper < lower:
            raise ValueError("max_height must be >= min_height.")
        object.__setattr__(self, "min_height", lower)
        object.__setattr__(self, "max_height", upper)
        object.__setattr__(self, "pre_nms_limit", _limit(self.pre_nms_limit, "pre_nms_limit"))
        object.__setattr__(self, "max_detections", _limit(self.max_detections, "max_detections"))
        mode, threshold, groups = _validate_nms_options(self.nms_mode, self.nms_iou_threshold, self.nms_groups)
        object.__setattr__(self, "nms_mode", mode)
        object.__setattr__(self, "nms_iou_threshold", threshold)
        object.__setattr__(self, "nms_groups", groups)

    def cache_key(self) -> tuple:
        """Return a hashable policy key; frame/source revisions belong to callers."""

        return (
            self.score_threshold, tuple(sorted(self.per_class_scores.items())),
            self.min_distance, self.max_distance, self.min_height, self.max_height,
            self.pre_nms_limit, self.max_detections, self.nms_mode,
            self.nms_iou_threshold, self.nms_groups,
        )

    def __hash__(self) -> int:
        return hash(self.cache_key())


@dataclass(frozen=True)
class FilterResult:
    """Retained predictions, monotonic stage counts and removed-ID provenance.

    Counts describe loaded, score-passing, spatial-passing, pre-NMS-limited,
    NMS-passing and final output sets. Reasons contain only removed identifiers:
    ``score``, ``distance``, ``height``, ``pre_nms_limit``, ``nms`` or
    ``max_detections``. ``suppressed_by`` maps NMS-removed identifiers to the
    corresponding keeper identifier, before the final output cap is applied.
    Display visibility does not contribute to these counts.
    """

    detections: tuple[Detection, ...]
    loaded_count: int
    score_count: int
    spatial_count: int
    pre_nms_count: int
    nms_count: int
    output_count: int
    reasons: dict[str, str]
    suppressed_by: dict[str, str]


def filter_predictions(frame: PredictionFrame, config: FilterConfig) -> FilterResult:
    """Filter the original prediction frame without mutating any input.

    The processing order is score -> distance/center-height -> stable score
    order -> pre-NMS limit -> NMS -> output limit. Equal scores preserve source
    order. Re-running with relaxed settings therefore recovers original boxes.
    """

    if not isinstance(config, FilterConfig):
        raise TypeError("config must be a FilterConfig.")
    detections = tuple(frame.detections)
    identifiers = [detection.identifier for detection in detections]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("Detection identifiers must be nonempty strings.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Detection identifiers must be unique within a frame.")
    reasons = {}
    score_passing = []
    for detection in detections:
        score = _number(detection.score, "Detection score", minimum=0, maximum=1)
        threshold = config.per_class_scores.get(detection.box.class_name, config.score_threshold)
        if score >= threshold:
            score_passing.append(detection)
        else:
            reasons[detection.identifier] = "score"

    spatial_passing = []
    for detection in score_passing:
        center = np.asarray(detection.box.center, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Detection box center must be finite with shape (3,).")
        distance = float(np.hypot(center[0], center[1]))
        if distance < config.min_distance or (config.max_distance is not None and distance > config.max_distance):
            reasons[detection.identifier] = "distance"
        elif ((config.min_height is not None and center[2] < config.min_height)
                or (config.max_height is not None and center[2] > config.max_height)):
            reasons[detection.identifier] = "height"
        else:
            spatial_passing.append(detection)
    ordered = sorted(spatial_passing, key=lambda detection: -detection.score)
    before_nms = ordered if config.pre_nms_limit is None else ordered[:config.pre_nms_limit]
    for detection in ordered[len(before_nms):]:
        reasons[detection.identifier] = "pre_nms_limit"
    nms = non_maximum_suppression(
        before_nms, mode=config.nms_mode, iou_threshold=config.nms_iou_threshold,
        groups=config.nms_groups,
    )
    after_nms = tuple(before_nms[index] for index in nms.kept_indices)
    suppressed_by = {}
    for removed, keeper in nms.suppressed_by.items():
        identifier = before_nms[removed].identifier
        reasons[identifier] = "nms"
        suppressed_by[identifier] = before_nms[keeper].identifier
    retained = after_nms if config.max_detections is None else after_nms[:config.max_detections]
    for detection in after_nms[len(retained):]:
        reasons[detection.identifier] = "max_detections"
    return FilterResult(
        detections=tuple(retained),
        loaded_count=len(detections),
        score_count=len(score_passing),
        spatial_count=len(spatial_passing),
        pre_nms_count=len(before_nms),
        nms_count=len(after_nms),
        output_count=len(retained),
        reasons=reasons,
        suppressed_by=suppressed_by,
    )


__all__ = ["FilterConfig", "FilterResult", "filter_predictions"]
