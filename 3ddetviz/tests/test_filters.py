"""Independent geometric examples and prediction-filter pipeline behavior."""

from dataclasses import FrozenInstanceError
import subprocess
import sys

import numpy as np
import pytest

from dataset_utils import Box3D
from detviz.predictions.models import Detection, PredictionFrame
from detviz.processing import (
    DEFAULT_NMS_GROUPS, FilterConfig, bev_iou, filter_predictions,
    non_maximum_suppression,
)


def box(center=(0, 0, 0), size=(2, 2, 2), rotation=None, class_name="car"):
    return Box3D("box", class_name,
        np.array(center, dtype=float), np.array(size, dtype=float),
        np.eye(3) if rotation is None else rotation,
    )


def detection(identifier, score=0.8, *, center=(0, 0, 0), size=(2, 2, 2), rotation=None, class_name="car"):
    return Detection(box(center, size, rotation, class_name), score, identifier)


def frame(*detections):
    return PredictionFrame("sample", tuple(detections))


def ids(result):
    return tuple(detection.identifier for detection in result.detections)


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_hand_computed_axis_aligned_bev_iou_and_disjoint_contacts():
    first = box()
    assert bev_iou(first, first) == pytest.approx(1)
    # Two 2x2 squares offset by one metre: intersection 2, union 6.
    assert bev_iou(first, box(center=(1, 0, 0))) == pytest.approx(1 / 3)
    assert bev_iou(first, box(center=(2, 0, 0))) == 0
    assert bev_iou(first, box(center=(3, 0, 0))) == 0
    assert bev_iou(first, box(center=(2, 2, 0))) == 0


def test_yaw_uses_rotated_footprints_instead_of_axis_aligned_bounds():
    first = box(size=(4, 2, 2))
    # Perpendicular 4x2 rectangles intersect in a 2x2 square: 4/(8+8-4).
    assert bev_iou(first, box(size=(4, 2, 2), rotation=rotation_z(np.pi / 2))) == pytest.approx(1 / 3)
    # A square rotated by 45 degrees has IoU 1/sqrt(2) with the original.
    assert bev_iou(box(), box(rotation=rotation_z(np.pi / 4))) == pytest.approx(1 / np.sqrt(2))


def test_pitch_and_roll_keep_height_contributions_to_bev_footprint():
    first = box(size=(4, 2, 6))
    # Pitch swaps length/height: a 6x2 footprint contains the original 4x2.
    assert bev_iou(first, box(size=(4, 2, 6), rotation=rotation_y(np.pi / 2))) == pytest.approx(2 / 3)
    # Roll swaps width/height: a 4x6 footprint contains the original 4x2.
    assert bev_iou(first, box(size=(4, 2, 6), rotation=rotation_x(np.pi / 2))) == pytest.approx(1 / 3)


def test_combined_pitch_roll_footprint_area_matches_projected_surface_formula():
    length, width, height = 4, 2, 6
    rotation = rotation_z(0.3) @ rotation_y(0.6) @ rotation_x(-0.4)
    tilted = box(size=(length, width, height), rotation=rotation)
    # Orthographic area is the sum of the XY projections of one face from
    # each pair of opposing faces. The tilted box fits wholly inside 100x100.
    expected_area = (
        length * width * abs(rotation[2, 2])
        + width * height * abs(rotation[2, 0])
        + length * height * abs(rotation[2, 1])
    )
    assert bev_iou(tilted, box(size=(100, 100, 1))) == pytest.approx(expected_area / 10_000)


def test_bev_overlap_ignores_vertical_separation_for_upright_boxes():
    assert bev_iou(box(center=(0, 0, -100)), box(center=(0, 0, 100), size=(2, 2, 100))) == pytest.approx(1)
    detections = (detection("low", 0.7, center=(0, 0, -100)), detection("high", 0.9, center=(0, 0, 100)))
    result = non_maximum_suppression(detections, mode="class", iou_threshold=0.5)
    assert result.kept_indices == (1,)
    assert result.suppressed_by == {0: 1}


def test_overlap_is_symmetric_and_stable_with_large_coordinate_translation():
    first = box(size=(4, 2, 3), rotation=rotation_z(0.3))
    second = box(center=(1, 0.5, 0), size=(3, 2, 1), rotation=rotation_z(-0.5))
    overlap = bev_iou(first, second)
    assert 0 < overlap < 1
    assert bev_iou(second, first) == pytest.approx(overlap, abs=1e-14)
    shift = np.array([1e8, -2e8, 100])
    translated_first = box(center=first.center + shift, size=first.size, rotation=first.rotation)
    translated_second = box(center=second.center + shift, size=second.size, rotation=second.rotation)
    assert bev_iou(translated_first, translated_second) == pytest.approx(overlap, abs=1e-8)


@pytest.mark.parametrize("mode,kept", [
    ("off", (1, 0, 2)), ("class", (1, 2)), ("agnostic", (1,)), ("groups", (1, 2)),
])
def test_nms_modes_are_explicit_and_return_original_indices(mode, kept):
    detections = (
        detection("car-low", 0.7), detection("car-high", 0.9),
        detection("pedestrian", 0.5, class_name="pedestrian"),
    )
    result = non_maximum_suppression(detections, mode=mode)
    assert result.kept_indices == kept
    if mode == "off":
        assert result.suppressed_by == {}
    elif mode == "agnostic":
        assert result.suppressed_by == {0: 1, 2: 1}
    else:
        assert result.suppressed_by == {0: 1}


def test_default_groups_suppress_centerpoint_task_members_together():
    classes = (
        "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
        "motorcycle", "bicycle", "pedestrian", "traffic_cone", "unknown-a", "unknown-b",
    )
    detections = tuple(detection(name, 1 - index * .05, class_name=name) for index, name in enumerate(classes))
    result = non_maximum_suppression(detections, mode="groups")
    assert tuple(detections[index].identifier for index in result.kept_indices) == (
        "car", "truck", "bus", "barrier", "motorcycle", "pedestrian", "unknown-a", "unknown-b",
    )
    assert result.suppressed_by == {2: 1, 4: 3, 7: 6, 9: 8}


def test_custom_groups_and_unlisted_singletons():
    detections = (detection("car"), detection("truck", class_name="truck"), detection("bike", class_name="bicycle"))
    result = non_maximum_suppression(detections, mode="groups", groups=(("car", "truck"),))
    assert result.kept_indices == (0, 2)
    assert result.suppressed_by == {1: 0}
    assert non_maximum_suppression(detections, mode="groups", groups=()).kept_indices == (0, 1, 2)


def test_exact_iou_threshold_is_retained_and_score_ties_preserve_source_order():
    detections = (detection("z-first", .8), detection("a-second", .8, center=(1, 0, 0)))
    equal = non_maximum_suppression(detections, mode="class", iou_threshold=1 / 3)
    assert equal.kept_indices == (0, 1)
    below = non_maximum_suppression(detections, mode="class", iou_threshold=np.nextafter(1 / 3, 0))
    assert below.kept_indices == (0,)
    assert below.suppressed_by == {1: 0}
    identical = non_maximum_suppression((detection("z-first"), detection("a-second")), iou_threshold=1)
    assert identical.kept_indices == (0, 1)


def test_suppressed_boxes_do_not_suppress_later_candidates():
    detections = (
        detection("a", .9), detection("b", .8, center=(.5, 0, 0)),
        detection("c", .7, center=(1, 0, 0)),
    )
    result = non_maximum_suppression(detections, iou_threshold=.5)
    assert result.kept_indices == (0, 2)
    assert result.suppressed_by == {1: 0}


def test_score_overrides_are_inclusive_and_replace_global_threshold():
    predictions = frame(
        detection("car", .3), detection("truck", .3, class_name="truck"),
        detection("bike", .3, class_name="bicycle"), detection("truck-low", .2, class_name="truck"),
    )
    config = FilterConfig(score_threshold=.5, per_class_scores={"car": .4, "truck": .3})
    result = filter_predictions(predictions, config)
    assert ids(result) == ("truck",)
    assert result.reasons == {"car": "score", "bike": "score", "truck-low": "score"}
    assert result.score_count == 1


def test_distance_is_horizontal_and_height_uses_box_center_with_inclusive_bounds():
    predictions = frame(
        detection("boundary", .1, center=(3, 4, 1), size=(2, 2, 100)),
        detection("below", .5, center=(3, 4, -1)),
        detection("above", .5, center=(3, 4, 2)),
        detection("near", .5, center=(2, 0, 0)),
        detection("far", .5, center=(6, 0, 0)),
    )
    result = filter_predictions(predictions, FilterConfig(min_distance=5, max_distance=5, min_height=-1, max_height=1))
    assert ids(result) == ("below", "boundary")
    assert result.reasons == {"above": "height", "near": "distance", "far": "distance"}
    vertical = frame(detection("high-but-near", center=(3, 4, 1000)))
    assert ids(filter_predictions(vertical, FilterConfig(max_distance=5))) == ("high-but-near",)


def test_pipeline_counts_and_provenance_cover_each_processing_stage():
    predictions = frame(
        detection("low-score", .05),
        detection("far", .95, center=(100, 0, 0)),
        detection("below", .94, center=(0, 0, -3)),
        detection("above", .93, center=(0, 0, 3)),
        detection("a", .9), detection("b", .8),
        detection("c", .7, center=(10, 0, 0)),
        detection("d", .6, center=(20, 0, 0)),
        detection("e", .5, center=(30, 0, 0)),
    )
    config = FilterConfig(max_distance=50, min_height=-1, max_height=1, pre_nms_limit=4, max_detections=2, nms_mode="class")
    result = filter_predictions(predictions, config)
    assert ids(result) == ("a", "c")
    assert (
        result.loaded_count, result.score_count, result.spatial_count,
        result.pre_nms_count, result.nms_count, result.output_count,
    ) == (9, 8, 5, 4, 3, 2)
    assert result.reasons == {
        "low-score": "score", "far": "distance", "below": "height", "above": "height",
        "e": "pre_nms_limit", "b": "nms", "d": "max_detections",
    }
    assert result.suppressed_by == {"b": "a"}
    assert set(ids(result)).isdisjoint(result.reasons)
    assert len(result.reasons) + result.output_count == result.loaded_count


def test_pre_nms_and_output_limits_are_global_stable_score_limits():
    predictions = frame(
        detection("low", .2), detection("z-first", .9, center=(10, 0, 0)),
        detection("a-second", .9, center=(20, 0, 0)), detection("medium", .8, center=(30, 0, 0)),
    )
    result = filter_predictions(predictions, FilterConfig(pre_nms_limit=3, max_detections=2))
    assert ids(result) == ("z-first", "a-second")
    assert result.reasons == {"low": "pre_nms_limit", "medium": "max_detections"}
    assert result.pre_nms_count == result.nms_count == 3


@pytest.mark.parametrize("config,pre,nms", [
    (FilterConfig(pre_nms_limit=0), 0, 0), (FilterConfig(max_detections=0), 1, 1),
])
def test_zero_limits_intentionally_return_no_predictions(config, pre, nms):
    result = filter_predictions(frame(detection("one")), config)
    assert ids(result) == ()
    assert result.pre_nms_count == pre
    assert result.nms_count == nms
    assert result.output_count == 0


def test_relaxing_filters_and_disabling_nms_recovers_untouched_original_data():
    original = frame(detection("first", .9), detection("second", .4))
    centers = [item.box.center.copy() for item in original.detections]
    scores = tuple(item.score for item in original.detections)
    assert ids(filter_predictions(original, FilterConfig(nms_mode="class"))) == ("first",)
    assert ids(filter_predictions(original, FilterConfig(score_threshold=.95))) == ()
    restored = filter_predictions(original, FilterConfig(score_threshold=0, nms_mode="off"))
    assert ids(restored) == ("first", "second")
    assert all(output is source for output, source in zip(restored.detections, original.detections))
    assert tuple(item.score for item in original.detections) == scores
    for source, expected in zip(original.detections, centers):
        np.testing.assert_array_equal(source.box.center, expected)
        assert not source.box.center.flags.writeable


def test_empty_frame_has_zero_stage_counts_in_every_nms_mode():
    for mode in ("off", "class", "agnostic", "groups"):
        result = filter_predictions(frame(), FilterConfig(nms_mode=mode))
        assert result.detections == ()
        assert result.reasons == result.suppressed_by == {}
        assert (result.loaded_count, result.score_count, result.spatial_count, result.pre_nms_count, result.nms_count, result.output_count) == (0, 0, 0, 0, 0, 0)


def test_configuration_preserves_immutable_independent_settings():
    scores = {"truck": .3, "car": .5}
    groups = [["car", "truck"]]
    config = FilterConfig(per_class_scores=scores, nms_groups=groups)
    scores["car"] = .9
    groups[0].append("pedestrian")
    assert config.per_class_scores["car"] == .5
    assert config.nms_groups == (("car", "truck"),)
    equivalent = FilterConfig(per_class_scores={"car": .5, "truck": .3}, nms_groups=(("car", "truck"),))
    assert config == equivalent
    with pytest.raises(TypeError):
        config.per_class_scores["car"] = .8
    with pytest.raises(FrozenInstanceError):
        config.score_threshold = .8
    assert FilterConfig().nms_groups == DEFAULT_NMS_GROUPS


@pytest.mark.parametrize("settings", [
    {"score_threshold": -.1}, {"score_threshold": 1.1}, {"score_threshold": np.nan},
    {"score_threshold": True}, {"score_threshold": ".1"},
    {"per_class_scores": []}, {"per_class_scores": {"": .1}},
    {"per_class_scores": {"car": np.inf}}, {"per_class_scores": {"car": -1}},
    {"min_distance": -1}, {"max_distance": -1}, {"max_distance": np.inf},
    {"min_distance": 10, "max_distance": 9}, {"min_distance": np.nan},
    {"min_height": np.nan}, {"max_height": np.inf}, {"min_height": 2, "max_height": 1},
    {"pre_nms_limit": -1}, {"pre_nms_limit": 1.5}, {"pre_nms_limit": True},
    {"max_detections": -1}, {"max_detections": 1.5}, {"max_detections": np.inf},
    {"nms_mode": "3d"}, {"nms_iou_threshold": -1}, {"nms_iou_threshold": 1.1},
    {"nms_iou_threshold": np.nan}, {"nms_iou_threshold": True},
    {"nms_groups": "car"}, {"nms_groups": ("car",)}, {"nms_groups": ((),)},
    {"nms_groups": (("car", "car"),)}, {"nms_groups": (("car",), ("car",))},
    {"nms_groups": (("",),)}, {"nms_groups": None},
])
def test_invalid_filter_configuration_is_rejected_before_processing(settings):
    with pytest.raises(ValueError):
        FilterConfig(**settings)


def test_invalid_geometry_and_standalone_nms_settings_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        bev_iou(box(size=(0, 2, 2)), box())
    with pytest.raises(ValueError, match="finite"):
        bev_iou(box(center=(np.nan, 0, 0)), box())
    with pytest.raises(ValueError, match="proper"):
        bev_iou(box(rotation=np.diag([1, 1, -1])), box())
    with pytest.raises(ValueError, match="nms_mode"):
        non_maximum_suppression((), mode="bad")
    with pytest.raises(ValueError, match="nms_iou_threshold"):
        non_maximum_suppression((), iou_threshold=np.inf)


def test_processing_import_and_nms_do_not_require_torch_pcdet_or_cuda():
    script = """
import importlib.abc
import sys
class NoDetectorFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'pcdet'}:
            raise AssertionError('Unexpected framework import: ' + fullname)
sys.meta_path.insert(0, NoDetectorFrameworks())
import numpy as np
from dataset_utils import Box3D
from detviz.predictions.models import Detection, PredictionFrame
from detviz.processing import FilterConfig, filter_predictions
box = Box3D('box', 'car', np.zeros(3), np.ones(3), np.eye(3))
frame = PredictionFrame('sample', (Detection(box, .9, 'a'), Detection(box, .8, 'b')))
result = filter_predictions(frame, FilterConfig(nms_mode='class'))
assert result.output_count == 1
assert result.suppressed_by == {'b': 'a'}
assert 'torch' not in sys.modules and 'pcdet' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
