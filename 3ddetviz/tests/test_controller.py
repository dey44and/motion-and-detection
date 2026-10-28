"""Review composes canonical observations without a dataset adapter layer."""

from dataclasses import replace

import numpy as np

from dataset_utils import Box3D, Modality
from threedviz.dataloader import create_dataset
from detviz.controller import ReviewFrame, compose_review, refilter
from detviz.predictions import Detection, InMemoryPredictionSource, PredictionFrame, PredictionInfo
from detviz.processing.filters import FilterConfig


def observation():
    with create_dataset("demo") as loader:
        scene = loader.list_scenes()[0]
        frame_id = loader.list_frames(scene.scene_id, modality=Modality.POINT_CLOUD)[0].frame_id
        return loader.load_point_cloud(frame_id)


def source(frame, empty=False):
    detections = () if empty else (
        Detection(Box3D("high", "car", [3, 0, 1], [4, 2, 1.5], np.eye(3)), .9, "high"),
        Detection(Box3D("overlap", "car", [3.1, 0, 1], [4, 2, 1.5], np.eye(3)), .7, "overlap"),
        Detection(Box3D("low", "custom", [25, 0, 1], [4, 2, 1.5], np.eye(3)), .2, "low"),
    )
    return InMemoryPredictionSource([PredictionFrame(frame.info.frame_id, detections)],
                                    info=PredictionInfo("test", "memory", "lidar"))


def test_review_preserves_canonical_observation_and_original_predictions():
    frame = observation()
    with source(frame) as predictions:
        review = compose_review(frame, source=predictions, config=FilterConfig(score_threshold=.8))
        assert isinstance(review, ReviewFrame)
        assert review.observation is frame
        assert review.predictions is predictions.get_frame(frame.info.frame_id)
        assert [item.identifier for item in review.filtered.detections] == ["high"]
        suppressed = refilter(review, FilterConfig(score_threshold=0, max_distance=10, nms_mode="class"))
        assert [item.identifier for item in suppressed.filtered.detections] == ["high"]
        relaxed = refilter(suppressed, FilterConfig(score_threshold=0))
        assert len(relaxed.filtered.detections) == 3
        assert relaxed.observation is frame and relaxed.predictions is review.predictions
    assert frame.points.shape[1] == 3  # payload lifetime is independent of its loader/source


def test_missing_source_missing_frame_and_empty_predictions_are_distinct():
    frame = observation()
    assert compose_review(frame).predictions is None
    with source(frame, empty=True) as predictions:
        empty = compose_review(frame, source=predictions)
        assert empty.predictions.detections == () and empty.filtered.output_count == 0
        other = replace(frame, info=replace(frame.info, frame_id="other-frame"))
        missing = compose_review(other, source=predictions)
        assert missing.predictions is None and missing.filtered is None


def test_demo_composition_is_deterministic_and_has_overlapping_candidates():
    frame = observation()
    first, repeated = compose_review(frame, demo=True), compose_review(frame, demo=True)
    assert len(first.predictions.detections) > len(frame.annotations.boxes)
    np.testing.assert_array_equal(first.predictions.detections[0].box.center,
                                  repeated.predictions.detections[0].box.center)
    after_nms = refilter(first, FilterConfig(score_threshold=0, nms_mode="class"))
    assert after_nms.filtered.output_count < first.filtered.output_count
    assert after_nms.observation is frame
