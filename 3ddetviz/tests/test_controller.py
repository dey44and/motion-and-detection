"""Review composition preserves dataset data while joining/filtering predictions."""

from dataclasses import fields, replace

import numpy as np
import pytest

from threedviz.dataloader import BoundingBox3D, DatasetAdapter, DatasetError, FrameData, FrameInfo, SceneInfo, create_dataset

from detviz.controller import ReviewDatasetAdapter, ReviewFrame, refilter
from detviz.predictions.base import PredictionSource
from detviz.predictions.models import Detection, PredictionFrame, PredictionInfo
from detviz.processing.filters import FilterConfig


def box(class_name="car", x=0, token="box"):
    return BoundingBox3D(np.array([x, 0., 1.]), np.array([4., 2., 1.5]), np.eye(3), class_name, token)


class MemoryDataset(DatasetAdapter):
    def __init__(self, *, plain_prepared_frame=False):
        self.plain_prepared_frame = plain_prepared_frame
        self.load_calls = []
        self.prepare_calls = []
        self.split_calls = []
        self.scenes = (SceneInfo("train-scene", "Train", frame_count=1), SceneInfo("val-scene", "Val", frame_count=3))
        self.infos = {
            "train-scene": (FrameInfo("train-frame", "train-scene", 0, 0),),
            "val-scene": tuple(FrameInfo(f"val-{index}", "val-scene", index, index * 500_000) for index in range(3)),
        }
        self.frames = {
            info.token: FrameData(info, np.array([[1., 2., 3.], [4., 5., 6.]]), (box(x=20),),
                                  intensity=np.array([10., 20.]), lidar_timestamp_us=info.timestamp_us,
                                  ring_indices=np.array([0, 1]), reference_sensor_to_world=np.eye(4))
            for infos in self.infos.values() for info in infos
        }

    @property
    def class_names(self):
        return ("car", "pedestrian")

    @property
    def supports_camera_rgb(self):
        return True

    @property
    def supports_point_timing(self):
        return True

    def list_scenes(self):
        return self.scenes

    def list_scenes_for_split(self, split="all"):
        self.split_calls.append(split)
        if split == "all":
            return self.scenes
        if split == "val":
            return (self.scenes[1],)
        raise DatasetError(f"Unsupported fixture split {split!r}")

    def list_frames(self, scene_token):
        if scene_token not in self.infos:
            raise DatasetError("Unknown fixture scene")
        return self.infos[scene_token]

    def load_frame(self, token):
        self.load_calls.append(token)
        if token not in self.frames:
            raise DatasetError("Unknown fixture frame")
        return self.frames[token]

    def prepare_frame(self, frame, *, camera_rgb=False, point_timing=False):
        self.prepare_calls.append((camera_rgb, point_timing))
        attributes = {item.name: getattr(frame, item.name) for item in fields(FrameData)}
        if camera_rgb:
            attributes["rgb_colors"] = np.full(frame.points.shape, .5)
            attributes["rgb_valid_mask"] = np.ones(len(frame.points), dtype=bool)
        if point_timing:
            attributes["raw_points"] = frame.points + [1, 0, 0]
            attributes["point_timestamps_us"] = np.array([frame.info.timestamp_us, frame.info.timestamp_us + 10])
        if self.plain_prepared_frame:
            return FrameData(**attributes)
        return replace(frame, **attributes)


class MemoryPredictions(PredictionSource):
    def __init__(self, frames):
        self.frames = frames
        self.queries = []
        self.closed = False

    @property
    def info(self):
        return PredictionInfo("Fixture predictions", "fixture", "lidar", "unknown")

    @property
    def class_names(self):
        return ("car", "custom-object")

    @property
    def frame_tokens(self):
        return frozenset(self.frames)

    def get_frame(self, token):
        self.queries.append(token)
        return self.frames.get(token)

    def close(self):
        self.closed = True


def predictions(token="val-0"):
    return PredictionFrame(token, (
        Detection(box(x=3, token="high"), .9, "high"),
        Detection(box(x=3.1, token="overlap"), .7, "overlap"),
        Detection(box("custom-object", x=25, token="low"), .2, "low"),
    ))


def test_validation_split_and_optional_capabilities_delegate_to_dataset():
    base = MemoryDataset()
    adapter = ReviewDatasetAdapter(base)
    assert adapter.list_scenes() == (base.scenes[1],)
    assert base.split_calls == ["val"]
    assert adapter.list_scenes_for_split("all") is base.scenes
    assert adapter.list_frames("val-scene") is base.infos["val-scene"]
    assert adapter.supports_camera_rgb and adapter.supports_point_timing
    assert adapter.class_names == base.class_names
    with pytest.raises(DatasetError, match="Unsupported fixture split"):
        ReviewDatasetAdapter(base, split="test").list_scenes()


def test_filtering_preserves_ground_truth_cloud_and_original_predictions():
    base = MemoryDataset()
    source_frame = predictions()
    source = MemoryPredictions({"val-0": source_frame})
    adapter = ReviewDatasetAdapter(base, source=source, config=FilterConfig(score_threshold=.8))
    original = base.frames["val-0"]
    before = original.points.copy()
    loaded = adapter.load_frame("val-0")
    assert isinstance(loaded, ReviewFrame)
    assert loaded.points is original.points and loaded.boxes is original.boxes
    assert loaded.intensity is original.intensity and loaded.ring_indices is original.ring_indices
    assert loaded.reference_sensor_to_world is original.reference_sensor_to_world
    assert loaded.predictions is not None and len(loaded.predictions.detections) == 3
    assert [d.identifier for d in loaded.filtered.detections] == ["high"]
    strict = refilter(loaded, FilterConfig(score_threshold=0, max_distance=10, nms_mode="class"))
    assert [d.identifier for d in strict.filtered.detections] == ["high"]
    relaxed = refilter(strict, FilterConfig(score_threshold=0, nms_mode="off"))
    assert [d.identifier for d in relaxed.filtered.detections] == ["high", "overlap", "low"]
    assert strict.predictions is loaded.predictions and relaxed.predictions is loaded.predictions
    assert strict.points is original.points and strict.boxes is original.boxes
    assert relaxed.points is original.points and relaxed.boxes is original.boxes
    np.testing.assert_array_equal(original.points, before)
    assert len(source_frame.detections) == 3 and not source_frame.detections[0].box.center.flags.writeable
    assert adapter.class_names == ("car", "pedestrian", "custom-object")


def test_no_source_missing_frame_and_explicit_empty_predictions_are_distinct():
    base = MemoryDataset()
    plain = ReviewDatasetAdapter(base).load_frame("val-0")
    assert plain.predictions is None and plain.filtered is None
    source = MemoryPredictions({"val-0": PredictionFrame("val-0", ())})
    adapter = ReviewDatasetAdapter(base, source=source)
    empty = adapter.load_frame("val-0")
    missing = adapter.load_frame("val-1")
    assert empty.predictions is not None and empty.predictions.detections == ()
    assert empty.filtered.loaded_count == empty.filtered.output_count == 0
    assert missing.predictions is None and missing.filtered is None
    assert source.frame_tokens == {"val-0"}
    assert empty.boxes is base.frames["val-0"].boxes
    assert missing.boxes is base.frames["val-1"].boxes


@pytest.mark.parametrize("plain_prepared_frame", [False, True])
def test_optional_preparation_retains_review_data_for_existing_and_custom_adapters(plain_prepared_frame):
    base = MemoryDataset(plain_prepared_frame=plain_prepared_frame)
    adapter = ReviewDatasetAdapter(base, source=MemoryPredictions({"val-0": predictions()}),
                                   config=FilterConfig(score_threshold=.8))
    loaded = adapter.load_frame("val-0")
    prepared = adapter.prepare_frame(loaded, camera_rgb=True, point_timing=True)
    assert isinstance(prepared, ReviewFrame)
    assert base.prepare_calls == [(True, True)]
    assert prepared.predictions.sample_token == loaded.info.token
    assert [d.identifier for d in prepared.filtered.detections] == ["high"]
    assert prepared.points is loaded.points and prepared.boxes is loaded.boxes
    assert prepared.reference_sensor_to_world is loaded.reference_sensor_to_world
    np.testing.assert_allclose(prepared.raw_points, loaded.points + [1, 0, 0])
    np.testing.assert_allclose(prepared.rgb_colors, .5)
    assert prepared.rgb_valid_mask.all()


def test_two_frame_dataset_cache_reuses_cloud_but_applies_current_filter_settings():
    base = MemoryDataset()
    source = MemoryPredictions({"val-0": predictions()})
    adapter = ReviewDatasetAdapter(base, source=source)
    first = adapter.load_frame("val-0")
    adapter.load_frame("val-1")
    adapter.config = FilterConfig(score_threshold=.8)
    updated = adapter.load_frame("val-0")
    assert first.points is updated.points
    assert updated.filtered.output_count == 1
    assert base.load_calls == ["val-0", "val-1"]
    adapter.load_frame("val-2")
    adapter.load_frame("val-1")
    assert base.load_calls == ["val-0", "val-1", "val-2", "val-1"]
    assert not source.closed


def test_demo_composition_supplies_overlapping_candidates_and_preserves_ground_truth():
    base = create_dataset("demo")
    adapter = ReviewDatasetAdapter(base, demo=True)
    scenes = adapter.list_scenes()
    frame = adapter.load_frame(adapter.list_frames(scenes[0].token)[0].token)
    assert len(frame.predictions.detections) > len(frame.boxes)
    after_nms = refilter(frame, FilterConfig(score_threshold=0, nms_mode="class"))
    assert after_nms.filtered.output_count < frame.filtered.output_count
    assert after_nms.points is frame.points and after_nms.boxes is frame.boxes
    repeated = adapter.load_frame(frame.info.token)
    np.testing.assert_array_equal(repeated.points, frame.points)
    assert [d.identifier for d in repeated.predictions.detections] == [d.identifier for d in frame.predictions.detections]
