"""Prediction imports preserve identity, coordinates, coverage, and saved stage."""

from __future__ import annotations

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from threedviz.dataloader import DemoLoader
from visual_utils import Box3D, DatasetValidationError, FrameInfo, Modality, PointCloudFrame

from detviz.predictions import (
    DemoPredictionSource, Detection, InMemoryPredictionSource, PredictionCancelled, PredictionError,
    PredictionFrame, PredictionInfo, available_prediction_formats,
    make_demo_predictions, normalize_frame, open_predictions,
    register_prediction_format,
)


def make_box(center, size, rotation, class_id, annotation_id="box"):
    return Box3D(annotation_id, class_id, center, size, rotation)


def annotation(token="sample-a", **changes):
    result = {
        "sample_token": token, "translation": [10, 20, 3], "size": [2, 4, 1.5],
        "rotation": [np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
        "detection_name": "car", "detection_score": 0.8, "velocity": [2, 3],
    }
    result.update(changes)
    return result


@pytest.fixture
def json_path(tmp_path):
    path = tmp_path / "results_nusc.json"
    path.write_text(json.dumps({
        "results": {"sample-a": [annotation()], "sample-empty": []},
        "meta": {"use_lidar": True},
    }), encoding="utf-8")
    return path


def open_json(path, tmp_path, **kwargs):
    return open_predictions(path, cache_dir=tmp_path / "cache", **kwargs)


def pcdet_entry(token="sample-a", names=True):
    entry = {
        "boxes_lidar": np.array([[1, 2, 3, 4, 2, 1.5, np.pi / 2, 2, 3]], dtype=np.float32),
        "score": np.array([0.75], dtype=np.float32),
        "pred_labels": np.array([2], dtype=np.int64),
        "metadata": {"token": token, "tf_glob_from_lidar": np.eye(4)},
        "frame_id": np.str_("filename.pcd"),
    }
    if names:
        entry["name"] = np.array(["car"])
    return entry


def write_pickle(tmp_path, entries, protocol=4):
    path = tmp_path / "result.pkl"
    with path.open("wb") as stream:
        pickle.dump(entries, stream, protocol=protocol)
    return path


def test_official_json_world_dimensions_rotation_identity_and_empty_coverage(json_path, tmp_path):
    updates = []
    source = open_json(json_path, tmp_path, progress=lambda fraction, message: updates.append((fraction, message)))
    assert source.info.coordinate_frame == "world"
    assert source.info.stage == "unknown"
    assert source.info.format == "nuscenes_json"
    assert "already contain" in source.info.metadata["producer_note"]
    assert source.frame_tokens == frozenset({"sample-a", "sample-empty"})
    assert source.class_names == ("car",)
    detection = source.get_frame("sample-a").detections[0]
    np.testing.assert_allclose(detection.box.size, [4, 2, 1.5])
    np.testing.assert_allclose(detection.box.rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(detection.velocity, [2, 3, 0])
    assert detection.identifier == "sample-a:0"
    assert detection.score == 0.8
    assert source.get_frame("sample-empty") == PredictionFrame("sample-empty", ())
    assert source.get_frame("not-covered") is None
    assert updates[0][0] == 0 and updates[-1][0] == 1
    assert all(0 <= fraction <= 1 for fraction, _ in updates)
    assert source.cache_path.is_file()
    source.close()
    source.close()
    with pytest.raises(PredictionError, match="closed"):
        source.get_frame("sample-a")


def test_embedded_metadata_after_results_and_sidecar_override(json_path, tmp_path):
    data = json.loads(json_path.read_text())
    data["meta"]["detviz"] = {"stage": "pre_nms", "name": "Raw model"}
    json_path.write_text(json.dumps(data))
    source = open_json(json_path, tmp_path)
    assert source.info.stage == "pre_nms"
    assert source.info.name == "Raw model"
    source.close()
    sidecar = json_path.with_suffix(".detviz.json")
    sidecar.write_text(json.dumps({"stage": "post_nms", "name": "Final model"}))
    source = open_json(json_path, tmp_path)
    assert source.info.stage == "post_nms"
    assert source.info.name == "Final model"


@pytest.mark.parametrize("changes, message", [
    ({"detection_score": 1.01}, "score"),
    ({"detection_score": float("nan")}, "score|import"),
    ({"translation": [0, 1, float("inf")]}, "finite|import"),
    ({"size": [2, 0, 1]}, "positive"),
    ({"rotation": [0, 0, 0, 0]}, "quaternion"),
    ({"sample_token": "wrong"}, "inconsistent"),
    ({"detection_name": ""}, "class name"),
    ({"velocity": [1, 2, 3]}, "velocity"),
])
def test_malformed_json_detections_fail_without_partial_cache(tmp_path, changes, message):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"results": {"sample-a": [annotation(**changes)]}}))
    with pytest.raises(PredictionError, match=message):
        open_json(path, tmp_path)
    assert not list((tmp_path / "cache").glob("*.sqlite3"))
    assert not list((tmp_path / "cache").glob("*.tmp"))


@pytest.mark.parametrize("text", [
    '{}', '{"results": []}', '{"results":{"sample-a":null}}',
    '{"results":{"sample-a":[],"sample-a":[]}}',
    '{"results":{},"results":{}}', '{"results":{"sample-a":[]}',
])
def test_invalid_json_structure_and_duplicate_frames_are_rejected(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(PredictionError):
        open_json(path, tmp_path)


def test_cache_reused_without_parsing_again_and_invalidated_by_input(json_path, tmp_path, monkeypatch):
    from detviz.predictions import nuscenes_json

    source = open_json(json_path, tmp_path)
    cache = source.cache_path
    source.close()
    original = nuscenes_json._stream_frames

    def no_reimport(*args, **kwargs):
        raise AssertionError("A valid cache must not parse the JSON again.")

    monkeypatch.setattr(nuscenes_json, "_stream_frames", no_reimport)
    cached = open_json(json_path, tmp_path)
    assert cached.cache_path == cache
    assert cached.get_frame("sample-a").detections[0].score == 0.8
    cached.close()
    monkeypatch.setattr(nuscenes_json, "_stream_frames", original)
    data = json.loads(json_path.read_text())
    data["results"]["sample-a"][0]["detection_score"] = 0.95
    json_path.write_text(json.dumps(data))
    changed = open_json(json_path, tmp_path)
    assert changed.cache_path != cache
    assert changed.get_frame("sample-a").detections[0].score == 0.95


def test_cache_hit_does_not_open_the_export_payload(json_path, tmp_path, monkeypatch):
    from pathlib import Path
    with open_json(json_path, tmp_path):
        pass
    original = Path.open
    def guarded_open(path, *args, **kwargs):
        if path == json_path:
            raise AssertionError("Cached export payload must not be reread.")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    with open_json(json_path, tmp_path) as cached:
        assert cached.get_frame("sample-a").detections[0].score == .8


def test_cancelled_import_leaves_no_published_or_partial_cache(json_path, tmp_path):
    state = {"cancelled": False}

    def progress(fraction, message):
        state["cancelled"] = True

    with pytest.raises(PredictionCancelled):
        open_json(json_path, tmp_path, progress=progress, cancelled=lambda: state["cancelled"])
    assert not list((tmp_path / "cache").iterdir())


def test_cached_frames_can_be_read_from_multiple_threads(json_path, tmp_path):
    source = open_json(json_path, tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(source.get_frame, ["sample-a"] * 20))
    assert all(record.detections[0].identifier == "sample-a:0" for record in records)


@pytest.mark.parametrize("protocol", [4, 5])
def test_numpy_pickle_export_uses_token_not_filename_and_optional_velocity(tmp_path, protocol):
    entry = pcdet_entry()
    empty = {"boxes_lidar": np.empty((0, 7)), "score": np.empty(0), "metadata": {"token": "empty"}}
    path = write_pickle(tmp_path, [entry, empty], protocol)
    source = open_json(path, tmp_path)
    assert source.info.coordinate_frame == "lidar" and source.info.stage == "unknown"
    assert source.frame_tokens == frozenset({"sample-a", "empty"})
    detection = source.get_frame("sample-a").detections[0]
    np.testing.assert_allclose(detection.box.center, [1, 2, 3])
    np.testing.assert_allclose(detection.box.size, [4, 2, 1.5])
    np.testing.assert_allclose(detection.box.rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
    np.testing.assert_allclose(detection.velocity, [2, 3, 0])
    assert source.get_frame("empty").detections == ()


def test_numeric_labels_require_explicit_mapping_and_are_one_based(tmp_path):
    path = write_pickle(tmp_path, [pcdet_entry(names=False)])
    with pytest.raises(PredictionError, match="explicit class_names"):
        open_json(path, tmp_path)
    sidecar = type(path)(str(path) + ".detviz.json")
    sidecar.write_text(json.dumps({"class_names": ["truck", "car"], "stage": "pre_nms"}))
    source = open_json(path, tmp_path)
    assert source.get_frame("sample-a").detections[0].box.class_id == "car"
    assert source.info.stage == "pre_nms"
    bad = pcdet_entry(names=False)
    bad["pred_labels"][0] = 0
    path = write_pickle(tmp_path, [bad])
    with pytest.raises(PredictionError, match="1-based"):
        open_json(path, tmp_path)


def test_pickle_envelope_metadata_and_seven_value_boxes(tmp_path):
    entry = pcdet_entry(names=False)
    entry["boxes_lidar"] = entry["boxes_lidar"][:, :7]
    path = write_pickle(tmp_path, {"frames": [entry], "metadata": {"class_names": ["bus", "car"], "stage": "pre_nms"}})
    source = open_json(path, tmp_path)
    assert source.get_frame("sample-a").detections[0].velocity is None
    assert source.info.stage == "pre_nms"


def test_pickle_arbitrary_globals_are_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"

    class ArbitraryObject:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

    path = write_pickle(tmp_path, [ArbitraryObject()])
    with pytest.raises(PredictionError, match="Unsupported pickle global"):
        open_json(path, tmp_path)
    assert not marker.exists()


@pytest.mark.parametrize("change", ["missing_token", "bad_shape", "bad_score", "duplicate"])
def test_malformed_pickle_frames_are_rejected(tmp_path, change):
    entry = pcdet_entry()
    entries = [entry]
    if change == "missing_token":
        entry["metadata"] = {}
    elif change == "bad_shape":
        entry["boxes_lidar"] = np.zeros((1, 8))
    elif change == "bad_score":
        entry["score"] = np.ones(2)
    else:
        entries.append(entry)
    path = write_pickle(tmp_path, entries)
    with pytest.raises(PredictionError):
        open_json(path, tmp_path)


def test_normalization_preserves_full_rotation_and_rotates_velocity():
    sensor_to_world = np.eye(4)
    sensor_to_world[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    sensor_to_world[:3, 3] = [10, -20, 30]
    local_center, local_velocity = np.array([1, 2, 3]), np.array([2, -1, 4])
    local_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotation = sensor_to_world[:3, :3]
    box = make_box(rotation @ local_center + sensor_to_world[:3, 3], np.array([4, 2, 1]), rotation @ local_rotation, "car")
    predictions = PredictionFrame("frame", (Detection(box, 0.5, "frame:0", rotation @ local_velocity),))
    frame = PointCloudFrame(FrameInfo("frame", "scene", 0, 0, {Modality.POINT_CLOUD}), "test", "reference", np.empty((0, 3)), world_frame_id="world", T_world_from_reference=sensor_to_world)
    normalized = normalize_frame(predictions, frame, coordinate_frame="world")
    result = normalized.detections[0]
    np.testing.assert_allclose(result.box.center, local_center)
    np.testing.assert_allclose(result.box.rotation, local_rotation)
    np.testing.assert_allclose(result.velocity, local_velocity)
    assert result.identifier == "frame:0" and result.score == 0.5
    np.testing.assert_array_equal(result.box.size, box.size)
    assert not result.box.size.flags.writeable
    with pytest.raises(PredictionError, match="no T_world_from_reference"):
        normalize_frame(predictions, replace(frame, world_frame_id=None, T_world_from_reference=None), coordinate_frame="world")
    with pytest.raises(PredictionError, match="frame IDs"):
        normalize_frame(predictions, replace(frame, info=replace(frame.info, frame_id="other")))


def test_detection_payloads_are_copied_and_immutable():
    center = np.array([1., 2., 3.])
    detection = Detection(make_box(center, np.ones(3), np.eye(3), "car"), 0.5, "sample:0")
    center[:] = 999
    np.testing.assert_array_equal(detection.box.center, [1, 2, 3])
    with pytest.raises(ValueError):
        detection.box.center[0] = 0
    with pytest.raises(DatasetValidationError, match="rotation"):
        Detection(make_box(center, np.ones(3), np.diag([1, 1, -1]), "car"), 0.5, "sample:0")


def test_demo_predictions_are_deterministic_with_overlap_and_scores():
    dataset = DemoLoader()
    token = dataset.list_frames(dataset.list_scenes()[0].scene_id, modality=Modality.POINT_CLOUD)[0].frame_id
    frame = dataset.load_point_cloud(token)
    predictions = make_demo_predictions(frame)
    assert len(predictions.detections) > len(frame.annotations.boxes)
    assert len({detection.score for detection in predictions.detections}) > 1
    repeated = make_demo_predictions(frame)
    np.testing.assert_array_equal(predictions.detections[0].box.center, repeated.detections[0].box.center)
    source = DemoPredictionSource(dataset)
    assert token in source.frame_tokens
    assert source.get_frame("absent") is None
    assert source.info.stage == "pre_nms"


def test_registry_supports_additional_prediction_sources(tmp_path, monkeypatch):
    from detviz.predictions import factory

    monkeypatch.setattr(factory, "_REGISTRY", dict(factory._REGISTRY))
    source = DemoPredictionSource(DemoLoader())
    path = tmp_path / "custom.format"
    path.touch()
    register_prediction_format("custom", lambda *args, **kwargs: source)
    assert "custom" in available_prediction_formats()
    assert open_predictions(path, format="custom") is source
    with pytest.raises(ValueError, match="already registered"):
        register_prediction_format("custom", lambda: source)
    with pytest.raises(PredictionError, match="infer"):
        open_predictions(path)


def test_incompatible_coordinate_declaration_and_invalid_stage_rejected(json_path, tmp_path):
    sidecar = type(json_path)(str(json_path) + ".detviz.json")
    sidecar.write_text(json.dumps({"coordinate_frame": "lidar"}))
    with pytest.raises(PredictionError, match="incompatible"):
        open_json(json_path, tmp_path)
    sidecar.write_text(json.dumps({"stage": "guessed"}))
    with pytest.raises(PredictionError, match="stage"):
        open_json(json_path, tmp_path)


@pytest.mark.parametrize("name", ["auto", "json", "nuscenes", "pickle", "pcdet", " JSON "])
def test_prediction_aliases_cannot_hide_registered_openers(name):
    with pytest.raises(ValueError, match="reserved"):
        register_prediction_format(name, lambda *args, **kwargs: None)


def test_memory_source_preserves_empty_coverage_identity_and_stable_classes():
    frame = PredictionFrame("custom/frame", (
        Detection(make_box(np.zeros(3), np.ones(3), np.eye(3), "cone"), .8, "one"),
        Detection(make_box(np.zeros(3), np.ones(3), np.eye(3), "car"), .4, "two"),
    ))
    empty = PredictionFrame("custom/empty", ())
    info = PredictionInfo("In-process detector", "memory", "lidar", "pre_nms")
    with InMemoryPredictionSource(iter([frame, empty]), info=info) as source:
        assert source.info is info
        assert source.class_names == ("cone", "car")
        assert source.frame_tokens == {"custom/frame", "custom/empty"}
        assert source.get_frame("custom/frame") is frame
        assert source.get_frame("custom/empty") is empty
        assert source.get_frame("absent") is None
    source.close()
    assert source.frame_tokens == {"custom/frame", "custom/empty"}
    with pytest.raises(PredictionError, match="closed"):
        source.get_frame("custom/frame")


def test_memory_source_rejects_duplicate_frame_ids():
    frame = PredictionFrame("duplicate", ())
    with pytest.raises(PredictionError, match="Duplicate"):
        InMemoryPredictionSource([frame, frame], info=PredictionInfo("test", "memory", "lidar"))
