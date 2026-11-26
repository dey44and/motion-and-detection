"""Streaming, invalidation, and atomic publication of the nuScenes index."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from threading import Event

import pytest

from visual_utils import DatasetValidationError
from visual_utils.nuscenes import metadata


@pytest.fixture
def tables(tmp_path, monkeypatch):
    root = tmp_path / "dataset with spaces" / "v1.0-mini"
    root.mkdir(parents=True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache with spaces"))
    samples = {"one": "A", "two": "B", "three": "A"}
    records = {
        "sample": [{"token": token, "scene_token": scene} for token, scene in samples.items()],
        "ego_pose": [{"token": "pose2", "timestamp": 2}, {"token": "pose1", "timestamp": 1}],
        "sample_data": [
            {"token": "camera", "sample_token": "two", "is_key_frame": True, "calibrated_sensor_token": "b"},
            {"token": "scan", "sample_token": "one", "is_key_frame": True, "calibrated_sensor_token": "a"},
            {"token": "sweep", "sample_token": "three", "is_key_frame": False, "calibrated_sensor_token": "c"},
            {"token": "linked", "sample_token": "unknown", "calibrated_sensor_token": []},
        ],
        "sample_annotation": [
            {"token": "box2", "sample_token": "one", "label": "car"},
            {"token": "box1", "sample_token": "one", "label": "bus"},
            {"token": "box3", "sample_token": "two", "label": "car"},
        ],
        "instance": [],
        "category": [],
    }
    for name, values in records.items():
        prefix = " \n" * 5_000 if name == "sample_data" else ""
        (root / f"{name}.json").write_text(prefix + json.dumps(values), encoding="utf-8")
    return root, samples, tmp_path / "cache with spaces" / "visual-utils" / "nuscenes"


def annotation_class(record):
    return record["label"]


def test_index_access_and_warm_reuse(tables, monkeypatch):
    root, samples, cache = tables
    updates = []
    snapshot = metadata.metadata_source_snapshot(root)
    with closing(metadata.open_metadata_index(root, samples, annotation_class, progress=updates.append, source_snapshot=snapshot)) as index:
        path = index.path
        assert list(index.poses) == ["pose2", "pose1"]
        assert index.poses["pose1"]["timestamp"] == 1
        assert index.poses.get("missing") is None
        assert len(index.sample_data) == 4
        assert index.sample_data["linked"]["calibrated_sensor_token"] == []
        assert [row["token"] for row in index.keyframes()] == ["camera", "scan"]
        assert [row["token"] for row in index.scene_sample_data("A")] == ["scan", "sweep"]
        assert list(index.scene_sample_data("unknown")) == []
        assert index.calibration_tokens() == ("a", "b", "c")
        assert index.annotation_classes() == ("bus", "car")
        assert [(row["token"], name) for row, name in index.annotations("one")] == [("box2", "car"), ("box1", "bus")]
        assert index.annotations("three") == ()
        with ThreadPoolExecutor(max_workers=1) as worker:
            assert worker.submit(lambda: index.sample_data["sweep"]["sample_token"]).result(timeout=5) == "three"
    assert list(cache.glob("*.sqlite3")) == [path]
    assert updates == ["Indexing nuScenes ego_pose", "Indexing nuScenes sample_data", "Indexing nuScenes sample_annotation", "Finalizing nuScenes metadata index"]

    def forbidden(*args):
        pytest.fail("A warm lookup must not parse the source tables.")

    monkeypatch.setattr(metadata, "_records", forbidden)
    with closing(metadata.open_metadata_index(root, samples, forbidden)) as reused:
        assert reused.path == path
        assert reused.sample_data["camera"]["sample_token"] == "two"


def test_source_snapshot_and_optional_annotation_invalidation(tables):
    root, samples, cache = tables
    original = metadata.metadata_source_snapshot(root)
    labels = root / "sample_annotation.json"
    labels.unlink()
    with pytest.raises(DatasetValidationError, match="changed while the dataset was opening"):
        metadata.open_metadata_index(root, samples, annotation_class, source_snapshot=original)
    with closing(metadata.open_metadata_index(root, samples, annotation_class)) as first:
        assert first.annotations("one") == ()
        labels.write_text('[{"sample_token":"one","label":"new"}]', encoding="utf-8")
        with closing(metadata.open_metadata_index(root, samples, annotation_class)) as second:
            assert second.path != first.path
            assert second.annotation_classes() == ("new",)
            assert first.annotations("one") == ()

    before = metadata.metadata_source_snapshot(root)
    (root / "category.json").write_text('[{"token":"new"}]', encoding="utf-8")
    assert metadata.metadata_source_snapshot(root) != before
    assert len(list(cache.glob("*.sqlite3"))) == 2


@pytest.mark.parametrize("values", [
    '{}', '{"item": {"token": "a"}}', '[3]', '[] false', '[{"token":"a"}',
    '[{"token":', '[{"token":"a"}, {"token":"a"}]',
])
def test_malformed_or_duplicate_data_never_publishes_a_cache(tables, values):
    root, samples, cache = tables
    (root / "ego_pose.json").write_text(values, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="ego_pose"):
        metadata.open_metadata_index(root, samples, annotation_class)
    assert not list(cache.glob("*.sqlite3"))
    assert not list(cache.glob("*.partial*"))


def test_mutating_source_mid_build_leaves_no_cache(tables):
    root, samples, cache = tables

    def progress(message):
        if message == "Indexing nuScenes sample_annotation":
            (root / "category.json").write_text('[{"token":"new"}]', encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="changed during indexing"):
        metadata.open_metadata_index(root, samples, annotation_class, progress=progress)
    assert not list(cache.glob("*.sqlite3"))
    assert not list(cache.glob("*.partial*"))


def test_interrupted_build_releases_lock_and_removes_partial_files(tables):
    root, samples, cache = tables

    def cancel(message):
        if message == "Indexing nuScenes sample_data":
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        metadata.open_metadata_index(root, samples, annotation_class, progress=cancel)
    assert not list(cache.glob("*.partial*"))
    assert not list(cache.glob("*.sqlite3"))
    with closing(metadata.open_metadata_index(root, samples, annotation_class)) as index:
        assert len(index.poses) == 2


def test_two_callers_share_one_build(tables):
    root, samples, cache = tables
    started, waiting, release = Event(), Event(), Event()

    def first(message):
        if message == "Indexing nuScenes ego_pose":
            started.set()
            assert release.wait(5)

    def second(message):
        assert message == "Waiting for the nuScenes metadata index"
        waiting.set()

    def build(progress):
        with closing(metadata.open_metadata_index(root, samples, annotation_class, progress=progress)) as index:
            return index.path

    with ThreadPoolExecutor(max_workers=2) as workers:
        one = workers.submit(build, first)
        assert started.wait(5)
        two = workers.submit(build, second)
        try:
            assert waiting.wait(5)
        finally:
            release.set()
        assert one.result(timeout=5) == two.result(timeout=5)
    assert len(list(cache.glob("*.sqlite3"))) == 1


def test_invalid_cache_rebuild_and_unavailable_cache_error(tables, tmp_path, monkeypatch):
    root, samples, _ = tables
    with closing(metadata.open_metadata_index(root, samples, annotation_class)) as index:
        path = index.path
    path.write_bytes(b"incomplete metadata cache")
    with closing(metadata.open_metadata_index(root, samples, annotation_class)) as rebuilt:
        assert rebuilt.path == path
        assert rebuilt.annotations("one")[0][1] == "car"

    invalid_home = tmp_path / "cache file"
    invalid_home.write_bytes(b"")
    monkeypatch.setenv("XDG_CACHE_HOME", str(invalid_home))
    with pytest.raises(metadata.MetadataCacheError, match="Cannot access nuScenes metadata cache"):
        metadata.open_metadata_index(root, samples, annotation_class)
