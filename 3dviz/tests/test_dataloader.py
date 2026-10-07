"""Exercise the native raw JSON format and transformations without downloads."""

import json
from pathlib import Path

import numpy as np
import pytest

from threedviz.dataloader import (
    DETECTION_CLASSES,
    DatasetAdapter,
    DatasetError,
    DemoAdapter,
    available_datasets,
    create_dataset,
    register_dataset,
)


@pytest.fixture
def nuscenes_root(tmp_path):
    root = tmp_path / "nuscenes"
    table_root = root / "v1.0-mini"
    table_root.mkdir(parents=True)
    s = np.sqrt(0.5)
    tables = {
        "scene": [
            {"token": "scene-1", "name": "scene-001", "description": "A tiny fixture", "nbr_samples": 99},
            {"token": "scene-empty", "name": "scene-002"},
        ],
        # No devkit-enriched sample.data or sample.anns keys.
        "sample": [
            {"token": "sample-2", "scene_token": "scene-1", "timestamp": 2_000_000},
            {"token": "sample-1", "scene_token": "scene-1", "timestamp": 1_000_000},
            {"token": "sample-no-lidar", "scene_token": "scene-1", "timestamp": 500_000},
        ],
        "sensor": [
            {"token": "lidar", "channel": "LIDAR_TOP", "modality": "lidar"},
            {"token": "radar", "channel": "RADAR_FRONT", "modality": "radar"},
        ],
        "calibrated_sensor": [
            {"token": "cal-lidar", "sensor_token": "lidar", "translation": [1, 2, 3], "rotation": [s, s, 0, 0]},
            {"token": "cal-radar", "sensor_token": "radar", "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]},
        ],
        "ego_pose": [{"token": "pose", "translation": [10, 20, 30], "rotation": [s, 0, 0, s]}],
        "sample_data": [
            {"token": "lidar-2", "sample_token": "sample-2", "is_key_frame": True,
             "calibrated_sensor_token": "cal-lidar", "ego_pose_token": "pose",
             "filename": "samples/LIDAR_TOP/second.bin"},
            {"token": "radar-1", "sample_token": "sample-1", "is_key_frame": True,
             "calibrated_sensor_token": "cal-radar", "ego_pose_token": "pose", "filename": "unread-radar.pcd"},
            {"token": "lidar-1", "sample_token": "sample-1", "is_key_frame": True,
             "calibrated_sensor_token": "cal-lidar", "ego_pose_token": "pose",
             "filename": "samples/LIDAR_TOP/first.bin"},
            {"token": "sweep", "sample_token": "sample-1", "is_key_frame": False,
             "calibrated_sensor_token": "unused-sweep-calibration", "filename": "unread-sweep.bin"},
        ],
        "category": [
            {"token": "category-car", "name": "vehicle.car"},
            {"token": "category-stroller", "name": "human.pedestrian.stroller"},
        ],
        "instance": [
            {"token": "instance-car", "category_token": "category-car"},
            {"token": "instance-stroller", "category_token": "category-stroller"},
        ],
        "sample_annotation": [
            {"token": "annotation-car", "sample_token": "sample-1", "instance_token": "instance-car",
             "translation": [14, 25, 38], "size": [2, 4, 1.5], "rotation": [s, 0, s, 0]},
            {"token": "annotation-stroller", "sample_token": "sample-1", "instance_token": "instance-stroller",
             "translation": [14, 25, 38], "size": [0.5, 1, 1.2], "rotation": [1, 0, 0, 0]},
        ],
    }
    for name, records in tables.items():
        (table_root / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")
    points_root = root / "samples" / "LIDAR_TOP"
    points_root.mkdir(parents=True)
    scan = np.array([[1, 2, 3, 44, 5], [-4, 5, -6, 99, 2]], dtype="<f4")
    scan.tofile(points_root / "first.bin")
    scan.tofile(points_root / "second.bin")
    return root


def rewrite_table(root: Path, name: str, records) -> None:
    (root / "v1.0-mini" / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")


def test_native_tables_scene_and_frame_indices(nuscenes_root):
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    assert isinstance(adapter, DatasetAdapter)
    scenes = adapter.list_scenes()
    assert [(scene.token, scene.frame_count) for scene in scenes] == [("scene-1", 2), ("scene-empty", 0)]
    assert scenes[0].description == "A tiny fixture"
    frames = adapter.list_frames("scene-1")
    assert [(frame.token, frame.index, frame.timestamp_us) for frame in frames] == [
        ("sample-1", 0, 1_000_000), ("sample-2", 1, 2_000_000),
    ]
    assert adapter.list_frames("scene-empty") == ()
    assert len(adapter.load_frame("sample-2").boxes) == 0


def test_full_global_to_ego_to_lidar_transform_and_dimensions(nuscenes_root):
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    frame = adapter.load_frame("sample-1")
    assert frame.info.token == "sample-1"
    np.testing.assert_array_equal(frame.points, [[1, 2, 3], [-4, 5, -6]])
    np.testing.assert_array_equal(frame.intensity, [44, 99])
    assert frame.points.flags.c_contiguous
    assert frame.intensity.flags.c_contiguous
    car = frame.boxes[0]
    np.testing.assert_allclose(car.center, [4, 5, 6], atol=1e-12)
    np.testing.assert_array_equal(car.size, [4, 2, 1.5])
    np.testing.assert_allclose(car.rotation, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert car.class_name == "car"
    assert car.token == "annotation-car"
    assert frame.boxes[1].class_name == "human.pedestrian.stroller"
    assert adapter.class_names[:10] == DETECTION_CLASSES
    assert "human.pedestrian.stroller" in adapter.class_names


def test_version_directory_input_and_sensor_files_inside_version(nuscenes_root):
    table_root = nuscenes_root / "v1.0-mini"
    assert create_dataset("nuscenes", root=table_root).load_frame("sample-1").points.shape == (2, 3)
    (nuscenes_root / "samples").rename(table_root / "samples")
    assert create_dataset("nuscenes", root=nuscenes_root).load_frame("sample-1").points.shape == (2, 3)
    assert create_dataset("nuscenes", root=table_root).load_frame("sample-1").points.shape == (2, 3)


@pytest.mark.parametrize("missing_tables", [False, True])
def test_test_release_without_ground_truth(nuscenes_root, missing_tables):
    for table in ("sample_annotation", "instance", "category"):
        if missing_tables:
            (nuscenes_root / "v1.0-mini" / f"{table}.json").unlink()
        else:
            rewrite_table(nuscenes_root, table, [])
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    assert adapter.load_frame("sample-1").boxes == ()
    assert adapter.class_names == DETECTION_CLASSES


def test_lidar_loading_is_lazy_and_missing_file_error_is_actionable(nuscenes_root):
    path = nuscenes_root / "samples" / "LIDAR_TOP" / "first.bin"
    path.unlink()
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    assert len(adapter.list_frames("scene-1")) == 2
    with pytest.raises(DatasetError, match="Extract the sensor files"):
        adapter.load_frame("sample-1")


@pytest.mark.parametrize("data", [b"", b"\x00" * 19, b"\x00" * 21])
def test_truncated_or_empty_lidar_is_rejected(nuscenes_root, data):
    (nuscenes_root / "samples" / "LIDAR_TOP" / "first.bin").write_bytes(data)
    with pytest.raises(DatasetError, match="20 bytes per point"):
        create_dataset("nuscenes", root=nuscenes_root).load_frame("sample-1")


def test_nonfinite_lidar_and_zero_quaternion_are_rejected(nuscenes_root):
    path = nuscenes_root / "samples" / "LIDAR_TOP" / "first.bin"
    np.array([[np.nan, 0, 0, 0, 0]], dtype="<f4").tofile(path)
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    with pytest.raises(DatasetError, match="NaN/Inf"):
        adapter.load_frame("sample-1")
    np.zeros((1, 5), dtype="<f4").tofile(path)
    rewrite_table(nuscenes_root, "ego_pose", [{"token": "pose", "rotation": [0, 0, 0, 0], "translation": [0, 0, 0]}])
    with pytest.raises(DatasetError, match="nonzero norm"):
        create_dataset("nuscenes", root=nuscenes_root).load_frame("sample-1")


def test_metadata_errors_are_actionable(nuscenes_root):
    with pytest.raises(DatasetError, match="dataset root"):
        create_dataset("nuscenes")
    with pytest.raises(DatasetError, match="check --version"):
        create_dataset("nuscenes", root=nuscenes_root, version="v1.0-trainval")
    rewrite_table(nuscenes_root, "sample", {"wrong": "format"})
    with pytest.raises(DatasetError, match="JSON array"):
        create_dataset("nuscenes", root=nuscenes_root)


def test_unknown_scene_and_frame(nuscenes_root):
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    with pytest.raises(DatasetError, match="Unknown nuScenes scene"):
        adapter.list_frames("missing")
    with pytest.raises(DatasetError, match="Unknown nuScenes frame"):
        adapter.load_frame("missing")


def test_factory_registry_is_extensible(monkeypatch):
    from threedviz.dataloader import factory

    monkeypatch.setattr(factory, "_REGISTRY", dict(factory._REGISTRY))
    passed = {}

    def make_adapter(*, root, version):
        passed.update(root=root, version=version)
        return DemoAdapter()

    register_dataset("custom-fixture", make_adapter)
    assert "custom-fixture" in available_datasets()
    assert isinstance(create_dataset(" Custom-Fixture ", root="data", version="release"), DemoAdapter)
    assert passed == {"root": "data", "version": "release"}
    with pytest.raises(ValueError, match="already registered"):
        register_dataset("custom-fixture", make_adapter)
    with pytest.raises(ValueError, match="Available datasets"):
        create_dataset("nonexistent")
    with pytest.raises(TypeError, match="callable"):
        register_dataset("broken", None)


def test_demo_is_deterministic_and_spans_all_detection_classes():
    adapter = create_dataset("demo")
    seen_classes = set()
    for scene in adapter.list_scenes():
        frames = adapter.list_frames(scene.token)
        assert len(frames) == scene.frame_count
        first = adapter.load_frame(frames[0].token)
        repeated = create_dataset("demo").load_frame(frames[0].token)
        np.testing.assert_array_equal(first.points, repeated.points)
        np.testing.assert_array_equal(first.intensity, repeated.intensity)
        assert first.points.shape[1] == 3
        assert first.intensity.shape == (len(first.points),)
        assert np.isfinite(first.points).all()
        seen_classes.update(box.class_name for box in first.boxes)
        following = adapter.load_frame(frames[1].token)
        assert not np.array_equal(first.boxes[0].center, following.boxes[0].center)
    assert seen_classes == set(DETECTION_CLASSES)
