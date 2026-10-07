"""Native-table fixtures exercise calibration, camera ordering and failure paths."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cameraviz.dataloader import (
    CAMERA_LAYOUT, DETECTION_CLASSES, BoundingBox3D, CameraFrame, CameraInfo,
    DatasetAdapter, DatasetError, DemoAdapter, available_datasets, create_dataset,
    register_dataset,
)


@pytest.fixture
def nuscenes_root(tmp_path):
    root = tmp_path / "nuscenes"
    table_root = root / "v1.0-mini"
    table_root.mkdir(parents=True)
    s = np.sqrt(.5)
    tables = {
        "scene": [
            {"token": "scene-1", "name": "scene-001", "description": "A camera fixture", "nbr_samples": 99},
            {"token": "scene-empty", "name": "scene-002"},
        ],
        "sample": [
            {"token": "sample-2", "scene_token": "scene-1", "timestamp": 2_000_000},
            {"token": "sample-1", "scene_token": "scene-1", "timestamp": 1_000_000},
            {"token": "sample-without-camera", "scene_token": "scene-1", "timestamp": 500_000},
        ],
        "sensor": [
            {"token": "front", "channel": "CAM_FRONT", "modality": "camera"},
            {"token": "front-left", "channel": "CAM_FRONT_LEFT", "modality": "camera"},
            {"token": "back-right", "channel": "CAM_BACK_RIGHT", "modality": "camera"},
            {"token": "lidar", "channel": "LIDAR_TOP", "modality": "lidar"},
        ],
        "calibrated_sensor": [
            {"token": f"cal-{token}", "sensor_token": token, "translation": [1, 2, 3],
             "rotation": [s, s, 0, 0], "camera_intrinsic": [[200, 0, 4], [0, 250, 3], [0, 0, 1]]}
            for token in ("front", "front-left", "back-right", "lidar")
        ],
        "ego_pose": [
            {"token": "camera-pose-1", "translation": [10, 20, 30], "rotation": [s, 0, 0, s]},
            {"token": "camera-pose-2", "translation": [40, 50, 60], "rotation": [1, 0, 0, 0]},
        ],
        "sample_data": [
            {"token": "camera-front-2", "sample_token": "sample-2", "is_key_frame": True,
             "calibrated_sensor_token": "cal-front", "ego_pose_token": "camera-pose-2",
             "timestamp": 1_985_000, "filename": "samples/CAM_FRONT/second.png", "width": 8, "height": 6},
            {"token": "camera-front-1", "sample_token": "sample-1", "is_key_frame": True,
             "calibrated_sensor_token": "cal-front", "ego_pose_token": "camera-pose-1",
             "timestamp": 985_000, "filename": "samples/CAM_FRONT/first.png", "width": 8, "height": 6},
            {"token": "camera-left-1", "sample_token": "sample-1", "is_key_frame": True,
             "calibrated_sensor_token": "cal-front-left", "ego_pose_token": "camera-pose-1",
             "timestamp": 982_000, "filename": "samples/CAM_FRONT_LEFT/first.png", "width": 8, "height": 6},
            {"token": "camera-back-right-2", "sample_token": "sample-2", "is_key_frame": True,
             "calibrated_sensor_token": "cal-back-right", "ego_pose_token": "camera-pose-2",
             "timestamp": 1_987_000, "filename": "samples/CAM_BACK_RIGHT/second.png", "width": 8, "height": 6},
            {"token": "lidar-1", "sample_token": "sample-1", "is_key_frame": True,
             "calibrated_sensor_token": "cal-lidar", "filename": "unread-lidar.bin"},
            {"token": "ignored-sweep", "sample_token": "sample-1", "is_key_frame": False,
             "calibrated_sensor_token": "unused-calibration", "filename": "unread-sweep.jpg"},
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
             "translation": [15, 26, 38], "size": [.5, 1, 1.2], "rotation": [1, 0, 0, 0]},
        ],
    }
    for name, records in tables.items():
        (table_root / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")
    for data in tables["sample_data"]:
        if data.get("filename", "").endswith(".png"):
            path = root / data["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6), (30, 60, 90)).save(path)
    return root


def read_table(root: Path, name: str):
    return json.loads((root / "v1.0-mini" / f"{name}.json").read_text(encoding="utf-8"))


def rewrite_table(root: Path, name: str, records):
    (root / "v1.0-mini" / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")


def test_native_scene_and_frame_indices_and_partial_camera_grid(nuscenes_root):
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    assert isinstance(adapter, DatasetAdapter)
    scenes = adapter.list_scenes()
    assert [(scene.token, scene.frame_count) for scene in scenes] == [("scene-1", 2), ("scene-empty", 0)]
    assert scenes[0].description == "A camera fixture"
    assert [(frame.token, frame.index, frame.timestamp_us) for frame in adapter.list_frames("scene-1")] == [
        ("sample-1", 0, 1_000_000), ("sample-2", 1, 2_000_000),
    ]
    assert adapter.list_frames("scene-empty") == ()
    assert [info.channel for info in adapter.camera_layout] == [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ]
    assert [(info.row, info.column) for info in adapter.camera_layout] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert [camera.info.channel for camera in adapter.load_frame("sample-1").cameras] == ["CAM_FRONT_LEFT", "CAM_FRONT"]
    assert [camera.info.channel for camera in adapter.load_frame("sample-2").cameras] == ["CAM_FRONT", "CAM_BACK_RIGHT"]


def test_actual_camera_exposure_pose_and_global_annotation_transform(nuscenes_root):
    frame = create_dataset("nuscenes", root=nuscenes_root).load_frame("sample-1")
    camera = next(camera for camera in frame.cameras if camera.info.channel == "CAM_FRONT")
    assert camera.timestamp_us == 985_000
    assert camera.timestamp_us != frame.info.timestamp_us
    np.testing.assert_allclose(camera.sensor_to_world[:3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-12)
    np.testing.assert_allclose(camera.sensor_to_world[:3, 3], [8, 21, 33], atol=1e-12)
    np.testing.assert_array_equal(camera.intrinsics, [[200, 0, 4], [0, 250, 3], [0, 0, 1]])
    np.testing.assert_array_equal(camera.image[0, 0], [30, 60, 90])
    assert camera.image.shape == (6, 8, 3)
    car = frame.boxes[0]
    np.testing.assert_array_equal(car.center, [14, 25, 38])
    np.testing.assert_array_equal(car.size, [4, 2, 1.5])
    np.testing.assert_allclose(car.rotation, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)
    assert (car.class_name, car.token) == ("car", "annotation-car")
    assert frame.boxes[1].class_name == "human.pedestrian.stroller"
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    assert adapter.class_names[:10] == DETECTION_CLASSES
    assert "human.pedestrian.stroller" in adapter.class_names


def test_version_directory_input_and_images_inside_version(nuscenes_root):
    table_root = nuscenes_root / "v1.0-mini"
    assert create_dataset("nuscenes", root=table_root).load_frame("sample-1").cameras
    (nuscenes_root / "samples").rename(table_root / "samples")
    assert create_dataset("nuscenes", root=nuscenes_root).load_frame("sample-1").cameras
    assert create_dataset("nuscenes", root=table_root).load_frame("sample-1").cameras


@pytest.mark.parametrize("remove", [False, True])
def test_test_release_without_annotations(nuscenes_root, remove):
    for table in ("sample_annotation", "category", "instance"):
        if remove:
            (nuscenes_root / "v1.0-mini" / f"{table}.json").unlink()
        else:
            rewrite_table(nuscenes_root, table, [])
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    assert adapter.class_names == DETECTION_CLASSES
    assert adapter.load_frame("sample-1").boxes == ()


def test_camera_loading_lazy_missing_declared_image_reports_path(nuscenes_root):
    missing = nuscenes_root / "samples" / "CAM_FRONT" / "first.png"
    missing.unlink()
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    assert len(adapter.list_frames("scene-1")) == 2
    with pytest.raises(DatasetError, match="Extract the camera sensor files") as failure:
        adapter.load_frame("sample-1")
    assert str(missing) in str(failure.value)


def test_cache_reuses_images_and_is_bounded(nuscenes_root, monkeypatch):
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    real_open = Image.open
    opened = []

    def counting_open(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", counting_open)
    first = adapter.load_frame("sample-1")
    repeated = adapter.load_frame("sample-1")
    assert len(opened) == 2
    assert first.cameras[0].image is repeated.cameras[0].image
    for index in range(20):
        filename = f"samples/CAM_FRONT/extra-{index}.png"
        Image.new("RGB", (8, 6), (index, 0, 0)).save(nuscenes_root / filename)
        adapter._image(filename)
    assert len(adapter._image_cache) == 16
    assert not any(path.name == "first.png" for path in adapter._image_cache)


def test_invalid_camera_file_and_dimensions_are_actionable(nuscenes_root):
    filename = nuscenes_root / "samples" / "CAM_FRONT" / "first.png"
    filename.write_bytes(b"not an image")
    with pytest.raises(DatasetError, match="Cannot decode camera image"):
        create_dataset("nuscenes", root=nuscenes_root).load_frame("sample-1")
    Image.new("RGB", (10, 6)).save(filename)
    with pytest.raises(DatasetError, match="dimensions disagree"):
        create_dataset("nuscenes", root=nuscenes_root).load_frame("sample-1")


@pytest.mark.parametrize("name,change,match", [
    ("ego_pose", lambda data: data[0].update(rotation=[0, 0, 0, 0]), "nonzero norm"),
    ("calibrated_sensor", lambda data: data[0].update(camera_intrinsic=[[0, 0, 0]] * 3), "positive focal"),
    ("sample_annotation", lambda data: data[0].update(size=[0, 1, 1]), "dimensions must be positive"),
    ("sample_data", lambda data: data[1].update(timestamp=True), "integer timestamp"),
])
def test_invalid_sensor_and_annotation_metadata(nuscenes_root, name, change, match):
    records = read_table(nuscenes_root, name)
    change(records)
    rewrite_table(nuscenes_root, name, records)
    with pytest.raises(DatasetError, match=match):
        create_dataset("nuscenes", root=nuscenes_root).load_frame("sample-1")


def test_duplicate_camera_keyframe_rejected(nuscenes_root):
    records = read_table(nuscenes_root, "sample_data")
    records.append({**records[1], "token": "duplicate-camera"})
    rewrite_table(nuscenes_root, "sample_data", records)
    with pytest.raises(DatasetError, match="Multiple CAM_FRONT"):
        create_dataset("nuscenes", root=nuscenes_root)


def test_metadata_errors_and_unknown_selection(nuscenes_root):
    with pytest.raises(DatasetError, match="dataset root"):
        create_dataset("nuscenes")
    with pytest.raises(DatasetError, match="check --version"):
        create_dataset("nuscenes", root=nuscenes_root, version="v1.0-trainval")
    adapter = create_dataset("nuscenes", root=nuscenes_root)
    with pytest.raises(DatasetError, match="Unknown nuScenes scene"):
        adapter.list_frames("missing")
    with pytest.raises(DatasetError, match="Unknown nuScenes frame"):
        adapter.load_frame("missing")
    rewrite_table(nuscenes_root, "sample", {"not": "an array"})
    with pytest.raises(DatasetError, match="JSON array"):
        create_dataset("nuscenes", root=nuscenes_root)


def test_factory_registration_extensible_and_guards_invalid_constructors(monkeypatch):
    from cameraviz.dataloader import factory

    monkeypatch.setattr(factory, "_REGISTRY", dict(factory._REGISTRY))
    passed = {}

    def constructor(*, root, version):
        passed.update(root=root, version=version)
        return DemoAdapter()

    register_dataset("custom-fixture", constructor)
    assert "custom-fixture" in available_datasets()
    assert isinstance(create_dataset(" Custom-Fixture ", root="data", version="release"), DemoAdapter)
    assert passed == {"root": "data", "version": "release"}
    with pytest.raises(ValueError, match="already registered"):
        register_dataset("custom-fixture", constructor)
    with pytest.raises(ValueError, match="Available datasets"):
        create_dataset("missing")
    with pytest.raises(TypeError, match="callable"):
        register_dataset("invalid", None)
    register_dataset("bad-result", lambda **kwargs: object())
    with pytest.raises(TypeError, match="return a DatasetAdapter"):
        create_dataset("bad-result")


def test_demo_images_deterministic_calibrated_and_moving():
    adapter = create_dataset("demo")
    assert len(adapter.list_scenes()) == 2
    for scene in adapter.list_scenes():
        frames = adapter.list_frames(scene.token)
        assert len(frames) == scene.frame_count
        first = adapter.load_frame(frames[0].token)
        repeated = DemoAdapter().load_frame(frames[0].token)
        assert len(first.cameras) == 6
        assert first.cameras[0].image.shape == (540, 960, 3)
        assert {box.class_name for box in first.boxes} == set(DETECTION_CLASSES)
        for left, right in zip(first.cameras, repeated.cameras):
            np.testing.assert_array_equal(left.image, right.image)
            assert not left.image.flags.writeable
            assert left.info in CAMERA_LAYOUT
        following = adapter.load_frame(frames[1].token)
        assert not np.array_equal(first.boxes[0].center, following.boxes[0].center)
        assert not np.array_equal(first.cameras[0].image, following.cameras[0].image)
        # Every object is visible in at least one calibrated demo image.
        for box in first.boxes:
            visible = False
            for camera in first.cameras:
                optical = (box.center - camera.sensor_to_world[:3, 3]) @ camera.sensor_to_world[:3, :3]
                if optical[2] <= 0:
                    continue
                pixel_h = camera.intrinsics @ optical
                u, v = pixel_h[:2] / pixel_h[2]
                visible |= 0 <= u < camera.image.shape[1] and 0 <= v < camera.image.shape[0]
            assert visible, box.class_name


def test_models_freeze_metadata_and_protect_geometry_and_images():
    center = np.zeros(3)
    box = BoundingBox3D(center, np.ones(3), np.eye(3), "car")
    center[0] = 100
    assert box.center[0] == 0
    with pytest.raises(ValueError):
        box.center[0] = 1
    with pytest.raises(FrozenInstanceError):
        box.class_name = "truck"
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    camera = CameraFrame(CameraInfo("test", "Test", 0, 0), image, np.eye(3), np.eye(4), 0)
    image[:] = 255
    assert camera.image.max() == 0
    with pytest.raises(ValueError, match="proper rotation"):
        BoundingBox3D(np.zeros(3), np.ones(3), -np.eye(3), "car")
