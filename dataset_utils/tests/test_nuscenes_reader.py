"""Exercise the shared reader without either visualization application."""

import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from dataset_utils.errors import DatasetError
from dataset_utils.models import PointCloudFrame
from dataset_utils.nuscenes import NuScenesCameraFrame, NuScenesReader


@pytest.fixture
def native_root(tmp_path):
    root = tmp_path / "nuscenes"
    metadata = root / "v1.0-mini"
    metadata.mkdir(parents=True)
    tables = {
        "scene": [{"token": "scene", "name": "scene-0103"}],
        "sample": [
            {"token": "both", "scene_token": "scene", "timestamp": 1_000_000},
            {"token": "camera-only", "scene_token": "scene", "timestamp": 2_000_000},
            {"token": "lidar-only", "scene_token": "scene", "timestamp": 3_000_000},
        ],
        "sensor": [
            {"token": "lidar", "channel": "LIDAR_TOP", "modality": "lidar"},
            {"token": "camera", "channel": "CAM_FRONT", "modality": "camera"},
        ],
        "calibrated_sensor": [
            {"token": "lidar-cal", "sensor_token": "lidar", "rotation": [1, 0, 0, 0], "translation": [1, 2, 3]},
            {"token": "camera-cal", "sensor_token": "camera", "rotation": [1, 0, 0, 0], "translation": [0, 0, 0],
             "camera_intrinsic": [[10, 0, 2], [0, 10, 2], [0, 0, 1]]},
        ],
        "ego_pose": [
            {"token": "pose-before", "timestamp": 950_000, "rotation": [1, 0, 0, 0], "translation": [9, 0, 0]},
            {"token": "pose-lidar", "timestamp": 1_000_000, "rotation": [1, 0, 0, 0], "translation": [10, 0, 0]},
            {"token": "pose-camera", "timestamp": 975_000, "rotation": [1, 0, 0, 0], "translation": [12, 0, 0]},
        ],
        "sample_data": [
            {"token": "lidar-before", "sample_token": "both", "is_key_frame": False,
             "calibrated_sensor_token": "lidar-cal", "ego_pose_token": "pose-before", "timestamp": 950_000,
             "filename": "unused-sweep.bin"},
            {"token": "lidar-both", "sample_token": "both", "is_key_frame": True,
             "calibrated_sensor_token": "lidar-cal", "ego_pose_token": "pose-lidar", "timestamp": 1_000_000,
             "filename": "samples/LIDAR_TOP/both.bin", "prev": "lidar-before"},
            {"token": "lidar-only-data", "sample_token": "lidar-only", "is_key_frame": True,
             "calibrated_sensor_token": "lidar-cal", "ego_pose_token": "pose-lidar",
             "filename": "samples/LIDAR_TOP/only.bin"},
            {"token": "camera-before", "sample_token": "both", "is_key_frame": False,
             "calibrated_sensor_token": "camera-cal", "ego_pose_token": "pose-before", "timestamp": 950_000,
             "filename": "samples/CAM_FRONT/frame.png", "width": 4, "height": 4},
            {"token": "camera-both", "sample_token": "both", "is_key_frame": True,
             "calibrated_sensor_token": "camera-cal", "ego_pose_token": "pose-camera", "timestamp": 975_000,
             "filename": "samples/CAM_FRONT/frame.png", "width": 4, "height": 4},
            {"token": "camera-only-data", "sample_token": "camera-only", "is_key_frame": True,
             "calibrated_sensor_token": "camera-cal", "ego_pose_token": "pose-camera", "timestamp": 1_975_000,
             "filename": "samples/CAM_FRONT/frame.png", "width": 4, "height": 4},
        ],
        "sample_annotation": [
            {"token": "box", "sample_token": "both", "category_name": "vehicle.car",
             "translation": [13, 4, 8], "size": [2, 4, 1], "rotation": [1, 0, 0, 0]},
        ],
    }
    for name, records in tables.items():
        (metadata / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")
    lidar_path = root / "samples/LIDAR_TOP"
    lidar_path.mkdir(parents=True)
    scan = np.array([[0, 0, 10, 3, 0], [1, 0, 10, 4, 1]], dtype="<f4")
    scan.tofile(lidar_path / "both.bin")
    scan.tofile(lidar_path / "only.bin")
    camera_path = root / "samples/CAM_FRONT/frame.png"
    camera_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), (30, 60, 90)).save(camera_path)
    return root


@pytest.mark.parametrize("sensor,tokens", [
    ("lidar", ["both", "lidar-only"]), ("camera", ["both", "camera-only"]),
])
def test_mode_selects_keyframe_index_without_loading_payloads(native_root, monkeypatch, sensor, tokens):
    def forbidden(*args, **kwargs):
        raise AssertionError("Enumerating sensor metadata must not read payloads.")

    monkeypatch.setattr(Image, "open", forbidden)
    monkeypatch.setattr(np, "fromfile", forbidden)
    reader = NuScenesReader(root=native_root, sensor=sensor)
    assert reader.list_scenes()[0].frame_count == 2
    frames = reader.list_frames("scene")
    assert [frame.token for frame in frames] == tokens
    assert [frame.index for frame in frames] == [0, 1]
    assert reader._image_cache == {}


def test_both_modalities_share_tables_and_keep_their_coordinate_frames(native_root, monkeypatch):
    read_table = NuScenesReader._read_table
    read_counts = Counter()

    def counted(self, name, **kwargs):
        read_counts[name] += 1
        return read_table(self, name, **kwargs)

    monkeypatch.setattr(NuScenesReader, "_read_table", counted)
    reader = NuScenesReader(root=native_root)
    cloud = reader.load_frame("both")
    camera_frame = reader.load_camera_frame("both")
    assert isinstance(cloud, PointCloudFrame)
    assert isinstance(camera_frame, NuScenesCameraFrame)
    assert all(count == 1 for count in read_counts.values())
    assert read_counts["sample_data"] == 1
    np.testing.assert_array_equal(cloud.boxes[0].center, [2, 2, 5])
    np.testing.assert_array_equal(camera_frame.boxes[0].center, [13, 4, 8])
    np.testing.assert_array_equal(cloud.boxes[0].size, [4, 2, 1])
    camera = camera_frame.cameras[0]
    assert camera.timestamp_us == 975_000
    np.testing.assert_array_equal(camera.sensor_to_world[:3, 3], [12, 0, 0])
    assert camera.image is reader.load_camera_frame("both").cameras[0].image
    assert not camera.image.flags.writeable
    assert len(reader._cameras_by_scene["scene"]["CAM_FRONT"]) == 3
    timed = reader.prepare_frame(cloud, point_timing=True)
    assert timed.raw_points.shape == cloud.points.shape
    assert timed.point_timestamps_us.shape == (len(cloud.points),)
    assert timed.rgb_colors is None
    assert reader._image_cache  # Timing does not clear camera payload cache.


@pytest.mark.parametrize("sensor", ["camera", "lidar"])
def test_wrong_modality_and_unknown_tokens_raise_shared_error(native_root, sensor):
    reader = NuScenesReader(root=native_root, sensor=sensor)
    with pytest.raises(DatasetError, match="Unknown nuScenes frame"):
        reader.load_frame("camera-only")
    with pytest.raises(DatasetError, match="Unknown nuScenes frame"):
        reader.load_camera_frame("lidar-only")
    with pytest.raises(DatasetError, match="Unknown nuScenes frame"):
        reader.load_camera_frame("missing")
    # Constructor enumeration mode never hides the other available payload.
    assert reader.load_camera_frame("camera-only").cameras
    assert reader.load_frame("lidar-only").points.shape == (2, 3)


def test_camera_only_extract_ignores_unavailable_lidar_payload_and_pose(native_root):
    for path in (native_root / "samples/LIDAR_TOP").iterdir():
        path.unlink()
    metadata = native_root / "v1.0-mini/sample_data.json"
    records = json.loads(metadata.read_text())
    for data in records:
        if data["calibrated_sensor_token"] == "lidar-cal":
            data.pop("ego_pose_token", None)
    metadata.write_text(json.dumps(records))
    reader = NuScenesReader(root=native_root, sensor="camera")
    assert len(reader.list_frames("scene")) == 2
    assert reader.load_camera_frame("both").cameras


def test_lidar_only_extract_does_not_decode_images(native_root, monkeypatch):
    (native_root / "samples/CAM_FRONT/frame.png").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Ordinary LiDAR viewing must not decode camera images.")

    monkeypatch.setattr(Image, "open", forbidden)
    reader = NuScenesReader(root=native_root)
    assert reader.load_frame("both").points.shape == (2, 3)
    assert reader._image_cache == {}


def test_invalid_sensor_is_actionable(native_root):
    with pytest.raises(DatasetError, match="expected 'lidar' or 'camera'"):
        NuScenesReader(root=native_root, sensor="radar")


@pytest.mark.parametrize("active,inactive,bad_token,bad_timestamp", [
    ("lidar", "camera", "camera-only", "invalid"),
    ("camera", "lidar", "lidar-only", True),
])
def test_inactive_modality_timestamp_error_is_deferred_to_cross_read(
    native_root, active, inactive, bad_token, bad_timestamp
):
    path = native_root / "v1.0-mini/sample.json"
    records = json.loads(path.read_text())
    next(sample for sample in records if sample["token"] == bad_token)["timestamp"] = bad_timestamp
    path.write_text(json.dumps(records))
    reader = NuScenesReader(root=native_root, sensor=active)
    assert reader.list_scenes()[0].frame_count == 2
    assert len(reader.list_frames("scene")) == 2
    active_load = reader.load_frame if active == "lidar" else reader.load_camera_frame
    assert active_load("both").info.token == "both"
    inactive_load = reader.load_frame if inactive == "lidar" else reader.load_camera_frame
    with pytest.raises(DatasetError, match=f"Invalid {inactive} sample timestamps.*{bad_token}"):
        inactive_load(bad_token)
    # The same invalid metadata remains an eager error when selected for
    # enumeration, rather than silently losing a frame from the active view.
    with pytest.raises(DatasetError, match=f"Invalid {inactive} sample timestamps"):
        NuScenesReader(root=native_root, sensor=inactive)


def test_camera_view_retains_keyframes_and_prepares_full_sweeps_only_on_request(native_root, monkeypatch):
    # Cover both scan returns, including the earlier sweep's exposure pose.
    Image.new("RGB", (8, 8), (30, 60, 90)).save(native_root / "samples/CAM_FRONT/frame.png")
    path = native_root / "v1.0-mini/sample_data.json"
    records = json.loads(path.read_text())
    for data in records:
        if data["calibrated_sensor_token"] == "camera-cal":
            data.update(width=8, height=8)
    path.write_text(json.dumps(records))
    reader = NuScenesReader(root=native_root, sensor="camera")
    assert set(reader._sample_data) == {
        "lidar-both", "lidar-only-data", "camera-both", "camera-only-data",
    }
    assert not reader._data_by_scene
    assert not reader._lidar_sweeps_by_scene
    assert len(reader._cameras_by_scene["scene"]["CAM_FRONT"]) == 2
    assert not reader._sweep_indexes_ready
    original_poses, original_calibrations, original_annotations = reader._poses, reader._calibrations, reader._annotations
    read_table = NuScenesReader._read_table
    read_counts = Counter()

    def counted(self, name, **kwargs):
        read_counts[name] += 1
        return read_table(self, name, **kwargs)

    monkeypatch.setattr(NuScenesReader, "_read_table", counted)
    camera = reader.load_camera_frame("both").cameras[0]
    cloud = reader.load_frame("both")
    assert reader.prepare_frame(cloud) is cloud
    assert not read_counts
    assert not reader._sweep_indexes_ready
    timed = reader.prepare_frame(cloud, point_timing=True)
    assert read_counts["sample_data"] == 1
    assert reader._sweep_indexes_ready
    assert set(reader._sample_data).issuperset({"lidar-before", "camera-before"})
    assert reader._data_by_scene["scene"]
    assert reader._poses is original_poses
    assert reader._calibrations is original_calibrations
    assert reader._annotations is original_annotations
    assert reader.sensor == "camera"
    assert [frame.token for frame in reader.list_frames("scene")] == ["both", "camera-only"]
    assert reader.load_camera_frame("both").cameras[0].image is camera.image
    np.testing.assert_array_equal(timed.point_timestamps_us, reader.prepare_frame(cloud, point_timing=True).point_timestamps_us)
    assert read_counts["sample_data"] == 1  # Reuse the deferred full index.
    assert set(read_counts) == {"sample_data"}  # No second pose/annotation load.
    colors = reader.prepare_frame(timed, camera_rgb=True)
    assert colors.rgb_colors.shape == (len(cloud.points), 3)
    assert colors.rgb_valid_mask.all()
    np.testing.assert_allclose(colors.rgb_colors, [[30 / 255, 60 / 255, 90 / 255]] * len(cloud.points))
