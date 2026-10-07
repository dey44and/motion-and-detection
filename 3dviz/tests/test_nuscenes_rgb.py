"""Small native fixtures exercise lazy camera loading and exposure poses."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from threedviz.dataloader import DatasetError, create_dataset


@pytest.fixture
def rgb_nuscenes_root(tmp_path):
    root = tmp_path / "nuscenes"
    metadata = root / "v1.0-mini"
    metadata.mkdir(parents=True)
    tables = {
        "scene": [{"token": "scene", "name": "RGB fixture"}],
        "sample": [{"token": "sample", "scene_token": "scene", "timestamp": 1_000_100}],
        "sensor": [
            {"token": "lidar", "channel": "LIDAR_TOP", "modality": "lidar"},
            {"token": "camera", "channel": "CAM_FRONT", "modality": "camera"},
        ],
        "calibrated_sensor": [
            {"token": "lidar-cal", "sensor_token": "lidar", "rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
            {"token": "camera-cal", "sensor_token": "camera", "rotation": [1, 0, 0, 0], "translation": [0, 0, 0],
             "camera_intrinsic": [[10, 0, 3], [0, 10, 3], [0, 0, 1]]},
        ],
        "ego_pose": [
            {"token": "pose-before", "timestamp": 950_000, "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]},
            {"token": "pose-lidar", "timestamp": 1_000_000, "translation": [1, 0, 0], "rotation": [1, 0, 0, 0]},
            # Deliberately differs from interpolation of the lidar-only poses.
            {"token": "pose-camera", "timestamp": 975_000, "translation": [0.25, 0, 0], "rotation": [1, 0, 0, 0]},
        ],
        "sample_data": [
            {"token": "sweep", "sample_token": "sample", "is_key_frame": False,
             "calibrated_sensor_token": "lidar-cal", "ego_pose_token": "pose-before", "timestamp": 950_000,
             "filename": "sweeps/LIDAR_TOP/previous.bin", "prev": "", "next": "lidar-frame"},
            {"token": "lidar-frame", "sample_token": "sample", "is_key_frame": True,
             "calibrated_sensor_token": "lidar-cal", "ego_pose_token": "pose-lidar", "timestamp": 1_000_000,
             "filename": "samples/LIDAR_TOP/frame.bin", "prev": "sweep", "next": ""},
            {"token": "camera-sweep", "sample_token": "sample", "is_key_frame": False,
             "calibrated_sensor_token": "camera-cal", "ego_pose_token": "pose-camera", "timestamp": 975_000,
             "filename": "sweeps/CAM_FRONT/previous.png", "width": 8, "height": 6},
            {"token": "camera-frame", "sample_token": "sample", "is_key_frame": True,
             "calibrated_sensor_token": "camera-cal", "ego_pose_token": "pose-lidar", "timestamp": 1_000_000,
             "filename": "samples/CAM_FRONT/frame.png", "width": 8, "height": 6},
        ],
    }
    for table, records in tables.items():
        (metadata / f"{table}.json").write_text(json.dumps(records), encoding="utf-8")
    scan = np.array([[0, 0, 10, 10, 0], [2, 0, 10, 20, 1], [-2, 0, 10, 30, 2]], dtype="<f4")
    for filename in ("sweeps/LIDAR_TOP/previous.bin", "samples/LIDAR_TOP/frame.bin"):
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        scan.tofile(path)
    gradient = np.zeros((6, 8, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.arange(8, dtype=np.uint8) * 30
    gradient[:, :, 1] = np.arange(6, dtype=np.uint8)[:, None] * 20
    for filename, pixels in (
        ("sweeps/CAM_FRONT/previous.png", gradient),
        ("samples/CAM_FRONT/frame.png", np.full((6, 8, 3), [0, 0, 240], dtype=np.uint8)),
    ):
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
    return root


def _rewrite(root: Path, table: str, update):
    path = root / "v1.0-mini" / f"{table}.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(update(records)), encoding="utf-8")


def test_camera_metadata_and_images_remain_lazy(rgb_nuscenes_root, monkeypatch):
    def unexpected_image(*args, **kwargs):
        raise AssertionError("Basic loading and timing must not open images.")

    monkeypatch.setattr(Image, "open", unexpected_image)
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    assert adapter.supports_camera_rgb
    assert adapter.supports_point_timing
    assert len(adapter.list_frames("scene")) == 1
    frame = adapter.load_frame("sample")
    assert frame.info.timestamp_us == 1_000_100
    assert frame.lidar_timestamp_us == 1_000_000
    np.testing.assert_array_equal(frame.ring_indices, [0, 1, 2])
    assert frame.point_timestamps_us is None
    assert frame.raw_points is None
    assert frame.rgb_colors is None
    assert adapter.prepare_frame(frame) is frame
    timed = adapter.prepare_frame(frame, point_timing=True)
    assert timed.rgb_colors is None
    assert timed.point_timestamps_us.shape == (3,)
    assert timed.raw_points.shape == (3, 3)
    assert np.all((timed.point_timestamps_us >= 950_000) & (timed.point_timestamps_us <= 1_000_000))
    # Both timing and projection must leave the reference cloud intact.
    np.testing.assert_array_equal(timed.points, frame.points)


def test_per_point_nearest_images_include_nonkeyframes_and_exact_camera_pose(rgb_nuscenes_root, monkeypatch):
    from threedviz.processing import motion

    monkeypatch.setattr(motion, "nominal_point_timestamps", lambda *args, **kwargs: np.array([950_000, 975_000, 999_999]))
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    frame = adapter.load_frame("sample")
    prepared = adapter.prepare_frame(frame, camera_rgb=True)
    np.testing.assert_array_equal(prepared.rgb_valid_mask, [True, True, True])
    # Lidar reference x=1 and actual camera exposure x=.25 give u=3.75/5.75.
    # Using a shared sample pose, or adding compensation twice, changes colors.
    np.testing.assert_allclose(prepared.rgb_colors, np.array([[112.5, 60, 0], [172.5, 60, 0], [0, 0, 240]]) / 255)
    assert len(adapter._image_cache) == 2
    assert adapter.prepare_frame(prepared, camera_rgb=True) is prepared
    np.testing.assert_array_equal(prepared.points, frame.points)
    trajectory = adapter._scene_trajectory("scene")
    point_poses = trajectory.at(prepared.point_timestamps_us)
    world_from_raw = np.einsum("nij,nj->ni", point_poses[:, :3, :3], prepared.raw_points) + point_poses[:, :3, 3]
    np.testing.assert_allclose(world_from_raw, frame.points + [1, 0, 0], atol=1e-6)


def test_missing_images_fail_only_when_rgb_is_requested(rgb_nuscenes_root):
    (rgb_nuscenes_root / "sweeps/CAM_FRONT/previous.png").unlink()
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    frame = adapter.load_frame("sample")
    adapter.prepare_frame(frame, point_timing=True)
    with pytest.raises(DatasetError, match="Cannot prepare camera RGB.*CAM_FRONT.*Extract nuScenes camera"):
        adapter.prepare_frame(frame, camera_rgb=True)
    assert frame.rgb_colors is None
    np.testing.assert_array_equal(adapter.load_frame("sample").points, frame.points)


def test_no_camera_metadata_does_not_advertise_rgb(rgb_nuscenes_root):
    _rewrite(rgb_nuscenes_root, "sample_data", lambda records: [record for record in records if "camera" not in record["token"]])
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    assert adapter.supports_point_timing
    assert not adapter.supports_camera_rgb
    with pytest.raises(DatasetError, match="No camera images near this lidar scan"):
        adapter.prepare_frame(adapter.load_frame("sample"), camera_rgb=True)


@pytest.mark.parametrize("ring", [-1, 32, 1.5, np.nan])
def test_invalid_ring_does_not_break_basic_cloud(rgb_nuscenes_root, ring):
    path = rgb_nuscenes_root / "samples/LIDAR_TOP/frame.bin"
    scan = np.fromfile(path, dtype="<f4").reshape(-1, 5)
    scan[0, 4] = ring
    scan.tofile(path)
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    frame = adapter.load_frame("sample")
    assert frame.ring_indices is None
    with pytest.raises(DatasetError, match="ring indices"):
        adapter.prepare_frame(frame, point_timing=True)


def test_sweep_interval_uses_previous_lidar_timestamp_and_first_next_gap(rgb_nuscenes_root):
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    assert adapter._sweep_interval(adapter._sample_data["lidar-frame"], "scene") == (950_000, 1_000_000)
    assert adapter._sweep_interval(adapter._sample_data["sweep"], "scene") == (900_000, 950_000)


def test_scene_phase_comes_from_first_complete_sweep_and_stays_fixed(rgb_nuscenes_root):
    phase = 0.2
    scan = np.zeros((1085 * 32, 5), dtype="<f4")
    scan[:, :2] = [30 * np.cos(phase), 30 * np.sin(phase)]
    scan[:, 4] = np.tile(np.arange(32), 1085)
    path = rgb_nuscenes_root / "sweeps/LIDAR_TOP/previous.bin"
    scan.tofile(path)
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    frame = adapter.load_frame("sample")
    adapter.prepare_frame(frame, point_timing=True)
    assert adapter._nominal_phases["scene"] == pytest.approx(phase)
    scan[:, :2] = [30 * np.cos(0.9), 30 * np.sin(0.9)]
    scan.tofile(path)
    adapter.prepare_frame(frame, point_timing=True)
    assert adapter._nominal_phases["scene"] == pytest.approx(phase)


def test_invalid_camera_calibration_is_reported_only_on_rgb_request(rgb_nuscenes_root):
    def invalid_camera(records):
        records[1]["camera_intrinsic"][0][0] = 0
        return records

    _rewrite(rgb_nuscenes_root, "calibrated_sensor", invalid_camera)
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    frame = adapter.load_frame("sample")
    assert adapter.prepare_frame(frame, point_timing=True).rgb_colors is None
    with pytest.raises(DatasetError, match="Cannot prepare camera RGB.*positive focal lengths"):
        adapter.prepare_frame(frame, camera_rgb=True)


def test_scene_trajectory_uses_all_sensor_poses_and_rejects_missing_timing_on_request(rgb_nuscenes_root):
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    np.testing.assert_allclose(adapter._scene_trajectory("scene").at(np.array([975_000]))[0, :3, 3], [0.25, 0, 0])
    _rewrite(rgb_nuscenes_root, "sample_data", lambda records: [{key: value for key, value in record.items() if key != "timestamp"} for record in records])
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    assert not adapter.supports_point_timing
    frame = adapter.load_frame("sample")
    assert frame.lidar_timestamp_us == frame.info.timestamp_us
    with pytest.raises(DatasetError, match="valid LIDAR_TOP sample_data timestamp"):
        adapter.prepare_frame(frame, point_timing=True)


def test_decoded_image_cache_is_bounded(rgb_nuscenes_root, tmp_path):
    adapter = create_dataset("nuscenes", root=rgb_nuscenes_root)
    paths = []
    for index in range(20):
        path = tmp_path / f"camera-{index}.png"
        Image.fromarray(np.full((1, 1, 3), index, dtype=np.uint8)).save(path)
        paths.append(path)
        adapter._load_camera_image(path)
    assert len(adapter._image_cache) == 16
    assert paths[0] not in adapter._image_cache
    first_cached = adapter._load_camera_image(paths[-1])
    assert adapter._load_camera_image(paths[-1]) is first_cached
