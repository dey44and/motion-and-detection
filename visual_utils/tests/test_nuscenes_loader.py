"""Conformance tests for normalized nuScenes observations and preparation."""

import json
import shutil

import numpy as np
from PIL import Image
import pytest

from visual_utils.config import DatasetConfig
from visual_utils.models import (
    AnnotationStatus, Capability, Modality, PointCloudPreparation, TimingQuality,
)
from visual_utils.errors import (
    DataUnavailableError, DatasetClosedError, DatasetConfigurationError,
    DatasetIOError, DatasetValidationError, UnknownIdentifierError,
)
from visual_utils.nuscenes.loader import NuScenesLoader


@pytest.fixture
def contract_root(tmp_path):
    root = tmp_path / "nuscenes"
    metadata = root / "v1.0-mini"
    metadata.mkdir(parents=True)
    tables = {
        "scene": [{"token": "scene", "name": "scene-0103"}],
        "sample": [
            {"token": "both", "scene_token": "scene", "timestamp": 1_000_100},
            {"token": "camera-only", "scene_token": "scene", "timestamp": 2_000_100},
            {"token": "lidar-only", "scene_token": "scene", "timestamp": 3_000_100},
        ],
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
            {"token": "before", "timestamp": 950_000, "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]},
            {"token": "lidar-pose", "timestamp": 1_000_000, "translation": [1, 0, 0], "rotation": [1, 0, 0, 0]},
            {"token": "camera-pose", "timestamp": 975_000, "translation": [.25, 0, 0], "rotation": [1, 0, 0, 0]},
        ],
        "sample_data": [
            {"token": "sweep", "sample_token": "both", "is_key_frame": False, "timestamp": 950_000,
             "calibrated_sensor_token": "lidar-cal", "ego_pose_token": "before", "filename": "unused.bin"},
            {"token": "lidar-data", "sample_token": "both", "is_key_frame": True, "timestamp": 1_000_000,
             "calibrated_sensor_token": "lidar-cal", "ego_pose_token": "lidar-pose", "prev": "sweep",
             "filename": "samples/LIDAR_TOP/points.bin"},
            {"token": "only-lidar-data", "sample_token": "lidar-only", "is_key_frame": True,
             "calibrated_sensor_token": "lidar-cal", "ego_pose_token": "lidar-pose",
             "filename": "samples/LIDAR_TOP/points.bin"},
            {"token": "camera-data", "sample_token": "both", "is_key_frame": True, "timestamp": 975_000,
             "calibrated_sensor_token": "camera-cal", "ego_pose_token": "camera-pose", "width": 8, "height": 8,
             "filename": "samples/CAM_FRONT/image.png"},
            {"token": "only-camera-data", "sample_token": "camera-only", "is_key_frame": True, "timestamp": 1_975_000,
             "calibrated_sensor_token": "camera-cal", "ego_pose_token": "camera-pose", "width": 8, "height": 8,
             "filename": "samples/CAM_FRONT/image.png"},
        ],
        "category": [
            {"token": "adult-cat", "name": "human.pedestrian.adult"},
            {"token": "child-cat", "name": "human.pedestrian.child"},
        ],
        "instance": [
            {"token": "adult-instance", "category_token": "adult-cat"},
            {"token": "child-instance", "category_token": "child-cat"},
        ],
        "sample_annotation": [
            {"token": "adult-box", "sample_token": "both", "instance_token": "adult-instance",
             "translation": [1, 0, 10], "size": [1, 2, 3], "rotation": [1, 0, 0, 0]},
            {"token": "child-box", "sample_token": "both", "instance_token": "child-instance",
             "translation": [2, 0, 10], "size": [1, 1, 2], "rotation": [1, 0, 0, 0]},
        ],
    }
    for name, records in tables.items():
        (metadata / f"{name}.json").write_text(json.dumps(records))
    lidar = root / "samples/LIDAR_TOP/points.bin"
    lidar.parent.mkdir(parents=True)
    np.array([[0, 0, 10, 20, 0], [1, 0, 10, 40, 1]], dtype="<f4").tofile(lidar)
    camera = root / "samples/CAM_FRONT/image.png"
    camera.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(camera)
    return root


def update(root, name, transform):
    path = root / "v1.0-mini" / f"{name}.json"
    records = json.loads(path.read_text())
    transform(records)
    path.write_text(json.dumps(records))


def test_metadata_and_modality_catalog_are_lazy_native_and_dense(contract_root, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata must not decode sensor payloads.")

    monkeypatch.setattr(Image, "open", forbidden)
    monkeypatch.setattr(np, "fromfile", forbidden)
    loader = NuScenesLoader(DatasetConfig(root=contract_root))
    assert not loader._readers
    metadata = loader.metadata
    assert metadata.dataset_id == "nuscenes:v1.0-mini"
    assert metadata.capabilities == frozenset(Capability)
    assert set(loader._readers) == {Modality.CAMERAS}
    assert not loader.native_reader(Modality.CAMERAS)._data_by_scene
    classes = {item.class_id: item for item in metadata.classes}
    assert classes["human.pedestrian.adult"].display_group == "pedestrian"
    assert classes["human.pedestrian.child"].display_group == "pedestrian"
    scenes = loader.list_scenes()
    assert dict(scenes[0].frame_counts) == {Modality.POINT_CLOUD: 2, Modality.CAMERAS: 2}
    point_frames = loader.list_frames("scene", modality=Modality.POINT_CLOUD)
    cameras = loader.list_frames("scene", modality=Modality.CAMERAS)
    assert [(frame.frame_id, frame.index) for frame in point_frames] == [("both", 0), ("lidar-only", 1)]
    assert [(frame.frame_id, frame.index) for frame in cameras] == [("both", 0), ("camera-only", 1)]
    assert loader.native_reader(Modality.POINT_CLOUD) is loader.native_reader(Modality.CAMERAS)
    assert point_frames[0] == cameras[0]
    assert loader.list_scenes(split="train") == ()
    with pytest.raises(DatasetConfigurationError):
        loader.list_scenes(split="not-a-split")


def test_native_box_identity_anchored_geometry_and_pinhole_projection(contract_root):
    s = float(np.sqrt(.5))
    update(contract_root, "calibrated_sensor", lambda rows: rows[0].update(rotation=[s, 0, 0, s], translation=[1, 2, 3]))
    loader = NuScenesLoader(DatasetConfig(root=contract_root))
    cloud = loader.load_point_cloud("both")
    cameras = loader.load_cameras("both")
    assert cloud.reference_frame_id == "sensor:LIDAR_TOP:lidar-data"
    assert cloud.world_frame_id == cameras.reference_frame_id
    assert cloud.info.timestamp_us == 1_000_100
    assert cloud.reference_pose_timestamp_us == 1_000_000
    assert (cloud.scan_start_us, cloud.scan_end_us) == (950_000, 1_000_000)
    assert [box.class_id for box in cloud.annotations.boxes] == ["human.pedestrian.adult", "human.pedestrian.child"]
    assert [box.instance_id for box in cloud.annotations.boxes] == ["adult-instance", "child-instance"]
    assert [box.annotation_id for box in cloud.annotations.boxes] == ["adult-box", "child-box"]
    np.testing.assert_array_equal(cloud.annotations.boxes[0].size, [2, 1, 3])
    transform = cloud.T_world_from_reference
    box = cloud.annotations.boxes[0]
    np.testing.assert_allclose(transform[:3, :3] @ box.center + transform[:3, 3], cameras.annotations.boxes[0].center, atol=1e-12)
    np.testing.assert_allclose(transform[:3, :3] @ box.rotation, cameras.annotations.boxes[0].rotation, atol=1e-12)
    camera = cameras.cameras[0]
    assert camera.timestamp_us == camera.pose_timestamp_us == 975_000
    assert camera.exposure_interval_us is None
    result = camera.projection_model.project_reference(np.array([[.25, 0, 10], [.25, 0, -10]]))
    np.testing.assert_array_equal(result.valid_mask, [True, False])
    np.testing.assert_array_equal(result.depth, [10, -10])
    np.testing.assert_allclose(result.uv[0], [3, 3])
    assert not cloud.points.flags.writeable
    assert not cloud.attributes["intensity"].flags.writeable


def test_payload_indices_belong_to_their_own_modality_list(contract_root):
    update(contract_root, "sample", lambda rows: rows[1].update(timestamp=500_100))
    loader = NuScenesLoader(DatasetConfig(root=contract_root))
    camera_info = loader.list_frames("scene", modality=Modality.CAMERAS)[1]
    point_info = loader.list_frames("scene", modality=Modality.POINT_CLOUD)[0]
    assert camera_info.frame_id == point_info.frame_id == "both"
    assert camera_info.timestamp_us == point_info.timestamp_us
    assert loader.load_cameras("both").info.index == 1
    assert loader.load_point_cloud("both").info.index == 0
    assert loader.load_point_cloud("lidar-only").reference_frame_id != loader.load_point_cloud("both").reference_frame_id


@pytest.mark.parametrize("state", ["zero", "absent", "withheld"])
def test_annotation_availability_distinguishes_zero_absent_and_withheld(contract_root, state):
    release = "v1.0-mini"
    if state == "zero":
        update(contract_root, "sample_annotation", lambda rows: rows.clear())
        expected = AnnotationStatus.AVAILABLE
    elif state == "absent":
        (contract_root / release / "sample_annotation.json").unlink()
        expected = AnnotationStatus.NOT_PROVIDED
    else:
        release = "v1.0-test"
        (contract_root / "v1.0-mini").rename(contract_root / release)
        expected = AnnotationStatus.WITHHELD
    loader = NuScenesLoader(DatasetConfig(root=contract_root, release=release))
    cloud = loader.load_point_cloud("both")
    camera = loader.load_cameras("both")
    assert cloud.annotations.status == camera.annotations.status == expected
    assert cloud.annotations.boxes == camera.annotations.boxes == ()
    assert (Capability.BOXES_3D in loader.metadata.capabilities) == (state == "zero")


def test_optional_preparation_estimated_atomic_and_preserves_valid_black(contract_root):
    loader = NuScenesLoader(DatasetConfig(root=contract_root))
    cloud = loader.load_point_cloud("both")
    assert loader.prepare_point_cloud(cloud, request=PointCloudPreparation()) is cloud
    prepared = loader.prepare_point_cloud(cloud, request=PointCloudPreparation(rgb=True, timestamps=True, acquisition_geometry=True))
    assert len({id(reader) for reader in loader._readers.values()}) == 1
    assert prepared.info is cloud.info
    assert prepared.annotations is cloud.annotations
    assert set(prepared.attributes) == set(cloud.attributes)
    for key in cloud.attributes:
        np.testing.assert_array_equal(prepared.attributes[key], cloud.attributes[key])
    np.testing.assert_array_equal(prepared.points, cloud.points)
    assert cloud.rgb is cloud.timing is cloud.acquisition is None
    assert prepared.timing.quality == TimingQuality.ESTIMATED
    assert prepared.timing.valid_mask.all()
    assert "nominal" in prepared.timing.provenance.method_id
    assert prepared.acquisition.valid_mask.all()
    assert prepared.acquisition.reference_frame_id != cloud.reference_frame_id
    assert prepared.rgb.valid_mask.all()
    np.testing.assert_array_equal(prepared.rgb.values, np.zeros((len(cloud.points), 3)))
    assert loader.prepare_point_cloud(prepared, request=PointCloudPreparation(rgb=True)) is prepared


def test_known_absence_missing_file_and_corruption_are_distinct(contract_root):
    loader = NuScenesLoader(DatasetConfig(root=contract_root))
    with pytest.raises(UnknownIdentifierError):
        loader.load_point_cloud("unknown")
    with pytest.raises(DataUnavailableError):
        loader.load_point_cloud("camera-only")
    with pytest.raises(DataUnavailableError):
        loader.load_cameras("lidar-only")
    image_path = contract_root / "samples/CAM_FRONT/image.png"
    image_path.unlink()
    with pytest.raises(DatasetIOError, match="image not found"):
        loader.load_cameras("both")
    cloud = loader.load_point_cloud("both")
    with pytest.raises(DatasetIOError):
        loader.prepare_point_cloud(cloud, request=PointCloudPreparation(rgb=True))
    assert cloud.rgb is cloud.timing is cloud.acquisition is None
    (contract_root / "samples/LIDAR_TOP/points.bin").write_bytes(b"broken")
    with pytest.raises(DatasetValidationError, match="20 bytes"):
        loader.load_point_cloud("both")


def test_uncalibrated_images_are_valid_and_missing_pose_time_stays_unknown(contract_root):
    def remove_calibration(rows):
        rows[1].pop("camera_intrinsic")
        rows[1].pop("rotation")
        rows[1].pop("translation")

    update(contract_root, "calibrated_sensor", remove_calibration)
    update(contract_root, "sample_data", lambda rows: rows[3].pop("ego_pose_token"))
    loader = NuScenesLoader(DatasetConfig(root=contract_root))
    camera = loader.load_cameras("both").cameras[0]
    assert camera.intrinsics is camera.T_reference_from_camera is camera.projection_model is None
    assert camera.pose_timestamp_us is None
    assert camera.timestamp_us == 975_000
    assert camera.image.shape == (8, 8, 3)
    cloud = loader.load_point_cloud("lidar-only")
    # Its recorded ego pose timestamp is known even though sample_data lacks it.
    assert cloud.reference_pose_timestamp_us == 1_000_000
    assert cloud.scan_end_us is None


def test_per_frame_missing_timing_and_no_camera_records_are_not_fabricated(contract_root):
    update(contract_root, "sample_data", lambda rows: rows.__setitem__(slice(None), [row for row in rows if row["calibrated_sensor_token"] != "camera-cal"]))
    loader = NuScenesLoader(DatasetConfig(root=contract_root))
    assert Capability.RGB_COLORING not in loader.metadata.capabilities
    assert Capability.POINT_TIMESTAMPS in loader.metadata.capabilities
    cloud = loader.load_point_cloud("lidar-only")
    assert cloud.scan_start_us is cloud.scan_end_us is None
    with pytest.raises(DataUnavailableError, match="valid LIDAR_TOP sample_data timestamp"):
        loader.prepare_point_cloud(cloud, request=PointCloudPreparation(timestamps=True))
    with pytest.raises(DataUnavailableError):
        loader.load_cameras("both")


def test_profiles_relocate_stably_and_lifecycle_preserves_records(contract_root, tmp_path):
    loader = NuScenesLoader(DatasetConfig(root=contract_root))
    cloud = loader.load_point_cloud("both")
    camera = loader.load_cameras("both").cameras[0]
    relocated = tmp_path / "relocated"
    shutil.copytree(contract_root, relocated)
    other = NuScenesLoader(DatasetConfig(root=tmp_path, source_paths={"native": relocated}))
    assert other.metadata.dataset_id == loader.metadata.dataset_id
    assert other.load_point_cloud("both").source_id == cloud.source_id
    assert other.load_point_cloud("both").reference_frame_id == cloud.reference_frame_id
    loader.close()
    loader.close()
    assert loader.metadata.dataset_id == "nuscenes:v1.0-mini"
    np.testing.assert_array_equal(camera.image[0, 0], [0, 0, 0])
    assert np.isfinite(cloud.points).all()
    with pytest.raises(DatasetClosedError):
        loader.list_scenes()
    with pytest.raises(DatasetClosedError):
        loader.load_point_cloud("both")
    with pytest.raises(DatasetClosedError):
        loader.prepare_point_cloud(cloud, request=PointCloudPreparation())


@pytest.mark.parametrize("kwargs", [
    {"release": "unknown"}, {"storage_format": "pickle"},
    {"options": {"ignored": True}}, {"source_paths": {"typo": "."}},
])
def test_unknown_provider_configuration_rejected(contract_root, kwargs):
    with pytest.raises(DatasetConfigurationError):
        NuScenesLoader(DatasetConfig(root=contract_root, **kwargs))


@pytest.mark.parametrize("active,bad_token", [(Modality.POINT_CLOUD, "camera-only"), (Modality.CAMERAS, "lidar-only")])
def test_bad_inactive_frame_time_does_not_block_active_contract_view(contract_root, active, bad_token):
    update(contract_root, "sample", lambda rows: next(row for row in rows if row["token"] == bad_token).update(timestamp=True))
    loader = NuScenesLoader(DatasetConfig(root=contract_root))
    assert len(loader.list_frames("scene", modality=active)) == 2
    if active == Modality.POINT_CLOUD:
        assert loader.load_point_cloud("both").points.shape == (2, 3)
    else:
        assert loader.load_cameras("both").cameras
