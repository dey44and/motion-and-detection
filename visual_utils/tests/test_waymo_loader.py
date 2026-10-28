"""The normalized Waymo provider over independent processed/native sources."""

import io
import pickle
import shutil
import struct
import zlib
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from visual_utils.config import DatasetConfig
from visual_utils.models import (
    AnnotationAlignment, AnnotationStatus, Capability, Modality, PointCloudPreparation,
)
from visual_utils.errors import (
    DataUnavailableError, DatasetClosedError, DatasetConfigurationError,
    DatasetIOError, DatasetValidationError, UnknownIdentifierError, UnsupportedCapabilityError,
)
from visual_utils.waymo import NativeWaymoReader
from visual_utils.waymo.camera import WaymoCameraProjection
from visual_utils.waymo.loader import WaymoLoader
from visual_utils.waymo.proto import FrameProto
from visual_utils.waymo.records import masked_crc32c


def _varint(value):
    value &= (1 << 64) - 1
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    return bytes(result) + bytes([value])


def _matrix(value, integer=False):
    array = np.array(value, dtype=np.int32 if integer else "<f4")
    data = b"".join(_varint(int(item)) for item in array.ravel()) if integer else array.tobytes()
    shape = b"".join(b"\x08" + _varint(size) for size in array.shape)
    return zlib.compress(b"\x0a" + _varint(len(data)) + data + b"\x12" + _varint(len(shape)) + shape)


def _transform(message, x=0):
    matrix = np.eye(4)
    matrix[0, 3] = x
    message.transform.extend(matrix.ravel())


def _proto(index=0, labels=True):
    frame = FrameProto()
    frame.context.name = "context"
    frame.timestamp_micros = 1_000_000 + index * 100_000
    _transform(frame.pose, 2)
    calibration = frame.context.camera_calibrations.add()
    calibration.name = 1
    calibration.intrinsic.extend([100, 100, 1, .5, 0, 0, 0, 0, 0])
    calibration.width, calibration.height = 2, 1
    calibration.rolling_shutter_direction = 2
    _transform(calibration.extrinsic)
    camera = frame.images.add()
    camera.name = 1
    _transform(camera.pose, 2)
    camera.pose_timestamp = frame.timestamp_micros / 1e6
    camera.shutter = .001
    camera.camera_trigger_time = camera.pose_timestamp - .005
    camera.camera_readout_done_time = camera.pose_timestamp + .005
    stream = io.BytesIO()
    Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]]], np.uint8)).save(stream, "PNG")
    camera.image = stream.getvalue()
    for name in (1, 2):
        calibration = frame.context.laser_calibrations.add()
        calibration.name = name
        calibration.beam_inclinations.append(0)
        _transform(calibration.extrinsic)
        laser = frame.lasers.add()
        laser.name = name
        laser.ri_return1.range_image_compressed = _matrix([[[10, name, 2, -1], [0, 0, 0, -1]]])
        if name == 1:
            laser.ri_return1.range_image_pose_compressed = _matrix([[[0, 0, 0, 5, 0, 0], [0, 0, 0, 5, 0, 0]]])
            laser.ri_return1.camera_projection_compressed = _matrix([[[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]], True)
            laser.ri_return2.range_image_compressed = _matrix([[[0, 0, 0, -1], [20, 3, 4, 1]]])
            laser.ri_return2.camera_projection_compressed = _matrix([[[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]], True)
    if labels:
        for label_type, identifier in ((1, "vehicle"), (3, "sign")):
            label = frame.laser_labels.add()
            label.type, label.id = label_type, identifier
            label.box.center_x = 5
            label.box.length, label.box.width, label.box.height = 4, 2, 1
            if label_type == 1:
                label.most_visible_camera_name = "FRONT"
                label.camera_synced_box.CopyFrom(label.box)
                label.camera_synced_box.center_x = 6
    return frame


def _record(frame):
    payload = frame.SerializeToString()
    length = struct.pack("<Q", len(payload))
    return length + struct.pack("<I", masked_crc32c(length)) + payload + struct.pack("<I", masked_crc32c(payload))


@pytest.fixture
def sources(tmp_path):
    native = tmp_path / "arbitrary-native-name"
    folder = native / "validation"
    folder.mkdir(parents=True)
    (folder / "segment-context_with_camera_labels.tfrecord").write_bytes(b"".join(_record(_proto(index)) for index in range(3)))
    processed = tmp_path / "point-workspace" / "arbitrary-processed-name"
    sequence = "segment-context_with_camera_labels"
    directory = processed / sequence
    directory.mkdir(parents=True)
    reader = NativeWaymoReader(native)
    records = []
    for index in (0, 2):
        frame = reader.read_frame("context", index)
        points = frame.points()
        np.save(directory / f"{index:04d}.npy", np.column_stack([points.points, points.intensity, points.elongation, points.nlz]).astype(np.float32))
        records.append({
            "point_cloud": {"lidar_sequence": sequence, "sample_idx": index, "num_features": 5},
            "frame_id": f"{sequence}_{index:03d}",
            "metadata": {"context_name": "context", "timestamp_micros": frame.timestamp_us},
            "pose": frame.vehicle_to_world,
            "annos": {"name": np.array([box.class_name for box in frame.boxes]),
                      "obj_ids": np.array([box.identifier for box in frame.boxes]),
                      "gt_boxes_lidar": np.array([[*box.center, *box.size, box.heading, 0, 0] for box in frame.boxes])},
        })
    (directory / f"{sequence}.pkl").write_bytes(pickle.dumps(records))
    splits = processed.parent / "ImageSets"
    splits.mkdir()
    (splits / "val.txt").write_text(sequence + ".tfrecord\n")
    (splits / "test.txt").write_text("segment-not-installed.tfrecord\n")
    return native, processed


def _loader(sources, *, processed=True, native=True, options=None):
    paths = {}
    if processed:
        paths["processed"] = sources[1]
    if native:
        paths["native"] = sources[0]
    return WaymoLoader(DatasetConfig(storage_format="auto", source_paths=paths, options=options or {}))


def test_native_and_processed_profiles_share_ids_but_keep_modality_indices(sources):
    loader = _loader(sources)
    assert loader.reader is not None
    scenes = loader.list_scenes(split="val")
    assert scenes[0].scene_id == "context" and scenes[0].frame_counts[Modality.POINT_CLOUD] is None
    clouds = loader.list_frames("context", modality=Modality.POINT_CLOUD)
    cameras = loader.list_frames("context", modality=Modality.CAMERAS)
    assert [frame.frame_id for frame in clouds] == [cameras[0].frame_id, cameras[2].frame_id]
    assert clouds[1].index == 1 and cameras[2].index == 2
    assert loader.list_scenes()[0].frame_counts[Modality.POINT_CLOUD] == 2
    assert loader.list_scenes()[0].frame_counts[Modality.CAMERAS] == 3
    assert loader.list_scenes(split="test") == ()
    with pytest.raises(DataUnavailableError):
        loader.load_point_cloud(cameras[1].frame_id)


def test_preparation_preserves_baseline_and_records_partial_coverage(sources):
    loader = _loader(sources)
    assert loader.metadata.feature_coverage[Capability.ACQUISITION_GEOMETRY] == "partial"
    info = loader.list_frames("context", modality=Modality.POINT_CLOUD)[1]
    frame = loader.load_point_cloud(info.frame_id)
    assert set(frame.attributes) == {"intensity", "elongation", "nlz"}
    assert frame.reference_pose_timestamp_us is None and frame.scan_start_us == info.timestamp_us
    assert frame.annotations.status == AnnotationStatus.AVAILABLE
    assert frame.annotations.timing.timestamp_us is None
    prepared = loader.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=True, acquisition_geometry=True))
    assert prepared.points is frame.points and prepared.annotations is frame.annotations
    np.testing.assert_array_equal(prepared.rgb.valid_mask, [True, True, False])
    np.testing.assert_array_equal(prepared.acquisition.valid_mask, [True, True, False])
    assert prepared.timing is None and frame.rgb is None and frame.acquisition is None
    assert loader.prepare_point_cloud(prepared, request=PointCloudPreparation(rgb=True)) is prepared
    with pytest.raises(UnsupportedCapabilityError):
        loader.prepare_point_cloud(frame, request=PointCloudPreparation(timestamps=True))


def test_native_cameras_keep_local_reference_and_mixed_synced_annotations(sources):
    loader = _loader(sources, processed=False)
    info = loader.list_frames("context", modality=Modality.CAMERAS)[0]
    frame = loader.load_cameras(info.frame_id)
    assert frame.reference_frame_id == info.frame_id + ":vehicle-reference"
    assert frame.reference_pose_timestamp_us is None
    assert frame.cameras[0].sensor_id == "CAM_FRONT"
    assert frame.cameras[0].pose_timestamp_us == 1_000_000
    assert frame.cameras[0].timestamp_kind == "camera_trigger"
    assert frame.cameras[0].timestamp_us == 995_000
    override = frame.cameras[0].annotations
    assert override.timing.alignment == AnnotationAlignment.MIXED
    assert [box.timing_override.alignment for box in override.boxes] == [AnnotationAlignment.CAMERA_SYNCHRONIZED, AnnotationAlignment.REFERENCE_TIME]
    assert all(box.timing_override.timestamp_us is None for box in override.boxes)
    assert override.boxes[0].center[0] == 6 and frame.annotations.boxes[0].center[0] == 5
    projection = frame.cameras[0].projection_model
    loader.close()
    assert np.isfinite(projection.project_reference(np.array([[5., 0, 0]])).uv).all()
    assert loader.metadata.dataset_id
    with pytest.raises(DatasetClosedError):
        loader.list_scenes()


def test_processed_only_mode_requires_no_native_inputs(sources):
    loader = _loader(sources, native=False)
    assert Capability.CAMERAS not in loader.metadata.capabilities
    frame = loader.load_point_cloud(loader.list_frames("context", modality=Modality.POINT_CLOUD)[0].frame_id)
    with pytest.raises(UnsupportedCapabilityError):
        loader.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=True))
    with pytest.raises(UnsupportedCapabilityError):
        loader.list_frames("context", modality=Modality.CAMERAS)


def test_wrong_native_row_order_is_rejected_without_partial_preparation(sources):
    loader = _loader(sources)
    info = loader.list_frames("context", modality=Modality.POINT_CLOUD)[0]
    directory = sources[1] / "segment-context_with_camera_labels"
    values = np.load(directory / "0000.npy")
    np.save(directory / "0000.npy", values[::-1])
    frame = loader.load_point_cloud(info.frame_id)
    with pytest.raises(DatasetValidationError, match="correspondence"):
        loader.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=True, acquisition_geometry=True))
    assert frame.rgb is None and frame.acquisition is None


@pytest.mark.parametrize("component", ["translation", "rotation"])
@pytest.mark.parametrize("preparation", [PointCloudPreparation(rgb=True), PointCloudPreparation(acquisition_geometry=True)])
def test_processed_pose_mismatch_cannot_be_used_for_native_preparation(sources, component, preparation):
    metadata = sources[1] / "segment-context_with_camera_labels/segment-context_with_camera_labels.pkl"
    records = pickle.loads(metadata.read_bytes())
    if component == "translation":
        records[0]["pose"][0, 3] += 100
    else:
        angle = .01
        records[0]["pose"][:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                                       [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    metadata.write_bytes(pickle.dumps(records))
    loader = _loader(sources)
    cloud = loader.load_point_cloud("waymo:context:1000000")
    cameras = loader.load_cameras(cloud.info.frame_id)
    assert cloud.reference_frame_id != cameras.reference_frame_id
    assert len(cameras.cameras) == 1  # Native camera reads remain independent of the processed pose.
    with pytest.raises(DatasetValidationError, match="reference pose correspondence"):
        loader.prepare_point_cloud(cloud, request=preparation)
    assert cloud.rgb is None and cloud.acquisition is None


def test_processed_pose_pairing_accepts_float32_global_translation_rounding(sources):
    metadata = sources[1] / "segment-context_with_camera_labels/segment-context_with_camera_labels.pkl"
    records = pickle.loads(metadata.read_bytes())
    records[0]["pose"][0, 3] += .001
    metadata.write_bytes(pickle.dumps(records))
    loader = _loader(sources)
    cloud = loader.load_point_cloud("waymo:context:1000000")
    prepared = loader.prepare_point_cloud(cloud, request=PointCloudPreparation(rgb=True, acquisition_geometry=True))
    assert prepared.rgb.valid_mask.any() and prepared.acquisition.valid_mask.any()


def test_processed_source_identity_tracks_profile_without_absolute_paths(sources, tmp_path):
    first = _loader(sources).load_point_cloud("waymo:context:1000000")
    alternate = tmp_path / "alternate-profile"
    relocated = tmp_path / "another-machine" / sources[1].name
    shutil.copytree(sources[1], alternate)
    shutil.copytree(sources[1], relocated)
    second = _loader((sources[0], alternate)).load_point_cloud(first.info.frame_id)
    third = _loader((sources[0], relocated)).load_point_cloud(first.info.frame_id)
    assert first.source_id != second.source_id and first.reference_frame_id != second.reference_frame_id
    assert first.source_id == third.source_id and first.reference_frame_id == third.reference_frame_id
    assert sources[1].name in first.source_id and str(tmp_path) not in first.source_id


def test_processed_missing_tracking_ids_retains_annotation_ids_only(sources):
    metadata = sources[1] / "segment-context_with_camera_labels/segment-context_with_camera_labels.pkl"
    records = pickle.loads(metadata.read_bytes())
    records[0]["annos"].pop("obj_ids")
    metadata.write_bytes(pickle.dumps(records))
    loader = _loader(sources)
    annotations = loader.load_point_cloud("waymo:context:1000000").annotations
    assert len({box.annotation_id for box in annotations.boxes}) == len(annotations.boxes)
    assert all(box.annotation_id and box.instance_id is None for box in annotations.boxes)
    assert [box.instance_id for box in loader.load_point_cloud("waymo:context:1200000").annotations.boxes] == ["vehicle", "sign"]


def test_withheld_missing_and_empty_annotations_are_distinct(sources):
    test_folder = sources[0] / "testing"
    test_folder.mkdir()
    frame = _proto(labels=False)
    frame.context.name = "test-context"
    (test_folder / "segment-test-context.tfrecord").write_bytes(_record(frame))
    loader = _loader(sources, processed=False)
    assert loader.load_point_cloud("waymo:test-context:1000000").annotations.status == AnnotationStatus.WITHHELD
    metadata = sources[1] / "segment-context_with_camera_labels/segment-context_with_camera_labels.pkl"
    records = pickle.loads(metadata.read_bytes())
    records[0].pop("annos")
    records[1]["annos"] = {"name": np.array([], dtype=str), "gt_boxes_lidar": np.empty((0, 7))}
    metadata.write_bytes(pickle.dumps(records))
    loader = _loader(sources)
    assert loader.load_point_cloud("waymo:context:1000000").annotations.status == AnnotationStatus.NOT_PROVIDED
    empty = loader.load_point_cloud("waymo:context:1200000").annotations
    assert empty.status == AnnotationStatus.AVAILABLE and empty.boxes == ()


def test_missing_files_and_unknown_ids_use_contract_errors(sources):
    loader = _loader(sources)
    (sources[1] / "segment-context_with_camera_labels/0000.npy").unlink()
    with pytest.raises(DatasetIOError):
        loader.load_point_cloud("waymo:context:1000000")
    with pytest.raises(UnknownIdentifierError):
        loader.load_cameras("waymo:unknown:1000000")


def test_additional_classes_are_declared_without_eager_metadata_loading(sources):
    metadata = sources[1] / "segment-context_with_camera_labels/segment-context_with_camera_labels.pkl"
    records = pickle.loads(metadata.read_bytes())
    records[0]["annos"]["name"] = np.array(["Vehicle", "Custom"])
    metadata.write_bytes(pickle.dumps(records))
    loader = _loader(sources)
    with pytest.raises(DatasetValidationError, match="additional_classes"):
        loader.load_point_cloud("waymo:context:1000000")
    configured = _loader(sources, options={"additional_classes": ["Custom"]})
    assert "Custom" in {item.class_id for item in configured.metadata.classes}
    assert configured.load_point_cloud("waymo:context:1000000").annotations.boxes[1].class_id == "Custom"


def test_rgb_success_is_not_published_when_acquisition_fails(sources, monkeypatch):
    loader = _loader(sources)
    frame = loader.load_point_cloud("waymo:context:1000000")
    native = loader._native(frame.info)
    loader._pointset(frame.info)  # Keep row correspondence fixed for this staged failure.
    without_poses = replace(native, lasers=tuple(replace(laser, returns=tuple(
        replace(item, range_image_pose_compressed=b"") for item in laser.returns)) for laser in native.lasers))
    monkeypatch.setattr(loader, "_native", lambda info, **kwargs: without_poses)
    with pytest.raises(DataUnavailableError, match="acquisition geometry"):
        loader.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=True, acquisition_geometry=True))
    assert frame.rgb is None and frame.acquisition is None


@pytest.mark.parametrize("kwargs", [
    {"storage_format": "kitti"}, {"release": "1.6.1"}, {"source_paths": {"other": "/tmp"}},
    {"options": {"unknown": True}}, {"options": {"additional_classes": "Custom"}},
    {"storage_format": "native", "options": {"processed_tag": "tag"}},
])
def test_unknown_configuration_is_not_ignored(sources, kwargs):
    values = {"root": sources[0], **kwargs}
    with pytest.raises(DatasetConfigurationError):
        WaymoLoader(DatasetConfig(**values))


def test_existing_directory_without_waymo_data_is_configuration_error(tmp_path):
    with pytest.raises(DatasetConfigurationError, match="TFRecords not found"):
        WaymoLoader(DatasetConfig(root=tmp_path, storage_format="native"))


def test_native_point_loading_does_not_decode_jpegs_or_validate_camera_calibration(sources, monkeypatch):
    frame = _proto()
    frame.images[0].image = b"invalid-jpeg-unused-by-point-read"
    frame.context.camera_calibrations[0].intrinsic[:] = []
    path = sources[0] / "validation/segment-context_with_camera_labels.tfrecord"
    path.write_bytes(_record(frame))
    loader = _loader(sources, processed=False)
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("Point reads must not decode camera images."))
    cloud = loader.load_point_cloud("waymo:context:1000000")
    assert cloud.points.shape == (3, 3) and len(cloud.annotations.boxes) == 2
    assert all(box.timing_override is None for box in cloud.annotations.boxes)


def test_native_camera_loading_ignores_invalid_laser_metadata_and_never_decodes_ranges(sources, monkeypatch):
    frame = _proto()
    frame.context.laser_calibrations[0].extrinsic.transform[:] = []
    frame.lasers[0].ri_return1.range_image_compressed = b"invalid-unused-range"
    path = sources[0] / "validation/segment-context_with_camera_labels.tfrecord"
    path.write_bytes(_record(frame))
    loader = _loader(sources, processed=False)
    import visual_utils.waymo.loader as implementation
    monkeypatch.setattr(implementation, "decode_lidar", lambda *args: pytest.fail("Camera reads must not decode range images."))
    cameras = loader.load_cameras("waymo:context:1000000")
    assert len(cameras.cameras) == 1 and len(cameras.annotations.boxes) == 2
    assert cameras.cameras[0].annotations.timing.alignment == AnnotationAlignment.MIXED


def test_native_reader_modality_cache_keys_do_not_return_incomplete_other_payload(sources):
    reader = NativeWaymoReader(sources[0])
    points = reader.read_frame("context", 0, include_cameras=False)
    cameras = reader.read_frame("context", 0, include_points=False)
    assert points.lasers and points.cameras == ()
    assert cameras.cameras and cameras.lasers == () and cameras.laser_calibrations == ()
    combined = reader.read_frame("context", 0)
    assert combined.cameras and combined.lasers
    assert len(reader._frames) == 2
    with pytest.raises(ValueError):
        combined.cameras[0].image_rgb.setflags(write=True)


def test_failed_rolling_shutter_projection_has_unknown_depth_not_false_negative(sources, monkeypatch):
    loader = _loader(sources, processed=False)
    info = loader.list_frames("context", modality=Modality.CAMERAS)[0]
    native = loader._native(info, include_points=False)
    camera = native.cameras[0]
    factor = WaymoCameraProjection(camera)._time_factor
    velocity = np.zeros(6)
    velocity[1] = 5 / factor  # Singular shutter equation Jacobian for a point 5 m ahead.
    native = replace(native, cameras=(replace(camera, velocity=velocity),))
    monkeypatch.setattr(loader, "_native", lambda *args, **kwargs: native)
    projection = loader.load_cameras("waymo:context:1000000").cameras[0].projection_model
    result = projection.project_reference(np.array([[5., 0, 0], [-5., 0, 0]]))
    assert np.isnan(result.uv).all() and not result.valid_mask.any()
    assert np.isnan(result.depth[0]) and result.depth[1] == -5


def test_native_point_read_distinguishes_missing_lasers_from_valid_empty_ranges(sources):
    path = sources[0] / "validation/segment-context_with_camera_labels.tfrecord"
    proto = _proto()
    proto.ClearField("lasers")
    path.write_bytes(_record(proto))
    loader = _loader(sources, processed=False)
    with pytest.raises(DataUnavailableError, match="no LiDAR observations"):
        loader.load_point_cloud("waymo:context:1000000")
    assert loader.load_cameras("waymo:context:1000000").cameras
    proto = _proto()
    for laser in proto.lasers:
        for item in (laser.ri_return1, laser.ri_return2):
            if item.range_image_compressed:
                item.range_image_compressed = _matrix([[[0, 0, 0, -1], [0, 0, 0, -1]]])
    path.write_bytes(_record(proto))
    empty = _loader(sources, processed=False).load_point_cloud("waymo:context:1000000")
    assert empty.points.shape == (0, 3)


@pytest.mark.parametrize("selected", ["parent", "workspace", "processed"])
def test_automatic_source_discovery_prefers_processed_and_pairs_native_records(sources, tmp_path, selected):
    from visual_utils.waymo.processed import DEFAULT_PROCESSED_TAG

    root = tmp_path / "dataset"
    workspace = root / "waymo_pcdet_workspace"
    processed = workspace / DEFAULT_PROCESSED_TAG
    native = root / "waymo_og_format_unpacked"
    shutil.copytree(sources[1], processed)
    shutil.copytree(sources[0], native)
    path = {"parent": root, "workspace": workspace, "processed": processed}[selected]
    with WaymoLoader(DatasetConfig(root=path)) as loader:
        frame = loader.load_point_cloud("waymo:context:1200000")
        assert frame.source_id.endswith(DEFAULT_PROCESSED_TAG)
        assert frame.info.index == 1  # Display index differs from original NPY index 2.
        prepared = loader.prepare_point_cloud(frame, request=PointCloudPreparation(rgb=True))
        assert prepared.rgb.valid_mask.any()
    with WaymoLoader(DatasetConfig(root=root, storage_format="native")) as loader:
        assert loader.load_point_cloud(frame.info.frame_id).source_id == "waymo:native:all-lidars-both-returns"


def test_malformed_processed_source_does_not_fall_back_to_valid_native_data(sources, tmp_path):
    from visual_utils.waymo.processed import DEFAULT_PROCESSED_TAG

    root = tmp_path / "dataset"
    shutil.copytree(sources[0], root / "waymo_og_format_unpacked")
    (root / "waymo_pcdet_workspace" / DEFAULT_PROCESSED_TAG).mkdir(parents=True)
    with pytest.raises(DatasetValidationError, match="No complete OpenPCDet"):
        WaymoLoader(DatasetConfig(root=root))


def test_ambiguous_processed_sources_require_an_explicit_tag(sources, tmp_path):
    root = tmp_path / "workspace"
    first, second = root / "waymo_processed_data_first", root / "waymo_processed_data_second"
    shutil.copytree(sources[1], first)
    shutil.copytree(sources[1], second)
    with pytest.raises(DatasetConfigurationError, match="Several Waymo processed directories"):
        WaymoLoader(DatasetConfig(root=root))
    with WaymoLoader(DatasetConfig(root=root, options={"processed_tag": second.name})) as loader:
        assert loader.load_point_cloud("waymo:context:1000000").source_id.endswith(second.name)


def test_explicit_processed_tag_can_select_a_nonstandard_directory_name(sources, tmp_path):
    root = tmp_path / "workspace"
    profile = root / "research-profile"
    shutil.copytree(sources[1], profile)
    with WaymoLoader(DatasetConfig(root=root, options={"processed_tag": profile.name})) as loader:
        assert loader.load_point_cloud("waymo:context:1000000").source_id.endswith(profile.name)


def test_processed_metadata_stays_lazy_and_sparse_indices_select_original_files(sources, monkeypatch):
    from visual_utils.waymo import processed

    reads = []
    original = processed._NumpyMetadataUnpickler.load

    def observe(stream):
        reads.append(True)
        return original(stream)

    monkeypatch.setattr(processed._NumpyMetadataUnpickler, "load", observe)
    with _loader(sources, native=False) as loader:
        assert loader.list_scenes()[0].frame_counts[Modality.POINT_CLOUD] is None
        assert not reads
        frames = loader.list_frames("context", modality=Modality.POINT_CLOUD)
        assert len(reads) == 1
        cloud = loader.load_point_cloud(frames[1].frame_id)
        assert len(reads) == 1 and cloud.info.index == 1
        expected = np.load(sources[1] / "segment-context_with_camera_labels/0002.npy")
        np.testing.assert_array_equal(cloud.points, expected[:, :3])
        np.testing.assert_array_equal(cloud.attributes["nlz"], expected[:, 5])


def test_processed_metadata_rejects_non_numpy_pickle_globals(sources):
    from pathlib import Path

    metadata = sources[1] / "segment-context_with_camera_labels/segment-context_with_camera_labels.pkl"
    metadata.write_bytes(pickle.dumps(Path("unexpected-object")))
    with _loader(sources, native=False) as loader:
        with pytest.raises(DatasetValidationError, match="Unsupported Waymo metadata pickle global"):
            loader.list_frames("context", modality=Modality.POINT_CLOUD)


def test_ground_truth_database_directory_is_not_a_processed_cloud_profile(sources, tmp_path):
    root = tmp_path / "dataset"
    shutil.copytree(sources[0], root / "waymo_og_format_unpacked")
    (root / "waymo_pcdet_workspace/waymo_processed_data_v0_5_0_gt_database_train").mkdir(parents=True)
    with WaymoLoader(DatasetConfig(root=root)) as loader:
        assert loader.load_point_cloud("waymo:context:1000000").source_id == "waymo:native:all-lidars-both-returns"
