"""Synthetic native protos verify framing, row alignment, motion, and RGB."""

from __future__ import annotations

import io
import struct
import zlib
import warnings
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from dataset_utils.waymo import NativeWaymoError, NativeWaymoReader, colorize_points, decode_lidar
from dataset_utils.waymo.proto import FrameProto, decode_matrix, parse_frame
from dataset_utils.waymo.range_image import rotation_rpy
from dataset_utils.waymo.records import MAX_RECORD_BYTES, crc32c, index_records, masked_crc32c, read_record


def varint(value):
    value &= (1 << 64) - 1
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    return bytes(result) + bytes([value])


def matrix(values, *, integer=False, packed_shape=False):
    values = np.asarray(values, dtype=np.int32 if integer else np.float32)
    data = b"".join(varint(int(value)) for value in values.ravel()) if integer else values.astype("<f4").tobytes()
    dimensions = b"".join(varint(dimension) for dimension in values.shape)
    shape = b"\x0a" + varint(len(dimensions)) + dimensions if packed_shape else b"".join(b"\x08" + varint(dimension) for dimension in values.shape)
    return zlib.compress(b"\x0a" + varint(len(data)) + data + b"\x12" + varint(len(shape)) + shape)


def transform(message, translation=(0, 0, 0)):
    value = np.eye(4)
    value[:3, 3] = translation
    message.transform.extend(value.ravel())


def make_frame(timestamp=1_000_000):
    frame = FrameProto()
    frame.context.name = "context"
    frame.timestamp_micros = timestamp
    transform(frame.pose, [2, 0, 0])
    for name in (1, 2):
        calibration = frame.context.camera_calibrations.add()
        calibration.name = name
        calibration.intrinsic.extend([100, 100, 1, .5, 0, 0, 0, 0, 0])
        calibration.width, calibration.height = 2, 1
        calibration.rolling_shutter_direction = 2
        transform(calibration.extrinsic)
        image = frame.images.add()
        image.name = name
        image.pose_timestamp = timestamp / 1e6
        image.shutter = .001
        image.camera_trigger_time = image.pose_timestamp - .005
        image.camera_readout_done_time = image.pose_timestamp + .005
        transform(image.pose, [2, 0, 0])
        pixels = np.array([[[255, 0, 0], [0, 255, 0]]] if name == 1 else [[[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        image.image = stream.getvalue()
    for name in (1, 2):
        calibration = frame.context.laser_calibrations.add()
        calibration.name = name
        transform(calibration.extrinsic, [1, 0, 0] if name == 2 else [0, 0, 0])
        calibration.beam_inclinations.append(0)
    # Deliberately add lasers out of order: output must follow official order.
    side = frame.lasers.add()
    side.name = 2
    side.ri_return1.range_image_compressed = matrix([[[1, 10, 20, -1], [0, -1, -1, -1]]])
    top = frame.lasers.add()
    top.name = 1
    top.ri_return1.range_image_compressed = matrix([[[10, 1, 2, -1], [0, -1, -1, -1]]])
    top.ri_return2.range_image_compressed = matrix([[[0, -1, -1, -1], [20, 3, 4, 1]]])
    top.ri_return1.range_image_pose_compressed = matrix([[[0, 0, 0, 5, 0, 0], [0, 0, 0, 5, 0, 0]]])
    top.ri_return1.camera_projection_compressed = matrix([[[1, 0, 0, 2, 1, 0], [0, 0, 0, 0, 0, 0]]], integer=True)
    top.ri_return2.camera_projection_compressed = matrix([[[0, 0, 0, 0, 0, 0], [2, 1, 0, 1, 0, 0]]], integer=True)
    label = frame.laser_labels.add()
    label.type, label.id = 1, "vehicle"
    label.box.center_x = 5
    label.box.length, label.box.width, label.box.height, label.box.heading = 4, 2, 1, .3
    # Empty optional synced boxes occur in real v1.4.2 data for invisible labels.
    label.camera_synced_box.SetInParent()
    return frame


def record(payload):
    length = struct.pack("<Q", len(payload))
    return length + struct.pack("<I", masked_crc32c(length)) + payload + struct.pack("<I", masked_crc32c(payload))


@pytest.fixture
def native_root(tmp_path):
    folder = tmp_path / "waymo_og_format_unpacked" / "validation"
    folder.mkdir(parents=True)
    path = folder / "segment-context_with_camera_labels.tfrecord"
    path.write_bytes(b"".join(record(make_frame(1_000_000 + index * 100_000).SerializeToString()) for index in range(3)))
    return tmp_path, path


def test_reader_catalogue_and_headers_are_lazy(native_root, monkeypatch):
    from dataset_utils.waymo import reader

    root, _ = native_root
    adapter = NativeWaymoReader(root, split="val")
    assert [(item.context_name, item.split) for item in adapter.list_sequences()] == [("context", "validation")]
    monkeypatch.setattr(reader.Image, "open", lambda *args, **kwargs: pytest.fail("Headers must not decode JPEGs."))
    headers = adapter.list_frame_headers("context")
    assert [(header.frame_index, header.timestamp_us) for header in headers] == [(0, 1_000_000), (1, 1_100_000), (2, 1_200_000)]
    assert adapter.frame_timestamps("context") == (1_000_000, 1_100_000, 1_200_000)
    assert adapter._frames == {}


def test_native_frame_metadata_and_lru_bound(native_root):
    adapter = NativeWaymoReader(native_root[0])
    frame = adapter.read_frame("context", 0, timestamp_us=1_000_000)
    assert frame is adapter.read_frame("context", 0)
    assert len(frame.cameras) == 2
    camera = frame.cameras[0]
    assert camera.image_rgb.dtype == np.uint8 and camera.image_rgb.shape == (1, 2, 3)
    np.testing.assert_array_equal(camera.intrinsic, [100, 100, 1, .5, 0, 0, 0, 0, 0])
    assert camera.rolling_shutter_direction == 2
    assert frame.boxes[0].class_name == "Vehicle"
    np.testing.assert_array_equal(frame.boxes[0].size, [4, 2, 1])
    assert frame.boxes[0].camera_synced_box is None
    adapter.read_frame("context", 1)
    adapter.read_frame("context", 2)
    assert len(adapter._frames) == 2
    assert ("context", 0) not in adapter._frames


def test_duplicate_camera_calibrations_are_rejected():
    from dataset_utils.waymo.reader import frame_from_proto

    value = make_frame()
    duplicate = value.context.camera_calibrations.add()
    duplicate.CopyFrom(value.context.camera_calibrations[0])
    duplicate.extrinsic.transform[3] = 100
    with pytest.raises(NativeWaymoError, match="camera calibrations require unique"):
        frame_from_proto(value)


@pytest.mark.parametrize("raw_only", [False, True])
def test_float32_coordinate_overflow_is_rejected_without_warnings(raw_only):
    from dataset_utils.waymo.reader import frame_from_proto

    value = make_frame()
    value.context.laser_calibrations[0].extrinsic.transform[3] = 4e38 if raw_only else 1e40
    if raw_only:
        # Motion compensation brings the reference cloud into float32 range,
        # while the acquisition-frame raw coordinates still overflow it.
        poses = np.zeros((1, 2, 6), dtype=np.float32)
        poses[..., 3] = -3e38
        value.lasers[1].ri_return1.range_image_pose_compressed = matrix(poses)
    frame = frame_from_proto(value)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(NativeWaymoError, match="finite float32"):
            decode_lidar(frame)


@pytest.mark.parametrize("index,timestamp,match", [(-1, None, "outside"), (3, None, "outside"), (True, None, "integer"), (0, 0, "timestamp")])
def test_invalid_frame_identity_is_actionable(native_root, index, timestamp, match):
    adapter = NativeWaymoReader(native_root[0])
    with pytest.raises(NativeWaymoError, match=match):
        adapter.read_frame("context", index, timestamp)
    with pytest.raises(NativeWaymoError, match="unavailable"):
        adapter.read_frame("unknown", 0)


def test_both_returns_keep_their_own_camera_associations_and_raw_poses(native_root):
    frame = NativeWaymoReader(native_root[0]).read_frame("context", 0)
    points = decode_lidar(frame)
    np.testing.assert_allclose(points.points, [[3, 10, 0], [3, -20, 0], [1, 1, 0]], atol=1e-6)
    np.testing.assert_allclose(points.raw_points, [[0, 10, 0], [0, -20, 0], [1, 1, 0]], atol=1e-6)
    np.testing.assert_array_equal(points.raw_valid_mask, [True, True, False])
    np.testing.assert_array_equal(points.lidar_id, [1, 1, 2])
    np.testing.assert_array_equal(points.return_id, [0, 1, 0])
    np.testing.assert_array_equal(points.intensity, [1, 3, 10])
    np.testing.assert_array_equal(points.nlz, [-1, 1, -1])
    colors, valid = colorize_points(frame, points)
    np.testing.assert_allclose(colors, [[1, 0, 0], [1, 1, 0], [.5, .5, .5]])
    np.testing.assert_array_equal(valid, [True, True, False])


def test_second_association_used_when_first_pixel_invalid(native_root):
    frame = NativeWaymoReader(native_root[0]).read_frame("context", 0)
    points = decode_lidar(frame)
    projections = points.camera_projections.copy()
    projections[0, 1] = 100
    colors, valid = colorize_points(frame, replace(points, camera_projections=projections))
    np.testing.assert_array_equal(colors[0], [1, 1, 0])
    assert valid[0]


def test_missing_return_is_not_replaced_with_first_return_projection():
    from dataset_utils.waymo.reader import frame_from_proto

    value = make_frame()
    value.lasers[1].ri_return1.ClearField("range_image_compressed")
    value.lasers[1].ri_return2.ClearField("camera_projection_compressed")
    frame = frame_from_proto(value)
    points = decode_lidar(frame)
    np.testing.assert_array_equal(points.return_id, [1, 0])
    np.testing.assert_array_equal(points.camera_projections, np.zeros((2, 6)))
    assert not colorize_points(frame, points)[1].any()


def test_projection_shape_mismatch_rejected():
    from dataset_utils.waymo.reader import frame_from_proto

    value = make_frame()
    value.lasers[1].ri_return2.camera_projection_compressed = matrix(np.zeros((1, 1, 6)), integer=True)
    with pytest.raises(NativeWaymoError, match="corresponding lidar return"):
        decode_lidar(frame_from_proto(value))


@pytest.mark.parametrize("integer", [False, True])
@pytest.mark.parametrize("packed_shape", [False, True])
def test_matrix_packed_shapes_and_signed_integer_values(integer, packed_shape):
    values = np.array([[1, -1, 127], [128, -12345, 0]], dtype=np.int32 if integer else np.float32)
    decoded = decode_matrix(matrix(values, integer=integer, packed_shape=packed_shape), integer=integer)
    np.testing.assert_array_equal(decoded, values)


def test_packed_integer_boundaries_and_mixed_lengths():
    random = np.random.default_rng(17)
    values = random.integers(-(1 << 31), 1 << 31, size=(11, 73), dtype=np.int32)
    values[0, :8] = [0, 1, 127, 128, 16383, 16384, -(1 << 31), (1 << 31) - 1]
    np.testing.assert_array_equal(decode_matrix(matrix(values, integer=True), integer=True), values)


@pytest.mark.parametrize("payload", [b"", b"invalid zlib", zlib.compress(b"\x00"), zlib.compress(b"\x0a\x04\x00"), matrix([1, 2]) + b"trailing"])
def test_malformed_matrix_is_rejected(payload):
    with pytest.raises(NativeWaymoError):
        decode_matrix(payload)


def test_decompression_size_cap(monkeypatch):
    from dataset_utils.waymo import proto

    monkeypatch.setattr(proto, "MAX_MATRIX_BYTES", 32)
    with pytest.raises(NativeWaymoError, match="size limit"):
        decode_matrix(matrix(np.ones((100, 100))))


def test_known_crc32c_vector_and_record_corruption(native_root):
    assert crc32c(b"123456789") == 0xE3069283
    _, path = native_root
    indexed = index_records(path)
    assert len(indexed) == 3
    assert parse_frame(read_record(path, indexed[0])).context.name == "context"
    data = bytearray(path.read_bytes())
    data[20] ^= 1
    path.write_bytes(data)
    with pytest.raises(NativeWaymoError, match="payload CRC"):
        read_record(path, indexed[0])
    data[8] ^= 1
    path.write_bytes(data)
    with pytest.raises(NativeWaymoError, match="length CRC"):
        index_records(path)


@pytest.mark.parametrize("payload", [b"", b"short", record(b"x")[:-1]])
def test_empty_and_truncated_records_rejected(tmp_path, payload):
    path = tmp_path / "bad.tfrecord"
    path.write_bytes(payload)
    with pytest.raises(NativeWaymoError):
        index_records(path)


def test_record_size_limit_checked_before_reading_payload(tmp_path):
    length = struct.pack("<Q", MAX_RECORD_BYTES + 1)
    path = tmp_path / "bad.tfrecord"
    path.write_bytes(length + struct.pack("<I", masked_crc32c(length)))
    with pytest.raises(NativeWaymoError, match="record size"):
        index_records(path)


def test_context_mismatch_is_rejected_without_image_decoding(native_root):
    root, path = native_root
    path.rename(path.with_name("segment-wrong_with_camera_labels.tfrecord"))
    reader = NativeWaymoReader(root)
    with pytest.raises(NativeWaymoError, match="context"):
        reader.list_frame_headers("wrong")
    with pytest.raises(NativeWaymoError, match="context"):
        reader.read_frame("wrong", 0)


def test_full_rpy_transform_and_empty_lidar():
    from dataset_utils.waymo.reader import frame_from_proto

    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(rotation_rpy(0, 0, np.pi / 2), expected, atol=1e-15)
    frame = make_frame()
    frame.ClearField("lasers")
    points = frame_from_proto(frame).points()
    assert points.points.shape == (0, 3)
    assert points.camera_projections.shape == (0, 6)
