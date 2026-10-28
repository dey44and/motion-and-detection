"""Minimal compatible Waymo v1.6.1 protobuf descriptors and bounded matrices.

Field numbers follow the Apache-2.0 Waymo Open Dataset definitions:
https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/dataset.proto
https://github.com/waymo-research/waymo-open-dataset/blob/v1.6.1/src/waymo_open_dataset/label.proto

Only needed fields are described, in a private descriptor pool. TensorFlow,
generated Waymo Python modules, map/keypoint protos, and their imports are not
required. Unknown fields are discarded after parsing.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import DecodeError

from .models import NativeWaymoError

MAX_MATRIX_BYTES = 96 * 1024 * 1024
_PREFIX = ".dataset_utils.waymo."


def _make_messages():
    file = descriptor_pb2.FileDescriptorProto(name="dataset_utils_waymo_minimal.proto", package="dataset_utils.waymo", syntax="proto2")
    kind = descriptor_pb2.FieldDescriptorProto
    types = {"double": kind.TYPE_DOUBLE, "float": kind.TYPE_FLOAT, "int32": kind.TYPE_INT32,
             "int64": kind.TYPE_INT64, "bytes": kind.TYPE_BYTES, "string": kind.TYPE_STRING}

    def message(name, fields):
        description = file.message_type.add(name=name)
        for field_name, number, field_type, repeated in fields:
            field = description.field.add(name=field_name, number=number)
            field.label = kind.LABEL_REPEATED if repeated else kind.LABEL_OPTIONAL
            field.type = types.get(field_type, kind.TYPE_MESSAGE)
            if field_type not in types:
                field.type_name = _PREFIX + field_type
        return description

    message("Transform", [("transform", 1, "double", True)])
    message("Velocity", [("v_x", 1, "float", False), ("v_y", 2, "float", False), ("v_z", 3, "float", False),
                         ("w_x", 4, "double", False), ("w_y", 5, "double", False), ("w_z", 6, "double", False)])
    message("CameraCalibration", [("name", 1, "int32", False), ("intrinsic", 2, "double", True),
             ("extrinsic", 3, "Transform", False), ("width", 4, "int32", False), ("height", 5, "int32", False),
             ("rolling_shutter_direction", 6, "int32", False)])
    message("LaserCalibration", [("name", 1, "int32", False), ("beam_inclinations", 2, "double", True),
             ("beam_inclination_min", 3, "double", False), ("beam_inclination_max", 4, "double", False),
             ("extrinsic", 5, "Transform", False)])
    message("Context", [("name", 1, "string", False), ("camera_calibrations", 2, "CameraCalibration", True),
                        ("laser_calibrations", 3, "LaserCalibration", True)])
    message("RangeImage", [("range_image_compressed", 2, "bytes", False),
             ("camera_projection_compressed", 3, "bytes", False), ("range_image_pose_compressed", 4, "bytes", False)])
    message("Laser", [("name", 1, "int32", False), ("ri_return1", 2, "RangeImage", False), ("ri_return2", 3, "RangeImage", False)])
    message("CameraImage", [("name", 1, "int32", False), ("image", 2, "bytes", False),
             ("pose", 3, "Transform", False), ("velocity", 4, "Velocity", False), ("pose_timestamp", 5, "double", False),
             ("shutter", 6, "double", False), ("camera_trigger_time", 7, "double", False),
             ("camera_readout_done_time", 8, "double", False)])
    message("Box", [("center_x", 1, "double", False), ("center_y", 2, "double", False),
             ("center_z", 3, "double", False), ("width", 4, "double", False), ("length", 5, "double", False),
             ("height", 6, "double", False), ("heading", 7, "double", False)])
    message("Label", [("box", 1, "Box", False), ("type", 3, "int32", False), ("id", 4, "string", False),
             ("most_visible_camera_name", 11, "string", False), ("camera_synced_box", 12, "Box", False)])
    message("CameraLabels", [("name", 1, "int32", False), ("labels", 2, "Label", True)])
    message("Frame", [("context", 1, "Context", False), ("timestamp_micros", 2, "int64", False),
             ("pose", 3, "Transform", False), ("images", 4, "CameraImage", True), ("lasers", 5, "Laser", True),
             ("laser_labels", 6, "Label", True), ("camera_labels", 8, "CameraLabels", True),
             ("projected_lidar_labels", 9, "CameraLabels", True)])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    def create(name):
        descriptor = pool.FindMessageTypeByName("dataset_utils.waymo." + name)
        if hasattr(message_factory, "GetMessageClass"):
            return message_factory.GetMessageClass(descriptor)
        return message_factory.MessageFactory(pool).GetPrototype(descriptor)
    return {name: create(name) for name in ("Frame", "Context")}


_MESSAGES = _make_messages()
FrameProto = _MESSAGES["Frame"]
ContextProto = _MESSAGES["Context"]


def parse_frame(payload: bytes):
    frame = FrameProto()
    try:
        frame.ParseFromString(payload)
        frame.DiscardUnknownFields()
    except DecodeError as exc:
        raise NativeWaymoError(f"Invalid native Waymo Frame protobuf: {exc}") from exc
    if not frame.context.name or not frame.HasField("timestamp_micros") or frame.timestamp_micros < 0:
        raise NativeWaymoError("Native Frame is missing context.name or timestamp_micros.")
    return frame


def varint(data: bytes | memoryview, position: int, end: int | None = None):
    end = len(data) if end is None else end
    value = 0
    for shift in range(0, 70, 7):
        if position >= end:
            raise NativeWaymoError("Truncated protobuf varint.")
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            if shift == 63 and byte > 1:
                raise NativeWaymoError("Protobuf varint exceeds 64 bits.")
            return value, position
    raise NativeWaymoError("Malformed protobuf varint.")


def _fields(data):
    position = 0
    while position < len(data):
        tag, position = varint(data, position)
        number, wire = tag >> 3, tag & 7
        if not number:
            raise NativeWaymoError("Invalid protobuf field zero.")
        if wire == 0:
            value, position = varint(data, position)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[position:position + size]
            position += size
            if len(value) != size:
                raise NativeWaymoError("Truncated protobuf scalar.")
        elif wire == 2:
            size, position = varint(data, position)
            if size > len(data) - position:
                raise NativeWaymoError("Truncated protobuf byte field.")
            value = data[position:position + size]
            position += size
        else:
            raise NativeWaymoError(f"Unsupported protobuf wire type {wire}.")
        yield number, wire, value


def _packed_int32(data):
    """Decode packed protobuf integers in parallel, including signed 10-byte values."""
    octets = np.frombuffer(data, dtype=np.uint8)
    if not len(octets):
        return np.empty(0, dtype=np.int32)
    ends = np.flatnonzero(octets < 128)
    if not len(ends) or ends[-1] != len(octets) - 1:
        raise NativeWaymoError("Truncated packed integer matrix varint.")
    if len(ends) > MAX_MATRIX_BYTES // 4:
        raise NativeWaymoError("Waymo integer matrix exceeds the size limit.")
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts + 1
    if np.any(lengths > 10) or np.any((lengths == 10) & (octets[ends] > 1)):
        raise NativeWaymoError("Packed matrix varint exceeds 64 bits.")
    values = np.zeros(len(ends), dtype=np.uint64)
    for offset in range(int(lengths.max())):
        active = lengths > offset
        bits = (octets[starts[active] + offset] & 127).astype(np.uint64)
        values[active] |= bits << np.uint64(7 * offset)
    return values.astype(np.uint32).view(np.int32)


def decode_matrix(compressed: bytes, *, integer: bool = False) -> np.ndarray:
    """Decode bounded packed/unpacked Waymo matrices, preserving signed int32."""
    inflater = zlib.decompressobj()
    try:
        data = inflater.decompress(compressed, MAX_MATRIX_BYTES + 1)
        if len(data) > MAX_MATRIX_BYTES or inflater.unconsumed_tail:
            raise NativeWaymoError("Waymo matrix exceeds the decompression size limit.")
        if not inflater.eof or inflater.unused_data:
            raise NativeWaymoError("Truncated or trailing compressed Waymo matrix data.")
    except zlib.error as exc:
        raise NativeWaymoError(f"Invalid compressed Waymo matrix: {exc}") from exc
    pieces, shape = [], []
    for number, wire, value in _fields(memoryview(data)):
        if number == 1:
            if integer and wire == 0:
                pieces.append(np.array([((value & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000], dtype=np.int32))
            elif integer and wire == 2:
                pieces.append(_packed_int32(value))
            elif not integer and wire in (2, 5):
                if len(value) % 4:
                    raise NativeWaymoError("Packed float matrix has an incomplete element.")
                pieces.append(np.frombuffer(value, dtype="<f4"))
            else:
                raise NativeWaymoError("Waymo matrix data has an incompatible wire type.")
        elif number == 2 and wire == 2:
            for shape_number, shape_wire, item in _fields(value):
                if shape_number != 1:
                    continue
                if shape_wire == 0:
                    shape.append(item)
                elif shape_wire == 2:
                    position = 0
                    while position < len(item):
                        dimension, position = varint(item, position)
                        shape.append(dimension)
                else:
                    raise NativeWaymoError("Invalid Waymo matrix shape.")
    if not shape or len(shape) > 6 or any(dimension <= 0 for dimension in shape):
        raise NativeWaymoError("Waymo matrix requires a nonempty positive shape.")
    count = math.prod(shape)
    if count > MAX_MATRIX_BYTES // 4:
        raise NativeWaymoError("Waymo matrix dimensions exceed the size limit.")
    values = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int32 if integer else np.float32)
    if count != values.size:
        raise NativeWaymoError(f"Waymo matrix shape {shape} does not match {values.size} values.")
    if not integer and not np.isfinite(values).all():
        raise NativeWaymoError("Waymo matrix contains non-finite values.")
    return values.reshape(shape)
