"""Lazy native Waymo reader: one selected TFRecord sequence and two frames."""

from __future__ import annotations

import io
import math
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from .models import (
    NativeBox, NativeCamera, NativeCameraLabels, NativeFrame, NativeFrameHeader,
    NativeLaser, NativeLaserCalibration, NativeRangeImage, NativeSequence, NativeWaymoError,
)
from .proto import parse_frame
from .records import index_records, read_identity, read_record

_CLASSES = {1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4: "Cyclist"}
_SPLITS = {"train": "training", "val": "validation", "test": "testing", "all": None,
           "training": "training", "validation": "validation", "testing": "testing", None: None}


def _array(value, shape, context):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise NativeWaymoError(f"Invalid {context}.") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise NativeWaymoError(f"{context} must have finite shape {shape}.")
    return array


def _transform(value, context):
    matrix = _array(value, (16,), context).reshape(4, 4)
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-7)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5)):
        raise NativeWaymoError(f"{context} must be a proper rigid transform.")
    return matrix


def _box(label, *, image=False, box=None, include_camera_sync=True):
    if label.type not in _CLASSES:
        return None
    value = label.box if box is None else box
    center = _array([value.center_x, value.center_y, value.center_z], (3,), "Native box center")
    size = _array([value.length, value.width, value.height], (3,), "Native box LWH")
    if np.any(size[:2] <= 0) or (not image and size[2] <= 0) or size[2] < 0 or not math.isfinite(value.heading):
        raise NativeWaymoError("Native boxes require positive dimensions and finite heading.")
    synced = None
    if include_camera_sync and box is None and label.HasField("camera_synced_box") and label.camera_synced_box.ListFields():
        synced = _box(label, box=label.camera_synced_box)
    return NativeBox(center, size, float(value.heading), _CLASSES[label.type], label.id,
                     label.most_visible_camera_name, synced)


def frame_from_proto(value, *, include_points=True, include_cameras=True) -> NativeFrame:
    if type(include_points) is not bool or type(include_cameras) is not bool:
        raise NativeWaymoError("Native modality selection flags must be booleans.")
    pose = _transform(value.pose.transform, "Frame vehicle pose")
    camera_calibrations = {}
    for calibration in (value.context.camera_calibrations if include_cameras else ()):
        name = int(calibration.name)
        if name not in range(1, 6) or name in camera_calibrations:
            raise NativeWaymoError("Native camera calibrations require unique IDs 1–5.")
        camera_calibrations[name] = calibration
    cameras = []
    names = set()
    for image in (value.images if include_cameras else ()):
        name = int(image.name)
        if name not in range(1, 6) or name in names:
            raise NativeWaymoError("Native camera IDs must be unique official IDs 1–5.")
        names.add(name)
        calibration = camera_calibrations.get(name)
        if calibration is None:
            raise NativeWaymoError(f"Native camera {name} has no calibration.")
        intrinsic = _array(calibration.intrinsic, (9,), f"Camera {name} intrinsic")
        if np.any(intrinsic[:2] <= 0):
            raise NativeWaymoError(f"Camera {name} focal lengths must be positive.")
        if calibration.width <= 0 or calibration.height <= 0 or calibration.width * calibration.height > 32_000_000:
            raise NativeWaymoError(f"Invalid camera {name} image dimensions.")
        try:
            with Image.open(io.BytesIO(image.image)) as decoded:
                if decoded.size != (calibration.width, calibration.height):
                    raise NativeWaymoError(f"Camera {name} image dimensions do not match calibration.")
                rgb = decoded.convert("RGB")
                # Immutable bytes backing lets normalized camera records reuse
                # decoded pixels safely, even after cache eviction or close.
                pixels = np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(rgb.height, rgb.width, 3)
        except (OSError, ValueError) as exc:
            raise NativeWaymoError(f"Cannot decode native camera {name} image: {exc}") from exc
        timing = [image.pose_timestamp, image.shutter, image.camera_trigger_time, image.camera_readout_done_time]
        if not np.isfinite(timing).all() or image.shutter < 0 or image.camera_readout_done_time < image.camera_trigger_time:
            raise NativeWaymoError(f"Invalid camera {name} exposure timing.")
        velocity = _array([image.velocity.v_x, image.velocity.v_y, image.velocity.v_z,
                           image.velocity.w_x, image.velocity.w_y, image.velocity.w_z], (6,), "Camera velocity")
        cameras.append(NativeCamera(name, pixels, intrinsic,
            _transform(calibration.extrinsic.transform, f"Camera {name} extrinsic"),
            _transform(image.pose.transform, f"Camera {name} vehicle pose"),
            float(timing[0]), velocity, float(timing[1]), float(timing[2]), float(timing[3]),
            int(calibration.rolling_shutter_direction)))
    calibrations = []
    calibration_ids = set()
    for calibration in (value.context.laser_calibrations if include_points else ()):
        name = int(calibration.name)
        if name not in range(1, 6) or name in calibration_ids:
            raise NativeWaymoError("Native laser calibrations require unique IDs 1–5.")
        calibration_ids.add(name)
        inclinations = np.asarray(calibration.beam_inclinations, dtype=np.float64)
        if not np.isfinite(inclinations).all() or not np.isfinite([calibration.beam_inclination_min, calibration.beam_inclination_max]).all():
            raise NativeWaymoError("Native beam inclinations must be finite.")
        calibrations.append(NativeLaserCalibration(name,
            _transform(calibration.extrinsic.transform, f"Laser {name} extrinsic"), inclinations,
            float(calibration.beam_inclination_min), float(calibration.beam_inclination_max)))
    lasers = []
    laser_ids = set()
    for laser in (value.lasers if include_points else ()):
        name = int(laser.name)
        if name not in calibration_ids or name in laser_ids:
            raise NativeWaymoError(f"Laser {name} has missing or duplicate calibration.")
        laser_ids.add(name)
        returns = tuple(NativeRangeImage(item.range_image_compressed, item.camera_projection_compressed,
                                        item.range_image_pose_compressed)
                        for item in (laser.ri_return1, laser.ri_return2))
        lasers.append(NativeLaser(name, returns))

    def labels(records):
        return tuple(box for label in records if (box := _box(label, include_camera_sync=include_cameras)) is not None)

    def camera_labels(records):
        return tuple(NativeCameraLabels(int(group.name), tuple(
            box for label in group.labels if (box := _box(label, image=True)) is not None)) for group in records)

    return NativeFrame(value.context.name, int(value.timestamp_micros), pose,
        tuple(sorted(cameras, key=lambda camera: camera.name)), labels(value.laser_labels),
        tuple(sorted(lasers, key=lambda laser: laser.name)), tuple(sorted(calibrations, key=lambda item: item.name)),
        camera_labels(value.camera_labels) if include_cameras else (),
        camera_labels(value.projected_lidar_labels) if include_cameras else ())


def find_native_root(root: Path) -> Path | None:
    """Find TFRecords by filenames, including a paired dataset beside a workspace."""
    candidates = (root, root / "waymo_og_format_unpacked",
                  *(ancestor / "waymo_og_format_unpacked" for ancestor in root.parents))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_dir() and any(
            next(folder.glob("*.tfrecord"), None) is not None
            for folder in (candidate, *(candidate / split for split in ("training", "validation", "testing")))
        ):
            return candidate
    return None


class NativeWaymoReader:
    def __init__(self, root, *, split=None):
        try:
            requested_split = _SPLITS[split]
        except (KeyError, TypeError):
            raise NativeWaymoError(f"Unsupported native Waymo split {split!r}.") from None
        self.root = Path(root).expanduser().resolve()
        native_root = find_native_root(self.root)
        if native_root is None:
            raise NativeWaymoError(f"Native Waymo TFRecords not found under {self.root}. Select waymo_og_format_unpacked or its dataset parent.")
        self.native_root = native_root
        sequences = {}
        folders = [(native_root / name, name) for name in ("training", "validation", "testing")]
        folders.append((native_root, native_root.name if native_root.name in {"training", "validation", "testing"} else "all"))
        for folder, split_name in folders:
            if requested_split is not None and split_name not in {requested_split, "all"}:
                continue
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.tfrecord")):
                context = path.stem
                for prefix in ("training_", "validation_", "testing_"):
                    context = context.removeprefix(prefix)
                context = context.removeprefix("segment-").removesuffix("_with_camera_labels")
                if context in sequences:
                    raise NativeWaymoError(f"Multiple native TFRecords identify context {context!r}.")
                sequences[context] = NativeSequence(context, path, split_name)
        self._sequences = sequences
        self._offsets = {}
        self._headers = {}
        self._frames = OrderedDict()
        self._lock = threading.RLock()

    def close(self) -> None:
        """Release cached payloads and record indexes after pending reads finish."""
        with self._lock:
            self._frames.clear()
            self._headers.clear()
            self._offsets.clear()

    def list_sequences(self):
        return tuple(self._sequences[name] for name in sorted(self._sequences))

    def _sequence(self, context_name):
        try:
            return self._sequences[context_name]
        except KeyError:
            raise NativeWaymoError(f"Native TFRecord is unavailable for context {context_name!r}.") from None

    def _records(self, context_name):
        if context_name not in self._offsets:
            self._offsets[context_name] = index_records(self._sequence(context_name).path)
        return self._offsets[context_name]

    def list_frame_headers(self, context_name):
        with self._lock:
            if context_name not in self._headers:
                sequence = self._sequence(context_name)
                result = []
                try:
                    with sequence.path.open("rb") as stream:
                        for index, record in enumerate(self._records(context_name)):
                            context, timestamp = read_identity(stream, record)
                            if context != context_name:
                                raise NativeWaymoError("TFRecord context disagrees with its filename/catalogue.")
                            result.append(NativeFrameHeader(index, timestamp))
                except OSError as exc:
                    raise NativeWaymoError(f"Cannot read native sequence metadata: {exc}") from exc
                self._headers[context_name] = tuple(result)
            return self._headers[context_name]

    def frame_timestamps(self, context_name):
        return tuple(header.timestamp_us for header in self.list_frame_headers(context_name))

    def list_frames(self, context_name):
        return self.list_frame_headers(context_name)

    def read_frame(self, context_name, frame_index, timestamp_us=None, *, include_points=True, include_cameras=True):
        with self._lock:
            if type(include_points) is not bool or type(include_cameras) is not bool:
                raise NativeWaymoError("Native modality selection flags must be booleans.")
            if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
                raise NativeWaymoError("Native frame index must be an integer.")
            key = context_name, int(frame_index), include_points, include_cameras
            if key not in self._frames:
                records = self._records(context_name)
                if not 0 <= frame_index < len(records):
                    raise NativeWaymoError(f"Native frame index {frame_index} is outside 0–{len(records)-1}.")
                proto = parse_frame(read_record(self._sequence(context_name).path, records[frame_index]))
                if proto.context.name != context_name:
                    raise NativeWaymoError("Selected native Frame context does not match the requested context.")
                self._frames[key] = frame_from_proto(proto, include_points=include_points, include_cameras=include_cameras)
                while len(self._frames) > 2:
                    self._frames.popitem(last=False)
            frame = self._frames[key]
            if timestamp_us is not None and frame.timestamp_us != timestamp_us:
                raise NativeWaymoError(f"Native timestamp {frame.timestamp_us} does not match expected {timestamp_us}.")
            self._frames.move_to_end(key)
            return frame
