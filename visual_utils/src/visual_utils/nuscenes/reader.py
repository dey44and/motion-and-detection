"""Read native nuScenes metadata, LiDAR and images without the nuScenes SDK.

The selected sensor controls keyframe enumeration, not metadata indexing.
Frame tokens are nuScenes sample tokens. Sensor payloads are loaded lazily.
Native class identities are retained; the loader supplies detection display groups.

Format and coordinate conventions:
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DataUnavailableError, DatasetError, DatasetValidationError, UnknownIdentifierError
from ..models import (
    AcquisitionGeometry, AnnotationAlignment, AnnotationSet, AnnotationStatus,
    AnnotationTiming, Box3D, CameraFrameSet, CameraObservation, FrameInfo,
    Modality, PointCloudFrame, PointColors, PointTiming, ProcessingProvenance,
    SceneInfo, TimingQuality,
)
from ..projection import PinholeProjection

_POINT_SOURCE = "nuscenes:native:LIDAR_TOP:compensated-v1"
_CAMERA_SOURCE = "nuscenes:native:camera-keyframes-v1"
_TIMING_ASSUMPTIONS = (
    "Point times are estimated from the nominal HDL-32E firing model, not recorded per return.",
    "One scene-stable azimuth phase and interpolated recorded ego poses undo existing ego compensation.",
    "A missing preceding sweep uses the next sweep interval or a nominal 50 ms interval.",
    "Returns in one firing column share its nominal time; object motion is not compensated.",
)

DETECTION_CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)

_CATEGORY_TO_DETECTION = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

CAMERA_CHANNELS = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
)
_CAMERA_CHANNELS = frozenset(CAMERA_CHANNELS)
_CAMERA_MAX_TIME_DELTA_US = 100_000


def _timestamp(value: Any) -> int | None:
    """Keep optional timing metadata from breaking ordinary lidar viewing."""

    if isinstance(value, bool):
        return None
    try:
        result = int(value)
        if result < 0 or result != value:
            return None
        return result
    except (TypeError, ValueError, OverflowError):
        return None


def _vector(value: Any, length: int, context: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"Invalid {context}: expected {length} finite numbers.") from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise DatasetError(f"Invalid {context}: expected {length} finite numbers.")
    return result


def _quaternion_matrix(value: Any, context: str) -> np.ndarray:
    """Convert a nuScenes [w, x, y, z] quaternion to a rotation matrix."""

    quaternion = _vector(value, 4, context)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise DatasetError(f"Invalid {context}: quaternion must have nonzero norm.")
    w, x, y, z = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _transform(record: dict[str, Any], context: str) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _quaternion_matrix(record.get("rotation"), f"{context} rotation")
    result[:3, 3] = _vector(record.get("translation"), 3, f"{context} translation")
    return result


class NuScenesReader:
    """Indexed lazy sensor reader for mini, trainval and test releases.

    ``sensor`` selects which keyframes appear in scene and frame lists. Both
    sensor indexes share the same JSON tables; reading camera frames never
    opens LiDAR files and reading ordinary LiDAR frames never opens images.
    """

    def __init__(self, *, root: str | Path | None, version: str = "v1.0-mini", sensor: str = "lidar", _validate_active_frames: bool = True):
        if sensor not in ("lidar", "camera"):
            raise DatasetError(f"Unknown nuScenes sensor {sensor!r}: expected 'lidar' or 'camera'.")
        self.sensor = sensor
        self._validate_active_frames = _validate_active_frames
        if root is None:
            raise DatasetError("nuScenes requires a dataset root. Set --root to your extracted nuScenes directory.")
        self.root = Path(root).expanduser().resolve()
        self.version = version
        if (self.root / version / "scene.json").is_file():
            self.table_root = self.root / version
            self._data_roots = (self.root, self.table_root)
        elif (self.root / "scene.json").is_file():
            self.table_root = self.root
            self._data_roots = (
                (self.root.parent, self.root) if self.root.name == version else (self.root, self.root.parent)
            )
        else:
            raise DatasetError(
                f"nuScenes metadata not found at {self.root / version}. "
                "Choose the extracted dataset root or its version directory, and check --version."
            )

        scenes = self._index_table("scene")
        samples = self._index_table("sample")
        self._known_frame_tokens = frozenset(samples)
        self._sample_scene_tokens = {token: sample.get("scene_token") for token, sample in samples.items()}
        self._calibrations = self._index_table("calibrated_sensor")
        self._poses = self._index_table("ego_pose")
        sensors = self._index_table("sensor")
        self._sensor_channels = tuple(
            (sensor.get("channel"), sensor.get("modality")) for sensor in sensors.values()
        )
        self._sensor_by_token = {
            token: (sensor.get("channel"), sensor.get("modality")) for token, sensor in sensors.items()
        }
        categories = self._index_table("category", optional=True)
        instances = self._index_table("instance", optional=True)

        # Reverse-index plain JSON; sample.data and sample.anns only exist after
        # SDK enrichment and are intentionally not required by this adapter.
        self._lidar_by_sample: dict[str, dict[str, Any]] = {}
        self._cameras_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        # Camera viewing keeps only usable keyframes. The much larger sweep
        # indexes are needed only by optional LiDAR timing/RGB preparation.
        if self.sensor == "lidar":
            self._sample_data = self._index_table("sample_data")
            data_records = self._sample_data.values()
        else:
            self._sample_data = {}
            data_records = self._read_table("sample_data")
        self._data_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lidar_sweeps_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._cameras_by_scene: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        self._has_camera_data = False
        for data in data_records:
            if self.sensor == "camera" and not data.get("is_key_frame", False):
                # Retain an availability flag for optional point coloring,
                # including extracts that contain only nonkey camera images.
                # No nonkey records or decoded images are retained here.
                try:
                    calibration = self._calibrations.get(data.get("calibrated_sensor_token"), {})
                    channel, modality = self._sensor_by_token.get(calibration.get("sensor_token"), (None, None))
                    self._has_camera_data |= channel in _CAMERA_CHANNELS and modality == "camera"
                except TypeError:
                    pass
                continue
            sample_token = data.get("sample_token")
            sample = samples.get(sample_token) if isinstance(sample_token, str) else None
            if sample is not None and self.sensor == "lidar":
                # A sensor record with unusable optional calibration may still
                # contribute a valid timestamped ego pose to the trajectory.
                self._data_by_scene[sample.get("scene_token")].append(data)
            # Optional sweep/camera metadata is validated when that feature is
            # requested. Existing lidar-only extracts remain useful.
            try:
                calibration = self._record(self._calibrations, data.get("calibrated_sensor_token"), "calibrated_sensor")
                sensor = self._record(sensors, calibration.get("sensor_token"), "sensor")
            except DatasetError:
                if data.get("is_key_frame", False):
                    raise
                continue
            channel = sensor.get("channel")
            if sample is None:
                if (channel == "LIDAR_TOP" or channel in _CAMERA_CHANNELS) and data.get("is_key_frame", False):
                    self._record(samples, sample_token, "sample")
                continue
            scene_token = sample.get("scene_token")
            if channel in _CAMERA_CHANNELS and sensor.get("modality") == "camera":
                self._has_camera_data = True
                self._cameras_by_scene[scene_token][channel].append(data)
                if data.get("is_key_frame", False):
                    if self.sensor == "camera":
                        self._retain_keyframe(data)
                    cameras = self._cameras_by_sample[sample_token]
                    if channel in cameras:
                        raise DatasetError(f"Multiple {channel} camera keyframes reference sample {sample_token!r}.")
                    cameras[channel] = data
            if channel != "LIDAR_TOP" or sensor.get("modality") != "lidar":
                continue
            if self.sensor == "lidar":
                self._lidar_sweeps_by_scene[scene_token].append(data)
            if data.get("is_key_frame", False):
                if self.sensor == "camera":
                    self._retain_keyframe(data)
                if sample_token in self._lidar_by_sample:
                    raise DatasetError(f"Multiple LIDAR_TOP keyframes reference sample {sample_token!r}.")
                self._lidar_by_sample[sample_token] = data

        # Keep only predecessor timestamps, not nonkeyframe records, for scan
        # intervals in ordinary camera-mode cross-modality point reads.
        previous_tokens = {data.get("prev") for data in self._lidar_by_sample.values() if data.get("prev")}
        previous_times = {
            data.get("token"): _timestamp(data.get("timestamp")) for data in data_records
            if data.get("token") in previous_tokens
        }
        self._lidar_sweep_start_us = {
            data["token"]: previous_times[data["prev"]] for data in self._lidar_by_sample.values()
            if data.get("prev") in previous_times and previous_times[data["prev"]] is not None
        }
        self._sweep_indexes_ready = self.sensor == "lidar"

        for channels in self._cameras_by_scene.values():
            for observations in channels.values():
                observations.sort(key=lambda record: _timestamp(record.get("timestamp")) or 0)
        for sweeps in self._lidar_sweeps_by_scene.values():
            sweeps.sort(key=lambda record: _timestamp(record.get("timestamp")) or 0)
        self._trajectories: dict[str, Any] = {}
        self._nominal_phases: dict[str, float] = {}
        self._image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

        self._annotations: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
        # Keep native category identity; the loader supplies display groups.
        native_class_names: set[str] = set()
        for category in categories.values():
            name = category.get("name")
            if not isinstance(name, str) or not name:
                raise DatasetError("Invalid category.json record: expected a nonempty name.")
            native_class_names.add(name)
        for annotation in self._read_table("sample_annotation", optional=True):
            sample_token = annotation.get("sample_token")
            self._record(samples, sample_token, "sample")
            name = annotation.get("category_name")
            if not name:
                instance = self._record(instances, annotation.get("instance_token"), "instance")
                category = self._record(categories, instance.get("category_token"), "category")
                name = category.get("name")
            if not isinstance(name, str) or not name:
                raise DatasetError(f"Invalid category for annotation {annotation.get('token')!r}.")
            native_class_names.add(name)
            self._annotations[sample_token].append((annotation, name))
        self._native_class_names = tuple(sorted(native_class_names))

        samples_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for token, sample in samples.items():
            scene_token = sample.get("scene_token")
            self._record(scenes, scene_token, "scene")
            samples_by_scene[scene_token].append(sample)

        self._frames: dict[str, FrameInfo] = {}
        self._frames_by_scene: dict[str, tuple[FrameInfo, ...]] = {}
        self._lidar_frames: dict[str, FrameInfo] = {}
        self._camera_frames: dict[str, FrameInfo] = {}
        self._frame_index_errors: dict[str, dict[str, str]] = {"lidar": {}, "camera": {}}
        self._frame_index_error_scenes: dict[str, dict[str, str]] = {"lidar": {}, "camera": {}}
        scene_infos = []
        for scene_token, scene in scenes.items():
            scene_samples = samples_by_scene[scene_token]
            lidar_frames = self._scene_frame_infos(scene_samples, scene_token, "lidar")
            camera_frames = self._scene_frame_infos(scene_samples, scene_token, "camera")
            self._lidar_frames.update((frame.frame_id, frame) for frame in lidar_frames)
            self._camera_frames.update((frame.frame_id, frame) for frame in camera_frames)
            frames = lidar_frames if self.sensor == "lidar" else camera_frames
            self._frames_by_scene[scene_token] = frames
            self._frames.update((frame.frame_id, frame) for frame in frames)
            scene_infos.append(SceneInfo(
                scene_id=scene_token,
                display_name=scene.get("name", scene_token),
                description=scene.get("description", ""),
                frame_counts={
                    Modality.POINT_CLOUD: None if scene_token in self._frame_index_error_scenes["lidar"] else len(lidar_frames),
                    Modality.CAMERAS: None if scene_token in self._frame_index_error_scenes["camera"] else len(camera_frames),
                },
            ))
        self._scenes = tuple(sorted(scene_infos, key=lambda scene: (scene.display_name, scene.scene_id)))

    def _retain_keyframe(self, data: dict[str, Any]) -> None:
        token = data.get("token")
        if not isinstance(token, str) or not token:
            raise DatasetError("Invalid sample_data.json record: expected a nonempty token.")
        if token in self._sample_data:
            raise DatasetError(f"Duplicate token {token!r} in sample_data.json.")
        self._sample_data[token] = data

    def _scene_frame_infos(
        self, samples: list[dict[str, Any]], scene_token: str, modality: str
    ) -> tuple[FrameInfo, ...]:
        keyframes = self._lidar_by_sample if modality == "lidar" else self._cameras_by_sample
        relevant = [sample for sample in samples if sample["token"] in keyframes]
        for sample in relevant:
            if _timestamp(sample.get("timestamp")) is None:
                message = (
                    f"Invalid {modality} sample timestamps for scene {scene_token!r}: "
                    f"sample {sample['token']!r} requires a nonnegative integer timestamp."
                )
                if modality == self.sensor and self._validate_active_frames:
                    raise DatasetError(message)
                # Stable chronological indices cannot be assigned to this
                # modality's scene. Surface its error only on a cross read.
                self._frame_index_errors[modality].update((item["token"], message) for item in relevant)
                self._frame_index_error_scenes[modality][scene_token] = message
                return ()
        relevant.sort(key=lambda sample: (int(sample["timestamp"]), sample["token"]))
        return tuple(
            FrameInfo(sample["token"], scene_token, index, int(sample["timestamp"]),
                      frozenset(mode for mode, index in ((Modality.POINT_CLOUD, self._lidar_by_sample),
                                                         (Modality.CAMERAS, self._cameras_by_sample))
                                if sample["token"] in index))
            for index, sample in enumerate(relevant)
        )

    def _ensure_sweep_indexes(self) -> None:
        """Add motion/RGB indexes without duplicating large common tables.

        Camera enumeration remains camera enumeration after this upgrade.
        Existing poses, calibration, annotations, and decoded images are reused.
        """
        if self._sweep_indexes_ready:
            return
        sample_data = self._index_table("sample_data")
        for token, previous in self._sample_data.items():
            if sample_data.get(token) != previous:
                raise DatasetError(f"nuScenes sample_data keyframe {token!r} changed after it was indexed.")
        data_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        lidar_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        cameras_by_scene: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for data in sample_data.values():
            scene_token = self._sample_scene_tokens.get(data.get("sample_token"))
            if scene_token is None:
                continue
            data_by_scene[scene_token].append(data)
            try:
                calibration = self._record(self._calibrations, data.get("calibrated_sensor_token"), "calibrated_sensor")
                channel, modality = self._sensor_by_token[calibration.get("sensor_token")]
            except (DatasetError, KeyError, TypeError):
                if data.get("is_key_frame", False):
                    raise DatasetError(f"Invalid sensor calibration for nuScenes sample_data {data.get('token')!r}.") from None
                continue
            if channel == "LIDAR_TOP" and modality == "lidar":
                lidar_by_scene[scene_token].append(data)
            elif channel in _CAMERA_CHANNELS and modality == "camera":
                cameras_by_scene[scene_token][channel].append(data)
        for sweeps in lidar_by_scene.values():
            sweeps.sort(key=lambda record: _timestamp(record.get("timestamp")) or 0)
        for channels in cameras_by_scene.values():
            for observations in channels.values():
                observations.sort(key=lambda record: _timestamp(record.get("timestamp")) or 0)
        # Commit only after indexing succeeds; optional preparation is atomic.
        self._sample_data = sample_data
        self._data_by_scene = data_by_scene
        self._lidar_sweeps_by_scene = lidar_by_scene
        self._cameras_by_scene = cameras_by_scene
        self._sweep_indexes_ready = True

    def _read_table(self, name: str, *, optional: bool = False) -> list[dict[str, Any]]:
        path = self.table_root / f"{name}.json"
        if optional and not path.exists():
            return []
        try:
            with path.open(encoding="utf-8") as stream:
                records = json.load(stream)
        except (OSError, ValueError) as exc:
            raise DatasetError(f"Cannot read nuScenes table {path}: {exc}") from exc
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise DatasetError(f"Invalid nuScenes table {path}: expected a JSON array of records.")
        return records

    def _index_table(self, name: str, *, optional: bool = False) -> dict[str, dict[str, Any]]:
        result = {}
        for record in self._read_table(name, optional=optional):
            token = record.get("token")
            if not isinstance(token, str) or not token:
                raise DatasetError(f"Invalid {name}.json record: expected a nonempty token.")
            if token in result:
                raise DatasetError(f"Duplicate token {token!r} in {name}.json.")
            result[token] = record
        return result

    @staticmethod
    def _record(index: dict, token: Any, table: str) -> dict[str, Any]:
        try:
            return index[token]
        except (KeyError, TypeError):
            raise DatasetError(f"nuScenes metadata references missing {table} token {token!r}.") from None

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._native_class_names

    @property
    def supports_point_timing(self) -> bool:
        return any(_timestamp(data.get("timestamp")) is not None for data in self._lidar_by_sample.values())

    @property
    def supports_camera_rgb(self) -> bool:
        return self.supports_point_timing and any(self._cameras_by_scene.values())

    def list_scenes(self) -> tuple[SceneInfo, ...]:
        return self._scenes

    @property
    def available_splits(self) -> tuple[str, ...]:
        if self.version.endswith("mini"):
            return ("all", "train", "val", "mini_train", "mini_val")
        if self.version.endswith("test"):
            return ("all", "test")
        return ("all", "train", "val")

    def list_scenes_for_split(self, split: str = "all") -> tuple[SceneInfo, ...]:
        from .splits import scene_names_for_split

        names = scene_names_for_split(split, self.version)
        return self._scenes if names is None else tuple(scene for scene in self._scenes if scene.display_name in names)

    def list_frames(self, scene_token: str) -> tuple[FrameInfo, ...]:
        try:
            return self._frames_by_scene[scene_token]
        except KeyError:
            raise DatasetError(f"Unknown nuScenes scene {scene_token!r}.") from None

    def _point_file(self, filename: Any) -> Path:
        if not isinstance(filename, str) or not filename:
            raise DatasetError("LIDAR_TOP sample_data record has no valid filename.")
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise DatasetError(f"Invalid nuScenes lidar filename {filename!r}: expected a relative path inside the dataset.")
        candidates = tuple(root / filename for root in self._data_roots)
        for path in candidates:
            if path.is_file():
                return path
        checked = ", ".join(str(path) for path in candidates)
        raise DatasetError(
            f"nuScenes lidar file is missing. Checked: {checked}. "
            "Extract the sensor files (samples/LIDAR_TOP) under your dataset root."
        )

    def _annotation_set(self, frame_id, reference, timestamp, boxes):
        if self.version.endswith("test"):
            status = AnnotationStatus.WITHHELD
        elif not (self.table_root / "sample_annotation.json").is_file():
            status = AnnotationStatus.NOT_PROVIDED
        else:
            status = AnnotationStatus.AVAILABLE
        return AnnotationSet(
            reference, status, boxes if status == AnnotationStatus.AVAILABLE else (),
            AnnotationTiming(timestamp, AnnotationAlignment.REFERENCE_TIME),
        )

    def _pose_timestamp(self, data):
        pose = self._poses.get(data.get("ego_pose_token"))
        if pose is None:
            return None
        return _timestamp(pose.get("timestamp")) if "timestamp" in pose else _timestamp(data.get("timestamp"))

    def _frame_info(self, frame_id: str, modality: Modality) -> FrameInfo:
        if frame_id not in self._known_frame_tokens:
            raise UnknownIdentifierError(f"Unknown nuScenes frame {frame_id!r}.")
        sensor = "lidar" if modality == Modality.POINT_CLOUD else "camera"
        error = self._frame_index_errors[sensor].get(frame_id)
        if error is not None:
            raise DatasetValidationError(error)
        frames = self._lidar_frames if modality == Modality.POINT_CLOUD else self._camera_frames
        if frame_id not in frames:
            raise DataUnavailableError(f"nuScenes frame {frame_id!r} has no {modality.value} observation.")
        return frames[frame_id]

    def load_frame(self, frame_token: str) -> PointCloudFrame:
        frame_info = self._frame_info(frame_token, Modality.POINT_CLOUD)
        data = self._lidar_by_sample[frame_token]
        path = self._point_file(data.get("filename"))
        try:
            byte_count = path.stat().st_size
            if byte_count == 0 or byte_count % 20:
                raise DatasetError(
                    f"Malformed nuScenes lidar file {path}: expected nonempty records of "
                    "five float32 values (x, y, z, intensity, ring), 20 bytes per point."
                )
            scan = np.fromfile(path, dtype="<f4").reshape(-1, 5)
        except OSError as exc:
            raise DatasetError(f"Cannot read nuScenes lidar file {path}: {exc}") from exc
        if not np.isfinite(scan[:, :4]).all():
            raise DatasetError(f"Malformed nuScenes lidar file {path}: points or intensities contain NaN/Inf.")

        calibration = self._record(self._calibrations, data.get("calibrated_sensor_token"), "calibrated_sensor")
        pose = self._record(self._poses, data.get("ego_pose_token"), "ego_pose")
        sensor_to_ego = _quaternion_matrix(calibration.get("rotation"), "calibrated sensor rotation")
        ego_to_global = _quaternion_matrix(pose.get("rotation"), "ego pose rotation")
        sensor_translation = _vector(calibration.get("translation"), 3, "calibrated sensor translation")
        ego_translation = _vector(pose.get("translation"), 3, "ego pose translation")
        global_to_sensor = sensor_to_ego.T @ ego_to_global.T
        reference_sensor_to_world = np.eye(4)
        reference_sensor_to_world[:3, :3] = ego_to_global @ sensor_to_ego
        reference_sensor_to_world[:3, 3] = ego_to_global @ sensor_translation + ego_translation

        boxes = []
        for box in self._world_boxes(frame_token):
            center = sensor_to_ego.T @ (
                ego_to_global.T @ (box.center - ego_translation) - sensor_translation
            )
            boxes.append(Box3D(
                center=center,
                size=box.size,
                rotation=global_to_sensor @ box.rotation,
                class_id=box.class_id,
                annotation_id=box.annotation_id,
                instance_id=box.instance_id,
            ))
        attributes = {"intensity": np.ascontiguousarray(scan[:, 3])}
        if (np.isfinite(scan[:, 4]).all() and np.all(scan[:, 4] == np.floor(scan[:, 4]))
                and np.all((scan[:, 4] >= 0) & (scan[:, 4] < 32))):
            attributes["ring"] = np.ascontiguousarray(scan[:, 4], dtype=np.int32)
        reference = f"sensor:LIDAR_TOP:{data['token']}"
        end = _timestamp(data.get("timestamp"))
        start = self._lidar_sweep_start_us.get(data["token"])
        if start is not None and (end is None or start >= end):
            start = None
        return PointCloudFrame(
            info=frame_info, source_id=_POINT_SOURCE, reference_frame_id=reference,
            points=np.ascontiguousarray(scan[:, :3]), attributes=attributes,
            annotations=self._annotation_set(frame_token, reference, frame_info.timestamp_us, tuple(boxes)),
            reference_pose_timestamp_us=self._pose_timestamp(data),
            world_frame_id=f"world:scene:{frame_info.scene_id}", T_world_from_reference=reference_sensor_to_world,
            scan_start_us=start, scan_end_us=end,
        )

    def _world_boxes(self, frame_token: str) -> tuple[Box3D, ...]:
        boxes = []
        for annotation, class_name in self._annotations.get(frame_token, ()):
            token = annotation.get("token", "")
            size = _vector(annotation.get("size"), 3, f"box {token!r} size")
            if np.any(size <= 0):
                raise DatasetError(f"Invalid box {token!r} size: dimensions must be positive.")
            boxes.append(Box3D(
                center=_vector(annotation.get("translation"), 3, f"box {token!r} translation"),
                size=size[[1, 0, 2]],
                rotation=_quaternion_matrix(annotation.get("rotation"), f"box {token!r} rotation"),
                class_id=class_name,
                annotation_id=token,
                instance_id=annotation.get("instance_token") or None,
            ))
        return tuple(boxes)

    def _image_path(self, filename: Any, *, must_exist: bool = True) -> Path:
        if not isinstance(filename, str) or not filename:
            raise DatasetError("Invalid camera sample_data: expected an image filename.")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise DatasetError(f"Invalid nuScenes image filename {filename!r}: expected a relative path inside the dataset.")
        candidates = tuple(root / relative for root in self._data_roots)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is not None:
            return path
        if not must_exist:
            return candidates[0]
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise DatasetError(
            f"Camera image not found: {searched}. Extract the camera sensor files into your nuScenes root."
        )

    def _image(self, filename: Any) -> np.ndarray:
        """Decode a dataset-relative image through the shared bounded cache."""
        return self._load_camera_image(self._image_path(filename))

    @staticmethod
    def _camera_intrinsics(calibration: dict[str, Any], channel: str) -> np.ndarray:
        try:
            intrinsics = np.array(calibration.get("camera_intrinsic"), dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DatasetError(f"Invalid {channel} camera intrinsics: expected a finite 3-by-3 matrix.") from exc
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise DatasetError(f"Invalid {channel} camera intrinsics: expected a finite 3-by-3 matrix.")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise DatasetError(f"Invalid {channel} camera intrinsics: expected positive focal lengths.")
        if not np.allclose(intrinsics[2], [0, 0, 1], rtol=0, atol=1e-9) or not np.isclose(intrinsics[1, 0], 0, rtol=0, atol=1e-9):
            raise DatasetError(f"Invalid {channel} camera intrinsics: expected the standard pinhole matrix form.")
        intrinsics.setflags(write=False)
        return intrinsics

    @staticmethod
    def _camera_dimensions(data: dict[str, Any], *, image: np.ndarray | None = None) -> tuple[int, int]:
        dimensions = []
        for name, axis in (("width", 1), ("height", 0)):
            value = data.get(name)
            if value is None and image is not None:
                value = image.shape[axis]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DatasetError(f"Invalid camera image dimensions for {data.get('filename')!r}: expected positive integer width and height.")
            if image is not None and value != image.shape[axis]:
                raise DatasetError(f"Camera image dimensions disagree with metadata for {data.get('filename')!r}.")
            dimensions.append(value)
        return dimensions[0], dimensions[1]

    def load_camera_frame(self, frame_id: str) -> CameraFrameSet:
        info = self._frame_info(frame_id, Modality.CAMERAS)
        reference = f"world:scene:{info.scene_id}"
        cameras = []
        for channel in CAMERA_CHANNELS:
            data = self._cameras_by_sample[frame_id].get(channel)
            if data is None:
                continue
            calibration = self._record(self._calibrations, data.get("calibrated_sensor_token"), "calibrated_sensor")
            image = self._image(data.get("filename"))
            self._camera_dimensions(data, image=image)
            timestamp = data.get("timestamp")
            if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0):
                raise DatasetValidationError(f"Invalid {channel} image timestamp in frame {frame_id!r}: expected an integer timestamp or None.")
            intrinsics = None
            if calibration.get("camera_intrinsic") is not None:
                intrinsics = self._camera_intrinsics(calibration, channel)
            pose = None
            if data.get("ego_pose_token"):
                pose = self._record(self._poses, data["ego_pose_token"], "ego_pose")
            sensor_to_world = None
            if pose is not None and ("rotation" in pose or "translation" in pose) and ("rotation" in calibration or "translation" in calibration):
                sensor_to_world = _transform(pose, f"{channel} ego pose") @ _transform(calibration, f"{channel} calibration")
            projection = None if intrinsics is None or sensor_to_world is None else PinholeProjection(
                intrinsics=intrinsics, T_reference_from_camera=sensor_to_world,
            )
            cameras.append(CameraObservation(
                sensor_id=channel, image=image, timestamp_us=timestamp,
                timestamp_kind="recorded_image_timestamp", pose_timestamp_us=self._pose_timestamp(data),
                intrinsics=intrinsics, T_reference_from_camera=sensor_to_world, projection_model=projection,
            ))
        return CameraFrameSet(
            info=info, source_id=_CAMERA_SOURCE,
            reference_frame_id=reference, world_frame_id=reference, T_world_from_reference=np.eye(4),
            cameras=tuple(cameras),
            annotations=self._annotation_set(frame_id, reference, info.timestamp_us, self._world_boxes(frame_id)),
        )

    def _scene_trajectory(self, scene_token: str):
        from .motion import PoseTrajectory

        self._ensure_sweep_indexes()
        if scene_token in self._trajectories:
            return self._trajectories[scene_token]
        poses_by_timestamp = {}
        for data in self._data_by_scene[scene_token]:
            pose = self._poses.get(data.get("ego_pose_token"))
            if pose is None:
                continue
            timestamp = _timestamp(pose.get("timestamp", data.get("timestamp")))
            if timestamp is None:
                continue
            try:
                _transform(pose, "ego pose")
                translation = _vector(pose.get("translation"), 3, "ego pose translation")
                quaternion = _vector(pose.get("rotation"), 4, "ego pose rotation")
            except DatasetError:
                continue
            poses_by_timestamp[timestamp] = (translation, quaternion)
        if not poses_by_timestamp:
            raise DatasetError(
                "Point timing requires valid ego poses and timestamps from the nuScenes sample_data table."
            )
        timestamps = sorted(poses_by_timestamp)
        try:
            trajectory = PoseTrajectory(
                np.asarray(timestamps, dtype=np.int64),
                np.stack([poses_by_timestamp[t][0] for t in timestamps]),
                np.stack([poses_by_timestamp[t][1] for t in timestamps]),
            )
        except ValueError as exc:
            raise DatasetError(f"Cannot prepare point timing for scene {scene_token!r}: {exc}") from exc
        self._trajectories[scene_token] = trajectory
        return trajectory

    def _sweep_interval(self, data: dict[str, Any], scene_token: str) -> tuple[int, int]:
        self._ensure_sweep_indexes()
        end = _timestamp(data.get("timestamp"))
        if end is None:
            raise DatasetError("Point timing requires a valid LIDAR_TOP sample_data timestamp.")
        previous = self._sample_data.get(data.get("prev"))
        previous_time = _timestamp(previous.get("timestamp")) if previous is not None else None
        if previous_time is not None and previous_time < end:
            return previous_time, end
        # Plain JSON fixtures and partial extracts may omit the linked token.
        # Nearby indexed sweeps provide the same acquisition interval.
        sweep_times = sorted({
            timestamp for sweep in self._lidar_sweeps_by_scene[scene_token]
            if (timestamp := _timestamp(sweep.get("timestamp"))) is not None
        })
        earlier = [timestamp for timestamp in sweep_times if timestamp < end]
        if earlier:
            return earlier[-1], end
        following = self._sample_data.get(data.get("next"))
        following_time = _timestamp(following.get("timestamp")) if following is not None else None
        if following_time is None or following_time <= end:
            later = [timestamp for timestamp in sweep_times if timestamp > end]
            following_time = later[0] if later else None
        gap = following_time - end if following_time is not None else 50_000
        return end - gap, end

    def _scene_nominal_phase(self, frame: PointCloudFrame) -> float:
        """Use the first complete available sweep to anchor the scene's model."""

        from .motion import estimate_nominal_phase

        self._ensure_sweep_indexes()
        scene_token = frame.info.scene_id
        if scene_token not in self._nominal_phases:
            reference_points = frame.points
            for data in self._lidar_sweeps_by_scene[scene_token]:
                try:
                    path = self._point_file(data.get("filename"))
                    byte_count = path.stat().st_size
                    if byte_count % 20 or byte_count // 20 // 32 != 1085:
                        continue
                    scan = np.fromfile(path, dtype="<f4").reshape(-1, 5)
                except (DatasetError, OSError, ValueError):
                    continue
                if np.isfinite(scan[:, :3]).all():
                    reference_points = scan[:, :3]
                    break
            self._nominal_phases[scene_token] = estimate_nominal_phase(reference_points)
        return self._nominal_phases[scene_token]

    def _camera_observations(self, scene_token: str, point_timestamps_us: np.ndarray):
        from .projection import CameraObservation

        self._ensure_sweep_indexes()
        low = int(np.min(point_timestamps_us)) - _CAMERA_MAX_TIME_DELTA_US
        high = int(np.max(point_timestamps_us)) + _CAMERA_MAX_TIME_DELTA_US
        result = []
        for channel, records in self._cameras_by_scene[scene_token].items():
            for data in records:
                timestamp = _timestamp(data.get("timestamp"))
                if timestamp is None or timestamp < low or timestamp > high:
                    continue
                calibration = self._record(self._calibrations, data.get("calibrated_sensor_token"), "calibrated_sensor")
                pose = self._record(self._poses, data.get("ego_pose_token"), "ego_pose")
                intrinsics = self._camera_intrinsics(calibration, channel)
                width, height = self._camera_dimensions(data)
                path = self._image_path(data.get("filename"), must_exist=False)
                result.append(CameraObservation(
                    channel=channel,
                    timestamp_us=timestamp,
                    image_path=path,
                    intrinsics=intrinsics,
                    sensor_to_world=_transform(pose, f"{channel} ego pose") @ _transform(calibration, f"{channel} calibration"),
                    width=width,
                    height=height,
                ))
        if not result:
            raise DatasetError(
                "No camera images near this lidar scan. Extract the nuScenes camera samples and sweeps "
                "and check their sample_data timestamps."
            )
        return result

    def _load_camera_image(self, path: Path) -> np.ndarray:
        path = Path(path)
        if path in self._image_cache:
            self._image_cache.move_to_end(path)
            return self._image_cache[path]
        from PIL import Image

        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                # Immutable bytes keep cached images safe to share with the
                # normalized records without another multi-megabyte copy.
                pixels = np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(rgb.height, rgb.width, 3)
        except (OSError, ValueError) as exc:
            raise DatasetError(f"Cannot decode camera image {path}: {exc}") from exc
        pixels.setflags(write=False)
        self._image_cache[path] = pixels
        while len(self._image_cache) > 16:
            self._image_cache.popitem(last=False)
        return pixels

    def prepare_frame(self, frame: PointCloudFrame, *, camera_rgb: bool = False, point_timing: bool = False) -> PointCloudFrame:
        """Derive nominal firing times/raw returns, loading images only for RGB.

        Native nuScenes points are already ego-motion compensated. Undoing
        compensation and applying each return's ego pose produces the same
        world point; image alignment therefore uses the actual reference pose
        and each image's recorded pose without applying ego motion twice.
        """

        if not camera_rgb and not point_timing:
            return frame
        self._ensure_sweep_indexes()
        try:
            data = self._lidar_by_sample[frame.info.frame_id]
        except KeyError:
            raise DatasetError(f"Unknown nuScenes frame {frame.info.frame_id!r}.") from None
        from .motion import decompensate_points, nominal_point_timestamps

        calibration = self._record(self._calibrations, data.get("calibrated_sensor_token"), "calibrated_sensor")
        pose = self._record(self._poses, data.get("ego_pose_token"), "ego_pose")
        sensor_to_ego = _transform(calibration, "LIDAR_TOP calibration")
        sensor_to_world = _transform(pose, "LIDAR_TOP ego pose") @ sensor_to_ego
        start, end = self._sweep_interval(data, frame.info.scene_id)
        trajectory = self._scene_trajectory(frame.info.scene_id)
        if frame.timing is None or frame.acquisition is None:
            if frame.attributes.get("ring") is None:
                raise DatasetError("Nominal point timing requires integer HDL-32E ring indices in the range 0–31.")
            try:
                point_timestamps = nominal_point_timestamps(
                    frame.points, frame.attributes["ring"], start, end, trajectory, sensor_to_ego,
                    azimuth_phase_rad=self._scene_nominal_phase(frame),
                    reference_sensor_to_world=sensor_to_world,
                )
                raw_points = decompensate_points(
                    frame.points, point_timestamps, end, trajectory, sensor_to_ego,
                    reference_sensor_to_world=sensor_to_world,
                )
            except ValueError as exc:
                raise DatasetError(f"Cannot prepare nominal point timing: {exc}") from exc
            sources = (frame.source_id, "nuscenes:sample_data", "nuscenes:ego_pose", "nuscenes:calibrated_sensor")
            frame = replace(
                frame,
                timing=frame.timing or PointTiming(
                    point_timestamps, np.ones(len(frame.points), dtype=bool), TimingQuality.ESTIMATED,
                    ProcessingProvenance("nuscenes.hdl32e.nominal_timing.v1", sources, _TIMING_ASSUMPTIONS),
                ),
                acquisition=frame.acquisition or AcquisitionGeometry(
                    raw_points, np.ones(len(frame.points), dtype=bool), f"acquisition:LIDAR_TOP:{data['token']}",
                    ProcessingProvenance("nuscenes.hdl32e.inverse_ego_compensation.v1", sources, _TIMING_ASSUMPTIONS),
                ),
            )
        if camera_rgb and frame.rgb is None:
            from .projection import project_camera_rgb

            points_world = frame.points.astype(np.float64) @ sensor_to_world[:3, :3].T + sensor_to_world[:3, 3]
            try:
                cameras = self._camera_observations(frame.info.scene_id, frame.timing.timestamps_us)
                colors, valid = project_camera_rgb(
                    points_world, frame.timing.timestamps_us, cameras,
                    image_loader=self._load_camera_image, max_time_delta_us=_CAMERA_MAX_TIME_DELTA_US,
                )
            except (OSError, ValueError, DatasetError) as exc:
                raise DatasetError(
                    f"Cannot prepare camera RGB: {exc}. Extract nuScenes camera samples/sweeps under the dataset root."
                ) from exc
            sources = (frame.source_id, "nuscenes:sample_data", "nuscenes:ego_pose", "nuscenes:calibrated_sensor", _CAMERA_SOURCE)
            frame = replace(frame, rgb=PointColors(colors, valid, ProcessingProvenance(
                "nuscenes.camera_projection.nominal_hdl32e.v1", sources,
                _TIMING_ASSUMPTIONS + (
                    "Native undistorted camera images use their recorded exposure poses and pinhole calibration.",
                    "Depth competition is approximate; the validity mask identifies usable samples, not proven physical visibility.",
                ),
            )))
        return frame
