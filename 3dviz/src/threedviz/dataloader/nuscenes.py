"""Read the native nuScenes tables and lidar files without the nuScenes SDK.

Only LIDAR_TOP keyframes are exposed. Frame tokens are nuScenes sample tokens.
The ten detection categories are merged following the official devkit mapping;
other annotation categories keep their original names so no boxes are dropped.

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

from .base import DatasetAdapter, DatasetError
from .models import BoundingBox3D, FrameData, FrameInfo, SceneInfo

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

_CAMERA_CHANNELS = frozenset({
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
})
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


class NuScenesAdapter(DatasetAdapter):
    """An indexed, lazy point-cloud adapter for mini, trainval and test releases."""

    def __init__(self, *, root: str | Path | None, version: str = "v1.0-mini"):
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
        self._calibrations = self._index_table("calibrated_sensor")
        self._poses = self._index_table("ego_pose")
        sensors = self._index_table("sensor")
        categories = self._index_table("category", optional=True)
        instances = self._index_table("instance", optional=True)

        # Reverse-index plain JSON; sample.data and sample.anns only exist after
        # SDK enrichment and are intentionally not required by this adapter.
        self._lidar_by_sample: dict[str, dict[str, Any]] = {}
        self._sample_data = self._index_table("sample_data")
        self._data_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lidar_sweeps_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._cameras_by_scene: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for data in self._sample_data.values():
            sample_token = data.get("sample_token")
            sample = samples.get(sample_token) if isinstance(sample_token, str) else None
            if sample is not None:
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
                if channel == "LIDAR_TOP" and data.get("is_key_frame", False):
                    self._record(samples, sample_token, "sample")
                continue
            scene_token = sample.get("scene_token")
            if channel in _CAMERA_CHANNELS and sensor.get("modality") == "camera":
                self._cameras_by_scene[scene_token][channel].append(data)
            if channel != "LIDAR_TOP" or sensor.get("modality") != "lidar":
                continue
            self._lidar_sweeps_by_scene[scene_token].append(data)
            if data.get("is_key_frame", False):
                if sample_token in self._lidar_by_sample:
                    raise DatasetError(f"Multiple LIDAR_TOP keyframes reference sample {sample_token!r}.")
                self._lidar_by_sample[sample_token] = data

        for channels in self._cameras_by_scene.values():
            for observations in channels.values():
                observations.sort(key=lambda record: _timestamp(record.get("timestamp")) or 0)
        for sweeps in self._lidar_sweeps_by_scene.values():
            sweeps.sort(key=lambda record: _timestamp(record.get("timestamp")) or 0)
        self._trajectories: dict[str, Any] = {}
        self._nominal_phases: dict[str, float] = {}
        self._image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

        self._annotations: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
        class_names = set(DETECTION_CLASSES)
        for category in categories.values():
            name = category.get("name")
            if not isinstance(name, str) or not name:
                raise DatasetError("Invalid category.json record: expected a nonempty name.")
            class_names.add(_CATEGORY_TO_DETECTION.get(name, name))
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
            class_name = _CATEGORY_TO_DETECTION.get(name, name)
            class_names.add(class_name)
            self._annotations[sample_token].append((annotation, class_name))
        self._class_names = DETECTION_CLASSES + tuple(sorted(class_names.difference(DETECTION_CLASSES)))

        samples_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for token, sample in samples.items():
            scene_token = sample.get("scene_token")
            self._record(scenes, scene_token, "scene")
            if token in self._lidar_by_sample:
                samples_by_scene[scene_token].append(sample)

        self._frames: dict[str, FrameInfo] = {}
        self._frames_by_scene: dict[str, tuple[FrameInfo, ...]] = {}
        scene_infos = []
        for scene_token, scene in scenes.items():
            scene_samples = samples_by_scene[scene_token]
            try:
                scene_samples.sort(key=lambda sample: (int(sample["timestamp"]), sample["token"]))
                frames = tuple(
                    FrameInfo(sample["token"], scene_token, index, int(sample["timestamp"]))
                    for index, sample in enumerate(scene_samples)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetError(f"Invalid sample timestamps for scene {scene_token!r}.") from exc
            self._frames_by_scene[scene_token] = frames
            self._frames.update((frame.token, frame) for frame in frames)
            scene_infos.append(SceneInfo(
                token=scene_token,
                name=scene.get("name", scene_token),
                description=scene.get("description", ""),
                frame_count=len(frames),
            ))
        self._scenes = tuple(sorted(scene_infos, key=lambda scene: (scene.name, scene.token)))

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
        return self._class_names

    @property
    def supports_point_timing(self) -> bool:
        return any(_timestamp(data.get("timestamp")) is not None for data in self._lidar_by_sample.values())

    @property
    def supports_camera_rgb(self) -> bool:
        return self.supports_point_timing and any(self._cameras_by_scene.values())

    def list_scenes(self) -> tuple[SceneInfo, ...]:
        return self._scenes

    def list_frames(self, scene_token: str) -> tuple[FrameInfo, ...]:
        try:
            return self._frames_by_scene[scene_token]
        except KeyError:
            raise DatasetError(f"Unknown nuScenes scene {scene_token!r}.") from None

    def _point_file(self, filename: Any) -> Path:
        if not isinstance(filename, str) or not filename:
            raise DatasetError("LIDAR_TOP sample_data record has no valid filename.")
        candidates = tuple(root / filename for root in self._data_roots)
        for path in candidates:
            if path.is_file():
                return path
        checked = ", ".join(str(path) for path in candidates)
        raise DatasetError(
            f"nuScenes lidar file is missing. Checked: {checked}. "
            "Extract the sensor files (samples/LIDAR_TOP) under your dataset root."
        )

    def load_frame(self, frame_token: str) -> FrameData:
        try:
            frame_info = self._frames[frame_token]
        except KeyError:
            raise DatasetError(f"Unknown nuScenes frame {frame_token!r}.") from None
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

        boxes = []
        for annotation, class_name in self._annotations.get(frame_token, ()):
            token = annotation.get("token", "")
            global_center = _vector(annotation.get("translation"), 3, f"box {token!r} translation")
            wlh = _vector(annotation.get("size"), 3, f"box {token!r} size")
            if np.any(wlh <= 0):
                raise DatasetError(f"Invalid box {token!r} size: dimensions must be positive.")
            global_rotation = _quaternion_matrix(annotation.get("rotation"), f"box {token!r} rotation")
            center = sensor_to_ego.T @ (
                ego_to_global.T @ (global_center - ego_translation) - sensor_translation
            )
            boxes.append(BoundingBox3D(
                center=center,
                size=wlh[[1, 0, 2]],
                rotation=global_to_sensor @ global_rotation,
                class_name=class_name,
                token=token,
            ))
        return FrameData(
            info=frame_info,
            points=np.ascontiguousarray(scan[:, :3]),
            intensity=np.ascontiguousarray(scan[:, 3]),
            boxes=tuple(boxes),
            lidar_timestamp_us=(
                frame_info.timestamp_us if _timestamp(data.get("timestamp")) is None
                else _timestamp(data.get("timestamp"))
            ),
            ring_indices=(
                np.ascontiguousarray(scan[:, 4], dtype=np.int32)
                if np.isfinite(scan[:, 4]).all()
                and np.all(scan[:, 4] == np.floor(scan[:, 4]))
                and np.all((scan[:, 4] >= 0) & (scan[:, 4] < 32))
                else None
            ),
        )

    def _scene_trajectory(self, scene_token: str):
        from ..processing.motion import PoseTrajectory

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

    def _scene_nominal_phase(self, frame: FrameData) -> float:
        """Use the first complete available sweep to anchor the scene's model."""

        from ..processing.motion import estimate_nominal_phase

        scene_token = frame.info.scene_token
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
        from ..processing.projection import CameraObservation

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
                try:
                    intrinsics = np.asarray(calibration.get("camera_intrinsic"), dtype=np.float64)
                    width, height = int(data["width"]), int(data["height"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise DatasetError(f"Invalid {channel} camera calibration or image dimensions.") from exc
                if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or min(width, height) <= 0:
                    raise DatasetError(f"Invalid {channel} camera calibration or image dimensions.")
                filename = data.get("filename")
                if not isinstance(filename, str) or not filename:
                    raise DatasetError(f"{channel} camera record has no valid image filename.")
                candidates = tuple(root / filename for root in self._data_roots)
                path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
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
        if path in self._image_cache:
            self._image_cache.move_to_end(path)
            return self._image_cache[path]
        from PIL import Image

        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        self._image_cache[path] = pixels
        while len(self._image_cache) > 16:
            self._image_cache.popitem(last=False)
        return pixels

    def prepare_frame(self, frame: FrameData, *, camera_rgb: bool = False, point_timing: bool = False) -> FrameData:
        """Derive nominal firing times/raw returns, loading images only for RGB.

        Native nuScenes points are already ego-motion compensated. Undoing
        compensation and applying each return's ego pose produces the same
        world point; image alignment therefore uses the actual reference pose
        and each image's recorded pose without applying ego motion twice.
        """

        if not camera_rgb and not point_timing:
            return frame
        try:
            data = self._lidar_by_sample[frame.info.token]
        except KeyError:
            raise DatasetError(f"Unknown nuScenes frame {frame.info.token!r}.") from None
        from ..processing.motion import decompensate_points, nominal_point_timestamps

        calibration = self._record(self._calibrations, data.get("calibrated_sensor_token"), "calibrated_sensor")
        pose = self._record(self._poses, data.get("ego_pose_token"), "ego_pose")
        sensor_to_ego = _transform(calibration, "LIDAR_TOP calibration")
        sensor_to_world = _transform(pose, "LIDAR_TOP ego pose") @ sensor_to_ego
        start, end = self._sweep_interval(data, frame.info.scene_token)
        trajectory = self._scene_trajectory(frame.info.scene_token)
        if frame.point_timestamps_us is None or frame.raw_points is None:
            if frame.ring_indices is None:
                raise DatasetError("Nominal point timing requires integer HDL-32E ring indices in the range 0–31.")
            try:
                point_timestamps = nominal_point_timestamps(
                    frame.points, frame.ring_indices, start, end, trajectory, sensor_to_ego,
                    azimuth_phase_rad=self._scene_nominal_phase(frame),
                    reference_sensor_to_world=sensor_to_world,
                )
                raw_points = decompensate_points(
                    frame.points, point_timestamps, end, trajectory, sensor_to_ego,
                    reference_sensor_to_world=sensor_to_world,
                )
            except ValueError as exc:
                raise DatasetError(f"Cannot prepare nominal point timing: {exc}") from exc
            frame = replace(frame, point_timestamps_us=point_timestamps, raw_points=raw_points, lidar_timestamp_us=end)
        if camera_rgb and frame.rgb_colors is None:
            from ..processing.projection import project_camera_rgb

            points_world = frame.points.astype(np.float64) @ sensor_to_world[:3, :3].T + sensor_to_world[:3, 3]
            try:
                cameras = self._camera_observations(frame.info.scene_token, frame.point_timestamps_us)
                colors, valid = project_camera_rgb(
                    points_world, frame.point_timestamps_us, cameras,
                    image_loader=self._load_camera_image, max_time_delta_us=_CAMERA_MAX_TIME_DELTA_US,
                )
            except (OSError, ValueError) as exc:
                raise DatasetError(
                    f"Cannot prepare camera RGB: {exc}. Extract nuScenes camera samples/sweeps under the dataset root."
                ) from exc
            frame = replace(frame, rgb_colors=colors, rgb_valid_mask=valid)
        return frame
