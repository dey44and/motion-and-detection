"""Native nuScenes JSON/image adapter, independent of the nuScenes SDK.

Boxes use their global annotation coordinates. Each optical camera pose uses
its own sample_data ego_pose and calibrated_sensor, preserving exposure-time
ego motion rather than substituting the sample's LiDAR pose.
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import DatasetAdapter, DatasetError
from .models import BoundingBox3D, CameraFrame, CameraInfo, FrameData, FrameInfo, SceneInfo

DETECTION_CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)

CAMERA_LAYOUT = (
    CameraInfo("CAM_FRONT_LEFT", "Front left", 0, 0),
    CameraInfo("CAM_FRONT", "Front", 0, 1),
    CameraInfo("CAM_FRONT_RIGHT", "Front right", 0, 2),
    CameraInfo("CAM_BACK_LEFT", "Back left", 1, 0),
    CameraInfo("CAM_BACK", "Back", 1, 1),
    CameraInfo("CAM_BACK_RIGHT", "Back right", 1, 2),
)
_CAMERAS = {info.channel: info for info in CAMERA_LAYOUT}
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


def _vector(value: Any, length: int, context: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"Invalid {context}: expected {length} finite numbers.") from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise DatasetError(f"Invalid {context}: expected {length} finite numbers.")
    return result


def _quaternion_matrix(value: Any, context: str) -> np.ndarray:
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
    result = np.eye(4)
    result[:3, :3] = _quaternion_matrix(record.get("rotation"), f"{context} rotation")
    result[:3, 3] = _vector(record.get("translation"), 3, f"{context} translation")
    return result


def _timestamp(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetError(f"Invalid {context}: expected a nonnegative integer timestamp.")
    return value


class NuScenesAdapter(DatasetAdapter):
    """Keyframe camera images with a bounded lazy decoded-image cache."""

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
        sensors = self._index_table("sensor")
        self._calibrations = self._index_table("calibrated_sensor")
        self._poses = self._index_table("ego_pose")
        categories = self._index_table("category", optional=True)
        instances = self._index_table("instance", optional=True)
        self._cameras_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

        # Native JSON has no SDK-enriched sample.data/sample.anns fields.
        for data in self._read_table("sample_data"):
            if not data.get("is_key_frame", False):
                continue
            calibration = self._record(self._calibrations, data.get("calibrated_sensor_token"), "calibrated_sensor")
            sensor = self._record(sensors, calibration.get("sensor_token"), "sensor")
            channel = sensor.get("channel")
            if sensor.get("modality") != "camera" or channel not in _CAMERAS:
                continue
            sample_token = data.get("sample_token")
            self._record(samples, sample_token, "sample")
            cameras = self._cameras_by_sample[sample_token]
            if channel in cameras:
                raise DatasetError(f"Multiple {channel} camera keyframes reference sample {sample_token!r}.")
            cameras[channel] = data

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
            self._annotations[sample_token].append((annotation, class_name))
            class_names.add(class_name)
        self._class_names = DETECTION_CLASSES + tuple(sorted(class_names.difference(DETECTION_CLASSES)))

        samples_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for token, sample in samples.items():
            scene_token = sample.get("scene_token")
            self._record(scenes, scene_token, "scene")
            if token in self._cameras_by_sample:
                _timestamp(sample.get("timestamp"), f"sample {token!r}")
                samples_by_scene[scene_token].append(sample)
        self._frames: dict[str, FrameInfo] = {}
        self._frames_by_scene: dict[str, tuple[FrameInfo, ...]] = {}
        scene_infos = []
        for scene_token, scene in scenes.items():
            ordered = sorted(samples_by_scene[scene_token], key=lambda sample: (sample["timestamp"], sample["token"]))
            frames = tuple(
                FrameInfo(sample["token"], scene_token, index, sample["timestamp"])
                for index, sample in enumerate(ordered)
            )
            self._frames_by_scene[scene_token] = frames
            self._frames.update((frame.token, frame) for frame in frames)
            try:
                scene_infos.append(SceneInfo(
                    scene_token, scene.get("name", scene_token), scene.get("description", ""), len(frames)
                ))
            except (TypeError, ValueError) as exc:
                raise DatasetError(f"Invalid nuScenes scene {scene_token!r}: {exc}") from exc
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
                raise DatasetError(f"Duplicate {name} token {token!r}.")
            result[token] = record
        return result

    @staticmethod
    def _record(index: dict[str, dict[str, Any]], token: Any, table: str) -> dict[str, Any]:
        if not isinstance(token, str) or token not in index:
            raise DatasetError(f"Missing nuScenes {table} record for token {token!r}.")
        return index[token]

    def _image(self, filename: Any) -> np.ndarray:
        if not isinstance(filename, str) or not filename:
            raise DatasetError("Invalid camera sample_data: expected an image filename.")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise DatasetError(f"Invalid nuScenes image filename {filename!r}: expected a relative path inside the dataset.")
        path = next((root / relative for root in self._data_roots if (root / relative).is_file()), None)
        if path is None:
            searched = ", ".join(str(root / relative) for root in self._data_roots)
            raise DatasetError(f"Camera image not found: {searched}. Extract the camera sensor files into your nuScenes root.")
        if path in self._image_cache:
            self._image_cache.move_to_end(path)
            return self._image_cache[path]
        try:
            with Image.open(path) as image:
                array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        except (OSError, ValueError) as exc:
            raise DatasetError(f"Cannot decode camera image {path}: {exc}") from exc
        array.setflags(write=False)
        self._image_cache[path] = array
        if len(self._image_cache) > 16:
            self._image_cache.popitem(last=False)
        return array

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._class_names

    @property
    def camera_layout(self) -> tuple[CameraInfo, ...]:
        return CAMERA_LAYOUT

    def list_scenes(self) -> tuple[SceneInfo, ...]:
        return self._scenes

    def list_frames(self, scene_token: str) -> tuple[FrameInfo, ...]:
        try:
            return self._frames_by_scene[scene_token]
        except KeyError:
            raise DatasetError(f"Unknown nuScenes scene {scene_token!r}.") from None

    def load_frame(self, frame_token: str) -> FrameData:
        try:
            info = self._frames[frame_token]
        except KeyError:
            raise DatasetError(f"Unknown nuScenes frame {frame_token!r}.") from None
        cameras = []
        for camera_info in CAMERA_LAYOUT:
            data = self._cameras_by_sample[frame_token].get(camera_info.channel)
            if data is None:
                continue
            calibration = self._record(self._calibrations, data.get("calibrated_sensor_token"), "calibrated_sensor")
            pose = self._record(self._poses, data.get("ego_pose_token"), "ego_pose")
            image = self._image(data.get("filename"))
            width, height = data.get("width"), data.get("height")
            if width is not None and width != image.shape[1] or height is not None and height != image.shape[0]:
                raise DatasetError(f"Camera image dimensions disagree with metadata for {data.get('filename')!r}.")
            try:
                cameras.append(CameraFrame(
                    camera_info, image, calibration.get("camera_intrinsic"),
                    _transform(pose, "camera ego pose") @ _transform(calibration, "camera calibration"),
                    _timestamp(data.get("timestamp"), "camera exposure"),
                ))
            except (TypeError, ValueError) as exc:
                raise DatasetError(f"Invalid {camera_info.channel} camera metadata: {exc}") from exc
        boxes = []
        for annotation, class_name in self._annotations.get(frame_token, ()):
            size = _vector(annotation.get("size"), 3, "annotation size")
            try:
                boxes.append(BoundingBox3D(
                    _vector(annotation.get("translation"), 3, "annotation translation"),
                    size[[1, 0, 2]], _quaternion_matrix(annotation.get("rotation"), "annotation rotation"),
                    class_name, annotation.get("token", ""),
                ))
            except (TypeError, ValueError) as exc:
                raise DatasetError(f"Invalid annotation {annotation.get('token')!r}: {exc}") from exc
        return FrameData(info, tuple(cameras), tuple(boxes))
