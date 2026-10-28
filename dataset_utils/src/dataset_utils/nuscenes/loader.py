"""Normalized nuScenes loader over the SDK-independent native reader.

The baseline preserves native LIDAR_TOP rows and axes (X right, Y forward,
Z up). Its calibrated rigid transform relates that anchored sensor frame to
the scene's native world coordinates. Camera sets use those world coordinates
and expose undistorted optical-camera pinhole models at recorded image poses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

from ..config import DatasetConfig
from ..dataset import DatasetLoader
from ..models import (
    CameraFrameSet, Capability, ClassInfo, DatasetMetadata, FrameInfo, Modality,
    PointCloudFrame, PointCloudPreparation, SceneInfo, SensorInfo,
)
from ..errors import (
    DataUnavailableError, DatasetConfigurationError, DatasetError, DatasetIOError,
    DatasetValidationError, UnknownIdentifierError,
)
from .reader import (
    CAMERA_CHANNELS, DETECTION_CLASSES, NuScenesReader, _CATEGORY_TO_DETECTION,
)


_T = TypeVar("_T")
_RELEASES = frozenset({"v1.0-mini", "v1.0-trainval", "v1.0-test"})
_CAMERA_ROLES = {
    "CAM_FRONT_LEFT": "front_left", "CAM_FRONT": "front", "CAM_FRONT_RIGHT": "front_right",
    "CAM_BACK_RIGHT": "rear_right", "CAM_BACK": "rear", "CAM_BACK_LEFT": "rear_left",
}
_CAMERA_LABELS = {
    "CAM_FRONT_LEFT": "Front left", "CAM_FRONT": "Front", "CAM_FRONT_RIGHT": "Front right",
    "CAM_BACK_RIGHT": "Back right", "CAM_BACK": "Back", "CAM_BACK_LEFT": "Back left",
}

class NuScenesLoader(DatasetLoader):
    """Shared point-cloud/camera contract for native nuScenes releases.

    Accepted storage formats are ``auto`` and ``native``. ``source_paths``
    optionally selects an explicit ``native`` root; no provider options exist.
    Attributes ``intensity`` and ``ring`` preserve the native intensity values
    and zero-based HDL-32E laser-ring identifiers respectively.
    """

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self._release = config.release or "v1.0-mini"
        if self._release not in _RELEASES:
            raise DatasetConfigurationError(f"Unsupported nuScenes release {self._release!r}; expected one of {sorted(_RELEASES)}.")
        if config.storage_format not in ("auto", "native"):
            raise DatasetConfigurationError(f"Unsupported nuScenes storage_format {config.storage_format!r}; expected 'auto' or 'native'.")
        unknown_sources = set(config.source_paths).difference({"native"})
        if unknown_sources:
            raise DatasetConfigurationError(f"Unknown nuScenes source names: {sorted(unknown_sources)}; only 'native' is supported.")
        if config.options:
            raise DatasetConfigurationError(f"Unknown nuScenes options: {sorted(config.options)}; this provider defines no options.")
        self._root = config.source_paths.get("native", config.root)
        if self._root is None:
            raise DatasetConfigurationError("nuScenes requires a dataset root or an explicit 'native' source path.")
        self._root = Path(self._root)
        self._readers: dict[Modality, NuScenesReader] = {}
        self._metadata: DatasetMetadata | None = None

    @staticmethod
    def _call(operation: Callable[..., _T], *args, configuration: bool = False, **kwargs) -> _T:
        try:
            return operation(*args, **kwargs)
        except DatasetError as exc:
            # Translate native reader failures at the dataset boundary; keep their
            # original cause for debugging.
            if type(exc) is not DatasetError:
                raise
            message = str(exc)
            if configuration and ("metadata not found" in message or "dataset root" in message):
                raise DatasetConfigurationError(message) from exc
            cause: BaseException | None = exc
            while cause is not None:
                if isinstance(cause, OSError):
                    raise DatasetIOError(message) from exc
                cause = cause.__cause__
            if "file is missing" in message or "image not found" in message:
                raise DatasetIOError(message) from exc
            if (
                "requires integer HDL-32E ring indices" in message
                or "requires a valid LIDAR_TOP sample_data timestamp" in message
                or "requires valid ego poses and timestamps" in message
                or "No camera images near this lidar scan" in message
            ):
                raise DataUnavailableError(message) from exc
            raise DatasetValidationError(message) from exc

    def _reader_for(self, modality: Modality) -> NuScenesReader:
        try:
            modality = Modality(modality)
        except (TypeError, ValueError) as exc:
            raise DatasetConfigurationError(f"Unknown nuScenes modality {modality!r}.") from exc
        if modality not in self._readers:
            reader = next(iter(self._readers.values()), None)
            if reader is None:
                reader = self._call(
                    NuScenesReader, root=self._root, version=self._release, sensor="camera", configuration=True,
                    _validate_active_frames=False,
                )
            self._readers[modality] = reader
        return self._readers[modality]

    def native_reader(self, modality: Modality = Modality.POINT_CLOUD) -> NuScenesReader:
        """Access the native reader for dataset-specific research."""
        self._ensure_open()
        return self._reader_for(modality)

    @property
    def metadata(self) -> DatasetMetadata:
        if self._metadata is None:
            reader = next(iter(self._readers.values()), None) or self._reader_for(Modality.CAMERAS)
            sensors = []
            native_sensors = set(reader._sensor_channels)
            if ("LIDAR_TOP", "lidar") in native_sensors or reader._lidar_by_sample:
                sensors.append(SensorInfo(sensor_id="LIDAR_TOP", modality=Modality.POINT_CLOUD, display_name="LIDAR_TOP"))
            has_cameras = bool(reader._cameras_by_sample) or any(
                channel in _CAMERA_ROLES and modality == "camera" for channel, modality in native_sensors
            )
            if has_cameras:
                # A partial extract is still the standard native camera rig;
                # observations declare which of its channels are available.
                for channel in CAMERA_CHANNELS:
                    sensors.append(SensorInfo(
                        sensor_id=channel, modality=Modality.CAMERAS,
                        display_name=_CAMERA_LABELS[channel],
                        view_role=_CAMERA_ROLES[channel],
                    ))
            capabilities = set()
            if any(sensor.modality == Modality.POINT_CLOUD for sensor in sensors):
                capabilities.add(Capability.POINT_CLOUD)
            if any(sensor.modality == Modality.CAMERAS for sensor in sensors):
                capabilities.add(Capability.CAMERAS)
            if (reader.table_root / "sample_annotation.json").is_file() and not self._release.endswith("test"):
                capabilities.add(Capability.BOXES_3D)
            if Capability.POINT_CLOUD in capabilities and reader.supports_point_timing:
                capabilities.update({Capability.POINT_TIMESTAMPS, Capability.ACQUISITION_GEOMETRY})
                if reader._has_camera_data:
                    capabilities.add(Capability.RGB_COLORING)
            # The known native detection ontology keeps the ten legacy display
            # groups even in small extracts, without merging native identities.
            native_classes = set(_CATEGORY_TO_DETECTION).union(reader._native_class_names)
            group_order = {name: index for index, name in enumerate(DETECTION_CLASSES)}
            classes = tuple(ClassInfo(
                class_id=name, native_name=name, display_name=name,
                display_group=_CATEGORY_TO_DETECTION.get(name, name),
            ) for name in sorted(native_classes, key=lambda name: (
                group_order.get(_CATEGORY_TO_DETECTION.get(name, name), len(group_order)), name,
            )))
            self._metadata = DatasetMetadata(
                dataset_id=f"nuscenes:{self._release}", display_name="nuScenes",
                release=self._release, capabilities=frozenset(capabilities),
                classes=classes, sensors=tuple(sensors), splits=reader.available_splits,
                clock_domain="Unix microseconds",
                feature_coverage={capability: "varies" for capability in (
                    Capability.RGB_COLORING, Capability.POINT_TIMESTAMPS, Capability.ACQUISITION_GEOMETRY,
                ) if capability in capabilities},
            )
        return self._metadata

    def _catalog_reader(self) -> NuScenesReader:
        return next(iter(self._readers.values()), None) or self._reader_for(Modality.CAMERAS)

    def _list_scenes(self, *, split: str = "all") -> tuple[SceneInfo, ...]:
        try:
            return self._catalog_reader().list_scenes_for_split(split)
        except DatasetError as exc:
            raise DatasetConfigurationError(str(exc)) from exc

    def _list_frames(self, scene_id: str, *, modality: Modality) -> tuple[FrameInfo, ...]:
        reader = self._reader_for(modality)
        modality = Modality(modality)
        self._require(Capability.POINT_CLOUD if modality == Modality.POINT_CLOUD else Capability.CAMERAS)
        if scene_id not in reader._frames_by_scene:
            raise UnknownIdentifierError(f"Unknown nuScenes scene {scene_id!r}.")
        sensor = "lidar" if modality == Modality.POINT_CLOUD else "camera"
        error = reader._frame_index_error_scenes[sensor].get(scene_id)
        if error is not None:
            raise DatasetValidationError(error)
        frames = reader._lidar_frames if modality == Modality.POINT_CLOUD else reader._camera_frames
        return tuple(sorted(
            (frame for frame in frames.values() if frame.scene_id == scene_id), key=lambda frame: frame.index,
        ))

    def _load_point_cloud(self, frame_id: str) -> PointCloudFrame:
        reader = self._reader_for(Modality.POINT_CLOUD)
        return self._call(reader.load_frame, frame_id)

    def _load_cameras(self, frame_id: str) -> CameraFrameSet:
        reader = self._reader_for(Modality.CAMERAS)
        return self._call(reader.load_camera_frame, frame_id)

    def _prepare_point_cloud(self, frame: PointCloudFrame, *, request: PointCloudPreparation) -> PointCloudFrame:
        return self._call(
            self._reader_for(Modality.POINT_CLOUD).prepare_frame, frame,
            camera_rgb=request.rgb, point_timing=request.timestamps or request.acquisition_geometry,
        )

    def _close(self) -> None:
        for reader in set(self._readers.values()):
            reader._image_cache.clear()
            reader._trajectories.clear()
        self._readers.clear()
