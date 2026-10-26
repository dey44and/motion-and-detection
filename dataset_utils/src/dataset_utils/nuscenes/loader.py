"""Normalized nuScenes loader over the SDK-independent native reader.

The baseline preserves native LIDAR_TOP rows and axes (X right, Y forward,
Z up). Its calibrated rigid transform relates that anchored sensor frame to
the scene's native world coordinates. Camera sets use those world coordinates
and expose undistorted optical-camera pinhole models at recorded image poses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from ..contracts.config import DatasetConfig
from ..contracts.loader import DatasetLoader
from ..contracts.models import (
    AcquisitionGeometry, AnnotationAlignment, AnnotationSet, AnnotationStatus,
    AnnotationTiming, Box3D, CameraFrameSet, CameraObservation, Capability,
    ClassInfo, DatasetMetadata, FrameInfo, Modality, PointCloudFrame,
    PointCloudPreparation, PointColors, PointTiming, ProcessingProvenance,
    SceneInfo, SensorInfo, TimingQuality,
)
from ..contracts.projection import PinholeProjection
from ..errors import (
    DataUnavailableError, DatasetConfigurationError, DatasetError, DatasetIOError,
    DatasetValidationError, UnknownIdentifierError,
)
from ..models import FrameInfo as LegacyFrameInfo
from ..models import PointCloudFrame as LegacyPointCloudFrame
from .reader import (
    CAMERA_CHANNELS, DETECTION_CLASSES, NuScenesReader, _CATEGORY_TO_DETECTION, _timestamp, _transform,
)


_T = TypeVar("_T")
_RELEASES = frozenset({"v1.0-mini", "v1.0-trainval", "v1.0-test"})
_POINT_SOURCE = "nuscenes:native:LIDAR_TOP:compensated-v1"
_CAMERA_SOURCE = "nuscenes:native:camera-keyframes-v1"
_CAMERA_ROLES = {
    "CAM_FRONT_LEFT": "front_left", "CAM_FRONT": "front", "CAM_FRONT_RIGHT": "front_right",
    "CAM_BACK_RIGHT": "rear_right", "CAM_BACK": "rear", "CAM_BACK_LEFT": "rear_left",
}
_CAMERA_LABELS = {
    "CAM_FRONT_LEFT": "Front left", "CAM_FRONT": "Front", "CAM_FRONT_RIGHT": "Front right",
    "CAM_BACK_RIGHT": "Back right", "CAM_BACK": "Back", "CAM_BACK_LEFT": "Back left",
}
_TIMING_ASSUMPTIONS = (
    "Point times are estimated from the nominal HDL-32E firing model, not recorded per return.",
    "One scene-stable azimuth phase and interpolated recorded ego poses undo existing ego compensation.",
    "A missing preceding sweep uses the next sweep interval or a nominal 50 ms interval.",
    "Returns in one firing column share its nominal time; object motion is not compensated.",
)


class NuScenesLoader(DatasetLoader):
    """Version-1 point-cloud/camera contract for native nuScenes releases.

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
            # Existing readers remain compatibility APIs; translate their
            # errors at the normalized boundary without discarding the cause.
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
        """Compatibility hook; normalized consumers use the contract methods."""
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

    def list_scenes(self, *, split: str = "all") -> tuple[SceneInfo, ...]:
        reader = self._catalog_reader()
        try:
            scenes = reader.list_scenes_for_split(split)
        except DatasetError as exc:
            raise DatasetConfigurationError(str(exc)) from exc
        lidar_counts = Counter(frame.scene_token for frame in reader._lidar_frames.values())
        camera_counts = Counter(frame.scene_token for frame in reader._camera_frames.values())
        return tuple(SceneInfo(
            scene_id=scene.token, display_name=scene.name, description=scene.description,
            frame_counts={
                Modality.POINT_CLOUD: (
                    None if scene.token in reader._frame_index_error_scenes["lidar"] else lidar_counts[scene.token]
                ),
                Modality.CAMERAS: (
                    None if scene.token in reader._frame_index_error_scenes["camera"] else camera_counts[scene.token]
                ),
            },
        ) for scene in scenes)

    def _info(self, native, reader: NuScenesReader) -> FrameInfo:
        modalities = set()
        if native.token in reader._lidar_by_sample:
            modalities.add(Modality.POINT_CLOUD)
        if native.token in reader._cameras_by_sample:
            modalities.add(Modality.CAMERAS)
        return FrameInfo(
            frame_id=native.token, scene_id=native.scene_token, index=native.index,
            timestamp_us=native.timestamp_us, modalities=frozenset(modalities),
        )

    def list_frames(self, scene_id: str, *, modality: Modality) -> tuple[FrameInfo, ...]:
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
        return tuple(self._info(frame, reader) for frame in sorted(
            (frame for frame in frames.values() if frame.scene_token == scene_id), key=lambda frame: frame.index,
        ))

    @staticmethod
    def _check_frame(reader: NuScenesReader, frame_id: str, modality: Modality) -> None:
        if frame_id not in reader._known_frame_tokens:
            raise UnknownIdentifierError(f"Unknown nuScenes frame {frame_id!r}.")
        index = reader._lidar_by_sample if modality == Modality.POINT_CLOUD else reader._cameras_by_sample
        if frame_id not in index:
            raise DataUnavailableError(f"nuScenes frame {frame_id!r} has no {modality.value} observation.")

    def _annotations(self, reader: NuScenesReader, frame_id: str, reference: str, timestamp: int | None, boxes) -> AnnotationSet:
        if self._release.endswith("test"):
            status = AnnotationStatus.WITHHELD
        elif not (reader.table_root / "sample_annotation.json").is_file():
            status = AnnotationStatus.NOT_PROVIDED
        else:
            status = AnnotationStatus.AVAILABLE
        normalized_boxes = []
        if status == AnnotationStatus.AVAILABLE:
            records = reader._annotations.get(frame_id, ())
            names = reader._native_annotation_names.get(frame_id, ())
            if not len(boxes) == len(records) == len(names):
                raise DatasetValidationError(f"Inconsistent nuScenes annotation index for frame {frame_id!r}.")
            for box, (record, _), native_name in zip(boxes, records, names):
                if not isinstance(box.token, str) or not box.token:
                    raise DatasetValidationError(f"nuScenes annotation in frame {frame_id!r} has no native token.")
                normalized_boxes.append(Box3D(
                    annotation_id=box.token, class_id=native_name,
                    center=box.center, size=box.size, rotation=box.rotation,
                    instance_id=record.get("instance_token") or None,
                ))
        return AnnotationSet(
            reference_frame_id=reference, status=status, boxes=tuple(normalized_boxes),
            timing=AnnotationTiming(timestamp_us=timestamp, alignment=AnnotationAlignment.REFERENCE_TIME),
        )

    @staticmethod
    def _pose_timestamp(reader: NuScenesReader, data: dict) -> int | None:
        pose = reader._poses.get(data.get("ego_pose_token"))
        if pose is None:
            return None
        return _timestamp(pose.get("timestamp")) if "timestamp" in pose else _timestamp(data.get("timestamp"))

    def load_point_cloud(self, frame_id: str) -> PointCloudFrame:
        reader = self._reader_for(Modality.POINT_CLOUD)
        self._require(Capability.POINT_CLOUD)
        self._check_frame(reader, frame_id, Modality.POINT_CLOUD)
        native = self._call(reader.load_frame, frame_id)
        data = reader._lidar_by_sample[frame_id]
        reference = f"sensor:LIDAR_TOP:{data['token']}"
        attributes = {}
        if native.intensity is not None:
            attributes["intensity"] = native.intensity
        if native.ring_indices is not None:
            attributes["ring"] = native.ring_indices
        end = _timestamp(data.get("timestamp"))
        start = reader._lidar_sweep_start_us.get(data["token"])
        if start is not None and (end is None or start >= end):
            start = None
        return PointCloudFrame(
            info=self._info(native.info, reader), source_id=_POINT_SOURCE, points=native.points,
            reference_frame_id=reference, reference_pose_timestamp_us=self._pose_timestamp(reader, data),
            world_frame_id=f"world:scene:{native.info.scene_token}", T_world_from_reference=native.reference_sensor_to_world,
            scan_start_us=start, scan_end_us=end, attributes=attributes,
            annotations=self._annotations(reader, frame_id, reference, native.info.timestamp_us, native.boxes),
        )

    def load_cameras(self, frame_id: str) -> CameraFrameSet:
        reader = self._reader_for(Modality.CAMERAS)
        self._require(Capability.CAMERAS)
        self._check_frame(reader, frame_id, Modality.CAMERAS)
        error = reader._frame_index_errors["camera"].get(frame_id)
        if error is not None:
            raise DatasetValidationError(error)
        info = reader._camera_frames[frame_id]
        reference = f"world:scene:{info.scene_token}"
        cameras = []
        for channel in CAMERA_CHANNELS:
            data = reader._cameras_by_sample[frame_id].get(channel)
            if data is None:
                continue
            calibration = self._call(reader._record, reader._calibrations, data.get("calibrated_sensor_token"), "calibrated_sensor")
            image = self._call(reader._image, data.get("filename"))
            self._call(reader._camera_dimensions, data, image=image)
            timestamp = data.get("timestamp")
            if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0):
                raise DatasetValidationError(f"Invalid {channel} image timestamp in frame {frame_id!r}: expected an integer timestamp or None.")
            intrinsics = None
            if calibration.get("camera_intrinsic") is not None:
                intrinsics = self._call(reader._camera_intrinsics, calibration, channel)
            pose = None
            if data.get("ego_pose_token"):
                pose = self._call(reader._record, reader._poses, data["ego_pose_token"], "ego_pose")
            sensor_to_world = None
            if pose is not None and ("rotation" in pose or "translation" in pose) and ("rotation" in calibration or "translation" in calibration):
                sensor_to_world = self._call(_transform, pose, f"{channel} ego pose") @ self._call(_transform, calibration, f"{channel} calibration")
            projection = None if intrinsics is None or sensor_to_world is None else PinholeProjection(
                intrinsics=intrinsics, T_reference_from_camera=sensor_to_world,
            )
            cameras.append(CameraObservation(
                sensor_id=channel, image=image, timestamp_us=timestamp,
                timestamp_kind="recorded_image_timestamp", pose_timestamp_us=self._pose_timestamp(reader, data),
                intrinsics=intrinsics, T_reference_from_camera=sensor_to_world, projection_model=projection,
            ))
        return CameraFrameSet(
            info=self._info(info, reader), source_id=_CAMERA_SOURCE,
            reference_frame_id=reference, world_frame_id=reference, T_world_from_reference=np.eye(4),
            cameras=tuple(cameras),
            annotations=self._annotations(reader, frame_id, reference, info.timestamp_us, self._call(reader._world_boxes, frame_id)),
        )

    def prepare_point_cloud(self, frame: PointCloudFrame, *, request: PointCloudPreparation) -> PointCloudFrame:
        need_rgb = request.rgb and frame.rgb is None
        need_times = request.timestamps and frame.timing is None
        need_acquisition = request.acquisition_geometry and frame.acquisition is None
        if not any((need_rgb, need_times, need_acquisition)):
            return frame
        if need_rgb:
            self._require(Capability.RGB_COLORING)
        if need_times:
            self._require(Capability.POINT_TIMESTAMPS)
        if need_acquisition:
            self._require(Capability.ACQUISITION_GEOMETRY)
        reader = self._reader_for(Modality.POINT_CLOUD)
        native = LegacyPointCloudFrame(
            info=LegacyFrameInfo(frame.info.frame_id, frame.info.scene_id, frame.info.index, frame.info.timestamp_us),
            points=frame.points, boxes=(), intensity=frame.attributes.get("intensity"),
            ring_indices=frame.attributes.get("ring"), lidar_timestamp_us=frame.scan_end_us,
            point_timestamps_us=None if frame.timing is None else frame.timing.timestamps_us,
            raw_points=None if frame.acquisition is None else frame.acquisition.points,
            rgb_colors=None if frame.rgb is None else frame.rgb.values,
            rgb_valid_mask=None if frame.rgb is None else frame.rgb.valid_mask,
            reference_sensor_to_world=frame.T_world_from_reference,
        )
        prepared = self._call(reader.prepare_frame, native, camera_rgb=need_rgb, point_timing=need_times or need_acquisition)
        sources = (frame.source_id, "nuscenes:sample_data", "nuscenes:ego_pose", "nuscenes:calibrated_sensor")
        timing = frame.timing
        if timing is None and prepared.point_timestamps_us is not None:
            timing = PointTiming(
                timestamps_us=prepared.point_timestamps_us, valid_mask=np.ones(len(frame.points), dtype=bool),
                quality=TimingQuality.ESTIMATED,
                provenance=ProcessingProvenance(method_id="nuscenes.hdl32e.nominal_timing.v1", source_ids=sources, assumptions=_TIMING_ASSUMPTIONS),
            )
        acquisition = frame.acquisition
        if acquisition is None and prepared.raw_points is not None:
            acquisition = AcquisitionGeometry(
                points=prepared.raw_points, valid_mask=np.ones(len(frame.points), dtype=bool),
                reference_frame_id=f"acquisition:LIDAR_TOP:{reader._lidar_by_sample[frame.info.frame_id]['token']}",
                provenance=ProcessingProvenance(method_id="nuscenes.hdl32e.inverse_ego_compensation.v1", source_ids=sources, assumptions=_TIMING_ASSUMPTIONS),
            )
        rgb = frame.rgb
        if rgb is None and prepared.rgb_colors is not None:
            rgb = PointColors(
                values=prepared.rgb_colors, valid_mask=prepared.rgb_valid_mask,
                provenance=ProcessingProvenance(
                    method_id="nuscenes.camera_projection.nominal_hdl32e.v1", source_ids=sources + (_CAMERA_SOURCE,),
                    assumptions=_TIMING_ASSUMPTIONS + (
                        "Native undistorted camera images use their recorded exposure poses and pinhole calibration.",
                        "Depth competition is approximate; the validity mask identifies usable samples, not proven physical visibility.",
                    ),
                ),
            )
        return replace(frame, rgb=rgb, timing=timing, acquisition=acquisition)

    def _close(self) -> None:
        for reader in set(self._readers.values()):
            reader._image_cache.clear()
            reader._trajectories.clear()
            reader._preparation_reader = None
        self._readers.clear()
