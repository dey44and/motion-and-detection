"""Abstract loader with lifecycle, payload validation, and atomic preparation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
import hashlib

import numpy as np

from .config import DatasetConfig
from .models import (
    AnnotationStatus, CameraFrameSet, Capability, DatasetMetadata, FrameInfo,
    Modality, PointCloudFrame, PointCloudPreparation, SceneInfo,
)
from ..errors import (
    DatasetClosedError, DatasetConfigurationError, DatasetError, DatasetIOError,
    DatasetValidationError, UnsupportedCapabilityError,
)


def _digest_array(digest, value):
    if value is None:
        digest.update(b"none")
        return
    array = np.asarray(value)
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes(order="C"))


def _digest_annotations(digest, annotations):
    digest.update(repr((annotations.reference_frame_id, annotations.status, annotations.timing)).encode())
    for box in annotations.boxes:
        digest.update(repr((box.annotation_id, box.class_id, box.instance_id, box.timing_override)).encode())
        for value in (box.center, box.size, box.rotation):
            _digest_array(digest, value)


def _baseline_hash(frame):
    digest = hashlib.sha256()
    digest.update(repr((
        frame.info.frame_id, frame.info.scene_id, frame.info.index, frame.info.timestamp_us,
        sorted(mode.value for mode in frame.info.modalities),
        frame.source_id, frame.reference_frame_id, frame.reference_pose_timestamp_us,
        frame.world_frame_id, frame.scan_start_us, frame.scan_end_us,
    )).encode())
    _digest_array(digest, frame.points)
    _digest_array(digest, frame.T_world_from_reference)
    for key in sorted(frame.attributes):
        digest.update(key.encode())
        _digest_array(digest, frame.attributes[key])
    _digest_annotations(digest, frame.annotations)
    return digest.digest()


def _derived_hash(value):
    digest = hashlib.sha256()
    if value is None:
        return b"none"
    digest.update(repr(value.provenance).encode())
    for key in ("values", "timestamps_us", "points", "valid_mask"):
        if hasattr(value, key):
            _digest_array(digest, getattr(value, key))
    digest.update(repr((getattr(value, "quality", None), getattr(value, "reference_frame_id", None))).encode())
    return digest.digest()


def _whole_hash(frame):
    digest = hashlib.sha256(_baseline_hash(frame))
    for key in ("rgb", "timing", "acquisition"):
        digest.update(_derived_hash(getattr(frame, key)))
    return digest.digest()


_REQUESTS = (
    ("rgb", Capability.RGB_COLORING, "rgb"),
    ("timestamps", Capability.POINT_TIMESTAMPS, "timing"),
    ("acquisition_geometry", Capability.ACQUISITION_GEOMETRY, "acquisition"),
)


def _guard_operation(name, function):
    @wraps(function)
    def guarded(self, *args, **kwargs):
        self._ensure_open()
        metadata = self.metadata
        if not isinstance(metadata, DatasetMetadata) or metadata.contract_version != 1:
            raise DatasetConfigurationError("Loader must expose version-1 DatasetMetadata.")
        mode = None
        if name == "list_scenes":
            split = kwargs.get("split", "all")
            if split not in metadata.splits:
                raise DatasetConfigurationError(f"Unsupported split {split!r} for {metadata.dataset_id}.")
        elif name == "list_frames":
            try:
                mode = Modality(kwargs["modality"])
            except (KeyError, TypeError, ValueError) as error:
                raise DatasetConfigurationError("list_frames requires a supported modality.") from error
            kwargs["modality"] = mode
            self._require(Capability(mode.value))
        elif name == "load_point_cloud":
            self._require(Capability.POINT_CLOUD)
        elif name == "load_cameras":
            self._require(Capability.CAMERAS)
        elif name == "prepare_point_cloud":
            original = args[0] if len(args) == 1 else kwargs.get("frame") if not args else None
            if not isinstance(original, PointCloudFrame):
                raise DatasetValidationError("Preparation requires one PointCloudFrame issued by this loader.")
            request = kwargs.get("request")
            if not isinstance(request, PointCloudPreparation):
                raise DatasetValidationError("Preparation requires a PointCloudPreparation request.")
            self._validate_owned(original)
            for flag, capability, _field in _REQUESTS:
                if getattr(request, flag):
                    self._require(capability)
            if all(not getattr(request, flag) or getattr(original, field) is not None
                   for flag, _cap, field in _REQUESTS):
                return original

        try:
            result = function(self, *args, **kwargs)
        except DatasetError:
            raise
        except OSError as error:
            raise DatasetIOError(f"{metadata.dataset_id}: {name}: {error}") from error
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise DatasetValidationError(f"{metadata.dataset_id}: {name}: {error}") from error

        if name == "list_scenes":
            if not isinstance(result, tuple) or any(not isinstance(item, SceneInfo) for item in result):
                raise DatasetValidationError("list_scenes must return a tuple of SceneInfo.")
            if len({item.scene_id for item in result}) != len(result):
                raise DatasetValidationError("Scene identifiers must be unique.")
        elif name == "list_frames":
            if not isinstance(result, tuple) or any(not isinstance(item, FrameInfo) for item in result):
                raise DatasetValidationError("list_frames must return a tuple of FrameInfo.")
            scene_id = args[0] if args else kwargs.get("scene_id")
            if (len({item.frame_id for item in result}) != len(result)
                    or any(item.index != index or item.scene_id != scene_id or mode not in item.modalities
                           for index, item in enumerate(result))):
                raise DatasetValidationError("Frame list has invalid IDs, indices, scenes, or modalities.")
        elif name == "load_point_cloud":
            if not isinstance(result, PointCloudFrame):
                raise DatasetValidationError("load_point_cloud must return a PointCloudFrame.")
            frame_id = args[0] if args else kwargs.get("frame_id")
            if result.info.frame_id != frame_id:
                raise DatasetValidationError("Returned point cloud has a different frame identity.")
            self._validate_annotations(result.annotations)
            self._validate_point_capabilities(result)
            self._own_frame(result)
        elif name == "load_cameras":
            if not isinstance(result, CameraFrameSet):
                raise DatasetValidationError("load_cameras must return a CameraFrameSet.")
            frame_id = args[0] if args else kwargs.get("frame_id")
            if result.info.frame_id != frame_id:
                raise DatasetValidationError("Returned cameras have a different frame identity.")
            self._validate_annotations(result.annotations)
            camera_ids = [sensor.sensor_id for sensor in metadata.sensors if sensor.modality == Modality.CAMERAS]
            present = [camera.sensor_id for camera in result.cameras]
            if any(sensor_id not in camera_ids for sensor_id in present):
                raise DatasetValidationError("A returned camera is absent from dataset sensor metadata.")
            if present != [sensor_id for sensor_id in camera_ids if sensor_id in present]:
                raise DatasetValidationError("Camera observations must follow dataset sensor ordering.")
            for camera in result.cameras:
                if camera.annotations is not None:
                    self._validate_annotations(camera.annotations)
        elif name == "prepare_point_cloud":
            request = kwargs["request"]
            if not isinstance(result, PointCloudFrame) or _baseline_hash(result) != original._baseline:
                raise DatasetValidationError("Preparation changed baseline point rows, source, reference, or annotations.")
            for flag, _cap, field in _REQUESTS:
                if getattr(request, flag) and getattr(result, field) is None:
                    raise DatasetValidationError(f"Preparation did not return requested {field}.")
                if not getattr(request, flag) and getattr(original, field) is not None:
                    if _derived_hash(getattr(result, field)) != _derived_hash(getattr(original, field)):
                        raise DatasetValidationError(f"Preparation changed unrequested {field}.")
            self._validate_owned(original)
            self._validate_point_capabilities(result)
            self._own_frame(result)
        return result
    return guarded


class DatasetLoader(ABC):
    """Implement normalized observations; common guards enforce the contract."""

    def __init__(self, config: DatasetConfig):
        if not isinstance(config, DatasetConfig):
            raise DatasetConfigurationError("DatasetLoader requires DatasetConfig.")
        self.config = config
        self._closed = False
        self._owner_token = object()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in ("list_scenes", "list_frames", "load_point_cloud", "load_cameras", "prepare_point_cloud"):
            method = cls.__dict__.get(name)
            if method is not None and not getattr(method, "__isabstractmethod__", False):
                setattr(cls, name, _guard_operation(name, method))

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Return the immutable installation metadata snapshot."""

    @abstractmethod
    def list_scenes(self, *, split: str = "all") -> tuple[SceneInfo, ...]:
        """List scene metadata without decoding sensor payloads."""

    @abstractmethod
    def list_frames(self, scene_id: str, *, modality: Modality) -> tuple[FrameInfo, ...]:
        """List the modality's logical observations with dense display indices."""

    def _ensure_open(self):
        if getattr(self, "_closed", True):
            raise DatasetClosedError("Dataset loader is closed or was not initialized.")

    def _require(self, capability):
        self._ensure_open()
        if capability not in self.metadata.capabilities:
            raise UnsupportedCapabilityError(f"{self.metadata.dataset_id} does not offer {capability.value}.")

    def _validate_annotations(self, annotations):
        classes = {item.class_id for item in self.metadata.classes}
        if any(box.class_id not in classes for box in annotations.boxes):
            raise DatasetValidationError("An annotation references an undeclared native class.")
        if annotations.status == AnnotationStatus.AVAILABLE:
            self._require(Capability.BOXES_3D)

    def _own_frame(self, frame):
        object.__setattr__(frame, "_baseline", _baseline_hash(frame))
        object.__setattr__(frame, "_owner", (self._owner_token, _whole_hash(frame)))
        return frame

    def _validate_point_capabilities(self, frame):
        for field, capability in (("rgb", Capability.RGB_COLORING),
                                  ("timing", Capability.POINT_TIMESTAMPS),
                                  ("acquisition", Capability.ACQUISITION_GEOMETRY)):
            if getattr(frame, field) is not None:
                self._require(capability)

    def _validate_owned(self, frame):
        owner = frame._owner
        if (not isinstance(owner, tuple) or len(owner) != 2 or owner[0] is not self._owner_token
                or frame._baseline != _baseline_hash(frame) or owner[1] != _whole_hash(frame)):
            raise DatasetValidationError("Frame is foreign, reordered, modified, or belongs to another loader/configuration.")

    def load_point_cloud(self, frame_id: str) -> PointCloudFrame:
        self._require(Capability.POINT_CLOUD)
        raise UnsupportedCapabilityError("Provider must implement load_point_cloud.")

    def load_cameras(self, frame_id: str) -> CameraFrameSet:
        self._require(Capability.CAMERAS)
        raise UnsupportedCapabilityError("Provider must implement load_cameras.")

    def prepare_point_cloud(self, frame: PointCloudFrame, *, request: PointCloudPreparation) -> PointCloudFrame:
        self._ensure_open()
        if not isinstance(frame, PointCloudFrame) or not isinstance(request, PointCloudPreparation):
            raise DatasetValidationError("Invalid point-cloud preparation input.")
        self._validate_owned(frame)
        for flag, capability, field in _REQUESTS:
            if getattr(request, flag):
                self._require(capability)
                if getattr(frame, field) is None:
                    raise UnsupportedCapabilityError(f"Provider must prepare requested {field}.")
        return frame

    def _close(self):
        """Optional provider resource cleanup hook."""

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        try:
            self._close()
        except DatasetError:
            raise
        except Exception as error:
            raise DatasetIOError(f"Dataset resource cleanup failed: {error}") from error
        finally:
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
