"""Version-1 normalized observations, with explicit identity and geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from ..errors import DatasetValidationError


class Modality(str, Enum):
    POINT_CLOUD = "point_cloud"
    CAMERAS = "cameras"


class Capability(str, Enum):
    POINT_CLOUD = "point_cloud"
    CAMERAS = "cameras"
    BOXES_3D = "boxes_3d"
    RGB_COLORING = "rgb_coloring"
    POINT_TIMESTAMPS = "point_timestamps"
    ACQUISITION_GEOMETRY = "acquisition_geometry"


class AnnotationStatus(str, Enum):
    AVAILABLE = "available"
    NOT_PROVIDED = "not_provided"
    WITHHELD = "withheld"


class AnnotationAlignment(str, Enum):
    REFERENCE_TIME = "reference_time"
    CAMERA_SYNCHRONIZED = "camera_synchronized"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"
    MIXED = "mixed"


class TimingQuality(str, Enum):
    RECORDED = "recorded"
    ESTIMATED = "estimated"


def _text(value, name, *, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or not value:
        raise DatasetValidationError(f"{name} must be a nonempty string.")


def _integer(value, name, *, optional=False, nonnegative=False):
    if optional and value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise DatasetValidationError(f"{name} must be an integer.")
    if not -(2**63) <= value < 2**63 or (nonnegative and value < 0):
        raise DatasetValidationError(f"{name} is outside its permitted range.")


def _enum(kind, value):
    try:
        return kind(value)
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(f"Invalid {kind.__name__}: {value!r}.") from error


def _array(value, shape, name, *, dtype=None, finite=True, kind=None):
    try:
        source = np.asarray(value)
        if kind is not None and source.dtype.kind not in kind:
            raise DatasetValidationError(f"{name} has an invalid dtype {source.dtype}.")
        result = np.asarray(value, dtype=dtype)
        if result.shape != shape:
            raise DatasetValidationError(f"{name} must have shape {shape}, got {result.shape}.")
        if finite and not np.isfinite(result).all():
            raise DatasetValidationError(f"{name} must contain finite values.")
        # A read-only view can still alias a writable provider buffer. Reuse
        # only arrays backed by immutable bytes; all others become snapshots.
        owner = result
        while isinstance(owner, np.ndarray) and owner.base is not None:
            owner = owner.base
        if result.flags.writeable or not result.flags.c_contiguous or not isinstance(owner, bytes):
            result = np.frombuffer(result.tobytes(order="C"), dtype=result.dtype).reshape(result.shape)
        return result
    except (TypeError, ValueError, OverflowError) as error:
        if isinstance(error, DatasetValidationError):
            raise
        raise DatasetValidationError(f"Invalid {name}: {error}") from error


def _rotation(value, name):
    result = _array(value, (3, 3), name, dtype=np.float64)
    if (not np.allclose(result.T @ result, np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(result), 1, rtol=0, atol=1e-5)):
        raise DatasetValidationError(f"{name} must be a proper rotation.")
    return result


def _transform(value, name):
    result = _array(value, (4, 4), name, dtype=np.float64)
    _rotation(result[:3, :3], name)
    if not np.allclose(result[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise DatasetValidationError(f"{name} must be a rigid homogeneous transform.")
    return result


def _world(record):
    _text(record.reference_frame_id, "reference_frame_id")
    _integer(record.reference_pose_timestamp_us, "reference_pose_timestamp_us", optional=True)
    if (record.world_frame_id is None) != (record.T_world_from_reference is None):
        raise DatasetValidationError("world_frame_id and T_world_from_reference must be supplied together.")
    if record.world_frame_id is not None:
        _text(record.world_frame_id, "world_frame_id")
        object.__setattr__(record, "T_world_from_reference",
                           _transform(record.T_world_from_reference, "T_world_from_reference"))


@dataclass(frozen=True)
class ClassInfo:
    class_id: str
    native_name: str
    display_name: str
    display_group: str | None = None

    def __post_init__(self):
        for name in ("class_id", "native_name", "display_name"):
            _text(getattr(self, name), name)
        _text(self.display_group, "display_group", optional=True)


@dataclass(frozen=True)
class SensorInfo:
    sensor_id: str
    modality: Modality
    display_name: str
    view_role: str | None = None

    def __post_init__(self):
        _text(self.sensor_id, "sensor_id")
        _text(self.display_name, "display_name")
        _text(self.view_role, "view_role", optional=True)
        object.__setattr__(self, "modality", _enum(Modality, self.modality))


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    display_name: str
    capabilities: frozenset[Capability]
    classes: tuple[ClassInfo, ...] = ()
    sensors: tuple[SensorInfo, ...] = ()
    splits: tuple[str, ...] = ("all",)
    release: str | None = None
    clock_domain: str | None = None
    contract_version: int = 1
    feature_coverage: Mapping[Capability, str] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("dataset_id", "display_name"):
            _text(getattr(self, name), name)
        for name in ("release", "clock_domain"):
            _text(getattr(self, name), name, optional=True)
        _integer(self.contract_version, "contract_version", nonnegative=True)
        capabilities = frozenset(_enum(Capability, cap) for cap in self.capabilities)
        if not capabilities.intersection((Capability.POINT_CLOUD, Capability.CAMERAS)):
            raise DatasetValidationError("A dataset must offer point clouds or cameras.")
        derived = {Capability.RGB_COLORING, Capability.POINT_TIMESTAMPS, Capability.ACQUISITION_GEOMETRY}
        if capabilities.intersection(derived) and Capability.POINT_CLOUD not in capabilities:
            raise DatasetValidationError("Derived point capabilities require POINT_CLOUD.")
        for name, kind, key in (("classes", ClassInfo, "class_id"), ("sensors", SensorInfo, "sensor_id")):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, kind) for value in values):
                raise DatasetValidationError(f"{name} contains invalid records.")
            if len({getattr(value, key) for value in values}) != len(values):
                raise DatasetValidationError(f"{name} contains duplicate identifiers.")
            object.__setattr__(self, name, values)
        splits = tuple(self.splits)
        if "all" not in splits or len(set(splits)) != len(splits):
            raise DatasetValidationError("splits must be unique and include 'all'.")
        for split in splits:
            _text(split, "split")
        coverage = {_enum(Capability, key): value for key, value in self.feature_coverage.items()}
        if any(key not in capabilities or value not in {"full", "partial", "varies"} for key, value in coverage.items()):
            raise DatasetValidationError("feature_coverage must describe advertised capabilities.")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "feature_coverage", MappingProxyType(coverage))


@dataclass(frozen=True)
class SceneInfo:
    scene_id: str
    display_name: str
    description: str = ""
    frame_counts: Mapping[Modality, int | None] = field(default_factory=dict)

    def __post_init__(self):
        _text(self.scene_id, "scene_id")
        _text(self.display_name, "display_name")
        counts = {_enum(Modality, key): value for key, value in self.frame_counts.items()}
        for value in counts.values():
            _integer(value, "frame count", optional=True, nonnegative=True)
        object.__setattr__(self, "frame_counts", MappingProxyType(counts))


@dataclass(frozen=True)
class FrameInfo:
    frame_id: str
    scene_id: str
    index: int
    timestamp_us: int | None = None
    modalities: frozenset[Modality] = frozenset()

    def __post_init__(self):
        _text(self.frame_id, "frame_id")
        _text(self.scene_id, "scene_id")
        _integer(self.index, "index", nonnegative=True)
        _integer(self.timestamp_us, "timestamp_us", optional=True)
        modes = frozenset(_enum(Modality, mode) for mode in self.modalities)
        if not modes:
            raise DatasetValidationError("Frame modalities cannot be empty.")
        object.__setattr__(self, "modalities", modes)


@dataclass(frozen=True)
class AnnotationTiming:
    timestamp_us: int | None = None
    alignment: AnnotationAlignment = AnnotationAlignment.UNKNOWN

    def __post_init__(self):
        _integer(self.timestamp_us, "annotation timestamp", optional=True)
        alignment = _enum(AnnotationAlignment, self.alignment)
        if alignment == AnnotationAlignment.MIXED and self.timestamp_us is not None:
            raise DatasetValidationError("Mixed annotations have no single timestamp.")
        object.__setattr__(self, "alignment", alignment)


@dataclass(frozen=True, eq=False)
class Box3D:
    annotation_id: str
    class_id: str
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    instance_id: str | None = None
    timing_override: AnnotationTiming | None = None

    def __post_init__(self):
        _text(self.annotation_id, "annotation_id")
        _text(self.class_id, "class_id")
        _text(self.instance_id, "instance_id", optional=True)
        object.__setattr__(self, "center", _array(self.center, (3,), "box center", dtype=np.float64))
        size = _array(self.size, (3,), "box size", dtype=np.float64)
        if np.any(size <= 0):
            raise DatasetValidationError("Box LWH dimensions must be positive.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "rotation", _rotation(self.rotation, "box rotation"))
        if self.timing_override is not None and (
            not isinstance(self.timing_override, AnnotationTiming)
            or self.timing_override.alignment == AnnotationAlignment.MIXED
        ):
            raise DatasetValidationError("Box timing overrides must be non-mixed AnnotationTiming records.")


@dataclass(frozen=True)
class AnnotationSet:
    reference_frame_id: str
    status: AnnotationStatus = AnnotationStatus.NOT_PROVIDED
    boxes: tuple[Box3D, ...] = ()
    timing: AnnotationTiming = field(default_factory=AnnotationTiming)

    def __post_init__(self):
        _text(self.reference_frame_id, "annotation reference")
        status = _enum(AnnotationStatus, self.status)
        boxes = tuple(self.boxes)
        if any(not isinstance(box, Box3D) for box in boxes):
            raise DatasetValidationError("annotations must contain Box3D records.")
        if len({box.annotation_id for box in boxes}) != len(boxes):
            raise DatasetValidationError("Duplicate annotation IDs within one set.")
        if status != AnnotationStatus.AVAILABLE and boxes:
            raise DatasetValidationError("Unavailable/withheld annotation sets must be empty.")
        if not isinstance(self.timing, AnnotationTiming):
            raise DatasetValidationError("Annotation timing must be AnnotationTiming.")
        if self.timing.alignment == AnnotationAlignment.MIXED and any(box.timing_override is None for box in boxes):
            raise DatasetValidationError("Every box in a mixed set needs an explicit timing override.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "boxes", boxes)


@dataclass(frozen=True)
class ProcessingProvenance:
    method_id: str
    source_ids: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self):
        _text(self.method_id, "method_id")
        for name in ("source_ids", "assumptions"):
            values = tuple(getattr(self, name))
            for value in values:
                _text(value, name)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, eq=False)
class PointColors:
    values: np.ndarray
    valid_mask: np.ndarray
    provenance: ProcessingProvenance

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.ndim != 2 or values.shape[1] != 3:
            raise DatasetValidationError("RGB values must have shape (N,3).")
        values = _array(values, values.shape, "RGB values", kind="f")
        if np.any((values < 0) | (values > 1)):
            raise DatasetValidationError("RGB values must be in [0,1].")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", _array(self.valid_mask, (len(values),), "RGB mask", kind="b"))
        if not isinstance(self.provenance, ProcessingProvenance):
            raise DatasetValidationError("RGB provenance is required.")


@dataclass(frozen=True, eq=False)
class PointTiming:
    timestamps_us: np.ndarray
    valid_mask: np.ndarray
    quality: TimingQuality
    provenance: ProcessingProvenance

    def __post_init__(self):
        values = np.asarray(self.timestamps_us)
        if values.ndim != 1:
            raise DatasetValidationError("Point timestamps must have shape (N,).")
        values = _array(values, (len(values),), "point timestamps", dtype=np.int64, kind="iu")
        if np.asarray(self.timestamps_us).dtype.kind == "u" and np.any(np.asarray(self.timestamps_us) >= 2**63):
            raise DatasetValidationError("Point timestamps overflow int64.")
        object.__setattr__(self, "timestamps_us", values)
        object.__setattr__(self, "valid_mask", _array(self.valid_mask, values.shape, "timing mask", kind="b"))
        object.__setattr__(self, "quality", _enum(TimingQuality, self.quality))
        if not isinstance(self.provenance, ProcessingProvenance):
            raise DatasetValidationError("Timing provenance is required.")


@dataclass(frozen=True, eq=False)
class AcquisitionGeometry:
    points: np.ndarray
    valid_mask: np.ndarray
    reference_frame_id: str
    provenance: ProcessingProvenance

    def __post_init__(self):
        values = np.asarray(self.points)
        if values.ndim != 2 or values.shape[1] != 3:
            raise DatasetValidationError("Acquisition coordinates must have shape (N,3).")
        object.__setattr__(self, "points", _array(values, values.shape, "acquisition points", kind="f"))
        object.__setattr__(self, "valid_mask", _array(self.valid_mask, (len(values),), "acquisition mask", kind="b"))
        _text(self.reference_frame_id, "acquisition reference")
        if not isinstance(self.provenance, ProcessingProvenance):
            raise DatasetValidationError("Acquisition provenance is required.")


@dataclass(frozen=True)
class PointCloudPreparation:
    rgb: bool = False
    timestamps: bool = False
    acquisition_geometry: bool = False

    def __post_init__(self):
        if any(type(getattr(self, key)) is not bool for key in ("rgb", "timestamps", "acquisition_geometry")):
            raise DatasetValidationError("Preparation flags must be booleans.")


@dataclass(frozen=True, eq=False)
class PointCloudFrame:
    info: FrameInfo
    source_id: str
    reference_frame_id: str
    points: np.ndarray
    annotations: AnnotationSet | None = None
    attributes: Mapping[str, np.ndarray] = field(default_factory=dict)
    reference_pose_timestamp_us: int | None = None
    world_frame_id: str | None = None
    T_world_from_reference: np.ndarray | None = None
    scan_start_us: int | None = None
    scan_end_us: int | None = None
    rgb: PointColors | None = None
    timing: PointTiming | None = None
    acquisition: AcquisitionGeometry | None = None
    _owner: object = field(default=None, repr=False, compare=False)
    _baseline: bytes | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.info, FrameInfo) or Modality.POINT_CLOUD not in self.info.modalities:
            raise DatasetValidationError("Point-cloud info must declare its modality.")
        _text(self.source_id, "source_id")
        _world(self)
        points = np.asarray(self.points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise DatasetValidationError("Point-cloud points must have shape (N,3).")
        points = _array(points, points.shape, "points", kind="f")
        object.__setattr__(self, "points", points)
        attrs = {}
        for name, value in self.attributes.items():
            _text(name, "attribute name")
            values = np.asarray(value)
            if not values.ndim or values.shape[0] != len(points) or values.dtype.kind == "O":
                raise DatasetValidationError(f"Point attribute {name!r} is not row-aligned.")
            attrs[name] = _array(values, values.shape, f"attribute {name}", finite=values.dtype.kind in "fciu")
        object.__setattr__(self, "attributes", MappingProxyType(attrs))
        annotations = self.annotations if self.annotations is not None else AnnotationSet(self.reference_frame_id)
        if not isinstance(annotations, AnnotationSet) or annotations.reference_frame_id != self.reference_frame_id:
            raise DatasetValidationError("Point annotations must use the point reference.")
        object.__setattr__(self, "annotations", annotations)
        for key in ("scan_start_us", "scan_end_us"):
            _integer(getattr(self, key), key, optional=True)
        if self.scan_start_us is not None and self.scan_end_us is not None and self.scan_start_us > self.scan_end_us:
            raise DatasetValidationError("Scan start follows scan end.")
        for key, kind in (("rgb", PointColors), ("timing", PointTiming), ("acquisition", AcquisitionGeometry)):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, kind) or len(value.valid_mask) != len(points)):
                raise DatasetValidationError(f"{key} is not aligned with point rows.")


@dataclass(frozen=True, eq=False)
class ProjectionResult:
    uv: np.ndarray
    depth: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self):
        uv = np.asarray(self.uv)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise DatasetValidationError("Projected UV must have shape (N,2).")
        uv = _array(uv, uv.shape, "projected UV", dtype=np.float64, finite=False)
        depth = _array(self.depth, (len(uv),), "projection depth", dtype=np.float64, finite=False)
        valid = _array(self.valid_mask, (len(uv),), "projection mask", kind="b")
        if np.isinf(uv).any() or np.isinf(depth).any():
            raise DatasetValidationError("Undefined projections use NaN, not infinity.")
        if np.any(valid & (~np.isfinite(uv).all(axis=1) | ~np.isfinite(depth) | (depth <= 0))):
            raise DatasetValidationError("Valid projections require finite UV and positive depth.")
        object.__setattr__(self, "uv", uv)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "valid_mask", valid)


@runtime_checkable
class ProjectionModel(Protocol):
    def project_reference(self, points: np.ndarray) -> ProjectionResult: ...


@dataclass(frozen=True, eq=False)
class CameraObservation:
    sensor_id: str
    image: np.ndarray
    timestamp_us: int | None = None
    timestamp_kind: str | None = None
    exposure_interval_us: tuple[int, int] | None = None
    pose_timestamp_us: int | None = None
    intrinsics: np.ndarray | None = None
    T_reference_from_camera: np.ndarray | None = None
    projection_model: ProjectionModel | None = None
    annotations: AnnotationSet | None = None

    def __post_init__(self):
        _text(self.sensor_id, "sensor_id")
        image = np.asarray(self.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
            raise DatasetValidationError("Camera image must be nonempty uint8 RGB (H,W,3).")
        object.__setattr__(self, "image", _array(image, image.shape, "camera image", kind="u"))
        for key in ("timestamp_us", "pose_timestamp_us"):
            _integer(getattr(self, key), key, optional=True)
        _text(self.timestamp_kind, "timestamp_kind", optional=True)
        if self.timestamp_us is not None and self.timestamp_kind is None:
            raise DatasetValidationError("Known camera timestamp needs a documented timestamp_kind.")
        if self.exposure_interval_us is not None:
            interval = tuple(self.exposure_interval_us)
            if len(interval) != 2:
                raise DatasetValidationError("Exposure interval needs two timestamps.")
            for timestamp in interval:
                _integer(timestamp, "exposure interval")
            if interval[1] < interval[0]:
                raise DatasetValidationError("Exposure interval is reversed.")
            object.__setattr__(self, "exposure_interval_us", interval)
        if self.intrinsics is not None:
            matrix = _array(self.intrinsics, (3, 3), "intrinsics", dtype=np.float64)
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1], atol=1e-9, rtol=0):
                raise DatasetValidationError("Invalid pinhole intrinsic matrix.")
            object.__setattr__(self, "intrinsics", matrix)
        if self.T_reference_from_camera is not None:
            object.__setattr__(self, "T_reference_from_camera", _transform(self.T_reference_from_camera, "T_reference_from_camera"))
        if self.projection_model is not None and (
            not isinstance(self.projection_model, ProjectionModel)
            or not callable(getattr(self.projection_model, "project_reference", None))
        ):
            raise DatasetValidationError("Projection model must implement project_reference.")
        if self.annotations is not None and not isinstance(self.annotations, AnnotationSet):
            raise DatasetValidationError("Camera annotations must be AnnotationSet.")


@dataclass(frozen=True, eq=False)
class CameraFrameSet:
    info: FrameInfo
    source_id: str
    reference_frame_id: str
    cameras: tuple[CameraObservation, ...]
    annotations: AnnotationSet | None = None
    reference_pose_timestamp_us: int | None = None
    world_frame_id: str | None = None
    T_world_from_reference: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.info, FrameInfo) or Modality.CAMERAS not in self.info.modalities:
            raise DatasetValidationError("Camera-set info must declare CAMERAS.")
        _text(self.source_id, "source_id")
        _world(self)
        cameras = tuple(self.cameras)
        if not cameras or any(not isinstance(camera, CameraObservation) for camera in cameras):
            raise DatasetValidationError("A camera set requires at least one camera observation.")
        if len({camera.sensor_id for camera in cameras}) != len(cameras):
            raise DatasetValidationError("Duplicate camera IDs in camera set.")
        annotations = self.annotations if self.annotations is not None else AnnotationSet(self.reference_frame_id)
        for annotation in (annotations, *(camera.annotations for camera in cameras if camera.annotations is not None)):
            if not isinstance(annotation, AnnotationSet) or annotation.reference_frame_id != self.reference_frame_id:
                raise DatasetValidationError("All camera annotations must use the set reference.")
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "cameras", cameras)
