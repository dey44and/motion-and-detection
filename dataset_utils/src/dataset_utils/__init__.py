"""Dataset adapters and shared records for research visualization."""

from .config import DatasetConfig, load_dataset_config
from .dataset import DatasetLoader
from .errors import (
    DataUnavailableError, DatasetClosedError, DatasetConfigurationError,
    DatasetError, DatasetIOError, DatasetValidationError, UnknownIdentifierError,
    UnsupportedCapabilityError,
)
from .models import (
    AcquisitionGeometry, AnnotationAlignment, AnnotationSet, AnnotationStatus,
    AnnotationTiming, Box3D, CameraFrameSet, CameraObservation, Capability,
    ClassInfo, DatasetMetadata, FrameInfo, Modality, PointCloudFrame,
    PointCloudPreparation, PointColors, PointTiming, ProcessingProvenance,
    ProjectionModel, ProjectionResult, SceneInfo, SensorInfo, TimingQuality,
)
from .projection import PinholeProjection
from .registry import available_loaders, create_loader, register_loader

__version__ = "0.3.0"

__all__ = [
    "DatasetConfig", "load_dataset_config", "DatasetLoader", "available_loaders", "create_loader", "register_loader",
    "DataUnavailableError", "DatasetClosedError", "DatasetConfigurationError", "DatasetError",
    "DatasetIOError", "DatasetValidationError", "UnknownIdentifierError", "UnsupportedCapabilityError",
    "AcquisitionGeometry", "AnnotationAlignment", "AnnotationSet", "AnnotationStatus", "AnnotationTiming",
    "Box3D", "CameraFrameSet", "CameraObservation", "Capability", "ClassInfo", "DatasetMetadata", "FrameInfo",
    "Modality", "PointCloudFrame", "PointCloudPreparation", "PointColors", "PointTiming", "ProcessingProvenance",
    "ProjectionModel", "ProjectionResult", "SceneInfo", "SensorInfo", "TimingQuality", "PinholeProjection",
]
