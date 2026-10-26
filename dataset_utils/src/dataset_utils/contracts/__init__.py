"""Public version-1 dataset extension interface."""

from .config import DatasetConfig
from .loader import DatasetLoader
from .models import (
    AcquisitionGeometry, AnnotationAlignment, AnnotationSet, AnnotationStatus,
    AnnotationTiming, Box3D, CameraFrameSet, CameraObservation, Capability,
    ClassInfo, DatasetMetadata, FrameInfo, Modality, PointCloudFrame,
    PointCloudPreparation, PointColors, PointTiming, ProcessingProvenance,
    ProjectionModel, ProjectionResult, SceneInfo, SensorInfo, TimingQuality,
)
from .projection import PinholeProjection
from .registry import available_loaders, create_loader, register_loader

__all__ = [
    "AcquisitionGeometry", "AnnotationAlignment", "AnnotationSet", "AnnotationStatus",
    "AnnotationTiming", "Box3D", "CameraFrameSet", "CameraObservation", "Capability",
    "ClassInfo", "DatasetConfig", "DatasetLoader", "DatasetMetadata", "FrameInfo",
    "Modality", "PinholeProjection", "PointCloudFrame", "PointCloudPreparation",
    "PointColors", "PointTiming", "ProcessingProvenance", "ProjectionModel",
    "ProjectionResult", "SceneInfo", "SensorInfo", "TimingQuality",
    "available_loaders", "create_loader", "register_loader",
]
