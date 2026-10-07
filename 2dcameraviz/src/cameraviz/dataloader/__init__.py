"""Dataset adapters, neutral models and the registry factory."""

from .base import DatasetAdapter, DatasetError
from .models import BoundingBox3D, CameraFrame, CameraInfo, FrameData, FrameInfo, SceneInfo
from .factory import available_datasets, create_dataset, register_dataset
from .nuscenes import CAMERA_LAYOUT, DETECTION_CLASSES, NuScenesAdapter
from .demo import DemoAdapter

register_dataset("nuscenes", NuScenesAdapter)
register_dataset("demo", DemoAdapter)

__all__ = [
    "BoundingBox3D", "CameraFrame", "CameraInfo", "FrameData", "FrameInfo", "SceneInfo",
    "DatasetAdapter", "DatasetError", "NuScenesAdapter", "DemoAdapter", "CAMERA_LAYOUT", "DETECTION_CLASSES",
    "available_datasets", "create_dataset", "register_dataset",
]
