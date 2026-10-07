"""Dataset adapters, common geometry models, and the adapter registry factory."""

from .base import DatasetAdapter, DatasetError
from .demo import DemoAdapter
from .factory import available_datasets, create_dataset, register_dataset
from .models import BoundingBox3D, FrameData, FrameInfo, SceneInfo
from .nuscenes import DETECTION_CLASSES, NuScenesAdapter

register_dataset("nuscenes", NuScenesAdapter)
register_dataset("demo", DemoAdapter)

__all__ = [
    "BoundingBox3D", "DatasetAdapter", "DatasetError", "DETECTION_CLASSES",
    "DemoAdapter", "FrameData", "FrameInfo", "NuScenesAdapter", "SceneInfo",
    "available_datasets", "create_dataset", "register_dataset",
]
