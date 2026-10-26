"""Compatibility constructor over the generic shared camera bridge."""

from dataset_utils import DatasetConfig, create_loader

from .models import CameraInfo
from .shared import SharedCameraAdapter


WAYMO_CLASSES = ("Vehicle", "Pedestrian", "Sign", "Cyclist")
CAMERA_LAYOUT = (
    CameraInfo("CAM_FRONT_LEFT", "Front left", 0, 0),
    CameraInfo("CAM_FRONT", "Front", 0, 1),
    CameraInfo("CAM_FRONT_RIGHT", "Front right", 0, 2),
    CameraInfo("CAM_SIDE_LEFT", "Side left", 1, 0),
    CameraInfo("CAM_SIDE_RIGHT", "Side right", 1, 2),
)


class WaymoAdapter(SharedCameraAdapter):
    def __init__(self, *, root, version="v1.0-mini"):
        release = None if version == "v1.0-mini" else version
        super().__init__(create_loader("waymo", DatasetConfig(root=root, release=release)))


__all__ = ["CAMERA_LAYOUT", "WAYMO_CLASSES", "WaymoAdapter"]
