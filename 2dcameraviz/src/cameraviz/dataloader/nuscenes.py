"""Compatibility constructor over the generic shared camera bridge."""

from dataset_utils import DatasetConfig, create_loader
from dataset_utils.nuscenes import DETECTION_CLASSES

from .models import CameraInfo
from .shared import SharedCameraAdapter


CAMERA_LAYOUT = (
    CameraInfo("CAM_FRONT_LEFT", "Front left", 0, 0),
    CameraInfo("CAM_FRONT", "Front", 0, 1),
    CameraInfo("CAM_FRONT_RIGHT", "Front right", 0, 2),
    CameraInfo("CAM_BACK_RIGHT", "Back right", 1, 0),
    CameraInfo("CAM_BACK", "Back", 1, 1),
    CameraInfo("CAM_BACK_LEFT", "Back left", 1, 2),
)


class NuScenesAdapter(SharedCameraAdapter):
    def __init__(self, *, root, version="v1.0-mini"):
        super().__init__(create_loader("nuscenes", DatasetConfig(root=root, release=version)))
