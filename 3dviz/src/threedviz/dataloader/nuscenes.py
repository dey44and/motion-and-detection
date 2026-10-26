"""Compatibility constructor over the generic shared point-cloud bridge."""

from dataset_utils import create_loader
from dataset_utils.nuscenes import DETECTION_CLASSES

from .shared import SharedPointCloudAdapter, legacy_config


class NuScenesAdapter(SharedPointCloudAdapter):
    def __init__(self, *, root, version="v1.0-mini"):
        super().__init__(create_loader("nuscenes", legacy_config("nuscenes", root, version)))


__all__ = ["DETECTION_CLASSES", "NuScenesAdapter"]
