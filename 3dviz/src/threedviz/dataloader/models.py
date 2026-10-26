"""Compatibility exports for dataset-independent shared frame records."""

from dataset_utils.models import BoundingBox3D, FrameInfo, PointCloudFrame, SceneInfo

FrameData = PointCloudFrame

__all__ = ["BoundingBox3D", "FrameData", "FrameInfo", "PointCloudFrame", "SceneInfo"]
