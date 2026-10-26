"""Compatibility exports for Waymo storage helpers, now shared and GUI-free.

Normal viewer construction uses the shared loader registry and generic bridge.
"""

from dataset_utils.waymo import processed as _processed
from dataset_utils.waymo.processed import (
    DEFAULT_PROCESSED_TAG, WAYMO_CLASSES, NativeWaymoAdapter, WaymoAdapter, open_waymo,
)


def __getattr__(name):
    return getattr(_processed, name)


__all__ = ["DEFAULT_PROCESSED_TAG", "WAYMO_CLASSES", "NativeWaymoAdapter", "WaymoAdapter", "open_waymo"]
