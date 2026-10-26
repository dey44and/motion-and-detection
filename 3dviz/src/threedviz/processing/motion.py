"""Compatibility import for shared nuScenes motion utilities."""

import sys

from dataset_utils.nuscenes import motion as _implementation

# Preserve module identity for existing integrations and monkeypatches.
sys.modules[__name__] = _implementation
