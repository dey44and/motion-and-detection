"""Compatibility import for shared nuScenes projection utilities."""

import sys

from dataset_utils.nuscenes import projection as _implementation

# Preserve module identity for existing integrations and monkeypatches.
sys.modules[__name__] = _implementation
