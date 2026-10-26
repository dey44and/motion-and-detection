"""Compatibility import for shared official nuScenes scene memberships."""

import sys

from dataset_utils.nuscenes import splits as _implementation

sys.modules[__name__] = _implementation
