"""Raw firing-order timing for nuScenes HDL-32E LiDAR files.

Released rows retain all 32 firing slots per block in vertical ring order.
The laser fires in a different order, interleaving the lower and upper banks.
Use the metadata sweep timestamp as the end of the last firing cycle.
"""

from functools import lru_cache

import numpy as np


RINGS_PER_COLUMN = 32
COLUMN_PERIOD_NS = 46_080
FIRING_PERIOD_NS = 1_152
FIRING_INDEX_BY_RING = tuple(range(0, RINGS_PER_COLUMN, 2)) + tuple(range(1, RINGS_PER_COLUMN, 2))


def count_firing_columns(raw_points):
    """Validate the raw row layout before removing any points or reordering them."""
    if raw_points.ndim != 2 or raw_points.shape[1] != 5 or not len(raw_points):
        raise ValueError('Expected a nonempty array of raw nuScenes LiDAR rows with five columns')
    if len(raw_points) % RINGS_PER_COLUMN:
        raise ValueError('The raw LiDAR row count is not divisible by 32')
    rings = raw_points[:, 4].reshape(-1, RINGS_PER_COLUMN)
    if not np.all(rings == np.arange(RINGS_PER_COLUMN)):
        raise ValueError('The raw LiDAR rings must repeat 0 through 31 in each firing column')
    return len(rings)


@lru_cache(maxsize=32)
def point_ages_from_sweep_end(num_columns):
    """Return ages in seconds for every raw row, in released file order."""
    columns_to_end = np.arange(num_columns, 0, -1, dtype=np.int64)
    firing_offsets = np.asarray(FIRING_INDEX_BY_RING, dtype=np.int64) * FIRING_PERIOD_NS
    ages_ns = columns_to_end[:, None] * COLUMN_PERIOD_NS - firing_offsets
    ages_s = ages_ns.reshape(-1) * 1e-9
    ages_s.setflags(write=False)
    return ages_s
