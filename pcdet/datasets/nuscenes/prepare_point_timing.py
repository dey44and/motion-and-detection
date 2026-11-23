"""Add validated raw-order LiDAR timing to the existing nuScenes info splits.

One firing-column count per source sweep encodes its complete timing vector
without duplicating the LiDAR data or changing the split order.
"""

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tqdm import tqdm

from .hdl32e_timing import (
    COLUMN_PERIOD_NS, FIRING_INDEX_BY_RING, FIRING_PERIOD_NS, RINGS_PER_COLUMN, count_firing_columns,
)


def scan_column_count(lidar_path):
    try:
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        return count_firing_columns(points)
    except ValueError as exc:
        raise ValueError(f'{lidar_path}: {exc}') from exc


def add_timing(infos, root_path, workers, mode):
    paths = list(dict.fromkeys(scan['lidar_path'] for info in infos for scan in (info, *info['sweeps'])))
    counts = {}
    # Submit bounded batches to keep the full split out of the thread pool queue.
    with ThreadPoolExecutor(max_workers=workers) as executor, tqdm(
        total=len(paths), desc=f'{mode} LiDAR scans', mininterval=10,
    ) as progress:
        for start in range(0, len(paths), 1024):
            batch = paths[start:start + 1024]
            counts.update(zip(batch, executor.map(scan_column_count, (root_path / path for path in batch))))
            progress.update(len(batch))

    for info in infos:
        for scan in (info, *info['sweeps']):
            scan['hdl32e_num_columns'] = counts[scan['lidar_path']]

    return {
        'keyframes': len(infos),
        'unique_scans': len(counts),
        'min_firing_columns': min(counts.values()),
        'max_firing_columns': max(counts.values()),
    }


def prepare_point_timing(dataset_cfg, root_path, *, output_dir=None, workers=1):
    """Publish timing-aware copies of the existing train and validation infos."""
    output_dir = output_dir or root_path / 'point_time_hdl32e_raw_order'
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'mode': 'hdl32e_raw_order',
        'rings_per_column': RINGS_PER_COLUMN,
        'column_period_ns': COLUMN_PERIOD_NS,
        'firing_period_ns': FIRING_PERIOD_NS,
        'firing_index_by_ring': FIRING_INDEX_BY_RING,
        'sweep_timestamp_reference': 'end of final firing cycle',
        'model_time_feature': 'age relative to keyframe timestamp in seconds',
        'splits': {},
    }
    with TemporaryDirectory(dir=output_dir.parent, prefix=f'.{output_dir.name}-') as temp_dir:
        stage = Path(temp_dir)
        for mode in ('train', 'test'):
            infos = []
            sources = dataset_cfg.INFO_PATH[mode]
            for source in sources:
                with (root_path / source).open('rb') as handle:
                    infos.extend(pickle.load(handle))
            if not infos:
                raise ValueError(f'No input nuScenes infos for {mode}')
            split = dataset_cfg.DATA_SPLIT[mode]
            name = f'nuscenes_infos_{dataset_cfg.MAX_SWEEPS}sweeps_{split}.pkl'
            stats = add_timing(infos, root_path, workers, split)
            with (stage / name).open('wb') as handle:
                pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
            metadata['splits'][split] = {'output': name, 'sources': sources, **stats}
            print(f'{split}: {stats["keyframes"]:,} keyframes, {stats["unique_scans"]:,} unique scans')
            del infos
        (stage / 'timing.json').write_text(json.dumps(metadata, indent=2) + '\n')
        stage.rename(output_dir)
    print(f'Published timing-aware info files in {output_dir.resolve()}')
