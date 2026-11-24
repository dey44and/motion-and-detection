"""Count nuScenes CenterHead target collisions from the train and val info files.

Run from tools/ so the model YAML resolves its dataset base configuration.
Frame-cell counts treat the same BEV cell in different frames as separate cells.
"""

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pcdet.config import EasyDict, cfg_from_yaml_file


def load_targets(cfg, mode, class_ids, seen_tokens):
    data_cfg = cfg.DATA_CONFIG
    root_path = Path(data_cfg.DATA_PATH) / data_cfg.VERSION
    bounds = np.asarray(data_cfg.POINT_CLOUD_RANGE)
    min_lidar_points = data_cfg.get('FILTER_MIN_POINTS_IN_GT', 0)
    frames, labels, centers = [], [], []
    frame_count = 0

    for info_path in data_cfg.INFO_PATH[mode]:
        with (root_path / info_path).open('rb') as handle:
            infos = pickle.load(handle)
        for info in infos:
            token = info['token']
            if token in seen_tokens:
                raise ValueError(f'Duplicate nuScenes keyframe token {token}')
            seen_tokens.add(token)
            frame_count += 1

            boxes = info['gt_boxes']
            classes = np.fromiter((class_ids.get(name, -1) for name in info['gt_names']), dtype=np.int32)
            keep = (
                (classes >= 0)
                & (info['num_lidar_pts'] >= min_lidar_points)
                & ((boxes[:, :3] >= bounds[:3]) & (boxes[:, :3] <= bounds[3:])).all(axis=1)
                & (boxes[:, 3:5] > 0).all(axis=1)
            )
            labels.append(classes[keep])
            frames.append(np.full(keep.sum(), len(seen_tokens) - 1, dtype=np.int64))
            centers.append(boxes[keep, :2].astype(np.float32))
        del infos

    return frame_count, (np.concatenate(frames), np.concatenate(labels), np.concatenate(centers))


def grid_indices(centers, bounds, cell_size):
    grid_size = np.ceil((bounds[3:5] - bounds[:2]) / cell_size - 1e-9).astype(np.int64)
    xy = (centers - bounds[:2].astype(np.float32)) / np.float32(cell_size)
    # CenterHead clamps continuous coordinates before taking the integer cell.
    indices = np.clip(xy, 0, grid_size.astype(np.float32) - 0.5).astype(np.int64)
    return indices[:, 1] * grid_size[0] + indices[:, 0], grid_size


def count_collisions(frames, entities, cells, n_entities, n_cells):
    keys = (frames * n_entities + entities) * n_cells + cells
    unique_keys, counts = np.unique(keys, return_counts=True)
    occupied_entities = (unique_keys // n_cells) % n_entities
    collided = counts > 1
    multiplicity = counts[collided]
    collision_entities = occupied_entities[collided]
    collision_cells = np.bincount(collision_entities, minlength=n_entities)
    affected = np.bincount(collision_entities, weights=multiplicity, minlength=n_entities).astype(np.int64)
    pairs = multiplicity * (multiplicity - 1) // 2
    frame_entities = np.unique(unique_keys[collided] // n_cells)
    return {
        'targets': np.bincount(entities, minlength=n_entities),
        'occupied_frame_cells': np.bincount(occupied_entities, minlength=n_entities),
        'collision_frame_cells': collision_cells,
        'affected_targets': affected,
        'excess_targets': affected - collision_cells,
        'collision_pairs': np.bincount(collision_entities, weights=pairs, minlength=n_entities).astype(np.int64),
        'frames_with_collisions': np.bincount(frame_entities % n_entities, minlength=n_entities),
    }, np.unique(frame_entities // n_entities).size


def collision_rows(cfg, splits, cell_sizes, class_names):
    bounds = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    head_names = ['+'.join(classes) for classes in cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD]
    head_ids = {name: i for i, head in enumerate(cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD) for name in head}
    class_to_head = np.array([head_ids[name] for name in class_names])

    for split, (frame_count, (frames, labels, centers)) in splits.items():
        for cell_size in cell_sizes:
            cells, grid_size = grid_indices(centers, bounds, cell_size)
            n_cells = grid_size.prod()
            class_counts, _ = count_collisions(frames, labels, cells, len(class_names), n_cells)
            head_counts, collided_frames = count_collisions(
                frames, class_to_head[labels], cells, len(head_names), n_cells,
            )
            same_class_pairs = np.bincount(
                class_to_head, weights=class_counts['collision_pairs'], minlength=len(head_names),
            ).astype(np.int64)

            def row(level, entity, head, counts, same_pairs):
                targets = counts['targets']
                return {
                    'split': split, 'cell_size_m': cell_size, 'level': level,
                    'entity': entity, 'head': head, 'keyframes': frame_count,
                    **{key: int(value) for key, value in counts.items()},
                    'same_class_pairs': int(same_pairs),
                    'cross_class_pairs': int(counts['collision_pairs'] - same_pairs),
                    'affected_pct': round(100 * counts['affected_targets'] / targets, 6) if targets else 0.0,
                }

            total = {key: values.sum() for key, values in head_counts.items()}
            total['frames_with_collisions'] = collided_frames
            yield row('all_heads', 'all', '', total, same_class_pairs.sum())
            for i, name in enumerate(head_names):
                yield row('head', name, name, {key: values[i] for key, values in head_counts.items()}, same_class_pairs[i])
            for i, name in enumerate(class_names):
                counts = {key: values[i] for key, values in class_counts.items()}
                yield row('class', name, head_names[class_to_head[i]], counts, counts['collision_pairs'])


def save_heatmaps(cfg, targets, class_names, cell_size, frame_count, output_dir):
    _, labels, centers = targets
    bounds = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    cells, grid_size = grid_indices(centers, bounds, cell_size)
    nx, ny = grid_size
    n_cells = nx * ny
    counts = np.bincount(labels * n_cells + cells, minlength=len(class_names) * n_cells)
    counts = counts.reshape(len(class_names), ny, nx)
    scale = LogNorm(vmin=1, vmax=max(2, counts.max()))
    cmap = matplotlib.colormaps['viridis'].with_extremes(bad='#f3f4f6')

    n_columns = min(len(class_names), 5)
    n_rows = (len(class_names) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(3.0 * n_columns, 3.15 * n_rows + 0.9),
        sharex=True, sharey=True, squeeze=False, layout='constrained',
    )
    used_axes = axes.flat[:len(class_names)]
    for ax, label, image in zip(used_axes, class_names, counts):
        density = ax.imshow(
            np.ma.masked_equal(image, 0), origin='lower', interpolation='nearest',
            extent=(bounds[0], bounds[0] + nx * cell_size, bounds[1], bounds[1] + ny * cell_size),
            cmap=cmap, norm=scale, aspect='equal',
        )
        ax.set(xlim=(bounds[0], bounds[3]), ylim=(bounds[1], bounds[4]))
        ax.set_title(f'{label.replace("_", " ").capitalize()} ({image.sum():,})', fontsize=10)
        ax.set_xticks([-50, -25, 0, 25, 50])
        ax.set_yticks([-50, -25, 0, 25, 50])
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(class_names):]:
        ax.set_visible(False)

    fig.suptitle(
        f'nuScenes ground-truth object centers\n'
        f'Train + validation ({frame_count:,} keyframes) | {cell_size:g} m BEV cells', fontsize=14,
    )
    fig.supxlabel('Keyframe LiDAR x (m)', fontsize=10)
    fig.supylabel('Keyframe LiDAR y (m)', fontsize=10)
    fig.colorbar(density, ax=used_axes, shrink=0.83, label='Ground-truth centers per cell (log scale)')
    for suffix in ('pdf', 'png'):
        fig.savefig(output_dir / f'bev_center_heatmaps.{suffix}', dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg-file', default='cfgs/nuscenes_models/centerpoint_litept.yaml')
    parser.add_argument('--cell-sizes', type=float, nargs='+', default=[0.2, 0.4, 0.8])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not all(np.isfinite(size) and size > 0 for size in args.cell_sizes):
        parser.error('Cell sizes must be finite and positive')

    cfg = cfg_from_yaml_file(args.cfg_file, EasyDict())
    class_names = cfg.CLASS_NAMES
    class_ids = {name: i for i, name in enumerate(class_names)}
    seen_tokens = set()
    splits = {
        'train': load_targets(cfg, 'train', class_ids, seen_tokens),
        'val': load_targets(cfg, 'test', class_ids, seen_tokens),
    }
    combined = tuple(np.concatenate(arrays) for arrays in zip(splits['train'][1], splits['val'][1]))
    splits['trainval'] = len(seen_tokens), combined

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(collision_rows(cfg, splits, dict.fromkeys(args.cell_sizes), class_names))
    with (args.output_dir / 'centerhead_collisions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    voxel_size = next(proc.VOXEL_SIZE for proc in cfg.DATA_CONFIG.DATA_PROCESSOR if 'VOXEL_SIZE' in proc)
    model_cell_size = voxel_size[0] * cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
    save_heatmaps(cfg, combined, class_names, model_cell_size, len(seen_tokens), args.output_dir)
    for split, (frames, targets) in splits.items():
        print(f'{split}: {frames:,} keyframes, {len(targets[0]):,} eligible targets')
    print(f'Output directory: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
