#!/usr/bin/env bash
#SBATCH --job-name=litept-gt-db
#SBATCH --partition=main
#SBATCH --account=nedevschi
#SBATCH --gres=gpu:nvidia_h200_1g.18gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/nedevschi/andreii/nuScenes/litept-gt-db-%j.log

set -euo pipefail

repo_root=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
cd "$repo_root"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
data_root=${1:-"$repo_root/data/nuscenes"}

.venv/bin/python -u - "$data_root" <<'PY'
import sys
from pathlib import Path

import torch

from pcdet.config import EasyDict, cfg_from_yaml_file
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.utils.common_utils import create_logger

if not torch.cuda.is_available():
    raise RuntimeError('nuScenes GT database generation requires a CUDA device')

cfg = cfg_from_yaml_file('tools/cfgs/dataset_configs/nuscenes_dataset.yaml', EasyDict())
dataset = NuScenesDataset(
    dataset_cfg=cfg,
    class_names=None,
    training=True,
    root_path=Path(sys.argv[1]),
    logger=create_logger(),
)
if not dataset.infos:
    raise FileNotFoundError(f'No training infos found in {dataset.root_path}')

print(f'Generating LitePT GT database in {dataset.root_path.resolve()}')
dataset.create_groundtruth_database(max_sweeps=cfg.MAX_SWEEPS, suffix='withvelo_litept', overwrite=False)
PY
