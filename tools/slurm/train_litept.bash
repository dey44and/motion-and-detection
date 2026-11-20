#!/usr/bin/env bash
set -euo pipefail

# The Slurm job accepts an optional batch size and experiment tag.
variant=$1
batch_size=${2:-16}
experiment=${3:-"coarse-${variant}-b${batch_size}"}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
project_dir=/projects/nedevschi/andreii/motion-and-detection

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$project_dir/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$repo_root/tools"
args=(
    --cfg_file cfgs/nuscenes_models/centerpoint_litept.yaml
    --launcher none
    --batch_size "$batch_size"
    --workers 8
    --extra_tag "$experiment"
    --fix_random_seed
    --wo_gpu_stat
    --eval_each_epoch
    --use_amp
)

case "$variant" in
    gt-on)
        index=../data/nuscenes/v1.0-trainval/nuscenes_dbinfos_10sweeps_withvelo_litept.pkl
        if [[ ! -f "$index" ]]; then
            printf 'GT database index is missing at %s\n' "$index" >&2
            exit 1
        fi
        ;;
    gt-off)
        args+=(--set DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST "['placeholder', 'gt_sampling']")
        ;;
    *)
        printf 'Unknown LitePT training variant %s\n' "$variant" >&2
        exit 2
        ;;
esac

exec ../.venv/bin/python -u train.py "${args[@]}"
