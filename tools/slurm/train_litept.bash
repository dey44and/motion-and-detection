#!/usr/bin/env bash
set -euo pipefail

# The Slurm job accepts an optional batch size and experiment tag.
variant=$1
batch_size=${2:-16}
default_experiment="coarse-${variant}-b${batch_size}"
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
    gt-off|point-time-gt-off)
        overrides=(DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST "['placeholder', 'gt_sampling']")
        ;;
    *)
        printf 'Unknown LitePT training variant %s\n' "$variant" >&2
        exit 2
        ;;
esac

if [[ "$variant" == point-time-gt-off ]]; then
    default_experiment="${variant}-b${batch_size}"
    info_dir=point_time_hdl32e_raw_order
    train_info="$info_dir/nuscenes_infos_10sweeps_train.pkl"
    val_info="$info_dir/nuscenes_infos_10sweeps_val.pkl"
    for info in "$train_info" "$val_info"; do
        path="../data/nuscenes/v1.0-trainval/$info"
        if [[ ! -f "$path" ]]; then
            printf 'Point-timing info file is missing at %s\n' "$path" >&2
            exit 1
        fi
    done
    overrides+=(
        DATA_CONFIG.POINT_TIME_MODE hdl32e_raw_order
        DATA_CONFIG.INFO_PATH.train "['$train_info']"
        DATA_CONFIG.INFO_PATH.test "['$val_info']"
    )
fi

args+=(--extra_tag "${3:-$default_experiment}")
if [[ "$variant" != gt-on ]]; then
    args+=(--set "${overrides[@]}")
fi

exec ../.venv/bin/python -u train.py "${args[@]}"
