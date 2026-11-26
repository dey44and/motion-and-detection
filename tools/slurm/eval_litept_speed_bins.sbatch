#!/usr/bin/env bash
#SBATCH --job-name=litept-speed-bins
#SBATCH --partition=main
#SBATCH --account=nedevschi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/nedevschi/andreii/motion-and-detection/logs/%x-%j.log

set -euo pipefail
repo_root=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
cd "$repo_root"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

runs_dir="$repo_root/output/nuscenes_models/centerpoint_litept"
output_dir=${1:-"$runs_dir/speed-bins-epoch20-$SLURM_JOB_ID"}
model_args=()
for model in coarse-gt-on-b16 coarse-gt-off-b16 point-time-gt-off-b16; do
    predictions="$runs_dir/$model/eval/eval_with_train/epoch_20/val/final_result/data/results_nusc.json"
    model_args+=(--model "$model" "$predictions")
done

exec .venv/bin/python -u tools/run_nuscenes_speed_bins.py "${model_args[@]}" --output-dir "$output_dir"
