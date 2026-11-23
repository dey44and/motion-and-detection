#!/usr/bin/env bash
#SBATCH --job-name=litept-point-time-gt-off
#SBATCH --partition=main
#SBATCH --account=nedevschi
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_3g.71gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=/projects/nedevschi/andreii/motion-and-detection/logs/%x-%j.log

set -euo pipefail
repo_root=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
exec bash "$repo_root/tools/slurm/train_litept.bash" point-time-gt-off "$@"
