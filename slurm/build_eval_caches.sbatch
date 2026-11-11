#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J eval_caches
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 02:00:00
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# Build the fixed evaluation caches. CPU-only work, but it cannot run on a
# login node: RANGE throttles sustained per-user CPU there to about one core
# (the K400 train split fell from 29 to 2 clips/s after half an hour), and
# GPU-less jobs are killed, so it holds one H200 for its 26 cores.
# Finished splits are skipped; an unfinished split restarts from zero.
set -euo pipefail
cd /home/CLUSTER_USER/video_crw
source /home/CLUSTER_USER/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/CLUSTER_USER/conda-envs/video_next_prediction
python -u scripts/natural/build_eval_caches.py --workers "$SLURM_CPUS_PER_TASK"
