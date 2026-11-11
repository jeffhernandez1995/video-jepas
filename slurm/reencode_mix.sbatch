#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J reencode_mix
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 24:00:00
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# Re-encode the mix stores at short edge 256 / JPEG 85 into /scratch/CLUSTER_USER/datasets/mix256
# (scripts/range/reencode_mix.py; shared claims, so several jobs split the shards).
# CPU work; holds one H200 because RANGE reaps jobs without a GPU.
#   sbatch --export=ALL,SOURCES="k710_20pct k710 ssv2 egodex droid in1k pe_video cc12m vidaforge" slurm/reencode_mix.sbatch
set -euo pipefail
cd /home/CLUSTER_USER/video_crw
source /home/CLUSTER_USER/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/CLUSTER_USER/conda-envs/video_next_prediction
python -u scripts/range/reencode_mix.py --sources ${SOURCES:-k710_20pct k710 ssv2 egodex droid in1k pe_video cc12m vidaforge} \
  --workers "$SLURM_CPUS_PER_TASK"
