#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J lance_k710
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=300G
#SBATCH --gres=gpu:h200:4
#SBATCH -t 24:00:00
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# Build the K710 subset Lance store. This is CPU work, but RANGE kills jobs
# that hold no GPU, so it holds four H200s and their 104 CPUs. The smoke test
# measured ~0.16 clip/s per worker, so the 20% subset (~127k clips) takes
# about two hours at 104 workers. SUBSET selects the csv (20 by default).
set -euo pipefail
cd /home/CLUSTER_USER/video_crw
source /home/CLUSTER_USER/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/CLUSTER_USER/conda-envs/video_next_prediction
SUBSET=${SUBSET:-20}
CSV=/scratch/CLUSTER_USER/datasets/k710/train_${SUBSET}pct.csv
OUT=/scratch/CLUSTER_USER/datasets/k710/train_${SUBSET}pct.lance
if [ -f "$OUT.stats.json" ]; then echo "already built: $OUT"; exit 0; fi
if [ -d "$OUT" ]; then echo "partial store found, removing: $OUT"; rm -rf "$OUT"; fi
python -u scripts/range/build_lance_kinetics.py --csv "$CSV" --out "$OUT" --workers "$SLURM_CPUS_PER_TASK"
