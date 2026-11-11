#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J nlp_smoke
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem-per-cpu=9G
#SBATCH --gres=gpu:h200:2
#SBATCH -t 02:00:00
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# Throughput and correctness smoke for train_nlp.py: each size at its production
# per-GPU batch (T 128, S 64, B 32 clips) on 2 GPUs, 30 steps, real K710 store.
set -uo pipefail
cd /home/CLUSTER_USER/video_crw
PY=/scratch/CLUSTER_USER/conda-envs/levjepa-venv/bin
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 LE_VJEPA_INDEX_CACHE=/scratch/CLUSTER_USER/datasets/k710/.lance_index
for spec in ${SPECS:-vit_small:128:24 vit_tiny:256:50 vit_base:64:24}; do
  IFS=: read size batch workers arm <<< "$spec"; arm=${arm:-A}
  out=/scratch/CLUSTER_USER/runs/nlp_smoke/${size}_${arm}_${SLURM_JOB_ID}
  echo "=== $size arm $arm global batch $batch workers/rank $workers"
  gamma=""
  if [ "$arm" = B ] || [ "${GUARD:-0}" = 1 ]; then
    "$PY/torchrun" --standalone --nproc_per_node=2 scripts/natural/train_nlp.py --size "$size" --arm "$arm" --out "$out/calibration" \
      --batch "$batch" --workers "$workers" --calibrate-steps 10 --no-wandb ${EXTRA:-} 2>&1 | grep -E "calibration|Error|Traceback" | tail -5
    gamma="--gamma-file $out/calibration/calibration.json"
  fi
  "$PY/torchrun" --standalone --nproc_per_node=2 scripts/natural/train_nlp.py --size "$size" --arm "$arm" --out "$out" \
    --batch "$batch" --workers "$workers" --smoke-steps 30 --no-wandb $gamma ${EXTRA:-} 2>&1 | grep -vE "UserWarning|warnings.warn" | tail -12
  cat "$out/smoke.json" 2>/dev/null
done
