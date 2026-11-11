#!/usr/bin/env bash
#SBATCH -A commons
#SBATCH -p commons
#SBATCH -J probe_nots
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --gres=gpu:lovelace:4
#SBATCH -t 24:00:00
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# Paper-protocol frozen probe (probe_vjepa.py) of a released reference encoder on NOTS: 4 L40S,
# token-matched (FRAMES = 16 for tubelet-2 encoders, 8 for tubelet-1, 1 for DINOv2 on IN1K),
# one training clip per video. /scratch is the RANGE VAST volume, so datasets and outputs are
# shared with RANGE. probe_vjepa.py saves its state every epoch: submit a chain of links with
# --dependency=afterany and each later link resumes, or exits at once when the result exists.
#   sbatch -J probe_in1k_vjepa --export=ALL,CKPT=...,OUT=/scratch/CLUSTER_USER/runs/ref_vjepa,RUN=ref_vjepa,DATASET=in1k,FAMILY=vjepa,FRAMES=16 slurm/probe_vjepa_nots.sbatch
# Env: eval-nots, a venv over the training env (same torch 2.11 + cu128) with the eval-only
# packages pinned to Tamara's versions (timm, transformers, safetensors, einops, scikit-learn).
set -euo pipefail
cd /home/CLUSTER_USER/video_crw
: "${CKPT:?}" "${OUT:?}" "${RUN:?}" "${DATASET:?}" "${FAMILY:?}" "${FRAMES:?}"
PY=/scratch/CLUSTER_USER/conda-envs/eval-nots/bin/python
# NOTS has no W&B login: runs log offline to the shared /scratch/CLUSTER_USER/wandb; upload them later from
# RANGE (same scratch, logged in) with: wandb sync /scratch/CLUSTER_USER/wandb/wandb/offline-run-*
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 WANDB_DIR=/scratch/CLUSTER_USER/wandb WANDB_MODE=${WANDB_MODE:-offline}
tag=exact_${DATASET}$([ "$FRAMES" != 8 ] && echo "_${FRAMES}f" || true)
if [ -f "$OUT/evals/$tag.json" ]; then echo "$(date) $OUT/evals/$tag.json exists, nothing to do"; exit 0; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m torch.distributed.run --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" scripts/natural/probe_vjepa.py \
  --dataset "$DATASET" --family "$FAMILY" --checkpoint "$CKPT" --out "$OUT/evals" --run "$RUN" --frames "$FRAMES" \
  --train-segments 1 --epochs 20 --workers "${WORKERS:-20}" ${EXTRA:-}
