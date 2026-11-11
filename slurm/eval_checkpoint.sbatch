#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J nv_eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=300G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 03:00:00
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# Frozen-feature evaluation of one milestone checkpoint on one H200:
# extract K400 and IN1K features from the fixed caches, fit linear, attentive
# and conv-attentive probes, log to W&B, write <OUT>/evals/eNNN.json.
# --export: CKPT, OUT (trunk run dir), RUN (W&B group), EPOCH, FAMILY (levjepa).
# LeVJEPA checkpoints are read with LeVJEPA's own environment, which has
# their module's dependencies plus wandb.
set -euo pipefail
cd /home/CLUSTER_USER/video_crw
: "${CKPT:?}" "${OUT:?}" "${RUN:?}" "${EPOCH:?}"
PY=/scratch/CLUSTER_USER/conda-envs/levjepa-venv/bin/python
"$PY" -u scripts/natural/eval_checkpoint.py --checkpoint "$CKPT" --out "$OUT/evals" \
  --family "${FAMILY:-levjepa}" --run "$RUN" --epoch "$EPOCH" ${EXTRA:-} \
  && touch "$OUT/evals/e$(printf %03d "$EPOCH").done"
