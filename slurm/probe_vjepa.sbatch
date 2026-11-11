#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J probe_vjepa
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=208
#SBATCH --gres=gpu:h200:8
#SBATCH -t 10:00:00
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# V-JEPA exact frozen-probe protocol on one checkpoint, one dataset, one node.
# --export: CKPT, OUT (run dir), RUN (W&B group), DATASET (in1k|k400), EXTRA.
# Not --exclusive: the flag is inherited by smaller command-line overrides
# (a 2-GPU smoke test with it waited 9 h for an empty node while 1-GPU
# non-exclusive jobs started in minutes). 8 GPUs + 208 CPUs fills a node.
set -euo pipefail
cd /home/CLUSTER_USER/video_crw
: "${CKPT:?}" "${OUT:?}" "${RUN:?}" "${DATASET:?}"
PY=/scratch/CLUSTER_USER/conda-envs/levjepa-venv/bin
export OMP_NUM_THREADS=4 WANDB_DIR=/scratch/CLUSTER_USER/wandb
"$PY/python" -c "import timm, av" 2>/dev/null || UV_CACHE_DIR=/scratch/CLUSTER_USER/.cache/uv uv pip install -q --python "$PY/python" timm av
"$PY/torchrun" --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-8}" scripts/natural/probe_vjepa.py \
  --dataset "$DATASET" --checkpoint "$CKPT" --out "$OUT/evals" --run "$RUN" --workers 24 ${EXTRA:-} \
  && touch "$OUT/evals/exact_${DATASET}.done"
