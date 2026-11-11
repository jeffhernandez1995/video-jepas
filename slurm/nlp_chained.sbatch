#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J nlp
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=208
#SBATCH --mem-per-cpu=9G
#SBATCH --gres=gpu:h200:8
#SBATCH -t 24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# One 24-hour link of a chained next-latent-prediction run (scripts/natural/train_nlp.py).
# Chain with slurm/chain.sh. --export: SIZE (vit_tiny|vit_small|vit_base), OUT, ARM (A|B), EXTRA,
# GUARD=1 for A' (A plus B's calibrated coding-rate guard; B always has it).
# GPU and CPU counts come from the sbatch command line; CPUs are sized to the
# loader, not to the GPUs (docs/levjepa_loader_bottleneck.md): ViT-T and ViT-S
# get 104 CPUs on 2 and 4 GPUs, ViT-B 208 on 8. Not --exclusive, so 2- and
# 4-GPU jobs can start on partly used nodes.
# Direction B calibrates its coding-rate weight first, as in wave 2: the first
# link runs an unregularized B for 200 steps into $OUT/calibration, measures
# gamma = 0.1 x ||grad prediction|| / ||grad coding rate|| on the encoder at
# step 200, and the real run then starts from scratch with that gamma.
# The trainer resumes from $OUT/checkpoint_last.pt and exits at once when
# training_finished.json exists. SIGUSR1 (15 min before the limit) becomes
# $OUT/stop_requested, which rank 0 polls every 50 steps before saving.
set -uo pipefail
cd /home/CLUSTER_USER/video_crw
: "${SIZE:?set SIZE}" "${OUT:?set OUT}"
ARM=${ARM:-A}
PY=/scratch/CLUSTER_USER/conda-envs/levjepa-venv/bin
export OMP_NUM_THREADS=2 WANDB_DIR=/scratch/CLUSTER_USER/wandb PYTHONUNBUFFERED=1
export LE_VJEPA_INDEX_CACHE=/scratch/CLUSTER_USER/datasets/k710/.lance_index
[ -f "$OUT/training_finished.json" ] && { echo "already finished: $OUT"; exit 0; }
mkdir -p "$OUT" /scratch/CLUSTER_USER/wandb
GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
trap 'echo "$(date +%T) USR1: requesting checkpoint"; touch "$OUT/stop_requested"' USR1
GAMMA_ARGS=""
if [ "$ARM" = B ] || [ "${GUARD:-0}" = 1 ]; then
  CAL=$OUT/calibration/calibration.json
  if [ ! -f "$CAL" ]; then
    echo "$(date +%T) calibrating gamma (200 unregularized arm-$ARM steps)"
    rm -rf "$OUT/calibration"
    "$PY/torchrun" --standalone --nproc_per_node="$GPUS" scripts/natural/train_nlp.py \
      --size "$SIZE" --arm "$ARM" --out "$OUT/calibration" --calibrate-steps 200 --no-wandb ${EXTRA:-} \
      || { echo "calibration failed"; exit 1; }
  fi
  GAMMA_ARGS="--gamma-file $CAL"
  echo "gamma $(python3 -c "import json; print(json.load(open('$CAL'))['gamma'])")"
fi
"$PY/torchrun" --standalone --nproc_per_node="$GPUS" scripts/natural/train_nlp.py \
  --size "$SIZE" --arm "$ARM" --out "$OUT" $GAMMA_ARGS ${EXTRA:-} &
child=$!
wait "$child"; rc=$?
while [ "$rc" -gt 128 ] && kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
echo "trainer exited with $rc"
exit "$rc"
