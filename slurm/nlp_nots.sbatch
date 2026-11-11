#!/usr/bin/env bash
#SBATCH -A commons
#SBATCH -p commons
#SBATCH -J nlp_nots
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=480G
#SBATCH --gres=gpu:lovelace:4
#SBATCH -t 24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# One 24-hour link of a chained ViT next-latent-prediction run on NOTS (4 L40S),
# slurm/nlp_chained.sbatch adapted: same calibrate-then-train flow, resume from
# $OUT/checkpoint_last.pt, SIGUSR1 15 min before the limit -> $OUT/stop_requested,
# exit at once when training_finished.json exists. Chain with slurm/chain.sh.
# /scratch is the RANGE VAST volume, so data and outputs are shared with RANGE.
# --export: OUT, ARM (B|Q), SPEC (mix spec json or a .lance store), SET (--set
# overrides, default mask_ratio=0.9), CAL_SCALE (--calibrate-scale, default 0.1),
# SIZE (default vit_tiny), WORKERS (per rank, default 12). Memory: every loader worker holds
# a ~1.2 GB copy of the mix index and batches wait in /dev/shm (prefetch 4 x workers x
# per-rank batch), so 4 ranks x 12 workers needs ~150 GB; 2 ranks x 16 workers at 128 clips
# per rank went past 128 GB (smoke 2531877, 2026-11-04).
#   bash slurm/chain.sh 3 slurm/nlp_nots.sbatch -J nlpQm_mix3_vitt --export=ALL,ARM=Q,\
#     OUT=/scratch/CLUSTER_USER/runs/nlpQm_mix3_vitt,SPEC=scripts/natural/mixes/mix256/mix3.json,CAL_SCALE=0.0125
# NOTS has no W&B login: runs log offline to /scratch/CLUSTER_USER/wandb; sync from RANGE with
# wandb sync /scratch/CLUSTER_USER/wandb/wandb/offline-run-*
set -uo pipefail
cd /home/CLUSTER_USER/video_crw
: "${OUT:?set OUT}" "${ARM:?set ARM}" "${SPEC:?set SPEC}"
SIZE=${SIZE:-vit_tiny}
E=/scratch/CLUSTER_USER/conda-envs/video_next_prediction
PY=/scratch/CLUSTER_USER/conda-envs/eval-nots/bin/python
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export WANDB_DIR=/scratch/CLUSTER_USER/wandb WANDB_MODE=${WANDB_MODE:-offline}
export LE_VJEPA_INDEX_CACHE=/scratch/CLUSTER_USER/datasets/k710/.lance_index
[ -f "$OUT/training_finished.json" ] && { echo "already finished: $OUT"; exit 0; }
mkdir -p "$OUT" "$WANDB_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
trap 'echo "$(date +%T) USR1: requesting checkpoint"; touch "$OUT/stop_requested"' USR1
case "$SPEC" in *.lance) DATA="--data $SPEC" ;; *) DATA="--mix $SPEC" ;; esac
RUN="$PY -m torch.distributed.run --standalone --nproc_per_node=$GPUS scripts/natural/train_nlp.py \
  --size $SIZE --arm $ARM $DATA --workers ${WORKERS:-12}"
EXTRA="--set ${SET:-mask_ratio=0.9}"
CAL=$OUT/calibration/calibration.json
if [ ! -f "$CAL" ]; then
  echo "$(date +%T) calibrating gamma (200 unregularized arm-$ARM steps, scale ${CAL_SCALE:-0.1})"
  rm -rf "$OUT/calibration"
  $RUN --out "$OUT/calibration" --calibrate-steps 200 --calibrate-scale "${CAL_SCALE:-0.1}" --no-wandb $EXTRA \
    || { echo "calibration failed"; exit 1; }
fi
echo "gamma $(python3 -c "import json; print(json.load(open('$CAL'))['gamma'])")"
$RUN --out "$OUT" --gamma-file "$CAL" $EXTRA &
child=$!
wait "$child"; rc=$?
while [ "$rc" -gt 128 ] && kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
echo "trainer exited with $rc"
exit "$rc"
