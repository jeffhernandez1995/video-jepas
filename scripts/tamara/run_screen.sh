#!/usr/bin/env bash
# One ViT-T masked-B (h1/2/4, 90% tube masks) screen run on Tamara's A40s, the
# same recipe as nlpBm_k710_vitt, on a 256-px mix (scripts/natural/mixes/mix256/).
#   tmux new -d -s nlp_mix1 "DEVICES=0,1,2,3 PORT=29500 bash ~/video_crw/scripts/tamara/run_screen.sh \
#       nlpBm_mix1_vitt scripts/natural/mixes/mix256/mix1.json >> ~/mixbuild_logs/nlpBm_mix1_vitt.log 2>&1"
# Calibrates gamma first (200 unregularized steps), as slurm/nlp_chained.sbatch
# does, then trains 240 virtual epochs; resumes from checkpoint_last.pt if rerun.
# ARM=A runs lean masked A instead (Sinkhorn codes, 10 iterations, no guard),
# the recipe of nlpAm_k710_vitt; there is nothing to calibrate. SET adds --set
# overrides to the masked-B run (and its calibration), e.g. SET="guard=sigreg".
# ARM=Q (categorical random-projection targets) takes B's path: calibrated guard.
# SPEC may be a .lance store instead of a mix spec (the 20% K710 store:
# /scratch/CLUSTER_USER/datasets/mix256/k710_20pct/train_20pct.lance). CAL_SCALE sets
# --calibrate-scale (0.05 matches nlpBm_k710_vitt's guard, 0.0125 nlpBm_mix3_vitt's).
set -uo pipefail
NAME=$1 SPEC=$2
E=/scratch/CLUSTER_USER/envs/video_next_prediction
OUT=/scratch/CLUSTER_USER/runs/$NAME
DEVICES=${DEVICES:-0,1,2,3}
PORT=${PORT:-29500}
ARM=${ARM:-B}
NG=$(echo "$DEVICES" | tr , '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$DEVICES LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PATH=$E/bin:$PATH
export WANDB_DIR=/scratch/CLUSTER_USER/wandb LE_VJEPA_INDEX_CACHE=/scratch/CLUSTER_USER/cache/lance_index
cd "$HOME/video_crw"
mkdir -p "$OUT" "$WANDB_DIR"
[ -f "$OUT/training_finished.json" ] && { echo "already finished: $OUT"; exit 0; }
case "$SPEC" in *.lance) DATA="--data $SPEC" ;; *) DATA="--mix $SPEC" ;; esac
RUN="$E/bin/python -m torch.distributed.run --standalone --master-port $PORT --nproc_per_node=$NG scripts/natural/train_nlp.py --size vit_tiny --arm $ARM $DATA --workers 10"
if [ "$ARM" = A ]; then
  $RUN --out "$OUT" --set mask_ratio=0.9 sinkhorn_iters=10
  echo "$(date) trainer exited with $?"
  exit
fi
EXTRA="--set mask_ratio=0.9 ${SET:-}"     # SET="guard=sigreg" for the SIGReg guard
CAL=$OUT/calibration/calibration.json
if [ ! -f "$CAL" ]; then
  rm -rf "$OUT/calibration"
  $RUN --out "$OUT/calibration" --calibrate-steps 200 --calibrate-scale "${CAL_SCALE:-0.1}" --no-wandb $EXTRA \
    || { echo "calibration failed"; exit 1; }
fi
echo "$(date) gamma $(python3 -c "import json; print(json.load(open('$CAL'))['gamma'])")"
$RUN --out "$OUT" --gamma-file "$CAL" $EXTRA
echo "$(date) trainer exited with $?"
