#!/usr/bin/env bash
# One VideoFlexTok-target ViT-T run on Tamara's A40s (scripts/vft/train_vft.py), from a snapshot of the
# code taken at first launch (<out>/code_snapshot), so later edits to the tree never reach a running job.
# Resumes from checkpoint_last.pt if rerun.
#   tmux new -d -s vftN "DEVICES=0,1,2,3 PORT=29510 bash ~/video_crw/scripts/vft/run_vft.sh vftN_k710_vitt next \
#       >> ~/mixbuild_logs/vftN_k710_vitt.log 2>&1"
# LOSS=id (64,000-way, default) or digits (six FSQ digit cross-entropies); K = target positions (16).
set -uo pipefail
NAME=$1 ARM=$2
E=/scratch/CLUSTER_USER/envs/video_next_prediction
OUT=/scratch/CLUSTER_USER/runs/$NAME
DEVICES=${DEVICES:-0,1,2,3}
PORT=${PORT:-29510}
NG=$(echo "$DEVICES" | tr , '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$DEVICES LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PATH=$E/bin:$PATH
export WANDB_DIR=/scratch/CLUSTER_USER/wandb LE_VJEPA_INDEX_CACHE=/scratch/CLUSTER_USER/cache/lance_index
mkdir -p "$OUT" "$WANDB_DIR"
[ -f "$OUT/training_finished.json" ] && { echo "already finished: $OUT"; exit 0; }
S=$OUT/code_snapshot
if [ ! -d "$S" ]; then
  mkdir -p "$S/scripts" "$S/third_party"
  rsync -a "$HOME/video_crw/scripts/vft" "$HOME/video_crw/scripts/natural" "$S/scripts/"
  rsync -a --exclude=.git --exclude=__pycache__ "$HOME/video_crw/third_party/LeVJEPA" "$S/third_party/"
fi
cd "$S"
$E/bin/python -m torch.distributed.run --standalone --master-port "$PORT" --nproc_per_node="$NG" \
  scripts/vft/train_vft.py --arm "$ARM" --out "$OUT" --loss "${LOSS:-id}" --K "${K:-16}" \
  --workers "${WORKERS:-10}" ${EXTRA:-}
echo "$(date) trainer exited with $?"
