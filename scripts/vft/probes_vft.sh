#!/usr/bin/env bash
# K400 evaluation of the VideoFlexTok-target ViT-T arms (decided 2026-11-12):
#   e60   fast cached frozen-feature probe (eval_checkpoint.py --datasets k400, ~15 min on one GPU),
#         run alongside training on one GPU of the arm as soon as checkpoint_epoch060.pt exists;
#         the Q / Q-frame references get the same probe at e60 first, so the comparison is matched.
#   e240  fast probe again (cheap, matches the Q references' e240 fast numbers), then the V-JEPA paper
#         probe (probe_ours.sh k400: 8 frames, 1 train segment, 8x3 views, 20 epochs, all 8 GPUs)
#         one arm at a time, once every arm has finished training.
# Finished results are skipped, so it is safe to rerun.
#   tmux new -d -s vft_probes "bash ~/video_crw/scripts/vft/probes_vft.sh >> ~/mixbuild_logs/vft_probes.log 2>&1"
set -uo pipefail
E=/scratch/CLUSTER_USER/envs/video_next_prediction
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PATH=$E/bin:$PATH WANDB_DIR=/scratch/CLUSTER_USER/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/video_crw"
R=/scratch/CLUSTER_USER/runs L=$HOME/mixbuild_logs
declare -A GPU=([vftN_k710_vitt]=3 [vftS_k710_vitt]=5 [vftM_k710_vitt]=7 [nlpQm_k710_vitt]=5 [nlpQfm_k710_vitt]=3)

fast() {   # $1 run, $2 epoch
  local ck=$R/$1/$(printf checkpoint_epoch%03d.pt "$2") tag=$(printf e%03d_full_student_k400 "$2")
  [ -f "$R/$1/evals/$tag.json" ] && return 0
  echo "$(date +'%F %T') fast $1 e$2 on GPU ${GPU[$1]}"
  CUDA_VISIBLE_DEVICES=${GPU[$1]} python scripts/natural/eval_checkpoint.py --family nlp --checkpoint "$ck" \
    --out "$R/$1/evals" --run "$1" --epoch "$2" --datasets k400 --batch 128 --workers 8 \
    > "$L/fast_$1_e$2.log" 2>&1
  echo "$(date +'%F %T') fast $1 e$2 exited $?: $(grep -m1 '^k400 ' "$L/fast_$1_e$2.log")"
}

ARMS=(vftN_k710_vitt vftS_k710_vitt vftM_k710_vitt)
fast nlpQm_k710_vitt 60
fast nlpQfm_k710_vitt 60
todo=("${ARMS[@]}")
while [ ${#todo[@]} -gt 0 ]; do               # e60 fast probes, in the order the milestones appear
  left=()
  for r in "${todo[@]}"; do
    if [ -f $R/$r/checkpoint_epoch060.pt ]; then sleep 60; fast $r 60; else left+=("$r"); fi
  done
  todo=("${left[@]}"); [ ${#todo[@]} -gt 0 ] && sleep 600
done
for r in "${ARMS[@]}"; do
  until [ -f $R/$r/training_finished.json ]; do sleep 900; done
done
echo "$(date +'%F %T') all arms finished training"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
for r in "${ARMS[@]}"; do fast $r 240; done
for r in "${ARMS[@]}"; do bash scripts/tamara/probe_ours.sh k400 $R/$r/checkpoint_epoch240.pt; done
echo "$(date +'%F %T') vft probes finished"
