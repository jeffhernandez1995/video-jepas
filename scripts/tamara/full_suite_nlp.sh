#!/usr/bin/env bash
# The baselines' full evaluation suite for our ViT checkpoints on Tamara's 8 A40s (student
# encoder, e240 by default), the same scripts and settings as the six references: DAVIS-2017,
# physics linear probes (IntPhys 2, InfLevel-Lab), the IntPhys 2 paper probe (Esmati et al.,
# 16 frames), then the paper-protocol attentive probes K400 / SSv2 / IN1K (probe_ours.sh, all 8
# GPUs, one at a time). Surprise is left out (dropped 2026-11-06). Waits for every run's
# training_finished.json; finished results are skipped, so it is safe to rerun.
#   tmux new -d -s suite_h1 "bash ~/video_crw/scripts/tamara/full_suite_nlp.sh nlpQh1m_k710_vitt \
#       nlpQfh1m_k710_vitt >> ~/mixbuild_logs/full_suite_h1.log 2>&1"
set -uo pipefail
E=/scratch/CLUSTER_USER/envs/video_next_prediction
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PATH=$E/bin:$PATH WANDB_DIR=/scratch/CLUSTER_USER/wandb
cd "$HOME/video_crw"
L=/home/CLUSTER_USER/mixbuild_logs
EPOCH=${EPOCH:-240}
RUNS=("$@")
for r in "${RUNS[@]}"; do
  until [ -f /scratch/CLUSTER_USER/runs/$r/training_finished.json ]; do
    echo "$(date) waiting for $r to finish training"; sleep 600
  done
done
# phase A needs the GPUs the trainers used to be released
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
small() {   # $1 run, $2 gpu
  local R=/scratch/CLUSTER_USER/runs/$1 g=$2
  local CK=$R/$(printf checkpoint_epoch%03d.pt "$EPOCH")
  [ -f $R/davis2017.json ] || CUDA_VISIBLE_DEVICES=$g python scripts/natural/eval_davis.py --family nlp \
    --checkpoint $CK --out $R > $L/davis_$1.log 2>&1; echo "$(date +%T) $1 davis $?"
  [ -f $R/physics_probe.json ] || CUDA_VISIBLE_DEVICES=$g python scripts/natural/eval_physics.py --family nlp \
    --checkpoint $CK --out $R > $L/physics_$1.log 2>&1; echo "$(date +%T) $1 physics $?"
  ls $R/intphys2_paper_probe_*.json >/dev/null 2>&1 || CUDA_VISIBLE_DEVICES=$g python scripts/natural/eval_intphys2_paper.py \
    --family nlp --checkpoint $CK --out $R --frames 16 > $L/ip2paper_$1.log 2>&1; echo "$(date +%T) $1 ip2paper $?"
}
echo "$(date) phase A: ${RUNS[*]}"
g=0
for r in "${RUNS[@]}"; do small "$r" $((g % 8)) & g=$((g + 1)); done
wait
echo "$(date) phase B"
for ds in k400 ssv2 in1k; do for r in "${RUNS[@]}"; do
  bash scripts/tamara/probe_ours.sh $ds /scratch/CLUSTER_USER/runs/$r/$(printf checkpoint_epoch%03d.pt "$EPOCH")
done; done
echo "$(date) full suite finished: ${RUNS[*]}"
