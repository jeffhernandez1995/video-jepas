#!/usr/bin/env bash
# Paper-protocol probes (probe_vjepa.py, token-matched: 8 temporal slots per segment) run one
# after another on all 8 A40s, after an optional tmux session has ended. Each job is
# "dataset family checkpoint frames"; finished jobs (result JSON present) are skipped.
#   tmux new -d -s probe_q2 "bash ~/video_crw/scripts/tamara/probe_queue.sh --after probe_refs \
#       'k400 videomae /scratch/CLUSTER_USER/models/videomae_large/model.safetensors 16' ... >> ~/mixbuild_logs/probe_queue.log 2>&1"
set -uo pipefail
E=/scratch/CLUSTER_USER/envs/video_next_prediction
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PATH=$E/bin:$PATH WANDB_DIR=/scratch/CLUSTER_USER/wandb
cd "$HOME/video_crw"
if [ "${1:-}" = --after ]; then
  echo "$(date) waiting for tmux session $2"
  while tmux has-session -t "=$2" 2>/dev/null; do sleep 120; done
  shift 2
fi
for job in "$@"; do
  read -r ds fam ckpt frames <<< "$job"
  out=/scratch/CLUSTER_USER/runs/ref_$fam/evals; tag=exact_${ds}$([ "$frames" != 8 ] && echo "_${frames}f")
  [ -f "$out/$tag.json" ] && { echo "$(date) skip $fam $ds (done)"; continue; }
  echo "$(date) start $fam $ds $frames frames"
  "$E/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/natural/probe_vjepa.py \
    --dataset "$ds" --family "$fam" --checkpoint "$ckpt" --out "$out" --run "ref_$fam" --frames "$frames" \
    --train-segments 1 --epochs 20 --workers 12 > "/home/CLUSTER_USER/mixbuild_logs/probe_${fam}_${ds}.log" 2>&1
  echo "$(date) $fam $ds exited $?"
done
