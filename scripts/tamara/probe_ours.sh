#!/usr/bin/env bash
# Paper-protocol probe (probe_vjepa.py) of one of our next-latent-prediction checkpoints on all
# 8 A40s, after an optional tmux session has ended; token-matched 8 frames (tubelet 1), student
# weights (EVAL_WEIGHTS=teacher for the EMA). Skips when the result JSON exists.
#   tmux new -d -s probe_ours "bash ~/video_crw/scripts/tamara/probe_ours.sh --after probe_q2 ssv2 \
#       /scratch/CLUSTER_USER/runs/nlpBm_mix3_vitl_sigreg/checkpoint_epoch240.pt >> ~/mixbuild_logs/probe_ours.log 2>&1"
set -uo pipefail
E=/scratch/CLUSTER_USER/envs/video_next_prediction
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PATH=$E/bin:$PATH WANDB_DIR=/scratch/CLUSTER_USER/wandb
cd "$HOME/video_crw"
if [ "${1:-}" = --after ]; then
  echo "$(date) waiting for tmux session $2"
  while tmux has-session -t "=$2" 2>/dev/null; do sleep 120; done
  shift 2
fi
ds=$1; ckpt=$2; run=$(basename "$(dirname "$ckpt")")
out=$(dirname "$ckpt")/evals
[ -f "$out/exact_$ds.json" ] && { echo "$(date) skip $run $ds (done)"; exit 0; }
until [ -f "$ckpt" ] && grep -q done /home/CLUSTER_USER/mixbuild_logs/ckpt240_copy.log 2>/dev/null; do
  echo "$(date) waiting for $ckpt"; sleep 120
done
echo "$(date) start $run $ds"
"$E/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/natural/probe_vjepa.py \
  --dataset "$ds" --family nlp --checkpoint "$ckpt" --out "$out" --run "$run" --frames 8 \
  --train-segments 1 --epochs 20 --workers 12 > "/home/CLUSTER_USER/mixbuild_logs/probe_${run}_${ds}.log" 2>&1
echo "$(date) $run $ds exited $?"
