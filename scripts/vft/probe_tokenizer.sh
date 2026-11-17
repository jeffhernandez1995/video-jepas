#!/usr/bin/env bash
# V-JEPA paper-protocol probe (DS=k400 default, or ssv2) (probe_vjepa.py, all 8 A40s) of VideoFlexTok itself
# (EPFL-VILAB/videoflextok_d18_d18_k600, the VFT ViT-T's target tokenizer) as a reference encoder:
# --family videoflextok (scripts/natural/eval_checkpoint.py), 9 frames at 7.5 fps (the tokenizer takes
# 1 + 4k frames; 9 -> 3 groups, 1536 tokens of width 1152, vs 8 x 196 for our 8-frame ViTs), 1 train
# segment, 8 x 3 test views (SSv2 2 x 3), 20 epochs, attentive + mean-pooled linear. Runs with the videoflextok venv,
# after the tmux session given with --after (default vft_probes) has ended. Skips when done.
#   tmux new -d -s probe_vfttok "bash ~/video_crw/scripts/vft/probe_tokenizer.sh >> ~/mixbuild_logs/probe_vfttok.log 2>&1"
set -uo pipefail
E=/scratch/CLUSTER_USER/envs/video_next_prediction
PY=/scratch/CLUSTER_USER/envs/videoflextok/bin/python
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PATH=$E/bin:$PATH WANDB_DIR=/scratch/CLUSTER_USER/wandb
export HF_HUB_OFFLINE=1
cd "$HOME/video_crw"
DS=${DS:-k400}
AFTER=vft_probes
[ "${1:-}" = --after ] && { AFTER=$2; shift 2; }
while tmux has-session -t "=$AFTER" 2>/dev/null; do sleep 300; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
out=/scratch/CLUSTER_USER/runs/ref_videoflextok/evals
[ -f "$out/exact_${DS}_9f.json" ] && { echo "$(date) skip videoflextok $DS (done)"; exit 0; }
echo "$(date) start videoflextok $DS 9 frames"
"$PY" -W ignore -m torch.distributed.run --standalone --nproc_per_node=8 scripts/natural/probe_vjepa.py \
  --dataset "$DS" --family videoflextok --checkpoint /scratch/CLUSTER_USER/models/videoflextok/videoflextok_d18_d18_k600 \
  --out "$out" --run ref_videoflextok --frames 9 --train-segments 1 --epochs 20 --workers 12 \
  > /home/CLUSTER_USER/mixbuild_logs/probe_videoflextok_$DS.log 2>&1
echo "$(date) videoflextok $DS exited $?"
