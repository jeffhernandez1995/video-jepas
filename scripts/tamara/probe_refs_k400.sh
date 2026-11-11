#!/usr/bin/env bash
# Paper-protocol K400 probes (probe_vjepa.py: 8 segments x 3 crops at test, one clip per
# video in training) on the released reference encoders, token-matched at 8 temporal slots:
# tubelet-2 V-JEPA 2 gets 16 frames, tubelet-1 LeVJEPA 8. Run one after the other on all 8 A40s.
set -uo pipefail
E=/scratch/CLUSTER_USER/envs/video_next_prediction
M=/scratch/CLUSTER_USER/models
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PATH=$E/bin:$PATH WANDB_DIR=/scratch/CLUSTER_USER/wandb
cd "$HOME/video_crw"
probe() {   # family checkpoint frames
  out=/scratch/CLUSTER_USER/runs/ref_$1/evals; tag=exact_k400$([ "$3" != 8 ] && echo "_${3}f")
  [ -f "$out/$tag.json" ] && { echo "$(date) skip $1 (done)"; return; }
  echo "$(date) start $1 k400 $3 frames"
  "$E/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/natural/probe_vjepa.py \
    --dataset k400 --family "$1" --checkpoint "$2" --out "$out" --run "ref_$1" --frames "$3" \
    --train-segments 1 --epochs 20 --workers 12 > "/home/CLUSTER_USER/mixbuild_logs/probe_${1}_k400.log" 2>&1
  echo "$(date) $1 exited $?"
}
probe vjepa2 $M/vjepa2_vitl/vitl.pt 16
probe levjepa_hf $M/levjepa_videomix_large/model.safetensors 8
