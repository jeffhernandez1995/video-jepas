#!/usr/bin/env bash
# Intuitive-physics linear probes (scripts/natural/eval_physics.py: IntPhys 2, InfLevel-Lab) on the six reference encoders,
# one GPU each, in parallel; results in /scratch/CLUSTER_USER/runs/ref_<family>/physics_probe.json.
set -uo pipefail
E=/scratch/CLUSTER_USER/envs/video_next_prediction
M=/scratch/CLUSTER_USER/models
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$HOME/video_crw"
g=0
for spec in "vjepa $M/vjepa_vitl16/vitl16.pth.tar" "vjepa2 $M/vjepa2_vitl/vitl.pt" \
            "levjepa_hf $M/levjepa_videomix_large/model.safetensors" "videomae $M/videomae_large/model.safetensors" \
            "videomaev2 $M/videomaev2_large/model.safetensors" "dinov2 $M/dinov2_large/model.safetensors"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$g "$E/bin/python" scripts/natural/eval_physics.py --family "$1" --checkpoint "$2" \
    --out "/scratch/CLUSTER_USER/runs/ref_$1" > "/home/CLUSTER_USER/mixbuild_logs/physics_$1.log" 2>&1 &
  g=$((g + 1))
done
wait
echo "$(date) all physics probes finished"
