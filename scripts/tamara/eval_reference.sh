#!/usr/bin/env bash
# Our frozen-feature protocol (scripts/natural/eval_checkpoint.py, full datasets) on the
# released reference encoders, to measure how far it sits from the papers' protocol:
# V-JEPA ViT-L/16, V-JEPA 2 ViT-L and LeVJEPA-VideoMix-Large. One (model, dataset) pair
# per A40, at most one per GPU at a time; results in /scratch/CLUSTER_USER/runs/ref_<family>/evals/
# as e000_full_<dataset>.json. Needs the eval caches in /scratch/CLUSTER_USER/datasets/eval_caches.
#   tmux new -d -s evalref "bash ~/video_crw/scripts/tamara/eval_reference.sh >> ~/mixbuild_logs/evalref.log 2>&1"
set -uo pipefail
E=/scratch/CLUSTER_USER/envs/video_next_prediction
M=/scratch/CLUSTER_USER/models
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PATH=$E/bin:$PATH WANDB_DIR=/scratch/CLUSTER_USER/wandb
cd "$HOME/video_crw"
declare -A CKPT=([vjepa]=$M/vjepa_vitl16/vitl16.pth.tar [vjepa2]=$M/vjepa2_vitl/vitl.pt
                 [levjepa_hf]=$M/levjepa_videomix_large/model.safetensors)
jobs=()                                   # IN1K first: the longest extraction (1.28M images)
for ds in in1k k400 ssv2; do for fam in vjepa vjepa2 levjepa_hf; do jobs+=("$fam $ds"); done; done
run() {                                   # $1 = GPU, $2 = "family dataset"
  read -r fam ds <<< "$2"
  out=/scratch/CLUSTER_USER/runs/ref_$fam/evals
  [ -f "$out/e000_full_$ds.json" ] && { echo "$(date +%T) skip $fam $ds (done)"; return; }
  echo "$(date +%T) GPU $1: $fam $ds"
  CUDA_VISIBLE_DEVICES=$1 python scripts/natural/eval_checkpoint.py --checkpoint "${CKPT[$fam]}" \
    --out "$out" --family "$fam" --run "ref_$fam" --epoch 0 --datasets "$ds" --workers 10 --no-wandb \
    > "/home/CLUSTER_USER/mixbuild_logs/evalref_${fam}_$ds.log" 2>&1
  echo "$(date +%T) GPU $1: $fam $ds exited $?"
}
gpu_jobs() { local i=$1; while [ "$i" -lt "${#jobs[@]}" ]; do run "$2" "${jobs[$i]}"; i=$((i + 8)); done; }
for g in 0 1 2 3 4 5 6 7; do gpu_jobs "$g" "$g" & done
wait
echo "$(date +%T) all reference evaluations finished"
