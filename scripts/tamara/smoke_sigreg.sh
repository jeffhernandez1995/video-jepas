#!/usr/bin/env bash
# Smoke test of the SIGReg guard: ViT-B arm B on 2 GPUs (exercises the cross-rank ECF
# average), small batch so it can share GPUs: 200-step calibration, then 300 guarded steps.
set -uo pipefail
E=/scratch/CLUSTER_USER/envs/video_next_prediction
OUT=/scratch/CLUSTER_USER/runs/smoke_sigreg_vitb
export CUDA_VISIBLE_DEVICES=${DEVICES:-0,1} LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PATH=$E/bin:$PATH
export LE_VJEPA_INDEX_CACHE=/scratch/CLUSTER_USER/cache/lance_index
cd "$HOME/video_crw"
rm -rf "$OUT"; mkdir -p "$OUT"
RUN="$E/bin/python -m torch.distributed.run --standalone --master-port ${PORT:-29511} --nproc_per_node=2 scripts/natural/train_nlp.py --size vit_base --arm B --mix scripts/natural/mixes/mix256/k710full.json --workers 6 --batch 32"
SET='mask_ratio=0.9 guard="sigreg"'
$RUN --out "$OUT/calibration" --calibrate-steps 200 --no-wandb --set $SET || { echo "calibration failed"; exit 1; }
cat "$OUT/calibration/calibration.json"
$RUN --out "$OUT" --gamma-file "$OUT/calibration/calibration.json" --smoke-steps 300 --no-wandb --set $SET
echo "$(date) smoke exited with $?"
