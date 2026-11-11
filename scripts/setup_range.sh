#!/usr/bin/env bash
# Run on a RANGE login node after scripts/sync_range.sh.
# The environment lives on /scratch because /home has a 50 GB quota and
# already holds miniconda3. Driver 580 / CUDA 13.0 on the H200 nodes accepts
# the cu128 PyTorch wheels used on Tamara.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
ENV=/scratch/CLUSTER_USER/conda-envs/video_next_prediction
source /home/CLUSTER_USER/miniconda3/etc/profile.d/conda.sh
if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" -c conda-forge python=3.12 pip
fi
conda activate "$ENV"
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0
python -m pip install -r "$repo_dir/requirements-crw.txt"
# Video decoding, the Lance frame store used by LeVJEPA's loader, and logging.
python -m pip install av pylance pyarrow wandb yt-dlp
mkdir -p /scratch/CLUSTER_USER/runs /scratch/CLUSTER_USER/datasets /scratch/CLUSTER_USER/logs
# Keep runs out of the home quota; the trainers write to runs/ relative to the repo.
if [ ! -e "$repo_dir/runs" ]; then ln -s /scratch/CLUSTER_USER/runs "$repo_dir/runs"; fi
export HF_HOME=/scratch/CLUSTER_USER/hf_home
python -m pip freeze > "$repo_dir/environment-range.freeze.txt"
python - <<'EOF'
import torch, av, lance
print("torch", torch.__version__, "cuda", torch.version.cuda, "av", av.__version__, "lance", lance.__version__)
EOF
