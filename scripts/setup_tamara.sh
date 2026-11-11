#!/usr/bin/env bash
# Run on Tamara after syncing the project. Keep existing Hugging Face credentials.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
source /home/CLUSTER_USER/miniconda/etc/profile.d/conda.sh
if [ ! -x /home/CLUSTER_USER/miniconda/envs/crw/bin/python ]; then
  mamba create -y -n crw -c conda-forge python=3.12 pip
fi
conda activate crw
python -m pip install -r "$repo_dir/requirements-crw.txt"
export HF_HUB_CACHE=/scratch/CLUSTER_USER/datasets/huggingface/hub
export HF_XET_CACHE=/scratch/CLUSTER_USER/datasets/huggingface/xet
python -m pip freeze > "$repo_dir/environment-crw.freeze.txt"
