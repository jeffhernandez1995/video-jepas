#!/usr/bin/env bash
# Sync the project to the RANGE cluster login node. Code lives in /home/CLUSTER_USER
# (50 GB quota); runs and datasets live under /scratch/CLUSTER_USER and are excluded.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
rsync -az --stats \
  --exclude='.git/' --exclude='.DS_Store' --exclude='__pycache__/' \
  --exclude='/runs/' --exclude='/data/' --exclude='/.venv/' --exclude='/old/' \
  "$repo_dir/" CLUSTER_USER@RANGE_HOST:/home/CLUSTER_USER/video_crw/
