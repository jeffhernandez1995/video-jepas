#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
rsync -az --stats \
  --exclude='.git/' --exclude='.DS_Store' --exclude='__pycache__/' \
  --exclude='/runs/' --exclude='/data/' --exclude='/.venv/' --exclude='/old/' \
  "$repo_dir/" CLUSTER_USER@TAMARA_HOST:/home/CLUSTER_USER/video_crw/
