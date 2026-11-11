#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J kinetics_dl
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH -t 24:00:00
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# Download the Kinetics-600 and Kinetics-700-2020 train/val archives from the
# CVDF mirror (the same lists Tamara's kinetics-downloader used), then extract.
# Run it on a LOGIN NODE under tmux: RANGE reaps jobs that hold no GPU, so the
# SBATCH header above is kept only for a GPU-backed rerun if ever needed.
#   ssh CLUSTER_USER@login2.RANGE_HOST   # pin one login node: tmux sessions are per host
#   tmux new -d -s kdl "bash scripts/range/download_kinetics.sbatch > /scratch/CLUSTER_USER/logs/kinetics_dl_login.log 2>&1"
# wget -c resumes and every finished stage leaves a sentinel, so rerunning
# after an interruption only does the remaining work.
set -euo pipefail
ROOT=/scratch/CLUSTER_USER/datasets/kinetics_targz
OUT=/scratch/CLUSTER_USER/datasets
mkdir -p "$ROOT" "$OUT"
fetch_list() {  # fetch_list <url of path list> <dest dir>
  local list=$1 dest=$2
  mkdir -p "$dest"
  [ -f "$dest/.complete" ] && { echo "skip $dest"; return; }
  wget -q -O "$dest/paths.txt" "$list"
  # Class names contain spaces and apostrophes ("acting in play", "...o'..."),
  # so split on newlines only; xargs' default quote parsing aborts otherwise.
  xargs -d '\n' -a "$dest/paths.txt" -P 4 -n 1 wget -c -q -P "$dest"
  # Every listed archive present and non-empty.
  while read -r u; do f="$dest/$(basename "$u")"; [ -s "$f" ] || { echo "missing $f"; exit 1; }; done < "$dest/paths.txt"
  touch "$dest/.complete"
}
fetch_list https://s3.amazonaws.com/kinetics/600/train/k600_train_path.txt "$ROOT/k600/train"
fetch_list https://s3.amazonaws.com/kinetics/600/val/k600_val_path.txt   "$ROOT/k600/val"
fetch_list https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt "$ROOT/k700_2020/train"
fetch_list https://s3.amazonaws.com/kinetics/700_2020/val/k700_2020_val_path.txt     "$ROOT/k700_2020/val"
mkdir -p "$ROOT/k600/annotations" "$ROOT/k700_2020/annotations"
wget -c -q -P "$ROOT/k600/annotations" https://s3.amazonaws.com/kinetics/600/annotations/train.csv https://s3.amazonaws.com/kinetics/600/annotations/val.csv
wget -c -q -P "$ROOT/k700_2020/annotations" https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv

extract() {  # extract <archive dir> <video dir>
  local src=$1 dst=$2
  mkdir -p "$dst"
  [ -f "$dst/.complete" ] && { echo "skip extract $dst"; return; }
  for f in "$src"/*.tar.gz; do
    m="$dst/.done_$(basename "$f")"
    [ -f "$m" ] && continue
    tar -xzf "$f" -C "$dst" && touch "$m"
  done
  touch "$dst/.complete"
}
extract "$ROOT/k600/train" "$OUT/k600/train"
extract "$ROOT/k600/val"   "$OUT/k600/val"
extract "$ROOT/k700_2020/train" "$OUT/k700_2020/train"
extract "$ROOT/k700_2020/val"   "$OUT/k700_2020/val"
cp -n "$ROOT/k600/annotations/"*.csv "$OUT/k600/" ; cp -n "$ROOT/k700_2020/annotations/"*.csv "$OUT/k700_2020/"
echo "train mp4s: k600 $(find "$OUT/k600/train" -name '*.mp4' | wc -l)  k700 $(find "$OUT/k700_2020/train" -name '*.mp4' | wc -l)"
touch "$OUT/.kinetics_download_finished"
