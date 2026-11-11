#!/usr/bin/env bash
# Parallel replacement for the sequential extraction stage of
# download_kinetics.sbatch: 8 concurrent tar streams per split, per-archive
# .done markers so it is resumable, and val archives are marked complete
# without extracting (exclusion lists come from the CSVs, not from files).
# Run on login2 under tmux:
#   tmux new -d -s kx "bash scripts/range/extract_kinetics_parallel.sh > /scratch/CLUSTER_USER/logs/extract_kinetics.log 2>&1"
set -euo pipefail
ROOT=/scratch/CLUSTER_USER/datasets/kinetics_targz
OUT=/scratch/CLUSTER_USER/datasets
PAR=${PAR:-8}
extract_split() {  # extract_split <archive dir> <video dir>
  local src=$1 dst=$2
  mkdir -p "$dst"
  [ -f "$dst/.complete" ] && { echo "skip $dst"; return; }
  echo "== $dst $(date -u +%FT%TZ): $(ls "$src"/*.tar.gz | wc -l) archives, $(ls -a "$dst" | grep -c '^\.done_' || true) done"
  ls "$src"/*.tar.gz | xargs -d '\n' -P "$PAR" -I{} sh -c '
    f="{}"; m="'"$dst"'/.done_$(basename "$f")"
    [ -f "$m" ] && exit 0
    tar -xzf "$f" -C "'"$dst"'" && touch "$m" || echo "FAILED $f"'
  local n_arch n_done
  n_arch=$(ls "$src"/*.tar.gz | wc -l); n_done=$(ls -a "$dst" | grep -c '^\.done_' || true)
  echo "   $n_done / $n_arch extracted $(date -u +%FT%TZ)"
  [ "$n_done" -eq "$n_arch" ] && touch "$dst/.complete"
}
extract_split "$ROOT/k600/train" "$OUT/k600/train"
extract_split "$ROOT/k700_2020/train" "$OUT/k700_2020/train"
for v in k600/val k700_2020/val; do mkdir -p "$OUT/$v"; touch "$OUT/$v/.complete" "$OUT/$v/.skipped_not_extracted"; done
cp -n "$ROOT/k600/annotations/"*.csv "$OUT/k600/" 2>/dev/null || true
cp -n "$ROOT/k700_2020/annotations/"*.csv "$OUT/k700_2020/" 2>/dev/null || true
if [ -f "$OUT/k600/train/.complete" ] && [ -f "$OUT/k700_2020/train/.complete" ]; then
  echo "train mp4s: k600 $(find "$OUT/k600/train" -maxdepth 1 -name '*.mp4' | wc -l)  k700 $(find "$OUT/k700_2020/train" -maxdepth 1 -name '*.mp4' | wc -l)"
  touch "$OUT/.kinetics_download_finished"
  echo "sentinel written $(date -u +%FT%TZ)"
else
  echo "some archives failed; rerun to retry"; exit 1
fi
