#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J build_mix
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=300G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 24:00:00
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# Mix shard builder on RANGE (CPU work; RANGE reaps jobs that hold no GPU, so
# it holds one H200). SOURCES is an ordered list; each source is built with
# build_lance_mix.py --claim-all, which shares its claim queue with the Tamara
# loops (scripts/tamara/build_mix_remote.sh), so any number of these jobs and
# loops can run at once. k710 (the 80% of K710 outside the 20% store) and
# egodex (300 GB zips) are RANGE-side sources; the rest are also built remotely.
# in1k and cc12m are image sources (build_lance_images.py, same claim scheme).
#   sbatch --export=ALL,SOURCES="k710 egodex pe_video vidaforge" slurm/build_mix.sbatch
set -euo pipefail
cd /home/CLUSTER_USER/video_crw
source /home/CLUSTER_USER/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/CLUSTER_USER/conda-envs/video_next_prediction
SOURCES=${SOURCES:-"k710 egodex cc12m pe_video vidaforge droid ssv2"}
W=$SLURM_CPUS_PER_TASK
# Sources still downloading list no units, so a single pass can skip them for
# good; repeat passes until every source is complete or the time limit hits.
remaining() {  # remaining <src>: units listed but not built (0 also when raw is incomplete)
  case $1 in
    in1k|cc12m) n=$(python scripts/range/build_lance_images.py --source "$1" --list | wc -l) ;;
    *) n=$(python scripts/range/build_lance_mix.py --source "$1" --list | wc -l) ;;
  esac
  echo $(( n - $(ls /scratch/CLUSTER_USER/datasets/mix/$1/*.stats.json 2>/dev/null | wc -l) ))
}
for pass in $(seq 1 100); do
  for src in $SOURCES; do
    case $src in
      in1k|cc12m) python -u scripts/range/build_lance_images.py --source "$src" --claim-all --workers "$W" ;;
      *) python -u scripts/range/build_lance_mix.py --source "$src" --claim-all --workers "$W" ;;
    esac
  done
  left=0
  for src in $SOURCES; do r=$(remaining "$src"); [ "$r" -gt 0 ] && left=$((left + r)); done
  raw_done=1
  for src in $SOURCES; do
    case $src in k710|in1k) ;; *) [ -e /scratch/CLUSTER_USER/datasets/raw/$src/.complete ] || [ -e /scratch/CLUSTER_USER/datasets/raw/$src/extracted/.complete ] || raw_done=0 ;; esac
  done
  echo "$(date) pass $pass: $left units unbuilt or claimed elsewhere, raw complete=$raw_done"
  [ $raw_done = 1 ] && [ $left -le 0 ] && break
  sleep 300
done
