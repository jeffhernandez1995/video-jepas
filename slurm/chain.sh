#!/usr/bin/env bash
# Submit <count> copies of an sbatch script, each starting after the previous
# one ends in any state (afterany), so a training run outlives RANGE's
# 24-hour job limit. The script must resume from its own last checkpoint and
# exit immediately when it finds its finished sentinel.
#   bash slurm/chain.sh 3 slurm/nlp_chained.sbatch --export=ALL,SIZE=vit_tiny,ARM=B,OUT=/scratch/CLUSTER_USER/runs/nlpB_k710_vitt
set -euo pipefail
count=$1; shift
script=$1; shift
prev=""
for i in $(seq 1 "$count"); do
  if [ -z "$prev" ]; then
    jid=$(sbatch --parsable "$@" "$script")
  else
    jid=$(sbatch --parsable --dependency=afterany:"$prev" "$@" "$script")
  fi
  echo "link $i: job $jid${prev:+ after $prev}"
  prev=$jid
done
