#!/usr/bin/env bash
# Submit a frozen evaluation job for every milestone checkpoint a trunk writes.
# The trunk never waits: this only reads checkpoints that already exist.
# Run on login2 under tmux:
#   tmux new -d -s evalwatch "bash scripts/range/watch_milestones.sh levjepa_k710_vitt levjepa_k710_vits levjepa_k710_vitb > /scratch/CLUSTER_USER/logs/watch_milestones.log 2>&1"
# Milestones are pretraining epochs 60/120/180/240. LeVJEPA names checkpoints
# by the zero-based epoch in progress when the step-based save fires, so
# epoch N is epoch-<N-1>.ckpt. A checkpoint is used once it has been
# unmodified for 3 minutes. One .submitted sentinel per (run, epoch) makes
# restarts of this watcher safe. Exits when every run has all evaluations
# submitted, or after 5 days.
set -uo pipefail
cd /home/CLUSTER_USER/video_crw
RUNS=("$@"); MILESTONES=(60 120 180 240); FAMILY=${FAMILY:-levjepa}
deadline=$(( $(date +%s) + 5*86400 ))
mkdir -p /scratch/CLUSTER_USER/logs
while [ "$(date +%s)" -lt "$deadline" ]; do
  pending=0
  for run in "${RUNS[@]}"; do
    out=/scratch/CLUSTER_USER/runs/$run; mkdir -p "$out/evals"
    for ep in "${MILESTONES[@]}"; do
      tag=$(printf e%03d "$ep"); [ -f "$out/evals/$tag.submitted" ] && continue
      pending=$((pending+1))
      name=$(printf 'epoch-%04d' $((ep-1)))
      ckpt=$(find "$out/spt" -name "$name*.ckpt" -mmin +3 -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
      [ -z "$ckpt" ] && continue
      jid=$(sbatch --parsable -J "eval_${run}_$tag" --exclude=bg5u16g1 \
            --export=ALL,CKPT="$ckpt",OUT="$out",RUN="$run",EPOCH="$ep",FAMILY="$FAMILY",EXTRA="--full" slurm/eval_checkpoint.sbatch) \
        && { echo "$(date -u +%FT%TZ) $run $tag -> job $jid ($ckpt)"; echo "$jid $ckpt" > "$out/evals/$tag.submitted"; }
    done
  done
  [ "$pending" -eq 0 ] && { echo "$(date -u +%FT%TZ) all evaluations submitted"; exit 0; }
  sleep 300
done
echo "watcher deadline reached"
