#!/usr/bin/env bash
# One-shot status check for a chained run on RANGE: Slurm state, the newest
# log's progress or errors, and whether the W&B run has logged any loss yet.
#   bash scripts/range/check_run.sh <job name> <wandb run id>
# Prints a final line LOSSES_LOGGED=yes|no so a scheduler can stop polling.
set -uo pipefail
NAME=$1; RUN=$2
echo "== $(date -u +%FT%TZ) job name=$NAME wandb=$RUN"
squeue -u CLUSTER_USER -n "$NAME" -h -o "queue: %i %T elapsed=%M reason/node=%R start=%S" || true
sacct -u CLUSTER_USER --name "$NAME" -S now-3days -X -n -o JobID,State,Elapsed,ExitCode | tail -3 | sed 's/^/sacct: /'
LOG=$(ls -t /scratch/CLUSTER_USER/logs/${NAME}_*.out 2>/dev/null | head -1)
if [ -n "${LOG:-}" ]; then
  echo "log: $LOG"
  grep -E "Traceback|Error|error:|CANCELLED|Killed|OOM|busy or unavailable" "$LOG" | sort | uniq -c | sort -rn | head -4 | cut -c1-200
  grep -E "loss|Epoch|step" "$LOG" | tail -2 | cut -c1-200
fi
/scratch/CLUSTER_USER/conda-envs/levjepa-venv/bin/python - "$RUN" <<'EOF'
import sys, wandb
try:
    r = wandb.Api(timeout=30).run(f"jeffhernandez1995/video_next_prediction/{sys.argv[1]}")
    losses = {k: v for k, v in r.summary._json_dict.items() if "loss" in k.lower() and isinstance(v, (int, float))}
    print("wandb:", r.state, "last step", r.lastHistoryStep, {k: round(v, 4) for k, v in list(losses.items())[:6]})
    print("LOSSES_LOGGED=" + ("yes" if losses else "no"))
except Exception as e:
    print("wandb: run not found yet (", str(e)[:80], ")")
    print("LOSSES_LOGGED=no")
EOF
