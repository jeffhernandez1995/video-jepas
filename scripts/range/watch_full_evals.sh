#!/usr/bin/env bash
# Submit full-protocol cached evaluations (complete IN1K/K400 train sets, LR
# chosen on held-out train items) for every milestone checkpoint of the given
# runs. Waits for the full caches first. Run on login2 under tmux:
#   tmux new -d -s fullevals "FAMILY=nlp bash scripts/range/watch_full_evals.sh nlpA_k710_vitt nlpA_k710_vits nlpA_k710_vitb nlpB_k710_vitt nlpB_k710_vits nlpB_k710_vitb >> /scratch/CLUSTER_USER/logs/watch_full_evals_nlp.log 2>&1"
#   tmux new -d -s fullevals_a2 "FAMILY=nlp bash scripts/range/watch_full_evals.sh nlpA2_k710_vits >> /scratch/CLUSTER_USER/logs/watch_full_evals_a2.log 2>&1"
#   tmux new -d -s fullevals_m "FAMILY=nlp bash scripts/range/watch_full_evals.sh nlpBm_k710_vitt nlpAm_k710_vitt nlpBm0_k710_vitt >> /scratch/CLUSTER_USER/logs/watch_full_evals_m.log 2>&1"
#   tmux new -d -s fullevals_lv "FAMILY=levjepa bash scripts/range/watch_full_evals.sh levjepa_k710_vitt levjepa_k710_vits levjepa_k710_vitb >> /scratch/CLUSTER_USER/logs/watch_full_evals_levjepa.log 2>&1"
# nlp: student and teacher encoders of checkpoint_epochNNN.pt, results in <run>/evals.
# levjepa: raw training weights of epoch-(NNN-1).ckpt (never the no-warm-up EMA), results in <run>_raw/evals.
# One .submitted sentinel per (run, epoch, weights) makes restarts safe.
set -uo pipefail
cd /home/CLUSTER_USER/video_crw
FAMILY=${FAMILY:-nlp}
read -r -a MILESTONES <<< "${MILESTONES:-60 120 180 240}"
if [ "$FAMILY" = nlp ]; then read -r -a WEIGHTS <<< "${WEIGHTS:-student teacher}"; else WEIGHTS=(raw); fi
C=/scratch/CLUSTER_USER/datasets/eval_caches
until [ -f "$C/k400_train_full.json" ] && [ -f "$C/in1k_train_full.json" ]; do
  echo "$(date -u +%FT%TZ) waiting for full caches"; sleep 600
done
deadline=$(( $(date +%s) + 12*86400 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  pending=0
  for run in "$@"; do
    src=/scratch/CLUSTER_USER/runs/$run
    out=$src; [ "$FAMILY" = levjepa ] && out=${src}_raw
    mkdir -p "$out/evals"
    for ep in "${MILESTONES[@]}"; do
      for w in "${WEIGHTS[@]}"; do
        tag=$(printf 'e%03d_full_%s' "$ep" "$w")
        [ -f "$out/evals/$tag.submitted" ] && continue
        pending=$((pending+1))
        if [ "$FAMILY" = nlp ]; then
          ckpt=$src/$(printf 'checkpoint_epoch%03d.pt' "$ep")
          [ -n "$(find "$ckpt" -mmin +3 2>/dev/null)" ] || continue
        else
          ckpt=$(find "$src/spt" -name "$(printf 'epoch-%04d' $((ep-1)))*.ckpt" 2>/dev/null | head -1)
          [ -n "$ckpt" ] || continue
        fi
        limit=06:00:00; [[ $run == *vitl* ]] && limit=12:00:00   # ViT-L K400 extraction alone takes ~3 h
        jid=$(sbatch --parsable -J "evalfull_${run}_$tag" --exclude=bg5u16g1 --cpus-per-task=32 --mem=500G -t $limit \
              --export=ALL,EVAL_WEIGHTS="$w",CKPT="$ckpt",OUT="$out",RUN="$run",EPOCH="$ep",FAMILY="$FAMILY",EXTRA="--full" \
              slurm/eval_checkpoint.sbatch) \
          && { echo "$(date -u +%FT%TZ) $run $tag -> job $jid ($ckpt)"; echo "$jid $ckpt" > "$out/evals/$tag.submitted"; }
      done
    done
  done
  [ "$pending" -eq 0 ] && { echo "$(date -u +%FT%TZ) all evaluations submitted"; exit 0; }
  sleep 300
done
echo "watcher deadline reached"
