#!/usr/bin/env bash
# One-line-per-item status of the natural-video runs on RANGE, used by the
# event monitor to detect changes (diff of successive snapshots) and by
# humans as a quick overview. Output is deterministic and sorted.
cd /home/CLUSTER_USER/video_crw
for r in vitt vits vitb; do
  run=levjepa_k710_$r
  st=$(squeue -u CLUSTER_USER -n "$run" -h -o "%T" | grep -v -x PENDING | head -1)
  [ -z "$st" ] && st=$(squeue -u CLUSTER_USER -n "$run" -h -o "%T" | head -1)
  [ -z "$st" ] && st=$(sacct -u CLUSTER_USER --name "$run" -S now-7days -X -n -o State | tail -1 | tr -d ' ')
  f=$(ls -t /scratch/CLUSTER_USER/logs/${run}_*.out 2>/dev/null | head -1)
  ep=$(grep -oE "\[Epoch [0-9]+/240\]" "$f" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
  err=$(grep -c -E "Traceback|oom_kill|Killed|busy or unavailable" "$f" 2>/dev/null)
  fin=$([ -f /scratch/CLUSTER_USER/runs/$run/training_finished ] && echo finished || echo running)
  echo "trunk $run state=${st:-none} epoch=${ep:-?} errors=${err:-0} $fin"
  for j in /scratch/CLUSTER_USER/runs/$run/evals/e*.json; do
    [ -f "$j" ] && python3 -c "import json; d=json.load(open('$j')); s=d['scores']; print('eval $run e%03d'%d['pretrain_epoch'], ' '.join(f'{k}={v}' for k,v in sorted(s.items())))"
  done
  for s in /scratch/CLUSTER_USER/runs/$run/evals/e*.submitted; do
    [ -f "$s" ] || continue; tag=$(basename "$s" .submitted); [ -f "/scratch/CLUSTER_USER/runs/$run/evals/$tag.json" ] && continue
    jid=$(cut -d' ' -f1 "$s"); jst=$(squeue -j "$jid" -h -o %T 2>/dev/null); [ -z "$jst" ] && jst=$(sacct -j "$jid" -X -n -o State | tr -d ' ')
    echo "evaljob $run $tag job=$jid state=$jst"
  done
done
echo "watcher $(tmux ls 2>/dev/null | grep -c evalwatch) home_cache=$(du -sh ~/.cache/stable-pretraining 2>/dev/null | cut -f1)"
