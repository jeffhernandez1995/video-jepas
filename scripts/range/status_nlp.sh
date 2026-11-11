#!/usr/bin/env bash
# One line per item for the next-latent-prediction runs and all full-protocol
# evaluations (ours and the LeVJEPA raw re-evaluations). Deterministic order.
cd /home/CLUSTER_USER/video_crw
for run in nlpA_k710_vitt nlpA_k710_vits nlpA_k710_vitb nlpB_k710_vitt nlpB_k710_vits nlpB_k710_vitb nlpA2_k710_vits nlpBm_k710_vitt nlpAm_k710_vitt nlpBm0_k710_vitt nlpBm1_k710_vitt nlpAm1_k710_vitt; do
  dir=/scratch/CLUSTER_USER/runs/$run
  st=$(squeue -u CLUSTER_USER -n "$run" -h -o "%T" | grep -v -x PENDING | head -1)
  [ -z "$st" ] && st=$(squeue -u CLUSTER_USER -n "$run" -h -o "%T" | head -1)
  [ -z "$st" ] && st=$(sacct -u CLUSTER_USER --name "$run" -S now-14days -X -n -o State 2>/dev/null | tail -1 | tr -d ' ')
  f=$(ls -t /scratch/CLUSTER_USER/logs/${run}_*.out 2>/dev/null | head -1)
  ep=$(grep -oE '"epoch": [0-9.]+' "$f" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
  err=$(grep -c -E "Traceback|oom_kill|Killed|busy or unavailable|FloatingPointError" "$f" 2>/dev/null)
  fin=$([ -f "$dir/training_finished.json" ] && echo finished || echo running)
  gam=$([ -f "$dir/calibration/calibration.json" ] && python3 -c "import json; print('gamma=%.3g' % json.load(open('$dir/calibration/calibration.json'))['gamma'])")
  echo "nlp $run state=${st:-none} epoch=${ep:-?} errors=${err:-0} $fin $gam"
done
for d in /scratch/CLUSTER_USER/runs/nlp[AB]*_k710_* /scratch/CLUSTER_USER/runs/levjepa_k710_*_raw; do
  for j in "$d"/evals/e*_full*.json; do
    [ -f "$j" ] && python3 -c "import json,sys; d=json.load(open('$j')); s=d['scores']; print('full', '$(basename "$d")', d['tag'], ' '.join(f'{k}={v}' for k,v in sorted(s.items())))"
  done
  for s in "$d"/evals/e*_full*.submitted; do
    [ -f "$s" ] || continue; tag=$(basename "$s" .submitted); [ -f "$d/evals/$tag.json" ] && continue
    jid=$(cut -d' ' -f1 "$s"); jst=$(squeue -j "$jid" -h -o %T 2>/dev/null); [ -z "$jst" ] && jst=$(sacct -j "$jid" -X -n -o State | tr -d ' ')
    echo "fulljob $(basename "$d") $tag job=$jid state=$jst"
  done
done
echo "watchers $(tmux ls 2>/dev/null | grep -c fullevals)"
