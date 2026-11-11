#!/usr/bin/env bash
# Build mix shards on Tamara's idle CPUs from raw data that lives on RANGE.
# Tamara's scratch is full, so everything is staged in /dev/shm (756 GB):
# claim a shard on RANGE (the same atomic mkdir claim build_lance_mix.py
# --claim-all uses, so RANGE jobs and this loop never build the same shard),
# pull its raw files, build it with scripts/range/build_lance_mix.py, push the
# shard back (stats json last: it marks the shard complete), delete the copy.
#
# Needs the persistent Tamara->RANGE ControlMaster (opened from the laptop
# with agent forwarding, see scripts/range/transfer_from_tamara.sh):
#   ssh -A CLUSTER_USER@TAMARA_HOST 'ssh -M -S ~/.ssh/range_master.sock -o ControlPersist=yes -fN CLUSTER_USER@RANGE_HOST'
# Launch K loops (each with 104/K decode workers):
#   for k in 0 1 2 3; do tmux new -d -s mix$k "WID=$k WORKERS=26 bash ~/video_crw/scripts/tamara/build_mix_remote.sh pe_video vidaforge droid ssv2 >> ~/mixbuild_logs/w$k.log 2>&1"; done
# A source is finished for a loop when RANGE has raw/<source>/.complete (SSv2:
# raw/ssv2/extracted/.complete) and a
# full pass over its units found nothing left to claim.
set -uo pipefail
R=CLUSTER_USER@RANGE_HOST
S=$HOME/.ssh/range_master.sock
SSH="ssh -n -S $S -o BatchMode=yes $R"   # -n: never read the unit list on stdin
RS="rsync -a -W --no-compress -e 'ssh -S $S -o BatchMode=yes'"
PY=/home/CLUSTER_USER/venvs/mixbuild/bin/python
BUILDER=$HOME/video_crw/scripts/range/build_lance_mix.py
RRAW=/scratch/CLUSTER_USER/datasets/raw
ROUT=/scratch/CLUSTER_USER/datasets/mix
RENV='source /home/CLUSTER_USER/miniconda3/etc/profile.d/conda.sh; conda activate /scratch/CLUSTER_USER/conda-envs/video_next_prediction'
WID=${WID:-0}
WORKERS=${WORKERS:-26}
SHM=/dev/shm/mixbuild
RAW=$SHM/raw                   # mirrors RRAW's layout for the files we hold
OUT=$SHM/out_$WID
mkdir -p "$RAW" "$OUT" "$SHM/locks"
# Raw paths shared by many units: pulled once under a lock and kept.
shared() { case $1 in ssv2/*|droid/meta/episodes) return 0 ;; *) return 1 ;; esac; }

pull() {  # pull <relative raw path>
  local f=$1
  if shared "$f"; then
    ( flock 9
      [ -e "$RAW/$f.ok" ] && exit 0
      mkdir -p "$RAW/$(dirname "$f")"
      eval "$RS" "$R:$RRAW/$f" "$RAW/$(dirname "$f")/" && touch "$RAW/$f.ok"
    ) 9>"$SHM/locks/$(echo "$f" | tr / _).lock"
  else
    mkdir -p "$RAW/$(dirname "$f")"
    eval "$RS" "$R:$RRAW/$f" "$RAW/$(dirname "$f")/"
  fi
}

for src in "$@"; do
  while true; do
    marker=$RRAW/$src/.complete; [ "$src" = ssv2 ] && marker=$RRAW/ssv2/extracted/.complete
    complete=$($SSH "test -e $marker && echo yes || echo no")
    list=$($SSH "$RENV; python /home/CLUSTER_USER/video_crw/scripts/range/build_lance_mix.py --source $src --list" 2>/dev/null)
    built=0
    # One listing per pass (per-unit ssh tests cost ~1 s each, 30 min for PE Video).
    skip=$($SSH "cd $ROUT/$src 2>/dev/null && { ls | sed -n 's/\.lance\.stats\.json\$//p'; ls .claims 2>/dev/null; }")
    while IFS=$'\t' read -r -u 3 name files; do   # fd 3: nothing inside can consume the list
      [ -z "$name" ] && continue
      grep -qxF "$name" <<< "$skip" && continue
      $SSH "mkdir -p $ROUT/$src/.claims && mkdir $ROUT/$src/.claims/$name 2>/dev/null && echo 'tamara w$WID '\"\$(date)\" > $ROUT/$src/.claims/$name/owner" || continue
      echo "$(date) w$WID $src/$name"
      ok=1
      for f in ${files//,/ }; do pull "$f" || ok=0; done
      if [ $ok = 1 ] && "$PY" "$BUILDER" --source "$src" --units "$name" --raw "$RAW" --out "$OUT" --workers "$WORKERS" < /dev/null; then
        eval "$RS" "$OUT/$src/$name.lance" "$OUT/$src/$name.lance.prov.parquet" "$R:$ROUT/$src/" &&
          eval "$RS" "$OUT/$src/$name.lance.stats.json" "$R:$ROUT/$src/" && built=$((built + 1)) ||
          { echo "push failed $src/$name"; $SSH "rm -rf $ROUT/$src/.claims/$name"; }
      else
        echo "build failed $src/$name; releasing claim"
        $SSH "rm -rf $ROUT/$src/.claims/$name"
      fi
      rm -rf "$OUT/$src/$name.lance"* "$OUT/$src/.tmp_$name.lance"
      for f in ${files//,/ }; do shared "$f" || rm -f "$RAW/$f"; done
    done 3<<< "$list"
    if [ "$complete" = yes ] && [ $built = 0 ]; then echo "$(date) w$WID $src finished"; break; fi
    [ $built = 0 ] && sleep 300       # raw still downloading: wait for more units
  done
done
echo "$(date) w$WID all sources done"
