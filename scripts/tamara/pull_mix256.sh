#!/usr/bin/env bash
# Copy the 256-px mix stores (scripts/range/reencode_mix.py) from RANGE to the
# same path on Tamara and consolidate them there, one pass per call:
#   bash ~/video_crw/scripts/tamara/pull_mix256.sh k710_20pct k710 ssv2 egodex droid in1k pe_video
# Only finished shards are copied (stats json present); rerun until RANGE has
# all of a source. Needs the Tamara->RANGE ControlMaster (see build_mix_remote.sh).
set -uo pipefail
R=CLUSTER_USER@RANGE_HOST
S=$HOME/.ssh/range_master.sock
D=/scratch/CLUSTER_USER/datasets/mix256
E=/scratch/CLUSTER_USER/envs/video_next_prediction
mkdir -p "$D"
for src in "$@"; do
  mkdir -p "$D/$src"
  # names of finished shards on RANGE, then copy each shard dir + sidecars (stats last)
  names=$(ssh -n -S "$S" "$R" "cd $D/$src 2>/dev/null && ls | sed -n 's/\.lance\.stats\.json\$//p'")
  for n in $names; do
    [ -e "$D/$src/$n.lance.stats.json" ] && continue
    echo "$n"
  done | xargs -P 6 -I{} sh -c "rsync -a -W --no-compress -e 'ssh -S $S' $R:$D/$src/{}.lance $R:$D/$src/{}.lance.prov.parquet $R:$D/$src/{}.lance.index.npz $D/$src/ 2>/dev/null; rsync -a -e 'ssh -S $S' $R:$D/$src/{}.lance.stats.json $D/$src/"
  echo "$(date) $src: $(ls $D/$src/*.stats.json 2>/dev/null | wc -l) shards on Tamara"
  [ "$src" = k710_20pct ] && continue        # a plain store, read directly (no consolidation)
  MIX_ROOT=$D LD_LIBRARY_PATH=$E/lib "$E/bin/python" "$HOME/video_crw/scripts/range/consolidate_mix.py" "$src"
done
