#!/usr/bin/env bash
# Push ImageNet and Kinetics-400 from Tamara to RANGE over the campus network.
# Authentication is the laptop's key via agent forwarding (ssh -A), which
# only lives as long as that session. So the script first opens a persistent
# ControlMaster connection to range while the forwarded agent is available;
# every later ssh/rsync multiplexes over it and never authenticates again.
# Launch from the laptop:
#   ssh -A CLUSTER_USER@TAMARA_HOST "tmux new -d -s xfer 'bash /home/CLUSTER_USER/video_crw/scripts/range/transfer_from_tamara.sh'"
# Resumable: every rsync is per class directory with --partial, so rerunning
# only moves what is missing. Verification at the end compares file counts
# and total bytes per split on both machines.
set -euo pipefail
R=CLUSTER_USER@RANGE_HOST
SOCK=$HOME/.ssh/range_master.sock
SRC=/scratch/CLUSTER_USER/datasets
DST=/scratch/CLUSTER_USER/datasets
PAR=${PAR:-6}   # concurrent rsync streams; keep modest on the shared login node
LOG=/scratch/CLUSTER_USER/transfer_range_logs
mkdir -p "$LOG"
if ! ssh -S "$SOCK" -O check "$R" 2>/dev/null; then
  ssh -M -S "$SOCK" -o ControlPersist=yes -o BatchMode=yes -fN "$R"
fi
SSH="ssh -S $SOCK"
RS="rsync -a --partial -W --no-compress --info=stats0 -e '$SSH'"
exec > >(tee -a "$LOG/console.log") 2>&1

$SSH "$R" "mkdir -p $DST/imagenet/train $DST/imagenet/val $DST/k400/train $DST/k400/val $DST/k400/test $DST/k400/annotations $DST/k400/replacement"

push_split() {  # push_split <relative dir with class subdirs>
  local rel=$1
  echo "== $rel  $(date -u +%FT%TZ)"
  # A single failed class must not abort the whole run (set -e + xargs would);
  # failures are logged, and the second pass below retries anything missing.
  ls "$SRC/$rel" | xargs -P "$PAR" -I{} sh -c "$RS '$SRC/$rel/{}/' '$R:$DST/$rel/{}/' 2>>'$LOG/$(echo "$rel" | tr / _).err'" \
    || echo "   $rel: some classes failed on first pass, see $LOG" | tee -a "$LOG/failures.txt"
}
export SSH

for pass in 1 2; do
  echo "#### pass $pass  $(date -u +%FT%TZ)"
  push_split imagenet/train
  push_split imagenet/val
  # Loader files that sit beside the class folders.
  eval $RS --exclude=train --exclude=val "$SRC/imagenet/" "$R:$DST/imagenet/" || true
  push_split k400/train
  push_split k400/val
  push_split k400/test
  eval $RS "$SRC/k400/annotations/" "$R:$DST/k400/annotations/" || true
  eval $RS "$SRC/k400/replacement/" "$R:$DST/k400/replacement/" || true
  eval $RS "$SRC/k400/readme.md" "$R:$DST/k400/readme.md" || true
done

echo "== verify  $(date -u +%FT%TZ)"
for rel in imagenet/train imagenet/val k400/train k400/val k400/test; do
  local_stat=$(find "$SRC/$rel" -type f -printf '%s\n' | awk '{n++; s+=$1} END {print n" files "s" bytes"}')
  remote_stat=$($SSH "$R" "find $DST/$rel -type f -printf '%s\n' | awk '{n++; s+=\$1} END {print n\" files \"s\" bytes\"}'")
  status=MATCH; [ "$local_stat" = "$remote_stat" ] || status=MISMATCH
  echo "$rel: tamara $local_stat | range $remote_stat | $status" | tee -a "$LOG/verify.txt"
done
echo "== done  $(date -u +%FT%TZ)"
