#!/usr/bin/env bash
# When vftN_k710_vitt finishes and frees GPUs 0-3, move vftS and vftM from 2 GPUs (accum 2) to 4 GPUs
# each (accum 1): request a stop (checkpoint_last.pt at the next 50-step check), wait for each trainer
# to exit, and resume it from its own code snapshot. The resume is exact: global batch 256 and micro
# batch 64 per GPU are unchanged, and WindowSampler cuts the same 256-clip global batches in the same
# order for any GPU count (the batch offset in checkpoint_last.pt carries over); the optimizers keep
# no world-size dependent state.
#   tmux new -d -s vft_widen "bash ~/video_crw/scripts/vft/widen_after_next.sh >> ~/mixbuild_logs/vft_widen.log 2>&1"
set -uo pipefail
R=/scratch/CLUSTER_USER/runs L=$HOME/mixbuild_logs
cd "$HOME/video_crw"
until [ -f $R/vftN_k710_vitt/training_finished.json ]; do sleep 300; done
while tmux has-session -t =vftN 2>/dev/null; do sleep 30; done
echo "$(date +'%F %T') vftN finished; GPUs 0-3: $(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader | wc -l) processes"
while [ "$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do sleep 30; done
widen() {   # $1 name, $2 arm, $3 devices, $4 port
  local out=$R/$1_k710_vitt
  [ -f $out/training_finished.json ] && { echo "$(date +'%F %T') $1 already finished"; return; }
  touch $out/stop_requested
  while tmux has-session -t "=$1" 2>/dev/null; do sleep 15; done
  [ -f $out/stop_requested ] && { echo "$(date +'%F %T') $1 exited without honouring the stop; not relaunching"; return; }
  echo "$(date +'%F %T') $1 stopped at: $(grep -E 'resumed|milestone|^\{"step' $L/$1_k710_vitt.log | tail -1 | cut -c1-120); relaunching on $3"
  tmux new -d -s "$1" "DEVICES=$3 PORT=$4 bash scripts/vft/run_vft.sh $1_k710_vitt $2 >> $L/$1_k710_vitt.log 2>&1"
}
widen vftS same 0,1,4,5 29511 &
widen vftM multi 2,3,6,7 29512 &
wait
echo "$(date +'%F %T') widened"
