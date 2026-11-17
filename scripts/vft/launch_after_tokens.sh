#!/usr/bin/env bash
# Waits for vft_tokenize.py to finish all 8 ranks, checks the shard count, writes the train/val copy
# diagnostic, then starts the three VideoFlexTok-target ViT-T arms (64,000-way id loss, K = 16):
#   vftN_k710_vitt next  GPUs 0-3 | vftS_k710_vitt same GPUs 4,5 (accum 2) | vftM_k710_vitt multi GPUs 6,7 (accum 2)
#   tmux new -d -s vft_launch "bash ~/video_crw/scripts/vft/launch_after_tokens.sh >> ~/mixbuild_logs/vft_launch.log 2>&1"
set -uo pipefail
O=/scratch/CLUSTER_USER/datasets/vft_tokens
E=/scratch/CLUSTER_USER/envs/video_next_prediction
L=$HOME/mixbuild_logs
until grep -q "all ranks finished" $O/run_all.log 2>/dev/null; do sleep 300; done
n=$(ls $O/train/train_*.npz | grep -vc tmp)
echo "$(date) tokenization finished: $n train shards"
[ "$n" -eq 222 ] || { echo "expected 222 train shards, found $n: not launching"; exit 1; }
cd $HOME/video_crw
LD_LIBRARY_PATH=$E/lib $E/bin/python scripts/vft/diag_tokens.py --windows 50000 --K 16 > $O/diag_tokens_K16.txt 2>&1
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
tmux new -d -s vftN "DEVICES=0,1,2,3 PORT=29510 bash scripts/vft/run_vft.sh vftN_k710_vitt next >> $L/vftN_k710_vitt.log 2>&1"
tmux new -d -s vftS "DEVICES=4,5 PORT=29511 EXTRA='--accum 2' bash scripts/vft/run_vft.sh vftS_k710_vitt same >> $L/vftS_k710_vitt.log 2>&1"
tmux new -d -s vftM "DEVICES=6,7 PORT=29512 EXTRA='--accum 2' bash scripts/vft/run_vft.sh vftM_k710_vitt multi >> $L/vftM_k710_vitt.log 2>&1"
echo "$(date) launched vftN, vftS, vftM"
