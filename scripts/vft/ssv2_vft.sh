#!/usr/bin/env bash
# SSv2 paper-protocol probes (probe_ours.sh ssv2: 8 frames, 1 train segment, 2 x 3 test views, 20 epochs,
# student e240, all 8 GPUs, the setup of the Q / Q-frame references), one after the other:
# next and same-time first, then multi, then VideoFlexTok's own encoder (9 frames) as the upper bound.
#   tmux new -d -s vft_ssv2 "bash ~/video_crw/scripts/vft/ssv2_vft.sh >> ~/mixbuild_logs/vft_ssv2.log 2>&1"
set -uo pipefail
cd "$HOME/video_crw"
for r in vftN vftS vftM; do bash scripts/tamara/probe_ours.sh ssv2 /scratch/CLUSTER_USER/runs/${r}_k710_vitt/checkpoint_epoch240.pt; done
DS=ssv2 bash scripts/vft/probe_tokenizer.sh --after none
echo "$(date) vft ssv2 probes finished"
