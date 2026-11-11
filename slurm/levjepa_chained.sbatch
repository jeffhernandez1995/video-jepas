#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J levjepa
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=26
#SBATCH --mem-per-cpu=9G
#SBATCH --gres=gpu:h200:8
#SBATCH -t 24:00:00
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# One 24-hour link of a chained LeVJEPA pretraining run (their trainer,
# their recipe, our K710 Lance store), following their own Slurm script:
# one srun task per GPU, Lightning DDP. Chain with slurm/chain.sh.
#
# --export variables:
#   OUT     run directory on scratch (hydra run dir; checkpoints/last.ckpt)
#   MODEL   vit_small | vit_base | vit_large   (default vit_base)
#   DATA    hydra data config (default k710)
#   EPOCHS  default 240, LeVJEPA's epoch-matched protocol on 20% K710
#   EXTRA   further hydra overrides
# Memory must be requested explicitly: the partition default is 1 GB per CPU,
# which OOM-killed the loader workers of a 2-GPU job at 52 GB. 9 GB per CPU is
# the node's share (2 TB / 208 CPUs); --exclusive used to hide this.
# GPU count is set on the sbatch command line and must match in both flags,
# e.g. --ntasks-per-node=2 --gres=gpu:h200:2 (defaults above: 8). Each task
# gets 26 CPUs, the per-GPU share of a node. Not --exclusive, so 2- and 4-GPU
# jobs can start on partly used nodes.
# Effective batch is held at 3072 = nodes * gpus * 96 * accumulate.
# Checkpoints every 540 optimizer steps = 15 epochs of the 20% K710 store
# (36 steps per epoch at batch 3072), so epochs 60/120/180/240 land exactly
# on a kept checkpoint for evaluation; save_top_k=-1 keeps them all.
# Where checkpoints go: stable-pretraining ignores checkpoint.dirpath and
# redirects every ModelCheckpoint into <SPT_CACHE_DIR>/runs/<date>/<time>/<id>/
# checkpoints, defaulting to ~/.cache in HOME (50 GB quota). Each trunk
# therefore gets its own cache at $OUT/spt, and a chain link resumes from the
# newest last*.ckpt anywhere under it (spt's own auto-resume is keyed by Slurm
# job id, which changes per link). Milestones are epoch-0059/0119/0179/0239.
# Lightning's SLURM plugin would requeue on SIGUSR1, which conflicts with the
# chain, so no --signal here; checkpoints land every CKPT_EVERY optimizer steps
# (Lightning forbids combining that with a per-epoch interval).
set -euo pipefail
cd /home/CLUSTER_USER/video_crw/third_party/LeVJEPA
export UV_PROJECT_ENVIRONMENT=/scratch/CLUSTER_USER/conda-envs/levjepa-venv
export LEVJEPA_DATA_ROOT=/scratch/CLUSTER_USER/datasets
export WANDB_DIR=/scratch/CLUSTER_USER/wandb OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${OUT:?set OUT}"
MODEL=${MODEL:-vit_base}; DATA=${DATA:-k710}; EPOCHS=${EPOCHS:-240}
GPUS=${SLURM_NTASKS_PER_NODE:-8}
ACCUM=$(( 3072 / (SLURM_JOB_NUM_NODES * GPUS * 96) ))
CKPT_EVERY=${CKPT_EVERY:-540}
# W&B run name and id are the run directory's basename, so every chained
# link resumes the same run (resume=allow) instead of opening a new one.
RUN=$(basename "$OUT")
# Warm the page cache: training reads 16 random rows per clip from a 505 GB
# Lance store, and VAST serves cold random reads slowly (38-77 clips/s with
# three trunks sharing the store, independent of worker count). A sequential
# read in the background pulls the store into this node's page cache, charged
# to this job's memory, which is why 9 GB per CPU is requested (2-GPU jobs
# should be submitted with --mem-per-cpu=12G so the whole store fits).
STORE=/scratch/CLUSTER_USER/datasets/k710/train_20pct.lance
[ "${DATA:-k710}" = k710 ] && ( find "$STORE" -type f -print0 | xargs -0 -P 6 -n 4 cat > /dev/null; echo "page cache warm $(date +%T)" ) &
mkdir -p "$OUT/spt" /scratch/CLUSTER_USER/wandb
export SPT_CACHE_DIR="$OUT/spt"
if [ -f "$OUT/training_finished" ]; then echo "already finished: $OUT"; exit 0; fi
LAST=$(find "$OUT/spt" -name 'last*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
resume=""; [ -n "$LAST" ] && { echo "resuming from $LAST"; resume="resume.ckpt_path=$LAST resume.weights_only=false"; }
srun "$UV_PROJECT_ENVIRONMENT/bin/python" main.py \
  data="$DATA" model.name="$MODEL" \
  hydra.run.dir="$OUT" checkpoint.dirpath="$OUT/checkpoints" \
  checkpoint.every_n_train_steps="$CKPT_EVERY" checkpoint.every_n_epochs=0 \
  trainer.devices="$GPUS" trainer.num_nodes="$SLURM_JOB_NUM_NODES" \
  trainer.max_epochs="$EPOCHS" accumulate_grad_batches="$ACCUM" \
  loader.clips_per_video=1 loader.num_workers=20 \
  wandb.enabled=true wandb.config.project=video_next_prediction \
  +wandb.config.entity=jeffhernandez1995 wandb.config.name="$RUN" \
  +wandb.config.id="$RUN" +wandb.config.resume=allow \
  +wandb.config.save_dir=/scratch/CLUSTER_USER/wandb \
  "+wandb.config.tags=[natural-video,levjepa,$DATA,$MODEL]" \
  $resume ${EXTRA:-} && touch "$OUT/training_finished"
