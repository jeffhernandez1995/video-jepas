#!/usr/bin/env bash
#SBATCH -A SLURM_ACCT
#SBATCH -p commons
#SBATCH -J nlp_mn
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=208
#SBATCH --gres=gpu:h200:8
#SBATCH -t 24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH -o /scratch/CLUSTER_USER/logs/%x_%j.out
# Multi-node link of a chained next-latent-prediction run (scripts/natural/train_nlp.py),
# same contract as slurm/nlp_chained.sbatch (SIZE, OUT, ARM, EXTRA; chain with
# slurm/chain.sh), one torchrun agent per node with a c10d rendezvous on the
# first node and InfiniBand on, as in old/slurm/train_multinode.sbatch. -N sets the
# node count (group cap 32 GPUs, so at most -N 4). The trainer shards data by
# global rank and keeps the global batch, so per-GPU batch = 256 / GPUs.
# Gamma calibration (arm B) runs on this allocation too, before training.
set -uo pipefail
cd /home/CLUSTER_USER/video_crw
: "${SIZE:?set SIZE}" "${OUT:?set OUT}"
ARM=${ARM:-B}
PY=/scratch/CLUSTER_USER/conda-envs/levjepa-venv/bin
export OMP_NUM_THREADS=2 WANDB_DIR=/scratch/CLUSTER_USER/wandb PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_IB_DISABLE=0
export LE_VJEPA_INDEX_CACHE=/scratch/CLUSTER_USER/datasets/k710/.lance_index
unset LD_PRELOAD; export XALT_EXECUTABLE_TRACKING=no
[ -f "$OUT/training_finished.json" ] && { echo "already finished: $OUT"; exit 0; }
mkdir -p "$OUT" /scratch/CLUSTER_USER/wandb
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "nodes:"; scontrol show hostnames "$SLURM_JOB_NODELIST" | nl
trap 'echo "$(date +%T) USR1: requesting checkpoint"; touch "$OUT/stop_requested"' USR1
launch() {  # launch <train_nlp args...>
  srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c "
    $PY/torchrun --nnodes=\$SLURM_NNODES --nproc-per-node=8 --node_rank=\$SLURM_NODEID \
      --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_id=$SLURM_JOB_ID-\$RANDOM_TAG \
      scripts/natural/train_nlp.py --size $SIZE --arm $ARM $*" &
  child=$!
  wait "$child"; rc=$?
  while [ "$rc" -gt 128 ] && kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
  return $rc
}
export RANDOM_TAG=cal
GAMMA_ARGS=""
if [ "$ARM" = B ] || [ "${GUARD:-0}" = 1 ]; then
  CAL=$OUT/calibration/calibration.json
  if [ ! -f "$CAL" ]; then
    echo "$(date +%T) calibrating gamma (200 unregularized arm-$ARM steps)"
    rm -rf "$OUT/calibration"
    launch --out "$OUT/calibration" --calibrate-steps 200 --no-wandb ${EXTRA:-} || { echo "calibration failed"; exit 1; }
  fi
  GAMMA_ARGS="--gamma-file $CAL"
  echo "gamma $(python3 -c "import json; print(json.load(open('$CAL'))['gamma'])")"
fi
export RANDOM_TAG=train
launch --out "$OUT" $GAMMA_ARGS ${EXTRA:-}; rc=$?
echo "trainer exited with $rc"
exit "$rc"
