# RANGE cluster: access, layout, policies, chained jobs

Set up 2026-09-16. RANGE is the university's Slurm cluster with H200 nodes; Tamara
stays the interactive 8×A40 box and the permanent home of every dataset.

## Access

```bash
ssh CLUSTER_USER@RANGE_HOST                       # round-robins between login2 and login3
ssh -J CLUSTER_USER@RANGE_HOST CLUSTER_USER@login2       # pin a host; login2/login3 resolve only inside
```

tmux sessions live on one node, so anything left running under tmux
(downloads, transfers, Lance builds) is started on `login2` through the
ProxyJump form above and found again the same way.

Same key as Tamara from the laptop. Tamara-to-range transfers authenticate
with the laptop's key through agent forwarding (`ssh -A`), the same way the
earlier Geo-Thought copy was done; neither server holds a key for the
other. Because the forwarded agent disappears when that session ends,
`scripts/range/transfer_from_tamara.sh` first opens a persistent
ControlMaster connection (`~/.ssh/range_master.sock` on Tamara) while the
agent is available, and every rsync stream multiplexes over it afterwards.
Range's host key was verified against the laptop's known_hosts before
being trusted on Tamara (2026-09-16). To reopen the master by hand:

```bash
ssh -A CLUSTER_USER@TAMARA_HOST 'ssh -M -S ~/.ssh/range_master.sock -o ControlPersist=yes -fN CLUSTER_USER@RANGE_HOST'
```

Sync code and build the environment exactly as for Tamara:

```bash
bash scripts/sync_range.sh                 # laptop -> range:/home/CLUSTER_USER/video_crw
ssh CLUSTER_USER@RANGE_HOST
cd /home/CLUSTER_USER/video_crw && bash scripts/setup_range.sh
```

`setup_range.sh` creates `/scratch/CLUSTER_USER/conda-envs/video_next_prediction`
(Python 3.12, torch 2.11 cu128, torchvision 0.26, PyAV, Lance, wandb) and
symlinks `runs/` to `/scratch/CLUSTER_USER/runs`. Home has a 50 GB quota with
miniconda3 already in it, so nothing large goes there.

## Layout

| Path | Purpose | Quota | Policy |
| --- | --- | --- | --- |
| `/home/CLUSTER_USER/video_crw` | code only | 50 GB home | backed up, slow, never job I/O |
| `/scratch/CLUSTER_USER/datasets` | staged datasets | none | **purged after 14 days without access**; may be purged earlier when full |
| `/scratch/CLUSTER_USER/runs`, `/scratch/CLUSTER_USER/logs` | run outputs, Slurm logs | none | same purge |
| `/scratch/CLUSTER_USER/conda-envs` | environments | none | same purge; rebuild with `setup_range.sh` |
| `/tmp` on compute nodes | 872 GB local NVMe | none | wiped at job end |

The 14-day purge is the fact that shapes everything. Rules:

- Tamara holds the master copy of every dataset. Range scratch is a cache.
  Re-stage with `scripts/range/transfer_from_tamara.sh` (resumable, verifies
  counts) whenever something is missing.
- Anything produced on range that must survive (Lance stores built from
  the Kinetics downloads, final checkpoints, probe results) gets copied back
  to Tamara by the job that produced it. Raw Kinetics archives on range are
  deleted once their Lance store exists and is mirrored.
- Checkpoints under `/scratch/CLUSTER_USER/runs` are touched by every chained job
  link, so an active run is never at risk; a paused run older than two
  weeks is.
- `cp`, not `mv`, when reorganizing on scratch, so access times refresh.

## Slurm

- Account `SLURM_ACCT`, partition `commons`, QOS `normal`. Every script needs
  `-A SLURM_ACCT -p commons`.
- Max walltime **24 hours**. No per-user job-count limit found;
  `MaxJobCount` 10,000 cluster-wide.
- Group limit **32 concurrent GPUs** across all of `SLURM_ACCT`'s running jobs.
  That is four H200 nodes at once, shared with whoever else is in the
  group. Check with `squeue -A SLURM_ACCT`.
- Nodes: 8 × (8 H200 143 GB, 208 CPUs, 2 TB RAM); 2 × (4 H100); 2 mixed
  nodes with 24/47 GB cards. Request with `--gres=gpu:h200:8`.
- Driver 580, CUDA 13.0 reported by `nvidia-smi`; cu128 wheels run. The
  `module`/`ml` system exists in login shells (`bash -lc`) with CUDA
  12.4 to 12.9 and NCCL 2.26 modules, but our environment ships its own
  CUDA libraries through pip, so no module is loaded. Conda and apptainer
  are available.
- Compute nodes and login nodes both have outbound internet, but jobs that
  hold no GPU get killed, so downloads, extraction and other I/O-bound
  work run on a login node under tmux (`tmux new -d -s <name> '<command>'`),
  written to be resumable.
- Login nodes throttle sustained CPU use per user to about one core after
  roughly half an hour (processes are niced and share ~100% CPU; a
  16-worker decode fell from 29 to 2 clips/s). CPU-heavy preparation
  (decoding, Lance builds, evaluation caches) must therefore run as a job
  that holds a GPU for its 26 cores, e.g. `slurm/build_eval_caches.sbatch`.
  Light watchers and waiters are fine on the login node.

### Chained jobs

Every training run is submitted as a chain of 24-hour links:

```bash
bash slurm/chain.sh 3 slurm/nlp_chained.sbatch \
  --export=ALL,SIZE=vit_tiny,ARM=B,OUT=/scratch/CLUSTER_USER/runs/nlpB_k710_vitt
```

`chain.sh` submits N copies with `--dependency=afterany`, so each link
starts when the previous one ends for any reason. The link script resumes
from `checkpoint_last.pt`, exits immediately if `training_finished.json`
exists, and forwards `SIGUSR1` (sent 15 minutes before the limit) to the
trainer so it can checkpoint. Trainers already checkpoint every epoch, and a
K710 epoch is minutes, so a lost link costs at most one epoch.

Size the chain to the run: at ViT-B on 20% K710, 240 epochs is about two
days on one node, so `chain.sh 3`. Over-provisioned links exit in seconds.

### Multi-node

`slurm/nlp_multinode.sbatch` runs one `torchrun` agent per node with a
c10d rendezvous on the first node. Set `-N` and chain it the same way. With
the 32-GPU group cap, four nodes is the ceiling and only when nobody else
in the group is running. ViT-L is the only planned run that needs it.

## Data plan

| Dataset | Source | Where | Status |
| --- | --- | --- | --- |
| ImageNet-1K | Tamara `/scratch/CLUSTER_USER/datasets/imagenet` (146 GB, 1.33M files) | push with `transfer_from_tamara.sh` in tmux session `xfer` on Tamara | running since 2026-09-16 19:30 UTC |
| Kinetics-400 | Tamara `/scratch/CLUSTER_USER/datasets/k400` (433 GB, 261k mp4) | same script, after ImageNet | queued in the same run |
| Kinetics-600, 700-2020 train+val | CVDF S3 archives | `scripts/range/download_kinetics.sbatch` run on a login node, tmux session `kdl` | running since 2026-09-16, about 140 MB/s |
| TAP-Vid-DAVIS | public GCS, 1.67 GB | `scripts/range/download_tapvid.sh` on a login node | done, `/scratch/CLUSTER_USER/datasets/tapvid/tapvid_davis.pkl` |
| SSv2 | Qualcomm, account-gated | manual download of the 20bn webm archives and labels to `/scratch/CLUSTER_USER/datasets/ssv2` | needs your login |
| Walking Tours | YouTube via LeVJEPA's `download_walking_tours.sh` (yt-dlp) | later, only for the VideoMix reproduction | not started |
| PE-Video | Hugging Face | later, only for VideoMix | not started |

### K710 pipeline (automated on login2, 2026-09-17)

1. `scripts/range/extract_kinetics_parallel.sh` (tmux `kx`): 8 parallel tar
   streams for K600 and K700-2020 train; val archives are marked complete
   without extraction since exclusion works from the CSV lists. Writes
   `/scratch/CLUSTER_USER/datasets/.kinetics_download_finished`.
2. `scripts/range/build_k710.py` (tmux `k710`, waits for the sentinel):
   union of the three train sets, dedup by YouTube id, minus every id in any
   val or test list of any version (K600/K700 test lists fetched to
   `kinetics_meta/`), labels via VideoMAEv2's `misc/` label maps (checked
   against the identical-label merge), nested class-balanced 5/10/20/50%
   subsets with seed 0. Output `/scratch/CLUSTER_USER/datasets/k710/*.csv` and
   `stats.json`.
3. `scripts/range/build_lance_k710.sbatch` (tmux `lance_submit` submits it
   when `train_20pct.csv` appears): LeVJEPA's frame-per-row Lance schema at
   15 fps, short edge 384, JPEG 90, 128 rows per clip, built with PyAV by
   `scripts/range/build_lance_kinetics.py`. Holds 4 H200s for their 104
   CPUs; about two hours for the 20% subset (~750 GB). Output
   `/scratch/CLUSTER_USER/datasets/k710/train_20pct.lance`.

The Lance store is what both LeVJEPA and our trainer read, and it is
mirrored to Tamara once that scratch is cleaned; the raw archives are then
deleted from range. The environment needs `LD_LIBRARY_PATH` to prefer the
env's `libstdc++` (installed as an `activate.d` hook) or Lance's import of
sqlite3 fails against the system library.

Tamara's own scratch is 98% full; the 5 TB of retired synthetic feature
caches has to go before the Lance mirrors come back.
