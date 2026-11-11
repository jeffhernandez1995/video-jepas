# Why the LeVJEPA trunks are loader-bound (and why ViT-Tiny is the slowest)

Measured on the three K710 trunks, 2026-09-21 to 2026-09-23, from the
timestamps of their 15-epoch checkpoints. The store is the 20% K710 Lance
store (111,566 clips, 505 GB, 15 fps frames stored as JPEG at short edge
384), one clip per video per epoch.

## The numbers

| Trunk | GPUs | CPUs | Loader workers | min / epoch | clips / s | clips / s / GPU | clips / s / core | 240 epochs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-Base | 8 | 208 | 160 | 7.0 | 266 | 33 | 1.28 | 28 h (done) |
| ViT-Small | 4 | 104 | 80 | 12.8 | 145 | 36 | 1.40 | 51 h |
| ViT-Tiny | 2 | 52 | 40 | 24.5 | 76 | 38 | 1.46 | 98 h |

Throughput per GPU is the same for all three models, 33 to 38 clips per
second, and it is really throughput per 26 CPUs: each task gets the
per-GPU share of a node (208 CPUs / 8). The model does not matter at all.
The wall-clock ordering Base < Small < Tiny is simply the CPU ordering
208 > 104 > 52.

The GPU side is close to idle. With 95% token drop, ViT-Base's global view
is about 160 tokens per clip and each of the ten 96-pixel local views is
about 30. That is a few hundred tokens per clip through an encoder that
could take tens of thousands, and the trunks report about 5% real GPU
utilization. Removing the GPUs would not change the step rate; removing the
CPUs would.

## Where the CPU goes

Each clip costs roughly 0.7 s of one core (1.3 to 1.5 clips per second per
core above, with workers sharing cores with the main process). The cost is:

1. **16 JPEG decodes at 384 × 683.** The store was built at LeVJEPA's own
   resolution (short edge 384) but the global crop is 224 and the local
   crops are 96, so every frame is decoded at 2 to 3× more pixels than it
   is used at. Decode time scales with pixels.
2. **11 crops per clip, bicubic, with color jitter including hue.** Two
   global (224) and ten local (96) views, each a resized crop of all 16
   frames with brightness, contrast, saturation and hue jitter. Hue jitter
   is the expensive one: an RGB→HSV→RGB round trip in float on the CPU.
   LeVJEPA's config comment says dropping the hue term makes the transform
   about 3× cheaper, and also says why they keep it (it denies the encoder
   a colour-histogram shortcut).
3. **Not storage.** Warming the page cache so the whole store sits in node
   RAM (the `find | cat` in `slurm/levjepa_chained.sbatch`) made no
   measurable difference to the step rate; the cold-read rates of 38 to 77
   clips per second seen early on were the same order as the decode limit.

LeVJEPA's own published throughput on a consumer GPU works out to about 115
clips per second, so this regime is the recipe's normal one, not something
broken in our setup.

## What "should" have happened

If training were compute-bound the order would be Tiny < Small < Base in
wall clock and the 2/4/8 GPU split would be sensible. Because it is
loader-bound, the split is exactly backwards: the smallest model got the
fewest cores and takes 3.5× longer than the largest. In hindsight the right
allocation was the same CPU count for every trunk regardless of model, for
example `--cpus-per-task=104 --gres=gpu:h200:2` for ViT-Tiny (Slurm gives
CPUs per task, not per GPU; the job is not `--exclusive`, so it can still
share a node). That would have put all three at about 28 hours.

## Fixes, in order of payoff

These are for the next round of LeVJEPA runs (ViT-L, VideoMix, or any
rerun). The current trunks are left to finish as they are; rerunning them
with a different loader would not change their results, only the clock.

1. **Match CPUs to the loader, not to the GPUs.** Immediate, no code: give
   every LeVJEPA job 104 to 208 CPUs. Verified 2026-09-23 that `commons`
   has no `MaxCPUsPerGPU`/`DefCpuPerGPU` and `sbatch --test-only` accepts
   2 GPUs with 104 or 208 CPUs; the 26-per-GPU figure was our own choice
   copied from LeVJEPA's Slurm script. 104 CPUs (half a node) is the
   practical ceiling, since a 208-CPU request needs a node with no other
   job on it. `loader.num_workers` must be raised with it (currently 20
   per task) and memory stays per CPU (`--mem-per-cpu`).
2. **Build a 256 store.** `scripts/range/build_lance_kinetics.py --short 256`
   (adding the flag) cuts decode and resize work by about 2.2× and the store
   to about 230 GB. The 224 global crop is unaffected; only the
   `global_crops_scale: [0.8, 1.0]` random-resized crop loses a little
   headroom above 224, which is what VideoMAE and SlowFast accept by
   storing Kinetics at short side 256 or 320.
3. **Decode and augment on the GPU.** Workers return JPEG bytes; the
   training step calls `torchvision.io.decode_jpeg(device='cuda')` on the
   batch, then does the 11 crops and the jitter as batched tensor ops
   (Kornia or plain torch; hue jitter in particular is cheap on a GPU). At
   5% utilization the GPUs have the room. NVIDIA DALI is the packaged
   version of the same idea. This removes the CPU from the loop entirely
   and is the only fix that gets LeVJEPA to a GPU-bound regime.
4. **Drop hue jitter** (`color_jitter_hue: 0`) if we are willing to deviate
   from the published recipe. We are not, for the matched arm.

## Why our own trainer is not in this situation

Next-latent prediction reads the same store but uses one shared 224 crop,
no color jitter and all 16 frames as tokens. That is about 150 ms of CPU per
clip from the 384 store and about 70 ms from a 256 store, so 26 cores supply
several hundred clips per second per GPU, while a full-token ViT-S consumes
tens. It will be GPU-bound, which is the regime the scaling comparison
assumes. The loader cost above belongs to LeVJEPA's multi-view recipe.
