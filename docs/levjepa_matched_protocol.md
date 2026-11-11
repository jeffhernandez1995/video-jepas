# Arm-to-arm with LeVJEPA, plus scaling

Written 2026-09-15. Supersedes the data and evaluation sections of
`natural_video_plan.md`; the two recipes (A and B) are unchanged.

## LeVJEPA's protocol, as published

- Pretraining data: a class-balanced 20% subsample of K710 (K400 + K600 +
  K700 train, validation overlap removed), following V-JEPA. About 130k
  clips. Their data-scaling run uses VideoMix (1.8M clips: K710, SSv2,
  Walking Tours, PE-Video).
- Models: ViT-S/B/L at patch 16, tubelet 1, RoPE, block-causal. Clips of
  16 frames at 224px (3,136 tokens), 95% uniform token dropping, 4 local
  views at 96px. 240 epochs, effective batch 3,072. Polyak average of the
  encoder at 0.9999 is what they evaluate.
- Evaluation: V-JEPA's frozen attentive probe without modification. IN1K
  with the image repeated along time, SSv2, and K400 by linear probe on
  mean-pooled tokens. Probes run on 8 temporal slots.
- Headline numbers to beat, ViT-B on 20% K710: epoch-matched 51.2 IN1K /
  30.4 SSv2 at 4.8 ExaFLOPs; FLOP-matched (1,085 epochs) 61.0 IN1K / 40.4
  SSv2 / 44.6 K400-linear. V-JEPA 2 retrained on the same data costs 36.4
  ExaFLOPs at ViT-B. ViT-L on VideoMix: 69.5 / 55.0.
- What they do not report: any correspondence or tracking evaluation.
  Their discussion names it as untested.

## Our compute position

Next-latent prediction processes every token of the clip plus a no-grad
teacher pass and a predictor. Per clip at 16 frames and ViT-B that is about
70 ExaFLOPs over 240 epochs on 20% K710, twice V-JEPA 2 and about 15×
LeVJEPA. We cannot drop 95% of tokens: the targets are dense. The honest
comparison is therefore the one LeVJEPA itself makes against V-JEPA 2:

- Epoch-matched: same data, 240 epochs, batch 3,072, our cost versus theirs
  stated in ExaFLOPs.
- FLOP-matched: LeVJEPA granted our budget, which is their 1,085-epoch
  configuration or longer. Their FLOP-matched numbers are the bar.

On eight A40s at roughly 0.4 PFLOP/s effective: ViT-S about 0.5 days,
ViT-B about 2 days, ViT-L about 7 days at 16 frames. At 8 frames, half.
LeVJEPA baselines retrained by us cost a twentieth of that.

## Matched pretraining

Identical for our A, our B, and the retrained LeVJEPA:

- Data: our own K710 rebuild (2026-09-19, `scripts/range/build_k710.py`,
  stats in `/scratch/CLUSTER_USER/datasets/k710/stats.json`): union of the K400,
  K600 and K700-2020 training sets deduplicated by YouTube id, minus every
  id in any validation *or test* list of any version (209k ids), labels in
  the 710 space via VideoMAEv2's maps with their identical-label merge.
  Result: 557,769 clips over all 710 classes (class sizes 227 to 1,598);
  VideoMAEv2's K710 has ~658k because it keeps test-set ids and we are
  missing ~18k unavailable downloads. The 20% subset is a per-class seeded
  prefix (seed 0) of 111,565 clips, nested inside the 5/10/50% subsets,
  with zero overlap against K400 val. The only unmapped label is K600's
  "chopping vegetables" (520 clips), which has no class in the 710 space.
  Both methods pretrain on exactly this subset; it differs from LeVJEPA's
  published subset (theirs is 20% of the VideoMAEv2 K710), so their
  numbers are a reference, not a matched baseline, until we retrain them.
- Storage: one Lance store per dataset, frame-per-row JPEG at 7.5 fps and
  short edge 256, which is LeVJEPA's own Walking Tours builder pointed at
  Kinetics. About 1 KB per frame-row of overhead plus 15 KB per JPEG, so
  20% K710 at 75 frames per clip is around 150 GB and full K400 about
  280 GB. Random temporal sampling then works for 240 epochs, and the
  LeVJEPA trainer runs on it unmodified for the baseline.
- Clips: 16 frames at stride 2 from the 7.5 fps store (about 2 seconds),
  224px RandomResizedCrop shared across frames, horizontal flip. Our
  methods see all 3,136 tokens; LeVJEPA sees its 5%.
- Schedule: 240 epochs, effective batch 3,072 via DDP over 8 GPUs with
  gradient accumulation, Muon at 1e-3 for ours (their AdamW recipe for
  theirs), 12 warmup epochs, cosine to 1e-5. Polyak average of the encoder
  at 0.9999 for evaluation, matching theirs; for A and B this is separate
  from the EMA teacher.
- Sizes: ViT-S, ViT-B, ViT-L. Two seeds at S and B; one at L.

## LeVJEPA trunks launched 2026-09-21

Their shipped recipe unchanged (flat LR after a 1,200-step warmup, AdamW
4e-4, 95% token drop, 10 local views, SIGReg 0.02, Polyak average 0.9999
evaluated), 240 epochs on our 20% K710 Lance store, effective batch 3,072
held by accumulation, 36 optimizer steps per epoch, 8,640 steps total.

| Run (W&B id) | Model | GPUs | Accumulation | Chain |
| --- | --- | ---: | ---: | --- |
| `levjepa_k710_vitt` | ViT-Tiny | 2 | 16 | jobs 268996, 268997 |
| `levjepa_k710_vits` | ViT-Small | 4 | 8 | jobs 268998, 268999 |
| `levjepa_k710_vitb` | ViT-Base | 8 | 4 | jobs 269000, 269001 |

Measured wall clock: ViT-Base 7.0 min/epoch (28 h, finished 2026-09-22),
ViT-Small 12.8 (51 h), ViT-Tiny 24.5 (98 h). The order is backwards
because the recipe is CPU-decode-bound and CPUs were allotted per GPU; see
`levjepa_loader_bottleneck.md`.

Cached-protocol scores, raw training weights (k400 lin-mean / att /
conv-att | in1k lin-mean / att / conv-att); the EMA readings are in W&B
under the original run names and are 10 to 17 points lower everywhere
(see "the EMA trap" below):

| Run | e60 | e120 | e180 | e240 |
| --- | --- | --- | --- | --- |
| ViT-T | 13.8/15.6/16.2 \| 12.4/17.4/21.7 | 17.3/19.4/20.1 \| 16.4/21.2/25.4 | 18.7/20.3/21.7 \| 18.1/23.2/26.6 | 19.2/20.2/22.1 \| 19.2/24.3/27.5 |
| ViT-S | 18.3/20.1/21.9 \| 18.3/23.7/26.9 | 22.1/24.3/25.5 \| 22.9/28.7/31.5 | 24.1/26.4/27.9 \| 25.7/31.5/34.1 | 25.7/28.1/29.5 \| 27.6/33.8/35.9 |
| ViT-B | 22.1/21.9/23.3 \| 23.0/25.3/28.1 | 25.7/27.4/27.8 \| 28.2/31.2/34.6 | 28.1/30.3/31.1 \| 31.1/35.2/38.1 | 28.9/32.0/32.3 \| 32.5/37.2/39.7 |

Checkpoints every 540 steps (15 epochs), all kept, so epochs 60, 120, 180
and 240 each have an exact checkpoint for evaluation. No cooldown branches
for LeVJEPA: its schedule is flat and the evaluated weights are the Polyak
average, so intermediate checkpoints are comparable as they are. W&B tags:
`natural-video`, `levjepa`, `k710`, model name. All runs created before
2026-09-16 were tagged `moving-mnist` retroactively.

### Chained evaluation (scaling-curve protocol)

`scripts/range/watch_milestones.sh` (tmux `evalwatch` on login2) submits
`slurm/eval_checkpoint.sbatch` for every milestone checkpoint a trunk
writes (`epoch-0059/0119/0179/0239.ckpt`); the trunk never waits. Each
evaluation uses one H200 and `scripts/natural/eval_checkpoint.py`:

- Fixed inputs from `scripts/natural/build_eval_caches.py`
  (`/scratch/CLUSTER_USER/datasets/eval_caches`): short edge 256, center crop 224,
  uint8, no augmentation. IN1K 300 images per class plus all 50k val; K400
  150 clips per class plus all 19,881 val, 8 frames at 7.5 fps from the
  clip center. Images are repeated 8 times along time.
- One extraction pass of the Polyak-averaged encoder (causal), keeping the
  CLS token, the mean patch token, and a pooled grid (video 8×4×4, image
  last slot 7×7).
- Probes fitted on the cached features, 20 epochs, best of two learning
  rates chosen on validation: linear on mean tokens, linear on CLS,
  attentive (one learned query, V-JEPA style), and conv-attentive (residual
  3×3 bottleneck on the grid, then attentive).
- Scores go to `<run>/evals/eNNN.json` and to W&B as
  `<run>/eval_eNNN` (group = trunk, tags `natural-video`, `eval`).

This is deliberately not V-JEPA's probe protocol, which trains the probe
with augmentation through the encoder and costs more than LeVJEPA's entire
pretraining per fit on IN1K. The cached protocol is for curves across
sizes and epochs; the exact protocol is run once per model on its final
checkpoint for the table.

One framework detail that bit us: stable-pretraining ignores
`checkpoint.dirpath` and writes every checkpoint under `SPT_CACHE_DIR`,
which defaults to `~/.cache` in the 50 GB home. The launch script sets it to
`<run>/spt`, resumes each chain link from the newest `last*.ckpt` there, and
the watcher searches the same tree.

## Matched evaluation

- V-JEPA attentive probe code, unmodified, on IN1K (present on scratch),
  SSv2 (to download), and K400 linear on mean-pooled tokens. Eight
  temporal slots. Layer sweep reported in the appendix; the main table
  uses the final layer, since that is what everyone else reports and the
  synthetic stage showed the final layer is where our recipe can fail.
- Our additional axis: TAP-Vid-DAVIS correspondence with the transport
  audit, direct matching at the final layer and the best layer, causal
  and full-context. Reported for LeVJEPA and V-JEPA 2 checkpoints too.
  This is the table they do not have.
- Per-horizon held-out cross-entropy is a within-run diagnostic only. It
  is not comparable across runs because the vocabulary is learned.

### Full cached protocol, raw weights (2026-09-29): the baseline for our runs

Complete IN1K (1.28M) and K400 (241k) training sets, LR chosen on 10k
held-out train items, validation only reported
(`eval_checkpoint.py --full`, `<run>_raw/evals/eNNN_full.json`). This is the
table the next-latent-prediction runs are compared against
(k400 lin-mean/att/conv-att | in1k lin-mean/att/conv-att):

| Run | e60 | e120 | e180 | e240 |
| --- | --- | --- | --- | --- |
| ViT-T | 15.6/20.0/22.2 \| 14.8/23.6/30.5 | 19.7/24.2/26.1 \| 19.3/28.6/34.6 | 21.4/25.8/27.9 \| 21.2/31.0/36.5 | 22.4/26.7/28.9 \| 22.4/32.3/37.1 |
| ViT-S | 21.7/25.4/27.2 \| 22.1/32.3/37.9 | 26.3/29.5/31.0 \| 27.2/38.0/42.6 | 28.7/31.8/33.6 \| 30.4/40.8/45.2 | 30.4/33.8/35.2 \| 32.1/43.2/47.1 |
| ViT-B | 26.6/25.6/27.5 \| 28.5/35.6/38.6 | 31.3/31.8/33.0 \| 34.2/41.4/43.9 | 33.8/34.2/36.3 \| 37.5/44.0/46.5 | 35.2/36.8/37.9 \| 38.9/45.8/47.8 |

The full training sets add 6 to 9 points over the subset protocol at the
same checkpoint (ViT-B e240 IN1K attentive 45.8 vs 37.2); the exact V-JEPA
probe on the same ViT-B checkpoint gives 48.8.

### Which weights to evaluate: the EMA trap (2026-09-24)

LeVJEPA's checkpoints carry `state_dict_ema`, and their README says that
is what to evaluate. Their `WeightEMA` callback (`callbacks.py`) has no
warm-up: it is initialised from the random weights at train start and
decays at 0.9999 per optimizer step. Our 240-epoch trunks take 8,640
steps, and 0.9999^8640 = 0.42, so **42% of the evaluated EMA was still
the initialisation**, the rest a lagged average weighted toward early
training. Every number first reported for these trunks (the EMA columns
below) measured that blend, not the model. Their own VideoMix runs are
~58k steps, where the lag is harmless; for a ~10k-step K710 run it costs
10 to 17 points. `EVAL_WEIGHTS=raw` in `scripts/natural/eval_checkpoint.py`
switches both evaluators to the training weights; raw is now the default
reading for every trunk and both are kept in W&B (`<run>_raw`).

Fix for every subsequent run (ours and any LeVJEPA retrain): warm up the
Polyak average, `decay_t = min(0.9999, (1+t)/(10+t))`, and evaluate raw
and averaged weights side by side.

### Exact V-JEPA protocol on the ViT-Base trunk (2026-09-24)

> **Invalid, 2026-10-26.** probe_vjepa.py called `ddp.module` directly, which skips
> DDP's gradient all-reduce: each of the 4 ranks trained its own probe on a quarter of
> the data, and the reported accuracy is the mean of those 4 probes. The numbers below
> (and the K400 exact < cached puzzle) predate the fix and must be rerun before use.

`scripts/natural/probe_vjepa.py` (V-JEPA frozen attentive probe, full
train sets, 20 epochs, AutoAugment / RandAugment, 8-slot still video for
IN1K, 8 segments × 3 crops joint attention for K400) on
`levjepa_k710_vitb` epoch 240, 4 H200s each, raw weights (EMA in
parentheses):

| | exact protocol (att / lin) | cached protocol (att / lin-mean) | LeVJEPA published ViT-B |
| --- | --- | --- | --- |
| IN1K | **48.81** / 20.90 (36.35 / 13.76) | 37.17 / 32.53 (25.85 / 22.79) | 51.2 att (epoch-matched, V=4) |
| K400 | 25.17 / **20.69** (14.39 / 11.24) | 32.04 / 28.89 (20.13 / 20.41) | 44.6 lin (FLOP-matched only) |

The 15-point IN1K gap to the paper was 12.5 points of EMA lag; what
remains is 2.4 points, with 111.5k vs ~130k clips (15% fewer optimizer
steps) pulling one way and the repo's V=10 local views against the
paper's V=4 (+2.6 in their ablation) the other. The trunk reproduces the
paper. On K400 the exact protocol still scores below the cached one
(25.2 vs 32.0 attentive): RandomResizedCrop(0.08–1)+RandAugment at train
against center crops at test, and one clip at train against eight jointly
attended at test, is a train/test shift the encoder does not fully
absorb; the published K400 number is FLOP-matched (4.5× the schedule) and
not comparable. Per-5-epoch curves are in W&B (`levjepa_k710_vitb_raw_probe_*`,
tag `eval-exact`). Both protocols keep the T < S < B ordering
(`docs/figures/levjepa_k710_curves.png`, `scripts/natural/plot_levjepa_curves.py`),
so the cached protocol stays the scaling-curve instrument and the exact
one is the headline number.

## Scaling experiments, AIMv2 style

AIMv2 scales two axes and plots validation loss against samples seen at
fixed size, and against FLOPs at fixed data, then shows the downstream
probe tracking both. It gets the data axis from intermediate checkpoints
of one long run plus a cooldown, not from separate subset runs.

- Model axis: ViT-S/B/L at fixed data and epochs, which is the matched
  sweep above. Add ViT-Ti as a fourth point since it costs nothing. Plot
  IN1K and SSv2 attentive accuracy and TAP-Vid PCK against pretraining
  ExaFLOPs, both methods on the same axes.
- Data axis: one ViT-B run under a warmup-stable-decay schedule with
  cooldown branches at 60, 120, 240, and 480 epochs on 20% K710, each
  branch a 10% cooldown. The 480-epoch branch is the FLOP-matched regime
  for LeVJEPA's 1,085 epochs. Same branches for LeVJEPA. Then one ViT-B
  at 240 epochs on full K710 as the fifth data point.
- Loss axis: our pretraining loss is not stationary across runs, so the
  AIMv2 loss curves are replaced by a frozen reference target: prototype
  assignments from the ViT-L teacher's final checkpoint, evaluated on a
  fixed 10k-clip held-out set. Every model's predictor is scored by a
  linear map fitted to that reference on a disjoint 10k clips. This gives
  one comparable loss number per checkpoint. It is a proxy and is
  labeled as one.
- Power-law fits on accuracy versus FLOPs only where the four sizes are
  monotone; report exponents with bootstrap intervals over probe seeds,
  and do not fit through the ViT-Ti point if it sits below the trend.

## Order of work

1. Retire the 5 TB of synthetic caches. Download K600, K700, SSv2,
   TAP-Vid-DAVIS. Install PyAV and decord. Build the K710-20% and SSv2
   Lance stores with LeVJEPA's builder. Get V-JEPA's probe code running on
   IN1K with the released LeVJEPA ViT-L to confirm the protocol
   reproduces their 69.5.
2. Retrain LeVJEPA ViT-S on K710-20% with their config to confirm the
   pipeline reproduces their ViT-S point. This is cheap and it is the
   calibration of everything after it.
3. A and B at ViT-S, seed 0, 16 frames, 240 epochs. Full evaluation
   including TAP-Vid. About a day. This is the go/no-go for the paper.
4. ViT-B pair, then ViT-L for A only, then seeds.
5. The WSD data-axis run at ViT-B for A and LeVJEPA.
6. Ablations from the natural-video plan, on ViT-S, in the order there.
