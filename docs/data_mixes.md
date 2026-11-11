# Pretraining data mixes 1–3

Started 2026-10-14. Goal: move next-latent prediction off the 20% K710 set
(111,565 clips, where both masked arms plateau by epoch 60) onto larger,
more diverse data, shaped after V-JEPA 2.1 (web video dominant, SSv2 boosted,
a separate image stream) and LeVJEPA's scaling corpus (K710 + SSv2 + PE Video).

## Sources

| Source | Raw | Hours | Store (7.5 fps, short edge ≤ 384) | Licence | Notes |
|---|---|---|---|---|---|
| K710 | on disk | ~1,300 | 20% store (15 fps, existing) + `k710.lance` for the other 80% | research | `train_rest80.csv` = full minus the 20% |
| SSv2 | 19 GB HF mirror (`morpheushoc/something-something-v2`) | 168 | `ssv2.lance` | Qualcomm research | train ids only; val is the probe |
| PE Video | 1.6 TB (`facebook/PE-Video` train + extended) | 4,625 | ~3.9 TB | CC-BY-NC | test split left out |
| VidaForge-3M | 2.4 TB (`VidaForge/VidaForge-3M`) | 6,475 | ~2.4 TB | Apache-2.0 | kept: real-world domain, best clip of each Cosmos duplicate group, ≥ 2.2 s |
| EgoDex | 1.7 TB (Apple CDN) | ~820 | ~0.8 TB | CC-BY-NC-ND | test split (their benchmark) left out |
| DROID | 0.4 TB (`lerobot/droid_1.0.1`) | ~510 per camera, 3 cameras | ~0.3 TB | Apache-2.0 | 320×180 AV1; cut at episode boundaries |
| ImageNet-1K | on disk | – | `in1k.lance` | research | Mix 1 only; see robustness evals |
| CC12M | 1.2 TB (`pixparse/cc12m-wds`) | – | `cc12m.lance` | other | the CLIP-style web image source |

Dropped: PASS (no people; replaced by CC12M), Walking Tours (the HF repo is
only a YouTube download script), Ego4D (licence approval), AgiBot (50 TB).

## Mixes (sampling weights; specs in `scripts/natural/mixes/`)

| | Mix 1: LeVJEPA+ | Mix 2: web-heavy | Mix 3: no PE Video |
|---|---|---|---|
| K710 | 0.25 | 0.05 | 0.10 |
| SSv2 | 0.15 | 0.15 | 0.15 |
| PE Video | 0.35 | 0.30 | – |
| VidaForge | – | 0.30 | 0.50 |
| EgoDex | 0.10 | 0.08 | 0.10 |
| DROID | 0.05 | 0.04 | 0.05 |
| Images | 0.10 IN1K | 0.08 CC12M | 0.10 CC12M |

Images are static 16-frame clips (one crop, repeated) through the unchanged
model; under tube masking this is latent inpainting, like the h=0 control.
A virtual epoch is 111,565 clips with fixed per-source counts, so 240 epochs is
step-matched to every K710-20% run. Weights are a starting point for the
ViT-T screen, not tuned.

## Pipeline

1. `scripts/range/download_mix.sh` (RANGE login node, tmux): raw downloads to
   `/scratch/CLUSTER_USER/datasets/raw/`.
2. `scripts/range/build_lance_mix.py` / `build_lance_images.py`: one Lance shard
   per raw unit, shared atomic claims, so RANGE jobs (`slurm/build_mix.sbatch`)
   and Tamara loops (`scripts/tamara/build_mix_remote.sh`, staged in
   `/dev/shm`, pushed over the ControlMaster) split the work.
3. `scripts/range/consolidate_mix.py`: hard-links finished shards into one
   Lance dataset per source (opening ~2,000 shards per worker was the loader
   bottleneck), with a pinned version and a precomputed episode index.
4. `scripts/natural/mix_data.py` + `train_nlp.py --mix scripts/natural/mixes/mixN.json`.

## Evaluation additions

- SSv2: full train and val sets (8 frames uniform over the clip). Evaluation
  is full datasets only (2026-10-15); the per-class subset caches are unused.
- ImageNet robustness: IN1K probes scored without refitting on IN-v2, Sketch,
  IN-R and IN-A (R and A restricted to their 200 classes). Needed because Mix 1
  pretrains on IN1K images.

## Runs (2026-10-15)

All masked B h1/2/4 (90% tube masks), 240 virtual epochs of 111,565 clips
(26.8M clips, the step budget of every K710-20% run). Evaluation only on the
full datasets (K400, IN1K, SSv2 train/val; IN-v2/R/A/Sketch from the IN1K
probes), on RANGE.

| Run | Data | Where |
|---|---|---|
| nlpBm_mix1_vitt, nlpBm_k710full_vitt | 256-px stores | Tamara, 4 A40 each (~238 clips/s, ~31 h) |
| nlpBm_mix3_vitt | 256-px stores | Tamara round 2, 8 A40 (~420 clips/s, ~18 h) |
| nlpBm_mix2_vitt | 256-px stores | RANGE, 2 H200 |
| nlpBm_mix2_vitb, nlpBm_mix1_vitb | 384-px stores | RANGE, 8 H200, 3 chained links |
| nlpBm_k710_vitb | 20% K710 store (384 px) | RANGE, 8 H200: the matched comparison with levjepa_k710_vitb |
| nlpBm_mix2_vitl | 384-px stores | RANGE, one 8 H200 node, 6 chained links |
| nlpAm_mix3_vitt | 256-px stores | Tamara, 8 A40 (~146 clips/s, ~51 h): masked A (lean, 10 Sinkhorn iterations, no guard) on the screen winner, started 2026-10-19 |

Masked B everywhere first; masked A is re-run only on the mix that wins the
screen. Mix 3 won at e60 (ViT-T attentive vs full K710: K400 32.2 vs 28.8,
IN1K 37.9 vs 34.4, SSv2 25.8 vs 19.3; Mixes 1 and 2 were below full K710).
Mix 3 masked B ends at e240 at 29.2/32.3/23.1 (K400/IN1K/SSv2 attentive; full
K710 28.2/33.9/18.0), having dropped after e60 and stayed flat from e120. Mix 3
masked A is flat throughout (e060 26.5/30.0/20.1, e240 28.1/29.8/23.1): below
masked B on K400 and IN1K at every epoch, level on SSv2 by e180. The LeVJEPA comparison is shown on one dataset, 20% K710, where all
three LeVJEPA sizes were already retrained under the matched protocol.

The 256-px copies (`scripts/range/reencode_mix.py`, `/scratch/CLUSTER_USER/datasets/mix256`)
exist because Tamara has 5.8 TB free and Mix 2 needs 10.9 TB at 384 px (6.4 TB
at 256). Tamara's NFS home is not usable for training (47 random reads/s).

## Collapse at scale and the guard (2026-10-19)

ViT-B (K710 and Mix 1) and ViT-L peak at e60 and then decline; the teacher's
feature effective rank falls 20-30% after e60 (ViT-B K710: 551 at e60, 381 at
e150) while the prediction loss drops, and the coding rate barely moves
(2222 -> 2190). Data and hyperparameters were checked and are not the cause
(identical configs; decoding verified per source; no rotation metadata in 400
sampled raw K710 videos; PE Video is low-motion stock footage with 13%
duplicate frames, which explains Mixes 1-2 losing the screen but not the decline).

Bug: `calibrate_gamma` summed the guard gradient over ranks, but the coding
rate's autograd all_reduce already makes each rank's gradient global, so gamma
was world-size times too small. Applied guard = 0.1 / world of the prediction
gradient: 1/8 of intended on every 8-GPU run (all ViT-B/L, Mix 3 ViT-T), 1/2 to
1/4 on the other ViT-T runs. The only ViT-T run on 8 GPUs (Mix 3) is also the
only one whose rank and probes fall after e60 (rank 136 -> 124; IN1K attentive
37.9 -> 33.0). Fixed in `nlp.calibrate_gamma` (mean over ranks); existing runs
keep their gamma from file.

Tests (ViT-B, 20% K710, matched to nlpBm_k710_vitb / levjepa_k710_vitb):

| Run | Guard | Jobs |
|---|---|---|
| nlpBm_k710_vitb_sigreg | SIGReg on raw patch features (`guard=sigreg`), calibrated at 0.1 | 333549-551 |
| nlpBm_k710_vitb_crfix | coding rate, calibrated correctly (8x the old gamma) | 333552-554 |

Success: feature rank holds after e60 and e120 probes match or beat e60.

SIGReg result (gamma 2.07e-6, 483 clips/s, 38 GB, data wait ~0): teacher feature
rank rises monotonically (82 at e5, 234 at e60, 287 at e120, 306 at e222), lower
in absolute terms than the old run (537 at e60, falling to ~405 from e150) but
with no collapse. Attentive probes, student, K400 / IN1K / SSv2:

| Epoch | SIGReg | coding rate, fixed | old guard (nlpBm_k710_vitb) | LeVJEPA ViT-B (raw; K400/IN1K only) |
|---|---|---|---|---|
| 60 | 38.9 / 43.1 / 34.0 | 39.5 / 38.7 / 36.7 | 39.3 / 42.5 / 34.5 | 25.6 / 35.6 |
| 120 | 41.6 / 45.2 / 37.2 | 42.1 / 40.2 / 39.7 | 37.9 / 39.9 / 33.6 | 31.8 / 41.4 |
| 180 | 42.2 / 44.8 / 38.2 | 43.5 / 39.9 / 40.6 | 35.9 / 36.4 / 30.2 | 34.2 / 44.0 |
| 240 | 42.4 / 44.1 / 38.8 | 44.2 / 39.5 / 40.6 | 35.9 / 35.5 / 30.1 | 36.8 / 45.8 |

Coding rate, fixed (gamma 7.8e-5, 12x the old 6.3e-6; 488 clips/s, 38 GB): feature
rank flat at ~650 from e15 to e240. Both fixes stop the decline. The coding rate
is best on video (K400 +1.8, SSv2 +1.8 over SIGReg at e240) but 4.6 lower on IN1K
and below SIGReg on every IN1K robustness set; SIGReg is within 1.7 of LeVJEPA on
IN1K and 5.6 above it on K400.

Follow-up (2026-10-22): nlpBm_mix3sigreg_vitt, the Mix 3 ViT-T screen recipe with
`guard=sigreg`, on Tamara's 8 A40s (`SET=guard=sigreg` in run_screen.sh), to compare
with nlpBm_mix3_vitt (old guard, declined after e60) and nlpAm_mix3_vitt.
Gamma 1.77e-6, 419-446 clips/s, 18 h. Attentive K400 / IN1K / SSv2, student:

| Epoch | Mix 3 SIGReg | Mix 3 old guard | Mix 3 masked A | full K710 |
|---|---|---|---|---|
| 60 | 32.4 / 37.8 / 27.3 | 32.2 / 37.9 / 25.8 | 26.5 / 30.0 / 20.1 | 28.8 / 34.4 / 19.3 |
| 120 | 33.0 / 37.8 / 29.1 | 29.3 / 33.0 / 22.0 | 26.4 / 29.0 / 21.5 | 28.5 / 33.1 / 18.0 |
| 240 | 32.6 / 34.5 / 29.0 | 29.2 / 32.3 / 23.1 | 28.1 / 29.8 / 23.1 | 28.2 / 33.9 / 18.0 |

K400 and SSv2 hold or rise after e60; IN1K holds to e120 and then slips 3.3. Feature
rank peaks at 114 (e90-e120) and ends at 108, lower than the old-guard run's 125-137,
so at ViT-T the rank number alone does not track probe quality.

Scale-up with SIGReg (2026-10-23), 384-px Mix 3, RANGE 8xH200:
nlpBm_mix3_vitb_sigreg (jobs 340586-587) and nlpBm_mix3_vitl_sigreg (340588-590).
ViT-B: gamma 3.3e-6, 473 clips/s, 38 GB, 16.9 h. ViT-L: gamma 2.4e-6, 257 clips/s,
59 GB; feature rank rises through e180 (236 at e60, 340 at e180). Attentive K400 / IN1K /
SSv2, student:

| Epoch | Mix 3 ViT-L SIGReg | Mix 3 ViT-B SIGReg | K710 20% ViT-B SIGReg |
|---|---|---|---|
| 60 | 38.6 / 42.1 / 35.7 | 37.5 / 42.5 / 35.0 | 38.9 / 43.1 / 34.0 |
| 120 | 42.8 / 46.0 / 40.5 | 41.0 / 45.2 / 39.1 | 41.6 / 45.2 / 37.2 |
| 180 | 45.9 / 49.9 / 43.6 | 43.0 / 45.9 / 42.2 | 42.2 / 44.8 / 38.2 |
| 240 | pending | 43.7 / 45.9 / 43.4 | 42.4 / 44.1 / 38.8 |

With SIGReg, Mix 3 beats 20% K710 at ViT-B (e240: +1.3 / +1.8 / +4.6) and ViT-L beats
ViT-B. Mix 3 ViT-L at e180 matches the released LeVJEPA-VideoMix-Large under our
protocol on K400 (45.9 vs 45.9) and nearly on SSv2 (43.6 vs 44.3); IN1K trails (49.9 vs 58.5).
nlpBm_mix2_vitb (old guard) was cancelled. nlpBm_mix2_vitl (old guard) was
stopped at epoch 186 on 2026-10-20 (links 328626-630 cancelled; checkpoints to
e180 and checkpoint_last kept); ViT-L reruns with whichever guard wins the tests.

## Reference checkpoints under our protocol (2026-10-23)

Our numbers are far below the papers' (V-JEPA ViT-L/16 reports roughly K400 80.8,
SSv2 69.5, IN1K 74.8 with its attentive-probe protocol) because of ~10x less
pretraining and a faster cached, augmentation-free probe protocol. To size the
protocol part, scripts/tamara/eval_reference.sh runs our full-dataset eval on the
released V-JEPA ViT-L/16, V-JEPA 2 ViT-L and LeVJEPA-VideoMix-Large weights on
Tamara (one A40 per model x dataset; results in /scratch/CLUSTER_USER/runs/ref_<family>/evals/).
Loaders (eval_checkpoint.py --family vjepa | vjepa2 | levjepa_hf) build the official
architectures from third_party/jepa and third_party/vjepa2 and load the target
encoders; tubelet-2 models get each of our 8 frames twice (their native 16-frame clip,
8 temporal slots) and the token mean as CLS. V-JEPA 2 runs at our 224 px, not its 256.

Results (2026-10-24), linear / attentive / conv-attentive, then IN-v2/R/Sketch/A attentive:

| Model | K400 | IN1K | SSv2 | IN-v2 / R / Sketch / A |
|---|---|---|---|---|
| V-JEPA ViT-L/16 | 59.9 / 63.3 / 63.4 | 57.2 / 58.7 / 59.5 | 44.7 / 58.4 / 59.2 | 46.9 / 22.8 / 12.0 / 6.3 |
| V-JEPA 2 ViT-L | 61.7 / 65.3 / 64.8 | 70.6 / 67.0 / 67.2 | 43.9 / 60.0 / 60.2 | 56.2 / 33.1 / 22.6 / 26.7 |
| LeVJEPA-VideoMix-Large | 42.3 / 45.9 / 46.7 | 51.9 / 58.5 / 60.0 | 31.2 / 44.3 / 46.3 | 45.5 / 18.8 / 9.8 / 3.4 |

V-JEPA ViT-L's paper numbers (attentive, ~80.8 / 74.8 / 69.5) sit ~17 / 16 / 11 points
above ours on K400 / IN1K / SSv2: that is the protocol gap (16 frames x multi-clip x
3 views, augmentation, longer probe training). V-JEPA 2's IN1K linear (70.6) beating our
attentive (67.0) also says our 20-epoch, 2-LR attentive probe under-fits. The gap is a
rough offset, not a constant; a paper-comparable number needs probe_vjepa.py-style evals.

## Evaluation tiers (decided 2026-10-26)

- Analysis, ablations, scaling curves: the fast cached protocol above
  (eval_checkpoint.py, full datasets), applied identically to our runs and to the
  reference encoders (`--family levjepa_hf | vjepa | vjepa2`). Valid for relative
  comparisons because every method gets the same protocol.
- Final results table only: V-JEPA's paper protocol via probe_vjepa.py on the final
  checkpoints and the references, token-matched as in LeVJEPA (decided 2026-10-27):
  every model gets 8 temporal slots per segment, i.e. 8 frames for tubelet-1 encoders
  (ours, LeVJEPA) and 16 for tubelet-2 (V-JEPA, V-JEPA 2), at ~7.5 fps. Test views
  as in V-JEPA/LeVJEPA: K400 8 segments x 3 crops (attended jointly across segments),
  SSv2 2 x 3, IN1K 1 center crop. Training: one random clip per video
  (`--train-segments 1`). K400 is reported attentive and linear (LeVJEPA's K400 metric).
  To do: SSv2 is not in probe_vjepa.py yet (and raw SSv2 videos are not on Tamara);
  K400 16x8x3 costs ~3 days and IN1K ~2 days per ViT-L on Tamara's 8 A40s.
- Check of the pipeline first (2026-10-27): V-JEPA ViT-L, K400, 16 frames, 1 training
  segment, 8x3 validation (`exact_k400_16f.json`, 14.7 h on 8 A40s, CPU-decode bound):
  attentive 60.0 / 65.9 / 70.0 / 71.4 at probe epochs 5 / 10 / 15 / 20 (linear 62.8),
  against the paper's ~80.8 and our cached protocol's 63.3. Still rising at epoch 20;
  the rest of the gap is likely the 8-segment joint training (`--train-segments 8`).
  The first attempt crashed on a probe_vjepa.py bug (no DDP gradient sync, fixed
  2026-10-26; see levjepa_matched_protocol.md).

## Baselines for the final table (decided 2026-10-27)

Released self-supervised encoders only (no label-finetuned weights: VideoMAE
`*-finetuned-kinetics`, TimeSformer and ViViT are excluded), all frozen under the
token-matched paper probe, ViT-L class:

| Model | Source | Frames (slots) | `--family` |
|---|---|---|---|
| V-JEPA ViT-L/16 | dl.fbaipublicfiles.com/jepa/vitl16 | 16 (8) | vjepa |
| V-JEPA 2 ViT-L | dl.fbaipublicfiles.com/vjepa2/vitl.pt | 16 (8) | vjepa2 |
| LeVJEPA-VideoMix-Large | HF galilai-group | 8 (8) | levjepa_hf |
| VideoMAE ViT-L (pretrain only, K400 1600 ep) | HF MCG-NJU/videomae-large | 16 (8) | videomae |
| VideoMAE V2 ViT-L (UnlabeledHybrid-1M, 800 ep) | HF OpenGVLab/VideoMAEv2-Large | 16 (8) | videomaev2 |
| DINOv2 ViT-L/14 (image SSL, per frame) | HF facebook/dinov2-large | 8 (8) | dinov2 |

Paper-probe results, complete 2026-11-03 (1 training segment, K400 8x3 and SSv2 2x3 views,
20 epochs, attentive / linear):

| Model | K400 | SSv2 | IN1K |
|---|---|---|---|
| V-JEPA-L | 71.4 / 62.8 | 67.5 / 40.4 | 75.6 / 49.8 |
| V-JEPA 2-L | 73.9 / 65.1 | 68.5 / 35.2 | 81.0 / 69.9 |
| LeVJEPA-VideoMix-L | 62.9 / 22.7 | 50.0 / 16.6 | 69.3 / 28.5 |
| VideoMAE-L | 76.1 / 45.9 | 63.8 / 24.8 | 71.7 / 39.4 |
| VideoMAE V2-L (label-supervised, see below) | 81.2 / 80.8 | 55.3 / 29.2 | 65.5 / 45.5 |
| DINOv2-L (per frame) | 82.3 / 73.1 | 45.1 / 23.1 | 86.3 / 83.9 |
| Ours, Mix 3 ViT-L SIGReg e240 (student) | 59.2 / 33.3 | 47.9 / 20.1 | 62.8 / 31.4 |

The 1-clip protocol costs V-JEPA more than VideoMAE (published 80.8 vs 77.8; ours 71.4 vs 76.1).
Ours is last on K400 and IN1K and sixth on SSv2 (above DINOv2 only); the cached probe
underestimated the paper probe by 12 points on K400 (47.4 vs 59.2) and 3 on SSv2.
IN1K for V-JEPA, V-JEPA 2, LeVJEPA and ours resumed on Tamara (8 A40s) from per-epoch states
saved on NOTS (4 L40S): same global batch and step count, different per-GPU batch.

Caveat, VideoMAE V2-L (HF OpenGVLab/VideoMAEv2-Large, 2026-10-29): the card says 800 epochs of
self-supervised pretraining on UnlabeledHybrid-1M, but the weights are the fine-tuning
architecture (fc_norm + mean pooling, head removed, no decoder) rather than a pretraining encoder
(final norm + pixel decoder), and the VideoMAE V2 zoo's ViT-S/B/L-size models are distilled from a
K710-fine-tuned ViT-g. K400 linear 80.8 = attentive 81.2 (VideoMAE-L: 45.9 vs 76.1) fits K710 label
supervision. Treat it as label-supervised: out of the self-supervised comparison on K400 / SSv2, or
footnoted as such (its IN1K 65.5 and SSv2 55.3, both below VideoMAE-L, fit the same reading).
V-JEPA 2's first K400 run hung at its epoch-5 validation (probe_vjepa.py was edited under it; its
spawned workers re-import the file); the rerun is the number above.

Plus the controlled comparison (ours vs LeVJEPA retrained on 20% K710, ViT-T/S/B).
Weights in /scratch/CLUSTER_USER/models on Tamara; the VideoMAE V2 architecture is vendored in
third_party/videomaev2_hf (no HF remote code). Queue: scripts/tamara/probe_queue.sh.

## Tracking and intuitive physics (2026-10-27)

- DAVIS-2017 val label propagation, scripts/natural/eval_davis.py: DINO protocol (480p,
  L2-normalized patch features, first frame + last 7 frames as context, top-5, temperature
  0.1, 12-patch radius), official davis2017-evaluation J&F. Token-matched windows of 8 slots;
  224-px position tables interpolated to the 480p grid; RoPE models get height / width positions
  rescaled onto their pretraining grid (default since 2026-10-27, as V-JEPA 2's AC predictor does;
  extrapolating instead cost V-JEPA 2 6.9 and LeVJEPA 3.0 J&F). ~7 min per ViT-L.
  J&F (J / F): DINOv2-L 66.8 (64.8 / 68.8), VideoMAE-L 55.4 (54.9 / 56.0), LeVJEPA-L 50.8
  (49.7 / 51.9), V-JEPA-L 47.9 (47.0 / 48.9), V-JEPA 2-L 47.0 (44.8 / 49.2), VideoMAE V2-L
  40.2 (39.2 / 41.2; its vendored attention now uses fused SDPA, the explicit version ran out
  of memory at 480p). Ours (Mix 3 ViT-L SIGReg e240) 55.5 (54.0 / 57.0), level with VideoMAE-L.
- Physics, scripts/natural/eval_physics.py: plausible vs implausible linear probe per category
  on mean-pooled features, stratified 60/40, 5 seeds (Esmati et al. 2026, Fig. 2), on IntPhys 2
  Main (1,012 videos; the original IntPhys 2019 is no longer downloadable) and InfLevel-Lab
  (5,772 videos; InfLevel's authors intend zero-shot use only, the probe follows the paper).
  Results (2026-10-28), split by matched set (the default; average over categories):

  | Model | IntPhys 2 | InfLevel avg | continuity | gravity | solidity |
  |---|---|---|---|---|---|
  | V-JEPA-L | 50.3 | 68.6 | 98.2 | 56.6 | 51.1 |
  | V-JEPA 2-L | 50.3 | 68.5 | 98.4 | 55.8 | 51.5 |
  | LeVJEPA-L | 51.9 | 66.5 | 97.6 | 53.5 | 48.4 |
  | VideoMAE-L | 51.0 | 68.9 | 97.9 | 57.3 | 51.5 |
  | VideoMAE V2-L | 50.6 | 66.6 | 94.9 | 55.5 | 49.5 |
  | DINOv2-L | 51.0 | 67.2 | 98.5 | 53.4 | 49.7 |
  | Ours e240 | n.a. | 65.8 | 96.3 | 51.9 | 49.4 |

  The first fit split per video at random: siblings of a matched set (near-identical, opposite
  labels) then sit on both sides and every model scored below chance (IntPhys 2 23-30%,
  InfLevel solidity 37-42%); kept as physics_probe_random.json. With the grouped split every
  model is at chance on IntPhys 2, and InfLevel continuity (~98% even for the image model
  DINOv2) is a low-level visibility cue, not physics. So this probe does not separate the
  models; the paper's V-JEPA 2-L 71.4 (average over its benchmarks) uses the original IntPhys,
  per-block probes and 256 px, none of which we can match exactly. The zero-shot surprise
  metric below is the physics number to rely on.
  Zero-shot surprise results (2026-10-29), pair-wise %, max surprise and min over context lengths
  (best single context in brackets):

  | Model | IntPhys 2 avg | InfLevel avg | InfLevel continuity / gravity / solidity |
  |---|---|---|---|
  | V-JEPA 2-L | 54.1 (53.5) | 54.2 (54.6) | 64.1 / 49.5 / 49.1 |
  | V-JEPA-L | 51.1 (51.9) | 49.8 (53.8) | 52.0 / 50.2 / 47.1 |
  | VideoMAE-L (pixels) | 50.8 (51.2) | 56.4 (57.1) | 70.5 / 48.1 / 50.4 |
  | Ours e240, unmasked | 49.9 | 48.2 | 44.4 / 52.5 / 47.6 |
  | Ours e240, masked (diagnostic) | 52.6 | 52.4 | 56.1 / 50.1 / 50.9 |

  Ours (2026-10-30) has no context sweep: 16-frame windows at 224 px, stride 4 (step 4 on
  IntPhys 2, 2 on InfLevel), teacher targets. Unmasked = full past visible, each next frame
  predicted (the baselines' setting); masked = the training 90% tube masks, scored at hidden
  sites, mean over 4 fixed masks. Both are at chance. Its next-frame error (0.78-0.82, squared
  distance of normalized features) is worse than copying the last frame (0.46-0.54).

  IntPhys 2 paper (best of ~12 hyper-parameter runs): V-JEPA 2-h 57.5, V-JEPA-h + RoPE 53.8,
  VideoMAE V2-g 53.8. Garrido et al.'s InfLevel numbers (V-JEPA 62% flagship, 72.1 permanence for
  their ViT-L) are for their own pretrained models, not the released V-JEPA-L. Gravity and
  solidity are at chance for every model; only continuity carries signal, and the pixel model
  leads on it.
  Zero-shot surprise, scripts/natural/eval_surprise.py (run 2026-10-28): IntPhys 2 per the
  official IntPhys 2 surprise code (6 fps, V-JEPA 2 48-frame windows at 256 px with contexts
  12-42, V-JEPA / VideoMAE 16-frame windows at 224 with contexts 4-14, max-context mode), InfLevel
  per Garrido et al. (their pair CSVs, post-priming frames, V-JEPA 32-frame windows contexts 4-20,
  VideoMAE 16-frame windows contexts 2-10). V-JEPA 2 runs through the official vjepa2 modules (the
  IntPhys 2 repo's copy has its own RoPE module); the official IntPhys 2 wrapper sizes its time
  masks for 224 px even at 256, which we do not copy. Predictors checked against trivial
  baselines on IntPhys 2: L1 0.55-0.61 vs 0.63-0.66 for the mean target token and 0.75-0.84 for
  copying the last context frame. Pair-wise accuracy per context, best context and min over
  contexts, on max and mean surprise.
  Zero-shot surprise (Garrido et al. 2025, third_party/jepa-intuitive-physics) is possible only for
  models with a predictor or decoder: ours, V-JEPA, V-JEPA 2, VideoMAE (pixel decoder).

## Comparison with published numbers (2026-10-28)

Published frozen attentive probes (K400 16x8x3, SSv2 16x2x3, 224 px unless noted) against ours:

| Model | K400 paper / ours | SSv2 paper / ours | IN1K paper / ours | Source |
|---|---|---|---|---|
| V-JEPA ViT-L | 80.8 / 71.4 (1 train seg; cached 63.3) | 69.5 / 67.5 (cached 58.4) | 74.8 / 75.6 (cached 58.7) | V-JEPA Tab. 5 |
| V-JEPA 2 ViT-L | 85.1 / 73.9 (cached 65.3) | 73.7 / 68.5 (cached 60.0) | 83.5 / 81.0 (cached 67.0) | V-JEPA 2 Tab. 4: 4-layer probe, 256 px |
| LeVJEPA-VideoMix-L | not reported / 62.9 (cached 45.9) | 55.0 / 50.0 (cached 44.3) | 69.5 / 69.3 (cached 58.5) | LeVJEPA, ViT-L 100 ep combined corpus |
| VideoMAE ViT-L | 77.8 / 76.1 | 65.5 / 63.8 | 71.1 / 71.7 | V-JEPA Tab. 5 |
| VideoMAE V2 | ViT-g only: 71.2 / 82.8; ours (L) 81.2 | ViT-g: 61.2 / 56.1; ours (L) 55.3 | ViT-g: 71.4; ours (L) 65.5 | V-JEPA Tab. 6 / V-JEPA 2 Tab. 4 |
| DINOv2 | ViT-g only: 83.4; ours (L) 82.3 | ViT-g: 50.6; ours (L) 45.1 | ViT-g: 86.2; ours (L) 86.3 | V-JEPA Tab. 6 |

IN1K and SSv2 reproduce within ~1-5 points; K400 is 9-11 points low for V-JEPA and V-JEPA 2,
from the single training segment (VideoMAE loses 1.7).

Physics, Esmati et al. 2026 (arXiv 2606.05328), bar heights read from the paper's SVGs
(Fig. 2 and App. D Fig. 14), against our grouped-split probe:

| | V-JEPA 2-L paper / ours | VideoMAE-L paper / ours |
|---|---|---|
| InfLevel continuity | 97.3 / 98.4 | 94.7 / 97.9 |
| InfLevel gravity | 56.4 / 55.8 | 57.3 / 57.3 |
| InfLevel solidity | 52.5 / 51.5 | 54.3 / 51.5 |
| IntPhys 2 permanence | 63.0 / 49.0 | 64.5 / 50.2 |
| IntPhys 2 immutability | 67.8 / 50.0 | 65.5 / 52.3 |
| IntPhys 2 solidity | 56.8 / 50.7 | 55.1 / 53.2 |
| IntPhys 2 continuity | 61.3 / 51.7 | 61.2 / 48.3 |
| IntPhys (original) perm / shape / cont | 81.9 / 73.2 / 61.9 / n.a. | 66.8 / 68.6 / 60.8 / n.a. |

InfLevel reproduces within ~1-3 points (kept on our protocol, eval_physics.py).

IntPhys 2 now follows the paper's own code (decided 2026-10-28; the original IntPhys it reports in
Fig. 2 is not downloadable, so matching the protocol is what we can defend):
scripts/natural/eval_intphys2_paper.py, after github.com/ParsaEsmati/videodiffusionphysics
(encode_video_encoder.py + probe_pairwise.py). Native clip per encoder sampled with linspace over
the whole video and squashed square (V-JEPA 2 64 f / 256 px, others 16 f / 224), fp32; mean-pooled
output of all blocks + final output; one Linear(D,2) per block, Adam 1e-3, batch 4, 50 epochs, raw
features; per condition; scenario-grouped (SceneIndex) 60/40; seeds 0-4; mean over blocks.
The paper text says 60/40 and batch 4, its code defaults (used by its README) 80/20 and 32.

| IntPhys 2, paper / ours pair-wise | permanence | immutability | solidity | continuity | average |
|---|---|---|---|---|---|
| V-JEPA 2-L | 63.0 / 62.1 | 67.8 / 67.1 | 56.8 / 53.5 | 61.3 / 65.2 | 62.2 / 62.0 |
| VideoMAE-L | 64.5 / 62.4 | 65.5 / 63.8 | 55.1 / 55.3 | 61.2 / 65.5 | 61.6 / 61.8 |

All six references (pair-wise, mean over blocks; per-video in brackets, all at chance):

| IntPhys 2 probe | permanence | immutability | solidity | continuity | average |
|---|---|---|---|---|---|
| LeVJEPA-L | 64.4 | 65.7 | 52.7 | 68.5 | 62.8 (51.1) |
| V-JEPA-L | 65.6 | 67.4 | 52.1 | 64.7 | 62.5 (50.3) |
| VideoMAE V2-L | 62.9 | 65.3 | 55.1 | 65.2 | 62.1 (51.7) |
| V-JEPA 2-L | 62.1 | 67.1 | 53.5 | 65.2 | 62.0 (51.6) |
| VideoMAE-L | 62.4 | 63.8 | 55.3 | 65.5 | 61.8 (50.9) |
| DINOv2-L | 49.7 | 54.2 | 56.1 | 51.3 | 52.8 (50.3) |
| Ours e240 (16 f / 224) | 51.8 | 57.0 | 52.9 | 59.3 | 55.2 (50.3) |

The video encoders tie at 62 +- 0.5; the image model is ~10 points lower. Ours sits between:
only its last 2-3 blocks carry signal (per-video 52-53%), the rest are at chance.

App. D Fig. 14 is labelled per-video accuracy, but its values are the pair-wise ones: per-video
accuracy from the same probes is 51.6 (V-JEPA 2) / 50.9 (VideoMAE), i.e. chance. Scene-to-scene
variation swamps a global threshold; comparing the possible and impossible videos of one scene
removes it. We report pair-wise for IntPhys 2 and note the label. Diagnostics that ruled out other
explanations: our sklearn probe on every block's mean-pooled 7.5 fps / 224 px features is at chance
per-video for every block (V-JEPA 2 best block 51.9, VideoMAE 51.6;
scripts/natural/eval_physics_layers.py).

## Direction Q, ViT-T on 20% K710: full suite (2026-11-09)

nlpQm_k710_vitt (Q: 32 random directions x 8 bins, cross-entropy, clip-context teacher) and
nlpQfm_k710_vitt (Q-frame: each target frame encoded alone), 240 epochs, masked (0.9),
coding-rate guard at scale 0.05; e240 student encoder, the baselines' scripts and settings.
Surprise dropped for now (Q's predictor only ever saw masked input: unmasked surprise is above
chance cross-entropy; eval_surprise.py has a Q path, unused).

| e240 student | K400 att / lin | SSv2 att / lin | IN1K att / lin | DAVIS J&F (J / F) | InfLevel avg (cont / grav / sol) | IntPhys 2 pair-wise (per-video) |
|---|---|---|---|---|---|---|
| Q, ViT-T | 28.5 / 17.7 | 18.8 / 7.5 | 30.5 / 15.8 | 42.1 (41.7 / 42.4) | 62.5 (85.1 / 52.8 / 49.7) | 50.8 (50.1) |
| Q-frame, ViT-T | 29.5 / 18.0 | 18.7 / 7.0 | 29.9 / 15.2 | 48.3 (47.1 / 49.5) | 62.1 (85.1 / 53.0 / 48.3) | 49.9 (50.0) |
| Ours B, Mix 3 ViT-L SIGReg | 59.2 / 33.3 | 47.9 / 20.1 | 62.8 / 31.4 | 55.5 (54.0 / 57.0) | 65.8 (96.3 / 51.9 / 49.4) | 55.2 (50.3) |

Cached probes (attentive / conv-attentive), with the regression reference nlpBm_k710_vitt
(K400 / IN1K only; its evals predate the SSv2 cache):

| Epoch | Q K400 | Q IN1K | Q SSv2 | Q-frame K400 | Q-frame IN1K | Q-frame SSv2 | B K400 | B IN1K |
|---|---|---|---|---|---|---|---|---|
| 60 | 27.3 / 28.3 | 32.3 / 38.2 | 18.4 / 23.8 | 27.9 / 29.7 | 33.3 / 39.3 | 18.2 / 24.5 | 27.1 / 27.5 | 31.6 / 37.9 |
| 120 | 28.4 / 28.7 | 32.9 / 38.8 | 18.9 / 24.1 | 27.8 / 29.9 | 33.8 / 39.6 | 18.3 / 24.4 | 27.5 / 28.4 | 31.8 / 38.3 |
| 180 | 27.7 / 29.9 | 33.4 / 39.3 | 18.2 / 23.5 | 27.9 / 29.5 | 33.1 / 39.0 | 18.0 / 22.4 | 27.4 / 27.8 | 32.1 / 38.3 |
| 240 | 29.0 / 29.8 | 33.9 / 39.5 | 19.4 / 24.5 | 27.6 / 29.2 | 32.8 / 38.9 | 17.7 / 23.2 | 27.5 / 28.4 | 32.0 / 38.6 |

The full-suite rows have no same-size control yet (nlpBm_k710_vitt has only the cached probes).

## ViT-L plan

- Memory is not the reason for multiple nodes: ViT-L (304M) with 90% masking
  at global batch 256 is 32 clips per GPU on one 8xH200 node, well inside 141 GB.
  `slurm/nlp_multinode.sbatch` (c10d rendezvous, IB) runs the same trainer on
  2-4 nodes when wall-clock matters; the trainer keeps the global batch.
- Cost: ~3.5x ViT-B per clip (304M vs 86M parameters). Hours for a run =
  26.8M x epochs/240 / (3.5 x ViT-B clips/s per node x nodes); fill in once the
  first ViT-B link reports its throughput.
- Data: the mix the ViT-T screen and the ViT-B e60/e120 evals favour; 384-px stores.
- Scheduling: our fair-share (0.25) makes whole multi-node allocations slow to
  start (8 H200 nodes in commons, ~300 jobs pending). Default: one node, chained
  24 h links; ask for 2 nodes only if the single-node estimate exceeds the time
  left before the deadline.
- Learning rate: every size so far used Muon lr 1e-3; check ViT-L with a short
  smoke (loss and coding rate over the first few thousand steps) before
  committing the full run.

## Open items

- Contamination: VidaForge's source videos (LLaVA-OneVision-2, YouTube) and PE
  Video may overlap K400 val; check before any headline number.
- GPU access: RANGE fair-share is 0.25 with ~290 jobs pending.
