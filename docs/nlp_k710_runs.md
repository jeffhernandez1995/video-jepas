# Next-latent prediction on 20% K710: Directions A and B at T/S/B

Six runs: `nlpA_k710_{vitt,vits,vitb}` (categorical) and
`nlpB_k710_{vitt,vits,vitb}` (regression with the calibrated coding-rate
guard), each on the same GPUs as its LeVJEPA counterpart (2/4/8).

Written 2026-09-28. The arm-to-arm counterpart of the LeVJEPA trunks in
`levjepa_matched_protocol.md`. Code: `scripts/natural/nlp.py` (model, targets,
losses), `scripts/natural/train_nlp.py` (DDP trainer),
`slurm/nlp_chained.sbatch` (24-hour links), `slurm/nlp_smoke.sbatch`.

## What is identical to the LeVJEPA arm

- **Data**: the same Lance store (`k710/train_20pct.lance`, 111.5k clips), one
  clip per video per epoch, 16 frames at stride 2 of the 15 fps store
  (7.5 fps, about 2 s), LeVJEPA's clean global view (RandomResizedCrop 224,
  area 0.8 to 1.0, bicubic), ImageNet normalization. No photometric
  augmentation, no flips.
- **Encoder**: LeVJEPA's own `VisionTransformer` class and constructor
  arguments (patch 16, 16 frames, tubelet 1, RoPE, block-causal, CLS prefix),
  so checkpoints load into the same evaluators unchanged.
- **Schedule length**: 240 epochs; milestones at 60/120/180/240.

## What differs, by design

- **All 3,136 patch tokens** are inputs; LeVJEPA sees 5%. From frame 2 on,
  every token is also a target.
- **Objective (Direction A)**: EMA teacher (full encoder, final layer, L2
  normalized); linear prototypes (K = 16,384) on teacher features, trained by
  their own clustering loss; Sinkhorn-Knopp targets at temperature 0.06;
  student cross-entropy at 0.12 against the code at (t+h, i) for
  h in {1, 2, 4}, weights 1 / 0.15 / 0.15. Frame 0 is burn-in context only.
- **Predictor**: block-causal self-attention over the student's final-layer
  tokens, width = encoder width, a 4-block shared trunk and a 2-block branch
  run once per horizon with a learned horizon embedding. Output at (t, i)
  predicts (t+h, i). This transforms memory (the synthetic stage's
  requirement) at about 0.8x an encoder pass instead of the 3x that one
  query token per target would cost at 3,136 tokens and three horizons.
- **Balancing is per spatial site**, pooled over the global batch and the 14
  target frames: 256 x 14 = 3,584 samples per site for 16,384 codes, the same
  samples-per-code ratio (about 1:4) as the synthetic per-position arm that
  produced tracking. Per-(t, i) balancing would give 256 samples for 16,384
  codes, which forces every target to spread over at least 64 codes. Column
  sums (196 x 16,384 per iteration) and the stabilizing column max are
  all-reduced; logits are never gathered.
- **Batch 256 clips** (LeVJEPA 3,072): 435 optimizer steps per epoch, 104k in
  total, which an EMA-teacher method needs. Muon on encoder and predictor
  matrices, AdamW elsewhere, lr 1e-3, 5 warmup epochs, cosine to 1e-5,
  weight decay 0.05, clip 1.0, bf16. Teacher momentum cosine 0.996 to 1.

## Weight averaging: no Polyak average; evaluate student and teacher

The LeVJEPA evaluation EMA had no warm-up and was 42% initialization at the
end of training (the EMA trap). Here there is no separate evaluation
average. The EMA teacher is itself an average of the student, but its
momentum starts at 0.996 (a window of about 250 steps), so the
initialization is gone within the first two thousand steps, and it tightens
only as training converges. Every milestone evaluates both the student
(raw) and the teacher encoder, and both are reported.

## Video storage: no rebuild

LeVJEPA was loader-bound because each clip cost about 0.7 core-seconds
(16 JPEG decodes plus 11 views with hue jitter) while its GPUs saw 5% of the
tokens. Here one clip is one crop, about 0.1 core-seconds, and the GPU cost
per clip is roughly 20x LeVJEPA's (all tokens, a teacher pass, a predictor,
a 16k-way head). The runs should therefore be GPU-bound on the existing
384-short-edge store, as long as CPUs are allotted per job rather than per
GPU (ViT-T 104 CPUs on 2 GPUs, ViT-S 104 on 4, ViT-B 208 on 8). The trainer
logs `data_wait_frac`; if it rises above a few percent, the fixes in
`levjepa_loader_bottleneck.md` apply (a 256-short-edge store, GPU JPEG
decoding), in that order.

## Evaluation

- **Full cached protocol** at every milestone, student and teacher: the
  complete IN1K (1.28M) and K400 (241k) training sets decoded once into
  fixed caches (`build_eval_caches.py --only k400_train_full in1k_train_full`,
  about 480 GB), one encoder pass, linear (mean and CLS), attentive and
  conv-attentive probes. The learning rate is now chosen on 10k held-out
  training items and validation is only reported, never used for selection
  (the subset protocol selected on validation). Submitted automatically by
  `scripts/range/watch_full_evals.sh`.
- **The LeVJEPA trunks are re-evaluated on the same protocol** with raw
  weights (`FAMILY=levjepa`), so both arms are compared on one ruler.
- **Exact V-JEPA probe** (`probe_vjepa.py`) at epoch 240 for the headline
  IN1K number, as done for LeVJEPA ViT-B (48.81 raw).
- Deferred: the layer sweep (layers 6/9/12) that the synthetic stage found
  informative for autoregressive encoders; TAP-Vid-DAVIS correspondence;
  SSv2 once available.

## Direction B

Same encoder, teacher, predictor, data and schedule as A. The loss is the
squared distance between the normalized prediction and the normalized
teacher feature at (t+h, i), plus the coding-rate guard
-gamma * 0.5 logdet(I + D/0.05^2 Cov) on 4,096 sampled final-layer student
tokens per GPU, with the covariance all-reduced over ranks (autograd-aware),
so the guard sees the global batch.

Gamma is calibrated per model size exactly as in wave 2: an unregularized B
run of 200 steps, then gamma = 0.1 x ||d prediction / d encoder|| /
||d coding rate / d encoder|| on the 200th batch, both unclipped, prediction
gradients averaged over ranks and coding-rate gradients summed (it is one
global quantity). The first link of each B chain does this into
`<run>/calibration/calibration.json`, then starts the real run from scratch
with that gamma. B is the control that decides whether the categorical head
does more than prevent collapse; it is cheaper than A by the prototype head
and the Sinkhorn pass.

## Step cost and the two kernel changes (2026-09-29)

First link of `nlpA_k710_vitt` on 2 H200s: 114 clips/s (57 per GPU), data
wait 0.02%, 102 GiB peak, about 65 hours for 240 epochs. A per-section
benchmark (`scripts/natural/bench_nlp.py`, one 47 GB GPU) showed the encoder
is not where the time goes: for ViT-T at batch 32 the step was 1,193 ms, of
which the student encoder forward+backward is 275 ms, the teacher 117 ms, the
per-site Sinkhorn 208 ms, and the predictor with the 16k-way student and
cluster cross-entropies about 590 ms. Those are memory-bound fp32 passes over
(rows x 16,384) tensors. Two changes, same math, now defaults:

- **flex-attention** for the block-causal mask (`nlp.enable_flex_attention`,
  on unless `--no-flex`): skips future-frame blocks; teacher features agree
  with dense-mask SDPA to cosine 1.00000, max |diff| 5e-4 in bf16.
- **compiled element-wise tails** of the soft cross-entropies and the
  Sinkhorn iterations (`nlp.fused`, on unless `NLP_FUSE=0`). Inductor needs a
  C++ compiler, which RANGE compute nodes lack outside a login shell; `nlp.py`
  points `CXX` at `/opt/apps/software/GCCcore/14.2.0/bin/g++` itself.

| Model (batch) | Baseline ms/step | Optimized ms/step | Speedup |
| --- | ---: | ---: | ---: |
| ViT-T (32) | 1,193 | 754 | 1.58x |
| ViT-S (16) | 896 | 635 | 1.41x |
| ViT-B (8) | 779 | 605 | 1.29x |

Every link that starts from 2026-09-29 uses them; the ViT-T A link already
running keeps the old kernels until its 24-hour limit.

## Direction A collapsed; A' adds B's guard (2026-09-30)

A's probes fell between e60 and e120 at ViT-S (K400 linear 14.4 to 12.4, IN1K
conv-attentive 26.4 to 23.6). The batch code perplexity had been 16k (all
codes) until epoch 32, then fell steadily to 1.1k by epoch 160, while the
prediction loss kept falling. ViT-T followed the same path, somewhat later.

The prototypes in the milestone checkpoints show that this is a collapse of
the representation's dimensions, not of the codebook. Their effective rank
(exp entropy of the spectrum of the unit-normalized prototype matrix) fell
from 368 of 384 at epoch 15 to 183 at 60, 57 at 120 and 42 at 150. The median
cosine to the nearest other prototype rose from 0.21 to 0.63. Prototype norms
barely vary (coefficient of variation 3%) and slightly shrink, the bias range
stays about 1, and no two prototypes are duplicates (none above 0.99 cosine).
The prototypes are fit to the teacher's features, so they span the space
those features occupy, which is about 42 dimensions by epoch 150.

Per-site Sinkhorn balancing keeps codes in use, but it cannot stop features
from collapsing into a few dimensions. A low-dimensional teacher is easier to
predict, so the prediction loss rewards the collapse. B has the coding-rate
guard against exactly this: its coding rate holds (T about 530, S about 1,100)
and its probes plateau but never fall.

A' (`nlpA2_k710_vits`, 4 GPUs) is A plus:

- **B's coding-rate guard** on the student's final-layer tokens, with gamma
  calibrated exactly as for B (`GUARD=1` in `nlp_chained.sbatch`: 200
  unregularized A steps, then 0.1 x the gradient-norm ratio). The cluster
  term reaches only the prototypes, so the prediction gradient on the encoder
  is A's alone.
- **Sinkhorn iterations raised from 3 to 10** (`--set sinkhorn_iters=10`). A's
  target entropy had fallen to 1.2 nats, below the log(16,384 / 3,584) = 1.5
  that a converged balance allows, so 3 iterations were no longer balancing.
- **Two logged ranks** (every 20 steps, all runs): `feature_erank` (centered,
  on 8,192 of rank 0's teacher tokens) and, for A, `prototype_erank`. The
  collapse is then visible within 40 epochs instead of at the e120 probe.

Schedule and everything else are A's. Pass criterion: prototype and feature
rank hold past epoch 40, and the e60 full probe at least matches A-S e60
(14.4/16.5/17.7 K400, 16.0/20.7/26.4 IN1K). If the rank holds but probes do
not improve, the next change is K = 4,096, so that each per-site Sinkhorn
problem has more samples than codes. The A runs at T, S and B were cancelled.
Their checkpoints are kept: A-T at epoch 105, A-S at 161.

## Direction B plateaus because copying solves the task (2026-10-01)

B-T finished at 15.6/18.1/18.7 K400 and 14.1/21.2/25.2 IN1K (linear /
attentive / conv-attentive), unchanged since e60 and 7 to 12 points below
LeVJEPA-T. B-S gained about 3 points from e60 to e180 and then stopped: it
finished at 21.5/22.7/23.2 K400 and 21.2/27.8/31.0 IN1K, against LeVJEPA-S's
30.4/33.8/35.2 and 32.1/43.2/47.1. The
guard did its job: the teacher's feature rank at the end of B-S is 243 of 384.

`scripts/natural/diag_copy_baseline.py` compares the trained predictor with
a trivial one that copies the teacher's feature at the same site of the
current frame, on the same (source, target) pairs as training (6 training
clips, CPU; squared distance between unit vectors, so 2 is orthogonal):

| Checkpoint | h | Predictor | Copy | Clip mean | Predictor / copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| B-T e60 | 1 | 0.042 | 0.043 | 0.64 | 1.00 |
| B-T e60 | 4 | 0.100 | 0.152 | 0.62 | 0.66 |
| B-T e240 | 1 | 0.030 | 0.042 | 0.51 | 0.71 |
| B-T e240 | 4 | 0.106 | 0.188 | 0.50 | 0.56 |
| B-S e60 | 1 | 0.025 | 0.021 | 0.60 | 1.22 |
| B-S e60 | 4 | 0.063 | 0.072 | 0.60 | 0.88 |
| B-S e240 | 1 | 0.043 | 0.044 | 0.82 | 0.99 |
| B-S e240 | 4 | 0.135 | 0.146 | 0.82 | 0.92 |

At 7.5 fps a site's feature one frame later has cosine 0.98 to 0.99 with its
current value. Copying therefore removes 93 to 97% of the h = 1 loss, which
carries 77% of the objective's weight, and the trained ViT-S predictor is no
better than copying at any horizon. The encoder sees every token, so nothing
forces it to infer content it was not shown. Direction A has the same
structure (the code at (t+1, i) is usually the code at (t, i)), so A' can
remove the collapse but should not be expected to pass this ceiling.

B at ViT-B was cancelled (never started). Candidate changes, cheapest first:
longer and equally weighted horizons with a lower frame rate (config only);
weighting the loss by how much the target changed; hiding most input tokens
from the student while the teacher sees all (the LeVJEPA/V-JEPA ingredient).

## A' result: the guard did not prevent the collapse (2026-10-05)

`nlpA2_k710_vits` (A + calibrated coding-rate guard, gamma 1.11e-5, Sinkhorn
10 iterations) finished all 240 epochs (2 links, about 39 h on 4 H200s).

| Epoch | Prototype rank | A-S prototype rank | Teacher feature rank | Code perplexity | Coding rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 371 | 368 | 196 | 16,309 | 1,010 |
| 30 | 324 | 314 | 215 | 16,303 | 1,051 |
| 45 | 244 | 248 | 191 | 16,307 | 1,043 |
| 60 | 171 | 183 | 170 | 16,302 | 1,029 |
| 120 | 45 | 57 | 112 | 16,143 | 993 |
| 240 | 26 | n/a | 110 | 15,946 | 979 |

Full protocol, student (K400 linear / attentive / conv-attentive | IN1K):

| Epoch | A' | A (no guard) | B |
| ---: | --- | --- | --- |
| 60 | 14.0/16.1/16.8 \| 15.5/20.9/26.1 | 14.4/16.5/17.7 \| 16.0/20.7/26.4 | 18.9/20.6/20.3 \| 18.4/25.7/29.1 |
| 120 | 11.7/15.0/15.4 \| 12.6/17.8/22.1 | 12.4/13.7/15.0 \| 13.3/18.6/23.6 | 21.1/22.0/22.2 \| 20.5/28.0/31.5 |
| 180 | 11.5/15.0/15.4 \| 11.9/16.1/19.4 | cancelled | 21.7/22.7/23.1 \| 21.3/28.5/31.5 |
| 240 | 11.4/14.6/15.2 \| 11.8/15.5/18.9 | cancelled | 21.5/22.7/23.2 \| 21.2/27.8/31.0 |

The teacher stays within 0.6 of the student at every milestone.

- Ten Sinkhorn iterations fixed the code usage: perplexity stayed at 16k
  for the whole run, where A fell to 1.2k.
- The prototype rank fell almost exactly as in A, and the teacher's feature
  rank fell from 215 at epoch 30 to 110. The coding rate held near 1,000
  throughout, so a guard on the student's coding rate does not stop the
  teacher's features from losing dimensions here. The probes fall from e60 in
  step with them.
- A' fails both criteria: at e60 it does not beat A-S (K400 0.4 to 0.9 lower)
  and it ends 10 points below B on K400 and 12 on IN1K conv-attentive.

With balanced codes and the guard on, the categorical objective still
degrades the representation. Together with B's copy-baseline result, this
points at the task (copying the current site predicts the next frame) rather
than at the collapse controls. Direction A in its all-tokens form is closed.

## Masked next-latent prediction: three ViT-T arms (2026-10-05)

Masking is the only change; data, encoder, EMA teacher (full clip), predictor,
horizons, burn-in, optimizers and schedule are as before.

- **Mask** (`nlp.tube_block_masks`): per clip, rectangles of 15 to 40% of the
  14 x 14 grid (aspect 0.75 to 1.5) are hidden until 20 of 196 sites remain
  (89.8% hidden). The same sites are hidden in every frame (a tube), so a
  hidden site cannot be copied from its own past. Blocks rather than random
  scatter keep a hidden site's neighbours hidden too: 85% of hidden sites
  have all four neighbours hidden.
- **Student encoder** (`nlp.encode_visible`): only the 20 visible sites of
  each frame (320 tokens plus CLS), with the encoder's own RoPE positions and
  block-causal mask. With every site visible it matches the full encoder
  exactly (max difference 0).
- **Predictor**: visible encoder outputs are scattered into the full grid and
  hidden positions get a learned mask token. The predictor is unchanged: its
  output at (t, i) predicts the teacher at (t + h, i), block-causal, so a
  prediction uses only visible tokens from frames up to t.
- **Loss**: at hidden sites only. B regresses the normalized teacher feature;
  A uses the per-site Sinkhorn codes of the full-clip teacher (balancing
  unchanged) with cross-entropy gathered at hidden rows.

| Run | Arm | Horizons | Guard | Sinkhorn |
| --- | --- | --- | --- | --- |
| `nlpBm_k710_vitt` | B | 1, 2, 4 | calibrated (as B) | n/a |
| `nlpAm_k710_vitt` | A (lean) | 1, 2, 4 | none | 10 iterations |
| `nlpBm0_k710_vitt` | B | 0 only | calibrated | n/a |

The h = 0 arm is causal V-JEPA: it predicts the hidden sites of the current
frame from the visible past and present. If masked B with h = 1, 2, 4 does
not beat it, forecasting adds nothing beyond masking. Compare at e60 with B-T
(15.8 K400 linear) and LeVJEPA-T (15.6), and watch whether the curves still
rise after e60, where B and A stalled. All on 2 H200s, 3 chained links each.
