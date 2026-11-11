# Next-latent prediction on natural video

Written 2026-09-15. Consolidates the synthetic-benchmark findings into two
frozen recipes and a transfer plan. The remaining synthetic runs (predictor
fix, seed 1, transport audits on wave-1 checkpoints) only decide the
predictor form; everything else below is fixed now.

## What transfers from the synthetic stage

Established, and carried over unchanged:

- Causal next-latent prediction with frame-level block-causal attention, one
  shared crop per clip, no augmentation invariance, burn-in on the first
  frame, causal evaluation only. Bidirectional readout is dead.
- The EMA teacher is the tokenizer. No external discrete video tokenizer,
  ever: it would add a full network in front of each batch and hand back
  reconstruction-shaped codes.
- The predictor must transform memory. Cross-attention-only predictors at
  small depth push the prediction task into the encoder's final layer and
  collapse it. Use self-attention over memory plus queries, or a
  CAPI-depth cross-attention predictor. Depth is the knob, not width.
- Per-position Sinkhorn is the default for the categorical head. Pooled
  balancing leaks position and the leak grows through training.
- Multi-horizon queries (1, 2, 4) with weight 0.15 help both the objective
  and the probes once the predictor exists.
- The categorical objective produces correspondence; regression does not.
  Transport at layer 5 of the categorical arms is above the supervised
  digit classifier; the regression arm is at random level.
- Static tokens stay in the loss. Dropping redundant *inputs* is free;
  dropping their targets costs at the final layer.
- Linear probes are uninformative for autoregressive encoders. Attentive
  probes with a layer sweep, reported at the best layer and the final
  layer, are the protocol.
- Seed spread on ViT-Tiny reached six points at 80 epochs. Every adopted
  comparison needs two seeds, and single-seed gaps under five points are
  unresolved.

Established but not carried over: the pixel k-means codebook. It is the
strongest final-layer result on the synthetic data because its target is
stationary, and it is hopeless on natural patches. Its role on natural
video is the stationarity ablation, done with a frozen teacher snapshot
instead of pixels.

Not established: the coding-rate guard fixes teacher variance decay but
not final-layer collapse; registers are confounded with the predictor
change; the 3× γ arm was the only one above the guard's baseline at layer
12 and is single-seed. These become ablations, not defaults.

## The two directions

**Direction A, categorical next-latent prediction.** Student encoder,
EMA teacher at the final layer, memory-transforming predictor, horizons
{1, 2, 4}, linear prototype head on L2-normalized teacher features trained
by its own clustering loss, per-position Sinkhorn-Knopp targets at
temperature 0.06, student cross-entropy at 0.12, K = 16,384. This is the
candidate. It is CPC's objective with CAPI's target and a causal encoder,
and it is the recipe that produced tracking.

**Direction B, regression next-latent prediction with a spread guard.**
Same encoder, teacher and predictor; squared distance between normalized
prediction and normalized teacher feature; coding-rate regularizer on the
student encoder's final-layer features at the calibrated γ. This is the
SimDINO analogue and the control that decides whether the categorical head
is doing more than preventing collapse. It is cheaper by the prototype
head and the Sinkhorn pass, which at K = 16,384 is not negligible.

Both directions run at every scale and on every dataset below. A is
reported; B is the ablation that must accompany it.

## Data

Tamara state on 2026-09-15: Kinetics-400 is complete on scratch
(241,237 train and 19,881 val mp4s, 433 GB, class folders). SSv2 and
TAP-Vid-DAVIS are absent and need downloading. No video decoder is
installed in the training environment: torchvision 0.26 has no
`read_video`, and PyAV is pip-installable. The host has 104 CPUs and
1.5 TB RAM, so pre-decoding is cheap. Scratch has 389 GB free; about
5 TB of feature caches from finished synthetic experiments are
retirable.

- **Kinetics-400 first, because it is already here.** Appearance-dominated,
  the standard comparison point for V-JEPA-style methods. The first A/B
  pair runs on it while SSv2 downloads.
- **SSv2 is where the claim lives.** Its labels require motion direction,
  so it is the dataset where a temporal objective has to beat a
  view-invariance objective. 169k training clips. Start the download now;
  the K400 pipeline transfers unchanged.
- Clips: 8 frames at 160px short side with center crop to 160×160, patch
  16, so 100 patches per frame and 800 tokens per clip. Stride 4 on SSv2
  (about 2.7 s at 12 fps), stride 8 on K400. 16 frames is the follow-up
  once packing is in.
- Preprocessing: pre-decode four temporal windows per video at 160px,
  deterministic by (video id, window), stored as JPEG frames at quality
  90 in sharded archives with an index. Raw uint8 for K400 at four
  windows is about 590 GB, more than scratch has; JPEG is about 120 GB
  and decodes in loader workers on 104 CPUs. This keeps the
  fixed-benchmark discipline from the synthetic stage and removes video
  decoding from the training loop.
- Probe fitting: features for 241k clips at 800 tokens cannot be cached,
  so attentive probes run with the frozen encoder online. Fit on a
  balanced 60k-clip subset of K400 train (150 per class), evaluate on the
  full 19.9k val with the center window. Readouts: attentive pooling over
  all tokens with one query and class queries, as in V-JEPA, plus the
  convolutional variant on the 10×10 grid.
- Augmentation: one RandomResizedCrop per clip, area 0.4 to 1.0, shared
  across frames, plus horizontal flip on K400 only. Nothing else.
- The blank mask is all ones. No data-specific code paths.

## Model and budget

- ViT-Small/16, 12 blocks, width 384, RoPE, block-causal, 4 registers per
  frame inside the causal structure. ViT-Tiny has no headroom for a
  16k-way head on natural patches.
- Predictor: 6 blocks of width 384 with self-attention over memory plus
  queries under the causal mask, queries never attended by memory. Depth
  6 is half the encoder, which is CAPI's ratio.
- Head for A: normalized bottleneck to 256 dimensions, then 16,384
  prototypes; the logits tensor for 256 clips × 600 queries × 16,384 is
  handled by the chunked cross-entropy from wave 1.
- Teacher momentum: cosine from 0.996 with a floor at 0.9995 so the
  teacher never freezes; the floor is an ablation against the current
  schedule on the first SSv2 run.
- Batch 256 clips, Muon on transformer matrices at 1e-3, AdamW elsewhere,
  weight decay 0.05, 5 warmup epochs, cosine to 1e-5, bf16.
- Budget: wave 2 ran ViT-Tiny at 1568 tokens at about 80 clips per second
  per A40 including teacher and predictor. ViT-Small has about four times
  the FLOPs per token and half the tokens, so roughly 35 to 40 clips per
  second per GPU. On four GPUs one K400 epoch (241k clips) is about
  25 minutes and 100 epochs about 42 hours; on eight GPUs half that. Two
  arms run concurrently on four GPUs each. This is the multi-GPU path the
  synthetic stage deliberately avoided.
- DDP details. Per-position Sinkhorn needs the global batch per position,
  but only its marginals: each of the three iterations all-reduces the
  positions × codes column sums (800 × 16,384 floats, 52 MB), and the
  row normalization is local. Never all-gather the logits. The
  coding-rate covariance all-reduces a 384 × 384 sum. The clustering loss
  averages through DDP as usual.
- Head memory at K = 16,384: 1,400 queries per clip over the three
  horizons, so per-GPU batch 64 gives a 2.9 GB bf16 logit tensor, handled
  by the chunked cross-entropy from wave 1.

## Evaluation

- **Attentive probe** on SSv2 and K400 with the fixed 7×7 regional heads,
  causal encoding, layers 3, 6, 9, 12, reported at the best layer and at
  layer 12, two seeds. The linear probe is recorded and ignored.
- **Correspondence** on TAP-Vid-DAVIS: the transport audit re-pointed at
  real annotations, direct matching at layers 6 and 12, PCK at 8 and 16
  pixels plus average Jaccard, causal and full-context conditions. This
  is where the synthetic transport result either generalizes or does not.
- **Dynamics diagnostic**: per-horizon cross-entropy on held-out clips,
  and the position and label mutual information of the codes.
- **Baselines at matched compute**: random initialization; the existing
  LeVJEPA masked-prediction recipe on the same clips, which is the
  view-invariance objective this is meant to beat; and frozen DINOv2-S as
  the external correspondence reference, evaluated but not trained.

## Ablations, run on SSv2 only, single seed unless promoted

| Ablation | Question it answers |
| --- | --- |
| Self-attention predictor at depth 3 versus 6 versus cross-attention at 12 | Where the final-layer collapse boundary is on real data |
| Registers 0 versus 4 per frame | The confound left open in wave 2 |
| Coding-rate guard on A | Whether the guard helps once the predictor is right |
| Pooled versus per-position Sinkhorn | Whether the position leak recurs with natural patches |
| K in {4,096, 16,384, 65,536} | The vocabulary-scaling question from the start of this thread |
| Horizons {1} versus {1, 2, 4} versus {1, 2, 4, 8} | Multi-horizon on a multimodal future |
| Frozen-teacher codebook | Stationarity versus loss form, on feature codes |
| Redundant-input dropping with varlen attention | The speedup, on data with real static background |
| Momentum floor 0.9995 versus cosine to 1.0 | Whether the frozen-teacher end of training is the drift |
| 16 frames with packing | The path to learned temporal chunking |

## Order of work

1. Frame caches for SSv2 and K400, the DDP path, and TAP-Vid-DAVIS
   loaders, while the last synthetic runs finish.
2. A and B on SSv2, two seeds each, 100 epochs, with the LeVJEPA and
   random baselines. This is the result.
3. The ablation table on SSv2, prioritized by the predictor row first,
   since it decides the default for everything after it.
4. A on K400 with the winning configuration, for the literature comparison.
5. Learned temporal chunking, only after packing exists and only on the
   winning direction.
