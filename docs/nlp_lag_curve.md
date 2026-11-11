# How long a gap before copying stops solving the target (2026-11-04)

`scripts/natural/diag_lag_curve.py`, Mix 3 video sources, 96 random training-eligible
episodes per source, e240 ViT-L teacher (`nlpBm_mix3_vitl_sigreg`). Outputs in
`/scratch/CLUSTER_USER/diag/lag_curve/` (Tamara and RANGE): `episode_lengths.json`, `lag_curve.json`,
`cuts.jpg`.

## Episode lengths: the mix tops out at 8.5 s

| Source | Weight | Continuous video |
|---|---|---|
| VidaForge | 0.50 | one shot per clip, at most ~8.5 s (11% of source videos have a second clip, possibly a different shot) |
| SSv2 | 0.15 | median 3.7 s (p10 2.5, p90 5.2) |
| K710 | 0.10 | one 8.5 s episode per video (raw clips ~10 s) |
| EgoDex | 0.10 | consecutive episodes stitch: median 8.5 s, 18% of videos >= 16 s |
| DROID | 0.05 | consecutive episodes stitch: median 17 s, 73% >= 16 s, 17% >= 32 s |

Stores cut videos into 64-row episodes (8.5 s at 7.5 fps), but for ~75% of the mix that is
the whole clip. About 5% of training clips could come from >= 16 s of continuous video.
Horizons of tens of seconds need other data.

## Share of the loss removed by copying the same site, teacher features per frame (T = 1)

`removed = 1 - copy / mean`, where mean is the window's normalized mean feature (knows nothing
about the target). All pairs within a shot; static / moving = patch pixel change below the
source's median / above its 90th percentile; nn_global = oracle copy from the best site anywhere
in the earlier frame.

| Gap | K710 | SSv2 | VidaForge | EgoDex | DROID |
|---|---|---|---|---|---|
| 0.13 s | 58 | 62 | 63 | 80 | 75 |
| 0.53 s | 29 | 28 | 33 | 61 | 58 |
| 1.07 s | 16 | 10 | 19 | 51 | 48 |
| 2.13 s | 4 | -6 | 8 | 43 | 38 |
| 4.27 s | -7 | -14 | -2 | 35 | 30 |
| 6.40 s | -11 | n/a | -5 | 27 | 27 |
| 6.40 s, static | 7 | 1 (4.3 s) | 15 | 38 | 40 |
| 6.40 s, moving | -38 | -36 (4.3 s) | -34 | 1 | -34 |
| 6.40 s, nn_global | 24 | 24 (4.3 s) | 27 | 58 | 46 |

On the web sources (75% of the mix) copying stops helping after 1.5 to 4 s; on moving patches it
is worse than the mean beyond ~0.3 s. Static patches stay partly copyable (7-15% at 6.4 s), and
the fixed-camera sources (EgoDex, DROID) keep 27% at 6.4 s. The content is usually still somewhere
in view (nn_global 24-58% at 6.4 s), so long gaps turn copying into search.

## The training targets are far less copyable than the per-frame features

The training teacher encodes 16-frame clips, block-causal. Encoding the first 16 frames of each
window that way, copying removes much less (per frame / clip, %):

| Gap (frames) | K710 | SSv2 | VidaForge | EgoDex | DROID |
|---|---|---|---|---|---|
| 1 | 59 / 29 | 61 / 26 | 62 / 29 | 80 / 51 | 75 / 45 |
| 4 | 30 / 4 | 27 / 1 | 29 / 6 | 59 / 31 | 60 / 28 |
| 8 | 17 / -6 | 9 / -9 | 14 / -5 | 48 / 22 | 50 / 18 |

Same-site cosine one frame apart is 0.52-0.71 in clip context (0.80-0.90 per frame). The 0.98 and
93-97% in `nlp_k710_runs.md` came from the unmasked B teachers: that objective made its own target
space copyable. For the masked ViT-L, copying the true current feature (squared distance
0.59-0.95 at h = 1) is in the same range as the predictor's h1 training loss (0.88), which never
sees the hidden site; the clip mean is 1.17-1.34.

Part of the change between frames is a clock. One frame repeated 16 times (no motion), encoded as
a clip: same-site cosine 0.966 one position apart, 0.918 at 4, 0.842 at 8, 0.632 at 15; frame 0
(which attends only to itself) has cosine ~0.70 with every later position. Over h <= 4 the drift is
small next to the content change, but targets carry frame position as well as content.

## Random-projection bins as categorical targets (`scripts/natural/diag_bins.py`)

Candidate target "Q": 32 fixed random unit directions, each projection cut into 8 bins. 32 episodes
per source, first 64 frames, e240 teacher; raw tokens and unit-normalized targets (x sqrt(D)) give
the same numbers to 0.01, so one row per mode. Clip = 16-frame block-causal clip, frame 0 dropped.

| | per frame (T = 1) | 16-frame clip |
|---|---|---|
| projection std (mean, range over directions) | 0.87-0.94 (0.76-1.07) | 0.95-0.99 (0.86-1.10) |
| abs skew / excess kurtosis | <= 0.09 / -0.07 to 0.00 | <= 0.05 / -0.06 to -0.01 |
| fixed N(0,1)-quantile bins: entropy / log 8, mean (worst direction) | 0.94-0.97 (0.67-0.86) | 0.98-0.99 (0.90-0.97) |
| largest single-bin mass (uniform = 0.125) | 0.32-0.51 | 0.21-0.30 |
| joint code of 4 directions, perplexity of 4,096: fixed / empirical bins | 2,100-3,100 / 3,740-3,920 | 3,420-3,870 / 3,910-3,970 |

Projections are close to N(0, 1) with mild anisotropy (random projections of high-dimensional
features look Gaussian anyway; the per-direction std is the informative part). Fixed Gaussian bins
are near-uniform for clip-context targets; per-frame features have lopsided directions, so bins
should come from running per-direction quantiles (or a running standardization) instead.

Bin persistence at the same site, empirical bins, gain over chance = (P(same bin) - 1/8) / (7/8):

| Gap | per frame: web (K710, SSv2, VidaForge) | per frame: EgoDex, DROID | clip: web | clip: EgoDex, DROID |
|---|---|---|---|---|
| 0.13 s | 0.30-0.32 | 0.37-0.40 | 0.11-0.13 | 0.16-0.18 |
| 0.53 s | 0.17-0.18 | 0.27-0.32 | 0.06-0.08 | 0.12-0.13 |
| 1.07 s | 0.13-0.15 | 0.22-0.28 | 0.05-0.06 | 0.10-0.11 |
| 1.6-1.9 s | 0.10-0.13 | 0.19-0.27 | 0.03-0.05 | 0.09 |

In the clip-context target space a patch's bin is barely more likely than chance to survive one
frame (gain 0.11-0.18): copying gives a categorical predictor almost nothing, but the targets also
move with frame-to-frame changes that may be motion or may be noise. Per-frame features are more
stable and content-like, and their copy advantage fades to 0.10-0.13 on web video by ~1.5 s.

## Caveats

- The cut rule (lag-1 copy error above half the mean baseline and 4x the episode median) over-flags:
  most flagged pairs in `cuts.jpg` are fast hand motion or camera shake, a few are real cuts.
  "Within shot" differs from "all pairs" by at most 2 points, so this does not move the tables.
- 96 episodes per source; SEM of removed at 8 frames is 2-3 points. Center square crop, not the
  training RandomResizedCrop. SSv2 has few windows beyond 3 s (59 at 3.2 s, 20 at 4.3 s).
