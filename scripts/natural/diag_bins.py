"""Would random-projection bins of the teacher's patch features make balanced, predictable
categorical targets? (2026-11-04)

The candidate target ("Q"): M fixed random unit directions a_m; a patch feature y gets the code
(bin_1(a_1 . y), ..., bin_M(a_M . y)), each projection cut into K bins. If the features were
isotropic N(0, I) (what SIGReg tests for), every projection would be N(0, 1) and the bins could be
the fixed N(0, 1) quantiles. This measures, on the frames sampled by diag_lag_curve.py:

  moments    per-direction mean, std, |skew|, excess kurtosis of the projections
  fixed      usage of the fixed N(0, 1)-quantile bins: entropy / log K (mean and worst direction)
             and the largest single-bin mass
  joint      perplexity of the joint code of the first 4 directions (K^4 = 4,096 codes), fixed bins
             and per-direction empirical quantile bins
  stay       P(same empirical bin at the same site d frames later), chance = 1 / K, and the
             normalized gain (stay - 1/K) / (1 - 1/K)

for the teacher's raw patch tokens (final LayerNorm output) and its unit-normalized features (the
regression targets, scaled by sqrt(D)), each encoded per frame (T = 1) and as 16-frame clips
(block-causal, frame 0 dropped as in training).

  python scripts/natural/diag_bins.py --checkpoint <run>/checkpoint_epoch240.pt --data <lag_curve dir>
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_lag_curve as lag  # noqa: E402
import nlp  # noqa: E402

LAGS = dict(frame=[1, 4, 8, 12, 20, 32], clip=[1, 4, 8, 14])      # frames at 7.5 fps


@torch.no_grad()
def tokens(teacher, fr):
    """Raw patch tokens (B, T, S, D) of uint8 frames (B, T, C, H, W)."""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        tok = teacher(nlp.normalize_video(fr.cuda()))[:, 1:]
    return tok.float().reshape(fr.shape[0], fr.shape[1], nlp.SITES, -1)


def gauss_edges(k):
    return torch.distributions.Normal(0., 1.).icdf(torch.arange(1, k) / k)


def entropy(counts):
    p = counts / counts.sum(-1, keepdim=True)
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def summarize(proj, k, lags):
    """proj: (E, T, S, M) projections of E episodes; returns the metrics of the module docstring."""
    x = proj.flatten(0, 2)                                                   # N, M
    mu, sd = x.mean(0), x.std(0)
    z = (x - mu) / sd
    fixed = torch.bucketize(x, gauss_edges(k))                               # N, M
    emp_edges = torch.quantile(x[torch.randperm(len(x))[:200_000]], torch.arange(1, k) / k, dim=0).T  # M, K-1
    emp = torch.stack([torch.bucketize(x[:, m].contiguous(), emp_edges[m]) for m in range(x.shape[1])], 1)
    cnt = torch.stack([torch.bincount(fixed[:, m], minlength=k) for m in range(x.shape[1])]).float()
    h = entropy(cnt) / math.log(k)

    def joint(b):
        code = (b[:, :4] * torch.tensor([k ** 3, k ** 2, k, 1])).sum(1)
        return float(entropy(torch.bincount(code, minlength=k ** 4).float()).exp())

    eb = emp.reshape(proj.shape)
    stay = {}
    for d in lags:
        if d < proj.shape[1]:
            s = float((eb[:, :-d] == eb[:, d:]).float().mean())
            stay[str(d)] = dict(seconds=round(d / lag.FPS, 2), stay=round(s, 4), gain=round((s - 1 / k) / (1 - 1 / k), 4))
    return dict(tokens=len(x), mean_abs_mean=round(float(mu.abs().mean()), 4), mean_std=round(float(sd.mean()), 4),
                std_range=[round(float(sd.min()), 4), round(float(sd.max()), 4)],
                mean_abs_skew=round(float((z ** 3).mean(0).abs().mean()), 4),
                mean_excess_kurtosis=round(float(((z ** 4).mean(0) - 3).mean()), 4),
                fixed_entropy=round(float(h.mean()), 4), fixed_entropy_worst=round(float(h.min()), 4),
                fixed_max_bin_mass=round(float((cnt / cnt.sum(1, keepdim=True)).max()), 4),
                joint4_perplexity_fixed=round(joint(fixed), 1), joint4_perplexity_empirical=round(joint(emp), 1),
                stay=stay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=32)
    p.add_argument('--frames', type=int, default=64)
    p.add_argument('--dirs', type=int, default=32)
    p.add_argument('--bins', type=int, default=8)
    a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    teacher = nlp.build_encoder(ck['config'])
    teacher.load_state_dict(ck['teacher'])
    teacher.cuda().eval().requires_grad_(False)
    dim = teacher.embed_dim
    dirs = F.normalize(torch.randn(dim, a.dirs, generator=torch.Generator().manual_seed(0)), dim=0).cuda()
    out = dict(checkpoint=str(a.checkpoint), dirs=a.dirs, bins=a.bins, sources={})
    for f in sorted(a.data.glob('*.pt')):
        data = torch.load(f, weights_only=False)
        acc = {(feat, mode): [] for feat in ('raw', 'unit') for mode in ('frame', 'clip')}
        for e in data['episodes'][:a.episodes]:
            fr = torch.stack([lag.to_224(b) for b in e['frames'][:a.frames]])
            if len(fr) < 16:
                continue
            per = torch.cat([tokens(teacher, fr[i:i + 64, None]) for i in range(0, len(fr), 64)])[:, 0][None]
            clip = tokens(teacher, fr[None, :16])[:, 1:]                       # frame 0 dropped
            for mode, t in (('frame', per), ('clip', clip)):
                acc[('raw', mode)].append((t @ dirs).cpu())
                acc[('unit', mode)].append((F.normalize(t, dim=-1) @ dirs * math.sqrt(dim)).cpu())
        res = {}
        for (feat, mode), v in acc.items():
            n = min(x.shape[1] for x in v)                                   # common window length
            res[f'{feat}_{mode}'] = summarize(torch.cat([x[:, :n] for x in v]), a.bins, LAGS[mode])
        out['sources'][data['source']] = res
        print(f"\n{data['source']} ({len(acc[('raw', 'frame')])} episodes)")
        print('  features     std (min-max)       |skew|  kurt  fixed H (worst)  max bin  joint4 fixed/emp   stay gain by lag')
        for key, r in res.items():
            st = '  '.join(f"{v['seconds']}s:{v['gain']:.2f}" for v in r['stay'].values())
            print(f"  {key:<11} {r['mean_std']:6.3f} ({r['std_range'][0]:.2f}-{r['std_range'][1]:.2f})  "
                  f"{r['mean_abs_skew']:5.2f} {r['mean_excess_kurtosis']:5.2f}   {r['fixed_entropy']:.3f} ({r['fixed_entropy_worst']:.3f})"
                  f"     {r['fixed_max_bin_mass']:.3f}   {r['joint4_perplexity_fixed']:6.0f}/{r['joint4_perplexity_empirical']:6.0f}   {st}")
        (a.data / 'bins.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
