"""IntPhys 2 plausibility probe under the exact protocol of Esmati et al. 2026 (arXiv 2606.05328,
Fig. 2 / App. D Fig. 14), following their code (github.com/ParsaEsmati/videodiffusionphysics:
encode_video_encoder.py, probe_pairwise.py). The original IntPhys dev set they report in Fig. 2 is
no longer downloadable, so IntPhys 2 (Main, 1,012 videos) is the benchmark we can match.

Features (encode_video_encoder.py): each whole video is sampled to the encoder's native clip
length with torch.linspace, resized (squashed) to a square at the native resolution, bilinear,
ImageNet-normalized, run once in fp32. Every transformer block's output plus the encoder's final
output is mean-pooled over all tokens -> [blocks + 1, D] per video (V-JEPA 2: 64 frames at 256 px;
VideoMAE: 16 at 224; NATIVE below for the rest).

Probe (probe_pairwise.py): one Linear(D, 2) per block, trained jointly on the mean cross-entropy
over blocks, Adam lr 1e-3, no weight decay, raw (unstandardized) features, 50 epochs, evaluated
after the last epoch. Separate probes per condition. Split by scenario (IntPhys 2 SceneIndex: 2
possible + 2 impossible videos): torch.randperm(seed) over the sorted scenario ids, the first
round(0.4 * n) held out. Seeds 0-4 set both the split and the probe init / batch order. Reported:
held-out accuracy averaged over blocks (the paper's "mean probe accuracy across blocks"), mean +-
SEM over seeds, both pair-wise (possible vs impossible margin within a scene, ties 0.5) and
per-video; also the best block and the final output.

Headline = pair-wise. App. D Fig. 14 is labelled per-video, but its numbers are the pair-wise ones:
on V-JEPA 2-L / VideoMAE-L this script gives pair-wise 62.0 / 61.8 averaged over conditions against
the figure's 62.2 / 61.6 (per condition within 0.2-4 points), while per-video is 51.6 / 50.9.

The paper text gives 60/40 and batch size 4; the code defaults (used by its README commands) are
--val-split 0.2 and --batch-size 32. We follow the paper text.

  python scripts/natural/eval_intphys2_paper.py --family vjepa2 --checkpoint <ckpt> --out <dir>
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_davis import Encoder  # noqa: E402

ROOT = Path('/scratch/CLUSTER_USER/datasets/intphys2/Main')
NATIVE = dict(vjepa2=(64, 256), vjepa=(16, 224), videomae=(16, 224), videomaev2=(16, 224),
              levjepa_hf=(16, 224), levjepa=(16, 224), dinov2=(16, 224), nlp=(None, 224))   # (frames, px)
# nlp: pass --frames = the run's training num_frames
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def items():
    """[(path, label 1=possible, condition, scenario)]"""
    with open(ROOT / 'metadata.csv') as f:
        return [(ROOT / r['file_name'], int(r['type'].endswith('_Possible')), r['condition'], r['SceneIndex'])
                for r in csv.DictReader(f)]


def load_clip(path, frames, size):
    """[1, 3, T, size, size]: linspace over the whole video, bilinear squash, ImageNet norm."""
    import av
    with av.open(str(path)) as c:
        st = c.streams.video[0]; st.thread_type = 'AUTO'
        all_frames = [f.to_ndarray(format='rgb24') for f in c.decode(st)]
    idx = torch.linspace(0, len(all_frames) - 1, frames).long()
    v = torch.from_numpy(np.stack([all_frames[i] for i in idx])).permute(0, 3, 1, 2).float() / 255   # T,C,H,W
    v = F.interpolate(v, size=(size, size), mode='bilinear', align_corners=False).permute(1, 0, 2, 3)
    return ((v - MEAN) / STD)[None]


def block_list(module):
    return max((m for m in module.modules() if isinstance(m, nn.ModuleList)), key=len)


@torch.no_grad()
def extract(family, checkpoint, frames, size):
    enc = Encoder(family, checkpoint)
    blocks = block_list(enc.enc)
    cap = {}

    def hook(i):
        def f(_, __, out):
            out = (out[0] if isinstance(out, tuple) else out).float()
            cap[i] = out.reshape(-1, out.shape[-1]).mean(0)      # all tokens (and frames, for DINOv2)
        return f
    for i, b in enumerate(blocks):
        b.register_forward_hook(hook(i))
    its, feats, t0 = items(), [], time.time()
    for v, (path, *_) in enumerate(its):
        cap.clear()
        clip = load_clip(path, frames, size).cuda()
        final = enc.tokens(clip).float().mean(1)[0]                     # final output, all tokens
        feats.append(torch.stack([cap[i] for i in range(len(blocks))] + [final]).cpu().numpy())
        if v % 100 == 0:
            print(f'intphys2: {v}/{len(its)} videos, {time.time() - t0:.0f}s', flush=True)
    return np.stack(feats), enc.info, len(blocks)


def probe(x, y, scen, seed, val_frac=0.4, batch=4, epochs=50, lr=1e-3):
    """x [N, K, D] -> per-block held-out per-video and pair-wise accuracy (lists of K)."""
    torch.manual_seed(seed); np.random.seed(seed)
    uniq = sorted(set(scen))
    order = torch.randperm(len(uniq), generator=torch.Generator().manual_seed(seed)).tolist()
    val = {uniq[order[i]] for i in range(max(1, int(round(len(uniq) * val_frac))))}
    tr = [i for i, s in enumerate(scen) if s not in val]
    va = [i for i, s in enumerate(scen) if s in val]
    K, D = x.shape[1:]
    heads = nn.ModuleList([nn.Linear(D, 2) for _ in range(K)]).cuda()
    opt = torch.optim.Adam(heads.parameters(), lr=lr)
    xt, yt = torch.from_numpy(x).float().cuda(), torch.tensor(y).cuda()
    fwd = lambda b: torch.stack([heads[k](xt[b, k]) for k in range(K)], 1)            # B,K,2
    for _ in range(epochs):
        perm = torch.tensor(tr)[torch.randperm(len(tr))]
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch].cuda()
            out = fwd(b)
            loss = torch.stack([F.cross_entropy(out[:, k], yt[b]) for k in range(K)]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = fwd(torch.tensor(va).cuda())
        acc = (out.argmax(2) == yt[va][:, None]).float().mean(0).cpu().numpy() * 100
        margin = (out[..., 1] - out[..., 0]).cpu().numpy()                             # V,K
    pair = []
    groups = defaultdict(list)
    for j, i in enumerate(va):
        groups[scen[i]].append((j, y[i]))
    for k in range(K):
        c = t = 0
        for g in groups.values():
            for p, lp in g:
                for n, ln in g:
                    if lp == 1 and ln == 0:
                        t += 1; c += 1.0 if margin[p, k] > margin[n, k] else 0.5 if margin[p, k] == margin[n, k] else 0
        pair.append(100 * c / t)
    return acc, np.array(pair)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True, choices=list(NATIVE))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frames', type=int, default=None, help='default: the encoder\'s native clip (NATIVE)')
    p.add_argument('--size', type=int, default=None)
    p.add_argument('--seeds', default='0,1,2,3,4')
    a = p.parse_args()
    frames, size = a.frames or NATIVE[a.family][0], a.size or NATIVE[a.family][1]
    assert frames, 'pass --frames (the training clip length)'
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / f'intphys2_paper_blocks_{frames}f_{size}px.npz'
    info = {}
    if cache.exists():
        feats = np.load(cache)['feats']
    else:
        feats, info, nb = extract(a.family, a.checkpoint, frames, size)
        np.savez(cache, feats=feats)
    its = items()
    res = dict(family=a.family, checkpoint=a.checkpoint, frames=frames, size=size, blocks_plus_final=feats.shape[1],
               protocol='Esmati et al. 2026 probe_pairwise.py: per-block Linear(D,2), Adam 1e-3, batch 4, 50 ep, '
                        'scenario-grouped 60/40, seeds ' + a.seeds + '; mean over blocks of per-video accuracy',
               conditions={})
    seeds = [int(s) for s in a.seeds.split(',')]
    for cond in sorted({it[2] for it in its}):
        idx = [i for i, it in enumerate(its) if it[2] == cond]
        x, y, scen = feats[idx], [its[i][1] for i in idx], [its[i][3] for i in idx]
        accs, pairs = zip(*(probe(x, y, scen, s) for s in seeds))
        accs, pairs = np.stack(accs), np.stack(pairs)                    # seeds, K
        per_seed, pair_seed = accs.mean(1), pairs.mean(1)
        res['conditions'][cond] = dict(
            pair_mean_over_blocks=round(float(pair_seed.mean()), 2),
            pair_sem=round(float(pair_seed.std(ddof=1) / np.sqrt(len(seeds))), 2),
            mean_over_blocks=round(float(per_seed.mean()), 2),
            sem=round(float(per_seed.std(ddof=1) / np.sqrt(len(seeds))), 2),
            best_block=int(accs.mean(0).argmax()), best_block_acc=round(float(accs.mean(0).max()), 2),
            final_output_acc=round(float(accs[:, -1].mean()), 2),
            per_block=[round(float(v), 2) for v in accs.mean(0)], n=len(idx))
        print(cond, json.dumps({k: v for k, v in res['conditions'][cond].items() if k != 'per_block'}), flush=True)
    res['pair_average'] = round(float(np.mean([c['pair_mean_over_blocks'] for c in res['conditions'].values()])), 2)
    res['per_video_average'] = round(float(np.mean([c['mean_over_blocks'] for c in res['conditions'].values()])), 2)
    res.update({k: v for k, v in info.items()})
    (a.out / f'intphys2_paper_probe_{frames}f_{size}px.json').write_text(json.dumps(res, indent=2) + '\n')
    print('average over conditions: pair-wise', res['pair_average'], 'per-video', res['per_video_average'])


if __name__ == '__main__':
    main()
