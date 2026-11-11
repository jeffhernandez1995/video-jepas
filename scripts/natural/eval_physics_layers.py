"""Per-block version of eval_physics.py: the same videos, windows and grouped 60/40 linear probe,
but with mean-pooled features from the output of every transformer block instead of only the
encoder's final (normed) output. Esmati et al. 2026 probe every block (their Fig. 4); this checks
whether plausibility is linearly readable at intermediate depth when the last layer is at chance.

Block outputs are captured with forward hooks on the encoder's block list (the largest ModuleList
in the module tree), averaged over each window's tokens and then over windows, weighted by the
real (unpadded) frames per window. Features are cached in <out>/physics_<dataset>_blocks.npz
[videos, blocks, D]; results go to <out>/physics_probe_blocks_<dataset>.json.

  python scripts/natural/eval_physics_layers.py --family vjepa2 --checkpoint <ckpt> --out <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_davis  # noqa: E402
import eval_physics as ep  # noqa: E402
from eval_davis import Encoder  # noqa: E402


def block_list(module):
    lists = [m for m in module.modules() if isinstance(m, torch.nn.ModuleList)]
    return max(lists, key=len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset', default='intphys2', choices=['intphys2', 'inflevel_lab'])
    p.add_argument('--height', type=int, default=224)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.dataset}{"" if a.height == 224 else f"_{a.height}px"}'
    cache = a.out / f'physics_{tag}_blocks.npz'
    its = ep.items(a.dataset)
    info = {}
    if cache.exists():
        feats = np.load(cache)['feats']
    else:
        enc = Encoder(a.family, a.checkpoint)
        info = enc.info
        blocks = block_list(enc.enc)
        acc = {}

        def hook(i):
            def f(_, __, out):
                out = out[0] if isinstance(out, tuple) else out
                acc.setdefault(i, []).append(out.float().mean(1))   # [batch, D] per window / chunk
            return f
        for i, b in enumerate(blocks):
            b.register_forward_hook(hook(i))
        step = 8 if a.dataset == 'intphys2' else 4
        span = eval_davis.SLOTS * enc.tubelet
        feats, t0 = [], time.time()
        for v, (path, *_) in enumerate(its):
            acc.clear()
            frames = ep.load_frames(path, step, a.height, enc.patch)
            enc.features(frames)
            n = len(frames)
            if enc.image:   # per-frame chunks of 16: each row is one frame
                f = torch.stack([torch.cat(acc[i]).mean(0) for i in range(len(blocks))])
            else:           # one row per window; weight by the window's real frames
                wts = torch.tensor([min(span, n - s) for s in range(0, n, span)], dtype=torch.float32, device='cuda')
                f = torch.stack([(torch.cat(acc[i]) * wts[:, None]).sum(0) / wts.sum() for i in range(len(blocks))])
            feats.append(f.cpu().numpy())
            if v % 100 == 0:
                print(f'{a.dataset}: {v}/{len(its)} videos, {time.time() - t0:.0f}s', flush=True)
        feats = np.stack(feats)
        np.savez(cache, feats=feats, paths=np.array([str(it[0]) for it in its]))
    labels, cats, groups = [it[1] for it in its], [it[2] for it in its], [it[3] for it in its]
    res = dict(family=a.family, checkpoint=a.checkpoint, dataset=a.dataset, height=a.height, split='grouped',
               blocks={})
    for i in range(feats.shape[1]):
        r = ep.probe(feats[:, i], labels, cats, groups, 'grouped')
        res['blocks'][i] = r
        print(f'block {i:2d}', json.dumps({k: (v['mean'] if isinstance(v, dict) else v) for k, v in r.items()}),
              flush=True)
    best = max(res['blocks'], key=lambda i: res['blocks'][i]['average'])
    res['best_block'] = best
    res['mean_over_blocks'] = round(float(np.mean([r['average'] for r in res['blocks'].values()])), 2)
    res.update({k: v for k, v in info.items()})
    (a.out / f'physics_probe_blocks_{tag}.json').write_text(json.dumps(res, indent=2) + '\n')
    print('best block', best, res['blocks'][best]['average'], 'mean over blocks', res['mean_over_blocks'])


if __name__ == '__main__':
    main()
