"""Copy and unigram diagnostics of the VideoFlexTok targets, per token position k (coarse to fine).

For each position k and adjacent group pair (g, g+1), g = 0..3 (group 0 = frame 0 alone):
  copy      P(token k of group g+1 == token k of group g): accuracy of the copy-last-step predictor
  copy_any  P(token k of group g+1 appears anywhere among group g's first K tokens)
  digits    the same copy rate per FSQ digit (8,8,8,5,5,5)
  unigram   entropy (nats, comparable with a cross-entropy loss) and top-1 frequency of position k over
            groups 1..4 (top-1 frequency = accuracy of always predicting the most common id)
Run on the train shards available (a subset of windows is enough) and on the K400 val windows.
  python diag_tokens.py --windows 20000 --K 16
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

TOKENS = Path('/scratch/CLUSTER_USER/datasets/vft_tokens')
LEVELS = np.array([8, 8, 8, 5, 5, 5])
BASIS = np.array([1, 8, 64, 512, 2560, 12800])


def stats(tok, K):
    tok = tok[:, :, :K].astype(np.int64)                      # N, 5, K
    a, b = tok[:, :-1], tok[:, 1:]                            # g, g+1
    copy = (a == b).mean(axis=(0, 1))
    copy_any = (b[..., :, None] == a[..., None, :]).any(-1).mean(axis=(0, 1))
    da = (a[..., None] // BASIS) % LEVELS; db = (b[..., None] // BASIS) % LEVELS
    digit_copy = (da == db).mean(axis=(0, 1))                 # K, 6
    ent, top1 = [], []
    for k in range(K):
        _, c = np.unique(tok[:, 1:, k], return_counts=True)
        p = c / c.sum()
        ent.append(float(-(p * np.log(p)).sum())); top1.append(float(p.max()))
    per_pair = (a == b).mean(axis=(0, 2))                     # copy rate per (g, g+1), averaged over k
    return dict(n=len(tok), copy=copy.tolist(), copy_any=copy_any.tolist(), digit_copy=digit_copy.tolist(),
                unigram_entropy_nats=ent, unigram_top1=top1, copy_by_group_pair=per_pair.tolist(),
                uniform_entropy_nats=float(np.log(64000)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--windows', type=int, default=20000)
    p.add_argument('--K', type=int, default=16)
    a = p.parse_args()
    out = {}
    for split in ('train', 'val'):
        files = sorted(glob.glob(str(TOKENS / split / f'{split}_*.npz')))
        if not files:
            continue
        tok = np.concatenate([np.load(f)['tokens'] for f in files])
        rng = np.random.default_rng(0)
        tok = tok[rng.choice(len(tok), min(a.windows, len(tok)), replace=False)]
        out[split] = stats(tok, a.K) | dict(shards=len(files))
        s = out[split]
        print(f'== {split}: {s["n"]} windows from {s["shards"]} shards')
        print(' k  copy  copy_any  digit_copy(mean)  unigram_H(nats)  top1')
        for k in range(a.K):
            print(f'{k + 1:2d}  {s["copy"][k]:.3f}  {s["copy_any"][k]:.3f}     {np.mean(s["digit_copy"][k]):.3f}'
                  f'            {s["unigram_entropy_nats"][k]:.2f}          {s["unigram_top1"][k]:.4f}')
        print(' copy rate by group pair (0->1, 1->2, 2->3, 3->4):', [round(v, 3) for v in s['copy_by_group_pair']])
    (TOKENS / f'diag_tokens_K{a.K}.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
