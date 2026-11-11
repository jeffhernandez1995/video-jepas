"""Time build_lance_mix.decode_item on a handful of raw videos (login-node sized).
  python scripts/range/bench_build_mix.py --glob '/scratch/.../*.webm' --n 96 --workers 8
"""
import argparse
import glob
import time
from multiprocessing import get_context

import build_lance_mix as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default=None)
    ap.add_argument('--tar', default=None, help='decode mp4 members of one tar instead')
    ap.add_argument('--n', type=int, default=96)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.tar:
        items = B._tar_items(a.tar)[:a.n]
    else:
        items = [(f, 'path', f, None) for f in sorted(glob.glob(a.glob))[:a.n]]
    t = time.time()
    with get_context('spawn').Pool(a.workers) as p:
        res = p.map(B.decode_item, items, chunksize=2)
    dt = time.time() - t
    eps = [e for _, r in res if r for e in r]
    rows = sum(len(e[1]) for e in eps)
    kb = sum(len(f[0]) for e in eps for f in e[1]) / max(rows, 1) / 1e3
    print(f'{len(items)} videos, {a.workers} workers, {dt:.1f}s: {len(eps)} episodes, {rows} rows, '
          f'failed {sum(r is None for _, r in res)}, frame {eps[0][1][0][1:] if eps else None}, {kb:.1f} KB/frame; '
          f'{rows / B.FPS / dt / a.workers:.2f} video-seconds per second per worker')


if __name__ == '__main__':
    main()
