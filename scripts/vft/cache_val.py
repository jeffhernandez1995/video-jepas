"""Held-out set for train_vft.py: a fixed random subset (seed 0) of the tokenized K400 val windows, with
the same 17 frames at the ViT-T's resolution (fixed center square crop, 224 px), so token loss and
accuracy can be measured on unseen videos during training.
Writes val_frames224_<N>.npy (N, 17, 3, 224, 224) uint8 and val_tokens_<N>.npy (N, 5, 256) uint16.
  python cache_val.py --n 4096 --procs 16
"""
import argparse
import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TOKENS = Path('/scratch/CLUSTER_USER/datasets/vft_tokens')
K400_VAL = '/scratch/CLUSTER_USER/datasets/k400/val'


def frames(job):
    import av
    rel, idx = job
    with av.open(f'{K400_VAL}/{rel}') as c:
        imgs = [fr.to_ndarray(format='rgb24') for fr in c.decode(c.streams.video[0])]
    x = torch.from_numpy(np.stack([imgs[j] for j in idx])).permute(0, 3, 1, 2).float()
    h, w = x.shape[-2:]
    s = min(h, w)
    y, xx = (h - s) // 2, (w - s) // 2
    x = F.interpolate(x[..., y:y + s, xx:xx + s], size=(224, 224), mode='bicubic', antialias=True, align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--procs', type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(1)
    files = sorted(glob.glob(str(TOKENS / 'val' / 'val_*.npz')))
    z = [np.load(f) for f in files if not f.endswith('_errors.json')]
    tok = np.concatenate([d['tokens'] for d in z]); rel = np.concatenate([d['path'] for d in z])
    fi = np.concatenate([d['frame_index'] for d in z])
    print(len(tok), 'val windows from', len(files), 'shards')
    pick = np.sort(np.random.default_rng(0).choice(len(tok), a.n, replace=False))
    out = np.lib.format.open_memmap(TOKENS / f'val_frames224_{a.n}.tmp.npy', mode='w+', dtype=np.uint8,
                                    shape=(a.n, 17, 3, 224, 224))
    with Pool(a.procs) as pool:
        for i, f in enumerate(pool.imap(frames, [(rel[j], fi[j]) for j in pick], chunksize=8)):
            out[i] = f
            if i % 512 == 0:
                print(i, flush=True)
    out.flush(); del out
    (TOKENS / f'val_frames224_{a.n}.tmp.npy').rename(TOKENS / f'val_frames224_{a.n}.npy')
    np.save(TOKENS / f'val_tokens_{a.n}.npy', tok[pick])
    np.save(TOKENS / f'val_pick_{a.n}.npy', pick)
    print('done')


if __name__ == '__main__':
    main()
