"""Precompute VideoFlexTok (EPFL-VILAB/videoflextok_d18_d18_k600) token targets, no augmentation.

Frames are what the ViT-T sees: stride 2 of the 15 fps store (7.5 fps), one fixed center square crop,
resized to 128 px for the tokenizer (the ViT-T gets the same region at 224 px). A 17-frame window gives
5 token groups (group 0 = frame 0, group g >= 1 = frames 4g-3..4g), 256 coarse-to-fine ids each
(FSQ 8,8,8,5,5,5 -> 64,000 ids, stored as uint16).

  train: the 20% K710 store, every episode with >= 34 rows (the ViT-T's clip filter), a window
         starting every 4 frames (8 store rows): start_row = 0, 8, 16, ... while start_row + 32 < len.
  val:   K400 val mp4s, one window centered in the video, frames picked at 7.5 fps by timestamp.

Deterministic: the VidTok KL-VAE uses its posterior mean (its default samples), TF32 matmuls/convs
for every split (fp32 vs TF32 ids agree ~90%, inside the ~60% agreement under +-1 gray level noise).
One process per GPU; output shards are written atomically and skipped if present, so reruns resume.
  CUDA_VISIBLE_DEVICES=r python vft_tokenize.py --split train --rank r --world 8 --out DIR
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'third_party' / 'LeVJEPA'))

CKPT = '/scratch/CLUSTER_USER/models/videoflextok/videoflextok_d18_d18_k600'
STORE = '/scratch/CLUSTER_USER/datasets/mix256/k710_20pct/train_20pct.lance'
K400_VAL = '/scratch/CLUSTER_USER/datasets/k400/val'
T, STRIDE, STEP, SIZE = 17, 2, 4, 128          # frames per window, store-row stride, window step (frames)
CHUNK = 500                                    # episodes (train) or videos (val) per shard


def center_square(frames, size):
    """uint8 (N, C, H, W) -> uint8 (N, C, size, size): one center square crop, bicubic + antialias."""
    h, w = frames.shape[-2:]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    v = frames[..., y:y + s, x:x + s].float()
    v = F.interpolate(v, size=(size, size), mode='bicubic', antialias=True, align_corners=False)
    return v.round().clamp(0, 255).to(torch.uint8)


class TrainEpisodes(torch.utils.data.Dataset):
    """Every stride-2 frame (phase 0) of one valid episode, center-cropped to 128 px."""
    def __init__(self, positions):
        from data.loader import VJEPAClipDataset
        self.ds = VJEPAClipDataset(STORE, num_frames=16, frame_stride=STRIDE, clips_per_video=1, random_crop=False)
        self.positions = positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        p = int(self.positions[i])
        n_rows = int(self.ds._ep_len[p])
        rows = np.arange(0, n_rows, STRIDE) + int(self.ds._ep_off[p])
        look = self.ds._fetch(rows)
        frames = torch.stack([self.ds._decode_frame(look(int(r))) for r in rows])
        starts = np.arange(0, len(rows) - T + 1, STEP)                     # in stride-2 frames
        return dict(key=p, episode=int(self.ds.valid_eps[p]), frames=center_square(frames, SIZE),
                    starts=starts)


class ValVideos(torch.utils.data.Dataset):
    """One centered 17-frame window at 7.5 fps from each K400 val mp4."""
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import av
        f = self.files[i]
        try:
            with av.open(f) as c:
                st = c.streams.video[0]
                ts, imgs = [], []
                for fr in c.decode(st):
                    ts.append(float(fr.pts * st.time_base)); imgs.append(fr.to_ndarray(format='rgb24'))
        except Exception as e:                                             # unreadable file: report, skip
            return dict(key=i, file=f, error=repr(e))
        ts = np.asarray(ts)
        if len(ts) < 2:
            return dict(key=i, file=f, error='fewer than 2 frames')
        want = (ts[0] + ts[-1]) / 2 + (np.arange(T) - (T - 1) / 2) / 7.5
        idx = np.abs(ts[None] - want[:, None]).argmin(1)
        frames = torch.from_numpy(np.stack([imgs[j] for j in idx])).permute(0, 3, 1, 2)
        return dict(key=i, file=f, frames=center_square(frames, SIZE), starts=np.array([0]),
                    frame_index=idx, times=ts[idx])


def load_model():
    from videoflextok.wrappers import VideoFlexTokFromHub
    model = VideoFlexTokFromHub.from_pretrained(CKPT).eval().cuda()
    n = 0
    for m in model.modules():
        if type(m).__name__ == 'DiagonalGaussianRegularizer':
            m.sample = False; n += 1
    assert n == 1, n
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return model


def write(dest, arrays):
    tmp = dest.with_suffix('.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--limit', type=int, default=0, help='first N items of this rank only (smoke test)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if a.split == 'train':
        from data.loader import VJEPAClipDataset
        n_items = len(VJEPAClipDataset(STORE, num_frames=16, frame_stride=STRIDE, clips_per_video=1,
                                       random_crop=False).valid_eps)
        items, make = None, TrainEpisodes
    else:
        items = sorted(glob.glob(f'{K400_VAL}/*/*.mp4'))
        n_items, make = len(items), ValVideos
    chunks = [c for c in range(0, n_items, CHUNK) if (c // CHUNK) % a.world == a.rank]
    if a.limit:
        chunks = chunks[:1]
    model = load_model()
    t0, done_windows = time.time(), 0
    for c in chunks:
        path = a.out / f'{a.split}_{c // CHUNK:05d}.npz'
        if path.exists():
            continue
        sel = np.arange(c, min(c + CHUNK, n_items))
        if a.limit:
            sel = sel[:a.limit]
        ds = make(sel if a.split == 'train' else [items[j] for j in sel])
        loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=4,
                                             multiprocessing_context='spawn')
        out = dict(tokens=[], key=[], start=[], episode=[], file=[], frame_index=[], label=[])
        errors, pending = [], []

        def flush():
            # Always a full batch: the last, partial batch of a shard is padded with copies of its final
            # window (outputs dropped). A new batch size makes torch.compile recompile the tokenizer; after
            # 8 recompiles dynamo falls back to eager, where flex attention runs dense and needs ~90 GiB.
            n = len(pending)
            ws = [w for w, _ in pending] + [pending[-1][0]] * (a.batch - n)
            x = torch.stack(ws).cuda(non_blocking=True)
            x = x.float().div(127.5).sub(1).permute(0, 2, 1, 3, 4)                    # B, C, T, H, W
            with torch.no_grad():
                ids = torch.cat(model.tokenize(x))[:n]                                # n, 5, 256
            assert ids.shape == (n, 5, 256) and int(ids.max()) < 65536, ids.shape
            out['tokens'].append(ids.to(torch.int32).cpu().numpy().astype(np.uint16))
            pending.clear()

        for it in loader:
            if 'error' in it:
                errors.append(dict(file=it['file'], error=it['error'])); continue
            for s in it['starts']:
                pending.append((it['frames'][s:s + T], None))
                out['key'].append(it['key'] + (c if a.split == 'val' else 0)); out['start'].append(int(s) * STRIDE)
                out['episode'].append(it.get('episode', -1))
                if a.split == 'val':
                    out['file'].append(os.path.relpath(it['file'], K400_VAL))
                    out['frame_index'].append(it['frame_index'])
                    out['label'].append(Path(it['file']).parent.name)
                if len(pending) == a.batch:
                    flush()
        if pending:
            flush()
        n = sum(len(t) for t in out['tokens'])
        done_windows += n
        arrays = dict(tokens=np.concatenate(out['tokens']), key=np.asarray(out['key'], np.int64),
                      start_row=np.asarray(out['start'], np.int32), episode=np.asarray(out['episode'], np.int64))
        if a.split == 'val':
            arrays.update(path=np.asarray(out['file']), label=np.asarray(out['label']),
                          frame_index=np.stack(out['frame_index']).astype(np.int32))
        if a.limit:
            path = a.out / f'smoke_{a.split}_{a.rank}.npz'
        write(path, arrays)
        if errors:
            (a.out / f'{path.stem}_errors.json').write_text(json.dumps(errors, indent=1))
        el = time.time() - t0
        print(f'{time.strftime("%H:%M:%S")} rank {a.rank} {path.name}: {n} windows, {len(errors)} errors; '
              f'{done_windows / el:.1f} windows/s', flush=True)
    print(f'{time.strftime("%H:%M:%S")} rank {a.rank} finished {a.split}', flush=True)


if __name__ == '__main__':
    main()
