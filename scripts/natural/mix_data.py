"""Weighted multi-source training data for the pretraining mixes (docs/data_mixes.md).

A mix spec (scripts/natural/mixes/*.json) names sources, each with a sampling
weight and one or more Lance datasets with a frame stride (the K710 20% store
is 15 fps and read at stride 2; consolidated mix sources, see
scripts/range/consolidate_mix.py, are 7.5 fps and read at stride 1, so every
source yields 16 frames at 7.5 fps). Image sources hold one image per row and
become static 16-frame clips (the same crop on every frame).

Each source is one index over its datasets: episode i -> (dataset, first row,
length). A virtual epoch is a fixed number of clips (default 111,565, the 20%
K710 store, so "240 epochs" is the same step budget as the K710 runs) with
fixed per-source counts in proportion to the weights. Within a source, epochs
walk consecutive slices of a seeded permutation (a new permutation per pass),
so no episode repeats before the source is exhausted, and every epoch is a pure
function of (seed, epoch): runs resume at any step.

Dataset versions are pinned once, at the start of a run, into
<out>/mix_resolved.json; resumes read that file, so shards consolidated later
cannot change the index under a running job.
"""
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MIX_DIR = Path(__file__).resolve().parent / 'mixes'


# ------------------------------------------------------------------ indexing
def episode_index(path):
    """(lengths, offsets) of a shard's episodes, cached as <shard>.index.npz beside it."""
    side = Path(str(path).rstrip('/') + '.index.npz')
    if side.exists():
        z = np.load(side)
        return z['lengths'], z['offsets']
    import lance
    ep = np.concatenate([b.column(0).to_numpy(zero_copy_only=False)
                         for b in lance.dataset(str(path)).to_batches(columns=['episode_idx'], batch_size=1 << 20)])
    starts = np.flatnonzero(np.concatenate(([True], ep[1:] != ep[:-1])))
    lengths = np.diff(np.concatenate((starts, [len(ep)]))).astype(np.int64)
    offsets = starts.astype(np.int64)
    tmp = side.with_name(side.name + f'.{os.getpid()}.tmp.npz')
    np.savez(tmp, lengths=lengths, offsets=offsets)
    os.replace(tmp, side)
    return lengths, offsets


def resolve(spec):
    """Pin every dataset of the spec to its current version and episode index.

    A group is either {"dataset": <source>.lance} (a consolidated source, see
    scripts/range/consolidate_mix.py, whose _members.json names the version
    and its precomputed _index_v<N>.npz) or {"path": <store>.lance} (a plain
    store such as the K710 20% store, indexed by scanning episode_idx).
    """
    import lance
    out = dict(name=spec['name'], epoch_clips=spec.get('epoch_clips', 111565), sources=[])
    for s in spec['sources']:
        groups = []
        for g in s['shards']:
            if 'dataset' in g:
                members = json.loads((Path(g['dataset']) / '_members.json').read_text())
                v = members['version']
                groups.append(dict(path=g['dataset'], version=v, shards=len(members['shards']),
                                   index=str(Path(g['dataset']) / f'_index_v{v}.npz'),
                                   frame_stride=g.get('frame_stride', 1)))
            else:
                groups.append(dict(path=g['path'], version=lance.dataset(g['path']).version, shards=1,
                                   index=None, frame_stride=g.get('frame_stride', 1)))
        out['sources'].append(dict(name=s['name'], kind=s.get('kind', 'video'), weight=s['weight'], groups=groups))
    return out


def load_resolved(out_dir, spec_path, main_rank):
    """Freeze the shard list for a run: written by rank 0 on the first start, reused on resume."""
    f = Path(out_dir) / 'mix_resolved.json'
    if not f.exists() and main_rank:
        spec = json.loads(Path(spec_path).read_text())
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix('.tmp')
        tmp.write_text(json.dumps(resolve(spec), indent=1) + '\n')
        os.replace(tmp, f)
    return f


# ------------------------------------------------------------------- sources
class _Datasets:
    """Lazily opened, version-pinned Lance datasets (one per group; few per source).

    Lance's default metadata cache keeps every touched fragment's file metadata, and
    random clips touch all ~3,800 fragments of a mix: each loader worker grew without
    a plateau (1 -> 7+ GB in an hour), so RANGE jobs with ~100-200 workers were
    OOM-killed. 64 MB caps a worker at ~0.6 GB with no measured slowdown."""
    CACHE = dict(metadata_cache_size_bytes=64 << 20, index_cache_size_bytes=16 << 20)
    def __init__(self, groups):
        self.specs = [(g['path'], g['version']) for g in groups]
        self._open = {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_open'] = {}
        return d

    def take(self, k, rows):
        import lance
        ds = self._open.get(k)
        if ds is None:
            path, version = self.specs[k]
            ds = self._open[k] = lance.dataset(path, version=version, **self.CACHE)
        return ds.take(rows, columns=['frame'])['frame'].to_pylist()


def _group_index(g):
    if g['index']:
        z = np.load(g['index'])
        return z['lengths'], z['offsets']
    return episode_index(g['path'])


def _decode(b):
    with Image.open(io.BytesIO(b)) as img:
        return torch.from_numpy(np.array(img.convert('RGB'), copy=True)).permute(2, 0, 1)


class ClipSource:
    """All episodes of one video source long enough for a num_frames x stride clip."""
    def __init__(self, name, groups, num_frames):
        self.name, self.num_frames = name, num_frames
        shard, first, length, stride = [], [], [], []
        for k, g in enumerate(groups):
            fs = g['frame_stride']
            lens, offs = _group_index(g)
            keep = lens >= num_frames * fs
            shard.append(np.full(int(keep.sum()), k, np.int32))
            first.append(offs[keep]); length.append(lens[keep])
            stride.append(np.full(int(keep.sum()), fs, np.int64))
        self.shards = _Datasets(groups)
        self.shard = np.concatenate(shard); self.first = np.concatenate(first)
        self.length = np.concatenate(length); self.stride = np.concatenate(stride)

    def __len__(self):
        return len(self.shard)

    def get(self, idx):
        """uint8 (T, C, H, W) clips for local episode indices, one random temporal crop each."""
        idx = np.asarray(idx)
        span = self.num_frames * self.stride[idx]
        start = (torch.rand(len(idx)).numpy() * (self.length[idx] - span + 1)).astype(np.int64)
        rows = self.first[idx, None] + start[:, None] + np.arange(self.num_frames)[None] * self.stride[idx, None]
        out = [None] * len(idx)
        for k in np.unique(self.shard[idx]):
            pos = np.flatnonzero(self.shard[idx] == k)
            uniq = np.unique(rows[pos])
            lookup = dict(zip(uniq.tolist(), self.shards.take(int(k), uniq.tolist())))
            for p in pos:
                out[p] = torch.stack([_decode(lookup[r]) for r in rows[p].tolist()])
        return out


class ImageSource:
    """One image per row; returned as a single frame, repeated to num_frames after cropping."""
    def __init__(self, name, groups, num_frames):
        self.name, self.num_frames = name, num_frames
        shard, row = [], []
        for k, g in enumerate(groups):
            lens, offs = _group_index(g)
            shard.append(np.full(len(offs), k, np.int32)); row.append(offs)
        self.shards = _Datasets(groups)
        self.shard = np.concatenate(shard); self.row = np.concatenate(row)

    def __len__(self):
        return len(self.shard)

    def get(self, idx):
        idx = np.asarray(idx)
        out = [None] * len(idx)
        for k in np.unique(self.shard[idx]):
            pos = np.flatnonzero(self.shard[idx] == k)
            for p, b in zip(pos, self.shards.take(int(k), self.row[idx[pos]].tolist())):
                out[p] = _decode(b)[None]                      # (1, C, H, W)
        return out


class MixDataset(torch.utils.data.Dataset):
    """Concatenated index over the sources; a batch costs one Lance take per touched shard."""
    def __init__(self, resolved, num_frames, transform):
        self.sources = [(ImageSource if s['kind'] == 'image' else ClipSource)(s['name'], s['groups'], num_frames)
                        for s in resolved['sources']]
        self.weights = [s['weight'] for s in resolved['sources']]
        self.sizes = [len(s) for s in self.sources]
        assert all(self.sizes), dict(zip([s.name for s in self.sources], self.sizes))
        self.cum = np.concatenate(([0], np.cumsum(self.sizes))).astype(np.int64)
        self.num_frames, self.transform = num_frames, transform

    def __len__(self):
        return int(self.cum[-1])

    def describe(self):
        return {s.name: dict(episodes=n, weight=w, datasets=[f'{p}@v{v}' for p, v in s.shards.specs])
                for s, n, w in zip(self.sources, self.sizes, self.weights)}

    def __getitem__(self, i):
        return self.__getitems__([i])[0]

    def __getitems__(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        src = np.searchsorted(self.cum, idx, side='right') - 1
        out = [None] * len(idx)
        for s in np.unique(src):
            pos = np.flatnonzero(src == s)
            for p, clip in zip(pos, self.sources[s].get(idx[pos] - self.cum[s])):
                clip = self.transform({'frame': clip})
                if clip.shape[0] == 1:                             # image: same crop on every frame
                    clip = clip.expand(self.num_frames, -1, -1, -1).contiguous()
                out[p] = clip
        return out


# ------------------------------------------------------------------- sampler
def _counts(weights, n):
    w = np.asarray(weights, dtype=np.float64)
    raw = w / w.sum() * n
    c = np.floor(raw).astype(np.int64)
    c[np.argsort(raw - c)[::-1][:n - c.sum()]] += 1               # largest remainder
    return c


class MixEpochSampler(torch.utils.data.Sampler):
    """Fixed per-source counts per virtual epoch; same rank sharding and resume
    semantics as train_nlp.EpochSampler (seeded, cut to whole global batches)."""
    def __init__(self, sizes, weights, epoch_clips, rank, world, per_rank, seed):
        self.sizes = [int(s) for s in sizes]
        self.base = np.concatenate(([0], np.cumsum(self.sizes)))[:-1]
        self.counts = _counts(weights, epoch_clips)
        self.rank, self.world, self.per_rank, self.seed = rank, world, per_rank, seed
        self.steps = int(epoch_clips) // (world * per_rank)
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch, skip=0):
        self.epoch, self.skip = epoch, skip

    def epoch_indices(self, epoch):
        parts = []
        for s, (size, n, base) in enumerate(zip(self.sizes, self.counts, self.base)):
            pos, got = epoch * int(n), []
            while sum(len(g) for g in got) < n:
                p, o = divmod(pos, size)
                g = torch.Generator().manual_seed((self.seed * 1_000_003 + s) * 100_003 + p)
                take = min(int(n) - sum(len(x) for x in got), size - o)
                got.append(torch.randperm(size, generator=g)[o:o + take] + int(base))
                pos += take
            parts += got
        idx = torch.cat(parts)
        g = torch.Generator().manual_seed(self.seed * 100003 + epoch)
        return idx[torch.randperm(len(idx), generator=g)]

    def __iter__(self):
        perm = self.epoch_indices(self.epoch)[:self.steps * self.world * self.per_rank]
        perm = perm.view(self.steps, self.world, self.per_rank)[self.skip:, self.rank]
        return iter(perm.flatten().tolist())

    def __len__(self):
        return (self.steps - self.skip) * self.per_rank
