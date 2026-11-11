"""Fixed evaluation inputs for every natural-video checkpoint.

Every checkpoint of every method is probed on byte-identical inputs, decoded
once: short edge resized to 256 (bilinear), center crop 224, uint8, no
augmentation. Written as .npy memmaps under /scratch/CLUSTER_USER/datasets/eval_caches.

  in1k_train  300 images per class, seeded   [300000, 224, 224, 3]
  in1k_train_full, k400_train_full   the complete training sets (--only)
  in1k_val    all 50,000                     [ 50000, 224, 224, 3]
  k400_train  150 clips per class, seeded    [ 60000, 8, 224, 224, 3]
  k400_val    all validation clips           [~19881, 8, 224, 224, 3]
  ssv2_train  150 clips per class, seeded    [ 26100, 8, 224, 224, 3]
  ssv2_val    all validation clips           [ 24777, 8, 224, 224, 3]
  ssv2_train_full                            the complete SSv2 train set (--only)
  in1k_v2, in1k_sketch, in1k_r, in1k_a       ImageNet robustness sets (eval only), labels in
              the IN1K index space; R and A cover 200 classes (in1k_r/a_classes.npy)

Video clips are 8 frames at ~7.5 fps (every round(fps/7.5)-th frame) from the
temporal center of the 10 s clip, matching the rate LeVJEPA trains at and its
8-slot evaluation length for per-frame tokenization. SSv2 clips are 2-6 s and
the action spans the whole clip, so their 8 frames are spread uniformly over
the full video (the usual SSv2 sampling) instead of a 1 s center window.
The ImageNet robustness sets come from the clip-benchmark webdataset tars;
their class names are matched to IN1K indices through IN-v2's 1000-name list. Labels use the sorted
class-folder order. Undecodable items keep valid=False and zero pixels.
Subsets are for the scaling curves; the final table number for a model uses
the full protocol separately.

Run on a login node under tmux (CPU only), resumable per split:
  python scripts/natural/build_eval_caches.py --workers 24
"""
import argparse
import json
import random
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

D = Path('/scratch/CLUSTER_USER/datasets')
OUT = D / 'eval_caches'
SIZE, SHORT, FRAMES, FPS = 224, 256, 8, 7.5


def resize_crop(img):
    from PIL import Image
    w, h = img.size
    s = SHORT / min(w, h)
    img = img.convert('RGB').resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return np.asarray(img.crop((l, t, l + SIZE, t + SIZE)), dtype=np.uint8)


def load_image(path):
    from PIL import Image
    with Image.open(path) as im:
        return resize_crop(im)


def load_clip(path):
    """8 frames at ~7.5 fps from the temporal center. When the container reports its
    frame count, decoding stops after the last needed frame (same frames as before)."""
    import av
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = 'AUTO'
        fps = float(st.average_rate or st.guessed_rate or 30)
        stride = max(1, round(fps / FPS))
        span = (FRAMES - 1) * stride + 1
        total = int(st.frames or 0)
        if total > 0:
            start = max(0, (total - span) // 2)
            want = {min(total - 1, start + i * stride) for i in range(FRAMES)}
            frames = {}
            for k, f in enumerate(c.decode(st)):
                if k in want:
                    frames[k] = f
                if k >= max(want):
                    break
            if len(frames) == len(want):
                idx = [min(total - 1, start + i * stride) for i in range(FRAMES)]
                return np.stack([resize_crop(frames[i].to_image()) for i in idx])
            c.seek(0)                            # frame count was wrong: fall back to decoding everything
        frames = [f for f in c.decode(st)]
    n = len(frames)
    start = max(0, (n - span) // 2)
    idx = [min(n - 1, start + i * stride) for i in range(FRAMES)]
    return np.stack([resize_crop(frames[i].to_image()) for i in idx])


def load_clip_uniform(path):
    """8 frames spread uniformly over the whole video (SSv2)."""
    import av
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = 'AUTO'
        frames = [f for f in c.decode(st)]
    n = len(frames)
    idx = [min(n - 1, int((i + 0.5) * n / FRAMES)) for i in range(FRAMES)]
    return np.stack([resize_crop(frames[i].to_image()) for i in idx])


def load_tar_image(spec):
    """'<tar>::<offset>::<size>': one image's bytes inside a webdataset tar."""
    import io
    from PIL import Image
    tar, off, size = spec.split('::')
    with open(tar, 'rb') as f:
        f.seek(int(off))
        data = f.read(int(size))
    with Image.open(io.BytesIO(data)) as im:
        return resize_crop(im)


LOADERS = {'image': lambda p: load_image(p), 'video': lambda p: load_clip(p),
           'video_uniform': lambda p: load_clip_uniform(p), 'tar_image': lambda p: load_tar_image(p)}


def work(chunk):
    """Decode a chunk of items and write them with one memmap open and one
    flush. Flushing per item costs an msync round trip to VAST each time and
    held ImageNet to 25 images/s."""
    target, kind, entries = chunk
    m = np.load(target, mmap_mode='r+')
    done = []
    for i, path in entries:
        try:
            m[i] = LOADERS[kind](path)
            done.append((i, True))
        except Exception:  # noqa: BLE001
            done.append((i, False))
    m.flush(); del m
    return done


def build(name, items, kind, workers):
    """items: list of (path, label)."""
    target = OUT / f'{name}.npy'
    meta = OUT / f'{name}.json'
    if meta.exists():
        print(f'{name}: already built', flush=True); return
    still = kind in ('image', 'tar_image')
    shape = (len(items), SIZE, SIZE, 3) if still else (len(items), FRAMES, SIZE, SIZE, 3)
    m = np.lib.format.open_memmap(target, mode='w+', dtype=np.uint8, shape=shape); del m
    size = 512 if still else 32
    entries = [(i, str(p)) for i, (p, _) in enumerate(items)]
    chunks = [(str(target), kind, entries[j:j + size]) for j in range(0, len(entries), size)]
    valid = np.zeros(len(items), dtype=bool)
    t0 = time.time(); k = 0; mark = 0
    with get_context('spawn').Pool(workers) as pool:
        for done in pool.imap_unordered(work, chunks):
            for i, ok in done:
                valid[i] = ok
            k += len(done)
            if k - mark >= 5000:
                mark = k; r = k / (time.time() - t0)
                print(f'  {name}: {k}/{len(items)} {r:.0f}/s eta {(len(items)-k)/r/60:.0f} min', flush=True)
    np.save(OUT / f'{name}_labels.npy', np.array([l for _, l in items], dtype=np.int16))
    np.save(OUT / f'{name}_valid.npy', valid)
    meta.write_text(json.dumps(dict(name=name, count=len(items), valid=int(valid.sum()), shape=list(shape),
                                    kind=kind, size=SIZE, short_edge=SHORT, frames=FRAMES, fps=FPS,
                                    seconds=time.time() - t0)) + '\n')
    print(f'{name}: {int(valid.sum())}/{len(items)} valid in {(time.time()-t0)/60:.1f} min', flush=True)


def class_items(root, pattern, per_class, seed):
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    rng = random.Random(seed)
    items = []
    for ci, c in enumerate(classes):
        files = sorted((root / c).glob(pattern))
        if per_class:
            files = rng.sample(files, min(per_class, len(files)))
        items += [(f, ci) for f in files]
    return classes, items


RAW = D / 'raw'


def ssv2_items(split, per_class, seed):
    """SSv2 (train.json / validation.json) -> [(webm path, label)], labels from labels.json."""
    labels = json.loads((RAW / 'ssv2/labels.json').read_text())
    rows = json.loads((RAW / f'ssv2/{split}.json').read_text())
    vids = RAW / 'ssv2/extracted/20bn-something-something-v2'
    items = [(vids / f"{r['id']}.webm", int(labels[r['template'].replace('[', '').replace(']', '')]))
             for r in rows]
    if per_class:
        rng = random.Random(seed)
        by = {}
        for it in items:
            by.setdefault(it[1], []).append(it)
        items = [x for c in sorted(by) for x in rng.sample(by[c], min(per_class, len(by[c])))]
    names = sorted(labels, key=lambda k: int(labels[k]))
    return names, items


def variant_items(name):
    """clip-benchmark wds tars -> [('tar::offset::size', IN1K label)], plus the IN1K indices the set covers."""
    import tarfile
    ref = (RAW / 'in1k_variants/imagenetv2/classnames.txt').read_text().splitlines()
    assert len(ref) == 1000
    root = RAW / 'in1k_variants' / name
    own = (root / 'classnames.txt').read_text().splitlines()
    if own == ref:
        to_in1k = list(range(1000))
    else:
        # Subsets list their classes in IN1K index order. CLIP's names repeat twice
        # ('missile' 657/744, 'sunglasses' 836/837), so a repeated name takes the
        # first candidate after the previous class, and the result must be increasing.
        to_in1k, prev = [], -1
        for n in own:
            cand = [i for i, r in enumerate(ref) if r == n and i > prev]
            assert cand, f'{name}: class {n!r} not in IN1K after index {prev}'
            prev = cand[0]
            to_in1k.append(prev)
    items = []
    for tar in sorted((root / 'test').glob('*.tar'), key=lambda p: int(p.stem)):
        with tarfile.open(tar) as tf:
            members = {m.name: m for m in tf}
            for m in sorted(members):
                if m.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    key = m.rsplit('.', 1)[0]
                    cls = int(tf.extractfile(members[key + '.cls']).read().decode().strip())
                    items.append((f'{tar}::{members[m].offset_data}::{members[m].size}', to_in1k[cls]))
    return sorted(set(to_in1k)), items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = [
        ('k400_val', D / 'k400/val', '*.mp4', None, 'video'),
        ('in1k_val', D / 'imagenet/val', '*.JPEG', None, 'image'),
        ('k400_train', D / 'k400/train', '*.mp4', 150, 'video'),
        ('in1k_train', D / 'imagenet/train', '*.JPEG', 300, 'image'),
        # complete training sets (~290 GB and ~193 GB), built only when named with --only
        ('k400_train_full', D / 'k400/train', '*.mp4', None, 'video'),
        ('in1k_train_full', D / 'imagenet/train', '*.JPEG', None, 'image'),
        ('ssv2_val', 'validation', None, None, 'video_uniform'),
        ('ssv2_train', 'train', None, 150, 'video_uniform'),
        ('ssv2_train_full', 'train', None, None, 'video_uniform'),
        ('in1k_v2', 'imagenetv2', None, None, 'tar_image'),
        ('in1k_sketch', 'imagenet_sketch', None, None, 'tar_image'),
        ('in1k_r', 'imagenet-r', None, None, 'tar_image'),
        ('in1k_a', 'imagenet-a', None, None, 'tar_image'),
    ]
    names = {}
    for name, root, pat, per, kind in plan:
        if a.only and name not in a.only:
            continue
        if not a.only and name.endswith('_full'):
            continue
        if kind == 'video_uniform':
            classes, items = ssv2_items(root, per, seed=0)
            names['ssv2'] = classes
        elif kind == 'tar_image':
            covered, items = variant_items(root)
            np.save(OUT / f'{name}_classes.npy', np.array(covered, dtype=np.int16))
        else:
            classes, items = class_items(root, pat, per, seed=0)
            names[name.split('_')[0]] = classes
        print(f'{name}: {len(items)} items', flush=True)
        build(name, items, kind, a.workers)
    for k, v in names.items():
        (OUT / f'{k}_classes.json').write_text(json.dumps(v) + '\n')


if __name__ == '__main__':
    main()
