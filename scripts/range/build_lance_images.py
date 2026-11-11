"""Lance shards of still images for the mixes' image stream (docs/data_mixes.md).

One row per image in the video stores' schema, so the same Lance readers work:
(episode_idx = image index in the shard, step_idx = 0, frame, h, w, label).
Images are re-encoded at short edge min(native, 384), JPEG 90, matching the
video frames; label is the ImageNet class index (sorted wnids) or -1.

  in1k   train_NNN: 20,000 ImageNet-1K train images (class folders under --in1k-root)
  cc12m  part_NNN: 8 consecutive CC12M webdataset tars (~40k images)

  python scripts/range/build_lance_images.py --source in1k --claim-all --workers 100 \
      --in1k-root /scratch/CLUSTER_USER/datasets/imagenet/train

Same conventions as build_lance_mix.py: shards at <out>/<source>/<name>.lance,
stats json written last, atomic mkdir claims in <out>/<source>/.claims/.
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from multiprocessing import get_context
from pathlib import Path

RAW = Path('/scratch/CLUSTER_USER/datasets/raw')
OUT = Path('/scratch/CLUSTER_USER/datasets/mix')
SHORT_MAX = 384
JPEG_Q = 90
IN1K_PER_SHARD = 20000
CC12M_TARS = 8


def _in1k_index(root):
    wnids = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert len(wnids) == 1000, len(wnids)
    cache = root.with_name(root.name + '.files.txt')     # VAST/NFS listings are slow; reuse
    if cache.exists():
        files = cache.read_text().split()
    else:
        files = sorted(str(p) for w in wnids for p in (root / w).iterdir())
        cache.write_text('\n'.join(files) + '\n')
    lab = {w: i for i, w in enumerate(wnids)}
    return [(f, lab[Path(f).parent.name]) for f in files]


def list_units(source, in1k_root):
    if source == 'in1k':
        n = len(_in1k_index(in1k_root))
        return [dict(name=f'train_{k // IN1K_PER_SHARD:03d}', lo=k) for k in range(0, n, IN1K_PER_SHARD)]
    tars = sorted((RAW / 'cc12m').glob('cc12m-train-*.tar'))
    return [dict(name=f'part_{k // CC12M_TARS:03d}', tars=[str(t) for t in tars[k:k + CC12M_TARS]])
            for k in range(0, len(tars), CC12M_TARS)]


def unit_items(source, u, in1k_root):
    if source == 'in1k':
        return [('path', f, lab, f) for f, lab in _in1k_index(in1k_root)[u['lo']:u['lo'] + IN1K_PER_SHARD]]
    items = []
    for t in u['tars']:
        with tarfile.open(t) as tf:
            items += [('range', (t, m.offset_data, m.size), -1, f'{Path(t).stem}/{m.name}')
                      for m in tf if m.isfile() and m.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    return items


def encode_item(item):
    from PIL import Image
    kind, arg, label, key = item
    try:
        if kind == 'path':
            img = Image.open(arg)
        else:
            path, off, size = arg
            with open(path, 'rb') as f:
                f.seek(off)
                img = Image.open(io.BytesIO(f.read(size)))
        img = img.convert('RGB')
        w, h = img.size
        if min(w, h) > SHORT_MAX:
            w, h = (round(w * SHORT_MAX / h), SHORT_MAX) if h <= w else (SHORT_MAX, round(h * SHORT_MAX / w))
            img = img.resize((w, h), Image.BICUBIC, reducing_gap=3.0)
        if min(w, h) < 64:
            return key, None
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=JPEG_Q)
        return key, (buf.getvalue(), h, w, label)
    except Exception as exc:  # noqa: BLE001
        print(f'  {key} failed: {exc!r}', flush=True)
        return key, None


def build_unit(source, u, workers, out_root, in1k_root):
    import lance
    import pyarrow as pa
    import pyarrow.parquet as pq
    name = u['name']
    final = out_root / source / f'{name}.lance'
    stats_path = Path(str(final) + '.stats.json')
    if stats_path.exists():
        return
    tmp = out_root / source / f'.tmp_{name}.lance'
    import shutil
    for d in (tmp, final):
        if d.exists():
            shutil.rmtree(d)
    final.parent.mkdir(parents=True, exist_ok=True)
    items = unit_items(source, u, in1k_root)
    schema = pa.schema([pa.field('episode_idx', pa.int32()), pa.field('step_idx', pa.int32()),
                        pa.field('frame', pa.binary()), pa.field('h', pa.int16()),
                        pa.field('w', pa.int16()), pa.field('label', pa.int16())])
    st = dict(items=len(items), failed=0, rows=0)
    keys = []
    t0 = time.time()

    def batches():
        buf = []
        with get_context('spawn').Pool(workers) as pool:
            for key, r in pool.imap(encode_item, items, chunksize=64):
                if r is None:
                    st['failed'] += 1; continue
                buf.append(r); keys.append(key)
                if len(buf) == 1024:
                    yield flush(buf); buf = []
        if buf:
            yield flush(buf)

    def flush(buf):
        n0 = st['rows']; st['rows'] += len(buf)
        return pa.RecordBatch.from_arrays([
            pa.array(range(n0, n0 + len(buf)), type=pa.int32()), pa.array([0] * len(buf), type=pa.int32()),
            pa.array([b for b, _, _, _ in buf], type=pa.binary()), pa.array([h for _, h, _, _ in buf], type=pa.int16()),
            pa.array([w for _, _, w, _ in buf], type=pa.int16()), pa.array([l for *_, l in buf], type=pa.int16())],
            schema=schema)

    lance.write_dataset(batches(), str(tmp), schema=schema, mode='create')
    assert lance.dataset(str(tmp)).count_rows() == st['rows']
    pq.write_table(pa.table(dict(episode_idx=list(range(len(keys))), key=keys)), str(final) + '.prov.parquet')
    os.rename(tmp, final)
    st.update(seconds=round(time.time() - t0, 1), source=source, name=name, host=os.uname().nodename)
    stats_path.write_text(json.dumps(st) + '\n')
    print(f'{source}/{name}: {st}', flush=True)


def main():
    global RAW
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, choices=['in1k', 'cc12m'])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--units', nargs='*', default=[])
    ap.add_argument('--claim-all', action='store_true')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 16)))
    ap.add_argument('--in1k-root', type=Path, default=Path('/scratch/CLUSTER_USER/datasets/imagenet/train'))
    ap.add_argument('--raw', type=Path, default=RAW)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    RAW = a.raw
    us = list_units(a.source, a.in1k_root)
    if a.list:
        for u in us:
            print(u['name'], ','.join(Path(t).relative_to(RAW).as_posix() for t in u.get('tars', [])) or '-', sep='\t')
        return
    by = {u['name']: u for u in us}
    for n in a.units:
        build_unit(a.source, by[n], a.workers, a.out, a.in1k_root)
    if a.claim_all:
        for u in us:
            d = a.out / a.source / '.claims'
            d.mkdir(parents=True, exist_ok=True)
            if (a.out / a.source / f"{u['name']}.lance.stats.json").exists():
                continue
            try:
                (d / u['name']).mkdir()
            except FileExistsError:
                continue
            (d / u['name'] / 'owner').write_text(f'{os.uname().nodename} pid {os.getpid()} {time.ctime()}\n')
            build_unit(a.source, u, a.workers, a.out, a.in1k_root)


if __name__ == '__main__':
    sys.exit(main())
