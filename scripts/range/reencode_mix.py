"""Re-encode the mix stores at short edge 256, JPEG 85, for ViT-T screens on Tamara.

ViT-T trains on 224 crops (RandomResizedCrop, area 0.8-1.0), so short edge 256
(V-JEPA 2's pretraining resolution) loses nothing it uses, and the stores shrink
from 11.0 TB to ~6.4 TB (measured per source: K710 0.45-0.49, PE Video 0.51,
VidaForge 0.73, EgoDex 0.42, DROID 0.88, SSv2 0.88, images 0.42-0.43), which is
what fits Tamara's scratch. Rows, episodes and their order are unchanged, so each
shard's provenance, stats and episode index are copied as they are.

One unit = one shard of <src>/<name>.lance (or, for k710_20pct, one slice of
200k rows of the 15 fps K710 20% store, re-assembled as one dataset). Output:
<out>/<src>/<name>.lance, stats json last, atomic claims like the builders, so
several Slurm jobs share the work. Consolidate afterwards with
  MIX_ROOT=/scratch/CLUSTER_USER/datasets/mix256 python scripts/range/consolidate_mix.py <sources>

  python scripts/range/reencode_mix.py --sources pe_video vidaforge ... --workers 100
"""
import argparse
import io
import json
import os
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

MIX = Path('/scratch/CLUSTER_USER/datasets/mix')
OUT = Path('/scratch/CLUSTER_USER/datasets/mix256')
K710_20 = Path('/scratch/CLUSTER_USER/datasets/k710/train_20pct.lance')
SHORT, QUALITY, CHUNK = 256, 85, 2048


def reencode(frames):
    from PIL import Image
    out = []
    for b in frames:
        with Image.open(io.BytesIO(b)) as im:
            im = im.convert('RGB')
            w, h = im.size
            s = min(w, h)
            if s > SHORT:
                im = im.resize((round(w * SHORT / s), round(h * SHORT / s)), Image.BICUBIC, reducing_gap=3.0)
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=QUALITY)
            out.append((buf.getvalue(), im.size[1], im.size[0]))
    return out


_DS = {}


def encode_span(args):
    """Worker: read rows [lo, hi) of a store itself and return them re-encoded as one batch."""
    import lance
    import pyarrow as pa
    path, lo, hi = args
    ds = _DS.get(path)
    if ds is None:
        _DS.clear()                         # one store at a time per worker
        ds = _DS[path] = lance.dataset(path)
    t = ds.to_table(offset=lo, limit=hi - lo).combine_chunks()
    enc = reencode(t['frame'].to_pylist())
    cols = {c: t[c].chunk(0) if t[c].num_chunks else t[c].combine_chunks() for c in t.column_names}
    cols['frame'] = pa.array([b for b, _, _ in enc], type=pa.binary())
    cols['h'] = pa.array([h for _, h, _ in enc], type=pa.int16())
    cols['w'] = pa.array([w for _, _, w in enc], type=pa.int16())
    return pa.RecordBatch.from_arrays([cols[f.name] for f in ds.schema], schema=ds.schema)


def _bounded(pool, spans):
    """Ordered results with at most 2 x workers chunks in flight. Pool.imap queues every
    result it finishes, which reached the job's 300 GB limit on the 505 GB K710 store."""
    window = 2 * pool._processes
    pending = []
    for sp in spans:
        pending.append(pool.apply_async(encode_span, (sp,)))
        if len(pending) >= window:
            yield pending.pop(0).get()
    while pending:
        yield pending.pop(0).get()


def convert(src_path, dst_path, pool):
    """Copy a Lance store row for row, re-encoding frames. One pool serves every store
    of the job: creating 104 spawn workers per shard cost more than the shard (5 min
    per 124k-row shard instead of ~10 s)."""
    import lance
    ds = lance.dataset(str(src_path))
    n, schema = ds.count_rows(), ds.schema
    tmp = dst_path.with_name('.tmp_' + dst_path.name)
    if tmp.exists():
        shutil.rmtree(tmp)
    spans = [(str(src_path), lo, min(lo + CHUNK, n)) for lo in range(0, n, CHUNK)]
    lance.write_dataset(_bounded(pool, spans), str(tmp), schema=schema, mode='create')
    assert lance.dataset(str(tmp)).count_rows() == n
    os.rename(tmp, dst_path)
    return n


def units(sources):
    out = []
    for s in sources:
        if s == 'k710_20pct':
            out.append(('k710_20pct', K710_20, Path('k710_20pct') / 'train_20pct.lance'))
            continue
        for p in sorted((MIX / s).glob('*.lance')):
            if Path(str(p) + '.stats.json').exists():
                out.append((s, p, Path(s) / p.name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 16)))
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    # Workers: maxtasksperchild because each worker's memory grows with every Lance
    # dataset it opens (two 300 GB jobs OOM-killed after ~1 h without it). Plain fork
    # deadlocked: replacement workers are forked after the parent has loaded Lance,
    # whose runtime is not fork-safe. forkserver forks every worker, replacements
    # included, from a clean server process, and starts them fast (spawn took ~6 min).
    pool = get_context('forkserver').Pool(a.workers, maxtasksperchild=25)
    for src, path, rel in units(a.sources):
        dst = a.out / rel
        if Path(str(dst) + '.stats.json').exists():
            continue
        claims = a.out / src / '.claims'
        claims.mkdir(parents=True, exist_ok=True)
        try:
            (claims / dst.stem).mkdir()
        except FileExistsError:
            continue
        (claims / dst.stem / 'owner').write_text(f'{os.uname().nodename} pid {os.getpid()} {time.ctime()}\n')
        t0 = time.time()
        if dst.exists():
            shutil.rmtree(dst)
        n = convert(path, dst, pool)
        for side in ('.prov.parquet', '.index.npz'):
            if Path(str(path) + side).exists():
                shutil.copy2(str(path) + side, str(dst) + side)
        st = json.loads(Path(str(path) + '.stats.json').read_text()) if Path(str(path) + '.stats.json').exists() else {}
        st.update(reencoded=dict(short_edge=SHORT, quality=QUALITY, rows=n, seconds=round(time.time() - t0, 1),
                                 host=os.uname().nodename, source_store=str(path)))
        Path(str(dst) + '.stats.json').write_text(json.dumps(st) + '\n')
        print(f'{rel}: {n} rows in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
