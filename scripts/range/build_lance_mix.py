"""Frame-per-row Lance shards for the pretraining mixes (docs/data_mixes.md).

Same row schema as build_lance_kinetics.py, so LeVJEPA's VJEPAClipDataset reads
the shards unchanged: (episode_idx, step_idx, frame, h, w, label), label -1.
Differences from the K710 store, all chosen to keep ~13k hours of new video
near 10 TB:
  - 7.5 fps instead of 15 (training reads every 2nd row of the 15 fps K710
    store, so a 7.5 fps shard is read at stride 1 and yields the same clip);
  - short edge min(native, 384): never upscaled (VidaForge is mostly 256p,
    SSv2 240p, DROID 180p);
  - long videos are cut into consecutive episodes of at most 64 rows (8.5 s,
    the K710 episode length), so one clip per episode samples time uniformly;
    remainders under 16 rows (one training clip) are dropped.
Frames are sampled by presentation time, not by frame index, so variable and
non-integer source rates (29.97, 25, 12) land on the same 7.5 fps grid.

One shard = one unit of raw input, named after the file it reads:
  ssv2       ids_NNN: 2,000 training videos (train.json ids only; val/test are probes)
  pe_video   train_NNNNNN / extended_NNNNNN: one 500-video tar
  vidaforge  shard-NNNNN: one tar, filtered with its meta parquet (real-world
             domain, best clip of each Cosmos semantic-duplicate group, >= 2.2 s)
  egodex     partK_NNN: 2,000 mp4s of one zip, read in place from the zip
  droid      <camera>_chunk-NNN_file-NNN: one concatenated camera video, cut at
             LeRobot episode boundaries
  k710       rest80_NNN: 2,000 clips of the 80% of K710 outside the existing
             20% store (k710/train_rest80.csv); RANGE only (mp4s live there)

  python scripts/range/build_lance_mix.py --source pe_video --list
  python scripts/range/build_lance_mix.py --source pe_video --units extended_000000 --workers 100
  python scripts/range/build_lance_mix.py --source pe_video --claim-all --workers 100

Shards go to <out>/<source>/<name>.lance via a temporary name, with a
provenance parquet (episode -> raw key and start time) and a stats json, which
is written last and marks the shard complete. --claim-all takes an atomic
mkdir claim in <out>/<source>/.claims/ before building, so any number of
builders (Slurm jobs here, the Tamara driver remotely) can share one queue.
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
import zipfile
from multiprocessing import get_context
from pathlib import Path

RAW = Path('/scratch/CLUSTER_USER/datasets/raw')
OUT = Path('/scratch/CLUSTER_USER/datasets/mix')
FPS = 7.5
EP_ROWS = 64
MIN_ROWS = 16
SHORT_MAX = 384
JPEG_Q = 90
PER_SHARD = 2000


# ---------------------------------------------------------------- work items
# A unit is one shard: {'name', 'files', ...}. 'name' is stable (derived from
# the raw file it reads, never from a list position), so several builders on
# different hosts can share one claim directory. 'files' are the raw paths
# (relative to RAW) the unit needs, which is what a remote builder must copy.
# An item is (key, reader, args, windows): the reader opens a path, a byte
# range of a tar, or a zip member; windows is None (whole video) or a list of
# (t0, t1) seconds to decode.

def _ssv2_units(gate):
    if gate and not (RAW / 'ssv2/extracted/.complete').exists():
        return []                               # archive still extracting
    n = len(json.loads((RAW / 'ssv2/train.json').read_text()))
    return [dict(name=f'ids_{k // PER_SHARD:03d}', files=['ssv2/train.json', 'ssv2/extracted'], lo=k)
            for k in range(0, n, PER_SHARD)]


def _pe_units(gate):
    return [dict(name=f'{split}_{t.stem}', files=[f'pe_video/{split}/{t.name}'])
            for split in ('train', 'extended') for t in sorted((RAW / 'pe_video' / split).glob('*.tar'))]


def _vidaforge_units(gate):
    return [dict(name=t.stem, files=[f'vidaforge/data/{t.name}', f'vidaforge/meta/{t.stem}.parquet'])
            for t in sorted((RAW / 'vidaforge/data').glob('shard-*.tar'))
            if (RAW / 'vidaforge/meta' / f'{t.stem}.parquet').exists()]


def _egodex_units(gate):
    out = []
    for z in sorted((RAW / 'egodex').glob('*.zip')):
        if not Path(str(z) + '.ok').exists():
            continue
        with zipfile.ZipFile(z) as zf:
            n = sum(1 for m in zf.namelist() if m.endswith('.mp4'))
        out += [dict(name=f'{z.stem}_{k // PER_SHARD:03d}', files=[f'egodex/{z.name}'], lo=k)
                for k in range(0, n, PER_SHARD)]
    return out


def _droid_units(gate):
    if gate and not (RAW / 'droid/.complete').exists():
        return []                               # episode metadata must be complete
    out = []
    index = _droid_index() if gate else None
    for v in sorted((RAW / 'droid/videos').glob('observation.images.*/chunk-*/file-*.mp4')):
        # The last ~10 files of each camera are referenced by no episode in
        # meta/episodes (29 of 813), so they have no boundaries to cut at: skip.
        if index is not None and (v.parts[-3], int(v.parent.name.split('-')[1]), int(v.stem.split('-')[1])) not in index:
            continue
        cam = v.parts[-3].split('.')[-1]
        out.append(dict(name=f'{cam}_{v.parent.name}_{v.stem}', files=[str(v.relative_to(RAW)), 'droid/meta/episodes']))
    return out


def _k710_units(gate):
    """The 80% of K710 outside the existing 20% store (train_rest80.csv), local to RANGE."""
    import csv
    rows = list(csv.DictReader(open('/scratch/CLUSTER_USER/datasets/k710/train_rest80.csv', newline='')))
    return [dict(name=f'rest80_{k // PER_SHARD:03d}', files=['(k710 mp4s on RANGE)'], lo=k)
            for k in range(0, len(rows), PER_SHARD)]


def list_units(source, gate=True):
    """Units whose raw files are fully present. gate=False skips the whole-source
    completeness markers (a remote builder holds only the files of one unit)."""
    return {'ssv2': _ssv2_units, 'pe_video': _pe_units, 'vidaforge': _vidaforge_units,
            'egodex': _egodex_units, 'droid': _droid_units, 'k710': _k710_units}[source](gate)


def _tar_items(tar_path):
    with tarfile.open(tar_path) as tf:
        return [(m.name, 'range', (str(tar_path), m.offset_data, m.size), None)
                for m in tf if m.isfile() and m.name.endswith('.mp4')]


def _vidaforge_items(tar_path):
    """Real-world clips, the best clip of each Cosmos semantic-duplicate group, >= 2.2 s."""
    import pyarrow.parquet as pq
    meta = RAW / 'vidaforge/meta' / (tar_path.stem + '.parquet')
    t = pq.read_table(meta, columns=['clip_id', 'tar_path', 'tar_offset', 'filesize_bytes', 'duration_sec',
                                     'tag_domain', 'cosmos_is_best_clip_in_group']).to_pylist()
    out = []
    for r in t:
        assert r['tar_path'].endswith(tar_path.name), (r['tar_path'], tar_path)
        if r['tag_domain'] != 'real_world' or not r['cosmos_is_best_clip_in_group'] or r['duration_sec'] < 2.2:
            continue
        out.append((r['clip_id'], 'range', (str(tar_path), int(r['tar_offset']), int(r['filesize_bytes'])), None))
    return out, len(t)


_DROID_INDEX = {}


def _droid_index():
    """(camera key, chunk, file) -> [(episode, from_s, to_s)], read once per process."""
    if not _DROID_INDEX:
        import pyarrow.parquet as pq
        for f in sorted((RAW / 'droid/meta/episodes').glob('chunk-*/file-*.parquet')):
            names = pq.read_schema(f).names
            cams = sorted({n.split('/')[1] for n in names if n.startswith('videos/observation.images.')})
            for cam in cams:
                pre = f'videos/{cam}/'
                cols = ['episode_index', pre + 'chunk_index', pre + 'file_index', pre + 'from_timestamp', pre + 'to_timestamp']
                t = pq.read_table(f, columns=cols).to_pydict()
                for e, c, fi, a, b in zip(*(t[k] for k in cols)):
                    _DROID_INDEX.setdefault((cam, int(c), int(fi)), []).append((int(e), float(a), float(b)))
    return _DROID_INDEX


def unit_items(source, u):
    """Expand one unit into decode items; returns (items, raw_count)."""
    if source == 'ssv2':
        ids = sorted((r['id'] for r in json.loads((RAW / 'ssv2/train.json').read_text())), key=int)
        vids = RAW / 'ssv2/extracted/20bn-something-something-v2'
        it = [(i, 'path', str(vids / f'{i}.webm'), None) for i in ids[u['lo']:u['lo'] + PER_SHARD]]
        return it, len(it)
    if source == 'k710':
        import csv
        rows = list(csv.DictReader(open('/scratch/CLUSTER_USER/datasets/k710/train_rest80.csv', newline='')))
        it = [(r['youtube_id'], 'path', r['path'], None) for r in rows[u['lo']:u['lo'] + PER_SHARD]]
        return it, len(it)
    if source == 'pe_video':
        it = _tar_items(RAW / u['files'][0]); return it, len(it)
    if source == 'vidaforge':
        return _vidaforge_items(RAW / u['files'][0])
    if source == 'egodex':
        z = RAW / u['files'][0]
        with zipfile.ZipFile(z) as zf:
            names = sorted(m for m in zf.namelist() if m.endswith('.mp4'))[u['lo']:u['lo'] + PER_SHARD]
        return [(n, 'zip', (str(z), n), None) for n in names], len(names)
    if source == 'droid':
        v = RAW / u['files'][0]
        cam, chunk, file = v.parts[-3], int(v.parent.name.split('-')[1]), int(v.stem.split('-')[1])
        wins = sorted(_droid_index().get((cam, chunk, file), []), key=lambda w: w[1])
        assert wins, f'no episodes indexed for {v}'
        return [(f'{cam}/ep{e}', 'path', str(v), [(x, y)]) for e, x, y in wins], len(wins)
    raise ValueError(source)


# ------------------------------------------------------------------- decoding
_ZIPS = {}


def _open(kind, arg):
    if kind == 'path':
        return arg
    if kind == 'range':
        path, off, size = arg
        with open(path, 'rb') as f:
            f.seek(off)
            return io.BytesIO(f.read(size))
    if kind == 'zip':
        path, name = arg
        if path not in _ZIPS:
            _ZIPS[path] = zipfile.ZipFile(path)
        return io.BytesIO(_ZIPS[path].read(name))
    raise ValueError(kind)


def _encode(frame):
    w, h = frame.width, frame.height
    s = min(w, h)
    if s > SHORT_MAX:
        w, h = (round(w * SHORT_MAX / h), SHORT_MAX) if h <= w else (SHORT_MAX, round(h * SHORT_MAX / w))
        w += w % 2; h += h % 2
    img = frame.reformat(width=w, height=h, format='rgb24', interpolation='AREA').to_image()
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=JPEG_Q)
    return buf.getvalue(), h, w


def decode_item(item):
    """-> (key, [(t_start, [(jpeg, h, w), ...]) per episode]) or (key, None) on failure."""
    import av
    import signal
    key, kind, arg, windows = item

    def _timeout(*_):
        raise TimeoutError('decode took over 300 s')
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(300)                           # backstop against any other hang in one video
    try:
        src = _open(kind, arg)
        eps = []
        with av.open(src) as c:
            st = c.streams.video[0]
            st.thread_type = 'AUTO'
            st.codec_context.thread_count = 2
            for t0, t1 in (windows or [(None, None)]):
                if t0 is not None and t0 > 0:
                    c.seek(int(t0 / st.time_base), stream=st, backward=True)
                frames, start, nxt = [], None, t0
                for fr in c.decode(st):
                    t = fr.time
                    if t is None:
                        continue
                    if t1 is not None and t >= t1 - 1e-6:
                        break
                    if t0 is not None and t < t0 - 1e-6:
                        continue
                    if nxt is None:
                        nxt = t
                    if t < nxt - 1e-3:
                        continue
                    # next grid time after t in O(1): a corrupt timestamp jump (seen in
                    # VidaForge) made a step-by-step loop spin for hours
                    nxt += (int((t + 1e-3 - nxt) * FPS) + 1) / FPS
                    if start is None:
                        start = t
                    frames.append(_encode(fr))
                for k in range(0, len(frames), EP_ROWS):
                    chunk = frames[k:k + EP_ROWS]
                    if len(chunk) >= MIN_ROWS:
                        eps.append((start + k / FPS, chunk))
        return key, eps
    except Exception as exc:  # noqa: BLE001 - one bad video must not kill a shard
        print(f'  {key} failed: {exc!r}', flush=True)
        return key, None
    finally:
        signal.alarm(0)


# -------------------------------------------------------------------- writing
def done(out_root, source, name):
    return (out_root / source / f'{name}.lance.stats.json').exists()


def claim(out_root, source, name):
    """Atomic claim shared by every builder that sees out_root (mkdir is atomic on VAST)."""
    d = out_root / source / '.claims'
    d.mkdir(parents=True, exist_ok=True)
    try:
        (d / name).mkdir()
    except FileExistsError:
        return False
    (d / name / 'owner').write_text(f'{os.uname().nodename} pid {os.getpid()} {time.ctime()}\n')
    return True


def build_unit(source, u, workers, out_root):
    import lance
    import pyarrow as pa
    import pyarrow.parquet as pq
    name = u['name']
    final = out_root / source / f'{name}.lance'
    stats_path = Path(str(final) + '.stats.json')
    if stats_path.exists():
        print(f'skip {source}/{name} (done)', flush=True)
        return
    tmp = out_root / source / f'.tmp_{name}.lance'
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    if final.exists():                      # renamed but stats never written: rebuild
        import shutil
        shutil.rmtree(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    items, raw_count = unit_items(source, u)
    schema = pa.schema([pa.field('episode_idx', pa.int32()), pa.field('step_idx', pa.int32()),
                        pa.field('frame', pa.binary()), pa.field('h', pa.int16()),
                        pa.field('w', pa.int16()), pa.field('label', pa.int16())])
    st = dict(raw=raw_count, items=len(items), failed=0, empty=0, episodes=0, rows=0)
    prov = dict(episode_idx=[], key=[], t_start=[])
    t0 = time.time()

    def batches():
        ep = 0
        with get_context('spawn').Pool(workers) as pool:
            for key, eps in pool.imap(decode_item, items, chunksize=2):
                if eps is None:
                    st['failed'] += 1; continue
                if not eps:
                    st['empty'] += 1; continue
                for ts, fr in eps:
                    n = len(fr)
                    st['episodes'] += 1; st['rows'] += n
                    prov['episode_idx'].append(ep); prov['key'].append(key); prov['t_start'].append(float(ts))
                    yield pa.RecordBatch.from_arrays([
                        pa.array([ep] * n, type=pa.int32()), pa.array(range(n), type=pa.int32()),
                        pa.array([b for b, _, _ in fr], type=pa.binary()),
                        pa.array([h for _, h, _ in fr], type=pa.int16()),
                        pa.array([w for _, _, w in fr], type=pa.int16()),
                        pa.array([-1] * n, type=pa.int16())], schema=schema)
                    ep += 1

    if not items:
        raise SystemExit(f'{source}/{name}: no items')
    lance.write_dataset(batches(), str(tmp), schema=schema, mode='create')
    rows = lance.dataset(str(tmp)).count_rows()
    assert rows == st['rows'], (rows, st)
    pq.write_table(pa.table(prov), str(final) + '.prov.parquet')
    os.rename(tmp, final)
    st.update(seconds=round(time.time() - t0, 1), hours_out=round(st['rows'] / FPS / 3600, 3),
              source=source, name=name, files=u['files'], host=os.uname().nodename)
    stats_path.write_text(json.dumps(st) + '\n')          # written last: marks the shard complete
    print(f'{source}/{name}: {st}', flush=True)


def main():
    global RAW
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, choices=['ssv2', 'pe_video', 'vidaforge', 'egodex', 'droid', 'k710'])
    ap.add_argument('--list', action='store_true', help='print "name<TAB>raw files" per unit and exit')
    ap.add_argument('--units', nargs='*', default=[], help='build these units (no claiming)')
    ap.add_argument('--claim-all', action='store_true', help='claim and build every unclaimed unit')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 16)))
    ap.add_argument('--raw', type=Path, default=RAW)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    RAW = a.raw
    us = list_units(a.source, gate=not a.units)
    if a.list:
        for u in us:
            print(u['name'], ','.join(u['files']), sep='\t')
        return
    by = {u['name']: u for u in us}
    for n in a.units:
        build_unit(a.source, by[n], a.workers, a.out)
    if a.claim_all:
        for u in us:
            if not done(a.out, a.source, u['name']) and claim(a.out, a.source, u['name']):
                build_unit(a.source, u, a.workers, a.out)


if __name__ == '__main__':
    sys.exit(main())
