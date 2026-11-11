"""Frame-per-row Lance store for Kinetics clips, LeVJEPA's convention.

Same schema and conventions as LeVJEPA's build_lance_walking_tours.py, which
its loader (third_party/LeVJEPA/data/loader.py) reads unchanged: rows
(episode_idx, step_idx, frame, h, w, label), stored at 15 fps, short edge 384,
JPEG quality 90, one episode per Kinetics clip of up to 128 stored frames
(8.5 s of the 10 s clip; training samples 16 frames at stride 2 = 7.5 fps).
That matches the published K710 frame-Lance derivative (128 rows per clip,
short edge 384, JPEG 90). Decoding uses PyAV instead of decord.

  python scripts/range/build_lance_kinetics.py \
      --csv /scratch/CLUSTER_USER/datasets/k710/train_20pct.csv \
      --out /scratch/CLUSTER_USER/datasets/k710/train_20pct.lance --workers 24

Resumable in the sense that a finished store is never overwritten; a partial
build must be deleted and restarted (Lance writes are transactional per batch).
"""
import argparse
import csv
import io
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

SHORT_EDGE = 384
JPEG_Q = 90
TARGET_FPS = 15.0
EP_LEN = 128


def encode_clip(task):
    """Decode one clip with PyAV, resample to ~15 fps, resize, JPEG-encode."""
    import av
    from PIL import Image
    ep_idx, path, label = task
    try:
        out = []
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            fps = float(stream.average_rate or stream.guessed_rate or 30)
            stride = max(1, round(fps / TARGET_FPS))
            for i, frame in enumerate(container.decode(stream)):
                if i % stride:
                    continue
                img = frame.to_image()
                w, h = img.size
                if h < w:
                    nh, nw = SHORT_EDGE, round(w * SHORT_EDGE / h)
                else:
                    nh, nw = round(h * SHORT_EDGE / w), SHORT_EDGE
                img = img.resize((nw, nh), Image.BILINEAR)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=JPEG_Q)
                out.append((len(out), buf.getvalue(), nh, nw))
                if len(out) >= EP_LEN:
                    break
        return ep_idx, label, out
    except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the build
        print(f'  clip {ep_idx} {path} failed: {exc}', flush=True)
        return ep_idx, label, None


def main():
    import lance
    import pyarrow as pa
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 16)))
    ap.add_argument('--min-rows', type=int, default=32, help='drop clips shorter than a 16x2 clip span')
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists; delete it to rebuild')
    with open(a.csv, newline='') as f:
        rows = list(csv.DictReader(f))
    tasks = [(i, r['path'], int(r['k710_label'])) for i, r in enumerate(rows)]
    print(f'{len(tasks)} clips from {a.csv}', flush=True)
    schema = pa.schema([
        pa.field('episode_idx', pa.int32()), pa.field('step_idx', pa.int32()),
        pa.field('frame', pa.binary()), pa.field('h', pa.int16()),
        pa.field('w', pa.int16()), pa.field('label', pa.int16()),
    ])
    t0 = time.time()
    written = dict(eps=0, rows=0, failed=0, short=0)

    def batches():
        ep = 0
        with get_context('spawn').Pool(a.workers) as pool:
            for _, label, frames in pool.imap(encode_clip, tasks, chunksize=4):
                if frames is None:
                    written['failed'] += 1; continue
                if len(frames) < a.min_rows:
                    written['short'] += 1; continue
                written['eps'] += 1; written['rows'] += len(frames)
                yield pa.RecordBatch.from_arrays([
                    pa.array([ep] * len(frames), type=pa.int32()),
                    pa.array([s for s, _, _, _ in frames], type=pa.int32()),
                    pa.array([b for _, b, _, _ in frames], type=pa.binary()),
                    pa.array([h for _, _, h, _ in frames], type=pa.int16()),
                    pa.array([w for _, _, _, w in frames], type=pa.int16()),
                    pa.array([label] * len(frames), type=pa.int16()),
                ], schema=schema)
                ep += 1
                if written['eps'] % 500 == 0:
                    el = time.time() - t0; rate = written['eps'] / el
                    print(f"  {written['eps']}/{len(tasks)} clips, {written['rows']/1e6:.2f}M rows, "
                          f"{rate:.2f} clip/s, eta {(len(tasks)-written['eps'])/rate/60:.0f} min", flush=True)

    lance.write_dataset(batches(), str(out), schema=schema, mode='create')
    ds = lance.dataset(str(out))
    print(f"wrote {out}: {written} in {(time.time()-t0)/60:.1f} min; store rows {ds.count_rows()}", flush=True)
    assert ds.count_rows() == written['rows']
    Path(str(out) + '.stats.json').write_text(__import__('json').dumps(dict(written, clips=len(tasks), csv=a.csv)) + '\n')


if __name__ == '__main__':
    sys.exit(main())
