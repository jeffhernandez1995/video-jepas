"""Re-encode mix shards to 256 px on Tamara, pulling the 384-px shards from RANGE.

For when RANGE re-encode jobs (scripts/range/reencode_mix.py) are stuck in the
queue: same conversion (reencode_mix.convert), written straight into Tamara's
copy of the 256-px stores (/scratch/CLUSTER_USER/datasets/mix256/<src>/), which
scripts/tamara/pull_mix256.sh then consolidates. Shards already on Tamara (built
here or pulled from RANGE) are skipped; the next shard is fetched into /dev/shm
while the current one converts. Claims are local to Tamara, so RANGE jobs may
redo some shards for RANGE's own copy.

  python scripts/tamara/reencode_remote.py --sources egodex droid in1k pe_video --workers 40
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

R = 'CLUSTER_USER@RANGE_HOST'
SOCK = os.path.expanduser('~/.ssh/range_master.sock')
SRC = '/scratch/CLUSTER_USER/datasets/mix'
DST = Path('/scratch/CLUSTER_USER/datasets/mix256')
STAGE = Path('/dev/shm/reencode')
SSH = ['ssh', '-n', '-S', SOCK, '-o', 'BatchMode=yes', R]


def remote_done(src):
    out = subprocess.run(SSH + [f"cd {SRC}/{src} && ls | sed -n 's/\\.lance\\.stats\\.json$//p'"],
                         capture_output=True, text=True, check=True).stdout
    return sorted(out.split())


def fetch(src, name):
    d = STAGE / src
    d.mkdir(parents=True, exist_ok=True)
    paths = [f'{R}:{SRC}/{src}/{name}.lance'] + [f'{R}:{SRC}/{src}/{name}.lance{s}'
                                                 for s in ('.stats.json', '.prov.parquet', '.index.npz')]
    return subprocess.Popen(['rsync', '-a', '-W', '--no-compress', '-e', f'ssh -S {SOCK} -o BatchMode=yes']
                            + paths + [str(d) + '/'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=40)
    a = ap.parse_args()
    # import before forking (workers unpickle reencode_mix.encode_span), but fork before
    # anything imports lance/pyarrow: reencode_mix imports them only inside functions
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'range'))
    import reencode_mix as R256
    pool = get_context('forkserver').Pool(a.workers, maxtasksperchild=25)
    for src in a.sources:
        (DST / src).mkdir(parents=True, exist_ok=True)
        claims = DST / src / '.claims_tamara'
        claims.mkdir(exist_ok=True)
        todo = []
        for n in remote_done(src):
            if (DST / src / f'{n}.lance.stats.json').exists():
                continue
            try:
                (claims / n).mkdir()
            except FileExistsError:
                continue
            todo.append(n)
        print(f'{time.ctime()} {src}: {len(todo)} shards to re-encode here', flush=True)
        nxt = fetch(src, todo[0]) if todo else None
        for i, n in enumerate(todo):
            cur = nxt
            if cur.wait() != 0:
                print(f'  fetch failed {src}/{n}: {cur.stderr.read().decode()[-300:]}', flush=True)
                shutil.rmtree(claims / n, ignore_errors=True)
                nxt = fetch(src, todo[i + 1]) if i + 1 < len(todo) else None
                continue
            nxt = fetch(src, todo[i + 1]) if i + 1 < len(todo) else None
            if (DST / src / f'{n}.lance.stats.json').exists():        # pulled from RANGE meanwhile
                shutil.rmtree(STAGE / src / f'{n}.lance', ignore_errors=True)
                continue
            t0 = time.time()
            st = STAGE / src / f'{n}.lance'
            dst = DST / src / f'{n}.lance'
            if dst.exists():
                shutil.rmtree(dst)
            rows = R256.convert(st, dst, pool)
            for s in ('.prov.parquet', '.index.npz'):
                if Path(str(st) + s).exists():
                    shutil.copy2(str(st) + s, str(dst) + s)
            stats = json.loads(Path(str(st) + '.stats.json').read_text())
            stats.update(reencoded=dict(short_edge=R256.SHORT, quality=R256.QUALITY, rows=rows,
                                        seconds=round(time.time() - t0, 1), host='tamara'))
            Path(str(dst) + '.stats.json').write_text(json.dumps(stats) + '\n')
            for f in STAGE.glob(f'{src}/{n}.lance*'):
                shutil.rmtree(f) if f.is_dir() else f.unlink()
            print(f'{src}/{n}: {rows} rows in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
