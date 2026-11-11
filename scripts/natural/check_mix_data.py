"""Login-node check of mix_data: index a small mix, draw batches through a spawn DataLoader,
and verify the sampler is deterministic, rank-disjoint and hits the per-source counts.
  python scripts/natural/check_mix_data.py --spec /scratch/CLUSTER_USER/tmp/mix_smoke.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mix_data  # noqa: E402
from train_nlp import GlobalCrop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--batches', type=int, default=4)
    a = ap.parse_args()
    resolved = mix_data.resolve(json.loads(Path(a.spec).read_text()))
    t = time.time()
    ds = mix_data.MixDataset(resolved, 16, GlobalCrop(224, [0.8, 1.0]))
    print('indexed in %.1fs' % (time.time() - t), json.dumps(ds.describe()))
    s0 = mix_data.MixEpochSampler(ds.sizes, ds.weights, 4096, 0, 2, 64, 0)
    s1 = mix_data.MixEpochSampler(ds.sizes, ds.weights, 4096, 1, 2, 64, 0)
    a0, b0, r1 = list(s0), list(s0), list(s1)
    assert a0 == b0, 'not deterministic'
    assert not set(a0) & set(r1), 'ranks overlap'
    src = np.searchsorted(ds.cum, np.array(a0 + r1), side='right') - 1
    print('per-source share', {ds.sources[i].name: round(float((src == i).mean()), 3) for i in range(len(ds.sources))},
          'counts', s0.counts.tolist())
    s0.set_epoch(0, skip=3)
    assert list(s0) == a0[3 * 64:], 'resume skip mismatch'
    e1 = s0.epoch_indices(1)
    assert not set(s0.epoch_indices(0).tolist()) & set(e1.tolist()) or min(ds.sizes) < 4096, 'epochs repeat early'
    loader = torch.utils.data.DataLoader(ds, batch_size=32, sampler=s1, num_workers=4, drop_last=True,
                                         multiprocessing_context='spawn', prefetch_factor=2)
    t = time.time()
    for i, x in enumerate(loader):
        print('batch', i, tuple(x.shape), x.dtype, '%.1fs' % (time.time() - t))
        if i + 1 == a.batches:
            break
    print('ok')


if __name__ == '__main__':
    main()
