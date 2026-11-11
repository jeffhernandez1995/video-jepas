"""Intuitive-physics linear probes on frozen features: plausible vs implausible video classification
on IntPhys 2 (Main split) and InfLevel-Lab, following Esmati et al. 2026 (arXiv 2606.05328, Fig. 2):
a linear probe per category on mean-pooled features, 60/40 train/test split, 5 seeds, held-out
accuracy averaged over categories (chance 50%).

The split is by matched set (the paper splits "scene-level features"): both benchmarks come in
sets of near-identical videos that differ only in plausibility (IntPhys 2: 4 per SceneIndex, 2
possible / 2 impossible; InfLevel: the 4 event codes of one camera / cover / object / direction).
A per-video random split puts siblings with opposite labels in train and test, and the probe then
scores below chance (IntPhys 2 23-30% for every reference); --split random keeps that variant.

Videos are sampled at ~7.5 fps (every 4th frame of InfLevel's 30 fps, every 8th of IntPhys 2's
60 fps), resized to 224 px high (width kept to the aspect ratio, a multiple of the patch size),
and encoded like the paper-probe tier (token-matched windows of 8 temporal slots; see
eval_davis.Encoder). A video's feature is the mean over all its output tokens. Features are cached
in <out>/physics_<dataset>_features.npz, so probes can be refit without the encoder.

Labels: IntPhys 2 metadata `type` (Possible / Impossible), category = `condition`. InfLevel-Lab
filename event type, from allenai/inflevel evaluator.py: continuity plausible vv, ii vs implausible
iv, vi; gravity and solidity plausible ui, cv vs implausible uv, ci.

  python scripts/natural/eval_physics.py --family vjepa --checkpoint <ckpt> --out <dir>
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_checkpoint as ec  # noqa: E402
from eval_davis import Encoder, MEAN, STD  # noqa: E402

D = Path('/scratch/CLUSTER_USER/datasets')
PLAUSIBLE = dict(continuity={'vv': 1, 'ii': 1, 'iv': 0, 'vi': 0},
                 gravity={'ui': 1, 'cv': 1, 'uv': 0, 'ci': 0},
                 solidity={'ui': 1, 'cv': 1, 'uv': 0, 'ci': 0})


def items(dataset):
    """[(path, label 1=plausible, category, matched set)]"""
    if dataset == 'intphys2':
        root = D / 'intphys2/Main'
        with open(root / 'metadata.csv') as f:
            return [(root / r['file_name'], int(r['type'].endswith('_Possible')), r['condition'],   # 1_/2_Possible
                     f"intphys2_{r['SceneIndex']}") for r in csv.DictReader(f)]
    out = []
    for cat, codes in PLAUSIBLE.items():
        for p in sorted((D / 'inflevel/inflevel_lab' / cat).glob('[!.]*.mp4')):   # skip macOS ._ files
            parts = p.stem.split('__')        # camera, category, cover, object, code[, direction]
            out.append((p, codes[parts[4]], cat, '__'.join(parts[:4] + parts[5:])))
    return out


def load_frames(path, step, height, patch):
    import av
    from PIL import Image
    with av.open(str(path)) as c:
        st = c.streams.video[0]; st.thread_type = 'AUTO'
        frames = [f.to_image() for i, f in enumerate(c.decode(st)) if i % step == 0]
    w0, h0 = frames[0].size
    width = max(patch, round(w0 * height / h0 / patch) * patch)
    x = torch.stack([torch.from_numpy(np.asarray(f.convert('RGB').resize((width, height), Image.BICUBIC)).copy())
                     for f in frames]).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN) / STD


def probe(feats, labels, cats, groups, split='grouped', seeds=5, train_frac=0.6):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.preprocessing import StandardScaler
    res = {}
    for c in sorted(set(cats)):
        idx = np.flatnonzero(np.array(cats) == c)
        x, y, g = feats[idx], np.array(labels)[idx], np.array(groups)[idx]
        accs = []
        for s in range(seeds):
            if split == 'grouped':
                tr, te = next(GroupShuffleSplit(1, train_size=train_frac, random_state=s).split(x, y, g))
                xtr, xte, ytr, yte = x[tr], x[te], y[tr], y[te]
            else:
                xtr, xte, ytr, yte = train_test_split(x, y, train_size=train_frac, stratify=y, random_state=s)
            sc = StandardScaler().fit(xtr)
            clf = LogisticRegression(max_iter=5000).fit(sc.transform(xtr), ytr)
            accs.append(100 * clf.score(sc.transform(xte), yte))
        res[c] = dict(mean=round(float(np.mean(accs)), 2), sem=round(float(np.std(accs, ddof=1) / np.sqrt(seeds)), 2),
                      n=len(idx), seeds=[round(a, 2) for a in accs])
    res['average'] = round(float(np.mean([v['mean'] for v in res.values()])), 2)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True, choices=list(ec.LOADERS))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=['intphys2', 'inflevel_lab'])
    p.add_argument('--split', choices=['grouped', 'random'], default='grouped',
                   help='60/40 split by matched set (default) or per video')
    p.add_argument('--limit', type=int, default=None, help='smoke test: videos per dataset')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    enc = None
    results = dict(family=a.family, checkpoint=a.checkpoint, split=a.split,
                   protocol=f'linear probe per category on mean-pooled frozen features, 60/40 split ({a.split}), '
                            '5 seeds (Esmati et al. 2026, Fig. 2); ~7.5 fps, 224 px high, token-matched 8-slot windows')
    for ds in a.datasets:
        cache = a.out / f'physics_{ds}_features{"_smoke" if a.limit else ""}.npz'
        its = items(ds)
        if a.limit:
            rng = np.random.default_rng(0)
            its = [its[i] for i in sorted(rng.choice(len(its), a.limit, replace=False))]
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            feats, labels, cats = z['feats'], list(z['labels']), list(z['cats'])
            group = {str(p): g for p, _, _, g in its}
            groups = [group[p] for p in z['paths']]
        else:
            enc = enc or Encoder(a.family, a.checkpoint)
            step = 8 if ds == 'intphys2' else 4
            feats, t0 = [], time.time()
            for i, (path, *_) in enumerate(its):
                f = enc.features(load_frames(path, step, 224, enc.patch))       # N,h,w,D
                feats.append(f.mean((0, 1, 2)).cpu().numpy())
                if i % 200 == 0:
                    print(f'{ds}: {i}/{len(its)} videos, {time.time() - t0:.0f}s', flush=True)
            feats = np.stack(feats); labels = [it[1] for it in its]; cats = [it[2] for it in its]
            groups = [it[3] for it in its]
            np.savez(cache, feats=feats, labels=np.array(labels), cats=np.array(cats),
                     paths=np.array([str(it[0]) for it in its]))
        results[ds] = probe(feats, labels, cats, groups, a.split)
        print(ds, json.dumps({k: (v['mean'] if isinstance(v, dict) else v) for k, v in results[ds].items()}), flush=True)
    if enc is not None:
        results.update({k: v for k, v in enc.info.items()})
    (a.out / f'physics_probe{"_random" if a.split == "random" else ""}{"_smoke" if a.limit else ""}.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
