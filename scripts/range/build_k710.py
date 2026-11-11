"""Assemble Kinetics-710 from the K400/K600/K700-2020 train sets on RANGE.

Rule (VideoMAEv2 / V-JEPA / LeVJEPA): union of the three training sets,
deduplicated by YouTube id, with every id that appears in ANY validation or
test list of ANY version removed, so probes on K400 val never see a
pretraining clip. Labels are mapped to the 710-class space with VideoMAEv2's
label maps; the mapping is checked against their identical-label merge.

Outputs under /scratch/CLUSTER_USER/datasets/k710/:
  train_full.csv            path,youtube_id,k710_label,k710_name,source
  train_{05,10,20,50}pct.csv nested class-balanced subsets, seed 0
  stats.json                counts, exclusions, conflicts, missing files
Run on a login node after the Kinetics extraction sentinel exists.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

D = Path('/scratch/CLUSTER_USER/datasets')
META = D / 'kinetics_meta'
SOURCES = {  # name: (train csv, [val/test csvs], video root(s), label map, 710 map)
    'k400': (D / 'k400/annotations/train.csv',
             [D / 'k400/annotations/val.csv', D / 'k400/annotations/test.csv'],
             [D / 'k400/train'], 'label_map_k400.txt', 'label_710to400.json'),
    'k600': (D / 'kinetics_targz/k600/annotations/train.csv',
             [D / 'kinetics_targz/k600/annotations/val.csv', META / 'k600/test.csv'],
             [D / 'k600/train'], 'label_map_k600.txt', 'label_710to600.json'),
    'k700': (D / 'kinetics_targz/k700_2020/annotations/train.csv',
             [D / 'kinetics_targz/k700_2020/annotations/val.csv', META / 'k700_2020/test.csv'],
             [D / 'k700_2020/train'], 'label_map_k700.txt', 'label_710to700.json'),
}
FRACTIONS = {'05': .05, '10': .10, '20': .20, '50': .50}


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def index_videos(roots):
    """youtube_id -> path. K400/K700 are class-foldered, K600 is flat.

    Listing a million files on VAST takes over an hour, so the listing is
    cached next to each root as <root>.files.txt (one path per line) and
    reused on later runs; delete the cache to force a fresh walk.
    """
    out = {}
    for root in roots:
        cache = root.with_name(root.name + '.files.txt')
        if cache.exists():
            paths = [Path(l) for l in cache.read_text().splitlines() if l]
        else:
            paths = list(root.rglob('*.mp4'))
            cache.write_text('\n'.join(map(str, paths)) + '\n')
        for p in paths:
            out.setdefault(p.name[:11], p)
    return out


def normalize(s):
    return ''.join(ch for ch in s.lower() if ch.isalnum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=D / 'k710')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    vm = META / 'videomaev2'
    k710_names = vm.joinpath('label_map_k710.txt').read_text().splitlines()
    merge = json.loads(vm.joinpath('k710_identical_label_merge.json').read_text())
    assert len(k710_names) == 710, len(k710_names)

    # Exclusion set: every id in any val/test list of any version.
    excluded = set()
    for name, (_, held, *_ ) in SOURCES.items():
        for path in held:
            excluded |= {r['youtube_id'] for r in read_csv(path)}
    stats = {'excluded_ids': len(excluded)}

    chosen = {}          # ytid -> dict(path, label, source)
    conflicts = 0
    for name, (train_csv, _, roots, label_map, map_json) in SOURCES.items():
        names = vm.joinpath(label_map).read_text().splitlines()
        to710 = json.loads(vm.joinpath(map_json).read_text())
        assert len(names) == len(to710), (name, len(names), len(to710))
        # Check the interpretation: version class j maps to k710 index to710[j],
        # whose name must equal the version name up to the identical-label merge.
        # The merge file renames some labels (e.g. "throwing ball" ->
        # "throwing ball (not baseball or softball)"); accept an exact match,
        # a match after renaming either side, and nothing else.
        bad = [(n, k710_names[i]) for n, i in zip(names, to710)
               if n != k710_names[i] and merge.get(n) != k710_names[i]
               and merge.get(k710_names[i]) != n]
        assert not bad, (name, bad[:5])
        label_of = {n: i for n, i in zip(names, to710)}
        # The version maps use the merged names ("balloon blowing" is listed
        # as "inflating balloons"), and one K600 label differs in case, so
        # resolve exact -> merged -> case/punctuation-insensitive.
        by_norm = {normalize(n): i for n, i in label_of.items()}
        # Final fallback: the 710-space names themselves. "hugging" merges to
        # "hugging (not baby)", which exists only in the 710 map, not K400's;
        # "archaeological excavation" is a 710 class missing from the K700 map.
        # "chopping vegetables" (K600) has no 710 class at all and stays unknown.
        k710_index = {n: i for i, n in enumerate(k710_names)}
        k710_by_norm = {normalize(n): i for n, i in k710_index.items()}

        def resolve(lab):
            for cand in (lab, merge.get(lab)):
                if cand is None: continue
                if cand in label_of: return label_of[cand]
                if cand in k710_index: return k710_index[cand]
            for cand in (lab, merge.get(lab)):
                if cand is None: continue
                if normalize(cand) in by_norm: return by_norm[normalize(cand)]
                if normalize(cand) in k710_by_norm: return k710_by_norm[normalize(cand)]
            return None
        files = index_videos(roots)
        rows = read_csv(train_csv)
        s = dict(csv_rows=len(rows), files_on_disk=len(files), missing_file=0,
                 excluded=0, duplicate=0, added=0, unknown_label=0, unknown_names=[])
        for r in rows:
            yt, lab = r['youtube_id'], r['label'].strip().strip('"')
            idx = resolve(lab)
            if idx is None:
                s['unknown_label'] += 1
                if lab not in s['unknown_names']: s['unknown_names'].append(lab)
                continue
            label_of[lab] = idx
            if yt in excluded:
                s['excluded'] += 1; continue
            if yt not in files:
                s['missing_file'] += 1; continue
            if yt in chosen:
                s['duplicate'] += 1
                if chosen[yt]['label'] != label_of[lab]:
                    conflicts += 1
                continue
            chosen[yt] = dict(path=str(files[yt]), label=label_of[lab], source=name)
            s['added'] += 1
        stats[name] = s
        print(name, s, flush=True)
    stats['label_conflicts_between_versions'] = conflicts
    stats['train_full'] = len(chosen)

    by_class = defaultdict(list)
    for yt, v in chosen.items():
        by_class[v['label']].append(yt)
    stats['classes_present'] = len(by_class)
    stats['class_size_min_max'] = [min(map(len, by_class.values())), max(map(len, by_class.values()))]

    def write(path, ids):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['path', 'youtube_id', 'k710_label', 'k710_name', 'source'])
            for yt in ids:
                v = chosen[yt]
                w.writerow([v['path'], yt, v['label'], k710_names[v['label']], v['source']])

    order_all = sorted(chosen)
    write(a.out / 'train_full.csv', order_all)
    # Nested class-balanced subsets: one seeded shuffle per class, prefixes.
    rng = random.Random(a.seed)
    shuffled = {c: rng.sample(ids, len(ids)) for c, ids in sorted(by_class.items())}
    for tag, frac in FRACTIONS.items():
        ids = []
        for c, lst in shuffled.items():
            ids.extend(lst[:max(1, round(frac * len(lst)))])
        write(a.out / f'train_{tag}pct.csv', ids)
        stats[f'train_{tag}pct'] = len(ids)
    stats['source_counts'] = dict(Counter(v['source'] for v in chosen.values()))
    stats['seed'] = a.seed
    stats['rule'] = ('union of K400/K600/K700-2020 train, dedup by youtube id, '
                     'minus any id in any val or test list of any version; '
                     'labels via VideoMAEv2 label maps; subsets are per-class seeded prefixes')
    (a.out / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
