"""Fold finished mix shards into one Lance dataset per source: <mix>/<source>.lance.

Opening a Lance dataset costs ~0.1 s per shard on VAST, so a loader sampling
uniformly over ~2,000 PE Video shards would spend its time opening datasets.
Each shard is a single fragment with a single data file, so consolidation is
metadata only: the data file is hard-linked into <source>.lance/data/ and
registered as a new fragment (Overwrite for the first batch, Append after),
leaving the shard itself untouched. Appends never move existing rows, and
mix_data opens a pinned version, so running jobs are unaffected.

Beside each version v it writes _index_v<v>.npz (episode lengths and global row
offsets, assembled from the shards' own episode runs so episode ids that
restart at 0 in every shard cannot merge across a boundary) and _members.json
(shards in fragment order with their row ranges).

  python scripts/range/consolidate_mix.py pe_video vidaforge egodex droid ssv2 k710 in1k cc12m
  MIX_ROOT=/scratch/CLUSTER_USER/mix256 python scripts/range/consolidate_mix.py ...   # another root

Run it whenever more shards have finished; it only adds the new ones.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

MIX = Path(os.environ.get('MIX_ROOT', '/scratch/CLUSTER_USER/datasets/mix'))   # MIX_ROOT: e.g. Tamara's copy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'natural'))


def consolidate(source):
    import lance
    from lance import LanceOperation
    from lance.fragment import DataFile, FragmentMetadata
    from mix_data import episode_index
    root = MIX / source
    combined = MIX / f'{source}.lance'
    members_f = combined / '_members.json'
    members = json.loads(members_f.read_text()) if members_f.exists() else dict(version=0, shards=[])
    have = {m['name'] for m in members['shards']}
    new = sorted(p for p in root.glob('*.lance')
                 if Path(str(p) + '.stats.json').exists() and p.stem not in have)
    if not new:
        print(f'{source}: nothing new ({len(have)} shards, version {members["version"]})')
        return
    (combined / 'data').mkdir(parents=True, exist_ok=True)
    next_id = sum(1 for _ in members['shards'])
    next_row = members['shards'][-1]['row_end'] if members['shards'] else 0
    frags, schema, added = [], None, []
    for p in new:
        ds = lance.dataset(str(p))
        (frag,) = ds.get_fragments()
        md = frag.metadata
        (df,) = md.files
        src = p / 'data' / df.path
        dst = combined / 'data' / df.path
        if not dst.exists():
            os.link(src, dst)
        schema = schema or ds.schema
        assert ds.schema == schema, f'schema mismatch in {p}'
        frags.append(FragmentMetadata(id=next_id, physical_rows=md.physical_rows,
                                      files=[DataFile(df.path, df.fields, df.column_indices, df.file_major_version,
                                                      df.file_minor_version, df.file_size_bytes)]))
        lengths, offsets = episode_index(p)
        assert int(lengths.sum()) == md.physical_rows, p
        added.append(dict(name=p.stem, fragment=next_id, row_start=next_row, row_end=next_row + md.physical_rows,
                          episodes=len(lengths)))
        next_id += 1
        next_row += md.physical_rows
    if members['shards']:
        op = LanceOperation.Append(frags)
        ds = lance.LanceDataset.commit(str(combined), op, read_version=members['version'])
    else:
        op = LanceOperation.Overwrite(schema, frags)
        ds = lance.LanceDataset.commit(str(combined), op)
    version = ds.version
    assert ds.count_rows() == next_row, (ds.count_rows(), next_row)
    # Global episode index for this version: previous version's + the new shards'.
    if members['shards']:
        z = np.load(combined / f"_index_v{members['version']}.npz")
        lengths_all, offsets_all = [z['lengths']], [z['offsets']]
    else:
        lengths_all, offsets_all = [], []
    for m in added:
        lengths, offsets = episode_index(MIX / source / f"{m['name']}.lance")
        lengths_all.append(lengths)
        offsets_all.append(offsets + m['row_start'])
    np.savez(combined / f'_index_v{version}.npz', lengths=np.concatenate(lengths_all),
             offsets=np.concatenate(offsets_all))
    members = dict(version=version, rows=next_row, shards=members['shards'] + added)
    tmp = members_f.with_suffix('.tmp')
    tmp.write_text(json.dumps(members) + '\n')
    os.replace(tmp, members_f)
    print(f'{source}: +{len(added)} shards -> {len(members["shards"])} shards, {next_row} rows, version {version}')


if __name__ == '__main__':
    for s in sys.argv[1:]:
        consolidate(s)
