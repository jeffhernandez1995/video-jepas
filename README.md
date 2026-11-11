# Latent video prediction on natural video

Self-supervised video pretraining by predicting the future in a latent or token
space, compared against V-JEPA, V-JEPA 2, LeVJEPA, VideoMAE and DINOv2 under one
frozen-probe protocol.

## Layout

| Path | What |
|---|---|
| `scripts/natural/` | Model, trainer, diagnostics and evaluators (`nlp.py`, `train_nlp.py`, `eval_checkpoint.py`, `probe_vjepa.py`, `eval_davis.py`, `eval_physics.py`, `eval_surprise.py`) |
| `scripts/range/` | Dataset builds, watchers and status scripts for the RANGE cluster |
| `scripts/tamara/` | Probe queues and screens for Tamara |
| `slurm/` | Chained and multi-node job templates |
| `docs/` | Plans and results for the natural-video stage |
| `third_party/` | Pinned upstream code; each `<name>.commit` holds the revision to check out |

## Docs, in reading order

1. `docs/natural_video_plan.md`: what carried over from the synthetic stage and the two original recipes.
2. `docs/levjepa_matched_protocol.md`: the matched LeVJEPA baseline and the evaluation protocol.
3. `docs/nlp_k710_runs.md`: next-latent prediction on 20% K710 (arms A, B, masked variants).
4. `docs/data_mixes.md`: data mixes, scaling runs, reference models and the final-table probes.
5. `docs/nlp_lag_curve.md`: how long a time gap must be before copying stops working.
6. `docs/range_cluster.md`, `docs/levjepa_loader_bottleneck.md`: cluster and data-loading notes.

## Clusters

```bash
bash scripts/sync_tamara.sh   # then scripts/setup_tamara.sh on Tamara
bash scripts/sync_range.sh    # then scripts/setup_range.sh on a RANGE login node
```

Both syncs skip `runs/` and `old/`. Do not sync edited scripts while jobs that
use them are running.

## Site-specific values

The user name, cluster host names and Slurm account are replaced in commits by
`CLUSTER_USER`, `TAMARA_HOST`, `RANGE_HOST`, `NOTS_HOST` and `SLURM_ACCT`
(`scripts/private_filter.sh`, a git clean/smudge filter). Substitute your own
values to run the cluster scripts. The earlier synthetic-benchmark stage is
archived outside the repository.
