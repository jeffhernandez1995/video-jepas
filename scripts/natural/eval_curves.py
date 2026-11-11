"""One W&B run per (pretraining run, weights, protocol) holding every milestone's
frozen-probe scores against `pretrain_epoch`, so the workspace draws lines over
pretraining epochs instead of one bar per evaluation.

eval_checkpoint.py calls sync() after writing its JSON; sync() reads every
eNNN[_full][_weights].json in the evals directory and logs the epochs the curve
run does not have yet, so it is idempotent and repairs itself after a missed upload.

Backfill / repair (login node, any python with wandb):
  python scripts/natural/eval_curves.py /scratch/CLUSTER_USER/runs/nlp*_k710_* /scratch/CLUSTER_USER/runs/levjepa_k710_*_raw
Report with the six line panels (needs wandb-workspaces on PYTHONPATH):
  python scripts/natural/eval_curves.py --report
"""
import argparse
import json
import re
from pathlib import Path

ENTITY, PROJECT = 'jeffhernandez1995', 'video_next_prediction'
KEYS = ['k400/linear_mean', 'k400/attentive', 'k400/conv_attentive',
        'in1k/linear_mean', 'in1k/attentive', 'in1k/conv_attentive',
        'ssv2/linear_mean', 'ssv2/attentive', 'ssv2/conv_attentive',
        'in1k_v2/attentive', 'in1k_r/attentive', 'in1k_sketch/attentive']


def sync(evals, full, weights=None, wandb_dir='/scratch/CLUSTER_USER/wandb'):
    """evals: <run dir>/evals. weights: 'student' / 'teacher' for nlp, None for LeVJEPA (raw)."""
    import wandb
    evals = Path(evals)
    pattern = re.compile(r'e(\d{3})' + ('_full' if full else '') + (f'_{weights}' if weights else '') + r'\.json')
    points = {int(m.group(1)): json.loads(f.read_text()) for f in sorted(evals.glob('e*.json'))
              if (m := pattern.fullmatch(f.name))}
    if not points:
        return None
    first = next(iter(points.values()))
    run, w, protocol = evals.parent.name, weights or 'raw', 'full' if full else 'subset'
    rid = f'{run}_curve_{protocol}_{w}'
    try:
        done = {int(r['pretrain_epoch']) for r in wandb.Api().run(f'{ENTITY}/{PROJECT}/{rid}').scan_history(keys=['pretrain_epoch'])}
    except Exception:
        done = set()
    todo = sorted(set(points) - done)
    if not todo:
        return rid
    masked = bool(re.match(r'nlp[AB]\d?m', run))
    r = wandb.init(project=PROJECT, entity=ENTITY, id=rid, name=f'{run} ({w})', group='eval_curves', job_type='eval_curve',
                   resume='allow', dir=wandb_dir,
                   tags=['natural-video', 'eval-curve', f'eval-{protocol}', first.get('family', 'levjepa'), first.get('model', 'unknown'), f'weights-{w}']
                        + (['masked'] if masked else []),
                   config=dict(run=run, family=first.get('family', 'levjepa'), model=first.get('model', 'unknown'), weights=w, protocol=first['protocol'],
                               arm=first.get('arm'), masked=masked),
                   settings=wandb.Settings(disable_git=True, save_code=False, console='off'))
    r.define_metric('pretrain_epoch')
    r.define_metric('probe/*', step_metric='pretrain_epoch')
    for ep in todo:
        r.log({'pretrain_epoch': ep, **{f'probe/{k}': v for k, v in points[ep]['scores'].items()}})
    r.summary.update({f'final/{k}': v for k, v in points[max(points)]['scores'].items()} | {'final/pretrain_epoch': max(points)})
    r.finish()
    return rid


def report():
    import wandb_workspaces.reports.v2 as wr

    def grid(title, filters):
        panels = [wr.LinePlot(title=k, x='pretrain_epoch', y=[f'probe/{k}'], title_x='pretraining epoch', title_y='top-1 (%)',
                              smoothing_type='none', legend_position='south',
                              layout=wr.Layout(x=(i % 3) * 8, y=(i // 3) * 9, w=8, h=9)) for i, k in enumerate(KEYS)]
        return [wr.H2(text=title), wr.PanelGrid(runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name=title, filters=filters)],
                                                 panels=panels)]

    curve = "JobType == 'eval_curve' and Tags('tags') in ['eval-full']"
    blocks = [wr.P(text='Frozen-probe top-1 against pretraining epoch on 20% K710. One line per pretraining run; '
                        'points are appended by every finished evaluation job (scripts/natural/eval_curves.py).')]
    blocks += grid('ViT-T: masked next-latent prediction (student) vs LeVJEPA (raw)',
                   curve + " and Config('model') == 'vit_tiny' and Config('weights') in ['student', 'raw']"
                           " and (Config('masked') == True or Config('family') == 'levjepa')")
    blocks += grid('All runs, student / raw weights', curve + " and Config('weights') in ['student', 'raw']")
    blocks += grid('All runs, EMA teacher weights', curve + " and Config('weights') == 'teacher'")
    rep = wr.Report(entity=ENTITY, project=PROJECT, title='Next-latent prediction vs LeVJEPA: probe curves',
                    description='Live line plots of the full frozen-probe protocol.', blocks=blocks, width='fluid')
    rep.save()
    print(rep.url)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='*', type=Path)
    p.add_argument('--report', action='store_true')
    p.add_argument('--wandb-dir', default='/scratch/CLUSTER_USER/wandb')
    a = p.parse_args()
    for run in a.runs:
        for full in (True, False):
            for w in (None, 'student', 'teacher'):
                rid = sync(run / 'evals', full, w, a.wandb_dir)
                if rid:
                    print(rid, flush=True)
    if a.report:
        report()
