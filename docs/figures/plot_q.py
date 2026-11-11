"""Progress charts for direction Q (ViT-T): cached probes vs epoch, and training dynamics."""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

G = Path(__file__).parent
OUT = Path('/Users/CLUSTER_USER/Desktop/repos/video_crw/docs/figures')
OUT.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C = {'Q': '#2a78d6', 'Q-frame': '#eb6834', 'Regression (B)': '#1baf7a'}
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
    'axes.titlesize': 11, 'axes.titleweight': 'semibold', 'axes.titlecolor': INK,
    'axes.titlelocation': 'left', 'legend.frameon': False,
})


def parse(path):
    """Metric records with an epoch (calibration steps have none); keep the last monotone run."""
    rows = []
    for line in open(path, errors='ignore'):
        line = line.strip()
        if not line.startswith('{') or 'step' not in line[:10]:
            continue
        try:
            d = json.loads(line)
        except Exception:
            try:
                d = ast.literal_eval(line)
            except Exception:
                continue
        if 'epoch' in d:
            if rows and d['epoch'] < rows[-1]['epoch'] - 1:
                rows = []                       # a restart from scratch: keep the later run
            rows.append(d)
    return rows


def series(paths, key, win=25):
    rows = [r for p in paths for r in parse(G / 'logs' / p)]
    rows = sorted({round(r['epoch'], 4): r for r in rows if key in r}.values(), key=lambda r: r['epoch'])
    x = np.array([r['epoch'] for r in rows]); y = np.array([r[key] for r in rows], float)
    if win > 1 and len(y) > win:
        k = np.ones(win) / win
        ys = np.convolve(np.pad(y, (win // 2, win - 1 - win // 2), mode='edge'), k, mode='valid')
        return x, y, ys
    return x, y, y


def evals(run):
    out = {}
    for f in sorted((G / 'runs' / run / 'evals').glob('e*_full_student.json')):
        d = json.load(open(f))
        out[d['pretrain_epoch'] if 'pretrain_epoch' in d else d['epoch']] = d['scores']
    return out


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords='offset points', va='center',
                fontsize=9, color=INK, clip_on=False)
    ax.plot([x], [y], 'o', ms=6, color=color, mec=SURF, mew=2, zorder=5)


# ------------------------------------------------------------------ figure 1: cached probes
runs = {'Q': 'nlpQm_k710_vitt', 'Q-frame': 'nlpQfm_k710_vitt', 'Regression (B)': 'nlpBm_k710_vitt'}
E = {k: evals(v) for k, v in runs.items()}
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
for ax, (ds, title) in zip(axes, [('k400', 'K400'), ('in1k', 'ImageNet-1k'), ('ssv2', 'SSv2')]):
    offs = {}
    for name, ev in E.items():
        ep = [e for e in sorted(ev) if f'{ds}/attentive' in ev[e]]
        if not ep:
            continue
        y = [ev[e][f'{ds}/attentive'] for e in ep]
        ax.plot(ep, y, '-', lw=2, color=C[name], label=name, zorder=3)
        ax.plot(ep, y, 'o', ms=6, color=C[name], mec=SURF, mew=2, zorder=4)
        offs[name] = y[-1]
    # direct labels at the right end, nudged apart
    order = sorted(offs, key=offs.get)
    last = None
    for name in order:
        dy = 0
        if last is not None and abs(offs[name] - offs[last]) < 0.35:
            dy = 9
        ax.annotate(f'{name} {offs[name]:.1f}', (240, offs[name]), xytext=(8, dy), textcoords='offset points',
                    va='center', fontsize=9, color=INK)
        last = name
    ax.set_title(f'{title}, attentive probe')
    ax.set_xticks([60, 120, 180, 240]); ax.set_xlim(50, 300)
    ax.set_xlabel('pretraining epoch'); ax.set_ylabel('top-1 %')
    lo = min(offs.values()) if False else None
    ys_all = [v for ev in E.values() for e, v in ev.items() if f'{ds}/attentive' in v]
    vals = [v[f'{ds}/attentive'] for v in ys_all]
    mid = (max(vals) + min(vals)) / 2
    ax.set_ylim(mid - 1.6, mid + 1.6)                    # same 3.2-point span on every panel
    if ds == 'ssv2':
        ax.set_title('SSv2, attentive (no regression run yet)')
axes[0].legend(loc='upper left', fontsize=9)
fig.suptitle('ViT-T on 20% K710: cached frozen probes, student encoder (full train/val sets)',
             x=0.01, ha='left', fontsize=12, fontweight='semibold', color=INK)
fig.savefig(OUT / 'q_probes_vs_epoch.png', dpi=160)

# ------------------------------------------------------------------ figure 2: training dynamics
logs = {
    ('K710', 'Q'): ['nlpQm_k710_vitt.log'], ('K710', 'Q-frame'): ['nlpQfm_k710_vitt.log'],
    ('K710', 'Regression (B)'): ['nlpBm_k710_vitt_314483.out', 'nlpBm_k710_vitt_314484.out'],
    ('Mix 3', 'Q'): ['nlpQm_mix3_vitt_2532386.out'], ('Mix 3', 'Q-frame'): ['nlpQfm_mix3_vitt_2532389.out'],
}
metrics = [('acc1', 'next-frame bin accuracy (h=1)', '%'), ('feature_erank', 'teacher feature rank (of 192)', '')]
fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), constrained_layout=True, sharex=True)
for i, data in enumerate(['K710', 'Mix 3']):
    for j, (key, title, unit) in enumerate(metrics):
        ax = axes[i, j]
        ends = []
        for name in C:
            if (data, name) not in logs:
                continue
            x, y, ys = series(logs[(data, name)], key)
            if not len(x):
                continue
            scale = 100 if key == 'acc1' else 1
            ax.plot(x, y * scale, '-', lw=0.6, color=C[name], alpha=0.25)
            ax.plot(x, ys * scale, '-', lw=2, color=C[name], label=name)
            ends.append((name, x[-1], ys[-1] * scale))
        if key == 'acc1':
            ax.axhline(12.5, color=INK2, lw=0.8)
            ax.text(4, 13.3, 'chance 12.5%', fontsize=8.5, color=INK2)
            ax.set_ylim(10, 32)
        if key == 'bin_entropy':
            ax.set_ylim(0.94, 1.001)
        ends.sort(key=lambda t: t[2])
        prev = None
        for name, xe, ye in ends:
            dy = 0
            if prev is not None:
                span = ax.get_ylim()[1] - ax.get_ylim()[0]
                if abs(ye - prev) < 0.05 * span:
                    dy = 9
            fmt = f'{ye:.1f}' if key != 'bin_entropy' else f'{ye:.3f}'
            ax.annotate(f'{name} {fmt}', (xe, ye), xytext=(6, dy), textcoords='offset points',
                        va='center', fontsize=8.5, color=INK, clip_on=False)
            prev = ye
        ax.set_xlim(0, 285); ax.set_xticks([0, 60, 120, 180, 240])
        ax.set_title(f'{data}: {title}')
        if i == 1:
            ax.set_xlabel('pretraining epoch')
axes[0, 0].legend(loc='lower right', fontsize=9)
fig.suptitle('ViT-T training dynamics (thin: raw log every 20 steps; thick: 25-point mean)\n'
             'Bins stayed balanced throughout (normalized entropy 0.998-0.999, all four runs). '
             'No Mix 3 regression log on hand',
             x=0.01, ha='left', fontsize=11.5, fontweight='semibold', color=INK)
fig.savefig(OUT / 'q_training_dynamics.png', dpi=160)
print('wrote', OUT / 'q_probes_vs_epoch.png', OUT / 'q_training_dynamics.png')
for k, ev in E.items():
    print(k, {e: (ev[e].get('k400/attentive'), ev[e].get('in1k/attentive'), ev[e].get('ssv2/attentive')) for e in sorted(ev)})
