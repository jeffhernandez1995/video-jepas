"""Scaling curves of the LeVJEPA K710 trunks (cached probe protocol),
raw training weights versus the no-warm-up EMA that was evaluated first.
Numbers copied from <run>/evals/eNNN.json and <run>_raw/evals/eNNN.json.
  python scripts/natural/plot_levjepa_curves.py   -> docs/figures/levjepa_k710_curves.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYS = ['k400/linear_mean', 'k400/attentive', 'k400/conv_attentive',
        'in1k/linear_mean', 'in1k/attentive', 'in1k/conv_attentive']
RAW = {
    'ViT-T': {60: (13.76, 15.63, 16.17, 12.42, 17.35, 21.68), 120: (17.29, 19.44, 20.07, 16.38, 21.19, 25.37),
              180: (18.68, 20.33, 21.70, 18.11, 23.15, 26.57), 240: (19.16, 20.18, 22.12, 19.21, 24.27, 27.53)},
    'ViT-S': {60: (18.30, 20.10, 21.94, 18.28, 23.68, 26.88), 120: (22.05, 24.31, 25.46, 22.87, 28.74, 31.54),
              180: (24.07, 26.40, 27.90, 25.66, 31.53, 34.14), 240: (25.65, 28.07, 29.47, 27.58, 33.80, 35.87)},
    'ViT-B': {60: (22.05, 21.94, 23.29, 23.03, 25.30, 28.06), 120: (25.73, 27.42, 27.76, 28.23, 31.24, 34.56),
              180: (28.05, 30.32, 31.11, 31.11, 35.21, 38.05), 240: (28.89, 32.04, 32.30, 32.53, 37.17, 39.74)},
}
EMA = {
    'ViT-T': {60: (4.75, 7.00, 8.57, 2.88, 6.97, 13.60), 120: (6.98, 9.15, 10.15, 5.10, 9.96, 15.14),
              180: (9.27, 11.05, 11.79, 7.96, 12.77, 17.84), 240: (11.44, 13.25, 14.36, 10.30, 15.72, 20.25)},
    'ViT-S': {60: (6.24, 8.82, 9.99, 4.81, 9.22, 14.09), 120: (9.42, 11.60, 12.84, 8.38, 13.67, 18.00),
              180: (12.60, 14.78, 16.20, 12.29, 17.84, 22.27), 240: (16.10, 18.00, 19.77, 16.73, 22.55, 26.48)},
    'ViT-B': {60: (7.96, 8.84, 8.80, 7.05, 9.08, 13.31), 120: (11.80, 12.13, 12.48, 11.67, 14.09, 18.04),
              180: (16.50, 16.29, 17.18, 17.49, 19.68, 23.89), 240: (20.41, 20.13, 21.26, 22.79, 25.85, 29.21)},
}
COL = {'ViT-T': 'tab:green', 'ViT-S': 'tab:orange', 'ViT-B': 'tab:blue'}
# Exact V-JEPA probe protocol (scripts/natural/probe_vjepa.py) on ViT-B e240,
# and the only like-for-like published number: LeVJEPA ViT-B, 20% K710,
# 240 epochs, IN1K attentive probe = 51.2 (paper Table "attention", V=4;
# 50.7 bidirectional). Their K400 linear 44.6 is FLOP-matched (1,085 epochs)
# and is not drawn. ViT-S/L epoch-matched values exist only inside a PDF figure.
EXACT = {'in1k/attentive': {'EMA': 36.35, 'raw': 48.81}, 'k400/linear_mean': {'EMA': 11.24, 'raw': 20.69}}
PAPER = {'in1k/attentive': 51.2}

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for ax, key, i in zip(axes.flat, KEYS, range(6)):
    for m in RAW:
        e = sorted(RAW[m]); ax.plot(e, [RAW[m][x][i] for x in e], '-o', color=COL[m], label=f'{m} raw')
        e = sorted(EMA[m]); ax.plot(e, [EMA[m][x][i] for x in e], '--x', color=COL[m], alpha=.5, label=f'{m} EMA')
    if key in EXACT:
        for w, mk in (('EMA', 'v'), ('raw', '^')):
            v = EXACT[key][w]
            if v is not None:
                ax.plot([240], [v], mk, color='k', ms=9, label=f'ViT-B exact V-JEPA probe ({w})')
    if key in PAPER:
        ax.plot([240], [PAPER[key]], '*', color='crimson', ms=14, label='LeVJEPA paper ViT-B (exact probe)')
        ax.legend(fontsize=7, loc='lower right')
    ax.set_title(key); ax.set_xticks([60, 120, 180, 240]); ax.grid(alpha=.3)
    ax.set_ylabel('top-1 (%)') if i % 3 == 0 else None
    ax.set_xlabel('pretraining epoch') if i >= 3 else None
axes[0, 0].legend(fontsize=8, ncol=2)
fig.suptitle('LeVJEPA on 20% K710 (our rebuild): cached frozen-probe curves, raw weights vs no-warm-up EMA; black/red markers use the exact V-JEPA probe')
fig.tight_layout()
fig.savefig('docs/figures/levjepa_k710_curves.png', dpi=130)
print('wrote docs/figures/levjepa_k710_curves.png')
