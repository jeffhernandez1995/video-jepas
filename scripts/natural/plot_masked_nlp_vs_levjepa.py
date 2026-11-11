"""Masked next-latent prediction (A lean and B, each with horizons h1/2/4 and h=1 only; 90% tube-block masking) against
LeVJEPA on 20% K710, ViT-T, full cached probe protocol. Ours: student weights (the
EMA teacher is within ~1 everywhere). LeVJEPA: raw training weights. Numbers copied
from <run>/evals/eNNN_full*.json.
  python scripts/natural/plot_masked_nlp_vs_levjepa.py   -> docs/figures/masked_nlp_vs_levjepa_k710.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYS = ['k400/linear_mean', 'k400/attentive', 'k400/conv_attentive',
        'in1k/linear_mean', 'in1k/attentive', 'in1k/conv_attentive']
RUNS = {
    'LeVJEPA (raw)': {60: (15.63, 19.96, 22.20, 14.80, 23.57, 30.49), 120: (19.69, 24.20, 26.12, 19.27, 28.59, 34.56),
                      180: (21.44, 25.83, 27.94, 21.22, 31.01, 36.48), 240: (22.38, 26.72, 28.94, 22.38, 32.30, 37.11)},
    'masked B: regression + guard, h1/2/4': {60: (22.66, 27.10, 27.49, 19.83, 31.56, 37.87), 120: (23.07, 27.50, 28.44, 20.00, 31.82, 38.27),
                                     180: (22.77, 27.37, 27.79, 19.64, 32.09, 38.25), 240: (22.81, 27.49, 28.42, 19.72, 32.00, 38.63)},
    'masked A: categorical (lean), h1/2/4': {60: (21.04, 25.42, 26.86, 18.59, 29.99, 36.70), 120: (21.59, 25.42, 27.78, 18.96, 29.98, 36.51),
                                     180: (22.06, 27.11, 28.13, 18.81, 30.17, 36.54), 240: (21.90, 26.44, 28.26, 18.78, 30.28, 36.76)},
    'masked B, h=1 only': {60: (19.85, 24.41, 24.72, 17.16, 27.56, 34.68), 120: (19.44, 24.13, 24.58, 16.71, 27.50, 34.46),
                           180: (20.62, 25.09, 25.87, 17.11, 27.59, 34.87), 240: (20.74, 25.49, 26.32, 17.45, 28.46, 35.23)},
    'masked A, h=1 only': {60: (17.17, 22.55, 23.94, 15.15, 24.79, 30.09), 120: (15.57, 20.99, 22.31, 13.69, 22.12, 26.60),
                           180: (16.03, 20.71, 21.42, 13.73, 21.68, 24.44), 240: (16.40, 20.17, 21.57, 13.63, 20.93, 23.10)},
}
STYLE = {'LeVJEPA (raw)': ('-', 'o', 'tab:gray'),
         'masked B: regression + guard, h1/2/4': ('--', 's', 'tab:blue'),
         'masked A: categorical (lean), h1/2/4': ('-.', '^', 'tab:red'),
         'masked B, h=1 only': (':', 's', 'tab:cyan'),
         'masked A, h=1 only': (':', '^', 'tab:orange')}

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True)
for ax, key, i in zip(axes.flat, KEYS, range(6)):
    for name, pts in RUNS.items():
        ls, mk, c = STYLE[name]
        e = sorted(pts)
        ax.plot(e, [pts[x][i] for x in e], ls=ls, marker=mk, color=c, label=name)
    ax.set_title(key)
    ax.set_xticks([60, 120, 180, 240])
    ax.set_xlim(50, 250)
    ax.grid(alpha=.3)
    if i % 3 == 0:
        ax.set_ylabel('top-1 (%)')
    if i >= 3:
        ax.set_xlabel('pretraining epoch')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, frameon=False)
fig.suptitle('Masked next-latent prediction vs LeVJEPA, ViT-T on 20% K710: full frozen-probe protocol '
             '(ours = student weights)')
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig('docs/figures/masked_nlp_vs_levjepa_k710.png', dpi=130)
print('wrote docs/figures/masked_nlp_vs_levjepa_k710.png')
