"""Next-latent prediction (Directions A, A', B) against LeVJEPA on 20% K710, full
cached probe protocol (complete IN1K / K400 train sets, LR chosen on held-out train
items). Ours: student weights (the EMA teacher is within 0.6 everywhere).
LeVJEPA: raw training weights. Numbers copied from <run>/evals/eNNN_full*.json.
  python scripts/natural/plot_nlp_vs_levjepa.py   -> docs/figures/nlp_vs_levjepa_k710.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYS = ['k400/linear_mean', 'k400/attentive', 'k400/conv_attentive',
        'in1k/linear_mean', 'in1k/attentive', 'in1k/conv_attentive']
RUNS = {
    ('LeVJEPA', 'ViT-T'): {60: (15.63, 19.96, 22.20, 14.80, 23.57, 30.49), 120: (19.69, 24.20, 26.12, 19.27, 28.59, 34.56),
                           180: (21.44, 25.83, 27.94, 21.22, 31.01, 36.48), 240: (22.38, 26.72, 28.94, 22.38, 32.30, 37.11)},
    ('LeVJEPA', 'ViT-S'): {60: (21.69, 25.43, 27.24, 22.05, 32.25, 37.90), 120: (26.26, 29.53, 30.96, 27.18, 37.97, 42.58),
                           180: (28.65, 31.75, 33.62, 30.38, 40.80, 45.24), 240: (30.42, 33.77, 35.16, 32.07, 43.21, 47.06)},
    ('LeVJEPA', 'ViT-B'): {60: (26.63, 25.59, 27.51, 28.48, 35.63, 38.64), 120: (31.30, 31.80, 32.95, 34.19, 41.37, 43.87),
                           180: (33.83, 34.16, 36.31, 37.51, 44.04, 46.54), 240: (35.21, 36.81, 37.91, 38.94, 45.77, 47.75)},
    ('B', 'ViT-T'): {60: (15.78, 18.44, 18.71, 14.45, 22.20, 27.33), 120: (15.93, 18.67, 18.85, 14.13, 21.68, 24.80),
                     180: (15.66, 17.92, 18.92, 14.26, 21.50, 25.43), 240: (15.56, 18.12, 18.72, 14.13, 21.24, 25.21)},
    ('B', 'ViT-S'): {60: (18.92, 20.64, 20.31, 18.43, 25.67, 29.06), 120: (21.13, 22.00, 22.16, 20.49, 27.97, 31.46),
                     180: (21.66, 22.73, 23.07, 21.32, 28.47, 31.53), 240: (21.49, 22.70, 23.17, 21.23, 27.79, 30.95)},
    ('A', 'ViT-T'): {60: (12.00, 13.98, 15.57, 11.38, 15.31, 21.67)},
    ('A', 'ViT-S'): {60: (14.35, 16.46, 17.72, 16.04, 20.69, 26.39), 120: (12.39, 13.65, 15.00, 13.31, 18.57, 23.58)},
    ("A'", 'ViT-S'): {60: (14.00, 16.11, 16.80, 15.49, 20.87, 26.10), 120: (11.65, 15.02, 15.42, 12.57, 17.79, 22.05),
                      180: (11.48, 14.98, 15.42, 11.91, 16.11, 19.42), 240: (11.39, 14.63, 15.22, 11.84, 15.54, 18.86)},
}
COL = {'ViT-T': 'tab:green', 'ViT-S': 'tab:orange', 'ViT-B': 'tab:blue'}
STYLE = {'LeVJEPA': ('-', 'o', 1.0, 'LeVJEPA (raw)'),
         'B': ('--', 's', 1.0, 'B: regression + guard'),
         'A': (':', 'x', 0.8, 'A: categorical (cancelled)'),
         "A'": ('-.', '^', 1.0, "A': A + guard + 10 Sinkhorn"),}

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True)
for ax, key, i in zip(axes.flat, KEYS, range(6)):
    for (method, size), pts in RUNS.items():
        ls, mk, a, name = STYLE[method]
        e = sorted(pts)
        ax.plot(e, [pts[x][i] for x in e], ls=ls, marker=mk, color=COL[size], alpha=a, label=f'{name}, {size}')
    ax.set_title(key)
    ax.set_xticks([60, 120, 180, 240])
    ax.grid(alpha=.3)
    if i % 3 == 0:
        ax.set_ylabel('top-1 (%)')
    if i >= 3:
        ax.set_xlabel('pretraining epoch')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8, frameon=False)
fig.suptitle('Next-latent prediction vs LeVJEPA on 20% K710: full frozen-probe protocol '
             '(colour = model size, line style = method; ours = student weights)')
fig.tight_layout(rect=(0, 0.08, 1, 0.97))
fig.savefig('docs/figures/nlp_vs_levjepa_k710.png', dpi=130)
print('wrote docs/figures/nlp_vs_levjepa_k710.png')
