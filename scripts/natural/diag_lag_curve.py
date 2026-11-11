"""How long a gap must a prediction target be before copying stops solving it? (2026-11-04)

Three steps, run where the data and the GPUs are:
  lengths  (CPU, RANGE login): episode lengths per video source of a mix spec, read from the
           episode indexes (seconds at 7.5 fps), and the share of episodes and of hours long
           enough for each gap.
  sample   (CPU, RANGE login): --episodes random training-eligible episodes per video source,
           up to --max-frames consecutive frames at 7.5 fps from a random start, kept as the
           stored JPEG bytes in <out>/<source>.pt.
  curve    (GPU): teacher features of every sampled frame, then for each lag d (frames at
           7.5 fps) four trivial predictors of y_{t+d,i}, all squared distances between unit
           vectors (2 = orthogonal):
             copy       y_{t,i}, the shortcut
             mean       the window's normalized mean feature, which knows nothing about t+d
             nn_local   the best y_{t,j} with j within 2 sites of i (oracle: copy after a small move)
             nn_global  the best y_{t,j} over the frame (oracle: the content is still in view)
           removed = 1 - copy / mean is the share of the loss that copying removes.
           Subsets: all pairs; pairs within one shot (a cut is flagged where the lag-1 copy error
           jumps, see is_cut); static and moving sites within a shot (mean pixel change of the
           patch from t to t+1 below the source's median, or above its 90th percentile).

Frames are encoded one at a time (T = 1): the teacher is block-causal, so this is exactly its
feature for the first frame of a clip. To show what clip context changes, curve also encodes the
first 16 frames of each window as one clip and reports lags 1-15 both ways. Image sources are
skipped (a static clip is copied exactly). Frames are center-cropped to a square and resized to
224 (training uses RandomResizedCrop with area 0.8-1.0).

  python scripts/natural/diag_lag_curve.py lengths --mix scripts/natural/mixes/mix3.json --out <dir>
  python scripts/natural/diag_lag_curve.py sample  --mix scripts/natural/mixes/mix3.json --out <dir>
  python scripts/natural/diag_lag_curve.py curve   --checkpoint <run>/checkpoint_epoch240.pt --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mix_data  # noqa: E402

FPS = 7.5
LAGS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]       # frames at 7.5 fps: 0.13 s to 12.8 s
GAPS_S = [2.1, 4, 8, 16, 32, 64]
SUBSETS = ('all', 'shot', 'static', 'moving')
CUT_ABS, CUT_REL = 0.5, 4.0


def video_sources(mix):
    spec = json.loads(Path(mix).read_text())
    return [s for s in mix_data.resolve(spec)['sources'] if s['kind'] == 'video']


# ------------------------------------------------------------------- lengths
def lengths(a):
    out = {}
    for s in video_sources(a.mix):
        secs = np.concatenate([mix_data._group_index(g)[0] / (FPS * g['frame_stride']) for g in s['groups']])
        hours = secs.sum() / 3600
        q = np.percentile(secs, [10, 25, 50, 75, 90])
        out[s['name']] = dict(
            episodes=int(len(secs)), hours=round(float(hours), 1), mean_s=round(float(secs.mean()), 2),
            p10_s=round(q[0], 2), p25_s=round(q[1], 2), median_s=round(q[2], 2), p75_s=round(q[3], 2),
            p90_s=round(q[4], 2), max_s=round(float(secs.max()), 1),
            share_episodes_at_least={str(g): round(float((secs >= g).mean()), 4) for g in GAPS_S},
            share_hours_at_least={str(g): round(float(secs[secs >= g].sum() / 3600 / hours), 4) for g in GAPS_S})
        r = out[s['name']]
        print(f"{s['name']:>10}: {r['episodes']:>9,} episodes, {r['hours']:>8,.0f} h, median {r['median_s']:6.1f} s "
              f"(p10 {r['p10_s']:.1f}, p90 {r['p90_s']:.1f}), episodes >= 4/8/16 s: "
              + ' / '.join(f"{100 * r['share_episodes_at_least'][str(g)]:.0f}%" for g in (4, 8, 16)))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'episode_lengths.json').write_text(json.dumps(dict(mix=str(a.mix), fps=FPS, sources=out), indent=1) + '\n')


# -------------------------------------------------------------------- sample
def sample(a):
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for s in video_sources(a.mix):
        src = mix_data.ClipSource(s['name'], s['groups'], 16)     # the episodes training can draw
        eps = []
        for idx in rng.choice(len(src), size=min(a.episodes, len(src)), replace=False):
            length, stride = int(src.length[idx]), int(src.stride[idx])
            w = min((length - 1) // stride + 1, a.max_frames)
            off = int(rng.integers(0, length - (w - 1) * stride))
            rows = (int(src.first[idx]) + off + np.arange(w) * stride).tolist()
            eps.append(dict(episode=int(idx), seconds=length / (FPS * stride), start_s=off / (FPS * stride),
                            frames=src.shards.take(int(src.shard[idx]), rows)))
        torch.save(dict(source=s['name'], max_frames=a.max_frames, seed=a.seed, episodes=eps), a.out / f"{s['name']}.pt")
        n = [len(e['frames']) for e in eps]
        print(f"{s['name']}: {len(eps)} episodes, {sum(n)} frames, window median {int(np.median(n))} frames")


# --------------------------------------------------------------------- curve
def to_224(b):
    x = mix_data._decode(b)                                      # C,H,W uint8
    h, w = x.shape[1:]
    k = min(h, w)
    x = x[:, (h - k) // 2:(h - k) // 2 + k, (w - k) // 2:(w - k) // 2 + k].float()[None]
    x = F.interpolate(x, size=(224, 224), mode='bicubic', antialias=True, align_corners=False)
    return x[0].clamp_(0, 255).round_().to(torch.uint8)


def patch_motion(fr):
    """Mean absolute pixel change of each 16x16 patch from frame t to t+1: (W-1, 196)."""
    d = (fr[1:].float() - fr[:-1].float()).abs().mean(1, keepdim=True)
    return F.avg_pool2d(d, 16).flatten(1)


def is_cut(copy1, mean_err):
    """Lag-1 pairs whose copy error is at least half the mean baseline and 4x the episode's median."""
    return (copy1 > CUT_ABS * mean_err) & (copy1 > CUT_REL * copy1.median())


def local_mask(grid=14, r=2):
    rr, cc = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
    rr, cc = rr.flatten(), cc.flatten()
    return ((rr[:, None] - rr[None]).abs() <= r) & ((cc[:, None] - cc[None]).abs() <= r)


@torch.no_grad()
def features(teacher, nlp, fr, chunk=64):
    """Per-frame teacher features of uint8 frames (W, C, 224, 224): (W, 196, D), unit norm."""
    out = [nlp.teacher_features(teacher, nlp.normalize_video(fr[i:i + chunk, None].cuda()))[:, 0]
           for i in range(0, len(fr), chunk)]
    return torch.cat(out)


class Acc:
    """Sums per (lag, subset) over pooled pairs, plus per-episode removed shares for the SEM."""
    KEYS = ('copy', 'mean', 'nn_local', 'nn_global', 'cos')

    def __init__(self):
        self.s, self.ep = {}, {}

    def add(self, lag, sub, vals, sel, ep):
        n = int(sel.sum())
        if n == 0:
            return
        d = self.s.setdefault((lag, sub), dict(n=0, eps=set(), **{k: 0.0 for k in self.KEYS}))
        d['n'] += n
        d['eps'].add(ep)
        for k in self.KEYS:
            d[k] += float(vals[k][sel].sum())
        c, m = float(vals['copy'][sel].sum()), float(vals['mean'][sel].sum())
        self.ep.setdefault((lag, sub), []).append(1 - c / m)

    def table(self, oracles=True):
        out = {}
        for (lag, sub), d in sorted(self.s.items()):
            r = {k: d[k] / d['n'] for k in self.KEYS}
            per = np.array(self.ep[(lag, sub)])
            out.setdefault(str(lag), {})[sub] = dict(
                lag_s=round(lag / FPS, 3), pairs=d['n'], episodes=len(d['eps']),
                removed=round(1 - r['copy'] / r['mean'], 4),
                removed_sem=round(float(per.std(ddof=1) / np.sqrt(len(per))), 4) if len(per) > 1 else None,
                **(dict(removed_nn_local=round(1 - r['nn_local'] / r['mean'], 4),
                        removed_nn_global=round(1 - r['nn_global'] / r['mean'], 4)) if oracles else {}),
                **{k: round(v, 4) for k, v in r.items() if oracles or not k.startswith('nn_')})
        return out


def curve(a):
    import nlp
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    teacher = nlp.build_encoder(ck['config'])
    teacher.load_state_dict(ck['teacher'])
    teacher.cuda().eval().requires_grad_(False)
    near = local_mask().cuda()
    result, cuts_seen = dict(checkpoint=str(a.checkpoint), epoch=int(ck['epoch']), lags=LAGS, fps=FPS,
                             cut_rule=f'lag-1 copy > {CUT_ABS} x mean baseline and > {CUT_REL} x episode median',
                             sources={}), []
    for f in sorted(a.out.glob('*.pt')):
        data = torch.load(f, weights_only=False)
        name = data['source']
        frames = [torch.stack([to_224(b) for b in e['frames']]) for e in data['episodes']]
        mot = torch.cat([patch_motion(fr).flatten() for fr in frames if len(fr) > 1])
        lo, hi = float(mot.quantile(0.5)), float(mot.quantile(0.9))
        acc, clip_acc, n_cuts, seconds = Acc(), Acc(), 0, 0.0
        for ep, fr in enumerate(frames):
            w = len(fr)
            if w < 2:
                continue
            z = features(teacher, nlp, fr)                                   # W, S, D
            m = F.normalize(z.mean((0, 1)), dim=-1)
            mean_err = 2 - 2 * (z @ m)                                       # W, S
            copy1 = (2 - 2 * (z[:-1] * z[1:]).sum(-1)).mean(-1)              # W-1
            cut = is_cut(copy1, mean_err.mean())
            n_cuts += int(cut.sum())
            seconds += (w - 1) / FPS
            cuts_seen += [(name, ep, int(t)) for t in torch.nonzero(cut).flatten().tolist()]
            shot = torch.cat([cut.new_zeros(1, dtype=torch.long), cut.long().cumsum(0)])
            motion = patch_motion(fr).cuda()                                 # W-1, S
            for d in LAGS:
                if d >= w:
                    break
                zs, zt = z[:-d], z[d:]
                sims = torch.einsum('tsd,tjd->tsj', zt, zs)                  # target site s vs source site j
                vals = dict(copy=2 - 2 * (zs * zt).sum(-1), mean=mean_err[d:],
                            nn_local=2 - 2 * sims.masked_fill(~near, -2).amax(-1),
                            nn_global=2 - 2 * sims.amax(-1), cos=(zs * zt).sum(-1))
                same = (shot[:-d] == shot[d:]).cuda()[:, None].expand_as(vals['copy'])
                mv = motion[:w - d]
                for sub, sel in (('all', torch.ones_like(same)), ('shot', same),
                                 ('static', same & (mv < lo)), ('moving', same & (mv > hi))):
                    acc.add(d, sub, vals, sel, ep)
            if w >= 16:                                                      # clip context vs one frame
                zc = nlp.teacher_features(teacher, nlp.normalize_video(fr[None, :16].cuda()))[0]
                for d in range(1, 16):
                    for sub, zz in (('per_frame', z[:16]), ('clip16', zc)):
                        vals = dict(copy=2 - 2 * (zz[:-d] * zz[d:]).sum(-1),
                                    mean=2 - 2 * (zz[d:] @ F.normalize(zz.mean((0, 1)), dim=-1)),
                                    nn_local=torch.zeros_like(zz[d:, :, 0]), nn_global=torch.zeros_like(zz[d:, :, 0]),
                                    cos=(zz[:-d] * zz[d:]).sum(-1))
                        clip_acc.add(d, sub, vals, torch.ones_like(vals['copy'], dtype=torch.bool), ep)
        result['sources'][name] = dict(episodes=len(frames), seconds=round(seconds, 1), cuts=n_cuts,
                                       cuts_per_minute=round(60 * n_cuts / max(seconds, 1e-9), 2),
                                       motion_thresholds=dict(static_below=lo, moving_above=hi),
                                       lags=acc.table(), clip_context=clip_acc.table(oracles=False))
        print(f'\n{name}: {len(frames)} episodes, {seconds / 60:.1f} min, {n_cuts} cuts flagged '
              f'({result["sources"][name]["cuts_per_minute"]}/min). Share of loss removed by copying:')
        print('   lag      all    shot  static  moving   nn_global(shot)  episodes')
        for lag, r in acc.table().items():
            g = lambda k: f"{100 * r[k]['removed']:6.1f}" if k in r else '     -'  # noqa: E731
            print(f"{r['all']['lag_s']:6.2f} s {g('all')}  {g('shot')}  {g('static')}  {g('moving')}"
                  f"   {100 * r['shot']['removed_nn_global'] if 'shot' in r else float('nan'):10.1f}"
                  f"        {r['all']['episodes']}")
        (a.out / 'lag_curve.json').write_text(json.dumps(result, indent=1) + '\n')
    cut_grid(a, cuts_seen)


def cut_grid(a, cuts, n=24):
    """Frames t | t+1 of the first n flagged cuts, for checking the cut rule by eye."""
    from PIL import Image
    if not cuts:
        return
    rows, loaded = [], {}
    for name, ep, t in cuts[:n]:
        if name not in loaded:
            loaded[name] = torch.load(a.out / f'{name}.pt', weights_only=False)['episodes']
        e = loaded[name][ep]
        pair = [to_224(e['frames'][k]).permute(1, 2, 0).numpy() for k in (t, t + 1)]
        rows.append(np.concatenate(pair, 1)[::2, ::2])
    Image.fromarray(np.concatenate(rows, 0)).save(a.out / 'cuts.jpg', quality=85)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['lengths', 'sample', 'curve'])
    p.add_argument('--mix', type=Path, default=Path(__file__).resolve().parent / 'mixes' / 'mix3.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--episodes', type=int, default=96)
    p.add_argument('--max-frames', type=int, default=120)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    dict(lengths=lengths, sample=sample, curve=curve)[a.step](a)


if __name__ == '__main__':
    main()
