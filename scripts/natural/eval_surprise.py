"""Zero-shot intuitive-physics surprise (violation of expectation) for models that predict:
V-JEPA (predictor + target encoder), V-JEPA 2 (same), VideoMAE (pixel decoder), ours (block-causal
next-latent predictor + EMA teacher). Released encoders without a predictor or decoder (VideoMAE V2-L
distilled, LeVJEPA, DINOv2) have no surprise.

Ours (family nlp): the encoder is block-causal, so the predictor's output at frame t sees frames <= t
only; the surprise of a window is the model's own training loss for the one-step prediction, the
squared distance between the normalized h=1 prediction and the teacher's normalized feature of frame
t+1, over frames burnin+1..T-1, averaged over the window. Two variants, stored as two "contexts":
'full' (reported, the protocol of the other models: the past fully visible, the future predicted)
and 'masked' (as in training: 90% multi-block tube masks, error at hidden sites, the same 4 mask
seeds for every video, averaged). Training supervised hidden-site predictions only; the baselines'
predictors were likewise never trained on "every future frame hidden". Sampled at our training rate, 16 frames at
~15 fps: every 4th frame of IntPhys 2 (60 fps), every 2nd of InfLevel (30 fps); windows stride 4
(~0.27 s; the baselines' 2 frames at 6 fps are 0.33 s). Sanity check on IntPhys 2: masked error
0.99-1.00 (full: 0.78-0.82) against the 0.88 training h1 loss on natural video, 1.04-1.05 for
each frame's mean teacher feature, 0.46-0.54 for copying frame t (unavailable to the masked
predictor: a hidden site is hidden in every frame).
Arm Q (categorical targets) has no prediction in the teacher's space: its surprise is again its own
training loss, the cross-entropy of the teacher's bin codes (the checkpoint's frozen directions,
edges and running statistics) under the h=1 logits, averaged over the q_dirs directions.

Protocol, following the official code:
- IntPhys 2 Main: third_party/IntPhys2/prediction_evals (evals/intphys2/eval.py, configs
  vjepa_rope.yaml / vjepa_2.yaml / videomaev2.yaml): every 10th frame (6 fps), center square crop
  resized to the model resolution, sliding windows of `frames` with stride 2; for each context
  length C the first C frames are encoded and the rest predicted; "max-context mode" also scores
  the start of the video with every shorter even context (the break may come before frame C).
- InfLevel-Lab: third_party/jepa-intuitive-physics (Garrido et al. 2025; default_inflevel.yaml,
  inflevel_dataset.py): the pair lists in auxiliary_data_loading_files/inflevel, frames after the
  second priming segment (aligned to equal length within a pair), every 5th frame for V-JEPA (32-
  frame windows, contexts 4-20) and every 10th for VideoMAE (16-frame windows, contexts 2-10), the
  whole frame squashed square.
- Models: V-JEPA through Garrido's code (their V-JEPA eval), V-JEPA 2 through the official
  third_party/vjepa2 modules (the IntPhys 2 repo's own V-JEPA 2 copy has a different RoPE module;
  the official one is what our probes validated), VideoMAE through HF transformers.
- Surprise of a window: L1 between predicted and actual target tokens (JEPA: target encoder's
  output, layer-normed; MAE: per-patch normalized pixels, Garrido's is_mae path), averaged over
  tokens and channels. A video's surprise is the max (or mean) over windows.
- Score: pair-wise accuracy, impossible more surprising than possible (ties 0.5); IntPhys 2 pairs
  are all possible x impossible videos of a scene, InfLevel pairs are the CSV's. Reported per
  context length, for the best context, and for the per-video minimum over contexts.

Deviation: the IntPhys 2 wrapper builds its time masks for a 224-px grid whatever the resolution
(get_time_masks' spatial_dim default), so at V-JEPA 2's 256 px its context / target split is off;
here the masks use the actual grid.

  python scripts/natural/eval_surprise.py --family vjepa2 --checkpoint <ckpt> --dataset intphys2 --out <dir>
  python scripts/natural/eval_surprise.py --score --family vjepa2 --dataset intphys2 --out <dir>   # merge shards
"""
import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THIRD = Path(__file__).resolve().parents[2] / 'third_party'
D = Path('/scratch/CLUSTER_USER/datasets')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

CONFIG = {   # (family, dataset): frames per window, resolution, frame step, contexts, max-context mode
    ('vjepa', 'intphys2'): dict(frames=16, size=224, step=10, contexts=[4, 6, 8, 10, 12, 14], max_ctx=True),
    ('vjepa2', 'intphys2'): dict(frames=48, size=256, step=10, contexts=[12, 18, 24, 30, 36, 42], max_ctx=True),
    ('videomae', 'intphys2'): dict(frames=16, size=224, step=10, contexts=[4, 6, 8, 10, 12, 14], max_ctx=True),
    ('vjepa', 'inflevel_lab'): dict(frames=32, size=224, step=5, contexts=[4, 8, 12, 16, 20], max_ctx=False),
    ('vjepa2', 'inflevel_lab'): dict(frames=32, size=256, step=5, contexts=[4, 8, 12, 16, 20], max_ctx=False),
    ('videomae', 'inflevel_lab'): dict(frames=16, size=224, step=10, contexts=[2, 4, 6, 8, 10], max_ctx=False),
    ('nlp', 'intphys2'): dict(frames=16, size=224, step=4, stride=4, contexts=['full', 'masked'], max_ctx=False),
    ('nlp', 'inflevel_lab'): dict(frames=16, size=224, step=2, stride=4, contexts=['full', 'masked'], max_ctx=False),
}


# ------------------------------------------------------------------ data
def videos(dataset):
    """IntPhys 2: [(path, label, condition, scene)]; InfLevel: [(path, start frame, label, property, pair id)]."""
    if dataset == 'intphys2':
        root = D / 'intphys2/Main'
        with open(root / 'metadata.csv') as f:
            return [(root / r['file_name'], 0, int(r['type'].endswith('_Possible')), r['condition'], r['SceneIndex'])
                    for r in csv.DictReader(f)]
    out = []
    aux = THIRD / 'jepa-intuitive-physics/evaluation_code/auxiliary_data_loading_files/inflevel'
    for prop in ('continuity', 'solidity', 'gravity'):
        with open(aux / f'{prop}.csv') as f:
            for i, r in enumerate(csv.DictReader(f)):
                p1, p2 = D / 'inflevel/inflevel_lab' / r['vid1_path'], D / 'inflevel/inflevel_lab' / r['vid2_path']
                n1, n2 = n_frames(p1), n_frames(p2)
                e1, e2 = int(r['vid1_end_priming_2']), int(r['vid2_end_priming_2'])
                diff = (n1 - e1) - (n2 - e2)                  # inflevel_dataset.py: equal remaining length
                s1, s2 = e1 + max(diff, 0) + 1, e2 + max(-diff, 0) + 1
                out.append((p1, s1, int(r['vid1_label'] == 'possible'), prop, f'{prop}_{i}'))
                out.append((p2, s2, int(r['vid2_label'] == 'possible'), prop, f'{prop}_{i}'))
    return out


def n_frames(path):
    import av
    with av.open(str(path)) as c:
        return c.streams.video[0].frames


def load(path, start, step, size, square_crop):
    """[3, T, size, size] normalized: every `step`-th frame from `start`."""
    import av
    with av.open(str(path)) as c:
        st = c.streams.video[0]; st.thread_type = 'AUTO'
        fr = [f.to_ndarray(format='rgb24') for i, f in enumerate(c.decode(st)) if i >= start and (i - start) % step == 0]
    v = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).float() / 255          # T,C,H,W
    if square_crop:                                                                 # center square
        H, W = v.shape[-2:]; s = min(H, W)
        v = v[..., (H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]
    v = F.interpolate(v, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
    return (v.permute(1, 0, 2, 3) - MEAN) / STD


# ------------------------------------------------------------------ models
def time_masks(ctx, frames, grid, tubelet=2):
    n = grid * grid * frames // tubelet
    k = grid * grid * ctx // tubelet
    idx = torch.arange(n)
    return idx[:k], idx[k:], idx


def _clean(sd):
    return {k.replace('module.', ''): v for k, v in sd.items()}


def _load(model, sd):
    own = model.state_dict()
    sd = {k: (v if k not in own or own[k].shape == v.shape else own[k]) for k, v in sd.items()}   # sincos tables
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all('pos_embed' in m for m in missing), (missing[:5], unexpected[:5])


class JEPA:
    """V-JEPA (Garrido's jepa-intuitive-physics code) or V-JEPA 2 (official third_party/vjepa2 code)."""
    def __init__(self, family, checkpoint, frames, size):
        repo = (THIRD / 'jepa-intuitive-physics/evaluation_code' if family == 'vjepa' else THIRD / 'vjepa2')
        sys.path.insert(0, str(repo))
        self.official2 = family == 'vjepa2'
        try:
            import beartype  # noqa: F401
        except ImportError:   # rope.py uses it only as a runtime type-check decorator
            import types
            import typing
            bt = types.ModuleType('beartype'); bt.beartype = lambda f: f; bt.typing = typing
            sys.modules['beartype'], sys.modules['beartype.typing'] = bt, typing
        import src.models.vision_transformer as vit
        import src.models.predictor as vit_pred
        from src.masks.utils import apply_masks
        self.apply_masks = apply_masks
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        rope = family == 'vjepa2'
        enc_kw = dict(img_size=size, patch_size=16, num_frames=frames, tubelet_size=2, uniform_power=True,
                      use_sdpa=True)
        if rope:
            enc_kw['use_rope'] = True
        n_mask = sum(k.startswith('module.backbone.mask_tokens.') for k in ck['predictor'])
        pred_kw = dict(img_size=size, patch_size=16, num_frames=frames, tubelet_size=2, embed_dim=1024,
                       predictor_embed_dim=384, depth=12, num_heads=12 if rope else 16, uniform_power=True,
                       use_mask_tokens=True, num_mask_tokens=n_mask, zero_init_mask_tokens=True, use_sdpa=True)
        if rope:
            pred_kw['use_rope'] = True
        if self.official2:            # bare modules, weights under module.backbone.
            self.enc, self.tgt = vit.vit_large(**enc_kw), vit.vit_large(**enc_kw)
            self.pred = vit_pred.vit_predictor(**pred_kw)
            strip = lambda sd: {k.replace('module.backbone.', ''): v for k, v in sd.items()}
        else:                         # Garrido's MultiMask wrappers, weights under backbone.
            from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
            self.enc, self.tgt = MultiMaskWrapper(vit.vit_large(**enc_kw)), MultiMaskWrapper(vit.vit_large(**enc_kw))
            self.pred = PredictorMultiMaskWrapper(vit_pred.vit_predictor(**pred_kw))
            strip = _clean
        _load(self.enc, strip(ck['encoder']))
        _load(self.tgt, strip(ck['target_encoder']))
        _load(self.pred, strip(ck['predictor']))
        for m in (self.enc, self.tgt, self.pred):
            m.cuda().eval().requires_grad_(False)
        self.grid = size // 16
        self.info = dict(epoch=int(ck.get('epoch', -1)), num_mask_tokens=n_mask, pred_heads=pred_kw['num_heads'])

    @torch.no_grad()
    def targets(self, x):
        """Layer-normed target-encoder tokens of the whole window (independent of the context length)."""
        if self.official2:
            return F.layer_norm(self.tgt(x), (1024,))
        full = time_masks(2, x.shape[2], self.grid)[2][None].repeat(x.shape[0], 1).cuda()
        h = self.tgt(x, [full])[0]
        return F.layer_norm(h, (h.size(-1),))

    @torch.no_grad()
    def surprise(self, x, ctx, h=None):
        """x [B,3,T,H,W] -> [B] mean L1 over the predicted tokens; h: cached targets(x)."""
        B, T = x.shape[0], x.shape[2]
        m, m_, _ = (t[None].repeat(B, 1).cuda() for t in time_masks(ctx, T, self.grid))
        h = self.targets(x) if h is None else h
        if self.official2:
            targets = self.apply_masks(h, [m_])
            preds = self.pred(self.enc(x, masks=[m]), [m], [m_], mask_index=0)
        else:
            targets = self.apply_masks(h, [m_], concat=False)[0]
            preds = self.pred(self.enc(x, [m]), [targets], [m], [m_])[0]
        return F.l1_loss(preds.float(), targets.float(), reduction='none').mean((1, 2))


class MAE:
    """HF VideoMAEForPreTraining (MCG-NJU/videomae-large ships its decoder); Garrido's is_mae targets."""
    def __init__(self, checkpoint, frames, size):
        from transformers import VideoMAEForPreTraining
        assert frames == 16 and size == 224, 'fixed sincos tables: 16 frames at 224'
        self.model = VideoMAEForPreTraining.from_pretrained(str(Path(checkpoint).parent)).cuda().eval()
        self.grid, self.info = size // 16, dict(norm_pix_loss=self.model.config.norm_pix_loss)

    def targets(self, x):
        return None

    @torch.no_grad()
    def surprise(self, x, ctx, h=None):
        from einops import rearrange
        B, T = x.shape[0], x.shape[2]
        _, m_, _ = time_masks(ctx, T, self.grid)
        bool_pos = torch.zeros(B, self.grid * self.grid * T // 2, dtype=torch.bool, device=x.device)
        bool_pos[:, m_.cuda()] = True
        preds = self.model(pixel_values=x.transpose(1, 2), bool_masked_pos=bool_pos).logits     # B, Nm, 1536
        v = x * STD.view(1, 3, 1, 1, 1).cuda() + MEAN.view(1, 3, 1, 1, 1).cuda()               # [0, 1]
        p = rearrange(v, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=16, p2=16)
        p = (p - p.mean(-2, keepdim=True)) / (p.var(-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        targets = rearrange(p, 'b n p c -> b n (p c)')[bool_pos].reshape(B, -1, preds.shape[-1])
        return F.l1_loss(preds.float(), targets.float(), reduction='none').mean((1, 2))


class NLP:
    """Our next-latent-prediction checkpoint: student encoder + predictor + EMA teacher; the mask
    token (not in the milestone files) from mask_token_<tag>.pt beside the checkpoint."""
    def __init__(self, checkpoint, seeds=4):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import nlp
        self.nlp = nlp
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = self.cfg = ck['config']
        self.student = nlp.Student(cfg)
        self.student.encoder.load_state_dict(ck['encoder'])
        self.student.predictor.load_state_dict(ck['predictor'])
        if cfg['arm'] == 'Q':
            self.student.head.load_state_dict(ck['head'])
            for k, v in ck['quantizer'].items():
                getattr(self.student, k).copy_(v)
        mt = torch.load(Path(checkpoint).parent / f"mask_token_e{int(ck['epoch']):03d}.pt", map_location='cpu')
        assert mt['step'] == ck['step'], (mt['step'], ck['step'])
        self.student.mask_token.data.copy_(mt['mask_token'])
        self.teacher = nlp.build_encoder(cfg)
        self.teacher.load_state_dict(ck['teacher'])
        for m in (self.student, self.teacher):
            m.cuda().eval().requires_grad_(False)
        import numpy as np
        self.masks = [nlp.tube_block_masks(1, cfg['mask_ratio'], cfg['mask_block_scale'], cfg['mask_block_aspect'],
                                           np.random.default_rng(1000 + k)).cuda() for k in range(seeds)]
        self.info = dict(epoch=int(ck['epoch']), step=int(ck['step']), mask_ratio=cfg['mask_ratio'], mask_seeds=seeds,
                         burnin=cfg['burnin'])

    def targets(self, x):
        return self.nlp.teacher_features(self.teacher, x, per_frame=self.cfg.get('target_view') == 'frame')

    def error(self, pred, z):
        """Per-site h=1 training loss: pred B,f,S,D (predictor output), z B,f,S,D (teacher)."""
        if self.cfg['arm'] == 'Q':
            k = self.cfg['q_bins']
            codes = self.student.quantize(z)                                            # B, f, S, m
            logits = self.student.head(pred).float().view(*codes.shape, k)
            return F.cross_entropy(logits.reshape(-1, k), codes.reshape(-1), reduction='none').view(codes.shape).mean(-1)
        return (F.normalize(pred.float(), dim=-1) - z).square().sum(-1)

    @torch.no_grad()
    def surprise(self, x, ctx, z=None):
        z = self.targets(x) if z is None else z
        b, burn, enc = x.shape[0], self.cfg['burnin'], self.student.encoder
        S, lo = self.nlp.SITES, self.cfg['burnin'] + 1
        if ctx == 'full':
            pred = self.student.predictor(enc(x)[:, 1:])[1][:, burn:]                   # B, T-1-burn, S, D
            err = self.error(pred, z[:, lo:])                                           # B, f, S
            return err.mean((1, 2))
        out = []
        for vis1 in self.masks:
            vis = vis1.expand(b, -1)
            tok, ids = self.nlp.encode_visible(enc, x, vis)
            patches, d = tok[:, 1:], tok.shape[-1]
            memory = self.student.mask_token.to(patches.dtype).expand(b, x.shape[2] * S, d)
            memory = memory.scatter(1, ids.unsqueeze(-1).expand(-1, -1, d), patches)
            pred = self.student.predictor(memory)[1][:, burn:]
            err = self.error(pred, z[:, lo:])
            hidden = torch.ones(S, dtype=torch.bool, device=x.device).scatter_(0, vis1[0], False)
            out.append(err[:, :, hidden].mean((1, 2)))
        return torch.stack(out).mean(0)


# ------------------------------------------------------------------ extraction
@torch.no_grad()
def video_surprise(model, clip, cfg, chunk=12):
    """clip [3,T,H,W] -> {context: [windows]} (max-context mode prepends the shorter-context starts).
    Target tokens are computed once per window and each start context once per video: the same
    numbers as recomputing them per context length, as the official loop does."""
    F_, cs = cfg['frames'], cfg['contexts']
    clip = clip.cuda()
    if clip.shape[1] < F_:                                   # too short for one window: repeat the last frame
        clip = torch.cat([clip, clip[:, -1:].expand(-1, F_ - clip.shape[1], -1, -1)], 1)
    wins = clip.unfold(1, F_, cfg.get('stride', 2)).permute(1, 0, 4, 2, 3)      # W,3,F,H,W
    with torch.autocast('cuda', dtype=torch.bfloat16):
        start = {}
        if cfg['max_ctx']:
            first = clip[None, :, :F_]
            h0 = model.targets(first)
            start = {k: model.surprise(first, k, h0)[0] for k in range(2, max(cs), 2)}
        main = {c: [] for c in cs}
        chunk = 4 if isinstance(model, NLP) else chunk
        for i in range(0, len(wins), chunk):
            w = wins[i:i + chunk]
            h = model.targets(w)
            for c in cs:
                main[c].append(model.surprise(w, c, h))
    out = {}
    for c in cs:
        parts = [torch.stack([start[k] for k in range(2, c, 2)])] if cfg['max_ctx'] and c > 2 else []
        out[c] = torch.cat(parts + [torch.cat(main[c])]).float().cpu().numpy()
    return out


def score(rows, surprises, dataset, contexts):
    """Pair-wise accuracy per group: impossible more surprising than possible."""
    def pairs():
        if dataset == 'intphys2':
            by = {}
            for i, r in enumerate(rows):
                by.setdefault(r[4], []).append(i)
            for idx in by.values():
                for p in (i for i in idx if rows[i][2] == 1):
                    for q in (i for i in idx if rows[i][2] == 0):
                        yield rows[p][3], p, q
        else:
            by = {}
            for i, r in enumerate(rows):
                by.setdefault(r[4], []).append(i)
            for idx in by.values():
                p = [i for i in idx if rows[i][2] == 1]; q = [i for i in idx if rows[i][2] == 0]
                if p and q:
                    yield rows[p[0]][3], p[0], q[0]
    agg = {'max': np.max, 'mean': np.mean}
    res = {}
    for an, af in agg.items():
        s = {c: np.array([af(surprises[i][c]) for i in range(len(rows))]) for c in contexts}
        numeric = all(isinstance(c, int) for c in contexts)
        if numeric:
            s['min_ctx'] = np.min(np.stack([s[c] for c in contexts]), 0)
        for key in list(contexts) + (['min_ctx'] if numeric else []):
            acc = {}
            for g, p, q in pairs():
                acc.setdefault(g, []).append(1.0 if s[key][q] > s[key][p] else 0.5 if s[key][q] == s[key][p] else 0.0)
            per = {g: round(100 * float(np.mean(v)), 2) for g, v in sorted(acc.items())}
            per['average'] = round(float(np.mean(list(per.values()))), 2)
            res.setdefault(an, {})[str(key)] = per
        best = {g: max(res[an][str(c)][g] for c in contexts) for g in res[an][str(contexts[0])]}
        res[an]['best_ctx_per_group'] = best
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True, choices=['vjepa', 'vjepa2', 'videomae', 'nlp'])
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--dataset', required=True, choices=['intphys2', 'inflevel_lab'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=None, help='smoke test: first N videos')
    p.add_argument('--score', action='store_true', help='merge shard files and write the metrics')
    a = p.parse_args()
    cfg = CONFIG[(a.family, a.dataset)]
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f'surprise_{a.dataset}'
    rows = videos(a.dataset)
    if a.limit:
        rows = rows[:a.limit]
    if a.score:
        surprises = {}
        for f in sorted(a.out.glob(f'{tag}_shard*of{a.num_shards}.pkl')):
            surprises.update(pickle.load(open(f, 'rb')))
        assert len(surprises) == len(rows), (len(surprises), len(rows))
        res = dict(family=a.family, dataset=a.dataset, config=cfg, n_videos=len(rows),
                   metric='pair-wise accuracy, impossible more surprising than possible (ties 0.5)',
                   **score(rows, surprises, a.dataset, cfg['contexts']))
        (a.out / f'{tag}.json').write_text(json.dumps(res, indent=2) + '\n')
        for an in ('max', 'mean'):
            print(an, {k: v['average'] for k, v in res[an].items() if k != 'best_ctx_per_group'},
                  'best ctx', res[an]['best_ctx_per_group'])
        return
    model = (MAE(a.checkpoint, cfg['frames'], cfg['size']) if a.family == 'videomae'
             else NLP(a.checkpoint) if a.family == 'nlp'
             else JEPA(a.family, a.checkpoint, cfg['frames'], cfg['size']))
    print('model', model.info, 'config', cfg, flush=True)
    mine = list(range(a.shard, len(rows), a.num_shards))
    out_f = a.out / f'{tag}_shard{a.shard}of{a.num_shards}{"_smoke" if a.limit else ""}.pkl'
    done = pickle.load(open(out_f, 'rb')) if out_f.exists() else {}
    t0 = time.time()
    for n, i in enumerate(mine):
        if i in done:
            continue
        path, start, *_ = rows[i]
        clip = load(path, start, cfg['step'], cfg['size'], square_crop=a.dataset == 'intphys2')
        done[i] = video_surprise(model, clip, cfg)
        if n % 50 == 0:
            pickle.dump(done, open(out_f, 'wb'))
            print(f'{a.dataset} shard {a.shard}: {n}/{len(mine)} videos, {time.time() - t0:.0f}s', flush=True)
    pickle.dump(done, open(out_f, 'wb'))
    print('done', len(done), flush=True)


if __name__ == '__main__':
    main()
