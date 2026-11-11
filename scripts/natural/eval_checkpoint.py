"""Frozen-feature evaluation of one pretraining checkpoint on K400, IN1K and SSv2,
plus the IN1K probes scored on the ImageNet robustness sets.

One augmentation-free extraction pass over the fixed evaluation caches
(scripts/natural/build_eval_caches.py), then every probe is fitted on the
cached features, so three probe families and their learning-rate grids cost
one encoder pass. Probes are always fitted on the complete training sets and
scored on the complete validation sets (the per-class subset caches are not
used any more). This is the scaling-curve protocol; it is NOT V-JEPA's
probe-with-augmentation protocol, which is reserved for final table numbers.

Encoder input: video = 8 frames; image = the image repeated 8 times along
time (V-JEPA's adaptation). Block-causal encoders are run causally.
Features kept per item:
  cls    [D]            the clip-level readout token
  mean   [D]            mean of all patch tokens
  grid   video [8*4*4, D] (4x4 average-pooled per frame)
         image [7*7, D]   (last temporal slot, 7x7 average-pooled)
Probes (top-1 on the validation cache, best of the LR grid, val-selected):
  linear_mean, linear_cls   standardized features -> Linear
  attentive                 one learned query, cross-attention, MLP (V-JEPA style)
  conv_attentive            residual 3x3 conv bottleneck on the grid, then attentive
Each IN1K probe (the LR it selected) is also scored, without refitting, on
in1k_v2, in1k_sketch, in1k_r and in1k_a; R and A are 200-class subsets, so
their logits are restricted to the covered classes. --datasets picks the
probed sets (default: every one whose caches exist; K400 and IN1K required).

  python scripts/natural/eval_checkpoint.py --checkpoint <ckpt> --out <dir> \
      --family levjepa --run levjepa_k710_vitb --epoch 60
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CACHE = Path('/scratch/CLUSTER_USER/datasets/eval_caches')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


# ----------------------------------------------------------------- encoders
def load_levjepa(path):
    """EMA encoder weights from a LeVJEPA Lightning checkpoint."""
    sys.path.insert(0, '/home/CLUSTER_USER/video_crw/third_party/LeVJEPA')
    import module as vit_models
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['hyper_parameters']['model']
    enc = getattr(vit_models, cfg['name'])(
        img_size=cfg['img_size'], patch_size=cfg['patch_size'], num_frames=cfg['num_frames'],
        tubelet_size=cfg['tubelet_size'], use_rope=cfg.get('use_rope', True),
        token_drop_rate=0.0, attn_mode=cfg.get('attn_mode', 'full'))
    # EVAL_WEIGHTS=raw evaluates the training weights instead of the EMA.
    # LeVJEPA's WeightEMA has no warm-up: it starts from the random init and
    # decays 0.9999 per step, so after 8,640 steps 0.9999**8640 = 42% of the
    # EMA is still the initialization. Their eval default is the EMA anyway.
    want_raw = os.environ.get('EVAL_WEIGHTS', 'ema') == 'raw'
    source = 'state_dict_ema' if ck.get('state_dict_ema') and not want_raw else 'state_dict'
    sd = {k[len('encoder.'):]: v for k, v in ck[source].items() if k.startswith('encoder.')}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected and all('pos_embed' in m for m in missing), (missing, unexpected)
    info = dict(model=cfg['name'], weights=source, epoch=int(ck['epoch']), global_step=int(ck['global_step']),
                attn_mode=cfg.get('attn_mode'), embed_dim=enc.embed_dim)
    return enc.eval().requires_grad_(False), info


def load_nlp(path):
    """Encoder from a next-latent-prediction milestone (scripts/natural/train_nlp.py).
    EVAL_WEIGHTS=student (default; alias raw) or teacher (alias ema)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nlp
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    which = {'raw': 'student', 'ema': 'teacher'}.get(os.environ.get('EVAL_WEIGHTS', 'student'),
                                                     os.environ.get('EVAL_WEIGHTS', 'student'))
    assert which in ('student', 'teacher'), which
    enc = nlp.build_encoder(cfg)
    enc.load_state_dict(ck['encoder' if which == 'student' else 'teacher'], strict=True)
    info = dict(model=cfg['size'], weights=which, epoch=int(ck['epoch']), global_step=int(ck['step']),
                attn_mode='block_causal', embed_dim=enc.embed_dim, arm=cfg['arm'])
    return enc.eval().requires_grad_(False), info


THIRD_PARTY = Path('/home/CLUSTER_USER/video_crw/third_party')


class TubeletPairs(nn.Module):
    """A tubelet-2, CLS-free encoder (V-JEPA, V-JEPA 2) on our 8-frame layout: each frame
    is repeated twice, so the encoder sees its native 16-frame clip and returns 8 temporal
    slots of 14x14 tokens, the grid every probe here expects; the token mean stands in
    for the CLS token these encoders do not have (linear_cls then equals linear_mean)."""
    def __init__(self, enc):
        super().__init__()
        self.enc, self.embed_dim = enc, enc.embed_dim

    def forward(self, x):                                   # B,3,8,H,W
        tok = self.enc(x.repeat_interleave(2, dim=2))       # B,8*196,D
        return torch.cat([tok.mean(1, keepdim=True), tok], 1)


def _official_target_encoder(repo, path, build):
    """Target (EMA) encoder of an official facebookresearch checkpoint, keys cleaned as in
    their evals/video_classification_frozen loaders. weights_only: no pickled code runs."""
    sys.path.insert(0, str(THIRD_PARTY / repo))
    import importlib.util
    import types
    # vjepa2 imports timm's drop_path (identity at drop_path_rate 0, as in eval); stub timm
    # if the env lacks it rather than install it (it can pull in another torchvision)
    if importlib.util.find_spec('timm') is None:
        layers = types.ModuleType('timm.models.layers')
        layers.drop_path = lambda x, drop_prob=0., training=False: x
        for name, mod in (('timm', types.ModuleType('timm')), ('timm.models', types.ModuleType('timm.models')),
                          ('timm.models.layers', layers)):
            sys.modules.setdefault(name, mod)
    from src.models import vision_transformer as vit
    enc = build(vit)
    ck = torch.load(path, map_location='cpu', weights_only=True)
    sd = {k.replace('module.', '').replace('backbone.', ''): v for k, v in ck['target_encoder'].items()}
    enc.load_state_dict(sd, strict=True)
    return enc, int(ck.get('epoch', -1))


def load_vjepa(path):
    """V-JEPA ViT-L/16 (Bardes et al. 2024; facebookresearch/jepa vitl16.pth.tar), built as in
    configs/evals/vitl16_k400_16x8x3.yaml: 16 frames, tubelet 2, uniform_power, sincos pos."""
    enc, epoch = _official_target_encoder('jepa', path, lambda vit: vit.vit_large(
        img_size=224, num_frames=16, tubelet_size=2, uniform_power=True, use_sdpa=True))
    info = dict(model='vjepa_vitl16', weights='target_encoder', epoch=epoch, global_step=-1,
                attn_mode='full', embed_dim=enc.embed_dim)
    return TubeletPairs(enc).eval().requires_grad_(False), info


def load_vjepa2(path):
    """V-JEPA 2 ViT-L (facebookresearch/vjepa2 vitl.pt), built as in configs/eval/vitl/k400.yaml
    (RoPE, tubelet 2, uniform_power) but at our caches' 224 px instead of their 256."""
    enc, epoch = _official_target_encoder('vjepa2', path, lambda vit: vit.vit_large(
        img_size=224, num_frames=16, patch_size=16, tubelet_size=2, uniform_power=True, use_rope=True,
        use_sdpa=True))
    info = dict(model='vjepa2_vitl', weights='target_encoder', epoch=epoch, global_step=-1,
                attn_mode='full', embed_dim=enc.embed_dim)
    return TubeletPairs(enc).eval().requires_grad_(False), info


def load_levjepa_hf(path):
    """Released LeVJEPA weights (HF galilai-group/LeVJEPA-*: model.safetensors + config.json),
    loaded into the vendored LeVJEPA module, whose VisionTransformer the HF modeling code
    copies, so the HF remote code is never run. Tubelet 1 with a CLS token: our 8 frames
    give our layout directly; block-causal attention as in config.json."""
    from safetensors.torch import load_file
    sys.path.insert(0, str(THIRD_PARTY / 'LeVJEPA'))
    import module as vit_models
    cfg = json.loads((Path(path).parent / 'config.json').read_text())
    sizes = {768: 'vit_base', 1024: 'vit_large', 1280: 'vit_huge'}
    enc = getattr(vit_models, sizes[cfg['embed_dim']])(
        img_size=cfg['img_size'], patch_size=cfg['patch_size'], num_frames=cfg['num_frames'],
        tubelet_size=cfg['tubelet_size'], uniform_power=cfg['uniform_power'], use_rope=cfg['use_rope'],
        token_drop_rate=0.0, attn_mode=cfg['attn_mode'])
    sd = load_file(path)
    want = set(enc.state_dict())
    for prefix in ('', 'encoder.', 'model.', 'backbone.', 'model.encoder.'):   # whatever the HF wrapper prepends
        cand = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if want <= set(cand):
            break
    missing, unexpected = enc.load_state_dict({k: cand[k] for k in want if k in cand}, strict=False)
    assert not missing, (prefix, missing[:5])
    info = dict(model=f"levjepa_hf_{sizes[cfg['embed_dim']]}", weights='released', epoch=-1, global_step=-1,
                attn_mode=cfg['attn_mode'], embed_dim=enc.embed_dim)
    return enc.eval().requires_grad_(False), info


# Released baselines for the paper-probe table (probe_vjepa.py). Each returns an encoder that maps a
# normalized clip [B,3,T,H,W] to all its output tokens [B,N,D], with no CLS token (info cls=False):
# tubelet-2 video models take 16 frames (8 temporal slots), the image model DINOv2 takes 8 frames
# (one slot per frame), as in the token-matched protocol. They are not wired into the cached tier.
class _VideoMAE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.embed_dim = model, model.config.hidden_size

    def forward(self, x):                        # transformers' VideoMAE takes [B,T,C,H,W]
        return self.model(pixel_values=x.transpose(1, 2)).last_hidden_state


class _VideoMAEv2(nn.Module):
    """VideoMAEv2's ViT up to the last block, then its fc_norm applied per token (the released
    forward mean-pools first and returns one vector)."""
    def __init__(self, vit):
        super().__init__()
        self.vit, self.embed_dim = vit, vit.embed_dim

    def forward(self, x):
        v = self.vit
        t = v.patch_embed(x)
        t = t + v.pos_embed.to(t.device, t.dtype)
        for blk in v.blocks:
            t = blk(t)
        return v.fc_norm(t)


class _PerFrame(nn.Module):
    """An image encoder applied to every frame; patch tokens of all frames, in time order."""
    def __init__(self, model):
        super().__init__()
        self.model, self.embed_dim = model, model.config.hidden_size

    def forward(self, x):
        b, c, t, h, w = x.shape
        out = self.model(pixel_values=x.transpose(1, 2).reshape(b * t, c, h, w)).last_hidden_state
        return out[:, 1:].reshape(b, -1, out.shape[-1])  # drop each frame's CLS


def load_videomae(path):
    """VideoMAE (Tong et al. 2022) pretrained-only weights, e.g. HF MCG-NJU/videomae-large
    (self-supervised on K400, no labels); encoder of VideoMAEForPreTraining, final LayerNorm on."""
    from transformers import VideoMAEModel
    model = VideoMAEModel.from_pretrained(Path(path).parent, attn_implementation='sdpa')
    info = dict(model=f'videomae_{model.config.hidden_size}', weights='pretrained', epoch=-1, global_step=-1,
                attn_mode='full', embed_dim=model.config.hidden_size, cls=False, token_matched_frames=16)
    return _VideoMAE(model).eval().requires_grad_(False), info


def load_videomaev2(path):
    """VideoMAE V2 (Wang et al. 2023) released encoders, e.g. HF OpenGVLab/VideoMAEv2-Large
    (self-supervised on UnlabeledHybrid-1M), built from the vendored architecture."""
    from safetensors.torch import load_file
    sys.path.insert(0, str(THIRD_PARTY / 'videomaev2_hf'))
    import modeling_videomaev2 as vm2
    cfg = json.loads((Path(path).parent / 'config.json').read_text())['model_config']
    vit = vm2.VisionTransformer(**cfg)
    sd = {k[len('model.'):]: v for k, v in load_file(path).items()}
    vit.load_state_dict(sd, strict=True)
    info = dict(model=f"videomaev2_{cfg['embed_dim']}", weights='pretrained', epoch=-1, global_step=-1,
                attn_mode='full', embed_dim=cfg['embed_dim'], cls=False, token_matched_frames=16)
    return _VideoMAEv2(vit).eval().requires_grad_(False), info


def load_dinov2(path):
    """DINOv2 (Oquab et al. 2023), e.g. HF facebook/dinov2-large, applied per frame at 224 px
    (16x16 patches of 14 px per frame, position embeddings interpolated by the model)."""
    from transformers import Dinov2Model
    model = Dinov2Model.from_pretrained(Path(path).parent, attn_implementation='sdpa')
    info = dict(model=f'dinov2_{model.config.hidden_size}', weights='pretrained', epoch=-1, global_step=-1,
                attn_mode='full', embed_dim=model.config.hidden_size, cls=False, token_matched_frames=8)
    return _PerFrame(model).eval().requires_grad_(False), info


LOADERS = {'levjepa': load_levjepa, 'nlp': load_nlp, 'vjepa': load_vjepa, 'vjepa2': load_vjepa2,
           'levjepa_hf': load_levjepa_hf, 'videomae': load_videomae, 'videomaev2': load_videomaev2,
           'dinov2': load_dinov2}


class Cache(torch.utils.data.Dataset):
    def __init__(self, name, limit=None):
        self.x = np.load(CACHE / f'{name}.npy', mmap_mode='r')
        self.y = np.load(CACHE / f'{name}_labels.npy')
        keep = np.flatnonzero(np.load(CACHE / f'{name}_valid.npy'))
        self.idx = keep[:limit] if limit else keep

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return torch.from_numpy(np.ascontiguousarray(self.x[j])), int(self.y[j])


@torch.no_grad()
def extract(enc, name, batch, workers, limit=None):
    ds = Cache(name, limit)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    video = ds.x.ndim == 5
    n, d = len(ds), enc.embed_dim
    g_tokens = 8 * 16 if video else 49
    out = dict(cls=torch.empty(n, d), mean=torch.empty(n, d), grid=torch.empty(n, g_tokens, d, dtype=torch.half),
               y=torch.empty(n, dtype=torch.long))
    mean, std = MEAN.cuda(), STD.cuda()
    i = 0
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        if not video:
            x = x[:, None].expand(-1, 8, -1, -1, -1)
        x = x.permute(0, 4, 1, 2, 3).float().div_(255).sub_(mean).div_(std)  # B,3,T,H,W
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok = enc(x)
        b = tok.shape[0]
        patches = tok[:, 1:].float().reshape(b, 8, 14, 14, d)
        if video:
            g = F.adaptive_avg_pool2d(patches.flatten(0, 1).permute(0, 3, 1, 2), 4)  # B*8,D,4,4
            g = g.permute(0, 2, 3, 1).reshape(b, 8 * 16, d)
        else:
            g = F.adaptive_avg_pool2d(patches[:, -1].permute(0, 3, 1, 2), 7)          # B,D,7,7
            g = g.permute(0, 2, 3, 1).reshape(b, 49, d)
        out['cls'][i:i + b] = tok[:, 0].float().cpu()
        out['mean'][i:i + b] = patches.mean((1, 2, 3)).cpu()
        out['grid'][i:i + b] = g.half().cpu()
        out['y'][i:i + b] = y
        i += b
    assert i == n
    return out | dict(video=video)


# ------------------------------------------------------------------- probes
class Attentive(nn.Module):
    """V-JEPA style: one learned query cross-attends to the tokens."""
    def __init__(self, d, classes, conv=None):
        super().__init__()
        self.conv = conv
        self.norm = nn.LayerNorm(d)
        self.query = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(self.query, std=.02)
        self.attn = nn.MultiheadAttention(d, max(1, d // 64), batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, classes))

    def forward(self, tokens):
        if self.conv is not None:
            tokens = self.conv(tokens)
        t = self.norm(tokens)
        q = self.query.expand(len(t), -1, -1)
        q = q + self.attn(q, t, t, need_weights=False)[0]
        q = q + self.mlp(q)
        return self.head(q[:, 0])


class GridConv(nn.Module):
    """Residual 3x3 bottleneck over each frame's spatial grid."""
    def __init__(self, d, side, frames):
        super().__init__()
        self.side, self.frames = side, frames
        h = max(64, d // 4)
        self.net = nn.Sequential(nn.Conv2d(d, h, 3, padding=1), nn.GELU(), nn.Conv2d(h, d, 3, padding=1))

    def forward(self, tokens):
        b, n, d = tokens.shape
        g = tokens.reshape(b * self.frames, self.side, self.side, d).permute(0, 3, 1, 2)
        g = g + self.net(g)
        return g.permute(0, 2, 3, 1).reshape(b, n, d)


def _device(x):
    """Put a feature tensor on the GPU when it fits with room to spare (full IN1K grid for ViT-B is ~96 GB)."""
    free, _ = torch.cuda.mem_get_info()
    return x.cuda() if x.numel() * x.element_size() < 0.75 * free else x


def fit(make, xtr, ytr, xva, yva, lrs, epochs, batch=1024, wd=0.01, holdout=10000):
    """Train one probe per LR on the train split minus a fixed held-out slice, pick the LR
    on that slice, report validation accuracy of the picked probe (val never selects)."""
    holdout = min(holdout, len(xtr) // 10)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(xtr), generator=g)
    sel, fit_idx = perm[:holdout], perm[holdout:]
    xsel, ysel = xtr[sel], ytr[sel]
    x_all = _device(xtr)                         # one resident copy; batches index into it
    y_all = ytr.to(x_all.device)
    fit_idx = fit_idx.to(x_all.device)
    n = len(fit_idx)

    def accuracy(head, x, y):
        correct = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(x), 2048):
                correct += int((head(x[i:i + 2048].cuda().float()).argmax(-1).cpu() == y[i:i + 2048]).sum())
        return correct / len(x)

    best = dict(select_acc=-1.)
    for lr in lrs:
        torch.manual_seed(0)
        head = make().cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
        steps = epochs * ((n + batch - 1) // batch); step = 0
        for ep in range(epochs):
            head.train(); order = fit_idx[torch.randperm(n, device=fit_idx.device)]
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                for pg in opt.param_groups:
                    pg['lr'] = lr * min(1., (step + 1) / (0.05 * steps)) * 0.5 * (1 + np.cos(np.pi * step / steps))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = F.cross_entropy(head(x_all[idx].cuda(non_blocking=True).float()).float(),
                                           y_all[idx].cuda(non_blocking=True))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
        head.eval()
        select_acc = accuracy(head, xsel, ysel)
        if select_acc > best['select_acc']:
            best = dict(select_acc=select_acc, acc=accuracy(head, xva, yva), lr=lr, train_loss=float(loss), head=head)
    del x_all, y_all
    torch.cuda.empty_cache()
    return best


def probes(tr, va, classes, epochs):
    """-> (scores per probe, {probe: (head, feature key, mean, std)} for scoring other sets)."""
    d = tr['cls'].shape[-1]; res, heads = {}, {}
    for key in ('mean', 'cls'):
        mu, sd = tr[key].mean(0), tr[key].std(0).clamp_min(1e-6)
        r = fit(lambda: nn.Linear(d, classes), (tr[key] - mu) / sd, tr['y'],
                (va[key] - mu) / sd, va['y'], lrs=(1e-2, 1e-3), epochs=epochs, wd=0.)
        heads[f'linear_{key}'] = (r.pop('head'), key, mu, sd); res[f'linear_{key}'] = r
    side, frames = (4, 8) if tr['video'] else (7, 1)
    for name, conv in (('attentive', None), ('conv_attentive', lambda: GridConv(d, side, frames))):
        r = fit(lambda: Attentive(d, classes, conv() if conv else None), tr['grid'], tr['y'], va['grid'], va['y'],
                lrs=(1e-3, 3e-4), epochs=epochs)
        heads[name] = (r.pop('head'), 'grid', None, None); res[name] = r
    return res, heads


@torch.no_grad()
def score(heads, feats, covered):
    """Top-1 of fitted IN1K probes on another IN1K-labelled set, logits restricted to `covered`."""
    cols = torch.as_tensor(covered, dtype=torch.long).cuda()
    out = {}
    for name, (head, key, mu, sd) in heads.items():
        x = feats[key] if mu is None else (feats[key] - mu) / sd
        correct = 0
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(x), 2048):
                logits = head(x[i:i + 2048].cuda().float()).float()[:, cols]
                correct += int((cols[logits.argmax(-1)].cpu() == feats['y'][i:i + 2048]).sum())
        out[name] = correct / len(x)
    return out


VARIANTS = ('in1k_v2', 'in1k_sketch', 'in1k_r', 'in1k_a')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--family', default='levjepa', choices=list(LOADERS))
    p.add_argument('--run', required=True); p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--batch', type=int, default=256); p.add_argument('--workers', type=int, default=20)
    p.add_argument('--probe-epochs', type=int, default=20)
    p.add_argument('--limit', type=int, default=None, help='smoke test: items per cache')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--full', action='store_true', default=True,
                   help='kept for old callers: probes are always fitted on the complete training sets')
    p.add_argument('--datasets', nargs='*', default=None, help='subset of k400 in1k ssv2 (default: all with caches)')
    a = p.parse_args()
    a.full = True                      # full datasets only: the per-class subset protocol is retired
    a.out.mkdir(parents=True, exist_ok=True)
    enc, info = LOADERS[a.family](a.checkpoint)
    enc = enc.cuda()
    result = dict(run=a.run, pretrain_epoch=a.epoch, checkpoint=a.checkpoint, family=a.family, **info,
                  protocol=('cached frozen features, no augmentation, LR selected on 10k held-out train items; '
                            + ('full training sets' if a.full else 'IN1K 300/class, K400 and SSv2 150/class train subsets')),
                  scores={}, details={}, seconds={})
    sets = dict(k400=400, in1k=1000, ssv2=174)
    split = '_train_full' if a.full else '_train'
    names = a.datasets or [d for d in sets if (CACHE / f'{d}_val.json').exists() and (CACHE / f'{d}{split}.json').exists()]
    assert {'k400', 'in1k'} <= set(names) or a.datasets, names
    for ds in names:
        classes = sets[ds]
        t0 = time.time()
        tr = extract(enc, f'{ds}_train_full' if a.full else f'{ds}_train', a.batch, a.workers, a.limit)
        va = extract(enc, f'{ds}_val', a.batch, a.workers, a.limit)
        result['seconds'][f'{ds}_extract'] = time.time() - t0; t0 = time.time()
        fitted, heads = probes(tr, va, classes, a.probe_epochs)
        result['seconds'][f'{ds}_probes'] = time.time() - t0
        for k, v in fitted.items():
            result['scores'][f'{ds}/{k}'] = round(100 * v['acc'], 2); result['details'][f'{ds}/{k}'] = v
        result['counts'] = result.get('counts', {}) | {ds: [len(tr['y']), len(va['y'])]}
        print(ds, {k: round(100 * v['acc'], 2) for k, v in fitted.items()}, flush=True)
        del tr, va
        if ds == 'in1k':
            for vname in VARIANTS:
                if not (CACHE / f'{vname}.json').exists():
                    continue
                t0 = time.time()
                feats = extract(enc, vname, a.batch, a.workers, a.limit)
                accs = score(heads, feats, np.load(CACHE / f'{vname}_classes.npy'))
                for k, v in accs.items():
                    result['scores'][f'{vname}/{k}'] = round(100 * v, 2)
                result['counts'][vname] = len(feats['y'])
                result['seconds'][f'{vname}'] = time.time() - t0
                print(vname, {k: round(100 * v, 2) for k, v in accs.items()}, flush=True)
                del feats
        del heads
    tag = f'e{a.epoch:03d}' + ('_full' if a.full else '') + (f'_{info["weights"]}' if a.family == 'nlp' else '')
    if a.datasets:                     # one dataset per process (e.g. one GPU each): keep the files apart
        tag += '_' + '-'.join(a.datasets)
    result['tag'] = tag
    (a.out / f'{tag}.json').write_text(json.dumps(result, indent=2) + '\n')
    if not a.no_wandb:
        import wandb
        r = wandb.init(project='video_next_prediction', entity='jeffhernandez1995', id=f'{a.run}_eval_{tag}',
                       name=f'{a.run}/eval_{tag}', group=a.run, job_type='frozen_eval', resume='allow',
                       tags=['natural-video', 'eval', a.family, info['model']] + (['eval-full'] if a.full else []), dir='/scratch/CLUSTER_USER/wandb',
                       config={k: v for k, v in result.items() if k not in ('scores', 'details')})
        r.log({'pretrain_epoch': a.epoch, **{f'eval/{k}': v for k, v in result['scores'].items()}})
        r.summary.update(result['scores']); r.finish()
        # the same scores as one more point of this run's line over pretraining epochs
        try:
            from eval_curves import sync
            sync(a.out, a.full, info['weights'] if a.family == 'nlp' else None)
        except Exception as e:  # the JSON and the per-evaluation run are already written
            print(f'eval curve upload failed: {e!r}', flush=True)
    print(json.dumps(result['scores'], indent=2), flush=True)


if __name__ == '__main__':
    main()
