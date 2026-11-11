"""Next-latent prediction (NLP) on natural video: model, targets and losses.

Direction A (categorical, the candidate): a block-causal student encoder, a
predictor that transforms its memory, and a normalized head trained to predict,
at every patch site i of frame t, the online-clustered code of an EMA teacher's
final-layer feature at (t+h, i) for h in {1, 2, 4}. Codes come from a linear
prototype layer on the teacher features, trained by its own clustering loss,
with Sinkhorn-Knopp targets balanced per spatial site over the global batch and
the target frames (marginals all-reduced, logits never gathered).

Direction B (regression, the control): same encoder, teacher and predictor;
squared distance between the normalized prediction and the normalized teacher
feature, with an optional coding-rate guard on the student's final layer.

The encoder is LeVJEPA's VisionTransformer (third_party/LeVJEPA/module.py)
built with the LeVJEPA arm's arguments: patch 16, 16 frames, tubelet 1, RoPE,
block-causal. Checkpoints therefore load into the same evaluators. Without
masking nothing is dropped: all 3,136 patch tokens are inputs and, from frame 2
on, targets. Frame 0 is burn-in context only (never a source).

Direction Q (categorical on random projections, 2026-11-04): same encoder,
teacher and predictor; each teacher feature gets a code of q_dirs bins, one per
fixed random direction (projection standardized by running statistics, cut at the
N(0, 1) quantiles), and a linear head predicts one q_bins-way softmax per
direction, cross-entropy averaged over directions. Nothing in the quantizer is
learned, so it cannot collapse; the bins stay balanced while the teacher's
features stay spread (the guard), see docs/nlp_lag_curve.md. target_view='frame'
encodes every target frame alone (T = 1) instead of as part of the clip.

Masked variant (mask_ratio > 0): the student encoder sees only the visible
sites of a per-clip multi-block tube mask (the same sites hidden in every
frame, V-JEPA style), the predictor fills hidden sites with a learned mask
token, and the loss is taken at hidden sites only. The teacher still sees the
whole clip. With every past frame of a hidden site hidden too, the copy
shortcut (a site's feature one frame later has cosine ~0.98 with its current
value) is no longer available.

CAPI-derived pieces (normalized head, Sinkhorn): Meta Platforms, Apache-2.0,
see third_party/capi/LICENSE.
"""
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LEVJEPA = Path(__file__).resolve().parents[2] / 'third_party' / 'LeVJEPA'
sys.path.insert(0, str(LEVJEPA))
import module as vit  # noqa: E402

GRID = 14
SITES = GRID * GRID
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZES = ('vit_tiny', 'vit_small', 'vit_base', 'vit_large')


def config(size, arm='A', **over):
    assert size in SIZES and arm in ('A', 'B', 'Q')
    c = dict(
        size=size, arm=arm,
        # data: identical to the LeVJEPA arm's clean global view
        img_size=224, patch_size=16, num_frames=16, frame_stride=2, crop_scale=[0.8, 1.0],
        # objective
        horizons=[1, 2, 4], aux_weight=0.15, burnin=1,
        predictor_trunk=4, predictor_branch=2,
        vocab=16384, target_temp=0.06, student_temp=0.12, cluster_temp=0.12,
        sinkhorn_iters=3, balance='site',
        gamma=0.0, coding_eps=0.05, coding_sample=4096,
        # anti-collapse guard weighted by gamma: 'coding_rate' (subtracted) or 'sigreg'
        # (LeJEPA's isotropic-Gaussian test, added) on the same sampled raw patch features
        guard='coding_rate', sigreg_proj=1024, sigreg_knots=17,
        # masking (0 = off): multi-block tube masks, loss on hidden sites only
        mask_ratio=0.0, mask_block_scale=[0.15, 0.4], mask_block_aspect=[0.75, 1.5],
        # Q: q_dirs fixed random directions, each projection standardized by running statistics
        # and cut at the N(0, 1) quantiles into q_bins bins; one q_bins-way softmax per direction
        q_dirs=32, q_bins=8, q_momentum=0.99, q_seed=0,
        # teacher targets: 'clip' (the 16-frame block-causal clip) or 'frame' (each frame alone, T = 1)
        target_view='clip',
        # teacher
        ema_start=0.996,
        # optimization
        epochs=240, global_batch=256, lr=1e-3, min_lr_ratio=0.01, weight_decay=0.05,
        warmup_epochs=5, cluster_lr_mult=0.5, cluster_wd=0.01, clip_grad=1.0,
        ce_rows=16384,
    )
    c.update(over)
    assert c['balance'] == 'site', 'only per-site balancing is implemented'
    assert c['guard'] in ('coding_rate', 'sigreg'), c['guard']
    assert c['target_view'] in ('clip', 'frame'), c['target_view']
    return c


class _FlexF:
    """Stand-in for torch.nn.functional inside LeVJEPA's module.py: identical except that
    scaled_dot_product_attention with the block-causal bool mask runs as a compiled
    flex-attention kernel that skips the masked (future-frame) blocks instead of a dense
    masked kernel. Same math: one fused softmax(QK^T/sqrt(d)) V over the same allowed pairs."""
    def __init__(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        self._create, self._flex = create_block_mask, torch.compile(flex_attention, dynamic=False)
        self._masks = {}

    def __getattr__(self, name):
        return getattr(F, name)

    def _block_mask(self, attn_mask):
        """Flex block mask for a full-grid block-causal mask, or None for any other
        mask (the masked student's visible-token sequence), which then runs dense."""
        n = attn_mask.shape[-1]
        prefix = int(bool(attn_mask[0, 0, 0, -1]))          # CLS prefix row sees everything
        key = (n, prefix, attn_mask.device)
        if key not in self._masks:
            bm = None
            if (n - prefix) % SITES == 0:
                def allowed(b, h, q, kv):
                    fq, fk = (q - prefix) // SITES, (kv - prefix) // SITES
                    patch = (q >= prefix) & (kv >= prefix) & (fk <= fq)
                    return patch | (q < prefix)
                from torch.nn.attention.flex_attention import create_mask
                if torch.equal(create_mask(allowed, 1, 1, n, n, device=attn_mask.device), attn_mask[:1, :1]):
                    bm = self._create(allowed, None, None, n, n, device=attn_mask.device)
            self._masks[key] = bm
        return self._masks[key]

    def scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0, **kw):
        bm = None
        if attn_mask is not None and not dropout_p and not kw and q.is_cuda and attn_mask.dtype == torch.bool \
                and attn_mask.shape[0] == 1:
            bm = self._block_mask(attn_mask)
        if bm is None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, **kw)
        return self._flex(q, k, v, block_mask=bm)


def enable_flex_attention():
    """Route LeVJEPA's attention (encoder, teacher and predictor blocks) through flex-attention."""
    if not isinstance(vit.F, _FlexF):
        vit.F = _FlexF()


def build_encoder(cfg):
    return getattr(vit, cfg['size'])(
        img_size=cfg['img_size'], patch_size=cfg['patch_size'], num_frames=cfg['num_frames'],
        tubelet_size=1, use_rope=True, token_drop_rate=0.0, attn_mode='block_causal')


def normalize_video(x):
    """uint8 B,T,C,H,W -> float B,C,T,H,W, ImageNet-normalized (the LeVJEPA arm's input)."""
    x = x.permute(0, 2, 1, 3, 4).float().div_(255)
    mean = x.new_tensor(MEAN).view(1, 3, 1, 1, 1)
    std = x.new_tensor(STD).view(1, 3, 1, 1, 1)
    return x.sub_(mean).div_(std)


def tube_block_masks(b, ratio, scale, aspect, rng):
    """Visible sites (b, k) of multi-block tube masks, k = round(SITES x (1 - ratio)).

    Rectangles of area scale x SITES (uniform) and aspect ratio log-uniform in aspect
    are hidden until at most k sites remain uncovered; the uncovered sites stay
    visible and the shortfall is filled with random covered sites. Contiguous
    hidden blocks mean a hidden site's spatial neighbours are mostly hidden too
    (random scatter would let it copy a visible neighbour). Every clip gets its own
    mask; k is the same for all, so the visible sequence has a fixed length.
    """
    import numpy as np
    k = max(1, int(round(SITES * (1 - ratio))))
    out = np.empty((b, k), dtype=np.int64)
    la, lb = math.log(aspect[0]), math.log(aspect[1])
    for n in range(b):
        cover = np.zeros((GRID, GRID), dtype=bool)
        while (~cover).sum() > k:
            area = SITES * rng.uniform(scale[0], scale[1])
            a = math.exp(rng.uniform(la, lb))
            h = min(GRID, max(1, int(round(math.sqrt(area * a)))))
            w = min(GRID, max(1, int(round(math.sqrt(area / a)))))
            top, left = rng.integers(0, GRID - h + 1), rng.integers(0, GRID - w + 1)
            cover[top:top + h, left:left + w] = True
        free = np.flatnonzero(~cover.ravel())
        fill = rng.permutation(np.flatnonzero(cover.ravel()))[:k - len(free)]
        out[n] = np.sort(np.concatenate([free, fill]))
    return torch.from_numpy(out)


def encode_visible(enc, x, vis):
    """LeVJEPA VisionTransformer.forward on a given keep-set: vis (B, k) visible
    sites, kept in every frame (a tube). Same patch embedding, RoPE positions and
    block-causal mask as the encoder's own token dropping, so the weights are
    shared with the full-clip pass. Returns tokens (B, 1 + T k, D), CLS first,
    frame-major, and their grid ids (B, T k)."""
    assert enc.pos_embed is None and enc.attn_mode == 'block_causal'
    frames = x.shape[2] // enc.tubelet_size
    tok = enc.patch_embed(x)
    b, _, d = tok.shape
    ids = (torch.arange(frames, device=x.device)[None, :, None] * SITES + vis[:, None, :]).reshape(b, -1)
    tok = torch.gather(tok, 1, ids.unsqueeze(-1).expand(-1, -1, d))
    tok = torch.cat((enc.cls_token.expand(b, -1, -1).to(tok.dtype), tok), dim=1)
    # every clip keeps k sites per frame, so one mask (from clip 0) serves the batch
    mask = vit.build_block_causal_mask(frames, GRID, GRID, token_ids=ids[:1], num_prefix_tokens=1, device=x.device)
    for blk in enc.blocks:
        tok = blk(tok, T=frames, H_patches=GRID, W_patches=GRID, token_ids=ids, attn_mask=mask)
    return enc.norm(tok), ids


class NormalizedHead(nn.Module):
    """CAPI L2NormLinear: normalized input, weight-normed prototypes, scale initialized to 1."""
    def __init__(self, dim, k):
        super().__init__()
        layer = nn.Linear(dim, k, bias=False)
        nn.init.trunc_normal_(layer.weight, std=0.02)
        self.last_layer = nn.utils.parametrizations.weight_norm(layer)
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)

    def forward(self, x):
        return self.last_layer(F.normalize(x, dim=-1, eps=1e-12))


class Predictor(nn.Module):
    """Block-causal self-attention over the student's final-layer patch tokens.

    A shared trunk transforms memory (the synthetic stage showed the predictor
    must, or the prediction task collapses the encoder's last layer). A
    horizon-conditioned branch (shared weights, learned horizon embedding) then
    runs once per horizon on frames [0, T-h); its output at (t, i) is the
    prediction for (t+h, i). Cost: trunk + len(horizons) x branch blocks over
    3,136 tokens at most, far cheaper than one query token per target.
    """
    def __init__(self, dim, heads, trunk, branch, horizons, init):
        super().__init__()
        def block():
            b = vit.Block(dim=dim, num_heads=heads, mlp_ratio=4.0, qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), grid_size=GRID, use_rope=True)
            b.attn.has_cls_token = False
            return b
        self.trunk = nn.ModuleList(block() for _ in range(trunk))
        self.branch = nn.ModuleList(block() for _ in range(branch))
        self.horizon = nn.Embedding(len(horizons), dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.horizons = list(horizons)
        self.apply(init)
        nn.init.trunc_normal_(self.horizon.weight, std=0.02)
        for i, b in enumerate((*self.trunk, *self.branch), 1):
            b.attn.proj.weight.data.div_(math.sqrt(2.0 * i))
            b.mlp.fc2.weight.data.div_(math.sqrt(2.0 * i))

    @staticmethod
    def _run(blocks, x, frames):
        mask = vit.build_block_causal_mask(frames, GRID, GRID, num_prefix_tokens=0, device=x.device)
        for b in blocks:
            x = b(x, T=frames, H_patches=GRID, W_patches=GRID, attn_mask=mask)
        return x

    def forward(self, memory):
        b, n, d = memory.shape
        frames = n // SITES
        x = self._run(self.trunk, memory, frames)
        out = {}
        for k, h in enumerate(self.horizons):
            f = frames - h
            y = x[:, :f * SITES] + self.horizon.weight[k]
            out[h] = self.norm(self._run(self.branch, y, f)).reshape(b, f, SITES, d)
        return out


def _ce_tail(logits, target, inv_temp):
    return -(target.float() * F.log_softmax(logits * inv_temp, dim=-1)).sum()


def _sinkhorn_start(m, top):
    return (m - top[:, None]).exp()


def _sinkhorn_step(m, col):
    m = m / (col[:, None] + 1e-8)
    return m / (m.sum(2, keepdim=True) + 1e-8)


def _sinkhorn_out(m):
    entropy = -(m * m.clamp_min(1e-30).log()).sum(-1).sum()
    return m.to(torch.bfloat16), entropy, m.argmax(-1)


_FUSED = {}


def _toolchain():
    """Inductor needs a C++ compiler; RANGE compute nodes have none on PATH outside a login
    shell, but the module tree is mounted. Point CXX at it when nothing else is found."""
    import shutil
    if shutil.which(os.environ.get('CXX', 'g++')):
        return True
    for gxx in ('/opt/apps/software/GCCcore/14.2.0/bin/g++', '/opt/apps/software/GCCcore/13.3.0/bin/g++'):
        if os.path.exists(gxx):
            os.environ['PATH'] = os.path.dirname(gxx) + os.pathsep + os.environ.get('PATH', '')
            os.environ['CXX'] = gxx
            try:
                import torch._inductor.config as ic
                ic.cpp.cxx = (None, gxx)
            except Exception:  # noqa: BLE001
                pass
            return True
    return False


# Fused element-wise CE/Sinkhorn (1.3-1.6x per step, same math). NLP_FUSE=0 disables it.
FUSE = os.environ.get('NLP_FUSE', '1') == '1' and torch.cuda.is_available() and _toolchain()


def fused(fn):
    """torch.compile'd version of an element-wise helper (same math, fewer memory passes)."""
    if not FUSE:
        return fn
    if fn not in _FUSED:
        _FUSED[fn] = torch.compile(fn, dynamic=True) if torch.cuda.is_available() else fn
    return _FUSED[fn]


def _soft_ce(module, temp, x, target):
    """Sum over rows of -<target, log_softmax(module(x)/temp)>, in fp32."""
    with torch.autocast('cuda', enabled=False):
        logits = module(x.float())
        return fused(_ce_tail)(logits, target, 1.0 / temp)


def chunked_soft_ce(module, temp, x, target, rows, xi=None, ti=None):
    """Mean soft cross-entropy over the rows of x (N, D) against target (N, K), or
    over the row pairs (x[xi], target[ti]) when indices are given (gathered per
    chunk, so the selected targets are never copied whole).

    Each chunk is checkpointed, so the (N, K) logits are never stored for backward.
    """
    total = x.new_zeros((), dtype=torch.float32)
    n = len(x) if xi is None else len(xi)
    for i in range(0, n, rows):
        if xi is None:
            xa, ta = x[i:i + rows], target[i:i + rows]
        else:
            xa, ta = x[xi[i:i + rows]], target[ti[i:i + rows]]
        total = total + checkpoint(_soft_ce, module, temp, xa, ta, use_reentrant=False)
    return total / n


def _all_reduce(t, op=dist.ReduceOp.SUM):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=op)
    return t


@torch.no_grad()
def site_targets(cluster, z, temp, iters, site_chunk=49):
    """Sinkhorn-Knopp code assignments, balanced separately at every spatial site.

    z: B, F, S, D unit-norm teacher features at the target frames (local batch).
    Each site is its own balancing problem over (global batch x target frames):
    every code gets equal total mass at every site, so codes cannot encode where
    a patch is. Column sums (sites x K) and the stabilizing column max are
    all-reduced; rows are normalized locally. Returns B, F, S, K in bf16 plus
    batch diagnostics.
    """
    b, f, s, d = z.shape
    k = cluster.out_features
    out = torch.empty(b, f, s, k, device=z.device, dtype=torch.bfloat16)
    entropy = z.new_zeros((), dtype=torch.float32)
    usage = torch.zeros(k, device=z.device, dtype=torch.float32)
    with torch.autocast('cuda', enabled=False):
        for j in range(0, s, site_chunk):
            x = z[:, :, j:j + site_chunk].permute(2, 0, 1, 3).reshape(-1, b * f, d).float()
            m = cluster(x) / temp                                   # sites, samples, K
            top = _all_reduce(m.amax(1), dist.ReduceOp.MAX)
            m = fused(_sinkhorn_start)(m, top)
            for _ in range(iters):
                col = _all_reduce(m.sum(1))
                m = fused(_sinkhorn_step)(m, col)
            q, ent, codes = fused(_sinkhorn_out)(m)
            entropy += ent
            usage += torch.bincount(codes.flatten(), minlength=k).float()
            out[:, :, j:j + site_chunk] = q.reshape(-1, b, f, k).permute(1, 2, 0, 3)
    return out, dict(target_entropy=entropy / (b * f * s), usage=usage)


def coding_rate(z, eps):
    """0.5 logdet(I + D/eps^2 Cov) of unit-normalized rows, covariance over all ranks (autograd-aware).
    Always fp32: the D/eps^2 (~1.5e5 at D=384) amplification turns bf16 rounding in the
    covariance into a non-positive-definite matrix."""
    with torch.autocast('cuda', enabled=False):
        z = F.normalize(z.float(), dim=-1)
        d = z.shape[-1]
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if distributed:
            from torch.distributed.nn.functional import all_reduce as ar
            n = _all_reduce(torch.tensor(float(len(z)), device=z.device))
            mean = ar(z.sum(0)) / n
            zc = z - mean
            cov = ar(zc.T @ zc) / n
        else:
            zc = z - z.mean(0)
            cov = zc.T @ zc / len(z)
        matrix = torch.eye(d, device=z.device) + (d / eps ** 2) * cov
        chol, info = torch.linalg.cholesky_ex(matrix)
        if int(info):
            raise FloatingPointError(f'coding-rate Cholesky failed ({int(info)})')
        return chol.diagonal().log().sum()


def sigreg(z, seed, num_proj=1024, knots=17):
    """LeJEPA's sketched isotropic Gaussian regularizer (third_party/LeVJEPA/module.py
    SIGReg): the Epps-Pulley distance between the empirical characteristic function of
    raw rows z (N, D) along num_proj random unit directions and that of N(0, 1), times
    the global sample count. Unlike the coding rate, it does not saturate when a few
    directions dominate. Two departures from upstream, both for DDP: the directions come
    from a generator seeded identically on every rank (upstream draws them per rank, so
    its all-reduced ECFs can mix different directions), and the ECF average over ranks
    is autograd-aware, so each rank's gradient is the global one, as in coding_rate."""
    with torch.autocast('cuda', enabled=False):
        z = z.float()
        g = torch.Generator(device=z.device)
        g.manual_seed(seed)
        a = torch.randn(z.shape[-1], num_proj, device=z.device, generator=g)
        a = a / a.norm(dim=0)
        t = torch.linspace(0, 3, knots, device=z.device)
        phi = torch.exp(-t.square() / 2)
        w = torch.full((knots,), 6 / (knots - 1), device=z.device)
        w[[0, -1]] = 3 / (knots - 1)
        xt = (z @ a).unsqueeze(-1) * t                                    # N, P, K
        ecf = torch.stack((xt.cos().mean(0), xt.sin().mean(0)))           # 2, P, K
        n = len(z)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            from torch.distributed.nn.functional import all_reduce as ar
            ecf = ar(ecf) / dist.get_world_size()
            n *= dist.get_world_size()
        err = (ecf[0] - phi).square() + ecf[1].square()
        return (err @ (w * phi)).mean() * n


class Student(nn.Module):
    """Everything that receives gradients; wrapped in DDP by the trainer."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        d, heads = self.encoder.embed_dim, self.encoder.num_heads
        self.predictor = Predictor(d, heads, cfg['predictor_trunk'], cfg['predictor_branch'],
                                   cfg['horizons'], self.encoder._init_weights)
        if cfg['arm'] == 'A':
            self.head = NormalizedHead(d, cfg['vocab'])
            self.cluster = nn.Linear(d, cfg['vocab'])
            nn.init.normal_(self.cluster.weight, std=1.0)
            nn.init.zeros_(self.cluster.bias)
        if cfg['arm'] == 'Q':
            m, k = cfg['q_dirs'], cfg['q_bins']
            self.head = nn.Linear(d, m * k)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)
            g = torch.Generator().manual_seed(cfg['q_seed'])             # the same directions on every rank
            self.register_buffer('q_dirs', F.normalize(torch.randn(d, m, generator=g), dim=0))
            self.register_buffer('q_edges', torch.distributions.Normal(0., 1.).icdf(torch.arange(1, k) / k))
            self.register_buffer('q_mean', torch.zeros(m))
            self.register_buffer('q_var', torch.ones(m))
            self.register_buffer('q_seen', torch.zeros((), dtype=torch.long))
        if cfg['mask_ratio'] > 0:
            import numpy as np
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            self.mask_rng = np.random.default_rng(0)            # the trainer reseeds it per rank
        import numpy as np
        self.sigreg_rng = np.random.default_rng(12345)          # same on every rank: shared directions

    @property
    def first_target(self):
        return self.cfg['burnin'] + min(self.cfg['horizons'])

    def guard_value(self, rows):
        """The guard on sampled patch rows: SIGReg (minimized) or the coding rate (maximized)."""
        cfg = self.cfg
        if cfg['guard'] == 'sigreg':
            return sigreg(rows, int(self.sigreg_rng.integers(2 ** 62)), cfg['sigreg_proj'], cfg['sigreg_knots'])
        return coding_rate(rows, cfg['coding_eps'])

    @torch.no_grad()
    def quantize(self, z):
        """Q's codes (B, T, S, q_dirs) of teacher features z (B, T, S, D). In training the running
        per-direction mean and variance (global batch, all ranks) move toward this batch's first."""
        proj = z.float() @ self.q_dirs
        if self.training:
            flat = proj.reshape(-1, proj.shape[-1])
            s = _all_reduce(torch.cat([flat.sum(0), flat.square().sum(0),
                                       flat.new_full((1,), float(len(flat)))]))
            m = s[-1]
            mean = s[:proj.shape[-1]] / m
            var = (s[proj.shape[-1]:-1] / m - mean.square()).clamp_min(1e-8)
            mom = self.cfg['q_momentum'] if self.q_seen > 0 else 0.0
            self.q_mean.lerp_(mean, 1 - mom)
            self.q_var.lerp_(var, 1 - mom)
            self.q_seen += 1
        return torch.bucketize((proj - self.q_mean) * self.q_var.rsqrt(), self.q_edges)

    def encode(self, x):
        """Encoder pass. Returns (encoder tokens with CLS, predictor memory over the full
        grid, encoder patch tokens, patch tokens per frame, hidden-site mask (B, SITES) or None)."""
        cfg = self.cfg
        if cfg['mask_ratio'] <= 0:
            tok = self.encoder(x)
            return tok, tok[:, 1:], tok[:, 1:], SITES, None
        b = x.shape[0]
        vis = tube_block_masks(b, cfg['mask_ratio'], cfg['mask_block_scale'], cfg['mask_block_aspect'],
                               self.mask_rng).to(x.device, non_blocking=True)
        tok, ids = encode_visible(self.encoder, x, vis)
        patches = tok[:, 1:]
        d = patches.shape[-1]
        memory = self.mask_token.to(patches.dtype).expand(b, x.shape[2] * SITES, d)
        memory = memory.scatter(1, ids.unsqueeze(-1).expand(-1, -1, d), patches)
        hidden = torch.ones(b, SITES, dtype=torch.bool, device=x.device).scatter_(1, vis, False)
        return tok, memory, patches, vis.shape[1], hidden

    def forward(self, x, z, targets=None):
        """x: normalized B,C,T,H,W. z: teacher features B,T,S,D (unit norm, no grad).
        targets: A only, B,T-first_target,S,K codes for frames first_target..T-1."""
        cfg = self.cfg
        tok, memory, patches, per_frame, hidden = self.encode(x)
        pred = self.predictor(memory)
        first, parts, total, wsum = self.first_target, {}, 0.0, 0.0
        if cfg['arm'] == 'Q':
            codes = self.quantize(z)                                # B, T, S, q_dirs
            with torch.no_grad():
                c = codes[:, first:].reshape(-1, codes.shape[-1])
                cnt = torch.zeros(c.shape[1], cfg['q_bins'], device=c.device).scatter_add_(
                    1, c.T, torch.ones_like(c.T, dtype=torch.float))
                prob = cnt / cnt.sum(1, keepdim=True)
                parts['bin_entropy'] = (-(prob * prob.clamp_min(1e-12).log()).sum(1) / math.log(cfg['q_bins'])).mean()
        for h, p in pred.items():
            src = p[:, cfg['burnin']:]                              # sources t in [burnin, T-h)
            lo = cfg['burnin'] + h                                  # targets t+h in [burnin+h, T)
            if cfg['arm'] == 'A' and hidden is None:
                tgt = targets[:, lo - first:]
                loss = chunked_soft_ce(self.head, cfg['student_temp'], src.reshape(-1, src.shape[-1]),
                                       tgt.reshape(-1, tgt.shape[-1]), cfg['ce_rows'])
            elif cfg['arm'] == 'A':
                # hidden sites only; rows of the contiguous targets (B, T-first, S, K) by index
                b, f = src.shape[:2]
                bi, ti, si = torch.nonzero(hidden[:, None, :].expand(b, f, SITES), as_tuple=True)
                xi = (bi * f + ti) * SITES + si
                tix = (bi * targets.shape[1] + ti + lo - first) * SITES + si
                loss = chunked_soft_ce(self.head, cfg['student_temp'], src.reshape(-1, src.shape[-1]),
                                       targets.view(-1, targets.shape[-1]), cfg['ce_rows'], xi, tix)
            elif cfg['arm'] == 'Q':
                b, f = src.shape[:2]
                sel = (hidden[:, None, :].expand(b, f, SITES) if hidden is not None
                       else torch.ones(b, f, SITES, dtype=torch.bool, device=src.device))
                logits = self.head(src[sel]).float().view(-1, cfg['q_bins'])      # rows x dirs, bins
                tgt = codes[:, lo:][sel].reshape(-1)
                loss = F.cross_entropy(logits, tgt)
                parts[f'acc{h}'] = (logits.detach().argmax(-1) == tgt).float().mean()
            else:
                err = (F.normalize(src.float(), dim=-1) - z[:, lo:].float()).square().sum(-1)   # B, f, S
                if hidden is None:
                    loss = err.mean()
                else:
                    w = hidden[:, None, :].float()
                    loss = (err * w).sum() / (w.sum() * err.shape[1])
            w = 1.0 if h == cfg['horizons'][0] else cfg['aux_weight']
            total, wsum = total + w * loss, wsum + w
            parts[f'h{h}'] = loss.detach()
        loss = total / wsum
        parts['prediction'] = loss.detach()
        if cfg['arm'] == 'A':
            zt = z[:, first:].reshape(-1, z.shape[-1])
            cl = chunked_soft_ce(self.cluster, cfg['cluster_temp'], zt,
                                 targets.reshape(-1, targets.shape[-1]), cfg['ce_rows'])
            loss = loss + cl                     # touches only the prototypes: z and targets are detached
            parts['cluster'] = cl.detach()
        if cfg['gamma'] > 0:                     # B's guard; also A' (A plus the guard)
            flat = patches[:, first * per_frame:].reshape(-1, patches.shape[-1])
            pick = torch.randperm(len(flat), device=flat.device)[:cfg['coding_sample']]
            if cfg['guard'] == 'sigreg':
                reg = self.guard_value(flat[pick])
                loss = loss + cfg['gamma'] * reg
                parts['sigreg'] = reg.detach()
                with torch.no_grad():                    # logged for comparison with the coding-rate runs
                    parts['coding_rate'] = coding_rate(flat[pick].detach(), cfg['coding_eps'])
            else:
                rate = coding_rate(flat[pick], cfg['coding_eps'])
                loss = loss - cfg['gamma'] * rate
                parts['coding_rate'] = rate.detach()
        # CLS stays in the architecture (identical to the LeVJEPA arm) but not in the
        # objective; the zero term keeps DDP's reducer from waiting on its gradient.
        loss = loss + 0.0 * tok[:, 0].float().sum()
        return loss, parts


def calibrate_gamma(student, x, z, targets=None, scale=0.1):
    """The guard weight (coding rate or SIGReg, cfg['guard']), as in wave 2: scale x
    ||d prediction / d encoder|| over ||d guard / d encoder||, both unclipped, on one batch
    of an unregularized run. Both gradients are averaged over ranks, as DDP applies them.
    The guard all-reduces its statistic with an autograd-aware all_reduce whose backward
    sums over ranks, so each rank's guard gradient is already the global one. Until
    2026-10-19 the guard gradients were summed instead, which made gamma world-size times
    too small (calibration.json files written before then: the applied guard was
    scale / world of the prediction gradient, 1/8 of the intended 0.1 on 8 GPUs).
    For A (A', A plus the guard) the loss also holds the cluster term, which reaches only
    the prototypes, not the encoder."""
    cfg = student.cfg
    assert cfg['gamma'] == 0 and (cfg['arm'] in ('B', 'Q') or targets is not None)
    params = list(student.encoder.parameters())
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, _ = student(x, z, targets)
    g_pred = torch.autograd.grad(loss, params, allow_unused=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, _, patches, per_frame, _ = student.encode(x)
    flat = patches[:, student.first_target * per_frame:].reshape(-1, patches.shape[-1])
    pick = torch.randperm(len(flat), device=flat.device)[:cfg['coding_sample']]
    rate = student.guard_value(flat[pick])
    g_rate = torch.autograd.grad(rate, params, allow_unused=True)

    def norm(grads, average):
        v = torch.cat([(g if g is not None else torch.zeros_like(p)).float().flatten()
                       for g, p in zip(grads, params)])
        _all_reduce(v)
        if average:
            v /= world
        return float(v.norm())
    dn, rn = norm(g_pred, True), norm(g_rate, True)
    assert math.isfinite(dn) and math.isfinite(rn) and rn > 0
    return dict(gamma=scale * dn / rn, guard=cfg['guard'], prediction_grad_norm=dn, coding_grad_norm=rn,
                coding_rate=float(rate), prediction_loss=float(loss), scale=scale,
                grad_average='mean over ranks (fixed 2026-10-19)')


@torch.no_grad()
def effective_rank(x, center):
    """exp(entropy) of the normalized eigenvalue spectrum of x^T x (rows of x, fp32).
    Logged for the teacher's features (centered) and A's prototypes (unit rows, uncentered):
    A's collapse was the prototypes' rank falling from 368 to 42 of 384 over 150 epochs."""
    x = x.float()
    if center:
        x = x - x.mean(0)
    ev = torch.linalg.eigvalsh(x.T @ x).clamp_min(0)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))


@torch.no_grad()
def teacher_features(teacher, x, per_frame=False):
    """Unit-norm teacher features B, T, S, D of a normalized clip x (B, C, T, H, W): the clip
    encoded as one block-causal sequence, or with per_frame every frame encoded alone (T = 1)."""
    b = x.shape[0]
    if per_frame:
        x = x.transpose(1, 2).flatten(0, 1).unsqueeze(2)          # B*T, C, 1, H, W
    with torch.autocast('cuda', dtype=torch.bfloat16):
        tok = teacher(x)[:, 1:]
    return F.normalize(tok.float(), dim=-1).reshape(b, -1, SITES, tok.shape[-1])


@torch.no_grad()
def update_teacher(teacher, encoder, momentum):
    t = [p for p in teacher.parameters()]
    s = [p.detach() for p in encoder.parameters()]
    torch._foreach_lerp_(t, s, 1.0 - momentum)


def teacher_momentum(cfg, progress):
    return 1.0 - (1.0 - cfg['ema_start']) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def learning_rate(cfg, step, steps_per_epoch):
    total = cfg['epochs'] * steps_per_epoch
    warm = cfg['warmup_epochs'] * steps_per_epoch
    peak, low = cfg['lr'], cfg['lr'] * cfg['min_lr_ratio']
    if step < warm:
        return peak * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, total - warm))
    return low + (peak - low) * 0.5 * (1.0 + math.cos(math.pi * p))


def optimizers(student, cfg):
    """Muon on transformer matrices (encoder and predictor blocks), AdamW elsewhere,
    AdamW at half the rate for the prototypes."""
    muon, decay, no_decay, cluster = [], [], [], []
    for n, p in student.named_parameters():
        if n.startswith('cluster.'):
            cluster.append(p)
        elif p.ndim == 2 and ('.blocks.' in n or '.trunk.' in n or '.branch.' in n):
            muon.append(p)
        elif p.ndim >= 2 and 'original0' not in n and n != 'mask_token':
            decay.append(p)
        else:
            no_decay.append(p)
    opts = [torch.optim.Muon(muon, lr=cfg['lr'], weight_decay=cfg['weight_decay'], momentum=0.95,
                             adjust_lr_fn='match_rms_adamw'),
            torch.optim.AdamW([dict(params=decay, weight_decay=cfg['weight_decay']),
                               dict(params=no_decay, weight_decay=0.0)], lr=cfg['lr'], betas=(0.9, 0.95))]
    if cluster:
        opts.append(torch.optim.AdamW(cluster, lr=cfg['lr'] * cfg['cluster_lr_mult'],
                                      weight_decay=cfg['cluster_wd'], betas=(0.9, 0.95)))
    return opts


def set_lr(opts, cfg, lr):
    for i, o in enumerate(opts):
        mult = cfg['cluster_lr_mult'] if i == 2 else 1.0
        for g in o.param_groups:
            g['lr'] = lr * mult
