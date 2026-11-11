"""DAVIS-2017 semi-supervised video object segmentation by label propagation on frozen features
(the DINO / CRW protocol), scored with the official davis2017-evaluation J&F.

Frames stay at 480p, cropped to a multiple of the patch size. Patch features are L2-normalized;
each frame's label map is the affinity-weighted average of the labels of the first frame and of
the last 7 predicted frames (DINO's eval_video_segmentation.py: temperature 0.1, top-5 sources
per target patch, sources within a 12-patch radius), upsampled bilinearly to full resolution.

Encoders see the video the way the paper-probe tier does (token-matched, 8 temporal slots):
video encoders run over consecutive windows of 8 slots (8 frames at tubelet 1, 16 at tubelet 2;
the last window is padded with the last frame) and every frame takes the features of its slot;
the image encoder (DINOv2) runs per frame. Position embeddings that are tables for 224 px
(V-JEPA's sin-cos, VideoMAE / V2's 1-D sin-cos) are interpolated to the 480p grid. RoPE models
(V-JEPA 2, LeVJEPA, ours) get their height / width positions rescaled onto the pretraining grid
(16 for V-JEPA 2's 256 px, 14 for 224 px), as V-JEPA 2's own action-conditioned predictor does
(`h_mask *= self.grid_size / H`); temporal positions are left as they are. This is the default
(2026-10-27: DAVIS J&F V-JEPA 2 40.1 -> 47.0, LeVJEPA 47.8 -> 50.8 against extrapolating integer
positions); --no-rope-interp extrapolates instead.

  python scripts/natural/eval_davis.py --family vjepa --checkpoint <ckpt> --out <dir>
"""
import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_checkpoint as ec  # noqa: E402

DAVIS = Path('/scratch/CLUSTER_USER/datasets/davis2017/DAVIS')
EVAL_CODE = Path(__file__).resolve().parents[2] / 'third_party/davis2017-evaluation'
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SLOTS = 8


# ------------------------------------------------------------------ encoders
def _interp_table(table, slots, grid, size):
    """[1, slots*grid*grid, D] position table (224 px) -> [1, slots*h*w, D] for the (h, w) grid."""
    d = table.shape[-1]
    t = table.reshape(1, slots, grid, grid, d).permute(0, 4, 1, 2, 3)          # 1,D,T,g,g
    t = F.interpolate(t.float(), size=(slots, *size), mode='trilinear', align_corners=False)
    return t.permute(0, 2, 3, 4, 1).reshape(1, -1, d)


ROPE_GRID = dict(vjepa2=16, levjepa_hf=14, levjepa=14, nlp=14)   # pretraining patch grid per RoPE family


def interpolate_rope(enc, grid):
    """Rescale every RoPE attention's height / width positions from the input grid onto `grid`."""
    n = 0
    for m in enc.modules():
        if hasattr(m, 'separate_positions') and hasattr(m, 'grid_size'):
            orig = m.separate_positions

            def scaled(ids, H_patches=None, W_patches=None, _orig=orig):
                d, h, w = _orig(ids, H_patches, W_patches)
                if H_patches is not None and W_patches is not None:
                    h, w = h * (grid / H_patches), w * (grid / W_patches)
                return d, h, w
            m.separate_positions = scaled
            n += 1
    assert n, 'no RoPE attention found'
    return n


class Encoder:
    """Per-frame patch features [N, h, w, D] for one family; frames [N,3,H,W] normalized."""
    def __init__(self, family, checkpoint, rope_interp=True):
        enc, info = ec.LOADERS[family](checkpoint)
        self.family, self.info = family, info
        self.image = family == 'dinov2'
        self.tubelet = 2 if family in ('vjepa', 'vjepa2', 'videomae', 'videomaev2') else 1
        self.patch = 14 if self.image else 16
        self.cls = info.get('cls', True) and family not in ('vjepa', 'vjepa2')
        self.enc = (enc.enc if family in ('vjepa', 'vjepa2') else enc).cuda().eval()
        if rope_interp and family in ROPE_GRID:
            self.info['rope_interp_grid'] = ROPE_GRID[family]
            self.info['rope_layers'] = interpolate_rope(self.enc, ROPE_GRID[family])

    @torch.no_grad()
    def tokens(self, clip):                                   # clip [1,3,T,H,W] -> [1,N,D]
        f, h, w = self.family, clip.shape[-2] // self.patch, clip.shape[-1] // self.patch
        if f == 'videomae':
            m = self.enc.model
            x = m.embeddings.patch_embeddings.projection(clip).flatten(2).transpose(1, 2)
            pos = _interp_table(m.embeddings.position_embeddings, SLOTS, 14, (h, w))
            x = m.encoder(x + pos.to(x.device, x.dtype)).last_hidden_state
            return m.layernorm(x)
        if f == 'videomaev2':
            v = self.enc.vit
            x = v.patch_embed.proj(clip).flatten(2).transpose(1, 2)
            x = x + _interp_table(v.pos_embed, SLOTS, 14, (h, w)).to(x.device, x.dtype)
            for blk in v.blocks:
                x = blk(x)
            return v.fc_norm(x)
        tok = self.enc(clip)
        return tok[:, 1:] if self.cls else tok

    @torch.no_grad()
    def features(self, frames):
        n, _, H, W = frames.shape
        h, w = H // self.patch, W // self.patch
        out = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if self.image:
                for i in range(0, n, 16):
                    x = frames[i:i + 16].cuda().transpose(0, 1)[None]          # 1,3,t,H,W
                    out.append(self.enc(x).float().reshape(-1, h, w, self.info['embed_dim']))
                return torch.cat(out)
            span = SLOTS * self.tubelet
            for s in range(0, n, span):
                idx = [min(n - 1, s + k) for k in range(span)]
                clip = frames[idx].cuda().transpose(0, 1)[None]                 # 1,3,span,H,W
                tok = self.tokens(clip).float().reshape(SLOTS, h, w, -1)
                for k in range(min(span, n - s)):
                    out.append(tok[k // self.tubelet])
        return torch.stack(out)


# ------------------------------------------------------------- propagation
def neighborhood(h, w, radius):
    """[h*w (source), h*w (target)] mask of source patches within `radius` of the target."""
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    pos = torch.stack([yy.flatten(), xx.flatten()], 1).float()
    return (torch.cdist(pos, pos, p=float('inf')) <= radius).float().cuda()


def propagate(feats, first_seg, n_last=7, topk=5, temp=0.1, radius=12):
    """feats [N,h,w,D] L2-normalized; first_seg [C,h,w] one-hot -> soft labels [N,C,h,w]."""
    n, h, w, d = feats.shape
    f = F.normalize(feats.reshape(n, h * w, d), dim=-1)
    mask = neighborhood(h, w, radius)
    segs = [first_seg]
    que = queue.Queue(n_last)
    for t in range(1, n):
        ctx = [0] + [i for i, _ in list(que.queue)]
        ctx_segs = [first_seg] + [s for _, s in list(que.queue)]
        src = f[ctx]                                                         # K,hw,D
        aff = torch.exp(torch.einsum('qd,ksd->ksq', f[t], src) / temp) * mask  # K,hw(src),hw(tar)
        aff = aff.reshape(-1, h * w)
        kth = aff.topk(topk, dim=0).values[-1]
        aff = torch.where(aff >= kth, aff, torch.zeros_like(aff))
        aff = aff / aff.sum(0, keepdim=True).clamp_min(1e-12)
        lab = torch.cat([s.reshape(s.shape[0], -1) for s in ctx_segs], 1)      # C, K*hw
        seg = (lab @ aff).reshape(-1, h, w)
        if que.full():
            que.get()
        que.put((t, seg))
        segs.append(seg)
    return torch.stack(segs)


# ------------------------------------------------------------------- data
def load_video(name, patch):
    jpgs = sorted((DAVIS / 'JPEGImages/480p' / name).glob('*.jpg'))
    first = Image.open(DAVIS / 'Annotations/480p' / name / '00000.png')
    palette, lab0 = first.getpalette(), np.array(first)
    W, H = Image.open(jpgs[0]).size
    Hc, Wc = H - H % patch, W - W % patch                     # crop to a patch multiple
    frames = torch.stack([(torch.from_numpy(np.asarray(Image.open(p).convert('RGB'))[:Hc, :Wc].copy())
                           .permute(2, 0, 1).float() / 255 - MEAN) / STD for p in jpgs])
    return frames, lab0, palette, (H, W), (Hc, Wc), [p.stem for p in jpgs]


def one_hot_small(lab, n_obj, h, w):
    small = np.array(Image.fromarray(lab.astype(np.uint8)).resize((w, h), Image.NEAREST))
    return torch.from_numpy(np.eye(n_obj + 1, dtype=np.float32)[small]).permute(2, 0, 1).cuda()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True, choices=list(ec.LOADERS))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=None, help='smoke test: first N val videos')
    p.add_argument('--no-rope-interp', action='store_true', help='extrapolate RoPE positions past the pretraining grid')
    a = p.parse_args()
    enc = Encoder(a.family, a.checkpoint, not a.no_rope_interp)
    tag = 'davis2017' + ('_ropeextrap' if a.no_rope_interp and a.family in ROPE_GRID else '') + ('_smoke' if a.limit else '')
    pred_dir = a.out / f'{tag}_preds'
    videos = (DAVIS / 'ImageSets/2017/val.txt').read_text().split()[:a.limit]
    t0 = time.time()
    for v in videos:
        frames, lab0, palette, (H, W), (Hc, Wc), stems = load_video(v, enc.patch)
        h, w = Hc // enc.patch, Wc // enc.patch
        n_obj = int(lab0.max())
        feats = enc.features(frames)
        soft = propagate(feats, one_hot_small(lab0[:Hc, :Wc], n_obj, h, w))
        up = F.interpolate(soft, size=(Hc, Wc), mode='bilinear', align_corners=False)
        up = F.pad(up, (0, W - Wc, 0, H - Hc), mode='replicate')            # back to full frame
        labels = up.argmax(1).byte().cpu().numpy()
        labels[0] = lab0
        (pred_dir / v).mkdir(parents=True, exist_ok=True)
        for stem, lab in zip(stems, labels):
            im = Image.fromarray(lab); im.putpalette(palette)          # L + palette -> P
            im.save(pred_dir / v / f'{stem}.png')
        print(f'{v}: {len(stems)} frames, {n_obj} objects, grid {h}x{w}, {time.time() - t0:.0f}s', flush=True)
    sys.path.insert(0, str(EVAL_CODE))
    from davis2017.evaluation import DAVISEvaluation
    ev = DAVISEvaluation(davis_root=str(DAVIS), task='semi-supervised', gt_set='val',
                         sequences=videos if a.limit else 'all')
    m = ev.evaluate(str(pred_dir))
    J, Fm = float(np.mean(m['J']['M'])), float(np.mean(m['F']['M']))
    res = dict(family=a.family, checkpoint=a.checkpoint, **{k: v for k, v in enc.info.items()},
               protocol='DAVIS-2017 val semi-supervised label propagation (DINO: n_last 7, topk 5, temp 0.1, radius 12); '
                        f'token-matched, {SLOTS} temporal slots per window',
               videos=len(videos), J_mean=round(100 * J, 2), F_mean=round(100 * Fm, 2),
               JF_mean=round(100 * (J + Fm) / 2, 2), minutes=round((time.time() - t0) / 60, 1))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f'{tag}.json').write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: res[k] for k in ('J_mean', 'F_mean', 'JF_mean', 'videos', 'minutes')}), flush=True)


if __name__ == '__main__':
    main()
