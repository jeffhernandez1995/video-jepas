"""V-JEPA frozen attentive-probe protocol, as used by LeVJEPA, on our checkpoints.

Follows facebookresearch/jepa evals/{image,video}_classification_frozen and
configs/evals/vitl16_in1k.yaml, vitl16_k400_16x8x3.yaml:
  probe      AttentiveClassifier: one learnable query, one cross-attention
             block (12 heads, MLP ratio 4, pre-norm), then Linear. No LayerNorm
             on the pooled token. init_std 0.02.
  optimizer  AdamW, weight decay on weights only, cosine from lr to 0 over
             20 epochs, no warmup, grad clip 1.0, cross-entropy, bf16 encoder.
  IN1K       global batch 1024, lr 1e-3, wd 1e-3. Train: timm AutoAugment
             'original', bicubic, random erasing 0.25. Val: resize 256, center
             crop 224. Image repeated 8x along time (LeVJEPA's 8-slot eval).
  K400       global batch 256, lr 1e-3, wd 0.01. Train: one random 8-frame clip
             at ~7.5 fps, RandomResizedCrop (0.08,1) ratio (3/4,4/3), no flip,
             RandAugment rand-m7-n4-mstd0.5-inc1 and random erasing 0.25 applied
             identically to all frames. Val: 8 evenly spaced temporal segments
             x 3 spatial crops of the short-side-256 video; the probe attends
             across the 8 segments' tokens jointly (attend_across_segments) and
             softmax is averaged over the 3 views.
             A second, linear head on mean-pooled tokens is trained in the same
             pass: that is the K400 number LeVJEPA reports.
  SSv2       as K400 (lr 1e-3, wd 0.01, RandAugment, no flips), one random clip per video for
             training; val 2 temporal segments x 3 crops attended across segments. Frames are
             taken every 4 raw frames (V-JEPA's frame_step 4 at SSv2's 12 fps), clips decoded whole.
Token-matched evaluation (LeVJEPA; the final-table protocol): every encoder gets 8 temporal
slots per segment, --frames 8 for tubelet-1 encoders and --frames 16 for tubelet-2 ones, on
every dataset; IN1K images are repeated to that many frames. For a per-frame image encoder
(DINOv2) --frames 1 on IN1K gives the same probe result as 8 identical frames at 1/8 the cost:
the attentive probe and the mean pool are unchanged by duplicated tokens.
V-JEPA's own 16x8x3 setting (configs/evals/vitl16_k400_16x8x3.yaml, vitl16_in1k.yaml) is
--frames 16 --train-segments 8: 16-frame clips (images repeated 16x), and K400 training
on 8 segments per video attended jointly, each segment augmented independently, as
their attend_across_segments loader does. Tubelet-2 encoders without a CLS token
(--family vjepa, vjepa2) are then fed their raw 16 frames and all their tokens are pooled.
Launch with torchrun (slurm/probe_vjepa.sbatch):
  torchrun --nproc_per_node=8 scripts/natural/probe_vjepa.py --dataset in1k \
      --checkpoint <ckpt> --out <dir> --run levjepa_k710_vitb
"""
import argparse
import datetime
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_checkpoint import LOADERS  # noqa: E402  (LeVJEPA EMA encoder loader)

D = Path('/scratch/CLUSTER_USER/datasets')
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
FRAMES, FPS, RES = 8, 7.5, 224


# ------------------------------------------------------------------ probe
class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.q = nn.Linear(dim, dim); self.kv = nn.Linear(dim, 2 * dim); self.proj = nn.Linear(dim, dim)

    def forward(self, q, x):
        b, n, d = x.shape
        q = self.q(q).reshape(b, -1, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(x).reshape(b, n, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(b, -1, d))


class AttentiveClassifier(nn.Module):
    """jepa AttentivePooler(depth=1, complete_block=True) + Linear."""
    def __init__(self, dim, classes, heads=12, init_std=.02):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm_q = nn.LayerNorm(dim); self.norm_x = nn.LayerNorm(dim)
        self.xattn = CrossAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.head = nn.Linear(dim, classes)
        nn.init.trunc_normal_(self.query, std=init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=init_std); nn.init.zeros_(m.bias)

    def forward(self, x):
        q = self.query.expand(len(x), -1, -1)
        q = q + self.xattn(self.norm_q(q), self.norm_x(x))
        q = q + self.mlp(self.norm2(q))
        return self.head(q[:, 0])


# --------------------------------------------------------------- datasets
class Heads(nn.Module):
    """The attentive probe and the mean-pooled linear probe, trained together in one DDP module."""
    def __init__(self, attentive, linear):
        super().__init__()
        self.attentive, self.linear = attentive, linear

    def forward(self, tok):
        return self.attentive(tok), self.linear(tok.mean(1))


def norm_tensor(img_uint8_chw):  # [3,H,W] uint8 -> normalized float
    x = img_uint8_chw.float().div_(255)
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


class ImageNet(Dataset):
    def __init__(self, split, train):
        from timm.data import create_transform
        from torchvision import transforms as T
        root = D / 'imagenet' / split
        self.classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        self.items = [(p, i) for i, c in enumerate(self.classes) for p in sorted((root / c).glob('*.JPEG'))]
        if train:
            self.tf = create_transform(input_size=RES, is_training=True, auto_augment='original',
                                       interpolation='bicubic', re_prob=.25, re_mode='pixel', re_count=1,
                                       mean=MEAN, std=STD)
        else:
            self.tf = T.Compose([T.Resize(int(RES * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
                                 T.CenterCrop(RES), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        p, y = self.items[i]
        with Image.open(p) as im:
            x = self.tf(im.convert('RGB'))
        return x, y  # [3,H,W]; repeated to T frames on the GPU in encode()
        # (returning [3,T,H,W] from the workers made the 4-rank, 24-worker,
        # prefetch-4 pipeline hold ~1 TB of pinned batches; job 272237 OOM-killed)


def decode_window(path, start_frac=None, n=None, short=None, step=None, full=False):
    """Decode n frames at ~FPS from an mp4 starting at a fraction of the clip
    (random when None); optional short-side resize. Returns list of PIL images."""
    import av
    from PIL import Image
    n = n or FRAMES
    if full:
        return decode_full(path, start_frac, n, short, step)
    with av.open(path) as c:
        st = c.streams.video[0]; st.thread_type = 'AUTO'
        fps = float(st.average_rate or st.guessed_rate or 30)
        stride = step or max(1, round(fps / FPS)); span = (n - 1) * stride + 1
        total = st.frames or int((st.duration or 0) * st.time_base * fps) or int(10 * fps)
        max_start = max(0, total - span)
        start = int((random.random() if start_frac is None else start_frac) * max_start)
        if start > 0 and st.time_base:
            c.seek(int(start / fps / st.time_base), stream=st, backward=True, any_frame=False)
        out = []; idx = -1
        for fr in c.decode(st):
            idx += 1
            t = int(round(fr.pts * st.time_base * fps)) if fr.pts is not None else idx
            if t < start or (t - start) % stride: continue
            im = fr.to_image()
            if short:
                w, h = im.size; s = short / min(w, h)
                im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
            out.append(im)
            if len(out) == n: break
    while out and len(out) < n: out.append(out[-1])
    return out


def decode_full(path, start_frac, n, short, step):
    return decode_full_segments(path, [start_frac], n, short, step)[0]


def decode_full_segments(path, start_fracs, n, short, step):
    """Decode a short clip once and completely (SSv2 webm often reports no frame count), then for
    each start fraction (None: random) take n frames every `step` raw frames; past the end, the
    last frame repeats. Returns one list of PIL frames per start, or [] when nothing decodes."""
    import av
    from PIL import Image
    with av.open(path) as c:
        st = c.streams.video[0]; st.thread_type = 'AUTO'
        frames = list(c.decode(st))
    if not frames:
        return [[] for _ in start_fracs]
    total = len(frames); span = (n - 1) * step + 1
    cache, out = {}, []
    for sf in start_fracs:
        start = int((random.random() if sf is None else sf) * max(0, total - span))
        seg = []
        for k in range(n):
            j = min(total - 1, start + k * step)
            if j not in cache:
                im = frames[j].to_image()
                if short:
                    w, h = im.size; s = short / min(w, h)
                    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
                cache[j] = im
            seg.append(cache[j])
        out.append(seg)
    return out


def k400_items(split):
    root = D / 'k400' / split
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    return [(p, i) for i, c in enumerate(classes) for p in sorted((root / c).glob('*.mp4'))]


def ssv2_items(split):
    """SSv2 train / validation from the raw release (labels.json, <split>.json), as in build_eval_caches.py."""
    raw = D / 'raw/ssv2'
    labels = json.loads((raw / 'labels.json').read_text())
    rows = json.loads((raw / f'{split}.json').read_text())
    vids = raw / 'extracted/20bn-something-something-v2'
    return [(vids / f"{r['id']}.webm", int(labels[r['template'].replace('[', '').replace(']', '')])) for r in rows]


# Per-dataset video protocol, V-JEPA's configs (vitl16_k400_16x8x3, vitl16_ssv2_16x2x3):
# test segments, raw-frame step (None: ~7.5 fps from the container rate; V-JEPA's frame_step 4
# at K400's ~30 fps), whether to decode the whole clip, and the item lists.
VIDEO = dict(
    k400=dict(segs=8, step=None, full=False, classes=400, train=lambda: k400_items('train'), val=lambda: k400_items('val')),
    ssv2=dict(segs=2, step=4, full=True, classes=174, train=lambda: ssv2_items('train'), val=lambda: ssv2_items('validation')),
)


class VideoTrain(Dataset):
    """One random clip per video, or with segments > 1 one random clip in each of that many
    equal parts of the video, each augmented independently ([S,3,T,H,W]). No flips (SSv2 labels
    are direction-sensitive)."""
    def __init__(self, dataset, segments=1, frames=8):
        cfg = VIDEO[dataset]
        self.segments, self.frames = segments, frames   # workers are spawned: FRAMES is not set there
        self.step, self.full = cfg['step'], cfg['full']
        from timm.data.auto_augment import rand_augment_transform
        from timm.data.random_erasing import RandomErasing
        self.items = cfg['train']()
        self.ra = rand_augment_transform('rand-m7-n4-mstd0.5-inc1', dict(translate_const=int(RES * .45), img_mean=(124, 116, 104)))
        self.erase = RandomErasing(probability=.25, mode='pixel', max_count=1, device='cpu')

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        # An undecodable video is replaced by a random other one rather than dropped: with
        # synchronized gradients every rank must take every step with a non-empty batch.
        for _ in range(20):
            p, y = self.items[i]
            s = self.segments
            try:
                clips = [self.augment(decode_window(str(p), start_frac=None if s == 1 else (k + random.random()) / s,
                                                   n=self.frames, step=self.step, full=self.full))
                         for k in range(s)]
            except Exception:
                clips = [None]
            if all(c is not None for c in clips):
                return (clips[0] if s == 1 else torch.stack(clips)), y
            i = random.randrange(len(self.items))
        raise RuntimeError(f'20 undecodable videos in a row, last {p}')

    def augment(self, frames):
        from PIL import Image
        from torchvision.transforms import RandomResizedCrop
        from torchvision.transforms.functional import resized_crop, to_tensor
        if not frames:
            return None
        top, left, h, w = RandomResizedCrop.get_params(frames[0], (.08, 1.), (3 / 4, 4 / 3))
        frames = [resized_crop(f, top, left, h, w, [RES, RES], antialias=True) for f in frames]
        # Same RandAugment sample for every frame: augment one tall image, then split.
        tall = Image.new('RGB', (RES, RES * len(frames)))
        for k, f in enumerate(frames): tall.paste(f, (0, k * RES))
        tall = self.ra(tall)
        x = to_tensor(tall).reshape(3, len(frames), RES, RES)
        x = (x - torch.tensor(MEAN).view(3, 1, 1, 1)) / torch.tensor(STD).view(3, 1, 1, 1)
        x = self.erase(x.transpose(0, 1)).transpose(0, 1)  # one erase box applied per frame independently by timm; acceptable
        return x.contiguous()


class VideoVal(Dataset):
    """V-JEPA's test views: K400 8 temporal segments x 3 spatial crops, SSv2 2 x 3; short side 256."""
    VIEWS = 3

    def __init__(self, dataset, frames=8):
        cfg = VIDEO[dataset]
        self.frames, self.SEGS, self.step, self.full = frames, cfg['segs'], cfg['step'], cfg['full']
        self.items = cfg['val']()

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        from torchvision.transforms.functional import to_tensor
        p, y = self.items[i]
        segs = []
        try:
            starts = [(s + .5) / self.SEGS for s in range(self.SEGS)]
            if self.full:   # one decode for all segments
                segs = decode_full_segments(str(p), starts, self.frames, 256, self.step)
            else:
                segs = [decode_window(str(p), start_frac=sf, n=self.frames, short=256) for sf in starts]
        except Exception:
            return None
        if not all(segs): return None
        w, h = segs[0][0].size
        views = []
        for v in range(self.VIEWS):  # three crops along the long side
            if w >= h:
                left, top = round(v * (w - RES) / (self.VIEWS - 1)), (h - RES) // 2
            else:
                left, top = (w - RES) // 2, round(v * (h - RES) / (self.VIEWS - 1))
            clip = torch.stack([torch.stack([to_tensor(f.crop((left, top, left + RES, top + RES))) for f in seg], 1) for seg in segs])  # [S,3,T,H,W]
            views.append(clip)
        x = torch.stack(views)  # [V,S,3,T,H,W]
        x = (x - torch.tensor(MEAN).view(1, 1, 3, 1, 1, 1)) / torch.tensor(STD).view(1, 1, 3, 1, 1, 1)
        return x.contiguous(), y


def collate_skip(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.default_collate(batch) if batch else (torch.empty(0), torch.empty(0, dtype=torch.long))


# ------------------------------------------------------------------ train
def param_groups(model, wd):
    decay, no = [], []
    for n, p in model.named_parameters():
        (no if p.ndim < 2 or n.endswith('.bias') else decay).append(p)
    return [dict(params=decay, weight_decay=wd), dict(params=no, weight_decay=0.)]


HAS_CLS = True                    # False for the CLS-free tubelet-2 encoders (vjepa, vjepa2)


@torch.no_grad()
def encode(enc, x, chunk=64):  # x [B,3,T,H,W], [B,S,3,T,H,W] or image [B,3,H,W] -> tokens [B,N,D] float
    x = x.cuda(non_blocking=True)
    segs = x.shape[1] if x.dim() == 6 else 0
    if segs:                      # segments attended jointly: [B,S,...] -> [B,S*N,D]
        x = x.flatten(0, 1)
    if x.dim() == 4:  # image: repeat to a T-frame still video on the GPU
        x = x[:, :, None].expand(-1, -1, FRAMES, -1, -1)
    out = []
    for i in range(0, len(x), chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok = enc(x[i:i + chunk])
        out.append(tok[:, 1:].float() if HAS_CLS else tok.float())  # V-JEPA pools patch tokens only
    tok = torch.cat(out)
    return tok.reshape(-1, segs * tok.shape[1], tok.shape[2]) if segs else tok


def main():
    global FRAMES, HAS_CLS
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['in1k', 'k400', 'ssv2'], required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run', required=True); p.add_argument('--family', default='levjepa')
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--workers', type=int, default=20)
    p.add_argument('--limit', type=int, default=None); p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--frames', type=int, default=FRAMES, help='clip length; V-JEPA uses 16')
    p.add_argument('--train-segments', type=int, default=1,
                   help='K400 training segments attended jointly; V-JEPA 16x8x3 uses 8')
    a = p.parse_args()
    FRAMES = a.frames
    # 2 h: the decode-bound K400 validation can finish minutes apart across ranks
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=2)); rank = dist.get_rank(); world = dist.get_world_size()
    local = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local)
    torch.manual_seed(rank); random.seed(rank); np.random.seed(rank)
    enc, info = LOADERS[a.family](a.checkpoint)
    if a.family in ('vjepa', 'vjepa2'):   # unwrap the 8-frame adapter: raw clip in, all tokens out
        enc, HAS_CLS = enc.enc, False
    elif info.get('cls') is False:        # released baselines: already raw clip in, all tokens out
        HAS_CLS = False
    enc = enc.cuda(); dim = info['embed_dim']

    if a.dataset == 'in1k':
        tr, va, classes = ImageNet('train', True), ImageNet('val', False), 1000
        gbatch, lr, wd = 1024, 1e-3, 1e-3
    else:
        tr, va = VideoTrain(a.dataset, a.train_segments, FRAMES), VideoVal(a.dataset, FRAMES)
        classes = VIDEO[a.dataset]['classes']
        gbatch, lr, wd = 256, 1e-3, 1e-2   # both V-JEPA video configs: lr 1e-3, wd 0.01, 20 epochs
    if a.limit:
        tr.items = tr.items[::max(1, len(tr.items) // a.limit)]; va.items = va.items[::max(1, len(va.items) // a.limit)]
    bs = gbatch // world
    tl = DataLoader(tr, batch_size=bs, sampler=DistributedSampler(tr, shuffle=True, drop_last=True), num_workers=a.workers,
                    pin_memory=True, persistent_workers=True, prefetch_factor=4, collate_fn=collate_skip,
                    multiprocessing_context='spawn')
    vbs = 32 if a.dataset == 'in1k' else 1
    vl = DataLoader(va, batch_size=vbs, sampler=DistributedSampler(va, shuffle=False), num_workers=a.workers,
                    pin_memory=True, collate_fn=collate_skip, multiprocessing_context='spawn')

    probe = AttentiveClassifier(dim, classes, heads=dim // 64).cuda()   # the encoder's head count, as in V-JEPA (12 at B, 16 at L)
    linear = nn.Linear(dim, classes).cuda()  # K400 mean-pool linear probe (LeVJEPA's reported metric); fitted for IN1K too
    heads = Heads(probe, linear)
    # Calls go through ddp(...): calling ddp.module directly skips DDP's gradient all-reduce,
    # which (before 2026-10-26) trained 8 independent probes, one per rank on 1/8 of the data.
    ddp = nn.parallel.DistributedDataParallel(heads, device_ids=[local])
    opt = torch.optim.AdamW(param_groups(heads, wd), lr=lr, betas=(.9, .999))
    steps = a.epochs * len(tl); step = 0; start_ep = 0; acc = None; prev_minutes = 0.
    tag = f'exact_{a.dataset}' + (f'_{FRAMES}f' if FRAMES != 8 else '') + (
        f'_{a.train_segments}seg' if a.dataset != 'in1k' and a.train_segments > 1 else '')
    # Per-epoch state, so a job that hits the wall-clock limit continues in the next one (same
    # result as one long job: the sampler reshuffles from set_epoch(ep), the schedule from step).
    resume = a.out / f'{tag}.resume.pt'
    if resume.exists():
        st = torch.load(resume, map_location='cuda', weights_only=False)
        heads.load_state_dict(st['heads']); opt.load_state_dict(st['opt'])
        start_ep, step, acc, prev_minutes = st['epoch'], st['step'], st['acc'], st['minutes']
        if rank == 0:
            print(f'resumed from {resume}: epoch {start_ep}, step {step}', flush=True)
    t0 = time.time() - prev_minutes * 60
    run = None
    if rank == 0 and not a.no_wandb:
        import wandb
        run = wandb.init(project='video_next_prediction', entity='jeffhernandez1995', id=f'{a.run}_probe_{a.dataset}',
                         name=f'{a.run}/probe_{a.dataset}', group=a.run, job_type='exact_probe', resume='allow',
                         tags=['natural-video', 'eval-exact', a.family, info['model'], a.dataset],
                         dir='/scratch/CLUSTER_USER/wandb', config=dict(protocol='vjepa_frozen_attentive', epochs=a.epochs,
                         global_batch=gbatch, lr=lr, weight_decay=wd, **info))

    for ep in range(start_ep, a.epochs):
        tl.sampler.set_epoch(ep); ddp.train(); seen = 0; loss_sum = 0.; t_ep = time.time(); s_ep = step
        for x, y in tl:
            for g in opt.param_groups: g['lr'] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
            tok = encode(enc, x); y = y.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out_a, out_l = ddp(tok)
                loss = F.cross_entropy(out_a.float(), y) + F.cross_entropy(out_l.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(heads.parameters(), 1.0); opt.step()
            step += 1; seen += len(y); loss_sum += float(loss) * len(y)
            if rank == 0 and step % 50 == 0:
                el = time.time() - t_ep
                print(f'ep {ep} step {step}/{steps} loss {loss_sum/seen:.3f} {seen*world/el:.0f} samples/s '
                      f'eta {(steps-step)*el/(step-s_ep)/60:.0f} min', flush=True)
                if run: run.log(dict(step=step, epoch=ep, train_loss=loss_sum / seen, lr=opt.param_groups[0]['lr']))
        # validation each 5 epochs and at the end
        if ep % 5 == 4 or ep == a.epochs - 1:
            acc = validate(enc, heads, vl, a.dataset, rank)
            if rank == 0:
                print(f'== epoch {ep+1}: {acc}', flush=True)
                if run: run.log(dict(epoch=ep + 1, **{f'val/{k}': v for k, v in acc.items()}))
        if rank == 0 and ep < a.epochs - 1:
            a.out.mkdir(parents=True, exist_ok=True)
            torch.save(dict(heads=heads.state_dict(), opt=opt.state_dict(), epoch=ep + 1, step=step, acc=acc,
                            minutes=(time.time() - t0) / 60), str(resume) + '.tmp')
            os.replace(str(resume) + '.tmp', resume)
    if rank == 0:
        result = dict(run=a.run, dataset=a.dataset, protocol='vjepa_frozen_attentive_probe', epochs=a.epochs,
                      frames=FRAMES, train_segments=a.train_segments if a.dataset != 'in1k' else None,
                      global_batch=gbatch, lr=lr, weight_decay=wd, scores=acc, checkpoint=a.checkpoint, **info,
                      minutes=(time.time() - t0) / 60, train_items=len(tr), val_items=len(va))
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f'{tag}.json').write_text(json.dumps(result, indent=2) + '\n')
        resume.unlink(missing_ok=True)
        if run: run.summary.update({f'{a.dataset}/{k}': v for k, v in acc.items()}); run.finish()
        print(json.dumps(result['scores']), flush=True)
    dist.barrier(); dist.destroy_process_group()


@torch.no_grad()
def validate(enc, heads, vl, dataset, rank):
    heads.eval(); correct = dict(attentive=0, linear=0); n = 0
    for x, y in vl:
        if len(x) == 0: continue
        y = y.cuda()
        if dataset == 'in1k':
            tok = encode(enc, x)
            pa, pl = heads(tok)
        else:  # x [B,V,S,3,T,H,W]: attend across segments per view, average softmax over views
            b, v, s = x.shape[:3]
            flat = encode(enc, x.flatten(0, 2))            # [B*V*S,N,D]
            tok = flat.reshape(b, v, s, flat.shape[1], flat.shape[2])
            joint = tok.flatten(2, 3)  # [B,V,S*N,D]
            pa = torch.stack([F.softmax(heads.attentive(joint[:, i]), -1) for i in range(v)]).mean(0)
            pl = torch.stack([F.softmax(heads.linear(tok[:, i].mean((1, 2))), -1) for i in range(v)]).mean(0)
        correct['attentive'] += int((pa.argmax(-1) == y).sum()); correct['linear'] += int((pl.argmax(-1) == y).sum()); n += len(y)
    t = torch.tensor([correct['attentive'], correct['linear'], n], device='cuda'); dist.all_reduce(t)
    heads.train()
    return dict(attentive=round(100 * t[0].item() / t[2].item(), 2), linear=round(100 * t[1].item() / t[2].item(), 2), n=int(t[2]))



if __name__ == '__main__':
    main()
