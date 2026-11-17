"""ViT-T trained to predict fixed VideoFlexTok tokens (no teacher, no guard, no masking, no augmentation).

Targets come from scripts/vft/vft_tokenize.py: a 17-frame window at 7.5 fps (stride 2 of the 15 fps
20% K710 store) gives 5 token groups, group 0 = frame 0 and group g >= 1 = frames 4g-3..4g, each 256
coarse-to-fine ids of a 64,000-entry FSQ codebook (levels 8,8,8,5,5,5). The ViT-T sees the same 17
frames, the same fixed center square crop, at 224 px: LeVJEPA's VisionTransformer (patch 16, tubelet 1,
RoPE, block-causal), built by nlp.build_encoder so milestones load into the existing evaluators.

Context for group g is the encoder's patch tokens of group g's frames (frame 0 for g = 0, repeated to
four frames so every context has the same length); block-causal attention means they summarize frames
0..4g and nothing later. The CLS token sees the whole clip and is never used. A small causal decoder
head (self-attention over the K target positions, cross-attention to the context) predicts tokens 1..K
of the target group coarse to fine, teacher-forced on the true earlier tokens of that group. The head gets
a learned embedding of the target group's index: VideoFlexTok ids are slot-specific (a static clip gets
different ids in every group, scripts/vft/static_test.py), so the same id means different things per group.

  arm 'next'  : context g -> target g+1, g = 0..3 (the main run)
  arm 'same'  : context g -> target g, g = 1..4 (same targets as 'next', one step more context:
                distillation of the tokenizer with no forecasting)
  arm 'multi' : context g -> targets g+1..g+3 within the window, each horizon weighted equally

loss 'id': 64,000-way cross-entropy per position. loss 'digits': six cross-entropies, one per FSQ digit
(8,8,8,5,5,5), averaged. Both log id accuracy (all six digits right) and digit accuracy.

  torchrun --nproc_per_node=4 scripts/vft/train_vft.py --arm next --out /scratch/CLUSTER_USER/runs/vftN_k710_vitt
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'scripts' / 'natural'))
import nlp  # noqa: E402

sys.path.insert(0, str(nlp.LEVJEPA))
from data.loader import VJEPAClipDataset  # noqa: E402

TOKENS = Path('/scratch/CLUSTER_USER/datasets/vft_tokens')
STORE = '/scratch/CLUSTER_USER/datasets/mix256/k710_20pct/train_20pct.lance'
LEVELS = (8, 8, 8, 5, 5, 5)
BASIS = (1, 8, 64, 512, 2560, 12800)
VOCAB = 64000
GROUPS, T = 5, 17


def config(arm, **over):
    assert arm in ('next', 'same', 'multi'), arm
    c = dict(size='vit_tiny', arm='VFT', target=arm, loss='id', K=16,
             img_size=224, patch_size=16, num_frames=T, frame_stride=2, crop='center',
             head_layers=2, head_heads=4, head_mlp=4.0,
             epochs=240, global_batch=256, lr=1e-3, min_lr_ratio=0.01, weight_decay=0.05,
             warmup_epochs=5, clip_grad=1.0, cluster_lr_mult=1.0,
             tokenizer='EPFL-VILAB/videoflextok_d18_d18_k600', tokens=str(TOKENS))
    c.update(over)
    assert c['loss'] in ('id', 'digits') and 1 <= c['K'] <= 256
    return c


def pairs(arm):
    """(context group, target group, horizon index) triples for one window."""
    if arm == 'next':
        return [(g, g + 1, 1) for g in range(4)]
    if arm == 'same':
        return [(g, g, 0) for g in range(1, 5)]
    return [(g, g + h, h) for h in (1, 2, 3) for g in range(5 - h)]


def context_frames(g):
    return [0] * 4 if g == 0 else list(range(4 * g - 3, 4 * g + 1))


def digits(ids):
    out = [(ids // b) % l for b, l in zip(BASIS, LEVELS)]
    return torch.stack(out, -1)


class Block(nn.Module):
    def __init__(self, d, heads, mlp):
        super().__init__()
        self.heads = heads
        self.n1, self.n2, self.n3, self.nm = (nn.LayerNorm(d, eps=1e-6) for _ in range(4))
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.q, self.kv, self.xproj = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, int(d * mlp)), nn.Linear(int(d * mlp), d)

    def _split(self, x):
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x, mem):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        a = F.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v), is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        k, v = self.kv(self.nm(mem)).chunk(2, -1)
        a = F.scaled_dot_product_attention(self._split(self.q(self.n2(x))), self._split(k), self._split(v))
        x = x + self.xproj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.n3(x))))


class TokenHead(nn.Module):
    """Causal decoder over K target positions, cross-attending to one context's patch tokens."""
    def __init__(self, d, cfg):
        super().__init__()
        self.K, self.loss = cfg['K'], cfg['loss']
        self.digit_emb = nn.ModuleList(nn.Embedding(l, d) for l in LEVELS)
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, self.K, d))
        self.horizon = nn.Embedding(4, d)
        self.group = nn.Embedding(GROUPS, d)       # ids mean different things in every group slot
        self.blocks = nn.ModuleList(Block(d, cfg['head_heads'], cfg['head_mlp']) for _ in range(cfg['head_layers']))
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.out = nn.Linear(d, VOCAB if self.loss == 'id' else sum(LEVELS))
        for p in (self.bos, self.pos):
            nn.init.trunc_normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, mem, target, horizon, group):
        """mem (N, M, D) context tokens; target (N, K) ids; horizon, group (N,) -> logits (N, K, V)."""
        prev = sum(e(dg) for e, dg in zip(self.digit_emb, digits(target[:, :-1]).unbind(-1)))
        x = torch.cat([self.bos.expand(len(target), -1, -1), prev], 1) + self.pos + (self.horizon(horizon) + self.group(group))[:, None]
        for b in self.blocks:
            x = b(x, mem)
        return self.out(self.norm(x))


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = nlp.build_encoder(cfg)
        self.head = TokenHead(self.encoder.embed_dim, cfg)
        p = pairs(cfg['target'])
        self.register_buffer('ctx', torch.tensor([context_frames(g) for g, _, _ in p]), persistent=False)
        self.register_buffer('tgt', torch.tensor([t for _, t, _ in p]), persistent=False)
        self.register_buffer('hor', torch.tensor([h for _, _, h in p]), persistent=False)

    def forward(self, x, tokens):
        """x normalized (B, C, 17, H, W); tokens (B, 5, K) ids. Returns per-pair losses and stats."""
        b = x.shape[0]
        tok = self.encoder(x)[:, 1:]                                     # B, 17*196, D (CLS dropped)
        d = tok.shape[-1]
        frames = tok.view(b, T, nlp.SITES, d)
        P = len(self.tgt)
        mem = frames[:, self.ctx].reshape(b * P, 4 * nlp.SITES, d)       # B*P, 784, D
        target = tokens[:, self.tgt].reshape(b * P, -1)                  # B*P, K
        logits = self.head(mem, target, self.hor.repeat(b), self.tgt.repeat(b)).float()                     # B*P, K, V
        return logits, target


def token_stats(logits, target, loss_kind):
    """Per-position mean loss, id accuracy and digit accuracy, each (K,), summed over rows."""
    K = target.shape[1]
    dg = digits(target)
    if loss_kind == 'id':
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1), reduction='none').view(-1, K)
        pred = logits.argmax(-1)
        pdg = digits(pred)
        id_ok = pred == target
    else:
        parts, pdg, off = [], [], 0
        for i, l in enumerate(LEVELS):
            lg = logits[..., off:off + l]; off += l
            parts.append(F.cross_entropy(lg.reshape(-1, l), dg[..., i].reshape(-1), reduction='none').view(-1, K))
            pdg.append(lg.argmax(-1))
        ce = torch.stack(parts, -1).mean(-1)
        pdg = torch.stack(pdg, -1)
        id_ok = (pdg == dg).all(-1)
    return ce, id_ok.float(), (pdg == dg).float().mean(-1)


def load_tokens(split, K):
    files = sorted(glob.glob(str(TOKENS / split / f'{split}_*.npz')))
    assert files, f'no {split} token shards under {TOKENS}'
    out = dict(tokens=[], key=[], start_row=[])
    for f in files:
        z = np.load(f)
        out['tokens'].append(z['tokens'][:, :, :K]); out['key'].append(z['key']); out['start_row'].append(z['start_row'])
    return {k: np.concatenate(v) for k, v in out.items()}, len(files)


class Windows(torch.utils.data.Dataset):
    """17 frames of one tokenized window, the fixed center square crop at 224 px (uint8 T,C,H,W)."""
    def __init__(self, key, start_row, size):
        self.key, self.start_row, self.size = key, start_row, size
        self.ds = None

    def __len__(self):
        return len(self.key)

    def __getitem__(self, i):
        if self.ds is None:
            self.ds = VJEPAClipDataset(STORE, num_frames=16, frame_stride=2, clips_per_video=1, random_crop=False)
        p = int(self.key[i])
        rows = int(self.ds._ep_off[p]) + int(self.start_row[i]) + np.arange(T) * 2
        look = self.ds._fetch(rows)
        frames = torch.stack([self.ds._decode_frame(look(int(r))) for r in rows])
        h, w = frames.shape[-2:]
        s = min(h, w)
        y, x = (h - s) // 2, (w - s) // 2
        v = F.interpolate(frames[..., y:y + s, x:x + s].float(), size=(self.size, self.size), mode='bicubic',
                          antialias=True, align_corners=False)
        return v.round().clamp(0, 255).to(torch.uint8), i


class WindowSampler(torch.utils.data.Sampler):
    """One random tokenized window per episode per epoch (the random temporal crop), episodes in a seeded
    permutation, cut to whole global batches, sharded by rank, resumable at a batch offset."""
    def __init__(self, key, rank, world, per_rank, seed):
        order = np.argsort(key, kind='stable')
        eps, first, count = np.unique(key[order], return_index=True, return_counts=True)
        self.order, self.first, self.count = order, first, count
        self.rank, self.world, self.per_rank, self.seed = rank, world, per_rank, seed
        self.steps = len(eps) // (world * per_rank)
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch, skip=0):
        self.epoch, self.skip = epoch, skip

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 100003 + self.epoch)
        pick = self.order[self.first + (rng.random(len(self.first)) * self.count).astype(np.int64)]
        idx = pick[rng.permutation(len(pick))][:self.steps * self.world * self.per_rank]
        idx = idx.reshape(self.steps, self.world, self.per_rank)[self.skip:, self.rank].reshape(-1)
        return iter(idx.tolist())

    def __len__(self):
        return (self.steps - self.skip) * self.per_rank


@torch.no_grad()
def evaluate(model, frames, tokens, cfg, batch=32):
    """Held-out K400 val windows: per-position loss, id and digit accuracy (all ranks' shares summed)."""
    model.eval()
    K = cfg['K']
    acc = torch.zeros(3, K, device='cuda'); n = torch.zeros((), device='cuda')
    for i in range(0, len(frames), batch):
        x = nlp.normalize_video(torch.from_numpy(frames[i:i + batch]).cuda(non_blocking=True))
        t = torch.from_numpy(tokens[i:i + batch].astype(np.int64)).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, target = model(x, t)
        ce, idk, dgk = token_stats(logits, target, cfg['loss'])
        acc += torch.stack([ce.sum(0), idk.sum(0), dgk.sum(0)]); n += len(ce)
    nlp._all_reduce(acc); nlp._all_reduce(n)
    model.train()
    acc = (acc / n).tolist()
    return dict(loss=acc[0], id_acc=acc[1], digit_acc=acc[2], rows=int(n))


def summarize(prefix, per_pos):
    out = {}
    for name, v in per_pos.items():
        v = np.asarray(v)
        out[f'{prefix}{name}'] = float(v.mean())
        for k in (1, 2, 4, 8, 16):
            if k <= len(v):
                out[f'{prefix}{name}_k{k}'] = float(v[k - 1])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['next', 'same', 'multi'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--loss', choices=['id', 'digits'], default='id')
    p.add_argument('--K', type=int, default=16)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-every', type=int, default=60)
    p.add_argument('--eval-every', type=int, default=10, help='epochs between held-out token evals')
    p.add_argument('--val-windows', type=int, default=4096)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--no-wandb', action='store_true')
    a = p.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    main_rank = rank == 0
    cfg = config(a.arm, loss=a.loss, K=a.K, global_batch=a.batch)
    assert a.batch % (world * a.accum) == 0
    micro = a.batch // (world * a.accum)
    a.out.mkdir(parents=True, exist_ok=True)
    if main_rank:
        (a.out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        code = a.out / 'code'
        if not code.exists():
            code.mkdir()
            for f in [HERE / 'train_vft.py', HERE / 'vft_tokenize.py', REPO / 'scripts/natural/nlp.py']:
                shutil.copy2(f, code / f.name)

    nlp.enable_flex_attention()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = Model(cfg).cuda()
    opts = nlp.optimizers(model, cfg)
    params = [q for q in model.parameters() if q.requires_grad]

    tr, n_shards = load_tokens('train', cfg['K'])
    tokens_gpu = torch.from_numpy(tr['tokens'].astype(np.int32)).cuda()        # N, 5, K
    ds = Windows(tr['key'], tr['start_row'], cfg['img_size'])
    sampler = WindowSampler(tr['key'], rank, world, micro * a.accum, a.seed)
    steps_per_epoch = sampler.steps
    total_steps = cfg['epochs'] * steps_per_epoch
    loader = torch.utils.data.DataLoader(ds, batch_size=micro, sampler=sampler, drop_last=True,
                                         num_workers=a.workers, pin_memory=True, persistent_workers=True,
                                         prefetch_factor=4, multiprocessing_context='spawn')
    vf = TOKENS / f'val_frames224_{a.val_windows}.npy'
    val_frames = val_tokens = None
    if vf.exists():
        val_frames = np.load(vf, mmap_mode='r')
        vt = np.load(TOKENS / f'val_tokens_{a.val_windows}.npy')[:, :, :cfg['K']]
        share = np.array_split(np.arange(len(vt)), world)[rank]
        val_frames, val_tokens = val_frames[share[0]:share[-1] + 1], vt[share]
    if main_rank:
        print(f'{len(tr["key"])} train windows from {n_shards} shards, {len(np.unique(tr["key"]))} episodes; '
              f'{steps_per_epoch} steps/epoch; held-out windows: {0 if val_tokens is None else world * len(val_tokens)}',
              flush=True)

    epoch, step, skip = 0, 0, 0
    last = a.out / 'checkpoint_last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        assert ck['config'] == cfg, 'config changed since the run started'
        model.encoder.load_state_dict(ck['encoder']); model.head.load_state_dict(ck['head'])
        for o, s in zip(opts, ck['optimizers'], strict=True):
            o.load_state_dict(s)
        epoch, step, skip = ck['epoch'], ck['step'], ck['batch_in_epoch']
        if main_rank:
            print(f'resumed from {last}: epoch {epoch} step {step} batch {skip}', flush=True)
    ddp = DDP(model, device_ids=[local])

    run = None
    if main_rank and not a.no_wandb and not a.smoke_steps:
        import wandb
        run = wandb.init(project='video_next_prediction', entity='jeffhernandez1995', name=a.out.name, id=a.out.name,
                         resume='allow', group='vft_k710', job_type='pretraining', dir='/scratch/CLUSTER_USER/wandb', config=cfg)

    def save(path, full):
        obj = dict(config=cfg, encoder=model.encoder.state_dict(), head=model.head.state_dict(), epoch=epoch,
                   step=step, batch_in_epoch=batch_in_epoch)
        if full:
            obj.update(optimizers=[o.state_dict() for o in opts])
        tmp = path.with_suffix('.tmp')
        torch.save(obj, tmp); os.replace(tmp, path)

    stop_file = a.out / 'stop_requested'
    stopping, start = False, time.time()
    t_last, wait, window, meter = time.time(), 0.0, 0, {}
    batch_in_epoch = skip
    while epoch < cfg['epochs'] and not stopping:
        sampler.set_epoch(epoch, skip)
        batch_in_epoch, skip = skip, 0
        it = iter(loader)
        for _ in range(len(sampler) // (micro * a.accum)):
            lr = nlp.learning_rate(cfg, step, steps_per_epoch)
            nlp.set_lr(opts, cfg, lr)
            t0 = time.time()
            for i in range(a.accum):
                x, idx = next(it)
                if i == 0:
                    wait += time.time() - t0
                x = nlp.normalize_video(x.cuda(non_blocking=True))
                tok = tokens_gpu[idx.cuda(non_blocking=True)].long()
                ctx = ddp.no_sync() if i < a.accum - 1 else torch.enable_grad()
                with ctx:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits, target = ddp(x, tok)
                    ce, idk, dgk = token_stats(logits, target, cfg['loss'])
                    loss = ce.mean()
                    if cfg['target'] == 'multi':          # equal weight per horizon, not per pair
                        h = model.hor.repeat(len(x))
                        loss = torch.stack([ce[h == k].mean() for k in (1, 2, 3)]).mean()
                    (loss / a.accum).backward()
                for k, v in [('loss', loss), ('id_acc', idk.mean()), ('digit_acc', dgk.mean()),
                             ('id_acc_k1', idk[:, 0].mean()), ('loss_k1', ce[:, 0].mean())]:
                    meter[k] = meter.get(k, 0.0) + float(v) / a.accum
            grad = torch.nn.utils.clip_grad_norm_(params, cfg['clip_grad'])
            if not torch.isfinite(grad):
                raise FloatingPointError(f'non-finite gradient at step {step}')
            for o in opts:
                o.step(); o.zero_grad(set_to_none=True)
            step += 1; batch_in_epoch += 1; window += 1
            meter['grad_norm'] = meter.get('grad_norm', 0.0) + float(grad)
            if step % 20 == 0 or step == 1 or a.smoke_steps:
                vals = torch.tensor(list(meter.values()), device='cuda') / window
                nlp._all_reduce(vals)
                vals = (vals / world).tolist()
                dt = time.time() - t_last
                row = dict(zip(meter.keys(), vals), lr=lr, epoch=epoch + batch_in_epoch / steps_per_epoch,
                           clips_per_s=window * a.batch / dt, data_wait_frac=wait / dt,
                           peak_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
                if main_rank:
                    print(json.dumps(dict(step=step, **{k: round(v, 5) if isinstance(v, float) else v
                                                        for k, v in row.items()})), flush=True)
                    if run:
                        run.log({f'train/{k}': v for k, v in row.items()}, step=step)
                t_last, wait, window, meter = time.time(), 0.0, 0, {}
            if a.smoke_steps and step >= a.smoke_steps:
                stopping = True; break
            if step % 50 == 0:
                flag = torch.tensor(float(main_rank and stop_file.exists()), device='cuda')
                if float(nlp._all_reduce(flag)) > 0:
                    stopping = True; break
        else:
            epoch += 1; batch_in_epoch = 0
            if val_tokens is not None and (epoch % a.eval_every == 0 or epoch == 1 or epoch == cfg['epochs']):
                ev = evaluate(model, val_frames, val_tokens, cfg)
                if main_rank:
                    (a.out / 'evals').mkdir(exist_ok=True)
                    (a.out / 'evals' / f'heldout_e{epoch:03d}.json').write_text(json.dumps(dict(epoch=epoch, **ev)) + '\n')
                    s = summarize('heldout/', {k: ev[k] for k in ('loss', 'id_acc', 'digit_acc')})
                    print(json.dumps(dict(step=step, epoch=epoch, **{k: round(v, 5) for k, v in s.items()})), flush=True)
                    if run:
                        run.log(s, step=step)
        if main_rank:
            save(last, full=True)
            if not stopping and (epoch % a.save_every == 0 or epoch == cfg['epochs']):
                save(a.out / f'checkpoint_epoch{epoch:03d}.pt', full=False)
                print(f'milestone epoch {epoch} saved', flush=True)
        dist.barrier()

    if main_rank:
        if a.smoke_steps:
            (a.out / 'smoke.json').write_text(json.dumps(dict(steps=step, seconds=time.time() - start, world=world,
                                                              micro=micro, peak_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30)) + '\n')
        elif epoch >= cfg['epochs']:
            (a.out / 'training_finished.json').write_text(json.dumps(dict(epoch=epoch, step=step)) + '\n')
        if stop_file.exists():
            stop_file.unlink()
        if run:
            run.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
