"""DDP training of next-latent prediction on the K710 Lance store or a data mix (see nlp.py,
mix_data.py; --mix scripts/natural/mixes/mix2.json).

  torchrun --nproc_per_node=8 scripts/natural/train_nlp.py --size vit_base --arm A \
      --out /scratch/CLUSTER_USER/runs/nlp_k710_vitb

Same store, clip spec and epoch definition as the LeVJEPA arm: one 16-frame clip
per video per epoch at stride 2 of the 15 fps store (7.5 fps), LeVJEPA's clean
global crop (RandomResizedCrop 224, area 0.8-1.0, bicubic), no photometric
augmentation. Workers decode JPEGs and crop; normalization runs on the GPU.

Resumable at any step: checkpoint_last.pt (full state) every epoch and when
<out>/stop_requested appears (the chained Slurm script creates it on SIGUSR1).
Milestone weights (student encoder, teacher encoder, predictor, heads) go to
checkpoint_epochNNN.pt every --save-every epochs; evaluators read those.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nlp  # noqa: E402

sys.path.insert(0, str(nlp.LEVJEPA))
from data.loader import VJEPAClipDataset  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STORE = '/scratch/CLUSTER_USER/datasets/k710/train_20pct.lance'


class GlobalCrop:
    """LeVJEPA's clean global view: one RandomResizedCrop shared by all frames."""
    def __init__(self, size, scale):
        from torchvision.transforms import v2
        self.t = v2.RandomResizedCrop((size, size), scale=tuple(scale),
                                      interpolation=v2.InterpolationMode.BICUBIC, antialias=True)

    def __call__(self, sample):
        return self.t(sample['frame'])              # T,C,H,W uint8, same crop for every frame


class EpochSampler(torch.utils.data.Sampler):
    """Seeded permutation per epoch, cut to whole global batches, sharded by rank,
    and resumable at a batch offset inside the epoch."""
    def __init__(self, n, rank, world, per_rank, seed):
        self.n, self.rank, self.world, self.per_rank, self.seed = n, rank, world, per_rank, seed
        self.steps = n // (world * per_rank)
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch, skip=0):
        self.epoch, self.skip = epoch, skip

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        perm = torch.randperm(self.n, generator=g)[:self.steps * self.world * self.per_rank]
        perm = perm.view(self.steps, self.world, self.per_rank)[self.skip:, self.rank]
        return iter(perm.flatten().tolist())

    def __len__(self):
        return (self.steps - self.skip) * self.per_rank


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', required=True, choices=nlp.SIZES)
    p.add_argument('--arm', default='A', choices=['A', 'B', 'Q'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', default=STORE)
    p.add_argument('--mix', type=Path, help='mix spec json (scripts/natural/mixes/); replaces --data')
    p.add_argument('--epochs', type=int, default=240)
    p.add_argument('--batch', type=int, default=256, help='global batch (clips)')
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--workers', type=int, default=None, help='loader workers per rank')
    p.add_argument('--save-every', type=int, default=15)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--gamma-file', type=Path, help='read gamma from a calibration.json (B, or A with the guard)')
    p.add_argument('--calibrate-steps', type=int, default=0,
                   help='gamma 0: train N-1 steps, measure gamma on batch N, write calibration.json, exit')
    p.add_argument('--calibrate-scale', type=float, default=0.1,
                   help='guard gradient / prediction gradient at calibration. Runs calibrated before '
                        '2026-10-19 applied 0.1 / world: use that to match one (nlpBm_k710_vitt 0.05, nlpBm_mix3_vitt 0.0125)')
    p.add_argument('--smoke-steps', type=int, default=0, help='run N steps, save, report, exit')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--no-flex', action='store_true', help='dense-mask SDPA instead of flex-attention (same math)')
    p.add_argument('--set', nargs='*', default=[], help='config overrides key=json')
    a = p.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main_rank = rank == 0

    def _value(v):                       # JSON, or a bare string (guard=sigreg) that survives sbatch quoting
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    over = {k: _value(v) for k, v in (s.split('=', 1) for s in a.set)}
    if a.gamma_file:
        a.gamma = json.loads(a.gamma_file.read_text())['gamma']
    assert not a.calibrate_steps or a.gamma == 0, 'calibration is an unregularized run'

    cfg = nlp.config(a.size, a.arm, epochs=a.epochs, global_batch=a.batch, gamma=a.gamma, **over)
    per_rank = a.batch // (world * a.accum)
    assert per_rank * world * a.accum == a.batch, 'global batch must divide world x accum'
    micro = per_rank

    if not a.no_flex:
        nlp.enable_flex_attention()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    student = nlp.Student(cfg).cuda()
    if cfg['mask_ratio'] > 0:
        student.mask_rng = np.random.default_rng(a.seed * 1009 + rank)
    teacher = nlp.build_encoder(cfg).cuda()
    teacher.load_state_dict(student.encoder.state_dict())
    teacher.requires_grad_(False).eval()
    opts = nlp.optimizers(student, cfg)

    crop = GlobalCrop(cfg['img_size'], cfg['crop_scale'])
    mix = None
    if a.mix:
        import mix_data
        f = mix_data.load_resolved(a.out, a.mix, main_rank)
        dist.barrier()
        resolved = json.loads(f.read_text())
        ds = mix_data.MixDataset(resolved, cfg['num_frames'], crop)
        sampler = mix_data.MixEpochSampler(ds.sizes, ds.weights, resolved['epoch_clips'], rank, world,
                                           per_rank * a.accum, a.seed)
        mix = dict(name=resolved['name'], sources=ds.describe())
        if main_rank:
            print('mix', json.dumps(mix), flush=True)
    else:
        ds = VJEPAClipDataset(a.data, num_frames=cfg['num_frames'], frame_stride=cfg['frame_stride'],
                              clips_per_video=1, random_crop=True, transform=crop)
        sampler = EpochSampler(len(ds), rank, world, per_rank * a.accum, a.seed)
    steps_per_epoch = sampler.steps
    total_steps = cfg['epochs'] * steps_per_epoch
    cpus = len(os.sched_getaffinity(0))
    workers = a.workers if a.workers is not None else max(2, cpus // max(1, int(os.environ.get('LOCAL_WORLD_SIZE', world))) - 2)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=micro, sampler=sampler, drop_last=True, num_workers=workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4, multiprocessing_context='spawn')

    epoch, step, skip = 0, 0, 0
    last = a.out / 'checkpoint_last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        assert ck['config'] == cfg, 'config changed since the run started'
        student.load_state_dict(ck['student'])
        teacher.load_state_dict(ck['teacher'])
        for o, s in zip(opts, ck['optimizers'], strict=True):
            o.load_state_dict(s)
        epoch, step, skip = ck['epoch'], ck['step'], ck['batch_in_epoch']
        if main_rank:
            print(f'resumed from {last}: epoch {epoch} step {step} batch {skip}', flush=True)
        del ck
    elif main_rank:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'config.json').write_text(json.dumps(dict(config=cfg, args={k: str(v) for k, v in vars(a).items()},
                                                           world=world, per_rank=per_rank, clips=len(ds),
                                                           steps_per_epoch=steps_per_epoch, mix=mix), indent=2) + '\n')
        for f in ('scripts/natural/nlp.py', 'scripts/natural/train_nlp.py', 'scripts/natural/mix_data.py'):
            dest = a.out / 'code' / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / f, dest)
    dist.barrier()

    ddp = DDP(student, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    params = [q for q in student.parameters() if q.requires_grad]

    run = None
    if main_rank and not a.no_wandb and not a.smoke_steps:
        import wandb
        name = a.out.name
        run = wandb.init(project='video_next_prediction', entity='jeffhernandez1995', id=name, name=name,
                         resume='allow', group='nlp_mix' if mix else 'nlp_k710', job_type='pretraining', dir='/scratch/CLUSTER_USER/wandb',
                         tags=['natural-video', 'nlp', mix['name'] if mix else 'k710', a.size, f'arm-{a.arm}']
                         + (['guard'] if a.arm == 'A' and a.gamma > 0 else [])
                         + (['sigreg'] if cfg['guard'] == 'sigreg' else [])
                         + ([f"mask{cfg['mask_ratio']}"] if cfg['mask_ratio'] > 0 else [])
                         + (['frame-targets'] if cfg['target_view'] == 'frame' else []),
                         config=dict(cfg, world=world, per_rank=per_rank, clips=len(ds), steps_per_epoch=steps_per_epoch,
                                     mix=mix))

    def save(path, full):
        obj = dict(config=cfg, epoch=epoch, step=step, batch_in_epoch=batch_in_epoch,
                   encoder=student.encoder.state_dict(), teacher=teacher.state_dict(),
                   predictor=student.predictor.state_dict())
        if cfg['arm'] == 'A':
            obj.update(head=student.head.state_dict(), cluster=student.cluster.state_dict())
        if cfg['arm'] == 'Q':
            obj.update(head=student.head.state_dict(),
                       quantizer={k: getattr(student, k) for k in ('q_dirs', 'q_edges', 'q_mean', 'q_var', 'q_seen')})
        if full:
            obj.update(student=student.state_dict(), optimizers=[o.state_dict() for o in opts])
        atomic_save(obj, path)

    stop_file = a.out / 'stop_requested'
    usage = torch.zeros(cfg['vocab'], device='cuda') if cfg['arm'] == 'A' else None
    t_last, wait, window, meter = time.time(), 0.0, 0, {}
    start = time.time()
    stopping = False
    batch_in_epoch = skip
    while epoch < cfg['epochs'] and not stopping:
        sampler.set_epoch(epoch, skip)
        batch_in_epoch = skip
        skip = 0
        it = iter(loader)
        for _ in range(len(sampler) // (micro * a.accum)):
            lr = nlp.learning_rate(cfg, step, steps_per_epoch)
            nlp.set_lr(opts, cfg, lr)
            t0 = time.time()
            for i in range(a.accum):
                x = next(it)
                if i == 0:
                    wait += time.time() - t0
                x = nlp.normalize_video(x.cuda(non_blocking=True))
                z = nlp.teacher_features(teacher, x, per_frame=cfg['target_view'] == 'frame')
                if i == 0 and ((step + 1) % 20 == 0 or step == 0 or a.smoke_steps):   # a step that logs
                    flat = z.reshape(-1, z.shape[-1])
                    feature_erank = nlp.effective_rank(flat[torch.randperm(len(flat), device=flat.device)[:8192]],
                                                       center=True)
                targets = None
                if cfg['arm'] == 'A':
                    targets, diag = nlp.site_targets(student.cluster, z[:, student.first_target:],
                                                     cfg['target_temp'], cfg['sinkhorn_iters'])
                    usage += diag['usage']
                    meter['target_entropy'] = meter.get('target_entropy', 0.0) + float(diag['target_entropy']) / a.accum
                if a.calibrate_steps and step + 1 == a.calibrate_steps:
                    cal = nlp.calibrate_gamma(student, x, z, targets, scale=a.calibrate_scale)
                    if main_rank:
                        cal.update(step=step + 1, size=a.size, global_batch=a.batch, world=world)
                        (a.out / 'calibration.json').write_text(json.dumps(cal, indent=2) + '\n')
                        print('calibration', json.dumps(cal), flush=True)
                    dist.barrier()
                    dist.destroy_process_group()
                    return
                ctx = ddp.no_sync() if i < a.accum - 1 else torch.enable_grad()
                with ctx, torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, parts = ddp(x, z, targets)
                    (loss / a.accum).backward()
                for k, v in parts.items():
                    meter[k] = meter.get(k, 0.0) + float(v) / a.accum
                del x, z, targets
            grad = torch.nn.utils.clip_grad_norm_(params, cfg['clip_grad'])
            if not torch.isfinite(grad):
                raise FloatingPointError(f'non-finite gradient at step {step}')
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            momentum = nlp.teacher_momentum(cfg, (step + 1) / total_steps)
            nlp.update_teacher(teacher, student.encoder, momentum)
            step += 1
            batch_in_epoch += 1
            window += 1
            meter['grad_norm'] = meter.get('grad_norm', 0.0) + float(grad)

            if step % 20 == 0 or step == 1 or a.smoke_steps:
                vals = torch.tensor(list(meter.values()), device='cuda') / window
                nlp._all_reduce(vals)
                vals = (vals / world).tolist()
                dt = time.time() - t_last
                row = dict(zip(meter.keys(), vals), lr=lr, momentum=momentum, epoch=epoch + batch_in_epoch / steps_per_epoch,
                           clips_per_s=window * a.batch / dt, data_wait_frac=wait / dt,
                           peak_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
                row['feature_erank'] = feature_erank     # rank 0's batch; the spectrum is not all-reduced
                if usage is not None:
                    row['prototype_erank'] = nlp.effective_rank(
                        torch.nn.functional.normalize(student.cluster.weight, dim=-1), center=False)
                    u = usage.clone()
                    nlp._all_reduce(u)
                    prob = u / u.sum()
                    row['code_perplexity'] = float(torch.exp(-(prob * prob.clamp_min(1e-30).log()).sum()))
                    row['dead_code_frac'] = float((u == 0).float().mean())
                    usage.zero_()
                if main_rank:
                    print(json.dumps(dict(step=step, **{k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})), flush=True)
                    if run:
                        run.log({f'train/{k}': v for k, v in row.items()}, step=step)
                t_last, wait, window, meter = time.time(), 0.0, 0, {}

            if a.smoke_steps and step >= a.smoke_steps:
                stopping = True
                break
            if step % 50 == 0:
                flag = torch.tensor(float(main_rank and stop_file.exists()), device='cuda')
                if float(nlp._all_reduce(flag)) > 0:
                    stopping = True
                    break
        else:
            epoch += 1
            batch_in_epoch = 0
        if main_rank:
            save(last, full=True)
            if not stopping and (epoch % a.save_every == 0 or epoch == cfg['epochs']):
                save(a.out / f'checkpoint_epoch{epoch:03d}.pt', full=False)
                print(f'milestone epoch {epoch} saved', flush=True)
        dist.barrier()

    if main_rank:
        if a.smoke_steps:
            (a.out / 'smoke.json').write_text(json.dumps(dict(steps=step, seconds=time.time() - start,
                                                              world=world, per_rank=per_rank, workers=workers,
                                                              peak_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30)) + '\n')
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
