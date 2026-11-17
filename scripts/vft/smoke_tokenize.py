"""Smoke test: VideoFlexTok (d18_d18_k600) on K710-20% clips as the ViT-T will see them.

17 frames at stride 2 of the 15 fps store (7.5 fps), fixed center square crop resized to 128 px,
[-1, 1]. Prints token shapes, dtype/range, determinism, and tokenize throughput for a few batch sizes.
  CUDA_VISIBLE_DEVICES=0 /scratch/CLUSTER_USER/envs/videoflextok/bin/python scripts/vft/smoke_tokenize.py
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'third_party' / 'LeVJEPA'))
from data.loader import VJEPAClipDataset  # noqa: E402
from videoflextok.wrappers import VideoFlexTokFromHub  # noqa: E402

CKPT = '/scratch/CLUSTER_USER/models/videoflextok/videoflextok_d18_d18_k600'   # local copy; nothing is fetched
STORE = '/scratch/CLUSTER_USER/datasets/mix256/k710_20pct/train_20pct.lance'


def center_square(frames, size):
    """uint8 (T, C, H, W) -> float (C, T, size, size) in [-1, 1]; one center crop for all frames."""
    t, c, h, w = frames.shape
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    v = frames[:, :, y:y + s, x:x + s].float().div(255)
    v = F.interpolate(v, size=(size, size), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)
    return v.mul(2).sub(1).permute(1, 0, 2, 3).contiguous()


def deterministic(model):
    """Posterior mean instead of a sample in the VidTok KL-VAE (its default draws a random latent)."""
    n = 0
    for m in model.modules():
        if type(m).__name__ == 'DiagonalGaussianRegularizer':
            m.sample = False; n += 1
    return n


def agree(a, b):
    a = torch.cat([t[0] for t in a]); b = torch.cat([t[0] for t in b])   # (clips*5, 256)
    return round(float((a[..., :16] == b[..., :16]).float().mean()), 3), round(float((a == b).float().mean()), 3)


def main():
    model = VideoFlexTokFromHub.from_pretrained(CKPT).eval().cuda()
    print('gaussian regularizers set to mode:', deterministic(model))
    ds = VJEPAClipDataset(STORE, num_frames=17, frame_stride=2, clips_per_video=1, random_crop=False)
    clips = torch.stack([center_square(ds[i * 997]['frame'], 128) for i in range(32)]).cuda()
    vae = next(m for m in model.modules() if type(m).__name__ == 'VidTokVAE')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad():
        ref = model.tokenize(clips[:16])
        print('fp32 repeat agreement (first16, all):', agree(ref, model.tokenize(clips[:16])))
        noisy = model.tokenize((clips[:16] + 2 / 255 * torch.randn_like(clips[:16])).clamp(-1, 1))
        print('fp32 with +-1 gray level input noise:', agree(ref, noisy))
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        print('tf32 vs fp32:', agree(ref, model.tokenize(clips[:16])))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            print('bf16 vs fp32:', agree(ref, model.tokenize(clips[:16])))
        t0 = ref[0][0]
        print('adjacent-group same id (first16, all):',
              [(round(float((t0[g, :16] == t0[g + 1, :16]).float().mean()), 2)) for g in range(4)])
        for tf32 in [False, True]:
            torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
            for bs in [8, 16]:
                x = clips[:bs]
                model.tokenize(x); torch.cuda.synchronize(); t = time.time()
                for _ in range(3): vae._encode(x)
                torch.cuda.synchronize(); tv = (time.time() - t) / 3; t = time.time()
                for _ in range(3): model.tokenize(x)
                torch.cuda.synchronize(); tt = (time.time() - t) / 3
                print(f'tf32={tf32} batch {bs}: {bs / tt:.1f} clips/s total, VAE share {tv / tt:.0%}, '
                      f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')


if __name__ == '__main__':
    main()
