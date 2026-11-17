"""How stable are VideoFlexTok targets under imperceptible input noise: whole 64k ids vs the 6 FSQ digits
(levels 8,8,8,5,5,5), and how far a flipped digit moves. Same clips and crop as smoke_tokenize.py."""
import sys, torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
from smoke_tokenize import CKPT, STORE, VJEPAClipDataset, VideoFlexTokFromHub, center_square, deterministic

LEVELS = torch.tensor([8, 8, 8, 5, 5, 5])
BASIS = torch.cumprod(torch.cat([torch.tensor([1]), LEVELS[:-1]]), 0)


def digits(ids):
    return (ids[..., None] // BASIS.to(ids.device)) % LEVELS.to(ids.device)


model = VideoFlexTokFromHub.from_pretrained(CKPT).eval().cuda(); deterministic(model)
ds = VJEPAClipDataset(STORE, num_frames=17, frame_stride=2, clips_per_video=1, random_crop=False)
clips = torch.stack([center_square(ds[i * 997]['frame'], 128) for i in range(32)]).cuda()
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
with torch.no_grad():
    ref = torch.cat([t[0] for t in model.tokenize(clips)])                       # (160, 256)
    for name, x in [('+-1 gray level noise', clips + 2 / 255 * torch.randn_like(clips)),
                    ('1-pixel shift', torch.roll(clips, 1, dims=-1))]:
        out = torch.cat([t[0] for t in model.tokenize(x.clamp(-1, 1))])
        d0, d1 = digits(ref), digits(out)
        for lo, hi in [(0, 16), (0, 256)]:
            same_id = (ref[:, lo:hi] == out[:, lo:hi]).float().mean()
            same_digit = (d0[:, lo:hi] == d1[:, lo:hi]).float().mean(dim=(0, 1))
            within1 = ((d0[:, lo:hi] - d1[:, lo:hi]).abs() <= 1).float().mean()
            print(f'{name}, positions {lo}-{hi}: same id {same_id:.3f}; same digit per FSQ dim',
                  [round(float(v), 3) for v in same_digit], f'; digit within +-1 {within1:.3f}')
