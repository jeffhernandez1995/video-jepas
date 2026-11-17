"""Do token ids mean the same thing in every group? Tokenize a static clip (one frame repeated 17 times):
if ids were group-independent, groups 1..4 (identical content) would share ids."""
import sys, torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
from smoke_tokenize import CKPT, STORE, VJEPAClipDataset, VideoFlexTokFromHub, center_square, deterministic
model = VideoFlexTokFromHub.from_pretrained(CKPT).eval().cuda(); deterministic(model)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
ds = VJEPAClipDataset(STORE, num_frames=17, frame_stride=2, clips_per_video=1, random_crop=False)
clips = torch.stack([center_square(ds[i * 997]['frame'], 128) for i in range(8)])
static = clips[:, :, 8:9].expand(-1, -1, 17, -1, -1).contiguous()
with torch.no_grad():
    for name, x in [('static', static), ('real', clips)]:
        t = torch.cat([v[0] for v in model.tokenize(x.cuda())])          # 40, 256 -> (8 clips, 5 groups)
        t = t.view(8, 5, 256)
        same = [[round(float((t[:, i, :16] == t[:, j, :16]).float().mean()), 2) for j in range(5)] for i in range(5)]
        print(name, 'share of first-16 ids equal between groups i, j:'); [print('  ', r) for r in same]
