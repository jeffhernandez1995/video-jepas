"""How much of Direction B's prediction task is solved by copying the current frame?

For a B checkpoint: the trained predictor's loss at each horizon next to two trivial
predictors measured on the same clips and the same (source, target) pairs as training:
copy (the teacher's own feature at (t, i) as the prediction for (t+h, i)) and mean (the
unit-normalized mean teacher feature). All are squared distances between unit vectors.
CPU is enough for a few clips: python diag_copy_baseline.py <run dir> [epoch] [clips]
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nlp  # noqa: E402
from train_nlp import STORE, GlobalCrop, VJEPAClipDataset  # noqa: E402


@torch.no_grad()
def main():
    run = Path(sys.argv[1])
    epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    clips = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(run / f'checkpoint_epoch{epoch:03d}.pt', map_location='cpu', weights_only=False)
    cfg = dict(ck['config'], gamma=0.0)
    student = nlp.Student(cfg).to(dev).eval()
    student.encoder.load_state_dict(ck['encoder'])
    student.predictor.load_state_dict(ck['predictor'])
    teacher = nlp.build_encoder(cfg).to(dev).eval()
    teacher.load_state_dict(ck['teacher'])
    ds = VJEPAClipDataset(STORE, num_frames=cfg['num_frames'], frame_stride=cfg['frame_stride'],
                          clips_per_video=1, random_crop=True, transform=GlobalCrop(cfg['img_size'], cfg['crop_scale']))
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:clips].tolist()
    rows = {}
    for i in idx:
        x = nlp.normalize_video(ds[i][None].to(dev))
        z = nlp.teacher_features(teacher, x)                       # 1, T, S, D unit norm
        _, parts = student(x, z)
        mean = F.normalize(z[:, cfg['burnin']:].mean((1, 2)), dim=-1)[:, None, None]
        for h in cfg['horizons']:
            src, tgt = z[:, cfg['burnin']:z.shape[1] - h], z[:, cfg['burnin'] + h:]
            vals = dict(predictor=float(parts[f'h{h}']),
                        copy=float((src - tgt).square().sum(-1).mean()),
                        mean=float((mean - tgt).square().sum(-1).mean()),
                        cos_copy=float((src * tgt).sum(-1).mean()))
            for k, v in vals.items():
                rows.setdefault((h, k), []).append(v)
    print(f'{run.name} epoch {epoch}, {clips} clips ({dev})')
    print('h   predictor   copy     mean(clip)   copy cosine   predictor/copy')
    for h in cfg['horizons']:
        m = {k: sum(rows[(h, k)]) / clips for k in ('predictor', 'copy', 'mean', 'cos_copy')}
        print(f"{h}   {m['predictor']:.4f}      {m['copy']:.4f}   {m['mean']:.4f}       {m['cos_copy']:.4f}        {m['predictor'] / m['copy']:.2f}")


if __name__ == '__main__':
    main()
