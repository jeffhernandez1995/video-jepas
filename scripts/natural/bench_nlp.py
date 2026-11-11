"""Per-section timing of one training step (single GPU): baseline (dense-mask SDPA, unfused) vs optimized
(flex-attention + compiled element-wise CE/Sinkhorn),
plus an output-equality check between the two. python scripts/natural/bench_nlp.py vit_tiny 32"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nlp  # noqa: E402

size, batch = sys.argv[1], int(sys.argv[2])
cfg = nlp.config(size, 'A')
torch.backends.cuda.matmul.allow_tf32 = True
x = nlp.normalize_video(torch.randint(0, 256, (batch, 16, 3, 224, 224), dtype=torch.uint8,
                                      generator=torch.Generator().manual_seed(0)).cuda())
results, outputs = {}, {}
FUSE = nlp.fused
for mode in ('sdpa', 'flex'):
    nlp.fused = FUSE if mode == 'flex' else (lambda fn: fn)
    if mode == 'flex':
        nlp.enable_flex_attention()
    torch.manual_seed(0)
    s = nlp.Student(cfg).cuda()
    t = nlp.build_encoder(cfg).cuda()
    t.load_state_dict(s.encoder.state_dict())
    times = {}

    def sec(name, fn):
        torch.cuda.synchronize(); t0 = time.time(); r = fn(); torch.cuda.synchronize()
        times[name] = times.get(name, 0.0) + time.time() - t0
        return r

    for it in range(6):
        if it == 1:
            times.clear()                                   # first iteration: compile and warm-up
        z = sec('teacher_fwd', lambda: nlp.teacher_features(t, x))
        tg, _ = sec('sinkhorn', lambda: nlp.site_targets(s.cluster, z[:, 2:], 0.06, 3))

        def step():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = s(x, z, tg)
            loss.backward()
        sec('student_fwd_bwd', step)

        def enc():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = s.encoder(x)
            out.float().square().mean().backward()
        sec('encoder_only_fwd_bwd', enc)
        s.zero_grad(set_to_none=True)
    outputs[mode] = z[:2].float().cpu()
    per = {k: 1000 * v / 5 for k, v in times.items()}
    step_ms = per['teacher_fwd'] + per['sinkhorn'] + per['student_fwd_bwd']
    results[mode] = dict(per, step_ms=step_ms, clips_per_s=batch / step_ms * 1000,
                         peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    torch.cuda.reset_peak_memory_stats()
    del s, t, z, tg
    torch.cuda.empty_cache()
    print(mode, {k: round(v, 1) for k, v in results[mode].items()}, flush=True)
diff = (outputs['sdpa'] - outputs['flex']).abs().max().item()
cos = torch.nn.functional.cosine_similarity(outputs['sdpa'].flatten(0, 2), outputs['flex'].flatten(0, 2), dim=-1).min().item()
print(f'{size} batch {batch}: speedup {results["sdpa"]["step_ms"] / results["flex"]["step_ms"]:.2f}x; '
      f'teacher features max|diff| {diff:.2e}, min cosine {cos:.5f}', flush=True)
