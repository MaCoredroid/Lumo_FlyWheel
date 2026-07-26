#!/usr/bin/env python3
"""Capture bench for the SAMPLER HALF of the S1 (=2) region at live vocab:
apply_top_k_top_p sorts on [rows, 248320] + softmax + q-consume (patched
random_sample) + div/argmax, twice (bonus [4,V] + a [84,V] constraints-style
top-k/p pass). The only region piece never captured in isolation. Run inside
a PATCHED container (patcher applied first).

Run: docker run --rm --gpus all -v $PWD:/w --entrypoint bash <image> -lc \\
  'python3 /w/scripts/fr10_phase4_patch_vllm_tree_gdn.py >/dev/null 2>&1; \\
   python3 /w/scripts/fr13_s1_sampler_half_capture_bench.py'
"""
import torch
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p, random_sample, fr13_sg_set_q, fr13_sg_fill_q)

assert torch.cuda.is_available()
dev = torch.device("cuda")
V = 248320
g = torch.Generator(device=dev); g.manual_seed(3)

bonus_logits = torch.randn(4, V, device=dev, generator=g)
tls_logits = torch.randn(84, V, device=dev, generator=g)
k_t = torch.full((4,), 20, device=dev)
p_t = torch.full((4,), 0.95, device=dev)
k84 = torch.full((84,), 20, device=dev)
p84 = torch.full((84,), 0.95, device=dev)
qs = torch.empty(4, V, device=dev, dtype=torch.float32)

def region():
    # tls constraints-style pass (the in-forward apply_sampling_constraints class)
    _ = apply_top_k_top_p(tls_logits.clone(), k84, p84)
    # bonus path: top-k/p -> softmax -> q-consume -> gumbel argmax
    lg = apply_top_k_top_p(bonus_logits.clone(), k_t, p_t)
    probs = lg.softmax(dim=-1, dtype=torch.float32)
    return random_sample(probs, {})

# warm (also compiles sort paths)
fr13_sg_fill_q(qs, None); fr13_sg_set_q(qs)
out_ref = region()
torch.cuda.synchronize()
print("eager warm OK", out_ref[:4].tolist())

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    fr13_sg_fill_q(qs, None); fr13_sg_set_q(qs)
    _ = region()
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

gph = torch.cuda.CUDAGraph()
prev = torch.cuda.current_stream()
fr13_sg_fill_q(qs, None); fr13_sg_set_q(qs)
torch.cuda.set_stream(s)
try:
    gph.capture_begin()
    out_cap = region()
    gph.capture_end()
except Exception as e:
    try: gph.capture_end()
    except Exception: pass
    try: gph.reset()
    except Exception: pass
    torch.cuda.set_stream(prev)
    torch.cuda.synchronize()
    raise SystemExit(f"CAPTURE FAILED: {type(e).__name__}: {str(e)[:150]}")
torch.cuda.set_stream(prev)
for i in range(3):
    fr13_sg_fill_q(qs, None)
    gph.replay()
torch.cuda.synchronize()
print("replayed x3, out:", out_cap[:4].tolist())
print(">>> PASS — sampler half captures clean at live vocab")
