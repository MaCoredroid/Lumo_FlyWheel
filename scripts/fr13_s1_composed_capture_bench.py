#!/usr/bin/env python3
"""COMPOSED one-capture bench: the full =2 region content in ONE graph —
tls top-k/p pass + bonus sampler (topk/softmax/random_sample+q) + TAW defer
walk + device products + device committer (real 48-layer geometry). Every
piece passes captured ALONE; if the composition reproduces the live
cudaErrorStreamCaptureInvalidated, boot-roulette becomes minute-scale local
bisection (comment out phases via PHASES env: e.g. PHASES=tls,bonus,walk).

Run (patched container):
  docker run --rm --gpus all -v $PWD:/w -v $PWD:/workspace \\
    -e PYTHONPATH=/w/src --entrypoint bash <image> -lc \\
    'python3 /workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py >/dev/null 2>&1; \\
     PHASES=all python3 /w/scripts/fr13_s1_composed_capture_bench.py'
"""
import os, sys, importlib.util
import torch

sys.path.insert(0, "/w/scripts")
spec = importlib.util.spec_from_file_location(
    "_fr13_device_multidraft_kernel", "/w/scripts/fr13_device_multidraft_kernel.py")
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)
sys.modules["_fr13_device_multidraft_kernel"] = dm
from lumo_flywheel_serving import fr10_gdn_tree_kernel as tk
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p, random_sample, fr13_sg_set_q, fr13_sg_fill_q)

assert torch.cuda.is_available()
dev = torch.device("cuda")
PHASES = os.environ.get("PHASES", "all")
def on(p): return PHASES == "all" or p in PHASES.split(",")

V, B = 248320, 4
parents = [-1, -1, 0, 0, 2, 2, 4] + list(range(4, 18))   # tail6-like 21 nodes
NC = len(parents)
TOT = NC * B
MAXSPEC = 20
ROW_CAP = MAXSPEC + 1

g = torch.Generator(device=dev); g.manual_seed(9)
# walk inputs
drafts = torch.randint(0, V, (TOT,), device=dev, generator=g)
tl = torch.randn(TOT, V, device=dev, generator=g)
sl = torch.randn(TOT, V, device=dev, generator=g)
bonus_ids = torch.randint(0, V, (B, 1), device=dev, generator=g)
ndt = torch.full((B,), NC, dtype=torch.long, device=dev)
tpi = torch.tensor(parents * B, device=dev)
# sampler inputs
bonus_logits = torch.randn(B, V, device=dev, generator=g)
k_t = torch.full((B,), 20, device=dev); p_t = torch.full((B,), 0.95, device=dev)
k84 = torch.full((TOT,), 20, device=dev); p84 = torch.full((TOT,), 0.95, device=dev)
# committer inputs (real geometry)
L, RING, num_kh, dim_k, num_vh, dim_v, ROWS = 48, 24, 16, 128, 32, 128, 128
gg = torch.Generator().manual_seed(5)
banks = [torch.randn(ROWS, num_vh, dim_k, dim_v, generator=gg).to(dev) for _ in range(L)]
k_rings = torch.randn(L, B, RING, num_kh, dim_k, generator=gg).to(dev, torch.bfloat16)
v_rings = torch.randn(L, B, RING, num_vh, dim_v, generator=gg).to(dev, torch.bfloat16)
a_rings = torch.randn(L, B, RING, num_vh, generator=gg).to(dev, torch.bfloat16)
b_rings = torch.randn(L, B, RING, num_vh, generator=gg).to(dev, torch.bfloat16)
A_logs = [torch.randn(num_vh, generator=gg).to(dev) for _ in range(L)]
dt_biases = [torch.randn(num_vh, generator=gg).to(dev) for _ in range(L)]
ssi48 = torch.randint(0, ROWS - 1, (L, B, 6), generator=gg).to(dev, torch.int32)
# statics (wrapper-style)
uni = torch.empty(B, ROW_CAP, 3, device=dev)
qs = torch.empty(B, V, device=dev, dtype=torch.float32)
otid = torch.empty(B, ROW_CAP, dtype=torch.long, device=dev)
atr = torch.empty(B, dtype=torch.long, device=dev)
dm.fr13_sg_set_topology(tpi, ndt)

def prefill():
    dm.fr13_sg_set_uniforms(uni)
    dm.fr13_sg_fill_uniforms(B, ROW_CAP, dev, None)
    fr13_sg_fill_q(qs, None); fr13_sg_set_q(qs)

def region():
    if on("tls"):
        _ = apply_top_k_top_p(tl.clone(), k84, p84)
    if on("bonus"):
        lg = apply_top_k_top_p(bonus_logits.clone(), k_t, p_t)
        probs = lg.softmax(dim=-1, dtype=torch.float32)
        _ = random_sample(probs, {})
    out = None
    if on("walk"):
        out = dm.fr13_taw_commit(
            ndt, drafts, tpi, tl, sl, bonus_ids, MAXSPEC,
            generators=None, defer_materialize=True)
        if on("products"):
            gdn_paths, _r = dm.fr13_taw_products_device(
                out[0], out[1], out[2], out[3], otid, atr)
            if on("commit"):
                lens = out[3].clamp(max=12)
                tk._fr13_native_committer_all_layers_device(
                    banks_list=banks, spec_state_indices=ssi48,
                    accepted_paths=gdn_paths, accepted_lens=lens,
                    k_rings=k_rings, v_rings=v_rings, a_rings=a_rings,
                    b_rings=b_rings, A_logs=A_logs, dt_biases=dt_biases,
                    num_layers=L, num_spec_decodes=B, output_scale=1.0,
                    use_qk_l2norm_in_kernel=True, burn_node_bank=False)
    return out

# eager warm
prefill(); _ = region(); torch.cuda.synchronize()
print(f"eager warm OK (PHASES={PHASES})", flush=True)
# side-stream warm (wrapper pattern)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
prefill()
with torch.cuda.stream(s):
    _ = region()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
# capture
gph = torch.cuda.CUDAGraph()
prev = torch.cuda.current_stream()
prefill()
torch.cuda.set_stream(s)
try:
    gph.capture_begin(capture_error_mode="thread_local")
    _ = region()
    gph.capture_end()
except Exception as e:
    try: gph.capture_end()
    except Exception: pass
    try: gph.reset()
    except Exception: pass
    torch.cuda.set_stream(prev); torch.cuda.synchronize()
    raise SystemExit(f"CAPTURE FAILED (PHASES={PHASES}): {type(e).__name__}: {str(e)[:140]}")
torch.cuda.set_stream(prev)
for _ in range(3):
    dm.fr13_sg_fill_uniforms(B, ROW_CAP, dev, None)
    fr13_sg_fill_q(qs, None)
    gph.replay()
torch.cuda.synchronize()
print(f">>> PASS — composed capture clean (PHASES={PHASES})", flush=True)
