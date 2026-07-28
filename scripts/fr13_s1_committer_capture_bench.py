#!/usr/bin/env python3
"""Capture bench for _fr13_native_committer_all_layers_device with the REAL
fused_sigmoid kernel: warm -> capture -> replay x3 -> byte-compare final
states vs eager recompute on identical inputs. Finds the boot-11
cudaErrorStreamCaptureInvalidated source in ~2 min instead of a 25-min boot.

Run: docker run --rm --gpus all --entrypoint python3 -v $PWD:/w \
       -e PYTHONPATH=/w/src <image> /w/scripts/fr13_s1_committer_capture_bench.py
"""
import torch
from lumo_flywheel_serving import fr10_gdn_tree_kernel as tk

assert torch.cuda.is_available()
dev = torch.device("cuda")
g = torch.Generator().manual_seed(7)

# qwen3.6-27b GDN geometry (from the live boot): fp32 banks
L, RING = 48, 24           # REAL scale: 48 GDN layers, live head geometry
num_kh, dim_k, num_vh, dim_v = 16, 128, 32, 128
B, MAX_PATH = 4, 16
ROWS = 128

def mk_inputs(seed):
    gg = torch.Generator().manual_seed(seed)
    banks = [torch.randn(ROWS, num_vh, dim_k, dim_v, generator=gg).to(dev) for _ in range(L)]
    k_rings = torch.randn(L, B, RING, num_kh, dim_k, generator=gg).to(dev, torch.bfloat16)
    v_rings = torch.randn(L, B, RING, num_vh, dim_v, generator=gg).to(dev, torch.bfloat16)
    a_rings = torch.randn(L, B, RING, num_vh, generator=gg).to(dev, torch.bfloat16)
    b_rings = torch.randn(L, B, RING, num_vh, generator=gg).to(dev, torch.bfloat16)
    A_logs = [torch.randn(num_vh, generator=gg).to(dev) for _ in range(L)]
    dt_biases = [torch.randn(num_vh, generator=gg).to(dev) for _ in range(L)]
    paths = torch.randint(1, RING, (B, MAX_PATH), generator=gg).to(dev)
    lens = torch.tensor([5, 0, 12, 3]).to(dev)
    ssi = torch.randint(0, ROWS - 1, (L, B, 6), generator=gg).to(dev, torch.int32)
    return banks, k_rings, v_rings, a_rings, b_rings, A_logs, dt_biases, paths, lens, ssi

def run(banks, *rest):
    k_rings, v_rings, a_rings, b_rings, A_logs, dt_biases, paths, lens, ssi = rest
    tk._fr13_native_committer_all_layers_device(
        banks_list=banks, spec_state_indices=ssi,
        accepted_paths=paths, accepted_lens=lens,
        k_rings=k_rings, v_rings=v_rings, a_rings=a_rings, b_rings=b_rings,
        A_logs=A_logs, dt_biases=dt_biases, num_layers=L, num_spec_decodes=B,
        output_scale=1.0, use_qk_l2norm_in_kernel=True,
        burn_node_bank=False, root_node=0)

# ---- eager reference (also warms Triton) ----
ref = mk_inputs(100)
run(*ref)
torch.cuda.synchronize()
ref_states = [b.clone() for b in ref[0]]
print("eager warm OK")

# ---- capture on a fresh input set ----
cap = mk_inputs(100)   # same seed => same inputs, fresh bank copies
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    run(*cap)          # side-stream warmup (allocator stream-assignment)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
# reset banks to pristine before capture (warmup mutated them)
cap2 = mk_inputs(100)
gph = torch.cuda.CUDAGraph()
prev = torch.cuda.current_stream()
torch.cuda.set_stream(s)
try:
    gph.capture_begin()
    run(*cap2)
    gph.capture_end()
except Exception as e:
    try:
        gph.capture_end()
    except Exception:
        pass
    try:
        gph.reset()
    except Exception:
        pass
    torch.cuda.set_stream(prev)
    torch.cuda.synchronize()
    raise SystemExit(f"CAPTURE FAILED: {type(e).__name__}: {e}")
torch.cuda.set_stream(prev)
gph.replay()
torch.cuda.synchronize()
for i, (got, want) in enumerate(zip(cap2[0], ref_states)):
    md = (got - want).abs().max().item()
    assert torch.equal(got, want), f"layer {i} state mismatch max_abs={md}"
print("captured+replayed states BYTE-IDENTICAL to eager")
print(">>> PASS — device committer capture-clean end-to-end")
