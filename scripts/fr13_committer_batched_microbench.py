"""FR13 committer micro-benchmark: time the batched committer vs the per-layer loop DIRECTLY
(wall timing = host+GPU, cuda-synced), isolated from the agentic server / async / metric plumbing
that made the live measurement intractable. Same synthetic tensors both arms => the batched-vs-per-layer
delta is the host-overhead reduction (192 gathers+48 layout-recomputes -> 4 gathers+1 layout). Run in the
vLLM container: docker exec <cb3-container> python3 /workspace/scripts/fr13_committer_batched_microbench.py
"""
import os, time, torch
import sys
sys.path.insert(0, "/workspace/src")
from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
    _fr13_native_committer_replay,
    _fr13_native_committer_all_layers_batched,
)

dev = "cuda"
L, B, N = 48, 4, 21          # layers, batch(spec decodes), tree nodes
ACCEPT = 5                    # accepted path length
NKH, NVH, DK, DV = 16, 32, 128, 128
BANK_ROWS = 64
SPEC_COLS = 22
PATH_COLS = 12
dt = torch.bfloat16

torch.manual_seed(0)
k_rings = torch.randn(L, B, N, NKH, DK, device=dev, dtype=dt)
v_rings = torch.randn(L, B, N, NVH, DV, device=dev, dtype=dt)
a_rings = torch.randn(L, B, N, NVH, device=dev, dtype=dt)
b_rings = torch.rand(L, B, N, NVH, device=dev, dtype=dt)
A_logs = torch.randn(L, NVH, device=dev, dtype=torch.float32)
dt_biases = torch.randn(L, NVH, device=dev, dtype=torch.float32)
banks = [torch.zeros(BANK_ROWS, NVH, DK, DV, device=dev, dtype=torch.float32) for _ in range(L)]
# spec_state_indices [L,B,SPEC_COLS] int32; col0 = a valid running row per (L,b)
ssi = torch.randint(0, BANK_ROWS, (L, B, SPEC_COLS), device=dev, dtype=torch.int32)
accepted_paths = torch.randint(0, N, (B, PATH_COLS), device=dev, dtype=torch.int64)
accepted_lens = torch.full((B,), ACCEPT, device=dev, dtype=torch.int64)
scale = 1.0


def run_perlayer():
    for _L in range(L):
        _fr13_native_committer_replay(
            state_bank=banks[_L], spec_state_indices=ssi[_L],
            accepted_paths=accepted_paths, accepted_lens=accepted_lens,
            k_ring=k_rings[_L], v_ring=v_rings[_L], a_ring=a_rings[_L], b_ring=b_rings[_L],
            A_log=A_logs[_L], dt_bias=dt_biases[_L], num_spec_decodes=B,
            output_scale=scale, use_qk_l2norm_in_kernel=True,
            burn_node_bank=True, spec_cols=SPEC_COLS,
        )


def run_batched():
    _fr13_native_committer_all_layers_batched(
        banks_list=banks, spec_state_indices=ssi,
        accepted_paths=accepted_paths, accepted_lens=accepted_lens,
        k_rings=k_rings, v_rings=v_rings, a_rings=a_rings, b_rings=b_rings,
        A_logs=A_logs, dt_biases=dt_biases, num_layers=L, num_spec_decodes=B,
        output_scale=scale, use_qk_l2norm_in_kernel=True, burn_node_bank=True,
    )


def bench(fn, name, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:10s} = {ms:7.2f} ms/commit")
    return ms


print(f"shapes: L={L} B={B} N={N} accept={ACCEPT} heads(k/v)={NKH}/{NVH} dim(k/v)={DK}/{DV}")
try:
    pl = bench(run_perlayer, "per-layer")
    ba = bench(run_batched, "batched")
    print(f"  => batched is {pl/ba:.2f}x faster ({pl:.1f} -> {ba:.1f} ms, -{pl-ba:.1f} ms host)")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("BENCH FAILED (likely a fused_sigmoid shape constraint) -- adjust shapes:", e)
