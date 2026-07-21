"""FR13 graph-committer PORT test: exercises the REAL _fr13_native_committer_all_layers_graph against the
per-layer and batched committers (synthetic rings). Gates: byte-identity (graph == per-layer == batched) +
speed. Distinct rows per (layer) so no CTA race. Run in the vLLM container:
  docker run --rm --gpus all --entrypoint python3 -v <repo>:/workspace <img> \
      /workspace/scripts/fr13_committer_graph_port_test.py
"""
import sys, time, torch
sys.path.insert(0, "/workspace/src")
from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
    _fr13_native_committer_replay,
    _fr13_native_committer_all_layers_batched,
    _fr13_native_committer_all_layers_graph,
)

dev = "cuda"
L, B, N = 48, 4, 21
ACCEPT = 5
NKH, NVH, DK, DV = 16, 32, 128, 128
BANK_ROWS = 128
SPEC_COLS = 22
PATH_COLS = 12
MAX_PATH = 8            # >= 1 + ACCEPT
dt = torch.bfloat16
torch.manual_seed(0)

k_rings = torch.randn(L, B, N, NKH, DK, device=dev, dtype=dt)
v_rings = torch.randn(L, B, N, NVH, DV, device=dev, dtype=dt)
a_rings = torch.randn(L, B, N, NVH, device=dev, dtype=dt)
b_rings = torch.rand(L, B, N, NVH, device=dev, dtype=dt)
A_logs = torch.randn(L, NVH, device=dev, dtype=torch.float32)
dt_biases = torch.randn(L, NVH, device=dev, dtype=torch.float32)
bank_init = [torch.randn(BANK_ROWS, NVH, DK, DV, device=dev, dtype=torch.float32) for _ in range(L)]
# DISTINCT rows per layer (col0 running row + spec rows) so a single fused_sigmoid call never races
ssi = torch.stack([
    torch.randperm(BANK_ROWS - 1, device=dev)[:B * SPEC_COLS].reshape(B, SPEC_COLS).to(torch.int32)
    for _ in range(L)
])
accepted_paths = torch.randint(0, N, (B, PATH_COLS), device=dev, dtype=torch.int64)
accepted_lens = torch.full((B,), ACCEPT, device=dev, dtype=torch.int64)
scale = 1.0


def reset(banks):
    for _L in range(L):
        banks[_L].copy_(bank_init[_L])


def run_perlayer(banks):
    for _L in range(L):
        _fr13_native_committer_replay(
            state_bank=banks[_L], spec_state_indices=ssi[_L],
            accepted_paths=accepted_paths, accepted_lens=accepted_lens,
            k_ring=k_rings[_L], v_ring=v_rings[_L], a_ring=a_rings[_L], b_ring=b_rings[_L],
            A_log=A_logs[_L], dt_bias=dt_biases[_L], num_spec_decodes=B,
            output_scale=scale, use_qk_l2norm_in_kernel=True, burn_node_bank=True, spec_cols=SPEC_COLS,
        )


def run_batched(banks):
    _fr13_native_committer_all_layers_batched(
        banks_list=banks, spec_state_indices=ssi, accepted_paths=accepted_paths, accepted_lens=accepted_lens,
        k_rings=k_rings, v_rings=v_rings, a_rings=a_rings, b_rings=b_rings, A_logs=A_logs, dt_biases=dt_biases,
        num_layers=L, num_spec_decodes=B, output_scale=scale, use_qk_l2norm_in_kernel=True, burn_node_bank=True,
    )


def run_graph(banks):
    _fr13_native_committer_all_layers_graph(
        banks_list=banks, spec_state_indices=ssi, accepted_paths=accepted_paths, accepted_lens=accepted_lens,
        k_rings=k_rings, v_rings=v_rings, a_rings=a_rings, b_rings=b_rings, A_logs=A_logs, dt_biases=dt_biases,
        num_layers=L, num_spec_decodes=B, output_scale=scale, use_qk_l2norm_in_kernel=True, burn_node_bank=True,
        max_path=MAX_PATH, max_b=B,
    )


# ---- byte-identity ----
ba = [b.clone() for b in bank_init]; reset(ba); run_perlayer(ba)
bb = [b.clone() for b in bank_init]; reset(bb); run_batched(bb)
bc = [b.clone() for b in bank_init]; reset(bc); run_graph(bc)   # captures on first call
torch.cuda.synchronize()
d_ab = max((ba[_L] - bb[_L]).abs().max().item() for _L in range(L))
d_ac = max((ba[_L] - bc[_L]).abs().max().item() for _L in range(L))
print(f"byte-identity  per-layer vs batched : {d_ab:.3e}  {'IDENTICAL' if d_ab==0 else 'DIFF'}")
print(f"byte-identity  per-layer vs GRAPH   : {d_ac:.3e}  {'IDENTICAL' if d_ac==0 else 'DIFF'}  <== gate")


def bench(fn, banks, name, iters=30, warmup=5):
    for _ in range(warmup):
        fn(banks)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn(banks)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:10s} = {ms:7.2f} ms/commit")
    return ms


print("--- speed (real committer functions) ---")
pl = bench(run_perlayer, [b.clone() for b in bank_init], "per-layer")
ba2 = bench(run_batched, [b.clone() for b in bank_init], "batched")
gr = bench(run_graph, bc, "graph")   # reuse bc so the captured graph (keyed on id) is hit
print(f"  => graph {pl/gr:.2f}x vs per-layer, {ba2/gr:.2f}x vs batched  ({pl:.1f}/{ba2:.1f} -> {gr:.1f} ms)")
