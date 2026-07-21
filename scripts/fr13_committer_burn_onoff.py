"""FR13: measure the committer with burn ON vs OFF for per-layer / batched / graph, to answer
'is burn-off enough or do we need the graph?'. Isolated micro-bench (real committer fns, synthetic rings),
cuda-event/wall timed. Compare to native inline committer ~7ms."""
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
MAX_PATH = 8
torch.manual_seed(0)
k_rings = torch.randn(L, B, N, NKH, DK, device=dev, dtype=torch.bfloat16)
v_rings = torch.randn(L, B, N, NVH, DV, device=dev, dtype=torch.bfloat16)
a_rings = torch.randn(L, B, N, NVH, device=dev, dtype=torch.bfloat16)
b_rings = torch.rand(L, B, N, NVH, device=dev, dtype=torch.bfloat16)
A_logs = torch.randn(L, NVH, device=dev)
dt_biases = torch.randn(L, NVH, device=dev)
bank_init = [torch.randn(BANK_ROWS, NVH, DK, DV, device=dev) for _ in range(L)]
ssi = torch.stack([torch.randperm(BANK_ROWS - 1, device=dev)[:B * SPEC_COLS].reshape(B, SPEC_COLS).to(torch.int32) for _ in range(L)])
accepted_paths = torch.randint(0, N, (B, PATH_COLS), device=dev, dtype=torch.int64)
accepted_lens = torch.full((B,), ACCEPT, device=dev, dtype=torch.int64)
scale = 1.0
banks = [b.clone() for b in bank_init]


def per_layer(burn):
    for _L in range(L):
        _fr13_native_committer_replay(
            state_bank=banks[_L], spec_state_indices=ssi[_L], accepted_paths=accepted_paths,
            accepted_lens=accepted_lens, k_ring=k_rings[_L], v_ring=v_rings[_L], a_ring=a_rings[_L],
            b_ring=b_rings[_L], A_log=A_logs[_L], dt_bias=dt_biases[_L], num_spec_decodes=B,
            output_scale=scale, use_qk_l2norm_in_kernel=True, burn_node_bank=burn, spec_cols=SPEC_COLS)


def batched(burn):
    _fr13_native_committer_all_layers_batched(
        banks_list=banks, spec_state_indices=ssi, accepted_paths=accepted_paths, accepted_lens=accepted_lens,
        k_rings=k_rings, v_rings=v_rings, a_rings=a_rings, b_rings=b_rings, A_logs=A_logs, dt_biases=dt_biases,
        num_layers=L, num_spec_decodes=B, output_scale=scale, use_qk_l2norm_in_kernel=True, burn_node_bank=burn)


def graph(burn):
    _fr13_native_committer_all_layers_graph(
        banks_list=banks, spec_state_indices=ssi, accepted_paths=accepted_paths, accepted_lens=accepted_lens,
        k_rings=k_rings, v_rings=v_rings, a_rings=a_rings, b_rings=b_rings, A_logs=A_logs, dt_biases=dt_biases,
        num_layers=L, num_spec_decodes=B, output_scale=scale, use_qk_l2norm_in_kernel=True, burn_node_bank=burn,
        max_path=MAX_PATH)


def bench(fn, burn, iters=40, warmup=8):
    for _ in range(warmup):
        fn(burn)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn(burn)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


print(f"shapes: L={L} B={B} accept={ACCEPT}   (native inline committer ~7ms reference)")
print(f"{'committer':12s} {'burn-ON':>9s} {'burn-OFF':>9s} {'burn cost':>10s}")
for name, fn in [("per-layer", per_layer), ("batched", batched), ("graph", graph)]:
    on = bench(fn, True)
    off = bench(fn, False)
    print(f"{name:12s} {on:8.2f}ms {off:8.2f}ms {on-off:9.2f}ms")
