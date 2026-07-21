"""FR13 committer CUDA-graph micro-bench (isolated). Captures the 48-layer fused_sigmoid committer loop
into a CUDA graph using FIXED shapes (state-neutral padding, validated byte-identical), replays per commit
-> kills the ~24ms 48-launch dispatch. Gates: (1) byte-identity graph-banks == per-layer-varlen-banks;
(2) speed graph-replay vs per-layer. Run in a fresh container (no model):
  docker run --rm --gpus all --entrypoint python3 -v <repo>:/workspace <img> \
      /workspace/scripts/fr13_committer_graph_microbench.py
"""
import torch, time
from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating_delta_rule_update as sg

dev = "cuda"
L, B = 48, 4
NKH, NVH, DK, DV = 16, 32, 128, 128
ROWS = 64
dt = torch.bfloat16
MAX_PATH = 12                       # fixed per-request pad (>= 1 + max accept)
MAXT = B * MAX_PATH
real = [3, 6, 2, 5]                 # per-request accepted-path lengths
torch.manual_seed(0)

A_logs = torch.randn(L, NVH, device=dev)
dt_biases = torch.randn(L, NVH, device=dev)
K = torch.randn(L, B, MAX_PATH, NKH, DK, device=dev, dtype=dt)
V = torch.randn(L, B, MAX_PATH, NVH, DV, device=dev, dtype=dt)
Am = torch.randn(L, B, MAX_PATH, NVH, device=dev, dtype=dt)
Bm = torch.rand(L, B, MAX_PATH, NVH, device=dev, dtype=dt)
bank_init = torch.randn(L, ROWS, NVH, DK, DV, device=dev)
# DISTINCT per request (matches reality: each request owns its own running state row).
# random col0 can collide -> two segments write the same row in one fused_sigmoid call -> CTA race.
col0 = torch.randperm(ROWS, device=dev)[:B].to(torch.int32)


def run_perlayer(banks):
    """Baseline: per-layer, real (variable) lengths -- the current committer."""
    T = sum(real)
    cu = torch.tensor([0] + list(torch.tensor(real).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    # ssi cols = MAX_PATH (match the fixed loop exactly; only col0 fill is read, so extra cols are inert)
    ssi = torch.zeros(B, MAX_PATH, device=dev, dtype=torch.int32)
    for b in range(B):
        ssi[b, :] = col0[b]
    for _L in range(L):
        k = torch.cat([K[_L, b, :real[b]] for b in range(B)], 0).reshape(1, T, NKH, DK).contiguous()
        v = torch.cat([V[_L, b, :real[b]] for b in range(B)], 0).reshape(1, T, NVH, DV).contiguous()
        a = torch.cat([Am[_L, b, :real[b]] for b in range(B)], 0).reshape(1, T, NVH).contiguous()
        bb = torch.cat([Bm[_L, b, :real[b]] for b in range(B)], 0).reshape(1, T, NVH).contiguous()
        q = torch.zeros(1, T, NKH, DK, device=dev, dtype=dt)
        sg(A_log=A_logs[_L], a=a, b=bb, dt_bias=dt_biases[_L], q=q, k=k, v=v, scale=1.0,
           initial_state=banks[_L], inplace_final_state=True, cu_seqlens=cu,
           ssm_state_indices=ssi, use_qk_l2norm_in_kernel=True)


# ---- FIXED-shape buffers for the graph (persistent, graph-stable addresses) ----
kbuf = torch.zeros(L, MAXT, NKH, DK, device=dev, dtype=dt)
vbuf = torch.zeros(L, MAXT, NVH, DV, device=dev, dtype=dt)
abuf = torch.full((L, MAXT, NVH), -1e4, device=dev, dtype=dt)   # default state-neutral
bbuf = torch.zeros(L, MAXT, NVH, device=dev, dtype=dt)
qbuf = torch.zeros(1, MAXT, NKH, DK, device=dev, dtype=dt)
cu_fixed = torch.tensor([i * MAX_PATH for i in range(B + 1)], device=dev, dtype=torch.int32)
ssi_fixed = torch.zeros(B, MAX_PATH, device=dev, dtype=torch.int32)
banks_g = [bank_init[_L].clone() for _L in range(L)]


def fill_fixed():
    """Populate the fixed buffers from the current commit's data (state-neutral padding)."""
    abuf.fill_(-1e4); bbuf.zero_(); kbuf.zero_(); vbuf.zero_()
    for b in range(B):
        s = b * MAX_PATH
        for _L in range(L):
            kbuf[_L, s:s + real[b]] = K[_L, b, :real[b]]
            vbuf[_L, s:s + real[b]] = V[_L, b, :real[b]]
            abuf[_L, s:s + real[b]] = Am[_L, b, :real[b]]
            bbuf[_L, s:s + real[b]] = Bm[_L, b, :real[b]]
        ssi_fixed[b, :] = col0[b]


def committer_fixed_loop():
    for _L in range(L):
        sg(A_log=A_logs[_L], a=abuf[_L].reshape(1, MAXT, NVH), b=bbuf[_L].reshape(1, MAXT, NVH),
           dt_bias=dt_biases[_L], q=qbuf, k=kbuf[_L].reshape(1, MAXT, NKH, DK),
           v=vbuf[_L].reshape(1, MAXT, NVH, DV), scale=1.0, initial_state=banks_g[_L],
           inplace_final_state=True, cu_seqlens=cu_fixed, ssm_state_indices=ssi_fixed,
           use_qk_l2norm_in_kernel=True)


def bench(fn, name, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:18s} = {ms:7.2f} ms/commit")
    return ms


def reset_banks_g():
    for _L in range(L):
        banks_g[_L].copy_(bank_init[_L])


rows = col0.tolist()


def maxdiff(a, b):
    return max((a[_L][rows] - b[_L][rows]).abs().max().item() for _L in range(L))


def tag(mx):
    return "IDENTICAL" if mx == 0 else ("~floor" if mx < 1e-3 else "DIVERGE")


# ---- byte-identity #1: per-layer(varlen) vs fixed-shape-EAGER(state-neutral pad) ----
bk_ref = [bank_init[_L].clone() for _L in range(L)]
run_perlayer(bk_ref)
reset_banks_g(); fill_fixed(); committer_fixed_loop()
bk_fixed_eager = [banks_g[_L].clone() for _L in range(L)]
mx1 = maxdiff(bk_ref, bk_fixed_eager)
print(f"[1] varlen        vs fixed-eager : max_diff={mx1:.3e} {tag(mx1)}")

# ---- CUDA-graph capture of the fixed-shape loop ----
reset_banks_g(); fill_fixed()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        committer_fixed_loop()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    committer_fixed_loop()


def run_graph():
    fill_fixed()      # gather/copy (dynamic, outside graph) -- host part
    g.replay()        # 48 fused_sigmoid as ONE replay -- kills dispatch


# ---- byte-identity #2: fixed-eager vs GRAPH-replay ----
reset_banks_g(); fill_fixed(); g.replay(); torch.cuda.synchronize()
bk_graph=[banks_g[_L].clone() for _L in range(L)]
mx2=maxdiff(bk_fixed_eager,bk_graph)
print(f"[2] fixed-eager   vs graph-replay: max_diff={mx2:.3e} {tag(mx2)}")
mx3=maxdiff(bk_ref,bk_graph)
print(f"[3] varlen        vs graph-replay: max_diff={mx3:.3e} {tag(mx3)}  <== the real gate")

print("--- speed ---")
pl = bench(lambda: run_perlayer([b.clone() for b in bank_init]), "per-layer varlen")
gf = bench(run_graph, "graph replay")
print(f"  => graph is {pl/gf:.2f}x faster ({pl:.1f} -> {gf:.1f} ms)")
