"""FR13 gate (2): does ONE captured committer graph stay byte-identical across replays with DIFFERENT
accept-lengths (T varies) AND different active-batch B (fewer real requests, rest dummy-neutral)? This is
the real-world condition -- every decode step has different accepts/B. If any replay diverges from the
varlen committer for that (B, real), the port is unsafe.
Run: docker run --rm --gpus all --entrypoint python3 -v <repo>:/workspace <img> \
     /workspace/scripts/fr13_committer_graph_varying.py
"""
import torch
from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating_delta_rule_update as sg

dev = "cuda"
L, MAX_B = 48, 4
NKH, NVH, DK, DV = 16, 32, 128, 128
ROWS = 64
dt = torch.bfloat16
MAX_PATH = 12
MAXT = MAX_B * MAX_PATH
SCRATCH = ROWS - 1                     # reserved row for dummy segments; never a real request row
torch.manual_seed(0)

A_logs = torch.randn(L, NVH, device=dev)
dt_biases = torch.randn(L, NVH, device=dev)
# oversized per (layer, request-slot, token) activation pool; slice per test case
K = torch.randn(L, MAX_B, MAX_PATH, NKH, DK, device=dev, dtype=dt)
V = torch.randn(L, MAX_B, MAX_PATH, NVH, DV, device=dev, dtype=dt)
Am = torch.randn(L, MAX_B, MAX_PATH, NVH, device=dev, dtype=dt)
Bm = torch.rand(L, MAX_B, MAX_PATH, NVH, device=dev, dtype=dt)
bank_init = torch.randn(L, ROWS, NVH, DK, DV, device=dev)
col0 = torch.randperm(ROWS - 1, device=dev)[:MAX_B].to(torch.int32)   # distinct real rows, none == SCRATCH

kbuf = torch.zeros(L, MAXT, NKH, DK, device=dev, dtype=dt)
vbuf = torch.zeros(L, MAXT, NVH, DV, device=dev, dtype=dt)
abuf = torch.full((L, MAXT, NVH), -1e4, device=dev, dtype=dt)
bbuf = torch.zeros(L, MAXT, NVH, device=dev, dtype=dt)
qbuf = torch.zeros(1, MAXT, NKH, DK, device=dev, dtype=dt)
cu_fixed = torch.tensor([i * MAX_PATH for i in range(MAX_B + 1)], device=dev, dtype=torch.int32)
ssi_fixed = torch.zeros(MAX_B, MAX_PATH, device=dev, dtype=torch.int32)
banks_g = [bank_init[_L].clone() for _L in range(L)]


def fill_fixed(B, real):
    """Full re-neutralize (cheap GPU memset) then overwrite real slots; dummy slots b>=B stay neutral->SCRATCH."""
    abuf.fill_(-1e4); bbuf.zero_(); kbuf.zero_(); vbuf.zero_()
    ssi_fixed.fill_(SCRATCH)
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


def run_varlen(banks, B, real):
    T = sum(real[:B])
    cu = torch.tensor([0] + list(torch.tensor(real[:B]).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    ssi = torch.zeros(B, MAX_PATH, device=dev, dtype=torch.int32)
    for b in range(B):
        ssi[b, :] = col0[b]
    for _L in range(L):
        k = torch.cat([K[_L, b, :real[b]] for b in range(B)], 0).reshape(1, T, NKH, DK).contiguous()
        v = torch.cat([V[_L, b, :real[b]] for b in range(B)], 0).reshape(1, T, NVH, DV).contiguous()
        a = torch.cat([Am[_L, b, :real[b]] for b in range(B)], 0).reshape(1, T, NVH).contiguous()
        bb = torch.cat([Bm[_L, b, :real[b]] for b in range(B)], 0).reshape(1, T, NVH).contiguous()
        sg(A_log=A_logs[_L], a=a, b=bb, dt_bias=dt_biases[_L], q=torch.zeros(1, T, NKH, DK, device=dev, dtype=dt),
           k=k, v=v, scale=1.0, initial_state=banks[_L], inplace_final_state=True, cu_seqlens=cu,
           ssm_state_indices=ssi, use_qk_l2norm_in_kernel=True)


def reset_banks_g():
    for _L in range(L):
        banks_g[_L].copy_(bank_init[_L])


# capture ONCE with the (4,[3,6,2,5]) shape; the graph is shape-invariant to T/B (always MAXT tokens)
fill_fixed(4, [3, 6, 2, 5])
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        committer_fixed_loop()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    committer_fixed_loop()

cases = [
    (4, [3, 6, 2, 5]),      # capture case
    (4, [11, 1, 7, 4]),     # larger, different T
    (4, [1, 1, 1, 1]),      # minimal T (shrink -> R-B stale-pad test)
    (4, [12, 12, 12, 12]),  # MAX_PATH boundary (all full)
    (2, [5, 3, 0, 0]),      # fewer active requests -> dummy pad (R-C)
    (1, [8, 0, 0, 0]),      # single request
    (3, [12, 2, 6, 0]),     # mixed
]
print(f"col0={col0.tolist()} SCRATCH={SCRATCH}")
allok = True
for B, real in cases:
    bk_ref = [bank_init[_L].clone() for _L in range(L)]
    run_varlen(bk_ref, B, real)
    reset_banks_g(); fill_fixed(B, real); g.replay(); torch.cuda.synchronize()
    rows = col0[:B].tolist()
    mx = max((bk_ref[_L][rows] - banks_g[_L][rows]).abs().max().item() for _L in range(L))
    ok = (mx == 0.0)
    allok &= ok
    print(f"  B={B} real={real} -> max_diff={mx:.3e}  {'IDENTICAL' if ok else 'DIVERGE'}")
print("ALL BYTE-IDENTICAL across varying T/B" if allok else "SOME DIVERGED -- port unsafe")
