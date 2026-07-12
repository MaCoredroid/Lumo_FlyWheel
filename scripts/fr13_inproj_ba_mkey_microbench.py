"""Standalone GB10 microbench: is the bf16 in_proj_ba GEMM M-keyed (row-0 output differs at M=9 vs M=1)?
Decisive, crash-free (no server, no capture). Same-input row across M => any diff = cuBLASLt M-dependent
kernel selection = the co-residency seed candidate. Run INSIDE the vllm container (GPU)."""
import torch, glob
from safetensors import safe_open
dev = "cuda"
f = glob.glob("/models/qwen3.6-27b-fp8/layers-0.safetensors")[0]
W = {}
with safe_open(f, framework="pt", device="cpu") as sf:
    for nm in ("a", "b"):
        W[nm] = sf.get_tensor(f"model.language_model.layers.0.linear_attn.in_proj_{nm}.weight").to(dev)
print("in_proj_a dtype", W["a"].dtype, "shape", tuple(W["a"].shape), flush=True)
g = torch.Generator(device=dev).manual_seed(1313)
X = (torch.randn(16, 5120, generator=g, device=dev, dtype=torch.bfloat16) * 0.1)

def gemm(Wm, M):
    return torch.matmul(X[:M], Wm.t())  # [M, 48]

print("=== THRESHOLD SWEEP: row-0 output at M vs M=1 (find the kernel-switch point) ===", flush=True)
for nm in ("a", "b"):
    base = gemm(W[nm], 1)[0].float()
    row = []
    for M in range(1, 13):
        d = (gemm(W[nm], M)[0].float() - base).abs().max().item()
        row.append(f"M{M}={d:.1e}")
    print(f"  in_proj_{nm} row0: " + "  ".join(row), flush=True)

# ---- FIX VALIDATION: compute all 9 rows via <=CHUNK-row sub-GEMMs vs the served M=9 vs native M=1 ----
def chunked(Wm, N, chunk):
    outs = []
    for s in range(0, N, chunk):
        outs.append(torch.matmul(X[s:min(s + chunk, N)], Wm.t()))
    return torch.cat(outs, 0)

def bmm_fix(Wm, N):
    Xb = X[:N].unsqueeze(1)                       # [N,1,5120] -> M=1 per batch
    Wb = Wm.t().unsqueeze(0).expand(N, -1, -1)    # [N,5120,48]
    return torch.bmm(Xb, Wb).squeeze(1)          # [N,48], ONE launch

print("=== FIX VALIDATION (N=9): each form vs per-row M=1 (native ground truth) ===", flush=True)
for nm in ("a", "b"):
    native_perrow = torch.cat([torch.matmul(X[i:i+1], W[nm].t()) for i in range(9)], 0)
    served9 = gemm(W[nm], 9)
    forms = {
        "served(M=9)": served9,
        "chunk=1": chunked(W[nm], 9, 1),
        "chunk=8": chunked(W[nm], 9, 8),
        "bmm(M=1xN)": bmm_fix(W[nm], 9),
    }
    for label, out in forms.items():
        d = (out.float() - native_perrow.float()).abs().max().item()
        print(f"  in_proj_{nm} {label:14s} vs-native = {d:.3e}", flush=True)
# all-rows check: does an M=8 GEMM give ALL 8 rows == per-row M=1?
for nm in ("a", "b"):
    m8 = gemm(W[nm], 8)
    pr = torch.cat([torch.matmul(X[i:i+1], W[nm].t()) for i in range(8)], 0)
    print(f"  in_proj_{nm} M=8 ALL-8-rows vs per-row-M1 = {(m8.float()-pr.float()).abs().max().item():.3e}", flush=True)
