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

print("=== FIX VALIDATION (N=9 rows): chunked-<=CHUNK vs per-row M=1 (native ground truth) ===", flush=True)
for chunk in (1, 4, 5):
    for nm in ("a", "b"):
        served9 = gemm(W[nm], 9)                      # what the tree does now (M=9)
        native_perrow = torch.cat([gemm(W[nm], 1) if False else torch.matmul(X[i:i+1], W[nm].t()) for i in range(9)], 0)
        fixed = chunked(W[nm], 9, chunk)
        d_served = (served9.float() - native_perrow.float()).abs().max().item()
        d_fixed = (fixed.float() - native_perrow.float()).abs().max().item()
        print(f"  chunk={chunk} in_proj_{nm}: served(M=9)-vs-native={d_served:.3e}  FIXED(chunk)-vs-native={d_fixed:.3e}", flush=True)
