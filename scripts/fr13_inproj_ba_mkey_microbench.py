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

print("=== bf16 in_proj_ba M-keying: row-0 output at M vs M=1 (same input row X[0]) ===", flush=True)
any_nonzero = False
for nm in ("a", "b"):
    outs = {M: gemm(W[nm], M) for M in (1, 5, 9, 16)}
    base = outs[1][0].float()
    for M in (5, 9, 16):
        d = (outs[M][0].float() - base).abs().max().item()
        rel = d / (base.abs().max().item() + 1e-30)
        print(f"  in_proj_{nm} row0: M={M} vs M=1  max_abs={d:.3e}  rel={rel:.2e}", flush=True)
        if d > 0:
            any_nonzero = True
    # also compare a middle row (row 4) M=9 vs M=16 (both contain row 4)
    d49 = (gemm(W[nm], 9)[4].float() - gemm(W[nm], 16)[4].float()).abs().max().item()
    print(f"  in_proj_{nm} row4: M=9 vs M=16 max_abs={d49:.3e}", flush=True)
print(f"VERDICT: in_proj_ba is {'M-KEYED (seed candidate CONFIRMED)' if any_nonzero else 'ROW-INDEPENDENT (in_proj_ba REFUTED as seed)'}", flush=True)
