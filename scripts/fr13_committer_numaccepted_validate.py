"""FR13 graph-capture ENABLER validation: does fused_sigmoid with FIXED-shape padding + num_accepted_tokens
give the BYTE-IDENTICAL committed state as the current variable-length cu_seqlens committer? If yes, the
committer is CUDA-graph-capturable (fixed shapes) -> the 48-launch dispatch (~24ms) can be replayed as one
graph. If no, the num_accepted_tokens semantics differ and the padding approach is wrong.
Run: docker run --rm --gpus all --entrypoint python3 -v <repo>:/workspace <img> \
     /workspace/scripts/fr13_committer_numaccepted_validate.py
"""
import torch
from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating_delta_rule_update as sg

dev = "cuda"
B = 4
NKH, NVH, DK, DV = 16, 32, 128, 128
BANK_ROWS = 64
dt = torch.bfloat16
torch.manual_seed(0)

# per-request real path lengths (1 root + accepted); vary them
real_len = [3, 6, 2, 5]           # T_b for each request
MAX_T = 8                          # fixed pad target (>= max real_len)
A_log = torch.randn(NVH, device=dev, dtype=torch.float32)
dt_bias = torch.randn(NVH, device=dev, dtype=torch.float32)
scale = 1.0

# shared initial bank + indices (same for both runs)
bank0 = torch.randn(BANK_ROWS, NVH, DK, DV, device=dev, dtype=torch.float32)
ssi_col0 = torch.randint(0, BANK_ROWS, (B,), device=dev, dtype=torch.int32)

# per-request token activations (random); we'll slice/pad from a max buffer so both runs use SAME values
kbuf = torch.randn(B, MAX_T, NKH, DK, device=dev, dtype=dt)
vbuf = torch.randn(B, MAX_T, NVH, DV, device=dev, dtype=dt)
abuf = torch.randn(B, MAX_T, NVH, device=dev, dtype=dt)
bbuf = torch.rand(B, MAX_T, NVH, device=dev, dtype=dt)


def run_varlen():
    """CURRENT committer style: cu_seqlens = exact real lengths, no padding, no num_accepted."""
    bank = bank0.clone()
    T = sum(real_len)
    k = torch.cat([kbuf[b, :real_len[b]] for b in range(B)], 0).reshape(1, T, NKH, DK).contiguous()
    v = torch.cat([vbuf[b, :real_len[b]] for b in range(B)], 0).reshape(1, T, NVH, DV).contiguous()
    a = torch.cat([abuf[b, :real_len[b]] for b in range(B)], 0).reshape(1, T, NVH).contiguous()
    bb = torch.cat([bbuf[b, :real_len[b]] for b in range(B)], 0).reshape(1, T, NVH).contiguous()
    q = torch.zeros(1, T, NKH, DK, device=dev, dtype=dt)
    cu = torch.tensor([0] + list(torch.tensor(real_len).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    ssi = torch.zeros(B, MAX_T, device=dev, dtype=torch.int32)
    for b in range(B):
        ssi[b, :] = ssi_col0[b]
    sg(A_log=A_log, a=a, b=bb, dt_bias=dt_bias, q=q, k=k, v=v, scale=scale,
       initial_state=bank, inplace_final_state=True, cu_seqlens=cu,
       ssm_state_indices=ssi, use_qk_l2norm_in_kernel=True)
    return bank


def run_fixed_numaccepted():
    """GRAPH-CAPTURABLE style: FIXED MAX_T per request (padded), num_accepted_tokens = real lengths."""
    bank = bank0.clone()
    T = B * MAX_T
    # pad each request to MAX_T (pad tokens are whatever's in the buffer beyond real_len; truncated by num_accepted)
    k = kbuf.reshape(1, T, NKH, DK).contiguous()
    v = vbuf.reshape(1, T, NVH, DV).contiguous()
    a = abuf.reshape(1, T, NVH).contiguous()
    bb = bbuf.reshape(1, T, NVH).contiguous()
    q = torch.zeros(1, T, NKH, DK, device=dev, dtype=dt)
    cu = torch.tensor([i * MAX_T for i in range(B + 1)], device=dev, dtype=torch.int32)
    ssi = torch.zeros(B, MAX_T, device=dev, dtype=torch.int32)
    for b in range(B):
        ssi[b, :] = ssi_col0[b]
    nacc = torch.tensor(real_len, device=dev, dtype=torch.int32)
    sg(A_log=A_log, a=a, b=bb, dt_bias=dt_bias, q=q, k=k, v=v, scale=scale,
       initial_state=bank, inplace_final_state=True, cu_seqlens=cu,
       ssm_state_indices=ssi, num_accepted_tokens=nacc, use_qk_l2norm_in_kernel=True)
    return bank


A = run_varlen()
Bnk = run_fixed_numaccepted()
diff = (A - Bnk).abs()
# only compare the rows that were actually committed (ssi_col0 rows)
rows = ssi_col0.tolist()
mx = diff[rows].max().item()
print(f"real_len={real_len} MAX_T={MAX_T}")
print(f"committed-row max|varlen - fixed+numaccepted| = {mx:.3e}")
print("=> BYTE-IDENTICAL (graph-capturable)" if mx == 0.0
      else ("=> within-floor (~ok)" if mx < 1e-3 else "=> DIVERGES -- padding/num_accepted semantics differ"))
