"""Per-path native recurrence — CORRECTED multi-seq convention (v2).

Kernel read (fused_sigmoid_gating.py): the INIT read uses ssm_state_indices[i_n,0] (non-spec) and
SKIPS (return) if state_idx<=0 (slot 0 = NULL). The INPLACE final-state write reads
ssm_state_indices[i_n*stride_seq + i_t] for EVERY token i_t -> a 1D idx of length N reads OOB for
multi-token paths (Test B crash root). Fix: inplace_final_state=False routes the final state to a
token-position buffer (no ssm_state_indices in the write) -> no OOB; the init read still needs valid
slots >=1. Bank = [NULL_slot0, h0_path0, h0_path1, ...]; idx = [1,2,...,N].

Run INSIDE the vllm container (GPU). Deterministic.
"""
import torch
from vllm.model_executor.layers.fla.ops import (  # noqa
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)

NUM_KH, NUM_VH, DIM = 16, 48, 128
KEY_DIM, VAL_DIM = NUM_KH * DIM, NUM_VH * DIM
CONV_DIM = KEY_DIM * 2 + VAL_DIM
SCALE = DIM ** -0.5


def split4(mixed_qkv, D):
    q = mixed_qkv[:, 0:KEY_DIM].reshape(1, D, NUM_KH, DIM).contiguous()
    k = mixed_qkv[:, KEY_DIM:2 * KEY_DIM].reshape(1, D, NUM_KH, DIM).contiguous()
    v = mixed_qkv[:, 2 * KEY_DIM:].reshape(1, D, NUM_VH, DIM).contiguous()
    return q, k, v


def packed_seq(mixed_qkv, a, b, A_log, dt_bias, D, h0):
    dev = "cuda"
    ssm = h0.clone().unsqueeze(0)
    idx = torch.zeros(1, device=dev, dtype=torch.int32)
    outs = []
    for i in range(D):
        ob = torch.zeros(1, 1, NUM_VH, DIM, device=dev, dtype=torch.bfloat16)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv[i:i + 1].contiguous(), a=a[i:i + 1].contiguous(),
            b=b[i:i + 1].contiguous(), A_log=A_log, dt_bias=dt_bias, scale=SCALE,
            initial_state=ssm, out=ob, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
        outs.append(ob[0, 0].clone())
    return torch.stack(outs, 0)


def sig_multi_v2(mixed_qkv, a, b, A_log, dt_bias, seg_lens, states):
    """CORRECTED: NULL slot-0 bank, idx slots 1..N, inplace_final_state=False (no OOB final write)."""
    dev = "cuda"
    D = int(sum(seg_lens)); N = len(seg_lens)
    q, k, v = split4(mixed_qkv, D)
    aa = a.reshape(1, D, NUM_VH).contiguous()
    bb = b.reshape(1, D, NUM_VH).contiguous()
    null = torch.zeros_like(states[0])
    ssm = torch.stack([null] + list(states), 0).contiguous()          # [N+1, NUM_VH, DIM, DIM]
    cu = torch.tensor([0] + list(torch.tensor(seg_lens).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    idx = torch.arange(1, N + 1, device=dev, dtype=torch.int32)        # slots 1..N (slot 0 = NULL)
    out = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=aa, b=bb, dt_bias=dt_bias, q=q, k=k, v=v, scale=SCALE,
        initial_state=ssm, inplace_final_state=False, cu_seqlens=cu,
        ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
    core = out[0] if isinstance(out, (tuple, list)) else out
    return core.reshape(D, NUM_VH, DIM), cu


def main():
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1313)
    mk = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.bfloat16)
    A_log = mk(NUM_VH); dt_bias = mk(NUM_VH)

    # Test A: single chain, non-zero h0
    D = 6
    mixed = mk(D, CONV_DIM) * 0.3
    a = mk(D, NUM_VH) * 0.5; b = mk(D, NUM_VH) * 0.5
    h0 = mk(NUM_VH, DIM, DIM).float() * 0.2
    nospec = packed_seq(mixed, a, b, A_log, dt_bias, D, h0).float()
    specA, _ = sig_multi_v2(mixed, a, b, A_log, dt_bias, [D], [h0])
    eA = (specA.float() - nospec).abs().max().item()
    print(f"=== Test A (single, non-zero h0, slot 1): max|d|={eA:.3e} ===", flush=True)

    # Test B: multi-path, each its own h0, DIFFERENT lengths
    seg = [2, 3, 5]
    Dt = sum(seg)
    mixedB = mk(Dt, CONV_DIM) * 0.3
    aB = mk(Dt, NUM_VH) * 0.5; bB = mk(Dt, NUM_VH) * 0.5
    states = [mk(NUM_VH, DIM, DIM).float() * 0.2 for _ in seg]
    specB, cu = sig_multi_v2(mixedB, aB, bB, A_log, dt_bias, seg, states)
    print(f"=== Test B (multi-path, seg={seg}, cu={cu.tolist()}) ===", flush=True)
    allok = True
    for s in range(len(seg)):
        lo, hi = int(cu[s]), int(cu[s + 1])
        ref = packed_seq(mixedB[lo:hi], aB[lo:hi], bB[lo:hi], A_log, dt_bias, hi - lo, states[s]).float()
        mx = (specB[lo:hi].float() - ref).abs().max().item()
        allok &= (mx == 0.0)
        print(f"  path{s} (len {hi-lo}) vs standalone packed_decode: max|d|={mx:.3e}", flush=True)

    print("\n=== VERDICT ===", flush=True)
    okA = eA < 1e-5
    print(f"  Test A: {'PASS' if okA else 'FAIL'} (max|d|={eA:.2e})", flush=True)
    print(f"  Test B (multi-path independent + native-exact): {'PASS' if allok else 'FAIL'}", flush=True)
    if okA and allok:
        print("  => multi-path batched native call CONVENTION SOLVED. Build the per-path fix.", flush=True)
    else:
        print("  => still broken; iterate convention.", flush=True)


if __name__ == "__main__":
    main()
