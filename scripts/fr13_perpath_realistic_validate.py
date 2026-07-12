"""RED-TEAM the fix mechanism under REALISTIC conditions (the 0.0 validation used zero-state, single path).

Test A: NON-ZERO initial_state (committed prefix h0). Does fused_sigmoid_gating chain still == packed_decode
        sequential when both start from a non-zero h0?
Test B: MULTI-PATH in one call. Present 3 paths (different lengths, each its OWN state row = copy of h0) as
        cu_seqlens segments to fused_sigmoid_gating; each path's last-token output must == a standalone
        packed_decode run of that path. Confirms sequences are INDEPENDENT (no cross-path contamination)
        and each is native-exact -- the actual fix layout.

Run INSIDE the vllm container (GPU). Deterministic. If either != 0.0, the fix needs rework BEFORE building.
"""
import torch
from vllm.model_executor.layers.fla.ops import (
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
    """native no-spec: sequential packed_decode from initial state h0 (shape [NUM_VH,DIM,DIM])."""
    dev = "cuda"
    ssm = h0.clone().unsqueeze(0)  # [1,NUM_VH,DIM,DIM]
    idx = torch.zeros(1, device=dev, dtype=torch.int32)
    outs = []
    for i in range(D):
        ob = torch.zeros(1, 1, NUM_VH, DIM, device=dev, dtype=torch.bfloat16)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv[i:i+1].contiguous(), a=a[i:i+1].contiguous(),
            b=b[i:i+1].contiguous(), A_log=A_log, dt_bias=dt_bias, scale=SCALE,
            initial_state=ssm, out=ob, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
        outs.append(ob[0, 0].clone())
    return torch.stack(outs, 0)  # [D,NUM_VH,DIM]


def sig_multi(mixed_qkv, a, b, A_log, dt_bias, seg_lens, states):
    """fused_sigmoid_gating with MULTIPLE cu_seqlens segments; states[s] = per-segment initial state row."""
    dev = "cuda"
    D = int(sum(seg_lens))
    q, k, v = split4(mixed_qkv, D)
    aa = a.reshape(1, D, NUM_VH).contiguous()
    bb = b.reshape(1, D, NUM_VH).contiguous()
    ssm = torch.stack(states, 0).contiguous()  # [nseg, NUM_VH, DIM, DIM]
    cu = torch.tensor([0] + list(torch.tensor(seg_lens).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    idx = torch.arange(len(seg_lens), device=dev, dtype=torch.int32)
    out = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=aa, b=bb, dt_bias=dt_bias, q=q, k=k, v=v, scale=SCALE,
        initial_state=ssm, inplace_final_state=True, cu_seqlens=cu,
        ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
    core = out[0] if isinstance(out, (tuple, list)) else out
    return core.reshape(D, NUM_VH, DIM), cu


def main():
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1313)
    mk = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.bfloat16)
    A_log = mk(NUM_VH); dt_bias = mk(NUM_VH)

    # ---- Test A: non-zero initial state ----
    D = 6
    mixed = mk(D, CONV_DIM) * 0.3
    a = mk(D, NUM_VH) * 0.5; b = mk(D, NUM_VH) * 0.5
    h0 = mk(NUM_VH, DIM, DIM).float() * 0.2   # NON-ZERO committed prefix state
    nospec = packed_seq(mixed, a, b, A_log, dt_bias, D, h0).float()
    specA, _ = sig_multi(mixed, a, b, A_log, dt_bias, [D], [h0])
    eA = (specA.float() - nospec)
    print(f"=== Test A: non-zero h0, single chain D={D} ===", flush=True)
    print(f"  spec-chain vs packed_decode: max|d|={eA.abs().max().item():.3e} mean|d|={eA.abs().mean().item():.3e}",
          flush=True)

    # ---- Test B: multi-path in ONE call, each its own h0 copy ----
    seg = [2, 3, 5]                     # three paths of different lengths (like tree paths)
    Dt = sum(seg)
    mixedB = mk(Dt, CONV_DIM) * 0.3
    aB = mk(Dt, NUM_VH) * 0.5; bB = mk(Dt, NUM_VH) * 0.5
    states = [mk(NUM_VH, DIM, DIM).float() * 0.2 for _ in seg]   # distinct h0 per path (general case)
    specB, cu = sig_multi(mixedB, aB, bB, A_log, dt_bias, seg, states)
    print(f"\n=== Test B: multi-path one call, seg_lens={seg}, cu={cu.tolist()} ===", flush=True)
    allok = True
    for s in range(len(seg)):
        lo, hi = int(cu[s]), int(cu[s + 1])
        ref = packed_seq(mixedB[lo:hi], aB[lo:hi], bB[lo:hi], A_log, dt_bias, hi - lo, states[s]).float()
        e = (specB[lo:hi].float() - ref)
        mx = e.abs().max().item()
        allok &= (mx == 0.0)
        print(f"  path{s} (len {hi-lo}) last-token & full seq vs standalone packed_decode: max|d|={mx:.3e}",
              flush=True)

    print("\n=== VERDICT ===", flush=True)
    okA = eA.abs().max().item() < 1e-5
    print(f"  Test A (non-zero h0): {'PASS' if okA else 'FAIL'} (max|d|={eA.abs().max().item():.2e})", flush=True)
    print(f"  Test B (multi-path independent + native-exact): {'PASS' if allok else 'FAIL'}", flush=True)
    if okA and allok:
        print("  => fix mechanism holds under REALISTIC conditions (non-zero state + multi-path). Build it.",
              flush=True)
    else:
        print("  => fix mechanism BREAKS under realistic conditions. Do NOT build yet; investigate.", flush=True)


if __name__ == "__main__":
    main()
