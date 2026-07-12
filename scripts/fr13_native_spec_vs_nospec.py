"""FIX-MECHANISM go/no-go: does native fused_sigmoid_gating (spec, per-chain via cu_seqlens) match
native packed_decode (no-spec ground truth) at garble-free scale (~6e-5, << tree's 9e-4)?

If yes: decomposing the tree into root->leaf PATHS and running each path's recurrence through the native
spec kernel (fused_sigmoid_gating) gets each committed token to the SAME ~6e-5 gap that native MTP/E5 has
(which is garble-free) -- so the tree becomes garble-free while KEEPING branches. This validates the fix
mechanism before implementing it.

Run INSIDE the vllm container (GPU). Deterministic.
"""
import torch
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)

NUM_KH, NUM_VH, DIM = 16, 48, 128
KEY_DIM = NUM_KH * DIM
VAL_DIM = NUM_VH * DIM
CONV_DIM = KEY_DIM * 2 + VAL_DIM
SCALE = DIM ** -0.5


def native_nospec(mixed_qkv, a, b, A_log, dt_bias, D):
    dev = "cuda"
    ssm = torch.zeros(1, NUM_VH, DIM, DIM, device=dev, dtype=torch.float32)
    idx = torch.zeros(1, device=dev, dtype=torch.int32)
    outs = []
    for i in range(D):
        out_buf = torch.zeros(1, 1, NUM_VH, DIM, device=dev, dtype=torch.bfloat16)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv[i:i+1].contiguous(), a=a[i:i+1].contiguous(),
            b=b[i:i+1].contiguous(), A_log=A_log, dt_bias=dt_bias, scale=SCALE,
            initial_state=ssm, out=out_buf, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
        outs.append(out_buf[0, 0].clone())
    return torch.stack(outs, 0)  # [D, NUM_VH, DIM]


def native_spec_chain(mixed_qkv, a, b, A_log, dt_bias, D):
    dev = "cuda"
    # 4D [B=1, T, H, K] as the kernel expects (B,T,H,K,V = *k.shape, v.shape[-1])
    q = mixed_qkv[:, 0:KEY_DIM].reshape(1, D, NUM_KH, DIM).contiguous()
    k = mixed_qkv[:, KEY_DIM:2 * KEY_DIM].reshape(1, D, NUM_KH, DIM).contiguous()
    v = mixed_qkv[:, 2 * KEY_DIM:].reshape(1, D, NUM_VH, DIM).contiguous()
    aa = a.reshape(1, D, NUM_VH).contiguous()
    bb = b.reshape(1, D, NUM_VH).contiguous()
    ssm = torch.zeros(1, NUM_VH, DIM, DIM, device=dev, dtype=torch.float32)
    cu = torch.tensor([0, D], device=dev, dtype=torch.int32)
    idx = torch.zeros(1, device=dev, dtype=torch.int32)
    nacc = torch.tensor([D], device=dev, dtype=torch.int32)
    for kwargs in ({}, {"num_accepted_tokens": nacc}):
        try:
            out = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log, a=aa, b=bb, dt_bias=dt_bias,
                q=q, k=k, v=v, scale=SCALE, initial_state=ssm, inplace_final_state=True,
                cu_seqlens=cu, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True, **kwargs)
            core = out[0] if isinstance(out, (tuple, list)) else out
            print(f"spec ran (kwargs={list(kwargs)}); out shape={tuple(core.shape)}", flush=True)
            return core.reshape(D, NUM_VH, DIM)
        except Exception as e:
            print(f"spec kwargs={list(kwargs)} failed: {repr(e)[:150]}", flush=True)
    raise RuntimeError("fused_sigmoid_gating failed")


def main():
    dev = "cuda"
    D = 8  # a tree path length (cat8 max depth ~5-8)
    g = torch.Generator(device=dev).manual_seed(1313)
    mixed_qkv = torch.randn(D, CONV_DIM, generator=g, device=dev, dtype=torch.bfloat16) * 0.3
    a = torch.randn(D, NUM_VH, generator=g, device=dev, dtype=torch.bfloat16) * 0.5
    b = torch.randn(D, NUM_VH, generator=g, device=dev, dtype=torch.bfloat16) * 0.5
    A_log = torch.randn(NUM_VH, generator=g, device=dev, dtype=torch.bfloat16)
    dt_bias = torch.randn(NUM_VH, generator=g, device=dev, dtype=torch.bfloat16)

    nospec = native_nospec(mixed_qkv, a, b, A_log, dt_bias, D).float()
    spec = native_spec_chain(mixed_qkv, a, b, A_log, dt_bias, D).float()

    e = (spec - nospec)
    print(f"\n=== native SPEC (fused_sigmoid_gating chain) vs native NO-SPEC (packed_decode seq), D={D} ===",
          flush=True)
    print(f"  max|d|={e.abs().max().item():.3e}  mean|d|={e.abs().mean().item():.3e}  "
          f"signed_mean={e.mean().item():+.3e}", flush=True)
    print(f"  (tree custom kernel gap vs no-spec was ~9e-4 mean; garble-free target ~6e-5)", flush=True)
    print("\n=== VERDICT ===", flush=True)
    m = e.abs().mean().item()
    if m < 1e-4:
        print(f"  native spec-vs-nospec mean|d|={m:.2e} << tree 9e-4 => running tree paths through", flush=True)
        print(f"  fused_sigmoid_gating gets each committed token to native-MTP scale = GARBLE-FREE.", flush=True)
        print(f"  FIX MECHANISM VALIDATED. Proceed to per-path implementation.", flush=True)
    else:
        print(f"  native spec-vs-nospec mean|d|={m:.2e} NOT << 9e-4 => fused_sigmoid_gating is NOT close", flush=True)
        print(f"  enough to no-spec; the per-path fix would inherit its gap. Reconsider (use packed_decode", flush=True)
        print(f"  per-node instead, or re-examine).", flush=True)


if __name__ == "__main__":
    main()
