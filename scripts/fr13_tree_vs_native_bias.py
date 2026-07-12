"""DECISIVE red-team: is the tree _gdn_node_step diff vs ACTUAL native
fused_recurrent_gated_delta_rule_packed_decode BIASED (same-sign) or unbiased?

Last cycle's ladder measured tree-DEFAULT vs tree-SCAN_ALIGN (both OUR kernel) and found it unbiased,
ASSUMING SCAN_ALIGN==native. This closes that gap: feed IDENTICAL post-conv mixed_qkv to
(1) native packed_decode run SEQUENTIALLY (= no-spec ground truth) and
(2) our _gdn_node_step chain (default AND SCAN_ALIGN), head 0,
and measure signed_mean (BIAS) + pos_frac. If tree-default-vs-native is same-sign biased => the user's
'fix the bias' is RIGHT and the reframe is wrong. If unbiased => reframe holds.

GUARD: the tree-vs-native diff must be O(1e-5) (realization scale). If it's O(1) the layout is wrong.
Run INSIDE the vllm container (GPU). Deterministic.
"""
import torch, triton, triton.language as tl
from lumo_flywheel_serving.fr10_gdn_tree_kernel import _gdn_node_step
from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule_packed_decode

NUM_KH, NUM_VH, DIM = 16, 48, 128
KEY_DIM = NUM_KH * DIM        # 2048
VAL_DIM = NUM_VH * DIM        # 6144
CONV_DIM = KEY_DIM * 2 + VAL_DIM  # 10240
SCALE = DIM ** -0.5
BLOCK_V = 128


@triton.jit
def _chain_kernel(q, k, v, raw_a, raw_b, A_log, dt_bias, out,
                  D: tl.constexpr, DIM_K: tl.constexpr, DIM_V: tl.constexpr,
                  BLOCK_V: tl.constexpr, OUTPUT_SCALE: tl.constexpr, SCAN_ALIGN: tl.constexpr):
    offs_k = tl.arange(0, DIM_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V
    state = tl.zeros((BLOCK_V, DIM_K), dtype=tl.float32)
    b_a_log = tl.load(A_log).to(tl.float32)
    b_dt_bias = tl.load(dt_bias).to(tl.float32)
    for i in tl.static_range(0, D):
        b_q = tl.load(q + i * DIM_K + offs_k).to(tl.float32)
        b_k = tl.load(k + i * DIM_K + offs_k).to(tl.float32)
        b_v = tl.load(v + i * DIM_V + offs_v, mask=v_mask, other=0.0).to(tl.float32)
        b_ra = tl.load(raw_a + i).to(tl.float32)
        b_rb = tl.load(raw_b + i).to(tl.float32)
        state, out_i = _gdn_node_step(
            state, b_q, b_k, b_v, b_rb, b_rb, b_ra, b_rb, b_a_log, b_dt_bias,
            OUTPUT_SCALE=OUTPUT_SCALE, USE_QK_L2NORM_IN_KERNEL=True,
            RAW_GATING=True, SCAN_ALIGN=SCAN_ALIGN)
        tl.store(out + i * DIM_V + offs_v, out_i, mask=v_mask)


def tree_chain(q0, k0, v0, a0, b0, A_log0, dt_bias0, D, align):
    out = torch.zeros(D, DIM, device="cuda", dtype=torch.float32)
    _chain_kernel[(1,)](q0.contiguous(), k0.contiguous(), v0.contiguous(),
                        a0.contiguous(), b0.contiguous(), A_log0, dt_bias0, out,
                        D=D, DIM_K=DIM, DIM_V=DIM, BLOCK_V=BLOCK_V,
                        OUTPUT_SCALE=SCALE, SCAN_ALIGN=align, num_warps=8)
    torch.cuda.synchronize()
    return out


def native_sequential(mixed_qkv, a, b, A_log, dt_bias, D):
    dev = "cuda"
    for sd in (torch.float32, torch.bfloat16):
        try:
            ssm = torch.zeros(1, NUM_VH, DIM, DIM, device=dev, dtype=sd)
            outs = []
            idx = torch.zeros(1, device=dev, dtype=torch.int32)
            for i in range(D):
                out_buf = torch.zeros(1, 1, NUM_VH, DIM, device=dev, dtype=torch.bfloat16)
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=mixed_qkv[i:i+1].contiguous(),
                    a=a[i:i+1].contiguous(), b=b[i:i+1].contiguous(),
                    A_log=A_log, dt_bias=dt_bias, scale=SCALE,
                    initial_state=ssm, out=out_buf, ssm_state_indices=idx,
                    use_qk_l2norm_in_kernel=True)
                outs.append(out_buf[0, 0].clone())  # [NUM_VH, DIM]
            print(f"native ran with state dtype={sd}", flush=True)
            return torch.stack(outs, 0)  # [D, NUM_VH, DIM]
        except Exception as e:
            print(f"native state dtype {sd} failed: {repr(e)[:160]}", flush=True)
    raise RuntimeError("native packed_decode failed for both state dtypes")


def main():
    dev = "cuda"
    D = 32
    g = torch.Generator(device=dev).manual_seed(1313)
    mixed_qkv = torch.randn(D, CONV_DIM, generator=g, device=dev, dtype=torch.bfloat16) * 0.3
    a = torch.randn(D, NUM_VH, generator=g, device=dev, dtype=torch.bfloat16) * 0.5
    b = torch.randn(D, NUM_VH, generator=g, device=dev, dtype=torch.bfloat16) * 0.5
    A_log = torch.randn(NUM_VH, generator=g, device=dev, dtype=torch.bfloat16)
    dt_bias = torch.randn(NUM_VH, generator=g, device=dev, dtype=torch.bfloat16)

    native = native_sequential(mixed_qkv, a, b, A_log, dt_bias, D)   # [D, NUM_VH, DIM]
    native0 = native[:, 0, :].float()                                # v-head 0 -> [D, DIM]

    # head 0 slices: q head0 = [0:128], k head0 = [KEY_DIM:KEY_DIM+128], v head0 = [2*KEY_DIM:2*KEY_DIM+128]
    q0 = mixed_qkv[:, 0:DIM]
    k0 = mixed_qkv[:, KEY_DIM:KEY_DIM + DIM]
    v0 = mixed_qkv[:, 2 * KEY_DIM:2 * KEY_DIM + DIM]
    a0 = a[:, 0].contiguous()
    b0 = b[:, 0].contiguous()
    A0 = A_log[0:1].contiguous()
    dt0 = dt_bias[0:1].contiguous()

    tree_def = tree_chain(q0, k0, v0, a0, b0, A0, dt0, D, False).float()
    tree_aln = tree_chain(q0, k0, v0, a0, b0, A0, dt0, D, True).float()

    def report(name, tree):
        e = (tree - native0)
        nz = e[e != 0]
        pf = (nz > 0).float().mean().item() if nz.numel() else float("nan")
        print(f"  {name:22s} max|d|={e.abs().max().item():.3e}  mean|d|={e.abs().mean().item():.3e}  "
              f"signed_mean(BIAS)={e.mean().item():+.3e}  pos_frac={pf:.4f}", flush=True)
        return e.abs().max().item()

    print(f"\n=== tree _gdn_node_step vs NATIVE packed_decode (head 0, D={D}) ===", flush=True)
    mx_def = report("tree-DEFAULT vs native", tree_def)
    mx_aln = report("tree-SCAN_ALIGN vs native", tree_aln)

    print("\n=== GUARD + VERDICT ===", flush=True)
    if mx_def > 0.05:
        print(f"  GUARD FAIL: tree-vs-native max|d|={mx_def:.2e} is O(1), NOT realization-scale.", flush=True)
        print("  Layout/gating mismatch => DO NOT TRUST the bias verdict. Fix the harness.", flush=True)
        return
    print(f"  GUARD OK (max|d|={mx_def:.2e} ~ realization scale). Bias verdict is trustworthy.", flush=True)
    print("  If tree-DEFAULT pos_frac ~0.5 & signed_mean~0: seed is UNBIASED (reframe holds).", flush=True)
    print("  If pos_frac strongly !=0.5 & signed_mean large same-sign: seed is BIASED (fix the bias).", flush=True)


if __name__ == "__main__":
    main()
