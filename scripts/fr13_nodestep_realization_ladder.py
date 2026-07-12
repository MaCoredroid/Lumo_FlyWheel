"""Decisive GB10 realization ladder: does the node-step realization diff (tree default vs
native fused_recurrent) amplify over depth, and is it BIASED (same-sign) or unbiased?

Phase 1 (no layout risk -- both are OUR _gdn_node_step): run a depth-D spine chain with
SCAN_ALIGN=False (tree default: rsqrt, fp32-beta, fp32-carry) vs SCAN_ALIGN=True (native
realization: div, bf16-beta, bf16-carry). Report per-depth |out diff|, signed_mean (BIAS),
and how the carried-STATE divergence grows with depth (the amplifier).

Phase 2 (guarded): compare tree default & SCAN_ALIGN chains against native
fused_recurrent_gated_delta_rule_packed_decode on identical inputs (one head). A pytorch fp32
replica guards the layout: tree-default MUST match the replica or the layout is wrong (don't trust).

Run INSIDE the vllm container (GPU, triton). Deterministic, no server, no boot variance.
"""
import torch, triton, triton.language as tl
from lumo_flywheel_serving.fr10_gdn_tree_kernel import _gdn_node_step

DIM_K = 128
DIM_V = 128
BLOCK_V = 128
SCALE = DIM_K ** -0.5


@triton.jit
def _chain_kernel(q, k, v, raw_a, raw_b, A_log, dt_bias, out, state_out,
                  D: tl.constexpr, DIM_K: tl.constexpr, DIM_V: tl.constexpr,
                  BLOCK_V: tl.constexpr, OUTPUT_SCALE: tl.constexpr,
                  SCAN_ALIGN: tl.constexpr):
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
            RAW_GATING=True, SCAN_ALIGN=SCAN_ALIGN,
        )
        tl.store(out + i * DIM_V + offs_v, out_i, mask=v_mask)
    tl.store(state_out + offs_v[:, None] * DIM_K + offs_k[None, :], state, mask=v_mask[:, None])


def run_chain(q, k, v, ra, rb, A_log, dt_bias, D, align):
    out = torch.zeros(D, DIM_V, device="cuda", dtype=torch.float32)
    st = torch.zeros(BLOCK_V, DIM_K, device="cuda", dtype=torch.float32)
    _chain_kernel[(1,)](q, k, v, ra, rb, A_log, dt_bias, out, st,
                        D=D, DIM_K=DIM_K, DIM_V=DIM_V, BLOCK_V=BLOCK_V,
                        OUTPUT_SCALE=SCALE, SCAN_ALIGN=align, num_warps=8)
    torch.cuda.synchronize()
    return out, st[:DIM_V]


def main():
    dev = "cuda"
    D = 64  # depth (layers-worth of recurrence steps)
    g = torch.Generator(device=dev).manual_seed(1313)
    q = torch.randn(D, DIM_K, generator=g, device=dev, dtype=torch.bfloat16) * 0.3
    k = torch.randn(D, DIM_K, generator=g, device=dev, dtype=torch.bfloat16) * 0.3
    v = torch.randn(D, DIM_V, generator=g, device=dev, dtype=torch.bfloat16) * 0.3
    ra = torch.randn(D, generator=g, device=dev, dtype=torch.bfloat16) * 0.5
    rb = torch.randn(D, generator=g, device=dev, dtype=torch.bfloat16) * 0.5
    A_log = torch.randn(1, generator=g, device=dev, dtype=torch.bfloat16)
    dt_bias = torch.randn(1, generator=g, device=dev, dtype=torch.bfloat16)

    out_def, st_def = run_chain(q, k, v, ra, rb, A_log, dt_bias, D, False)
    out_aln, st_aln = run_chain(q, k, v, ra, rb, A_log, dt_bias, D, True)

    print(f"=== PHASE 1: tree DEFAULT (fp32) vs SCAN_ALIGN (native realization), depth D={D} ===", flush=True)
    print("  depth |  max|out diff|  signed_mean(BIAS)   pos_frac", flush=True)
    for d in [0, 3, 7, 15, 31, 47, 63]:
        e = (out_def[d] - out_aln[d]).float()
        nz = e[e != 0]
        pf = (nz > 0).float().mean().item() if nz.numel() else float("nan")
        print(f"   {d:3d}  |  {e.abs().max().item():.3e}     {e.mean().item():+.3e}      {pf:.3f}", flush=True)
    # cumulative over all depths
    eall = (out_def - out_aln).float()
    print(f"  ALL-DEPTH out: mean|d|={eall.abs().mean().item():.3e}  signed_mean(BIAS)={eall.mean().item():+.3e}  "
          f"pos_frac={(eall[eall!=0]>0).float().mean().item():.4f}", flush=True)
    es = (st_def - st_aln).float()
    print(f"  FINAL STATE:   max|d|={es.abs().max().item():.3e}  signed_mean(BIAS)={es.mean().item():+.3e}  "
          f"pos_frac={(es[es!=0]>0).float().mean().item():.4f}", flush=True)

    # does the STATE divergence GROW with depth? (the amplifier signature)
    print("\n=== amplification: carried-state divergence growth (run chains to depth d, compare) ===", flush=True)
    prev = None
    for d in [1, 2, 4, 8, 16, 32, 64]:
        _, sd = run_chain(q, k, v, ra, rb, A_log, dt_bias, d, False)
        _, sa = run_chain(q, k, v, ra, rb, A_log, dt_bias, d, True)
        div = (sd - sa).float().abs().max().item()
        ratio = f"{div/prev:.2f}x" if prev and prev > 0 else "--"
        print(f"  depth={d:2d}  max|state diff|={div:.3e}  growth={ratio}", flush=True)
        prev = div

    print("\n=== VERDICT (phase 1) ===", flush=True)
    print(f"  If signed_mean ~ 0 (pos_frac~0.5) and state-diff grows ~linearly: SCAN_ALIGN change is", flush=True)
    print(f"  unbiased rounding, weakly amplifying -> realization-matching won't zero garble, and the", flush=True)
    print(f"  amplifier is the lever. If signed_mean is large same-sign and state-diff grows FAST:", flush=True)
    print(f"  SCAN_ALIGN (K1 bf16 carry) injects a systematic drift -> tree DEFAULT (fp32) is the right", flush=True)
    print(f"  carry and the seed is elsewhere.", flush=True)


if __name__ == "__main__":
    main()
