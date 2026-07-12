"""Decisive GB10 microbench: is tl.rsqrt systematically BIASED vs true 1/sqrt (div-by-sqrt)?

SEAM d in _gdn_node_step: tree default uses  b_q * tl.rsqrt(sum(b_q*b_q)+1e-6)
                          native  uses        b_q / tl.sqrt(sum(b_q*b_q)+1e-6)
If tl.rsqrt is a hardware reciprocal-sqrt APPROXIMATION with a consistent-sign error,
that is a SYSTEMATIC same-sign per-node/per-layer bias -> amplifies over ~48 layers -> garble.
This is the 'it's a bias, fix the bias' seed candidate. Unbiased (zero-mean) rounding would NOT amplify.

Runs INSIDE the vllm container (GPU, triton). No server, no boot variance -> deterministic.
Reports SIGNED mean error (the bias) and max |err| for rsqrt vs div, both against fp64 truth.
"""
import torch, triton, triton.language as tl

DIM_K = 128  # GDN head dim
N = 200000   # many random l2norm calls


@triton.jit
def _rsqrt_kernel(x_ptr, out_rsqrt, out_div, D: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + pid * D + offs).to(tl.float32)
    ss = tl.sum(x * x) + 1e-6
    # SEAM d, our default path (fast reciprocal-sqrt)
    r_rsqrt = x * tl.rsqrt(ss)
    # SEAM d, native path (IEEE div by IEEE sqrt)
    r_div = x / tl.sqrt(ss)
    tl.store(out_rsqrt + pid * D + offs, r_rsqrt)
    tl.store(out_div + pid * D + offs, r_div)


def main():
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1313)
    # representative GDN q/k magnitudes (post in_proj, pre-l2norm ~ O(1))
    x = torch.randn(N, DIM_K, generator=g, device=dev, dtype=torch.float32)
    out_rsqrt = torch.empty_like(x)
    out_div = torch.empty_like(x)
    _rsqrt_kernel[(N,)](x, out_rsqrt, out_div, D=DIM_K, num_warps=8)
    torch.cuda.synchronize()

    # fp64 truth (correctly-rounded reference)
    x64 = x.double()
    ss64 = (x64 * x64).sum(dim=1, keepdim=True) + 1e-6
    truth = (x64 / ss64.sqrt()).float()

    err_rsqrt = (out_rsqrt - truth)
    err_div = (out_div - truth)

    def stats(name, e):
        signed_mean = e.mean().item()          # BIAS (same-sign accumulation): nonzero => systematic
        abs_mean = e.abs().mean().item()
        mx = e.abs().max().item()
        # signed-fraction: of the nonzero errors, what frac are positive? 0.5 => unbiased
        nz = e[e != 0]
        pos_frac = (nz > 0).float().mean().item() if nz.numel() else float("nan")
        print(f"  {name:14s} signed_mean(BIAS)={signed_mean:+.3e}  mean|e|={abs_mean:.3e}  "
              f"max|e|={mx:.3e}  pos_frac={pos_frac:.4f}  (0.5=unbiased)", flush=True)
        return signed_mean, abs_mean

    print(f"=== SEAM d rsqrt-vs-div bias (N={N} vectors, DIM_K={DIM_K}) ===", flush=True)
    sm_r, am_r = stats("rsqrt(default)", err_rsqrt)
    sm_d, am_d = stats("div(native)", err_div)

    # direct rsqrt-vs-div diff (what SCAN_ALIGN seam-d actually changes)
    d = (out_rsqrt - out_div)
    print(f"  rsqrt - div    signed_mean={d.mean().item():+.3e}  mean|e|={d.abs().mean().item():.3e}  "
          f"max|e|={d.abs().max().item():.3e}", flush=True)

    print("\n=== VERDICT ===", flush=True)
    print(f"  rsqrt BIAS / div BIAS ratio = {abs(sm_r)/max(abs(sm_d),1e-30):.2f}x", flush=True)
    if abs(sm_r) > 5 * abs(sm_d) and abs(sm_r) > 1e-9:
        print("  => rsqrt is SYSTEMATICALLY BIASED vs div. Seam-d (rsqrt->div) is a real "
              "same-sign seed. Worth an ISOLATED live garble gate (no K1/e confound).", flush=True)
    else:
        print("  => rsqrt bias ~ div bias (both near correctly-rounded). Seam-d is NOT the "
              "systematic seed; the amplifying bias is elsewhere (e/K1 are bf16 rounding=unbiased).", flush=True)


if __name__ == "__main__":
    main()
