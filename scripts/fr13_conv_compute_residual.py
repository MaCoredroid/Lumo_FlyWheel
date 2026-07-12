"""FR13 garble localization: does the TREE causal-conv COMPUTE differ from
native ``causal_conv1d_update`` on IDENTICAL per-node windows?

Context (all deterministic, established elsewhere):
  - GDN recurrent scan is the known drift (~9e-4 unbiased, tree _gdn_node_step
    vs native packed_decode).
  - in_proj (bf16+fp8) and the conv WINDOW indexing are proven
    co-residency-clean / bit-exact.
  - OPEN: does the 4-tap depthwise causal-conv ARITHMETIC also differ?

The tree conv computes each tap as ``(x_bf16 * w_bf16).to(bf16).to(f32)`` then
fp32-accumulates (native-bf16-taps arm, the FR11-overturning fix). Native
``causal_conv1d_update`` loads bf16 and accumulates products in fp32 WITHOUT
rounding each product to bf16. bf16*bf16 fits exactly in fp32, so native's
per-tap product is EXACT; the tree ROUNDS each tap product to bf16 first. That
per-tap bf16 rounding is the candidate "second seed".

This script feeds IDENTICAL windows (real conv weight [10240,1,4] bf16, no bias,
silu activation) to:
  (a) native causal_conv1d_update  (the incumbent decode kernel)
  (b) tree legacy_tree_conv_taps_acc_reference (bf16 taps, the tree COMPUTE)
  (c) tree fused_tree_conv_taps_acc            (the shipped fused form)
  (d) an exact-fp32 window reference           (native-driver validation)

Two topologies:
  PART 1  linear chain, native driven SEQUENTIALLY with its own conv_state
          (validates the native driver + window ordering end to end).
  PART 2  cat8 tree parent=[-1,-1,0,0,2,2,4,6]; per-node ancestor windows fed
          batched to native (conv_state = 3 ancestor taps) and to the tree
          taps functions.

Verdict: nonzero unbiased diff ~1e-4..1e-3 => conv is a SECOND seed;
0.0 / <<1e-5 => conv compute is clean/negligible.

Run INSIDE the vllm container (GPU). Deterministic, no server.
"""
import torch
from safetensors import safe_open

from lumo_flywheel_serving.fr13_tree_conv_fused import (
    fused_tree_conv_taps_acc,
    legacy_tree_conv_taps_acc_reference,
    tree_paths_from_parent,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)

DEV = "cuda"
LAYER = "/models/qwen3.6-27b-fp8/layers-0.safetensors"
DIM = 10240   # CONV_DIM = 2*key_dim + val_dim
WIDTH = 4
ACT = "silu"


def load_conv_weight():
    with safe_open(LAYER, framework="pt", device="cpu") as sf:
        w = sf.get_tensor(
            "model.language_model.layers.0.linear_attn.conv1d.weight"
        )  # [10240, 1, 4] bf16
        keys = set(sf.keys())
    has_bias = (
        "model.language_model.layers.0.linear_attn.conv1d.bias" in keys
    )
    w2 = w.squeeze(1).contiguous().to(DEV)  # [dim, width] bf16
    assert w2.shape == (DIM, WIDTH), w2.shape
    return w2, has_bias


def silu_f32(t):
    return torch.nn.functional.silu(t)


def taps_fp32_exact(window_bf16, weight_bf16):
    """Exact-fp32 window reference (no per-tap bf16 rounding).

    window_bf16: [n, width, dim] bf16 ; weight_bf16: [dim, width] bf16.
    bf16 values are exact in fp32, so this equals native's per-tap product
    (native accumulates exact fp32 products). Returns pre-activation acc [n,dim].
    """
    wf = window_bf16.float()
    kf = weight_bf16.float().t().unsqueeze(0)  # [1, width, dim]
    prod = wf * kf                              # exact fp32 products
    return prod.sum(dim=1)                      # [n, dim]


def report(name, a, b):
    """a,b compared in fp32. Returns (max, mean, signed_mean)."""
    e = (a.float() - b.float())
    mx = e.abs().max().item()
    mn = e.abs().mean().item()
    sm = e.mean().item()
    nz = e[e != 0]
    pf = (nz > 0).float().mean().item() if nz.numel() else float("nan")
    print(
        f"  {name:42s} max|d|={mx:.3e}  mean|d|={mn:.3e}  "
        f"signed_mean={sm:+.3e}  pos_frac={pf:.4f}",
        flush=True,
    )
    return mx, mn, sm


def native_batched(windows_bf16, weight):
    """Native causal_conv1d_update over per-node windows [n,width,dim].

    conv_state = first width-1 taps (state col j = window[:,j]); x = last tap.
    Block 0 is the null block (skipped), so real blocks are 1..n and
    conv_state_indices = [1..n]. Returns native output [n, dim] bf16.
    """
    n = windows_bf16.size(0)
    state_len = WIDTH - 1
    conv_state = torch.zeros(n + 1, DIM, state_len, device=DEV, dtype=torch.bfloat16)
    # block b+1 holds node b's prior taps: col j = window[b, j]
    conv_state[1:] = windows_bf16[:, :state_len, :].permute(0, 2, 1).contiguous()
    idx = torch.arange(1, n + 1, device=DEV, dtype=torch.int32)
    x_t = windows_bf16[:, WIDTH - 1, :].contiguous().clone()  # [n, dim]
    out = causal_conv1d_update(
        x_t, conv_state, weight, bias=None, activation=ACT,
        conv_state_indices=idx,
    )
    return out  # [n, dim] bf16


# ---------------------------------------------------------------------------
# PART 1: linear chain, native driven sequentially with its own conv_state.
# ---------------------------------------------------------------------------
def part1_chain(weight):
    L = 64
    g = torch.Generator(device=DEV).manual_seed(1313)
    x = torch.randn(L, DIM, generator=g, device=DEV, dtype=torch.bfloat16) * 0.3

    # native: sequential decode with a single persistent conv_state.
    # block 0 = null (skipped); real state lives in block 1.
    conv_state = torch.zeros(2, DIM, WIDTH - 1, device=DEV, dtype=torch.bfloat16)
    idx = torch.tensor([1], device=DEV, dtype=torch.int32)
    native_out = torch.empty(L, DIM, device=DEV, dtype=torch.bfloat16)
    for i in range(L):
        x_i = x[i : i + 1].contiguous().clone()  # kernel overwrites its x
        o = causal_conv1d_update(
            x_i, conv_state, weight, bias=None, activation=ACT,
            conv_state_indices=idx,
        )
        native_out[i] = o[0]

    # build the identical per-node windows [L, width, dim] (zero-pad start)
    windows = torch.zeros(L, WIDTH, DIM, device=DEV, dtype=torch.bfloat16)
    for i in range(L):
        for k in range(WIDTH):
            src = i - (WIDTH - 1) + k
            if src >= 0:
                windows[i, k] = x[src]

    xb = windows[:, WIDTH - 1, :].contiguous()  # current tokens [L, dim]

    # (b) tree legacy bf16 taps
    acc_legacy = legacy_tree_conv_taps_acc_reference(
        window=windows, conv_weights=weight, bias=None, x=xb,
        tap_dtype=torch.bfloat16,
    )
    tree_legacy = silu_f32(acc_legacy).to(torch.bfloat16)

    # (c) tree fused bf16 taps
    acc_fused = fused_tree_conv_taps_acc(
        window=windows, conv_weights=weight, bias=None
    )
    tree_fused = silu_f32(acc_fused).to(torch.bfloat16)

    # (d) exact-fp32 window reference
    acc_ref = taps_fp32_exact(windows, weight)
    ref_out = silu_f32(acc_ref).to(torch.bfloat16)

    print(f"\n=== PART 1: LINEAR CHAIN (L={L}, sequential native) ===", flush=True)
    print(
        f"  native |out| max={native_out.abs().max().item():.3e} "
        f"mean={native_out.abs().mean().item():.3e}  "
        f"nonzero_frac={(native_out != 0).float().mean().item():.3f}",
        flush=True,
    )
    # red-team: native driver + window ordering validation
    mx_valid, _, _ = report("native vs exact-fp32 window ref", native_out, ref_out)
    report("tree-legacy vs exact-fp32 window ref", tree_legacy, ref_out)
    report("tree-fused  vs exact-fp32 window ref", tree_fused, ref_out)
    mx_fl, _, _ = report("tree-fused vs tree-legacy (bit-exact bar)", tree_fused, tree_legacy)
    mx_ans, mn_ans, sm_ans = report(">>> tree-legacy vs NATIVE (THE ANSWER)", tree_legacy, native_out)
    report(">>> tree-fused  vs NATIVE", tree_fused, native_out)
    return {
        "valid_native_vs_ref_max": mx_valid,
        "fused_vs_legacy_max": mx_fl,
        "ans_max": mx_ans,
        "ans_mean": mn_ans,
        "ans_signed": sm_ans,
        "native_out": native_out,
        "windows": windows,
    }


# ---------------------------------------------------------------------------
# PART 2: cat8 tree, per-node ancestor windows (batched native update).
# ---------------------------------------------------------------------------
def part2_cat8(weight):
    parent = [-1, -1, 0, 0, 2, 2, 4, 6]
    tree_n = len(parent)
    paths = tree_paths_from_parent(parent)
    g = torch.Generator(device=DEV).manual_seed(2027)
    xnode = torch.randn(
        tree_n, DIM, generator=g, device=DEV, dtype=torch.bfloat16
    ) * 0.3

    # per-node window = last WIDTH tokens of (zeros ++ path-token rows)
    windows = torch.zeros(tree_n, WIDTH, DIM, device=DEV, dtype=torch.bfloat16)
    for node in range(tree_n):
        path = paths[node]  # root..node node-ids
        toks = [torch.zeros(DIM, device=DEV, dtype=torch.bfloat16)] * (WIDTH - 1)
        toks = toks + [xnode[p] for p in path]
        last = toks[-WIDTH:]
        for k in range(WIDTH):
            windows[node, k] = last[k]

    xb = windows[:, WIDTH - 1, :].contiguous()

    native_out = native_batched(windows, weight)

    acc_legacy = legacy_tree_conv_taps_acc_reference(
        window=windows, conv_weights=weight, bias=None, x=xb,
        tap_dtype=torch.bfloat16,
    )
    tree_legacy = silu_f32(acc_legacy).to(torch.bfloat16)
    acc_fused = fused_tree_conv_taps_acc(
        window=windows, conv_weights=weight, bias=None
    )
    tree_fused = silu_f32(acc_fused).to(torch.bfloat16)
    acc_ref = taps_fp32_exact(windows, weight)
    ref_out = silu_f32(acc_ref).to(torch.bfloat16)

    print(
        f"\n=== PART 2: cat8 tree parent={parent} (batched native) ===",
        flush=True,
    )
    print(
        f"  native |out| max={native_out.abs().max().item():.3e} "
        f"mean={native_out.abs().mean().item():.3e}  "
        f"nonzero_frac={(native_out != 0).float().mean().item():.3f}",
        flush=True,
    )
    mx_valid, _, _ = report("native vs exact-fp32 window ref", native_out, ref_out)
    report("tree-legacy vs exact-fp32 window ref", tree_legacy, ref_out)
    mx_fl, _, _ = report("tree-fused vs tree-legacy (bit-exact bar)", tree_fused, tree_legacy)
    mx_ans, mn_ans, sm_ans = report(">>> tree-legacy vs NATIVE (THE ANSWER)", tree_legacy, native_out)
    report(">>> tree-fused  vs NATIVE", tree_fused, native_out)
    return {
        "valid_native_vs_ref_max": mx_valid,
        "fused_vs_legacy_max": mx_fl,
        "ans_max": mx_ans,
        "ans_mean": mn_ans,
        "ans_signed": sm_ans,
    }


def robustness_sweep(weight):
    """Multi-seed / multi-scale: prove the 0.0 is not seed-luck or a bf16-cast
    washout. Reports max over seeds of (tree vs native) and the exact-equal
    element fraction. A real ~1e-3 compute seed would break bit-equality on
    many of the ~10240*n elements per seed."""
    print("\n=== ROBUSTNESS SWEEP (cat8, tree-legacy vs native) ===", flush=True)
    parent = [-1, -1, 0, 0, 2, 2, 4, 6]
    tree_n = len(parent)
    paths = tree_paths_from_parent(parent)
    worst = 0.0
    worst_eqfrac = 1.0
    for seed in (1, 7, 42, 99, 1234, 31337):
        for scale in (0.1, 0.3, 1.0):
            g = torch.Generator(device=DEV).manual_seed(seed)
            xnode = torch.randn(
                tree_n, DIM, generator=g, device=DEV, dtype=torch.bfloat16
            ) * scale
            windows = torch.zeros(tree_n, WIDTH, DIM, device=DEV, dtype=torch.bfloat16)
            for node in range(tree_n):
                path = paths[node]
                toks = [torch.zeros(DIM, device=DEV, dtype=torch.bfloat16)] * (WIDTH - 1)
                toks = toks + [xnode[p] for p in path]
                last = toks[-WIDTH:]
                for k in range(WIDTH):
                    windows[node, k] = last[k]
            xb = windows[:, WIDTH - 1, :].contiguous()
            native_out = native_batched(windows, weight)
            acc_legacy = legacy_tree_conv_taps_acc_reference(
                window=windows, conv_weights=weight, bias=None, x=xb,
                tap_dtype=torch.bfloat16,
            )
            tree_legacy = silu_f32(acc_legacy).to(torch.bfloat16)
            e = (tree_legacy.float() - native_out.float())
            mx = e.abs().max().item()
            eqfrac = (tree_legacy == native_out).float().mean().item()
            worst = max(worst, mx)
            worst_eqfrac = min(worst_eqfrac, eqfrac)
    print(
        f"  over 6 seeds x 3 scales: worst tree-vs-native max|d|={worst:.3e}  "
        f"min exact-equal element frac={worst_eqfrac:.6f}",
        flush=True,
    )
    return worst


def main():
    torch.manual_seed(0)
    weight, has_bias = load_conv_weight()
    print(f"conv weight [dim,width]={tuple(weight.shape)} {weight.dtype}  has_bias={has_bias}", flush=True)
    if has_bias:
        print("  WARNING: model has conv bias but this bench runs bias=None (matches native call site)", flush=True)

    r1 = part1_chain(weight)
    r2 = part2_cat8(weight)
    sweep_worst = robustness_sweep(weight)

    print("\n=== RED-TEAM GUARDS ===", flush=True)
    ok = True
    # 1) native produced non-trivial nonzero output
    if r1["native_out"].abs().max().item() < 1e-3:
        print("  GUARD FAIL: native output ~0, kernel did not run.", flush=True)
        ok = False
    # 2) native must match the exact-fp32 window ref to output-cast scale
    #    (else window ordering / driver is wrong -> answer untrustworthy)
    for tag, r in (("chain", r1), ("cat8", r2)):
        if r["valid_native_vs_ref_max"] > 5e-2:
            print(
                f"  GUARD FAIL ({tag}): native vs exact-fp32 ref max|d|="
                f"{r['valid_native_vs_ref_max']:.2e} is O(1) -> ordering/driver "
                "mismatch. Answer NOT trustworthy.",
                flush=True,
            )
            ok = False
        else:
            print(
                f"  GUARD OK ({tag}): native vs fp32 ref max|d|="
                f"{r['valid_native_vs_ref_max']:.2e} (~output-cast scale).",
                flush=True,
            )
    # 3) fused must be bit-exact to legacy (the module's own bar)
    for tag, r in (("chain", r1), ("cat8", r2)):
        print(
            f"  fused==legacy ({tag}): max|d|={r['fused_vs_legacy_max']:.2e} "
            f"({'BIT-EXACT' if r['fused_vs_legacy_max'] == 0.0 else 'DIFF'})",
            flush=True,
        )

    print("\n=== VERDICT (tree conv COMPUTE vs native) ===", flush=True)
    print(
        f"  chain: max|d|={r1['ans_max']:.3e} mean|d|={r1['ans_mean']:.3e} "
        f"signed={r1['ans_signed']:+.3e}",
        flush=True,
    )
    print(
        f"  cat8 : max|d|={r2['ans_max']:.3e} mean|d|={r2['ans_mean']:.3e} "
        f"signed={r2['ans_signed']:+.3e}",
        flush=True,
    )
    print(f"  robustness sweep worst tree-vs-native max|d|={sweep_worst:.3e}", flush=True)
    if not ok:
        print("  >>> BLOCKED: a guard failed; numbers above are NOT trustworthy.", flush=True)
    elif max(r1["ans_max"], r2["ans_max"], sweep_worst) < 1e-5:
        print("  >>> CONV COMPUTE CLEAN/NEGLIGIBLE (<<1e-5).", flush=True)
    else:
        print("  >>> CONV COMPUTE IS A SECOND SEED (nonzero ~1e-4..1e-3).", flush=True)


if __name__ == "__main__":
    main()
