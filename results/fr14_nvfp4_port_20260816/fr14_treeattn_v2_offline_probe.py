#!/usr/bin/env python3
"""FR14 TreeAttention-v2 offline byte + timing probe (no serve, no gate window).

Two jobs, one harness:

  1. BYTE PROBE. Assert the candidate arm is byte-identical to the reference arm
     on output AND LSE, over many seeds and context lengths, at the REAL fixed32
     geometry -- 32 tree rows, 24 query heads, 4 KV heads, head_dim 256, bf16,
     paged KV with 1024-row pages, and a real 32x32 tree bias. Not a toy shape.

  2. TIMING READOUT / THE DISCRIMINATOR. The sealed gqa_pair .so exports BOTH
     private launchers, so the same binary yields two points on the geometry
     curve without building anything:

         qrow16    48 CTAs, kBlockM=16, 1 warp   -> 12x KV re-staging
         gqa_pair  12 CTAs, kBlockM=64, 4 warps  ->  3x KV re-staging

     Fitting T = alpha*staged_bytes + beta*mma_work/CTAs to those two points
     PREDICTS T(G=3) and T(G=6) before either kernel exists. See
     treeattn_v2_design.md section 8 for the pre-registered decision rule --
     this harness produces the numbers that rule consumes.

Arm selection is by the campaign's own mechanism and nothing else: the arm IS the
tree_bias batch stride, applied as a zero-copy `as_strided` retag exactly as
`_fr13_fa2_qrow32_b1_candidate_tree_bias` does in the sealed gate, so the operand
under test is the one that was qualified rather than a re-derived equivalent.

Usage:
  python3 fr14_treeattn_v2_offline_probe.py --so <candidate.so> [--json out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys

import torch

# The arm registry, mirrored from the patcher's _FR13_FA2_QROW32_B1_ARMS. The
# sentinel is the ASCII tag 'FR13' (0x46523133) plus an arm ordinal.
QROW16_REFERENCE_SENTINEL = 1179791667
ARMS = {
    "qrow16": {"sentinel": 1179791667, "num_splits": 0,
               "ctas_b1": 48, "block_m": 16, "warps": 1, "kv_lanes": 12},
    "nosplit": {"sentinel": 1179791668, "num_splits": 0,
                "ctas_b1": 24, "block_m": 32, "warps": 2, "kv_lanes": 6},
    "gqa_pair": {"sentinel": 1179791670, "num_splits": 0,
                 "ctas_b1": 12, "block_m": 64, "warps": 4, "kv_lanes": 3},
}

# Fixed32 production geometry. Every one of these is asserted by the in-binary
# gate, so a drift here fails loudly in the kernel rather than silently here.
TREE_ROWS = 32
Q_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
PAGE = 1024
BLOCK_N = 64
FULL_ATTN_LAYERS = 16


def build_case(seq_len: int, seed: int, device: str = "cuda"):
    """One decode step's worth of real operands at the canonical B1 geometry."""
    gen = torch.Generator(device=device).manual_seed(seed)
    num_blocks = math.ceil(seq_len / PAGE) + 1

    q = torch.randn(TREE_ROWS, Q_HEADS, HEAD_DIM, generator=gen,
                    device=device, dtype=torch.bfloat16) * 0.1

    # ILKV layout: [num_blocks, 2, PAGE, KV_HEADS, HEAD_DIM]. Slicing dim 1
    # gives k/v the (2*PAGE*KV*D, KV*D, D, 1) strides the gate demands.
    kv = torch.randn(num_blocks, 2, PAGE, KV_HEADS, HEAD_DIM, generator=gen,
                     device=device, dtype=torch.bfloat16) * 0.1
    k_cache, v_cache = kv[:, 0], kv[:, 1]
    assert tuple(k_cache.stride()) == (2 * PAGE * KV_HEADS * HEAD_DIM,
                                       KV_HEADS * HEAD_DIM, HEAD_DIM, 1), \
        f"ILKV stride contract drifted: {k_cache.stride()}"

    block_table = torch.arange(num_blocks, device=device,
                               dtype=torch.int32).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, TREE_ROWS], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([seq_len], device=device, dtype=torch.int32)

    # A real speculative tree mask: row i may attend to its ancestors only.
    # Sign and magnitude matter -- a zero bias would hide masking defects.
    bias = torch.full((TREE_ROWS, TREE_ROWS), float("-inf"), device=device,
                      dtype=torch.float32)
    parent = [-1] + [max(0, (i - 1) // 2) for i in range(1, TREE_ROWS)]
    for i in range(TREE_ROWS):
        j = i
        while j != -1:
            bias[i, j] = 0.0
            j = parent[j]
    assert tuple(bias.stride()) == (TREE_ROWS, 1)
    return dict(q=q, k=k_cache, v=v_cache, block_table=block_table,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                seqused_k=seqused_k, tree_bias=bias, seq_len=seq_len)


def tag(tree_bias: torch.Tensor, sentinel: int) -> torch.Tensor:
    """The arm IS the batch stride. Zero-copy retag, as the sealed gate does."""
    base = tree_bias[0] if tree_bias.ndim == 3 else tree_bias
    tagged = torch.as_strided(base, size=(1, TREE_ROWS, TREE_ROWS),
                              stride=(sentinel, TREE_ROWS, 1))
    assert int(tagged.stride(0)) == sentinel
    assert tagged.data_ptr() == base.data_ptr(), "retag copied the operand"
    return tagged


def run(case, sentinel: int, num_splits: int):
    out = torch.empty_like(case["q"])
    res = torch.ops._vllm_fa2_C.varlen_fwd_tree_bias(
        case["q"], case["k"], case["v"], out,
        case["cu_seqlens_q"], case["cu_seqlens_k"], case["seqused_k"],
        None, case["block_table"], None,
        TREE_ROWS, case["seq_len"],
        0.0, 1.0 / math.sqrt(HEAD_DIM), False, False,
        -1, -1, 0.0, False,
        num_splits, tag(case["tree_bias"], sentinel), None,
    )
    return out, res[1]


def raw(t: torch.Tensor):
    """Raw bytes as a uint8 array -- the comparison is on BYTES, not values, so
    NaN/-0.0 cannot launder a mismatch past it."""
    return t.detach().contiguous().view(torch.uint8).cpu().numpy()


def byte_mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        raise RuntimeError("comparison contract drifted")
    return int((raw(a) != raw(b)).sum())


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(raw(t).tobytes()).hexdigest()[:16]


def time_arm(case, sentinel, num_splits, iters=30, warmup=10):
    for _ in range(warmup):
        run(case, sentinel, num_splits)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        run(case, sentinel, num_splits)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def staged_gb_per_step(ctas: int, seq_len: int) -> float:
    n_blocks = math.ceil(seq_len / BLOCK_N)
    tile = BLOCK_N * HEAD_DIM * 2 * 2  # K and V, bf16
    return ctas * n_blocks * tile * FULL_ATTN_LAYERS / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--json")
    ap.add_argument("--reference", default="qrow16")
    ap.add_argument("--candidate", default="gqa_pair")
    ap.add_argument("--seq-lens", default="20480,23000,32768,40960")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    torch.ops.load_library(args.so)
    sha = hashlib.sha256(open(args.so, "rb").read()).hexdigest()
    report = {"schema": "fr14.treeattn_v2.offline_probe.v1",
              "so": args.so, "so_sha256": sha,
              "reference_arm": args.reference, "candidate_arm": args.candidate,
              "geometry": {"tree_rows": TREE_ROWS, "q_heads": Q_HEADS,
                           "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
                           "page": PAGE, "block_n": BLOCK_N,
                           "full_attn_layers": FULL_ATTN_LAYERS},
              "cases": [], "timing": []}

    ref, cand = ARMS[args.reference], ARMS[args.candidate]
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    mismatch_total = 0

    # --- 1. byte probe -------------------------------------------------
    for seq_len in seq_lens:
        for seed in range(args.seeds):
            case = build_case(seq_len, seed)
            o_r, l_r = run(case, ref["sentinel"], ref["num_splits"])
            o_c, l_c = run(case, cand["sentinel"], cand["num_splits"])
            ob = byte_mismatches(o_r, o_c)
            lb = byte_mismatches(l_r, l_c)
            mismatch_total += ob + lb
            report["cases"].append(
                {"seq_len": seq_len, "seed": seed,
                 "output_byte_mismatches": ob, "lse_byte_mismatches": lb,
                 "reference_output_sha16": digest(o_r),
                 "candidate_output_sha16": digest(o_c),
                 "reference_lse_sha16": digest(l_r),
                 "candidate_lse_sha16": digest(l_c)})
            del case, o_r, l_r, o_c, l_c
            torch.cuda.empty_cache()

    report["output_raw_byte_mismatches"] = mismatch_total
    report["byte_probe_pass"] = mismatch_total == 0

    # --- 2. timing / the discriminator ---------------------------------
    for seq_len in seq_lens:
        case = build_case(seq_len, 0)
        row = {"seq_len": seq_len}
        for name in (args.reference, args.candidate):
            a = ARMS[name]
            ms = time_arm(case, a["sentinel"], a["num_splits"])
            row[name] = {
                "ms_per_layer_call": ms,
                "ms_per_step_16_layers": ms * FULL_ATTN_LAYERS,
                "ctas_b1": a["ctas_b1"],
                "kv_lanes": a["kv_lanes"],
                "staged_gb_per_step": round(
                    staged_gb_per_step(a["ctas_b1"], seq_len), 3),
            }
        report["timing"].append(row)
        del case
        torch.cuda.empty_cache()

    # --- 3. the fit: solve alpha, beta; predict G=3 and G=6 ------------
    # T = alpha*staged_bytes + beta/CTAs. Two arms, two unknowns, exact solve.
    fits = []
    for row in report["timing"]:
        r, c = row[args.reference], row[args.candidate]
        s1, s2 = r["staged_gb_per_step"], c["staged_gb_per_step"]
        p1, p2 = 1.0 / r["ctas_b1"], 1.0 / c["ctas_b1"]
        t1, t2 = r["ms_per_step_16_layers"], c["ms_per_step_16_layers"]
        det = s1 * p2 - s2 * p1
        if abs(det) < 1e-12:
            continue
        alpha = (t1 * p2 - t2 * p1) / det
        beta = (s1 * t2 - s2 * t1) / det
        pred = {}
        for g, ctas in ((2, 12), (3, 8), (6, 4)):
            pred[f"G{g}"] = round(
                alpha * staged_gb_per_step(ctas, row["seq_len"])
                + beta / ctas, 3)
        fits.append({"seq_len": row["seq_len"],
                     "alpha_ms_per_staged_gb": round(alpha, 4),
                     "beta_ms_times_ctas": round(beta, 4),
                     "predicted_ms_per_step": pred,
                     "verdict": ("head-merge WINS past G=2"
                                 if pred["G3"] < pred["G2"]
                                 else "head-merge EXHAUSTED at G=2")})
    report["fit"] = fits

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        open(args.json, "w").write(text + "\n")
    print(text)
    return 0 if report["byte_probe_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
