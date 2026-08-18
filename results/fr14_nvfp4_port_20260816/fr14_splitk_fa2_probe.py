#!/usr/bin/env python3
"""FR14 split-K FA2: determinism gate + ULP characterization + timing.

This is the Tier-B counterpart of fr14_treeattn_v2_offline_probe.py, and it is
deliberately the SAME harness with ONE assert swapped. The scaffolding --
build_case()'s real fixed32 operands (32 tree rows, 24 query heads, 4 KV heads,
head_dim 256, bf16, paged KV with 1024-row pages, a real ancestry tree bias),
tag()'s zero-copy `as_strided` arm retag, run()'s call into
torch.ops._vllm_fa2_C.varlen_fwd_tree_bias, and time_arm() -- is reused
verbatim, because the operand under test must be the one the promoted arm was
qualified on rather than a re-derived equivalent. What changes is the verdict:

  BYTE PROBE (there)      : assert 0 byte mismatches vs the reference.
  CHARACTERIZATION (here) : byte identity CANNOT hold and is not the bar.

Split-K partitions each context walk four ways; every split keeps its own
running max and partial denominator, and the combine rescales by
exp(m_split - m_global) before summing. That is a different summation order for
the same real numbers, so the result differs in the last bits and the honest
question is HOW MUCH, not WHETHER. Three things are measured:

  1. DETERMINISM -- a hard gate, and the one that can fail. The combine visits
     splits in index order with a fixed shuffle-butterfly reduction and no
     atomics, so repeated runs on identical inputs must be BITWISE identical.
     Repeats are run in-process (fresh split accumulators each call, so their
     device addresses move between runs -- an address-dependent reduction would
     show up here) and the per-run digests are emitted so a second PROCESS can
     be compared against the first.

  2. ULP CHARACTERIZATION -- output and LSE max-abs delta, plus a ULP histogram
     computed on the monotone integer ordering of the float bits, so "1 ulp" is
     one representable step and not a relative-error proxy. Last-few-ulp is the
     expected scale for a reassociated softmax; anything beyond that is a BUG,
     not rounding, and the STOP condition says so.

  3. ARGMAX-FLIP RATE -- the only number that speaks to behaviour. Attention
     output is projected through a fixed, seeded, logit-like two-stage
     projection and the argmax is compared between arms. Reported beside the
     top1-top2 margin at the flipping rows, because a flip at a margin of 1e-4
     is a coin that was already balanced on its edge.

Timing is measured against BOTH the promoted gqa_pair arm and the qrow16
reference at the same four context lengths the fit was solved on, so the
measurement can be checked against treeattn_v2_design.md 12's prediction rather
than merely reported.

Usage:
  python3 fr14_splitk_fa2_probe.py --so <candidate.so> [--json out.json]
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
ARMS = {
    "qrow16": {"sentinel": 1179791667, "num_splits": 0,
               "ctas_b1": 48, "block_m": 16, "warps": 1, "kv_lanes": 12,
               "context_splits": 1},
    "gqa_pair": {"sentinel": 1179791670, "num_splits": 0,
                 "ctas_b1": 12, "block_m": 64, "warps": 4, "kv_lanes": 3,
                 "context_splits": 1},
    # FR14 Tier-B: same traits as gqa_pair, context walk split four ways.
    # 3 head pairs * 4 splits * 4 KV heads = 48 CTAs; each CTA still stages one
    # tile for two heads, so staged bytes per step are gqa_pair's, not qrow16's.
    "gqa_pair_splitk": {"sentinel": 1179791671, "num_splits": 4,
                        "ctas_b1": 48, "block_m": 64, "warps": 4,
                        "kv_lanes": 3, "context_splits": 4},
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

# The logit-like projection. Not the served o_proj/lm_head -- those live in an
# NVFP4 checkpoint this probe deliberately does not load -- but a projection of
# the same SHAPE and conditioning, so the flip rate it reports is a plausible
# per-position flip rate rather than an arbitrary one. It is fixed and seeded,
# so both arms see exactly the same projection and the comparison is clean.
PROJ_HIDDEN = 4096
PROJ_VOCAB = 32768
PROJ_SEED = 20260818


# OPERAND SCALE. The banked byte probe drew q, k and v as randn * 0.1. For a
# BYTE gate that is fine -- byte identity does not care what the numbers are.
# For a Tier-B characterization it is the wrong regime and a badly lenient one:
# at 0.1 the pre-softmax logits have std ~0.01, i.e. an essentially UNIFORM
# attention distribution, which is the easiest possible case for a split-K
# rescale because every split's running max is nearly the same.
#
# Real attention is peaked. Measured from 16 banked FR13_TREE_ATTN_OP_CAPTURE
# artifacts of the served model (fr13_wy_gateA_20260608T163915Z, layer 7 among
# others; same softmax scale 1/sqrt(256) = 0.0625 this probe uses):
#
#     query std  1.234  (1.145 .. 1.291)
#     key   std  1.404  (1.185 .. 1.570)
#     value std  1.731  (0.685 .. 5.740)
#     pre-softmax logit std ~3.6, max ~+13, min ~-8
#
# In that regime one split routinely holds the global max and the others are
# rescaled by exp(-10) or smaller -- the case where a reassociated softmax can
# actually lose bits. Both regimes are run and reported separately, so the
# 0.1 numbers remain comparable with the banked byte probe and the "real"
# numbers are the ones the verdict rests on.
OPERAND_SCALES = {
    "legacy0p1": {"q": 0.1, "k": 0.1, "v": 0.1,
                  "note": "the banked byte probe's scale; near-uniform softmax"},
    "captured": {"q": 1.234, "k": 1.404, "v": 1.731,
                 "note": "measured from banked TREE_ATTN op captures of the "
                         "served model"},
}


def build_case(seq_len: int, seed: int, device: str = "cuda",
               scale: str = "captured"):
    """One decode step's worth of real operands at the canonical B1 geometry."""
    gen = torch.Generator(device=device).manual_seed(seed)
    num_blocks = math.ceil(seq_len / PAGE) + 1
    s = OPERAND_SCALES[scale]

    q = torch.randn(TREE_ROWS, Q_HEADS, HEAD_DIM, generator=gen,
                    device=device, dtype=torch.bfloat16) * s["q"]

    # ILKV layout: [num_blocks, 2, PAGE, KV_HEADS, HEAD_DIM]. Slicing dim 1
    # gives k/v the (2*PAGE*KV*D, KV*D, D, 1) strides the gate demands.
    kv = torch.randn(num_blocks, 2, PAGE, KV_HEADS, HEAD_DIM, generator=gen,
                     device=device, dtype=torch.bfloat16)
    kv[:, 0] *= s["k"]
    kv[:, 1] *= s["v"]
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


def run_arm(case, name: str):
    a = ARMS[name]
    return run(case, a["sentinel"], a["num_splits"])


def raw(t: torch.Tensor):
    """Raw bytes as a uint8 array -- the comparison is on BYTES, not values, so
    NaN/-0.0 cannot launder a mismatch past it."""
    return t.detach().contiguous().view(torch.uint8).cpu().numpy()


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(raw(t).tobytes()).hexdigest()[:16]


def byte_mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        raise RuntimeError("comparison contract drifted")
    return int((raw(a) != raw(b)).sum())


def _monotone_bits(t: torch.Tensor) -> torch.Tensor:
    """Map float bits to a monotone integer ordering.

    IEEE-754 floats of the same width compare like sign-magnitude integers, so
    flipping the ordering of the negative half turns "adjacent representable
    values" into "adjacent integers". That is what makes the difference below a
    ULP COUNT -- one step of the actual float grid -- rather than a relative
    error dressed up as one.
    """
    if t.dtype == torch.bfloat16:
        bits = t.contiguous().view(torch.int16).to(torch.int64)
        sign_floor = 1 << 15
    elif t.dtype == torch.float32:
        bits = t.contiguous().view(torch.int32).to(torch.int64)
        sign_floor = 1 << 31
    else:
        raise RuntimeError(f"no ULP ordering defined for {t.dtype}")
    return torch.where(bits < 0, sign_floor - bits, bits)


def ulp_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """ULP distance distribution between two same-shaped float tensors."""
    finite = torch.isfinite(a.float()) & torch.isfinite(b.float())
    d = (_monotone_bits(a) - _monotone_bits(b)).abs()
    d_finite = d[finite]
    abs_delta = (a.float() - b.float()).abs()
    abs_delta_finite = abs_delta[finite]
    ref_mag = a.float().abs()[finite]
    hist = {}
    for k in (0, 1, 2, 3, 4):
        hist[str(k)] = int((d_finite == k).sum())
    hist[">4"] = int((d_finite > 4).sum())
    return {
        "elements": int(finite.numel()),
        "finite_elements": int(finite.sum()),
        "nonfinite_disagreements": int(
            (torch.isfinite(a.float()) != torch.isfinite(b.float())).sum()),
        "byte_mismatches": byte_mismatches(a, b),
        "max_ulp": int(d_finite.max()) if d_finite.numel() else 0,
        "mean_ulp": float(d_finite.double().mean()) if d_finite.numel() else 0.0,
        "p99_ulp": (int(torch.quantile(d_finite.double(), 0.99).item())
                    if d_finite.numel() else 0),
        "ulp_histogram": hist,
        "max_abs_delta": (float(abs_delta_finite.max())
                          if abs_delta_finite.numel() else 0.0),
        "max_rel_delta": (float((abs_delta_finite
                                 / ref_mag.clamp_min(1e-30)).max())
                          if abs_delta_finite.numel() else 0.0),
        "reference_max_abs": (float(ref_mag.max())
                              if ref_mag.numel() else 0.0),
    }


def exact_attention(case, device: str = "cuda"):
    """A float64 dense reference for the SAME operands the kernels see.

    This is the number that decides whether a Tier-B reassociation is a
    degradation or an improvement. "Split-K differs from the served kernel" is
    not by itself a finding -- the served kernel is not exact either. What
    matters is which one is CLOSER to the attention both are approximating, and
    for that a reference outside both is required.

    Built to the kernel's own contract, not to a textbook: the tree bias lands
    on columns [seqlen_k - 32, seqlen_k), which is exactly the window
    apply_tree_bias() computes from `context_len = actual_seqlen_k -
    query_rows`, and the scale is the same 1/sqrt(256).
    """
    seq_len = case["seq_len"]
    k = case["k"].reshape(-1, KV_HEADS, HEAD_DIM)[:seq_len].double()
    v = case["v"].reshape(-1, KV_HEADS, HEAD_DIM)[:seq_len].double()
    q = case["q"].double()
    bias = case["tree_bias"].double()
    scale = 1.0 / math.sqrt(HEAD_DIM)
    context_len = seq_len - TREE_ROWS
    out = torch.empty(TREE_ROWS, Q_HEADS, HEAD_DIM, device=device,
                      dtype=torch.float64)
    lse = torch.empty(Q_HEADS, TREE_ROWS, device=device, dtype=torch.float64)
    for h in range(Q_HEADS):
        kv = h // (Q_HEADS // KV_HEADS)
        s = (q[:, h, :] @ k[:, kv, :].T) * scale        # (32, seq_len)
        s[:, context_len:context_len + TREE_ROWS] += bias
        m = s.max(dim=-1, keepdim=True).values
        e = (s - m).exp()
        denom = e.sum(dim=-1, keepdim=True)
        out[:, h, :] = (e / denom) @ v[:, kv, :]
        lse[h, :] = (m.squeeze(-1) + denom.squeeze(-1).log())
        del s, e
    return out, lse


def error_vs_exact(arm_out, arm_lse, exact_out, exact_lse) -> dict:
    """How far an arm's bf16/fp32 result sits from the float64 truth.

    The bf16 quantization floor is reported beside it, because an output error
    at or below the floor means the arm is as close to exact as a bf16 tensor
    can represent, and the remaining difference between two such arms is not a
    difference in accuracy at all.
    """
    a = arm_out.double()
    d = (a - exact_out).abs()
    floor = (exact_out.to(torch.bfloat16).double() - exact_out).abs()
    dl = (arm_lse.double() - exact_lse).abs()
    return {
        "output_max_abs_error": float(d.max()),
        "output_rms_error": float(d.pow(2).mean().sqrt()),
        "output_bf16_quantization_floor_max": float(floor.max()),
        "output_errors_at_or_below_floor": float(
            (d <= floor + 1e-30).double().mean()),
        "lse_max_abs_error": float(dl.max()),
        "lse_rms_error": float(dl.pow(2).mean().sqrt()),
    }


def make_projection(device: str = "cuda"):
    """A fixed logit-like projection: (heads*d) -> hidden -> vocab.

    Scaled 1/sqrt(fan_in) at each stage so the logit spread lands in the range a
    real head produces, which is what makes the top1-top2 margins -- and hence
    the flip rate -- interpretable.
    """
    gen = torch.Generator(device=device).manual_seed(PROJ_SEED)
    o_proj = torch.randn(Q_HEADS * HEAD_DIM, PROJ_HIDDEN, generator=gen,
                         device=device, dtype=torch.bfloat16)
    o_proj *= (Q_HEADS * HEAD_DIM) ** -0.5
    lm_head = torch.randn(PROJ_HIDDEN, PROJ_VOCAB, generator=gen,
                          device=device, dtype=torch.bfloat16)
    lm_head *= PROJ_HIDDEN ** -0.5
    return o_proj, lm_head


def logits_of(attn_out: torch.Tensor, proj) -> torch.Tensor:
    o_proj, lm_head = proj
    hidden = (attn_out.reshape(TREE_ROWS, Q_HEADS * HEAD_DIM) @ o_proj)
    return (hidden @ lm_head).float()


def argmax_flips(ref_out, cand_out, proj) -> dict:
    lr = logits_of(ref_out, proj)
    lc = logits_of(cand_out, proj)
    ar = lr.argmax(dim=-1)
    ac = lc.argmax(dim=-1)
    flipped = ar != ac
    top2 = lr.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1])
    return {
        "rows": int(ar.numel()),
        "flips": int(flipped.sum()),
        "reference_margin_min": float(margin.min()),
        "reference_margin_median": float(margin.median()),
        "flipped_margin_max": (float(margin[flipped].max())
                               if int(flipped.sum()) else None),
        "logit_max_abs_delta": float((lr - lc).abs().max()),
        "logit_scale_p99": float(lr.abs().flatten()
                                 .quantile(0.99)),
    }


def time_arm(case, name, iters=30, warmup=10):
    a = ARMS[name]
    for _ in range(warmup):
        run(case, a["sentinel"], a["num_splits"])
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        run(case, a["sentinel"], a["num_splits"])
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def staged_gb_per_step(ctas: int, seq_len: int, context_splits: int = 1) -> float:
    """Bytes staged into smem per step.

    A split-K CTA walks only 1/context_splits of the tiles, so the arm's staged
    bytes are (CTAs / splits) x tiles x tile_bytes -- which is why split-K buys
    parallelism WITHOUT buying back the re-staging that head-merging was priced
    against.
    """
    n_blocks = math.ceil(seq_len / BLOCK_N)
    tile = BLOCK_N * HEAD_DIM * 2 * 2  # K and V, bf16
    return (ctas / context_splits) * n_blocks * tile * FULL_ATTN_LAYERS / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--json")
    ap.add_argument("--reference", default="gqa_pair")
    ap.add_argument("--candidate", default="gqa_pair_splitk")
    ap.add_argument("--seq-lens", default="20480,23000,32768,40960")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--determinism-reps", type=int, default=8)
    ap.add_argument("--exact-seq-lens", default="20480",
                    help="context lengths at which to build the float64 dense "
                         "reference; empty disables it")
    ap.add_argument("--exact-seeds", type=int, default=2)
    ap.add_argument("--scales", default="captured,legacy0p1",
                    help="comma-separated operand-scale regimes to "
                         "characterize; the first is used for timing")
    ap.add_argument("--process-tag", default="p0",
                    help="label for this process; run twice and diff the "
                         "recorded digests to test across processes")
    args = ap.parse_args()

    torch.ops.load_library(args.so)
    sha = hashlib.sha256(open(args.so, "rb").read()).hexdigest()
    report = {
        "schema": "fr14.splitk_fa2.probe.v1",
        "so": args.so, "so_sha256": sha,
        "process_tag": args.process_tag,
        "reference_arm": args.reference, "candidate_arm": args.candidate,
        "arms": {n: ARMS[n] for n in (args.reference, args.candidate)},
        "geometry": {"tree_rows": TREE_ROWS, "q_heads": Q_HEADS,
                     "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
                     "page": PAGE, "block_n": BLOCK_N,
                     "full_attn_layers": FULL_ATTN_LAYERS},
        "projection": {"hidden": PROJ_HIDDEN, "vocab": PROJ_VOCAB,
                       "seed": PROJ_SEED},
        "operand_scales": OPERAND_SCALES,
        "determinism": [], "characterization": [], "timing": [],
    }
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    scales = [s for s in args.scales.split(",") if s]
    for s in scales:
        if s not in OPERAND_SCALES:
            raise SystemExit(f"unknown operand scale {s!r}")
    proj = make_projection()

    # --- 1. DETERMINISM GATE (hard) ------------------------------------
    # Each call allocates its own split accumulators, so their device
    # addresses differ between repeats; a reduction that depended on
    # allocation order or on atomics would diverge here.
    determinism_pass = True
    for scale in scales:
      for seq_len in seq_lens:
        for seed in range(min(args.seeds, 2)):
            case = build_case(seq_len, seed, scale=scale)
            o_digests, l_digests = [], []
            for rep in range(args.determinism_reps):
                o, l = run_arm(case, args.candidate)
                o_digests.append(digest(o))
                l_digests.append(digest(l))
                # Interleave the reference arm so the candidate never sees the
                # same allocator or launch state twice.
                run_arm(case, args.reference)
                del o, l
            ok = (len(set(o_digests)) == 1 and len(set(l_digests)) == 1)
            determinism_pass = determinism_pass and ok
            report["determinism"].append({
                "scale": scale, "seq_len": seq_len, "seed": seed,
                "reps": args.determinism_reps,
                "output_sha16": o_digests[0], "lse_sha16": l_digests[0],
                "distinct_output_digests": len(set(o_digests)),
                "distinct_lse_digests": len(set(l_digests)),
                "bitwise_identical": ok,
            })
            del case
            torch.cuda.empty_cache()
    report["determinism_pass"] = determinism_pass

    # --- 2. ULP CHARACTERIZATION vs the served arm ---------------------
    per_scale = {}
    for scale in scales:
      worst = {"output_max_ulp": 0, "lse_max_ulp": 0,
               "output_max_abs_delta": 0.0, "lse_max_abs_delta": 0.0}
      flips_total = flip_rows_total = 0
      for seq_len in seq_lens:
        for seed in range(args.seeds):
            case = build_case(seq_len, seed, scale=scale)
            o_r, l_r = run_arm(case, args.reference)
            o_c, l_c = run_arm(case, args.candidate)
            row = {
                "scale": scale, "seq_len": seq_len, "seed": seed,
                "output": ulp_stats(o_r, o_c),
                "lse": ulp_stats(l_r, l_c),
                "argmax": argmax_flips(o_r, o_c, proj),
            }
            report["characterization"].append(row)
            worst["output_max_ulp"] = max(worst["output_max_ulp"],
                                          row["output"]["max_ulp"])
            worst["lse_max_ulp"] = max(worst["lse_max_ulp"],
                                       row["lse"]["max_ulp"])
            worst["output_max_abs_delta"] = max(
                worst["output_max_abs_delta"], row["output"]["max_abs_delta"])
            worst["lse_max_abs_delta"] = max(
                worst["lse_max_abs_delta"], row["lse"]["max_abs_delta"])
            flips_total += row["argmax"]["flips"]
            flip_rows_total += row["argmax"]["rows"]
            del case, o_r, l_r, o_c, l_c
            torch.cuda.empty_cache()
      per_scale[scale] = dict(
          worst,
          argmax_flip_rows=flips_total,
          argmax_total_rows=flip_rows_total,
          argmax_flip_rate=(flips_total / flip_rows_total
                            if flip_rows_total else 0.0),
      )
    report["characterization_summary"] = per_scale

    # --- 2b. WHICH ARM IS CLOSER TO EXACT? -----------------------------
    # The decisive comparison, and the one a byte gate can never make: both
    # arms approximate the same attention, so the question is not whether they
    # differ from each other but which sits closer to the float64 truth.
    report["exact_reference"] = []
    exact_lens = [int(s) for s in args.exact_seq_lens.split(",") if s.strip()]
    for scale in scales:
        for seq_len in exact_lens:
            for seed in range(args.exact_seeds):
                case = build_case(seq_len, seed, scale=scale)
                ex_o, ex_l = exact_attention(case)
                row = {"scale": scale, "seq_len": seq_len, "seed": seed}
                for name in (args.reference, args.candidate):
                    o, l = run_arm(case, name)
                    row[name] = error_vs_exact(o, l, ex_o, ex_l)
                    del o, l
                row["candidate_closer_output"] = bool(
                    row[args.candidate]["output_rms_error"]
                    <= row[args.reference]["output_rms_error"])
                row["candidate_closer_lse"] = bool(
                    row[args.candidate]["lse_rms_error"]
                    <= row[args.reference]["lse_rms_error"])
                report["exact_reference"].append(row)
                del case, ex_o, ex_l
                torch.cuda.empty_cache()

    # --- 3. TIMING vs the served arm and the qrow16 reference ----------
    timing_scale = scales[0]
    report["timing_scale"] = timing_scale
    for seq_len in seq_lens:
        case = build_case(seq_len, 0, scale=timing_scale)
        row = {"seq_len": seq_len}
        for name in ("qrow16", args.reference, args.candidate):
            a = ARMS[name]
            ms = time_arm(case, name)
            row[name] = {
                "ms_per_layer_call": ms,
                "ms_per_step_16_layers": ms * FULL_ATTN_LAYERS,
                "ctas_b1": a["ctas_b1"],
                "context_splits": a["context_splits"],
                "staged_gb_per_step": round(
                    staged_gb_per_step(a["ctas_b1"], seq_len,
                                       a["context_splits"]), 3),
            }
        report["timing"].append(row)
        del case
        torch.cuda.empty_cache()

    # --- 4. the fit, and what it predicted for 48 CTAs -----------------
    # T = alpha*staged_GB + beta/CTAs, solved on the SAME two arms
    # treeattn_v2_design.md section 8 solved it on, then evaluated at this
    # arm's (staged bytes, CTAs). Reported beside the measurement, never
    # instead of it.
    for row in report["timing"]:
        r, c = row["qrow16"], row[args.reference]
        s1, s2 = r["staged_gb_per_step"], c["staged_gb_per_step"]
        p1, p2 = 1.0 / r["ctas_b1"], 1.0 / c["ctas_b1"]
        t1, t2 = r["ms_per_step_16_layers"], c["ms_per_step_16_layers"]
        det = s1 * p2 - s2 * p1
        if abs(det) < 1e-12:
            continue
        alpha = (t1 * p2 - t2 * p1) / det
        beta = (s1 * t2 - s2 * t1) / det
        cand = row[args.candidate]
        predicted = alpha * cand["staged_gb_per_step"] + beta / cand["ctas_b1"]
        row["fit"] = {
            "alpha_ms_per_staged_gb": round(alpha, 4),
            "beta_ms_times_ctas": round(beta, 4),
            "predicted_candidate_ms_per_step": round(predicted, 3),
            "measured_candidate_ms_per_step": round(
                cand["ms_per_step_16_layers"], 3),
            "measured_minus_predicted_ms": round(
                cand["ms_per_step_16_layers"] - predicted, 3),
            "measured_delta_vs_reference_ms": round(
                cand["ms_per_step_16_layers"]
                - c["ms_per_step_16_layers"], 3),
        }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        open(args.json, "w").write(text + "\n")
    print("== FR14 split-K FA2 probe ==")
    print(f"  determinism_pass      : {report['determinism_pass']}")
    for scale, summary in report["characterization_summary"].items():
        print(f"  -- operand scale: {scale} --")
        print(f"     output max/mean ulp : {summary['output_max_ulp']}")
        print(f"     lse    max ulp      : {summary['lse_max_ulp']}")
        print(f"     output max abs delta: {summary['output_max_abs_delta']:.3e}")
        print(f"     lse    max abs delta: {summary['lse_max_abs_delta']:.3e}")
        print(f"     argmax flips        : {summary['argmax_flip_rows']}"
              f"/{summary['argmax_total_rows']}"
              f" ({summary['argmax_flip_rate']:.4%})")
    for row in report["exact_reference"]:
        r, c = row[args.reference], row[args.candidate]
        print(f"  exact  scale={row['scale']:<10s} L={row['seq_len']} "
              f"seed={row['seed']}  rms_err {args.reference}="
              f"{r['output_rms_error']:.3e}  {args.candidate}="
              f"{c['output_rms_error']:.3e}  "
              f"candidate_closer={row['candidate_closer_output']}")
    for row in report["timing"]:
        f = row.get("fit", {})
        print(f"  L={row['seq_len']:6d}  qrow16="
              f"{row['qrow16']['ms_per_step_16_layers']:7.3f}  "
              f"{args.reference}={row[args.reference]['ms_per_step_16_layers']:7.3f}  "
              f"{args.candidate}={row[args.candidate]['ms_per_step_16_layers']:7.3f}  "
              f"predicted={f.get('predicted_candidate_ms_per_step')}  "
              f"delta={f.get('measured_delta_vs_reference_ms')}")
    if args.json:
        print(f"  json -> {args.json}")
    # Determinism is the only hard gate here. The ULP and flip numbers are
    # CHARACTERIZATION: they are for the note and for Mark, not for a
    # threshold this script invents.
    return 0 if report["determinism_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
