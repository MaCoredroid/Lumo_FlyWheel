#!/usr/bin/env python3
"""FR13 B4 - attribute the width-4 'other' bucket, the last unnamed 17.9%.

DIAGNOSTIC. NOT CITABLE. Reads the CUPTI-attached 539-step width-4 capture, so
every absolute ms is profiler-perturbed and is an UPPER bound.

WHAT 'other' IS
---------------
`scripts/fr13_b4_prefill_gaps_reduce.py::_family` names nine kernel families and
sends everything else to "other": 64.43 s of the 360.19 s window (17.9%), split
38.38 s inside pure-decode steps and 26.05 s outside. Nothing has ever been
aimed at it because nothing had ever opened it. This tool opens it.

METHOD
------
Same step index, same family rule, same plain-sum convention as the parent
reducer (plain sum == union on this trace: 140.01 s vs 139.89 s). Each step is
tagged with its batch width read off the tree-FA2 launch grid -- gridY IS the
batch, proven by the absence of any flash_fwd_splitkv_combine_kernel
(num_splits == 1) and by the gridY=4 population being exactly 226 x 16 launches
against 225(+1 boundary) width-4 steps x 16 attention layers.

Width scaling is then MEASURED, not asserted: w4/w2 for each kernel. w1 has only
4 steps in the capture and is never used as a base.
"""
from __future__ import annotations
import argparse, bisect, collections, json, re, sqlite3
from pathlib import Path

SCHEMA = "fr13.b4_other_bucket.v1"
STEP_RANGE = "fr13.fixed32.step"
CAUSAL_SIG = ", (bool)1, (bool)0, (bool)0, (bool)0, (bool)1"
MAX_PLAUSIBLE_STEP_NS = 60_000_000_000

# width-4 reference points (see results/fr13_b4_width4_nsys_20260813)
W4_STEP_WALL_PROFILED_MS = 429.333
W4_STEP_WALL_SEALED_MS = 411.05


def _family(sn: str, dn: str) -> str:
    if sn == "flash_fwd_splitkv_kernel":
        return "FA2 causal" if CAUSAL_SIG in dn else "FA2 tree"
    if sn == "device_kernel" and "cutlass_3x_gemm_fp8_blockwise" in dn:
        return "GEMM fp8 blockwise"
    if sn == "_tree_gdn_path_kernel":
        return "GDN tree scan"
    if sn in {"chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "chunk_fwd_kernel_o",
              "recompute_w_u_fwd_kernel", "chunk_scaled_dot_kkt_fwd_kernel",
              "chunk_local_cumsum_scalar_kernel", "merge_16x16_to_64x64_inverse_kernel"}:
        return "GDN chunked"
    if sn == "fused_sigmoid_gating_delta_rule_update_kernel":
        return "GDN delta-rule"
    if sn == "kernel_unified_attention_2d":
        return "unified attention"
    if sn.startswith("nvjet") or sn == "Kernel2":
        return "bf16 GEMM"
    return "other"


# ---- sub-classification of the 'other' bucket -------------------------------
# Ordered; first match wins. Each class carries the reducibility band that the
# artifact defends in prose, NOT a number invented per kernel.
CLASSES: list[tuple[str, re.Pattern, tuple[float, float], str]] = [
    ("mandatory_fused_pipeline",
     re.compile(r"silu_and_mul_per_block_quant|per_token_group_quant_8bit"
                r"|_causal_conv1d_fwd_kernel|_fused_post_conv_kernel"
                r"|triton_(poi|red|per)_fused.*(rms_norm|quant|silu|mean_pow)"),
     (0.00, 0.25),
     "activation + fp8 quantisation + conv that any implementation of this model "
     "must run; already fused, already near its own traffic floor"),
    ("sampling_and_verification",
     re.compile(r"_topk_topp_kernel|cunn_SoftMax|apply_repetition_penalties"
                r"|tensor_kernel_scan_innermost_dim|RadixSort|radix_sort|sortKeyValue"
                r"|reduce_kernel.*(ArgMax|MaxOps|max_functor)"),
     (0.35, 0.70),
     "sampler math. The work is required; the SHAPE is not -- torch's generic "
     "cumsum/softmax path runs beside vLLM's already-fused _topk_topp_kernel"),
    ("copies_fills_cats",
     re.compile(r"direct_copy_kernel|FillFunctor|CatArrayBatchedCopy|batch_memcpy"
                r"|copy_kernel|_zero_kv_blocks"),
     (0.55, 0.90),
     "pure tensor plumbing: dtype/layout copies, buffer zero-fills, concatenations. "
     "No mandatory tensor exists that only these kernels can produce"),
    ("gather_scatter_index",
     re.compile(r"vectorized_gather_kernel|_scatter_gather_elementwise|indexSelect"
                r"|index_elementwise_kernel|IndexKernel|scatter"),
     (0.40, 0.75),
     "tree/committer bookkeeping: node gathers, accepted-path scatters, index "
     "selects. Real addressing work, but re-materialised per op instead of fused"),
    ("fr13_state_plumbing",
     re.compile(r"^_fr13_"),
     (0.20, 0.60),
     "fixed32-specific conv staging (pregather / direct col0). 1 launch/step each; "
     "already the consolidated form of what used to be 48"),
    ("elementwise_math",
     re.compile(r"elementwise_kernel|CUDAFunctor|BinaryFunctor|MulFunctor|DivFunctor"
                r"|AddFunctor|where_kernel|compare_scalar|unary|triton_poi_fused"
                r"|triton_red_fused|triton_per_fused"),
     (0.45, 0.80),
     "chained elementwise ops, each a separate DRAM round trip of a tensor the "
     "next op immediately re-reads; the classic fusion target"),
    ("reductions_norms",
     re.compile(r"reduce_kernel|norm|Norm"), (0.30, 0.65),
     "reductions and norms outside the fused triton path"),
    ("unclassified", re.compile(r".*"), (0.00, 0.50),
     "not matched by any rule; treated as mostly-mandatory to stay conservative"),
]


def classify(sn: str, dn: str) -> str:
    blob = f"{sn} {dn}"
    for name, pat, _, _ in CLASSES:
        if pat.search(blob):
            return name
    return "unclassified"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)

    steps = [(int(s), int(e)) for s, e in conn.execute(
        """SELECT n.start, n.end FROM NVTX_EVENTS n
           LEFT JOIN StringIds si ON si.id = n.textId
           WHERE COALESCE(si.value, n.text) = ? AND n.end IS NOT NULL
           ORDER BY n.start""", (STEP_RANGE,))]
    steps = sorted(s for s in steps if 0 < s[1] - s[0] <= MAX_PLAUSIBLE_STEP_NS)
    if not steps:
        raise SystemExit("FAIL-CLOSED: no decode step ranges")
    starts = [s for s, _ in steps]

    # width of each step := gridY of the non-causal tree FA2 launches inside it
    width: dict[int, int] = {}
    for t, gy in conn.execute(
            """SELECT k.start, k.gridY FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON s.id = k.shortName
               JOIN StringIds d ON d.id = k.demangledName
               WHERE s.value = 'flash_fwd_splitkv_kernel' AND d.value NOT LIKE ?
                 AND k.gridX = 1 AND k.gridZ = 24""", (f"%{CAUSAL_SIG}%",)):
        i = bisect.bisect_right(starts, int(t)) - 1
        if i >= 0 and int(t) < steps[i][1]:
            width[i] = max(width.get(i, 0), int(gy))
    hist = collections.Counter(width.values())
    if not hist:
        raise SystemExit("FAIL-CLOSED: could not width-tag any step")
    # fail-closed: the width histogram must reproduce the census one
    if sorted(hist) != [1, 2, 3, 4] or hist[4] != 225:
        raise SystemExit(f"FAIL-CLOSED: width histogram {dict(hist)} does not match census")

    fam = collections.defaultdict(lambda: collections.defaultdict(int))
    kern = collections.defaultdict(lambda: collections.defaultdict(int))
    kn = collections.defaultdict(lambda: collections.defaultdict(int))
    out_ns = collections.defaultdict(int)
    for st, dur, sn, dn in conn.execute(
            """SELECT k.start, k.end - k.start, s.value, d.value
               FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON s.id = k.shortName
               JOIN StringIds d ON d.id = k.demangledName"""):
        i = bisect.bisect_right(starts, st) - 1
        inside = i >= 0 and st < steps[i][1]
        f = _family(sn, dn)
        key = (sn, dn[:200])
        if not inside:
            if f == "other":
                out_ns[key] += dur
            continue
        w = width.get(i)
        if not w:
            continue
        fam[f][w] += dur
        if f == "other":
            kern[key][w] += dur
            kn[key][w] += 1

    n = {w: hist[w] for w in hist}
    per = lambda dd, w: dd.get(w, 0) / 1e6 / n.get(w, 1)

    rows = []
    for key in set(kern) | set(out_ns):
        dd = kern.get(key, {})
        sn, dn = key
        w2, w4 = per(dd, 2), per(dd, 4)
        rows.append({
            "short_name": sn, "demangled": dn, "class": classify(sn, dn),
            "ms_step_w1": per(dd, 1), "ms_step_w2": w2,
            "ms_step_w3": per(dd, 3), "ms_step_w4": w4,
            "inst_step_w4": kn[key].get(4, 0) / n.get(4, 1),
            "w4_over_w2": (w4 / w2) if w2 else None,
            "out_of_step_s": out_ns.get(key, 0) / 1e9,
        })
    rows.sort(key=lambda r: -r["ms_step_w4"])

    bands = {c: b for c, _, b, _ in CLASSES}
    notes = {c: t for c, _, _, t in CLASSES}
    cls = collections.defaultdict(lambda: {"ms_step_w4": 0.0, "ms_step_w2": 0.0,
                                           "inst_step_w4": 0.0, "kernels": 0,
                                           "out_of_step_s": 0.0})
    for r in rows:
        c = cls[r["class"]]
        c["ms_step_w4"] += r["ms_step_w4"]; c["ms_step_w2"] += r["ms_step_w2"]
        c["inst_step_w4"] += r["inst_step_w4"]; c["kernels"] += 1
        c["out_of_step_s"] += r["out_of_step_s"]
    for c, v in cls.items():
        lo, hi = bands[c]
        v["w4_over_w2"] = (v["ms_step_w4"] / v["ms_step_w2"]) if v["ms_step_w2"] else None
        v["width_scaling"] = ("width-scaling" if (v["w4_over_w2"] or 0) > 1.5
                              else "width-invariant" if (v["w4_over_w2"] or 0) < 1.2
                              else "sub-linear")
        v["reducible_band"] = [lo, hi]
        v["reducible_ms_step_w4"] = [v["ms_step_w4"] * lo, v["ms_step_w4"] * hi]
        v["note"] = notes[c]

    oos = sum(r["out_of_step_s"] for r in rows)
    if abs(oos - 26.048) > 0.05:
        raise SystemExit(f"FAIL-CLOSED: out-of-step 'other' is {oos:.3f} s, "
                         f"gaps.json publishes 26.05 s")
    tot_w4 = sum(v["ms_step_w4"] for v in cls.values())
    red_lo = sum(v["reducible_ms_step_w4"][0] for v in cls.values())
    red_hi = sum(v["reducible_ms_step_w4"][1] for v in cls.values())
    # CUPTI: the whole width-4 inflation is 429.333 - 411.05 = 18.28 ms/step.
    # Lower bound on the bucket = all of it charged to this bucket.
    cupti = W4_STEP_WALL_PROFILED_MS - W4_STEP_WALL_SEALED_MS
    share = tot_w4 / sum(sum(dd.get(4, 0) for dd in [d]) / 1e6 / n[4] for d in fam.values())

    doc = {
        "schema": SCHEMA, "acceptance_valid": False, "citable": False,
        "diagnostic_only": True, "source_sqlite": a.sqlite,
        "width_histogram": dict(sorted(hist.items())),
        "family_ms_step_by_width": {
            f: {f"w{w}": per(dd, w) for w in (1, 2, 3, 4)} for f, dd in fam.items()},
        "in_step_total_ms_step_by_width": {
            f"w{w}": sum(per(dd, w) for dd in fam.values()) for w in (1, 2, 3, 4)},
        "step_wall_ms_by_width_profiled": {
            f"w{w}": sum(steps[i][1] - steps[i][0] for i in width if width[i] == w)
                     / 1e6 / n.get(w, 1) for w in (1, 2, 3, 4)},
        "other_bucket": {
            "out_of_step_reconciles_to_gaps_json": abs(
                sum(r["out_of_step_s"] for r in rows) - 26.048) < 0.05,
            "ms_step_w4_profiled": tot_w4,
            "share_of_in_step_busy_w4": share,
            "share_of_profiled_step_wall_w4": tot_w4 / W4_STEP_WALL_PROFILED_MS,
            "distinct_kernels": len(rows),
            "inst_step_w4": sum(r["inst_step_w4"] for r in rows),
            "out_of_step_s": sum(r["out_of_step_s"] for r in rows),
            "cupti_bound": {
                "width4_cupti_inflation_ms_step": cupti,
                "ms_step_w4_lower_bound_all_cupti_charged_here": tot_w4 - cupti,
                "ms_step_w4_upper_bound_no_cupti_charged_here": tot_w4,
                "central_pro_rata": tot_w4 - cupti * share,
                "note": ("The parent attribution locates CUPTI cost almost entirely in the "
                         "wall residual, but this bucket holds ~3.06 M tiny kernel instances, "
                         "where per-launch profiler cost concentrates. Band, not point."),
            },
            "classes": dict(sorted(cls.items(), key=lambda kv: -kv[1]["ms_step_w4"])),
            "reducible_ms_step_w4": [red_lo, red_hi],
            "reducible_pct_of_width4_wall": [red_lo / W4_STEP_WALL_SEALED_MS * 100,
                                             red_hi / W4_STEP_WALL_SEALED_MS * 100],
        },
        "kernels": rows,
        "does_not_claim": [
            "No timing/acceptance claim. CUPTI-attached arm; absolute ms are upper bounds.",
            "Reducibility bands are ENGINEERING JUDGEMENT per class, defended in README.md; "
            "they are not measured. No candidate has been built or tested.",
            "A saving of X ms/step inside a decode step moves total window wall by ~0.59 X "
            "(gaps.json base_dilution).",
        ],
    }
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {a.out}")
    print(f"other bucket @w4 = {tot_w4:.3f} ms/step profiled "
          f"({tot_w4 / W4_STEP_WALL_PROFILED_MS * 100:.1f}% of the 429.3 ms profiled step)")
    print(f"reducible band  = {red_lo:.1f} .. {red_hi:.1f} ms/step "
          f"({red_lo / W4_STEP_WALL_SEALED_MS * 100:.1f}%..{red_hi / W4_STEP_WALL_SEALED_MS * 100:.1f}% of 411.05)")
    for c, v in sorted(cls.items(), key=lambda kv: -kv[1]["ms_step_w4"]):
        print(f"  {c:28s} {v['ms_step_w4']:7.3f} ms/step  w4/w2={v['w4_over_w2'] or 0:.2f} "
              f"{v['width_scaling']:16s} inst/step={v['inst_step_w4']:8.0f} "
              f"red={v['reducible_ms_step_w4'][0]:5.1f}..{v['reducible_ms_step_w4'][1]:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
