#!/usr/bin/env python3
"""E1 campaign aggregate (frozen estimand/precision rule, e1/E1_FREEZE.md; launch red-team 'unfinished work'). Reads every
cell_result.json of a campaign root (attempt dirs cell_NN_bB_arm_batch_aK), keeps ONLY SEALED measured cells (terminal_seal.sealed, status VALID; a preliminary summary or an unsealed/INVALID attempt is
REFUSED — listed as a failed attempt, never counted; INSUFFICIENT_
SUPPORT and INVALID/NOT_QUALIFIED cells are listed, never replaced or lengthened), and computes per batch condition b in
{B1, B4}: the per-cell aggregate rate (tokens per wall-second, primary), the per-arm mean over the three blocks, and for each
contrast (tree − native-5, tree − native-11, native-11 − native-5) the paired whole-block differences d_1..d_3, a bootstrap
(10 000 resamples with replacement of the 3 paired blocks, fixed seed) 95 % percentile interval [L, U], and the precision
statistic (U − L) / (2 · R̄_b) with R̄_b = the mean native-5 rate over the three blocks (the baseline-arm denominator). A
contrast whose three paired blocks are not all measured is reported 'not estimable' (no unpaired replacement); a contrast
with (U − L)/(2 R̄_b) > 0.10 is reported 'imprecise' with its interval. Diagnostics (>1.5 s intervals, trimmed rates) are
carried through unchanged and never enter the primary. Writes <root>/aggregate.json (+ prints a table).
Usage: e1_aggregate.py <campaign_root> [--cells e1_cells.v1.json] [--target 0.10] [--resamples 10000] [--seed 20260921]"""
import argparse, glob, json, os, random, sys
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--cells", default=None); ap.add_argument("--target", type=float, default=0.10); ap.add_argument("--resamples", type=int, default=10000); ap.add_argument("--seed", type=int, default=20260921); a = ap.parse_args()
    cells_p = a.cells or os.path.join(a.root, "campaign_snapshot", "e1_cells.json"); cells = json.load(open(cells_p))["cells"]
    table = []
    for c in cells:
        base = "cell_%02d_b%d_%s_%s" % (c["index"], c["block"], c["arm"], c["batch"]); atts = sorted(d for d in os.listdir(a.root) if d.startswith(base + "_a") and os.path.isdir(os.path.join(a.root, d)))
        rec = {"index": c["index"], "block": c["block"], "arm": c["arm"], "batch": c["batch"], "attempts": atts, "status": "NOT_ATTEMPTED", "rate": None, "diag": None, "support": None}
        for d in atts:   # the SEALED measured attempt (at most one) wins; unsealed/INVALID attempts are listed as failed (F1)
            p = os.path.join(a.root, d, "cell_result.json")
            if not os.path.exists(p): rec["status"] = "FAILED_ATTEMPT_NO_TERMINAL_RESULT"; rec["attempt"] = d; continue
            r = json.load(open(p)); sealed = bool((r.get("terminal_seal") or {}).get("sealed"))
            if not sealed: rec["status"] = "FAILED_ATTEMPT_UNSEALED (refused: no terminal seal)"; rec["attempt"] = d; continue
            rec["status"] = r.get("status"); rec["attempt"] = d; rec["terminal_seal_utc"] = r["terminal_seal"].get("utc")
            if r.get("status") == "VALID" and not r.get("rate_not_eligible"): rec["rate"] = (r.get("primary") or {}).get("tokens_per_wall_second"); rec["support"] = {k: (r.get("primary") or {}).get(k) for k in ("n_usable", "sum_wall_s_unique_physical_steps", "sum_emitted_tokens_api_bound_pure_support")}; rec["diag"] = r.get("diagnostic_over_cap"); rec["floor"] = r.get("floor"); break
            if r.get("status") in ("INSUFFICIENT_SUPPORT", "NOT_QUALIFIED_NOT_TIMED"): break
        table.append(rec)
    out = {"campaign": a.root, "cells": table, "rule": "paired whole-boot blocks; B1 and B4 separately; bootstrap of the 3 paired blocks (with replacement, %d resamples, seed %d); precision = (U-L)/(2*mean native-5 rate); target %.2f; no unpaired replacement; INSUFFICIENT_SUPPORT/INVALID/NOT_QUALIFIED cells listed only" % (a.resamples, a.seed, a.target), "by_batch": {}}
    rng = random.Random(a.seed)
    for b in ("B1", "B4"):
        rates = {(r["arm"], r["block"]): r["rate"] for r in table if r["batch"] == b}
        arms = {arm: [rates.get((arm, k)) for k in (1, 2, 3)] for arm in ("native-5", "native-11", "tree")}
        means = {arm: (sum(v) / len(v) if all(x is not None for x in v) else None) for arm, v in arms.items()}
        res = {"per_arm_block_rates": arms, "per_arm_mean_over_blocks": means, "contrasts": {}}
        Rb = means.get("native-5")
        for name, (x, y) in {"tree_minus_native5": ("tree", "native-5"), "tree_minus_native11": ("tree", "native-11"), "native11_minus_native5": ("native-11", "native-5")}.items():
            ds = [(arms[x][k], arms[y][k]) for k in range(3)]
            if any(p is None or q is None for p, q in ds) or Rb is None: res["contrasts"][name] = {"status": "not estimable (a paired block is not measured on both arms; no unpaired replacement)", "paired_blocks": ds}; continue
            d = [p - q for p, q in ds]; boots = []
            for _ in range(a.resamples):
                smp = [d[rng.randrange(3)] for _ in range(3)]; boots.append(sum(smp) / 3)
            boots.sort(); L = boots[int(0.025 * a.resamples)]; U = boots[int(0.975 * a.resamples) - 1]; prec = (U - L) / (2 * Rb)
            res["contrasts"][name] = {"paired_block_differences": d, "mean_difference": sum(d) / 3, "bootstrap_95pct_interval": [L, U], "baseline_native5_mean_rate": Rb, "precision_half_width_over_baseline": prec, "target": a.target, "status": ("within target" if prec <= a.target else "IMPRECISE (target missed; reported with its interval; no conclusion fabricated)")}
        out["by_batch"][b] = res
    json.dump(out, open(os.path.join(a.root, "aggregate.json"), "w"), indent=1)
    for r in table: print("cell %2d b%d %-9s %s %-22s rate=%s" % (r["index"], r["block"], r["arm"], r["batch"], r["status"], (None if r["rate"] is None else round(r["rate"], 3))))
    for b, res in out["by_batch"].items():
        for n, c in res["contrasts"].items(): print(b, n, c.get("status"), c.get("bootstrap_95pct_interval"), c.get("precision_half_width_over_baseline"))
if __name__ == "__main__":
    main()
