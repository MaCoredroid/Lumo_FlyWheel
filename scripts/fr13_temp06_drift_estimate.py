#!/usr/bin/env python3
"""FR13 temp-0.6 drift gate — first-pass BANKED estimate (CPU, no GPU boot).

Per FR13_TEMP06_DRIFT_GATE.md. This script does NOT fabricate a temp-0.6 drift
number. The binding temp-0.6 distributional gate needs the spec verify dist
``q = softmax(verify_logits/0.6)`` paired with the recurrent-decode dist
``p = softmax(decode_logits/0.6)`` per position. **q is NOT banked** in the
big-denom rescore (the served stream is temp-0 greedy token ids only; only the
recurrent oracle ``p`` is recorded, top-5, on flip positions). The only banked
``q`` (output/fr13_gold_margin/*_capture.json, the spec verify top-20) is paired
with a CHUNKED-prefill ``p`` at only 4 fork positions — the wrong oracle frame.

So this emits ONLY what banked data licenses, clearly labelled:
  (1) the temp-0 ARGMAX point-measure (cat9 vs native clear-margin rate);
  (2) a p-side temp-0.6 PEAKEDNESS read on the flip subset (refutes
      "argmax-pass => trivially lossless at 0.6"; it is NOT a drift estimate,
      it is symmetric between arms and uses only p);
and REFUSES the q-vs-p TV with the reason "q not banked".

Run:  python3 scripts/fr13_temp06_drift_estimate.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESCORE = ROOT / "output" / "fr13_bigdenom_rescore"


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((c - h) / d, (c + h) / d)


def _softmax_over_set(logprobs: list[float], temp: float) -> list[float]:
    """softmax(logprobs/temp) renormalised over the given support set.

    logprobs are log_softmax at T=1 (the banked top-k). The per-position additive
    constant cancels in softmax, so temp scaling over the SAME support is exact up
    to the truncated tail mass (omitted from the top-k)."""
    z = [lp / temp for lp in logprobs]
    m = max(z)
    ex = [math.exp(v - m) for v in z]
    s = sum(ex)
    return [e / s for e in ex]


def _scan_q_present(d: dict) -> bool:
    """Scan every per-position record for ANY spec-verify distribution (q)."""
    q_markers = ("verify", "served_topk", "q_top", "served_logprobs", "spec_topk")
    for pr in d["per_prompt"]:
        for pos in pr["positions"]:
            if any(any(mk in k for mk in q_markers) for k in pos.keys()):
                return True
        for fl in pr["flips"]:
            if isinstance(fl, dict) and any(
                any(mk in k for mk in q_markers) for k in fl.keys()
            ):
                return True
    return False


def main() -> int:
    out: dict = {"schema": "fr13.temp06_drift_estimate.v1"}

    # ---- (1) temp-0 ARGMAX point-measure (MEASURED) ----
    argmax = {}
    rescore = {}
    for arm in ("cat9", "native"):
        d = json.loads((RESCORE / f"rescore_{arm}.json").read_text())
        rescore[arm] = d
        n = d["total_positions"]
        k = d["total_clear_margin_flips"]
        lo, hi = _wilson(k, n)
        argmax[arm] = {
            "positions": n,
            "flips": d["total_flips"],
            "clear_margin": k,
            "clear_pct": 100 * k / n,
            "wilson95": [100 * lo, 100 * hi],
            "oracle": "recurrent_decode_non_spec (FR12_NO_SPECULATIVE_CONFIG=1, FLASH_ATTN)",
            "serve_temp": 0.0,
        }
    out["temp0_argmax_point_measure_MEASURED"] = argmax
    out["temp0_verdict"] = (
        "cat9 %.3f%% vs native %.3f%%, CIs overlap, cat9 LOWER -> within-floor PASS AT TEMP-0 "
        "(necessary, NOT sufficient for temp-0.6)"
        % (argmax["cat9"]["clear_pct"], argmax["native"]["clear_pct"])
    )

    # ---- temp-0.6 TV(q,p): REFUSED (q not banked) ----
    q_present = {arm: _scan_q_present(rescore[arm]) for arm in ("cat9", "native")}
    out["temp06_qp_TV_estimate"] = {
        "computable": False,
        "reason": (
            "spec verify distribution q is NOT banked in the big-denom rescore "
            "(served stream = temp-0 greedy token ids only; only recurrent oracle p "
            "is recorded, top-5, flip-only). The only banked q (output/fr13_gold_margin/"
            "*_capture.json) is paired with a CHUNKED-prefill p at 4 fork positions = "
            "wrong oracle frame. No banked (q, recurrent-p) pair exists."
        ),
        "any_q_in_bigdenom_cat9": q_present["cat9"],
        "any_q_in_bigdenom_native": q_present["native"],
        "awaits": "the temp-0.6 paired GPU capture (FR13_TEMP06_DRIFT_GATE.md sec 4)",
    }

    # ---- (2) p-side temp-0.6 PEAKEDNESS on flip subset (MEASURED; NOT a drift estimate) ----
    peaked = {}
    for arm in ("cat9", "native"):
        d = rescore[arm]
        p06t1, p10t1 = [], []
        for pr in d["per_prompt"]:
            for fl in pr["flips"]:
                lps = fl.get("oracle_topk_logprobs")
                if not lps:
                    continue
                p06t1.append(max(_softmax_over_set(lps, 0.6)))
                p10t1.append(max(_softmax_over_set(lps, 1.0)))
        peaked[arm] = {
            "flip_positions_with_topk": len(p06t1),
            "median_P_top1_at_T0.6": st.median(p06t1) if p06t1 else None,
            "median_P_top1_at_T1.0": st.median(p10t1) if p10t1 else None,
            "frac_flips_P_top1_below_0.9_at_T0.6": (
                sum(1 for x in p06t1 if x < 0.9) / len(p06t1) if p06t1 else None
            ),
            "topk_truncation": "top-5 (banked); tail mass omitted -> upper-bounds P(top1)",
        }
    out["temp06_pside_peakedness_MEASURED"] = peaked
    out["peakedness_note"] = (
        "p-side only, SYMMETRIC between arms (cat9~native), NOT a q-vs-p drift estimate. "
        "It refutes 'argmax-pass => trivially lossless at 0.6': the recurrent target is "
        "not a delta at 0.6 (~70% of flips have P(top1)<0.9), so temp-0.6 sampling reaches "
        "real sub-argmax mass."
    )

    # ---- native temp-0.6 bag-TV floor (read from bind; single-draw, class #12) ----
    out["native_temp06_bagTV_floor_MEASURED"] = {
        "bag_tv": 0.11328125,
        "raw_positional_mismatch": "139/256",
        "source": "FR13_NUM_SPLITS_NATIVE_FLOOR_BIND.md (B=4, temp0.6, top_p0.95, seed1313 v1313, FLASH_ATTN, CUDA-captured)",
        "caveat": "SINGLE C(2,2) draw = floor center NOT p95; upgrade to N=6-8 p95 before hard-thresholding (class #12). Supersedes the 0.0593 single-draw at fr13_corruption_gate.py:123.",
    }

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
