#!/usr/bin/env python3
"""Reduce the FR13_APC_CACHE_AB_GENREGION probe JSONL into a per-layer verdict.

This is the SOUND APC-losslessness reducer for the GENERATED-region two-phase
gate. It matches cache-ON records (req=R2: the GDN boundary state APC RESTORES on
a cache-HIT re-prefill of a GENERATED block, h_on = the FR13_APC_SSM_LEAF_SRC
value) against cache-OFF records (req=R3: R3's NATIVE chunked cache-OFF running
state, last_recurrent_state at the boundary-completing step, h_off = the true
incumbent with chunk-carry) at the SAME ABSOLUTE boundary_pos. APC is lossless
iff |h_on - h_off| ~ 0 within bf16 ULP AND argmax(h_on) == argmax(h_off) across
all GDN layers at the generated-region boundary.

Match (req-keyed, absolute-boundary keyed; NO total_len-suffix arithmetic):
  * ON records (phase="on", req == --on-req): boundary_pos = the absolute
    int(context_lens[r]) the restore happened at.
  * OFF records (phase="off", req == --off-req): boundary_pos = the absolute
    block boundary k*block the completing chunk landed on.
  * For each layer, join ON@(layer, target) to OFF@(layer, target) on the EXACT
    --target-boundary (e.g. 3072). Both must reference the same absolute token
    position. This replaces the round-1 boundary_pos = total_len - suffix_len
    derivation, which was unkeyed (ON total_len was None).

Guards (any failing => VACUOUS, never LOSSLESS/FAIL):
  * --target-boundary > --prompt-tokens  (the boundary is in the GENERATED
    region; otherwise the gate trivially certifies prefill-cache where the
    SSM_LEAF_SRC fix is irrelevant).
  * n_matched_at_target > 0  (an ON and an OFF record both exist at exactly the
    target boundary; defeats round-1's "PASS off a 1024/2048 match while 3072
    unmeasured").
  * an OFF record exists at exactly --target-boundary.
  * ON record's ssm_max_abs > 0 (a real restored nonzero boundary).

Verdict per (layer, target):
  fp_max_abs   = max_i |h_on.ssm_fp[i] - h_off.ssm_fp[i]| over the >=64-element
                 widened stride sample (exact per-element fp32 diff at the SAME
                 flatten indices in both arms);
  argmax_match = (ssm_argmax_idx_on == ssm_argmax_idx_off); the binding axis per
                 scalar-metric-blindspot. A nonzero argmax disagreement at the
                 target = NOT lossless. (A single-ULP flip at exactly one channel
                 with otherwise within-ULP fp is the FR13 FA2 floor ambiguity --
                 reported as ARGMAX_FLIP for user escalation, not auto-FAIL.)
  scalar gaps  = |ssm_max_abs_on - ssm_max_abs_off|, |ssm_sum_abs_on -
                 ssm_sum_abs_off| (secondary sanity).
LOSSLESS iff fp_max_abs <= ulp AND argmax matches across all layers at target,
with all guards satisfied.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def _fp_max_abs(a: list | None, b: list | None) -> float | None:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return max(abs(float(a[i]) - float(b[i])) for i in range(n))


def _stats(xs: list[float]) -> dict[str, Any]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "max": max(xs),
        "median": statistics.median(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", help="fr13_apc_cache_ab.jsonl from the probe")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--ulp", type=float, default=0.0078125,
        help="bf16-ULP-ish lossless threshold on the per-element fp diff "
             "(default 0.0078125 ~ 1 bf16 ULP at O(1) magnitude)",
    )
    ap.add_argument(
        "--prompt-tokens", type=int, required=True,
        help="len(P_token_ids); the matched boundary MUST be > this (the "
             "GENERATED region guard).",
    )
    ap.add_argument(
        "--target-boundary", type=int, required=True,
        help="the absolute block boundary to certify (e.g. 3072). ON and OFF "
             "must BOTH have a record at exactly this boundary.",
    )
    ap.add_argument(
        "--on-req", default="R2",
        help="the driver-pinned req_id for the cache-ON (h_on) phase.",
    )
    ap.add_argument(
        "--off-req", default="R3",
        help="the driver-pinned req_id for the cache-OFF (h_off) phase.",
    )
    args = ap.parse_args()

    target = int(args.target_boundary)
    prompt_tokens = int(args.prompt_tokens)

    recs = _load(Path(args.jsonl))

    # Phase + req discriminated. Records before v2 lacked req_id; if req_id is
    # absent we still match by phase (single-req boots), but the v2 probe stamps
    # it and we prefer the req gate.
    def _req_ok(r: dict, want: str) -> bool:
        rid = r.get("req_id")
        if rid is None:
            return True  # legacy / single-req: phase already discriminates
        return str(rid) == str(want)

    on = [
        r for r in recs
        if r.get("phase") == "on" and _req_ok(r, args.on_req)
    ]
    off = [
        r for r in recs
        if r.get("phase") == "off" and _req_ok(r, args.off_req)
    ]

    # Index OFF by (layer, boundary_pos).
    off_by_key: dict[tuple, dict] = {}
    for r in off:
        bp = r.get("boundary_pos")
        if bp is None:
            continue
        off_by_key[(str(r.get("layer")), int(bp))] = r

    # ---- vacuity guards (computed up-front; reported even if no match) ----
    generated_region_ok = target > prompt_tokens
    off_exists_at_target = any(
        int(r.get("boundary_pos")) == target
        for r in off
        if r.get("boundary_pos") is not None
    )
    on_exists_at_target = any(
        int(r.get("boundary_pos")) == target
        for r in on
        if r.get("boundary_pos") is not None
    )

    matched: list[dict[str, Any]] = []
    unmatched_on: list[dict[str, Any]] = []
    for r in on:
        bp = r.get("boundary_pos")
        if bp is None:
            unmatched_on.append({
                "layer": str(r.get("layer")),
                "reason": "ON record has no boundary_pos",
            })
            continue
        bp = int(bp)
        if bp != target:
            # Only certify at the target boundary. Off-target ON records are
            # ignored (not a failure; the gate is target-specific).
            continue
        L = str(r.get("layer"))
        o = off_by_key.get((L, target))
        if o is None:
            unmatched_on.append({
                "layer": L,
                "boundary_pos": bp,
                "reason": "no OFF record at (layer, target)",
            })
            continue
        fp_d = _fp_max_abs(r.get("ssm_fp"), o.get("ssm_fp"))
        amax_on = r.get("ssm_argmax_idx")
        amax_off = o.get("ssm_argmax_idx")
        argmax_match = (
            amax_on is not None and amax_off is not None
            and int(amax_on) == int(amax_off)
        )
        matched.append({
            "layer": L,
            "boundary_pos": bp,
            "state_row_on": r.get("state_row"),
            "state_row_off": o.get("state_row"),
            "block_size": r.get("block_size"),
            "ssm_fp_max_abs": fp_d,
            "argmax_match": bool(argmax_match),
            "ssm_argmax_idx_on": amax_on,
            "ssm_argmax_idx_off": amax_off,
            "ssm_argmax_val_on": r.get("ssm_argmax_val"),
            "ssm_argmax_val_off": o.get("ssm_argmax_val"),
            "ssm_numel_on": r.get("ssm_numel"),
            "ssm_numel_off": o.get("ssm_numel"),
            "n_fp_elems": (
                min(len(r.get("ssm_fp") or []), len(o.get("ssm_fp") or []))
            ),
            "ssm_max_abs_on": r.get("ssm_max_abs"),
            "ssm_max_abs_gap": (
                None if r.get("ssm_max_abs") is None or o.get("ssm_max_abs") is None
                else abs(float(r["ssm_max_abs"]) - float(o["ssm_max_abs"]))
            ),
            "ssm_sum_abs_gap": (
                None if r.get("ssm_sum_abs") is None or o.get("ssm_sum_abs") is None
                else abs(float(r["ssm_sum_abs"]) - float(o["ssm_sum_abs"]))
            ),
        })

    n_matched_at_target = len(matched)

    # per-layer rollup of the per-element fp diff (the binding axis).
    by_layer: dict[str, list[float]] = {}
    argmax_flips: list[dict[str, Any]] = []
    for m in matched:
        if m["ssm_fp_max_abs"] is not None:
            by_layer.setdefault(m["layer"], []).append(m["ssm_fp_max_abs"])
        if not m["argmax_match"]:
            argmax_flips.append({
                "layer": m["layer"],
                "boundary_pos": m["boundary_pos"],
                "argmax_idx_on": m["ssm_argmax_idx_on"],
                "argmax_idx_off": m["ssm_argmax_idx_off"],
                "fp_max_abs": m["ssm_fp_max_abs"],
            })
    per_layer = {L: _stats(v) for L, v in sorted(by_layer.items())}

    fp_all = [m["ssm_fp_max_abs"] for m in matched if m["ssm_fp_max_abs"] is not None]
    maxgap_all = [m["ssm_max_abs_gap"] for m in matched if m["ssm_max_abs_gap"] is not None]
    sumgap_all = [m["ssm_sum_abs_gap"] for m in matched if m["ssm_sum_abs_gap"] is not None]

    overall_fp_max = max(fp_all) if fp_all else None
    all_argmax_match = all(m["argmax_match"] for m in matched) if matched else False
    # ON restored boundary must be a real nonzero seed.
    on_nonzero = all(
        (m["ssm_max_abs_on"] is None) or (float(m["ssm_max_abs_on"]) > 0.0)
        for m in matched
    ) if matched else False

    vacuous = (
        not generated_region_ok
        or not off_exists_at_target
        or not on_exists_at_target
        or n_matched_at_target <= 0
    )

    lossless = (
        (not vacuous)
        and overall_fp_max is not None
        and overall_fp_max <= args.ulp
        and all_argmax_match
        and on_nonzero
    )

    # The single-channel-ULP-flip ambiguity (FR13 FA2 floor): fp within ULP but
    # argmax flips at >=1 layer -> escalate, do not auto-FAIL.
    argmax_flip_within_ulp = (
        (not vacuous)
        and overall_fp_max is not None
        and overall_fp_max <= args.ulp
        and not all_argmax_match
    )

    if vacuous:
        verdict_label = "VACUOUS"
    elif lossless:
        verdict_label = "LOSSLESS"
    elif argmax_flip_within_ulp:
        verdict_label = "ARGMAX_FLIP_WITHIN_ULP_ESCALATE"
    else:
        verdict_label = "NOT_LOSSLESS"

    verdict = {
        "schema": "fr13.apc_cache_ab.reduce.v2",
        "src": str(args.jsonl),
        "verdict": verdict_label,
        "prompt_tokens": prompt_tokens,
        "target_boundary": target,
        "on_req": args.on_req,
        "off_req": args.off_req,
        "n_records": len(recs),
        "n_on": len(on),
        "n_off": len(off),
        "n_matched_at_target": n_matched_at_target,
        "n_unmatched_on": len(unmatched_on),
        "ulp_threshold": args.ulp,
        "guards": {
            "generated_region_ok": bool(generated_region_ok),
            "off_record_exists_at_target": bool(off_exists_at_target),
            "on_record_exists_at_target": bool(on_exists_at_target),
            "n_matched_at_target_gt0": bool(n_matched_at_target > 0),
            "on_restored_nonzero": bool(on_nonzero),
        },
        "OVERALL_ssm_fp_max_abs": overall_fp_max,
        "OVERALL_ssm_fp_stats": _stats(fp_all),
        "OVERALL_ssm_max_abs_gap_stats": _stats(maxgap_all),
        "OVERALL_ssm_sum_abs_gap_stats": _stats(sumgap_all),
        "all_argmax_match": bool(all_argmax_match),
        "argmax_flips": argmax_flips[:40],
        "per_layer_ssm_fp_max_abs": per_layer,
        "LOSSLESS": bool(lossless),
        "VACUOUS": bool(vacuous),
        "verdict_note": (
            "LOSSLESS iff ON(req=" + str(args.on_req) + ")@(layer," + str(target)
            + ") restored boundary == OFF(req=" + str(args.off_req)
            + ")@(layer," + str(target) + ") native chunked cache-OFF "
            "last_recurrent_state within bf16 ULP AND per-channel argmax matches, "
            "with the boundary in the GENERATED region (> prompt_tokens) and an "
            "OFF record present at exactly the target. ssm_fp is the >=64-element "
            "widened stride sample (exact per-element diff). argmax is the binding "
            "axis; a within-ULP argmax flip is escalated, not auto-FAILED. SSM-"
            "STATE ONLY: conv-window losslessness is NOT certified here (separate "
            "gate)."
        ),
        "matched_sample": matched[:40],
        "unmatched_on_sample": unmatched_on[:20],
    }
    out = json.dumps(verdict, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
