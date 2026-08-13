#!/usr/bin/env python3
"""The GQA-pair FA2 RE-TEST at the width-4 operating point: paired windowed delta.

WHY THIS TOOL EXISTS
--------------------
The exact4 GQA-pair timing pair (2026-08-11) returned a NULL whose
kernel-attributable residual was a +1.92 ms/step COST, indistinguishable from
the disclosed candidate-side sentinel-retag overhead.  That null was measured
where FA2 is small.  The width-4 Nsight attribution then established that at the
width-4 operating point FA2 is 3.26x LARGER -- 69.75 ms of a ~411 ms
batch-conditioned step wall -- so the null was taken against a kernel a third of
its operating size.

This tool reduces a two-arm pool16 pair AT that operating point and judges it
against thresholds the campaign wrote down BEFORE this run existed.

WHAT IT DOES NOT DO: RE-DERIVE THE WINDOW
-----------------------------------------
Every windowed quantity here comes from `reduce_window_arm` in
scripts/fr13_b4_width4_window_reduce.py, imported and called unchanged.  That
function owns the exact bracket (real Prometheus snapshots at ledger admission
events), the zero-tolerance census identity, and the drain-exclusion proof.  This
tool adds ONLY the paired contrast and the verdict.  If the window math is
wrong, it is wrong identically on both arms and cancels in the delta.

THE STATISTICAL FRAME -- READ THIS BEFORE QUOTING ANY NUMBER
------------------------------------------------------------
This is n=1 PAIRED DRAW.  It is a SCREEN, not a significance test, and the tool
refuses to pretend otherwise:

  * The sealed MDE (4.20 ms Hydra27 / 6.42 ms Tail23) is `t * SD / sqrt(n)` at
    n=4 over BETWEEN-PASS variation.  It is the half-width of a one-sided 95%
    interval on a four-pass mean.  It is NOT the uncertainty of a single paired
    difference, and this run does not produce a variance estimate of its own.
  * So the MDE is used here as a REFERENCE SCALE with a pre-registered meaning,
    not as a p-value.  The campaign's own falsification clause
    (results/fr13_b4_width4_nsys_20260813/README.md section 3, rank 1) reads:
    "if a width-4 FA2 re-test moves step_wall_ms by less than 6.42 ms, the
    gqa_pair mapping is genuinely null at width 4 too and rank 1 dies."  That
    sentence was written before this measurement, which is exactly why it is the
    threshold used.
  * The between-pass SD is carried alongside every delta so a reader can see how
    many sealed SDs the observed effect is worth.  A single paired difference is
    NOT distributed as that SD -- pairing removes shared host and task-difficulty
    variation -- but no decomposition into paired and unpaired components is
    derivable from the sealed campaign, whose two arms per pass are different
    TOPOLOGIES rather than stock/candidate.  So the SD is an anchor, published
    with that caveat, and never converted into an interval.

THE DIRECTION OF THE BIAS
-------------------------
The candidate arm ALSO pays the per-target-layer bias retag that SELECTS the
kernel: the batch stride IS the dispatch predicate, so there is no way to serve
the candidate kernel with the stock operand.  That cost is charged to the
candidate, so the pair can only UNDERSTATE a gain.  A positive verdict is
therefore sound; a null/regression verdict is confounded by harness overhead and
must not be read as a pure kernel result.  The exact4 pair measured that
overhead at ~1.92 ms/step; it is reported here as context, never subtracted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_b4_width4_window_reduce as w4  # noqa: E402
import fr13_b4_batch_conditioned_wall as bcw  # noqa: E402

SCHEMA = "fr13.b4_gqa_pair_width4_timing_pair.v1"
RUN_CLASS = "b4_gqa_pair_width4_retest"
TITLE = "B4 GQA-PAIR FA2 RE-TEST AT THE WIDTH-4 OPERATING POINT"

# The sealed batch-conditioned width-4 wall and its four-pass MDE. Read from the
# artifact rather than retyped: a hand-copied threshold is a threshold that can
# drift away from the evidence it claims to come from.
SEALED_MDE_RELPATH = (
    "results/fr13_b4_width4_nsys_20260813/fr13_b4_batch_conditioned_wall.json"
)
SEALED_MDE_SCHEMA = "fr13.b4_batch_conditioned_wall.v1"

# The campaign's own pre-registered sizing of the lever, from
# results/fr13_b4_width4_nsys_20260813/README.md section 3 rank 1:
#   "recovering even 10% of FA2 is 7.0 ms/step and clears the MDE"
FA2_MS_PER_STEP_AT_WIDTH4 = 69.74838122735675
FA2_RECOVERY_TARGET_FRACTION = 0.10

# The exact4 pair's measured candidate-side overhead, reported as context only.
# MARK'S RULING 2026-08-13. candidate_scope moved off the sealed
# final_fixed32_b4_full_graph_only token to a b3-inclusive one when the
# qualified operating points widened to the FULL graphs at widths 3 AND 4.
# READERS MUST ACCEPT BOTH: the banked +29.50 ms/step width-4 pair
# (output/fr13_b4_gqa_width4_timing_20260813T104433Z) was recorded under the
# sealed token and must stay re-reducible byte for byte. WRITERS emit only the
# new one. The .so identity pins are untouched by the widening.
SEALED_B4_CANDIDATE_SCOPE = "final_fixed32_b4_full_graph_only"
B34_CANDIDATE_SCOPE = "final_fixed32_b34_full_graph_only"
ACCEPTED_CANDIDATE_SCOPES = (SEALED_B4_CANDIDATE_SCOPE, B34_CANDIDATE_SCOPE)

EXACT4_RETAG_OVERHEAD_MS_PER_STEP = 1.92
EXACT4_PRIOR_VERDICT = (
    "NULL on speed: kernel-attributable residual +1.92 ms/step sfwd COST, "
    "indistinguishable from the disclosed sentinel-retag overhead "
    "(output/fr13_fa2_gqa_pair_b4_timing_20260811T015257Z @ 8940cd1ba)"
)

# Lower is better for these; higher is better for the rest.
LOWER_IS_BETTER = ("step_wall_ms",)
DELTA_FIELDS = (
    "step_wall_ms",
    "per_request_step_tps",
    "measured_tps_fullstep_wall",
    "events_per_step",
    "accept_per_event",
    "committed_per_event",
    "prefill_frac",
    "retained_wall_fraction",
    "floor_ratio",
)

CLAIMS: tuple[str, ...] = (
    "the PAIRED difference in windowed width-4 step wall and per-request step "
    "TPS between a stock-FA2-dispatch arm and a GQA-pair-dispatch arm that "
    "loaded the IDENTICAL pinned binary, served the IDENTICAL 128-query-row B4 "
    "geometry over the IDENTICAL byte-pinned 16-task pool at 4 slots, and "
    "differed in exactly one environment variable",
)

DOES_NOT_CLAIM: tuple[str, ...] = (
    "statistical significance. This is n=1 paired draw. It produces NO variance "
    "estimate of its own, and the sealed MDE it is judged against is a "
    "four-pass between-pass quantity, not the uncertainty of a single paired "
    "difference. A delta clearing the MDE is grounds for a four-pass paired "
    "campaign, NOT a promotion",
    "a pure kernel result when the verdict is null or negative. The candidate "
    "arm additionally pays the per-target-layer bias retag that selects the "
    "kernel -- the batch stride IS the dispatch predicate -- so the pair is "
    "conservative against the candidate by construction. A gain is therefore "
    "understated and sound; a null is confounded by harness overhead",
    "whole-arm throughput. Inherited in full from the width-4 window class: the "
    "drain phase is excluded BY CONSTRUCTION and the windowed rate must never "
    "be multiplied by an arm wall",
    "exact4 comparability. Different task set, different bracket estimator, and "
    "a windowed rather than whole-arm reduction. The 2026-08-11 exact4 verdict "
    "is carried here as context for what is being re-tested, not as a baseline "
    "this run is differenced against",
    "a cap verdict. b4_cap_applicable is false at B4 context sizes and nothing "
    "here changes that",
    "an agent-quality verdict. Resolve rate is neither read nor recorded. "
    "exact16 as QUALITY CONTROL at batched milestones is a separate gate",
    "a promotion or a citable seal. The width-4 window class is an INSTRUMENT "
    "with citable=false, and a screen built on it inherits that",
)


class PairError(RuntimeError):
    """The evidence does not support a paired width-4 contrast."""


# --------------------------------------------------------------------------- #
# pre-registered thresholds                                                     #
# --------------------------------------------------------------------------- #
def load_sealed_thresholds(repo: Path, mode: str) -> dict[str, Any]:
    """Read the four-pass MDE for this topology out of the sealed artifact."""
    path = repo / SEALED_MDE_RELPATH
    if not path.is_file() or path.is_symlink():
        raise PairError(
            f"no sealed batch-conditioned wall artifact at {path}; the verdict "
            "thresholds are DEFINED by it and are not reconstructible here"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SEALED_MDE_SCHEMA:
        raise PairError(
            f"sealed threshold artifact schema is {payload.get('schema')!r}, "
            f"expected {SEALED_MDE_SCHEMA!r}"
        )
    pooled = payload.get("pooled") or {}
    if mode not in pooled:
        raise PairError(
            f"the sealed artifact carries no width-4 wall for topology {mode!r}; "
            f"it has {sorted(pooled)}"
        )
    block = pooled[mode].get("batch_conditioned_full_width") or {}
    for key in ("mean_ms", "sd_ms", "cv", "n", "mde_ms", "mde_fraction"):
        value = block.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PairError(f"sealed width-4 wall {key}={value!r} is not numeric")
    if int(block["n"]) != 4:
        raise PairError(
            f"sealed width-4 wall was built from n={block['n']}, expected 4"
        )
    # THE BLENDED BASIS AND ITS OWN MDE.  The window reducer's `step_wall_ms` is
    # the width-BLENDED window mean (only ~70% of window steps are width 4, and
    # width-3 steps run ~50 ms cheaper), so a delta measured on it must be judged
    # against the BLEND's MDE, not the batch-conditioned one.  Judging a blended
    # delta against the conditioned threshold would silently apply a threshold
    # ~9% too small.  Both bases are carried, each with its matching threshold.
    blend = pooled[mode].get("sealed_rescaled_blend") or {}
    for key in ("mean_ms", "sd_ms", "mde_ms"):
        value = blend.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PairError(f"sealed blended wall {key}={value!r} is not numeric")
    target_ms = FA2_MS_PER_STEP_AT_WIDTH4 * FA2_RECOVERY_TARGET_FRACTION
    return {
        "blended_basis": {
            "sealed_step_wall_ms": float(blend["mean_ms"]),
            "sealed_between_pass_sd_ms": float(blend["sd_ms"]),
            "mde_ms": float(blend["mde_ms"]),
            "role": (
                "the width-BLENDED window mean the width-4 window reducer emits "
                "directly. Reported with its OWN MDE so the comparison is "
                "basis-matched; the primary judgement uses the batch-conditioned "
                "basis the falsification clause was written on"
            ),
        },
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "topology": mode,
        "sealed_width4_step_wall_ms": float(block["mean_ms"]),
        "sealed_between_pass_sd_ms": float(block["sd_ms"]),
        "sealed_between_pass_cv": float(block["cv"]),
        "sealed_pass_count": int(block["n"]),
        "mde_ms": float(block["mde_ms"]),
        "mde_fraction": float(block["mde_fraction"]),
        "mde_critical": block.get("critical"),
        "mde_definition": (
            "t * SD / sqrt(n) at n=4 over BETWEEN-PASS variation: the half-width "
            "of a one-sided 95% interval on a four-pass mean. NOT the "
            "uncertainty of a single paired difference"
        ),
        "fa2_ms_per_step_at_width4": FA2_MS_PER_STEP_AT_WIDTH4,
        "fa2_recovery_target_fraction": FA2_RECOVERY_TARGET_FRACTION,
        "fa2_recovery_target_ms": target_ms,
        "pre_registered_falsification": (
            "results/fr13_b4_width4_nsys_20260813/README.md section 3 rank 1: "
            "'if a width-4 FA2 re-test moves step_wall_ms by less than 6.42 ms, "
            "the gqa_pair mapping is genuinely null at width 4 too and rank 1 "
            "dies'. Written before this measurement existed"
        ),
    }


# --------------------------------------------------------------------------- #
# paired contrast                                                               #
# --------------------------------------------------------------------------- #
def delta_block(stock: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Signed deltas, oriented so POSITIVE always means the candidate is better."""
    out: dict[str, Any] = {}
    for field in DELTA_FIELDS:
        s = stock.get(field)
        c = candidate.get(field)
        if not isinstance(s, (int, float)) or isinstance(s, bool):
            continue
        if not isinstance(c, (int, float)) or isinstance(c, bool):
            continue
        s = float(s)
        c = float(c)
        improvement = (s - c) if field in LOWER_IS_BETTER else (c - s)
        out[field] = {
            "stock": s,
            "candidate": c,
            "candidate_minus_stock": c - s,
            "improvement": improvement,
            "improvement_pct_of_stock": (improvement / s * 100.0) if s else None,
            "orientation": (
                "lower is better" if field in LOWER_IS_BETTER else "higher is better"
            ),
        }
    return out


def batch_conditioned(
    arm_record: dict[str, Any], sidecar_dir: Path, arm_name: str
) -> dict[str, Any]:
    """Condition this arm's windowed wall on served batch width == slots.

    The window is a POOL-DEPTH window, not a per-step batch filter: inside it the
    engine batch is width 4 on only ~63-70% of steps. The SFWD per-step samples
    sidecar carries wall AND width per step, so the width-4 steps can be selected
    exactly -- and `bcw.analyse_arm` re-proves that the selection IS the sealed
    counter bracket (step count and wall sum must both agree) before returning.
    """
    try:
        result = bcw.analyse_arm(arm_record, sidecar_dir, arm_name=arm_name)
    except (bcw.WallError, OSError, json.JSONDecodeError, KeyError) as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}
    if "width_full_step_wall_ms" not in result:
        return {
            "available": False,
            "reason": (
                "no step inside the window was served at the full slot width; "
                "there is no batch-conditioned wall to compute"
            ),
        }
    return {"available": True, **result}


def treated_widths(candidate_engagement: dict[str, Any] | None) -> tuple[int, ...]:
    """Which widths the CANDIDATE ARM actually served, read off its own record.

    This must never be inferred from the shape of the timing data. Before
    Mark's 2026-08-13 ruling the candidate was served at exactly one operating
    point and `max(width)` happened to name it; after the ruling a run whose
    credential carries the width-3 padded byte evidence serves widths 3 AND 4,
    and `max(width)` would classify a TREATED width as a placebo and then hand
    it to difference-in-differences as the control. That does not fail loudly
    -- it silently subtracts the effect from itself and prints "placebo clean"
    over contaminated rows.

    So the treated set comes from `candidate_scope_widths`, which the runtime
    emits as the widths that run was AUTHORISED to serve. An engagement
    predating the field carries only the scope token; the sealed
    width-4 token means (4,), and the b3-inclusive token without the field is
    ambiguous and is refused rather than guessed.
    """
    if not candidate_engagement:
        return ()
    declared = candidate_engagement.get("candidate_scope_widths")
    if declared is not None:
        if (
            not isinstance(declared, list)
            or not declared
            or not all(isinstance(width, int) for width in declared)
        ):
            raise PairError("candidate engagement scope widths drifted")
        return tuple(sorted(declared))
    scope = candidate_engagement.get("candidate_scope")
    if scope == SEALED_B4_CANDIDATE_SCOPE:
        return (4,)
    raise PairError(
        "candidate engagement declares scope "
        + repr(scope)
        + " without candidate_scope_widths; the treated widths cannot be "
        "resolved and must not be guessed"
    )


def per_width_placebo(
    stock_bc: dict[str, Any],
    candidate_bc: dict[str, Any],
    treated: tuple[int, ...],
) -> dict[str, Any]:
    """The strongest causal evidence this design produces, and it is free.

    The candidate kernel is served at a DECLARED set of operating points --
    the final fixed32 FULL graphs at the authorised widths. Every other width
    inside the same window runs the IDENTICAL stock dispatch on BOTH arms,
    because the candidate's scope excludes it. Until 2026-08-13 that scope was
    `final_fixed32_b4_full_graph_only` and the untreated widths were 1..3;
    under Mark's ruling a run whose credential carries the width-3 padded byte
    evidence declares `final_fixed32_b34_full_graph_only` and width 3 is
    TREATED, so the placebo set shrinks to widths 1 and 2. The placebo
    argument is unchanged in kind -- only the boundary moved -- and the
    treated set is READ OFF THE ENGAGEMENT RECORD, never inferred from the
    widths present in the data.

    So the untreated widths are a NATURAL PLACEBO, drawn from the same arms,
    the same hosts, the same window and the same wall chain as the treated
    width. If a step-wall gain shows up at widths where the code is identical,
    the contrast is measuring arm-to-arm confound rather than the kernel, and
    the headline delta must not be believed. If the gain appears only at the
    treated widths, the effect is localised to the intervention.

    DIFFERENCE-IN-DIFFERENCES. The widest UNTREATED width is the control: its
    arm-to-arm delta estimates the confound that also contaminates the treated
    width. Subtracting it (additive) or dividing it out (multiplicative) gives a
    confound-corrected effect. Both are reported because neither model is
    established, and the honest read is the range they span.
    """
    s_by = stock_bc.get("by_width") or {}
    c_by = candidate_bc.get("by_width") or {}
    widths = sorted({int(w) for w in s_by} & {int(w) for w in c_by})
    if not widths:
        return {"available": False, "reason": "no shared served widths"}
    if not treated:
        return {
            "available": False,
            "reason": "the candidate engagement declared no treated widths",
        }
    engaged = [w for w in widths if w in treated]
    if not engaged:
        return {
            "available": False,
            "reason": (
                "no treated width "
                + repr(list(treated))
                + " appears in both arms' batch-conditioned detail"
            ),
        }
    # The headline row is the widest TREATED width -- the operating point the
    # campaign is sized on. The narrower treated widths are reported as
    # treated too, and are excluded from the control set below.
    headline = max(engaged)
    rows = []
    for w in widths:
        s = s_by[str(w)]
        c = c_by[str(w)]
        rows.append(
            {
                "width": w,
                "candidate_engaged": w in treated,
                "role": "treated" if w in treated else "placebo",
                "stock_mean_ms": s["mean_ms"],
                "stock_steps": s["steps"],
                "candidate_mean_ms": c["mean_ms"],
                "candidate_steps": c["steps"],
                "candidate_minus_stock_ms": c["mean_ms"] - s["mean_ms"],
                "improvement_ms": s["mean_ms"] - c["mean_ms"],
            }
        )
    controls = [r for r in rows if not r["candidate_engaged"] and r["stock_steps"] >= 100]
    treated_row = next(r for r in rows if r["width"] == headline)
    out: dict[str, Any] = {
        "available": True,
        "treated_width": headline,
        "treated_widths": list(treated),
        "rows": rows,
        "placebo_note": (
            "widths outside the declared treated set run the IDENTICAL stock "
            "dispatch on both arms (they are outside candidate_scope, "
            "whichever of "
            + " / ".join(ACCEPTED_CANDIDATE_SCOPES)
            + " the engagement declared), so any delta there is arm-to-arm "
            "confound, not kernel"
        ),
        "min_control_steps_for_did": 100,
    }
    if not controls:
        out["difference_in_differences"] = {
            "available": False,
            "reason": "no untreated width carries enough steps to serve as a control",
        }
        return out
    control = max(controls, key=lambda r: r["width"])
    raw = treated_row["candidate_minus_stock_ms"]
    ctrl = control["candidate_minus_stock_ms"]
    ratio = (
        control["candidate_mean_ms"] / control["stock_mean_ms"]
        if control["stock_mean_ms"]
        else None
    )
    out["difference_in_differences"] = {
        "available": True,
        "control_width": control["width"],
        "control_steps": {
            "stock": control["stock_steps"],
            "candidate": control["candidate_steps"],
        },
        "raw_treated_delta_ms": raw,
        "control_delta_ms": ctrl,
        "additive_effect_ms": raw - ctrl,
        "multiplicative_effect_ms": (
            treated_row["candidate_mean_ms"] - treated_row["stock_mean_ms"] * ratio
            if ratio
            else None
        ),
        "confound_direction": (
            "the control shows the candidate arm SLOWER where the code is "
            "identical, so the confound works AGAINST the candidate and the raw "
            "delta UNDERSTATES the kernel effect"
            if ctrl > 0
            else "the control shows the candidate arm faster where the code is "
            "identical, so part of the raw delta is arm-to-arm confound and the "
            "raw delta OVERSTATES the kernel effect"
        ),
        "interpretation": (
            "additive and multiplicative are two models of the same confound; "
            "neither is established, so the honest effect is the range they span"
        ),
    }
    # Width scaling: the quantity the width-4 attribution said FA2 owns.
    # Measured against the next width DOWN in the data, treated or not: this is
    # a scaling cost, not a causal contrast, and the sealed width-4 lineage
    # reads it as 3 -> 4.
    narrower = [w for w in widths if w < headline]
    if narrower:
        prev = max(narrower)
        s_scale = s_by[str(headline)]["mean_ms"] - s_by[str(prev)]["mean_ms"]
        c_scale = c_by[str(headline)]["mean_ms"] - c_by[str(prev)]["mean_ms"]
        out["width_scaling_cost"] = {
            "from_width": prev,
            "to_width": headline,
            "stock_ms": s_scale,
            "candidate_ms": c_scale,
            "removed_ms": s_scale - c_scale,
            "removed_fraction": (s_scale - c_scale) / s_scale if s_scale else None,
            "nsys_attributed_fa2_width4_cost_ms": 48.4,
            "corroboration_note": (
                "the STOCK arm's width-scaling cost is an INDEPENDENT reproduction "
                "of the Nsight-attributed FA2 width-4 cost, measured by a "
                "different instrument (per-step wall samples vs kernel "
                "attribution). Agreement is evidence the basis is sound"
            ),
        }
    return out


def judge(step_wall_improvement_ms: float, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered thresholds. No threshold is chosen here."""
    mde = thresholds["mde_ms"]
    target = thresholds["fa2_recovery_target_ms"]
    sd = thresholds["sealed_between_pass_sd_ms"]
    d = step_wall_improvement_ms

    if d >= target:
        band = "GAIN_AT_OR_ABOVE_10PCT_FA2_TARGET"
        disposition = "REVERSES_THE_EXACT4_NULL"
        reading = (
            "the candidate recovers at least the 10% of FA2 the attribution "
            "sized as the lever's target. At n=1 this is a SCREEN result: it is "
            "grounds to fund a four-pass paired campaign, not a promotion"
        )
    elif d >= mde:
        band = "GAIN_CLEARS_FOUR_PASS_MDE_BELOW_TARGET"
        disposition = "REVERSES_THE_EXACT4_NULL"
        reading = (
            "the candidate moves the width-4 step wall by more than the sealed "
            "four-pass MDE but less than the 10%-of-FA2 target. The campaign's "
            "own falsification clause is NOT met, so the lever survives. At n=1 "
            "this is a screen: confirm with a four-pass paired campaign"
        )
    elif d > 0.0:
        band = "GAIN_BELOW_FOUR_PASS_MDE"
        disposition = "CONFIRMS_THE_EXACT4_NULL"
        reading = (
            "the candidate is nominally faster but by less than the sealed "
            "four-pass MDE. The pre-registered falsification clause is MET: at "
            "width 4 the gqa_pair mapping is null on the evidence it was asked "
            "to produce, and the FA2 headroom needs a different geometry"
        )
    else:
        band = "NO_GAIN_OR_REGRESSION"
        disposition = "CONFIRMS_THE_EXACT4_NULL"
        reading = (
            "the candidate is not faster at the width-4 operating point. "
            "Combined with the 2026-08-11 exact4 null this closes the gqa_pair "
            "lever at B4: the kernel is byte-legal and the machinery is proven, "
            "but the mapping does not convert FA2's structural headroom into "
            "step wall at either regime"
        )
    return {
        "step_wall_improvement_ms": d,
        "band": band,
        "lever_disposition": disposition,
        "reading": reading,
        "clears_four_pass_mde": d >= mde,
        "clears_10pct_fa2_target": d >= target,
        "four_pass_mde_ms": mde,
        "fa2_recovery_target_ms": target,
        "sealed_between_pass_sd_ms": sd,
        "improvement_in_sealed_sd_units": (d / sd) if sd else None,
        "sd_units_caveat": (
            "the sealed SD is BETWEEN-PASS across four passes whose two arms are "
            "different TOPOLOGIES, not stock/candidate. Pairing stock against "
            "candidate inside one pass removes shared host and task-difficulty "
            "variation, so the true SD of THIS difference is smaller than the "
            "sealed SD by an amount this evidence cannot quantify. The ratio is "
            "an anchor for judgement, never an interval"
        ),
        "n_paired_draws": 1,
        "is_significance_test": False,
    }


# --------------------------------------------------------------------------- #
# engagement provenance                                                         #
# --------------------------------------------------------------------------- #
def read_engagement(arm_dir: Path) -> dict[str, Any] | None:
    path = arm_dir / "logs" / "fr13_fa2_qrow32_b4_production_engagement.json"
    if not path.is_file() or path.is_symlink():
        return None
    payload = json.loads(path.read_text(encoding="ascii"))
    return {
        "status": payload.get("status"),
        "runtime_mode": payload.get("runtime_mode"),
        "arm": payload.get("arm"),
        "layer_count": payload.get("layer_count"),
        "selector_sentinel": payload.get("selector_sentinel"),
        "batch_size": payload.get("batch_size"),
        "total_query_rows": payload.get("total_query_rows"),
        "task_count": payload.get("task_count"),
        "subset_sha256": payload.get("subset_sha256"),
        "candidate_served": payload.get("candidate_served"),
        "candidate_scope": payload.get("candidate_scope"),
        # Load-bearing, not decorative: this is what decides the treated set,
        # and therefore which rows are eligible to be the DiD control.
        "candidate_scope_widths": payload.get("candidate_scope_widths"),
        "fallback_allowed": payload.get("fallback_allowed"),
        "bypass_counts": payload.get("bypass_counts"),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def validate_pair_engagement(
    stock_engagement: dict[str, Any] | None,
    candidate_engagement: dict[str, Any] | None,
    *,
    expected_task_count: int,
) -> None:
    if stock_engagement is not None:
        raise PairError(
            "the stock-dispatch arm emitted a GQA-pair engagement record; the "
            "sentinel leaked across the pair and the contrast is invalid"
        )
    if candidate_engagement is None:
        raise PairError(
            "the candidate arm emitted no GQA-pair engagement record; it did not "
            "serve the kernel and there is nothing to time"
        )
    e = candidate_engagement
    if (
        e.get("status") != "ENGAGED"
        or e.get("runtime_mode") != "FULL"
        or e.get("arm") != "gqa_pair"
        or e.get("layer_count") != 16
        or e.get("batch_size") != 4
        or e.get("total_query_rows") != 128
        or e.get("candidate_served") is not True
        or e.get("fallback_allowed") is not False
        or e.get("candidate_scope") not in ACCEPTED_CANDIDATE_SCOPES
    ):
        raise PairError(
            "the candidate arm did not serve the GQA-pair kernel on every target "
            f"layer of the qualified B4 FULL graph: {e!r}"
        )
    if e.get("task_count") != expected_task_count:
        raise PairError(
            f"the candidate engagement records task_count={e.get('task_count')!r}, "
            f"expected the {expected_task_count}-task pool"
        )


# --------------------------------------------------------------------------- #
# payload                                                                       #
# --------------------------------------------------------------------------- #
def build_payload(
    *,
    runroot: Path,
    stock_dir: Path,
    candidate_dir: Path,
    mode: str,
    source_commit: str,
    subset: Path,
    thresholds: dict[str, Any],
    provenance: dict[str, Any],
    repo: Path,
) -> dict[str, Any]:
    stock_record = w4.reduce_window_arm(stock_dir, mode=mode, pass_index=0)
    candidate_record = w4.reduce_window_arm(candidate_dir, mode=mode, pass_index=0)

    excluded = [
        {"role": role, "arm": r["arm"], "reason": r["exclusion_reason"]}
        for role, r in (("stock", stock_record), ("candidate", candidate_record))
        if not r.get("included")
    ]
    if excluded:
        return {
            "schema": SCHEMA,
            "run_class": RUN_CLASS,
            "title": TITLE,
            "verdict": "NOT_EVALUATED_NO_WINDOW",
            "analysis_valid": False,
            "citable": False,
            "runroot": str(runroot),
            "source_commit": source_commit,
            "topology": mode,
            "excluded_arms": excluded,
            "arms": {"stock": stock_record, "candidate": candidate_record},
            "thresholds": thresholds,
            "provenance": provenance,
            "claims": list(CLAIMS),
            "does_not_claim": list(DOES_NOT_CLAIM),
        }

    stock_w = stock_record["windowed"]
    candidate_w = candidate_record["windowed"]
    deltas = delta_block(stock_w, candidate_w)

    # ---- basis selection -------------------------------------------------- #
    # PRIMARY is the batch-conditioned (width == slots) wall, because that is the
    # basis the campaign's pre-registered falsification clause was written on and
    # because conditioning on width removes the width-mix variance (the sealed
    # MDE improves from 4.60 to 4.20 ms on Hydra27). If the per-step samples are
    # unavailable the tool FALLS BACK to the blended basis and says so in the
    # verdict rather than silently judging a blended delta against a conditioned
    # threshold.
    sidecar_dir = repo / "output" / "fr13_sfwd_sidecar"
    stock_bc = batch_conditioned(stock_record, sidecar_dir, stock_dir.name)
    candidate_bc = batch_conditioned(candidate_record, sidecar_dir, candidate_dir.name)

    if stock_bc["available"] and candidate_bc["available"]:
        basis = "batch_conditioned_width4"
        stock_wall = float(stock_bc["width_full_step_wall_ms"])
        candidate_wall = float(candidate_bc["width_full_step_wall_ms"])
        basis_thresholds = thresholds
        basis_note = (
            "width == slots steps only, selected from the SFWD per-step samples "
            "sidecar and re-proved against the sealed counter bracket"
        )
    else:
        basis = "width_blended_window"
        stock_wall = float(stock_w["step_wall_ms"])
        candidate_wall = float(candidate_w["step_wall_ms"])
        basis_thresholds = {
            **thresholds,
            "mde_ms": thresholds["blended_basis"]["mde_ms"],
            "sealed_between_pass_sd_ms": thresholds["blended_basis"][
                "sealed_between_pass_sd_ms"
            ],
            "sealed_width4_step_wall_ms": thresholds["blended_basis"][
                "sealed_step_wall_ms"
            ],
        }
        basis_note = (
            "FALLBACK: the batch-conditioned basis was unavailable, so the "
            "width-BLENDED window mean is judged against the BLEND's own MDE. "
            "This is a less sensitive test on a noisier statistic"
        )

    step_wall_basis = {
        "basis": basis,
        "basis_note": basis_note,
        "stock_step_wall_ms": stock_wall,
        "candidate_step_wall_ms": candidate_wall,
        "improvement_ms": stock_wall - candidate_wall,
        "improvement_pct": (
            (stock_wall - candidate_wall) / stock_wall * 100.0 if stock_wall else None
        ),
        "mde_ms_for_this_basis": basis_thresholds["mde_ms"],
        "stock_detail": stock_bc,
        "candidate_detail": candidate_bc,
        "blended_basis_contrast": {
            "stock_step_wall_ms": float(stock_w["step_wall_ms"]),
            "candidate_step_wall_ms": float(candidate_w["step_wall_ms"]),
            "improvement_ms": float(stock_w["step_wall_ms"])
            - float(candidate_w["step_wall_ms"]),
            "mde_ms_for_this_basis": thresholds["blended_basis"]["mde_ms"],
        },
    }
    verdict = judge(step_wall_basis["improvement_ms"], basis_thresholds)
    verdict["basis"] = basis
    verdict["basis_note"] = basis_note

    placebo = (
        per_width_placebo(
            stock_bc,
            candidate_bc,
            treated_widths(provenance.get("candidate_engagement")),
        )
        if (stock_bc["available"] and candidate_bc["available"])
        else {"available": False, "reason": "no batch-conditioned detail"}
    )
    step_wall_basis["per_width_placebo"] = placebo
    # A gain at a width where the candidate is NOT served would mean the headline
    # is measuring arm-to-arm confound. Surface that in the verdict rather than
    # leaving it for a reader to notice.
    if placebo.get("available"):
        leaks = [
            r
            for r in placebo["rows"]
            if not r["candidate_engaged"]
            and r["stock_steps"] >= placebo["min_control_steps_for_did"]
            and r["improvement_ms"] >= basis_thresholds["mde_ms"]
        ]
        verdict["placebo_clean"] = not leaks
        verdict["placebo_leak_widths"] = [r["width"] for r in leaks]
        verdict["placebo_reading"] = (
            "no untreated width shows a gain at or above the MDE, so the effect "
            "is localised to the one operating point where the candidate is "
            "served"
            if not leaks
            else "an untreated width shows a gain at or above the MDE: the "
            "headline delta is contaminated by arm-to-arm confound and must NOT "
            "be read as a kernel result"
        )

    subset_ids = sorted(
        json.loads(subset.read_text(encoding="ascii"))["instance_ids"]
    )
    validate_pair_engagement(
        provenance.get("stock_engagement"),
        provenance.get("candidate_engagement"),
        expected_task_count=len(subset_ids),
    )

    return {
        "schema": SCHEMA,
        "run_class": RUN_CLASS,
        "title": TITLE,
        "verdict": verdict["lever_disposition"],
        "analysis_valid": True,
        "citable": False,
        "citable_reason": (
            "n=1 paired screen built on the width-4 window INSTRUMENT class "
            "(itself citable=false). It reports a delta against pre-registered "
            "thresholds; it promotes nothing and seals nothing"
        ),
        "formal_floor_acceptance_eligible": False,
        "b4_cap_applicable": False,
        "runroot": str(runroot),
        "source_commit": source_commit,
        "topology": mode,
        "task_pool": len(subset_ids),
        "slots": stock_record.get("slots"),
        "subset": str(subset),
        "subset_sha256": hashlib.sha256(subset.read_bytes()).hexdigest(),
        "only_arm_delta": (
            "FA2_stock_dispatch_to_qrow32_gqa_pair_with_candidate_side_bias_retag"
        ),
        "arm_delta_disclosure": {
            "served_dispatch": "stock_fa2 -> qrow32_gqa_pair",
            "candidate_only_overhead": (
                "per-target-layer empty_strided((4,32,32),(131092,32,1)) + copy_ "
                "recorded into the captured graph and replayed every step"
            ),
            "bias_operand": "2d_broadcast_tile -> 4_plane_batch_strided",
            "overhead_charged_to": "candidate",
            "bias_direction": "conservative_against_candidate",
            "candidate_scope": (
                (provenance.get("candidate_engagement") or {}).get(
                    "candidate_scope"
                )
            ),
            "candidate_scope_accepted": list(ACCEPTED_CANDIDATE_SCOPES),
            "regression_verdict_is_confounded_by_harness": True,
            "exact4_measured_overhead_ms_per_step": (
                EXACT4_RETAG_OVERHEAD_MS_PER_STEP
            ),
            "overhead_is_never_subtracted": True,
        },
        "prior_verdict_under_retest": {
            "regime": "exact4 (pool == slots, events/step ~1.15-1.22)",
            "verdict": EXACT4_PRIOR_VERDICT,
            "why_retested": (
                "at the width-4 operating point FA2 is 3.26x larger (69.75 ms of "
                "a ~411 ms batch-conditioned step wall, ~52 ms of structural "
                "headroom above its bandwidth floor), and the candidate's "
                "mechanism scales with exactly the quantity that grew"
            ),
        },
        "step_wall_basis": step_wall_basis,
        "primary_statistic": "step_wall_ms",
        "primary_statistic_reason": (
            "the width holds the served batch fixed, so the aggregate is no "
            "longer a free parameter the schedule can inflate, and the sealed "
            "MDE that this lever was pre-registered against is stated on step "
            "wall. per_request_step_tps is reported alongside as the operating "
            "point's own primary statistic"
        ),
        "verdict_detail": verdict,
        "deltas": deltas,
        "windowed": {"stock": stock_w, "candidate": candidate_w},
        "thresholds": thresholds,
        "provenance": provenance,
        "arms": {"stock": stock_record, "candidate": candidate_record},
        "claims": list(CLAIMS),
        "does_not_claim": list(DOES_NOT_CLAIM),
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"{payload['title']} -- {payload['verdict']}",
        f"  runroot:  {payload['runroot']}",
        f"  topology: {payload.get('topology')}   citable={payload['citable']}",
        "",
    ]
    if not payload.get("analysis_valid"):
        lines.append("  NO WINDOW -- arms excluded:")
        for row in payload.get("excluded_arms", []):
            lines.append(f"    {row['role']:10} {row['arm']}: {row['reason']}")
        return "\n".join(lines)

    t = payload["thresholds"]
    lines += [
        f"  sealed width-4 wall ({t['topology']}): "
        f"{t['sealed_width4_step_wall_ms']:.2f} ms  "
        f"(4-pass SD {t['sealed_between_pass_sd_ms']:.2f}, "
        f"MDE {t['mde_ms']:.2f} ms)",
        f"  10%-of-FA2 target: {t['fa2_recovery_target_ms']:.2f} ms",
        "",
        f"  {'statistic':28} {'stock':>10} {'candidate':>10} {'delta':>10} {'%':>8}",
    ]
    for field in DELTA_FIELDS:
        row = payload["deltas"].get(field)
        if row is None:
            continue
        pct = row["improvement_pct_of_stock"]
        pct_s = f"{pct:+7.2f}%" if pct is not None else "      --"
        lines.append(
            f"  {field:28} {row['stock']:>10.3f} {row['candidate']:>10.3f} "
            f"{row['improvement']:>+10.3f} {pct_s:>8}"
        )
    b = payload["step_wall_basis"]
    lines += [
        "",
        f"  JUDGED BASIS: {b['basis']}  (MDE {b['mde_ms_for_this_basis']:.2f} ms)",
        f"    {b['basis_note']}",
        f"    stock {b['stock_step_wall_ms']:.2f} ms  ->  "
        f"candidate {b['candidate_step_wall_ms']:.2f} ms",
    ]
    bl = b["blended_basis_contrast"]
    lines.append(
        f"  blended-basis contrast: {bl['improvement_ms']:+.3f} ms "
        f"(its own MDE {bl['mde_ms_for_this_basis']:.2f} ms)"
    )
    pl = b.get("per_width_placebo") or {}
    if pl.get("available"):
        lines += [
            "",
            "  PER-WIDTH PLACEBO (candidate serves ONLY widths "
            + ", ".join(str(w) for w in pl.get("treated_widths", []))
            + ")",
        ]
        lines.append(
            f"    {'width':>5} {'stock ms':>10} {'cand ms':>10} {'impr ms':>9} {'role':>8}"
        )
        for r in pl["rows"]:
            lines.append(
                f"    {r['width']:>5} {r['stock_mean_ms']:>10.2f} "
                f"{r['candidate_mean_ms']:>10.2f} {r['improvement_ms']:>+9.2f} "
                f"{r['role']:>8}"
            )
        did = pl.get("difference_in_differences") or {}
        if did.get("available"):
            lines.append(
                f"    DiD vs width {did['control_width']}: additive "
                f"{did['additive_effect_ms']:+.2f} ms, multiplicative "
                f"{did['multiplicative_effect_ms']:+.2f} ms"
            )
        ws = pl.get("width_scaling_cost") or {}
        if ws:
            lines.append(
                f"    width {ws['from_width']}->{ws['to_width']} cost: stock "
                f"{ws['stock_ms']:+.2f} ms (nsys FA2 "
                f"{ws['nsys_attributed_fa2_width4_cost_ms']:+.1f}), candidate "
                f"{ws['candidate_ms']:+.2f} ms, removed {ws['removed_ms']:+.2f} ms"
            )
    v = payload["verdict_detail"]
    if "placebo_clean" in v:
        lines.append(f"  placebo clean: {v['placebo_clean']}  {v['placebo_reading']}")
    sd_units = v.get("improvement_in_sealed_sd_units")
    sd_note = f" ({sd_units:+.2f} sealed SD)" if sd_units is not None else ""
    lines += [
        "",
        f"  step-wall improvement: {v['step_wall_improvement_ms']:+.3f} ms{sd_note}",
        f"  band:                  {v['band']}",
        f"  clears 4-pass MDE:     {v['clears_four_pass_mde']}",
        f"  clears 10% FA2 target: {v['clears_10pct_fa2_target']}",
        "",
        "  n=1 paired draw -- SCREEN, not a significance test.",
        f"  {v['reading']}",
        "",
    ]
    return "\n".join(lines)


def self_check(repo: Path) -> int:
    """Resolve everything the run will need BEFORE any GPU time is spent."""
    problems: list[str] = []
    for mode in ("hydra27_fixed32", "tail6_fixed32"):
        try:
            t = load_sealed_thresholds(repo, mode)
            if not (t["mde_ms"] > 0 and t["sealed_width4_step_wall_ms"] > 0):
                problems.append(f"{mode}: non-positive sealed thresholds")
            print(
                f"  {mode:18} sealed wall {t['sealed_width4_step_wall_ms']:.2f} ms  "
                f"SD {t['sealed_between_pass_sd_ms']:.2f}  "
                f"MDE {t['mde_ms']:.2f} ms  "
                f"target {t['fa2_recovery_target_ms']:.2f} ms"
            )
        except (PairError, OSError, json.JSONDecodeError) as error:
            problems.append(f"{mode}: {error}")
    for name in ("reduce_window_arm", "Width4Error", "MIN_WINDOW_STEPS"):
        if not hasattr(w4, name):
            problems.append(f"width-4 window reducer has no {name}")
    subset = repo / "config/fr13_fixed32/subset_b4_sixteen.json"
    if not subset.is_file():
        problems.append(f"missing {subset}")
    else:
        ids = json.loads(subset.read_text(encoding="ascii"))["instance_ids"]
        if len(ids) != 16:
            problems.append(f"pool subset holds {len(ids)} ids, expected 16")
    if problems:
        for problem in problems:
            print(f"class-9 FAIL-LOUD [width4-pair self-check]: {problem}", file=sys.stderr)
        return 2
    print("  width-4 pair reducer self-check OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reduce a two-arm GQA-pair width-4 timing pair."
    )
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--runroot", type=Path, default=None)
    parser.add_argument("--stock-arm", type=Path, default=None)
    parser.add_argument("--candidate-arm", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="hydra27_fixed32")
    parser.add_argument("--source-commit", type=str, default="")
    parser.add_argument(
        "--subset", type=Path, default=Path("config/fr13_fixed32/subset_b4_sixteen.json")
    )
    parser.add_argument("--dual-gate-sha256", type=str, default="")
    parser.add_argument("--dual-gate-binding-sha256", type=str, default="")
    parser.add_argument("--candidate-sidecar-sha256", type=str, default="")
    parser.add_argument("--patch-source-sha256", type=str, default="")
    parser.add_argument("--repo", type=Path, default=SCRIPTS.parent)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check(args.repo)

    for name in ("runroot", "stock_arm", "candidate_arm"):
        if getattr(args, name) is None:
            print(
                f"class-9 FAIL-LOUD [width4-pair]: --{name.replace('_', '-')} is required",
                file=sys.stderr,
            )
            return 2

    try:
        thresholds = load_sealed_thresholds(args.repo, args.mode)
        provenance = {
            "source_commit": args.source_commit,
            "dual_gate_sha256": args.dual_gate_sha256,
            "dual_gate_binding_sha256": args.dual_gate_binding_sha256,
            "candidate_sidecar_sha256": args.candidate_sidecar_sha256,
            "patch_source_sha256": args.patch_source_sha256,
            "stock_arm_dir": str(args.stock_arm),
            "candidate_arm_dir": str(args.candidate_arm),
            "stock_engagement": read_engagement(args.stock_arm),
            "candidate_engagement": read_engagement(args.candidate_arm),
        }
        payload = build_payload(
            runroot=args.runroot,
            stock_dir=args.stock_arm,
            candidate_dir=args.candidate_arm,
            mode=args.mode,
            source_commit=args.source_commit,
            subset=args.subset,
            thresholds=thresholds,
            provenance=provenance,
            repo=args.repo,
        )
    except (PairError, w4.Width4Error, OSError, json.JSONDecodeError) as error:
        print(f"class-9 FAIL-LOUD [width4-pair]: {error}", file=sys.stderr)
        return 2

    out = args.out or (args.runroot / "width4_timing_pair.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(out)
    print(render(payload))
    print(f"  wrote {out}")
    return 0 if payload["analysis_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
