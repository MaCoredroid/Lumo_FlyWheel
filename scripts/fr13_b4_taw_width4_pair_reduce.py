#!/usr/bin/env python3
"""The TAW native-precompute production lever, paired at the width-4 point.

WHY THIS TOOL EXISTS
--------------------
The width-4 Nsight attribution (results/fr13_b4_width4_nsys_20260813/) ranked
three levers and then listed a fourth it could NOT rank, in README section 3:

    4. CFWD small-kernel consolidation *(new, unranked)* -- 37.1 ms/step of
       elementwise + other across 321k tiny instances per step. Not yet a lever
       because no candidate exists, but it is the third-largest addressable pile
       and nothing has ever been aimed at it.

The shape-pinned batched TAW committer IS that candidate arriving. Its whole
mechanism is collapsing the commit walk from twelve per-level launches to one
(`exact_commit_launches` 12 -> 1, `walk_levels` 12 -> 1 in the pinned tensor-call
census), which is exactly the "many tiny instances" quantity the attribution
sized. So this pair asks the attribution's own open question with the instrument
the campaign already sealed for width-4 contrasts.

WHAT IT DOES NOT DO: RE-DERIVE THE WINDOW
-----------------------------------------
Every windowed quantity here comes from `reduce_window_arm` in
scripts/fr13_b4_width4_window_reduce.py, imported and called UNCHANGED, exactly
as the GQA-pair sibling does. That function owns the exact bracket (real
Prometheus snapshots at ledger admission events), the zero-tolerance census
identity, and the drain-exclusion proof. The windowing math is lever-agnostic:
it discovers arms from the refill ledger, the work census and deploy_speed, none
of which know which lever produced them. This tool adds ONLY the TAW credential
checks, the paired contrast and the verdict.

THERE IS NO PRE-REGISTERED TAW EFFECT SIZE, AND THIS TOOL WILL NOT INVENT ONE
----------------------------------------------------------------------------
The GQA-pair re-test could be judged against a number the campaign wrote down
before it ran ("recovering even 10% of FA2 is 7.0 ms/step and clears the MDE").
No such sentence exists for CFWD consolidation -- the attribution explicitly
left it UNRANKED for want of a candidate. So the only pre-registered threshold
this screen has is the sealed four-pass MDE, and that is the only threshold it
uses. The addressable pile is reported beside it as SCALE, never as a target.

And the scale is reported with the arithmetic that matters: 10% of the 37.1
ms/step pile is ~3.7 ms/step, which is BELOW the Hydra27 MDE (4.20) and far
below Tail23's (6.42). A one-tenth consolidation is therefore INVISIBLE to this
instrument at n=1. A null here means "not resolvable at this size", not "no
effect", and the verdict text says so rather than leaving a reader to work it
out.

THE STRATA, AND WHY THE PLACEBO IS WEAKER HERE THAN FOR FA2
-----------------------------------------------------------
The FA2 GQA-pair candidate served ONE width (later two), so the untreated widths
inside the same window were a free natural placebo drawn from the same arms and
the same hosts. The TAW production selector is authorised at every batch its
PASS bundle qualified at or above the pinned minimum -- {2, 3, 4} for the
current bundle -- so the ONLY untreated width is 1, and width-1 steps inside a
pool-depth width-4 window are rare. The per-width strata are still reported,
still labelled treated/placebo from the arm's OWN engagement record, and a
difference-in-differences is still computed WHEN the control carries enough
steps; but the honest expectation is that it usually will not, and this tool
says "unavailable" rather than differencing against a handful of steps.

THE DIRECTION OF THE BIAS
-------------------------
Unlike the GQA-pair pair, the candidate arm here pays NO disclosed selector-side
retag: the TAW production selector changes which commit routine runs, not the
operand handed to a shared kernel. What the candidate arm does additionally pay
is the bundle validation at boot and one observer-accounting write at final
flush, both outside the timed window. So the pair is close to unbiased, and this
tool refuses to claim it is conservative in either direction, because it is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_b4_width4_window_reduce as w4  # noqa: E402
import fr13_b4_batch_conditioned_wall as bcw  # noqa: E402

SCHEMA = "fr13.b4_taw_native_production_width4_timing_pair.v1"
RUN_CLASS = "b4_taw_native_production_width4_screen"
TITLE = "B4 TAW NATIVE-PRECOMPUTE PRODUCTION AT THE WIDTH-4 OPERATING POINT"

ENGAGEMENT_RELPATH = (
    "logs/fr13_fixed32_taw_native_precompute.production_engagement.json"
)
ENGAGEMENT_SCHEMA = "fr13.fixed32.taw_native_precompute.production_engagement.v1"
BUNDLE_SCHEMA = "fr13.fixed32.taw_native_precompute.pass_bundle.v1"
TAW_CANDIDATE = "fixed32_all_parent_commit_v2"
TAW_PRODUCTION_ROUTE = "fixed32_native_precompute_production_candidate_return"

# The sealed batch-conditioned width-4 wall and its four-pass MDE. Read from the
# artifact rather than retyped: a hand-copied threshold is a threshold that can
# drift away from the evidence it claims to come from. This is the SAME sealed
# artifact the GQA-pair sibling judges on, because it is a property of the
# operating point and the instrument, not of any lever.
SEALED_MDE_RELPATH = (
    "results/fr13_b4_width4_nsys_20260813/fr13_b4_batch_conditioned_wall.json"
)
SEALED_MDE_SCHEMA = "fr13.b4_batch_conditioned_wall.v1"

# The Nsight attribution that sized the addressable pile. Also READ, not retyped.
SEALED_ATTRIBUTION_RELPATH = (
    "results/fr13_b4_width4_nsys_20260813/attribution_final.json"
)
# The two cfwd kernel groups README section 3 item 4 added together to get the
# "37.1 ms/step across 321k tiny instances" figure.
CFWD_SMALL_KERNEL_GROUPS = ("elementwise", "other")
ILLUSTRATIVE_RECOVERY_FRACTION = 0.10

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
    "TPS between an arm whose TAW commit ran the exact reference route and an "
    "arm whose TAW commit ran the byte-qualified native-precompute production "
    "candidate, where both arms served the IDENTICAL 128-query-row B4 geometry "
    "over the IDENTICAL byte-pinned 16-task pool at 4 slots, pinned the "
    "IDENTICAL FA2 production configuration, and differed in exactly one "
    "environment variable",
)

DOES_NOT_CLAIM: tuple[str, ...] = (
    "statistical significance. This is n=1 paired draw. It produces NO variance "
    "estimate of its own, and the sealed MDE it is judged against is a "
    "four-pass between-pass quantity, not the uncertainty of a single paired "
    "difference. A delta clearing the MDE is grounds for a four-pass paired "
    "campaign, NOT a promotion",
    "a pre-registered effect size. The width-4 attribution left CFWD "
    "small-kernel consolidation UNRANKED for want of a candidate, so no "
    "TAW-specific threshold was ever written down. The addressable pile is "
    "reported as SCALE only, and the sole decision threshold is the sealed MDE",
    "that a null means no effect. Ten percent of the 37.1 ms/step addressable "
    "pile is below the Hydra27 MDE and far below the Tail23 MDE, so a real "
    "consolidation of that size is INVISIBLE to this instrument at n=1",
    "a strong placebo. The TAW production selector is authorised at every "
    "qualified batch at or above the pinned minimum, so the untreated set is "
    "width 1 alone and it is usually too thin inside a width-4 window to serve "
    "as a difference-in-differences control",
    "a numerics verdict. Byte equality between the candidate and the exact "
    "reference is established by the raw-byte gate this run PRESENTS, never by "
    "this run: the production arm returns candidate products and compares "
    "nothing",
    "whole-arm throughput. Inherited in full from the width-4 window class: the "
    "drain phase is excluded BY CONSTRUCTION and the windowed rate must never "
    "be multiplied by an arm wall",
    "a cap verdict. b4_cap_applicable is false at B4 context sizes and nothing "
    "here changes that",
    "an agent-quality verdict. Resolve rate is neither read nor recorded. "
    "exact16 as QUALITY CONTROL at batched milestones is a separate gate",
    "a promotion or a citable seal. The width-4 window class is an INSTRUMENT "
    "with citable=false, and a screen built on it inherits that",
)


class PairError(RuntimeError):
    """The evidence does not support a paired TAW width-4 contrast."""


# --------------------------------------------------------------------------- #
# pre-registered thresholds and the addressable pile                            #
# --------------------------------------------------------------------------- #
def load_addressable_pile(repo: Path) -> dict[str, Any]:
    """Size the lever's target from the sealed attribution, by reading it."""
    path = repo / SEALED_ATTRIBUTION_RELPATH
    if not path.is_file() or path.is_symlink():
        raise PairError(
            f"no sealed width-4 attribution at {path}; the lever's addressable "
            "pile is DEFINED by it and is not reconstructible here"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    phases = payload.get("phase_kernels") or {}
    cfwd = phases.get("cfwd") or {}
    groups = cfwd.get("groups_ms_per_step") or {}
    if not groups:
        raise PairError("sealed attribution carries no cfwd kernel groups")
    detail: dict[str, Any] = {}
    total_ms = 0.0
    total_instances = 0
    for name in CFWD_SMALL_KERNEL_GROUPS:
        block = groups.get(name)
        if not isinstance(block, dict):
            raise PairError(f"sealed attribution has no cfwd group {name!r}")
        ms = block.get("ms_per_step")
        instances = block.get("instances")
        if not isinstance(ms, (int, float)) or isinstance(ms, bool):
            raise PairError(f"cfwd group {name!r} ms_per_step={ms!r} is not numeric")
        if not isinstance(instances, int) or isinstance(instances, bool):
            raise PairError(f"cfwd group {name!r} instances={instances!r} is not an int")
        detail[name] = {"ms_per_step": float(ms), "instances": int(instances)}
        total_ms += float(ms)
        total_instances += int(instances)
    phase = (payload.get("phase_projection") or {}).get("cfwd") or {}
    return {
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cfwd_phase_ms_per_step": phase.get("ms_per_step"),
        "cfwd_phase_span_ms_per_step": phase.get("span_ms_per_step"),
        "small_kernel_groups": list(CFWD_SMALL_KERNEL_GROUPS),
        "small_kernel_detail": detail,
        "small_kernel_ms_per_step": total_ms,
        "small_kernel_instances": total_instances,
        "role": (
            "SCALE, not a target. results/fr13_b4_width4_nsys_20260813/README.md "
            "section 3 item 4 left CFWD small-kernel consolidation UNRANKED "
            "because no candidate existed. This run is that candidate; the pile "
            "is what it is aimed at, and nothing about its size was "
            "pre-registered as a threshold"
        ),
    }


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
    # THE BLENDED BASIS AND ITS OWN MDE. The window reducer's `step_wall_ms` is
    # the width-BLENDED window mean, so a delta measured on it must be judged
    # against the BLEND's MDE. Judging a blended delta against the conditioned
    # threshold would silently apply a threshold ~9% too small. Both bases are
    # carried, each with its matching threshold -- identical reasoning, and
    # identical numbers, to the GQA-pair sibling, because the instrument is the
    # same instrument.
    blend = pooled[mode].get("sealed_rescaled_blend") or {}
    for key in ("mean_ms", "sd_ms", "mde_ms"):
        value = blend.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PairError(f"sealed blended wall {key}={value!r} is not numeric")
    pile = load_addressable_pile(repo)
    illustrative = pile["small_kernel_ms_per_step"] * ILLUSTRATIVE_RECOVERY_FRACTION
    return {
        "blended_basis": {
            "sealed_step_wall_ms": float(blend["mean_ms"]),
            "sealed_between_pass_sd_ms": float(blend["sd_ms"]),
            "mde_ms": float(blend["mde_ms"]),
            "role": (
                "the width-BLENDED window mean the width-4 window reducer emits "
                "directly. Reported with its OWN MDE so the comparison is "
                "basis-matched; the primary judgement uses the batch-conditioned "
                "basis, which is the basis the sealed campaign is stated on"
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
        "addressable_pile": pile,
        "illustrative_tenth_of_pile_ms": illustrative,
        "instrument_can_resolve_a_tenth_of_the_pile": illustrative >= float(
            block["mde_ms"]
        ),
        "no_pre_registered_effect_size": (
            "results/fr13_b4_width4_nsys_20260813/README.md section 3 item 4 "
            "records CFWD small-kernel consolidation as UNRANKED, 'not yet a "
            "lever because no candidate exists'. No TAW effect size was ever "
            "pre-registered, so the sealed MDE is the only threshold applied "
            "and the pile is reported as scale"
        ),
    }


# --------------------------------------------------------------------------- #
# TAW credential identities                                                     #
# --------------------------------------------------------------------------- #
def read_engagement(arm_dir: Path) -> dict[str, Any] | None:
    """Read one arm's TAW production engagement record, or report its absence.

    Absence is EVIDENCE here, not a missing file: the stock arm's correctness is
    exactly that the kernel had nothing to write.
    """
    path = arm_dir / ENGAGEMENT_RELPATH
    if not path.is_file() or path.is_symlink():
        return None
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise PairError(f"{path} is not a JSON object")
    return {
        **payload,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def authorised_widths(engagement: dict[str, Any]) -> tuple[int, ...]:
    """Which widths the candidate arm was AUTHORISED to serve, from its record.

    AUTHORISED, not observed. A width the bundle qualified but which happened to
    carry no events arm-wide is still a width the selector WOULD have served, so
    calling it a placebo control would hand difference-in-differences a stratum
    that is only untreated by accident. The engagement record carries both, and
    the authorised set is the one that decides roles; the observed set is
    checked to be a subset of it and is reported for audit.
    """
    qualified = engagement.get("qualified_batches")
    minimum = engagement.get("pinned_min_batch")
    if (
        not isinstance(qualified, list)
        or not qualified
        or not all(isinstance(batch, int) for batch in qualified)
        or not isinstance(minimum, int)
        or isinstance(minimum, bool)
    ):
        raise PairError(
            "the TAW engagement does not declare a usable qualified batch set; "
            "the treated widths cannot be resolved and must not be guessed"
        )
    authorised = tuple(sorted(batch for batch in qualified if batch >= minimum))
    if not authorised:
        raise PairError(
            "the TAW engagement authorises no batch at or above its pinned "
            "minimum; there is no treated width"
        )
    served = engagement.get("served_candidate_calls_by_batch")
    if not isinstance(served, dict) or not served:
        raise PairError("the TAW engagement records no served-candidate calls")
    stray = sorted(int(key) for key in served if int(key) not in authorised)
    if stray:
        raise PairError(
            f"the TAW engagement records served calls at unauthorised batches {stray}"
        )
    return authorised


def validate_pair_engagement(
    stock_engagement: dict[str, Any] | None,
    candidate_engagement: dict[str, Any] | None,
    *,
    expected_mode: str,
    expected_bundle_sha256: str,
    expected_source_contract_sha256: str,
) -> None:
    """The single-variable delta, proved from the two arms' own records."""
    if stock_engagement is not None:
        raise PairError(
            "the stock arm emitted a TAW production engagement record; the "
            "candidate selector leaked across the pair and the contrast is "
            "invalid"
        )
    if candidate_engagement is None:
        raise PairError(
            "the candidate arm emitted no TAW production engagement record; it "
            "did not serve the candidate commit and there is nothing to time"
        )
    e = candidate_engagement
    if (
        e.get("schema") != ENGAGEMENT_SCHEMA
        or e.get("status") != "ENGAGED"
        or e.get("candidate") != TAW_CANDIDATE
        or e.get("route") != TAW_PRODUCTION_ROUTE
        or e.get("candidate_returned") is not True
        or e.get("reference_returned") is not False
        or e.get("observer_accounting_only") is not True
        or e.get("flush_action") != "final"
        or e.get("finalized_by_fixed32_flush") is not True
        or e.get("mode") != expected_mode
    ):
        raise PairError(
            "the candidate arm did not serve the TAW native production "
            f"candidate on the declared mode: {e!r}"
        )
    if e.get("source_contract_sha256") != expected_source_contract_sha256:
        raise PairError(
            "the candidate arm's TAW source contract "
            f"{e.get('source_contract_sha256')!r} is not the contract this "
            f"verdict was bound to {expected_source_contract_sha256!r}"
        )
    if expected_bundle_sha256 and e.get("production_bundle_sha256") != (
        expected_bundle_sha256
    ):
        raise PairError(
            "the candidate arm served PASS bundle "
            f"{e.get('production_bundle_sha256')!r}, not the bundle the runner "
            f"validated at preflight {expected_bundle_sha256!r}"
        )
    served = e.get("served_candidate_calls")
    if not isinstance(served, int) or isinstance(served, bool) or served < 1:
        raise PairError(
            f"the candidate arm records served_candidate_calls={served!r}"
        )


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
    """Condition this arm's windowed wall on served batch width == slots."""
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


MIN_CONTROL_STEPS_FOR_DID = 100


def per_width_strata(
    stock_bc: dict[str, Any],
    candidate_bc: dict[str, Any],
    treated: tuple[int, ...],
) -> dict[str, Any]:
    """Per-width detail, with roles read off the engagement record.

    The strata are reported for EVERY shared served width, labelled treated or
    placebo from the authorised set. Where a placebo width carries enough steps
    it is used as a difference-in-differences control; where it does not, the
    correction is declared unavailable rather than computed from noise.

    Be clear about what this design can and cannot do. For the FA2 GQA-pair
    lever the untreated widths were a genuinely strong natural placebo. Here the
    TAW selector is authorised at every qualified batch at or above the pinned
    minimum, so the placebo set is width 1 alone, and width-1 steps inside a
    pool-depth width-4 window are rare. The check is kept because a leak at
    width 1 would still be decisive evidence of arm-to-arm confound; it is just
    much less likely to be ARMED than its FA2 counterpart, and pretending
    otherwise would overstate the design.
    """
    s_by = stock_bc.get("by_width") or {}
    c_by = candidate_bc.get("by_width") or {}
    widths = sorted({int(w) for w in s_by} & {int(w) for w in c_by})
    if not widths:
        return {"available": False, "reason": "no shared served widths"}
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
    controls = [
        r
        for r in rows
        if not r["candidate_engaged"] and r["stock_steps"] >= MIN_CONTROL_STEPS_FOR_DID
    ]
    treated_row = next(r for r in rows if r["width"] == headline)
    out: dict[str, Any] = {
        "available": True,
        "treated_width": headline,
        "treated_widths": list(treated),
        "rows": rows,
        "placebo_note": (
            "widths outside the authorised set run the IDENTICAL exact "
            "reference commit on both arms, so any delta there is arm-to-arm "
            "confound, not kernel. The TAW selector authorises every qualified "
            "batch at or above its pinned minimum, so that set is small"
        ),
        "min_control_steps_for_did": MIN_CONTROL_STEPS_FOR_DID,
    }
    if not controls:
        out["difference_in_differences"] = {
            "available": False,
            "reason": (
                "no untreated width carries enough steps to serve as a control. "
                "This is the EXPECTED case for the TAW lever, not a defect: the "
                "authorised set leaves only width 1 untreated"
            ),
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
    return out


def judge(step_wall_improvement_ms: float, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Apply the one pre-registered threshold there is. None is chosen here."""
    mde = thresholds["mde_ms"]
    sd = thresholds["sealed_between_pass_sd_ms"]
    pile = thresholds["addressable_pile"]["small_kernel_ms_per_step"]
    tenth = thresholds["illustrative_tenth_of_pile_ms"]
    d = step_wall_improvement_ms

    if d >= mde:
        band = "GAIN_CLEARS_FOUR_PASS_MDE"
        disposition = "LEVER_SURVIVES_FUND_A_FOUR_PASS_CAMPAIGN"
        reading = (
            "the candidate moves the width-4 step wall by more than the sealed "
            "four-pass MDE. At n=1 this is a SCREEN result: it is grounds to "
            "fund a four-pass paired campaign, not a promotion"
        )
    elif d > 0.0:
        band = "GAIN_BELOW_FOUR_PASS_MDE"
        disposition = "NOT_RESOLVED_AT_THIS_SIZE"
        reading = (
            "the candidate is nominally faster but by less than the sealed "
            "four-pass MDE. Read this as UNRESOLVED, not as null: the "
            f"addressable pile is {pile:.1f} ms/step and a tenth of it "
            f"({tenth:.1f} ms/step) is already below the MDE, so an effect of "
            "the size the attribution thought plausible could not have been "
            "seen here. Resolving it needs passes, not a different lever"
        )
    else:
        band = "NO_GAIN_OR_REGRESSION"
        disposition = "NOT_RESOLVED_AT_THIS_SIZE"
        reading = (
            "the candidate is not faster at the width-4 operating point. That "
            "is a real observation and it is not a null: with an MDE of "
            f"{mde:.2f} ms against a {pile:.1f} ms/step addressable pile, this "
            "single draw cannot distinguish a small gain, no effect, and a "
            "small regression. A four-pass campaign is what would separate them"
        )
    return {
        "step_wall_improvement_ms": d,
        "band": band,
        "lever_disposition": disposition,
        "reading": reading,
        "clears_four_pass_mde": d >= mde,
        "four_pass_mde_ms": mde,
        "addressable_pile_ms_per_step": pile,
        "illustrative_tenth_of_pile_ms": tenth,
        "instrument_can_resolve_a_tenth_of_the_pile": bool(tenth >= mde),
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
    # The windowing is delegated WHOLE and UNCHANGED.
    stock_record = w4.reduce_window_arm(stock_dir, mode=mode, pass_index=0)
    candidate_record = w4.reduce_window_arm(candidate_dir, mode=mode, pass_index=0)

    # Credentials are validated BEFORE any number is reported, so an invalid
    # pair never prints a delta a reader could quote.
    validate_pair_engagement(
        provenance.get("stock_engagement"),
        provenance.get("candidate_engagement"),
        expected_mode=mode,
        expected_bundle_sha256=provenance.get("production_bundle_sha256") or "",
        expected_source_contract_sha256=(
            provenance.get("source_contract_sha256") or ""
        ),
    )
    treated = authorised_widths(provenance["candidate_engagement"])

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

    strata = (
        per_width_strata(stock_bc, candidate_bc, treated)
        if (stock_bc["available"] and candidate_bc["available"])
        else {"available": False, "reason": "no batch-conditioned detail"}
    )
    step_wall_basis["per_width_strata"] = strata
    if strata.get("available"):
        leaks = [
            r
            for r in strata["rows"]
            if not r["candidate_engaged"]
            and r["stock_steps"] >= strata["min_control_steps_for_did"]
            and r["improvement_ms"] >= basis_thresholds["mde_ms"]
        ]
        armed = any(
            not r["candidate_engaged"]
            and r["stock_steps"] >= strata["min_control_steps_for_did"]
            for r in strata["rows"]
        )
        verdict["placebo_armed"] = armed
        verdict["placebo_clean"] = not leaks
        verdict["placebo_leak_widths"] = [r["width"] for r in leaks]
        verdict["placebo_reading"] = (
            "no untreated width carries enough steps to test, so the placebo "
            "check is NOT ARMED and 'clean' here means 'unexamined'. That is "
            "the expected state for this lever"
            if not armed
            else (
                "no untreated width shows a gain at or above the MDE, so the "
                "effect is localised to the widths where the candidate is served"
                if not leaks
                else "an untreated width shows a gain at or above the MDE: the "
                "headline delta is contaminated by arm-to-arm confound and must "
                "NOT be read as a kernel result"
            )
        )

    subset_ids = sorted(json.loads(subset.read_text(encoding="ascii"))["instance_ids"])

    return {
        "schema": SCHEMA,
        "run_class": RUN_CLASS,
        "title": TITLE,
        "verdict": verdict["lever_disposition"],
        "analysis_valid": True,
        "citable": False,
        "citable_reason": (
            "n=1 paired screen built on the width-4 window INSTRUMENT class "
            "(itself citable=false). It reports a delta against one "
            "pre-registered threshold; it promotes nothing and seals nothing"
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
        "only_arm_delta": "TAW_exact_reference_commit_to_native_precompute_production",
        "arm_delta_disclosure": {
            "served_commit": (
                "fixed32_pytorch_exact_float_triton_integer_commit -> "
                + TAW_PRODUCTION_ROUTE
            ),
            "candidate_only_overhead": (
                "PASS bundle validation at boot and one observer-accounting "
                "write at final flush, both OUTSIDE the timed window"
            ),
            "bias_direction": (
                "approximately unbiased. Unlike the GQA-pair pair there is no "
                "selector-side retag charged to the candidate inside the "
                "measured region, so this tool does NOT claim the pair is "
                "conservative in either direction"
            ),
            "fa2_configuration": (
                "PINNED IDENTICALLY ON BOTH ARMS to the promoted production "
                "default (gqa_pair). The FA2 candidate became the registry "
                "default at 32e240e15, so leaving it unset would have let the "
                "two arms differ in FA2 as well as in TAW and the delta would "
                "no longer be single-variable"
            ),
            "candidate_widths": None,
        },
        "step_wall_basis": step_wall_basis,
        "primary_statistic": "step_wall_ms",
        "primary_statistic_reason": (
            "the width holds the served batch fixed, so the aggregate is no "
            "longer a free parameter the schedule can inflate, and the sealed "
            "MDE this screen is judged against is stated on step wall. "
            "per_request_step_tps is reported alongside as the operating "
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
    pile = t["addressable_pile"]
    lines += [
        f"  sealed width-4 wall ({t['topology']}): "
        f"{t['sealed_width4_step_wall_ms']:.2f} ms  "
        f"(4-pass SD {t['sealed_between_pass_sd_ms']:.2f}, "
        f"MDE {t['mde_ms']:.2f} ms)",
        f"  addressable CFWD small-kernel pile: "
        f"{pile['small_kernel_ms_per_step']:.2f} ms/step across "
        f"{pile['small_kernel_instances']} instances  "
        f"(a tenth = {t['illustrative_tenth_of_pile_ms']:.2f} ms, "
        f"resolvable={t['instrument_can_resolve_a_tenth_of_the_pile']})",
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
    pl = b.get("per_width_strata") or {}
    if pl.get("available"):
        lines += [
            "",
            "  PER-WIDTH STRATA (candidate authorised at widths "
            + ", ".join(str(w) for w in pl.get("treated_widths", []))
            + ")",
            f"    {'width':>5} {'stock ms':>10} {'cand ms':>10} {'impr ms':>9} {'role':>8}",
        ]
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
        else:
            lines.append(f"    DiD unavailable: {did.get('reason')}")
    v = payload["verdict_detail"]
    if "placebo_clean" in v:
        lines.append(
            f"  placebo armed: {v['placebo_armed']}  clean: {v['placebo_clean']}  "
            f"{v['placebo_reading']}"
        )
    sd_units = v.get("improvement_in_sealed_sd_units")
    sd_note = f" ({sd_units:+.2f} sealed SD)" if sd_units is not None else ""
    lines += [
        "",
        f"  step-wall improvement: {v['step_wall_improvement_ms']:+.3f} ms{sd_note}",
        f"  band:                  {v['band']}",
        f"  clears 4-pass MDE:     {v['clears_four_pass_mde']}",
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
            pile = t["addressable_pile"]
            print(
                f"  {mode:18} sealed wall {t['sealed_width4_step_wall_ms']:.2f} ms  "
                f"SD {t['sealed_between_pass_sd_ms']:.2f}  "
                f"MDE {t['mde_ms']:.2f} ms  "
                f"pile {pile['small_kernel_ms_per_step']:.2f} ms  "
                f"tenth {t['illustrative_tenth_of_pile_ms']:.2f} ms  "
                f"resolvable={t['instrument_can_resolve_a_tenth_of_the_pile']}"
            )
        except (PairError, OSError, json.JSONDecodeError) as error:
            problems.append(f"{mode}: {error}")
    for name in ("reduce_window_arm", "Width4Error", "MIN_WINDOW_STEPS"):
        if not hasattr(w4, name):
            problems.append(f"width-4 window reducer has no {name}")
    if not hasattr(bcw, "analyse_arm"):
        problems.append("batch-conditioned wall module has no analyse_arm")
    subset = repo / "config/fr13_fixed32/subset_b4_sixteen.json"
    if not subset.is_file():
        problems.append(f"missing {subset}")
    else:
        ids = json.loads(subset.read_text(encoding="ascii"))["instance_ids"]
        if len(ids) != 16:
            problems.append(f"pool subset holds {len(ids)} ids, expected 16")
    # The engagement artifact this verdict is DEFINED by must be something the
    # runtime actually writes. Six campaign fossils were runners bound to an
    # artifact nothing ever wrote, so the emitter is resolved here, on CPU.
    kernel = repo / "scripts/fr13_device_multidraft_kernel.py"
    if not kernel.is_file():
        problems.append(f"missing {kernel}")
    else:
        text = kernel.read_text(encoding="utf-8")
        for needle in (
            "def fr13_fixed32_taw_native_production_engagement_finalize",
            ENGAGEMENT_SCHEMA,
            ENGAGEMENT_RELPATH.rsplit("/", 1)[-1],
            TAW_PRODUCTION_ROUTE,
        ):
            if needle not in text:
                problems.append(f"kernel module does not emit {needle!r}")
    if problems:
        for problem in problems:
            print(f"class-9 FAIL-LOUD [taw-width4-pair self-check]: {problem}", file=sys.stderr)
        return 2
    print("  TAW width-4 pair reducer self-check OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reduce a two-arm TAW native production width-4 timing pair."
    )
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--runroot", type=Path, default=None)
    parser.add_argument("--stock-arm", type=Path, default=None)
    parser.add_argument("--candidate-arm", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="tail6_fixed32")
    parser.add_argument("--source-commit", type=str, default="")
    parser.add_argument(
        "--subset", type=Path, default=Path("config/fr13_fixed32/subset_b4_sixteen.json")
    )
    parser.add_argument("--production-bundle-sha256", type=str, default="")
    parser.add_argument("--byte-gate-sha256", type=str, default="")
    parser.add_argument("--source-contract-sha256", type=str, default="")
    parser.add_argument("--runner-sha256", type=str, default="")
    parser.add_argument("--kernel-source-sha256", type=str, default="")
    parser.add_argument("--repo", type=Path, default=SCRIPTS.parent)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check(args.repo)

    for name in ("runroot", "stock_arm", "candidate_arm"):
        if getattr(args, name) is None:
            print(
                f"class-9 FAIL-LOUD [taw-width4-pair]: "
                f"--{name.replace('_', '-')} is required",
                file=sys.stderr,
            )
            return 2

    try:
        thresholds = load_sealed_thresholds(args.repo, args.mode)
        provenance = {
            "source_commit": args.source_commit,
            "production_bundle_sha256": args.production_bundle_sha256,
            "byte_gate_sha256": args.byte_gate_sha256,
            "source_contract_sha256": args.source_contract_sha256,
            "runner_sha256": args.runner_sha256,
            "kernel_source_sha256": args.kernel_source_sha256,
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
        candidate_engagement = provenance.get("candidate_engagement") or {}
        payload.setdefault("arm_delta_disclosure", {})
        if isinstance(payload.get("arm_delta_disclosure"), dict):
            payload["arm_delta_disclosure"]["candidate_widths"] = list(
                authorised_widths(candidate_engagement)
            )
    except (PairError, w4.Width4Error, OSError, json.JSONDecodeError) as error:
        print(f"class-9 FAIL-LOUD [taw-width4-pair]: {error}", file=sys.stderr)
        return 2

    out = args.out or (args.runroot / "taw_width4_timing_pair.json")
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
