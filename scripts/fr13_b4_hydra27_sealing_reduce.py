#!/usr/bin/env python3
"""Reduce the Hydra27-only paired sealing campaign to a citable verdict.

WHAT THIS TURNS INTO WHAT
    The width-4 screen returned REVERSES_THE_EXACT4_NULL at n=1 and said in its
    own verdict_detail: "At n=1 this is a SCREEN result: it is grounds to fund a
    four-pass paired campaign, not a promotion."  This reducer consumes that
    campaign -- N paired passes, each one stock arm against the padded GQA-pair
    candidate arm at the width-4 operating point on Hydra27 -- and reports the
    BETWEEN-PASS interval that a single pass cannot produce.

WHY BETWEEN-PASS AND NOT WITHIN-RUN
    The pair reducer's own uncertainty anchor is a sealed between-pass SD taken
    across four passes whose two arms are different TOPOLOGIES, and it flags that
    honestly: pairing stock against candidate INSIDE one pass removes shared host
    and task-difficulty variation, so the true SD of that difference is smaller
    by an amount one pass cannot quantify.  This campaign measures that SD
    directly -- n paired differences, one per pass -- so the interval is finally
    matched to the estimator.

WHAT MAKES A CAMPAIGN ADMISSIBLE
    * N in {4, 16}.  Only those land on the repo's pinned one-sided t criticals
      (df 3 or 15).  A short campaign is a SCREEN: NOT_EVALUATED_INSUFFICIENT_
      PASSES, citable=false.  No df=1 critical is invented to rescue a short run.
    * ARM-ORDER BALANCE.  The two arms here are stock and candidate, so arm
      position aliases directly into the contrast.  A campaign whose SC and CS
      pass counts differ is refused: it cannot separate "candidate is faster"
      from "the second arm of a pass is faster".
    * ONE STACK.  Every included pass must carry the same source commit, dual
      gate, candidate binary, subset, topology and candidate scope widths.  A
      campaign assembled from passes that served different stacks is not a
      campaign, it is a mixture.

WHAT IT DOES NOT CLAIM
    Hydra27 only.  Not Tail23, not the formal statistical hardware-floor
    acceptance gate, and not the exact16 agent-quality band.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The one-sided 95% Student-t criticals, imported rather than retyped so this
# reducer and the formal floor gate can never drift apart.
from fr13_b4_floor_gate_reduce import T95_ONE_SIDED  # noqa: E402

SCHEMA = "fr13.b4_hydra27_sealing_campaign.v1"
CLASSIFICATION = "real_swe_verified_pool16_b4_hydra27_paired_sealing_campaign"
PAIR_SCHEMA = "fr13.b4_gqa_pair_width4_timing_pair.v1"
TOPOLOGY = "hydra27_fixed32"
LOGICAL_TOPOLOGY = "Hydra27"
ADMISSIBLE_PASS_COUNTS = (4, 16)
# The sealed four-pass MDE for this topology, reported as an anchor. The
# campaign's OWN interval is the verdict; the sealed MDE is context.
SEALED_HYDRA27_MDE_MS = 4.204845067020671
POOL16_SUBSET_SHA256 = (
    "47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c"
)
EXPECTED_SCOPE_WIDTHS = [3, 4]
PASS_DIR_RE = re.compile(r"^run_(\d{2})$")


class SealError(ValueError):
    """The campaign is not the balanced, single-stack evidence it must be."""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        raise SealError("a sample SD needs at least two passes")
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def one_sided_interval(values: list[float]) -> dict[str, Any]:
    """Mean, SD and the one-sided 95% lower bound on the paired difference."""
    n = len(values)
    df = n - 1
    if df not in T95_ONE_SIDED:
        raise SealError(
            f"no pinned one-sided critical for df={df}; admissible n are "
            f"{ADMISSIBLE_PASS_COUNTS}"
        )
    mu = _mean(values)
    sd = _sample_sd(values)
    half = T95_ONE_SIDED[df] * sd / math.sqrt(n)
    return {
        "n": n,
        "mean_ms": mu,
        "sd_ms": sd,
        "t_0_95_one_sided": T95_ONE_SIDED[df],
        "half_width_ms": half,
        "lower_bound_ms": mu - half,
        "excludes_zero": (mu - half) > 0.0,
    }


def _load_pass(run_dir: Path) -> dict[str, Any]:
    """Read one pass: its pair verdict plus the arm order it actually ran."""
    pair_path = run_dir / "width4_timing_pair.json"
    meta_path = run_dir / "launcher_meta.txt"
    if not pair_path.is_file():
        raise SealError("pass has no width4_timing_pair.json")
    payload = json.loads(pair_path.read_text())
    if payload.get("schema") != PAIR_SCHEMA:
        raise SealError(f"pair schema drifted: {payload.get('schema')!r}")
    if not meta_path.is_file():
        raise SealError("pass has no launcher_meta.txt")
    meta: dict[str, str] = {}
    for line in meta_path.read_text().splitlines():
        if "=" in line and not line.startswith("arm="):
            key, _, value = line.partition("=")
            meta.setdefault(key.strip(), value.strip())
    arm_order = meta.get("arm_order")
    if arm_order not in ("SC", "CS"):
        raise SealError(
            "pass does not record a valid arm_order; it predates the paired "
            "campaign runner and its arm position cannot be established"
        )
    detail = payload.get("verdict_detail") or {}
    improvement = detail.get("step_wall_improvement_ms")
    if not isinstance(improvement, (int, float)):
        raise SealError("pass carries no batch-conditioned width-4 improvement")
    basis = payload.get("step_wall_basis") or {}
    per_width = basis.get("per_width_placebo") or {}
    if not per_width.get("available"):
        raise SealError("pass carries no per-width strata")
    return {
        "run_dir": str(run_dir),
        "arm_order": arm_order,
        "pass_index": meta.get("pass_index"),
        "improvement_ms": float(improvement),
        "verdict": payload.get("verdict"),
        "placebo_clean": bool(detail.get("placebo_clean")),
        "per_width_rows": per_width.get("rows") or [],
        "treated_widths": per_width.get("treated_widths") or [],
        "difference_in_differences": per_width.get("difference_in_differences"),
        "source_commit": payload.get("source_commit"),
        "topology": payload.get("topology"),
        "subset_sha256": payload.get("subset_sha256"),
        "provenance": payload.get("provenance") or {},
    }


def _stack_key(record: dict[str, Any]) -> tuple:
    prov = record["provenance"]
    engagement = prov.get("candidate_engagement") or {}
    return (
        record["source_commit"],
        record["topology"],
        record["subset_sha256"],
        prov.get("dual_gate_sha256"),
        prov.get("candidate_so_sha256") or engagement.get("candidate_so_sha256"),
        tuple(engagement.get("candidate_scope_widths") or ()),
        engagement.get("candidate_scope"),
    )


def collect(campaign_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for child in sorted(campaign_root.iterdir()):
        if not child.is_dir() or not PASS_DIR_RE.match(child.name):
            continue
        try:
            included.append(_load_pass(child))
        except (SealError, OSError, json.JSONDecodeError, KeyError) as error:
            excluded.append(
                {"run_dir": child.name, "reason": f"{type(error).__name__}: {error}"}
            )
    return included, excluded


def per_width_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-width strata across passes.

    A width is only reported with an interval if it appears in EVERY included
    pass; a width present in some passes and not others would otherwise let the
    pass composition, rather than the kernel, move its mean.
    """
    per_pass: list[dict[int, dict[str, Any]]] = [
        {int(row["width"]): row for row in rec["per_width_rows"]} for rec in records
    ]
    common = set(per_pass[0])
    for table in per_pass[1:]:
        common &= set(table)
    rows = []
    for width in sorted(common):
        improvements = [table[width]["improvement_ms"] for table in per_pass]
        treated = all(table[width]["candidate_engaged"] for table in per_pass)
        mixed = treated is False and any(
            table[width]["candidate_engaged"] for table in per_pass
        )
        entry: dict[str, Any] = {
            "width": width,
            "role": "treated" if treated else "placebo",
            "stock_steps_total": sum(table[width]["stock_steps"] for table in per_pass),
            "candidate_steps_total": sum(
                table[width]["candidate_steps"] for table in per_pass
            ),
            "per_pass_improvement_ms": improvements,
        }
        if mixed:
            entry["role"] = "INCONSISTENT"
            entry["note"] = (
                "this width was treated in some passes and not others; the "
                "campaign does not pool it"
            )
        else:
            entry.update(one_sided_interval(improvements))
        rows.append(entry)
    widths_dropped = sorted(
        {w for table in per_pass for w in table} - common
    )
    return {
        "rows": rows,
        "widths_present_in_every_pass": sorted(common),
        "widths_dropped_for_uneven_presence": widths_dropped,
        "note": (
            "widths outside the declared treated set run the IDENTICAL stock "
            "dispatch on both arms, so a gain there is arm-to-arm confound and "
            "not kernel; the treated set is read off each pass's engagement "
            "record, never inferred from the data"
        ),
    }


def reduce_campaign(
    campaign_root: Path,
    *,
    source_commit: str,
    dual_gate_sha256: str,
    min_passes: int,
) -> dict[str, Any]:
    included, excluded = collect(campaign_root)
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "campaign_root": str(campaign_root),
        "topology": TOPOLOGY,
        "logical_topology": LOGICAL_TOPOLOGY,
        "source_commit": source_commit,
        "dual_gate_sha256": dual_gate_sha256,
        "passes_included": len(included),
        "passes_excluded": excluded,
        "sealed_reference_mde_ms": SEALED_HYDRA27_MDE_MS,
        "does_not_claim": [
            "Tail23 -- this campaign seals Hydra27 only, per Mark's 2026-08-13 ruling",
            "the formal statistical hardware-floor acceptance gate",
            "the exact16 agent-quality band (exact16 remains QUALITY CONTROL)",
            "any claim about widths outside the declared candidate scope",
        ],
    }
    if not included:
        out["verdict"] = "NOT_EVALUATED_NO_ADMISSIBLE_PASSES"
        out["citable"] = False
        return out

    stacks = {_stack_key(record) for record in included}
    if len(stacks) != 1:
        raise SealError(
            "included passes did not all serve the same stack; a campaign "
            "assembled from different stacks is a mixture, not a campaign"
        )
    record0 = included[0]
    if record0["topology"] != TOPOLOGY:
        raise SealError(f"campaign topology drifted: {record0['topology']!r}")
    if record0["source_commit"] != source_commit:
        raise SealError("pass source commit does not match the campaign commit")
    if (record0["provenance"] or {}).get("dual_gate_sha256") != dual_gate_sha256:
        raise SealError("pass dual gate does not match the campaign dual gate")
    if record0["subset_sha256"] != POOL16_SUBSET_SHA256:
        raise SealError("campaign did not run the byte-pinned pool16 evidence set")
    scope_widths = list(record0["treated_widths"])
    if scope_widths != EXPECTED_SCOPE_WIDTHS:
        out["scope_widths_note"] = (
            f"treated widths are {scope_widths}, not the expected "
            f"{EXPECTED_SCOPE_WIDTHS}; reported, not assumed"
        )
    out["treated_widths"] = scope_widths
    out["candidate_scope"] = (
        (record0["provenance"].get("candidate_engagement") or {}).get("candidate_scope")
    )

    orders = [record["arm_order"] for record in included]
    balance = {"SC": orders.count("SC"), "CS": orders.count("CS")}
    out["arm_order_balance"] = balance
    out["arm_order_per_pass"] = orders

    if len(included) < min_passes or len(included) not in ADMISSIBLE_PASS_COUNTS:
        out["verdict"] = "NOT_EVALUATED_INSUFFICIENT_PASSES"
        out["citable"] = False
        out["citable_reason"] = (
            f"{len(included)} admissible passes; the pinned one-sided criticals "
            f"exist only for n in {ADMISSIBLE_PASS_COUNTS} and no df=1 constant "
            "is invented to rescue a short campaign"
        )
        return out
    if balance["SC"] != balance["CS"]:
        out["verdict"] = "NOT_EVALUATED_ARM_ORDER_UNBALANCED"
        out["citable"] = False
        out["citable_reason"] = (
            f"arm order is unbalanced ({balance}); with stock and candidate as "
            "the two arms, position aliases directly into the contrast, so an "
            "unbalanced campaign cannot separate the kernel from the warm-host "
            "advantage of running second"
        )
        return out

    improvements = [record["improvement_ms"] for record in included]
    interval = one_sided_interval(improvements)
    out["batch_conditioned_width4"] = interval
    out["per_width"] = per_width_table(included)
    out["placebo_clean_every_pass"] = all(r["placebo_clean"] for r in included)
    out["per_pass"] = [
        {
            "run_dir": Path(r["run_dir"]).name,
            "arm_order": r["arm_order"],
            "improvement_ms": r["improvement_ms"],
            "verdict": r["verdict"],
            "placebo_clean": r["placebo_clean"],
        }
        for r in included
    ]

    if interval["excludes_zero"] and out["placebo_clean_every_pass"]:
        out["verdict"] = "SEALED_HYDRA27_GAIN"
        out["citable"] = True
        out["citable_reason"] = (
            "n paired passes with balanced arm order on one stack; the one-sided "
            "95% lower bound on the paired batch-conditioned width-4 improvement "
            "excludes zero and no untreated width shows a gain in any pass"
        )
    elif interval["excludes_zero"]:
        out["verdict"] = "GAIN_WITH_PLACEBO_LEAK"
        out["citable"] = False
        out["citable_reason"] = (
            "the interval excludes zero but at least one pass shows a gain at an "
            "untreated width, so the contrast is measuring arm-to-arm confound "
            "as well as kernel"
        )
    else:
        out["verdict"] = "NO_SEALED_GAIN"
        out["citable"] = False
        out["citable_reason"] = (
            "the one-sided 95% lower bound on the paired improvement includes "
            "zero"
        )
    return out


def self_check() -> int:
    """Resolve every constant and exercise the math before any GPU time."""
    assert T95_ONE_SIDED[3] == 2.3533634348018264, "df=3 critical drifted"
    assert T95_ONE_SIDED[15] == 1.7530503556925547, "df=15 critical drifted"
    exact = one_sided_interval([10.0, 10.0, 10.0, 10.0])
    assert exact["sd_ms"] == 0.0 and exact["excludes_zero"], exact
    spread = one_sided_interval([0.0, 10.0, -10.0, 0.0])
    assert not spread["excludes_zero"], spread
    known = one_sided_interval([1.0, 2.0, 3.0, 4.0])
    assert abs(known["mean_ms"] - 2.5) < 1e-12, known
    assert abs(known["sd_ms"] - 1.2909944487358056) < 1e-12, known
    # t * sd / sqrt(n) = 2.3533634348018264 * 1.2909944487358056 / 2
    assert abs(known["half_width_ms"] - 1.519089565093493) < 1e-9, known
    assert abs(known["lower_bound_ms"] - 0.980910434906507) < 1e-9, known
    for bad in ([1.0], [1.0, 2.0, 3.0]):
        try:
            one_sided_interval(bad)
        except SealError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"n={len(bad)} must not yield an interval")
    print("width-4 Hydra27 sealing reducer self-check OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--dual-gate-sha256")
    parser.add_argument("--min-passes", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check()
    missing = [
        name
        for name, value in (
            ("--campaign-root", args.campaign_root),
            ("--source-commit", args.source_commit),
            ("--dual-gate-sha256", args.dual_gate_sha256),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    try:
        payload = reduce_campaign(
            args.campaign_root,
            source_commit=args.source_commit,
            dual_gate_sha256=args.dual_gate_sha256,
            min_passes=args.min_passes,
        )
    except SealError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 3
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.out.write_text(text)
    print(text)
    return 0 if payload.get("citable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
