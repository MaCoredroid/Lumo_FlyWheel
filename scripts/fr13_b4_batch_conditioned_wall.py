#!/usr/bin/env python3
"""Condition the width-4 step wall on ENGINE BATCH WIDTH, from banked evidence.

WHAT THIS CLOSES
----------------
`results/fr13_b4_refill_citable_20260812/width4_window.md` §3 names one gap as
the largest remaining, and declares it out of reach:

    A true `batch == 4` filter is **not derivable from this evidence**: the
    census resolves batch width per step but carries **no per-step wall**, so no
    wall statistic can be conditioned on it. This is the single largest
    remaining gap and it is what §4 asks for.

It is derivable, and it needs no new instrumentation. The premise is true of the
work census but not of the evidence set: the SFWD timer's per-step samples
sidecar (`fr13.sfwd_per_step_samples.v2`, written next to every arm) carries

    wall_fwd_indices[]  absolute forward-step index
    wall_ms[]           that step's measured full-step wall
    wall_drafts[]       that step's spec events == the engine batch width

so wall and width are already in one record, keyed by the same absolute step
index the census and the counter bracket use. Verified on the banked pool16
tail23 pass-0 arm: the sidecar's per-step width equals the census `batch_size`
on all 9385 shared steps, with zero mismatches.

WHY THE ANSWER MOVES, AND IN WHICH DIRECTION
--------------------------------------------
The sealed window is a POOL-DEPTH window, and §3 already publishes the residual:
inside it the engine batch was width 4 on only 62.5%-70.0% of steps. So the
sealed `step_wall_ms` is a width-blend, and width-3 steps are much cheaper than
width-4 steps. Conditioning removes that dilution and the number goes UP.

TWO EFFECTS, REPORTED SEPARATELY
--------------------------------
The gap between the sealed number and a width-4 mean is NOT all width, and
collapsing the two would overstate the width effect by about a quarter:

1. BASIS. The sealed `step_wall_ms` is a cross-population rescale,
   `(wall_seconds / wall_drafts) * (forward_drafts / forward_steps)` -- per-event
   wall from the WALL-BRACKETED population times events/step from the ALL-STEPS
   population. That is the documented "step_wall_ms and events_per_step do not
   share a basis" caveat, and it alone accounts for about +5.4 ms. The direct
   mean of the same wall samples is the like-for-like blend.
2. WIDTH. Direct blend -> width-4 only. About +18 ms.

Conditioning also DISSOLVES caveat 1 rather than merely measuring it: once the
population is `width == 4`, events/step is exactly 4.0 by construction, so
per-event and per-step wall differ by exactly 4 and no cross-population rescale
exists to disagree about.

This tool is OFFLINE and READ-ONLY over banked arms. It touches no GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

# The repo's pinned one-sided Student-t critical at df=3 (4 passes).
T95_ONE_SIDED_DF3 = 2.3534


class WallError(RuntimeError):
    """The batch-conditioned wall cannot be computed from this evidence."""


def _samples_for(sidecar_dir: Path, arm: str) -> Path:
    hits = sorted(glob.glob(str(sidecar_dir / f"{arm}.json.samples.*")))
    if len(hits) != 1:
        raise WallError(
            f"expected exactly one per-step samples sidecar for {arm}, "
            f"found {len(hits)}"
        )
    return Path(hits[0])


def _arm_dir_name(mode: str, pass_index: int) -> str:
    prefix = "tail6_fixed32" if mode == "tail6_fixed32" else "hydra27_fixed32"
    return f"{prefix}_pool{pass_index}"


def analyse_arm(
    arm_record: dict[str, Any],
    sidecar_dir: Path,
    arm_name: str | None = None,
) -> dict[str, Any]:
    """Condition an arm's windowed wall on the served batch width.

    `arm_name` overrides the pool16 campaign's `<mode>_pool<N>` naming so other
    run classes -- the GQA-pair width-4 timing pair, whose arms carry the served
    dispatch in their names -- can reuse this analysis unchanged. Defaulting it
    keeps every existing caller byte-identical in behaviour.
    """
    mode = arm_record["mode"]
    arm = arm_name or _arm_dir_name(mode, arm_record["pass_index"])
    doc = json.loads(_samples_for(sidecar_dir, arm).read_text())
    if doc.get("schema") != "fr13.sfwd_per_step_samples.v2":
        raise WallError(f"{arm}: unexpected samples schema {doc.get('schema')!r}")
    if doc.get("samples_capped"):
        # A capped sample array is a SUBSET of the steps and its mean is not the
        # population mean. Refuse rather than quietly average a truncation.
        raise WallError(f"{arm}: per-step samples were capped; refusing")

    gate = arm_record["bracket_reduction"]["work_census_gate"]
    lo = gate["census_first_forward_step"]
    hi = gate["census_last_forward_step"]

    idx = doc["wall_fwd_indices"]
    ms = doc["wall_ms"]
    width = doc["wall_drafts"]
    if not (len(idx) == len(ms) == len(width)):
        raise WallError(f"{arm}: per-step sample arrays are ragged")

    by_width: dict[int, list[float]] = {}
    total: list[float] = []
    for i, m, w in zip(idx, ms, width):
        if lo <= i <= hi:
            by_width.setdefault(int(w), []).append(float(m))
            total.append(float(m))
    if not total:
        raise WallError(f"{arm}: no wall samples inside the sealed window")

    # HARD CHECK: this sample selection must BE the sealed counter bracket, not
    # merely resemble it. If the sums disagree the window was mis-selected and
    # every conditioned mean below would be drawn from the wrong steps.
    raw = arm_record["raw_counter_delta_window"]
    want_steps = raw["vllm:fr13_decode_step_wall_steps_total"]
    want_seconds = raw["vllm:fr13_decode_step_wall_seconds_total"]
    got_seconds = sum(total) / 1000.0
    if len(total) != int(want_steps):
        raise WallError(
            f"{arm}: selected {len(total)} wall samples but the sealed bracket "
            f"counted {int(want_steps)}"
        )
    if not math.isclose(got_seconds, want_seconds, rel_tol=1e-6):
        raise WallError(
            f"{arm}: selected wall sums to {got_seconds:.6f}s but the sealed "
            f"bracket counted {want_seconds:.6f}s"
        )

    slots = arm_record["slots"]
    full = by_width.get(slots, [])
    out: dict[str, Any] = {
        "arm": arm,
        "mode": mode,
        "pass_index": arm_record["pass_index"],
        "window_step_range": [lo, hi],
        "wall_samples_in_window": len(total),
        "counter_bracket_agrees": True,
        "sealed_step_wall_ms": arm_record["windowed"]["step_wall_ms"],
        "direct_blend_step_wall_ms": st.mean(total),
        "by_width": {
            str(w): {
                "steps": len(v),
                "fraction": len(v) / len(total),
                "mean_ms": st.mean(v),
                "sd_ms": st.stdev(v) if len(v) > 1 else 0.0,
            }
            for w, v in sorted(by_width.items())
        },
    }
    if full:
        out["width_full_step_wall_ms"] = st.mean(full)
        out["width_full_steps"] = len(full)
        out["width_full_fraction"] = len(full) / len(total)
        out["basis_effect_ms"] = out["direct_blend_step_wall_ms"] - out["sealed_step_wall_ms"]
        out["width_effect_ms"] = out["width_full_step_wall_ms"] - out["direct_blend_step_wall_ms"]
        out["total_correction_ms"] = out["width_full_step_wall_ms"] - out["sealed_step_wall_ms"]
    return out


def _pooled(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = st.mean(values)
    sd = st.stdev(values) if n > 1 else 0.0
    cv = sd / mean if mean else 0.0
    # MDE = CV * t / sqrt(n): the smallest effect separable at this pass count.
    mde_frac = cv * T95_ONE_SIDED_DF3 / math.sqrt(n) if n else 0.0
    return {
        "n": n,
        "mean_ms": mean,
        "sd_ms": sd,
        "cv": cv,
        "mde_fraction": mde_frac,
        "mde_ms": mde_frac * mean,
        "critical": "T95_ONE_SIDED[3]" if n == 4 else "no_pinned_critical",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--operating-point",
        default="results/fr13_b4_refill_citable_20260812/"
        "fr13_b4_width4_operating_point.json",
    )
    p.add_argument("--sidecar-dir", default="output/fr13_sfwd_sidecar")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    art = json.loads(Path(args.operating_point).read_text())
    sidecar_dir = Path(args.sidecar_dir)

    arms = [analyse_arm(a, sidecar_dir) for a in art["arms"]]
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for a in arms:
        by_mode.setdefault(a["mode"], []).append(a)

    pooled: dict[str, Any] = {}
    for mode, rows in by_mode.items():
        pooled[mode] = {
            "sealed_rescaled_blend": _pooled([r["sealed_step_wall_ms"] for r in rows]),
            "direct_blend": _pooled([r["direct_blend_step_wall_ms"] for r in rows]),
            "batch_conditioned_full_width": _pooled(
                [r["width_full_step_wall_ms"] for r in rows]
            ),
            "mean_basis_effect_ms": st.mean([r["basis_effect_ms"] for r in rows]),
            "mean_width_effect_ms": st.mean([r["width_effect_ms"] for r in rows]),
            "mean_total_correction_ms": st.mean(
                [r["total_correction_ms"] for r in rows]
            ),
            "mean_full_width_fraction": st.mean(
                [r["width_full_fraction"] for r in rows]
            ),
        }

    doc = {
        "schema": "fr13.b4_batch_conditioned_wall.v1",
        "citable": False,
        "citable_reason": (
            "Re-reduction of a non-citable instrument class over the same "
            "banked arms; inherits b4_width4_operating_point's citable=false."
        ),
        "analysis_only": True,
        "gpu_touched": False,
        "source_operating_point": args.operating_point,
        "claims": [
            "The step wall CAN be conditioned on engine batch width from "
            "already-banked evidence: the SFWD per-step samples sidecar carries "
            "per-step wall and per-step width keyed by absolute forward-step "
            "index. width4_window.md §3 called this not derivable because the "
            "work census carries no per-step wall; the sidecar does.",
            "The sealed windowed step_wall_ms understates the true width-4 "
            "operating point, because the pool-depth window is a width blend "
            "and width-3 steps are ~50 ms cheaper.",
        ],
        "does_not_claim": [
            "No new operating point is sealed. This class is not citable and "
            "does not revise any sealed verdict.",
            "No throughput claim. Conditioning on width selects steps, so the "
            "result is a per-step cost at width 4, never a delivered rate.",
            "No causal claim that width CAUSES the extra wall; co-residency and "
            "context length move together across these arms.",
        ],
        "arms": arms,
        "pooled": pooled,
    }
    text = json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except WallError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(2)
