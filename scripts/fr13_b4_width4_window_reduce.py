#!/usr/bin/env python3
"""The DEPTH-WINDOW operating point: measure the pool while it IS a pool.

WHY THIS TOOL EXISTS
--------------------
The sealed pool16 campaign (output/fr13_b4_pool16_refill_gate_20260812T115625Z,
NOT CITABLE) established that a 16-task pool at 4 slots is not one operating
point but a WALL-BLENDED MIXTURE of two regimes:

  * a full-width phase, pool depth == slots, backfill available; and
  * a drain phase that begins when the last task is admitted and nothing remains
    behind the running four -- structurally the exact4 wave, 4->3->2->1.

Arm-level `events_per_step` is therefore a MIXTURE WEIGHT, not a rate.  Across
the campaign's four passes it ran 3.04 -> 1.67 while the median task duration
stayed flat (576-861 s); what moved was the tail (longest task 2841 s -> 9493 s).
One straggler sets a whole arm's measured co-residency.  That is why the 3.2
depth floor excluded passes 2 and 3 and the campaign sealed NOT_EVALUATED.

The recoverable truth is that the full-width phase of EVERY served arm is still
on disk, exactly bracketed, including the arms the depth floor excluded -- their
full-width phases are just as real, only shorter relative to their drains.  This
tool reduces that phase and nothing else.

WHAT MAKES THE WINDOW EXACT RATHER THAN INTERPOLATED
----------------------------------------------------
Three artifacts line up on the same events, so no wall->step interpolation is
ever performed:

  1. `swe_out/verified/fr13_task_refill_ledger.jsonl` timestamps every admit and
     complete with the resulting pool `depth` and the count still `pending`.
  2. `swe_out/verified/per_task/<id>/vllm_metrics_{pre,post}.txt` are REAL
     Prometheus snapshots taken at those same task boundaries.  Both window
     edges are ledger events that own a snapshot, so the windowed counter delta
     is a genuine bracket -- the same envelope estimator fr13_measure already
     uses for a staggered arm, closed earlier.
  3. `logs/fr13_fixed32_work_census.jsonl` is one record per forward step with
     an absolute `forward_step_index`, and the bracket's own
     `fr13_decode_forward_gpu_steps_total` says which steps those are.

So the census cross-check is not a plausibility check, it is an identity, and it
is enforced with EXACT equality and no tolerance, exactly as
fr13_measure.cmd_deploy_speed enforces it for a whole arm.

THE WINDOW
----------
Over the ledger events ordered as fr13_floor_gate._reduce_refill_ledger orders
them:

    open  = the first event at which depth == slots
    close = the first `complete` event carrying pending == 0

`close` is the moment the pool stops being a pool: depth falls below slots and
there is nothing left unstarted to refill it, so it can never return.  The tool
PROVES that rather than assuming it -- `_window_occupancy` asserts that not one
interval outside [open, close] carries depth >= slots, an exact structural
assertion with no tolerance.  The consequence is that the window contains 100%
of the arm's full-width wall, and the drain is excluded BY CONSTRUCTION.

WHY THAT EXCLUSION IS HONEST AND NOT CHERRY-PICKING
---------------------------------------------------
The boundary is defined by ADMISSION STRUCTURE ONLY -- depth and pending, both
integers written by the runner before any timing was reduced.  No measured rate
enters the definition, so the window cannot be slid to a favourable number.  It
is the same information the ledger's own `full_width_fraction` is built from,
and the tool recomputes that fraction from the raw events and requires it to
agree with the window it derived.

What the class buys with that exclusion is stated, not hidden: it is NOT
whole-arm throughput and must never be quoted as the arm's delivered rate.  See
`does_not_claim`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fr13_b4_timing_math import TimingMathError, phase_breakdown  # noqa: E402


def _load(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / relative)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


measure = _load("fr13_measure_for_width4", "fr13_measure.py")
reducer = _load("fr13_b4_floor_gate_reduce_for_width4", "fr13_b4_floor_gate_reduce.py")

COUNTERS: tuple[str, ...] = tuple(measure.COUNTERS)
T95_ONE_SIDED: dict[int, float] = dict(reducer.T95_ONE_SIDED)
TOPOLOGIES: tuple[str, ...] = tuple(reducer.TOPOLOGIES)

SCHEMA = "fr13.b4_width4_operating_point.v1"
CLASSIFICATION = "real_swe_verified_b4_width4_operating_point"
RUN_CLASS = "b4_width4_operating_point"
TITLE = "B4 WIDTH-4 OPERATING POINT (depth window)"

LEDGER_EVENTS_RELPATH = "swe_out/verified/fr13_task_refill_ledger.jsonl"
LEDGER_SUMMARY_RELPATH = "swe_out/verified/fr13_task_refill_summary.json"
WORK_CENSUS_RELPATH = "logs/fr13_fixed32_work_census.jsonl"
DEPLOY_SPEED_RELPATH = "deploy_speed_fullwall.json"
ARM_COMPLETE_MARKER = "arm_ended_at.txt"
TASK_REFILL_SUMMARY_SCHEMA = reducer.TASK_REFILL_SUMMARY_SCHEMA

# ADMISSIBILITY, pinned as a constant rather than fitted to the data it will
# judge.  A window must span at least this many pure-decode forward steps to be
# treated as a draw.  Rationale, from the measurement's own requirements and not
# from the observed spread: at the ~385 ms/step this stack measures, 1000 steps
# is ~6.4 minutes of sustained full-width serving, and at the ~0.88 wall-chain
# retention these arms record it leaves ~880 independent wall samples behind the
# step mean.  Below that the step wall is a small-sample statistic and the class
# refuses rather than quoting it.
#
# HONESTY NOTE, restated in the emitted artifact: on the 2026-08-12 campaign this
# floor did no work.  The shortest of the eight windows spans 4152 steps, 4.15x
# the floor.  The floor is a guard for future short-window arms, not a filter
# that shaped this result.
MIN_WINDOW_STEPS = 1000

# The depth floor that excluded passes 2 and 3 at arm level is DELIBERATELY NOT
# applied here.  `time_weighted_mean_depth >= 3.2` asks "was this arm mostly a
# pool?", which is a question about the MIXTURE WEIGHT -- precisely the quantity
# this class removes by construction.  Applying it would discard the full-width
# phases of arms that have perfectly good ones, on the grounds that their drains
# were long.  The ledger's structural invariants ARE still gated below
# (schema, slots, task_count, completed, aborted, peak_depth), because those ask
# whether the arm was a valid pool run at all.
APPLIES_ARM_LEVEL_DEPTH_FLOOR = False

CLAIMS: tuple[str, ...] = (
    "the per-request step TPS, aggregate step TPS, events/step and step wall "
    "delivered by the shipped fixed32 B4 stack DURING the phase in which a "
    "16-task pool holds its full slot width with backfill still available, "
    "bracketed by real Prometheus snapshots at ledger admission events and "
    "cross-checked step-for-step against the work census",
)

DOES_NOT_CLAIM: tuple[str, ...] = (
    "whole-arm throughput. This is the operating point of ONE PHASE of an arm. "
    "The drain phase -- which begins when the last task is admitted and nothing "
    "remains behind the running four -- is excluded BY CONSTRUCTION, and on this "
    "campaign it was 18% to 65% of arm wall. Multiplying a windowed rate by an "
    "arm wall would overstate delivered work, in one case by nearly 3x. The "
    "arm-level blended numbers remain the only answer to 'what did the arm "
    "deliver', and they are carried alongside every windowed value here",
    "exact4 comparability. Different task set (16 vs its first 4), different "
    "bracket estimator (envelope vs widest-nested), and -- newly and most "
    "importantly -- a different WINDOW: the exact4 arms are reduced whole, drain "
    "included, because at pool == slots the drain IS the arm. Comparing a "
    "windowed pool16 number against a whole-arm exact4 number compares a phase "
    "to a mixture. Any such contrast is descriptive and must say so",
    "that the window is a per-step engine-batch filter. It is a POOL-DEPTH "
    "window: it selects the wall during which four tasks were admitted and "
    "unfinished. Inside it the ENGINE batch was still width 4 on only 63-70% of "
    "steps, width 3 on 24-33%, and narrower on the rest, because an admitted "
    "task is repeatedly between requests while the agent runs tools locally. "
    "That residual is published per arm as census_event_counts_by_batch. A true "
    "batch==4 filter is NOT derivable from this evidence: the census resolves "
    "batch width per step but carries no per-step wall, so no wall statistic can "
    "be conditioned on it",
    "a cap verdict. b4_cap_applicable is false at B4 context sizes and nothing "
    "here changes that: 35-38k contexts imply 42-49 GB mandatory bytes and a "
    "155-179 ms floor, above the 137.607 ms one-sided cap",
    "an agent-quality verdict. Resolve rate is neither read nor recorded by this "
    "tool. exact16 as QUALITY CONTROL at batched milestones (Mark, 2026-08-10) is "
    "a separate gate over the same subset and is untouched",
    "that step_wall_ms and events_per_step share one basis. Inherited unchanged "
    "from the arm-level class: the wall mean is over wall-chain-bracketed steps, "
    "the event rate over all census steps. retained_wall_fraction is published "
    "per window and is LOWER inside the window (0.877-0.896) than over the whole "
    "arm (0.906-0.932), because continuous admission resets the wall chain more "
    "often and admission is densest exactly here. The first step after a reset "
    "carries no wall sample by construction, so whether those steps are slower in "
    "wall terms is NOT measurable from this evidence",
    "that prefill_frac and per_request_decode_tps are windowed on the same footing "
    "as the step-keyed statistics. Those two derive from request_*_seconds_sum "
    "counters that increment only when a REQUEST COMPLETES, so requests in flight "
    "at the window close contribute their wall to the denominator of nothing and "
    "their tokens to neither side. The step-keyed statistics -- step_wall_ms, "
    "events_per_step, aggregate and per-request step TPS -- have no such boundary "
    "bias. Both are emitted; only the step-keyed ones carry intervals",
    "a promotion, a floor-acceptance verdict, or a citable seal. This class exists "
    "to give the step-wall levers a stable target to be judged against. It is an "
    "INSTRUMENT, and formal_floor_acceptance_eligible is false by construction",
)


class Width4Error(RuntimeError):
    """The evidence does not support a width-4 window reduction."""


# --------------------------------------------------------------------------- #
# ledger                                                                        #
# --------------------------------------------------------------------------- #
def read_ledger_events(arm_dir: Path) -> list[dict[str, Any]]:
    """Read the raw admission events, refusing anything structurally unusable."""
    path = arm_dir / LEDGER_EVENTS_RELPATH
    if not path.is_file() or path.is_symlink():
        raise Width4Error(
            f"no admission ledger at {path}; the window is DEFINED by admission "
            "events, so an arm without one cannot be windowed at all"
        )
    events: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise Width4Error(f"{path}:{lineno} is not JSON: {error}") from error
        for field in ("event", "instance_id", "depth", "pending", "t_s"):
            if field not in record:
                raise Width4Error(f"{path}:{lineno} has no {field!r}")
        if record["event"] not in ("admit", "complete"):
            raise Width4Error(
                f"{path}:{lineno} carries unknown event {record['event']!r}"
            )
        for field in ("depth", "pending"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Width4Error(
                    f"{path}:{lineno} {field}={value!r} is not a non-negative int"
                )
        t_s = record["t_s"]
        if isinstance(t_s, bool) or not isinstance(t_s, (int, float)):
            raise Width4Error(f"{path}:{lineno} t_s={t_s!r} is not numeric")
        if not math.isfinite(float(t_s)):
            raise Width4Error(f"{path}:{lineno} t_s is not finite")
        events.append(record)
    if not events:
        raise Width4Error(f"{path} is empty")
    return events


def order_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order exactly as fr13_floor_gate._reduce_refill_ledger does.

    Admits sort before completes at an identical timestamp, so a same-instant
    handoff is read as the refill it is rather than as a dip below width.
    """
    return sorted(events, key=lambda e: (e["t_s"], e["event"] != "admit"))


def reduce_ledger_occupancy(
    events: list[dict[str, Any]], concurrency: int
) -> dict[str, float]:
    """Port of fr13_floor_gate._reduce_refill_ledger, arithmetic unchanged.

    Reproduced rather than imported: fr13_floor_gate is 12901 lines and pulls in
    the serving stack.  The house pattern is a second independent read of the
    same bytes, and `verify_ledger_summary` requires this to agree with what the
    runner wrote.
    """
    ordered = order_events(events)
    span = 0.0
    weighted = 0.0
    full_width = 0.0
    previous_t = ordered[0]["t_s"] if ordered else 0.0
    depth = 0
    for event in ordered:
        width = event["t_s"] - previous_t
        if width > 0:
            span += width
            weighted += depth * width
            if depth >= concurrency:
                full_width += width
        depth = event["depth"]
        previous_t = event["t_s"]
    return {
        "arm_wall_s": span,
        "time_weighted_mean_depth": (weighted / span) if span > 0 else 0.0,
        "full_width_fraction": (full_width / span) if span > 0 else 0.0,
    }


def verify_ledger_summary(
    arm_dir: Path, events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Gate the ledger's structural invariants and recompute its occupancy.

    The arm-level DEPTH and FULL-WIDTH floors are deliberately absent -- see
    APPLIES_ARM_LEVEL_DEPTH_FLOOR.  Everything here asks "was this a valid pool
    run", never "was it a fast one".
    """
    path = arm_dir / LEDGER_SUMMARY_RELPATH
    summary = reducer.exact_json(path, label=f"{arm_dir.name} pool ledger summary")
    if summary.get("schema") != TASK_REFILL_SUMMARY_SCHEMA:
        raise Width4Error(
            f"pool ledger schema is {summary.get('schema')!r}, "
            f"expected {TASK_REFILL_SUMMARY_SCHEMA!r}"
        )
    slots = summary.get("slots")
    task_count = summary.get("task_count")
    for label, value in (("slots", slots), ("task_count", task_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Width4Error(f"pool ledger {label}={value!r} is not a positive int")
    if task_count <= slots:
        raise Width4Error(
            f"a pool must be LARGER than its slot count ({task_count} tasks, "
            f"{slots} slots); at parity there is no refill phase to window"
        )
    if summary.get("aborted") is not False:
        raise Width4Error("pool run aborted; its evidence is partial")
    if summary.get("completed") != task_count:
        raise Width4Error(
            f"pool completed {summary.get('completed')!r} of {task_count} tasks"
        )
    peak = summary.get("peak_depth")
    if not isinstance(peak, int) or isinstance(peak, bool) or peak > slots:
        raise Width4Error(
            f"peak in-flight depth {peak!r} exceeds the slot count {slots}"
        )
    admits = sum(1 for e in events if e["event"] == "admit")
    completes = sum(1 for e in events if e["event"] == "complete")
    if admits != task_count or completes != task_count:
        raise Width4Error(
            f"ledger holds {admits} admits and {completes} completes for a "
            f"{task_count}-task pool"
        )
    recomputed = reduce_ledger_occupancy(events, slots)
    for key in ("arm_wall_s", "time_weighted_mean_depth", "full_width_fraction"):
        recorded = summary.get(key)
        if not isinstance(recorded, (int, float)) or isinstance(recorded, bool):
            raise Width4Error(f"pool ledger {key}={recorded!r} is not numeric")
        if not math.isclose(float(recorded), recomputed[key], rel_tol=1e-9, abs_tol=1e-9):
            raise Width4Error(
                f"pool ledger {key} recomputes to {recomputed[key]!r}, not the "
                f"recorded {recorded!r}; the summary is not supported by its events"
            )
    return {
        "slots": slots,
        "task_count": task_count,
        "peak_depth": peak,
        "recomputed": recomputed,
        "recorded_full_width_fraction": float(summary["full_width_fraction"]),
        "recorded_time_weighted_mean_depth": float(
            summary["time_weighted_mean_depth"]
        ),
    }


# --------------------------------------------------------------------------- #
# the window                                                                    #
# --------------------------------------------------------------------------- #
def derive_width4_window(
    events: list[dict[str, Any]], slots: int
) -> dict[str, Any]:
    """Locate the full-width phase from admission structure alone.

    open  -- the first event at which depth == slots.
    close -- the first `complete` carrying pending == 0, i.e. the first time the
             width falls with nothing unstarted left to restore it.

    Only `depth` and `pending` are read.  No measured rate participates, so the
    boundary cannot be slid toward a nicer number.
    """
    ordered = order_events(events)
    open_index = next(
        (i for i, e in enumerate(ordered) if e["depth"] == slots), None
    )
    if open_index is None:
        raise Width4Error(
            f"the pool never reached its full width of {slots}; this arm has no "
            "width-4 phase to measure and is NOT a short window -- it is no window"
        )
    close_index = next(
        (
            i
            for i, e in enumerate(ordered)
            if i > open_index and e["event"] == "complete" and e["pending"] == 0
        ),
        None,
    )
    if close_index is None:
        raise Width4Error(
            "no completion with pending == 0 follows the full-width entry; the "
            "ledger does not record the end of the refill phase"
        )
    occupancy = _window_occupancy(ordered, open_index, close_index, slots)
    last_admit = max(
        (i for i, e in enumerate(ordered) if e["event"] == "admit"), default=-1
    )
    if not open_index <= last_admit < close_index:
        raise Width4Error(
            f"the last admit sits at index {last_admit}, outside the window "
            f"[{open_index}, {close_index}); the window must contain the whole "
            "admission sequence or it is not the refill phase"
        )
    return {
        "open_index": open_index,
        "close_index": close_index,
        "open_event": dict(ordered[open_index]),
        "close_event": dict(ordered[close_index]),
        "open_t_s": float(ordered[open_index]["t_s"]),
        "close_t_s": float(ordered[close_index]["t_s"]),
        "window_wall_s": float(ordered[close_index]["t_s"])
        - float(ordered[open_index]["t_s"]),
        "opening_task": ordered[open_index]["instance_id"],
        "closing_task": ordered[close_index]["instance_id"],
        "last_admit_index": last_admit,
        "definition": (
            "open = first event at depth == slots; close = first complete at "
            "pending == 0. Structural only: depth and pending, no measured rate."
        ),
        **occupancy,
    }


def _window_occupancy(
    ordered: list[dict[str, Any]], open_index: int, close_index: int, slots: int
) -> dict[str, Any]:
    """Prove the drain exclusion instead of asserting it.

    Accumulates full-width wall INSIDE and OUTSIDE the window separately and
    requires the outside sum to be exactly zero.  If any interval outside
    [open, close] carried depth >= slots, the window would not contain the whole
    full-width phase and every claim built on it would be wrong.  Exact
    comparison against 0.0 -- no tolerance, because the quantity is a sum of
    non-negative terms and is either empty or it is not.
    """
    inside = 0.0
    outside = 0.0
    span = 0.0
    depth = 0
    previous_t = ordered[0]["t_s"]
    for index, event in enumerate(ordered):
        width = event["t_s"] - previous_t
        if width > 0:
            span += width
            if depth >= slots:
                if open_index < index <= close_index:
                    inside += width
                else:
                    outside += width
        depth = event["depth"]
        previous_t = event["t_s"]
    if outside != 0.0:
        raise Width4Error(
            f"{outside!r} s of full-width wall lies OUTSIDE the derived window; "
            "the pool returned to full width after the refill phase ended, which "
            "contradicts the window definition"
        )
    window_wall = ordered[close_index]["t_s"] - ordered[open_index]["t_s"]
    return {
        "full_width_wall_s_inside_window": inside,
        "full_width_wall_s_outside_window": outside,
        "arm_wall_s_from_events": span,
        "window_wall_fraction_of_arm": (window_wall / span) if span > 0 else 0.0,
        "depth_at_slots_fraction_within_window": (
            inside / window_wall if window_wall > 0 else 0.0
        ),
        "drain_wall_s_excluded_by_construction": span - window_wall,
        "drain_wall_fraction_excluded_by_construction": (
            (span - window_wall) / span if span > 0 else 0.0
        ),
    }


# --------------------------------------------------------------------------- #
# the windowed counter bracket                                                  #
# --------------------------------------------------------------------------- #
def window_counter_bracket(
    arm_dir: Path, ordered: list[dict[str, Any]], window: dict[str, Any]
) -> dict[str, Any]:
    """Envelope max(post) - min(pre), closed at the window edge.

    Identical in form to fr13_measure._bracket_reduce's staggered branch.  The
    only difference is membership: `pre` snapshots come from the tasks admitted
    at or before the window opens, `post` snapshots from the tasks completed at
    or before it closes.  Both edges are ledger events that own a real snapshot,
    so nothing is interpolated.

    Counters are ENGINE-GLOBAL and monotone, so work done inside the window by
    tasks still running at the close is correctly included -- the bracket is a
    time bracket, not a per-task sum.
    """
    open_index = window["open_index"]
    close_index = window["close_index"]
    opened = [
        e["instance_id"]
        for e in ordered[: open_index + 1]
        if e["event"] == "admit"
    ]
    closed = [
        e["instance_id"]
        for e in ordered[: close_index + 1]
        if e["event"] == "complete"
    ]
    if not opened:
        raise Width4Error("no task is admitted at or before the window opens")
    if not closed:
        raise Width4Error("no task completes at or before the window closes")

    task_root = arm_dir / "swe_out" / "verified" / "per_task"
    if not task_root.is_dir():
        raise Width4Error(f"no per-task bracket directory at {task_root}")

    def snapshot(instance_id: str, which: str) -> dict[str, float]:
        path = task_root / instance_id / f"vllm_metrics_{which}.txt"
        if not path.is_file() or path.is_symlink():
            raise Width4Error(f"missing {which} bracket snapshot at {path}")
        return measure._scrape_metrics_file(str(path))

    pre = {i: snapshot(i, "pre") for i in opened}
    post = {i: snapshot(i, "post") for i in closed}

    delta: dict[str, float] = {}
    origin: dict[str, float] = {}
    for counter in COUNTERS:
        low = min(pre[i][counter] for i in pre)
        high = max(post[i][counter] for i in post)
        if high < low:
            raise Width4Error(
                f"{counter} closes at {high!r} below its open {low!r}; the "
                "bracket is not monotone and the snapshots are not orderable"
            )
        origin[counter] = low
        delta[counter] = high - low
    return {
        "delta": delta,
        "origin": origin,
        "pre_tasks": sorted(opened),
        "post_tasks": sorted(closed),
        "distinct_bracket_origins": len(
            {tuple(pre[i][c] for c in COUNTERS) for i in pre}
        ),
        "topology": "staggered",
        "basis": (
            "the ENVELOPE max(post)-min(pre) over the per-task /metrics snapshots "
            "bounding the depth window: pre from the tasks admitted at or before "
            "depth reached slots, post from the tasks completed at or before the "
            "refill phase ended. Engine-global monotone counters, so in-flight "
            "tasks' work inside the window is included"
        ),
    }


def window_census_counts(
    path: Path, first_step: int, step_count: int
) -> dict[str, Any]:
    """Census counts restricted to the window's forward-step range.

    Guards are those of fr13_measure._work_census_counts, applied to the records
    inside the range.  A record missing from the range is fatal: the range comes
    from a counter delta, so a hole means the census and the counters disagree
    about what the engine did.
    """
    if not path.is_file() or path.is_symlink():
        raise Width4Error(f"no work census at {path}")
    if step_count <= 0:
        raise Width4Error(f"window spans {step_count} forward steps")
    last_step = first_step + step_count - 1
    by_batch: dict[int, int] = {}
    seen: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise Width4Error(f"{path}:{lineno} is not JSON: {error}") from error
            if not record.get("event_complete"):
                continue
            index = record.get("forward_step_index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise Width4Error(
                    f"{path}:{lineno} forward_step_index={index!r} is not an int"
                )
            if index < first_step or index > last_step:
                continue
            if index in seen:
                raise Width4Error(
                    f"{path}:{lineno} repeats forward_step_index {index}"
                )
            seen.add(index)
            batch = record.get("batch_size")
            if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
                raise Width4Error(
                    f"{path}:{lineno} batch_size={batch!r} is not a positive int"
                )
            purity = record.get("batch_purity") or {}
            if purity.get("mixed_pseudo_rows"):
                raise Width4Error(
                    f"{path}:{lineno} records mixed pseudo rows; not a pure "
                    "decode step"
                )
            by_batch[batch] = by_batch.get(batch, 0) + 1
    steps = len(seen)
    if steps != step_count:
        missing = step_count - steps
        raise Width4Error(
            f"the window spans forward steps [{first_step}, {last_step}] "
            f"({step_count} steps) but the census holds {steps} of them "
            f"({missing} missing); counters and census disagree"
        )
    events = sum(size * count for size, count in by_batch.items())
    return {
        "path": str(path),
        "first_forward_step": first_step,
        "last_forward_step": last_step,
        "steps": steps,
        "events": events,
        "events_per_step": events / steps,
        "event_counts_by_batch": {str(k): by_batch[k] for k in sorted(by_batch)},
    }


def census_gate(delta: dict[str, float], census: dict[str, Any]) -> dict[str, Any]:
    """Exact equality, no tolerance -- as fr13_measure.cmd_deploy_speed does it."""
    gate_steps = delta.get(measure.M_DECODE_FWD_GPU_STEPS, 0.0)
    gate_events = delta.get(measure.M_DECODE_FWD_GPU_DRAFTS, 0.0)
    if gate_steps != float(census["steps"]) or gate_events != float(census["events"]):
        raise Width4Error(
            "the windowed counter bracket is not supported by the engine-side "
            f"record: counters say {gate_steps!r} steps / {gate_events!r} events, "
            f"the census says {census['steps']!r} / {census['events']!r}"
        )
    return {
        "status": "pass",
        "basis": "exact equality, no tolerance",
        "census_path": census["path"],
        "census_first_forward_step": census["first_forward_step"],
        "census_last_forward_step": census["last_forward_step"],
        "census_steps": census["steps"],
        "census_events": census["events"],
        "census_events_per_step": census["events_per_step"],
        "census_event_counts_by_batch": census["event_counts_by_batch"],
    }


# --------------------------------------------------------------------------- #
# the windowed speed record                                                     #
# --------------------------------------------------------------------------- #
def build_speed_record(delta: dict[str, float], floor_ms: float) -> dict[str, Any]:
    """Every formula is fr13_measure.cmd_deploy_speed's, applied to the window.

    Nothing is re-derived: the identities are then enforced by
    fr13_b4_timing_math.phase_breakdown, the same function the arm-level reducer
    runs, so a windowed record that does not reconcile raises rather than ships.
    """

    def need(counter: str) -> float:
        value = delta.get(counter, 0.0)
        if not math.isfinite(value) or value <= 0:
            raise Width4Error(
                f"windowed delta for {counter} is {value!r}; the window records "
                "no work on a counter every derived rate depends on"
            )
        return value

    fwd_steps = need(measure.M_DECODE_FWD_GPU_STEPS)
    fwd_drafts = need(measure.M_DECODE_FWD_GPU_DRAFTS)
    fwd_seconds = need(measure.M_DECODE_FWD_GPU_S)
    wall_seconds = need(measure.M_STEP_WALL_S)
    wall_drafts = need(measure.M_STEP_WALL_DRAFTS)
    wall_steps = need(measure.M_STEP_WALL_STEPS)
    drafts = need(measure.M_DRAFTS)
    accepted = need(measure.M_ACCEPTED)

    events_per_step = fwd_drafts / fwd_steps
    wall_s_per_event = wall_seconds / wall_drafts
    step_wall_ms = wall_s_per_event * 1000.0 * events_per_step
    accept_per_event = accepted / drafts
    committed_per_event = accept_per_event + 1.0
    record: dict[str, Any] = {
        "events_per_step": events_per_step,
        "wall_s_per_event": wall_s_per_event,
        "step_wall_ms": step_wall_ms,
        "accept_per_event": accept_per_event,
        "committed_per_event": committed_per_event,
        "measured_tps_fullstep_wall": committed_per_event / wall_s_per_event,
        "s_per_fwd_gpu": fwd_seconds / fwd_drafts,
        "s_per_fwd_gpu_per_forward": fwd_seconds / fwd_steps,
        "drafter_gpu_ms_per_step": delta[measure.M_DRAFTER_GPU_S] / fwd_steps * 1000.0,
        "committer_gpu_ms_per_step": delta[measure.M_COMMITTER_GPU_S]
        / fwd_steps
        * 1000.0,
        "floor_ms": floor_ms,
        "floor_ratio": step_wall_ms / floor_ms,
        "retained_wall_fraction": wall_steps / fwd_steps,
        "generation_tokens": delta.get(measure.M_GEN_TOK, 0.0),
        "wall_steps_measured": wall_steps,
        "forward_steps_measured": fwd_steps,
    }
    decode_s = delta.get(measure.M_DECODE_S, 0.0)
    prefill_s = delta.get(measure.M_PREFILL_S, 0.0)
    record["prefill_frac"] = (prefill_s / decode_s) if decode_s > 0 else None
    tpot_count = delta.get(measure.M_TPOT_COUNT, 0.0)
    tpot_sum = delta.get(measure.M_TPOT_SUM, 0.0)
    record["per_request_decode_tps"] = (
        (tpot_count / tpot_sum) if tpot_sum > 0 else None
    )
    record["completion_keyed_note"] = (
        "prefill_frac and per_request_decode_tps come from request_*_seconds_sum "
        "counters that advance only when a request COMPLETES, so requests in "
        "flight at the window edges bias them. The step-keyed statistics above do "
        "not share that bias."
    )
    return record


# --------------------------------------------------------------------------- #
# per-arm reduction                                                             #
# --------------------------------------------------------------------------- #
def reduce_window_arm(
    arm_dir: Path, *, mode: str, pass_index: int
) -> dict[str, Any]:
    """Reduce ONE arm's width-4 window, or record exactly why it has none.

    Never raises for an evidence-level defect: an arm without a usable window is
    excluded with a named reason so the count of surviving draws is auditable.
    """
    record: dict[str, Any] = {
        "arm": arm_dir.name,
        "mode": mode,
        "pass_index": pass_index,
        "included": False,
        "exclusion_reason": None,
    }
    try:
        if not (arm_dir / ARM_COMPLETE_MARKER).is_file():
            raise Width4Error(
                "arm is still serving (no arm_ended_at.txt); its ledger and census "
                "are still being appended to and any window would be a torn read"
            )
        speed_path = arm_dir / DEPLOY_SPEED_RELPATH
        arm_speed = reducer.exact_json(speed_path, label=f"{arm_dir.name} deploy speed")
        floor_ms = arm_speed.get("floor_ms")
        if not isinstance(floor_ms, (int, float)) or isinstance(floor_ms, bool):
            raise Width4Error(f"{speed_path} carries no numeric floor_ms")

        events = read_ledger_events(arm_dir)
        ledger = verify_ledger_summary(arm_dir, events)
        slots = ledger["slots"]
        ordered = order_events(events)
        window = derive_width4_window(ordered, slots)
        bracket = window_counter_bracket(arm_dir, ordered, window)
        delta = bracket["delta"]

        # The bracket's own forward-step counter says WHICH census steps the
        # window is: its origin is the first index, its delta the count. Both
        # must be whole -- a fractional step count would mean the counter and
        # the census are not measuring the same thing.
        origin_steps = float(bracket["origin"][measure.M_DECODE_FWD_GPU_STEPS])
        delta_steps = float(delta[measure.M_DECODE_FWD_GPU_STEPS])
        if not origin_steps.is_integer() or not delta_steps.is_integer():
            raise Width4Error(
                "forward-step counters are not integral over the window "
                f"(origin {origin_steps!r}, delta {delta_steps!r})"
            )
        first_step = int(origin_steps)
        step_count = int(delta_steps)

        census = window_census_counts(
            arm_dir / WORK_CENSUS_RELPATH, first_step, step_count
        )
        gate = census_gate(delta, census)

        if step_count < MIN_WINDOW_STEPS:
            raise Width4Error(
                f"window spans {step_count} forward steps, below the pinned "
                f"admissibility floor of {MIN_WINDOW_STEPS}"
            )

        speed = build_speed_record(delta, float(floor_ms))
        breakdown = phase_breakdown(speed, f"{arm_dir.name} width4 window")

        record.update(
            {
                "included": True,
                "slots": slots,
                "task_count": ledger["task_count"],
                "window": window,
                "bracket_reduction": {
                    "topology": bracket["topology"],
                    "basis": bracket["basis"],
                    "distinct_bracket_origins": bracket["distinct_bracket_origins"],
                    "pre_tasks": bracket["pre_tasks"],
                    "post_tasks": bracket["post_tasks"],
                    "work_census_gate": gate,
                },
                "raw_counter_delta_window": {c: delta[c] for c in COUNTERS},
                "windowed": {
                    "events_per_step": speed["events_per_step"],
                    "step_wall_ms": speed["step_wall_ms"],
                    "measured_tps_fullstep_wall": speed["measured_tps_fullstep_wall"],
                    "per_request_step_tps": breakdown["per_request_step_tps"],
                    "committed_per_event": speed["committed_per_event"],
                    "accept_per_event": speed["accept_per_event"],
                    "prefill_frac": speed["prefill_frac"],
                    "per_request_decode_tps": speed["per_request_decode_tps"],
                    "retained_wall_fraction": speed["retained_wall_fraction"],
                    "floor_ratio": speed["floor_ratio"],
                    "window_steps": step_count,
                    "window_wall_s": window["window_wall_s"],
                    "window_wall_fraction_of_arm": window[
                        "window_wall_fraction_of_arm"
                    ],
                    "window_step_fraction_of_arm": (
                        step_count / arm_speed["bracket_reduction"]["work_census_gate"][
                            "census_steps"
                        ]
                        if arm_speed.get("bracket_reduction", {})
                        .get("work_census_gate", {})
                        .get("census_steps")
                        else None
                    ),
                    "generation_tokens": speed["generation_tokens"],
                },
                "phase_breakdown": breakdown,
                "arm_level_blended": {
                    "events_per_step": arm_speed.get("events_per_step"),
                    "step_wall_ms": arm_speed.get("step_wall_ms"),
                    "measured_tps_fullstep_wall": arm_speed.get(
                        "measured_tps_fullstep_wall"
                    ),
                    "per_request_step_tps": (
                        arm_speed["measured_tps_fullstep_wall"]
                        / arm_speed["events_per_step"]
                        if arm_speed.get("events_per_step")
                        else None
                    ),
                    "prefill_frac": arm_speed.get("prefill_frac"),
                    "retained_wall_fraction": (
                        arm_speed.get("wall_steps_measured", 0)
                        / arm_speed["bracket_reduction"]["work_census_gate"][
                            "census_steps"
                        ]
                        if arm_speed.get("bracket_reduction", {})
                        .get("work_census_gate", {})
                        .get("census_steps")
                        else None
                    ),
                    "role": "context only -- the mixture this class decomposes",
                },
                "pool_ledger": {
                    "recomputed": ledger["recomputed"],
                    "recorded_full_width_fraction": ledger[
                        "recorded_full_width_fraction"
                    ],
                    "recorded_time_weighted_mean_depth": ledger[
                        "recorded_time_weighted_mean_depth"
                    ],
                    "arm_level_depth_floor_applied": APPLIES_ARM_LEVEL_DEPTH_FLOOR,
                    "arm_level_depth_floor_note": (
                        "the 3.2 time-weighted depth floor is NOT applied: it "
                        "gates the mixture weight, which this class removes by "
                        "construction. Structural pool invariants ARE gated"
                    ),
                },
            }
        )
    except (Width4Error, TimingMathError, reducer.B4GateError, json.JSONDecodeError, OSError) as error:
        record["included"] = False
        record["exclusion_reason"] = f"{type(error).__name__}: {error}"
    return record


# --------------------------------------------------------------------------- #
# statistics                                                                    #
# --------------------------------------------------------------------------- #
INTERVAL_FIELDS: tuple[str, ...] = (
    "per_request_step_tps",
    "measured_tps_fullstep_wall",
    "events_per_step",
    "step_wall_ms",
)
PRIMARY_STATISTIC = "per_request_step_tps"
PRIMARY_STATISTIC_REASON = (
    "Mark's pinned-effective-batch ruling (2026-08-12): pin an effective batch and "
    "optimize per-request TPS under it. The window IS the pin -- it holds the "
    "served width at slots -- so the aggregate is no longer a free parameter the "
    "schedule can inflate, and what remains to optimize is the rate one request "
    "experiences. This inverts the pool16 class, which made the aggregate primary "
    "because co-residency was exactly what it was measuring; here co-residency is "
    "held fixed by the window instead of measured, so the 3c6d663d6 lesson applies "
    "in full force and per_request_step_tps is primary."
)


def interval_or_reason(values: list[float]) -> dict[str, Any]:
    """One-sided interval via the repo's pinned criticals, or a named refusal."""
    try:
        return {"status": "evaluated", **reducer.cluster_interval(values)}
    except reducer.B4GateError as error:
        count = len(values)
        point = statistics.fmean(values) if values else None
        sample_sd = statistics.stdev(values) if count > 1 else None
        return {
            "status": "no_pinned_critical",
            "reason": str(error),
            "cluster_count": count,
            "df": count - 1,
            "point_estimate": point,
            "sample_sd_across_draws": sample_sd,
            "cv": (sample_sd / point) if (sample_sd and point) else None,
            "values": list(values),
        }


def summarize_draws(label: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    included = [r for r in records if r.get("included")]
    out: dict[str, Any] = {
        "label": label,
        "draw_count": len(included),
        "arms": [
            {
                "arm": r["arm"],
                "pass_index": r["pass_index"],
                "mode": r["mode"],
                **{k: r["windowed"][k] for k in INTERVAL_FIELDS},
                "window_steps": r["windowed"]["window_steps"],
                "window_wall_fraction_of_arm": r["windowed"][
                    "window_wall_fraction_of_arm"
                ],
            }
            for r in included
        ],
        "excluded": [
            {"arm": r["arm"], "pass_index": r["pass_index"], "reason": r["exclusion_reason"]}
            for r in records
            if not r.get("included")
        ],
        "statistics": {},
    }
    for field in INTERVAL_FIELDS:
        values = [float(r["windowed"][field]) for r in included]
        entry = interval_or_reason(values)
        entry["role"] = "primary" if field == PRIMARY_STATISTIC else "reported"
        out["statistics"][field] = entry
    return out


# --------------------------------------------------------------------------- #
# discovery + payload                                                           #
# --------------------------------------------------------------------------- #
def discover_arms(gate_root: Path) -> list[dict[str, Any]]:
    if not gate_root.is_dir():
        raise Width4Error(f"{gate_root} is not a campaign root")
    records: list[dict[str, Any]] = []
    for pass_dir in sorted(p for p in gate_root.glob("pass_*") if p.is_dir()):
        try:
            pass_index = int(pass_dir.name.split("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise Width4Error(f"{pass_dir} is not a pass_NN directory") from error
        for mode in TOPOLOGIES:
            for arm_dir in sorted(pass_dir.glob(f"{mode}_*")):
                if arm_dir.is_dir() and not arm_dir.is_symlink():
                    records.append(
                        reduce_window_arm(arm_dir, mode=mode, pass_index=pass_index)
                    )
    if not records:
        raise Width4Error(f"no pass_NN/<mode>_* arms under {gate_root}")
    return records


def build_payload(
    *, gate_root: Path, source_commit: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in TOPOLOGIES}
    for record in records:
        by_mode.setdefault(record["mode"], []).append(record)

    topologies = {
        mode: summarize_draws(mode, rows) for mode, rows in by_mode.items() if rows
    }
    included = [r for r in records if r.get("included")]

    # POOLED: 8 windows are NOT 8 independent draws. The two arms of a pass share
    # a host and a task-difficulty draw (design.md section 9), so the independent
    # unit is the pass. Per topology that is n=4, df=3 -- exactly the pinned
    # critical. Pooled across topologies it is n=8, df=7, which the repo does NOT
    # pin, so no pooled interval is emitted rather than a critical being invented.
    pooled = {
        "draw_count": len(included),
        "independence_note": (
            "the two arms inside a pass share host conditions and one "
            "task-difficulty draw, so the independent unit is the PASS. Per "
            "topology that is n=4 (df=3), which the repo pins. Pooled across both "
            "topologies it is n=8 (df=7), which the repo does NOT pin"
        ),
        "statistics": {
            field: interval_or_reason(
                [float(r["windowed"][field]) for r in included]
            )
            for field in INTERVAL_FIELDS
        },
    }

    window_steps = [r["windowed"]["window_steps"] for r in included]
    admissibility = {
        "min_window_steps": MIN_WINDOW_STEPS,
        "min_window_steps_basis": (
            "pinned as a constant, not fitted: at the ~385 ms/step this stack "
            "measures it is ~6.4 minutes of sustained full-width serving and "
            "leaves ~880 wall-chain-retained samples behind the step mean"
        ),
        "observed_min_window_steps": min(window_steps) if window_steps else None,
        "observed_max_window_steps": max(window_steps) if window_steps else None,
        "floor_did_work": bool(window_steps) and min(window_steps) < MIN_WINDOW_STEPS,
        "floor_note": (
            "on this campaign the floor excluded nothing -- the shortest window is "
            f"{min(window_steps)} steps, {min(window_steps) / MIN_WINDOW_STEPS:.2f}x "
            "the floor. It is a guard for future short-window arms, not a filter "
            "that shaped this result"
        )
        if window_steps
        else "no admitted window",
        "arm_level_depth_floor_applied": APPLIES_ARM_LEVEL_DEPTH_FLOOR,
        "census_gate": "exact equality between counter delta and census, no tolerance",
    }

    all_included = bool(included) and len(included) == len(records)
    per_topology_intervals = all(
        summary["statistics"][PRIMARY_STATISTIC]["status"] == "evaluated"
        for summary in topologies.values()
    )
    analysis_valid = (
        all_included
        and len(topologies) == len(TOPOLOGIES)
        and per_topology_intervals
    )
    verdict = "WINDOWED" if analysis_valid else "NOT_EVALUATED_INSUFFICIENT_WINDOWS"

    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "run_class": RUN_CLASS,
        "title": TITLE,
        "gate_root": str(gate_root),
        "source_commit": source_commit,
        "verdict": verdict,
        "analysis_valid": analysis_valid,
        # An INSTRUMENT, never a floor-acceptance verdict.
        "citable": False,
        "citable_reason": (
            "this class is an instrument for the step-wall levers, not a sealed "
            "claim. It reports intervals so a lever can be judged against them; it "
            "does not itself promote anything"
        ),
        "formal_floor_acceptance_eligible": False,
        "b4_cap_applicable": False,
        "b4_cap_reason": (
            "35-38k contexts imply 42-49 GB mandatory bytes and a 155-179 ms floor, "
            "above the 137.607 ms one-sided cap"
        ),
        "primary_statistic": PRIMARY_STATISTIC,
        "primary_statistic_reason": PRIMARY_STATISTIC_REASON,
        "claims": list(CLAIMS),
        "does_not_claim": list(DOES_NOT_CLAIM),
        "window_definition": {
            "open": "first ledger event at which depth == slots",
            "close": "first `complete` ledger event carrying pending == 0",
            "inputs": "depth and pending only; no measured rate participates",
            "drain_exclusion": (
                "by construction. _window_occupancy PROVES that zero full-width "
                "wall lies outside the window, by exact comparison against 0.0"
            ),
            "bracket": (
                "real Prometheus snapshots at both edges; nothing interpolated"
            ),
            "census_cross_check": (
                "counter step delta indexes the census range; step count and event "
                "sum must match with exact equality"
            ),
        },
        "admissibility": admissibility,
        "uncertainty_model": {
            "interval": "one-sided 95% Student-t on between-pass variation",
            "criticals": {str(k): v for k, v in sorted(T95_ONE_SIDED.items())},
            "admissible_n": "exactly 4 or 16 included draws per cluster",
            "note": (
                "no critical is invented for an unpinned df; a cluster that does "
                "not fit reports status no_pinned_critical with its point estimate "
                "and CV and no bounds"
            ),
        },
        "topologies": topologies,
        "pooled": pooled,
        "arms": records,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"{payload['title']} -- {payload['verdict']}",
        f"  gate_root: {payload['gate_root']}",
        f"  run_class: {payload['run_class']}  citable={payload['citable']}",
        "",
        f"  window: {payload['window_definition']['open']}",
        f"       -> {payload['window_definition']['close']}",
        "",
    ]
    header = (
        f"  {'pass':>4} {'mode':16} {'eps':>7} {'per-req':>8} {'agg':>7} "
        f"{'step ms':>8} {'steps':>6} {'wall%':>6}"
    )
    lines.append(header)
    for record in payload["arms"]:
        if not record.get("included"):
            lines.append(
                f"  {record['pass_index']:>4} {record['mode']:16} EXCLUDED: "
                f"{record['exclusion_reason']}"
            )
            continue
        w = record["windowed"]
        lines.append(
            f"  {record['pass_index']:>4} {record['mode']:16} "
            f"{w['events_per_step']:>7.3f} {w['per_request_step_tps']:>8.2f} "
            f"{w['measured_tps_fullstep_wall']:>7.2f} {w['step_wall_ms']:>8.1f} "
            f"{w['window_steps']:>6} {w['window_wall_fraction_of_arm'] * 100:>5.1f}%"
        )
    lines.append("")
    for mode, summary in sorted(payload["topologies"].items()):
        lines.append(f"  {mode}  (n={summary['draw_count']})")
        for field in INTERVAL_FIELDS:
            entry = summary["statistics"][field]
            role = "*" if entry.get("role") == "primary" else " "
            if entry["status"] == "evaluated":
                lines.append(
                    f"   {role} {field:28} {entry['point_estimate']:>9.3f}  "
                    f"[{entry['l95']:.3f}, {entry['u95']:.3f}]  "
                    f"CV {entry['cv'] * 100:.2f}%"
                )
            else:
                cv = entry.get("cv")
                lines.append(
                    f"   {role} {field:28} {entry['point_estimate']:>9.3f}  "
                    f"(no interval: df={entry['df']})"
                    + (f"  CV {cv * 100:.2f}%" if cv else "")
                )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reduce the width-4 depth window of every arm of a pool campaign."
    )
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--source-commit", type=str, default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        records = discover_arms(args.gate_root)
    except Width4Error as error:
        print(f"class-9 FAIL-LOUD [b4-width4-window]: {error}", file=sys.stderr)
        return 2

    payload = build_payload(
        gate_root=args.gate_root,
        source_commit=args.source_commit,
        records=records,
    )
    out = args.out or (args.gate_root / "fr13_b4_width4_operating_point.json")
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
