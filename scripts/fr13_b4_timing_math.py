#!/usr/bin/env python3
"""Reconcile fixed32 B4 full-wall and GPU phase timing units."""

from __future__ import annotations

import math
from typing import Any


class TimingMathError(ValueError):
    """A timing record mixes units or fails an arithmetic identity."""


def positive(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimingMathError(f"{key} is missing from full-wall timing evidence")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise TimingMathError(f"{key} is not finite and positive")
    return value


def phase_breakdown(record: dict[str, Any], label: str) -> dict[str, float]:
    wall_ms = positive(record, "step_wall_ms")
    wall_tps = positive(record, "measured_tps_fullstep_wall")
    events_per_step = positive(record, "events_per_step")
    wall_s_per_event = positive(record, "wall_s_per_event")
    sfwd_s_per_event = positive(record, "s_per_fwd_gpu")
    sfwd_s_per_step = positive(record, "s_per_fwd_gpu_per_forward")
    if not math.isclose(
        sfwd_s_per_step,
        sfwd_s_per_event * events_per_step,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise TimingMathError(f"{label} SFWD event/step units do not reconcile")
    if not math.isclose(
        wall_ms,
        wall_s_per_event * events_per_step * 1000.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise TimingMathError(f"{label} wall event/step units do not reconcile")

    committed_per_event = positive(record, "committed_per_event")
    accepted_per_event = positive(record, "accept_per_event")
    if not math.isclose(
        committed_per_event,
        accepted_per_event + 1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TimingMathError(
            f"{label} accepted and committed token rates do not reconcile"
        )
    if not math.isclose(
        wall_tps,
        committed_per_event / wall_s_per_event,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise TimingMathError(f"{label} full-wall TPS does not reconcile")
    if not math.isclose(
        positive(record, "floor_ratio"),
        wall_ms / positive(record, "floor_ms"),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise TimingMathError(f"{label} hardware-floor ratio does not reconcile")

    sfwd_ms = sfwd_s_per_step * 1000.0
    dfwd_ms = positive(record, "drafter_gpu_ms_per_step")
    cfwd_ms = positive(record, "committer_gpu_ms_per_step")
    gpu_component_ms = sfwd_ms + dfwd_ms + cfwd_ms
    other_wall_ms = wall_ms - gpu_component_ms
    if not math.isfinite(other_wall_ms) or other_wall_ms < 0:
        raise TimingMathError(f"{label} phase components exceed full-step wall time")
    if not math.isclose(
        gpu_component_ms + other_wall_ms,
        wall_ms,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise TimingMathError(f"{label} phase breakdown does not reconcile")
    # The BATCH-INVARIANT rate: committed tokens per second for ONE request,
    # i.e. what a single agent actually experiences. The aggregate
    # measured_tps_fullstep_wall multiplies this by co-residency, so an arm can
    # post a large aggregate gain purely by holding more requests resident --
    # scheduling luck -- while every individual request got SLOWER. Promotion
    # must not reward that, so both rates are reported side by side.
    per_request_step_tps = committed_per_event / (wall_ms / 1000.0)
    if not math.isclose(
        per_request_step_tps,
        wall_tps / events_per_step,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise TimingMathError(
            f"{label} per-request and aggregate step throughput do not reconcile"
        )
    return {
        "events_per_step": events_per_step,
        "wall_ms_per_event": wall_s_per_event * 1000.0,
        "sfwd_gpu_ms_per_event": sfwd_s_per_event * 1000.0,
        "sfwd_gpu_ms_per_step": sfwd_ms,
        "dfwd_gpu_ms_per_step": dfwd_ms,
        "cfwd_gpu_ms_per_step": cfwd_ms,
        "gpu_component_ms_per_step": gpu_component_ms,
        "other_wall_ms_per_step": other_wall_ms,
        "measured_tps_fullstep_wall": wall_tps,
        "per_request_step_tps": per_request_step_tps,
    }


def promotion_verdict(
    stock: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Decide whether a candidate arm is PROMOTABLE against its stock arm.

    Two conditions, both required:

      1. NON-REGRESSION on the batch-invariant rate: the candidate's
         per_request_step_tps must be at least the stock's. This is the
         condition that cannot be bought with co-residency.
      2. A real AGGREGATE gain: measured_tps_fullstep_wall must improve.

    Condition 1 exists because of the measured two-M case, which cleared 2 on
    its own and would have been promoted on the aggregate alone: +17.2%
    aggregate TPS while per-request step TPS fell 2.96%. The whole aggregate
    gain was extra co-residency the scheduler happened to supply, not faster
    service; every individual agent was slower. See
    results/fr13_b4_promotion_criterion_20260809/README.md.

    Both inputs are phase_breakdown() results, so the two rates have already
    been reconciled against the recorded counters.
    """
    stock_per_request = positive(stock, "per_request_step_tps")
    candidate_per_request = positive(candidate, "per_request_step_tps")
    stock_aggregate = positive(stock, "measured_tps_fullstep_wall")
    candidate_aggregate = positive(candidate, "measured_tps_fullstep_wall")

    per_request_delta_frac = (
        candidate_per_request - stock_per_request
    ) / stock_per_request
    aggregate_delta_frac = (
        candidate_aggregate - stock_aggregate
    ) / stock_aggregate

    per_request_non_regression = candidate_per_request >= stock_per_request
    aggregate_gain = candidate_aggregate > stock_aggregate
    eligible = per_request_non_regression and aggregate_gain

    if eligible:
        reason = "aggregate gain with no per-request regression"
    elif aggregate_gain and not per_request_non_regression:
        reason = (
            "aggregate gain is co-residency, not speed: per-request step TPS "
            f"regressed {per_request_delta_frac * -100.0:.2f}% while aggregate "
            f"rose {aggregate_delta_frac * 100.0:.2f}%"
        )
    elif per_request_non_regression:
        reason = "no aggregate throughput gain"
    else:
        reason = "regression on both aggregate and per-request throughput"

    return {
        "promotion_eligible": eligible,
        "per_request_non_regression": per_request_non_regression,
        "aggregate_gain": aggregate_gain,
        "stock_per_request_step_tps": stock_per_request,
        "candidate_per_request_step_tps": candidate_per_request,
        "per_request_step_tps_delta_frac": per_request_delta_frac,
        "stock_measured_tps_fullstep_wall": stock_aggregate,
        "candidate_measured_tps_fullstep_wall": candidate_aggregate,
        "measured_tps_fullstep_wall_delta_frac": aggregate_delta_frac,
        "reason": reason,
    }
