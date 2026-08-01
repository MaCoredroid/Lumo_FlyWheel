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
    return {
        "events_per_step": events_per_step,
        "wall_ms_per_event": wall_s_per_event * 1000.0,
        "sfwd_gpu_ms_per_event": sfwd_s_per_event * 1000.0,
        "sfwd_gpu_ms_per_step": sfwd_ms,
        "dfwd_gpu_ms_per_step": dfwd_ms,
        "cfwd_gpu_ms_per_step": cfwd_ms,
        "gpu_component_ms_per_step": gpu_component_ms,
        "other_wall_ms_per_step": other_wall_ms,
    }
