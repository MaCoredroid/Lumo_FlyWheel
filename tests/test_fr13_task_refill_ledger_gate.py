"""The admission ledger is REQUIRED EVIDENCE for a task-pool run.

A pool run's whole citable claim is that it held the served width at the slot
count instead of letting it decay as sessions ended.  That claim lives only in
the admission ledger, so the gate requires the ledger to be present, internally
consistent, recomputable from its raw events, and above an occupancy floor --
and it must never let the pool size widen the floor-gate concurrency bounds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fr13_floor_gate as floor_gate  # noqa: E402


def _write_pool(
    arm_dir: Path,
    *,
    slots: int = 4,
    task_count: int = 16,
    hold_s: float = 100.0,
    idle_s: float = 0.0,
    summary_overrides: dict | None = None,
) -> Path:
    """Write a ledger whose depth rises to `slots`, holds, then drains.

    `idle_s` inserts a stretch at depth 1 before the drain, which is how a pool
    that failed to keep its slots full looks.
    """
    root = arm_dir / "swe_out" / "verified"
    root.mkdir(parents=True, exist_ok=True)

    events = []
    t = 0.0
    depth = 0
    for index in range(slots):  # fill every slot
        depth += 1
        events.append({"event": "admit", "t_s": t, "depth": depth, "i": index})
        t += 1e-3
    t += hold_s  # run at full width
    if idle_s > 0.0:
        for _ in range(slots - 1):  # drop to depth 1 and sit there
            depth -= 1
            events.append({"event": "complete", "t_s": t, "depth": depth})
            t += 1e-3
        t += idle_s
    while depth > 0:  # drain
        depth -= 1
        events.append({"event": "complete", "t_s": t, "depth": depth})
        t += 1e-3

    ledger = root / "fr13_task_refill_ledger.jsonl"
    ledger.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    recomputed = floor_gate._reduce_refill_ledger(events, slots)
    summary = {
        "schema": floor_gate.TASK_REFILL_SUMMARY_SCHEMA,
        "task_count": task_count,
        "slots": slots,
        "arm_wall_s": recomputed["arm_wall_s"],
        "time_weighted_mean_depth": recomputed["time_weighted_mean_depth"],
        "full_width_fraction": recomputed["full_width_fraction"],
        "admissions": slots,
        "peak_depth": slots,
        "completed": task_count,
        "aborted": False,
    }
    summary.update(summary_overrides or {})
    (root / "fr13_task_refill_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return arm_dir


def test_a_non_pool_arm_is_simply_absent(tmp_path: Path) -> None:
    """The default ThreadPoolExecutor.map path leaves no ledger and is fine."""
    (tmp_path / "swe_out" / "verified").mkdir(parents=True)
    result = floor_gate.validate_task_refill_ledger(
        tmp_path, concurrency=4, task_count=4
    )
    assert result == {"status": "absent", "pool_run": False}


def test_a_healthy_pool_run_passes_and_reports_its_occupancy(tmp_path: Path) -> None:
    _write_pool(tmp_path)
    result = floor_gate.validate_task_refill_ledger(
        tmp_path, concurrency=4, task_count=16
    )
    assert result["status"] == "pass"
    assert result["pool_run"] is True
    assert result["slots"] == 4
    assert result["peak_depth"] == 4
    assert result["time_weighted_mean_depth"] >= floor_gate.MIN_POOL_TIME_WEIGHTED_DEPTH
    assert result["full_width_fraction"] >= floor_gate.MIN_POOL_FULL_WIDTH_FRACTION
    # The ledger measures worker-thread depth, not engine co-residency, and the
    # artifact has to say so rather than let a reader mistake the two.
    assert "UPPER BOUND" in result["depth_basis"]


def test_half_present_pool_evidence_is_fatal(tmp_path: Path) -> None:
    _write_pool(tmp_path)
    (tmp_path / "swe_out" / "verified" / "fr13_task_refill_summary.json").unlink()
    with pytest.raises(floor_gate.GateError, match="half-present"):
        floor_gate.validate_task_refill_ledger(tmp_path, concurrency=4, task_count=16)


def test_a_pool_may_never_widen_the_concurrency_bounds(tmp_path: Path) -> None:
    """THE invariant: the floor-gate bounds are fed the SLOT count.

    A 16-task pool served on 4 slots must be checked against 4, never 16 --
    otherwise `steps <= drafts <= concurrency * steps` would be 4x looser and a
    pool run could launder a draft-rate violation.
    """
    _write_pool(tmp_path, slots=4, task_count=16)
    # Presenting the pool size as the serving concurrency is refused.
    with pytest.raises(floor_gate.GateError, match="may never widen them"):
        floor_gate.validate_task_refill_ledger(
            tmp_path, concurrency=16, task_count=16
        )
    # And the accepted record carries the slot count, not the pool size.
    result = floor_gate.validate_task_refill_ledger(
        tmp_path, concurrency=4, task_count=16
    )
    assert result["slots"] == 4
    assert result["task_count"] == 16


def test_a_pool_no_larger_than_its_slots_is_not_a_pool(tmp_path: Path) -> None:
    _write_pool(tmp_path, slots=4, task_count=4)
    with pytest.raises(floor_gate.GateError, match="must be LARGER"):
        floor_gate.validate_task_refill_ledger(tmp_path, concurrency=4, task_count=4)


def test_depth_below_the_floor_is_fatal_and_names_the_value(tmp_path: Path) -> None:
    """A pool that did not hold its width is not citable as a pool run."""
    _write_pool(tmp_path, hold_s=1.0, idle_s=400.0)
    with pytest.raises(floor_gate.GateError) as excinfo:
        floor_gate.validate_task_refill_ledger(tmp_path, concurrency=4, task_count=16)
    message = str(excinfo.value)
    assert "time-weighted mean depth" in message
    assert "below the pool floor 3.2" in message


def test_a_forged_summary_cannot_beat_the_ledger(tmp_path: Path) -> None:
    """The summary is recomputed from the raw events, so editing it fails."""
    _write_pool(
        tmp_path,
        hold_s=1.0,
        idle_s=400.0,
        summary_overrides={
            "time_weighted_mean_depth": 3.99,
            "full_width_fraction": 0.99,
        },
    )
    with pytest.raises(floor_gate.GateError, match="disagrees with the ledger"):
        floor_gate.validate_task_refill_ledger(tmp_path, concurrency=4, task_count=16)


def test_an_in_flight_invariant_violation_is_fatal(tmp_path: Path) -> None:
    _write_pool(tmp_path, summary_overrides={"peak_depth": 5})
    with pytest.raises(floor_gate.GateError, match="exceeds the slot count"):
        floor_gate.validate_task_refill_ledger(tmp_path, concurrency=4, task_count=16)


def test_an_aborted_or_short_pool_run_is_fatal(tmp_path: Path) -> None:
    _write_pool(tmp_path, summary_overrides={"aborted": True})
    with pytest.raises(floor_gate.GateError, match="aborted"):
        floor_gate.validate_task_refill_ledger(tmp_path, concurrency=4, task_count=16)

    _write_pool(tmp_path, summary_overrides={"completed": 15})
    with pytest.raises(floor_gate.GateError, match="completed 15 of 16"):
        floor_gate.validate_task_refill_ledger(tmp_path, concurrency=4, task_count=16)
