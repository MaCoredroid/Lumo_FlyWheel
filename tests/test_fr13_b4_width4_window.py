"""The width-4 depth window: what it measures, and what it refuses to measure.

The pool16 campaign sealed NOT CITABLE because an arm's events/step is a MIXTURE
WEIGHT -- a wall-blend of a full-width phase and an exact4-shaped drain -- rather
than an operating point.  This class windows the full-width phase out of the
blend.  These tests pin the three things that make that honest:

  1. the window is derived from ADMISSION STRUCTURE ONLY (depth, pending), so it
     cannot be slid toward a favourable number;
  2. the drain exclusion is PROVEN per arm, not asserted, by requiring that zero
     full-width wall lies outside the derived window;
  3. every gate is exact -- no tolerances anywhere -- and every exclusion is
     named.

Fixtures are synthetic ledgers, censuses and Prometheus brackets built so that
the derived rates are exactly known in closed form (per_request_step_tps == 12.5
by construction), which is what lets a boundary bug show up as a wrong NUMBER
rather than only as a wrong shape.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(SCRIPTS))
width4 = _load("fr13_b4_width4_window_reduce", "scripts/fr13_b4_width4_window_reduce.py")
measure = width4.measure

M = measure

# Per-step constants the fixture is built on.  They are chosen so that
# per_request_step_tps = committed_per_event / step_wall_s = 5.0 / 0.400 = 12.5
# exactly, independent of how many steps or events the window contains -- so any
# boundary error moves events_per_step and the aggregate while leaving the
# per-request rate pinned, which is precisely the discrimination this class needs.
STEP_WALL_S = 0.400
SFWD_S_PER_STEP = 0.100
DFWD_S_PER_STEP = 0.045
CFWD_S_PER_STEP = 0.055
ACCEPT_PER_EVENT = 4.0
FLOOR_MS = 109.336011018


def batch_at(step: int) -> int:
    """A width profile that is mostly 4 but not uniformly 4.

    Deliberately not constant: a window that silently swallowed drain steps would
    otherwise be invisible in events_per_step.
    """
    return 3 if step % 5 == 0 else 4


def events_through(step: int) -> int:
    """Cumulative census events strictly before `step`."""
    return sum(batch_at(s) for s in range(step))


def counters_at(step: int) -> dict[str, float]:
    """Every engine counter as a closed-form function of the forward step."""
    events = events_through(step)
    return {
        M.M_DECODE_FWD_GPU_STEPS: float(step),
        M.M_DECODE_FWD_GPU_DRAFTS: float(events),
        M.M_DECODE_FWD_GPU_S: SFWD_S_PER_STEP * step,
        M.M_DRAFTER_GPU_S: DFWD_S_PER_STEP * step,
        M.M_DRAFTER_GPU_SPANS: float(step),
        M.M_COMMITTER_GPU_S: CFWD_S_PER_STEP * step,
        M.M_COMMITTER_GPU_SPANS: float(step),
        M.M_STEP_WALL_STEPS: float(step),
        M.M_STEP_WALL_DRAFTS: float(events),
        M.M_STEP_WALL_S: STEP_WALL_S * step,
        M.M_DRAFTS: float(events),
        M.M_ACCEPTED: ACCEPT_PER_EVENT * events,
        M.M_DRAFT_TOK: 31.0 * events,
        M.M_GEN_TOK: 5.0 * events,
        M.M_DECODE_S: 1.0 * events,
        M.M_PREFILL_S: 0.2 * events,
        M.M_TPOT_COUNT: float(max(step // 100, 1)),
        M.M_TPOT_SUM: 0.08 * max(step // 100, 1),
    }


def write_metrics(path: Path, step: int) -> None:
    lines = ["# HELP synthetic", "# TYPE synthetic counter"]
    for name, value in counters_at(step).items():
        lines.append(f'{name}{{engine="0",model_name="synthetic"}} {value!r}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


#: (event, instance, depth, pending, t_s, forward_step)
CLEAN_TIMELINE: tuple[tuple[str, str, int, int, float, int], ...] = (
    ("admit", "t0", 1, 7, 0.001, 0),
    ("admit", "t1", 2, 6, 0.002, 0),
    ("admit", "t2", 3, 5, 0.003, 0),
    ("admit", "t3", 4, 4, 0.004, 0),
    ("complete", "t0", 3, 4, 100.0, 250),
    ("admit", "t4", 4, 3, 100.001, 280),
    ("complete", "t1", 3, 3, 200.0, 500),
    ("admit", "t5", 4, 2, 200.001, 530),
    ("complete", "t2", 3, 2, 300.0, 750),
    ("admit", "t6", 4, 1, 300.001, 780),
    ("complete", "t3", 3, 1, 400.0, 1000),
    ("admit", "t7", 4, 0, 400.001, 1030),
    # pending is 0 here: the pool has stopped being a pool.  WINDOW CLOSES.
    ("complete", "t4", 3, 0, 500.0, 1250),
    ("complete", "t5", 2, 0, 600.0, 1500),
    ("complete", "t6", 1, 0, 700.0, 1750),
    ("complete", "t7", 0, 0, 800.0, 2000),
)
CLEAN_WINDOW_STEPS = 1250
CLEAN_TOTAL_STEPS = 2000


def build_arm(
    arm_dir: Path,
    timeline: tuple[tuple[str, str, int, int, float, int], ...] = CLEAN_TIMELINE,
    *,
    slots: int = 4,
    total_steps: int = CLEAN_TOTAL_STEPS,
    ended: bool = True,
    census_skip: set[int] | None = None,
    census_batch_override: dict[int, int] | None = None,
    summary_override: dict[str, Any] | None = None,
) -> Path:
    """Materialise a synthetic arm that the real reducer can read end to end."""
    verified = arm_dir / "swe_out" / "verified"
    (verified / "per_task").mkdir(parents=True, exist_ok=True)
    (arm_dir / "logs").mkdir(parents=True, exist_ok=True)

    events = [
        {
            "event": event,
            "instance_id": instance,
            "depth": depth,
            "pending": pending,
            "t_s": t_s,
        }
        for event, instance, depth, pending, t_s, _ in timeline
    ]
    (verified / "fr13_task_refill_ledger.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in events), encoding="utf-8"
    )

    occupancy = width4.reduce_ledger_occupancy(events, slots)
    task_ids = sorted({instance for _, instance, _, _, _, _ in timeline})
    summary = {
        "schema": width4.TASK_REFILL_SUMMARY_SCHEMA,
        "slots": slots,
        "task_count": len(task_ids),
        "admissions": sum(1 for e in events if e["event"] == "admit"),
        "completed": sum(1 for e in events if e["event"] == "complete"),
        "aborted": False,
        "peak_depth": max(e["depth"] for e in events),
        "arm_wall_s": occupancy["arm_wall_s"],
        "time_weighted_mean_depth": occupancy["time_weighted_mean_depth"],
        "full_width_fraction": occupancy["full_width_fraction"],
    }
    summary.update(summary_override or {})
    (verified / "fr13_task_refill_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    for event, instance, _, _, _, step in timeline:
        if event == "admit":
            starts[instance] = step
        else:
            ends[instance] = step
    for instance in task_ids:
        task_dir = verified / "per_task" / instance
        task_dir.mkdir(parents=True, exist_ok=True)
        write_metrics(task_dir / "vllm_metrics_pre.txt", starts[instance])
        write_metrics(task_dir / "vllm_metrics_post.txt", ends[instance])

    skip = census_skip or set()
    override = census_batch_override or {}
    with (arm_dir / "logs" / "fr13_fixed32_work_census.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for step in range(total_steps):
            if step in skip:
                continue
            record = {
                "schema": "fr13-fixed32-work-census-v12",
                "event_complete": True,
                "forward_step_index": step,
                "batch_size": override.get(step, batch_at(step)),
                "batch_purity": {"mixed_pseudo_rows": 0},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    whole = counters_at(total_steps)
    arm_events = whole[M.M_DECODE_FWD_GPU_DRAFTS]
    arm_eps = arm_events / total_steps
    (arm_dir / "deploy_speed_fullwall.json").write_text(
        json.dumps(
            {
                "schema": "fr13.measure.deploy_speed.v1",
                "floor_ms": FLOOR_MS,
                "events_per_step": arm_eps,
                "step_wall_ms": STEP_WALL_S * 1000.0,
                "measured_tps_fullstep_wall": (ACCEPT_PER_EVENT + 1.0) / STEP_WALL_S,
                "prefill_frac": 0.2,
                "wall_steps_measured": float(total_steps),
                "bracket_reduction": {
                    "topology": "staggered",
                    "work_census_gate": {
                        "status": "pass",
                        "census_steps": total_steps,
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if ended:
        (arm_dir / "arm_ended_at.txt").write_text("2026-08-12T13:32:00Z\n", encoding="utf-8")
    return arm_dir


@pytest.fixture()
def clean_arm(tmp_path: Path) -> Path:
    return build_arm(tmp_path / "tail6_fixed32_pool0")


# --------------------------------------------------------------------------- #
# 1. the clean window                                                          #
# --------------------------------------------------------------------------- #
def test_clean_window_opens_at_full_width_and_closes_when_the_pool_stops_being_one(
    clean_arm: Path,
) -> None:
    events = width4.read_ledger_events(clean_arm)
    window = width4.derive_width4_window(width4.order_events(events), 4)
    # Opens at the FOURTH admit -- the first event at which depth == slots.
    assert window["opening_task"] == "t3"
    assert window["open_event"]["depth"] == 4
    # Closes at the first completion with nothing left unstarted behind it.
    assert window["closing_task"] == "t4"
    assert window["close_event"]["pending"] == 0
    assert window["close_event"]["event"] == "complete"
    assert window["close_t_s"] == pytest.approx(500.0)


def test_the_window_carries_all_of_the_arms_full_width_wall_and_the_drain_carries_none(
    clean_arm: Path,
) -> None:
    """The drain exclusion is PROVEN here, not asserted.

    If any full-width wall fell outside the window, the windowed rate would be a
    sample of the phase rather than the whole of it, and the class's central
    claim would be false.
    """
    events = width4.read_ledger_events(clean_arm)
    window = width4.derive_width4_window(width4.order_events(events), 4)
    assert window["full_width_wall_s_outside_window"] == 0.0
    recomputed = width4.reduce_ledger_occupancy(events, 4)
    assert window["full_width_wall_s_inside_window"] == pytest.approx(
        recomputed["full_width_fraction"] * recomputed["arm_wall_s"], rel=1e-12
    )
    # And the excluded drain is real, and named as excluded.
    assert window["drain_wall_s_excluded_by_construction"] > 0.0
    assert window["drain_wall_fraction_excluded_by_construction"] == pytest.approx(
        1.0 - window["window_wall_fraction_of_arm"], rel=1e-12
    )


def test_a_full_arm_reduction_recovers_the_closed_form_per_request_rate(
    clean_arm: Path,
) -> None:
    record = width4.reduce_window_arm(clean_arm, mode="tail6_fixed32", pass_index=0)
    assert record["included"], record["exclusion_reason"]
    windowed = record["windowed"]
    # Closed form: committed_per_event / step_wall_s = 5.0 / 0.400.
    assert windowed["per_request_step_tps"] == pytest.approx(12.5, rel=1e-12)
    assert windowed["step_wall_ms"] == pytest.approx(400.0, rel=1e-12)
    assert windowed["window_steps"] == CLEAN_WINDOW_STEPS
    expected_events = events_through(CLEAN_WINDOW_STEPS)
    assert windowed["events_per_step"] == pytest.approx(
        expected_events / CLEAN_WINDOW_STEPS, rel=1e-12
    )
    # aggregate == events_per_step * per_request, the identity phase_breakdown pins.
    assert windowed["measured_tps_fullstep_wall"] == pytest.approx(
        windowed["events_per_step"] * windowed["per_request_step_tps"], rel=1e-9
    )


def test_the_windowed_bracket_is_a_strict_prefix_of_the_arm_and_excludes_drain_steps(
    clean_arm: Path,
) -> None:
    record = width4.reduce_window_arm(clean_arm, mode="tail6_fixed32", pass_index=0)
    gate = record["bracket_reduction"]["work_census_gate"]
    assert gate["census_first_forward_step"] == 0
    assert gate["census_last_forward_step"] == CLEAN_WINDOW_STEPS - 1
    assert gate["census_steps"] == CLEAN_WINDOW_STEPS < CLEAN_TOTAL_STEPS
    # The post side is only the tasks that finished inside the window.
    assert record["bracket_reduction"]["post_tasks"] == ["t0", "t1", "t2", "t3", "t4"]
    # The pre side is only the tasks admitted by the time width was reached.
    assert record["bracket_reduction"]["pre_tasks"] == ["t0", "t1", "t2", "t3"]


# --------------------------------------------------------------------------- #
# 2. degenerate arms -- excluded with a NAMED reason, never silently            #
# --------------------------------------------------------------------------- #
#: A 6-task pool at 4 slots that only ever held three of them -- one worker never
#: took a task.  It IS a pool (more tasks than slots), it ran to completion, and
#: it has no full-width phase at all.
NO_WIDTH_TIMELINE = (
    ("admit", "t0", 1, 5, 0.001, 0),
    ("admit", "t1", 2, 4, 0.002, 0),
    ("admit", "t2", 3, 3, 0.003, 0),
    ("complete", "t0", 2, 3, 100.0, 200),
    ("admit", "t3", 3, 2, 100.001, 220),
    ("complete", "t1", 2, 2, 200.0, 400),
    ("admit", "t4", 3, 1, 200.001, 420),
    ("complete", "t2", 2, 1, 300.0, 600),
    ("admit", "t5", 3, 0, 300.001, 620),
    ("complete", "t3", 2, 0, 400.0, 800),
    ("complete", "t4", 1, 0, 500.0, 1000),
    ("complete", "t5", 0, 0, 600.0, 1200),
)
NO_WIDTH_TOTAL_STEPS = 1200


def test_an_arm_that_never_reached_full_width_has_NO_window_not_a_short_one(
    tmp_path: Path,
) -> None:
    """Three slots' worth of occupancy at a four-slot pool is not a narrow window.

    It is the absence of the phase this class measures, and the reducer must say
    so rather than reducing whatever it can find.
    """
    arm = build_arm(
        tmp_path / "tail6_fixed32_pool0", NO_WIDTH_TIMELINE, total_steps=NO_WIDTH_TOTAL_STEPS
    )
    record = width4.reduce_window_arm(arm, mode="tail6_fixed32", pass_index=0)
    assert record["included"] is False
    assert "never reached its full width" in record["exclusion_reason"]
    assert "no window" in record["exclusion_reason"]


#: A straggler-heavy arm: the pool drains for most of the arm, so the full-width
#: phase is genuinely short.  The window still EXISTS and is still exact -- what
#: makes it inadmissible is only its step count.
SHORT_WINDOW_TIMELINE = (
    ("admit", "t0", 1, 7, 0.001, 0),
    ("admit", "t1", 2, 6, 0.002, 0),
    ("admit", "t2", 3, 5, 0.003, 0),
    ("admit", "t3", 4, 4, 0.004, 0),
    ("complete", "t0", 3, 4, 10.0, 25),
    ("admit", "t4", 4, 3, 10.001, 28),
    ("complete", "t1", 3, 3, 20.0, 50),
    ("admit", "t5", 4, 2, 20.001, 53),
    ("complete", "t2", 3, 2, 30.0, 75),
    ("admit", "t6", 4, 1, 30.001, 78),
    ("complete", "t3", 3, 1, 40.0, 100),
    ("admit", "t7", 4, 0, 40.001, 103),
    ("complete", "t4", 3, 0, 50.0, 125),
    # the straggler tail: three long tasks drain for 9000 s
    ("complete", "t5", 2, 0, 3050.0, 800),
    ("complete", "t6", 1, 0, 6050.0, 1500),
    ("complete", "t7", 0, 0, 9050.0, 2000),
)


def test_a_straggler_heavy_arm_has_a_real_but_inadmissible_window(
    tmp_path: Path,
) -> None:
    arm = build_arm(tmp_path / "tail6_fixed32_pool0", SHORT_WINDOW_TIMELINE)
    events = width4.read_ledger_events(arm)
    window = width4.derive_width4_window(width4.order_events(events), 4)
    # The window is real and still carries 100% of the full-width wall...
    assert window["full_width_wall_s_outside_window"] == 0.0
    assert window["drain_wall_fraction_excluded_by_construction"] > 0.98
    # ...but 125 steps is below the pinned floor, so it is not a draw.
    record = width4.reduce_window_arm(arm, mode="tail6_fixed32", pass_index=0)
    assert record["included"] is False
    assert f"below the pinned admissibility floor of {width4.MIN_WINDOW_STEPS}" in (
        record["exclusion_reason"]
    )
    assert "125 forward steps" in record["exclusion_reason"]


def test_a_still_serving_arm_is_excluded_rather_than_read_torn(tmp_path: Path) -> None:
    arm = build_arm(tmp_path / "tail6_fixed32_pool0", ended=False)
    record = width4.reduce_window_arm(arm, mode="tail6_fixed32", pass_index=0)
    assert record["included"] is False
    assert "still serving" in record["exclusion_reason"]


# --------------------------------------------------------------------------- #
# 3. window-boundary edge cases                                                 #
# --------------------------------------------------------------------------- #
def test_the_close_is_the_FIRST_pending_zero_completion_not_a_later_one(
    clean_arm: Path,
) -> None:
    """t5, t6 and t7 also complete at pending == 0.  Taking any of them would
    swallow drain wall into the operating point."""
    events = width4.read_ledger_events(clean_arm)
    ordered = width4.order_events(events)
    window = width4.derive_width4_window(ordered, 4)
    later = [
        i
        for i, e in enumerate(ordered)
        if e["event"] == "complete" and e["pending"] == 0
    ]
    assert len(later) == 4
    assert window["close_index"] == later[0]


def test_the_window_must_contain_the_whole_admission_sequence(clean_arm: Path) -> None:
    events = width4.read_ledger_events(clean_arm)
    ordered = width4.order_events(events)
    window = width4.derive_width4_window(ordered, 4)
    last_admit = max(
        i for i, e in enumerate(ordered) if e["event"] == "admit"
    )
    assert window["open_index"] <= last_admit < window["close_index"]


def test_full_width_wall_after_the_close_is_refused_rather_than_averaged_in() -> None:
    """A ledger where the pool returns to full width after the refill phase ends.

    Physically impossible for this runner, which is exactly why it must raise: it
    would mean the window definition no longer matches the admission mechanics,
    and every rate built on it would be quietly wrong.
    """
    ordered = [
        {"event": "admit", "instance_id": "t0", "depth": 1, "pending": 3, "t_s": 0.0},
        {"event": "admit", "instance_id": "t1", "depth": 2, "pending": 2, "t_s": 0.1},
        {"event": "admit", "instance_id": "t2", "depth": 3, "pending": 1, "t_s": 0.2},
        {"event": "admit", "instance_id": "t3", "depth": 4, "pending": 0, "t_s": 0.3},
        {"event": "complete", "instance_id": "t0", "depth": 3, "pending": 0, "t_s": 100.0},
        # the impossible re-entry: depth back to 4 with nothing pending
        {"event": "admit", "instance_id": "t4", "depth": 4, "pending": 0, "t_s": 150.0},
        {"event": "complete", "instance_id": "t1", "depth": 3, "pending": 0, "t_s": 200.0},
        {"event": "complete", "instance_id": "t2", "depth": 2, "pending": 0, "t_s": 300.0},
        {"event": "complete", "instance_id": "t3", "depth": 1, "pending": 0, "t_s": 400.0},
        {"event": "complete", "instance_id": "t4", "depth": 0, "pending": 0, "t_s": 500.0},
    ]
    with pytest.raises(width4.Width4Error) as excinfo:
        width4.derive_width4_window(ordered, 4)
    assert "OUTSIDE the derived window" in str(excinfo.value)


def test_a_census_hole_inside_the_window_is_fatal(tmp_path: Path) -> None:
    arm = build_arm(
        tmp_path / "tail6_fixed32_pool0", census_skip={7, 900}
    )
    record = width4.reduce_window_arm(arm, mode="tail6_fixed32", pass_index=0)
    assert record["included"] is False
    assert "counters and census disagree" in record["exclusion_reason"]


def test_the_census_gate_is_exact_equality_with_no_tolerance(tmp_path: Path) -> None:
    """One event out of ~4700 must fail it.

    A tolerance here would let a misaligned window pass while quietly measuring
    the wrong steps, which is the whole failure mode this class exists to avoid.
    """
    arm = build_arm(
        tmp_path / "tail6_fixed32_pool0",
        census_batch_override={11: batch_at(11) + 1},
    )
    record = width4.reduce_window_arm(arm, mode="tail6_fixed32", pass_index=0)
    assert record["included"] is False
    assert "not supported by the engine-side record" in record["exclusion_reason"]


def test_a_census_step_outside_the_window_cannot_affect_the_gate(
    tmp_path: Path,
) -> None:
    """Perturbing a DRAIN step must be invisible: the window never reads it."""
    arm = build_arm(
        tmp_path / "tail6_fixed32_pool0",
        census_batch_override={CLEAN_WINDOW_STEPS + 40: 1},
    )
    record = width4.reduce_window_arm(arm, mode="tail6_fixed32", pass_index=0)
    assert record["included"], record["exclusion_reason"]
    assert record["windowed"]["per_request_step_tps"] == pytest.approx(12.5, rel=1e-12)


def test_the_ledger_summary_is_recomputed_and_must_agree(tmp_path: Path) -> None:
    """The runner's own summary is not trusted on its word -- the house pattern."""
    arm = build_arm(
        tmp_path / "tail6_fixed32_pool0",
        summary_override={"full_width_fraction": 0.99},
    )
    record = width4.reduce_window_arm(arm, mode="tail6_fixed32", pass_index=0)
    assert record["included"] is False
    assert "not supported by its events" in record["exclusion_reason"]


def test_a_pool_no_larger_than_its_slots_has_no_refill_phase(tmp_path: Path) -> None:
    arm = build_arm(tmp_path / "tail6_fixed32_pool0", summary_override={"slots": 8})
    record = width4.reduce_window_arm(arm, mode="tail6_fixed32", pass_index=0)
    assert record["included"] is False
    assert "must be LARGER than its slot count" in record["exclusion_reason"]


# --------------------------------------------------------------------------- #
# 4. the arm-level depth floor is deliberately NOT inherited                    #
# --------------------------------------------------------------------------- #
def test_the_3_2_arm_level_depth_floor_is_not_applied_because_it_gates_the_mixture(
    tmp_path: Path,
) -> None:
    """This is the point of the class.

    The straggler arm's time-weighted mean depth is far below the 3.2 floor that
    excluded pool16 passes 2 and 3 at arm level.  Its full-width phase is still a
    real full-width phase; only its DRAIN is long.  So the depth floor must not
    be what rejects it -- and here it is rejected purely on window length, with
    the depth never consulted.
    """
    assert width4.APPLIES_ARM_LEVEL_DEPTH_FLOOR is False
    events = width4.read_ledger_events(
        build_arm(tmp_path / "tail6_fixed32_pool0", SHORT_WINDOW_TIMELINE)
    )
    occupancy = width4.reduce_ledger_occupancy(events, 4)
    assert occupancy["time_weighted_mean_depth"] < 3.2
    record = width4.reduce_window_arm(
        tmp_path / "tail6_fixed32_pool0", mode="tail6_fixed32", pass_index=0
    )
    assert "depth" not in record["exclusion_reason"]
    assert "admissibility floor" in record["exclusion_reason"]


def test_a_long_window_arm_below_the_arm_level_depth_floor_is_still_admitted(
    tmp_path: Path,
) -> None:
    """A window-length-passing arm is admitted even with a drain-dominated arm.

    Built from the clean timeline with a 9000 s straggler tail bolted on: the
    arm-level depth collapses, the window is untouched.
    """
    timeline = CLEAN_TIMELINE[:-3] + (
        ("complete", "t5", 2, 0, 3600.0, 1500),
        ("complete", "t6", 1, 0, 7200.0, 1750),
        ("complete", "t7", 0, 0, 10800.0, 2000),
    )
    arm = build_arm(tmp_path / "tail6_fixed32_pool0", timeline)
    occupancy = width4.reduce_ledger_occupancy(width4.read_ledger_events(arm), 4)
    assert occupancy["time_weighted_mean_depth"] < 3.2
    record = width4.reduce_window_arm(arm, mode="tail6_fixed32", pass_index=0)
    assert record["included"], record["exclusion_reason"]
    # Unchanged from the clean arm: the drain cannot move the operating point.
    assert record["windowed"]["per_request_step_tps"] == pytest.approx(12.5, rel=1e-12)
    assert record["windowed"]["window_steps"] == CLEAN_WINDOW_STEPS


# --------------------------------------------------------------------------- #
# 5. statistics -- the pinned criticals, and no invented ones                    #
# --------------------------------------------------------------------------- #
def test_four_draws_get_an_interval_and_seven_df_gets_a_named_refusal() -> None:
    four = width4.interval_or_reason([15.0, 16.0, 15.5, 16.5])
    assert four["status"] == "evaluated"
    assert four["df"] == 3
    assert four["t_0_95_one_sided"] == width4.T95_ONE_SIDED[3]
    assert four["l95"] < four["point_estimate"] < four["u95"]

    eight = width4.interval_or_reason([15.0, 16.0, 15.5, 16.5, 15.2, 16.2, 15.7, 16.7])
    assert eight["status"] == "no_pinned_critical"
    assert eight["df"] == 7
    assert "l95" not in eight and "u95" not in eight
    # A point estimate and a CV are still reported -- refusing the interval is
    # not the same as withholding the measurement.
    assert eight["point_estimate"] == pytest.approx(15.85)
    assert eight["cv"] is not None


def test_per_request_step_tps_is_the_primary_statistic_of_this_class() -> None:
    """Inverted back from pool16's aggregate-primary, and for a stated reason:
    the window PINS co-residency instead of measuring it."""
    assert width4.PRIMARY_STATISTIC == "per_request_step_tps"
    assert width4.PRIMARY_STATISTIC in width4.INTERVAL_FIELDS
    assert "pinned-effective-batch" in width4.PRIMARY_STATISTIC_REASON


# --------------------------------------------------------------------------- #
# 6. class round-trip                                                           #
# --------------------------------------------------------------------------- #
def _campaign(tmp_path: Path, passes: int = 4) -> Path:
    root = tmp_path / "gate"
    for index in range(passes):
        for mode in ("tail6_fixed32", "hydra27_fixed32"):
            build_arm(root / f"pass_{index:02d}" / f"{mode}_pool{index}")
    return root


def test_class_round_trip_through_json(tmp_path: Path) -> None:
    root = _campaign(tmp_path)
    records = width4.discover_arms(root)
    assert len(records) == 8
    assert all(r["included"] for r in records)
    payload = width4.build_payload(
        gate_root=root, source_commit="deadbeef", records=records
    )
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    restored = json.loads(text)
    assert restored == payload

    assert restored["schema"] == "fr13.b4_width4_operating_point.v1"
    assert restored["classification"] == "real_swe_verified_b4_width4_operating_point"
    assert restored["run_class"] == "b4_width4_operating_point"
    assert restored["verdict"] == "WINDOWED"
    assert restored["analysis_valid"] is True
    # An instrument never seals anything.
    assert restored["citable"] is False
    assert restored["formal_floor_acceptance_eligible"] is False
    assert restored["b4_cap_applicable"] is False
    assert width4.render(payload)


def test_the_class_tokens_do_not_collide_with_the_neighbouring_classes() -> None:
    reducer = width4.reducer
    others = {spec["schema"] for spec in reducer.RUN_CLASSES.values()}
    labels = {spec["classification"] for spec in reducer.RUN_CLASSES.values()}
    assert width4.SCHEMA not in others
    assert width4.CLASSIFICATION not in labels
    assert width4.RUN_CLASS not in reducer.RUN_CLASSES


def test_this_class_does_not_mutate_the_neighbouring_run_class_registry() -> None:
    """The pool16 and exact4 contracts are sealed evidence.

    A new class must be a SIBLING, never an edit to a registry that sealed
    artifacts already cite.
    """
    assert set(width4.reducer.RUN_CLASSES) == {
        "exact4_formal_floor",
        "pool16_refill_timing",
    }


def test_does_not_claim_names_every_exclusion_this_class_makes(
    tmp_path: Path,
) -> None:
    payload = width4.build_payload(
        gate_root=tmp_path,
        source_commit="",
        records=width4.discover_arms(_campaign(tmp_path)),
    )
    disclaimers = payload["does_not_claim"]
    assert isinstance(disclaimers, list)
    assert all(isinstance(item, str) and item for item in disclaimers)
    joined = " ".join(disclaimers).lower()
    # The four exclusions a reader could otherwise be misled by.
    assert "whole-arm throughput" in joined
    assert "by construction" in joined
    assert "exact4 comparability" in joined
    assert "batch filter" in joined or "per-step engine-batch filter" in joined
    assert "request completes" in joined


def test_the_admissibility_floor_is_reported_with_whether_it_did_any_work(
    tmp_path: Path,
) -> None:
    """A floor that excluded nothing must say so, not imply it was load-bearing."""
    payload = width4.build_payload(
        gate_root=tmp_path,
        source_commit="",
        records=width4.discover_arms(_campaign(tmp_path)),
    )
    admissibility = payload["admissibility"]
    assert admissibility["min_window_steps"] == width4.MIN_WINDOW_STEPS
    assert admissibility["observed_min_window_steps"] == CLEAN_WINDOW_STEPS
    assert admissibility["floor_did_work"] is False
    assert admissibility["arm_level_depth_floor_applied"] is False


def test_a_single_defective_arm_costs_the_campaign_its_verdict(tmp_path: Path) -> None:
    root = _campaign(tmp_path)
    build_arm(
        root / "pass_02" / "hydra27_fixed32_pool2",
        NO_WIDTH_TIMELINE,
        total_steps=NO_WIDTH_TOTAL_STEPS,
    )
    payload = width4.build_payload(
        gate_root=root, source_commit="", records=width4.discover_arms(root)
    )
    assert payload["analysis_valid"] is False
    assert payload["verdict"] == "NOT_EVALUATED_INSUFFICIENT_WINDOWS"
    excluded = [a for a in payload["arms"] if not a["included"]]
    assert len(excluded) == 1
    assert "never reached its full width" in excluded[0]["exclusion_reason"]


def test_the_cli_writes_the_artifact_atomically_and_reports_its_verdict(
    tmp_path: Path,
) -> None:
    root = _campaign(tmp_path)
    out = tmp_path / "nested" / "width4.json"
    code = width4.main(
        ["--gate-root", str(root), "--source-commit", "abc123", "--out", str(out)]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source_commit"] == "abc123"
    assert payload["verdict"] == "WINDOWED"
    # No temp file survives a successful write.
    assert not (out.with_name(out.name + ".tmp")).exists()
