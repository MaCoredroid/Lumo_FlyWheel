"""The c5 seam gate, validated against the banked 100-arm corpus.

The corridor [0.40, 0.70] was pre-registered from that sweep before the module
existed; these tests re-derive the sweep independently and check that the gate
reproduces it, flags the four task-aggregate detections, and adds the fifth
only where the doc says it needs within-task resolution.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr14_c5_seam_gate as gate  # noqa: E402

# The banked accept ladders from the exact16 serve, by flush generation.
# gen 2 / 4 / 6 close tasks 1 / 2 / 3; task 3 is the enumeration degeneration.
EXACT16_LADDERS = {
    1: [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    2: [53, 152, 154, 136, 97, 197, 57, 36, 24, 18, 12, 119, 0, 0, 0, 0],
    4: [236, 734, 765, 598, 424, 768, 210, 145, 85, 65, 64, 710, 0, 0, 0, 0],
    6: [410, 1029, 1583, 1495, 1234, 2798, 398, 376, 319, 171, 101, 1006, 0, 0, 0, 0],
}

# The five corpus degenerations (C8), and the class each one is.
KNOWN_DEGENERATIONS = {
    ("fr14_promoab_Gp5_20260818T174541Z", "astropy__astropy-13236"): "low",
    ("fr14_promoab_Cqc16_20260819T222438Z", "astropy__astropy-13236"): "low",
    ("fr14_hydra27_lever_pair_20260817T130251Z", "astropy__astropy-13033"): "high",
    ("fr14_promoab_Ch27_20260819T064150Z", "astropy__astropy-13236"): "high",
    # the fifth: inside the corridor at task-aggregate resolution, by 0.017
    ("fr14_promoab_Ch27_20260819T064150Z", "astropy__astropy-13033"): "windowed-only",
}


def _corpus() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for runroot in sorted(glob.glob(str(REPO / "output" / "fr14_*"))):
        for row in gate.sweep_runroot(Path(runroot)):
            rows.append((Path(runroot).name, row))
    return rows


def _corpus_or_skip() -> list[tuple[str, dict]]:
    rows = _corpus()
    if len(rows) < 50:
        pytest.skip(f"banked corpus unavailable ({len(rows)} arms)")
    return rows


def test_the_corridor_is_the_pre_registered_one() -> None:
    """Changing it is a re-registration, not an edit."""
    assert gate.C5_CORRIDOR_LOW == 0.40
    assert gate.C5_CORRIDOR_HIGH == 0.70
    assert gate.C5_NUMERATOR_POSITION == 5, "position 5 is the MTP/Arctic seam"
    assert gate.C5_DENOMINATOR_POSITION == 4


def test_the_sweep_reproduces_the_hundred_arm_corpus() -> None:
    rows = _corpus_or_skip()
    usable = [row for _run, row in rows if row["c5"] is not None]
    assert len(usable) >= 95, f"only {len(usable)} usable task-arms"
    values = sorted(row["c5"] for row in usable)
    assert 0.27 <= values[0] <= 0.28, f"low end moved: {values[0]}"
    assert 0.89 <= values[-1] <= 0.90, f"high end moved: {values[-1]}"
    median = values[len(values) // 2]
    assert 0.54 <= median <= 0.55, f"median moved: {median}"


def test_every_flag_is_a_known_degeneration_and_no_healthy_arm_flags() -> None:
    """Zero false positives on 95 healthy arms -- the property that makes this
    worth emitting at all."""
    rows = _corpus_or_skip()
    flagged = {
        (run, row["label"]): row
        for run, row in rows
        if row["c5"] is not None and row["verdict"] != "in-corridor"
    }
    assert len(flagged) == 4, f"expected 4 task-aggregate detections, got {sorted(flagged)}"
    for key, row in flagged.items():
        assert key in KNOWN_DEGENERATIONS, f"FALSE POSITIVE on a healthy arm: {key}"
        expected = KNOWN_DEGENERATIONS[key]
        assert expected in row["verdict"], (
            f"{key} classified {row['verdict']}, corpus says {expected}"
        )


def test_the_fifth_degeneration_hides_at_task_aggregate_resolution() -> None:
    """Stated, not glossed: the corridor's high side is thin.

    13033/Ch27 reads 0.6855 -- inside [0.40, 0.70] by 0.0145. The gate does NOT
    catch it per-task, which is exactly why the windowed variant exists.
    """
    rows = _corpus_or_skip()
    match = [
        row
        for run, row in rows
        if run == "fr14_promoab_Ch27_20260819T064150Z"
        and row["label"] == "astropy__astropy-13033"
    ]
    if not match:
        pytest.skip("that runroot is not banked here")
    c5 = match[0]["c5"]
    assert 0.68 <= c5 <= 0.69
    assert match[0]["verdict"] == "in-corridor"
    assert gate.C5_CORRIDOR_HIGH - c5 < 0.02, "the high margin is thin by design"


# --- the windowed (F4) variant ---------------------------------------------
def test_the_ladder_reconstructs_the_stock_per_position_counter() -> None:
    """The identity the windowed variant rests on: density -> survival."""
    ladder = EXACT16_LADDERS[6]
    per_pos = gate.per_position_from_ladder(ladder)
    for index in range(len(ladder)):
        assert per_pos[index] == sum(ladder[index + 1 :])
    # against the scraped counter of the same run (positions 0..5 carry the
    # +26/+17/+10/+5/+2/+1 truncated-emission residual; 6+ are exact)
    scraped = {0: 10484, 1: 9464, 2: 7888, 3: 6398, 4: 5167, 5: 2370,
               6: 1973, 7: 1597, 8: 1278, 9: 1107, 10: 1006, 11: 0}
    residual = [per_pos[i] - scraped[i] for i in range(12)]
    assert residual == [26, 17, 10, 5, 2, 1, 0, 0, 0, 0, 0, 0]
    assert sum(residual) == 61, "the same +61 the token total carried"


def test_the_windowed_variant_catches_the_degenerating_task() -> None:
    """F4: within-task resolution, free -- no extra scrape.

    The ladder is already drained at every flush boundary, so consecutive
    sidecars give a window per task.
    """
    windows = []
    generations = [1, 2, 4, 6]
    for start, end in zip(generations, generations[1:]):
        windows.append(
            gate.c5_from_ladders(
                EXACT16_LADDERS[start],
                EXACT16_LADDERS[end],
                label=f"gen{start}->gen{end}",
            )
        )
    assert windows[0]["verdict"] == "in-corridor"
    assert windows[1]["verdict"] == "in-corridor"
    # task 3 is the enumeration degeneration
    assert windows[2]["verdict"] == "DEGENERATION-SHAPE:low(cache-lost-thread/enumeration)"
    # ...and it agrees with the scraped task-aggregate to four decimals
    assert abs(windows[2]["c5"] - 0.3499) < 0.0002
    assert windows[2]["source"] == "accept_ladder_window"


def test_aggregating_the_whole_run_hides_the_degeneration() -> None:
    """Why the window is not optional.

    Cumulated over the run, the same data reads 0.4587 -- comfortably inside
    the corridor. A run-level c5 would have reported this serve as healthy.
    """
    whole = gate.c5_from_ladders(
        EXACT16_LADDERS[1], EXACT16_LADDERS[6], label="whole-run"
    )
    assert whole["verdict"] == "in-corridor"
    assert 0.45 <= whole["c5"] <= 0.47


# --- it flags, it does not refuse ------------------------------------------
def test_the_gate_flags_and_never_refuses() -> None:
    """A non-zero exit here would make this a refusal by the back door."""
    runroot = REPO / "output" / "fr14_promoab_Cqc16_20260819T222438Z"
    if not runroot.exists():
        pytest.skip("exact16 runroot not banked here")
    rows = gate.sweep_runroot(runroot)
    assert any(row["verdict"] != "in-corridor" for row in rows), (
        "this runroot must produce a flag, or the test proves nothing"
    )
    assert gate.main([str(runroot)]) == 0, "the gate must exit 0 even when flagging"


def test_an_empty_window_reports_no_signal_not_zero() -> None:
    """A window with no position-4 acceptance carries no seam information.

    Reporting 0.0 would invent a DEGENERATION-SHAPE:low out of an absence --
    the measured-zero-versus-absent mistake this campaign keeps paying for.
    """
    flat = [0] * 16
    result = gate.c5_from_ladders(flat, flat, label="empty")
    assert result["c5"] is None
    assert result["verdict"] == "no-signal:no-denominator"
    assert result["verdict"] != gate.classify(0.0)


def test_classification_boundaries_are_inclusive_of_the_corridor() -> None:
    assert gate.classify(0.40) == "in-corridor"
    assert gate.classify(0.70) == "in-corridor"
    assert gate.classify(0.3999).startswith("DEGENERATION-SHAPE:low")
    assert gate.classify(0.7001).startswith("DEGENERATION-SHAPE:high")


def test_a_scrape_without_the_counter_is_an_error_not_a_verdict() -> None:
    with pytest.raises(gate.C5Error):
        gate.parse_per_position("# nothing useful here\n")


# --- E-A's audit (pass 162): the tiny-denominator artifact ------------------
# c5 is a ratio of counts. On trivially short tasks the denominator is small
# enough that a perfectly healthy arm reads below the corridor by chance: E-A
# found two CLEAN arms with 8-11 requests reading 0.3627 and 0.3876. The gate
# now refuses to give a corridor verdict below a derived minimum.
def test_the_minimum_denominator_is_derived_not_picked() -> None:
    """150 is where a median-healthy arm stops reaching the floor at 3 sigma."""
    import math

    median = 0.5425
    se = lambda n: math.sqrt(median * (1 - median) / n)
    assert median - 3 * se(100) < gate.C5_CORRIDOR_LOW, "at n=100 chance reaches the floor"
    assert median - 3 * se(gate.C5_MIN_DENOMINATOR) > gate.C5_CORRIDOR_LOW, (
        "at the threshold it must not"
    )
    assert gate.C5_MIN_DENOMINATOR == 150


def test_every_port_arm_keeps_its_verdict_under_the_threshold() -> None:
    """The threshold must not cost the calibration population its signal."""
    rows = _corpus_or_skip()
    usable = [row for _run, row in rows if row["c5"] is not None]
    assert len(usable) >= 95, f"the threshold silenced arms: {len(usable)}"
    smallest = min(row["delta_pos4"] for row in usable)
    assert smallest >= gate.C5_MIN_DENOMINATOR, (
        f"an arm below the threshold kept a verdict: d4={smallest}"
    )
    assert smallest == 215, (
        "the port's minimum denominator moved; re-check the threshold margin"
    )


@pytest.mark.parametrize("c5_value", [0.3627, 0.3876])
def test_the_EA_artifact_readings_go_no_signal_on_a_short_arm(c5_value: float) -> None:
    """MUTATION PROOF: E-A's two below-corridor readings, on a short arm.

    Reconstructed at their reported ratios with a denominator typical of an
    8-11 request task. Both would have been flagged DEGENERATION-SHAPE before;
    both are now no-signal.
    """
    short_denominator = 60
    assert short_denominator < gate.C5_MIN_DENOMINATOR
    result = gate._c5(
        short_denominator, round(short_denominator * c5_value), label="short-clean"
    )
    assert result["c5"] is None
    assert result["verdict"] == "no-signal:insufficient-denominator"
    assert result["delta_pos4"] == short_denominator, "the evidence is still emitted"
    assert result["min_denominator"] == gate.C5_MIN_DENOMINATOR


@pytest.mark.parametrize("c5_value", [0.3627, 0.3876])
def test_the_same_ratio_on_a_full_length_task_keeps_its_verdict(
    c5_value: float,
) -> None:
    """The threshold acts on the DENOMINATOR, not on the value.

    Same ratio, a full-length task's denominator: still flagged. Otherwise the
    hardening would be a way to silence real detections.
    """
    long_denominator = 600
    result = gate._c5(
        long_denominator, round(long_denominator * c5_value), label="long-degenerate"
    )
    assert result["c5"] is not None
    assert result["verdict"].startswith("DEGENERATION-SHAPE:low")


def test_the_real_degenerations_survive_the_threshold() -> None:
    """The four task-aggregate detections must not be silenced by hardening."""
    rows = _corpus_or_skip()
    flagged = [
        row
        for _run, row in rows
        if row["c5"] is not None and row["verdict"] != "in-corridor"
    ]
    assert len(flagged) == 4, f"hardening changed the detections: {len(flagged)}"
    for row in flagged:
        assert row["delta_pos4"] >= gate.C5_MIN_DENOMINATOR


def test_the_corridor_is_UNCHANGED_on_exclusive_bracket_data() -> None:
    """E-A asked whether [0.40, 0.70] moves. On the port's 100 exclusive arms it
    does not: healthy span [0.4517, 0.6688], the same numbers the corridor was
    pre-registered from. Recorded as an assertion so a future re-derivation is
    a deliberate, versioned re-registration.

    NOT a full answer: the 229 exclusive PRE-PORT arms are not in this tree, so
    the bounds may still move on the wider population. Reported, not retuned.
    """
    rows = _corpus_or_skip()
    healthy = [
        row
        for run, row in rows
        if row["c5"] is not None
        and (run, row["label"]) not in KNOWN_DEGENERATIONS
    ]
    values = sorted(row["c5"] for row in healthy)
    assert 0.451 <= values[0] <= 0.452, f"healthy floor moved: {values[0]}"
    assert 0.668 <= values[-1] <= 0.669, f"healthy ceiling moved: {values[-1]}"
    assert values[0] > gate.C5_CORRIDOR_LOW
    assert values[-1] < gate.C5_CORRIDOR_HIGH
