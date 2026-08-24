"""The c5 seam gate, validated against the banked 100-arm corpus.

The corridor [0.40, 0.70] was pre-registered from that sweep before the module
existed; these tests re-derive the sweep independently and check that the gate
reproduces it, flags the four task-aggregate detections, and adds the fifth
only where the doc says it needs within-task resolution.
"""

from __future__ import annotations

import glob
import shutil
import tempfile
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


# THE CALIBRATION SET AND THE BANK ARE DIFFERENT POPULATIONS, and conflating
# them was half of this instrument's own defect. Everything that DERIVES a
# corridor statistic reads the pinned set; everything that CHECKS an arm against
# the corridor reads the whole bank. A new arm can therefore be flagged but can
# never move the thing it is being flagged against.
def _calibration() -> list[tuple[str, dict]]:
    rows = gate.calibration_corpus(REPO / "output")
    if len(rows) < 50:
        pytest.skip(f"pinned calibration corpus unavailable ({len(rows)} arms)")
    return rows


def _all_arms(*, include_inapplicable: bool = False) -> list[tuple[str, dict]]:
    """Every FR14 arm on disk, applicability-aware."""
    return gate.sweep_output_root(
        REPO / "output", include_inapplicable=include_inapplicable
    )


def test_the_corridor_is_the_pre_registered_one() -> None:
    """Changing it is a re-registration, not an edit."""
    assert gate.C5_CORRIDOR_LOW == 0.40
    assert gate.C5_CORRIDOR_HIGH == 0.70
    assert gate.C5_NUMERATOR_POSITION == 5, "position 5 is the MTP/Arctic seam"
    assert gate.C5_DENOMINATOR_POSITION == 4


def test_the_sweep_reproduces_the_pinned_calibration_corpus() -> None:
    rows = _calibration()
    assert len(rows) == gate.C5_CALIBRATION_ROWS
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
    rows = _calibration()
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
    rows = _calibration()
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
    rows = _calibration()
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
    rows = _calibration()
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
    rows = _calibration()
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


# --------------------------------------------------------------------------- #
# APPLICABILITY: a drafter with no seam has no c5, not a low one               #
# --------------------------------------------------------------------------- #
# THE DEFECT THIS CLOSES. The sweep globbed every fr14_* arm and computed the
# seam conditional over all of them, including four MTP-5 chain-drafter arms
# where position 5 does not exist. delta_pos5 is structurally 0 there, so each
# read c5 = 0.0000 -- four manufactured floors below every real degeneration in
# the bank, ranking the healthiest arms as its worst. A statistic calibrated on
# one topology and applied blindly to another reads the healthy case as the
# pathological one.
def _write_runroot(root: Path, *, boot: str | None, probe: str | None = None) -> Path:
    """A runroot with one bracketed task and whatever evidence is named."""
    arm = root / "arm"
    task = arm / "swe_out" / "verified" / "per_task" / "astropy__astropy-13236"
    task.mkdir(parents=True, exist_ok=True)
    def scrape(pos4: int, pos5: int) -> str:
        rows = [
            'vllm:spec_decode_num_accepted_tokens_per_pos_total'
            f'{{engine="0",position="{index}"}} {value}'
            for index, value in ((4, pos4), (5, pos5))
        ]
        return "\n".join(rows) + "\n"
    (task / "vllm_metrics_pre.txt").write_text(scrape(0, 0), encoding="utf-8")
    (task / "vllm_metrics_post.txt").write_text(scrape(1000, 550), encoding="utf-8")
    if boot is not None:
        (arm / "boot_log_snapshot.txt").write_text(boot, encoding="utf-8")
    if probe is not None:
        (root / "MTP5_PROBE.txt").write_text(probe, encoding="utf-8")
    return root


def test_a_chain_drafter_arm_contributes_nothing_and_flags_nothing(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """MUTATION PROOF. num_spec_tokens=5 -> positions 0..4 -> no seam.

    The arm is present, bracketed, and perfectly healthy. It must contribute NO
    row at all -- not a zero, which is what made it look like the worst
    degeneration in the corpus.
    """
    root = _write_runroot(
        Path(tempfile.mkdtemp(prefix="c5-chain-")),
        boot="EngineArgs(... num_spec_tokens=5, method='mtp' ...)\n",
    )
    try:
        applicability = gate.seam_applicability(root)
        assert applicability["applicable"] is False
        assert "no-seam:num_spec_tokens=5" in applicability["reason"]
        assert gate.sweep_runroot(root) == []
        declared = gate.sweep_runroot(root, include_inapplicable=True)
        assert len(declared) == 1
        assert declared[0]["c5"] is None
        assert declared[0]["verdict"].startswith("not-applicable:")
        assert declared[0]["seam_applicable"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_arms_own_declaration_outranks_the_inference() -> None:
    """The MTP-5 probes stamp c5_applicable=NO. A declaration is evidence."""
    root = Path(tempfile.mkdtemp(prefix="c5-declared-"))
    try:
        _write_runroot(
            root,
            boot="num_spec_tokens=31\n",
            probe="c5_applicable=NO -- c5 is a SEAM conditional and a chain "
            "drafter has no seam\n",
        )
        applicability = gate.seam_applicability(root)
        assert applicability["applicable"] is False
        assert applicability["reason"] == "declared:c5_applicable=NO"
        assert gate.sweep_runroot(root) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_tree_drafter_arm_outside_the_corridor_still_flags() -> None:
    """MUTATION PROOF, the other direction: the exclusion must not silence."""
    root = Path(tempfile.mkdtemp(prefix="c5-tree-"))
    try:
        arm = root / "arm"
        task = arm / "per_task" / "astropy__astropy-13236"
        task.mkdir(parents=True)
        def scrape(pos4: int, pos5: int) -> str:
            return (
                'vllm:spec_decode_num_accepted_tokens_per_pos_total'
                f'{{engine="0",position="4"}} {pos4}\n'
                'vllm:spec_decode_num_accepted_tokens_per_pos_total'
                f'{{engine="0",position="5"}} {pos5}\n'
            )
        (task / "vllm_metrics_pre.txt").write_text(scrape(0, 0), encoding="utf-8")
        # 900/1000 = 0.90, well above the corridor: a cache-driving loop shape.
        (task / "vllm_metrics_post.txt").write_text(
            scrape(1000, 900), encoding="utf-8"
        )
        (arm / "boot_log_snapshot.txt").write_text(
            "num_spec_tokens=31\n", encoding="utf-8"
        )
        applicability = gate.seam_applicability(root)
        assert applicability["applicable"] is True
        rows = gate.sweep_runroot(root)
        assert len(rows) == 1
        assert rows[0]["c5"] == pytest.approx(0.90)
        assert rows[0]["verdict"].startswith("DEGENERATION-SHAPE:high")
        assert rows[0]["seam_applicable"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_an_arm_whose_drafter_cannot_be_identified_contributes_nothing() -> None:
    """UNDETERMINED IS NOT APPLICABLE. Admitting it silently is how this began."""
    root = Path(tempfile.mkdtemp(prefix="c5-unknown-"))
    try:
        _write_runroot(root, boot=None)
        applicability = gate.seam_applicability(root)
        assert applicability["applicable"] is False
        assert applicability["reason"].startswith("undetermined:")
        assert gate.sweep_runroot(root) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_two_drafter_widths_in_one_runroot_is_ambiguous_not_averaged() -> None:
    root = Path(tempfile.mkdtemp(prefix="c5-ambiguous-"))
    try:
        _write_runroot(root, boot="num_spec_tokens=31\nnum_spec_tokens=5\n")
        applicability = gate.seam_applicability(root)
        assert applicability["applicable"] is False
        assert "ambiguous:multiple-drafter-widths=[5, 31]" in applicability["reason"]
        assert gate.sweep_runroot(root) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_banked_chain_arms_are_excluded_from_the_live_bank() -> None:
    """The four real MTP-5 arms, on disk, contribute nothing to any sweep."""
    excluded = [
        (run, row)
        for run, row in _all_arms(include_inapplicable=True)
        if row["c5"] is None and row["verdict"].startswith("not-applicable:")
    ]
    assert len(excluded) == 4, f"expected the four MTP-5 arms, got {excluded}"
    assert all("mtp5" in run for run, _row in excluded)
    # ...and not one of them appears in an ordinary sweep
    assert all(row["c5"] is not None for _run, row in _all_arms())
    # nor does any arm anywhere read the manufactured 0.0
    assert all(row["c5"] != 0.0 for _run, row in _all_arms())


# --------------------------------------------------------------------------- #
# THE CALIBRATION SET IS PINNED, SO ABSORPTION IS LOUD                         #
# --------------------------------------------------------------------------- #
def test_the_calibration_digest_is_pinned() -> None:
    """MUTATION PROOF. A corridor that drifts with every run is not
    pre-registered.

    Cqc10's ten legitimate rows shifted aggregates that predate them simply by
    existing. The calibration set is now an enumerated, digest-pinned
    population: an arm entering it, or one of its values moving, changes this
    digest and fails here rather than silently re-deriving the corridor.
    """
    rows = _calibration()
    assert len(rows) == gate.C5_CALIBRATION_ROWS == 99
    assert gate.calibration_digest(rows) == gate.C5_CALIBRATION_SHA256
    assert len(gate.C5_CALIBRATION_RUNROOTS) == 47
    assert len(set(gate.C5_CALIBRATION_RUNROOTS)) == 47


def test_new_arms_are_checked_against_the_corridor_not_absorbed_into_it() -> None:
    """The bank is strictly larger than the calibration set, and stays outside."""
    pinned = set(gate.C5_CALIBRATION_RUNROOTS)
    live = {run for run, _row in _all_arms()}
    assert len(live - pinned) >= 1, "no post-registration arms on disk to check"
    # the arms that post-date the pre-registration are absent from it
    for late in ("fr14_promoab_Cqc10_20260824T074813Z",):
        assert late in live, f"{late} is not being checked at all"
        assert late not in pinned, f"{late} was absorbed into the calibration set"
    # and adding them does not move a single calibration number
    assert gate.calibration_digest(_calibration()) == gate.C5_CALIBRATION_SHA256


def test_absorbing_an_arm_into_the_calibration_set_is_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proof the digest can actually fail -- a pin that cannot is not a pin."""
    monkeypatch.setattr(
        gate,
        "C5_CALIBRATION_RUNROOTS",
        gate.C5_CALIBRATION_RUNROOTS + ("fr14_promoab_Cqc10_20260824T074813Z",),
    )
    widened = gate.calibration_corpus(REPO / "output")
    assert len(widened) > gate.C5_CALIBRATION_ROWS
    assert gate.calibration_digest(widened) != gate.C5_CALIBRATION_SHA256


def test_the_pinned_set_carries_every_known_degeneration() -> None:
    """The corridor's evidence must still contain what it was derived to catch."""
    rows = _calibration()
    present = {(run, row["label"]) for run, row in rows}
    for key in KNOWN_DEGENERATIONS:
        assert key in present, f"the calibration set lost {key}"
