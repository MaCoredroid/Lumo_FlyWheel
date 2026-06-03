from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr10_equivalence_gate import (
    GateThresholds,
    NegativeControl,
    StateParityRow,
    TokenRecord,
    compare_records,
    default_fr10_negative_controls,
    evaluate_accumulation,
    evaluate_flip_distribution_equivalence,
    evaluate_flip_margins,
    evaluate_negative_controls,
    evaluate_state_parity,
    evaluate_three_way_gate,
    first_token_diff,
    load_margin_artifact,
    load_token_artifact,
)


def _record(batch: str, choice: int, tokens: list[int]) -> TokenRecord:
    return TokenRecord(
        batch=batch,
        choice_index=choice,
        prompt=f"prompt-{choice}",
        token_ids=tuple(tokens),
    )


def test_first_token_diff_reports_token_and_length_changes() -> None:
    assert first_token_diff([1, 2, 3], [1, 4, 3]) == 1
    assert first_token_diff([1, 2], [1, 2, 3]) == 2
    assert first_token_diff([1, 2], [1, 2]) is None


def test_committed_ar_vs_p0_artifact_records_batch_shape_gap() -> None:
    non_mtp = load_token_artifact("output/fr10_cu130_non_mtp_ar_greedy/greedy_tokens.json")
    naive_mtp = load_token_artifact(
        "output/fr10_p0_cu130_boot_batchinv/fr10_cu130_p0_s1_batchinv_greedy_tokens.json"
    )

    all_flips = compare_records(non_mtp, naive_mtp)
    b1_flips = compare_records(non_mtp, naive_mtp, batches={"b1"})
    b4_flips = compare_records(non_mtp, naive_mtp, batches={"b4"})

    assert len(b1_flips) == 0
    assert len(b4_flips) == 1
    assert len(all_flips) == 1
    assert b4_flips[0].choice_index == 0
    assert b4_flips[0].position == 2


def test_committed_ar_vs_p0_calibration_flip_margin_is_enforced() -> None:
    non_mtp = load_token_artifact("output/fr10_cu130_non_mtp_ar_greedy/greedy_tokens.json")
    naive_mtp = load_token_artifact(
        "output/fr10_p0_cu130_boot_batchinv/fr10_cu130_p0_s1_batchinv_greedy_tokens.json"
    )
    margins = load_margin_artifact(
        "output/fr10_cu130_non_mtp_ar_greedy/ar_vs_p0_mtp5_flip_margins.json"
    )
    flips = compare_records(non_mtp, naive_mtp, batches={"b4"})
    flips = [
        type(flip)(
            **{
                **flip.__dict__,
                "margin": margins[flip.key][flip.position],
            }
        )
        for flip in flips
    ]

    assert len(flips) == 1
    assert flips[0].choice_index == 0
    assert flips[0].position == 2
    report = evaluate_flip_margins(flips)
    assert not report.passed
    assert report.metrics["max_margin"] == pytest.approx(0.25)
    assert report.metrics["max_margin"] > GateThresholds().margin_indifference
    assert any("high-margin flip" in violation for violation in report.violations)


def test_flip_margin_gate_rejects_high_margin_flip() -> None:
    flips = [
        compare_records(
            {("b1", 0): _record("b1", 0, [1, 2, 3])},
            {("b1", 0): _record("b1", 0, [1, 9, 3])},
        )[0]
    ]
    flips = [
        type(flips[0])(
            **{
                **flips[0].__dict__,
                "margin": 1e-2,
            }
        )
    ]

    report = evaluate_flip_margins(flips, thresholds=GateThresholds(margin_indifference=6e-5))

    assert not report.passed
    assert any("high-margin flip" in violation for violation in report.violations)


def test_flip_distribution_gate_detects_structured_extra_tree_flips() -> None:
    baseline = [
        compare_records(
            {("b1", 0): _record("b1", 0, [1, 2, 3])},
            {("b1", 0): _record("b1", 0, [1, 9, 3])},
        )[0]
    ]
    tree = []
    for idx in range(6):
        tree.extend(
            compare_records(
                {("b1", idx): _record("b1", idx, [1, 2, 3, 4])},
                {("b1", idx): _record("b1", idx, [1, 2, 3, 9])},
            )
        )

    report = evaluate_flip_distribution_equivalence(
        tree_flips=tree,
        baseline_flips=baseline,
        total_records=16,
        thresholds=GateThresholds(flip_rate_slack=1 / 16, position_tv_slack=0.25),
    )

    assert not report.passed
    assert any("flip rate" in violation or "position TV" in violation for violation in report.violations)


def test_state_parity_gate_is_contamination_microscope() -> None:
    clean = [
        StateParityRow(layer=0, node=0, max_state_abs=7e-6, max_output_abs=7e-9),
        StateParityRow(layer=1, node=2, max_state_abs=8e-6, max_output_abs=8e-9),
    ]
    contaminated = clean + [
        StateParityRow(layer=4, node=6, max_state_abs=0.56, max_output_abs=0.56)
    ]

    assert evaluate_state_parity(clean).passed
    report = evaluate_state_parity(contaminated)
    assert not report.passed
    assert any("state parity fail" in violation for violation in report.violations)


def test_accumulation_gate_detects_position_drift() -> None:
    rows = [
        StateParityRow(layer=0, node=0, position=pos, max_state_abs=pos * 1e-4)
        for pos in range(10)
    ]

    report = evaluate_accumulation(rows, thresholds=GateThresholds(accumulation_slope_abs=1e-6))

    assert not report.passed
    assert any("accumulates" in violation for violation in report.violations)


def test_required_negative_controls_fail_loudly() -> None:
    report = evaluate_negative_controls(default_fr10_negative_controls())

    assert report.passed


def test_negative_control_gate_rejects_too_loose_detector() -> None:
    report = evaluate_negative_controls([
        NegativeControl("too-small-leak", observed_delta=1e-7, must_exceed=2e-5)
    ])

    assert not report.passed
    assert any("did not fail loudly" in violation for violation in report.violations)


def test_three_way_gate_passes_when_tree_is_no_worse_than_baseline_and_state_clean() -> None:
    non_mtp = {
        ("b1", 0): _record("b1", 0, [1, 2, 3]),
        ("b1", 1): _record("b1", 1, [4, 5, 6]),
    }
    naive_mtp = {
        ("b1", 0): _record("b1", 0, [1, 9, 3]),
        ("b1", 1): _record("b1", 1, [4, 5, 6]),
    }
    tree_mtp = {
        ("b1", 0): _record("b1", 0, [1, 9, 3]),
        ("b1", 1): _record("b1", 1, [4, 5, 6]),
    }

    report = evaluate_three_way_gate(
        non_mtp=non_mtp,
        naive_mtp=naive_mtp,
        tree_mtp=tree_mtp,
        batches={"b1"},
        baseline_flip_margins={("b1", 0): {1: 1e-6}},
        tree_flip_margins={("b1", 0): {1: 1e-6}},
        state_rows=[StateParityRow(layer=0, node=0, max_state_abs=7e-6, max_output_abs=7e-9)],
        negative_controls=default_fr10_negative_controls(),
    )

    assert report.passed
    assert report.metrics["summary"]["baseline_flips_vs_non_mtp"] == 1
    assert report.metrics["summary"]["tree_flips_vs_non_mtp"] == 1


def test_three_way_gate_rejects_tree_worse_than_baseline() -> None:
    non_mtp = {
        ("b1", 0): _record("b1", 0, [1, 2, 3]),
        ("b1", 1): _record("b1", 1, [4, 5, 6]),
        ("b1", 2): _record("b1", 2, [7, 8, 9]),
    }
    naive_mtp = {
        ("b1", 0): _record("b1", 0, [1, 2, 3]),
        ("b1", 1): _record("b1", 1, [4, 5, 6]),
        ("b1", 2): _record("b1", 2, [7, 8, 9]),
    }
    tree_mtp = {
        ("b1", 0): _record("b1", 0, [1, 2, 0]),
        ("b1", 1): _record("b1", 1, [4, 0, 6]),
        ("b1", 2): _record("b1", 2, [0, 8, 9]),
    }

    report = evaluate_three_way_gate(
        non_mtp=non_mtp,
        naive_mtp=naive_mtp,
        tree_mtp=tree_mtp,
        batches={"b1"},
        tree_flip_margins={
            ("b1", 0): {2: 1e-6},
            ("b1", 1): {1: 1e-6},
            ("b1", 2): {0: 1e-6},
        },
        state_rows=[StateParityRow(layer=0, node=0, max_state_abs=7e-6, max_output_abs=7e-9)],
        negative_controls=default_fr10_negative_controls(),
    )

    assert not report.passed
    assert any("tree flip rate" in violation for violation in report.violations)
