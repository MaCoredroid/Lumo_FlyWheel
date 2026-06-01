from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_spec_teacher_forced.py"
spec = importlib.util.spec_from_file_location("measure_spec_teacher_forced", SCRIPT)
measure = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(measure)


def test_draft_top1_chain_e5() -> None:
    row = {"draft": [[10, 11, 12, 13, 14]]}

    assert measure.draft_top1_chain(row, "e5") == [10, 11, 12, 13, 14]


def test_draft_top1_chain_tree_spine_a_even_nodes() -> None:
    row = {"draft": [[10, 20, 11, 21, 12, 22, 13, 23, 14, 24]]}

    assert measure.draft_top1_chain(row, "tree") == [10, 11, 12, 13, 14]


def _measurement(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "reference_sha256": "abc",
        "start_token": 16,
        "depth": 5,
        "rows": rows,
    }


def test_forced_acceptance_uses_live_measurement_lcp() -> None:
    rows = [
        {"forced_position": 16, "draft_top1": [10, 11, 99, 13, 14], "teacher_forced_accept_lcp": 2},
        {"forced_position": 17, "draft_top1": [11, 12, 13, 14, 15], "teacher_forced_accept_lcp": 5},
        {"forced_position": 18, "draft_top1": [12, 13, 14, 15, 16], "teacher_forced_accept_lcp": 4},
    ]

    assert measure.forced_acceptance_lcps(rows, depth=5) == [2, 5, 4]

    summary = measure.forced_acceptance_summary(_measurement(rows))

    assert summary["available"] is True
    assert summary["avg"] == 11 / 3
    assert summary["per_position"] == [1.0, 1.0, 2 / 3, 2 / 3, 1 / 3]


def test_compare_verdict_divergence_only_when_draft_and_target_match() -> None:
    rows = [
        {
            "forced_position": 16,
            "draft_top1": [1, 2, 3, 4, 5],
            "target_argmax_token_id": 1,
        }
    ]

    summary = measure.compare_measurements(_measurement(rows), _measurement(rows))

    assert summary["verdict"] == "divergence-only"
    assert summary["draft_mismatches"] == 0
    assert summary["target_mismatches"] == 0
    assert summary["e5_forced_acceptance"]["available"] is False
    assert summary["tree_forced_acceptance"]["available"] is False


def test_compare_verdict_proposer_bug_when_only_draft_differs() -> None:
    e5 = _measurement(
        [{"forced_position": 16, "draft_top1": [1, 2, 3, 4, 5], "target_argmax_token_id": 1}]
    )
    tree = _measurement(
        [{"forced_position": 16, "draft_top1": [9, 2, 3, 4, 5], "target_argmax_token_id": 1}]
    )

    summary = measure.compare_measurements(e5, tree)

    assert summary["verdict"] == "proposer-bug"
    assert summary["draft_mismatches"] == 1
    assert summary["target_mismatches"] == 0


def test_compare_verdict_target_fundamental_when_argmax_differs() -> None:
    e5 = _measurement(
        [{"forced_position": 16, "draft_top1": [1, 2, 3, 4, 5], "target_argmax_token_id": 1}]
    )
    tree = _measurement(
        [{"forced_position": 16, "draft_top1": [9, 2, 3, 4, 5], "target_argmax_token_id": 8}]
    )

    summary = measure.compare_measurements(e5, tree)

    assert summary["verdict"] == "target-fundamental"
    assert summary["draft_mismatches"] == 1
    assert summary["target_mismatches"] == 1


def test_compare_forced_acceptance_uses_paired_row_limit() -> None:
    e5 = _measurement(
        [
            {
                "forced_position": 16,
                "draft_top1": [1, 2, 3, 4, 5],
                "target_argmax_token_id": 1,
                "teacher_forced_accept_lcp": 5,
            },
            {
                "forced_position": 17,
                "draft_top1": [1, 2, 3, 4, 5],
                "target_argmax_token_id": 1,
                "teacher_forced_accept_lcp": 0,
            },
        ]
    )
    tree = _measurement(
        [
            {
                "forced_position": 16,
                "draft_top1": [1, 2, 3, 4, 5],
                "target_argmax_token_id": 1,
                "teacher_forced_accept_lcp": 1,
            }
        ]
    )

    summary = measure.compare_measurements(e5, tree)

    assert summary["n_rows"] == 1
    assert summary["e5_forced_acceptance"]["avg"] == 5
    assert summary["tree_forced_acceptance"]["avg"] == 1
