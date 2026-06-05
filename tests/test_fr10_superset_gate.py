from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr10_superset_gate import (
    AcceptanceEvent,
    diagnose_spine_degradation,
    evaluate_enforced_superset_gate,
    evaluate_path0_sequence_gate,
    evaluate_strict_win_gate,
    evaluate_superset_hard_gate,
    evaluate_total_acceptance_gate,
    load_spec_trace,
    load_tree_accept_trace,
    path0_runtime_nodes,
    runtime_node_map,
)


CATERPILLAR_TREE = (
    "[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), "
    "(0, 0, 0, 0, 0), (0, 1), (0, 0, 1), "
    "(0, 0, 0, 1), (0, 0, 0, 0, 1)]"
)


def test_runtime_tree_mapping_resolves_depth_branches() -> None:
    node_map = runtime_node_map(CATERPILLAR_TREE)

    assert node_map[0] == (0,)
    assert node_map[1] == (0, 0)
    assert node_map[2] == (0, 1)
    assert node_map[4] == (0, 0, 1)
    assert path0_runtime_nodes(CATERPILLAR_TREE) == (0, 1, 3, 5, 7)


def test_tier1_external_path0_gate_catches_native_degradation(tmp_path: Path) -> None:
    native_path = tmp_path / "native_per_req_spec_trace.jsonl"
    tree_path = tmp_path / "tree_path_lcp_max.jsonl"
    native_path.write_text(
        "\n".join(
            [
                json.dumps({"rid": "r0", "acc": 5, "ts": 1.0}),
                json.dumps({"rid": "r0", "acc": 4, "ts": 2.0}),
            ]
        )
        + "\n"
    )
    # Internally this tree can still be self-consistent, but its path0 only
    # reaches depth 2. This is the external degradation FR10 must reject.
    tree_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "tree_sample_accept",
                        "req_index": 0,
                        "accepted_len": 2,
                        "accepted_node_ids": [0, 1],
                        "draft_token_ids": [10, 11, 12, 13, 14],
                        "ts": 1.1,
                    }
                ),
                json.dumps(
                    {
                        "event": "tree_sample_accept",
                        "req_index": 0,
                        "accepted_len": 2,
                        "accepted_node_ids": [0, 1],
                        "draft_token_ids": [20, 21, 22, 23, 24],
                        "ts": 2.1,
                    }
                ),
            ]
        )
        + "\n"
    )

    report = evaluate_path0_sequence_gate(
        native_events=load_spec_trace(native_path),
        tree_events=load_tree_accept_trace(
            tree_path,
            speculative_token_tree=CATERPILLAR_TREE,
        ),
    )

    assert not report.passed
    assert report.metrics["native_avg"] == pytest.approx(4.5)
    assert report.metrics["tree_path0_avg"] == pytest.approx(2.0)
    assert report.metrics["first_diff"] == 0
    assert "path0 accepted sequence differs" in report.violations[0]


def test_tier2_total_acceptance_gate_requires_tree_not_below_native() -> None:
    native = [
        AcceptanceEvent("r", 0, accepted_len=3, path0_len=3),
        AcceptanceEvent("r", 1, accepted_len=3, path0_len=3),
    ]
    tree_bad = [
        AcceptanceEvent("r", 0, accepted_len=2, path0_len=2),
        AcceptanceEvent("r", 1, accepted_len=2, path0_len=2),
    ]
    tree_good = [
        AcceptanceEvent("r", 0, accepted_len=3, path0_len=3),
        AcceptanceEvent("r", 1, accepted_len=4, path0_len=3),
    ]

    bad = evaluate_total_acceptance_gate(native_events=native, tree_events=tree_bad)
    good = evaluate_total_acceptance_gate(native_events=native, tree_events=tree_good)

    assert not bad.passed
    assert bad.metrics["tree_minus_native_avg"] == pytest.approx(-1.0)
    assert good.passed
    assert good.metrics["tree_minus_native_avg"] == pytest.approx(0.5)


def test_superset_hard_gate_requires_tree_ge_path0_ge_native_per_event() -> None:
    native = [
        AcceptanceEvent("r", 0, accepted_len=2, path0_len=2),
        AcceptanceEvent("r", 1, accepted_len=3, path0_len=3),
    ]
    tree_good = [
        AcceptanceEvent("r", 0, accepted_len=3, path0_len=2),
        AcceptanceEvent("r", 1, accepted_len=4, path0_len=3),
    ]
    tree_bad = [
        AcceptanceEvent("r", 0, accepted_len=1, path0_len=2),
        AcceptanceEvent("r", 1, accepted_len=4, path0_len=2),
    ]

    good = evaluate_superset_hard_gate(native_events=native, tree_events=tree_good)
    bad = evaluate_superset_hard_gate(native_events=native, tree_events=tree_bad)

    assert good.passed
    assert not bad.passed
    assert bad.metrics["violations"] == 2
    assert "tree accepted below path0" in bad.violations[0]
    assert "path0 accepted below native" in bad.violations[1]


def test_strict_win_gate_requires_bootstrap_ci_lower_bound_above_zero() -> None:
    native = [
        AcceptanceEvent("r", idx, accepted_len=2, path0_len=2)
        for idx in range(40)
    ]
    tree_win = [
        AcceptanceEvent("r", idx, accepted_len=3, path0_len=2)
        for idx in range(40)
    ]
    tree_tie = [
        AcceptanceEvent("r", idx, accepted_len=2, path0_len=2)
        for idx in range(40)
    ]
    tree_loss = [
        AcceptanceEvent("r", idx, accepted_len=1, path0_len=2)
        for idx in range(40)
    ]

    win = evaluate_strict_win_gate(
        native_events=native,
        tree_events=tree_win,
        bootstrap_samples=500,
    )
    tie = evaluate_strict_win_gate(
        native_events=native,
        tree_events=tree_tie,
        bootstrap_samples=500,
    )
    loss = evaluate_strict_win_gate(
        native_events=native,
        tree_events=tree_loss,
        bootstrap_samples=500,
    )

    assert win.passed
    assert win.metrics["ci_low"] > 0.0
    assert not tie.passed
    assert tie.metrics["ci_low"] == pytest.approx(0.0)
    assert not loss.passed
    assert loss.metrics["ci_low"] < 0.0


def test_enforced_superset_gate_fails_powered_tie_negative_control() -> None:
    native = [
        AcceptanceEvent("r", idx, accepted_len=2, path0_len=2)
        for idx in range(20)
    ]
    tree_tie = [
        AcceptanceEvent("r", idx, accepted_len=2, path0_len=2)
        for idx in range(20)
    ]

    report = evaluate_enforced_superset_gate(
        native_events=native,
        tree_events=tree_tie,
        bootstrap_samples=500,
    )

    assert not report.passed
    assert report.metrics["hard"]["violations"] == 0
    assert report.metrics["strict_win"]["tree_minus_native_avg"] == pytest.approx(0.0)
    assert "strict tree win not statistically proven" in report.violations[0]


def test_diagnostic_splits_draft_corruption_from_commit_bug() -> None:
    native = [
        AcceptanceEvent(
            "r",
            0,
            accepted_len=4,
            path0_len=4,
            draft_token_ids=(10, 11, 12, 13),
        )
    ]
    tree_draft_bad = [
        AcceptanceEvent(
            "r",
            0,
            accepted_len=2,
            path0_len=2,
            draft_token_ids=(10, 11, 99, 13),
        )
    ]
    tree_commit_bad = [
        AcceptanceEvent(
            "r",
            0,
            accepted_len=2,
            path0_len=2,
            draft_token_ids=(10, 11, 12, 13),
        )
    ]

    draft_report = diagnose_spine_degradation(
        native_events=native,
        tree_events=tree_draft_bad,
    )
    commit_report = diagnose_spine_degradation(
        native_events=native,
        tree_events=tree_commit_bad,
    )

    assert not draft_report.passed
    assert draft_report.metrics["classification"] == "draft_side_recurrent_state_corruption"
    assert not commit_report.passed
    assert commit_report.metrics["classification"] == "commit_or_position_bookkeeping_bug"
