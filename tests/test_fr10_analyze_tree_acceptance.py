from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fr10_analyze_tree_acceptance.py"
spec = importlib.util.spec_from_file_location("fr10_analyze_tree_acceptance", SCRIPT)
analyzer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analyzer)


def test_runtime_sorted_tree_branch_usage(tmp_path: Path) -> None:
    trace = tmp_path / "tree_path_lcp_max.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "tree_sample_accept",
                        "accepted_len": 2,
                        "accepted_node_ids": [0, 2],
                    }
                ),
                json.dumps(
                    {
                        "event": "tree_sample_accept",
                        "accepted_len": 3,
                        "accepted_node_ids": [0, 1, 4],
                    }
                ),
                json.dumps(
                    {
                        "event": "tree_sample_accept",
                        "accepted_len": 2,
                        "accepted_node_ids": [0, 1],
                    }
                ),
            ]
        )
        + "\n"
    )
    tree = (
        "[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), "
        "(0, 0, 0, 0, 0), (0, 1), (0, 0, 1), "
        "(0, 0, 0, 1), (0, 0, 0, 0, 1)]"
    )

    summary = analyzer.analyze_trace(trace, speculative_token_tree=tree)

    assert summary["runtime_tree_node_map"]["2"] == [0, 1]
    assert summary["runtime_tree_node_map"]["4"] == [0, 0, 1]
    assert summary["branch_node_usage_labeled"] == {
        "2": {"tree_path": [0, 1], "count": 1},
        "4": {"tree_path": [0, 0, 1], "count": 1},
    }
    assert summary["branch_event_count"] == 2
    assert summary["branch_event_fraction_of_all_rows"] == pytest.approx(2 / 3)
