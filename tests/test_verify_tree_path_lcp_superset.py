from pathlib import Path

from scripts.verify_tree_path_lcp_superset import _load_rows, summarize_rows, verify_rows


def test_verify_rows_accepts_superset_log() -> None:
    ok, errors = verify_rows([
        {
            "accepted_len": 2,
            "path0_lcp": 1,
            "superset_violation": False,
            "path_scores": [
                {"leaf": 3, "lcp": 1},
                {"leaf": 4, "lcp": 2},
            ],
        }
    ])

    assert ok
    assert errors == []


def test_verify_rows_rejects_underaccepted_winner() -> None:
    ok, errors = verify_rows([
        {
            "accepted_len": 1,
            "path0_lcp": 2,
            "superset_violation": True,
            "path_scores": [
                {"leaf": 3, "lcp": 2},
                {"leaf": 4, "lcp": 1},
            ],
        }
    ])

    assert not ok
    assert any("superset violation" in error for error in errors)
    assert any("does not equal max path lcp" in error for error in errors)


def test_summarize_rows_reports_path0_and_recovery() -> None:
    summary = summarize_rows([
        {
            "accepted_len": 2,
            "path0_lcp": 1,
            "winner_leaf": 4,
            "path_scores": [
                {"leaf": 3, "lcp": 1},
                {"leaf": 4, "lcp": 2},
            ],
        },
        {
            "accepted_len": 3,
            "path0_lcp": 3,
            "winner_leaf": 8,
            "path_scores": [
                {"leaf": 8, "lcp": 3},
                {"leaf": 9, "lcp": 1},
            ],
        },
    ])

    assert summary["rows"] == 2
    assert summary["accepted_total"] == 5
    assert summary["path0_total"] == 4
    assert summary["avg_accepted_len"] == 2.5
    assert summary["avg_path0_lcp"] == 2.0
    assert summary["recovery_event_count"] == 1
    assert summary["recovered_token_total"] == 1
    assert summary["winner_index_counts"] == {"1": 1, "0": 1}


def test_load_rows_skips_idle_batch_slots(tmp_path: Path) -> None:
    trace = tmp_path / "tree_path_lcp_max.jsonl"
    trace.write_text(
        "\n".join([
            (
                '{"event":"tree_path_lcp_max","node_count":0,'
                '"accepted_len":0,"path0_lcp":0,"path_scores":[]}'
            ),
            (
                '{"event":"tree_path_lcp_max","node_count":10,'
                '"accepted_len":1,"path0_lcp":1,'
                '"path_scores":[{"leaf":8,"lcp":1}]}'
            ),
        ])
        + "\n",
        encoding="utf-8",
    )

    rows = _load_rows(trace)

    assert len(rows) == 1
    assert rows[0]["node_count"] == 10
