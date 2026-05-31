from scripts.verify_tree_path_lcp_superset import verify_rows


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
