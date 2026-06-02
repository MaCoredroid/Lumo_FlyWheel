from pathlib import Path

from scripts.verify_independent_winner_trace import summarize


def test_summarize_independent_winner_trace_accepts_clean_recovery(tmp_path: Path) -> None:
    trace = tmp_path / "winner.jsonl"
    trace.write_text(
        '{"winner_acc": 3, "winner_spine": 1, "counts": {"0": 1, "1": 3}, "copy": {"missing": 0}}\n'
        '{"winner_acc": 2, "winner_spine": 0, "counts": {"0": 2, "1": 1}, "copy": {"missing": 0}}\n',
        encoding="utf-8",
    )

    summary = summarize(trace)

    assert summary["rows"] == 2
    assert summary["superset_violations"] == 0
    assert summary["copy_missing_sum"] == 0
    assert summary["hidden_winner_suppressed_events"] == 0
    assert summary["winner_nonzero_spine_events"] == 1
    assert summary["recovered_token_total"] == 2


def test_summarize_independent_winner_trace_reports_violations(tmp_path: Path) -> None:
    trace = tmp_path / "winner.jsonl"
    trace.write_text(
        '{"winner_acc": 1, "winner_spine": 0, "counts": {"0": 1, "1": 3}, "copy": {"missing": 2}}\n',
        encoding="utf-8",
    )

    summary = summarize(trace)

    assert summary["superset_violations"] == 1
    assert summary["copy_missing_sum"] == 2


def test_summarize_independent_winner_trace_reports_suppressed_hidden_winners(tmp_path: Path) -> None:
    trace = tmp_path / "winner.jsonl"
    trace.write_text(
        '{"winner_acc": 1, "winner_spine": 0, "candidate_winner_spine": 1, '
        '"candidate_winner_acc": 3, "hidden_winner_suppressed_reason": "no_lossless_selector", '
        '"policy": "lossless", "selector_enabled": false, "lossless_public_stream": true, '
        '"counts": {"0": 1, "1": 3}, "copy": {"missing": 0}}\n',
        encoding="utf-8",
    )

    summary = summarize(trace)

    assert summary["superset_violations"] == 0
    assert summary["lossless_suppressed_superset_events"] == 1
    assert summary["hidden_winner_suppressed_events"] == 1
    assert summary["hidden_recovery_opportunity_total"] == 2
    assert summary["lossless_public_stream_events"] == 1
    assert summary["non_lossless_public_stream_events"] == 0
    assert summary["policies"] == {"lossless": 1}
    assert summary["examples"][0]["reason"] == "hidden winner suppressed"


def test_summarize_independent_winner_trace_still_reports_unlabeled_best_of_mismatch(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "winner.jsonl"
    trace.write_text(
        '{"winner_acc": 1, "winner_spine": 0, "candidate_winner_spine": 1, '
        '"candidate_winner_acc": 3, "hidden_winner_suppressed_reason": "stochastic_sampling", '
        '"counts": {"0": 1, "1": 3}, "copy": {"missing": 0}}\n',
        encoding="utf-8",
    )

    summary = summarize(trace)

    assert summary["superset_violations"] == 1
    assert summary["lossless_suppressed_superset_events"] == 0
    assert summary["non_lossless_public_stream_events"] == 1
