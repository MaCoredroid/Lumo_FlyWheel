from report_filters.cli import render_filters


def test_cli_normalizes_separator_heavy_labels() -> None:
    assert (
        render_filters("Ops---Latency__Summary,Slack Alerts")
        == "ops latency summary,slack alerts"
    )


def test_cli_drops_blank_entries() -> None:
    assert render_filters("  , API__Errors ") == "api errors"
