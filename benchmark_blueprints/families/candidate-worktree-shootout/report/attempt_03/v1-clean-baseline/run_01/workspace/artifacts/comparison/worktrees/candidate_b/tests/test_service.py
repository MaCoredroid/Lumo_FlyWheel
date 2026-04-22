from report_filters.service import compile_filters


def test_compile_filters_handles_basic_whitespace() -> None:
    assert compile_filters([" Already Clean "]) == ["already clean"]


def test_compile_filters_normalizes_separator_heavy_labels() -> None:
    assert compile_filters(["Ops---Latency__Summary", " Slack Alerts "]) == [
        "ops latency summary",
        "slack alerts",
    ]
