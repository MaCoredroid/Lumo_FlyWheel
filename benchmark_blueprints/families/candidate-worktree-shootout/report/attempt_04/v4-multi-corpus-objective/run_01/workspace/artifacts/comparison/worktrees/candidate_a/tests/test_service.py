from report_filters.service import compile_filters


def test_compile_filters_handles_basic_whitespace() -> None:
    assert compile_filters([" Already Clean "]) == ["already clean"]
