from __future__ import annotations

from release_readiness.core.formatting import format_count, join_with_commas, pad_cell


def test_format_count_singular() -> None:
    assert format_count(1, singular="item", plural="items") == "1 item"


def test_format_count_plural() -> None:
    assert format_count(2, singular="item", plural="items") == "2 items"
    assert format_count(17, singular="owner", plural="owners") == "17 owners"


def test_join_with_commas_empty() -> None:
    assert join_with_commas([]) == ""


def test_join_with_commas_one_two_three() -> None:
    assert join_with_commas(["a"]) == "a"
    assert join_with_commas(["a", "b"]) == "a and b"
    assert join_with_commas(["a", "b", "c"]) == "a, b, and c"


def test_pad_cell() -> None:
    assert pad_cell("hi", 5) == "hi   "
    assert pad_cell("overflow", 3) == "overflow"
