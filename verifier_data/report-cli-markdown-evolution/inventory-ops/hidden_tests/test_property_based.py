"""Layer 2 — property-based hidden tests for inventory-ops."""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from conftest import direct_markdown


_owner_st = st.text(
    alphabet=st.characters(min_codepoint=65, max_codepoint=122, whitelist_categories=("Lu", "Ll")),
    min_size=1,
    max_size=10,
)
_label_st = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122, whitelist_categories=("Ll",)),
    min_size=3,
    max_size=16,
)


@st.composite
def _records(draw) -> list[dict[str, object]]:
    count = draw(st.integers(min_value=1, max_value=7))
    return [
        {
            "owner": draw(_owner_st),
            "label": draw(_label_st),
            "count": draw(st.integers(min_value=1, max_value=12)),
        }
        for _ in range(count)
    ]


def _section_rows(output: str) -> list[str]:
    block = output.split("## Sections", 1)[1].split("## Owner Totals", 1)[0]
    return [line for line in block.splitlines() if line.startswith("| ")][2:]


def _owner_rows(output: str) -> list[str]:
    block = output.split("## Owner Totals", 1)[1]
    return [line for line in block.splitlines() if line.startswith("| ")][2:]


@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_markdown_rendering_is_deterministic(records) -> None:
    assert direct_markdown(records) == direct_markdown(records)


@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_sections_table_row_count_matches_input(records) -> None:
    assert len(_section_rows(direct_markdown(records))) == len(records)


@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_owner_totals_cover_all_runtime_owners(records) -> None:
    output = direct_markdown(records)
    owner_rows = "\n".join(_owner_rows(output))

    for owner in {str(record["owner"]) for record in records}:
        assert owner in owner_rows


@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_owner_totals_are_sorted_desc_then_name(records) -> None:
    totals: dict[str, int] = {}
    for record in records:
        totals[str(record["owner"])] = totals.get(str(record["owner"]), 0) + int(record["count"])

    expected_order = [
        owner
        for owner, _count in sorted(
            totals.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )
    ]
    output = direct_markdown(records)
    actual_order = [row.split("|")[1].strip() for row in _owner_rows(output)]

    assert actual_order == expected_order
