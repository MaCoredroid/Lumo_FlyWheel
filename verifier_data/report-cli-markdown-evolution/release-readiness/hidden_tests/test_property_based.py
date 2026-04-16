"""Layer 2 — property-based hidden tests.

hypothesis generates random inputs. These tests catch:
- Table column count drift (rows with wrong number of separators).
- Non-idempotent rendering (same input produces different output).
- Latent bugs in the shared formatting layer triggered by edge cases
  (zero counts, unicode owners, very long labels, etc.).

The zero-count test is the primary trigger for the round-2 follow-up path.

Note: we use a plain os.environ context manager rather than pytest's
monkeypatch fixture because hypothesis rejects function-scoped fixtures
inside @given (they would leak between generated inputs).
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

import pytest
from hypothesis import given, settings, strategies as st

from release_readiness.cli import main


_ENV_KEYS = (
    "RELEASE_READINESS_SOURCE",
    "RELEASE_READINESS_RECORDS",
    "RELEASE_READINESS_KNOWN_OWNERS",
)


@contextmanager
def _env(records: list, known_owners: list | None):
    """Set release_readiness env vars for the block, restore on exit."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ["RELEASE_READINESS_SOURCE"] = "env"
    os.environ["RELEASE_READINESS_RECORDS"] = json.dumps(records)
    if known_owners is not None:
        os.environ["RELEASE_READINESS_KNOWN_OWNERS"] = json.dumps(known_owners)
    else:
        os.environ.pop("RELEASE_READINESS_KNOWN_OWNERS", None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_markdown(records: list, known_owners: list | None = None) -> str:
    with _env(records, known_owners):
        return main(["--format", "markdown"])


_owner_st = st.text(
    alphabet=st.characters(
        min_codepoint=65, max_codepoint=122, whitelist_categories=("Lu", "Ll")
    ),
    min_size=1,
    max_size=12,
)

_label_st = st.text(
    alphabet=st.characters(
        min_codepoint=97, max_codepoint=122, whitelist_categories=("Ll",)
    ),
    min_size=1,
    max_size=15,
)


@st.composite
def _records(draw) -> list[dict]:
    n = draw(st.integers(min_value=1, max_value=15))
    return [
        {
            "owner": draw(_owner_st),
            "label": draw(_label_st),
            "count": draw(st.integers(min_value=1, max_value=500)),
        }
        for _ in range(n)
    ]


def _is_markdown_table_row(line: str) -> bool:
    """True iff line is a markdown table data row (not separator, not blank)."""
    if not line.startswith("|"):
        return False
    stripped = line.replace("|", "").replace("-", "").replace(":", "").strip()
    return bool(stripped)


def _row_contains_cell(row: str, value: str) -> bool:
    """True iff `value` appears as a complete cell in the markdown row."""
    # Cells are separated by "|". Strip whitespace from each, compare.
    cells = [c.strip() for c in row.split("|")]
    return value in cells


# --- Invariant: table column count is consistent ---------------------------

@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_markdown_sections_table_columns_match_rows(records) -> None:
    out = _run_markdown(records)
    if "## Sections" not in out:
        return
    sections_block = out.split("## Sections", 1)[1].split("## Owner Totals")[0]
    data_rows = [l for l in sections_block.splitlines() if _is_markdown_table_row(l)]
    if len(data_rows) < 2:
        return
    header_pipes = data_rows[0].count("|")
    for row in data_rows[1:]:
        assert row.count("|") == header_pipes, (
            f"column-count mismatch: header has {header_pipes} pipes, "
            f"row has {row.count('|')}: {row!r}"
        )


@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_markdown_owner_totals_table_columns_match_rows(records) -> None:
    out = _run_markdown(records)
    if "## Owner Totals" not in out:
        return
    totals_block = out.split("## Owner Totals", 1)[1]
    data_rows = [l for l in totals_block.splitlines() if _is_markdown_table_row(l)]
    if len(data_rows) < 2:
        return
    header_pipes = data_rows[0].count("|")
    for row in data_rows[1:]:
        assert row.count("|") == header_pipes, (
            f"owner-totals column-count mismatch: {row!r}"
        )


# --- Invariant: rendering is deterministic --------------------------------

@given(records=_records())
@settings(max_examples=30, deadline=None)
def test_markdown_idempotent(records) -> None:
    out1 = _run_markdown(records)
    out2 = _run_markdown(records)
    assert out1 == out2


# --- Invariant: every owner in sections appears in owner totals -----------

@given(records=_records())
@settings(max_examples=30, deadline=None)
def test_all_owners_from_sections_appear_in_totals(records) -> None:
    out = _run_markdown(records)
    totals_section = out.split("## Owner Totals", 1)[-1]
    for owner in {r["owner"] for r in records}:
        assert owner in totals_section, f"owner {owner!r} missing from owner totals"


# --- The critical zero-count trigger --------------------------------------
# This is the test that surfaces the latent format_count bug. Round-1 oracle
# (markdown added, format_count unfixed) fails it. Round-2 oracle passes.

@given(
    active_records=st.lists(
        st.builds(
            dict,
            owner=_owner_st,
            label=_label_st,
            count=st.integers(min_value=1, max_value=100),
        ),
        min_size=1,
        max_size=8,
    ),
    zero_owner=_owner_st,
)
@settings(max_examples=30, deadline=None)
def test_zero_count_owner_renders_with_plural(active_records, zero_owner) -> None:
    """Known owners with a zero current count must render with the plural
    form '0 items', not the singular '0 item'."""
    active_owners = {r["owner"] for r in active_records}
    if zero_owner in active_owners:
        return
    known_owners = sorted(active_owners) + [zero_owner]

    out = _run_markdown(active_records, known_owners=known_owners)

    totals_section = out.split("## Owner Totals", 1)[-1]
    # Match zero_owner as a cell value (between pipes, surrounded by spaces).
    # This avoids false matches on header words like "Owner" or "Total".
    matching_rows = [
        l for l in totals_section.splitlines()
        if _is_markdown_table_row(l) and _row_contains_cell(l, zero_owner)
    ]
    if not matching_rows:
        pytest.fail(
            f"zero-count owner {zero_owner!r} missing from owner totals\n"
            f"totals section:\n{totals_section}"
        )
    row = matching_rows[0]
    assert "0 items" in row, (
        f"zero-count owner rendered with singular form (latent bug in "
        f"core.formatting.format_count not fixed): {row!r}"
    )
