"""Layer 2 — property-based hidden tests for incident-triage."""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from conftest import direct_markdown
from report_app.fixtures import ACK_SLA_MINUTES
from report_app.service import build_triage_summary


_service_st = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122, whitelist_categories=("Ll",)),
    min_size=4,
    max_size=18,
)
_owner_st = st.text(
    alphabet=st.characters(min_codepoint=65, max_codepoint=122, whitelist_categories=("Lu", "Ll")),
    min_size=1,
    max_size=8,
)
_severity_st = st.sampled_from(["sev1", "sev2", "sev3"])


@st.composite
def _minutes(draw, severity: str) -> int:
    threshold = ACK_SLA_MINUTES[severity]
    value = draw(st.integers(min_value=0, max_value=90))
    while value == threshold:
        value = draw(st.integers(min_value=0, max_value=90))
    return value


@st.composite
def _records(draw) -> list[dict[str, object]]:
    count = draw(st.integers(min_value=1, max_value=6))
    records: list[dict[str, object]] = []
    for _ in range(count):
        severity = draw(_severity_st)
        records.append(
            {
                "service": draw(_service_st),
                "severity": severity,
                "owner": draw(_owner_st),
                "minutes_open": draw(_minutes(severity)),
                "acked": draw(st.booleans()),
            }
        )
    return records


def _queue_rows(output: str) -> list[str]:
    block = output.split("## Active Queue", 1)[1].split("## Owner Load", 1)[0]
    return [line for line in block.splitlines() if line.startswith("| ")][2:]


def _owner_rows(output: str) -> list[str]:
    block = output.split("## Owner Load", 1)[1]
    return [line for line in block.splitlines() if line.startswith("| ")][2:]


@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_markdown_rendering_is_deterministic(records) -> None:
    assert direct_markdown(records) == direct_markdown(records)


@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_active_queue_table_row_count_matches_input(records) -> None:
    assert len(_queue_rows(direct_markdown(records))) == len(records)


@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_owner_load_covers_all_runtime_owners(records) -> None:
    output = direct_markdown(records)
    owner_rows = "\n".join(_owner_rows(output))

    for owner in {str(record["owner"]) for record in records}:
        assert owner in owner_rows


@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_breach_count_matches_runtime_records(records) -> None:
    summary = build_triage_summary(records)
    expected = 0
    for record in records:
        if record["acked"]:
            continue
        expected += int(record["minutes_open"] > ACK_SLA_MINUTES[str(record["severity"])])

    assert summary["breached_count"] == expected


@given(records=_records())
@settings(max_examples=40, deadline=None)
def test_owner_load_rows_follow_summary_sort(records) -> None:
    summary = build_triage_summary(records)
    output = direct_markdown(records)
    actual_order = [row.split("|")[1].strip() for row in _owner_rows(output)]
    expected_order = [entry["owner"] for entry in summary["owner_load"]]

    assert actual_order == expected_order
