from __future__ import annotations

from release_readiness.core.aggregation import build_report, compute_owner_totals
from release_readiness.core.model import Section


def _sections(*triples: tuple[str, str, int]) -> tuple[Section, ...]:
    return tuple(Section(owner=o, label=lbl, count=c) for o, lbl, c in triples)


def test_owner_totals_sum_correctly() -> None:
    totals = compute_owner_totals(
        _sections(("Sam", "a", 2), ("Rin", "b", 1), ("Sam", "c", 3))
    )
    by_owner = {t.owner: t.total for t in totals}
    assert by_owner == {"Sam": 5, "Rin": 1}


def test_owner_totals_sorted_by_total_desc_then_name_asc() -> None:
    totals = compute_owner_totals(
        _sections(("B", "x", 2), ("A", "x", 2), ("C", "x", 1))
    )
    assert [t.owner for t in totals] == ["A", "B", "C"]


def test_build_report_defaults_known_owners_to_present_owners() -> None:
    report = build_report(
        title="T",
        sections=_sections(("Sam", "x", 1), ("Rin", "y", 2)),
    )
    assert set(report.known_owners) == {"Sam", "Rin"}


def test_build_report_respects_explicit_known_owners() -> None:
    report = build_report(
        title="T",
        sections=_sections(("Sam", "x", 1)),
        known_owners=("Sam", "Rin", "Alex"),
    )
    assert report.known_owners == ("Sam", "Rin", "Alex")
