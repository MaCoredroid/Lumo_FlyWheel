"""Aggregation pipeline.

Pure functions that turn raw Section tuples into a fully-populated Report.
No I/O. No side effects. Deterministic for determinism's sake — the order of
owner totals is stable across runs given the same input.
"""
from __future__ import annotations

from collections import defaultdict

from release_readiness.core.model import OwnerTotal, Report, Section


def compute_owner_totals(sections: tuple[Section, ...]) -> tuple[OwnerTotal, ...]:
    """Sum counts per owner, sorted by total desc, then owner name asc."""
    totals: dict[str, int] = defaultdict(int)
    for section in sections:
        totals[section.owner] += section.count
    return tuple(
        OwnerTotal(owner=owner, total=total)
        for owner, total in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    )


def build_report(
    title: str,
    sections: tuple[Section, ...],
    known_owners: tuple[str, ...] = (),
) -> Report:
    """Build a fully-populated Report.

    `known_owners` is intended for the caller to pass the superset of all
    owners seen in the current reporting period. Currently no adapter
    populates it, so it defaults to the set of owners present in the
    sections.
    """
    owner_totals = compute_owner_totals(sections)
    # Default: known_owners = whoever currently has at least one section.
    # Adapters that track cross-period owner history may override.
    if not known_owners:
        known_owners = tuple(ot.owner for ot in owner_totals)
    return Report(
        title=title,
        sections=sections,
        owner_totals=owner_totals,
        known_owners=known_owners,
    )
