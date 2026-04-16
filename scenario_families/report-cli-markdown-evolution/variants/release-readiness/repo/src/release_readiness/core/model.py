"""Domain model for release-readiness reports."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    """A single section of the report.

    A section represents one tracked item (a blocked rollout, a hotfix, etc.)
    assigned to an owner. Count is the current queued/outstanding work for
    that item.
    """

    owner: str
    label: str
    count: int

    def __post_init__(self) -> None:
        if not self.owner.strip():
            raise ValueError("owner must be non-empty")
        if not self.label.strip():
            raise ValueError("label must be non-empty")
        if self.count < 0:
            raise ValueError("count must be non-negative")


@dataclass(frozen=True)
class OwnerTotal:
    """Aggregated total per owner across all sections."""

    owner: str
    total: int


@dataclass(frozen=True)
class Report:
    """The fully-aggregated report handed to a renderer."""

    title: str
    sections: tuple[Section, ...]
    owner_totals: tuple[OwnerTotal, ...]
    known_owners: tuple[str, ...] = field(default=())
    """All owners ever seen in the current reporting period, including those
    whose current count is zero. Populated by the aggregation layer."""


def sections_from_iterable(records: Iterable[dict[str, object]]) -> tuple[Section, ...]:
    """Build Section tuples from raw dict records.

    Raises ValueError on malformed records.
    """
    out: list[Section] = []
    for record in records:
        try:
            owner = str(record["owner"])
            label = str(record["label"])
            raw_count = record["count"]
            if not isinstance(raw_count, (int, str)):
                raise TypeError(f"count must be int or str, got {type(raw_count).__name__}")
            count = int(raw_count)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed record: {record!r}") from exc
        out.append(Section(owner=owner, label=label, count=count))
    return tuple(out)
