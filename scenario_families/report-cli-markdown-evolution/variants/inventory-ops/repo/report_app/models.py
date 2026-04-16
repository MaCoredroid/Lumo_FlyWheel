from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    owner: str
    label: str
    count: int

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "Section":
        return cls(
            owner=str(payload["owner"]),
            label=str(payload["label"]),
            count=int(payload["count"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "owner": self.owner,
            "label": self.label,
            "count": self.count,
        }


@dataclass(frozen=True)
class OwnerTotal:
    owner: str
    count: int

    def as_dict(self) -> dict[str, object]:
        return {"owner": self.owner, "count": self.count}


@dataclass(frozen=True)
class QueueSummary:
    section_count: int
    total_items: int
    top_owner: OwnerTotal | None
    owner_totals: tuple[OwnerTotal, ...]

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "section_count": self.section_count,
            "total_items": self.total_items,
            "owner_totals": [entry.as_dict() for entry in self.owner_totals],
        }
        payload["top_owner"] = None if self.top_owner is None else self.top_owner.as_dict()
        return payload
