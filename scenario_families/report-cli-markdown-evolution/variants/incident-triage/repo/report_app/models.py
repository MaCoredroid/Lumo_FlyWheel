from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Incident:
    service: str
    severity: str
    owner: str
    minutes_open: int
    acked: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "Incident":
        return cls(
            service=str(payload["service"]),
            severity=str(payload["severity"]),
            owner=str(payload["owner"]),
            minutes_open=int(payload["minutes_open"]),
            acked=bool(payload["acked"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "service": self.service,
            "severity": self.severity,
            "owner": self.owner,
            "minutes_open": self.minutes_open,
            "acked": self.acked,
        }

    def is_ack_sla_breached(self, sla_minutes: Mapping[str, int]) -> bool:
        if self.acked:
            return False
        return self.minutes_open > int(sla_minutes[self.severity])


@dataclass(frozen=True)
class OwnerLoad:
    owner: str
    count: int
    breached: int

    def as_dict(self) -> dict[str, object]:
        return {
            "owner": self.owner,
            "count": self.count,
            "breached": self.breached,
        }


@dataclass(frozen=True)
class TriageSummary:
    incident_count: int
    breached_count: int
    highest_load_owner: dict[str, object] | None
    owner_load: tuple[OwnerLoad, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "incident_count": self.incident_count,
            "breached_count": self.breached_count,
            "highest_load_owner": self.highest_load_owner,
            "owner_load": [entry.as_dict() for entry in self.owner_load],
        }
