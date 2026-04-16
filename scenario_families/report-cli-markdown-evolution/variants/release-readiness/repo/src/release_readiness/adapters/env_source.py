"""Environment source adapter.

Reads section records and known-owner history from environment variables.
Used by the test harness and by automation that wants inline inputs.
"""
from __future__ import annotations

import json
import os

from release_readiness.core.model import Section, sections_from_iterable

_ENV_RECORDS = "RELEASE_READINESS_RECORDS"
_ENV_KNOWN_OWNERS = "RELEASE_READINESS_KNOWN_OWNERS"


class EnvSource:
    def load(self) -> tuple[Section, ...]:
        raw = os.environ.get(_ENV_RECORDS)
        if not raw:
            return ()
        return sections_from_iterable(json.loads(raw))

    def known_owners(self) -> tuple[str, ...]:
        """All owners ever seen in the current reporting period.

        Feeds into Report.known_owners so renderers that show "all owners,
        including zero-count" can do so without the sections table alone.
        """
        raw = os.environ.get(_ENV_KNOWN_OWNERS)
        if not raw:
            return ()
        owners = json.loads(raw)
        if not isinstance(owners, list):
            raise ValueError(f"{_ENV_KNOWN_OWNERS} must be a JSON array")
        return tuple(str(o) for o in owners)
