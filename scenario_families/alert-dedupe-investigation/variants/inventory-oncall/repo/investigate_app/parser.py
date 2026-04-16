from __future__ import annotations

import json
from pathlib import Path


def load_events(path: str = "logs/failure.log") -> list[dict[str, str]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
