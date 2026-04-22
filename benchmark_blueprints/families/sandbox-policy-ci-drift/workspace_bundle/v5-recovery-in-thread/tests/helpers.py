
from __future__ import annotations

import os
from pathlib import Path


def fixture_path() -> Path:
    return Path(os.environ.get("CODEX_CONFIG_FIXTURE", "tests/fixtures/visible_config.toml"))
