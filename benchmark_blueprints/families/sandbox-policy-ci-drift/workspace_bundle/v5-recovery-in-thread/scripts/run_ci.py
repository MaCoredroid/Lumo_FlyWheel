
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    env = os.environ.copy()
    visible_fixture = Path("tests/fixtures/visible_config.toml")
    with tempfile.TemporaryDirectory(prefix="visible_fixture_") as tmp:
        temp_fixture = Path(tmp) / "visible_config.toml"
        shutil.copy(visible_fixture, temp_fixture)
        patched = temp_fixture.read_text(encoding="utf-8").replace(
            "workspace_write",
            "workspace-write",
        )
        temp_fixture.write_text(patched, encoding="utf-8")
        env["CODEX_CONFIG_FIXTURE"] = str(temp_fixture)
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
