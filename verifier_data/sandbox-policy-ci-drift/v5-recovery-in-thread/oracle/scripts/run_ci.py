
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"])


if __name__ == "__main__":
    raise SystemExit(main())
