from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package")
    args = parser.parse_args(argv)

    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    configured_package = config["tool"]["lumo_ci"]["package"]
    package_name = args.package or configured_package
    if package_name != "ci_app":
        print(f"ci package drift: {package_name}", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, "-m", "pytest", "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
