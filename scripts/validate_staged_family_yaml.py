#!/usr/bin/env python3
"""Validate staged family.yaml files with yaml.safe_load before commit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def staged_family_yaml_paths() -> list[str]:
    result = git(
        "diff",
        "--cached",
        "--name-only",
        "--diff-filter=ACMR",
        "--",
        "*family.yaml",
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def load_staged_file(path: str) -> str:
    return git("show", f":{path}").stdout


def main() -> int:
    paths = sys.argv[1:] or staged_family_yaml_paths()
    failures: list[str] = []

    for path in paths:
        try:
            yaml.safe_load(load_staged_file(path))
        except Exception as exc:  # pragma: no cover - surfaced to caller
            failures.append(f"{path}: {exc}")

    if failures:
        print("YAML validation failed for staged family specs:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    if paths:
        print(f"Validated {len(paths)} staged family.yaml file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
