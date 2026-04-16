from __future__ import annotations

from ci_app.service import run_checks


def main() -> str:
    return ",".join(run_checks())
