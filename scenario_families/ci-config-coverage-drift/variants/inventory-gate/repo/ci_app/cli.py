from __future__ import annotations

from ci_app.service import run_checks
from ci_app.workflow_preview import render_summary


def main() -> str:
    checks = run_checks()
    return f"{','.join(checks)} :: {render_summary(checks)}"
