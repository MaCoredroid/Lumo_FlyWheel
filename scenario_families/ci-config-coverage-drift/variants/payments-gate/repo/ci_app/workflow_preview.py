from __future__ import annotations

from collections.abc import Iterable

from ci_app.service import run_checks


PACKAGE_NAME = "payments_gate_legacy"


def expected_package_name() -> str:
    return PACKAGE_NAME


def workflow_command(package_name: str | None = None) -> str:
    package = package_name or expected_package_name()
    return f"python scripts/run_ci.py --package {package}"


def dispatch_job_ids(labels: Iterable[str] | None = None) -> list[str]:
    active_labels = list(run_checks() if labels is None else labels)
    package_slug = expected_package_name().replace("_", "-")
    return [
        f"{package_slug}-{label.strip().lower().replace('_', '-').replace('/', '-')}"
        for label in active_labels
    ]


def render_summary(labels: Iterable[str] | None = None) -> str:
    return " | ".join(dispatch_job_ids(labels))
