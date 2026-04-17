from __future__ import annotations

from collections.abc import Iterable

from ci_app.service import run_checks


PACKAGE_NAME = "inventory_gate_legacy"


def expected_package_name() -> str:
    return PACKAGE_NAME


def workflow_command(package_name: str | None = None) -> str:
    package = package_name or expected_package_name()
    return f"python scripts/run_ci.py --package {package}"


def _preview_slug(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("_", "-")
        .replace("/", "-")
        .replace(" ", "-")
    )


def preview_jobs(labels: Iterable[str] | None = None) -> list[dict[str, str]]:
    active_labels = list(run_checks() if labels is None else labels)
    package_slug = expected_package_name().replace("_", "-")
    return [
        {
            "job_id": f"{package_slug}-{_preview_slug(label)}",
            "artifact_name": f"inventory-{_preview_slug(label)}-report",
            "report_path": f"reports/{_preview_slug(label)}.json",
        }
        for label in active_labels
    ]


def render_summary(labels: Iterable[str] | None = None) -> str:
    return " | ".join(
        f'{job["job_id"]} => {job["artifact_name"]} @ {job["report_path"]}'
        for job in preview_jobs(labels)
    )
