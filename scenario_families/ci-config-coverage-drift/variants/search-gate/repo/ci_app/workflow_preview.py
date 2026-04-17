from __future__ import annotations

from collections.abc import Iterable

from ci_app.service import run_checks


PACKAGE_NAME = "search_gate_legacy"


def expected_package_name() -> str:
    return PACKAGE_NAME


def workflow_command(package_name: str | None = None) -> str:
    package = package_name or expected_package_name()
    return f"python scripts/run_ci.py --package {package}"


def _selector_tokens(value: str) -> list[str]:
    normalized = (
        value.strip()
        .lower()
        .replace("_", " ")
        .replace("/", " ")
        .replace(":", " ")
    )
    return [token for token in normalized.split() if token]


def _job_slug(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("_", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace(" ", "-")
    )


def preview_jobs(labels: Iterable[str] | None = None) -> list[dict[str, str]]:
    active_labels = list(run_checks() if labels is None else labels)
    package_slug = expected_package_name().replace("_", "-")
    jobs: list[dict[str, str]] = []
    for label in active_labels:
        selector = " and ".join(_selector_tokens(label))
        jobs.append(
            {
                "job_id": f"{package_slug}-{_job_slug(label)}",
                "pytest_selector": f"-k {selector}",
            }
        )
    return jobs


def render_summary(labels: Iterable[str] | None = None) -> str:
    return " | ".join(
        f'{job["job_id"]} => {job["pytest_selector"]}'
        for job in preview_jobs(labels)
    )
