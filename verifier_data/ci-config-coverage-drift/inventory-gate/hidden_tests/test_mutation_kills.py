from __future__ import annotations

import pytest

from ci_app.workflow_preview import preview_jobs


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (
            "__schema---drift__",
            {
                "job_id": "ci-app-schema-drift",
                "artifact_name": "inventory-schema-drift-report",
                "report_path": "reports/schema-drift.json",
            },
        ),
        (
            "..render@@boost..",
            {
                "job_id": "ci-app-render-boost",
                "artifact_name": "inventory-render-boost-report",
                "report_path": "reports/render-boost.json",
            },
        ),
        (
            "[stock]....recount__fr",
            {
                "job_id": "ci-app-stock-recount-fr",
                "artifact_name": "inventory-stock-recount-fr-report",
                "report_path": "reports/stock-recount-fr.json",
            },
        ),
    ],
)
def test_preview_jobs_collapse_boundary_punctuation_and_repeated_separators(
    label: str,
    expected: dict[str, str],
) -> None:
    assert preview_jobs([label]) == [expected]


def test_artifact_names_trim_boundary_punctuation() -> None:
    [job] = preview_jobs(["__schema-close__"])

    assert job["artifact_name"] == "inventory-schema-close-report"
    assert job["report_path"] == "reports/schema-close.json"
