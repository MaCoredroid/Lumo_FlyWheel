from __future__ import annotations

import re

import pytest

from ci_app.workflow_preview import preview_jobs, render_summary


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (
            " schema-check ",
            {
                "job_id": "ci-app-schema-check",
                "artifact_name": "inventory-schema-check-report",
                "report_path": "reports/schema-check.json",
            },
        ),
        (
            "render/check",
            {
                "job_id": "ci-app-render-check",
                "artifact_name": "inventory-render-check-report",
                "report_path": "reports/render-check.json",
            },
        ),
        (
            "stock_recount",
            {
                "job_id": "ci-app-stock-recount",
                "artifact_name": "inventory-stock-recount-report",
                "report_path": "reports/stock-recount.json",
            },
        ),
    ],
)
def test_preview_jobs_normalize_common_inventory_separator_forms(
    label: str,
    expected: dict[str, str],
) -> None:
    assert preview_jobs([label]) == [expected]


def test_render_summary_preserves_requested_order() -> None:
    summary = render_summary(["schema-check", "render/check", "stock_recount"])

    assert summary == (
        "ci-app-schema-check => inventory-schema-check-report @ reports/schema-check.json"
        " | ci-app-render-check => inventory-render-check-report @ reports/render-check.json"
        " | ci-app-stock-recount => inventory-stock-recount-report @ reports/stock-recount.json"
    )


def test_preview_jobs_keep_report_paths_under_reports_dir() -> None:
    [job] = preview_jobs(["schema:us"])

    assert re.fullmatch(r"reports/[a-z0-9-]+\.json", job["report_path"])
