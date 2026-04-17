from __future__ import annotations

from ci_app.workflow_preview import preview_jobs, render_summary


def test_preview_jobs_preserve_required_checks_and_report_paths() -> None:
    jobs = preview_jobs()

    assert jobs == [
        {
            "job_id": "ci-app-schema-check",
            "artifact_name": "inventory-schema-check-report",
            "report_path": "reports/schema-check.json",
        },
        {
            "job_id": "ci-app-render-check",
            "artifact_name": "inventory-render-check-report",
            "report_path": "reports/render-check.json",
        },
    ]


def test_preview_summary_mentions_every_required_label() -> None:
    summary = render_summary()

    assert "schema-check" in summary
    assert "render-check" in summary


def test_preview_jobs_normalize_simple_inventory_labels() -> None:
    assert preview_jobs(["stock/recount"]) == [
        {
            "job_id": "ci-app-stock-recount",
            "artifact_name": "inventory-stock-recount-report",
            "report_path": "reports/stock-recount.json",
        }
    ]
