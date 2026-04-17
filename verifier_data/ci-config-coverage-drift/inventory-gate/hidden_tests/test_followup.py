from __future__ import annotations

import pytest

from ci_app.workflow_preview import preview_jobs, render_summary


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (
            "stock...recount",
            {
                "job_id": "ci-app-stock-recount",
                "artifact_name": "inventory-stock-recount-report",
                "report_path": "reports/stock-recount.json",
            },
        ),
        (
            "render[beta]/fr",
            {
                "job_id": "ci-app-render-beta-fr",
                "artifact_name": "inventory-render-beta-fr-report",
                "report_path": "reports/render-beta-fr.json",
            },
        ),
        (
            "backfill::delta__nightly",
            {
                "job_id": "ci-app-backfill-delta-nightly",
                "artifact_name": "inventory-backfill-delta-nightly-report",
                "report_path": "reports/backfill-delta-nightly.json",
            },
        ),
    ],
)
def test_preview_jobs_strip_extra_punctuation_from_inventory_labels(
    label: str,
    expected: dict[str, str],
) -> None:
    assert preview_jobs([label]) == [expected]


def test_render_summary_never_leaks_double_hyphens_or_parent_dirs() -> None:
    summary = render_summary(["stock...recount", "render[beta]/fr"])

    assert "--" not in summary
    assert "../" not in summary
    assert "..json" not in summary
