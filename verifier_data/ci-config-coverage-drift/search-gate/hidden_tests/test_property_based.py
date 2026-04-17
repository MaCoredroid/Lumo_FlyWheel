from __future__ import annotations

import re

import pytest

from ci_app.workflow_preview import preview_jobs, render_summary


@pytest.mark.parametrize(
    ("label", "expected_job_id", "expected_selector"),
    [
        (" ranking-check ", "ci-app-ranking-check", "-k ranking and check"),
        ("fixture/check", "ci-app-fixture-check", "-k fixture and check"),
        ("ranking:locale_us", "ci-app-ranking-locale-us", "-k ranking and locale and us"),
    ],
)
def test_preview_jobs_normalize_common_separator_forms(
    label: str,
    expected_job_id: str,
    expected_selector: str,
) -> None:
    assert preview_jobs([label]) == [
        {
            "job_id": expected_job_id,
            "pytest_selector": expected_selector,
        }
    ]


def test_render_summary_preserves_requested_order() -> None:
    summary = render_summary(["ranking/check", "fixture:locale_us", "query boost"])

    assert summary == (
        "ci-app-ranking-check => -k ranking and check"
        " | ci-app-fixture-locale-us => -k fixture and locale and us"
        " | ci-app-query-boost => -k query and boost"
    )


def test_preview_jobs_emit_only_lowercase_alnum_selector_tokens_for_common_labels() -> None:
    [job] = preview_jobs(["ranking:locale_us"])

    selector = job["pytest_selector"].removeprefix("-k ")
    assert re.fullmatch(r"[a-z0-9 ]+( and [a-z0-9 ]+)*", selector)
