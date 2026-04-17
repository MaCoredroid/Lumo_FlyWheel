from __future__ import annotations

import pytest

from ci_app.workflow_preview import preview_jobs, render_summary


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("ranking...(canary)", {"job_id": "ci-app-ranking-canary", "pytest_selector": "-k ranking and canary"}),
        ("fixture[locale]/fr", {"job_id": "ci-app-fixture-locale-fr", "pytest_selector": "-k fixture and locale and fr"}),
        ("query.v2::drift", {"job_id": "ci-app-query-v2-drift", "pytest_selector": "-k query and v2 and drift"}),
    ],
)
def test_preview_jobs_strip_extra_punctuation_from_search_labels(
    label: str,
    expected: dict[str, str],
) -> None:
    assert preview_jobs([label]) == [expected]


def test_render_summary_never_leaks_brackets_or_dotted_tokens_into_selector_output() -> None:
    summary = render_summary(["ranking...(canary)", "fixture[locale]/fr"])

    assert "(" not in summary
    assert "[" not in summary
    assert ".." not in summary
