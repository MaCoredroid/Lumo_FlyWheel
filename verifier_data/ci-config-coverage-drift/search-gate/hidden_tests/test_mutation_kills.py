from __future__ import annotations

import pytest

from ci_app.workflow_preview import preview_jobs


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("__ranking---shadow__", {"job_id": "ci-app-ranking-shadow", "pytest_selector": "-k ranking and shadow"}),
        ("..fixture@@boost..", {"job_id": "ci-app-fixture-boost", "pytest_selector": "-k fixture and boost"}),
        ("[query]....locale__fr", {"job_id": "ci-app-query-locale-fr", "pytest_selector": "-k query and locale and fr"}),
    ],
)
def test_preview_jobs_collapse_boundary_punctuation_and_repeated_separators(
    label: str,
    expected: dict[str, str],
) -> None:
    assert preview_jobs([label]) == [expected]
