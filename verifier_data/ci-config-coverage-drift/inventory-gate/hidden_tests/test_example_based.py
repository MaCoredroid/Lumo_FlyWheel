from __future__ import annotations

import tomllib
from pathlib import Path

from ci_app.workflow_preview import expected_package_name, preview_jobs, workflow_command


def _package_name() -> str:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return config["tool"]["lumo_ci"]["package"]


def test_repo_no_longer_exposes_legacy_package_name() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert _package_name() == "ci_app"
    assert "inventory_gate_legacy" not in workflow


def test_expected_package_name_matches_pyproject() -> None:
    assert expected_package_name() == _package_name() == "ci_app"


def test_workflow_command_routes_through_make_ci() -> None:
    assert workflow_command() == "make ci"


def test_default_preview_jobs_use_ci_app_prefix_and_inventory_report_paths() -> None:
    assert preview_jobs() == [
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
