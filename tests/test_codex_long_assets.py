from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lumo_flywheel_serving.codex_long_assets import AssetPackError, validate_authored_asset_pack


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_authored_codex_long_asset_pack() -> None:
    summary = validate_authored_asset_pack(REPO_ROOT)

    assert summary.family_count == 5
    assert summary.variant_count == 15
    assert summary.has_freeze_artifacts is False
    assert summary.scenario_types == (
        "build_ci_breakage",
        "cross_layer_changes",
        "feature_evolution",
        "investigate_then_fix",
        "migration_refactor",
    )


def test_validate_authored_pack_detects_checksum_drift(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    drifted_test = (
        repo_copy
        / "scenario_families"
        / "report-cli-markdown-evolution"
        / "variants"
        / "inventory-ops"
        / "repo"
        / "tests"
        / "test_docs.py"
    )
    drifted_test.write_text(drifted_test.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")

    with pytest.raises(AssetPackError, match="Test checksum drift"):
        validate_authored_asset_pack(repo_copy)
