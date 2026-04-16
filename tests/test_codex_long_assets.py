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


def test_validate_authored_pack_rejects_untrusted_pytest_bootstrap(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    family_yaml = (
        repo_copy
        / "scenario_families"
        / "report-cli-markdown-evolution"
        / "family.yaml"
    )
    family_yaml.write_text(
        family_yaml.read_text(encoding="utf-8").replace(
            "import pytest;",
            "import os;",
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetPackError, match="import installed pytest"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_untrusted_ci_runner(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    ci_runner = (
        repo_copy
        / "scenario_families"
        / "ci-config-coverage-drift"
        / "variants"
        / "inventory-gate"
        / "repo"
        / "scripts"
        / "run_ci.py"
    )
    ci_runner.write_text(
        ci_runner.read_text(encoding="utf-8").replace(
            'subprocess.call([sys.executable, "-c", runner])',
            'subprocess.call([sys.executable, "-m", "pytest", "-q"])',
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetPackError, match="import installed pytest"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_allows_verify_sh_to_execute_milestone_helpers(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        'source /verifier/milestones/m1_markdown_rendered.sh',
        'bash /verifier/milestones/m1_markdown_rendered.sh',
        1,
    )
    verify_text = verify_text.replace(
        'source /verifier/milestones/m2_usage_doc_updated.sh',
        '. /verifier/milestones/m2_usage_doc_updated.sh',
        1,
    )
    verify_path.write_text(verify_text, encoding="utf-8")

    summary = validate_authored_asset_pack(repo_copy)

    assert summary.family_count == 5


def test_validate_authored_pack_rejects_missing_functional_check_result_consumption(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        '        check_phase2_pytest_suite() {\n'
        '          [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]\n'
        '        }\n\n',
        "",
        1,
    )
    verify_text = verify_text.replace(
        '        if ! check_phase2_pytest_suite; then\n'
        '          add_error "Phase 2 pytest suite did not pass"\n'
        '        fi\n\n',
        "",
        1,
    )
    verify_path.write_text(verify_text, encoding="utf-8")

    with pytest.raises(AssetPackError, match="must consume the trusted functional check result"):
        validate_authored_asset_pack(repo_copy)
