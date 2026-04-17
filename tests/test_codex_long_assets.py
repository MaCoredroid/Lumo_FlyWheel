from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from lumo_flywheel_serving.codex_long_assets import (
    AssetPackError,
    _hidden_test_node_to_path,
    _uses_trusted_pytest_entrypoint,
    validate_authored_asset_pack,
)
from lumo_flywheel_serving.task_orchestrator import get_variant_quality_contract


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


def test_report_cli_family_exposes_quality_contracts_for_all_variants() -> None:
    family_yaml = (
        REPO_ROOT
        / "scenario_families"
        / "report-cli-markdown-evolution"
        / "family.yaml"
    )
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))

    inventory_contract = get_variant_quality_contract(payload, "inventory-ops")
    incident_contract = get_variant_quality_contract(payload, "incident-triage")
    release_contract = get_variant_quality_contract(payload, "release-readiness")

    assert payload["grading_invariant"]["type"] == "hybrid"
    assert inventory_contract["tier"] == "small-investigative"
    assert inventory_contract["oracle"]["path"] == "oracle/solution.patch"
    assert inventory_contract["oracle"]["followup_path"] == "oracle/solution_followup.patch"
    assert inventory_contract["hidden_tests"]["path"] == (
        "verifier_data/report-cli-markdown-evolution/inventory-ops/hidden_tests"
    )
    assert inventory_contract["hidden_tests"]["entrypoint"] == "test_example_based.py"
    assert inventory_contract["red_team"]["path"] == (
        "verifier_data/report-cli-markdown-evolution/inventory-ops/red_team"
    )
    assert inventory_contract["red_team"]["exploits_required"] == 6
    assert incident_contract["tier"] == "standard"
    assert incident_contract["oracle"]["path"] == "oracle/solution.patch"
    assert incident_contract["oracle"]["followup_path"] == "oracle/solution_followup.patch"
    assert incident_contract["hidden_tests"]["path"] == (
        "verifier_data/report-cli-markdown-evolution/incident-triage/hidden_tests"
    )
    assert incident_contract["hidden_tests"]["entrypoint"] == "test_example_based.py"
    assert incident_contract["red_team"]["path"] == (
        "verifier_data/report-cli-markdown-evolution/incident-triage/red_team"
    )
    assert incident_contract["red_team"]["exploits_required"] == 6
    assert release_contract["tier"] == "pro"
    assert release_contract["oracle"]["path"] == "oracle/solution.patch"
    assert release_contract["oracle"]["followup_path"] == "oracle/solution_followup.patch"
    assert release_contract["hidden_tests"]["path"] == (
        "verifier_data/report-cli-markdown-evolution/release-readiness/hidden_tests"
    )
    assert release_contract["hidden_tests"]["entrypoint"] == "test_example_based.py"
    assert release_contract["red_team"]["path"] == (
        "verifier_data/report-cli-markdown-evolution/release-readiness/red_team"
    )
    assert release_contract["red_team"]["exploits_required"] == 6


def test_normalizer_family_exposes_mixed_quality_contract_for_billing_ledger() -> None:
    family_yaml = (
        REPO_ROOT
        / "scenario_families"
        / "normalizer-api-migration"
        / "family.yaml"
    )
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))

    alert_contract = get_variant_quality_contract(payload, "alert-routing")
    billing_contract = get_variant_quality_contract(payload, "billing-ledger")
    catalog_contract = get_variant_quality_contract(payload, "catalog-sync")

    assert payload["grading_invariant"]["type"] == "hybrid"
    assert alert_contract["hidden_tests"] == {}
    assert alert_contract["red_team"] == {}
    assert billing_contract["tier"] == "standard"
    assert billing_contract["oracle"]["path"] == "oracle/solution.patch"
    assert billing_contract["oracle"]["followup_path"] == "oracle/solution_followup.patch"
    assert billing_contract["hidden_tests"]["path"] == (
        "verifier_data/normalizer-api-migration/billing-ledger/hidden_tests"
    )
    assert billing_contract["hidden_tests"]["entrypoint"] == "test_example_based.py"
    assert billing_contract["red_team"]["path"] == (
        "verifier_data/normalizer-api-migration/billing-ledger/red_team"
    )
    assert billing_contract["red_team"]["exploits_required"] == 6
    assert catalog_contract["hidden_tests"] == {}
    assert catalog_contract["red_team"] == {}


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


def test_uses_trusted_pytest_entrypoint_rejects_dead_string_reference() -> None:
    command = (
        "cd /workspace && "
        "echo 'import pathlib, sys; import pytest; sys.path.insert(0, cwd); pytest.main([\"-q\"])' >/tmp/hint && "
        "python -c 'import os; raise SystemExit(0)'"
    )

    assert _uses_trusted_pytest_entrypoint(command) is False


def test_uses_trusted_pytest_entrypoint_accepts_python_c_payload() -> None:
    command = (
        "cd /workspace && "
        "python -c 'import pathlib, sys; cwd=str(pathlib.Path.cwd()); "
        "sys.path=[p for p in sys.path if p not in (\"\", cwd)]; import pytest; "
        "sys.path.insert(0, cwd); raise SystemExit(pytest.main([\"-q\"]))'"
    )

    assert _uses_trusted_pytest_entrypoint(command) is True


def test_uses_trusted_pytest_entrypoint_accepts_trusted_grader_pytest() -> None:
    command = "/grader/venv/bin/python -m pytest /verifier_data/hidden_tests -q"

    assert _uses_trusted_pytest_entrypoint(command) is True


def test_hidden_test_node_to_path_rejects_path_traversal(tmp_path: Path) -> None:
    hidden_tests_dir = tmp_path / "hidden_tests"
    hidden_tests_dir.mkdir()

    with pytest.raises(ValueError, match="must not escape hidden_tests"):
        _hidden_test_node_to_path(
            hidden_tests_dir,
            "tests/hidden/../../red_team/run_all.sh::test_round_one_green",
        )


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

    with pytest.raises(AssetPackError, match="trusted subprocess.call pytest path"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_ci_runner_early_success_shortcut(tmp_path: Path) -> None:
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
            'def main(argv: list[str] | None = None) -> int:\n',
            'def main(argv: list[str] | None = None) -> int:\n'
            '    if Path("pyproject.toml").exists():\n'
            '        return 0\n',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetPackError, match="early success shortcuts"):
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


def test_validate_authored_pack_rejects_comment_only_milestone_source(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        "source /verifier/milestones/m1_cli_markdown.sh",
        "# source /verifier/milestones/m1_cli_markdown.sh",
        1,
    )
    verify_path.write_text(verify_text, encoding="utf-8")

    with pytest.raises(AssetPackError, match="source and invoke milestone helper 'm1_cli_markdown'"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_comment_only_milestone_invocation(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        'if check_m1_cli_markdown "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then',
        '# check_m1_cli_markdown "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"\nif true; then',
        1,
    )
    verify_path.write_text(verify_text, encoding="utf-8")

    with pytest.raises(AssetPackError, match="source and invoke milestone helper 'm1_cli_markdown'"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_shadowed_milestone_helper(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        'source /verifier/milestones/m1_cli_markdown.sh\n',
        'source /verifier/milestones/m1_cli_markdown.sh\n'
        'check_m1_cli_markdown() {\n'
        '  return 0\n'
        '}\n',
        1,
    )
    verify_path.write_text(verify_text, encoding="utf-8")

    with pytest.raises(AssetPackError, match="source and invoke milestone helper 'm1_cli_markdown'"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_milestone_helper_called_without_gating_result(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        '  if check_m1_cli_markdown "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then\n'
        "    write_result '.milestones.m1_cli_markdown = true'\n"
        "  else\n"
        '    add_error "CLI still does not expose markdown output"\n'
        "  fi\n",
        '  check_m1_cli_markdown "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"\n'
        "  write_result '.milestones.m1_cli_markdown = true'\n",
        1,
    )
    verify_path.write_text(verify_text, encoding="utf-8")

    with pytest.raises(AssetPackError, match="source and invoke milestone helper 'm1_cli_markdown'"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_missing_functional_check_result_consumption(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        'check_phase2_pytest_suite() {\n'
        '  [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]\n'
        '}\n\n',
        "",
        1,
    )
    verify_text = verify_text.replace(
        'if ! check_phase2_pytest_suite; then\n'
        '  add_error "Phase 2 pytest suite did not pass"\n'
        'fi\n\n',
        "",
        1,
    )
    verify_path.write_text(verify_text, encoding="utf-8")

    with pytest.raises(AssetPackError, match="must consume the trusted functional check result"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_comment_only_functional_check_consumption(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        'check_phase2_pytest_suite() {\n'
        '  [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]\n'
        '}\n',
        'check_phase2_pytest_suite() {\n'
        '  # pytest_suite_exit_code is checked elsewhere\n'
        '  return 0\n'
        '}\n',
        1,
    )
    verify_path.write_text(verify_text, encoding="utf-8")

    with pytest.raises(AssetPackError, match="must consume the trusted functional check result"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_dead_string_functional_check_reference(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        'check_phase2_pytest_suite() {\n'
        '  [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]\n'
        '}\n',
        'check_phase2_pytest_suite() {\n'
        '  printf "%s\\n" "pytest_suite_exit_code"\n'
        '  return 0\n'
        '}\n',
        1,
    )
    verify_path.write_text(verify_text, encoding="utf-8")

    with pytest.raises(AssetPackError, match="must consume the trusted functional check result"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_unused_functional_check_reader(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        'check_phase2_pytest_suite() {\n'
        '  [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]\n'
        '}\n',
        'check_phase2_pytest_suite() {\n'
        '  return 0\n'
        '}\n'
        'check_unused_pytest_suite() {\n'
        '  [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]\n'
        '}\n',
        1,
    )
    verify_path.write_text(verify_text, encoding="utf-8")

    with pytest.raises(AssetPackError, match="must consume the trusted functional check result"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_requires_multi_step_milestones(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    family_yaml = repo_copy / "scenario_families" / "owner-field-cross-layer" / "family.yaml"
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))
    payload["milestones"] = payload["milestones"][:2]
    family_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(AssetPackError, match="at least 3 milestones"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_requires_multiple_breakage_surfaces(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    family_yaml = repo_copy / "scenario_families" / "owner-field-cross-layer" / "family.yaml"
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))
    payload["breakage_class"]["surfaces"] = ["store_layer"]
    family_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(AssetPackError, match="at least 3 breakage surfaces"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_shallow_cross_layer_variant_layout(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    docs_path = (
        repo_copy
        / "scenario_families"
        / "owner-field-cross-layer"
        / "variants"
        / "project-board"
        / "repo"
        / "docs"
        / "cli.md"
    )
    docs_path.unlink()

    with pytest.raises(AssetPackError, match="cross-layer coverage"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_shallow_ci_variant_layout(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    workflow_path = (
        repo_copy
        / "scenario_families"
        / "ci-config-coverage-drift"
        / "variants"
        / "inventory-gate"
        / "repo"
        / ".github"
        / "workflows"
        / "ci.yml"
    )
    workflow_path.unlink()

    with pytest.raises(AssetPackError, match="CI drift coverage"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_accepts_modern_variant_quality_assets(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    family_yaml = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "family.yaml"
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))
    payload["grading_invariant"]["type"] = "hybrid"
    payload["grading_invariant"]["functional_checks"] = [
        {
            "id": "hidden_round1",
            "command": "/grader/venv/bin/python -m pytest /verifier_data/hidden_tests -q",
            "description": "Run hidden tests from the trusted grader image",
            "timeout_seconds": 180,
        }
    ]
    payload["grading_invariant"]["expected_final_state"] = [
        {"oracle_parity_check": "Agent output matches the trusted oracle on hidden fixtures."}
    ]
    payload["milestones"] = [
        {
            "id": "m1_cli_markdown",
            "description": "CLI hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.2,
            "pass_rule": "all",
        },
        {
            "id": "m2_renderer_markdown",
            "description": "Renderer property tests pass",
            "test_nodes": ["tests/hidden/test_property.py::test_markdown_property"],
            "partial_credit": 0.35,
            "pass_rule": "all",
        },
        {
            "id": "m3_docs_updated",
            "description": "Follow-up hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.45,
            "pass_rule": "any",
        },
    ]
    payload["shortcut_resistance"] = {
        "generated_from": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team/",
        "min_exploits": 5,
        "mutation_score_floor": 0.85,
    }
    payload["difficulty_estimate"] = {
        "evidence_path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }

    for variant in payload["variants"]:
        variant_id = variant["variant_id"]
        variant["tier"] = "standard"
        variant["surfaces"] = ["cli", "renderer", "docs"]
        variant["oracle"] = {
            "path": "oracle/solution.patch",
            "followup_path": "oracle/solution_followup.patch",
            "source_commit": "abc1234",
        }
        variant["hidden_tests"] = {
            "path": f"verifier_data/report-cli-markdown-evolution/{variant_id}/hidden_tests",
            "entrypoint": "test_example.py",
            "milestone_map": {
                "m1_cli_markdown": ["tests/hidden/test_example.py::test_cli_markdown"],
                "m3_docs_updated": ["tests/hidden/test_followup.py::*"],
            },
        }
        variant["red_team"] = {
            "path": f"verifier_data/report-cli-markdown-evolution/{variant_id}/red_team",
            "exploits_required": 5,
        }

        variant_dir = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "variants" / variant_id
        hidden_tests_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "hidden_tests"
        red_team_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "red_team"
        hidden_tests_dir.mkdir(parents=True, exist_ok=True)
        red_team_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "oracle").mkdir(parents=True, exist_ok=True)

        (hidden_tests_dir / "test_example.py").write_text("def test_cli_markdown():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_property.py").write_text("def test_markdown_property():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_followup.py").write_text("def test_followup_round():\n    pass\n", encoding="utf-8")
        (red_team_dir / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (
            repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "calibration.json"
        ).write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")
        (variant_dir / "oracle" / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (variant_dir / "oracle" / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")

    family_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (repo_copy / "verifier_data" / "report-cli-markdown-evolution" / "variant_expectations.json").unlink()

    summary = validate_authored_asset_pack(repo_copy)

    assert summary.family_count == 5


def test_validate_authored_pack_accepts_family_level_template_quality_assets(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    family_yaml = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "family.yaml"
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))
    payload["grading_invariant"]["type"] = "hybrid"
    payload["grading_invariant"]["functional_checks"] = [
        {
            "id": "hidden_round1",
            "command": "/grader/venv/bin/python -m pytest /verifier_data/hidden_tests -q",
            "description": "Run hidden tests from the trusted grader image",
            "timeout_seconds": 180,
        }
    ]
    payload["grading_invariant"]["expected_final_state"] = [
        {"oracle_parity_check": "Agent output matches the trusted oracle on hidden fixtures."}
    ]
    payload["milestones"] = [
        {
            "id": "m1_cli_markdown",
            "description": "CLI hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.2,
            "pass_rule": "all",
        },
        {
            "id": "m2_renderer_markdown",
            "description": "Renderer property tests pass",
            "test_nodes": ["tests/hidden/test_property.py::test_markdown_property"],
            "partial_credit": 0.35,
            "pass_rule": "all",
        },
        {
            "id": "m3_docs_updated",
            "description": "Follow-up hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.45,
            "pass_rule": "any",
        },
    ]
    payload["hidden_tests"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/hidden_tests",
        "entrypoint": "test_example.py",
    }
    payload["red_team"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team",
        "exploits_required": 5,
    }
    payload["calibration"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }
    payload["shortcut_resistance"] = {
        "generated_from": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team/",
        "min_exploits": 5,
        "mutation_score_floor": 0.85,
    }
    payload["difficulty_estimate"] = {
        "evidence_path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }

    for variant in payload["variants"]:
        variant_id = variant["variant_id"]
        variant["tier"] = "standard"
        variant["surfaces"] = ["cli", "renderer", "docs"]
        variant["oracle"] = {
            "path": "oracle/solution.patch",
            "followup_path": "oracle/solution_followup.patch",
            "source_commit": "abc1234",
        }
        variant["hidden_tests"] = {
            "milestone_map": {
                "m1_cli_markdown": ["tests/hidden/test_example.py::test_cli_markdown"],
                "m3_docs_updated": ["tests/hidden/test_followup.py::*"],
            },
        }

        variant_dir = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "variants" / variant_id
        hidden_tests_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "hidden_tests"
        red_team_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "red_team"
        hidden_tests_dir.mkdir(parents=True, exist_ok=True)
        red_team_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "oracle").mkdir(parents=True, exist_ok=True)

        (hidden_tests_dir / "test_example.py").write_text("def test_cli_markdown():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_property.py").write_text("def test_markdown_property():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_followup.py").write_text("def test_followup_round():\n    pass\n", encoding="utf-8")
        (red_team_dir / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (
            repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "calibration.json"
        ).write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")
        (variant_dir / "oracle" / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (variant_dir / "oracle" / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")

    family_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (repo_copy / "verifier_data" / "report-cli-markdown-evolution" / "variant_expectations.json").unlink()

    summary = validate_authored_asset_pack(repo_copy)

    assert summary.family_count == 5


def test_validate_authored_pack_accepts_family_level_template_oracle_assets(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    family_yaml = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "family.yaml"
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))
    payload["grading_invariant"]["type"] = "hybrid"
    payload["grading_invariant"]["functional_checks"] = [
        {
            "id": "hidden_round1",
            "command": "/grader/venv/bin/python -m pytest /verifier_data/hidden_tests -q",
            "description": "Run hidden tests from the trusted grader image",
            "timeout_seconds": 180,
        }
    ]
    payload["grading_invariant"]["expected_final_state"] = [
        {"oracle_parity_check": "Agent output matches the trusted oracle on hidden fixtures."}
    ]
    payload["milestones"] = [
        {
            "id": "m1_cli_markdown",
            "description": "CLI hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.2,
            "pass_rule": "all",
        },
        {
            "id": "m2_renderer_markdown",
            "description": "Renderer property tests pass",
            "test_nodes": ["tests/hidden/test_property.py::test_markdown_property"],
            "partial_credit": 0.35,
            "pass_rule": "all",
        },
        {
            "id": "m3_docs_updated",
            "description": "Follow-up hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.45,
            "pass_rule": "any",
        },
    ]
    payload["oracle"] = {
        "path": "oracle/<variant_id>/solution.patch",
        "followup_path": "oracle/<variant_id>/solution_followup.patch",
        "source_commit": "abc1234",
    }
    payload["hidden_tests"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/hidden_tests",
        "entrypoint": "test_example.py",
    }
    payload["red_team"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team",
        "exploits_required": 5,
    }
    payload["calibration"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }
    payload["shortcut_resistance"] = {
        "generated_from": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team/",
        "min_exploits": 5,
        "mutation_score_floor": 0.85,
    }
    payload["difficulty_estimate"] = {
        "evidence_path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }

    for variant in payload["variants"]:
        variant_id = variant["variant_id"]
        variant["tier"] = "standard"
        variant["surfaces"] = ["cli", "renderer", "docs"]
        variant["hidden_tests"] = {
            "milestone_map": {
                "m1_cli_markdown": ["tests/hidden/test_example.py::test_cli_markdown"],
                "m3_docs_updated": ["tests/hidden/test_followup.py::*"],
            },
        }

        variant_dir = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "variants" / variant_id
        hidden_tests_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "hidden_tests"
        red_team_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "red_team"
        oracle_dir = variant_dir / "oracle" / variant_id
        hidden_tests_dir.mkdir(parents=True, exist_ok=True)
        red_team_dir.mkdir(parents=True, exist_ok=True)
        oracle_dir.mkdir(parents=True, exist_ok=True)

        (hidden_tests_dir / "test_example.py").write_text("def test_cli_markdown():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_property.py").write_text("def test_markdown_property():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_followup.py").write_text("def test_followup_round():\n    pass\n", encoding="utf-8")
        (red_team_dir / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (
            repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "calibration.json"
        ).write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")
        (oracle_dir / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (oracle_dir / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")

    family_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (repo_copy / "verifier_data" / "report-cli-markdown-evolution" / "variant_expectations.json").unlink()

    summary = validate_authored_asset_pack(repo_copy)

    assert summary.family_count == 5


def test_validate_authored_pack_rejects_missing_interactive_repo_brief_source(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    family_yaml = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "family.yaml"
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))
    payload["grading_invariant"]["type"] = "hybrid"
    payload["grading_invariant"]["functional_checks"] = [
        {
            "id": "hidden_round1",
            "command": "/grader/venv/bin/python -m pytest /verifier_data/hidden_tests -q",
            "description": "Run hidden tests from the trusted grader image",
            "timeout_seconds": 180,
        }
    ]
    payload["grading_invariant"]["expected_final_state"] = [
        {"oracle_parity_check": "Agent output matches the trusted oracle on hidden fixtures."}
    ]
    payload["milestones"] = [
        {
            "id": "m1_cli_markdown",
            "description": "CLI hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.2,
            "pass_rule": "all",
        },
        {
            "id": "m2_renderer_markdown",
            "description": "Renderer property tests pass",
            "test_nodes": ["tests/hidden/test_property.py::test_markdown_property"],
            "partial_credit": 0.35,
            "pass_rule": "all",
        },
        {
            "id": "m3_docs_updated",
            "description": "Follow-up hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.45,
            "pass_rule": "any",
        },
    ]
    payload["interactive"] = {
        "rounds": 2,
        "round_1": {
            "brief_source": "repo/followup.md",
            "grader_between_rounds": "verifier_data/report-cli-markdown-evolution/<variant_id>/hidden_tests/test_example.py",
        },
        "round_2": {
            "brief_source": "verifier_data/report-cli-markdown-evolution/<variant_id>/followup/brief.md",
            "inject_timing": "after_round_1_passes",
            "inject_mechanism": "append_to_AGENTS_md",
        },
    }
    payload["hidden_tests"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/hidden_tests",
        "entrypoint": "test_example.py",
    }
    payload["red_team"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team",
        "exploits_required": 5,
    }
    payload["calibration"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }
    payload["shortcut_resistance"] = {
        "generated_from": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team/",
        "min_exploits": 5,
        "mutation_score_floor": 0.85,
    }
    payload["difficulty_estimate"] = {
        "evidence_path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }

    for variant in payload["variants"]:
        variant_id = variant["variant_id"]
        variant["tier"] = "standard"
        variant["surfaces"] = ["cli", "renderer", "docs"]
        variant["oracle"] = {
            "path": "oracle/solution.patch",
            "followup_path": "oracle/solution_followup.patch",
            "source_commit": "abc1234",
        }
        variant["hidden_tests"] = {
            "milestone_map": {
                "m1_cli_markdown": ["tests/hidden/test_example.py::test_cli_markdown"],
                "m3_docs_updated": ["tests/hidden/test_followup.py::*"],
            },
        }

        variant_dir = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "variants" / variant_id
        repo_dir = variant_dir / "repo"
        hidden_tests_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "hidden_tests"
        red_team_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "red_team"
        followup_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "followup"
        hidden_tests_dir.mkdir(parents=True, exist_ok=True)
        red_team_dir.mkdir(parents=True, exist_ok=True)
        followup_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "oracle").mkdir(parents=True, exist_ok=True)

        (repo_dir / "AGENTS.md").write_text("agent brief\n", encoding="utf-8")
        (hidden_tests_dir / "test_example.py").write_text("def test_cli_markdown():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_property.py").write_text("def test_markdown_property():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_followup.py").write_text("def test_followup_round():\n    pass\n", encoding="utf-8")
        (red_team_dir / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (followup_dir / "brief.md").write_text("follow up\n", encoding="utf-8")
        (
            repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "calibration.json"
        ).write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")
        (variant_dir / "oracle" / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (variant_dir / "oracle" / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")

    family_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (repo_copy / "verifier_data" / "report-cli-markdown-evolution" / "variant_expectations.json").unlink()

    with pytest.raises(AssetPackError, match="Interactive asset 'brief_source' must resolve to a file"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_missing_declared_template_evidence_asset(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    family_yaml = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "family.yaml"
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))
    payload["grading_invariant"]["type"] = "hybrid"
    payload["grading_invariant"]["functional_checks"] = [
        {
            "id": "hidden_round1",
            "command": "/grader/venv/bin/python -m pytest /verifier_data/hidden_tests -q",
            "description": "Run hidden tests from the trusted grader image",
            "timeout_seconds": 180,
        }
    ]
    payload["grading_invariant"]["expected_final_state"] = [
        {"oracle_parity_check": "Agent output matches the trusted oracle on hidden fixtures."}
    ]
    payload["milestones"] = [
        {
            "id": "m1_cli_markdown",
            "description": "CLI hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.2,
            "pass_rule": "all",
        },
        {
            "id": "m2_renderer_markdown",
            "description": "Renderer property tests pass",
            "test_nodes": ["tests/hidden/test_property.py::test_markdown_property"],
            "partial_credit": 0.35,
            "pass_rule": "all",
        },
        {
            "id": "m3_docs_updated",
            "description": "Follow-up hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.45,
            "pass_rule": "any",
        },
    ]
    payload["hidden_tests"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/hidden_tests",
        "entrypoint": "test_example.py",
    }
    payload["shortcut_resistance"] = {
        "generated_from": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team/",
        "min_exploits": 5,
        "mutation_score_floor": 0.85,
    }
    payload["difficulty_estimate"] = {
        "evidence_path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }

    for variant in payload["variants"]:
        variant_id = variant["variant_id"]
        variant["tier"] = "standard"
        variant["surfaces"] = ["cli", "renderer", "docs"]
        variant["oracle"] = {
            "path": "oracle/solution.patch",
            "followup_path": "oracle/solution_followup.patch",
            "source_commit": "abc1234",
        }
        variant["hidden_tests"] = {
            "milestone_map": {
                "m1_cli_markdown": ["tests/hidden/test_example.py::test_cli_markdown"],
                "m3_docs_updated": ["tests/hidden/test_followup.py::*"],
            },
        }

        variant_dir = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "variants" / variant_id
        hidden_tests_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "hidden_tests"
        red_team_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "red_team"
        hidden_tests_dir.mkdir(parents=True, exist_ok=True)
        red_team_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "oracle").mkdir(parents=True, exist_ok=True)

        (hidden_tests_dir / "test_example.py").write_text("def test_cli_markdown():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_property.py").write_text("def test_markdown_property():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_followup.py").write_text("def test_followup_round():\n    pass\n", encoding="utf-8")
        (red_team_dir / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        if variant_id != "inventory-ops":
            (
                repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "calibration.json"
            ).write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")
        (variant_dir / "oracle" / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (variant_dir / "oracle" / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")

    family_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (repo_copy / "verifier_data" / "report-cli-markdown-evolution" / "variant_expectations.json").unlink()

    with pytest.raises(AssetPackError, match="Missing declared verifier_data asset"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_directory_calibration_asset(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    family_yaml = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "family.yaml"
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))
    payload["grading_invariant"]["type"] = "hybrid"
    payload["grading_invariant"]["functional_checks"] = [
        {
            "id": "hidden_round1",
            "command": "/grader/venv/bin/python -m pytest /verifier_data/hidden_tests -q",
            "description": "Run hidden tests from the trusted grader image",
            "timeout_seconds": 180,
        }
    ]
    payload["grading_invariant"]["expected_final_state"] = [
        {"oracle_parity_check": "Agent output matches the trusted oracle on hidden fixtures."}
    ]
    payload["milestones"] = [
        {
            "id": "m1_cli_markdown",
            "description": "CLI hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.2,
            "pass_rule": "all",
        },
        {
            "id": "m2_renderer_markdown",
            "description": "Renderer property tests pass",
            "test_nodes": ["tests/hidden/test_property.py::test_markdown_property"],
            "partial_credit": 0.35,
            "pass_rule": "all",
        },
        {
            "id": "m3_docs_updated",
            "description": "Follow-up hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.45,
            "pass_rule": "any",
        },
    ]
    payload["hidden_tests"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/hidden_tests",
        "entrypoint": "test_example.py",
    }
    payload["red_team"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team",
        "exploits_required": 5,
    }
    payload["calibration"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }
    payload["shortcut_resistance"] = {
        "generated_from": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team/",
        "min_exploits": 5,
        "mutation_score_floor": 0.85,
    }
    payload["difficulty_estimate"] = {
        "evidence_path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }

    for variant in payload["variants"]:
        variant_id = variant["variant_id"]
        variant["tier"] = "standard"
        variant["surfaces"] = ["cli", "renderer", "docs"]
        variant["oracle"] = {
            "path": "oracle/solution.patch",
            "followup_path": "oracle/solution_followup.patch",
            "source_commit": "abc1234",
        }
        variant["hidden_tests"] = {
            "milestone_map": {
                "m1_cli_markdown": ["tests/hidden/test_example.py::test_cli_markdown"],
                "m3_docs_updated": ["tests/hidden/test_followup.py::*"],
            },
        }

        variant_dir = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "variants" / variant_id
        hidden_tests_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "hidden_tests"
        red_team_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "red_team"
        hidden_tests_dir.mkdir(parents=True, exist_ok=True)
        red_team_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "oracle").mkdir(parents=True, exist_ok=True)

        (hidden_tests_dir / "test_example.py").write_text("def test_cli_markdown():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_property.py").write_text("def test_markdown_property():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_followup.py").write_text("def test_followup_round():\n    pass\n", encoding="utf-8")
        (red_team_dir / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (
            repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "calibration.json"
        ).mkdir(parents=True, exist_ok=True)
        (variant_dir / "oracle" / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (variant_dir / "oracle" / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")

    family_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (repo_copy / "verifier_data" / "report-cli-markdown-evolution" / "variant_expectations.json").unlink()

    with pytest.raises(AssetPackError, match="Calibration asset must resolve to a file"):
        validate_authored_asset_pack(repo_copy)


def test_validate_authored_pack_rejects_family_level_template_path_traversal(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    family_yaml = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "family.yaml"
    payload = yaml.safe_load(family_yaml.read_text(encoding="utf-8"))
    payload["grading_invariant"]["type"] = "hybrid"
    payload["grading_invariant"]["functional_checks"] = [
        {
            "id": "hidden_round1",
            "command": "/grader/venv/bin/python -m pytest /verifier_data/hidden_tests -q",
            "description": "Run hidden tests from the trusted grader image",
            "timeout_seconds": 180,
        }
    ]
    payload["grading_invariant"]["expected_final_state"] = [
        {"oracle_parity_check": "Agent output matches the trusted oracle on hidden fixtures."}
    ]
    payload["milestones"] = [
        {
            "id": "m1_cli_markdown",
            "description": "CLI hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.2,
            "pass_rule": "all",
        },
        {
            "id": "m2_renderer_markdown",
            "description": "Renderer property tests pass",
            "test_nodes": ["tests/hidden/test_property.py::test_markdown_property"],
            "partial_credit": 0.35,
            "pass_rule": "all",
        },
        {
            "id": "m3_docs_updated",
            "description": "Follow-up hidden tests pass",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.45,
            "pass_rule": "any",
        },
    ]
    payload["hidden_tests"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/../incident-triage/hidden_tests",
        "entrypoint": "test_example.py",
    }
    payload["red_team"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team",
        "exploits_required": 5,
    }
    payload["calibration"] = {
        "path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }
    payload["shortcut_resistance"] = {
        "generated_from": "verifier_data/report-cli-markdown-evolution/<variant_id>/red_team/",
        "min_exploits": 5,
        "mutation_score_floor": 0.85,
    }
    payload["difficulty_estimate"] = {
        "evidence_path": "verifier_data/report-cli-markdown-evolution/<variant_id>/calibration.json",
    }

    for variant in payload["variants"]:
        variant_id = variant["variant_id"]
        variant["tier"] = "standard"
        variant["surfaces"] = ["cli", "renderer", "docs"]
        variant["oracle"] = {
            "path": "oracle/solution.patch",
            "followup_path": "oracle/solution_followup.patch",
            "source_commit": "abc1234",
        }
        variant["hidden_tests"] = {
            "milestone_map": {
                "m1_cli_markdown": ["tests/hidden/test_example.py::test_cli_markdown"],
                "m3_docs_updated": ["tests/hidden/test_followup.py::*"],
            },
        }

        variant_dir = repo_copy / "scenario_families" / "report-cli-markdown-evolution" / "variants" / variant_id
        hidden_tests_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "hidden_tests"
        red_team_dir = repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "red_team"
        hidden_tests_dir.mkdir(parents=True, exist_ok=True)
        red_team_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "oracle").mkdir(parents=True, exist_ok=True)

        (hidden_tests_dir / "test_example.py").write_text("def test_cli_markdown():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_property.py").write_text("def test_markdown_property():\n    pass\n", encoding="utf-8")
        (hidden_tests_dir / "test_followup.py").write_text("def test_followup_round():\n    pass\n", encoding="utf-8")
        (red_team_dir / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (
            repo_copy / "verifier_data" / "report-cli-markdown-evolution" / variant_id / "calibration.json"
        ).write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")
        (variant_dir / "oracle" / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (variant_dir / "oracle" / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")

    family_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (repo_copy / "verifier_data" / "report-cli-markdown-evolution" / "variant_expectations.json").unlink()

    with pytest.raises(AssetPackError, match="must define variants\\[0\\]\\.hidden_tests\\.path"):
        validate_authored_asset_pack(repo_copy)
