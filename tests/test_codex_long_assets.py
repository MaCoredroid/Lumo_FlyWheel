from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from lumo_flywheel_serving.codex_long_assets import (
    AssetPackError,
    _uses_trusted_pytest_entrypoint,
    validate_authored_asset_pack,
)


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
        'if check_m1_cli_markdown "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then\n'
        "  write_result '.milestones.m1_cli_markdown = true'\n"
        "else\n"
        '  add_error "CLI still does not expose markdown output"\n'
        "fi\n",
        'check_m1_cli_markdown "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"\n'
        "write_result '.milestones.m1_cli_markdown = true'\n",
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


def test_validate_authored_pack_rejects_comment_only_functional_check_consumption(tmp_path: Path) -> None:
    repo_copy = tmp_path / "repo"
    shutil.copytree(REPO_ROOT, repo_copy)

    verify_path = repo_copy / "verifiers" / "report-cli-markdown-evolution" / "verify.sh"
    verify_text = verify_path.read_text(encoding="utf-8")
    verify_text = verify_text.replace(
        '        check_phase2_pytest_suite() {\n'
        '          [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]\n'
        '        }\n',
        '        check_phase2_pytest_suite() {\n'
        '          # pytest_suite_exit_code is checked elsewhere\n'
        '          return 0\n'
        '        }\n',
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
        '        check_phase2_pytest_suite() {\n'
        '          [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]\n'
        '        }\n',
        '        check_phase2_pytest_suite() {\n'
        '          printf "%s\\n" "pytest_suite_exit_code"\n'
        '          return 0\n'
        '        }\n',
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
        '        check_phase2_pytest_suite() {\n'
        '          [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]\n'
        '        }\n',
        '        check_phase2_pytest_suite() {\n'
        '          return 0\n'
        '        }\n'
        '        check_unused_pytest_suite() {\n'
        '          [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]\n'
        '        }\n',
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
        "generated_from": "verifier_data/report-cli-markdown-evolution/inventory-ops/red_team/",
        "min_exploits": 5,
        "mutation_score_floor": 0.85,
    }
    payload["difficulty_estimate"] = {
        "evidence_path": "verifier_data/report-cli-markdown-evolution/inventory-ops/calibration.json",
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
