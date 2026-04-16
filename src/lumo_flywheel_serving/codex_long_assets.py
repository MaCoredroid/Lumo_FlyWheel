from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from .data_pool import MIN_CODEX_LONG_FAMILIES, SCENARIO_TYPES
from .task_orchestrator import (
    ManifestMismatchError,
    TaskSpec,
    _render_variant_path_template,
    collect_declared_verifier_data_paths,
    get_variant_quality_contract,
    resolve_milestone_test_nodes,
    validate_family_spec,
)
from .yaml_utils import load_yaml_file


class AssetPackError(RuntimeError):
    """Raised when authored Codex-Long assets drift from the repo contract."""


@dataclass(frozen=True)
class AssetPackSummary:
    repo_root: Path
    family_count: int
    variant_count: int
    family_ids: tuple[str, ...]
    scenario_types: tuple[str, ...]
    has_freeze_artifacts: bool


def _checksum_manifest_for_dir(directory: Path) -> str:
    lines: list[str] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(directory.parent).as_posix()
        lines.append(f"{digest}  ./{rel}")
    return "\n".join(lines) + "\n"


def _uses_trusted_pytest_entrypoint(command: str) -> bool:
    normalized_command = _strip_shell_comments(command)
    if "pytest" not in normalized_command:
        return True
    trusted_grader_patterns = (
        "/grader/venv/bin/python -m pytest",
        "/grader/venv/bin/python3 -m pytest",
        "/grader/venv/bin/pytest",
    )
    if any(pattern in normalized_command for pattern in trusted_grader_patterns):
        return True
    if "python -m pytest" in normalized_command or "python3 -m pytest" in normalized_command:
        return False
    try:
        argv = shlex.split(normalized_command)
    except ValueError:
        return False
    code = None
    for index, token in enumerate(argv[:-2]):
        if token in {"python", "python3"} and argv[index + 1] == "-c":
            code = argv[index + 2]
            break
    if code is None:
        return False
    required = (
        "import pathlib, sys",
        "import pytest",
        'sys.path.insert(0, cwd)',
        'pytest.main(["-q"])',
    )
    return all(pattern in code for pattern in required)


def _is_phase2_only_expected_state(entry: object) -> bool:
    if isinstance(entry, str):
        combined = entry.lower()
    elif isinstance(entry, dict) and len(entry) == 1:
        key, value = next(iter(entry.items()))
        combined = f"{key} {value}".lower()
    else:
        return False
    markers = (
        "phase 2",
        "phase2",
        "functional check",
        "functional_checks",
        "exit 0",
        "round 1",
        "round1",
        "round 2",
        "round2",
        "pytest",
        "npm test",
        "cargo test",
        "go test",
        "gradle test",
        "mvn test",
        "hidden test",
        "hidden tests",
        "test suite passes",
        "tests pass",
    )
    return any(marker in combined for marker in markers)


def _strip_shell_comments(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line: list[str] = []
        in_single = False
        in_double = False
        for index, char in enumerate(raw_line):
            if char == "'" and not in_double:
                in_single = not in_single
                line.append(char)
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                line.append(char)
                continue
            if (
                char == "#"
                and not in_single
                and not in_double
                and (index == 0 or raw_line[index - 1].isspace())
            ):
                break
            line.append(char)
        lines.append("".join(line))
    return "\n".join(lines)


def _verify_references_milestone_helper(
    verify_text: str,
    milestone_id: str,
    helper_text: str,
) -> bool:
    normalized_verify = _strip_shell_comments(verify_text)
    normalized_helper = _strip_shell_comments(helper_text)
    helper_path = f"/verifier/milestones/{milestone_id}.sh"
    allowed_invocations = (
        f"source {helper_path}",
        f". {helper_path}",
        f"bash {helper_path}",
        f"sh {helper_path}",
        helper_path,
    )
    helper_function = f"check_{milestone_id}"
    helper_is_sourced = any(invocation in normalized_verify for invocation in allowed_invocations)
    helper_function_pattern = rf"(^|\n)\s*(?:function\s+)?{re.escape(helper_function)}\s*\("
    helper_defines_check = re.search(helper_function_pattern, normalized_helper) is not None
    helper_is_shadowed = re.search(helper_function_pattern, normalized_verify) is not None
    helper_result_is_gated = re.search(
        rf"""
        if\s+{re.escape(helper_function)}\b[^\n]*;\s*then
        (?:(?!\nfi\b).)*?
        write_result\s+['"]\.milestones\.{re.escape(milestone_id)}\s*=\s*true['"]
        (?:(?!\nfi\b).)*?
        \n\s*else\b
        (?:(?!\nfi\b).)*?
        \badd_error\b
        """,
        normalized_verify,
        flags=re.DOTALL | re.VERBOSE,
    ) is not None
    helper_is_invoked = re.search(
        rf"\b{re.escape(helper_function)}\b",
        normalized_verify,
    ) is not None
    return (
        helper_is_sourced
        and helper_defines_check
        and helper_is_invoked
        and helper_result_is_gated
        and not helper_is_shadowed
    )


def _verify_references_functional_check_result(
    verify_text: str,
    helper_texts: list[str],
    check_id: str,
) -> bool:
    needle = f"{check_id}_exit_code"
    normalized_verify = _strip_shell_comments(verify_text)
    normalized_helpers = [_strip_shell_comments(helper_text) for helper_text in helper_texts]
    path_pattern = rf'"[^"\n]*{re.escape(needle)}"'
    read_patterns = (
        rf"\bcat\s+{path_pattern}",
        rf"\bread\s+[A-Za-z_][A-Za-z0-9_]*\s*<\s*{path_pattern}",
        rf"\$\(\s*<\s*{path_pattern}\s*\)",
        rf"\bgrep\b[^\n]*{path_pattern}",
    )
    function_pattern = re.compile(
        r"(^|\n)\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{(?P<body>.*?)\n\s*\}",
        flags=re.DOTALL,
    )

    def reader_functions(text: str) -> set[str]:
        names: set[str] = set()
        for match in function_pattern.finditer(text):
            body = match.group("body")
            if any(re.search(pattern, body) for pattern in read_patterns):
                names.add(match.group(2))
        return names

    def gated_function_call(function_name: str) -> bool:
        negative_gate = re.search(
            rf"""
            if\s+!\s*{re.escape(function_name)}\b[^\n]*;\s*then
            (?:(?!\nfi\b).)*?
            \badd_error\b
            """,
            normalized_verify,
            flags=re.DOTALL | re.VERBOSE,
        ) is not None
        positive_else_gate = re.search(
            rf"""
            if\s+{re.escape(function_name)}\b[^\n]*;\s*then
            (?:(?!\nfi\b).)*?
            \n\s*else\b
            (?:(?!\nfi\b).)*?
            \badd_error\b
            """,
            normalized_verify,
            flags=re.DOTALL | re.VERBOSE,
        ) is not None
        return negative_gate or positive_else_gate

    reader_function_names = reader_functions(normalized_verify)
    for helper_text in normalized_helpers:
        reader_function_names.update(reader_functions(helper_text))

    return any(gated_function_call(function_name) for function_name in reader_function_names)


def _hidden_test_node_to_path(hidden_tests_dir: Path, test_node: str) -> Path:
    node_path = test_node.split("::", 1)[0]
    candidate = Path(node_path)
    if candidate.parts[:2] == ("tests", "hidden"):
        candidate = Path(*candidate.parts[2:])
    elif not candidate.parts:
        candidate = Path(candidate.name)
    return hidden_tests_dir / candidate


def _repo_source_files(repo_dir: Path) -> list[Path]:
    source_suffixes = {".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java"}
    files: list[Path] = []
    for path in repo_dir.rglob("*"):
        if not path.is_file() or path.suffix not in source_suffixes:
            continue
        rel = path.relative_to(repo_dir)
        if rel.parts and rel.parts[0] == "tests":
            continue
        files.append(path)
    return files


def _validate_variant_complexity(
    *,
    family_id: str,
    variant_id: str,
    scenario_type: str,
    repo_dir: Path,
) -> None:
    source_files = _repo_source_files(repo_dir)
    docs_files = [path for path in repo_dir.rglob("*") if path.is_file() and "docs" in path.relative_to(repo_dir).parts]
    test_files = [path for path in (repo_dir / "tests").rglob("*") if path.is_file()]
    workflow_files = [
        path
        for path in (repo_dir / ".github" / "workflows").rglob("*")
        if path.is_file()
    ]
    logs_files = [path for path in (repo_dir / "logs").rglob("*") if path.is_file()]
    config_files = [path for path in (repo_dir / "config").rglob("*") if path.is_file()]

    if scenario_type in {"feature_evolution", "migration_refactor", "cross_layer_changes"} and len(source_files) < 3:
        raise AssetPackError(
            f"Variant '{family_id}/{variant_id}' is too shallow for scenario_type='{scenario_type}': "
            "expected at least 3 non-test source files"
        )
    if scenario_type in {"feature_evolution", "migration_refactor", "investigate_then_fix", "cross_layer_changes"} and len(test_files) < 2:
        raise AssetPackError(
            f"Variant '{family_id}/{variant_id}' is too shallow for scenario_type='{scenario_type}': "
            "expected at least 2 test files"
        )
    if scenario_type == "feature_evolution" and not docs_files:
        raise AssetPackError(
            f"Variant '{family_id}/{variant_id}' must include docs/ content for feature-evolution benchmark coverage"
        )
    if scenario_type == "build_ci_breakage":
        if not (repo_dir / "Makefile").exists():
            raise AssetPackError(f"Variant '{family_id}/{variant_id}' must include Makefile for CI-contract coverage")
        if not (repo_dir / "scripts" / "run_ci.py").exists():
            raise AssetPackError(
                f"Variant '{family_id}/{variant_id}' must include scripts/run_ci.py for CI-contract coverage"
            )
        if not workflow_files:
            raise AssetPackError(
                f"Variant '{family_id}/{variant_id}' must include .github/workflows/ for CI drift coverage"
            )
    if scenario_type == "investigate_then_fix" and not logs_files:
        raise AssetPackError(
            f"Variant '{family_id}/{variant_id}' must include logs/ evidence for investigate-then-fix coverage"
        )
    if scenario_type == "cross_layer_changes":
        if not docs_files:
            raise AssetPackError(
                f"Variant '{family_id}/{variant_id}' must include docs/ content for cross-layer coverage"
            )
        if not config_files:
            raise AssetPackError(
                f"Variant '{family_id}/{variant_id}' must include config/ content for cross-layer coverage"
            )


def _is_sys_executable(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "executable"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _is_trusted_subprocess_call(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "call"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ):
        return False
    if len(node.args) != 1 or not isinstance(node.args[0], ast.List):
        return False
    args = node.args[0].elts
    if len(args) != 3:
        return False
    if not _is_sys_executable(args[0]):
        return False
    if not isinstance(args[1], ast.Constant) or args[1].value != "-c":
        return False
    return isinstance(args[2], ast.Name) and args[2].id == "runner"


def _uses_trusted_ci_runner_entrypoint(script_text: str) -> bool:
    required_patterns = (
        'sys.path=[p for p in sys.path if p not in ("", cwd)]',
        'import pytest',
        'pytest.main(["-q"])',
        'subprocess.call([sys.executable, "-c", runner])',
    )
    if any(pattern not in script_text for pattern in required_patterns):
        return False
    try:
        module = ast.parse(script_text)
    except SyntaxError:
        return False
    main_fn = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ),
        None,
    )
    if main_fn is None:
        return False
    returns = [node for node in ast.walk(main_fn) if isinstance(node, ast.Return)]
    if len(returns) != 2:
        return False
    has_guard_return = any(
        isinstance(node.value, ast.Constant) and node.value.value == 2
        for node in returns
    )
    has_trusted_return = any(_is_trusted_subprocess_call(node.value) for node in returns)
    return has_guard_return and has_trusted_return


def validate_authored_asset_pack(repo_root: str | Path) -> AssetPackSummary:
    root = Path(repo_root).resolve()
    scenario_families_dir = root / "scenario_families"
    verifiers_dir = root / "verifiers"
    verifier_data_dir = root / "verifier_data"
    split_assignment_path = root / "split_assignment.yaml"
    benchmark_manifest_path = root / "benchmark_manifest.lock"

    family_dirs = sorted(
        path
        for path in scenario_families_dir.iterdir()
        if path.is_dir()
    )
    if not family_dirs:
        raise AssetPackError("scenario_families/ does not contain any authored families")

    has_freeze_artifacts = split_assignment_path.exists() or benchmark_manifest_path.exists()
    if len(family_dirs) < MIN_CODEX_LONG_FAMILIES and has_freeze_artifacts:
        raise AssetPackError(
            "This repo carries fewer than the signed-off 35-family minimum, so "
            "split_assignment.yaml and benchmark_manifest.lock must not be present."
        )

    family_ids: list[str] = []
    scenario_types_seen: set[str] = set()
    variant_count = 0

    for family_dir in family_dirs:
        family_spec_path = family_dir / "family.yaml"
        if not family_spec_path.exists():
            raise AssetPackError(f"Missing family spec: {family_spec_path}")
        family_spec = load_yaml_file(family_spec_path) or {}
        if not isinstance(family_spec, dict):
            raise AssetPackError(f"Family spec must be a YAML mapping: {family_spec_path}")

        family_id = str(family_spec.get("family_id", ""))
        scenario_type = str(family_spec.get("scenario_type", ""))
        family_ids.append(family_id)
        scenario_types_seen.add(scenario_type)

        grading = family_spec.get("grading_invariant")
        if not isinstance(grading, dict):
            raise AssetPackError(f"grading_invariant must be a mapping: {family_spec_path}")
        breakage_class = family_spec.get("breakage_class")
        if not isinstance(breakage_class, dict):
            raise AssetPackError(f"breakage_class must be a mapping: {family_spec_path}")
        breakage_surfaces = breakage_class.get("surfaces")
        if not isinstance(breakage_surfaces, list):
            raise AssetPackError(f"breakage_class.surfaces must be a list: {family_spec_path}")
        if len(breakage_surfaces) < 3:
            raise AssetPackError(
                f"Family '{family_id}' must define at least 3 breakage surfaces so the benchmark stays reasoning-heavy"
            )
        functional_checks = grading.get("functional_checks")
        if not isinstance(functional_checks, list):
            raise AssetPackError(f"functional_checks must be a list: {family_spec_path}")
        for check in functional_checks:
            command = str(check.get("command", ""))
            if not _uses_trusted_pytest_entrypoint(command):
                raise AssetPackError(
                    "Functional check must import installed pytest before re-adding the workspace "
                    f"to sys.path: {family_spec_path}"
                )

        variants = family_spec.get("variants")
        if not isinstance(variants, list):
            raise AssetPackError(f"Family variants must be a list: {family_spec_path}")
        variant_ids = [str(variant.get("variant_id", "")) for variant in variants]
        variant_count += len(variant_ids)

        verifier_dir = verifiers_dir / family_id
        verify_path = verifier_dir / "verify.sh"
        if not verify_path.exists():
            raise AssetPackError(f"Missing verify.sh for family '{family_id}'")
        if not os.access(verify_path, os.X_OK):
            raise AssetPackError(f"verify.sh is not executable: {verify_path}")

        verify_text = verify_path.read_text(encoding="utf-8")
        if "/results/verify_result.json" not in verify_text:
            raise AssetPackError(f"verify.sh must write /results/verify_result.json: {verify_path}")

        milestones = family_spec.get("milestones")
        if not isinstance(milestones, list):
            raise AssetPackError(f"Family milestones must be a list: {family_spec_path}")
        if len(milestones) < 3:
            raise AssetPackError(
                f"Family '{family_id}' must define at least 3 milestones so the benchmark stays multi-step and non-trivial"
            )
        milestone_helper_texts: list[str] = []
        for milestone in milestones:
            milestone_id = str(milestone["id"])
            if milestone.get("check_script") is None:
                continue
            milestone_path = verifier_dir / "milestones" / f"{milestone_id}.sh"
            if not milestone_path.exists():
                raise AssetPackError(f"Missing milestone helper: {milestone_path}")
            if not os.access(milestone_path, os.X_OK):
                raise AssetPackError(f"Milestone helper is not executable: {milestone_path}")
            helper_text = milestone_path.read_text(encoding="utf-8")
            milestone_helper_texts.append(helper_text)
            if not _verify_references_milestone_helper(verify_text, milestone_id, helper_text):
                raise AssetPackError(
                    "verify.sh does not source and invoke milestone helper "
                    f"'{milestone_id}' for family '{family_id}'"
                )

        expected_final_state = grading.get("expected_final_state")
        if not isinstance(expected_final_state, list):
            raise AssetPackError(f"expected_final_state must be a list: {family_spec_path}")
        if any(_is_phase2_only_expected_state(entry) for entry in expected_final_state):
            for check in functional_checks:
                check_id = str(check.get("id", ""))
                if not _verify_references_functional_check_result(verify_text, milestone_helper_texts, check_id):
                    raise AssetPackError(
                    "verify.sh must consume the trusted functional check result "
                    f"for '{check_id}' in family '{family_id}'"
                    )

        expectations_path = verifier_data_dir / family_id / "variant_expectations.json"
        legacy_expectations: dict[str, object] = {}
        if expectations_path.exists():
            expectations = json.loads(expectations_path.read_text(encoding="utf-8"))
            expected_variants = expectations.get("variants")
            if not isinstance(expected_variants, dict):
                raise AssetPackError(f"variant_expectations.json must contain a 'variants' mapping: {expectations_path}")
            legacy_expectations = expected_variants

        for variant in variants:
            variant_id = str(variant["variant_id"])
            task = TaskSpec(
                track="codex_long",
                pool_or_split="train_long",
                scenario_id=f"{family_id}/{variant_id}",
                model_id="qwen3.5-27b",
                harness="codex",
                seed=1,
                family_id=family_id,
                variant_id=variant_id,
                image_digest="sha256:" + ("0" * 64),
                scenario_type=scenario_type,
                timeout_seconds=1,
            )
            try:
                validate_family_spec(task, family_spec)
            except ManifestMismatchError as exc:
                raise AssetPackError(str(exc)) from exc

            variant_dir = family_dir / "variants" / variant_id
            repo_dir = variant_dir / "repo"
            dockerfile_path = variant_dir / "Dockerfile"
            if not dockerfile_path.exists():
                raise AssetPackError(f"Missing Dockerfile for variant '{family_id}/{variant_id}'")
            dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
            for required in ("COPY repo/ /workspace/", "WORKDIR /workspace"):
                if required not in dockerfile_text:
                    raise AssetPackError(f"Dockerfile missing '{required}' for variant '{family_id}/{variant_id}'")
            for forbidden in ("verifiers", "verifier_data", "oracle"):
                if forbidden in dockerfile_text:
                    raise AssetPackError(
                        f"Dockerfile for '{family_id}/{variant_id}' unexpectedly references '{forbidden}'"
                    )

            agents_path = repo_dir / "AGENTS.md"
            if not agents_path.exists():
                raise AssetPackError(f"Missing AGENTS.md for variant '{family_id}/{variant_id}'")
            agents_text = agents_path.read_text(encoding="utf-8").lower()
            for forbidden in ("verifier", "oracle", "milestone"):
                if forbidden in agents_text:
                    raise AssetPackError(
                        f"AGENTS.md for '{family_id}/{variant_id}' leaks hidden grading detail '{forbidden}'"
                    )

            scenario_marker_path = repo_dir / ".scenario_variant"
            if scenario_marker_path.read_text(encoding="utf-8").strip() != variant_id:
                raise AssetPackError(f".scenario_variant mismatch for '{family_id}/{variant_id}'")

            tests_dir = repo_dir / "tests"
            if not tests_dir.exists():
                raise AssetPackError(f"Missing tests/ tree for variant '{family_id}/{variant_id}'")
            _validate_variant_complexity(
                family_id=family_id,
                variant_id=variant_id,
                scenario_type=scenario_type,
                repo_dir=repo_dir,
            )

            ci_runner_path = repo_dir / "scripts" / "run_ci.py"
            if ci_runner_path.exists():
                ci_runner_text = ci_runner_path.read_text(encoding="utf-8")
                if not _uses_trusted_ci_runner_entrypoint(ci_runner_text):
                    raise AssetPackError(
                        "Repo CI runner must use the guarded package-drift branch plus the trusted "
                        "subprocess.call pytest path without early success shortcuts "
                        f"for '{family_id}/{variant_id}'"
                    )

            declared_verifier_data_paths = collect_declared_verifier_data_paths(task, family_spec)
            for rel_path in declared_verifier_data_paths:
                resolved_path = root / rel_path
                if not resolved_path.exists():
                    raise AssetPackError(
                        f"Missing declared verifier_data asset for '{family_id}/{variant_id}': {resolved_path}"
                    )

            contract = get_variant_quality_contract(family_spec, variant_id)
            hidden_tests = contract.get("hidden_tests")
            if isinstance(hidden_tests, dict) and hidden_tests:
                hidden_tests_path = hidden_tests.get("path")
                if isinstance(hidden_tests_path, str) and hidden_tests_path.strip():
                    hidden_tests_dir = root / _render_variant_path_template(
                        hidden_tests_path,
                        family_id=family_id,
                        variant_id=variant_id,
                    )
                    if not hidden_tests_dir.exists():
                        raise AssetPackError(
                            f"Missing hidden_tests directory for '{family_id}/{variant_id}': {hidden_tests_dir}"
                        )
                    if hidden_tests.get("entrypoint") is not None:
                        entrypoint_path = hidden_tests_dir / str(hidden_tests["entrypoint"])
                        if not entrypoint_path.exists():
                            raise AssetPackError(
                                f"Missing hidden_tests entrypoint for '{family_id}/{variant_id}': {entrypoint_path}"
                            )
                    milestone_nodes = resolve_milestone_test_nodes(
                        family_spec,
                        family_id=family_id,
                        variant_id=variant_id,
                    )
                    for milestone_id, nodes in milestone_nodes.items():
                        for test_node in nodes:
                            test_path = _hidden_test_node_to_path(hidden_tests_dir, test_node)
                            if not test_path.exists():
                                raise AssetPackError(
                                    f"Hidden test node '{test_node}' for '{family_id}/{variant_id}/{milestone_id}' "
                                    f"does not resolve to an existing file under {hidden_tests_dir}"
                                )

            oracle = contract.get("oracle")
            if isinstance(oracle, dict) and oracle:
                for key in ("path", "followup_path"):
                    raw_path = oracle.get(key)
                    if raw_path is None:
                        continue
                    oracle_path = variant_dir / str(raw_path)
                    if not oracle_path.exists():
                        raise AssetPackError(
                            f"Missing oracle asset '{key}' for '{family_id}/{variant_id}': {oracle_path}"
                        )

            red_team = contract.get("red_team")
            if isinstance(red_team, dict) and red_team:
                red_team_path = red_team.get("path")
                if isinstance(red_team_path, str) and red_team_path.strip():
                    red_team_dir = root / _render_variant_path_template(
                        red_team_path,
                        family_id=family_id,
                        variant_id=variant_id,
                    )
                    if not red_team_dir.exists():
                        raise AssetPackError(
                            f"Missing red_team directory for '{family_id}/{variant_id}': {red_team_dir}"
                        )
                    if not (red_team_dir / "run_all.sh").exists():
                        raise AssetPackError(
                            f"red_team directory must include run_all.sh for '{family_id}/{variant_id}'"
                        )

            calibration = contract.get("calibration")
            if isinstance(calibration, dict) and calibration:
                calibration_path = calibration.get("path")
                if isinstance(calibration_path, str) and calibration_path.strip():
                    resolved_path = root / _render_variant_path_template(
                        calibration_path,
                        family_id=family_id,
                        variant_id=variant_id,
                    )
                    if not resolved_path.exists():
                        raise AssetPackError(
                            f"Missing calibration asset for '{family_id}/{variant_id}': {resolved_path}"
                        )

            if legacy_expectations:
                if variant_id not in legacy_expectations:
                    raise AssetPackError(
                        f"variant_expectations.json missing entry for '{family_id}/{variant_id}'"
                    )
                expectation_entry = legacy_expectations[variant_id]
                if not isinstance(expectation_entry, dict):
                    raise AssetPackError(
                        f"variant_expectations.json entry for '{family_id}/{variant_id}' must be a mapping"
                    )
                checksum_relpath = expectation_entry.get("checksum_file")
                if not isinstance(checksum_relpath, str) or not checksum_relpath:
                    raise AssetPackError(
                        f"variant_expectations.json missing checksum_file for '{family_id}/{variant_id}'"
                    )
                checksum_path = verifier_data_dir / family_id / checksum_relpath
                if not checksum_path.exists():
                    raise AssetPackError(f"Missing checksum manifest for '{family_id}/{variant_id}'")
                expected_manifest = checksum_path.read_text(encoding="utf-8")
                actual_manifest = _checksum_manifest_for_dir(tests_dir)
                if expected_manifest != actual_manifest:
                    raise AssetPackError(
                        f"Test checksum drift for '{family_id}/{variant_id}'; rerun the generator or update verifier_data."
                    )

    if scenario_types_seen != SCENARIO_TYPES:
        raise AssetPackError(
            "Initial authored pack must cover all five scenario types; "
            f"got {sorted(scenario_types_seen)}"
        )

    return AssetPackSummary(
        repo_root=root,
        family_count=len(family_dirs),
        variant_count=variant_count,
        family_ids=tuple(sorted(family_ids)),
        scenario_types=tuple(sorted(scenario_types_seen)),
        has_freeze_artifacts=has_freeze_artifacts,
    )


__all__ = [
    "AssetPackError",
    "AssetPackSummary",
    "validate_authored_asset_pack",
]
