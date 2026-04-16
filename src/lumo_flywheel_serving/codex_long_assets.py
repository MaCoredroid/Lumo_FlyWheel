from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .data_pool import MIN_CODEX_LONG_FAMILIES, SCENARIO_TYPES
from .task_orchestrator import TaskSpec, validate_family_spec
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
    if "pytest" not in command:
        return True
    if "python -m pytest" in command:
        return False
    required = (
        "import pathlib, sys",
        "import pytest",
        'sys.path.insert(0, cwd)',
        'pytest.main(["-q"])',
    )
    return all(pattern in command for pattern in required)


def _verify_references_milestone_helper(verify_text: str, milestone_id: str) -> bool:
    helper_path = f"/verifier/milestones/{milestone_id}.sh"
    allowed_invocations = (
        f"source {helper_path}",
        f". {helper_path}",
        f"bash {helper_path}",
        f"sh {helper_path}",
        helper_path,
    )
    return any(invocation in verify_text for invocation in allowed_invocations)


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
        for milestone in milestones:
            milestone_id = str(milestone["id"])
            milestone_path = verifier_dir / "milestones" / f"{milestone_id}.sh"
            if not milestone_path.exists():
                raise AssetPackError(f"Missing milestone helper: {milestone_path}")
            if not os.access(milestone_path, os.X_OK):
                raise AssetPackError(f"Milestone helper is not executable: {milestone_path}")
            if not _verify_references_milestone_helper(verify_text, milestone_id):
                raise AssetPackError(
                    f"verify.sh does not source or execute milestone helper '{milestone_id}' for family '{family_id}'"
                )

        expectations_path = verifier_data_dir / family_id / "variant_expectations.json"
        if not expectations_path.exists():
            raise AssetPackError(f"Missing variant expectations for family '{family_id}'")
        expectations = json.loads(expectations_path.read_text(encoding="utf-8"))
        expected_variants = expectations.get("variants")
        if not isinstance(expected_variants, dict):
            raise AssetPackError(f"variant_expectations.json must contain a 'variants' mapping: {expectations_path}")

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
            validate_family_spec(task, family_spec)

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

            ci_runner_path = repo_dir / "scripts" / "run_ci.py"
            if ci_runner_path.exists():
                ci_runner_text = ci_runner_path.read_text(encoding="utf-8")
                required_patterns = (
                    'sys.path=[p for p in sys.path if p not in ("", cwd)]',
                    'import pytest',
                    'pytest.main(["-q"])',
                    'subprocess.call([sys.executable, "-c", runner])',
                )
                if any(pattern not in ci_runner_text for pattern in required_patterns):
                    raise AssetPackError(
                        "Repo CI runner must import installed pytest before re-adding the workspace "
                        f"for '{family_id}/{variant_id}'"
                    )

            if variant_id not in expected_variants:
                raise AssetPackError(
                    f"variant_expectations.json missing entry for '{family_id}/{variant_id}'"
                )
            checksum_relpath = expected_variants[variant_id].get("checksum_file")
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
