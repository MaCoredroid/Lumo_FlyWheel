from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IGNORED_RUNTIME_PREFIXES = (
    ".pytest_cache/",
    "__pycache__/",
)
IGNORED_RUNTIME_SUFFIXES = (
    ".pyc",
    ".pyo",
)
IGNORED_RUNTIME_FILES = {
    ".DS_Store",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_tree(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for path in sorted(target.rglob("*")):
        rp = path.relative_to(target).as_posix()
        if _is_ignored_runtime_file(rp):
            continue
        if path.is_dir():
            h.update(b"D:" + rp.encode() + b"\x00")
        elif path.is_file():
            h.update(b"F:" + rp.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
    return h.hexdigest()


def _is_ignored_runtime_file(rel: str) -> bool:
    if rel in IGNORED_RUNTIME_FILES:
        return True
    if "/__pycache__/" in f"/{rel}/" or rel.startswith("__pycache__/"):
        return True
    if "/.pytest_cache/" in f"/{rel}/" or rel.startswith(".pytest_cache/"):
        return True
    if any(rel.startswith(prefix) for prefix in IGNORED_RUNTIME_PREFIXES):
        return True
    return rel.endswith(IGNORED_RUNTIME_SUFFIXES)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def verifier_root() -> Path:
    return repo_root() / "verifier_data/runbook-code-reconciliation"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_gold(variant_id: str) -> dict[str, Any]:
    return load_json(verifier_root() / variant_id / "gold_repair.json")


def load_manifest(variant_id: str) -> dict[str, Any]:
    return load_json(verifier_root() / variant_id / "workspace_manifest.json")


def read_text(workspace: Path, rel: str) -> str:
    path = workspace / rel
    if not path.exists():
        return ""
    return path.read_text()


def changed_files(workspace: Path, manifest: dict[str, Any]) -> list[str]:
    tracked = manifest["files"]
    changed: list[str] = []
    seen: set[str] = set()
    for rel, expected_sha in tracked.items():
        path = workspace / rel
        if not path.exists():
            changed.append(rel)
            seen.add(rel)
            continue
        if sha256_file(path) != expected_sha:
            changed.append(rel)
            seen.add(rel)
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace).as_posix()
        if rel in seen or rel in tracked:
            continue
        if _is_ignored_runtime_file(rel):
            continue
        changed.append(rel)
    return sorted(set(changed))


def writable_paths(gold: dict[str, Any]) -> set[str]:
    return set(gold["editable_files"])


def readonly_tree_hashes_ok(workspace: Path, gold: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for rel, expected in gold["readonly_tree_hashes"].items():
        actual = sha256_tree(workspace, rel)
        if actual != expected:
            mismatches.append(rel)
    return (not mismatches, mismatches)


@dataclass
class SurfaceStatus:
    runbook_current_command_ok: bool
    runbook_current_env_ok: bool
    runbook_marks_legacy_secondary: bool
    runbook_mentions_legacy: bool
    facts_file_exists: bool
    facts_exact_match: bool
    facts_evidence_commands_present: bool
    facts_evidence_commands_direct: bool
    notes_file_exists: bool
    notes_checked_directly_section: bool
    notes_inferred_section: bool
    notes_caveat_present: bool
    notes_direct_verification: bool
    notes_prefers_code_over_readme: bool
    deploy_file_exists: bool
    deploy_note_accurate: bool
    deploy_note_invents_behavior_change: bool


def load_facts(workspace: Path) -> dict[str, Any] | None:
    path = workspace / "artifacts/reconciliation_facts.json"
    if not path.exists():
        return None
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return None


def exact_fact_match(facts: dict[str, Any] | None, gold: dict[str, Any]) -> bool:
    if not isinstance(facts, dict):
        return False
    keys = [
        "primary_entrypoint",
        "legacy_alias",
        "current_flag",
        "deprecated_flag",
        "current_env_var",
        "deprecated_env_var",
    ]
    for key in keys:
        if facts.get(key) != gold[key]:
            return False
    commands = facts.get("evidence_commands")
    if not isinstance(commands, list):
        return False
    return all(cmd in commands for cmd in gold["required_evidence_commands"])


def inspect_surfaces(workspace: Path, gold: dict[str, Any]) -> SurfaceStatus:
    runbook = read_text(workspace, "docs/runbooks/release_preview.md")
    notes = read_text(workspace, "artifacts/verification_notes.md")
    deploy = read_text(workspace, "artifacts/deploy_note.md")
    facts = load_facts(workspace)

    required = gold["required_evidence_commands"]
    facts_commands = facts.get("evidence_commands", []) if isinstance(facts, dict) else []
    facts_evidence_commands_present = isinstance(facts_commands, list) and all(cmd in facts_commands for cmd in required)
    facts_evidence_commands_direct = facts_evidence_commands_present and any("--help" in cmd for cmd in facts_commands)

    runbook_current_command_ok = gold["expected_current_command"] in runbook and gold["primary_entrypoint"] in runbook
    runbook_current_env_ok = gold["current_env_var"] in runbook and gold["current_flag"] in runbook
    runbook_mentions_legacy = gold["legacy_alias_name"] in runbook or gold["deprecated_flag"] in runbook or gold["deprecated_env_var"] in runbook
    legacy_secondary_markers = ("deprecated", "compatibility", "secondary", "not primary")
    runbook_marks_legacy_secondary = not runbook_mentions_legacy or any(marker in runbook.lower() for marker in legacy_secondary_markers)

    notes_checked_directly_section = "## Checked directly" in notes
    notes_inferred_section = "## Inferred from code" in notes
    notes_caveat_present = "## Remaining caveats" in notes
    notes_direct_verification = all(cmd in notes for cmd in required[:2])
    notes_lower = notes.lower()
    notes_prefers_code_over_readme = "README" in notes and (
        "source of truth" in notes_lower
        or "contradict" in notes_lower
        or "overruled" in notes_lower
        or (
            "code" in notes_lower
            and "live help" in notes_lower
            and ("follow" in notes_lower or "authorit" in notes_lower)
        )
    )

    deploy_note_invents_behavior_change = any(
        phrase in deploy.lower()
        for phrase in ("removed the alias", "alias no longer works", "deprecated env var no longer works", "code behavior changed")
    )
    deploy_note_accurate = (
        gold["primary_entrypoint"] in deploy
        and gold["current_flag"] in deploy
        and gold["current_env_var"] in deploy
        and "compatibility" in deploy.lower()
        and not deploy_note_invents_behavior_change
    )

    return SurfaceStatus(
        runbook_current_command_ok=runbook_current_command_ok,
        runbook_current_env_ok=runbook_current_env_ok,
        runbook_marks_legacy_secondary=runbook_marks_legacy_secondary,
        runbook_mentions_legacy=runbook_mentions_legacy,
        facts_file_exists=facts is not None,
        facts_exact_match=exact_fact_match(facts, gold),
        facts_evidence_commands_present=facts_evidence_commands_present,
        facts_evidence_commands_direct=facts_evidence_commands_direct,
        notes_file_exists=bool(notes),
        notes_checked_directly_section=notes_checked_directly_section,
        notes_inferred_section=notes_inferred_section,
        notes_caveat_present=notes_caveat_present,
        notes_direct_verification=notes_direct_verification,
        notes_prefers_code_over_readme=notes_prefers_code_over_readme,
        deploy_file_exists=bool(deploy),
        deploy_note_accurate=deploy_note_accurate,
        deploy_note_invents_behavior_change=deploy_note_invents_behavior_change,
    )
