from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


IGNORED_RUNTIME_PREFIXES = (
    "repo/.pytest_cache/",
    "repo/__pycache__/",
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def family_root() -> Path:
    return repo_root() / "benchmark_blueprints/families/incident-retro-runbook-closure"


def verifier_root() -> Path:
    return repo_root() / "verifier_data/incident-retro-runbook-closure"


def load_gold(variant_id: str) -> dict[str, Any]:
    return load_json(verifier_root() / variant_id / "gold_repair.json")


def load_manifest(variant_id: str) -> dict[str, Any]:
    return load_json(verifier_root() / variant_id / "workspace_manifest.json")


def load_action_items(workspace: Path) -> dict[str, Any]:
    return load_json(workspace / "retro" / "action_items.json")


def load_automation(workspace: Path) -> dict[str, Any]:
    path = workspace / "repo/.codex/automations/queue-drain-watch/automation.toml"
    text = path.read_text()
    if tomllib is not None:
        parsed = tomllib.loads(text)
    else:
        parsed: dict[str, Any] = {}
        prompt_lines: list[str] = []
        in_prompt = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not in_prompt and line.startswith('prompt = """'):
                in_prompt = True
                continue
            if in_prompt:
                if line == '"""':
                    in_prompt = False
                    parsed["prompt"] = "\n".join(prompt_lines)
                    prompt_lines = []
                else:
                    prompt_lines.append(raw_line)
                continue
            if "=" not in raw_line:
                continue
            key, value = raw_line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                parsed[key] = value[1:-1]
            else:
                try:
                    parsed[key] = int(value)
                except ValueError:
                    parsed[key] = value
    prompt = parsed.get("prompt", "")
    return {"parsed": parsed, "prompt": prompt, "text": text}


def read_text(workspace: Path, rel: str) -> str:
    return (workspace / rel).read_text()


def render_expected_sequence(action_items: dict[str, Any]) -> list[str]:
    return list(action_items["verification_sequence"])


@dataclass
class SurfaceStatus:
    helper_command_matches_authority: bool
    helper_keeps_retired_primary: bool
    runbook_command_matches_authority: bool
    runbook_sequence_matches_authority: bool
    runbook_mentions_retired_command: bool
    automation_prompt_matches_authority: bool
    automation_uses_retired_command: bool
    automation_schedule_preserved: bool
    automation_destination_preserved: bool
    note_is_actionable_only: bool
    note_mentions_required_command: bool
    note_mentions_required_target: bool
    informational_notes_leaked: bool
    all_three_authoritative_surfaces_aligned: bool


def inspect_surfaces(workspace: Path, gold: dict[str, Any]) -> SurfaceStatus:
    action = load_action_items(workspace)
    current = action["verification_command"]
    retired = action["retired_command"]
    target = action["escalation_target"]

    helper_text = read_text(workspace, "repo/scripts/queue_drain_helper.py")
    runbook_text = read_text(workspace, "repo/runbooks/queue_drain.md")
    note_text = read_text(workspace, "repo/ops/notes/queue_drain_followup.md")
    automation = load_automation(workspace)
    prompt = automation["prompt"]
    parsed = automation["parsed"]

    helper_command_matches_authority = current in helper_text
    helper_keeps_retired_primary = retired in helper_text and current not in helper_text

    runbook_command_matches_authority = current in runbook_text and target in runbook_text
    runbook_sequence_matches_authority = all(
        step in runbook_text for step in render_expected_sequence(action)
    )
    runbook_mentions_retired_command = retired in runbook_text

    automation_prompt_matches_authority = current in prompt and target in prompt
    automation_uses_retired_command = retired in prompt
    expected_auto = gold["automation_expectations"]
    automation_schedule_preserved = parsed.get("schedule_minutes") == expected_auto["schedule_minutes"]
    automation_destination_preserved = parsed.get("destination") == expected_auto["destination"]

    informational_tokens = set(action.get("informational_note_tokens", []))
    applied_section = note_text.split("## Deferred informational retro notes", 1)[0]
    note_is_actionable_only = not any(token in applied_section for token in informational_tokens)
    note_mentions_required_command = current in note_text
    note_mentions_required_target = target in note_text
    informational_notes_leaked = any(
        token in corpus
        for token in informational_tokens
        for corpus in (runbook_text, helper_text, prompt, applied_section)
    )

    all_three_authoritative_surfaces_aligned = all(
        (
            helper_command_matches_authority,
            runbook_command_matches_authority,
            runbook_sequence_matches_authority,
            automation_prompt_matches_authority,
        )
    )

    return SurfaceStatus(
        helper_command_matches_authority=helper_command_matches_authority,
        helper_keeps_retired_primary=helper_keeps_retired_primary,
        runbook_command_matches_authority=runbook_command_matches_authority,
        runbook_sequence_matches_authority=runbook_sequence_matches_authority,
        runbook_mentions_retired_command=runbook_mentions_retired_command,
        automation_prompt_matches_authority=automation_prompt_matches_authority,
        automation_uses_retired_command=automation_uses_retired_command,
        automation_schedule_preserved=automation_schedule_preserved,
        automation_destination_preserved=automation_destination_preserved,
        note_is_actionable_only=note_is_actionable_only,
        note_mentions_required_command=note_mentions_required_command,
        note_mentions_required_target=note_mentions_required_target,
        informational_notes_leaked=informational_notes_leaked,
        all_three_authoritative_surfaces_aligned=all_three_authoritative_surfaces_aligned,
    )


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


def validate_followup_schema(workspace: Path) -> bool:
    note = read_text(workspace, "repo/ops/notes/queue_drain_followup.md")
    schema = load_json(workspace / "artifacts" / "ops_note_schema.json")
    return all(section in note for section in schema["required_lines"])


def validate_automation_schema(workspace: Path) -> bool:
    gold = load_json(workspace / "artifacts" / "automation_expectations.json")
    automation = load_automation(workspace)
    parsed = automation["parsed"]
    prompt = automation["prompt"]
    return (
        parsed.get("kind") == gold["kind"]
        and parsed.get("destination") == gold["destination"]
        and parsed.get("schedule_minutes") == gold["schedule_minutes"]
        and gold["expected_prompt_command"] in prompt
        and gold["expected_prompt_target"] in prompt
    )


def probe_paths() -> tuple[Path, Path]:
    vr = verifier_root()
    return vr / "probe_runs.jsonl", vr / "_probe_logs"
