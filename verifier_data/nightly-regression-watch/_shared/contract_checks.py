from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

REQUIRED_MILESTONES = ("M2_primary_fix", "M4_functional", "M5_e2e")
FAMILY_ID = "nightly-regression-watch"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def verifier_root() -> Path:
    return repo_root() / "verifier_data" / FAMILY_ID


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for item in sorted(target.rglob("*")):
        rp = item.relative_to(target).as_posix()
        if rp.endswith(".pyc") or "__pycache__" in rp or ".pytest_cache" in rp:
            continue
        if item.is_dir():
            h.update(b"D:" + rp.encode() + b"\x00")
        elif item.is_file():
            h.update(b"F:" + rp.encode() + b"\x00")
            h.update(sha256_file(item).encode() + b"\x00")
    return h.hexdigest()


def load_gold(variant_id: str) -> dict[str, Any]:
    return load_json(verifier_root() / variant_id / "gold_repair.json")


def load_manifest(variant_id: str) -> dict[str, Any]:
    return load_json(verifier_root() / variant_id / "workspace_manifest.json")


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
        if rel in tracked or rel in seen:
            continue
        if rel.endswith(".pyc") or "__pycache__" in rel or ".pytest_cache" in rel:
            continue
        changed.append(rel)
    return sorted(set(changed))


def readonly_tree_hashes_ok(workspace: Path, gold: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for rel, expected in gold["readonly_tree_hashes"].items():
        if sha256_tree(workspace, rel) != expected:
            mismatches.append(rel)
    return (not mismatches, mismatches)


def writable_paths(gold: dict[str, Any]) -> set[str]:
    return set(gold["editable_files"])


def parse_automation(workspace: Path) -> dict[str, Any]:
    path = workspace / "ops_digest/automation/nightly_regression_watch.toml"
    text = path.read_text()
    if tomllib is not None:
        return {"text": text, "parsed": tomllib.loads(text)}
    parsed: dict[str, Any] = {}
    prompt_lines: list[str] = []
    in_prompt = False
    for raw in text.splitlines():
        line = raw.strip()
        if not in_prompt and line.startswith('prompt = """'):
            in_prompt = True
            continue
        if in_prompt:
            if line == '"""':
                in_prompt = False
                parsed["prompt"] = "\n".join(prompt_lines)
                prompt_lines = []
            else:
                prompt_lines.append(raw)
            continue
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        parsed[key.strip()] = value.strip().strip('"')
    return {"text": text, "parsed": parsed}


def _load_module(workspace: Path, rel: str, name: str):
    path = workspace / rel
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def render_current_digest(workspace: Path) -> str:
    root_pkg = workspace / "ops_digest"
    _load_module(workspace, "ops_digest/__init__.py", "ops_digest")
    _load_module(workspace, "ops_digest/src/__init__.py", "ops_digest.src")
    _load_module(workspace, "ops_digest/src/schema.py", "ops_digest.src.schema")
    builder = _load_module(workspace, "ops_digest/src/digest_builder.py", "ops_digest.src.digest_builder")
    return builder.build_digest_markdown(builder.load_runs(root_pkg / "fixtures" / "runs"))


@dataclass
class SurfaceStatus:
    generated_digest_matches_output: bool
    automation_singleton_ok: bool
    automation_prompt_ok: bool
    automation_summary_ok: bool
    runbook_command_ok: bool
    runbook_wording_ok: bool
    schema_mentions_rollover: bool
    digest_mentions_latest_per_day: bool
    code_files_changed: bool


def inspect_surfaces(workspace: Path, manifest: dict[str, Any], gold: dict[str, Any]) -> SurfaceStatus:
    generated_path = workspace / gold["generated_digest_path"]
    current = render_current_digest(workspace)
    automation_dir = workspace / "ops_digest/automation"
    singleton_ok = sorted(p.name for p in automation_dir.glob("*.toml")) == ["nightly_regression_watch.toml"]
    auto = parse_automation(workspace)
    parsed = auto["parsed"]
    prompt = parsed.get("prompt", "")
    runbook = (workspace / "ops_digest/docs/escalation_runbook.md").read_text()
    schema_text = (workspace / "ops_digest/src/schema.py").read_text()
    builder_text = (workspace / "ops_digest/src/digest_builder.py").read_text()
    changed = changed_files(workspace, manifest)
    return SurfaceStatus(
        generated_digest_matches_output=generated_path.read_text() == current,
        automation_singleton_ok=singleton_ok,
        automation_prompt_ok=(
            "Action required" in prompt
            and "latest completed run for each `report_date`" in prompt
            and "flag anything marked fail" not in prompt
        ),
        automation_summary_ok=parsed.get("summary_title") == gold["automation_expectations"]["summary_title"],
        runbook_command_ok=gold["runbook_expectations"]["command"] in runbook,
        runbook_wording_ok=(
            "Action required" in runbook
            and "latest completed run for each `report_date`" in runbook
            and "--flag-any-fail" not in runbook
        ),
        schema_mentions_rollover=("final_verdict" in schema_text and "results" in schema_text),
        digest_mentions_latest_per_day=("latest" in builder_text and "report_date" in builder_text),
        code_files_changed=all(path in changed for path in gold["code_files"]),
    )


@dataclass
class HiddenStatus:
    required_milestone_blocks: bool
    advisory_non_blocking: bool
    latest_of_day_selected: bool
    mixed_milestone_shapes_parse: bool
    no_duplicate_same_day_lines: bool


def hidden_scenarios(workspace: Path) -> HiddenStatus:
    _load_module(workspace, "ops_digest/__init__.py", "ops_digest")
    _load_module(workspace, "ops_digest/src/__init__.py", "ops_digest.src")
    schema = _load_module(workspace, "ops_digest/src/schema.py", "ops_digest.src.schema")
    builder = _load_module(workspace, "ops_digest/src/digest_builder.py", "ops_digest.src.digest_builder")

    required_payload = {
        "run_id": "hidden-required",
        "report_date": "2026-05-01",
        "completed_at": "2026-05-01T03:00:00Z",
        "final_verdict": {"pass": True},
        "milestones": {"results": {"M2_primary_fix": {"status": "passed", "required": True}, "M4_functional": {"status": "missing", "required": True}, "M5_e2e": {"status": "passed", "required": True}}},
        "warnings": [],
    }
    advisory_payload = {
        "run_id": "hidden-advisory",
        "report_date": "2026-05-02",
        "completed_at": "2026-05-02T03:00:00Z",
        "final_verdict": {"pass": True},
        "milestones": {"results": {"M2_primary_fix": True, "M4_functional": True, "M5_e2e": True}},
        "warnings": ["advisory: retry ran late"],
    }
    older = {
        "run_id": "hidden-older",
        "report_date": "2026-05-03",
        "completed_at": "2026-05-03T01:00:00Z",
        "final_verdict": {"pass": False},
        "milestones": {"results": {"M2_primary_fix": False, "M4_functional": False, "M5_e2e": False}},
        "warnings": [],
    }
    newer = {
        "run_id": "hidden-newer",
        "report_date": "2026-05-03",
        "completed_at": "2026-05-03T04:00:00Z",
        "final_verdict": {"pass": True},
        "milestones": {"results": {"M2_primary_fix": {"passed_bool": True}, "M4_functional": {"passed": True}, "M5_e2e": {"status": "passed"}}},
        "warnings": ["advisory: summary delayed"],
    }

    required = schema.normalize_run(required_payload)
    advisory = schema.normalize_run(advisory_payload)
    latest = builder.select_latest_per_day([schema.normalize_run(older), schema.normalize_run(newer)])
    digest = builder.build_digest_markdown([schema.normalize_run(older), schema.normalize_run(newer)])

    return HiddenStatus(
        required_milestone_blocks=required["is_blocking"] and "M4_functional" in required["missing_required"],
        advisory_non_blocking=(not advisory["is_blocking"]) and advisory["label"] == "Healthy night",
        latest_of_day_selected=(len(latest) == 1 and latest[0]["run_id"] == "hidden-newer"),
        mixed_milestone_shapes_parse=(required["is_blocking"] and not advisory["is_blocking"]),
        no_duplicate_same_day_lines=(digest.count("2026-05-03") == 1),
    )
