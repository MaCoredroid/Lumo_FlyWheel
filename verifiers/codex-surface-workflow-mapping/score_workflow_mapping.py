#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA_ROOT = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

VERIFY_RESULT_VERSION = "cnb55.verify_result.v3"
SCHEMA_VERSION = "cnb55.workflow_mapping.v1"
IGNORED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
MAX_M_POINTS = 100
SCHEDULE_WORDS = ("weekday", "weekdays", "09:00", "cron", "monday", "tuesday", "wednesday", "thursday", "friday", "0 9")


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
    for entry in sorted(target.rglob("*")):
        if any(part in IGNORED_NAMES for part in entry.parts):
            continue
        if entry.suffix == ".pyc":
            continue
        rel_entry = entry.relative_to(target).as_posix()
        if entry.is_dir():
            h.update(f"D:{rel_entry}\n".encode())
        elif entry.is_file():
            h.update(f"F:{rel_entry}\n".encode())
            h.update(sha256_file(entry).encode())
            h.update(b"\n")
    return h.hexdigest()


def list_files(root: Path) -> set[str]:
    files: set[str] = set()
    for entry in root.rglob("*"):
        if entry.is_file():
            if any(part in IGNORED_NAMES for part in entry.parts):
                continue
            if entry.suffix == ".pyc":
                continue
            files.add(entry.relative_to(root).as_posix())
    return files


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def grounded(command: str, source_paths: list[str]) -> bool:
    if not command.strip():
        return False
    for rel in source_paths:
        path = AGENT_WS / rel
        if path.exists() and path.is_file() and command in path.read_text(encoding="utf-8"):
            return True
    return False


@dataclass
class ScoreState:
    errors: list[str] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    raw_score: int = 0
    raw_M_score: int = 0
    ceiling_cap: int = 100
    ceilings_applied: list[str] = field(default_factory=list)
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    shortcut_detected: bool = False
    milestones: dict[str, bool] = field(default_factory=dict)
    submission: dict[str, Any] | None = None

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.breakdown_bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        self.apply_ceiling("integrity_failure", 0)

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        capped = max(0, min(self.raw_M_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def trusted_final_state(state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected_test = gold["test_workflow_map_sha256"]
    actual_test = sha256_file(AGENT_WS / "tests" / "test_workflow_map.py")
    if actual_test != expected_test:
        state.raise_integrity("tests_modified")

    for shim in SHIM_FILES:
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    for rel, expected in gold["readonly_tree_hashes"].items():
        if sha256_tree(AGENT_WS, rel) != expected:
            state.raise_integrity("immutable_slice_mutated")
            break

    initial_files = set(manifest["files"])
    current_files = list_files(AGENT_WS)
    extras = []
    for rel in current_files:
        if rel in initial_files:
            continue
        if rel == "workflow_input.json":
            continue
        if rel.startswith("artifacts/"):
            continue
        extras.append(rel)
    if extras:
        state.raise_integrity("write_outside_whitelist")


def load_submission(state: ScoreState) -> None:
    target = AGENT_WS / "artifacts" / "workflow_map.json"
    if not target.exists():
        state.errors.append("artifacts/workflow_map.json missing")
        state.apply_ceiling("no_submission", 0)
        return
    state.add("phase2.submission_exists", 4)
    try:
        submission = load_json(target)
    except json.JSONDecodeError as exc:
        state.errors.append(f"invalid json: {exc}")
        state.apply_ceiling("malformed_submission", 10)
        return
    if not isinstance(submission, dict):
        state.errors.append("submission must be a JSON object")
        state.apply_ceiling("malformed_submission", 10)
        return
    state.submission = submission
    state.add("phase2.submission_parses", 6)


def check_visible(state: ScoreState) -> None:
    if state.submission is None:
        return
    sub = state.submission
    if sub.get("schema_version") == SCHEMA_VERSION and sub.get("variant_id") == VARIANT_ID:
        state.add("phase2.schema_and_variant", 4)
    else:
        state.errors.append("schema_version or variant_id mismatch")
        state.apply_ceiling("malformed_submission", 10)

    rendered = [
        AGENT_WS / "artifacts" / "SKILL.md",
        AGENT_WS / "artifacts" / "codex_triage.toml",
        AGENT_WS / "artifacts" / "automation_proposal.md",
        AGENT_WS / "artifacts" / "mapping_note.md",
    ]
    if all(path.exists() and path.read_text(encoding="utf-8").strip() for path in rendered):
        state.add("phase2.rendered_artifacts", 4)

    if all(isinstance(sub.get(name), dict) for name in ("skill", "toml", "automation", "mapping_note")):
        state.add("phase2.sections_present", 4)

    decisions = sub.get("mapping_note", {}).get("decisions", [])
    if isinstance(decisions, list) and {item.get("artifact") for item in decisions if isinstance(item, dict)} == {
        "skill",
        "toml",
        "automation",
        "mapping_note",
    }:
        state.add("phase2.mapping_decisions", 4)

    rejected = sub.get("rejected_candidates", [])
    if isinstance(rejected, list) and rejected:
        state.add("phase2.rejected_candidates", 4)

    proposal_text = (AGENT_WS / "artifacts" / "automation_proposal.md")
    if proposal_text.exists():
        text = proposal_text.read_text(encoding="utf-8")
        if "## Task" in text and "## Schedule" in text:
            state.add("phase2.task_schedule_split", 4)


def apply_hidden_scoring(state: ScoreState, gold: dict[str, Any]) -> None:
    if state.submission is None:
        return
    sub = state.submission
    skill = sub.get("skill", {})
    toml = sub.get("toml", {})
    automation = sub.get("automation", {})
    decisions = sub.get("mapping_note", {}).get("decisions", [])
    rejected = sub.get("rejected_candidates", [])

    expected_entrypoint = gold["expected_entrypoint_command"]
    expected_schedule = gold["expected_schedule_literal"]
    stale_commands = {item["command_literal"] for item in gold["rejected_candidates"]}

    entrypoints = {
        skill.get("entrypoint_command_literal"),
        toml.get("entrypoint_command_literal"),
        automation.get("command_literal"),
    }
    if len(entrypoints) == 1:
        if skill.get("entrypoint_command_literal") == expected_entrypoint:
            state.add("hidden.skill_entrypoint", 10)
        if toml.get("entrypoint_command_literal") == expected_entrypoint:
            state.add("hidden.toml_entrypoint", 10)
        if automation.get("command_literal") == expected_entrypoint:
            state.add("hidden.automation_entrypoint", 5)
    else:
        state.apply_ceiling("inconsistent_entrypoints", 0)

    if any(command in stale_commands for command in entrypoints if isinstance(command, str)):
        state.apply_ceiling("stale_entrypoint", 20)

    if automation.get("schedule_literal") == expected_schedule:
        state.add("hidden.schedule_literal", 10)
    else:
        state.apply_ceiling("wrong_schedule", 25)

    task_prompt = str(automation.get("task_prompt", "")).lower()
    if task_prompt and not any(token in task_prompt for token in SCHEDULE_WORDS):
        state.add("hidden.task_prompt_semantics", 5)
    else:
        state.apply_ceiling("schedule_in_prompt", 30)

    grounded_ok = True
    required_paths = set(gold["required_paths"])
    cited_paths: set[str] = set()
    if isinstance(decisions, list):
        for entry in decisions:
            if not isinstance(entry, dict):
                grounded_ok = False
                continue
            source_paths = entry.get("source_paths", [])
            command_literals = entry.get("command_literals", [])
            if not isinstance(source_paths, list) or not isinstance(command_literals, list):
                grounded_ok = False
                continue
            for rel in source_paths:
                cited_paths.add(rel)
            for command in command_literals:
                if not grounded(str(command), list(source_paths)):
                    grounded_ok = False
    else:
        grounded_ok = False
    if grounded_ok and required_paths.issubset(cited_paths):
        state.add("hidden.mapping_grounding", 15)
    else:
        state.apply_ceiling("ungrounded_mapping", 25)

    rejection_ok = True
    rejected_paths = {item.get("path") for item in rejected if isinstance(item, dict)}
    for expected in gold["rejected_candidates"]:
        if expected["path"] not in rejected_paths:
            rejection_ok = False
            break
    if rejection_ok:
        state.add("hidden.rejection_judgment", 10)
    else:
        if any(path.startswith("drafts/") for path in gold["required_paths"]):
            state.apply_ceiling("dirty_state_reuse", 35)
        if gold["incident_paths"]:
            state.apply_ceiling("incident_blind_reuse", 25)

    scope_text = " ".join(
        [
            str(skill.get("workflow_scope", "")),
            str(toml.get("workflow_scope", "")),
            str(automation.get("task_prompt", "")),
        ]
    ).lower()
    if all(keyword in scope_text for keyword in [item.lower() for item in gold["required_scope_keywords"]]):
        state.add("hidden.scope_specificity", 5)
    else:
        state.apply_ceiling("generic_mapping", 25)
        if gold["release_paths"]:
            state.apply_ceiling("objective_drift", 30)

    if len(cited_paths) < 4:
        state.apply_ceiling("generic_mapping", 25)


def compute_milestones(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    if state.submission is None:
        return {
            "M1_localization": False,
            "M2_primary_fix": False,
            "M3_invariants": False,
            "M4_functional": False,
            "M5_e2e": False,
        }
    decisions = state.submission.get("mapping_note", {}).get("decisions", [])
    cited_paths = set()
    if isinstance(decisions, list):
        for entry in decisions:
            if isinstance(entry, dict):
                cited_paths.update(entry.get("source_paths", []))
    m1 = len(cited_paths) >= 4
    m2 = state.submission.get("schema_version") == SCHEMA_VERSION and state.submission.get("variant_id") == VARIANT_ID
    m3 = state.integrity_flag == 0 and "stale_entrypoint" not in state.ceilings_applied
    commands = {
        state.submission.get("skill", {}).get("entrypoint_command_literal"),
        state.submission.get("toml", {}).get("entrypoint_command_literal"),
        state.submission.get("automation", {}).get("command_literal"),
    }
    m4 = m2 and len(commands) == 1 and next(iter(commands)) == gold["expected_entrypoint_command"]
    rejected_paths = {item.get("path") for item in state.submission.get("rejected_candidates", []) if isinstance(item, dict)}
    m5 = m4 and all(item["path"] in rejected_paths for item in gold["rejected_candidates"])
    return {
        "M1_localization": m1,
        "M2_primary_fix": m2,
        "M3_invariants": m3,
        "M4_functional": m4,
        "M5_e2e": m5,
    }


def write_result(state: ScoreState) -> None:
    milestone_weights = {
        "M1_localization": 0.10,
        "M2_primary_fix": 0.20,
        "M3_invariants": 0.20,
        "M4_functional": 0.20,
        "M5_e2e": 0.30,
    }
    m_aggregate = round(sum(milestone_weights[key] for key, passed in state.milestones.items() if passed), 4)
    payload = {
        "schema_version": VERIFY_RESULT_VERSION,
        "variant_id": VARIANT_ID,
        "P_benchmark": state.final_score(),
        "score": state.final_score(),
        "raw_score_pre_ceiling": state.raw_score,
        "M_training": state.final_m_training(),
        "pass": state.final_score() >= 70 and state.integrity_flag == 0,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "ceilings_applied": state.ceilings_applied,
        "errors": state.errors,
        "breakdown": state.breakdown,
        "breakdown_bands": state.breakdown_bands,
        "milestones": state.milestones,
        "milestone_vector": {
            "M_aggregate": m_aggregate,
            "slots": [
                {"slot": key, "weight": milestone_weights[key], "passed_bool": passed}
                for key, passed in state.milestones.items()
            ],
        },
    }
    RESULT_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    gold = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "gold_workflow.json")
    manifest = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "workspace_manifest.json")
    state = ScoreState()
    trusted_final_state(state, gold, manifest)
    load_submission(state)
    check_visible(state)
    apply_hidden_scoring(state, gold)
    state.milestones = compute_milestones(state, gold)
    write_result(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
