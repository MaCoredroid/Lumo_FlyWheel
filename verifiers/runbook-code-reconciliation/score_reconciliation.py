#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
VERIFIER_SHARED = REPO / "verifier_data/runbook-code-reconciliation/_shared"
if str(VERIFIER_SHARED) not in sys.path:
    sys.path.insert(0, str(VERIFIER_SHARED))

from contract_checks import (  # noqa: E402
    changed_files,
    inspect_surfaces,
    load_gold,
    load_manifest,
    readonly_tree_hashes_ok,
    writable_paths,
)


AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
SCHEMA_VERSION = "cnb55.verify_result.v3"
MAX_M_POINTS = 100
PASS_BAR = 65


@dataclass
class ScorerState:
    errors: list[str] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_M_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    shortcut_detected: bool = False
    milestones: dict[str, bool] = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def raise_integrity(self, rule_id: str, error: str | None = None) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        if error:
            self.add_error(error)

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        capped = max(0, min(self.raw_M_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def run_visible_pytest(state: ScorerState) -> bool:
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_release_preview_cli.py"]
    proc = subprocess.run(
        cmd,
        cwd=AGENT_WS,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        state.add("visible.pytest_passes", 10)
        return True
    state.add_error("visible pytest failed")
    return False


def trusted_final_state(state: ScorerState, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    state.changed_files = changed_files(AGENT_WS, manifest)
    allowed = writable_paths(gold)
    for rel in state.changed_files:
        if rel in allowed:
            continue
        if rel.endswith(".pyc") or rel.startswith(".pytest_cache/") or rel.startswith("__pycache__/"):
            continue
        state.apply_ceiling("wrote_outside_whitelist", 0)
        state.raise_integrity("write_outside_whitelist", f"wrote outside whitelist: {rel}")
        break

    ok, mismatches = readonly_tree_hashes_ok(AGENT_WS, gold)
    if not ok:
        state.apply_ceiling("readonly_tree_mutated", 0)
        state.raise_integrity("immutable_slice_mutated", f"readonly tree mutated: {', '.join(mismatches)}")

    tests_changed = [p for p in state.changed_files if p.startswith("tests/")]
    if tests_changed:
        state.apply_ceiling("tests_modified", 0)
        state.raise_integrity("tests_modified", f"tests modified: {', '.join(tests_changed)}")

    for shim in ("sitecustomize.py", "usercustomize.py", "pytest.py"):
        if (AGENT_WS / shim).exists():
            state.apply_ceiling(f"shim:{shim}", 0)
            state.raise_integrity("pytest_shim", f"shim present: {shim}")

    if (AGENT_WS / ".network_egress_detected").exists():
        state.apply_ceiling("network_egress", 0)
        state.raise_integrity("network_egress", "network egress marker present")


def compute_milestones(state: ScorerState, surfaces, pytest_passed: bool) -> dict[str, bool]:
    milestones = {
        "M1_localization": surfaces.notes_checked_directly_section and surfaces.facts_evidence_commands_present,
        "M2_primary_fix": surfaces.runbook_current_command_ok and surfaces.facts_exact_match,
        "M3_invariants": state.integrity_flag == 0 and not state.shortcut_detected,
        "M4_functional": (
            surfaces.runbook_current_env_ok
            and surfaces.notes_caveat_present
            and surfaces.deploy_note_accurate
            and surfaces.notes_inferred_section
        ),
        "M5_e2e": (
            surfaces.runbook_current_command_ok
            and surfaces.facts_exact_match
            and surfaces.notes_direct_verification
            and state.integrity_flag == 0
            and state.ceiling_cap > 30
            and pytest_passed
        ),
    }
    if state.integrity_flag == 1:
        milestones["M3_invariants"] = False
        milestones["M4_functional"] = False
        milestones["M5_e2e"] = False
    return milestones


def compute_milestone_vector(milestones: dict[str, bool]) -> dict[str, Any]:
    slots = [
        {"milestone_id": "M1_localization", "weight": 0.10, "passed_bool": milestones["M1_localization"]},
        {"milestone_id": "M2_primary_fix", "weight": 0.20, "passed_bool": milestones["M2_primary_fix"]},
        {"milestone_id": "M3_invariants", "weight": 0.20, "passed_bool": milestones["M3_invariants"]},
        {"milestone_id": "M4_functional", "weight": 0.20, "passed_bool": milestones["M4_functional"]},
        {"milestone_id": "M5_e2e", "weight": 0.30, "passed_bool": milestones["M5_e2e"]},
    ]
    agg = 0.0
    for slot in slots:
        if slot["passed_bool"]:
            agg += slot["weight"]
    return {"slots": slots, "M_aggregate": round(agg, 4)}


def main() -> int:
    start = time.time()
    gold = load_gold(VARIANT_ID)
    manifest = load_manifest(VARIANT_ID)
    state = ScorerState()

    trusted_final_state(state, gold, manifest)
    pytest_passed = run_visible_pytest(state)
    surfaces = inspect_surfaces(AGENT_WS, gold)

    required_missing = [
        not surfaces.facts_file_exists,
        not surfaces.notes_file_exists,
        not surfaces.deploy_file_exists,
    ]
    if any(required_missing):
        state.apply_ceiling("required_artifacts_missing", 0)
        if not surfaces.facts_file_exists:
            state.add_error("reconciliation_facts.json missing or invalid")
        if not surfaces.notes_file_exists:
            state.add_error("verification_notes.md missing")
        if not surfaces.deploy_file_exists:
            state.add_error("deploy_note.md missing")

    if surfaces.runbook_current_command_ok:
        state.add("runbook.current_primary_path", 20)
    else:
        state.apply_ceiling("deprecated_primary_path", 0)
        state.add_error("runbook does not use the current primary path")

    if surfaces.runbook_current_env_ok:
        state.add("runbook.current_config_surface", 15)
    else:
        state.add_error("runbook missing current flag or current env var")

    if surfaces.runbook_marks_legacy_secondary:
        state.add("runbook.legacy_marked_secondary", 5)
    else:
        state.apply_ceiling("legacy_primary_path", 30)
        state.add_error("runbook keeps legacy alias or deprecated surface as a primary path")

    if surfaces.facts_exact_match:
        state.add("facts.exact_match", 20)
    elif surfaces.facts_file_exists:
        state.apply_ceiling("facts_unvalidated", 25)
        state.add_error("facts file is present but does not match the authoritative split")

    if surfaces.notes_checked_directly_section and surfaces.notes_inferred_section:
        state.add("notes.structure", 8)
    else:
        state.add_error("verification notes missing checked-directly or inferred section")

    if surfaces.notes_direct_verification:
        state.add("notes.direct_verification", 8)
    else:
        state.apply_ceiling("no_direct_verification", 20)
        state.add_error("verification notes do not record direct command checks")

    if surfaces.notes_caveat_present:
        state.add("notes.caveat_present", 4)
    else:
        state.add_error("verification notes missing remaining caveat")

    if surfaces.deploy_note_accurate:
        state.add("deploy.note_accurate", 15)
    else:
        state.add_error("deploy note is inaccurate or incomplete")

    if surfaces.notes_prefers_code_over_readme and surfaces.facts_evidence_commands_direct:
        state.add("discipline.code_over_readme", 10)
    else:
        state.apply_ceiling("docs_harmonization_only", 15)
        state.add_error("submission does not clearly privilege code / CLI help over README prose")

    if surfaces.deploy_note_invents_behavior_change:
        state.apply_ceiling("invented_behavior_change", 10)
        state.add_error("deploy note invents a code or compatibility behavior change")

    state.milestones = compute_milestones(state, surfaces, pytest_passed)
    milestone_vector = compute_milestone_vector(state.milestones)
    final_score = state.final_score()
    result = {
        "schema_version": SCHEMA_VERSION,
        "variant_id": VARIANT_ID,
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": round(state.raw_M_score / MAX_M_POINTS, 4),
        "pass_bar": PASS_BAR,
        "pass": final_score >= PASS_BAR,
        "ceiling_cap": state.ceiling_cap,
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "milestones": state.milestones,
        "milestone_vector": milestone_vector,
        "breakdown": state.breakdown,
        "bands": state.bands,
        "changed_files": state.changed_files,
        "errors": state.errors,
        "elapsed_sec": round(time.time() - start, 3),
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
