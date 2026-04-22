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
VERIFIER_SHARED = REPO / "verifier_data/incident-retro-runbook-closure/_shared"
if str(VERIFIER_SHARED) not in sys.path:
    sys.path.insert(0, str(VERIFIER_SHARED))

from contract_checks import (  # noqa: E402
    changed_files,
    inspect_surfaces,
    load_gold,
    load_manifest,
    readonly_tree_hashes_ok,
    validate_automation_schema,
    validate_followup_schema,
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
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "repo/tests/test_queue_drain_helper.py",
        "repo/tests/test_runbook_contract.py",
    ]
    proc = subprocess.run(
        cmd,
        cwd=AGENT_WS,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        state.add("visible.pytest_passes", 25)
        return True
    state.add_error("visible pytest failed")
    state.breakdown["visible.pytest_output"] = 0
    return False


def trusted_final_state(state: ScorerState, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    state.changed_files = changed_files(AGENT_WS, manifest)
    allowed = writable_paths(gold)
    for rel in state.changed_files:
        if rel in allowed:
            continue
        if rel.endswith(".pyc") or rel.startswith("repo/.pytest_cache/") or rel.startswith("repo/__pycache__/"):
            continue
        state.apply_ceiling("wrote_outside_whitelist", 0)
        state.raise_integrity("write_outside_whitelist", f"wrote outside whitelist: {rel}")
        break

    ok, mismatches = readonly_tree_hashes_ok(AGENT_WS, gold)
    if not ok:
        state.apply_ceiling("readonly_tree_mutated", 0)
        state.raise_integrity("immutable_slice_mutated", f"readonly tree mutated: {', '.join(mismatches)}")

    tests_changed = [p for p in state.changed_files if p.startswith("repo/tests/")]
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
        "M1_localization": surfaces.helper_command_matches_authority and surfaces.runbook_command_matches_authority,
        "M2_primary_fix": pytest_passed and surfaces.helper_command_matches_authority,
        "M3_invariants": state.integrity_flag == 0 and not state.shortcut_detected,
        "M4_functional": (
            pytest_passed
            and surfaces.runbook_sequence_matches_authority
            and surfaces.automation_prompt_matches_authority
            and surfaces.note_is_actionable_only
        ),
        "M5_e2e": (
            pytest_passed
            and surfaces.all_three_authoritative_surfaces_aligned
            and surfaces.automation_schedule_preserved
            and surfaces.automation_destination_preserved
            and state.integrity_flag == 0
            and state.ceiling_cap > 30
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

    if surfaces.helper_command_matches_authority:
        state.add("helper.command_matches_authority", 12)
    else:
        state.add_error("helper command does not match authoritative action item")

    if not surfaces.helper_keeps_retired_primary:
        state.add("helper.no_retired_primary", 8)
    else:
        state.apply_ceiling("doc_only_closure", 10)
        state.add_error("helper still keeps retired command as the primary path")

    if surfaces.runbook_command_matches_authority:
        state.add("runbook.command_target_match", 10)
    else:
        state.add_error("runbook command or escalation target is stale")

    if surfaces.runbook_sequence_matches_authority:
        state.add("runbook.sequence_match", 10)
    else:
        state.add_error("runbook sequence does not match authoritative action items")

    if surfaces.automation_prompt_matches_authority:
        state.add("automation.prompt_matches_authority", 12)
    else:
        state.add_error("automation prompt does not match authoritative command/target")

    if surfaces.automation_schedule_preserved:
        state.add("automation.schedule_preserved", 4)
    else:
        state.apply_ceiling("schedule_drift", 25)
        state.add_error("automation schedule changed")

    if surfaces.automation_destination_preserved:
        state.add("automation.destination_preserved", 4)
    else:
        state.apply_ceiling("schedule_drift", 25)
        state.add_error("automation destination changed")

    if surfaces.note_is_actionable_only:
        state.add("note.actionable_only", 5)
    else:
        state.add_error("follow-up note includes informational-only retro content")

    if surfaces.note_mentions_required_command and surfaces.note_mentions_required_target:
        state.add("note.required_command_target", 5)
    else:
        state.add_error("follow-up note misses the required command or escalation target")

    if validate_automation_schema(AGENT_WS) and validate_followup_schema(AGENT_WS):
        state.add("invariants.schema_checks", 5)
    else:
        state.add_error("automation or follow-up schema validation failed")

    repaired_count = sum(
        [
            surfaces.helper_command_matches_authority,
            surfaces.runbook_command_matches_authority and surfaces.runbook_sequence_matches_authority,
            surfaces.automation_prompt_matches_authority,
        ]
    )
    if repaired_count < 3:
        state.apply_ceiling("partial_surface_repair", 25)
    if surfaces.runbook_mentions_retired_command or surfaces.automation_uses_retired_command:
        state.apply_ceiling("dual_command_path", 15)
    if not surfaces.all_three_authoritative_surfaces_aligned:
        state.apply_ceiling("ungrounded_authority", 20)
    if VARIANT_ID in {"v4-multi-corpus-objective", "v5-recovery-in-thread"} and not surfaces.runbook_sequence_matches_authority:
        state.apply_ceiling("objective_drift", 30)
    if surfaces.informational_notes_leaked:
        state.apply_ceiling("informational_note_promotion", 35)
        state.add_error("informational-only retro notes were promoted into a required repair surface")

    state.milestones = compute_milestones(state, surfaces, pytest_passed)
    milestone_vector = compute_milestone_vector(state.milestones)

    final_score = state.final_score()
    result = {
        "pass": bool(final_score >= PASS_BAR and state.milestones["M4_functional"] and state.integrity_flag == 0),
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_M_score,
        "milestones": state.milestones,
        "milestone_vector": milestone_vector,
        "breakdown": {**dict(sorted(state.breakdown.items())), "__bands": dict(sorted(state.bands.items()))},
        "ceilings_applied": sorted(state.ceilings_applied),
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": sorted(state.integrity_rules_fired),
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "changed_files": state.changed_files,
        "variant_id": VARIANT_ID,
        "wall_clock_seconds": int(round(time.time() - start)),
        "schema_version": SCHEMA_VERSION,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
