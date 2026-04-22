#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHARED = REPO / "verifier_data/nightly-regression-watch/_shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from contract_checks import (  # noqa: E402
    changed_files,
    hidden_scenarios,
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
PASS_BAR = 70


@dataclass
class State:
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

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def apply_ceiling(self, name: str, cap: int) -> None:
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def raise_integrity(self, rule_id: str, error: str | None = None) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        self.ceiling_cap = 0
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        if error:
            self.add_error(error)

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        capped = max(0, min(self.raw_M_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def run_pytest(paths: list[str], extra_path: str | None = None) -> bool:
    env = os.environ.copy()
    pythonpath = [str(AGENT_WS)]
    if extra_path:
        pythonpath.append(extra_path)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths],
        cwd=AGENT_WS,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def trusted_final_state(state: State, gold: dict, manifest: dict) -> list[str]:
    changed = changed_files(AGENT_WS, manifest)
    allowed = writable_paths(gold)
    for rel in changed:
        if rel in allowed:
            continue
        if rel.endswith(".pyc") or "__pycache__" in rel or ".pytest_cache" in rel:
            continue
        state.raise_integrity("write_outside_whitelist", f"wrote outside whitelist: {rel}")
        break

    ok, mismatches = readonly_tree_hashes_ok(AGENT_WS, gold)
    if not ok:
        state.raise_integrity("immutable_slice_mutated", "readonly tree mutated: " + ", ".join(mismatches))

    tests_changed = [rel for rel in changed if rel.startswith("ops_digest/tests/")]
    if tests_changed:
        state.raise_integrity("tests_modified", "tests modified: " + ", ".join(tests_changed))

    for shim in ("sitecustomize.py", "usercustomize.py", "pytest.py"):
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim", f"shim present: {shim}")

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress", "network egress marker present")
    return changed


def compute_milestones(state: State, visible_ok: bool, hidden_ok: bool, surfaces, hidden) -> dict[str, bool]:
    milestones = {
        "M1_localization": surfaces.schema_mentions_rollover and surfaces.digest_mentions_latest_per_day,
        "M2_primary_fix": visible_ok,
        "M3_invariants": state.integrity_flag == 0 and not state.shortcut_detected,
        "M4_functional": visible_ok and hidden_ok and hidden.required_milestone_blocks and hidden.advisory_non_blocking and hidden.latest_of_day_selected,
        "M5_e2e": visible_ok and hidden_ok and surfaces.generated_digest_matches_output and surfaces.automation_prompt_ok and surfaces.runbook_wording_ok and state.ceiling_cap > 30,
    }
    if state.integrity_flag == 1:
        milestones["M3_invariants"] = False
        milestones["M4_functional"] = False
        milestones["M5_e2e"] = False
    return milestones


def milestone_vector(milestones: dict[str, bool]) -> dict:
    slots = [
        {"milestone_id": "M1_localization", "weight": 0.10, "passed_bool": milestones["M1_localization"]},
        {"milestone_id": "M2_primary_fix", "weight": 0.20, "passed_bool": milestones["M2_primary_fix"]},
        {"milestone_id": "M3_invariants", "weight": 0.20, "passed_bool": milestones["M3_invariants"]},
        {"milestone_id": "M4_functional", "weight": 0.20, "passed_bool": milestones["M4_functional"]},
        {"milestone_id": "M5_e2e", "weight": 0.30, "passed_bool": milestones["M5_e2e"]},
    ]
    agg = round(sum(slot["weight"] for slot in slots if slot["passed_bool"]), 4)
    return {"slots": slots, "M_aggregate": agg}


def main() -> int:
    start = time.time()
    gold = load_gold(VARIANT_ID)
    manifest = load_manifest(VARIANT_ID)
    state = State()
    changed = trusted_final_state(state, gold, manifest)

    surfaces = inspect_surfaces(AGENT_WS, manifest, gold)
    hidden = hidden_scenarios(AGENT_WS)

    visible_ok = run_pytest(["ops_digest/tests"])
    if visible_ok:
        state.add("visible.pytest_passes", 20)
    else:
        state.add_error("visible pytest failed")

    classification_ok = hidden.required_milestone_blocks and hidden.advisory_non_blocking and hidden.mixed_milestone_shapes_parse
    if classification_ok:
        state.add("hidden.classification_logic", 20)
    else:
        state.apply_ceiling("missing_required_milestone_ignored", 30)
        state.add_error("classification logic still misses required-milestone or advisory contract")

    if hidden.latest_of_day_selected and hidden.no_duplicate_same_day_lines:
        state.add("hidden.latest_of_day", 15)
    else:
        state.apply_ceiling("latest_of_day_wrong", 35)
        state.add_error("latest-of-day or duplicate suppression still wrong")

    if surfaces.automation_singleton_ok and surfaces.automation_prompt_ok and surfaces.automation_summary_ok:
        state.add("automation.prompt_contract", 10)
    else:
        state.add_error("automation prompt/title contract not repaired in place")
        if not surfaces.automation_singleton_ok:
            state.apply_ceiling("second_automation_created", 20)

    if surfaces.generated_digest_matches_output:
        state.add("artifact.generated_digest", 10)
    else:
        state.apply_ceiling("stale_generated_digest", 20)
        state.add_error("generated digest does not match current code output")

    if surfaces.runbook_command_ok and surfaces.runbook_wording_ok:
        state.add("docs.runbook_contract", 10)
    else:
        state.add_error("runbook command or wording contract still stale")

    hidden_ok = run_pytest(
        [str((REPO / "verifier_data/nightly-regression-watch" / VARIANT_ID / "hidden_tests" / "test_hidden_contract.py").resolve())],
        extra_path=str((REPO / "verifier_data/nightly-regression-watch/_shared").resolve()),
    )
    if hidden_ok:
        state.add("hidden.pytest_passes", 15)
    else:
        state.add_error("hidden pytest contract failed")

    if not surfaces.code_files_changed and any(rel.endswith(".md") or rel.endswith(".toml") for rel in changed):
        state.apply_ceiling("docs_only_repair", 25)

    state.milestones = compute_milestones(state, visible_ok, hidden_ok, surfaces, hidden)
    final_score = state.final_score()
    result = {
        "pass": final_score >= PASS_BAR and state.integrity_flag == 0 and visible_ok and hidden_ok and surfaces.generated_digest_matches_output,
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_M_score,
        "milestones": state.milestones,
        "milestone_vector": milestone_vector(state.milestones),
        "breakdown": {**dict(sorted(state.breakdown.items())), "__bands": dict(sorted(state.bands.items()))},
        "ceilings_applied": sorted(state.ceilings_applied),
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": sorted(state.integrity_rules_fired),
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "wall_clock_seconds": int(round(time.time() - start)),
        "schema_version": SCHEMA_VERSION,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
