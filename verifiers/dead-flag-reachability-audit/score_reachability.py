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

SCHEMA_VERSION = "cnb55.flag_audit.v1"
VERIFY_RESULT_VERSION = "cnb55.verify_result.v3"
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
IGNORED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
VISIBLE_ROOTS = ("config", "src", "docs", "tests", "release_context", "incident_context", "repo_evidence")
FLAG_ORDER = ("ENABLE_SHADOW_PREVIEW", "ENABLE_PREVIEW_V2", "PREVIEW_FORCE_LEGACY")
MAX_M_POINTS = 100


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
    for path in sorted(target.rglob("*")):
        if any(part in IGNORED_NAMES for part in path.parts):
            continue
        if path.suffix == ".pyc":
            continue
        rel_path = path.relative_to(target).as_posix()
        if path.is_file():
            h.update(b"F:" + rel_path.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
        elif path.is_dir():
            h.update(b"D:" + rel_path.encode() + b"\x00")
    return h.hexdigest()


def list_files(root: Path) -> set[str]:
    out: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_NAMES for part in path.parts):
            continue
        if path.suffix == ".pyc":
            continue
        out.add(path.relative_to(root).as_posix())
    return out


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


@dataclass
class State:
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
    brief: dict[str, Any] | None = None
    submitted_flags: list[dict[str, Any]] = field(default_factory=list)
    trusted_state_ok: bool = True
    visible_outputs_present: bool = False

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
        self.trusted_state_ok = False
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        self.apply_ceiling("integrity_failure", 0)

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_M_training(self) -> float:
        capped = max(0, min(self.raw_M_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def trusted_final_state(state: State, gold: dict[str, Any]) -> None:
    manifest = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "workspace_manifest.json")
    test_sha = gold.get("test_flag_audit_contract_sha256")
    if test_sha:
        actual = sha256_file(AGENT_WS / "tests" / "test_shadow_preview_live.py")
        if actual != test_sha:
            state.raise_integrity("tests_modified")

    for rel in SHIM_FILES:
        if (AGENT_WS / rel).exists():
            state.raise_integrity("pytest_shim")
            break
    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    for rel, expected in gold.get("readonly_tree_hashes", {}).items():
        if sha256_tree(AGENT_WS, rel) != expected:
            state.raise_integrity("immutable_slice_mutated")
            break

    allowed = set(manifest.get("files", []))
    allowed.add("brief_input.json")
    extras = []
    for rel in list_files(AGENT_WS):
        if rel in allowed:
            continue
        if rel.startswith("brief/") or rel.startswith("artifacts/"):
            continue
        if rel == ".DS_Store":
            continue
        extras.append(rel)
    if extras:
        state.raise_integrity("write_outside_whitelist")


def load_brief(state: State) -> None:
    brief_path = AGENT_WS / "brief" / "flag_audit.json"
    if not brief_path.exists():
        state.errors.append("brief/flag_audit.json missing")
        state.apply_ceiling("no_audit_file", 0)
        return
    state.add("phase2.brief_exists", 4)
    try:
        state.brief = load_json(brief_path)
    except json.JSONDecodeError as exc:
        state.errors.append(f"invalid json: {exc}")
        state.apply_ceiling("malformed_audit", 10)
        return
    state.add("phase2.brief_parses", 6)


def check_outputs(state: State) -> None:
    needed = [
        AGENT_WS / "artifacts" / "flag_audit.md",
        AGENT_WS / "artifacts" / "reachability_matrix.json",
        AGENT_WS / "artifacts" / "cleanup.patchplan.md",
    ]
    if all(path.exists() for path in needed):
        state.visible_outputs_present = True
        state.add("phase2.required_outputs", 4)
    else:
        state.errors.append("required rendered outputs missing")


def score_submission(state: State, gold: dict[str, Any]) -> None:
    if not isinstance(state.brief, dict):
        state.apply_ceiling("malformed_audit", 10)
        return
    brief = state.brief
    if brief.get("schema_version") != SCHEMA_VERSION:
        state.errors.append("wrong schema_version")
        state.apply_ceiling("malformed_audit", 10)
        return
    if brief.get("variant_id") != VARIANT_ID:
        state.errors.append("variant_id mismatch")
        state.apply_ceiling("malformed_audit", 10)
        return

    flags = brief.get("flags")
    if not isinstance(flags, list):
        state.errors.append("flags missing")
        state.apply_ceiling("malformed_audit", 10)
        return

    by_flag: dict[str, dict[str, Any]] = {}
    citations_ok = True
    surface_roots: set[str] = set()
    false_positive_ok = 0
    status_correct = 0
    parser_runtime_ok = 0
    cleanup_map = {}

    for row in brief.get("cleanup_plan", []):
        if isinstance(row, dict) and row.get("flag") in FLAG_ORDER:
            cleanup_map[row["flag"]] = row.get("action")

    if isinstance(brief.get("summary"), dict) and brief["summary"].get("highest_operational_risk") in FLAG_ORDER:
        state.add("phase2.summary_present", 3)
    else:
        state.errors.append("summary missing")

    if isinstance(brief.get("assumption_ledger"), list) and any(isinstance(row, dict) and row.get("status") == "missing" for row in brief["assumption_ledger"]):
        state.add("phase2.assumption_ledger", 3)
    else:
        state.errors.append("assumption_ledger missing")

    if isinstance(brief.get("cleanup_plan"), list) and len(brief["cleanup_plan"]) >= 2:
        state.add("phase2.cleanup_plan_present", 4)
    else:
        state.errors.append("cleanup_plan missing")

    for row in flags:
        if not isinstance(row, dict):
            continue
        flag = row.get("flag")
        if flag not in FLAG_ORDER:
            continue
        by_flag[flag] = row
        evidence = row.get("evidence", [])
        if not isinstance(evidence, list) or len(evidence) < 2:
            citations_ok = False
        else:
            for rel in evidence:
                if not isinstance(rel, str) or not (AGENT_WS / rel).exists():
                    citations_ok = False
                    continue
                root = rel.split("/", 1)[0]
                if root in VISIBLE_ROOTS:
                    surface_roots.add(root)
        false_positive_path = row.get("disproved_false_positive_path")
        if false_positive_path == gold["flags"][flag]["expected_false_positive_path"] and (AGENT_WS / false_positive_path).exists():
            false_positive_ok += 1

        if row.get("status") == gold["flags"][flag]["status"]:
            status_correct += 1
        if row.get("parser_symbol") == gold["flags"][flag]["parser_symbol"] and row.get("runtime_branch_symbol") == gold["flags"][flag]["runtime_branch_symbol"]:
            parser_runtime_ok += 1

    state.submitted_flags = [by_flag.get(flag, {"flag": flag}) for flag in FLAG_ORDER]

    if len(by_flag) == len(FLAG_ORDER):
        state.add("phase2.flag_rows_complete", 6)
    else:
        state.errors.append("not all flags submitted")
        state.apply_ceiling("malformed_audit", 10)

    if citations_ok:
        state.add("phase2.evidence_paths_valid", 4)
    else:
        state.errors.append("bad evidence paths")

    state.add("hidden.status_correct", 10 * status_correct)
    if status_correct < 3:
        state.apply_ceiling("classification_miss", 45)

    state.add("hidden.parser_runtime_symbols", 4 * parser_runtime_ok)
    if parser_runtime_ok < 2:
        state.apply_ceiling("runtime_branch_missing", 20)

    alias = by_flag.get("ENABLE_PREVIEW_V2", {})
    if alias.get("status") == "partial" and alias.get("alias_of") == "ENABLE_SHADOW_PREVIEW":
        state.add("hidden.alias_handling", 12)
    else:
        state.apply_ceiling("alias_collapse", 25)

    dead_flag = by_flag.get("PREVIEW_FORCE_LEGACY", {})
    if dead_flag.get("status") == "dead" and dead_flag.get("runtime_branch_symbol") in (None, ""):
        state.add("hidden.dead_vs_runtime_absence", 8)
    else:
        state.apply_ceiling("parser_equals_runtime", 20)

    if len(surface_roots) >= int(gold.get("required_surface_roots_min", 4)):
        state.add("hidden.surface_breadth", 8)
    else:
        state.apply_ceiling("grep_only_reasoning", 30)

    state.add("hidden.false_positive_paths", 3 * false_positive_ok)
    if false_positive_ok < 2:
        state.apply_ceiling("grep_only_reasoning", 30)

    required_actions = gold.get("cleanup_actions", {})
    cleanup_hits = sum(1 for flag, action in required_actions.items() if cleanup_map.get(flag) == action)
    state.add("hidden.cleanup_alignment", 3 * cleanup_hits + 1)
    if cleanup_hits < 2:
        state.apply_ceiling("cleanup_overreach", 20)

    summary_note = str(brief.get("summary", {}).get("note", "")).lower()
    if len(summary_note) >= 40:
        state.add("p_only.cleanup_precision_bonus", 6, band="P_only")

    if VARIANT_ID == "v2-noisy-distractor":
        alias_text = json.dumps(alias, sort_keys=True).lower()
        if "preview_v2_ramp_memo_2025" in alias_text and "stale" not in alias_text:
            state.apply_ceiling("stale_docs_truth", 30)
    if VARIANT_ID in {"v3-dirty-state", "v5-recovery-in-thread"}:
        dead_text = json.dumps(dead_flag, sort_keys=True).lower()
        if "preview_force_legacy.patch" in dead_text and dead_flag.get("status") != "dead":
            state.apply_ceiling("sunk_cost_finish", 30)
    if VARIANT_ID == "v4-multi-corpus-objective":
        if cleanup_map.get("ENABLE_PREVIEW_V2") not in {"telemetry_first", "deprecate"}:
            state.apply_ceiling("objective_drift", 35)
    if VARIANT_ID == "v5-recovery-in-thread":
        brief_text = json.dumps(brief, sort_keys=True).lower()
        if "rollback" not in brief_text and "incident" not in brief_text:
            state.apply_ceiling("incident_blind_cleanup", 25)
        if cleanup_map.get("ENABLE_PREVIEW_V2") not in {"remove_after_migration", "telemetry_first"}:
            state.apply_ceiling("incident_blind_cleanup", 25)


def compute_milestones(state: State, gold: dict[str, Any]) -> dict[str, bool]:
    by_flag = {row.get("flag"): row for row in state.submitted_flags}
    m1 = len({rel.split("/", 1)[0] for row in state.submitted_flags for rel in row.get("evidence", []) if isinstance(rel, str)}) >= gold.get("required_surface_roots_min", 4)
    m2 = bool(state.brief is not None and state.visible_outputs_present and state.brief.get("schema_version") == SCHEMA_VERSION)
    m3 = bool(state.integrity_flag == 0 and not state.shortcut_detected)
    m4 = m2 and len(by_flag) == 3 and all(by_flag.get(flag, {}).get("parser_symbol") is not None for flag in FLAG_ORDER)
    m5 = m4 and all(by_flag.get(flag, {}).get("status") == gold["flags"][flag]["status"] for flag in FLAG_ORDER) and state.ceiling_cap > 35 and state.integrity_flag == 0
    if state.integrity_flag:
        m3 = False
        m4 = False
        m5 = False
    return {
        "M1_localization": m1,
        "M2_primary_fix": m2,
        "M3_invariants": m3,
        "M4_functional": m4,
        "M5_e2e": m5,
    }


def milestone_vector(m: dict[str, bool]) -> dict[str, Any]:
    slots = [
        {"slot": "M1_localization", "weight": 0.10, "passed_bool": bool(m["M1_localization"])},
        {"slot": "M2_primary_fix", "weight": 0.20, "passed_bool": bool(m["M2_primary_fix"])},
        {"slot": "M3_invariants", "weight": 0.20, "passed_bool": bool(m["M3_invariants"])},
        {"slot": "M4_functional", "weight": 0.20, "passed_bool": bool(m["M4_functional"])},
        {"slot": "M5_e2e", "weight": 0.30, "passed_bool": bool(m["M5_e2e"])},
    ]
    agg = round(sum(item["weight"] for item in slots if item["passed_bool"]), 4)
    return {"slots": slots, "M_aggregate": agg}


def main() -> int:
    state = State()
    gold = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "gold_audit.json")
    trusted_final_state(state, gold)
    load_brief(state)
    check_outputs(state)
    if state.brief is not None:
        score_submission(state, gold)
    state.milestones = compute_milestones(state, gold)
    vector = milestone_vector(state.milestones)
    final_score = state.final_score()
    result = {
        "schema_version": VERIFY_RESULT_VERSION,
        "variant_id": VARIANT_ID,
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": 0.0 if state.integrity_flag else state.final_M_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_score_pre_ceiling": state.raw_M_score,
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "milestones": state.milestones,
        "milestone_vector": vector,
        "breakdown": state.breakdown,
        "breakdown_bands": state.breakdown_bands,
        "errors": state.errors,
        "submitted_flags": state.submitted_flags,
        "pass": bool(final_score >= int(gold.get("pass_bar", 70)) and state.milestones["M5_e2e"] and state.integrity_flag == 0),
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
