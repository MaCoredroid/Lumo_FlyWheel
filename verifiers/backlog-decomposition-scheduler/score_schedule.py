#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA_ROOT = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

SCHEMA_VERSION = "cnb55.schedule_brief.v1"
VERIFY_RESULT_VERSION = "cnb55.verify_result.v3"
CONSTRAINT_TAGS = {"dependency", "capacity", "rollout", "objective", "incident"}
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
IGNORED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
READONLY_TREES = (
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "bin",
    "backlog",
    "repo_evidence",
    "release_context",
    "incident_context",
    "tests",
)

# Deterministic M-band max.
MAX_M_POINTS = 110


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
    for p in sorted(target.rglob("*")):
        if any(part in IGNORED_NAMES for part in p.parts):
            continue
        if p.suffix == ".pyc":
            continue
        rel_p = p.relative_to(target).as_posix()
        if p.is_file():
            h.update(b"F:" + rel_p.encode() + b"\x00")
            h.update(sha256_file(p).encode() + b"\x00")
        elif p.is_dir():
            h.update(b"D:" + rel_p.encode() + b"\x00")
    return h.hexdigest()


def list_files(root: Path) -> set[str]:
    out: set[str] = set()
    for p in root.rglob("*"):
        if p.is_file():
            if any(part in IGNORED_NAMES for part in p.parts):
                continue
            if p.suffix == ".pyc":
                continue
            out.add(p.relative_to(root).as_posix())
    return out


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


@dataclass
class ScorerState:
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
    manifest_files: set[str] = field(default_factory=set)
    trusted_state_ok: bool = True

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

    def final(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_M_training(self) -> float:
        capped = max(0, min(self.raw_M_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def load_gold(variant_id: str) -> dict[str, Any]:
    return load_json(VERIFIER_DATA_ROOT / variant_id / "gold_schedule.json")


def trusted_final_state(state: ScorerState, gold: dict[str, Any]) -> None:
    manifest = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "workspace_manifest.json")
    state.manifest_files = set(manifest.get("files", []))

    test_sha = gold.get("test_schedule_brief_sha256")
    if test_sha:
        actual = sha256_file(AGENT_WS / "tests" / "test_schedule_brief.py")
        if actual != test_sha:
            state.raise_integrity("tests_modified")

    for rel in SHIM_FILES:
        if (AGENT_WS / rel).exists():
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    expected_tree_hashes = gold.get("readonly_tree_hashes", {})
    for rel, expected in expected_tree_hashes.items():
        actual = sha256_tree(AGENT_WS, rel)
        if actual != expected:
            state.raise_integrity("immutable_slice_mutated")
            break

    current_files = list_files(AGENT_WS)
    whitelist_prefixes = ("brief/",)
    allowed = set(state.manifest_files)
    allowed.add("brief_input.json")
    extras = []
    for rel in current_files:
        if rel in allowed or rel.startswith(whitelist_prefixes):
            continue
        if rel == ".DS_Store":
            continue
        extras.append(rel)
    if extras:
        state.raise_integrity("write_outside_whitelist")


def load_brief(state: ScorerState) -> None:
    brief_path = AGENT_WS / "brief" / "schedule_brief.json"
    if not brief_path.exists():
        state.errors.append("brief/schedule_brief.json missing")
        state.apply_ceiling("no_schedule_file", 0)
        return
    state.add("phase2.brief_exists", 4)
    try:
        state.brief = load_json(brief_path)
    except json.JSONDecodeError as exc:
        state.errors.append(f"invalid json: {exc}")
        state.apply_ceiling("malformed_brief", 10)
        return
    state.add("phase2.brief_parses", 6)


def validate_schedule_structure(state: ScorerState, gold: dict[str, Any]) -> dict[str, int]:
    assert state.brief is not None
    brief = state.brief
    item_ids = [i["item_id"] for i in gold["items"]]

    if brief.get("schema_version") != SCHEMA_VERSION:
        state.errors.append("wrong schema_version")
        state.apply_ceiling("malformed_brief", 10)
        return {}
    if brief.get("variant_id") != VARIANT_ID:
        state.errors.append("variant_id mismatch")
        state.apply_ceiling("malformed_brief", 10)
        return {}

    objective_focus = brief.get("objective_focus", "")
    if isinstance(objective_focus, str) and objective_focus.strip():
        state.add("phase2.objective_focus_present", 4)
    else:
        state.errors.append("objective_focus missing")

    schedule = brief.get("schedule")
    if not isinstance(schedule, list):
        state.errors.append("schedule missing")
        state.apply_ceiling("malformed_brief", 10)
        return {}

    seen: set[str] = set()
    positions: dict[str, int] = {}
    valid_entries = 0
    citations_entries = 0
    valid_tags_entries = 0
    slots: list[int] = []
    for entry in schedule:
        if not isinstance(entry, dict):
            continue
        item_id = entry.get("item_id")
        slot = entry.get("slot")
        lane = entry.get("lane")
        summary = entry.get("summary")
        citations = entry.get("citations")
        tags = entry.get("constraint_tags")
        if item_id in item_ids and item_id not in seen:
            seen.add(item_id)
            if isinstance(slot, int) and slot >= 1:
                positions[item_id] = slot
                slots.append(slot)
            if isinstance(lane, str) and lane.strip() and isinstance(summary, str) and summary.strip():
                valid_entries += 1
            if isinstance(citations, list) and citations and all(isinstance(c, str) for c in citations):
                citations_entries += 1
            if isinstance(tags, list) and tags and set(tags).issubset(CONSTRAINT_TAGS):
                valid_tags_entries += 1

    if len(positions) == len(item_ids):
        state.add("phase2.schedule_length", 4)
    else:
        state.errors.append("schedule does not cover every backlog item exactly once")
        state.apply_ceiling("malformed_brief", 10)

    if valid_entries == len(item_ids):
        state.add("phase2.entry_shape", 4)

    if citations_entries >= max(1, len(item_ids) - 1):
        state.add("phase2.citations_present", 4)

    if valid_tags_entries >= max(1, len(item_ids) - 1):
        state.add("phase2.constraint_tags", 4)

    slot_set = set(slots)
    if slots and slot_set == set(range(1, max(slots) + 1)):
        state.add("phase2.schedule_slots", 4)
    else:
        state.errors.append("slots must be contiguous starting at 1")

    risk_gate = brief.get("risk_gate", {})
    if (
        isinstance(risk_gate, dict)
        and risk_gate.get("risky_item_id") == gold["risky_item_id"]
        and risk_gate.get("must_follow") == gold["risk_must_follow"]
        and isinstance(risk_gate.get("note"), str)
        and risk_gate["note"].strip()
    ):
        state.add("phase2.risk_gate_present", 4)
    else:
        state.errors.append("risk_gate missing or inconsistent")

    scarce_plan = brief.get("scarce_role_plan", {})
    if (
        isinstance(scarce_plan, dict)
        and scarce_plan.get("role") == gold["scarce_role_name"]
        and scarce_plan.get("protected_items") == gold["scarce_role_items"]
        and isinstance(scarce_plan.get("note"), str)
        and scarce_plan["note"].strip()
    ):
        state.add("phase2.scarce_role_plan", 4)
    else:
        state.errors.append("scarce_role_plan missing or inconsistent")

    ledger = brief.get("assumption_ledger")
    if isinstance(ledger, list) and any(isinstance(x, dict) and x.get("status") == "missing" for x in ledger):
        state.add("phase2.assumption_ledger", 4)
    else:
        state.errors.append("assumption_ledger missing a missing row")

    if state.trusted_state_ok:
        state.add("phase2.trusted_state", 4)

    return positions


def score_schedule(state: ScorerState, gold: dict[str, Any], positions: dict[str, int]) -> None:
    if not positions or state.brief is None:
        return

    brief = state.brief
    schedule = brief["schedule"]
    entries = {entry["item_id"]: entry for entry in schedule if isinstance(entry, dict) and "item_id" in entry}

    dep_total = len(gold["dependency_edges"])
    dep_ok = 0
    for before, after in gold["dependency_edges"]:
        if positions.get(before, 10**9) < positions.get(after, -1):
            dep_ok += 1
    if dep_total:
        state.add("behavioral.dependency_edges", round(18 * dep_ok / dep_total))

    scarce_items = gold["scarce_role_items"]
    slot_counts: dict[int, int] = {}
    for item_id in scarce_items:
        slot = positions.get(item_id)
        if slot is not None:
            slot_counts[slot] = slot_counts.get(slot, 0) + 1
    scarce_conflict = any(v > 1 for v in slot_counts.values())
    if not scarce_conflict:
        state.add("behavioral.scarce_role_protected", 12)

    risky_item = gold["risky_item_id"]
    risk_ok = all(positions.get(dep, 10**9) < positions.get(risky_item, -1) for dep in gold["risk_must_follow"])
    if risk_ok:
        state.add("behavioral.risk_gate", 10)

    objective_focus = str(brief.get("objective_focus", "")).lower()
    keywords = [kw.lower() for kw in gold.get("objective_keywords", [])]
    if any(kw in objective_focus for kw in keywords):
        state.add("objective.focus_named", 4)

    objective_pairs = gold.get("objective_order_pairs", [])
    objective_ok = True
    for earlier, later in objective_pairs:
        if positions.get(earlier, 10**9) >= positions.get(later, -1):
            objective_ok = False
            break
    if objective_ok:
        state.add("objective.ordering", 6)

    slot_targets = gold["slot_targets"]
    exact = sum(1 for item_id, slot in slot_targets.items() if positions.get(item_id) == slot)
    state.add("differential.slot_accuracy", round(8 * exact / max(1, len(slot_targets))))

    grounding_count = 0
    valid_grounding = 0
    for item_id, entry in entries.items():
        citations = entry.get("citations", [])
        if isinstance(citations, list) and citations:
            grounding_count += 1
            if all((AGENT_WS / rel).exists() for rel in citations):
                valid_grounding += 1
    state.add("property.grounded_entries", round(6 * valid_grounding / max(1, len(slot_targets))))

    critical_path = gold.get("critical_path", [])
    critical_ok = 0
    for i in range(len(critical_path) - 1):
        a = critical_path[i]
        b = critical_path[i + 1]
        if positions.get(a, 10**9) <= positions.get(b, -1):
            critical_ok += 1
    if critical_path:
        state.add("objective.critical_path", round(8 * critical_ok / max(1, len(critical_path) - 1)))

    # P-only shaping signal: reward correct prefix / partial reasoning without
    # letting it enter the training band.
    prefix_hits = 0
    for item_id in gold.get("preferred_prefix", []):
        if positions.get(item_id, 10**9) <= gold["slot_targets"].get(item_id, 10**9):
            prefix_hits += 1
    if gold.get("preferred_prefix"):
        state.add("partial_progress.heuristic", round(10 * prefix_hits / len(gold["preferred_prefix"])), band="P_only")

    if grounding_count < len(slot_targets) - 1:
        state.apply_ceiling("schedule_without_grounding", 25)

    hard_dep_broken = any(
        positions.get(before, 10**9) >= positions.get(after, -1)
        for before, after in gold.get("hard_dependency_edges", gold["dependency_edges"])
    )
    if hard_dep_broken:
        state.apply_ceiling("ignored_dependency_blocker", 30)

    if scarce_conflict:
        state.apply_ceiling("overloaded_scarce_specialist", 30)

    if not risk_ok:
        state.apply_ceiling("risky_rollout_frontloaded", 25)

    stale_item = gold.get("stale_priority_item")
    if stale_item:
        stale_after = gold.get("stale_anchor_after", [])
        if any(positions.get(stale_item, 10**9) <= positions.get(item, -1) for item in stale_after):
            state.apply_ceiling("stale_priority_anchor", 35)

    sunk_cost_item = gold.get("sunk_cost_item")
    if sunk_cost_item:
        if any(positions.get(sunk_cost_item, 10**9) < positions.get(item, -1) for item in gold.get("sunk_cost_after", [])):
            state.apply_ceiling("sunk_cost_finish", 40)

    if not objective_ok or (keywords and not any(kw in objective_focus for kw in keywords)):
        if gold.get("objective_drift_active", False):
            state.apply_ceiling("objective_drift", 45)

    incident_item = gold.get("incident_reselect_item")
    incident_guard = gold.get("incident_guard_item")
    if incident_item and incident_guard:
        if positions.get(incident_item, 10**9) <= positions.get(incident_guard, -1):
            state.apply_ceiling("incident_blind_reselect", 30)


def finalize_milestones(state: ScorerState) -> dict[str, Any]:
    breakdown = state.breakdown
    ceilings = set(state.ceilings_applied)
    state.milestones = {
        "M1_localization": breakdown.get("phase2.citations_present", 0) >= 4 or breakdown.get("property.grounded_entries", 0) >= 4,
        "M2_primary_fix": state.brief is not None and "malformed_brief" not in ceilings and "no_schedule_file" not in ceilings,
        "M3_invariants": state.integrity_flag == 0 and state.trusted_state_ok,
        "M4_functional": (
            breakdown.get("behavioral.dependency_edges", 0) >= 14
            and breakdown.get("behavioral.scarce_role_protected", 0) == 12
            and breakdown.get("behavioral.risk_gate", 0) == 10
        ),
        "M5_e2e": (
            breakdown.get("differential.slot_accuracy", 0) >= 6
            and "ignored_dependency_blocker" not in ceilings
            and "overloaded_scarce_specialist" not in ceilings
            and "risky_rollout_frontloaded" not in ceilings
            and state.integrity_flag == 0
        ),
    }
    slot_defs = [
        ("M1_localization", 0.10),
        ("M2_primary_fix", 0.20),
        ("M3_invariants", 0.20),
        ("M4_functional", 0.20),
        ("M5_e2e", 0.30),
    ]
    slots = []
    aggregate = 0.0
    for slot_id, weight in slot_defs:
        passed = bool(state.milestones[slot_id])
        slots.append({"slot_id": slot_id, "weight": weight, "passed_bool": passed})
        if passed:
            aggregate += weight
    return {"slots": slots, "M_aggregate": round(aggregate, 4)}


def main() -> int:
    start = time.time()
    gold = load_gold(VARIANT_ID)
    state = ScorerState()

    trusted_final_state(state, gold)
    load_brief(state)
    positions: dict[str, int] = {}
    if state.brief is not None:
        positions = validate_schedule_structure(state, gold)
        score_schedule(state, gold, positions)

    milestone_vector = finalize_milestones(state)
    pass_bar = int(gold.get("pass_bar", 40))
    result = {
        "pass": state.final() >= pass_bar and state.integrity_flag == 0,
        "score": state.final(),
        "P_benchmark": state.final(),
        "M_training": state.final_M_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_M_score,
        "milestones": state.milestones,
        "milestone_vector": milestone_vector,
        "breakdown": {"__bands": state.breakdown_bands, **state.breakdown},
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "wall_clock_seconds": int(time.time() - start),
        "schema_version": VERIFY_RESULT_VERSION,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
