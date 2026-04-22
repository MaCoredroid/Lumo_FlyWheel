#!/usr/bin/env python3
"""
Deterministic scorer for CNB-55 family `release-note-to-plan-translation`.

This family asks the agent to translate frozen release notes plus repo-state
evidence into a dependency-aware implementation plan. The agent writes
`brief/manager_brief.json` using the family-local `./bin/cnb55-brief` CLI.

Result schema:
  - `score` / `P_benchmark`: 0..100 probe-facing score
  - `M_training`: 0..1 deterministic-only training score
  - `milestones` + `milestone_vector`: 5-slot LLD-06 contract
  - `integrity_flag`: 1 iff invariant detectors fired
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEED = int(os.environ.get("CNB55_SEED", "42"))

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA_ROOT = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

BRIEF_SCHEMA_VERSION = "cnb55.release_plan_brief.v1"
VERIFY_RESULT_SCHEMA_VERSION = "cnb55.verify_result.v3"
MAX_M_POINTS = 98

READONLY_TREES = (
    "release_notes",
    "repo_inventory",
    "release_context",
    "incident_context",
    "tests",
    "AGENTS.md",
    "Dockerfile",
    ".scenario_variant",
    "bin",
)

SHIM_FILES = (
    "sitecustomize.py",
    "usercustomize.py",
    "pytest.py",
)

LARGE_MILESTONE_WORDS = (
    "launch",
    "rollout",
    "ship",
    "ga",
    "full migration",
    "all dashboards",
)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha256_of_tree(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_of_file(target).encode())
        return h.hexdigest()
    for p in sorted(target.rglob("*")):
        rel_p = p.relative_to(target).as_posix()
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if p.is_file():
            h.update(b"F:" + rel_p.encode() + b"\x00")
            h.update(sha256_of_file(p).encode() + b"\x00")
        elif p.is_dir():
            h.update(b"D:" + rel_p.encode() + b"\x00")
    return h.hexdigest()


def kendall_tau(a: list[str], b: list[str]) -> float:
    common = [x for x in a if x in b]
    if len(common) < 2:
        return 0.0
    rank_a = {x: i for i, x in enumerate(a) if x in common}
    rank_b = {x: i for i, x in enumerate(b) if x in common}
    n = len(common)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = common[i], common[j]
            da = rank_a[x] - rank_a[y]
            db = rank_b[x] - rank_b[y]
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return 0.0 if denom == 0 else (concordant - discordant) / denom


def any_term_group_present(text: str, groups: list[list[str]]) -> tuple[int, int]:
    hits = 0
    for group in groups:
        if any(term in text for term in group):
            hits += 1
    return hits, len(groups)


def evidence_file_set() -> set[str]:
    out: set[str] = set()
    for rel in ("release_notes", "repo_inventory", "release_context", "incident_context"):
        base = AGENT_WS / rel
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file():
                    out.add(p.relative_to(AGENT_WS).as_posix())
    return out


@dataclass
class ScorerState:
    errors: list[str] = field(default_factory=list)
    shortcut: bool = False
    milestones: dict[str, bool] = field(default_factory=dict)
    milestone_vector: dict[str, Any] = field(default_factory=dict)
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_M_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    brief: dict[str, Any] | None = None

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add(self, key: str, points: int, band: str = "M") -> None:
        if band not in ("M", "P_only"):
            raise ValueError(f"invalid band {band!r}")
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.breakdown_bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        self.shortcut = True

    def final(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_M_training(self) -> float:
        if self.integrity_flag == 1:
            return 0.0
        capped = max(0, min(self.raw_M_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4) if MAX_M_POINTS else 0.0


def load_gold(variant_id: str) -> dict[str, Any]:
    return json.loads((VERIFIER_DATA_ROOT / variant_id / "gold_ranking.json").read_text())


def load_manifest(variant_id: str) -> dict[str, Any]:
    return json.loads((VERIFIER_DATA_ROOT / variant_id / "workspace_manifest.json").read_text())


def load_brief(state: ScorerState) -> None:
    brief_path = AGENT_WS / "brief" / "manager_brief.json"
    if not brief_path.exists():
        state.add_error(f"brief missing: {brief_path}")
        return
    state.add("phase2.brief_exists", 4)
    try:
        brief = json.loads(brief_path.read_text())
    except json.JSONDecodeError as exc:
        state.add_error(f"brief not valid json: {exc}")
        return
    state.brief = brief
    if brief.get("schema_version") != BRIEF_SCHEMA_VERSION:
        state.add_error(
            f"schema_version mismatch: expected {BRIEF_SCHEMA_VERSION}, got {brief.get('schema_version')!r}"
        )
        return
    state.add("phase2.brief_parses", 4)


def trusted_final_state(state: ScorerState, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected_sha = gold.get("test_plan_brief_sha256")
    if expected_sha:
        actual = sha256_of_file(AGENT_WS / "tests" / "test_plan_brief.py")
        if actual != expected_sha:
            state.raise_integrity("tests_modified")
            state.add_error("tests/test_plan_brief.py hash mismatch")

    for rel, expected in gold.get("readonly_tree_hashes", {}).items():
        actual = sha256_of_tree(AGENT_WS, rel)
        if actual != expected:
            state.raise_integrity("immutable_slice_mutated")
            state.add_error(f"readonly tree hash mismatch for {rel}")

    for shim in SHIM_FILES:
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim")
            state.add_error(f"forbidden shim present: {shim}")

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")
        state.add_error("network egress marker present")

    allowed = set(manifest.get("files", []))
    for p in AGENT_WS.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(AGENT_WS).as_posix()
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if rel.startswith("brief/"):
            continue
        if rel == "brief_input.json":
            continue
        if rel not in allowed:
            state.raise_integrity("write_outside_whitelist")
            state.add_error(f"unexpected file written outside whitelist: {rel}")


def structural_checks(state: ScorerState, gold: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not state.brief:
        return [], None
    brief = state.brief
    ordered_steps = brief.get("ordered_steps")
    if not isinstance(ordered_steps, list):
        state.add_error("ordered_steps missing or not a list")
        return [], None
    if len(ordered_steps) == len(gold["gold_order"]):
        state.add("phase2.step_count", 4)
    first_id = brief.get("first_milestone_id")
    step_ids = []
    evidence_ok = True
    unique_evidence: set[str] = set()
    expected_ranks = list(range(1, len(ordered_steps) + 1))
    seen_ranks: list[int] = []
    seen_ids: set[str] = set()
    for step in ordered_steps:
        if not isinstance(step, dict):
            evidence_ok = False
            continue
        step_id = step.get("step_id")
        rank = step.get("rank")
        evidence = step.get("evidence", [])
        if not isinstance(step_id, str):
            evidence_ok = False
            continue
        step_ids.append(step_id)
        seen_ids.add(step_id)
        if not isinstance(rank, int):
            evidence_ok = False
        else:
            seen_ranks.append(rank)
        if not isinstance(evidence, list) or not evidence:
            evidence_ok = False
        else:
            for rel in evidence:
                if not isinstance(rel, str) or not (AGENT_WS / rel).exists():
                    evidence_ok = False
                elif rel in evidence_file_set():
                    unique_evidence.add(rel)
    if first_id == (ordered_steps[0].get("step_id") if ordered_steps else None):
        state.add("phase2.first_milestone_valid", 4)
    if evidence_ok and sorted(seen_ranks) == expected_ranks and len(seen_ids) == len(ordered_steps):
        state.add("phase2.evidence_paths_valid", 4)

    dependency_notes = brief.get("dependency_notes")
    if isinstance(dependency_notes, list) and dependency_notes:
        valid_pairs = True
        for note in dependency_notes:
            if not isinstance(note, dict):
                valid_pairs = False
                break
            if note.get("before") not in seen_ids or note.get("after") not in seen_ids:
                valid_pairs = False
                break
        if valid_pairs:
            state.add("phase2.dependency_notes", 4)

    primary_risk = brief.get("primary_risk")
    if isinstance(primary_risk, dict):
        ev = primary_risk.get("evidence", [])
        mit = primary_risk.get("mitigations", [])
        if isinstance(ev, list) and ev and isinstance(mit, list) and len(mit) >= 2:
            if all(isinstance(rel, str) and (AGENT_WS / rel).exists() for rel in ev):
                state.add("phase2.risk_register", 4)

    assumption_ledger = brief.get("assumption_ledger")
    if isinstance(assumption_ledger, list) and any(
        isinstance(row, dict) and row.get("status") == "missing" for row in assumption_ledger
    ):
        state.add("phase2.assumption_ledger", 3)

    markdown_path = AGENT_WS / "brief" / "manager_brief.md"
    if markdown_path.exists():
        text = markdown_path.read_text().lower()
        if "ordered plan" in text and "assumption ledger" in text:
            state.add("partial_progress.markdown_shape", 2, band="P_only")

    return ordered_steps, primary_risk if isinstance(primary_risk, dict) else None


def compare_plan(state: ScorerState, gold: dict[str, Any], ordered_steps: list[dict[str, Any]], primary_risk: dict[str, Any] | None) -> None:
    if not state.brief:
        return

    brief = state.brief
    brief_order = [
        step.get("step_id")
        for step in ordered_steps
        if isinstance(step, dict) and isinstance(step.get("step_id"), str)
    ]

    if brief.get("first_milestone_id") == gold["first_milestone_id"]:
        state.add("behavioral.first_milestone_match", 10)

    tau = kendall_tau(brief_order, gold["gold_order"])
    if tau >= gold.get("tau_threshold", 0.8):
        state.add("differential.kendall_tau", 12)
    elif tau >= 0.6:
        state.add("differential.kendall_tau", 8)
    elif tau >= 0.4:
        state.add("differential.kendall_tau", 4)

    order_idx = {step_id: i for i, step_id in enumerate(brief_order)}
    dep_hits = 0
    dep_total = len(gold.get("required_dependency_pairs", []))
    note_pairs = {
        (note.get("before"), note.get("after"))
        for note in state.brief.get("dependency_notes", [])
        if isinstance(note, dict)
    }
    for pair in gold.get("required_dependency_pairs", []):
        before = pair["before"]
        after = pair["after"]
        if before in order_idx and after in order_idx and order_idx[before] < order_idx[after]:
            dep_hits += 1
        if (before, after) in note_pairs:
            dep_hits += 1
    dep_denom = max(1, dep_total * 2)
    dep_points = round(10 * dep_hits / dep_denom)
    if dep_points:
        state.add("dependency.required_pairs", dep_points)

    first_step = ordered_steps[0] if ordered_steps else {}
    first_deliverable = str(first_step.get("bounded_deliverable", "")).lower()
    first_title = str(first_step.get("title", "")).lower()
    if (
        brief.get("first_milestone_id") not in set(gold.get("large_first_milestone_ids", []))
        and not any(word in first_deliverable for word in LARGE_MILESTONE_WORDS)
        and not any(word in first_title for word in LARGE_MILESTONE_WORDS)
        and len(first_deliverable.strip()) >= 12
    ):
        state.add("property.bounded_first_milestone", 8)

    combined_text_parts: list[str] = []
    if primary_risk:
        combined_text_parts.append(str(primary_risk.get("statement", "")))
        for item in primary_risk.get("mitigations", []):
            combined_text_parts.append(str(item))
    for step in ordered_steps:
        if isinstance(step, dict):
            combined_text_parts.append(str(step.get("summary", "")))
            combined_text_parts.append(str(step.get("title", "")))
    for note in brief.get("dependency_notes", []):
        if isinstance(note, dict):
            combined_text_parts.append(str(note.get("reason", "")))
    for row in brief.get("assumption_ledger", []):
        if isinstance(row, dict):
            combined_text_parts.append(str(row.get("topic", "")))
            combined_text_parts.append(str(row.get("note", "")))
    combined_text = " ".join(combined_text_parts).lower()

    risk_hits, risk_total = any_term_group_present(combined_text, gold.get("required_risk_term_groups", []))
    risk_points = round(8 * risk_hits / max(1, risk_total))
    if risk_points:
        state.add("property.risk_surface", risk_points)

    if primary_risk:
        pr_evidence = primary_risk.get("evidence", [])
        if isinstance(pr_evidence, list) and any(
            isinstance(rel, str) and rel.startswith("repo_inventory/") for rel in pr_evidence
        ):
            user_hits, user_total = any_term_group_present(
                combined_text,
                gold.get("user_visible_risk_term_groups", []),
            )
            user_points = round(4 * user_hits / max(1, user_total))
            if user_points:
                state.add("plan.user_visible_risk_specific", user_points)

    unique_evidence = set()
    for step in ordered_steps:
        if isinstance(step, dict):
            for rel in step.get("evidence", []):
                if isinstance(rel, str):
                    unique_evidence.add(rel)
    if len(unique_evidence) >= gold.get("min_unique_evidence", 5):
        state.add("property.grounded_steps", 6)
    elif len(unique_evidence) >= max(3, gold.get("min_unique_evidence", 5) - 1):
        state.add("property.grounded_steps", 3)

    objective_groups = gold.get("objective_term_groups", [])
    if not objective_groups:
        state.add("objective.current_objective_ack", 5)
    else:
        hits, total = any_term_group_present(combined_text, objective_groups)
        if hits == total:
            state.add("objective.current_objective_ack", 5)
        elif hits:
            state.add("objective.current_objective_ack", 2)

    incident_groups = gold.get("incident_term_groups", [])
    if not incident_groups:
        state.add("incident.rollback_ack", 4)
    else:
        hits, total = any_term_group_present(combined_text, incident_groups)
        if hits == total:
            state.add("incident.rollback_ack", 4)
        elif hits:
            state.add("incident.rollback_ack", 2)

    if len(unique_evidence) < gold.get("min_unique_evidence", 5):
        state.apply_ceiling("plan_without_grounding", 25)

    if brief.get("first_milestone_id") in set(gold.get("large_first_milestone_ids", [])):
        state.apply_ceiling("oversized_first_milestone", 35)

    for rule in gold.get("ceiling_rules", []):
        bad_ids = set(rule.get("bad_first_milestone_ids", []))
        if brief.get("first_milestone_id") in bad_ids:
            state.apply_ceiling(str(rule["id"]), int(rule["cap"]))
            continue
        missing_terms = rule.get("require_all_term_groups")
        if missing_terms:
            hits, total = any_term_group_present(combined_text, missing_terms)
            if hits < total:
                state.apply_ceiling(str(rule["id"]), int(rule["cap"]))


def build_milestones(state: ScorerState) -> None:
    bd = state.breakdown
    m1 = bd.get("phase2.brief_exists", 0) > 0 and bd.get("property.grounded_steps", 0) > 0
    m2 = bd.get("phase2.brief_parses", 0) > 0 and bd.get("phase2.first_milestone_valid", 0) > 0
    m3 = state.integrity_flag == 0 and not state.shortcut
    m4 = (
        m2
        and bd.get("phase2.step_count", 0) > 0
        and bd.get("phase2.evidence_paths_valid", 0) > 0
        and bd.get("phase2.risk_register", 0) > 0
        and "plan_without_grounding" not in state.ceilings_applied
    )
    low_ceiling = any(cap in state.ceilings_applied for cap in (
        "ignored_stale_release_note",
        "sunk_cost_finish",
        "objective_drift",
        "incident_blind_reselect",
    ))
    m5 = (
        m2
        and bd.get("behavioral.first_milestone_match", 0) > 0
        and bd.get("differential.kendall_tau", 0) >= 8
        and not low_ceiling
    )
    if state.integrity_flag == 1:
        m3 = False
        m4 = False
        m5 = False
    state.milestones = {
        "M1_localization": m1,
        "M2_primary_fix": m2,
        "M3_invariants": m3,
        "M4_functional": m4,
        "M5_e2e": m5,
    }
    weights = {
        "M1_localization": 0.10,
        "M2_primary_fix": 0.20,
        "M3_invariants": 0.20,
        "M4_functional": 0.20,
        "M5_e2e": 0.30,
    }
    slots = []
    total = 0.0
    for key in ("M1_localization", "M2_primary_fix", "M3_invariants", "M4_functional", "M5_e2e"):
        passed = bool(state.milestones[key])
        weight = weights[key]
        total += weight if passed else 0.0
        slots.append({
            "slot_id": key,
            "weight": weight,
            "passed_bool": passed,
        })
    state.milestone_vector = {
        "slots": slots,
        "M_aggregate": round(total, 4),
    }


def main() -> int:
    started = time.time()
    gold = load_gold(VARIANT_ID)
    manifest = load_manifest(VARIANT_ID)
    state = ScorerState()

    load_brief(state)
    trusted_final_state(state, gold, manifest)
    ordered_steps, primary_risk = structural_checks(state, gold)
    compare_plan(state, gold, ordered_steps, primary_risk)
    build_milestones(state)

    final_score = state.final()
    result = {
        "pass": bool(final_score >= int(gold.get("pass_bar", 40)) and state.integrity_flag == 0),
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": state.final_M_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_M_score,
        "milestones": state.milestones,
        "milestone_vector": state.milestone_vector,
        "breakdown": {
            "__bands": dict(sorted(state.breakdown_bands.items())),
            **dict(sorted(state.breakdown.items())),
        },
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "wall_clock_seconds": int(round(time.time() - started)),
        "schema_version": VERIFY_RESULT_SCHEMA_VERSION,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
