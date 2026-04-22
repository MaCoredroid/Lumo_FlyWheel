#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA_ROOT = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

SCHEMA_VERSION = "cnb55.evolution_plan.v1"
MAX_M_POINTS = 92
NUMERIC_RE = re.compile(r"\b\d+\b")
READONLY_TREES = (
    "focus_candidates",
    "round_history",
    "objective_metrics",
    "repo_state",
    "release_context",
    "incident_context",
    "tests",
    "AGENTS.md",
    "Dockerfile",
    ".scenario_variant",
    "bin",
)
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")


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
        rp = p.relative_to(target).as_posix()
        if p.is_file():
            h.update(b"F:" + rp.encode() + b"\x00")
            h.update(sha256_file(p).encode() + b"\x00")
        elif p.is_dir():
            h.update(b"D:" + rp.encode() + b"\x00")
    return h.hexdigest()


@dataclass
class ScorerState:
    errors: list[str] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_M_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    shortcut_detected: bool = False
    milestones: dict[str, bool] = field(default_factory=dict)
    brief: dict[str, Any] | None = None

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.breakdown_bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def apply_ceiling(self, cap: int, name: str) -> None:
        self.ceiling_cap = min(self.ceiling_cap, cap)
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        self.ceiling_cap = 0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def all_citations(brief: dict[str, Any]) -> list[str]:
    cites = list(brief.get("selected_focus", {}).get("citations", []))
    for row in brief.get("do_not_touch", []):
        cites.extend(row.get("citations", []))
    return cites


def existing_file(rel: str) -> bool:
    p = AGENT_WS / rel
    return p.exists() and p.is_file()


def load_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    gold = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "gold_plan.json")
    manifest = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "workspace_manifest.json")
    return gold, manifest


def check_integrity(state: ScorerState, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    allowed_write_files = {"brief_input.json"}
    for rel in READONLY_TREES:
        expected = gold["readonly_tree_hashes"].get(rel)
        if expected and sha256_tree(AGENT_WS, rel) != expected:
            state.raise_integrity("immutable_slice_mutated")
            break

    test_sha = manifest.get("test_evolution_plan_sha256")
    test_file = AGENT_WS / "tests" / "test_evolution_plan.py"
    if test_sha and test_file.exists() and sha256_file(test_file) != test_sha:
        state.raise_integrity("tests_modified")

    for shim in SHIM_FILES:
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    stray = []
    for p in AGENT_WS.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(AGENT_WS).as_posix()
        if rel.startswith("brief/"):
            continue
        if rel.startswith(".pytest_cache/") or rel.startswith("__pycache__/") or "/__pycache__/" in rel or rel.endswith(".pyc"):
            continue
        if rel in manifest["files"] or rel in allowed_write_files:
            continue
        stray.append(rel)
    if stray:
        state.raise_integrity("write_outside_whitelist")


def score_visible(state: ScorerState) -> None:
    brief_file = AGENT_WS / "brief" / "round_plan.json"
    if not brief_file.exists():
        state.apply_ceiling(0, "no_round_plan_file")
        return
    state.add("phase2.brief_exists", 4)
    try:
        brief = load_json(brief_file)
    except json.JSONDecodeError:
        state.apply_ceiling(10, "malformed_round_plan")
        state.add_error("invalid json in brief/round_plan.json")
        return

    state.brief = brief
    if brief.get("schema_version") != SCHEMA_VERSION:
        state.apply_ceiling(10, "malformed_round_plan")
        state.add_error("wrong schema_version")
        return
    state.add("phase2.brief_parses", 4)

    if brief.get("variant_id") == VARIANT_ID:
        state.add("phase2.variant_matches", 3)

    focus = brief.get("selected_focus", {})
    if isinstance(focus, dict) and all(isinstance(focus.get(k), str) and focus.get(k) for k in ("focus_id", "title", "why_now")):
        if isinstance(focus.get("citations"), list) and focus["citations"]:
            state.add("phase2.focus_exists", 4)

    dnt = brief.get("do_not_touch")
    if isinstance(dnt, list) and dnt and all(isinstance(row, dict) and row.get("area") and row.get("reason") for row in dnt):
        state.add("phase2.boundaries_exist", 4)

    metric = brief.get("partial_progress_metric", {})
    if isinstance(metric, dict) and all(isinstance(metric.get(k), str) and metric.get(k) for k in ("metric_id", "baseline", "target", "guardrail")):
        if isinstance(metric.get("measurement_plan"), list) and len(metric["measurement_plan"]) >= 2:
            state.add("phase2.metric_exists", 4)

    ledger = brief.get("assumption_ledger")
    if isinstance(ledger, list) and any(isinstance(row, dict) and row.get("status") == "missing" for row in ledger):
        state.add("phase2.assumption_missing", 2)


def score_hidden(state: ScorerState, gold: dict[str, Any]) -> None:
    if not state.brief:
        return
    brief = state.brief

    focus = brief["selected_focus"]
    focus_id = focus.get("focus_id")
    if focus_id == gold["accepted_focus_id"]:
        state.add("behavioral.focus_match", 15)
    if len([c for c in focus.get("citations", []) if existing_file(c)]) >= 2:
        state.add("grounding.focus_citations_valid", 8)
    else:
        state.apply_ceiling(20, "weak_focus_grounding")

    unlock_text = " ".join([focus.get("why_now", "")]).lower()
    if any(keyword.lower() in unlock_text for keyword in gold.get("unlock_keywords", [])):
        state.add("behavioral.unlocks_future", 6)

    boundaries = {row.get("area") for row in brief.get("do_not_touch", []) if isinstance(row, dict)}
    if all(boundary in boundaries for boundary in gold.get("expected_boundary_ids", [])):
        state.add("boundaries.required_present", 10)
    else:
        state.apply_ceiling(35, "boundary_missing")

    boundary_citations_ok = True
    for row in brief.get("do_not_touch", []):
        cites = row.get("citations", [])
        if not cites or not all(existing_file(c) for c in cites):
            boundary_citations_ok = False
            break
    if boundary_citations_ok and brief.get("do_not_touch"):
        state.add("boundaries.citations_valid", 4)

    all_cites = all_citations(brief)
    if all_cites and all(existing_file(c) for c in all_cites):
        state.add("grounding.all_citations_valid", 4)

    metric = brief["partial_progress_metric"]
    baseline = metric.get("baseline", "")
    target = metric.get("target", "")
    guardrail = metric.get("guardrail", "")
    plan = metric.get("measurement_plan", [])
    baseline_numeric = bool(NUMERIC_RE.search(baseline))
    target_numeric = bool(NUMERIC_RE.search(target))
    guardrail_good = len(guardrail.split()) >= 5 and any(word in guardrail.lower() for word in ("do not", "regress", "while", "until"))
    if baseline_numeric:
        state.add("metric.baseline_concrete", 4)
    if target_numeric:
        state.add("metric.target_concrete", 4)
    if guardrail_good:
        state.add("metric.guardrail_concrete", 4)
    if isinstance(plan, list) and len(plan) >= 2:
        state.add("metric.measurement_plan", 2)
    if not (baseline_numeric and target_numeric and guardrail_good):
        state.apply_ceiling(25, "vague_partial_progress_metric")

    ledger_text = " ".join(
        f"{row.get('topic', '')} {row.get('note', '')}".lower()
        for row in brief.get("assumption_ledger", [])
        if isinstance(row, dict)
    )
    if any(keyword.lower() in ledger_text for keyword in gold.get("expected_assumption_keywords", [])):
        state.add("assumptions.required_topic", 6)

    if VARIANT_ID == "v2-noisy-distractor" and focus_id == "contained-pager-cleanup":
        state.apply_ceiling(35, "anchored_on_contained_regression")
    if VARIANT_ID in {"v3-dirty-state", "v4-multi-corpus-objective"} and focus_id == "cold-start-cache-tuning":
        state.apply_ceiling(35, "sunk_cost_finish")
    if VARIANT_ID == "v4-multi-corpus-objective" and focus_id != "streaming-watermark-alignment":
        state.apply_ceiling(40, "objective_drift")
    if VARIANT_ID == "v5-recovery-in-thread" and focus_id == "streaming-watermark-alignment":
        state.apply_ceiling(30, "incident_blind_retry")


def compute_milestones(state: ScorerState) -> dict[str, bool]:
    focus_valid = state.breakdown.get("grounding.focus_citations_valid", 0) > 0
    brief_valid = state.breakdown.get("phase2.brief_exists", 0) > 0 and state.breakdown.get("phase2.brief_parses", 0) > 0
    boundaries = state.breakdown.get("boundaries.required_present", 0) > 0
    metric = state.breakdown.get("metric.baseline_concrete", 0) > 0 and state.breakdown.get("metric.target_concrete", 0) > 0 and state.breakdown.get("metric.guardrail_concrete", 0) > 0
    m = {
        "M1_localization": focus_valid,
        "M2_primary_fix": brief_valid and state.breakdown.get("phase2.variant_matches", 0) > 0,
        "M3_invariants": state.integrity_flag == 0 and not state.shortcut_detected,
        "M4_functional": brief_valid and boundaries and metric and state.integrity_flag == 0,
        "M5_e2e": state.breakdown.get("behavioral.focus_match", 0) > 0 and state.integrity_flag == 0 and state.ceiling_cap > 35,
    }
    if state.integrity_flag == 1:
        m["M3_invariants"] = False
        m["M4_functional"] = False
        m["M5_e2e"] = False
    return m


def build_result(state: ScorerState) -> dict[str, Any]:
    p_benchmark = max(0, min(state.ceiling_cap, state.raw_score))
    milestones = compute_milestones(state)
    weights = {
        "M1_localization": 0.10,
        "M2_primary_fix": 0.20,
        "M3_invariants": 0.20,
        "M4_functional": 0.20,
        "M5_e2e": 0.30,
    }
    m_agg = round(sum(weights[k] for k, passed in milestones.items() if passed), 4)
    m_training = 0.0 if state.integrity_flag else round(min(state.raw_M_score, state.ceiling_cap) / MAX_M_POINTS, 4)
    return {
        "pass": p_benchmark >= 70 and milestones["M5_e2e"],
        "score": p_benchmark,
        "P_benchmark": p_benchmark,
        "M_training": m_training,
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_M_score,
        "milestones": milestones,
        "milestone_vector": {
            "slots": [
                {"id": k, "weight": weights[k], "passed_bool": milestones[k]}
                for k in ("M1_localization", "M2_primary_fix", "M3_invariants", "M4_functional", "M5_e2e")
            ],
            "M_aggregate": m_agg,
        },
        "breakdown": {"__bands": state.breakdown_bands, **state.breakdown},
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "schema_version": "cnb55.verify_result.v3",
    }


def main() -> int:
    start = time.time()
    state = ScorerState()
    gold, manifest = load_contracts()
    check_integrity(state, gold, manifest)
    score_visible(state)
    score_hidden(state, gold)
    result = build_result(state)
    result["wall_clock_seconds"] = int(time.time() - start)
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
