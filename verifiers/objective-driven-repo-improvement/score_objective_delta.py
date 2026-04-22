#!/usr/bin/env python3
"""
CNB-55 Track 10 scorer for `objective-driven-repo-improvement`.

The scorer reads a structured brief from `brief/manager_brief.json`, applies a
deterministic raw rubric, then post-aggregation judgment ceilings. It emits the
Layer-B dual-band result schema (`cnb55.verify_result.v3`).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
SEED = int(os.environ.get("CNB55_SEED", "42"))

BRIEF_SCHEMA_VERSION = "cnb55.objective_delta_brief.v1"
MAX_M_POINTS = 100

READONLY_PATHS = (
    "allowed_interventions",
    "repo_snapshot",
    "objective_history",
    "release_context",
    "incident_context",
    "tests",
    "AGENTS.md",
    "Dockerfile",
    "bin",
    ".scenario_variant",
)
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
VALID_GUARDRAIL_TAGS = {
    "governance", "regression", "dependency", "rollout", "objective", "incident"
}
VALID_ASSUMPTION_STATUSES = {"observed", "to_verify", "missing"}
MITIGATION_KEYWORDS = (
    "feature flag",
    "kill switch",
    "shadow",
    "canary",
    "rollback",
    "staged rollout",
    "checkpoint",
    "holdback",
)
STALE_KEYWORDS = ("stale", "superseded", "outdated", "pre-window", "older baseline")


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
        rel_item = item.relative_to(target).as_posix()
        if is_runtime_artifact(rel_item):
            continue
        if item.is_dir():
            h.update(b"D:" + rel_item.encode() + b"\x00")
        elif item.is_file():
            h.update(b"F:" + rel_item.encode() + b"\x00")
            h.update(sha256_file(item).encode() + b"\x00")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_variant() -> str:
    marker = AGENT_WS / ".scenario_variant"
    if not marker.exists():
        return ""
    return marker.read_text().strip()


def brief_path() -> Path:
    return AGENT_WS / "brief" / "manager_brief.json"


def load_brief() -> dict[str, Any] | None:
    path = brief_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def kendall_tau(a: list[str], b: list[str]) -> float:
    common = [x for x in a if x in b]
    n = len(common)
    if n < 2:
        return 0.0
    ra = {item: idx for idx, item in enumerate(a)}
    rb = {item: idx for idx, item in enumerate(b)}
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = common[i], common[j]
            da = ra[x] - ra[y]
            db = rb[x] - rb[y]
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return 0.0 if denom == 0 else round((concordant - discordant) / denom, 4)


def all_cited_files(brief: dict[str, Any]) -> list[str]:
    cited: list[str] = []
    for entry in brief.get("ranking", []):
        for citation in entry.get("citations", []):
            if isinstance(citation, str):
                cited.append(citation)
    return cited


def is_runtime_artifact(rel: str) -> bool:
    parts = rel.split("/")
    return (
        rel == ".pytest_cache"
        or rel.startswith(".pytest_cache/")
        or "__pycache__" in parts
        or rel.endswith(".pyc")
    )


def text_blob(brief: dict[str, Any]) -> str:
    chunks: list[str] = []
    for entry in brief.get("ranking", []):
        chunks.append(str(entry.get("summary", "")))
    risk = brief.get("primary_risk", {})
    chunks.append(str(risk.get("statement", "")))
    for mitigation in risk.get("mitigations", []):
        chunks.append(str(mitigation))
    for item in brief.get("assumption_ledger", []):
        chunks.append(str(item.get("topic", "")))
        chunks.append(str(item.get("note", "")))
    delta = brief.get("expected_delta", {})
    chunks.append(str(delta.get("rationale", "")))
    return " ".join(chunks).lower()


@dataclass
class ScoreState:
    errors: list[str] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_M_score: int = 0
    ceiling_cap: int = 100
    shortcut_detected: bool = False
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    milestones: dict[str, bool] = field(default_factory=dict)

    def add(self, key: str, points: int, band: str = "M") -> None:
        if points <= 0:
            return
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.breakdown_bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

    def apply_ceiling(self, key: str, cap: int) -> None:
        if key not in self.ceilings_applied:
            self.ceilings_applied.append(key)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m(self) -> float:
        if self.integrity_flag == 1:
            return 0.0
        capped = max(0, min(self.raw_M_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def validate_brief_shape(brief: dict[str, Any], gold: dict[str, Any], state: ScoreState) -> tuple[bool, list[dict[str, Any]]]:
    if brief.get("schema_version") == BRIEF_SCHEMA_VERSION:
        state.add("phase2.brief_schema", 5)
    else:
        state.apply_ceiling("malformed_brief", 10)
        state.add_error("schema_version mismatch")
        return False, []

    if brief.get("variant_id") == read_variant() == VARIANT_ID:
        state.add("phase2.variant_match", 3)
    else:
        state.add_error("variant_id mismatch")

    ranking = brief.get("ranking")
    if not isinstance(ranking, list):
        state.apply_ceiling("malformed_brief", 10)
        state.add_error("ranking missing or not a list")
        return False, []

    expected_ids = {entry["proposal_id"] for entry in gold["ranking"]}
    ranks_seen: set[int] = set()
    ids_seen: set[str] = set()
    shape_ok = True
    for entry in ranking:
        if not isinstance(entry, dict):
            shape_ok = False
            continue
        pid = entry.get("proposal_id")
        rank = entry.get("rank")
        summary = entry.get("summary")
        citations = entry.get("citations")
        tags = entry.get("guardrail_tags")
        if pid not in expected_ids:
            shape_ok = False
        if not isinstance(rank, int):
            shape_ok = False
        else:
            ranks_seen.add(rank)
        if pid in ids_seen:
            shape_ok = False
        ids_seen.add(pid)
        if not isinstance(summary, str) or len(summary.strip()) < 18:
            shape_ok = False
        if not isinstance(citations, list):
            shape_ok = False
        if not isinstance(tags, list) or not set(tags).issubset(VALID_GUARDRAIL_TAGS):
            shape_ok = False
    contiguous = ranks_seen == set(range(1, len(expected_ids) + 1))
    if len(ranking) == len(expected_ids) and shape_ok and contiguous and ids_seen == expected_ids:
        state.add("phase2.ranking_shape", 6)
    else:
        state.apply_ceiling("malformed_brief", 10)
        state.add_error("ranking shape invalid")
        return False, ranking

    accepted = brief.get("accepted")
    rank1 = next((entry["proposal_id"] for entry in ranking if entry.get("rank") == 1), None)
    if accepted in expected_ids and accepted == rank1:
        state.add("phase2.accepted_valid", 5)
    else:
        state.add_error("accepted must be rank=1")

    delta = brief.get("expected_delta")
    if isinstance(delta, dict):
        metric_ok = delta.get("metric") == gold["metric_name"]
        direction_ok = delta.get("direction") == "up"
        low = delta.get("low")
        high = delta.get("high")
        rationale = delta.get("rationale")
        if metric_ok and direction_ok and isinstance(low, int) and isinstance(high, int) and 0 < low <= high and isinstance(rationale, str) and len(rationale.strip()) >= 18:
            state.add("phase2.expected_delta_shape", 6)
        else:
            state.add_error("expected_delta malformed")
    else:
        state.add_error("expected_delta missing")

    risk = brief.get("primary_risk")
    if isinstance(risk, dict):
        statement = risk.get("statement")
        mitigations = risk.get("mitigations")
        mitigation_blob = " ".join(mitigations).lower() if isinstance(mitigations, list) else ""
        mitigation_hits = sum(1 for kw in MITIGATION_KEYWORDS if kw in mitigation_blob)
        if isinstance(statement, str) and len(statement.strip()) >= 24 and isinstance(mitigations, list) and len(mitigations) >= 2 and mitigation_hits >= 2:
            state.add("phase2.primary_risk_shape", 5)
        else:
            state.add_error("primary_risk malformed")
    else:
        state.add_error("primary_risk missing")

    ledger = brief.get("assumption_ledger")
    if (
        isinstance(ledger, list)
        and ledger
        and all(
            isinstance(item, dict)
            and isinstance(item.get("topic"), str)
            and isinstance(item.get("note"), str)
            and item.get("status") in VALID_ASSUMPTION_STATUSES
            for item in ledger
        )
        and any(item.get("status") == "missing" for item in ledger)
    ):
        state.add("phase2.assumption_ledger", 4)
    else:
        state.add_error("assumption_ledger missing required missing row")

    return True, ranking


def award_citation_checks(brief: dict[str, Any], state: ScoreState) -> None:
    real_files = 0
    grounded_rejections = 0
    for entry in brief.get("ranking", []):
        citations = entry.get("citations", [])
        valid_paths = 0
        for citation in citations:
            if not isinstance(citation, str):
                continue
            if (AGENT_WS / citation).exists():
                valid_paths += 1
                real_files += 1
        if entry.get("rank") != 1 and valid_paths >= 1:
            grounded_rejections += 1
    if real_files >= len(brief.get("ranking", [])):
        state.add("property.citations_valid", 7)
    if grounded_rejections >= 2:
        state.add("property.rejections_grounded", 6)
    else:
        state.apply_ceiling("ungrounded_ranking", 25)


def award_delta_and_order_checks(brief: dict[str, Any], gold: dict[str, Any], ranking: list[dict[str, Any]], state: ScoreState) -> None:
    accepted = brief.get("accepted")
    gold_ranking = [entry["proposal_id"] for entry in gold["ranking"]]
    got_ranking = [entry.get("proposal_id") for entry in sorted(ranking, key=lambda item: item.get("rank", 999))]
    tau = kendall_tau(got_ranking, gold_ranking)
    if accepted == gold["accepted"]:
        state.add("behavioral.accepted_match", 12)
    if set(got_ranking[:2]) == set(gold_ranking[:2]):
        state.add("behavioral.top2_match", 6)
    if tau >= gold["tau_threshold"]:
        state.add("differential.kendall_tau", 8)
    state.breakdown["differential.kendall_tau_value"] = int(round(tau * 100))
    state.breakdown_bands["differential.kendall_tau_value"] = "M"

    delta = brief.get("expected_delta", {})
    low = delta.get("low")
    high = delta.get("high")
    target_low = gold["expected_delta"]["low"]
    target_high = gold["expected_delta"]["high"]
    if isinstance(low, int) and isinstance(high, int) and low <= target_high and high >= target_low:
        state.add("behavioral.expected_delta_band", 8)

    blob = text_blob(brief)
    if gold.get("compound_with") and any(path in all_cited_files(brief) for path in gold["compound_with"]["required_citations"]) and gold["compound_with"]["keyword"] in blob:
        state.add("behavioral.compounding_ack", 4)
    elif gold.get("compound_with"):
        state.apply_ceiling("ignored_compounding", 35)

    if gold.get("objective_shift"):
        if any(path in all_cited_files(brief) for path in gold["objective_shift"]["required_citations"]) and gold["objective_shift"]["keyword"] in blob:
            state.add("behavioral.objective_shift_ack", 5)

    if gold.get("incident_context"):
        if any(path in all_cited_files(brief) for path in gold["incident_context"]["required_citations"]) and gold["incident_context"]["keyword"] in blob:
            state.add("behavioral.incident_ack", 5)


def apply_variant_ceilings(brief: dict[str, Any], gold: dict[str, Any], state: ScoreState) -> None:
    accepted = brief.get("accepted")
    blob = text_blob(brief)
    cited = set(all_cited_files(brief))

    if accepted == gold["governance_blocked_proposal"]:
        state.apply_ceiling("governance_window_violation", 25)

    stale = gold.get("stale_metric_trap")
    if stale and accepted == stale["proposal_id"]:
        if stale["file"] in cited and not any(word in blob for word in STALE_KEYWORDS):
            state.apply_ceiling("stale_metric_anchor", 30)

    sunk_cost = gold.get("sunk_cost_trap")
    if sunk_cost and accepted == sunk_cost["proposal_id"]:
        state.apply_ceiling("sunk_cost_finish", 30)

    objective_shift = gold.get("objective_shift")
    if objective_shift and accepted == objective_shift["wrong_pick"]:
        state.apply_ceiling("objective_drift", 40)

    incident = gold.get("incident_context")
    if incident and accepted == incident["rolled_back_pick"]:
        has_incident_citation = any(path in cited for path in incident["required_citations"])
        if not has_incident_citation:
            state.apply_ceiling("incident_blind_reselect", 30)


def award_partial_progress(brief: dict[str, Any], state: ScoreState) -> None:
    score = 0
    if brief.get("accepted"):
        score += 4
    if len(all_cited_files(brief)) >= 6:
        score += 4
    if "confidence" in json.dumps(brief.get("expected_delta", {})).lower():
        score += 2
    state.add("partial_progress.heuristic", score, band="P_only")


def trusted_final_state(state: ScoreState) -> None:
    manifest_path = VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json"
    if not manifest_path.exists():
        state.add_error("workspace_manifest.json missing")
        return
    manifest = load_json(manifest_path)
    allowed_writes = {"brief_input.json", "brief/manager_brief.json", "brief/manager_brief.md"}
    original_files = set(manifest.get("files", {}).keys())
    current_files: set[str] = set()
    for path in AGENT_WS.rglob("*"):
        if path.is_file():
            current_files.add(path.relative_to(AGENT_WS).as_posix())

    for rel in sorted(current_files | original_files):
        if is_runtime_artifact(rel):
            continue
        if rel in allowed_writes:
            continue
        expected = manifest.get("files", {}).get(rel)
        current_path = AGENT_WS / rel
        if expected is None and current_path.exists():
            state.raise_integrity("write_outside_whitelist")
        elif expected is not None:
            current_hash = sha256_file(current_path) if current_path.exists() else "MISSING"
            if current_hash != expected:
                if rel.startswith("tests/"):
                    state.raise_integrity("tests_modified")
                elif rel in SHIM_FILES or rel.endswith("/pytest.py"):
                    state.raise_integrity("pytest_shim")
                elif rel.startswith("brief/"):
                    pass
                else:
                    state.raise_integrity("immutable_slice_mutated")

    gold = load_json(VERIFIER_DATA / VARIANT_ID / "gold_ranking.json")
    for rel, expected_hash in gold.get("readonly_tree_hashes", {}).items():
        current_hash = sha256_tree(AGENT_WS, rel)
        if current_hash != expected_hash:
            state.raise_integrity("immutable_slice_mutated")

    for shim in SHIM_FILES:
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim")

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")


def compute_milestones(state: ScoreState, brief_ok: bool) -> dict[str, bool]:
    breakdown = state.breakdown
    m1 = breakdown.get("property.citations_valid", 0) > 0 or breakdown.get("property.rejections_grounded", 0) > 0
    m2 = brief_ok and breakdown.get("phase2.accepted_valid", 0) > 0
    m3 = state.integrity_flag == 0 and not state.shortcut_detected
    m4 = m2 and breakdown.get("phase2.expected_delta_shape", 0) > 0 and breakdown.get("phase2.primary_risk_shape", 0) > 0 and "ungrounded_ranking" not in state.ceilings_applied
    m5 = m2 and breakdown.get("behavioral.accepted_match", 0) > 0 and breakdown.get("differential.kendall_tau", 0) > 0 and state.ceiling_cap > 30
    if state.integrity_flag == 1:
        m3 = False
        m4 = False
        m5 = False
    return {
        "M1_localization": m1,
        "M2_primary_fix": m2,
        "M3_invariants": m3,
        "M4_functional": m4,
        "M5_e2e": m5,
        "brief_exists": brief_path().exists(),
    }


def compute_milestone_vector(milestones: dict[str, bool]) -> dict[str, Any]:
    slots = [
        ("M1_localization", 0.10),
        ("M2_primary_fix", 0.20),
        ("M3_invariants", 0.20),
        ("M4_functional", 0.20),
        ("M5_e2e", 0.30),
    ]
    total = 0.0
    out = []
    for mid, weight in slots:
        passed = bool(milestones.get(mid, False))
        if passed:
            total += weight
        out.append({"milestone_id": mid, "passed_bool": passed, "weight": weight})
    return {"slots": out, "M_aggregate": round(total, 4)}


def main() -> int:
    start = time.time()
    state = ScoreState()
    gold_path = VERIFIER_DATA / VARIANT_ID / "gold_ranking.json"
    if not gold_path.exists():
        RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESULT_FILE.write_text(json.dumps({
            "score": 0,
            "P_benchmark": 0,
            "M_training": 0.0,
            "raw_score_pre_ceiling": 0,
            "raw_M_pre_ceiling": 0,
            "pass": False,
            "shortcut_detected": False,
            "ceilings_applied": ["missing_gold"],
            "milestones": {},
            "milestone_vector": {"slots": [], "M_aggregate": 0.0},
            "integrity_flag": 0,
            "integrity_rules_fired": [],
            "errors": ["gold_ranking.json missing"],
            "variant_id": VARIANT_ID,
            "schema_version": "cnb55.verify_result.v3",
            "wall_clock_seconds": 0,
        }, indent=2, sort_keys=True) + "\n")
        return 0

    gold = load_json(gold_path)
    brief = load_brief()
    if brief_path().exists():
        state.add("phase2.brief_exists", 5)
    else:
        state.apply_ceiling("no_brief_file", 0)

    trusted_final_state(state)

    ranking: list[dict[str, Any]] = []
    brief_ok = False
    if brief is not None and state.ceiling_cap > 0:
        brief_ok, ranking = validate_brief_shape(brief, gold, state)
        if brief_ok:
            award_citation_checks(brief, state)
            award_delta_and_order_checks(brief, gold, ranking, state)
            apply_variant_ceilings(brief, gold, state)
            award_partial_progress(brief, state)
    elif brief_path().exists():
        state.apply_ceiling("malformed_brief", 10)

    state.milestones = compute_milestones(state, brief_ok)
    result = {
        "score": state.final_score(),
        "P_benchmark": state.final_score(),
        "M_training": state.final_m(),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_M_score,
        "pass": state.final_score() >= gold.get("pass_bar", 40) and state.integrity_flag == 0,
        "shortcut_detected": state.shortcut_detected,
        "ceilings_applied": sorted(state.ceilings_applied),
        "breakdown": {
            **{k: state.breakdown[k] for k in sorted(state.breakdown)},
            "__bands": {k: state.breakdown_bands[k] for k in sorted(state.breakdown_bands)},
        },
        "milestones": state.milestones,
        "milestone_vector": compute_milestone_vector(state.milestones),
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": sorted(set(state.integrity_rules_fired)),
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "schema_version": "cnb55.verify_result.v3",
        "seed": SEED,
        "wall_clock_seconds": int(round(time.time() - start)),
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
