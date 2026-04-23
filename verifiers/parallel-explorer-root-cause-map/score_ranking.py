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

VERIFY_RESULT_SCHEMA = "cnb55.verify_result.v3"
MAX_M_POINTS = 90
MILESTONE_SLOTS = [
    ("M1_localization", 0.10),
    ("M2_primary_fix", 0.20),
    ("M3_invariants", 0.20),
    ("M4_functional", 0.20),
    ("M5_e2e", 0.30),
]
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def should_ignore_relpath(relpath: str) -> bool:
    parts = relpath.split("/")
    return (
        "__pycache__" in parts
        or ".pytest_cache" in parts
        or relpath.endswith(".pyc")
    )


def tree_hash(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    if target.is_file():
        if should_ignore_relpath(rel):
            return "IGNORED"
        return sha256_file(target)
    h = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        relpath = path.relative_to(target).as_posix()
        if should_ignore_relpath(relpath):
            continue
        if path.is_dir():
            h.update(f"D:{relpath}\n".encode())
        else:
            h.update(f"F:{relpath}\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\n")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def rel_is_allowed(relpath: str, allowed_paths: list[str]) -> bool:
    for allowed in allowed_paths:
        allowed = allowed.rstrip("/")
        if relpath == allowed or relpath.startswith(f"{allowed}/"):
            return True
    return False


def collect_workspace_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relpath = path.relative_to(root).as_posix()
            if should_ignore_relpath(relpath):
                continue
            files[relpath] = sha256_file(path)
    return files


@dataclass
class ScoreState:
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    milestones: dict[str, bool] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_m_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    shortcut_detected: bool = False
    submission_present: bool = False

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_m_score += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)

    def final_score(self) -> int:
        if self.integrity_flag == 1:
            return 0
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        if self.integrity_flag == 1:
            return 0.0
        capped = max(0, min(self.raw_m_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def baseline_and_integrity(state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    current_files = collect_workspace_files(AGENT_WS)
    baseline_files = manifest.get("file_hashes", {})
    allowed_write_paths = gold.get("allowed_write_paths", [])
    changed_paths: list[str] = []

    for relpath, sha in baseline_files.items():
        current_sha = current_files.get(relpath)
        if current_sha != sha:
            changed_paths.append(relpath)
    for relpath in current_files:
        if relpath not in baseline_files:
            changed_paths.append(relpath)

    state.submission_present = (
        (AGENT_WS / "brief/manager_brief.json").exists()
        or (AGENT_WS / "brief/manager_brief.md").exists()
        or (AGENT_WS / "brief_input.json").exists()
    )

    for relpath, expected_hash in gold.get("readonly_tree_hashes", {}).items():
        actual = tree_hash(AGENT_WS, relpath)
        if actual != expected_hash:
            state.raise_integrity("immutable_slice_mutated")
            if relpath == "tests":
                state.raise_integrity("tests_modified")

    for relpath in changed_paths:
        if not rel_is_allowed(relpath, allowed_write_paths):
            state.raise_integrity("write_outside_whitelist")
            break

    for shim in SHIM_FILES:
        if list(AGENT_WS.rglob(shim)):
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")


def read_brief(state: ScoreState) -> dict[str, Any] | None:
    path = AGENT_WS / "brief" / "manager_brief.json"
    if not path.exists():
        state.errors.append("missing brief/manager_brief.json")
        state.apply_ceiling("no_brief_file", 0)
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        state.errors.append(f"invalid JSON: {exc}")
        state.apply_ceiling("malformed_brief", 10)
        return None
    if data.get("schema_version") != "cnb55.root_cause_brief.v1":
        state.errors.append("unexpected schema_version")
        state.apply_ceiling("malformed_brief", 10)
    if data.get("variant_id") != VARIANT_ID:
        state.errors.append("variant_id mismatch")
        state.apply_ceiling("malformed_brief", 10)
    return data


def contiguous_ranks(ranking: list[dict[str, Any]]) -> bool:
    try:
        ranks = sorted(int(item.get("rank")) for item in ranking)
    except Exception:
        return False
    return ranks == list(range(1, len(ranking) + 1))


def unique_file_set(thread: dict[str, Any]) -> set[str]:
    return {str(path) for path in thread.get("files", [])}


def investigation_threads_non_overlapping(threads: list[dict[str, Any]]) -> bool:
    if len(threads) < 2:
        return False
    first = unique_file_set(threads[0])
    second = unique_file_set(threads[1])
    if not first or not second:
        return False
    return first != second and len(first & second) < min(len(first), len(second))


def find_ranking_entry(brief: dict[str, Any], suspect_id: str) -> dict[str, Any] | None:
    for entry in brief.get("ranking", []):
        if entry.get("suspect_id") == suspect_id:
            return entry
    return None


def ranking_text(entry: dict[str, Any] | None) -> str:
    if not entry:
        return ""
    parts = [
        str(entry.get("summary", "")),
        str(entry.get("file", "")),
        str(entry.get("symbol", "")),
        str(entry.get("role", "")),
        " ".join(str(p) for p in entry.get("evidence_paths", [])),
    ]
    return normalize_text(" ".join(parts))


def score_structure(state: ScoreState, brief: dict[str, Any], gold: dict[str, Any]) -> None:
    ranking = brief.get("ranking", [])
    if len(ranking) == 4 and contiguous_ranks(ranking) and all(entry.get("evidence_paths") for entry in ranking):
        state.add("structure.ranking_complete", 6)
    else:
        state.errors.append("ranking must contain 4 contiguous suspects")
        if len(ranking) == 4 and contiguous_ranks(ranking):
            state.apply_ceiling("grounding_stripped", 30)

    threads = brief.get("investigation_threads", [])
    if len(threads) >= 2 and investigation_threads_non_overlapping(threads):
        state.add("structure.investigation_threads", 6)
        state.milestones["M1_localization"] = True
    else:
        state.errors.append("need >=2 non-overlapping investigation threads")
        state.apply_ceiling("no_multi_thread_synthesis", 30)

    evidence_rows = brief.get("evidence_table", [])
    if len(evidence_rows) >= 4 and all(row.get("file") and row.get("symbol") and row.get("test_or_artifact") for row in evidence_rows[:4]):
        state.add("structure.evidence_table", 8)
    else:
        state.errors.append("evidence table missing rows or symbol/file refs")
        if len(evidence_rows) >= 4:
            state.apply_ceiling("grounding_stripped", 30)

    note = brief.get("verification_note", {})
    non_goals = brief.get("remediation_plan", {}).get("non_goals", [])
    if note.get("failing_assertion") and note.get("contradictory_artifact") and len(non_goals) >= 2:
        state.add("structure.verification_non_goals", 5)
    else:
        state.errors.append("verification note or remediation non-goals incomplete")


def score_primary_cause(state: ScoreState, brief: dict[str, Any], gold: dict[str, Any]) -> None:
    ranking = brief.get("ranking", [])
    accepted = brief.get("accepted")
    if not ranking:
        return

    top = ranking[0]
    if accepted == gold["primary_suspect_id"] and top.get("suspect_id") == accepted:
        state.add("primary.accepted_suspect", 10)
        state.milestones["M2_primary_fix"] = True
    else:
        state.errors.append("wrong primary suspect")
        state.apply_ceiling("wrong_primary_suspect", 20)
        if accepted == gold["renderer_suspect_id"]:
            state.apply_ceiling("renderer_primary_shortcut", 25)
            state.shortcut_detected = True

    if (
        top.get("file") == gold["primary_file"]
        and top.get("symbol") == gold["primary_symbol"]
        and top.get("role") == "primary"
    ):
        state.add("primary.file_symbol", 10)
    else:
        state.errors.append("top-ranked file/symbol mismatch")

    aggregation = find_ranking_entry(brief, gold["aggregation_suspect_id"])
    if aggregation and aggregation.get("file") == gold["aggregation_file"] and aggregation.get("symbol") == gold["aggregation_symbol"]:
        state.add("causal.aggregation_amplifier", 8)
    else:
        state.errors.append("aggregation suspect missing or wrong symbol")
        state.apply_ceiling("no_symbol_level_evidence", 30)

    assertion = normalize_text(brief.get("verification_note", {}).get("failing_assertion"))
    if normalize_text(gold["required_test_quote"]) in assertion:
        state.add("causal.exact_assertion", 8)
    else:
        state.errors.append("exact failing assertion not quoted")


def score_disproof(state: ScoreState, brief: dict[str, Any], gold: dict[str, Any]) -> None:
    renderer = find_ranking_entry(brief, gold["renderer_suspect_id"])
    if (
        renderer
        and renderer.get("file") == gold["renderer_file"]
        and renderer.get("symbol") == gold["renderer_symbol"]
        and renderer.get("role") == "downstream-only"
    ):
        state.add("disproof.renderer_downstream_only", 8)
    else:
        state.errors.append("renderer not ruled out correctly")
        if renderer and renderer.get("role") == "primary":
            state.apply_ceiling("renderer_primary_shortcut", 25)

    note = brief.get("verification_note", {})
    contradictory_path = note.get("contradictory_artifact", "")
    resolution = normalize_text(note.get("resolution", ""))
    expected_resolution_markers = ("stale", "downstream", "misleading", "symptom")
    if contradictory_path.endswith("renderer_hotfix.patch"):
        expected_resolution_markers = ("abandoned", "sunk-cost", "sunk cost", "hide", "masked")
    elif contradictory_path.endswith("hotfix_window.md"):
        expected_resolution_markers = ("constraint", "context", "hotfix", "scope")
    elif contradictory_path.endswith("renderer-rollback.md"):
        expected_resolution_markers = ("rollback", "failed", "upstream", "renderer-only")

    if contradictory_path == gold["contradictory_artifact_path"] and any(word in resolution for word in expected_resolution_markers):
        state.add("disproof.contradictory_artifact", 6)
        state.milestones["M4_functional"] = True
    else:
        state.errors.append("contradictory artifact not correctly disambiguated")
        state.apply_ceiling("missing_contradictory_disproof", 35)


def score_remediation(state: ScoreState, brief: dict[str, Any], gold: dict[str, Any]) -> None:
    plan = brief.get("remediation_plan", {})
    target_file = plan.get("patch_target_file")
    target_symbol = plan.get("patch_target_symbol")
    non_goals_text = normalize_text(" ".join(plan.get("non_goals", [])))

    if target_file == gold["patch_target_file"] and target_symbol == gold["patch_target_symbol"]:
        state.add("remediation.target", 10)
    else:
        state.errors.append("wrong remediation target")
        if target_file == gold["renderer_file"]:
            state.apply_ceiling("renderer_first_remediation", 35)
        if VARIANT_ID == "v4-multi-corpus-objective" and target_file == gold["aggregation_file"]:
            state.apply_ceiling("aggregation_rewrite_hotfix", 40)

    if "renderer" in non_goals_text and "aggregation" in non_goals_text:
        state.add("remediation.non_goals", 5)
        state.milestones["M5_e2e"] = True
    else:
        state.errors.append("non-goals do not rule out renderer and aggregation churn")

    if VARIANT_ID == "v3-dirty-state":
        blob = normalize_text(json.dumps(brief))
        if target_file == gold["renderer_file"] or "finish the renderer patch" in blob:
            state.apply_ceiling("sunk_cost_renderer_patch", 30)
    if VARIANT_ID == "v5-recovery-in-thread":
        blob = normalize_text(json.dumps(brief))
        if gold["required_context_path"] not in blob:
            state.apply_ceiling("incident_blind_rerender", 15)


def score_probe_only(state: ScoreState, brief: dict[str, Any]) -> None:
    ranking = brief.get("ranking", [])
    if sum(1 for entry in ranking if len(str(entry.get("summary", ""))) >= 60) >= 3:
        state.add("p_only.suspect_richness", 4, band="P_only")
    threads = brief.get("investigation_threads", [])
    if len(threads) >= 2 and all(len(str(thread.get("finding", ""))) >= 40 for thread in threads[:2]):
        state.add("p_only.cross_thread_synthesis", 3, band="P_only")
    plan = brief.get("remediation_plan", {})
    if len(plan.get("validation_steps", [])) >= 2 and len(str(plan.get("why_smallest_safe_patch", ""))) >= 30:
        state.add("p_only.plan_clarity", 3, band="P_only")


def build_milestone_vector(state: ScoreState) -> dict[str, Any]:
    if state.integrity_flag == 1:
        slots = [
            {"slot": name, "weight": weight, "passed_bool": False}
            for name, weight in MILESTONE_SLOTS
        ]
        return {"slots": slots, "M_aggregate": 0.0}

    milestone_status = {
        "M1_localization": state.milestones.get("M1_localization", False),
        "M2_primary_fix": state.milestones.get("M2_primary_fix", False),
        "M3_invariants": state.integrity_flag == 0,
        "M4_functional": state.milestones.get("M4_functional", False),
        "M5_e2e": state.milestones.get("M5_e2e", False),
    }
    slots = []
    aggregate = 0.0
    for name, weight in MILESTONE_SLOTS:
        passed = milestone_status[name]
        slots.append({"slot": name, "weight": weight, "passed_bool": passed})
        if passed:
            aggregate += weight
    return {"slots": slots, "M_aggregate": round(aggregate, 4)}


def main() -> int:
    state = ScoreState()
    gold = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "gold_ranking.json")
    manifest = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "workspace_manifest.json")

    baseline_and_integrity(state, gold, manifest)
    brief = read_brief(state)
    if brief is not None:
        score_structure(state, brief, gold)
        score_primary_cause(state, brief, gold)
        score_disproof(state, brief, gold)
        score_remediation(state, brief, gold)
        score_probe_only(state, brief)

    milestone_vector = build_milestone_vector(state)
    score = state.final_score()
    m_training = state.final_m_training()
    result = {
        "schema_version": VERIFY_RESULT_SCHEMA,
        "variant_id": VARIANT_ID,
        "score": score,
        "P_benchmark": score,
        "M_training": m_training,
        "raw_score_pre_ceiling": state.raw_score,
        "raw_m_points": state.raw_m_score,
        "pass": score >= gold.get("pass_bar", 75),
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "ceilings_applied": state.ceilings_applied,
        "breakdown": state.breakdown,
        "breakdown_bands": state.bands,
        "errors": state.errors,
        "milestones": {
            "M1_localization": milestone_vector["slots"][0]["passed_bool"],
            "M2_primary_fix": milestone_vector["slots"][1]["passed_bool"],
            "M3_invariants": milestone_vector["slots"][2]["passed_bool"],
            "M4_functional": milestone_vector["slots"][3]["passed_bool"],
            "M5_e2e": milestone_vector["slots"][4]["passed_bool"],
        },
        "milestone_vector": milestone_vector,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
