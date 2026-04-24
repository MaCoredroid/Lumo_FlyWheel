#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA_ROOT = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
VERIFY_SCHEMA = "cnb55.verify_result.v3"
MAX_M_POINTS = 100
PASS_BAR = 80
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
MILESTONE_SLOTS = [
    ("M1_localization", 0.10),
    ("M2_primary_fix", 0.20),
    ("M3_invariants", 0.20),
    ("M4_functional", 0.20),
    ("M5_e2e", 0.30),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    if target.is_file():
        return sha256_file(target)
    h = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        rel_path = path.relative_to(target).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_dir():
            h.update(f"D:{rel_path}\n".encode())
        else:
            h.update(f"F:{rel_path}\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\n")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def is_transient_runtime_file(relpath: str) -> bool:
    parts = relpath.split("/")
    return (
        "__pycache__" in parts
        or relpath.endswith(".pyc")
        or relpath.startswith(".pytest_cache/")
        or relpath == ".DS_Store"
        or relpath.endswith("/.DS_Store")
    )


def contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def has_release_contract_note(text: str) -> bool:
    omit_owner = (
        contains_any(text, ["omit", "omits", "omitted", "omitting"])
        and "owner" in text
        and contains_any(text, ["null", "owner: null"])
    )
    request_semantics = (
        contains_any(text, ["request-side", "filter", "owner=null", "owner = null"])
        and contains_any(text, ["unchanged", "still", "remain"])
        and contains_any(text, ["no filter", "unowned rows", "unowned"])
    )
    return omit_owner and request_semantics


def has_variant_objective_note(text: str) -> bool:
    if VARIANT_ID not in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        return True
    return contains_any(text, ["mobile", "ios", "client contract", "compatibility"])


def has_variant_incident_note(text: str) -> bool:
    if VARIANT_ID != "v5-recovery-in-thread":
        return True
    return contains_any(text, ["inc-742", "rollback"])


@dataclass
class ScoreState:
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    milestones: dict[str, bool] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_m_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    shortcut_detected: bool = False

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

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)

    def final_score(self) -> int:
        if self.integrity_flag:
            return 0
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        if self.integrity_flag:
            return 0.0
        capped = max(0, min(self.raw_m_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def trusted_final_state(state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    current_hashes: dict[str, str] = {}
    changed_paths: list[str] = []
    for path in sorted(AGENT_WS.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(AGENT_WS).as_posix()
        if is_transient_runtime_file(rel):
            continue
        current_hashes[rel] = sha256_file(path)
    baseline_hashes = manifest.get("file_hashes", {})
    for rel, expected_hash in baseline_hashes.items():
        actual = current_hashes.get(rel)
        if actual != expected_hash:
            changed_paths.append(rel)
    for rel in current_hashes:
        if rel not in baseline_hashes:
            changed_paths.append(rel)

    allowed = gold.get("allowed_write_paths", [])
    for rel in changed_paths:
        if not any(rel == path or rel.startswith(path.rstrip("/") + "/") for path in allowed):
            state.raise_integrity("write_outside_whitelist")
            if rel.startswith("tests/locked/"):
                state.raise_integrity("tests_modified")

    for rel, expected_hash in manifest.get("readonly_tree_hashes", {}).items():
        actual_hash = sha256_tree(AGENT_WS, rel)
        if actual_hash != expected_hash:
            rule = "tests_modified" if rel.startswith("tests/locked") else "immutable_slice_mutated"
            state.raise_integrity(rule)

    for shim in SHIM_FILES:
        if list(AGENT_WS.rglob(shim)):
            state.raise_integrity("pytest_shim")
            break
    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    return {"changed_paths": sorted(set(changed_paths))}


def run_unittest_targets(targets: list[str]) -> tuple[bool, str]:
    if str(AGENT_WS) not in sys.path:
        sys.path.insert(0, str(AGENT_WS))
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for target in targets:
        suite.addTests(loader.loadTestsFromName(target))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    return result.wasSuccessful(), stream.getvalue()


def run_visible_tests(state: ScoreState) -> bool:
    ok, output = run_unittest_targets(["tests.test_handlers", "tests.test_service"])
    if ok:
        state.add("visible.tests_pass", 20)
        return True
    state.add_error(output.strip()[-1200:])
    return False


def run_hidden_checks(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    module = import_module(
        "pr_thread_hidden_tests",
        VERIFIER_DATA_ROOT / "_shared" / "hidden_tests" / "test_contract_hidden.py",
    )
    diagnostics = module.evaluate(AGENT_WS)

    omit_ok = diagnostics["default_omits_owner"] and diagnostics["export_omits_owner"]
    if omit_ok:
        state.add("hidden.owner_omission", 15)
    else:
        state.apply_ceiling("null_owner_contract_unfixed", 30)

    order_ok = diagnostics["owner_order_preserved"] and diagnostics["equal_count_order_preserved"]
    if order_ok:
        state.add("hidden.stable_ordering", 15)
    else:
        state.apply_ceiling("unstable_unowned_order", 35)

    if diagnostics["request_semantics_preserved"]:
        state.add("hidden.request_semantics", 10)
    else:
        state.apply_ceiling("request_semantics_regression", 25)
        if VARIANT_ID in {"v3-dirty-state", "v4-multi-corpus-objective", "v5-recovery-in-thread"}:
            state.apply_ceiling("sunk_cost_finish", 25)

    if VARIANT_ID != "v1-clean-baseline" and diagnostics["global_owner_sort_detected"]:
        state.apply_ceiling("outdated_sort_resurrection", 20)

    return diagnostics


def validate_reply_schema(path: Path) -> tuple[bool, list[dict[str, Any]], str]:
    if not path.exists():
        return False, [], "review/reviewer_replies.json missing"
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return False, [], f"invalid JSON: {exc}"
    if not isinstance(payload, list):
        return False, [], "reply artifact must be a JSON array"
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            return False, [], f"reply #{index} is not an object"
        for key in ("thread_id", "status", "body", "evidence"):
            if key not in item:
                return False, [], f"reply #{index} missing key {key}"
        if item["status"] != "resolved":
            return False, [], f"reply #{index} has unsupported status {item['status']!r}"
        if not isinstance(item["evidence"], list) or not all(isinstance(v, str) for v in item["evidence"]):
            return False, [], f"reply #{index} evidence must be a list of strings"
    return True, payload, ""


def score_replies(state: ScoreState, gold: dict[str, Any]) -> bool:
    ok, replies, error = validate_reply_schema(AGENT_WS / "review" / "reviewer_replies.json")
    if ok:
        state.add("replies.schema_valid", 5)
    else:
        state.add_error(error)
        return False

    actionable = set(gold["actionable_thread_ids"])
    ignored = set(gold["ignored_thread_ids"])
    reply_ids = {item["thread_id"] for item in replies}
    if reply_ids == actionable and not (reply_ids & ignored):
        state.add("replies.actionable_only", 5)
        state.milestones["reply_targeting"] = True
    else:
        state.apply_ceiling("non_actionable_replied", 15)

    evidence_ok = True
    for item in replies:
        expected_evidence = gold["required_reply_evidence"].get(item["thread_id"], [])
        expected_phrases = gold.get("required_reply_phrases", {}).get(item["thread_id"], [])
        haystack = f"{item['body']} {' '.join(item['evidence'])}".lower()
        if len(item["body"].strip()) < 40:
            evidence_ok = False
        for evidence_path in expected_evidence:
            if evidence_path.lower() not in haystack:
                evidence_ok = False
        for phrase in expected_phrases:
            if phrase.lower() not in haystack:
                evidence_ok = False
    if evidence_ok:
        state.add("replies.evidence_grounded", 10)
        return True
    state.apply_ceiling("generic_replies", 20)
    return False


def score_release_note(state: ScoreState, gold: dict[str, Any]) -> bool:
    path = AGENT_WS / "docs" / "release_notes.md"
    if not path.exists():
        state.apply_ceiling("missing_release_note_contract", 25)
        return False
    text = path.read_text().lower()
    if not has_release_contract_note(text):
        state.apply_ceiling("missing_release_note_contract", 25)
        return False
    if not has_variant_objective_note(text):
        state.apply_ceiling("objective_drift", 25)
        return False
    if not has_variant_incident_note(text):
        state.apply_ceiling("incident_blind_reselect", 30)
        return False
    state.add("docs.release_note_contract", 10)
    return True


def score_test_updates(state: ScoreState, gold: dict[str, Any], changed_paths: list[str]) -> bool:
    required_names = gold["required_regression_test_names"]
    tests_changed = any(path in changed_paths for path in ("tests/test_handlers.py", "tests/test_service.py"))
    text = ""
    for relpath in ("tests/test_handlers.py", "tests/test_service.py"):
        path = AGENT_WS / relpath
        if path.exists():
            text += path.read_text()
    if tests_changed and all(name in text for name in required_names):
        state.add("tests.regression_expanded", 3)
        return True
    return False


def score_scope(state: ScoreState, changed_paths: list[str]) -> None:
    relevant = [path for path in changed_paths if not path.startswith("review/reviewer_replies.json")]
    if relevant and len(relevant) <= 8:
        state.add("scope.narrow_patch", 2)


def build_milestones(
    state: ScoreState,
    visible_ok: bool,
    hidden_diag: dict[str, bool],
    replies_ok: bool,
    release_ok: bool,
    tests_ok: bool,
) -> None:
    state.milestones["M1_localization"] = replies_ok or (AGENT_WS / "review" / "reviewer_replies.json").exists()
    state.milestones["M2_primary_fix"] = visible_ok and hidden_diag["default_omits_owner"]
    state.milestones["M3_invariants"] = state.integrity_flag == 0 and "non_actionable_replied" not in state.ceilings_applied
    state.milestones["M4_functional"] = (
        hidden_diag["default_omits_owner"]
        and hidden_diag["export_omits_owner"]
        and hidden_diag["owner_order_preserved"]
        and hidden_diag["equal_count_order_preserved"]
        and hidden_diag["request_semantics_preserved"]
        and release_ok
    )
    state.milestones["M5_e2e"] = state.milestones["M4_functional"] and replies_ok and tests_ok


def main() -> int:
    gold = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "gold_solution.json")
    manifest = load_json(VERIFIER_DATA_ROOT / VARIANT_ID / "workspace_manifest.json")

    state = ScoreState()
    trusted = trusted_final_state(state, gold, manifest)
    visible_ok = run_visible_tests(state)
    hidden_diag = run_hidden_checks(state, gold)
    release_ok = score_release_note(state, gold)
    replies_ok = score_replies(state, gold)
    tests_ok = score_test_updates(state, gold, trusted["changed_paths"])
    score_scope(state, trusted["changed_paths"])
    build_milestones(state, visible_ok, hidden_diag, replies_ok, release_ok, tests_ok)

    milestone_slots = []
    milestone_total = 0.0
    for milestone, weight in MILESTONE_SLOTS:
        passed = bool(state.milestones.get(milestone, False))
        milestone_slots.append({"slot_id": milestone, "weight": weight, "passed_bool": passed})
        if passed:
            milestone_total += weight

    final_score = state.final_score()
    result = {
        "schema_version": VERIFY_SCHEMA,
        "family_id": "pr-thread-contract-remediation",
        "variant_id": VARIANT_ID,
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "ceiling_cap": state.ceiling_cap,
        "pass_bar": PASS_BAR,
        "pass": bool(final_score >= PASS_BAR and state.milestones.get("M5_e2e", False) and state.integrity_flag == 0),
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "ceilings_applied": state.ceilings_applied,
        "breakdown": state.breakdown,
        "bands": state.bands,
        "milestones": state.milestones,
        "milestone_vector": {
            "slots": milestone_slots,
            "M_aggregate": round(milestone_total, 4),
        },
        "errors": state.errors,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
