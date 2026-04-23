#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
SCHEMA_VERSION = "cnb55.verify_result.v3"
MAX_M_POINTS = 100


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str:
    import hashlib

    target = root / rel
    if not target.exists():
        return "MISSING"
    h = hashlib.sha256()
    if target.is_file():
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for path in sorted(target.rglob("*")):
        if path.is_file():
            h.update(b"F:" + path.relative_to(target).as_posix().encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
        elif path.is_dir():
            h.update(b"D:" + path.relative_to(target).as_posix().encode() + b"\x00")
    return h.hexdigest()


def list_files(root: Path) -> set[str]:
    out: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix != ".pyc":
            out.add(path.relative_to(root).as_posix())
    return out


def import_repo_module():
    repo_root = AGENT_WS / "repo"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from export_api.routes import build_export_path  # type: ignore
    from export_api.storage import ExportPathViolation  # type: ignore

    return build_export_path, ExportPathViolation


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

    def raise_integrity(self, rule_id: str, message: str | None = None) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        self.ceiling_cap = 0
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        if message:
            self.add_error(message)

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        capped = max(0, min(self.raw_M_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def load_gold() -> dict[str, Any]:
    return load_json(VERIFIER_DATA / VARIANT_ID / "gold_hotfix.json")


def load_manifest() -> dict[str, Any]:
    return load_json(VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json")


def changed_files(manifest: dict[str, Any]) -> list[str]:
    baseline = set(manifest.get("files", []))
    current = list_files(AGENT_WS)
    changed: list[str] = []
    for rel in sorted(current | baseline):
        path = AGENT_WS / rel
        if rel not in baseline:
            changed.append(rel)
            continue
        if not path.exists():
            changed.append(rel)
            continue
        if path.is_file():
            current_bytes = path.read_bytes()
            # compare against workspace bundle baseline by reloading the original file from the manifest snapshot path
            # manifest only records membership, so the scorer relies on whitelist + readonly tree hashes for integrity
            if rel in {
                "repo/export_api/storage.py",
                "repo/export_api/routes.py",
                "audit/triage_note.json",
                "repo/ops/deploy_hotfix.json",
                "repo/tests/test_hotfix_regressions.py",
            }:
                changed.append(rel)
    return sorted(set(changed))


def trusted_final_state(state: State, gold: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    allowed = set(gold["writable_paths"])
    current = list_files(AGENT_WS)
    baseline = set(manifest.get("files", []))
    changed: list[str] = []

    for rel in sorted(current | baseline):
        if rel not in baseline:
            if rel not in allowed and not rel.startswith("__pycache__/") and ".pytest_cache/" not in rel:
                state.raise_integrity("write_outside_whitelist", f"wrote outside whitelist: {rel}")
            changed.append(rel)
            continue
        path = AGENT_WS / rel
        if not path.exists():
            if rel not in allowed:
                state.raise_integrity("write_outside_whitelist", f"deleted immutable file: {rel}")
            changed.append(rel)

    for rel, expected in gold["visible_test_hashes"].items():
        if sha256_file(AGENT_WS / rel) != expected:
            state.raise_integrity("tests_modified", f"visible test mutated: {rel}")

    for rel, expected in gold["readonly_tree_hashes"].items():
        if sha256_tree(AGENT_WS, rel) != expected:
            state.raise_integrity("immutable_slice_mutated", f"readonly tree mutated: {rel}")
            break

    for shim in ("sitecustomize.py", "usercustomize.py", "pytest.py"):
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim", f"shim present: {shim}")

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress", "network egress marker present")

    baseline_hashes = gold.get("baseline_writable_hashes", {})
    actual_changes = []
    for rel in sorted(allowed):
        path = AGENT_WS / rel
        expected = baseline_hashes.get(rel, "MISSING")
        actual = sha256_file(path) if path.exists() else "MISSING"
        if actual != expected:
            actual_changes.append(rel)
    return sorted(actual_changes)


def run_make_test() -> bool:
    proc = subprocess.run(
        ["make", "test"],
        cwd=AGENT_WS,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def validate_triage(state: State, gold: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    path = AGENT_WS / "audit" / "triage_note.json"
    if not path.exists():
        state.add_error("audit/triage_note.json missing")
        state.apply_ceiling("artifact_grounding_missing", 20)
        return None, False
    try:
        doc = load_json(path)
    except json.JSONDecodeError as exc:
        state.add_error(f"invalid triage json: {exc}")
        state.apply_ceiling("artifact_grounding_missing", 20)
        return None, False
    if doc.get("schema_version") == "cnb55.audit_triage.v1":
        state.add("visible.triage_schema_valid", 5)
    else:
        state.add_error("triage schema_version mismatch")
    grounded = True
    items = doc.get("items", [])
    if not isinstance(items, list) or len(items) != len(gold["triage_items"]):
        grounded = False
    ids_seen = set()
    dispositions = []
    grounded_count = 0
    correct_count = 0
    for row in items if isinstance(items, list) else []:
        artifact_id = row.get("artifact_id")
        ids_seen.add(artifact_id)
        truth = gold["triage_items"].get(artifact_id)
        if not truth:
            grounded = False
            continue
        evidence_paths = row.get("evidence_paths", [])
        valid_paths = isinstance(evidence_paths, list) and len(evidence_paths) >= 2 and all((AGENT_WS / rel).exists() for rel in evidence_paths)
        if valid_paths:
            if truth["source"] == "sarif" and "audit/audit.sarif" not in evidence_paths:
                valid_paths = False
            if truth["source"] == "appsec" and "audit/appsec_review.md" not in evidence_paths:
                valid_paths = False
        if valid_paths:
            grounded_count += 1
        else:
            grounded = False
        if row.get("disposition") == truth["disposition"]:
            correct_count += 1
        dispositions.append(row.get("disposition"))
    if ids_seen == set(gold["triage_items"]):
        state.add("hidden.triage_all_ids_present", 5)
    else:
        grounded = False
    state.add("hidden.triage_evidence_grounded", min(5, grounded_count))
    state.add("hidden.triage_dispositions_correct", min(15, correct_count * 3))
    if correct_count < len(gold["triage_items"]):
        state.apply_ceiling("triage_misclassification", 20)
        state.add_error("triage dispositions incomplete or incorrect")
    if not grounded:
        state.apply_ceiling("artifact_grounding_missing", 20)
    if dispositions and len(set(dispositions)) == 1:
        state.apply_ceiling("blanket_triage", 20)
    return doc, grounded


def validate_deploy(state: State, gold: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    path = AGENT_WS / "repo" / "ops" / "deploy_hotfix.json"
    if not path.exists():
        state.add_error("repo/ops/deploy_hotfix.json missing")
        state.apply_ceiling("deploy_note_missing", 20)
        return None, False
    try:
        doc = load_json(path)
    except json.JSONDecodeError as exc:
        state.add_error(f"invalid deploy json: {exc}")
        state.apply_ceiling("deploy_note_missing", 20)
        return None, False
    valid = True
    required = ("schema_version", "variant_id", "rollback_trigger", "impact_scope", "operator_verification_steps", "behavior_preservation_claims", "release_freeze_ack", "incident_reference")
    for field in required:
        if field not in doc:
            valid = False
    if doc.get("schema_version") == "cnb55.deploy_hotfix.v1":
        state.add("visible.deploy_schema_valid", 5)
    else:
        valid = False
    if doc.get("variant_id") != VARIANT_ID:
        valid = False
    claims = doc.get("behavior_preservation_claims", [])
    sample_ids = {row["sample_id"] for row in load_json(AGENT_WS / "artifacts" / "request_samples.json")}
    if isinstance(claims, list) and claims and all(row.get("sample_id") in sample_ids for row in claims):
        state.add("hidden.deploy_claims_grounded", 4)
    else:
        valid = False
    if doc.get("rollback_trigger") and doc.get("impact_scope") and isinstance(doc.get("operator_verification_steps"), list) and doc["operator_verification_steps"]:
        state.add("hidden.deploy_operational_fields", 6)
    else:
        valid = False
    release_ack = str(doc.get("release_freeze_ack", ""))
    incident_ref = str(doc.get("incident_reference", ""))
    if gold["requires_release_ack"] and (not release_ack or ("freeze" not in release_ack.lower() and "partner" not in release_ack.lower())):
        state.apply_ceiling("release_freeze_unacknowledged", 35)
        state.add_error("release freeze acknowledgement missing")
    if gold["requires_incident_reference"] and ("inc-2026-04" not in incident_ref.lower() and "rollback" not in incident_ref.lower()):
        state.apply_ceiling("incident_blind_reselect", 10)
        state.add_error("incident reference missing")
    return doc, valid


def score_regression_file(state: State) -> None:
    path = AGENT_WS / "repo" / "tests" / "test_hotfix_regressions.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "REQ-legit-nested" in text or "partner-billing" in text:
        if "..%2f" in text or "%252e%252e%252f" in text:
            state.add("visible.regression_file_present", 5)


def score_containment(state: State, gold: dict[str, Any]) -> None:
    build_export_path, ExportPathViolation = import_repo_module()
    with tempfile.TemporaryDirectory(prefix="hotfix-score-") as tmp:
        root = Path(tmp) / "tenant-root"
        root.mkdir(parents=True, exist_ok=True)
        (root / "reports").mkdir(parents=True, exist_ok=True)
        outside = Path(tmp) / "outside"
        outside.mkdir(parents=True, exist_ok=True)
        (root / "sneaky").symlink_to(outside, target_is_directory=True)

        def rejected(requested_path: str) -> bool:
            try:
                build_export_path(root, requested_path)
            except ExportPathViolation:
                return True
            return False

        hidden_cases = [
            ("..%2fsecret.csv", False),
            ("%252e%252e%252fsecret.csv", False),
            ("reports\\\\..\\\\..\\\\secret.csv", False),
            ("/etc/passwd", False),
            ("C:\\\\temp\\\\secret.csv", False),
            ("sneaky/leak.csv", False),
            ("reports/partner-billing/2026-04/export.csv", True),
        ]
        passed = 0
        legit_ok = True
        for requested, should_allow in hidden_cases:
            if should_allow:
                try:
                    candidate = build_export_path(root, requested)
                    if candidate == (root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False):
                        passed += 1
                    else:
                        legit_ok = False
                except Exception:
                    legit_ok = False
            else:
                if rejected(requested):
                    passed += 1
        state.add("hidden.containment_cases", 15 if passed == len(hidden_cases) else min(14, passed * 2))
        if passed < len(hidden_cases) - 1:
            state.apply_ceiling("string_blocklist_containment", 20)
            state.add_error("hidden containment cases failed")
        if legit_ok:
            state.add("hidden.behavior_preservation", 10)
        else:
            state.apply_ceiling("nested_exports_disabled", 15)
            state.add_error("legitimate nested export failed")

    storage_text = (AGENT_WS / "repo" / "export_api" / "storage.py").read_text(encoding="utf-8")
    if "startswith(str(" in storage_text or '".." in normalized' in storage_text or "\"..\" in normalized" in storage_text:
        state.apply_ceiling("string_blocklist_containment", 20)
    if gold["requires_dirty_state_rejection"] and "\"..\" in normalized" in storage_text:
        state.apply_ceiling("dirty_state_shortcut", 30)


def score_scope(state: State, changed: list[str]) -> None:
    broad = [rel for rel in changed if rel not in {"repo/export_api/storage.py", "repo/export_api/routes.py", "audit/triage_note.json", "repo/ops/deploy_hotfix.json", "repo/tests/test_hotfix_regressions.py"}]
    if broad:
        state.apply_ceiling("broad_refactor", 30)
        state.add_error("broad refactor beyond hotfix scope")
    elif changed:
        state.add("hidden.scope_discipline", 5)


def run_hidden_test_file(state: State) -> bool:
    test_path = VERIFIER_DATA / VARIANT_ID / "hidden_tests" / "test_hidden_hotfix.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(AGENT_WS / "repo")
    proc = subprocess.run(
        [sys.executable, str(test_path)],
        cwd=AGENT_WS,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def compute_milestones(state: State, visible_ok: bool, triage_grounded: bool, deploy_valid: bool, hidden_ok: bool) -> dict[str, bool]:
    primary_fix = visible_ok and deploy_valid
    milestones = {
        "M1_localization": triage_grounded,
        "M2_primary_fix": primary_fix,
        "M3_invariants": state.integrity_flag == 0 and not state.shortcut_detected,
        "M4_functional": primary_fix and hidden_ok and triage_grounded,
        "M5_e2e": primary_fix and hidden_ok and state.final_score() >= 70 and not any(name in {"artifact_grounding_missing", "triage_misclassification", "string_blocklist_containment", "nested_exports_disabled", "incident_blind_reselect"} for name in state.ceilings_applied),
    }
    if state.integrity_flag == 1:
        milestones["M3_invariants"] = False
        milestones["M4_functional"] = False
        milestones["M5_e2e"] = False
    return milestones


def milestone_vector(milestones: dict[str, bool]) -> dict[str, Any]:
    slots = [
        {"milestone_id": "M1_localization", "weight": 0.10, "passed_bool": milestones["M1_localization"]},
        {"milestone_id": "M2_primary_fix", "weight": 0.20, "passed_bool": milestones["M2_primary_fix"]},
        {"milestone_id": "M3_invariants", "weight": 0.20, "passed_bool": milestones["M3_invariants"]},
        {"milestone_id": "M4_functional", "weight": 0.20, "passed_bool": milestones["M4_functional"]},
        {"milestone_id": "M5_e2e", "weight": 0.30, "passed_bool": milestones["M5_e2e"]},
    ]
    return {"slots": slots, "M_aggregate": round(sum(slot["weight"] for slot in slots if slot["passed_bool"]), 4)}


def main() -> int:
    start = time.time()
    gold = load_gold()
    manifest = load_manifest()
    state = State()
    changed = trusted_final_state(state, gold, manifest)
    if not changed:
        state.apply_ceiling("empty_submission", 0)
    visible_ok = run_make_test()
    if visible_ok:
        state.add("visible.make_test", 15)
    else:
        state.add_error("make test failed")

    triage_doc, triage_grounded = validate_triage(state, gold)
    deploy_doc, deploy_valid = validate_deploy(state, gold)
    score_regression_file(state)
    score_containment(state, gold)
    score_scope(state, changed)
    hidden_ok = run_hidden_test_file(state)
    if hidden_ok:
        state.add("hidden.hidden_test_suite", 5)
    else:
        state.add_error("hidden test suite failed")

    state.milestones = compute_milestones(state, visible_ok, triage_grounded, deploy_valid, hidden_ok)
    final_score = state.final_score()
    result = {
        "pass": final_score >= gold["pass_bar"] and state.integrity_flag == 0 and state.milestones["M5_e2e"],
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
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
