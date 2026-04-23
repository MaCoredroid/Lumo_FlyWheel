#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace")).resolve()
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data")).resolve()
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json")).resolve()
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

VERIFY_SCHEMA = "cnb55.verify_result.v3"
MAX_M_POINTS = 100
MILESTONE_SLOTS = [
    ("M1_localization", 0.10),
    ("M2_primary_fix", 0.20),
    ("M3_invariants", 0.20),
    ("M4_functional", 0.20),
    ("M5_e2e", 0.30),
]
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path, rel: str) -> str:
    import hashlib

    target = root / rel
    if not target.exists():
        return "MISSING"
    if target.is_file():
        return sha256_file(target)
    digest = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        relpath = path.relative_to(target).as_posix()
        if path.is_dir():
            digest.update(f"D:{relpath}\n".encode())
        else:
            digest.update(f"F:{relpath}\n".encode())
            digest.update(sha256_file(path).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_workspace_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    return files


def rel_is_allowed(relpath: str, allowed_paths: list[str]) -> bool:
    for allowed in allowed_paths:
        allowed = allowed.rstrip("/")
        if relpath == allowed or relpath.startswith(f"{allowed}/"):
            return True
    return False


def run_pytest(test_paths: list[str], extra_env: dict[str, str] | None = None) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(AGENT_WS)
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *test_paths]
    proc = subprocess.run(
        cmd,
        cwd=AGENT_WS,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode == 0, proc.stdout


def run_command(args: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        args,
        cwd=AGENT_WS,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode == 0, proc.stdout


@dataclass
class ScoreState:
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    milestones: dict[str, bool] = field(default_factory=dict)
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
        if self.integrity_flag:
            return 0
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        if self.integrity_flag:
            return 0.0
        capped = max(0, min(self.raw_m_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)


def baseline_and_integrity(state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    current_files = collect_workspace_files(AGENT_WS)
    baseline_files = manifest.get("file_hashes", {})
    allowed_write_paths = gold.get("allowed_write_paths", [])
    changed_paths: list[str] = []
    for relpath, sha in baseline_files.items():
        if current_files.get(relpath) != sha:
            changed_paths.append(relpath)
    for relpath in current_files:
        if relpath not in baseline_files:
            changed_paths.append(relpath)

    for relpath in sorted(set(changed_paths)):
        if relpath == ".network_egress_detected":
            continue
        if relpath.startswith("__pycache__/") or "/__pycache__/" in relpath:
            continue
        if not rel_is_allowed(relpath, allowed_write_paths):
            state.raise_integrity("write_outside_whitelist")

    readonly_tree_hashes = gold.get("readonly_tree_hashes", {})
    for relpath, expected_hash in readonly_tree_hashes.items():
        if tree_hash(AGENT_WS, relpath) != expected_hash:
            state.raise_integrity("immutable_slice_mutated")
            if relpath.startswith("tests"):
                state.raise_integrity("tests_modified")

    for shim in SHIM_FILES:
        if list(AGENT_WS.rglob(shim)):
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    runtime_changed = sum(
        1
        for relpath in gold.get("runtime_paths", [])
        if current_files.get(relpath) != baseline_files.get(relpath)
    )
    proof_exists = (AGENT_WS / "artifacts/release_smoke_report.json").exists()
    state.submission_present = runtime_changed > 0 or proof_exists
    if runtime_changed >= 2:
        state.add("localize.surface_coverage", 5)
        state.milestones["M1_localization"] = True
    return {
        "runtime_changed": runtime_changed,
        "changed_paths": sorted(set(changed_paths)),
    }


def evaluate_alignment() -> dict[str, Any]:
    sys.path.insert(0, str(AGENT_WS))
    try:
        from deploy.check_release import evaluate_release_alignment  # type: ignore

        return evaluate_release_alignment(AGENT_WS, "staging")
    finally:
        if sys.path and sys.path[0] == str(AGENT_WS):
            sys.path.pop(0)


def validate_proof(gold: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    proof_path = AGENT_WS / "artifacts/release_smoke_report.json"
    if not proof_path.exists():
        return False, None
    try:
        payload = load_json(proof_path)
    except Exception:
        return False, None
    contract = gold["proof_contract"]
    ok = (
        payload.get("schema_version") == contract["schema_version"]
        and payload.get("variant_id") == VARIANT_ID
        and payload.get("target_environment") == "staging"
        and payload.get("artifact_manifest") == contract["artifact_manifest"]
        and payload.get("smoke_status") == "ok"
        and payload.get("ordered_checks") == contract["ordered_checks"]
        and payload.get("incident_guardrail") == contract["incident_guardrail"]
    )
    return ok, payload


def main() -> int:
    gold = load_json(VERIFIER_DATA / VARIANT_ID / "gold_release.json")
    manifest = load_json(VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json")
    state = ScoreState()

    integrity = baseline_and_integrity(state, gold, manifest)

    visible_ok, visible_output = run_pytest(
        ["tests/test_manifest_contract.py", "tests/test_release_driver.py"]
    )
    if visible_ok:
        state.add("phase2.visible_pytest", 15)
    else:
        state.errors.append("visible_pytest_failed")

    dry_run_ok, dry_run_output = run_command([sys.executable, "scripts/run_ci.py", "--mode", "release-dry-run"])
    if dry_run_ok:
        state.add("phase2.visible_dry_run", 15)
    else:
        state.errors.append("dry_run_failed")

    if visible_ok and dry_run_ok:
        state.milestones["M2_primary_fix"] = True
    else:
        state.apply_ceiling("visible_gate_incomplete", 10)

    alignment = evaluate_alignment()
    smoke_ok = bool(alignment.get("ok"))
    if smoke_ok:
        state.add("hidden.staging_smoke", 20)
    else:
        state.errors.append("staging_smoke_failed")
        state.apply_ceiling("staging_smoke_failed", 20)

    hidden_test = str(VERIFIER_DATA / VARIANT_ID / "hidden_tests" / "test_release_alignment_hidden.py")
    hidden_ok, hidden_output = run_pytest(
        [hidden_test],
        extra_env={
            "AGENT_WS": str(AGENT_WS),
            "VERIFIER_DATA": str(VERIFIER_DATA),
            "VARIANT_ID": VARIANT_ID,
        },
    )
    if hidden_ok:
        state.add("hidden.release_alignment_pack", 20)
    else:
        state.errors.append("hidden_alignment_pack_failed")
        state.apply_ceiling("hidden_alignment_pack_failed", 30)

    checks = alignment.get("checks", {})
    if checks.get("config") and checks.get("docs_base") and checks.get("variant_docs"):
        state.add("contract.config_docs", 10)
    else:
        state.apply_ceiling("config_or_docs_drift", 35)
    if not checks.get("variant_docs", True):
        state.apply_ceiling("variant_context_missed", 35)
    if not checks.get("no_prod_alias", True):
        state.apply_ceiling("prod_alias_retained", 25)

    proof_ok, proof_payload = validate_proof(gold)
    if proof_ok:
        state.add("proof.release_smoke_report", 15)
    else:
        state.errors.append("proof_artifact_missing_or_invalid")
        state.apply_ceiling("no_proof_artifact", 30)

    if state.integrity_flag == 0:
        state.milestones["M3_invariants"] = True
    if hidden_ok and checks.get("config") and checks.get("docs_base") and checks.get("variant_docs"):
        state.milestones["M4_functional"] = True
    if smoke_ok and proof_ok and state.integrity_flag == 0 and state.ceiling_cap > 30:
        state.milestones["M5_e2e"] = True

    milestone_vector = {"slots": [], "M_aggregate": 0.0}
    total = 0.0
    for key, weight in MILESTONE_SLOTS:
        passed = bool(state.milestones.get(key, False))
        milestone_vector["slots"].append(
            {"slot": key, "weight": weight, "passed_bool": passed}
        )
        if passed:
            total += weight
    if state.integrity_flag:
        total = 0.0
        for slot in milestone_vector["slots"]:
            if slot["slot"] in {"M3_invariants", "M4_functional", "M5_e2e"}:
                slot["passed_bool"] = False
    milestone_vector["M_aggregate"] = round(total, 4)

    result = {
        "schema_version": VERIFY_SCHEMA,
        "score": state.final_score(),
        "P_benchmark": state.final_score(),
        "M_training": state.final_m_training(),
        "raw_score_pre_ceiling": state.raw_score,
        "pass": bool(state.milestones.get("M5_e2e", False) and state.final_score() >= 60),
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "ceilings_applied": state.ceilings_applied,
        "breakdown": {**state.breakdown, "__bands": state.bands},
        "errors": state.errors,
        "milestones": {
            key: bool(state.milestones.get(key, False))
            for key, _ in MILESTONE_SLOTS
        },
        "milestone_vector": milestone_vector,
        "details": {
            "runtime_changed": integrity["runtime_changed"],
            "changed_paths": integrity["changed_paths"],
            "alignment_checks": checks,
            "proof_valid": proof_ok,
            "proof_payload": proof_payload,
            "visible_pytest_output": visible_output[-4000:],
            "dry_run_output": dry_run_output[-4000:],
            "hidden_pytest_output": hidden_output[-4000:],
        },
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
