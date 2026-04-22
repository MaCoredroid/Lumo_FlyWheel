#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA_ROOT = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

VERIFY_RESULT_SCHEMA = "cnb55.verify_result.v3"
MAX_M_POINTS = 87
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


def tree_hash(root: Path, rel: str) -> str:
    target = root / rel
    if not target.exists():
        return "MISSING"
    if target.is_file():
        return sha256_file(target)
    h = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        relpath = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(f"D:{relpath}\n".encode())
        else:
            h.update(f"F:{relpath}\n".encode())
            h.update(sha256_file(path).encode())
            h.update(b"\n")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def file_text(relpath: str) -> str:
    path = AGENT_WS / relpath
    if not path.exists():
        return ""
    return path.read_text()


def lower_text(relpath: str) -> str:
    return file_text(relpath).lower()


def bool_text_contains(text: str, phrases: list[str]) -> bool:
    return all(phrase.lower() in text for phrase in phrases)


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


def baseline_and_integrity(state: ScoreState, gold: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    current_files = collect_workspace_files(AGENT_WS)
    baseline_files = manifest.get("file_hashes", {})
    allowed_write_paths = gold.get("allowed_write_paths", [])
    runtime_paths = gold.get("runtime_paths", [])
    report_path = gold["integration_report_path"]
    proof_path = gold["proof_artifact_path"]

    changed_paths: list[str] = []
    for relpath, sha in baseline_files.items():
        current_sha = current_files.get(relpath)
        if current_sha != sha:
            changed_paths.append(relpath)
    for relpath in current_files:
        if relpath not in baseline_files:
            changed_paths.append(relpath)

    state.submission_present = any(rel_is_allowed(path, allowed_write_paths) for path in changed_paths)
    state.submission_present = state.submission_present or (AGENT_WS / report_path).exists() or (AGENT_WS / proof_path).exists()

    for relpath, expected_hash in gold.get("readonly_tree_hashes", {}).items():
        if tree_hash(AGENT_WS, relpath) != expected_hash:
            state.raise_integrity("immutable_slice_mutated")
            if relpath.startswith("services/api/tests") or relpath.startswith("apps/admin/tests"):
                state.raise_integrity("tests_modified")

    for relpath in changed_paths:
        if not rel_is_allowed(relpath, allowed_write_paths):
            state.raise_integrity("write_outside_whitelist")

    for shim in SHIM_FILES:
        if list(AGENT_WS.rglob(shim)):
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    if state.submission_present and state.integrity_flag == 0:
        state.add("integrity.clean_submission", 5)
        state.milestones["clean_submission"] = True

    runtime_changed = sum(1 for relpath in runtime_paths if current_files.get(relpath) != baseline_files.get(relpath))
    return {
        "changed_paths": sorted(set(changed_paths)),
        "runtime_changed": runtime_changed,
    }


def score_backend(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    review_text = file_text("services/api/src/review_state.py")
    routes_text = file_text("services/api/src/routes/releases.py")

    review_active_ok = (
        "human_review_required" in review_text
        and "ACTIVE_APPROVAL_STATES" in review_text
        and "manual_review" not in re.search(r"ACTIVE_APPROVAL_STATES\s*=\s*\{([^}]*)\}", review_text, re.S).group(1)
        if re.search(r"ACTIVE_APPROVAL_STATES\s*=\s*\{([^}]*)\}", review_text, re.S)
        else False
    )
    review_legacy_ok = '"manual_review": "human_review_required"' in review_text
    routes_ok = (
        "human_review_required" in routes_text
        and 'payload.get("approval_state")' in routes_text
        and "manual_review" not in routes_text
    )

    backend_aligned = review_active_ok and review_legacy_ok and routes_ok
    if backend_aligned:
        state.add("backend.alignment", 15)
        state.milestones["backend_aligned"] = True
    return {
        "backend_aligned": backend_aligned,
        "routes_ok": routes_ok,
        "review_legacy_ok": review_legacy_ok,
    }


def score_frontend(state: ScoreState) -> dict[str, bool]:
    form_text = file_text("apps/admin/src/components/ReleaseGateForm.tsx")
    api_text = file_text("apps/admin/src/lib/api.ts")
    form_ok = "human_review_required" in form_text and "manual_review" not in form_text
    api_ok = "human_review_required" in api_text and "manual_review" not in api_text
    frontend_aligned = form_ok and api_ok
    if frontend_aligned:
        state.add("frontend.alignment", 15)
        state.milestones["frontend_aligned"] = True
    return {
        "frontend_aligned": frontend_aligned,
    }


def score_docs(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    docs_text = lower_text("docs/runbooks/release_gating.md")
    docs_ok = bool_text_contains(docs_text, gold.get("required_docs_phrases", []))
    if docs_ok:
        state.add("docs.alignment", 10)
        state.milestones["docs_aligned"] = True
    return {
        "docs_aligned": docs_ok,
    }


def parse_json_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def score_proof_and_live_path(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    proof_path = AGENT_WS / gold["proof_artifact_path"]
    proof = parse_json_file(proof_path) if proof_path.exists() else None
    proof_exists = proof is not None
    proof_valid = False
    request_ok = False
    echo_ok = False
    if isinstance(proof, dict):
        req = proof.get("captured_request") or {}
        echo = proof.get("server_echo") or {}
        proof_valid = (
            proof.get("schema_version") == "cnb55.release_gate_capture.v1"
            and proof.get("variant_id") == VARIANT_ID
            and proof.get("seed_release_id") == gold.get("seed_release_id")
            and proof.get("proof_type") in {"dom_intercept", "request_echo_pair"}
        )
        request_ok = (
            isinstance(req, dict)
            and req.get("approval_state") == "human_review_required"
            and req.get("request_path") == gold.get("request_path")
            and "manual_review" not in json.dumps(req, sort_keys=True)
        )
        echo_ok = (
            isinstance(echo, dict)
            and echo.get("approval_state") == "human_review_required"
            and echo.get("echo_path") == gold.get("echo_path")
            and "manual_review" not in json.dumps(echo, sort_keys=True)
        )
    if proof_valid:
        state.add("proof.contract", 7)
        state.milestones["proof_valid"] = True
    if request_ok:
        state.add("live.request_correct", 15)
        state.milestones["live_request_correct"] = True
    if echo_ok:
        state.add("live.echo_correct", 15)
        state.milestones["live_echo_correct"] = True
    return {
        "proof_exists": proof_exists,
        "proof_valid": proof_valid,
        "request_ok": request_ok,
        "echo_ok": echo_ok,
    }


def score_report(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    report_path = AGENT_WS / gold["integration_report_path"]
    report_text = report_path.read_text().lower() if report_path.exists() else ""
    report_exists = report_path.exists()
    report_cross_surface = report_exists and bool_text_contains(
        report_text,
        ["backend", "frontend", "docs", "proof"],
    )
    report_variant_awareness = report_exists and bool_text_contains(
        report_text,
        gold.get("required_report_phrases", []),
    )
    if report_cross_surface:
        state.add("report.cross_surface", 8, band="P_only")
        state.milestones["report_cross_surface"] = True
    if report_variant_awareness:
        state.add("report.variant_awareness", 5, band="P_only")
        state.milestones["report_variant_awareness"] = True
    return {
        "report_exists": report_exists,
        "report_cross_surface": report_cross_surface,
        "report_variant_awareness": report_variant_awareness,
    }


def score_runtime_cleanliness(state: ScoreState, gold: dict[str, Any]) -> dict[str, bool]:
    clean = True
    for relpath in gold.get("runtime_paths_without_legacy_token", []):
        if "manual_review" in file_text(relpath):
            clean = False
    if clean:
        state.add("runtime.no_stale_token", 5)
        state.milestones["runtime_no_stale_token"] = True
    return {
        "runtime_clean": clean,
    }


def apply_ceilings(
    state: ScoreState,
    gold: dict[str, Any],
    baseline: dict[str, Any],
    backend: dict[str, bool],
    frontend: dict[str, bool],
    docs: dict[str, bool],
    live: dict[str, bool],
    report: dict[str, bool],
    runtime: dict[str, bool],
) -> None:
    surface_hits = sum(
        1 for ok in [backend["backend_aligned"], frontend["frontend_aligned"], docs["docs_aligned"]] if ok
    )
    if surface_hits < 3:
        state.apply_ceiling("surface_gap", 25)

    if backend["backend_aligned"] and (not frontend["frontend_aligned"] or not docs["docs_aligned"]):
        state.apply_ceiling("alias_only_fix", 30)

    if not live["proof_valid"] or not live["request_ok"] or not live["echo_ok"]:
        state.apply_ceiling("generic_proof", 35)

    if not runtime["runtime_clean"]:
        state.apply_ceiling("stale_runtime_token", 20)

    if not report["report_exists"]:
        state.apply_ceiling("missing_integration_report", 85)

    if gold.get("requires_release_order") and not report["report_variant_awareness"]:
        state.apply_ceiling("release_order_missed", 45)

    if gold.get("incident_ref") and not report["report_variant_awareness"]:
        state.apply_ceiling("incident_blind_recovery", 30)

    if baseline["runtime_changed"] == 0 and not report["report_exists"] and not live["proof_exists"]:
        state.apply_ceiling("no_submission", 0)


def finalize_milestones(state: ScoreState) -> None:
    m1 = bool(
        state.milestones.get("report_cross_surface")
        or (state.milestones.get("backend_aligned") and state.milestones.get("frontend_aligned"))
    )
    m2 = bool(state.milestones.get("backend_aligned") and state.milestones.get("frontend_aligned"))
    m3 = bool(state.integrity_flag == 0 and state.milestones.get("clean_submission"))
    m4 = bool(
        m2
        and state.milestones.get("docs_aligned")
        and state.milestones.get("proof_valid")
    )
    severe_ceiling = any(
        cap in state.ceilings_applied
        for cap in ["stale_runtime_token", "generic_proof", "incident_blind_recovery", "release_order_missed"]
    )
    m5 = bool(
        m4
        and state.milestones.get("live_request_correct")
        and state.milestones.get("live_echo_correct")
        and not severe_ceiling
        and state.integrity_flag == 0
    )

    if state.integrity_flag == 1:
        m3 = False
        m4 = False
        m5 = False

    state.milestones["M1_localization"] = m1
    state.milestones["M2_primary_fix"] = m2
    state.milestones["M3_invariants"] = m3
    state.milestones["M4_functional"] = m4
    state.milestones["M5_e2e"] = m5


def milestone_vector(state: ScoreState) -> dict[str, Any]:
    slots: list[dict[str, Any]] = []
    agg = 0.0
    for slot_id, weight in MILESTONE_SLOTS:
        passed = bool(state.milestones.get(slot_id, False))
        if passed:
            agg += weight
        slots.append({
            "slot_id": slot_id,
            "weight": weight,
            "passed_bool": passed,
        })
    return {
        "slots": slots,
        "M_aggregate": round(agg, 4),
    }


def main() -> int:
    started = time.time()
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    variant_root = VERIFIER_DATA_ROOT / VARIANT_ID
    gold_path = variant_root / "gold_release_state.json"
    manifest_path = variant_root / "workspace_manifest.json"
    if not gold_path.exists() or not manifest_path.exists():
        RESULT_FILE.write_text(json.dumps({
            "pass": False,
            "score": 0,
            "P_benchmark": 0,
            "M_training": 0.0,
            "raw_score_pre_ceiling": 0,
            "raw_M_pre_ceiling": 0,
            "milestones": {},
            "milestone_vector": {"slots": [], "M_aggregate": 0.0},
            "breakdown": {"__bands": {}},
            "ceilings_applied": ["missing_verifier_data"],
            "integrity_flag": 0,
            "integrity_rules_fired": [],
            "shortcut_detected": False,
            "errors": [f"missing verifier data for {VARIANT_ID}"],
            "variant_id": VARIANT_ID,
            "wall_clock_seconds": int(time.time() - started),
            "schema_version": VERIFY_RESULT_SCHEMA,
        }, indent=2, sort_keys=True))
        return 0

    gold = load_json(gold_path)
    manifest = load_json(manifest_path)
    state = ScoreState()

    baseline = baseline_and_integrity(state, gold, manifest)
    backend = score_backend(state, gold)
    frontend = score_frontend(state)
    docs = score_docs(state, gold)
    live = score_proof_and_live_path(state, gold)
    report = score_report(state, gold)
    runtime = score_runtime_cleanliness(state, gold)
    apply_ceilings(state, gold, baseline, backend, frontend, docs, live, report, runtime)
    finalize_milestones(state)

    final_score = state.final_score()
    m_training = state.final_m_training()
    result = {
        "pass": (
            final_score >= int(gold.get("pass_bar", 70))
            and state.integrity_flag == 0
            and state.milestones.get("M5_e2e", False)
        ),
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": m_training,
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_m_score,
        "milestones": dict(sorted(state.milestones.items())),
        "milestone_vector": milestone_vector(state),
        "breakdown": dict(sorted(state.breakdown.items())) | {"__bands": dict(sorted(state.bands.items()))},
        "ceilings_applied": sorted(state.ceilings_applied),
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": sorted(state.integrity_rules_fired),
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "wall_clock_seconds": int(time.time() - started),
        "schema_version": VERIFY_RESULT_SCHEMA,
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
