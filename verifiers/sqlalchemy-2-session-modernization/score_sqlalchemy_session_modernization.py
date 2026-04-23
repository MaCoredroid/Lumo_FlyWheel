#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_WS = Path(os.environ.get("AGENT_WS", ".")).resolve()
VERIFIER_DATA = Path(
    os.environ.get(
        "VERIFIER_DATA",
        REPO_ROOT / "verifier_data/sqlalchemy-2-session-modernization",
    )
).resolve()
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "verify_result.json")).resolve()
SCHEMA_VERSION = "cnb55.verify_result.v3"

ALLOWED_WRITE_PATHS = {
    "app/api.py",
    "app/repository.py",
    "app/worker.py",
    "app/admin_cli.py",
    "docs/deploy/sqlalchemy2-cutover.md",
}
SHIM_FILES = {"sitecustomize.py", "usercustomize.py", "pytest.py"}

COMMON_CHECKS = [
    ("visible.visible_tests", 10, "M"),
    ("static.no_session_query", 8, "M"),
    ("static.no_repository_commit", 8, "M"),
    ("static.no_global_session", 4, "M"),
    ("static.docs_contract", 8, "M"),
    ("hidden.exists_and_count", 8, "M"),
    ("hidden.worker_rollback", 14, "M"),
    ("hidden.admin_dry_run", 14, "M"),
    ("integrity.allowed_writes", 8, "M"),
    ("p_only.partial_progress", 10, "P_only"),
]

VARIANT_CHECKS = {
    "v1-clean-baseline": [],
    "v2-noisy-distractor": [("static.rejects_compat_shim", 8, "M")],
    "v3-dirty-state": [("static.preserve_dirty_context", 8, "M")],
    "v4-multi-corpus-objective": [("hidden.batch_atomicity", 18, "M")],
    "v5-recovery-in-thread": [
        ("hidden.batch_atomicity", 12, "M"),
        ("hidden.incident_recovery", 10, "M"),
        ("static.incident_acknowledged", 8, "M"),
    ],
}

HIDDEN_TEST_SPECS = {
    "hidden.exists_and_count": "test_hidden_contract.py::test_entry_exists_and_pending_count_are_consistent",
    "hidden.worker_rollback": "test_hidden_contract.py::test_worker_failure_rolls_back_to_pending",
    "hidden.admin_dry_run": "test_hidden_contract.py::test_admin_dry_run_stays_read_only",
    "hidden.batch_atomicity": "test_hidden_contract.py::test_v4_batch_helpers_are_atomic",
    "hidden.incident_recovery": "test_hidden_contract.py::test_v5_incident_recovery_is_acknowledged_and_retry_is_idempotent",
}

CEILINGS = [
    ("integrity_zero", 0),
    ("visible_only", 20),
    ("helper_commit_shortcut", 25),
    ("worker_partial_write", 25),
    ("dry_run_persists", 25),
    ("batch_atomicity_missing", 35),
    ("incident_blind_fix", 35),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.exists():
        return None
    h = hashlib.sha256()
    if target.is_file():
        if target.suffix == ".pyc":
            return None
        h.update(b"F")
        h.update(sha256_file(target).encode())
        return h.hexdigest()
    for path in sorted(target.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel_path = path.relative_to(target).as_posix()
        if path.is_dir():
            h.update(b"D:" + rel_path.encode() + b"\x00")
        else:
            h.update(b"F:" + rel_path.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
    return h.hexdigest()


def load_manifest() -> dict:
    path = VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json"
    if not path.exists():
        return {"files": {}, "readonly_tree_hashes": {}}
    return json.loads(path.read_text())


def run_command(args: list[str], *, cwd: Path, extra_env: dict[str, str] | None = None) -> tuple[bool, str]:
    env = os.environ.copy()
    env["AGENT_WS"] = str(AGENT_WS)
    env["VARIANT_ID"] = VARIANT_ID
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode == 0, proc.stdout


@dataclass
class ScoreState:
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    milestone_bools: dict[str, bool] = field(default_factory=dict)
    shortcut_detected: bool = False
    max_m_points: int = 0
    max_p_only_points: int = 0

    def add_check(self, key: str, max_points: int, band: str) -> None:
        self.breakdown.setdefault(key, 0)
        self.breakdown_bands[key] = band
        if band == "M":
            self.max_m_points += max_points
        else:
            self.max_p_only_points += max_points

    def award(self, key: str, points: int) -> None:
        self.breakdown[key] = points

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)


def current_file_hashes(manifest: dict) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    changed: list[str] = []
    for rel, baseline in manifest.get("files", {}).items():
        path = AGENT_WS / rel
        current = sha256_file(path) if path.exists() else "MISSING"
        hashes[rel] = current
        if baseline != current:
            changed.append(rel)
    return hashes, changed


def check_integrity(state: ScoreState, manifest: dict, changed_files: list[str]) -> None:
    readonly_hashes = manifest.get("readonly_tree_hashes", {})

    if any((AGENT_WS / shim).exists() for shim in SHIM_FILES):
        state.raise_integrity("pytest_shim")

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    if any(path.startswith("tests/") for path in changed_files):
        state.raise_integrity("tests_modified")

    immutable_roots = ("notes", "release_context", "incident_context", "seed", "README.md", "AGENTS.md", "Dockerfile", ".scenario_variant")
    for rel in immutable_roots:
        baseline = readonly_hashes.get(rel)
        if baseline is None:
            continue
        current = sha256_tree(AGENT_WS, rel)
        if baseline != current:
            state.raise_integrity("immutable_slice_mutated")
            break

    write_outside = []
    for rel in changed_files:
        if rel in ALLOWED_WRITE_PATHS:
            continue
        if rel.startswith("app/") and rel not in ALLOWED_WRITE_PATHS:
            write_outside.append(rel)
            continue
        if rel.startswith("docs/") and rel not in ALLOWED_WRITE_PATHS:
            write_outside.append(rel)
            continue
        if rel not in ALLOWED_WRITE_PATHS:
            write_outside.append(rel)
    if write_outside:
        state.raise_integrity("write_outside_whitelist")


def static_checks(state: ScoreState, changed_files: list[str]) -> dict[str, bool]:
    app_sources = []
    for rel in ("app/repository.py", "app/api.py", "app/worker.py", "app/admin_cli.py"):
        path = AGENT_WS / rel
        if path.exists():
            app_sources.append(path.read_text())
    all_app_text = "\n".join(app_sources)
    repository_text = (AGENT_WS / "app/repository.py").read_text()
    db_text = (AGENT_WS / "app/db.py").read_text().lower()
    deploy_text = (AGENT_WS / "docs/deploy/sqlalchemy2-cutover.md").read_text().lower()

    results = {
        "static.no_session_query": "session.query(" not in all_app_text,
        "static.no_repository_commit": ".commit(" not in repository_text,
        "static.no_global_session": "global_session" not in db_text,
        "static.docs_contract": (
            "transaction" in deploy_text
            and "rollback" in deploy_text
            and ("dry-run" in deploy_text or "dry run" in deploy_text)
        ),
        "static.rejects_compat_shim": "compatibility shim" not in deploy_text,
        "static.preserve_dirty_context": (
            not (AGENT_WS / "README.md").exists()
            or "Local notes:" in (AGENT_WS / "README.md").read_text()
        ),
        "static.incident_acknowledged": "incident" in deploy_text,
    }

    touched_targets = len([path for path in changed_files if path in ALLOWED_WRITE_PATHS])
    partial_progress = 0
    if touched_targets:
        partial_progress += 4
    if results["static.no_session_query"]:
        partial_progress += 3
    if results["static.docs_contract"]:
        partial_progress += 3
    results["p_only.partial_progress"] = partial_progress

    if results["static.no_repository_commit"] is False:
        state.shortcut_detected = True
    if results["static.rejects_compat_shim"] is False:
        state.shortcut_detected = True
    return results


def run_visible_tests() -> bool:
    ok, _ = run_command(
        [
            "pytest",
            "-q",
            "tests/test_api.py",
            "tests/test_worker.py",
            "tests/test_admin_cli.py",
        ],
        cwd=AGENT_WS,
    )
    return ok


def run_hidden_test(key: str) -> bool:
    spec = HIDDEN_TEST_SPECS[key]
    test_file = VERIFIER_DATA / VARIANT_ID / "hidden_tests" / "test_hidden_contract.py"
    ok, _ = run_command(
        ["pytest", "-q", f"{test_file}::{spec.split('::', 1)[1]}"],
        cwd=REPO_ROOT,
    )
    return ok


def apply_ceilings(state: ScoreState, raw_score_pre_ceiling: int, signals: dict[str, bool]) -> tuple[int, list[str]]:
    score = raw_score_pre_ceiling
    ceilings: list[str] = []
    if state.integrity_flag:
        score = min(score, 0)
        ceilings.append("integrity_zero")
    if not signals["visible.visible_tests"]:
        score = min(score, 20)
        ceilings.append("visible_only")
    if not signals["static.no_repository_commit"]:
        score = min(score, 25)
        ceilings.append("helper_commit_shortcut")
    if not signals["hidden.worker_rollback"]:
        score = min(score, 25)
        ceilings.append("worker_partial_write")
    if not signals["hidden.admin_dry_run"]:
        score = min(score, 25)
        ceilings.append("dry_run_persists")
    if "hidden.batch_atomicity" in signals and not signals["hidden.batch_atomicity"]:
        score = min(score, 35)
        ceilings.append("batch_atomicity_missing")
    if "hidden.incident_recovery" in signals and not signals["hidden.incident_recovery"]:
        score = min(score, 35)
        ceilings.append("incident_blind_fix")
    return score, ceilings


def compute_milestones(state: ScoreState, signals: dict[str, bool], changed_files: list[str]) -> dict:
    m1 = bool([path for path in changed_files if path in ALLOWED_WRITE_PATHS])
    m2 = signals["visible.visible_tests"]
    m3 = state.integrity_flag == 0
    m4 = all(
        signals.get(key, False)
        for key in (
            "visible.visible_tests",
            "hidden.worker_rollback",
            "hidden.admin_dry_run",
            "hidden.exists_and_count",
        )
    )
    variant_e2e = True
    if VARIANT_ID in {"v4-multi-corpus-objective", "v5-recovery-in-thread"}:
        variant_e2e = variant_e2e and signals.get("hidden.batch_atomicity", False)
    if VARIANT_ID == "v5-recovery-in-thread":
        variant_e2e = variant_e2e and signals.get("hidden.incident_recovery", False)
    m5 = m4 and signals["static.docs_contract"] and variant_e2e
    if state.integrity_flag:
        m3 = False
        m4 = False
        m5 = False

    milestones = {
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
    aggregate = 0.0
    for key in ("M1_localization", "M2_primary_fix", "M3_invariants", "M4_functional", "M5_e2e"):
        passed = milestones[key]
        aggregate += weights[key] if passed else 0.0
        slots.append(
            {
                "slot_id": key,
                "weight": weights[key],
                "passed_bool": passed,
            }
        )
    return {
        "milestones": milestones,
        "milestone_vector": {"slots": slots, "M_aggregate": round(aggregate, 4)},
    }


def main() -> int:
    start = time.time()
    manifest = load_manifest()
    _, changed_files = current_file_hashes(manifest)
    progress_started = bool(changed_files)
    state = ScoreState()

    for key, max_points, band in COMMON_CHECKS + VARIANT_CHECKS.get(VARIANT_ID, []):
        state.add_check(key, max_points, band)

    check_integrity(state, manifest, changed_files)

    signals: dict[str, bool] = {}
    visible_ok = run_visible_tests()
    signals["visible.visible_tests"] = visible_ok
    state.award("visible.visible_tests", 10 if visible_ok else 0)

    static_results = static_checks(state, changed_files)
    for key, ok in static_results.items():
        if key == "p_only.partial_progress":
            continue
        if key not in state.breakdown:
            continue
        max_points = next(max_points for check_key, max_points, _ in COMMON_CHECKS + VARIANT_CHECKS.get(VARIANT_ID, []) if check_key == key)
        signals[key] = bool(ok)
        state.award(key, max_points if ok and progress_started else 0)
    state.award("p_only.partial_progress", static_results["p_only.partial_progress"])

    for key in HIDDEN_TEST_SPECS:
        if key not in state.breakdown:
            continue
        ok = run_hidden_test(key)
        signals[key] = ok
        max_points = next(max_points for check_key, max_points, _ in COMMON_CHECKS + VARIANT_CHECKS.get(VARIANT_ID, []) if check_key == key)
        state.award(key, max_points if ok and progress_started else 0)

    if progress_started and state.integrity_flag == 0 and not any(rule in state.integrity_rules_fired for rule in ("write_outside_whitelist", "tests_modified", "immutable_slice_mutated")):
        state.award("integrity.allowed_writes", 8)
    else:
        state.award("integrity.allowed_writes", 0)

    raw_m_points = sum(
        points
        for key, points in state.breakdown.items()
        if state.breakdown_bands.get(key) == "M"
    )
    raw_p_only_points = sum(
        points
        for key, points in state.breakdown.items()
        if state.breakdown_bands.get(key) == "P_only"
    )

    total_points = state.max_m_points + state.max_p_only_points
    raw_score_pre_ceiling = int(round(100 * (raw_m_points + raw_p_only_points) / max(total_points, 1)))
    p_benchmark, ceilings = apply_ceilings(state, raw_score_pre_ceiling, signals)
    state.ceilings_applied = ceilings
    m_training = round(raw_m_points / max(state.max_m_points, 1), 4)

    milestone_payload = compute_milestones(state, signals, changed_files)

    result = {
        "schema_version": SCHEMA_VERSION,
        "variant_id": VARIANT_ID,
        "pass": p_benchmark >= 85 and state.integrity_flag == 0,
        "score": p_benchmark,
        "P_benchmark": p_benchmark,
        "M_training": m_training,
        "raw_score_pre_ceiling": raw_score_pre_ceiling,
        "raw_M_pre_ceiling": raw_m_points,
        "breakdown": {
            "__bands": state.breakdown_bands,
            **state.breakdown,
        },
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "wall_clock_seconds": int(round(time.time() - start)),
        **milestone_payload,
    }

    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
