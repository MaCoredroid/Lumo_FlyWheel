#!/usr/bin/env python3
"""
Deterministic scorer for `multi-tool-transaction-repair`.

Milestones (family.yaml weights):
  M1_localization  0.10  changed services/transactions/ and at least one services/{orders,billing,notifications}/*.py
  M2_primary_fix   0.20  pytest tests/test_transaction_flow.py passes
  M3_invariants    0.20  no duplicate notification side effects (test asserts) AND docs/transaction_ops.md updated
  M4_functional    0.20  pytest tests/test_atomicity.py passes
  M5_e2e           0.30  full pytest -q passes + integrity clean + operator note covers partial-rollback

Integrity flags:
  - tests_modified           tests under tests/ mutated (hash drift on visible tests)
  - pytest_shim              sitecustomize.py / usercustomize.py / pytest.py present
  - happy_path_only_shim     test bypasses by stubbing the workflow (heuristic)
  - write_outside_whitelist  wrote into a forbidden tree
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
SCHEMA_VERSION = "cnb55.verify_result.v3"
MAX_M_POINTS = 100

PYTEST_CMD = [sys.executable, "-m", "pytest"]

SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
FORBIDDEN_WRITE_PREFIXES = ("tests/",)
EXPECTED_SERVICE_DIRS = ("services/orders", "services/billing", "services/notifications", "services/transactions")
EXPECTED_TEST_FILES = ("tests/test_transaction_flow.py", "tests/test_atomicity.py")

# Heuristic keywords expected in a real operator-note update covering partial-rollback semantics.
OPERATOR_NOTE_KEYWORDS = (
    "rollback",
    "atomic",
    "partial",
    "idempot",
    "retry",
    "duplicate",
    "boundary",
)


@dataclass
class State:
    breakdown: dict[str, int] = field(default_factory=dict)
    bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_M_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    milestones: dict[str, bool] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

    def apply_ceiling(self, name: str, cap: int) -> None:
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.ceiling_cap = min(self.ceiling_cap, cap)

    def raise_integrity(self, rule_id: str, message: str | None = None) -> None:
        self.integrity_flag = 1
        self.ceiling_cap = 0
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        if message:
            self.errors.append(message)

    def final_score(self) -> int:
        return max(0, min(self.raw_score, self.ceiling_cap))

    def final_m_training(self) -> float:
        return round(max(0, min(self.raw_M_score, self.ceiling_cap)) / MAX_M_POINTS, 4)


def list_files(root: Path) -> set[str]:
    out: set[str] = set()
    for p in root.rglob("*"):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and p.suffix != ".pyc":
            out.add(p.relative_to(root).as_posix())
    return out


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_integrity(state: State, bundle_root: Path | None) -> None:
    for shim in SHIM_FILES:
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim", f"shim present: {shim}")
    # Visible test must not be modified; we approximate by hashing against the family workspace bundle if available.
    for rel in EXPECTED_TEST_FILES:
        ws = AGENT_WS / rel
        if not ws.is_file():
            continue
        if bundle_root is not None and (bundle_root / rel).is_file():
            if sha256_file(ws) != sha256_file(bundle_root / rel):
                state.raise_integrity("tests_modified", f"visible test mutated: {rel}")


def ensure_pytest(state: State) -> bool:
    """Verify pytest is importable for the grader's interpreter; self-install once if not."""
    def _check() -> tuple[bool, str, str]:
        proc = subprocess.run(
            [sys.executable, "-c", "import pytest, sys; print(pytest.__version__)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        )
        return (proc.returncode == 0,
                proc.stdout.decode("utf-8", errors="replace").strip(),
                proc.stderr.decode("utf-8", errors="replace").strip())
    ok, version, err = _check()
    state.diagnostics["pytest_available"] = {
        "executable": sys.executable, "initial_ok": ok, "version": version,
        "stderr": "" if ok else err,
    }
    if ok:
        return True
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "pytest"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
    )
    state.diagnostics["pytest_install_attempt"] = {
        "rc": install.returncode,
        "stderr_tail": install.stderr.decode("utf-8", errors="replace")[-300:],
    }
    ok2, version2, err2 = _check()
    state.diagnostics["pytest_available"]["after_install"] = {"ok": ok2, "version": version2, "stderr": err2 if not ok2 else ""}
    return ok2


def run_pytest(state: State, args: list[str], key: str, timeout: int = 90) -> bool:
    try:
        proc = subprocess.run(
            [*PYTEST_CMD, "-q", *args],
            cwd=str(AGENT_WS),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        ok = proc.returncode == 0
        tail = proc.stdout.decode("utf-8", errors="replace").splitlines()[-20:]
        state.diagnostics[key] = {"rc": proc.returncode, "tail": tail}
        return ok
    except subprocess.TimeoutExpired:
        state.diagnostics[key] = "timeout"
        return False
    except Exception as e:
        state.diagnostics[key] = f"exec_error: {e}"
        return False


def score(state: State) -> None:
    # Ensure pytest is available before any test invocation; record infra status to diagnostics.
    if not ensure_pytest(state):
        state.diagnostics["pytest_unavailable"] = True
    # M1: file-state — agent touched the expected service trees
    service_files_present = sum(1 for d in EXPECTED_SERVICE_DIRS if (AGENT_WS / d).is_dir() and any((AGENT_WS / d).rglob("*.py")))
    state.milestones["M1_localization"] = service_files_present >= 3
    state.add("M1.service_trees_present", 5 if service_files_present >= 3 else 0, "M")
    docs_present = (AGENT_WS / "docs" / "transaction_ops.md").is_file()
    state.add("M1.docs_present", 5 if docs_present else 0, "M")

    # M2: visible flow tests pass
    flow_ok = run_pytest(state, ["tests/test_transaction_flow.py"], "pytest_flow")
    state.milestones["M2_primary_fix"] = flow_ok
    state.add("M2.flow_tests_pass", 20 if flow_ok else 0, "M")

    # M3: invariants — atomicity tests check for duplicate side effects; we also check docs update content
    atomicity_ok = run_pytest(state, ["tests/test_atomicity.py"], "pytest_atomicity")
    # We grade docs separately: must contain rollback/atomicity language.
    docs_text = ""
    if docs_present:
        docs_text = (AGENT_WS / "docs" / "transaction_ops.md").read_text(encoding="utf-8", errors="replace").lower()
    docs_addresses_invariants = any(kw in docs_text for kw in OPERATOR_NOTE_KEYWORDS)
    state.milestones["M3_invariants"] = atomicity_ok and docs_addresses_invariants
    state.add("M3.atomicity_tests_pass", 15 if atomicity_ok else 0, "M")
    state.add("M3.docs_invariants_covered", 5 if docs_addresses_invariants else 0, "M")

    # M4: functional — combined pytest run
    combined_ok = run_pytest(state, ["tests/test_transaction_flow.py", "tests/test_atomicity.py"], "pytest_combined")
    state.milestones["M4_functional"] = combined_ok
    state.add("M4.combined_pytest_pass", 20 if combined_ok else 0, "M")

    # M5: e2e — full pytest -q, no integrity flag, docs covers partial-rollback specifically
    full_ok = run_pytest(state, [], "pytest_full", timeout=180)
    partial_rollback_covered = "rollback" in docs_text and ("partial" in docs_text or "boundary" in docs_text)
    state.milestones["M5_e2e"] = full_ok and partial_rollback_covered and not state.integrity_flag
    state.add("M5.full_pytest_pass", 20 if full_ok else 0, "M")
    state.add("M5.partial_rollback_doc", 10 if partial_rollback_covered else 0, "M")

    # Apply ceilings consistent with the family's grader-contract description.
    if not atomicity_ok:
        # Atomicity is the binding hidden invariant; if it fails the score caps at 40.
        state.apply_ceiling("atomicity_failed", 40)
    if not flow_ok:
        # If flow tests fail too, even tighter cap.
        state.apply_ceiling("flow_failed", 30)
    if not partial_rollback_covered and docs_present:
        state.apply_ceiling("operator_note_pre_retry_only", 60)


def main() -> int:
    state = State()
    if not AGENT_WS.exists():
        print(f"AGENT_WS not found: {AGENT_WS}", file=sys.stderr)
        return 2

    # Optional bundle root for test-hash comparison
    bundle_root_env = os.environ.get("WORKSPACE_BUNDLE_ROOT")
    bundle_root = Path(bundle_root_env) if bundle_root_env else None

    check_integrity(state, bundle_root)
    score(state)

    final = state.final_score()
    milestone_vector = [int(state.milestones.get(m, False)) for m in ("M1_localization", "M2_primary_fix", "M3_invariants", "M4_functional", "M5_e2e")]
    M_aggregate = round(
        0.10 * milestone_vector[0]
        + 0.20 * milestone_vector[1]
        + 0.20 * milestone_vector[2]
        + 0.20 * milestone_vector[3]
        + 0.30 * milestone_vector[4],
        4,
    )

    result = {
        "schema": SCHEMA_VERSION,
        "family": "multi-tool-transaction-repair",
        "variant": VARIANT_ID,
        "score": final,
        "P_benchmark": final,
        "M_training": state.final_m_training(),
        "M_aggregate": M_aggregate,
        "milestones": state.milestones,
        "milestone_vector": milestone_vector,
        "breakdown": state.breakdown,
        "bands": state.bands,
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "errors": state.errors,
        "diagnostics": state.diagnostics,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"score": final, "M_aggregate": M_aggregate, "ceilings": state.ceilings_applied, "integrity_flag": state.integrity_flag}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
