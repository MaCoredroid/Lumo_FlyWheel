#!/usr/bin/env python3
"""
Deterministic scorer for `incident-evidence-synthesis`.

Aligns with the existing CNB-55 grader contract used by policy-aware,
security-audit, etc. Family milestones (M1..M5, weights from family.yaml):
  M1_localization  0.10  packet/findings.json + packet/incident_packet.md exist
  M2_primary_fix   0.20  triggering_condition matches gold semantics
  M3_invariants    0.20  failed_guardrail named correctly + sections present
  M4_functional    0.20  highest-confidence follow-up matches gold semantics
  M5_e2e           0.30  unresolved ambiguity flagged + pytest passes + no integrity violation

Integrity flags:
  - write_outside_whitelist  agent wrote outside packet/ (or pre-existing files mutated)
  - tests_modified           tests/test_incident_packet.py mutated
  - pytest_shim              shim files present
  - missing_packet           packet/* missing entirely
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

# Gold answer fragments per variant; derived from artifacts/gold_findings.json shipped in the workspace bundle.
GOLD_TRIGGER_KEYWORDS = {
    "v1-clean-baseline": [
        # canonical "bulk-refund requests accepted without idempotency keys"
        ("bulk-refund", "bulk refund", "refund"),
        ("idempotency",),
    ],
}
GOLD_GUARDRAIL_KEYWORDS = {
    "v1-clean-baseline": [
        ("idempotency-required", "idempotency required", "idempotency"),
        ("legacy_batch_header", "legacy batch header", "batch_header"),
    ],
}
GOLD_FOLLOWUP_KEYWORDS = {
    "v1-clean-baseline": [
        ("reject", "block", "deny", "remove"),
        ("bypass", "header", "legacy", "idempotency"),
    ],
}
GOLD_AMBIGUITY_KEYWORDS = {
    "v1-clean-baseline": [
        ("worker", "replay"),
    ],
}

REQUIRED_PACKET_SECTIONS = ("trigger", "guardrail", "follow-up", "ambiguity")
ALLOWED_WRITE_PATHS_PREFIXES = ("packet/",)
PYTEST_CMD = [sys.executable, "-m", "pytest"]

SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")


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


def _matches_any(text: str, alternatives: tuple[str, ...]) -> bool:
    t = text.lower()
    return any(alt.lower() in t for alt in alternatives)


def _matches_all_groups(text: str, groups: list[tuple[str, ...]]) -> bool:
    return all(_matches_any(text, group) for group in groups)


def load_gold_from_ws() -> dict[str, Any]:
    """Gold lives in the workspace bundle for this family (not in verifier_data)."""
    candidate = AGENT_WS / "artifacts" / "gold_findings.json"
    if not candidate.is_file():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_files(root: Path) -> set[str]:
    out: set[str] = set()
    for p in root.rglob("*"):
        if p.is_file() and ".git" not in p.parts and p.suffix != ".pyc":
            out.add(p.relative_to(root).as_posix())
    return out


def check_integrity(state: State) -> None:
    # Allowlist: packet/* + any pre-existing files unchanged. Disallow shims.
    for shim in SHIM_FILES:
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim", f"shim present: {shim}")
    test_file = AGENT_WS / "tests" / "test_incident_packet.py"
    if test_file.is_file():
        # Approximate: must be unchanged-shape (still has the two test functions).
        text = test_file.read_text(encoding="utf-8", errors="replace")
        if "test_packet_outputs_exist_and_name_required_sections" not in text or "test_findings_shape_is_ranked_and_evidence_backed" not in text:
            state.raise_integrity("tests_modified", "tests/test_incident_packet.py altered")


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


def run_visible_tests(state: State) -> bool:
    test_path = AGENT_WS / "tests" / "test_incident_packet.py"
    if not test_path.is_file():
        state.diagnostics["pytest_visible"] = "test_file_missing"
        return False
    try:
        proc = subprocess.run(
            [*PYTEST_CMD, "-q", "tests/test_incident_packet.py"],
            cwd=str(AGENT_WS),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        ok = proc.returncode == 0
        tail = proc.stdout.decode("utf-8", errors="replace").splitlines()[-15:]
        state.diagnostics["pytest_visible"] = {"rc": proc.returncode, "tail": tail}
        return ok
    except Exception as e:
        state.diagnostics["pytest_visible"] = f"exec_error: {e}"
        return False


def score(state: State) -> None:
    # Ensure pytest is available before any test invocation; record infra status to diagnostics.
    if not ensure_pytest(state):
        state.diagnostics["pytest_unavailable"] = True
    gold = load_gold_from_ws()
    packet_md = AGENT_WS / "packet" / "incident_packet.md"
    findings_json = AGENT_WS / "packet" / "findings.json"

    # M1: localization (artifacts created in correct place)
    m1 = packet_md.is_file() and findings_json.is_file()
    state.milestones["M1_localization"] = m1
    if m1:
        state.add("M1.packet_files_present", 10, "M")
    else:
        # If both packet artifacts missing, this is structurally degenerate.
        state.apply_ceiling("missing_packet", 0)
        state.diagnostics["missing_artifacts"] = [
            ("packet/incident_packet.md", packet_md.is_file()),
            ("packet/findings.json", findings_json.is_file()),
        ]

    # Try to load findings.json content even if M1 partial
    findings: dict[str, Any] = {}
    findings_raw = ""
    if findings_json.is_file():
        try:
            findings_raw = findings_json.read_text(encoding="utf-8")
            findings = json.loads(findings_raw)
        except Exception as e:
            state.errors.append(f"findings.json invalid JSON: {e}")
            findings = {}

    packet_text = ""
    if packet_md.is_file():
        packet_text = packet_md.read_text(encoding="utf-8", errors="replace").lower()

    # Concatenated body for keyword matching: packet text + structured findings text.
    body = packet_text + "\n" + findings_raw.lower()

    # M2: triggering condition correct
    trig_ok = _matches_all_groups(body, GOLD_TRIGGER_KEYWORDS.get(VARIANT_ID, []))
    state.milestones["M2_primary_fix"] = trig_ok
    if trig_ok:
        state.add("M2.triggering_condition", 20, "M")
    else:
        state.diagnostics.setdefault("missed", []).append("triggering_condition keywords absent")

    # M3: invariants — required sections + failed guardrail named
    sections_present = all(s in packet_text for s in REQUIRED_PACKET_SECTIONS)
    guard_ok = _matches_all_groups(body, GOLD_GUARDRAIL_KEYWORDS.get(VARIANT_ID, []))
    state.milestones["M3_invariants"] = sections_present and guard_ok
    if sections_present:
        state.add("M3.sections_present", 10, "M")
    else:
        missing = [s for s in REQUIRED_PACKET_SECTIONS if s not in packet_text]
        state.diagnostics.setdefault("missed", []).append(f"sections_missing: {missing}")
    if guard_ok:
        state.add("M3.guardrail_named", 10, "M")
    else:
        state.diagnostics.setdefault("missed", []).append("failed_guardrail keywords absent")

    # M4: follow-up action correct
    followup_ok = _matches_all_groups(body, GOLD_FOLLOWUP_KEYWORDS.get(VARIANT_ID, []))
    state.milestones["M4_functional"] = followup_ok
    if followup_ok:
        state.add("M4.followup_action", 20, "M")
    else:
        state.diagnostics.setdefault("missed", []).append("highest_confidence_followup keywords absent")

    # M5: ambiguity + ranked_findings shape + pytest
    ambig_present_text = _matches_any(body, GOLD_AMBIGUITY_KEYWORDS[VARIANT_ID][0])
    ranked_ok = (
        isinstance(findings.get("ranked_findings"), list)
        and len(findings.get("ranked_findings") or []) > 0
        and isinstance(findings.get("unresolved_ambiguity"), (str, list, dict))
    )
    top_confidence_ok = False
    if ranked_ok:
        top = findings["ranked_findings"][0]
        top_confidence_ok = isinstance(top, dict) and top.get("confidence") in {"high", "medium", "low"} and bool(top.get("evidence"))

    pytest_ok = run_visible_tests(state)

    m5_partials = sum([bool(ambig_present_text), ranked_ok, top_confidence_ok, pytest_ok])
    state.milestones["M5_e2e"] = m5_partials >= 3 and not state.integrity_flag
    # Award up to 30 across these
    state.add("M5.ambiguity_text", 5 if ambig_present_text else 0, "M")
    state.add("M5.ranked_findings_shape", 10 if ranked_ok else 0, "M")
    state.add("M5.top_finding_confidence", 5 if top_confidence_ok else 0, "M")
    state.add("M5.visible_pytest_pass", 10 if pytest_ok else 0, "M")

    # Apply ceiling if guardrail incorrectly named (the spec's hidden check)
    if not guard_ok:
        state.apply_ceiling("guardrail_misnamed", 40)


def main() -> int:
    state = State()
    if not AGENT_WS.exists():
        print(f"AGENT_WS not found: {AGENT_WS}", file=sys.stderr)
        return 2

    check_integrity(state)
    score(state)

    final = state.final_score()
    m_training = state.final_m_training()
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
        "family": "incident-evidence-synthesis",
        "variant": VARIANT_ID,
        "score": final,
        "P_benchmark": final,
        "M_training": m_training,
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
