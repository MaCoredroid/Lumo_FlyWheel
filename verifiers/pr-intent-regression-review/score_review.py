#!/usr/bin/env python3
"""
Deterministic scorer for the `pr-intent-regression-review` family.

The agent writes `artifacts/review_packet.json` via `bin/cnb55-pr-review`.
This scorer reads only that canonical JSON deliverable plus trusted-final-state
signals from the workspace.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))

SCHEMA_VERSION = "cnb55.pr_review.v1"
RESULT_SCHEMA_VERSION = "cnb55.verify_result.v3"
PASS_BAR = 85

READONLY_TREES = (
    ".scenario_variant",
    "AGENTS.md",
    "Dockerfile",
    "bin",
    "repo",
    "review",
    "release_context",
    "incident_context",
)
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
ALLOWED_NEW_FILES = {"review_input.json"}
ALLOWED_WRITE_PREFIXES = ("artifacts/",)

MAX_M_POINTS = 90


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
        if item.is_file():
            h.update(b"F:" + rel_item.encode() + b"\x00")
            h.update(sha256_file(item).encode() + b"\x00")
        elif item.is_dir():
            h.update(b"D:" + rel_item.encode() + b"\x00")
    return h.hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


@dataclass
class State:
    errors: list[str] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    ceilings_applied: list[str] = field(default_factory=list)
    integrity_rules_fired: list[str] = field(default_factory=list)
    raw_score: int = 0
    raw_m_score: int = 0
    ceiling_cap: int = 100
    integrity_flag: int = 0
    shortcut_detected: bool = False
    milestones: dict[str, bool] = field(default_factory=dict)

    def add(self, key: str, points: int, *, band: str = "M") -> None:
        points = max(0, int(points))
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.breakdown_bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_m_score += points

    def cap(self, cap_value: int, reason: str) -> None:
        if reason not in self.ceilings_applied:
            self.ceilings_applied.append(reason)
        self.ceiling_cap = min(self.ceiling_cap, cap_value)
        if cap_value <= 30:
            self.shortcut_detected = True

    def raise_integrity(self, rule_id: str) -> None:
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        self.integrity_flag = 1


def trusted_final_state(state: State, workspace_manifest: dict[str, Any]) -> None:
    for rel in READONLY_TREES:
        expected = workspace_manifest.get("readonly_tree_hashes", {}).get(rel)
        if expected is None:
            continue
        if sha256_tree(AGENT_WS, rel) != expected:
            state.raise_integrity("immutable_slice_mutated")
            break

    repo_tests_expected = workspace_manifest.get("readonly_tree_hashes", {}).get("repo/tests")
    if repo_tests_expected and sha256_tree(AGENT_WS, "repo/tests") != repo_tests_expected:
        state.raise_integrity("tests_modified")

    for shim_name in SHIM_FILES:
        if any(AGENT_WS.rglob(shim_name)):
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    allowed_existing = set(workspace_manifest.get("files", []))
    for path in sorted(AGENT_WS.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(AGENT_WS).as_posix()
        if rel in allowed_existing:
            continue
        if rel in ALLOWED_NEW_FILES:
            continue
        if rel.startswith(ALLOWED_WRITE_PREFIXES):
            continue
        state.raise_integrity("write_outside_whitelist")
        break


def find_file_line_count(relpath: str) -> int:
    path = AGENT_WS / relpath
    if not path.exists():
        return 0
    return len(path.read_text().splitlines())


def finding_text(finding: dict[str, Any]) -> str:
    fields = [
        finding.get("title", ""),
        finding.get("impact", ""),
        finding.get("test_gap", ""),
    ]
    return normalize(" ".join(fields))


def keyword_groups_hit(text: str, groups: list[list[str]]) -> bool:
    for group in groups:
        if not any(normalize(token) in text for token in group):
            return False
    return True


def match_issue(finding: dict[str, Any], issue: dict[str, Any]) -> bool:
    changed_file = finding.get("changed_file")
    lines = finding.get("changed_lines", {})
    linked_surface = finding.get("linked_surface")
    evidence_paths = set(finding.get("evidence_paths", []))
    if changed_file != issue["changed_file"]:
        return False
    try:
        start = int(lines.get("start", -1))
        end = int(lines.get("end", -1))
    except (TypeError, ValueError):
        return False
    if not overlaps(start, end, issue["anchor"]["start"], issue["anchor"]["end"]):
        return False
    if linked_surface not in issue["allowed_linked_surfaces"]:
        return False
    if changed_file not in evidence_paths or linked_surface not in evidence_paths:
        return False
    if not keyword_groups_hit(finding_text(finding), issue["keyword_groups"]):
        return False
    return True


def score(packet: dict[str, Any], gold: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    state = State()
    trusted_final_state(state, manifest)

    review_packet = AGENT_WS / "artifacts" / "review_packet.json"
    findings_md = AGENT_WS / "artifacts" / "review_findings.md"
    summary_md = AGENT_WS / "artifacts" / "review_summary.md"
    if review_packet.exists():
        state.add("phase2.packet_present", 2)
    if findings_md.exists():
        state.add("phase2.findings_md_present", 1)
    if summary_md.exists():
        state.add("phase2.summary_md_present", 1)

    if packet.get("schema_version") == SCHEMA_VERSION:
        state.add("phase2.schema_version", 3)
    if packet.get("variant_id") == VARIANT_ID:
        state.add("phase2.variant_match", 3)

    findings = packet.get("findings", [])
    if isinstance(findings, list) and 2 <= len(findings) <= 4:
        state.add("phase2.finding_count", 4)

    summary_text = normalize(packet.get("summary", ""))
    intent_verdict = packet.get("intent_verdict")
    if intent_verdict == "does_not_match_intent":
        state.add("summary.intent_verdict", 5, band="P_only")
    if "json" in summary_text and ("default" in summary_text or "output json" in summary_text) and ("test" in summary_text or "coverage" in summary_text):
        state.add("summary.specificity", 5, band="P_only")

    matched_issue_ids: set[str] = set()
    matched_anchor_count = 0
    matched_cross_surface = 0
    distinct_evidence: set[str] = set()
    all_evidence_paths: set[str] = set()
    out_of_bundle = False
    stale_draft_cited = False
    release_context_cited = False
    incident_context_cited = False
    style_like_count = 0
    generated_noise_focus_count = 0

    expected_issues = gold["expected_issues"]

    for finding in findings:
        text = finding_text(finding)
        changed_file = str(finding.get("changed_file", ""))
        linked_surface = str(finding.get("linked_surface", ""))
        evidence_paths = set(str(p) for p in finding.get("evidence_paths", []))
        all_evidence_paths |= evidence_paths
        distinct_evidence |= evidence_paths

        if any(token in text for token in ("style", "format", "typo", "nit", "wording", "rename", "naming")):
            style_like_count += 1
        if changed_file.endswith("release_readiness.md") or changed_file.endswith("markdown_export.md"):
            generated_noise_focus_count += 1
        if changed_file.endswith("fixtures/release_readiness.md"):
            generated_noise_focus_count += 1
        if "review/stale_draft_comments.md" in evidence_paths or linked_surface == "review/stale_draft_comments.md":
            stale_draft_cited = True
        if any(path.startswith("release_context/") for path in evidence_paths) or linked_surface.startswith("release_context/"):
            release_context_cited = True
        if any(path.startswith("incident_context/") for path in evidence_paths) or linked_surface.startswith("incident_context/"):
            incident_context_cited = True

        max_line = find_file_line_count(changed_file)
        try:
            start = int(finding.get("changed_lines", {}).get("start", -1))
            end = int(finding.get("changed_lines", {}).get("end", -1))
        except (TypeError, ValueError):
            start, end = -1, -1
        if max_line > 0 and 1 <= start <= end <= max_line:
            matched_anchor_count += 1

        if linked_surface and linked_surface != changed_file:
            matched_cross_surface += 1

        for issue in expected_issues:
            if issue["id"] in matched_issue_ids:
                continue
            if match_issue(finding, issue):
                matched_issue_ids.add(issue["id"])
                state.add(issue["score_key"], issue["points"])
                break

    if matched_anchor_count >= 2:
        state.add("anchors.changed_lines", 8)
    elif matched_anchor_count == 1:
        state.add("anchors.changed_lines", 4)

    if matched_cross_surface >= 2:
        state.add("anchors.cross_surface_linkage", 8)
    elif matched_cross_surface == 1:
        state.add("anchors.cross_surface_linkage", 4)

    allowed_prefixes = tuple(gold["allowed_evidence_roots"])
    if all_evidence_paths and all(path.startswith(allowed_prefixes) for path in all_evidence_paths):
        state.add("integrity.evidence_discipline", 5)
    elif all_evidence_paths:
        out_of_bundle = True

    if (
        any(path.startswith("review/") for path in distinct_evidence)
        or any(path.startswith("release_context/") for path in distinct_evidence)
        or any(path.startswith("incident_context/") for path in distinct_evidence)
    ):
        state.add("context.grounded", 5)

    if len(findings) >= 2 and style_like_count >= 2:
        state.cap(10, "style_only_review")
    if generated_noise_focus_count >= 2:
        state.cap(15, "generated_fixture_fixation")
    if len(matched_issue_ids) == 0:
        state.cap(20, "unanchored_or_wrong_findings")
    if matched_cross_surface < 2:
        state.cap(25, "no_cross_surface_linkage")
    if "looks good" in summary_text or ("ci" in summary_text and "safe" in summary_text):
        state.cap(30, "ci_complacency")
    if out_of_bundle:
        state.cap(20, "out_of_bundle_evidence")
    if VARIANT_ID == "v3-dirty-state" and stale_draft_cited:
        state.cap(20, "stale_draft_parroting")
    if VARIANT_ID in {"v4-multi-corpus-objective", "v5-recovery-in-thread"} and not release_context_cited:
        state.cap(35, "release_context_blindness")
    if VARIANT_ID == "v5-recovery-in-thread":
        if not incident_context_cited or not any(token in summary_text for token in ("rollback", "reintroduc", "repeat", "again", "regression")):
            state.cap(30, "incident_blind_repeat")

    if state.integrity_flag == 1:
        state.shortcut_detected = True

    major_ok = {"default_output_regression", "json_renderer_contract_regression"}.issubset(matched_issue_ids)
    tests_ok = "missing_regression_tests" in matched_issue_ids
    summary_ok = intent_verdict == "does_not_match_intent"

    state.milestones = {
        "M1_localization": len(distinct_evidence) >= 3,
        "M2_primary_fix": review_packet.exists() and packet.get("schema_version") == SCHEMA_VERSION,
        "M3_invariants": state.integrity_flag == 0 and "style_only_review" not in state.ceilings_applied and "out_of_bundle_evidence" not in state.ceilings_applied,
        "M4_functional": len(matched_issue_ids) >= 2 and matched_anchor_count >= 2 and summary_ok,
        "M5_e2e": major_ok and tests_ok and matched_cross_surface >= 2 and summary_ok and state.ceiling_cap > 30,
    }
    if state.integrity_flag == 1:
        state.milestones["M3_invariants"] = False
        state.milestones["M4_functional"] = False
        state.milestones["M5_e2e"] = False

    p_score = min(state.raw_score, state.ceiling_cap)
    m_pre = min(state.raw_m_score, state.ceiling_cap)
    m_training = 0.0 if state.integrity_flag == 1 else round(max(0.0, min(1.0, m_pre / MAX_M_POINTS)), 4)
    pass_bool = (
        p_score >= PASS_BAR
        and state.milestones["M5_e2e"]
        and state.integrity_flag == 0
    )

    milestone_slots = [
        ("M1_localization", 0.10),
        ("M2_primary_fix", 0.20),
        ("M3_invariants", 0.20),
        ("M4_functional", 0.20),
        ("M5_e2e", 0.30),
    ]
    m_aggregate = 0.0
    slots = []
    for key, weight in milestone_slots:
        passed = bool(state.milestones.get(key, False))
        slots.append({"slot_id": key, "weight": weight, "passed_bool": passed})
        if passed:
            m_aggregate += weight

    return {
        "pass": pass_bool,
        "score": p_score,
        "P_benchmark": p_score,
        "M_training": m_training,
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_m_score,
        "milestones": state.milestones,
        "milestone_vector": {
            "slots": slots,
            "M_aggregate": round(m_aggregate, 4),
        },
        "breakdown": {
            "__bands": state.breakdown_bands,
            **state.breakdown,
        },
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "schema_version": RESULT_SCHEMA_VERSION,
        "wall_clock_seconds": int(time.time()),
    }


def main() -> int:
    start = time.time()
    gold = load_json(VERIFIER_DATA / VARIANT_ID / "gold_review.json")
    manifest = load_json(VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json")
    packet_path = AGENT_WS / "artifacts" / "review_packet.json"
    if not packet_path.exists():
        result = {
            "pass": False,
            "score": 0,
            "P_benchmark": 0,
            "M_training": 0.0,
            "raw_score_pre_ceiling": 0,
            "raw_M_pre_ceiling": 0,
            "milestones": {
                "M1_localization": False,
                "M2_primary_fix": False,
                "M3_invariants": True,
                "M4_functional": False,
                "M5_e2e": False,
            },
            "milestone_vector": {
                "slots": [
                    {"slot_id": "M1_localization", "weight": 0.10, "passed_bool": False},
                    {"slot_id": "M2_primary_fix", "weight": 0.20, "passed_bool": False},
                    {"slot_id": "M3_invariants", "weight": 0.20, "passed_bool": True},
                    {"slot_id": "M4_functional", "weight": 0.20, "passed_bool": False},
                    {"slot_id": "M5_e2e", "weight": 0.30, "passed_bool": False},
                ],
                "M_aggregate": 0.2,
            },
            "breakdown": {"__bands": {}},
            "ceilings_applied": [],
            "integrity_flag": 0,
            "integrity_rules_fired": [],
            "shortcut_detected": False,
            "errors": ["missing artifacts/review_packet.json"],
            "variant_id": VARIANT_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "wall_clock_seconds": int(time.time() - start),
        }
    else:
        try:
            packet = load_json(packet_path)
            result = score(packet, gold, manifest)
            result["wall_clock_seconds"] = int(time.time() - start)
        except Exception as exc:  # noqa: BLE001
            result = {
                "pass": False,
                "score": 0,
                "P_benchmark": 0,
                "M_training": 0.0,
                "raw_score_pre_ceiling": 0,
                "raw_M_pre_ceiling": 0,
                "milestones": {
                    "M1_localization": False,
                    "M2_primary_fix": False,
                    "M3_invariants": False,
                    "M4_functional": False,
                    "M5_e2e": False,
                },
                "milestone_vector": {"slots": [], "M_aggregate": 0.0},
                "breakdown": {"__bands": {}},
                "ceilings_applied": [],
                "integrity_flag": 0,
                "integrity_rules_fired": [],
                "shortcut_detected": False,
                "errors": [f"scorer_exception: {exc}"],
                "variant_id": VARIANT_ID,
                "schema_version": RESULT_SCHEMA_VERSION,
                "wall_clock_seconds": int(time.time() - start),
            }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
