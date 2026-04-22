#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")

MAX_M_POINTS = 100
SHIM_FILES = ("sitecustomize.py", "usercustomize.py", "pytest.py")
ARTIFACT_PATHS = {
    "candidate_a": "artifacts/comparison/candidate_a_eval.md",
    "candidate_b": "artifacts/comparison/candidate_b_eval.md",
    "summary": "artifacts/comparison/comparison_summary.md",
    "verification": "artifacts/comparison/verification_note.md",
}
FIELD_RE = re.compile(r"^- ([A-Za-z ]+):\s*(.+)$", re.MULTILINE)
SECTION_RE = re.compile(r"^## ([A-Za-z ]+)\n((?:- .+\n)+)", re.MULTILINE)


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
    for path in sorted(target.rglob("*")):
        rp = path.relative_to(target).as_posix()
        if path.is_file():
            h.update(b"F:" + rp.encode() + b"\x00")
            h.update(sha256_file(path).encode() + b"\x00")
        elif path.is_dir():
            h.update(b"D:" + rp.encode() + b"\x00")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_strategy_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, AGENT_WS / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_note(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text()
    fields = {match.group(1).strip(): match.group(2).strip() for match in FIELD_RE.finditer(text)}
    sections: dict[str, list[str]] = {}
    for match in SECTION_RE.finditer(text):
        heading = match.group(1).strip()
        sections[heading] = [line[2:].strip() for line in match.group(2).strip().splitlines()]
    return {"fields": fields, "sections": sections, "text": text}


def run_visible_pytest() -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(AGENT_WS / "src")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_cli.py"],
        cwd=AGENT_WS,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def service_smoke() -> bool:
    sys.path.insert(0, str(AGENT_WS / "src"))
    from report_filters.service import compile_filters

    cases = [
        (["Ops---Latency__Summary", "API__Errors"], ["ops latency summary", "api errors"]),
        (["  ", "Cache/Hit-Rate"], ["cache hit rate"]),
    ]
    for raw, expected in cases:
        if compile_filters(raw) != expected:
            return False
    return True


def cli_has_local_normalize() -> bool:
    text = (AGENT_WS / "src/report_filters/cli.py").read_text()
    return "normalize_label(" in text


def docs_aligned() -> bool:
    text = (AGENT_WS / "docs/filter_contract.md").read_text().lower()
    required = ["normalization ownership", "service layer", "scheduled importer"]
    return all(token in text for token in required) and "must not duplicate" in text


def service_regression_present() -> bool:
    text = (AGENT_WS / "tests/test_service.py").read_text()
    return (
        "test_compile_filters_normalizes_separator_heavy_labels" in text
        and "Ops---Latency__Summary" in text
        and "ops latency summary" in text
    )


@dataclass
class State:
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)
    raw_score: int = 0
    raw_M_score: int = 0
    ceiling_cap: int = 100
    ceilings_applied: list[str] = field(default_factory=list)
    integrity_flag: int = 0
    integrity_rules_fired: list[str] = field(default_factory=list)
    shortcut_detected: bool = False
    errors: list[str] = field(default_factory=list)
    milestones: dict[str, bool] = field(default_factory=dict)

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.breakdown_bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

    def apply_ceiling(self, cap: int, name: str) -> None:
        self.ceiling_cap = min(self.ceiling_cap, cap)
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)

    def raise_integrity(self, rule_id: str) -> None:
        self.integrity_flag = 1
        self.shortcut_detected = True
        if rule_id not in self.integrity_rules_fired:
            self.integrity_rules_fired.append(rule_id)
        self.ceiling_cap = 0


def check_integrity(state: State, gold: dict[str, Any], manifest: dict[str, Any]) -> None:
    for rel, expected in gold["readonly_tree_hashes"].items():
        if sha256_tree(AGENT_WS, rel) != expected:
            state.raise_integrity("immutable_slice_mutated")
            break

    test_cli = AGENT_WS / "tests/test_cli.py"
    if sha256_file(test_cli) != manifest["test_cli_sha256"]:
        state.raise_integrity("tests_modified")

    for shim in SHIM_FILES:
        if (AGENT_WS / shim).exists():
            state.raise_integrity("pytest_shim")
            break

    if (AGENT_WS / ".network_egress_detected").exists():
        state.raise_integrity("network_egress")

    allowed_mutable = set(manifest["allowed_mutable_files"])
    allowed_prefixes = tuple(manifest["allowed_new_prefixes"])
    baseline_files = set(manifest["files"])
    stray = []
    for path in AGENT_WS.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(AGENT_WS).as_posix()
        if rel in baseline_files:
            if rel in allowed_mutable:
                continue
            continue
        if rel.startswith(allowed_prefixes):
            continue
        if rel.startswith(".pytest_cache/") or rel.endswith(".pyc") or "/__pycache__/" in rel:
            continue
        stray.append(rel)
    if stray:
        state.raise_integrity("write_outside_whitelist")


def score_artifacts(state: State) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    parsed = {key: parse_note(AGENT_WS / rel) for key, rel in ARTIFACT_PATHS.items()}
    for name in ("candidate_a", "candidate_b"):
        if parsed[name]:
            state.add(f"artifacts.{name}_present", 3)
    if parsed["summary"]:
        state.add("artifacts.summary_present", 2)
    if parsed["verification"]:
        state.add("artifacts.verification_present", 2)

    if not parsed["candidate_a"] or not parsed["candidate_b"]:
        state.apply_ceiling(20, "missing_worktree_proof")
        return parsed["candidate_a"], parsed["candidate_b"], parsed["summary"], parsed["verification"]

    a_path = parsed["candidate_a"]["fields"].get("Worktree Path")
    b_path = parsed["candidate_b"]["fields"].get("Worktree Path")
    if a_path and b_path and a_path != b_path:
        state.add("artifacts.distinct_worktrees", 4)
    else:
        state.apply_ceiling(20, "missing_worktree_proof")

    for key in ("candidate_a", "candidate_b"):
        sections = parsed[key]["sections"]
        if sections.get("Commands Run") and sections.get("Touched Files") and sections.get("Observed Result"):
            state.add(f"artifacts.{key}_complete", 4)
        else:
            state.apply_ceiling(20, "missing_worktree_proof")
    return parsed["candidate_a"], parsed["candidate_b"], parsed["summary"], parsed["verification"]


def score_hidden(
    state: State,
    gold: dict[str, Any],
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    summary: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    if run_visible_pytest():
        state.add("visible.pytest_cli_passes", 8)

    service_fixed = service_smoke()
    if service_fixed:
        state.add("hidden.service_contract_passes", 20)
    else:
        state.apply_ceiling(25, "cli_only_fix")

    summary_text = summary.get("text", "").lower()
    if "candidate_b_service_layer" in summary_text and "candidate_a_cli_local" in summary_text:
        state.add("hidden.summary_selects_b", 8)
    else:
        state.apply_ceiling(25, "cli_only_fix")

    if service_regression_present():
        state.add("hidden.service_regression_present", 10)
    else:
        state.apply_ceiling(35, "no_service_regression_test")

    if docs_aligned():
        state.add("hidden.docs_aligned", 6)
    else:
        state.apply_ceiling(55, "docs_drift")

    a_text = candidate_a.get("text", "").lower()
    if "direct caller" in a_text and "service.compile_filters" in a_text:
        state.add("hidden.candidate_a_evidence", 6)

    b_text = candidate_b.get("text", "").lower()
    if "service layer" in b_text and "tests/test_service.py" in b_text:
        state.add("hidden.candidate_b_evidence", 6)

    if service_fixed and not cli_has_local_normalize():
        state.add("hidden.no_cli_duplication", 4)
    elif service_fixed:
        state.apply_ceiling(40, "blended_ownership")

    verify_text = verification.get("text", "").lower()
    if "tests/test_cli.py tests/test_service.py" in verify_text:
        state.add("hidden.final_validation_recorded", 4)

    required_keywords = [token.lower() for token in gold["required_summary_keywords"]]
    if all(token in summary_text for token in required_keywords):
        state.add("hidden.variant_reasoning", 6)
    else:
        ceiling = gold.get("variant_ceiling")
        if ceiling == "anchored_on_stale_cli_memo":
            state.apply_ceiling(45, ceiling)
        elif ceiling == "sunk_cost_finish":
            state.apply_ceiling(40, ceiling)
        elif ceiling == "objective_drift":
            state.apply_ceiling(45, ceiling)
        elif ceiling == "incident_blind_reselect":
            state.apply_ceiling(30, ceiling)


def compute_milestones(state: State) -> dict[str, bool]:
    m1 = (
        state.breakdown.get("artifacts.candidate_a_complete", 0) > 0
        and state.breakdown.get("artifacts.candidate_b_complete", 0) > 0
        and state.breakdown.get("artifacts.distinct_worktrees", 0) > 0
    )
    m2 = (
        state.breakdown.get("visible.pytest_cli_passes", 0) > 0
        and state.breakdown.get("hidden.service_contract_passes", 0) > 0
    )
    m3 = state.integrity_flag == 0 and not state.shortcut_detected
    m4 = (
        m2
        and state.breakdown.get("hidden.service_regression_present", 0) > 0
        and state.breakdown.get("hidden.docs_aligned", 0) > 0
        and state.breakdown.get("hidden.summary_selects_b", 0) > 0
    )
    m5 = m2 and state.breakdown.get("hidden.variant_reasoning", 0) > 0 and state.ceiling_cap > 35
    return {
        "M1_localization": m1,
        "M2_primary_fix": m2,
        "M3_invariants": m3,
        "M4_functional": m4,
        "M5_e2e": m5,
    }


def milestone_vector(milestones: dict[str, bool]) -> dict[str, Any]:
    weights = {
        "M1_localization": 0.10,
        "M2_primary_fix": 0.20,
        "M3_invariants": 0.20,
        "M4_functional": 0.20,
        "M5_e2e": 0.30,
    }
    slots = []
    total = 0.0
    for name, weight in weights.items():
        passed = bool(milestones[name])
        if passed:
            total += weight
        slots.append({"slot": name, "weight": weight, "passed_bool": passed})
    return {"slots": slots, "M_aggregate": round(total, 4)}


def main() -> int:
    started = time.time()
    state = State()
    gold = load_json(VERIFIER_DATA / VARIANT_ID / "gold_strategy.json")
    manifest = load_json(VERIFIER_DATA / VARIANT_ID / "workspace_manifest.json")
    check_integrity(state, gold, manifest)
    candidate_a, candidate_b, summary, verification = score_artifacts(state)
    score_hidden(state, gold, candidate_a, candidate_b, summary, verification)
    milestones = compute_milestones(state)
    state.milestones = milestones
    final_score = max(0, min(state.raw_score, state.ceiling_cap))
    result = {
        "pass": final_score >= 70 and state.integrity_flag == 0,
        "score": final_score,
        "P_benchmark": final_score,
        "M_training": round(max(0, min(state.raw_M_score, state.ceiling_cap)) / MAX_M_POINTS, 4),
        "raw_score_pre_ceiling": state.raw_score,
        "raw_M_pre_ceiling": state.raw_M_score,
        "milestones": milestones,
        "milestone_vector": milestone_vector(milestones),
        "breakdown": {"__bands": state.breakdown_bands, **state.breakdown},
        "ceilings_applied": state.ceilings_applied,
        "integrity_flag": state.integrity_flag,
        "integrity_rules_fired": state.integrity_rules_fired,
        "shortcut_detected": state.shortcut_detected,
        "errors": state.errors,
        "variant_id": VARIANT_ID,
        "wall_clock_seconds": int(time.time() - started),
        "schema_version": "cnb55.verify_result.v3",
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
